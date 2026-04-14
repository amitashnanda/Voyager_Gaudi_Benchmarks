                      
"""
PMPQ_sensitivity_hellaswag_hpu_real.py

PMPQ Phase 1 (Submodule-level): pruning-based sensitivity analysis for TinyLlama/Llama-style models on HellaSwag.

Key properties:
- Submodule-level sensitivities for exact linear submodules:
  self_attn.{q_proj,k_proj,v_proj,o_proj}, mlp.{gate_proj,up_proj,down_proj}
- Prunes ONLY nn.Linear.weight (no norms/embeddings), magnitude-based, per-submodule thresholding
- Sensitivity = Accuracy(baseline) - Accuracy(pruned_submodule)
  Positive sensitivity = accuracy drops when pruned → layer is important → keep in BF16
- Train split for sensitivity analysis (test labels unavailable, validation reserved for Phase 2)
- Single-card by default, no interactive prompts; structured for future distributed extension

Notes:
- This script produces module-name keyed sensitivities:
  e.g., "model.layers.0.self_attn.q_proj": 0.0123
- HellaSwag dataset: train (~39,905 samples), validation (~10,042 samples), test (unavailable labels)
- Default: 2000 train samples for sensitivity computation (configurable via --max_samples)
- Metric: Accuracy (4-choice commonsense reasoning) using length-normalized log-likelihood

Usage:
    PT_HPU_LAZY_MODE=1 python PMPQ_sensitivity_hellaswag_hpu_real.py --sparsity 0.3 --max_samples 2000
"""

import os
import gc
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

                                                            
try:
    import habana_frameworks.torch.hpu as hthpu
    import habana_frameworks.torch.core as htcore
    HABANA_AVAILABLE = True
except Exception:
    hthpu = None
    htcore = None
    HABANA_AVAILABLE = False


LLAMA_LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("PMPQ_PHASE1_HELLASWAG_SUBMODULE")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"phase1_hellaswag_submodule_{ts}.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if HABANA_AVAILABLE and hthpu.is_available():
        return torch.device("hpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def free_hpu_memory() -> None:
    gc.collect()
    try:
        if hasattr(torch, "hpu") and hasattr(torch.hpu, "empty_cache"):
            torch.hpu.empty_cache()
    except Exception:
        pass
    try:
        if HABANA_AVAILABLE:
            htcore.mark_step()
    except Exception:
        pass


def load_hellaswag_dataset(split: str = "train"):
    """
    Load HellaSwag dataset from HuggingFace.

    HellaSwag has train (~39905 samples), validation (~10042 samples).
    Test split labels are not publicly available.
    Use train for sensitivity, validation for final evaluation.
    """
    dataset = load_dataset("Rowan/hellaswag", split=split, trust_remote_code=True)
    return dataset


def preprocess_hellaswag_text(text: str) -> str:
    """
    Preprocess HellaSwag text (ctx_a, ctx_b, endings).
    Remove extra whitespace and normalize formatting.
    """
    text = text.strip()
    text = " ".join(text.split())
    return text


def cast_model_dtype(model: nn.Module, dtype: str) -> nn.Module:
    if dtype == "fp32":
        return model.to(dtype=torch.float32)
    if dtype == "bf16":
        return model.to(dtype=torch.bfloat16)
    if dtype == "fp16":
        return model.to(dtype=torch.float16)
    raise ValueError(f"Unsupported dtype: {dtype}")


def evaluate_hellaswag_accuracy(
    model: nn.Module,
    dataset,
    tokenizer,
    device: torch.device,
    use_mark_step: bool,
    logger: logging.Logger,
    desc: str,
    max_samples: int = 0,
    batch_size: int = 32,
) -> float:
    """
    Evaluate accuracy on HellaSwag using log-likelihood scoring on HPU with batched inference.

    For each sample:
      1. Context = ctx (ctx_a + ctx_b, preprocessed)
      2. For each of 4 candidate endings:
         - Tokenize context + " " + ending
         - Forward pass → logits
         - Compute length-normalized log-likelihood of ending tokens
      3. Pick ending with highest normalized log-likelihood (acc_norm)

    Batched processing: Processes batch_size samples in parallel for faster inference.

    This matches the lm-evaluation-harness methodology.
    TinyLlama-1.1B-3T official score: 59.20

    Returns: accuracy (0.0 to 100.0)
    """
    model.eval()
    model.to(device)

                                      
    if HABANA_AVAILABLE and device.type == "hpu":
        try:
            htcore.hpu_set_env()
        except Exception:
            pass
        try:
            if hasattr(htcore, "hpu_inference_initialize"):
                htcore.hpu_inference_initialize(model=model, mark_scales=True, mark_non_scales=False)
            else:
                htcore.hpu_initialize(model=model, mark_scales=True, mark_non_scales=False)
        except Exception:
            pass

    samples = dataset
    if max_samples > 0:
        n = min(max_samples, len(dataset))
        samples = dataset.select(range(n))

    correct = 0
    total = len(samples)

    t0 = time.time()

    with torch.no_grad():
                            
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_samples = [samples[i] for i in range(batch_start, batch_end)]
            actual_batch_size = len(batch_samples)

                                                           
            all_texts = []
            ctx_lens = []
            labels = []

            for sample in batch_samples:
                ctx = preprocess_hellaswag_text(sample["ctx"])
                endings = [preprocess_hellaswag_text(e) for e in sample["endings"]]
                label = int(sample["label"])
                labels.append(label)

                                    
                ctx_enc = tokenizer(ctx, add_special_tokens=True)
                ctx_len = len(ctx_enc["input_ids"])
                ctx_lens.append(ctx_len)

                                                             
                for ending in endings:
                    all_texts.append(ctx + " " + ending)

                                                                 
            batch = tokenizer(all_texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=2048)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

                                                             
            outputs = model(input_ids, attention_mask=attention_mask)

            if use_mark_step and HABANA_AVAILABLE and device.type == "hpu":
                htcore.mark_step()

                                                        
            logits = outputs.logits.float().cpu()
            input_ids_cpu = batch["input_ids"]
            attn_cpu = batch["attention_mask"]

                                              
            for idx in range(actual_batch_size):
                ctx_len = ctx_lens[idx]
                best_ll = float("-inf")
                best_idx = 0

                                                       
                for j in range(4):
                    candidate_idx = idx * 4 + j
                    seq_len = int(attn_cpu[candidate_idx].sum().item())
                    ending_len = seq_len - ctx_len

                    if ending_len <= 0:
                        continue

                                                          
                    shift_logits = logits[candidate_idx, ctx_len - 1 : seq_len - 1, :]
                    shift_labels = input_ids_cpu[candidate_idx, ctx_len : seq_len]

                    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                    token_lps = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

                                                          
                    avg_ll = token_lps.sum().item() / ending_len

                    if avg_ll > best_ll:
                        best_ll = avg_ll
                        best_idx = j

                if best_idx == labels[idx]:
                    correct += 1

    accuracy = correct / total * 100.0
    dt = time.time() - t0
    logger.info(f"[{desc}] accuracy={accuracy:.2f}% ({correct}/{total}), time={dt:.1f}s")

    model.to("cpu")
    free_hpu_memory()
    return accuracy


def list_candidate_linear_modules(model: nn.Module) -> List[str]:
    names = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and any(name.endswith(suf) for suf in LLAMA_LINEAR_SUFFIXES):
            names.append(name)
    names.sort()
    return names


def get_module_by_name(model: nn.Module, qualname: str) -> nn.Module:
    cur = model
    for part in qualname.split("."):
        cur = getattr(cur, part)
    return cur


def apply_magnitude_pruning_linear(linear: nn.Linear, sparsity: float) -> None:
    """
    Prune ONLY linear.weight by zeroing out the smallest-magnitude weights.
    Threshold is computed per-submodule by flattening the full weight tensor.
    """
    if not hasattr(linear, "weight") or linear.weight is None:
        return
    w = linear.weight.data
    flat = w.view(-1)
    n = flat.numel()
    keep = int(n * (1.0 - sparsity))
    if keep <= 0:
        linear.weight.data.zero_()
        return
    if keep >= n:
        return
                                        
    thresh = torch.topk(flat.abs(), k=keep, largest=True).values[-1]
    mask = (flat.abs() >= thresh).to(w.dtype)
    linear.weight.data.mul_(mask.view_as(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    ap.add_argument("--split", type=str, default="train", choices=["train", "validation"],
                    help="HellaSwag split for sensitivity (default: train)")
    ap.add_argument("--max_samples", type=int, default=2000,
                    help="Max samples from split (0 = full split; default: 2000)")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="Batch size for HellaSwag evaluation (default: 32)")
    ap.add_argument("--sparsity", type=float, default=0.30,
                    help="Pruning sparsity level (default: 0.30)")
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"],
                    help="Model dtype for sensitivity analysis (default: fp32)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default="Sensitivities_submodule",
                    help="Output directory for sensitivity JSON (default: Sensitivities_submodule)")
    ap.add_argument("--log_dir", type=str, default="logs_phase1_submodule",
                    help="Log directory (default: logs_phase1_submodule)")
    ap.add_argument("--max_modules", type=int, default=0,
                    help="Max modules to evaluate (0 = all; for debugging)")
    ap.add_argument("--use_mark_step", action="store_true",
                    help="Call htcore.mark_step() on HPU per iteration")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    logger = setup_logger(log_dir)

    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device} | Habana available: {HABANA_AVAILABLE}")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Split: {args.split} | max_samples={args.max_samples} | batch_size={args.batch_size}")
    logger.info(f"Sparsity: {args.sparsity:.2f} | dtype={args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_hellaswag_dataset(split=args.split)
    logger.info(f"Loaded HellaSwag {args.split} split: {len(dataset)} samples")

                       
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    base_model = cast_model_dtype(base_model, args.dtype)
    baseline_acc = evaluate_hellaswag_accuracy(
        base_model, dataset, tokenizer, device, args.use_mark_step, logger,
        desc="Baseline", max_samples=args.max_samples, batch_size=args.batch_size
    )
    del base_model
    free_hpu_memory()

                                                          
    probe_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    candidates = list_candidate_linear_modules(probe_model)
    del probe_model

    if args.max_modules and args.max_modules > 0:
        candidates = candidates[: args.max_modules]

    logger.info(f"Found {len(candidates)} candidate linear submodules.")

    sensitivities: Dict[str, float] = {}
    t_all = time.time()

    for i, mod_name in enumerate(candidates, 1):
        logger.info(f"[{i}/{len(candidates)}] Pruning submodule: {mod_name}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        model = cast_model_dtype(model, args.dtype)

        submod = get_module_by_name(model, mod_name)
        assert isinstance(submod, nn.Linear), f"Expected nn.Linear at {mod_name}, got {type(submod)}"

                                              
        model.to("cpu")
        apply_magnitude_pruning_linear(submod, args.sparsity)

        acc = evaluate_hellaswag_accuracy(
            model, dataset, tokenizer, device, args.use_mark_step, logger,
            desc=f"Pruned {mod_name}", max_samples=args.max_samples, batch_size=args.batch_size
        )

                                                 
                                                                      
        delta = float(baseline_acc - acc)
        sensitivities[mod_name] = delta
        logger.info(f"  Sensitivity[{mod_name}] = {delta:+.6f}")

        del model
        free_hpu_memory()

    dt_all = time.time() - t_all
    logger.info(f"Completed sensitivities in {dt_all:.1f}s")

                  
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_name_short = Path(args.model_name_or_path).name
    samples_str = f"n{args.max_samples}" if args.max_samples > 0 else "full"
    out_json = out_dir / f"submodule_sens_{model_name_short}_{args.split}_pruning_s{int(args.sparsity*100)}_{samples_str}_{ts}.json"

    payload = {
        "meta": {
            "model_name_or_path": args.model_name_or_path,
            "split": args.split,
            "max_samples": args.max_samples if args.max_samples > 0 else len(dataset),
            "batch_size": args.batch_size,
            "sparsity": args.sparsity,
            "dtype": args.dtype,
            "baseline_accuracy": baseline_acc,
            "timestamp": ts,
        },
        "sensitivities": sensitivities,
    }

    out_json.write_text(json.dumps(payload, indent=2))
    logger.info(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
