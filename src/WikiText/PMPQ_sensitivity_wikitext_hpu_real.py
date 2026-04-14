                      
"""
Phase1_submodule_sensitivity.py

PMPQ Phase 1 (Submodule-level): pruning-based sensitivity analysis for TinyLlama/Llama-style models.

Key properties:
- Submodule-level sensitivities for exact linear submodules:
  self_attn.{q_proj,k_proj,v_proj,o_proj}, mlp.{gate_proj,up_proj,down_proj}
- Prunes ONLY nn.Linear.weight (no norms/embeddings), magnitude-based, per-submodule thresholding
- Sensitivity = PPL(pruned_submodule) - PPL(baseline)
- Validation split for sensitivity analysis (test split reserved for Phase 2 evaluation)
- Single-card by default, no interactive prompts; structured for future distributed extension

Notes:
- This script is intended to produce module-name keyed sensitivities:
  e.g., "model.layers.0.self_attn.q_proj": 0.0123
"""

import os
import gc
import json
import math
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from itertools import chain
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator

                                                            
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
    logger = logging.getLogger("PMPQ_PHASE1_SUBMODULE")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"phase1_submodule_{ts}.log", mode="w")
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


def prepare_wikitext_dataset(tokenizer, split: str, block_size: int) -> torch.utils.data.Dataset:
                                                          
    try:
        raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception:
        raw_data = load_dataset("wikitext", "wikitext-2-v1", split=split)

    def tokenize_fn(examples):
        return tokenizer(examples["text"])

    tokenized = raw_data.map(
        tokenize_fn,
        batched=True,
        remove_columns=raw_data.column_names,
        desc=f"Tokenizing {split}",
    )

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(group_texts, batched=True, desc=f"Chunking {split} to {block_size}")
    return lm_dataset


def build_dataloader(dataset, batch_size: int, max_samples: Optional[int] = None):
    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        drop_last=False,
    )


def cast_model_dtype(model: nn.Module, dtype: str) -> nn.Module:
    if dtype == "fp32":
        return model.to(dtype=torch.float32)
    if dtype == "bf16":
        return model.to(dtype=torch.bfloat16)
    if dtype == "fp16":
        return model.to(dtype=torch.float16)
    raise ValueError(f"Unsupported dtype: {dtype}")


def evaluate_perplexity(
    model: nn.Module,
    dataloader,
    device: torch.device,
    use_mark_step: bool,
    logger: logging.Logger,
    desc: str,
) -> float:
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

    losses: List[float] = []
    t0 = time.time()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
                                                  
            out = model(**batch)
            loss = out.loss
            losses.append(float(loss.detach().cpu()))
            if use_mark_step and HABANA_AVAILABLE and device.type == "hpu":
                htcore.mark_step()

    avg_loss = float(np.mean(losses)) if losses else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    dt = time.time() - t0
    logger.info(f"[{desc}] avg_loss={avg_loss:.6f}, ppl={ppl:.4f}, time={dt:.1f}s")

    model.to("cpu")
    free_hpu_memory()
    return ppl


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
    ap.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--block_size", type=int, default=512)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=4)
    ap.add_argument("--max_eval_samples", type=int, default=0, help="0 = full split; else evaluate only first N blocks")
    ap.add_argument("--sparsity", type=float, default=0.30)
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", type=str, default="Sensitivities_submodule")
    ap.add_argument("--log_dir", type=str, default="logs_phase1_submodule")
    ap.add_argument("--max_modules", type=int, default=0, help="0 = all; else evaluate only first N modules (debug)")
    ap.add_argument("--use_mark_step", action="store_true", help="Call htcore.mark_step() on HPU per iteration")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    logger = setup_logger(log_dir)

    set_seed(args.seed)
    device = get_device()
    logger.info(f"Device: {device} | Habana available: {HABANA_AVAILABLE}")
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Split: {args.split} | block_size={args.block_size} | batch={args.per_device_eval_batch_size}")
    logger.info(f"Sparsity: {args.sparsity:.2f} | dtype={args.dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = prepare_wikitext_dataset(tokenizer, args.split, args.block_size)
    max_samples = args.max_eval_samples if args.max_eval_samples and args.max_eval_samples > 0 else None
    dataloader = build_dataloader(dataset, args.per_device_eval_batch_size, max_samples=max_samples)

              
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    base_model = cast_model_dtype(base_model, args.dtype)
    baseline_ppl = evaluate_perplexity(
        base_model, dataloader, device, args.use_mark_step, logger, desc="Baseline"
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

        ppl = evaluate_perplexity(
            model, dataloader, device, args.use_mark_step, logger, desc=f"Pruned {mod_name}"
        )
        delta = float(ppl - baseline_ppl)
        sensitivities[mod_name] = delta
        logger.info(f"  Sensitivity[{mod_name}] = {delta:+.6f}")

        del model
        free_hpu_memory()

    dt_all = time.time() - t_all
    logger.info(f"Completed sensitivities in {dt_all:.1f}s")

                  
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"submodule_sens_{Path(args.model_name_or_path).name}_{args.split}_pruning_s{int(args.sparsity*100)}_{ts}.json"

    payload = {
        "meta": {
            "model_name_or_path": args.model_name_or_path,
            "split": args.split,
            "block_size": args.block_size,
            "batch_size": args.per_device_eval_batch_size,
            "max_eval_samples": max_samples,
            "sparsity": args.sparsity,
            "dtype": args.dtype,
            "baseline_ppl": baseline_ppl,
            "timestamp": ts,
        },
        "sensitivities": sensitivities,
    }

    out_json.write_text(json.dumps(payload, indent=2))
    logger.info(f"Saved: {out_json}")


if __name__ == "__main__":
    main()


                                                             
                                                                              
                        
                     
                  
                      
                                    
                   