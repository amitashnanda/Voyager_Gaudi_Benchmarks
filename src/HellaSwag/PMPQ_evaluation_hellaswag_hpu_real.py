                      
"""
PMPQ_evaluation_hellaswag_hpu_real.py

HellaSwag evaluation with FP8 quantization using official Gaudi two-pass flow:
  MEASURE (prepare) -> forward-only calibration -> finalize_calibration
  QUANTIZE (convert) -> evaluate accuracy

Evaluates HellaSwag 4-choice multiple-choice accuracy using log-likelihood scoring.

Key differences from WikiText version:
- Uses HellaSwag accuracy instead of perplexity
- Calibration on train split (512 samples)
- Evaluation on validation split (~10,042 samples)
- Batched log-likelihood scoring for all 4 endings per sample
- Throughput measured in samples/second (not tokens/second)

References:
- Gaudi FP8 two-pass flow, LOG_LEVEL_INC, PT_HPU_WEIGHT_SHARING: see Habana FP8 inference doc.
"""

import os
import re
import gc
import json
import math
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.cluster import KMeans

import habana_frameworks.torch.hpu as hthpu
import habana_frameworks.torch.core as htcore
from neural_compressor.torch.quantization import FP8Config, prepare, convert, finalize_calibration


LLAMA_MLP_SUFFIXES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
LLAMA_ATTN_O_SUFFIX = ("self_attn.o_proj",)
LLAMA_ALL_LINEAR_SUFFIXES = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)

DEFAULT_BLOCKLIST = ["lm_head", "model.norm", "model.embed_tokens", "embed_tokens"]

DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}


def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("PMPQ_HELLASWAG_FP8")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"hellaswag_fp8_{ts}.log", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def free_hpu_memory() -> None:
    gc.collect()
    try:
        if hasattr(torch, "hpu") and hasattr(torch.hpu, "empty_cache"):
            torch.hpu.empty_cache()
    except Exception:
        pass
    try:
        htcore.mark_step()
    except Exception:
        pass


def get_device() -> torch.device:
    if hthpu.is_available():
        return torch.device("hpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def hpu_set_env_safe() -> None:
                                                                                         
    if hasattr(htcore, "hpu_inference_set_env"):
        htcore.hpu_inference_set_env()
    else:
        htcore.hpu_set_env()


def hpu_infer_init_safe(model: nn.Module) -> None:
                                                                             
    if hasattr(htcore, "hpu_inference_initialize"):
        htcore.hpu_inference_initialize(model=model, mark_scales=True, mark_non_scales=False)
    else:
        htcore.hpu_initialize(model=model, mark_scales=True, mark_non_scales=False)


def load_hellaswag_dataset(split: str):
    """Load HellaSwag dataset from Rowan/hellaswag."""
    return load_dataset("Rowan/hellaswag", split=split, trust_remote_code=True)


def preprocess_hellaswag_text(text: str) -> str:
    """
    Clean up HellaSwag text artifacts from WikiHow portion.
    Matches FAKE quantization script preprocessing exactly.
    """
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def cast_model_dtype(model: nn.Module, dtype: str) -> nn.Module:
    if dtype == "fp32":
        return model.to(dtype=torch.float32)
    if dtype == "bf16":
        return model.to(dtype=torch.bfloat16)
    if dtype == "fp16":
        return model.to(dtype=torch.float16)
    raise ValueError(f"Unsupported dtype: {dtype}")


def calibration_forward_hellaswag(
    model: nn.Module,
    dataset,
    tokenizer,
    device: torch.device,
    batch_size: int,
    use_mark_step: bool,
    max_samples: int = 512
) -> float:
    """
    Forward-only calibration on HellaSwag train split.
    Process batch_size samples at a time, tokenize all 4 endings per sample, run forward passes.
    No accuracy computation, just calibration.

    Returns calibration time in seconds.
    """
    model.eval()
    model.to(device)

                                                                            
    if device.type == "hpu":
        hpu_set_env_safe()
        hpu_infer_init_safe(model)

    num_samples = min(max_samples, len(dataset))

    t0 = time.time()
    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_samples = dataset.select(range(start_idx, end_idx))

                                              
            for sample in batch_samples:
                ctx = preprocess_hellaswag_text(sample["ctx"])
                endings = [preprocess_hellaswag_text(e) for e in sample["endings"]]

                                        
                for ending in endings:
                    full_text = ctx + " " + ending
                    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048)
                    input_ids = inputs["input_ids"].to(device)

                                                  
                    _ = model(input_ids=input_ids)

                    if use_mark_step and device.type == "hpu":
                        htcore.mark_step()

    dt = time.time() - t0

    model.to("cpu")
    free_hpu_memory()
    return dt


def evaluate_hellaswag_accuracy(
    model: nn.Module,
    dataset,
    tokenizer,
    device: torch.device,
    batch_size: int,
    use_mark_step: bool,
    logger: logging.Logger,
    use_hpu_init: bool = False
) -> Tuple[float, float]:
    """
    Evaluate HellaSwag accuracy using batched log-likelihood scoring on HPU.

    For each sample:
      1. Context = ctx (preprocessed)
      2. For each of 4 candidate endings:
         - Tokenize context + " " + ending
         - Forward pass → logits
         - Compute length-normalized log-likelihood of ending tokens
      3. Pick ending with highest normalized log-likelihood (acc_norm)

    Batched processing: Processes batch_size samples × 4 endings in parallel.
    This matches the FAKE quantization evaluation approach.

    Args:
        use_hpu_init: If True, call hpu_set_env_safe() and hpu_infer_init_safe().
                      Default False to match FAKE script (FP32/BF16 baseline needs no init).
                      Set True for FP8 converted models if needed.

    Returns (accuracy_percentage, eval_time_seconds).
    """
    model.eval()
    model.to(device)

                                                                              
                                                                                   
    if use_hpu_init and device.type == "hpu":
        hpu_set_env_safe()
        hpu_infer_init_safe(model)

    correct = 0
    total = len(dataset)

    logger.info(f"[HellaSwag] Evaluating {total} samples on HPU (batch_size={batch_size})...")
    logger.info(f"  Progress updates every ~100 samples...")

    t0 = time.time()

    with torch.no_grad():
                            
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_samples = [dataset[i] for i in range(batch_start, batch_end)]
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

                                                                                          
            if (batch_end % 128 == 0 or batch_end == total):
                current_acc = correct / batch_end * 100.0
                elapsed = time.time() - t0
                samples_per_sec = batch_end / elapsed if elapsed > 0 else 0
                logger.info(f"  [{batch_end}/{total}] acc={current_acc:.2f}% | "
                           f"{samples_per_sec:.1f} samples/s | {elapsed:.1f}s elapsed")
                                                   
                import sys
                sys.stdout.flush()
                sys.stderr.flush()

    dt = time.time() - t0
    accuracy = (correct / total * 100.0) if total > 0 else 0.0

    model.to("cpu")
    free_hpu_memory()
    return accuracy, dt


def load_sensitivities(path: Path) -> Dict[str, float]:
    obj = read_json(path)
    if isinstance(obj, dict) and "sensitivities" in obj:
        return {k: float(v) for k, v in obj["sensitivities"].items()}
    return {k: float(v) for k, v in obj.items()}


def filter_candidates(sens: Dict[str, float], target_family: str) -> List[Tuple[str, float]]:
    if target_family == "mlp_only":
        suf = LLAMA_MLP_SUFFIXES
    elif target_family == "attn_output_only":
        suf = LLAMA_ATTN_O_SUFFIX
    elif target_family == "all_linear":
        suf = LLAMA_ALL_LINEAR_SUFFIXES
    else:
        raise ValueError(f"Unknown target_family: {target_family}")

    out = [(k, v) for (k, v) in sens.items() if any(k.endswith(s) for s in suf)]
    out.sort(key=lambda kv: kv[0])
    return out


def select_fp8_modules(
    candidates: List[Tuple[str, float]],
    clustering: str,
    seed: int,
    max_fp8_modules: int,
) -> Tuple[List[str], Dict]:
    names = [n for (n, _) in candidates]
    vals = np.array([v for (_, v) in candidates], dtype=np.float32)
    meta: Dict = {"selection_mode": "clustered", "clustering": clustering, "n_clusters": 2}

    if len(names) == 0:
        return [], {**meta, "error": "no candidates"}

    if clustering == "percentile":
        med = float(np.median(vals))
        bf16_mask = vals >= med
        fp8_mask = ~bf16_mask
        meta.update({"median_threshold": med})
    elif clustering == "kmeans":
        km = KMeans(n_clusters=2, random_state=seed, n_init=10)
        labels = km.fit_predict(vals.reshape(-1, 1))
        mean0 = float(vals[labels == 0].mean())
        mean1 = float(vals[labels == 1].mean())
        bf16_cluster = 0 if mean0 >= mean1 else 1
        bf16_mask = labels == bf16_cluster
        fp8_mask = ~bf16_mask
        meta.update({
            "labels": labels.tolist(),
            "kmeans_means": [mean0, mean1],
            "bf16_cluster": int(bf16_cluster),
        })
    else:
        raise ValueError(f"Unsupported clustering: {clustering}")

    fp8_names = [names[i] for i in range(len(names)) if fp8_mask[i]]

    if max_fp8_modules and max_fp8_modules > 0 and len(fp8_names) > max_fp8_modules:
        fp8_pairs = [(names[i], float(vals[i])) for i in range(len(names)) if fp8_mask[i]]
        fp8_pairs.sort(key=lambda kv: kv[1])                         
        fp8_names = [n for (n, _) in fp8_pairs[:max_fp8_modules]]
        meta["max_fp8_modules_applied"] = int(max_fp8_modules)

    meta["candidate_count"] = len(names)
    meta["fp8_count"] = len(fp8_names)
    meta["bf16_count"] = len(names) - len(fp8_names)
    return fp8_names, meta


def infer_num_layers(model_name_or_path: str) -> int:
    cfg = AutoConfig.from_pretrained(model_name_or_path)
    if hasattr(cfg, "num_hidden_layers") and cfg.num_hidden_layers is not None:
        return int(cfg.num_hidden_layers)
                                
    for k in ("n_layer", "num_layers"):
        if hasattr(cfg, k):
            return int(getattr(cfg, k))
    return 0


def build_layer_precision_maps(fp8_modules: List[str], num_layers: int) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    per_layer_mods: Dict[int, List[str]] = {i: [] for i in range(num_layers)}
    for name in fp8_modules:
        m = re.search(r"model\.layers\.(\d+)\.", name)
        if not m:
            continue
        idx = int(m.group(1))
        if 0 <= idx < num_layers:
            short = name.split(f"model.layers.{idx}.", 1)[1]
            per_layer_mods[idx].append(short)

    layer_precision: Dict[str, str] = {}
    layer_fp8_detail: Dict[str, List[str]] = {}
    for i in range(num_layers):
        mods = sorted(per_layer_mods[i])
        layer_fp8_detail[str(i)] = mods
        layer_precision[str(i)] = "FP8" if len(mods) > 0 else "BF16"
    return layer_precision, layer_fp8_detail


def tensor_bytes_from_state_dict(model: nn.Module) -> Tuple[int, Dict[str, int]]:
    sd = model.state_dict()
    total = 0
    by_dtype: Dict[str, int] = {}
    for _, t in sd.items():
        if not torch.is_tensor(t):
            continue
        b = t.numel() * t.element_size()
        total += b
        dt = str(t.dtype)
        by_dtype[dt] = by_dtype.get(dt, 0) + b
    return total, by_dtype


def mb(x_bytes: int) -> float:
    return float(x_bytes) / (1024.0 * 1024.0)


def get_module_by_name(model: nn.Module, qualname: str) -> nn.Module:
    cur = model
    for part in qualname.split("."):
        cur = getattr(cur, part)
    return cur


def estimate_mixed_precision_size_bytes(
    model_name_or_path: str,
    hp_dtype: str,
    fp8_modules: List[str],
) -> Dict[str, float]:
    """
    Estimate parameter footprint assuming:
      - baseline FP32 = 4 bytes/param for all params
      - mixed = hp_dtype bytes/param for all params EXCEPT:
          selected FP8 modules' *weights* stored as 1 byte/param
      - biases remain at hp_dtype size
    """
    hp_b = DTYPE_BYTES[hp_dtype]
    base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to("cpu")
    total_params = sum(p.numel() for p in base_model.parameters())

    fp8_weight_params = 0
    fp8_bias_params = 0
    for name in fp8_modules:
        try:
            mod = get_module_by_name(base_model, name)
        except Exception:
            continue
        if isinstance(mod, nn.Linear) and hasattr(mod, "weight") and mod.weight is not None:
            fp8_weight_params += int(mod.weight.numel())
            if mod.bias is not None:
                fp8_bias_params += int(mod.bias.numel())

                                                                       
    base_hp_bytes = total_params * hp_b
    mixed_bytes_est = base_hp_bytes - (fp8_weight_params * hp_b) + (fp8_weight_params * 1)

    fp32_bytes = total_params * 4

    del base_model
    return {
        "total_params": float(total_params),
        "fp32_model_size_mb_est": mb(fp32_bytes),
        "hp_dtype_model_size_mb_est": mb(base_hp_bytes),
        "mixed_model_size_mb_est": mb(mixed_bytes_est),
        "fp8_weight_params": float(fp8_weight_params),
        "fp8_bias_params": float(fp8_bias_params),
        "compression_ratio_fp32_to_mixed_est": float(fp32_bytes / mixed_bytes_est) if mixed_bytes_est > 0 else float("inf"),
    }


def build_measure_cfg(dump_stats_path: str, allowlist_names: List[str], blocklist_names: List[str],
                      observer: str, measure_exclude: str, fp8_format: str, hp_dtype: str,
                      use_qdq: bool, fake_quant: bool, scale_format: str) -> Dict:
    return {
        "mode": "MEASURE",
        "observer": observer,
        "dump_stats_path": dump_stats_path,
        "allowlist": {"types": [], "names": allowlist_names},
        "blocklist": {"types": [], "names": blocklist_names},
        "measure_exclude": measure_exclude,
        "fp8_config": fp8_format,
        "hp_dtype": hp_dtype,
        "use_qdq": bool(use_qdq),
        "fake_quant": bool(fake_quant),
        "scale_format": scale_format,
    }


def build_quantize_cfg(dump_stats_path: str, allowlist_names: List[str], blocklist_names: List[str],
                       observer: str, measure_exclude: str, fp8_format: str, hp_dtype: str,
                       scale_method: str, input_backoff: float, weight_backoff: float,
                       use_qdq: bool, fake_quant: bool, scale_format: str) -> Dict:
    return {
        "mode": "QUANTIZE",
        "observer": observer,
        "dump_stats_path": dump_stats_path,
        "allowlist": {"types": [], "names": allowlist_names},
        "blocklist": {"types": [], "names": blocklist_names},
        "measure_exclude": measure_exclude,
        "fp8_config": fp8_format,
        "hp_dtype": hp_dtype,
        "scale_method": scale_method,
        "scale_params": {"input_backoff": float(input_backoff), "weight_backoff": float(weight_backoff)},
        "use_qdq": bool(use_qdq),
        "fake_quant": bool(fake_quant),
        "scale_format": scale_format,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_name_or_path", type=str,
                    default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    ap.add_argument("--sensitivity_json", type=str, required=True)

    ap.add_argument("--target_family", type=str, default="mlp_only",
                    choices=["mlp_only", "attn_output_only", "all_linear"])
    ap.add_argument("--clustering", type=str, default="percentile", choices=["percentile", "kmeans"])
    ap.add_argument("--max_fp8_modules", type=int, default=0)

    ap.add_argument("--fp8_format", type=str, default="E4M3", choices=["E4M3", "E5M2"])
    ap.add_argument("--hp_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--baseline_dtype", type=str, default="fp32", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--scale_method", type=str, default="maxabs_hw_opt_weight",
                    choices=["maxabs_hw", "maxabs_hw_opt_weight", "ACT_MAXABS_POW2_WEIGHTS_PCS_OPT_POW2"])
    ap.add_argument("--input_backoff", type=float, default=0.25)
    ap.add_argument("--weight_backoff", type=float, default=0.50)
    ap.add_argument("--observer", type=str, default="maxabs")
    ap.add_argument("--measure_exclude", type=str, default="OUTPUT", choices=["OUTPUT", "NONE"])
    ap.add_argument("--scale_format", type=str, default="scalar", choices=["scalar", "const", "CONST"])
    ap.add_argument("--use_qdq", action="store_true")
    ap.add_argument("--fake_quant", action="store_true")

    ap.add_argument("--calib_split", type=str, default="train", choices=["train", "validation"])
    ap.add_argument("--eval_split", type=str, default="validation", choices=["train", "validation"])
    ap.add_argument("--calib_max_samples", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--use_mark_step", action="store_true")

    ap.add_argument("--run_mode", type=str, default="both", choices=["measure", "quantize", "both"])
    ap.add_argument("--run_dir", type=str, default="")
    ap.add_argument("--log_dir", type=str, default="logs_hellaswag_fp8")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reuse_existing_configs", action="store_true",
                    help="If quantize-only and configs exist in run_dir, reuse them exactly (recommended).")

    args = ap.parse_args()

    os.environ.setdefault("LOG_LEVEL_INC", "1")

    device = get_device()
    if device.type != "hpu":
        raise RuntimeError("No HPU detected. This script is intended for Gaudi.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path("hellaswag_fp8_runs") / f"hellaswag_fp8_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(Path(args.log_dir))

    try:
        dev_name = hthpu.get_device_name()
    except Exception:
        dev_name = "HPU"

    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Device: {dev_name} | LOG_LEVEL_INC={os.environ.get('LOG_LEVEL_INC')}")
    logger.info(f"Run mode: {args.run_mode}")

                                       
    sens_path = Path(args.sensitivity_json)
    sens = load_sensitivities(sens_path)
    candidates = filter_candidates(sens, args.target_family)

                                      
    num_layers = infer_num_layers(args.model_name_or_path)
    logger.info(f"Model: {args.model_name_or_path} | num_layers={num_layers} | target_family={args.target_family}")
    logger.info(f"Candidates: {len(candidates)} (filtered submodules)")

           
    dump_stats_path = str((run_dir / "inc_output" / "measure").as_posix())
    measure_cfg_path = run_dir / "fp8_measure.json"
    quant_cfg_path = run_dir / "fp8_quantize.json"
    measure_meta_path = run_dir / "measure_summary.json"
    selection_meta_path = run_dir / "selection_meta.json"

                                                                                               
    fp8_modules: List[str] = []
    blocklist_names = list(DEFAULT_BLOCKLIST)
    selection_meta: Dict = {}

    if args.run_mode == "quantize" and args.reuse_existing_configs and measure_cfg_path.exists():
        logger.info("Reusing existing MEASURE config/selection from run_dir for consistency.")
        measure_cfg = read_json(measure_cfg_path)
        fp8_modules = list(measure_cfg.get("allowlist", {}).get("names", []))
        blocklist_names = list(measure_cfg.get("blocklist", {}).get("names", DEFAULT_BLOCKLIST))
        dump_stats_path = str(measure_cfg.get("dump_stats_path", dump_stats_path))
        selection_meta = read_json(selection_meta_path) if selection_meta_path.exists() else {}
    else:
        fp8_modules, selection_meta = select_fp8_modules(
            candidates=candidates,
            clustering=args.clustering,
            seed=args.seed,
            max_fp8_modules=args.max_fp8_modules,
        )

                                      
    layer_precision_map, layer_fp8_detail = build_layer_precision_maps(fp8_modules, num_layers)

    fp8_layers = [int(k) for k, v in layer_precision_map.items() if v == "FP8"]
    bf16_layers = [int(k) for k, v in layer_precision_map.items() if v == "BF16"]

    logger.info(f"Selection: clustering={args.clustering} | max_fp8_modules={args.max_fp8_modules}")
    if "median_threshold" in selection_meta:
        logger.info(f"Percentile split: median_threshold={selection_meta['median_threshold']:.6f}")
    if "kmeans_means" in selection_meta:
        logger.info(f"KMeans means={selection_meta['kmeans_means']} | bf16_cluster={selection_meta.get('bf16_cluster')}")

    logger.info(f"FP8 modules selected: {len(fp8_modules)}")
    logger.info(f"FP8 layers (any selected submodule): {len(fp8_layers)} -> {fp8_layers}")
    logger.info(f"BF16 layers (no selected submodule): {len(bf16_layers)} -> {bf16_layers}")

                                            
    for i in range(num_layers):
        mods = layer_fp8_detail.get(str(i), [])
        if mods:
            logger.info(f"Layer {i:02d}: FP8 submodules = {mods}")
        else:
            logger.info(f"Layer {i:02d}: BF16 (no FP8 submodules)")

                         
    write_json(selection_meta_path, selection_meta)

                                 
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading HellaSwag dataset: calib_split={args.calib_split}, eval_split={args.eval_split}")
    calib_dataset = load_hellaswag_dataset(args.calib_split)
    eval_dataset = load_hellaswag_dataset(args.eval_split)

    calib_samples = min(args.calib_max_samples, len(calib_dataset))
    eval_samples = len(eval_dataset)

    logger.info(f"Calibration samples: {calib_samples} (from {args.calib_split} split)")
    logger.info(f"Evaluation samples: {eval_samples} (from {args.eval_split} split)")

                                                                      
    size_est = estimate_mixed_precision_size_bytes(args.model_name_or_path, args.hp_dtype, fp8_modules)
    logger.info(f"Size estimate (FP32 baseline): {size_est['fp32_model_size_mb_est']:.2f} MB")
    logger.info(f"Size estimate (all {args.hp_dtype}): {size_est['hp_dtype_model_size_mb_est']:.2f} MB")
    logger.info(f"Size estimate (mixed {args.hp_dtype}+FP8 weights): {size_est['mixed_model_size_mb_est']:.2f} MB")
    logger.info(f"Estimated compression ratio (FP32 -> mixed): {size_est['compression_ratio_fp32_to_mixed_est']:.3f}x")

                                                               
                                                                   
    acc_baseline = None
    t_baseline = None
    throughput_baseline = None
    base_sd_bytes = None
    base_sd_by_dtype = None

    if args.run_mode in ("quantize", "both"):
        logger.info(f"=== Baseline Evaluation ({args.baseline_dtype}) ===")
        base_model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        base_model = cast_model_dtype(base_model, args.baseline_dtype)
        base_sd_bytes, base_sd_by_dtype = tensor_bytes_from_state_dict(base_model)

        acc_baseline, t_baseline = evaluate_hellaswag_accuracy(
            base_model, eval_dataset, tokenizer, device, args.batch_size, args.use_mark_step, logger
        )

        throughput_baseline = eval_samples / t_baseline if t_baseline > 0 else 0.0
        logger.info(f"[Baseline {args.baseline_dtype}] accuracy={acc_baseline:.2f}% time={t_baseline:.2f}s "
                    f"throughput={throughput_baseline:.2f} samples/s")
        logger.info(f"[Baseline state_dict] total={mb(base_sd_bytes):.2f} MB | by_dtype={ {k: round(mb(v),2) for k,v in base_sd_by_dtype.items()} }")

        del base_model
        free_hpu_memory()
    else:
        logger.info("=== Skipping Baseline Evaluation (MEASURE phase only does calibration) ===")

                              
    measure_cfg = build_measure_cfg(
        dump_stats_path=dump_stats_path,
        allowlist_names=fp8_modules,
        blocklist_names=blocklist_names,
        observer=args.observer,
        measure_exclude=args.measure_exclude,
        fp8_format=args.fp8_format,
        hp_dtype=args.hp_dtype,
        use_qdq=args.use_qdq,
        fake_quant=args.fake_quant,
        scale_format=args.scale_format,
    )
    quant_cfg = build_quantize_cfg(
        dump_stats_path=dump_stats_path,
        allowlist_names=fp8_modules,
        blocklist_names=blocklist_names,
        observer=args.observer,
        measure_exclude=args.measure_exclude,
        fp8_format=args.fp8_format,
        hp_dtype=args.hp_dtype,
        scale_method=args.scale_method,
        input_backoff=args.input_backoff,
        weight_backoff=args.weight_backoff,
        use_qdq=args.use_qdq,
        fake_quant=args.fake_quant,
        scale_format=args.scale_format,
    )

                                                   
    write_json(measure_cfg_path, measure_cfg)
    write_json(quant_cfg_path, quant_cfg)

    calib_time = None
    acc_fp8 = None
    t_fp8 = None
    quant_sd_bytes = None
    quant_sd_by_dtype = None

    if args.run_mode in ("measure", "both"):
        logger.info("=== MEASURE phase (prepare + forward-only calibration) ===")
        model_m = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        model_m = cast_model_dtype(model_m, args.hp_dtype)
        cfg_m = FP8Config.from_json_file(str(measure_cfg_path))
        model_m = prepare(model_m, cfg_m)

        calib_time = calibration_forward_hellaswag(
            model_m, calib_dataset, tokenizer, device, args.batch_size, args.use_mark_step, args.calib_max_samples
        )
        finalize_calibration(model_m)
        logger.info(f"[MEASURE] calibration_time_s={calib_time:.3f} dump_stats_path={dump_stats_path}")

                                  
        write_json(measure_meta_path, {
            "timestamp": ts,
            "calib_time_s": float(calib_time),
            "calib_samples": int(calib_samples),
            "batch_size": int(args.batch_size),
            "dump_stats_path": dump_stats_path,
        })

        del model_m
        free_hpu_memory()

    if args.run_mode in ("quantize", "both"):
        logger.info("=== QUANTIZE phase (convert + eval) ===")
        os.environ.setdefault("PT_HPU_WEIGHT_SHARING", "0")
        logger.info(f"PT_HPU_WEIGHT_SHARING={os.environ.get('PT_HPU_WEIGHT_SHARING')}")

                                                              
        if calib_time is None and measure_meta_path.exists():
            try:
                mmeta = read_json(measure_meta_path)
                calib_time = float(mmeta.get("calib_time_s"))
                logger.info(f"Loaded MEASURE metadata: calib_time_s={calib_time:.3f}")
            except Exception:
                pass

        model_q = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        model_q = cast_model_dtype(model_q, args.hp_dtype)

        cfg_q = FP8Config.from_json_file(str(quant_cfg_path))
        model_q = convert(model_q, cfg_q)

                                                            
        quant_sd_bytes, quant_sd_by_dtype = tensor_bytes_from_state_dict(model_q)

        acc_fp8, t_fp8 = evaluate_hellaswag_accuracy(
            model_q, eval_dataset, tokenizer, device, args.batch_size, args.use_mark_step, logger,
            use_hpu_init=True                                                     
        )

        throughput_fp8 = eval_samples / t_fp8 if t_fp8 > 0 else 0.0
        logger.info(f"[FP8 {args.fp8_format}] accuracy={acc_fp8:.2f}% time={t_fp8:.2f}s "
                    f"throughput={throughput_fp8:.2f} samples/s")
        logger.info(f"[FP8 state_dict] total={mb(quant_sd_bytes):.2f} MB | by_dtype={ {k: round(mb(v),2) for k,v in quant_sd_by_dtype.items()} }")

        del model_q
        free_hpu_memory()

                   
    acc_delta = (acc_fp8 - acc_baseline) if (acc_fp8 is not None) else None
    acc_delta_pct = (acc_delta / acc_baseline * 100.0) if (acc_delta is not None and acc_baseline > 0) else None

                  
    summary = {
        "timestamp": ts,
        "run_dir": str(run_dir),
        "model_name_or_path": args.model_name_or_path,
        "sensitivity_json": str(sens_path),
        "target_family": args.target_family,
        "clustering": args.clustering,
        "n_clusters": 2,
        "max_fp8_modules": args.max_fp8_modules,
        "fp8_modules": fp8_modules,
        "blocklist_names": blocklist_names,
        "layer_precision_map": layer_precision_map,
        "layer_fp8_submodules": layer_fp8_detail,
        "fp8_layers": fp8_layers,
        "bf16_layers": bf16_layers,
        "dump_stats_path": dump_stats_path,
        "measure_config_path": str(measure_cfg_path),
        "quant_config_path": str(quant_cfg_path),
        "fp8_format": args.fp8_format,
        "hp_dtype": args.hp_dtype,
        "baseline_dtype": args.baseline_dtype,
        "scale_method": args.scale_method,
        "scale_params": {"input_backoff": args.input_backoff, "weight_backoff": args.weight_backoff},
        "observer": args.observer,
        "measure_exclude": args.measure_exclude,
        "scale_format": args.scale_format,
        "use_qdq": args.use_qdq,
        "fake_quant": args.fake_quant,
        "calib_split": args.calib_split,
        "eval_split": args.eval_split,
        "calib_samples": int(calib_samples),
        "eval_samples": int(eval_samples),
        "batch_size": args.batch_size,
        "acc_baseline": float(acc_baseline) if acc_baseline is not None else None,
        "acc_fp8": float(acc_fp8) if acc_fp8 is not None else None,
        "acc_delta": float(acc_delta) if acc_delta is not None else None,
        "acc_delta_pct": float(acc_delta_pct) if acc_delta_pct is not None else None,
        "eval_time_baseline_s": float(t_baseline) if t_baseline is not None else None,
        "eval_time_fp8_s": float(t_fp8) if t_fp8 is not None else None,
        "throughput_baseline_samp_s": float(eval_samples / t_baseline) if t_baseline is not None and t_baseline > 0 else None,
        "throughput_fp8_samp_s": float(eval_samples / t_fp8) if t_fp8 is not None and t_fp8 > 0 else None,
        "calibration_time_s": float(calib_time) if calib_time is not None else None,
        "size_estimates": size_est,
        "baseline_state_dict_mb": float(mb(base_sd_bytes)) if base_sd_bytes is not None else None,
        "fp8_state_dict_mb": float(mb(quant_sd_bytes)) if quant_sd_bytes is not None else None,
        "warning": "Verify patched modules via LOG_LEVEL_INC=1 output: search for 'Patched modules'."
    }
    write_json(run_dir / "run_summary.json", summary)

                                          
    logger.info("=== SUMMARY ===")
    if acc_baseline is not None:
        logger.info(f"Baseline ({args.baseline_dtype}) Accuracy: {acc_baseline:.2f}%")
    if acc_fp8 is not None:
        logger.info(f"FP8 ({args.fp8_format}) Accuracy:        {acc_fp8:.2f}%")
        if acc_baseline is not None:
            logger.info(f"Delta: {acc_delta:+.2f}% ({acc_delta_pct:+.3f}%)")
    if t_baseline is not None and t_fp8 is not None:
        logger.info(f"Throughput baseline: {eval_samples/t_baseline:.2f} samples/s | FP8: {eval_samples/t_fp8:.2f} samples/s")
    if calib_time is not None:
        logger.info(f"Calibration time: {calib_time:.3f}s")
    logger.info(f"Est. compression FP32->mixed: {size_est['compression_ratio_fp32_to_mixed_est']:.3f}x")
    logger.info(f"Saved: {run_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()


                                
                        
                                                                                                                                                                                                                                                                                                                                                                                                                                          

                                 
                        
                                
                                                                                                                                                                                                                                                                                                                                                                                                                                                       
