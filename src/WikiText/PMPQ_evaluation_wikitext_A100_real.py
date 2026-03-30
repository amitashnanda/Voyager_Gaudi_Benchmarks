# Phase_2_evaluate_GroupWise_TinyLlama_WikiText_REAL.py
"""
Phase 2: PMPQ Group-Wise REAL Quantization & Evaluation - TinyLlama on WikiText-2
==================================================================================
QUANTIZATION TYPE: REAL
  FP16  -> weights stored as torch.float16  -- 2x real memory savings
  INT8  -> weights stored as torch.int8     -- 4x real memory savings
           + per-group FP16 scale (group-wise symmetric, GPTQ-style)
           INT8 is storage only; on forward pass, int8 is dequantized to FP16
           before matmul (weight-only quantization, same as GPTQ/bitsandbytes).

PMPQ -- Pruning-based Mixed-Precision Quantization
  Phase 1 sensitivity = normalized Frobenius norm of each layer's weights.
  Higher sensitivity -> assigned FP16 (higher precision, 2x savings).
  Lower sensitivity  -> assigned INT8 (more aggressive, 4x savings).

Evaluation Dataset:
  WikiText-2 TEST split (not train).
  Metric: Perplexity (lower is better).
  Additional metrics: tokens/s throughput, model size before/after.

Baseline Caching:
  FP32 baseline read from Models/baseline_wikitext_fp32.json
  (written by FAKE quant script or this script on first run).
  If skip selected and cache exists, loaded for comparison.

Model Saving (.bin HuggingFace format):
  Saved to Models/real_quant_TinyLlama_WikiText_<bits>_gs<gs>_<strategy>_PMPQ/
  Weights in .bin are dequantized to FP32 (HuggingFace standard).
  quant_config.json records FP16/INT8 layer assignment for re-quantization.

Speed:
  autocast(dtype=torch.float16) during evaluation.
  Quantization on CPU then model moves to GPU (avoids OOM).
  DataParallel applied AFTER quantization (required order).

Author: Mixed-Precision Quantization Team
Date: 2025-2026
"""

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
import os

HF_HOME = os.environ.get("HF_HOME", "/pscratch/sd/s/sreeb12/.cache/huggingface")
os.makedirs(HF_HOME, exist_ok=True)

os.environ.update({
    "HF_HOME":              HF_HOME,
    "HF_DATASETS_CACHE":    os.path.join(HF_HOME, "datasets"),
    "HF_HUB_CACHE":         os.path.join(HF_HOME, "hub"),
    "TOKENIZERS_PARALLELISM": "false",
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
})

print("Environment setup - cache:", HF_HOME)

# ============================================================================
# IMPORTS
# ============================================================================
import json, time, random, warnings, re
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from sklearn.cluster import KMeans, AgglomerativeClustering

warnings.filterwarnings('ignore')

print("All imports complete")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  |  {props.total_memory / 1024**3:.1f} GB")

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_GROUP_SIZE  = 128
SEQUENCE_LENGTH     = 512
EVAL_SPLIT          = "test"       # Phase 2 always uses test split
BASELINE_CACHE_DIR  = "Models"
BASELINE_CACHE_FILE = os.path.join(BASELINE_CACHE_DIR,
                                    "baseline_wikitext_fp32.json")

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
MODEL_KEY  = "TinyLlama-1.1B"

# Only FP16 and INT8 supported in pure PyTorch on A100:
#   bits >= 16  ->  FP16 (torch.float16)
#   bits <  16  ->  INT8 (torch.int8 + group-wise FP16 scales)
#   INT8 stores weights as int8; on forward pass dequantized to FP16 for matmul


# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device():
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def snap_bits(b: int) -> int:
    return 16 if b >= 16 else 8


def prompt_user(prompt_text, options, default=None):
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options, 1):
        marker = "-> " if opt == default else "   "
        print(f"{marker}{i}. {opt}")
    while True:
        try:
            raw = input(
                f"Enter choice (1-{len(options)}) "
                f"[{options.index(default)+1 if default else 1}]: "
            ).strip()
            if not raw and default: return default
            idx = int(raw) - 1
            if 0 <= idx < len(options): return options[idx]
            print(f"Enter 1-{len(options)}")
        except ValueError:
            print("Invalid input.")


def prompt_yes_no(prompt_text, default="yes"):
    d = "[Y/n]" if default == "yes" else "[y/N]"
    while True:
        inp = input(f"\n{prompt_text} {d}: ").strip().lower()
        if not inp:             return default == "yes"
        if inp in ("y","yes"):  return True
        if inp in ("n","no"):   return False
        print("Enter 'y' or 'n'")


def print_section(title):
    print(f"\n{'='*80}\n  {title}\n{'='*80}")


def format_duration(s):
    if s < 60:    return f"{s:.2f}s"
    if s < 3600:  return f"{int(s//60)}m {s%60:.2f}s"
    h = int(s//3600); m = int((s%3600)//60)
    return f"{h}h {m}m {s%60:.2f}s"


def get_model_size_mb(model):
    """Actual GPU/CPU memory footprint -- parameters + buffers."""
    total  = sum(p.nelement() * p.element_size() for p in model.parameters())
    total += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total / (1024 * 1024)


def prepare_wikitext_dataset(tokenizer, split="test", block_size=512):
    """
    Prepare WikiText-2 dataset with continuous tokenization and chunking.
    Matches FAKE quantization and HPU implementation exactly.

    Args:
        tokenizer: HuggingFace tokenizer
        split: "train", "validation", or "test"
        block_size: sequence length for chunks (default 512)

    Returns:
        Dataset with input_ids and labels
    """
    try:
        from itertools import chain
    except ImportError:
        pass

    try:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception:
        dataset = load_dataset("wikitext", "wikitext-2-v1", split=split)

    # Tokenize all text
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing {split}"
    )

    # Concatenate all texts and chunk into fixed-size blocks
    def group_texts(examples):
        # Concatenate all texts
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])

        # Drop the last chunk if it's smaller than block_size
        total_length = (total_length // block_size) * block_size

        # Split into chunks of block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result

    chunked = tokenized.map(
        group_texts,
        batched=True,
        desc=f"Chunking {split} into {block_size}-token blocks"
    )

    return chunked


# ============================================================================
# BASELINE CACHE
# ============================================================================

def load_baseline_cache():
    if not os.path.exists(BASELINE_CACHE_FILE):
        return None
    with open(BASELINE_CACHE_FILE, "r") as f:
        return json.load(f)


def save_baseline_cache(ppl, total_tokens, n_samples, eval_time, throughput,
                        device_str, num_gpus, timestamp):
    os.makedirs(BASELINE_CACHE_DIR, exist_ok=True)
    data = {
        "perplexity":    ppl,
        "total_tokens":  total_tokens,
        "n_samples":     n_samples,
        "eval_time_s":   eval_time,
        "throughput":    throughput,
        "model_key":     MODEL_KEY,
        "model_name":    MODEL_NAME,
        "dataset":       "WikiText-2",
        "split":         EVAL_SPLIT,
        "seqlen":        SEQUENCE_LENGTH,
        "device":        device_str,
        "num_gpus":      num_gpus,
        "quantization":  "FP32_baseline",
        "timestamp":     timestamp,
        "note":          "FP32 baseline stored for future skip-baseline comparisons"
    }
    with open(BASELINE_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Baseline results saved to: {BASELINE_CACHE_FILE}")


# ============================================================================
# CLUSTERING
# ============================================================================

def kmeans_clustering(sens, n_clusters=3):
    v = sens.reshape(-1, 1)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(v)
    cm = [(c, float(v[labels==c].mean())) for c in range(n_clusters)]
    cm.sort(key=lambda x: x[1], reverse=True)
    return labels, cm


def hierarchical_clustering(sens, n_clusters=3):
    v = sens.reshape(-1, 1)
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg.fit_predict(v)
    cm = [(c, float(v[labels==c].mean())) for c in range(n_clusters)]
    cm.sort(key=lambda x: x[1], reverse=True)
    return labels, cm


def percentile_clustering(sens, n_clusters=3):
    n = len(sens)
    pairs = sorted(enumerate(sens), key=lambda x: x[1], reverse=True)
    cs = n // n_clusters
    labels = np.zeros(n, dtype=int); cm = []
    for cid in range(n_clusters):
        s = cid * cs; e = s + cs if cid < n_clusters - 1 else n
        idxs = [pairs[i][0] for i in range(s, e)]
        for i in idxs: labels[i] = cid
        cm.append((cid, float(np.mean([sens[i] for i in idxs]))))
    cm.sort(key=lambda x: x[1], reverse=True)
    return labels, cm


# ============================================================================
# REAL QUANTIZATION -- FP16 and INT8 ONLY
# ============================================================================

class RealQuantizedLinearGroupWise(nn.Module):
    """
    REAL group-wise quantized linear layer.

    FP16 mode (nbits >= 16):
      Weight stored as torch.float16 -- 2x memory vs FP32.
      A100 Tensor Cores run FP16 matmuls natively.

    INT8 mode (nbits < 16):
      Weight stored as torch.int8 -- 4x memory vs FP32.
      Per-group FP16 scale (group_size weights -> 1 FP16 scale).
      Forward: INT8 weight dequantized to FP16 on-the-fly, then FP16 matmul.
      This is weight-only quantization -- same mechanism as GPTQ/bitsandbytes.
      INT8 is storage only; computation runs in FP16 on A100 Tensor Cores.
    """

    def __init__(self, linear: nn.Linear, nbits: int, group_size: int = 128):
        super().__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.nbits        = int(nbits)
        self.group_size   = group_size

        w    = linear.weight.detach().cpu().float()
        bias = linear.bias.detach().cpu().float() if linear.bias is not None else None

        if nbits >= 16:
            self.register_buffer('weight_fp16', w.half())
            self.register_buffer('bias_buf',
                                 bias.half() if bias is not None else None)
            self._mode = 'fp16'
        else:
            out_f, in_f = w.shape
            pad = (-in_f) % group_size
            if pad:
                w = torch.nn.functional.pad(w, (0, pad))
            in_f_padded = w.shape[1]
            num_groups  = in_f_padded // group_size
            wg          = w.view(out_f, num_groups, group_size)
            w_max       = wg.abs().amax(dim=-1, keepdim=True)
            scales      = (w_max / 127.0).clamp(min=1e-8)
            w_int8      = torch.round(wg / scales).clamp(-127, 127)
            w_int8      = w_int8.view(out_f, in_f_padded)[:, :in_f]
            self.register_buffer('weight_int8',  w_int8.to(torch.int8))
            self.register_buffer('scales_fp16',  scales.squeeze(-1).half())
            self.register_buffer('bias_buf',
                                 bias.half() if bias is not None else None)
            self._mode       = 'int8'
            self.in_f_padded = in_f_padded
            self.num_groups  = num_groups

    def _dequantize(self):
        gs    = self.group_size
        in_f  = self.in_features
        ngrps = (in_f + gs - 1) // gs
        sc    = self.scales_fp16[:, :ngrps].repeat_interleave(gs, dim=1)[:, :in_f]
        return self.weight_int8.half() * sc  # FP16 result

    def forward(self, x):
        if self._mode == 'fp16':
            w = self.weight_fp16.to(device=x.device, dtype=x.dtype)
        else:
            # Dequantize INT8 -> FP16 on the fly, then FP16 matmul
            w = self._dequantize().to(device=x.device, dtype=x.dtype)
        b = (self.bias_buf.to(device=x.device, dtype=x.dtype)
             if self.bias_buf is not None else None)
        return nn.functional.linear(x, w, b)

    def calculate_weight_bits(self):
        nw = self.in_features * self.out_features
        if self._mode == 'fp16':
            return nw * 32, nw * 16
        else:
            ngrps = self.out_features * (
                (self.in_features + self.group_size - 1) // self.group_size
            )
            return nw * 32, nw * 8 + ngrps * 16

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"mode={self._mode}, group_size={self.group_size}")


def _set_module(root, qualname, new_mod):
    parts = qualname.split(".")
    parent = root
    for p in parts[:-1]: parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def quantize_model_real(model, layer_bits_map, group_size=128,
                        is_data_parallel=False):
    actual = model.module if is_data_parallel else model
    targets = []
    for name, mod in actual.named_modules():
        if isinstance(mod, nn.Linear):
            m = re.search(r"(?:model\.|decoder\.)layers\.(\d+)\.", name)
            if m:
                layer_idx = int(m.group(1))
                targets.append((name, mod, layer_bits_map.get(layer_idx, 8)))

    fp16_c = sum(1 for _, _, b in targets if b >= 16)
    int8_c  = sum(1 for _, _, b in targets if b <  16)
    print(f"  Found {len(targets)} Linear layers to quantize (REAL storage)")
    print(f"  FP16 layers: {fp16_c}  (weights -> torch.float16, 2x savings)")
    print(f"  INT8 layers: {int8_c}  (weights -> torch.int8 storage, "
          f"dequant to FP16 on forward, 4x savings)")
    print(f"  Group size : {group_size}")

    total_orig = total_quant = 0
    for qualname, lm, nbits in tqdm(targets, desc="  Quantizing layers"):
        wrapper = RealQuantizedLinearGroupWise(lm, nbits, group_size)
        _set_module(actual, qualname, wrapper)
        o, q = wrapper.calculate_weight_bits()
        total_orig += o; total_quant += q
    return model, total_orig, total_quant


# ============================================================================
# WIKITEXT-2 PERPLEXITY EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_perplexity_wikitext(model, eval_dataset, device, eval_name="Perplexity", use_fp16=False, batch_size=4):
    """
    Evaluate perplexity using pre-prepared dataset.

    Args:
        model: nn.Module (can be wrapped in DataParallel)
        eval_dataset: HuggingFace dataset with input_ids and labels
        device: torch.device
        eval_name: label for progress bar
        use_fp16: If True, use FP16 autocast (for quantized model with real FP16/INT8 weights).
                  If False, use FP32 (for baseline FP32 model to get true FP32 throughput).
        batch_size: number of samples per batch (matches HPU per_device_batch_size)

    Returns:
        perplexity, eval_time_s, total_tokens, throughput (tokens/s), n_samples
    """
    n_samples = len(eval_dataset)
    seqlen = len(eval_dataset[0]["input_ids"])
    total_tokens = n_samples * seqlen

    precision_str = "FP16 autocast" if use_fp16 else "FP32"
    print(f"  Split: test  |  Samples: {n_samples}  |  Seqlen: {seqlen}  |  Total tokens: {total_tokens:,}  |  Precision: {precision_str}  |  Batch size: {batch_size}")

    model.eval()
    total_loss = 0.0
    loss_fct = nn.CrossEntropyLoss()
    start_time = time.time()

    # Process in batches
    num_batches = (n_samples + batch_size - 1) // batch_size

    with tqdm(total=num_batches, desc=eval_name, leave=False) as pbar:
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_samples = eval_dataset[batch_start:batch_end]

            # HuggingFace dataset slicing returns dict with lists, not list of dicts
            input_ids = torch.tensor(batch_samples["input_ids"]).to(device)
            labels = torch.tensor(batch_samples["labels"]).to(device)

            if use_fp16:
                # FP16 autocast for REAL quantization (FP16/INT8 weights)
                with autocast(dtype=torch.float16):
                    outputs = model(input_ids)
            else:
                # True FP32 for baseline (no autocast) - matches FAKE script
                outputs = model(input_ids)

            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Accumulate loss weighted by batch tokens
            batch_tokens = input_ids.size(0) * input_ids.size(1)
            total_loss += loss.item() * batch_tokens

            pbar.update(1)

    eval_time = time.time() - start_time
    ppl = np.exp(total_loss / total_tokens)
    throughput = total_tokens / eval_time

    return ppl, eval_time, total_tokens, throughput, n_samples


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print_section(
        "PHASE 2: PMPQ WikiText-2 Evaluation -- REAL QUANTIZATION (FP16 / INT8)")
    print("""
  REAL QUANTIZATION -- A100 native (pure PyTorch, no external libraries):
    FP16  ->  torch.float16 weights  -- 2x real memory savings
    INT8  ->  torch.int8 weights     -- 4x real memory savings
              INT8 is storage format only.
              On each forward pass: int8 weights dequantized to FP16,
              then FP16 matmul on A100 Tensor Cores.
              (weight-only quantization -- same as GPTQ/bitsandbytes)

  Bit options: only 16 (FP16) and 8 (INT8).
  Any other value snapped: >=16 -> FP16, <16 -> INT8.

  Evaluation: WikiText-2 TEST split (perplexity + tokens/s throughput).
  Baseline: loaded from Models/baseline_wikitext_fp32.json if skip selected.
  Model:    saved to Models/<n>/ in HuggingFace .bin sharded format.
    """)

    timing_log          = {}
    pipeline_start_time = time.time()

    set_seed(42)
    device   = pick_device()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"  Device: {device}  |  GPUs: {num_gpus}")

    # ==========================================================================
    # STEP 1: Load Sensitivity File
    # ==========================================================================
    print_section("STEP 1: LOAD SENSITIVITY FILE")
    t0 = time.time()

    sens_dir = Path("Sensitivities")
    if not sens_dir.exists():
        print("'Sensitivities' folder not found. Run Phase 1 first."); return
    sens_files = list(sens_dir.glob("sens_*.json"))
    if not sens_files:
        print("No sensitivity files found."); return

    names = [f.name for f in sens_files]
    print(f"Found {len(names)} sensitivity file(s):")
    for i, n in enumerate(names, 1): print(f"  {i}. {n}")

    selected = prompt_user("Select sensitivity file:", names, default=names[0])
    with open(sens_dir / selected, "r") as f:
        sd = json.load(f)

    if "sensitivities" in sd:
        sens    = sd["sensitivities"]
        meta    = sd.get("metadata", {})
        p1_gs   = meta.get("group_size", DEFAULT_GROUP_SIZE)
        p1_time = sd.get("timing", {}).get("sensitivity_computation_time_s")
        p1_ppl  = meta.get("fp32_perplexity_train")
    else:
        sens = sd; p1_gs = DEFAULT_GROUP_SIZE; p1_time = None; p1_ppl = None

    num_layers = max(int(k.split("_")[1]) for k in sens) + 1
    sv = np.array([sens[f"layer_{i}"] for i in range(num_layers)], dtype=np.float32)

    timing_log["sensitivity_file_loading_time_s"] = time.time() - t0
    print(f"\n  Loaded: {selected}  "
          f"({format_duration(timing_log['sensitivity_file_loading_time_s'])})")
    print(f"  Layers: {num_layers}  |  "
          f"Sensitivity: [{sv.min():.6f}, {sv.max():.6f}]")
    if p1_time:
        print(f"  Phase 1 sensitivity time: {format_duration(p1_time)}")
    if p1_ppl:
        print(f"  Phase 1 FP32 perplexity (train): {p1_ppl:.4f}")

    sf_lower = selected.lower()
    sensitivity_method = (
        "PMPQ"  if "pmpq"  in sf_lower else
        "PWCCA" if "pwcca" in sf_lower else
        "SVCCA" if "svcca" in sf_lower else
        "CKA"   if "cka"   in sf_lower else
        "CMPQ"
    )

    # ==========================================================================
    # STEP 2: Load Model
    # ==========================================================================
    print_section("STEP 2: LOADING MODEL")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32,
        device_map=None, low_cpu_mem_usage=True
    )

    timing_log["model_loading_time_s"] = time.time() - t0
    fp32_size_mb = get_model_size_mb(model)
    print(f"  Model loaded in {format_duration(timing_log['model_loading_time_s'])}")
    print(f"  FP32 model size (CPU): {fp32_size_mb:.2f} MB")

    # ==========================================================================
    # STEP 2b: Prepare Dataset
    # ==========================================================================
    print_section("STEP 2b: PREPARING DATASET")
    eval_dataset = prepare_wikitext_dataset(tokenizer, split=EVAL_SPLIT, block_size=SEQUENCE_LENGTH)
    num_samples = len(eval_dataset)
    total_tokens = num_samples * SEQUENCE_LENGTH
    print(f"WikiText-2 {EVAL_SPLIT}: {num_samples} samples × {SEQUENCE_LENGTH} tokens")
    print(f"Total tokens: {total_tokens:,}")

    # ==========================================================================
    # STEP 3: Clustering
    # ==========================================================================
    print_section("STEP 3: CLUSTERING CONFIGURATION")
    t0 = time.time()

    cc = prompt_user(
        "Select clustering strategy:",
        ["K-means (recommended)", "Percentile bucketing", "Hierarchical clustering"],
        default="K-means (recommended)"
    )
    ng = prompt_user(
        "Select number of groups:",
        ["3 groups (simpler)", "4 groups (finer control)"],
        default="3 groups (simpler)"
    )
    n_clusters = int(ng.split()[0])

    if "K-means" in cc:
        labels, cm = kmeans_clustering(sv, n_clusters); sn = "kmeans"
    elif "Percentile" in cc:
        labels, cm = percentile_clustering(sv, n_clusters); sn = "percentile"
    else:
        labels, cm = hierarchical_clustering(sv, n_clusters); sn = "hierarchical"

    timing_log["clustering_time_s"] = time.time() - t0
    print(f"\n  Clustering ({sn}) in {format_duration(timing_log['clustering_time_s'])}")
    for cid, cmean in cm:
        idxs = [i for i in range(num_layers) if labels[i] == cid]
        print(f"  Cluster {cid}: {len(idxs)} layers (mean sensitivity: {cmean:.6f})")

    # ==========================================================================
    # STEP 4: Bit-Width Allocation -- FP16 and INT8 ONLY
    # ==========================================================================
    print_section("STEP 4: BIT-WIDTH ALLOCATION")
    print("  REAL QUANTIZATION supports only FP16 and INT8 (pure PyTorch on A100).")
    print("  FP16 (16-bit) -> torch.float16 storage -- 2x memory savings")
    print("  INT8  (8-bit) -> torch.int8 storage    -- 4x memory savings")
    print("                   dequantized to FP16 on-the-fly for each forward pass")
    print("  Any other value snapped: >=16 -> FP16, <16 -> INT8")

    if n_clusters == 3:
        bit_options = [
            "[16, 8, 8]  -- recommended (FP16 sensitive, INT8 rest)",
            "[16, 16, 8] -- conservative (FP16 top-2, INT8 low)",
            "[8,  8, 8]  -- full INT8 (maximum compression)",
            "Custom (only 16 or 8 accepted)"
        ]
        bit_choice = prompt_user(
            "Select bit allocation (high sensitivity -> low sensitivity):",
            bit_options, default=bit_options[0]
        )
        preset_map = {
            "16, 8, 8":  ([16, 8, 8],  "16-8-8"),
            "16, 16, 8": ([16, 16, 8], "16-16-8"),
            "8,  8, 8":  ([8,  8, 8],  "8-8-8"),
        }
        matched = next((v for k, v in preset_map.items() if k in bit_choice), None)
        if matched:
            cluster_bits, alloc_name = matched
        else:
            raw = input("Enter 3 values comma-separated (e.g. 16,8,8): ").strip()
            cluster_bits = [snap_bits(int(b.strip())) for b in raw.split(",")]
            alloc_name = "custom"
    else:
        bit_options = [
            "[16, 8, 8, 8]  -- recommended",
            "[16, 16, 8, 8] -- conservative",
            "[8,  8, 8, 8]  -- full INT8 (max compression)",
            "[16, 16, 16, 8]-- very conservative",
            "Custom (only 16 or 8 accepted)"
        ]
        bit_choice = prompt_user(
            "Select bit allocation (high sensitivity -> low sensitivity):",
            bit_options, default=bit_options[0]
        )
        preset_map4 = {
            "16, 8, 8, 8":   ([16, 8, 8, 8],   "16-8-8-8"),
            "16, 16, 8, 8":  ([16, 16, 8, 8],  "16-16-8-8"),
            "8,  8, 8, 8":   ([8,  8, 8, 8],   "8-8-8-8"),
            "16, 16, 16, 8": ([16, 16, 16, 8], "16-16-16-8"),
        }
        matched = next((v for k, v in preset_map4.items() if k in bit_choice), None)
        if matched:
            cluster_bits, alloc_name = matched
        else:
            raw = input("Enter 4 values comma-separated (e.g. 16,8,8,8): ").strip()
            cluster_bits = [snap_bits(int(b.strip())) for b in raw.split(",")]
            alloc_name = "custom"

    final_bits = []
    for b in cluster_bits:
        snapped = snap_bits(b)
        if snapped != b:
            print(f"  {b}-bit NOT supported -- snapping to "
                  f"{snapped}-bit ({'FP16' if snapped==16 else 'INT8'})")
        final_bits.append(snapped)
    cluster_bits = final_bits

    gsc = prompt_user(
        f"Select quantization group size (Phase 1 used: {p1_gs}):",
        ["128 (standard, GPTQ default)", "64 (finer granularity)", "32 (finest)", "Custom"],
        default="128 (standard, GPTQ default)"
    )
    if "64" in gsc and "128" not in gsc:   group_size = 64
    elif "32" in gsc and "128" not in gsc: group_size = 32
    elif "Custom" in gsc: group_size = int(input("Enter group size: ").strip())
    else: group_size = 128

    layer_bits_map = {}
    for i in range(num_layers):
        rank = next(j for j, (cid, _) in enumerate(cm) if cid == labels[i])
        layer_bits_map[i] = cluster_bits[rank]

    fp16_layers = sum(1 for b in layer_bits_map.values() if b >= 16)
    int8_layers = sum(1 for b in layer_bits_map.values() if b <  16)

    print(f"\n  Layer bit allocation (PMPQ sensitivity-guided):")
    print(f"  Group size: {group_size}")
    print(f"\n  Summary:")
    print(f"    FP16 (16-bit): {fp16_layers}/{num_layers} layers -- 2x compression")
    print(f"    INT8  (8-bit): {int8_layers}/{num_layers} layers -- 4x compression "
          f"(storage, dequant to FP16 on forward)")
    print(f"\n  Layer-by-layer:")
    for i in range(num_layers):
        b  = layer_bits_map[i]
        rt = "FP16" if b >= 16 else "INT8"
        print(f"    Layer {i:2d}: {b:2d}-bit [{rt}]  (sensitivity: {sv[i]:.6f})")

    # ==========================================================================
    # STEP 5: FP32 Baseline (optional - loads from cache if skipped)
    # ==========================================================================
    cached_baseline = load_baseline_cache()
    if cached_baseline is not None:
        print(f"\n  Found cached baseline: {BASELINE_CACHE_FILE}")
        print(f"  Cached perplexity: {cached_baseline['perplexity']:.4f} "
              f"({cached_baseline['timestamp']})")

    run_bl = prompt_yes_no(
        "Run FP32 baseline evaluation? (No = load from cache file if available)",
        default="yes"
    )

    ppl_b = n_samples_b = total_tok_b = tp_b = None
    baseline_source = "computed"

    # Move model to GPU for baseline
    model = model.to(device)
    is_data_parallel = False
    if num_gpus > 1:
        print(f"\n  Wrapping with DataParallel across {num_gpus} GPUs...")
        model = nn.DataParallel(model)
        is_data_parallel = True

    if run_bl:
        print_section("STEP 5: EVALUATING FP32 BASELINE ON WikiText-2 TEST")
        t0 = time.time()
        ppl_b, _, total_tok_b, tp_b, n_samples_b = evaluate_perplexity_wikitext(
            model, eval_dataset, device,
            eval_name="FP32 Baseline",
            use_fp16=False  # True FP32 for baseline (matches FAKE script)
        )
        timing_log["fp32_evaluation_time_s"] = time.time() - t0

        timestamp_bl = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_baseline_cache(
            ppl_b, total_tok_b, n_samples_b,
            timing_log["fp32_evaluation_time_s"], tp_b,
            str(device), num_gpus, timestamp_bl
        )

        print(f"\n  FP32 Baseline (WikiText-2 test):")
        print(f"  Perplexity: {ppl_b:.4f}")
        print(f"  Samples   : {n_samples_b}  |  Tokens: {total_tok_b}")
        print(f"  Time      : {format_duration(timing_log['fp32_evaluation_time_s'])}")
        print(f"  Throughput: {tp_b:.2f} tokens/s")
    else:
        print_section("STEP 5: LOADING FP32 BASELINE FROM CACHE")
        if cached_baseline is None:
            print("  No cached baseline found.")
            print("  Run FAKE quant script first (it saves the baseline),")
            print("  or run this script with baseline evaluation enabled.")
            timing_log["fp32_evaluation_time_s"] = 0.0
        else:
            ppl_b       = cached_baseline["perplexity"]
            total_tok_b = cached_baseline["total_tokens"]
            n_samples_b = cached_baseline["n_samples"]
            tp_b        = cached_baseline["throughput"]
            timing_log["fp32_evaluation_time_s"] = cached_baseline["eval_time_s"]
            baseline_source = f"cached ({cached_baseline['timestamp']})"
            print(f"  Loaded from: {BASELINE_CACHE_FILE}")
            print(f"  Perplexity: {ppl_b:.4f}")
            print(f"  Throughput: {tp_b:.2f} tokens/s")
            print(f"  Cached on : {cached_baseline['timestamp']}")

    # ==========================================================================
    # STEP 6: Apply REAL Quantization
    # ==========================================================================
    print_section("STEP 6: APPLYING REAL GROUP-WISE MIXED-PRECISION QUANTIZATION")
    print(f"  Mode: REAL weight storage -- FP16 or INT8 -- group_size={group_size}")
    print(f"  INT8: stored as torch.int8, dequantized to FP16 on each forward pass")
    print(f"  Quantization runs on CPU then model moves to {num_gpus} GPU(s)\n")

    if is_data_parallel:
        model = model.module
        is_data_parallel = False
    model = model.cpu()

    tq0 = time.time()
    model, orig_bits, quant_bits = quantize_model_real(
        model, layer_bits_map, group_size=group_size, is_data_parallel=False
    )
    quantize_time_s = time.time() - tq0
    timing_log["quantization_application_time_s"] = quantize_time_s

    compression_ratio = orig_bits / quant_bits if quant_bits > 0 else float("inf")
    reduction_pct     = 100.0 * (1.0 - quant_bits / orig_bits) if orig_bits > 0 else 0.0

    model = model.to(device)
    quant_size_mb = get_model_size_mb(model)

    if num_gpus > 1:
        model = nn.DataParallel(model)
        is_data_parallel = True
        print(f"\n  DataParallel re-enabled across {num_gpus} GPUs after quantization")

    print(f"\n  Quantization complete in {format_duration(quantize_time_s)}")
    print(f"  FP32 model size (before) : {fp32_size_mb:.2f} MB")
    print(f"  Quantized model size (after): {quant_size_mb:.2f} MB  <- measured real")
    print(f"  Compression ratio        : {compression_ratio:.3f}x")
    print(f"  Size reduction           : {reduction_pct:.2f}%")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==========================================================================
    # STEP 7: Evaluate Real-Quantized Model
    # ==========================================================================
    print_section("STEP 7: EVALUATING REAL-QUANTIZED MODEL ON WikiText-2 TEST")
    print(f"  WikiText-2 test split  |  Metric: Perplexity + tokens/s\n")

    t0 = time.time()
    ppl_a, _, total_tok_a, tp_a, n_samples_a = evaluate_perplexity_wikitext(
        model, eval_dataset, device,
        eval_name="Real-Quantized Model",
        use_fp16=True  # FP16 autocast for real FP16/INT8 quantized weights
    )
    timing_log["quantized_evaluation_time_s"] = time.time() - t0

    print(f"\n  Real-Quantized Model (WikiText-2 test):")
    print(f"  Perplexity: {ppl_a:.4f}")
    print(f"  Samples   : {n_samples_a}  |  Tokens: {total_tok_a}")
    print(f"  Time      : {format_duration(timing_log['quantized_evaluation_time_s'])}")
    print(f"  Throughput: {tp_a:.2f} tokens/s")

    # ==========================================================================
    # STEP 8: Performance Analysis
    # ==========================================================================
    print_section("STEP 8: PERFORMANCE COMPARISON")
    timing_log["total_pipeline_time_s"] = time.time() - pipeline_start_time

    if ppl_b is not None:
        ppl_diff     = ppl_a - ppl_b
        ppl_diff_pct = (ppl_diff / ppl_b * 100.0) if ppl_b > 0 else 0.0
        tp_change    = ((tp_a / tp_b) - 1.0) * 100.0 if tp_b and tp_b > 0 else 0.0
        if ppl_diff_pct <= 5.0:    quality = "EXCELLENT -- Minimal degradation"
        elif ppl_diff_pct <= 10.0: quality = "GOOD -- Acceptable degradation"
        elif ppl_diff_pct <= 20.0: quality = "MODERATE -- Noticeable degradation"
        elif ppl_diff_pct <= 35.0: quality = "FAIR -- Significant degradation"
        else:                      quality = "POOR -- Severe degradation"
    else:
        ppl_diff = ppl_diff_pct = tp_change = None
        quality = "N/A (no baseline available)"

    print(f"""
====================================================================
  RESULTS -- PMPQ REAL QUANT GROUP-WISE (FP16/INT8) -- WikiText-2 TEST
====================================================================
  Model         : {MODEL_KEY}
  Quant type    : REAL (FP16 -> float16 | INT8 -> int8 storage + FP16 compute)
  Sensitivity   : {sensitivity_method}
  Clustering    : {sn}   |   Bits: {cluster_bits}
  Group size    : {group_size}      |   GPUs: {num_gpus}
====================================================================
  BASELINE (FP32) [{baseline_source}]:
    Perplexity  : {f"{ppl_b:.4f}" if ppl_b is not None else "N/A"}
    Throughput  : {f"{tp_b:.2f} tokens/s" if tp_b is not None else "N/A"}
====================================================================
  REAL-QUANTIZED (FP16/INT8):
    Perplexity  : {ppl_a:.4f}
    Throughput  : {tp_a:.2f} tokens/s
====================================================================
  DEGRADATION:
    Perplexity  : {f"{ppl_diff:+.4f} ({ppl_diff_pct:+.2f}%)" if ppl_diff is not None else "N/A"}
    Throughput  : {f"{tp_change:+.2f}%" if tp_change is not None else "N/A"}
    Quality     : {quality}
====================================================================
  COMPRESSION (real memory reduction):
    Ratio           : {compression_ratio:.3f}x
    FP32 size       : {fp32_size_mb:.2f} MB
    Quantized size  : {quant_size_mb:.2f} MB  <- measured real
====================================================================
""")

    print_section("TIMING SUMMARY")
    bl_str = (format_duration(timing_log['fp32_evaluation_time_s'])
              + (f" [{baseline_source}]" if not run_bl else ""))
    print(f"""
  Sensitivity File Loading  : {format_duration(timing_log['sensitivity_file_loading_time_s'])}
  Model Loading             : {format_duration(timing_log['model_loading_time_s'])}
  Clustering                : {format_duration(timing_log['clustering_time_s'])}
  FP32 Baseline Evaluation  : {bl_str}
  Quantization Application  : {format_duration(timing_log['quantization_application_time_s'])}
  Quantized Model Evaluation: {format_duration(timing_log['quantized_evaluation_time_s'])}
  -------------------------------------------------------
  Total Pipeline            : {format_duration(timing_log['total_pipeline_time_s'])}
  GPUs used                 : {num_gpus}
""")

    # ==========================================================================
    # STEP 9: Save Real-Quantized Model (.bin HuggingFace format)
    # ==========================================================================
    print_section("STEP 9: SAVING REAL-QUANTIZED MODEL")
    os.makedirs("Models", exist_ok=True)
    alloc_str      = "-".join(str(b) for b in cluster_bits)
    model_dir_name = (
        f"real_quant_TinyLlama_WikiText_{alloc_str}"
        f"_gs{group_size}_{sn}_{sensitivity_method}"
    )
    model_save_path = os.path.join("Models", model_dir_name)
    os.makedirs(model_save_path, exist_ok=True)

    actual_model = model.module if is_data_parallel else model

    hf_config   = AutoConfig.from_pretrained(MODEL_NAME)
    save_model  = AutoModelForCausalLM.from_config(hf_config)
    quant_state = actual_model.state_dict()
    save_state  = save_model.state_dict()
    for key in save_state.keys():
        if key in quant_state:
            save_state[key] = quant_state[key].float()
    save_model.load_state_dict(save_state, strict=True)

    save_model.save_pretrained(
        model_save_path, max_shard_size="500MB", safe_serialization=False
    )
    tokenizer.save_pretrained(model_save_path)

    quant_config = {
        "quantization_type":    "REAL_FP16_INT8",
        "description":          (
            "Weights dequantized to FP32 for .bin storage. "
            "Original inference: FP16 stored as float16; "
            "INT8 stored as int8, dequantized to FP16 on each forward pass. "
            "To restore: reload FP32 weights and re-run quantize_model_real()."
        ),
        "sensitivity_method":   sensitivity_method,
        "cluster_bits":         cluster_bits,
        "allocation_name":      alloc_name,
        "group_size":           group_size,
        "layer_bits_map":       {str(k): v for k, v in layer_bits_map.items()},
        "fp16_layers":          fp16_layers,
        "int8_layers":          int8_layers,
        "strategy_name":        sn,
        "model_key":            MODEL_KEY,
        "model_name":           MODEL_NAME,
        "ppl_after":            float(ppl_a),
        "ppl_before":           float(ppl_b) if ppl_b is not None else None,
        "baseline_source":      baseline_source,
        "eval_split":           EVAL_SPLIT,
        "fp32_size_mb":         float(fp32_size_mb),
        "quant_size_mb":        float(quant_size_mb),
        "compression_ratio":    float(compression_ratio),
        "reduction_pct":        float(reduction_pct),
        "timestamp":            datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(os.path.join(model_save_path, "quant_config.json"), "w") as qf:
        json.dump(quant_config, qf, indent=2)

    print(f"  Real-quantized model saved to: {model_save_path}/")
    print(f"    config.json              -- model architecture & hyperparameters")
    print(f"    tokenizer.json           -- tokenizer files")
    print(f"    pytorch_model-*.bin      -- sharded weights (dequantized to FP32)")
    print(f"    pytorch_model.bin.index.json -- shard index")
    print(f"    quant_config.json        -- PMPQ quantization metadata")
    print(f"  Note: .bin weights are FP32 (HuggingFace standard format).")
    print(f"  Reload + re-run quantize_model_real() to restore FP16/INT8 inference.")

    # ==========================================================================
    # STEP 10: Save Results Log
    # ==========================================================================
    print_section("STEP 10: SAVING RESULTS LOG")
    os.makedirs("Evaluation", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_filename = (
        f"real_quant_eval_TinyLlama_WikiText_{alloc_name}_"
        f"{sensitivity_method}_GroupWise_{timestamp}.txt"
    )
    log_path = os.path.join("Evaluation", log_filename)

    with open(log_path, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"REAL QUANTIZATION EVALUATION RESULTS "
                f"({sensitivity_method} + GROUP-WISE)\n")
        f.write(f"TinyLlama on WikiText-2 Test Set\n")
        f.write("="*80 + "\n\n")

        # CONFIGURATION
        f.write("="*80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("="*80 + "\n")
        f.write(f"Quantization Type: REAL "
                f"(FP16/INT8 -- weights physically stored in low precision)\n")
        f.write(f"  FP16: weights stored as torch.float16  -- 2x real memory savings\n")
        f.write(f"  INT8: weights stored as torch.int8     -- 4x real memory savings\n")
        f.write(f"        INT8 is storage only. On each forward pass:\n")
        f.write(f"        int8 weights dequantized to FP16, then FP16 matmul\n")
        f.write(f"        on A100 Tensor Cores. (weight-only quantization)\n")
        f.write(f"        Per-group FP16 scale: group_size={group_size} weights -> 1 scale\n")
        f.write(f"  No 4-bit/2-bit -- requires TorchAO/bitsandbytes for sub-INT8\n")
        f.write(f"Model: {MODEL_KEY}\n")
        f.write(f"Model HF Hub: {MODEL_NAME}\n")
        f.write(f"Task: Language Modeling (WikiText-2 -- Perplexity)\n")
        f.write(f"Dataset: WikiText-2\n")
        f.write(f"Evaluation Split: {EVAL_SPLIT} (test set -- used in Phase 2)\n")
        f.write(f"Sequence Length: {SEQUENCE_LENGTH}\n")
        f.write(f"Sensitivity Method: {sensitivity_method}\n")
        f.write(f"Sensitivity File: {selected}\n")
        f.write(f"Clustering Strategy: {sn}\n")
        f.write(f"Number of Clusters: {n_clusters}\n")
        f.write(f"Bit Allocation: {cluster_bits}\n")
        f.write(f"Allocation Name: {alloc_name}\n")
        f.write(f"Group Size: {group_size}\n")
        f.write(f"FP16 layers: {fp16_layers}/{num_layers}\n")
        f.write(f"INT8 layers: {int8_layers}/{num_layers}\n")
        f.write(f"Baseline Source: {baseline_source}\n")
        f.write(f"Baseline Cache File: {BASELINE_CACHE_FILE}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Num GPUs: {num_gpus}\n")
        if torch.cuda.is_available():
            for gi in range(num_gpus):
                f.write(f"  GPU {gi}: {torch.cuda.get_device_name(gi)}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Saved Model Folder: {model_save_path}\n\n")

        # COMPREHENSIVE TIMING LOG
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE TIMING LOG\n")
        f.write("="*80 + "\n")
        f.write(f"{'Step':<55} {'Time':<20} {'Seconds':<12}\n")
        f.write("-"*87 + "\n")
        for lbl, key in [
            ("Sensitivity File Loading",
             "sensitivity_file_loading_time_s"),
            ("Model Loading",
             "model_loading_time_s"),
            ("Clustering",
             "clustering_time_s"),
            (f"FP32 Baseline Eval [{baseline_source}]",
             "fp32_evaluation_time_s"),
            ("Real Quantization Application (FP16/INT8)",
             "quantization_application_time_s"),
            ("Real-Quantized Model Evaluation",
             "quantized_evaluation_time_s"),
        ]:
            v = timing_log[key]
            f.write(f"{lbl:<55} {format_duration(v):<20} {v:<12.4f}\n")
        f.write("-"*87 + "\n")
        tv = timing_log["total_pipeline_time_s"]
        f.write(f"{'TOTAL PIPELINE TIME':<55} {format_duration(tv):<20} "
                f"{tv:<12.4f}\n")
        if p1_time:
            f.write(f"\nPhase 1 Sensitivity Computation (from metadata):\n")
            f.write(f"  {format_duration(p1_time)}  ({p1_time:.4f}s)\n")
        f.write("\n")

        # METRICS BEFORE QUANTIZATION (FP32 Baseline)
        f.write("="*80 + "\n")
        f.write("METRICS BEFORE QUANTIZATION (FP32 Baseline)\n")
        f.write("="*80 + "\n")
        if ppl_b is not None:
            f.write(f"Perplexity: {ppl_b:.6f}\n")
            f.write(f"Samples: {n_samples_b}  |  Total Tokens: {total_tok_b}\n")
            f.write(f"Eval Time: {format_duration(timing_log['fp32_evaluation_time_s'])}\n")
            f.write(f"Throughput: {tp_b:.2f} tokens/s\n")
            f.write(f"Baseline Source: {baseline_source}\n")
            f.write(f"Baseline Cache File: {BASELINE_CACHE_FILE}\n\n")
        else:
            f.write("FP32 BASELINE -- NOT AVAILABLE (no cache, skip selected)\n\n")

        # METRICS AFTER QUANTIZATION (Group-Wise Real Mixed-Precision PTQ)
        f.write("="*80 + "\n")
        f.write("METRICS AFTER QUANTIZATION "
                "(Group-Wise Real Mixed-Precision PTQ)\n")
        f.write("="*80 + "\n")
        f.write(f"Perplexity: {ppl_a:.6f}\n")
        f.write(f"Samples: {n_samples_a}  |  Total Tokens: {total_tok_a}\n")
        f.write(f"Eval Time: {format_duration(timing_log['quantized_evaluation_time_s'])}\n")
        f.write(f"Throughput: {tp_a:.2f} tokens/s\n\n")

        # PERFORMANCE COMPARISON
        f.write("="*80 + "\n")
        f.write("PERFORMANCE COMPARISON\n")
        f.write("="*80 + "\n")
        if ppl_diff is not None:
            f.write(f"Perplexity Change: {ppl_diff:+.6f} ({ppl_diff_pct:+.2f}%)\n")
            f.write(f"Throughput Change: {tp_change:+.2f}%\n")
            f.write(f"Quality Assessment: {quality}\n\n")
        else:
            f.write("Not available (no baseline)\n\n")

        # COMPRESSION METRICS
        f.write("="*80 + "\n")
        f.write("COMPRESSION METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Quantization Type: REAL "
                f"(weights physically stored in low precision)\n")
        f.write(f"FP32 Model Size (measured before quantization): "
                f"{fp32_size_mb:.2f} MB\n")
        f.write(f"Quantized Model Size (measured after quantization): "
                f"{quant_size_mb:.2f} MB\n")
        f.write(f"  (FP16 weights for FP16 layers + INT8 weights + "
                f"FP16 group scales)\n")
        f.write(f"Compression Ratio: {compression_ratio:.3f}x\n")
        f.write(f"Size Reduction: {reduction_pct:.2f}%\n")
        f.write(f"Quantization Time: {quantize_time_s:.4f}s\n\n")

        # MACHINE-READABLE METRICS (KEY-VALUE)
        f.write("="*80 + "\n")
        f.write("MACHINE-READABLE METRICS (KEY-VALUE)\n")
        f.write("="*80 + "\n")
        for k, v in [
            ("quantization_type",  "REAL_FP16_INT8"),
            ("model_key",           MODEL_KEY),
            ("model_name",          MODEL_NAME),
            ("task",                "WikiText2_Language_Modeling"),
            ("eval_split",          EVAL_SPLIT),
            ("sequence_length",     SEQUENCE_LENGTH),
            ("device",              str(device)),
            ("num_gpus",            num_gpus),
            ("sensitivity_method",  sensitivity_method),
            ("clustering_strategy", sn),
            ("n_clusters",          n_clusters),
            ("allocation_name",     alloc_name),
            ("cluster_bits",        str(cluster_bits)),
            ("group_size",          group_size),
            ("fp16_layers",         fp16_layers),
            ("int8_layers",         int8_layers),
            ("baseline_source",     baseline_source),
        ]:
            f.write(f"{k}: {v}\n")
        if ppl_b is not None:
            f.write(f"ppl_before: {ppl_b:.6f}\n")
            f.write(f"total_tokens_before: {total_tok_b}\n")
            f.write(f"n_samples_before: {n_samples_b}\n")
            f.write(f"throughput_before: {tp_b:.2f}\n")
        f.write(f"ppl_after: {ppl_a:.6f}\n")
        f.write(f"total_tokens_after: {total_tok_a}\n")
        f.write(f"n_samples_after: {n_samples_a}\n")
        f.write(f"throughput_after: {tp_a:.2f}\n")
        if ppl_diff is not None:
            f.write(f"ppl_change: {ppl_diff:.6f}\n")
            f.write(f"ppl_change_pct: {ppl_diff_pct:.2f}\n")
            f.write(f"throughput_change_pct: {tp_change:.2f}\n")
        f.write(f"fp32_model_size_mb: {fp32_size_mb:.2f}\n")
        f.write(f"quant_model_size_mb: {quant_size_mb:.2f}\n")
        f.write(f"compression_ratio: {compression_ratio:.3f}\n")
        f.write(f"reduction_pct: {reduction_pct:.2f}\n")
        f.write(f"saved_model_folder: {model_save_path}\n")
        f.write(f"\n# Timing (seconds):\n")
        for k, v in timing_log.items():
            f.write(f"{k}: {v:.4f}\n")
        if p1_time:
            f.write(f"phase1_sensitivity_time_s: {p1_time:.4f}\n")
        if p1_ppl:
            f.write(f"phase1_fp32_ppl_train: {p1_ppl:.6f}\n")

        # LAYER BIT ALLOCATION
        f.write("\n" + "="*80 + "\n")
        f.write("LAYER BIT ALLOCATION\n")
        f.write("="*80 + "\n")
        for i in range(num_layers):
            b  = layer_bits_map[i]
            rt = "FP16" if b >= 16 else "INT8"
            f.write(f"  layer_{i:02d}: {b:2d}-bit [{rt}]  "
                    f"sensitivity: {sv[i]:.8f}\n")

        # METHOD NOTES
        f.write("\n" + "="*80 + "\n")
        f.write("METHOD NOTES\n")
        f.write("="*80 + "\n")
        f.write(f"Sensitivity Analysis Method: {sensitivity_method}\n")
        f.write(f"  PMPQ uses normalized Frobenius norm of weight matrices.\n")
        f.write(f"  sensitivity(layer_i) = mean(||W||_F / sqrt(|W|)) over\n")
        f.write(f"  all Linear modules in layer i.\n")
        f.write(f"  Higher score -> larger weight magnitude -> more sensitive\n")
        f.write(f"  -> needs higher bit-width (FP16) to preserve perplexity.\n")
        f.write(f"  Lower score  -> smaller magnitude -> can use INT8.\n\n")
        f.write(f"Quantization: REAL (weights stored in FP16 or INT8 on GPU)\n")
        f.write(f"  FP16 layers: weight stored as torch.float16.\n")
        f.write(f"               A100 Tensor Cores run FP16 matmul natively.\n")
        f.write(f"  INT8 layers: weight stored as torch.int8 (1 byte per weight).\n")
        f.write(f"               On each forward pass: int8 weights dequantized\n")
        f.write(f"               to FP16 using per-group scale, then FP16 matmul.\n")
        f.write(f"               This is weight-only quantization (same as GPTQ).\n")
        f.write(f"               INT8 is purely a storage format -- computation\n")
        f.write(f"               runs in FP16 on A100 Tensor Cores.\n\n")
        f.write(f"Group-Wise Quantization:\n")
        f.write(f"  group_size={group_size}: every {group_size} consecutive weights\n")
        f.write(f"  share one FP16 scale factor. Same approach as GPTQ.\n\n")
        f.write(f"Dataset: WikiText-2 Test Set\n")
        f.write(f"  Language modeling benchmark.\n")
        f.write(f"  Metric: Perplexity (lower is better).\n")
        f.write(f"  Additional metric: throughput in tokens/s.\n")
        f.write(f"  Note: Phase 1 calibration used train split ({SEQUENCE_LENGTH} tokens).\n")
        f.write(f"        Phase 2 evaluation always uses test split.\n\n")
        f.write(f"Clustering: {sn} with {n_clusters} groups\n")
        for rank, (cid, cmean) in enumerate(cm):
            lids = [i for i in range(num_layers) if labels[i] == cid]
            b    = cluster_bits[rank]
            rt   = "FP16" if b >= 16 else "INT8"
            f.write(f"  Cluster {cid} (rank {rank}, bits={b} [{rt}]): "
                    f"{len(lids)} layers, mean sensitivity={cmean:.6f}\n")
        f.write(f"\nModel saved in HuggingFace .bin sharded format:\n")
        f.write(f"  Folder: {model_save_path}/\n")
        f.write(f"  config.json              -- model architecture & hyperparameters\n")
        f.write(f"  tokenizer.json           -- tokenizer vocabulary & config\n")
        f.write(f"  pytorch_model-*.bin      -- weight shards (500MB each, FP32)\n")
        f.write(f"  pytorch_model.bin.index.json -- shard index\n")
        f.write(f"  quant_config.json        -- PMPQ quantization configuration\n")
        f.write(f"  Note: .bin files contain FP32 weights (HuggingFace standard).\n")
        f.write(f"  quant_config.json records FP16/INT8 layer assignment so\n")
        f.write(f"  quantization can be exactly reproduced.\n")

    print(f"  Results log saved to: {log_path}")
    print(f"\nPhase 2 (REAL) complete!")


if __name__ == "__main__":
    main()