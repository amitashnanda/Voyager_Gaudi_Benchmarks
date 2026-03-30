# Phase_2_evaluate_GroupWise_TinyLlama_WikiText_FAKE.py
"""
Phase 2: PMPQ Group-Wise Quantization & Evaluation - TinyLlama on WikiText-2
=============================================================================
QUANTIZATION TYPE: FAKE (Simulated)
  Weights remain FP32 throughout. Quantization is simulated:
  FP32 -> quantize -> dequantize -> store back as FP32.
  No real memory savings. Used to measure perplexity impact only.

PMPQ -- Pruning-based Mixed-Precision Quantization
  Phase 1 sensitivity = normalized Frobenius norm of each layer's weights.
  Higher sensitivity -> assigned higher bit-width.
  Lower sensitivity  -> assigned lower bit-width.

Evaluation Dataset:
  WikiText-2 TEST split (not train).
  Metric: Perplexity (lower is better).
  Additional metrics: tokens/s throughput, eval time.

Baseline Caching:
  FP32 baseline perplexity saved to Models/baseline_wikitext_fp32.json.
  If baseline skip selected, results loaded from that file.

Model Saving (.bin HuggingFace format):
  Saved to Models/fake_quant_TinyLlama_WikiText_<bits>_gs<gs>_<strategy>_PMPQ/
  containing: config.json, tokenizer.json, pytorch_model-*.bin, quant_config.json

Comprehensive timing logs are recorded for EVERY step.

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
from itertools import chain

import torch
import torch.nn as nn
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


# ============================================================================
# UTILITIES
# ============================================================================

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cuda_memory_info():
    """Get CUDA memory usage across all GPUs."""
    if not torch.cuda.is_available():
        return None

    num_gpus = torch.cuda.device_count()
    total_allocated = 0
    total_reserved = 0
    total_max_allocated = 0

    for i in range(num_gpus):
        total_allocated += torch.cuda.memory_allocated(i) / (1024 ** 2)  # MB
        total_reserved += torch.cuda.memory_reserved(i) / (1024 ** 2)  # MB
        total_max_allocated += torch.cuda.max_memory_allocated(i) / (1024 ** 2)  # MB

    return {
        'allocated_mb': total_allocated,
        'reserved_mb': total_reserved,
        'max_allocated_mb': total_max_allocated,
        'num_gpus': num_gpus
    }


def reset_cuda_memory():
    """Reset CUDA memory stats."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
            torch.cuda.empty_cache()


def prepare_wikitext_dataset(tokenizer, split="test", block_size=512):
    """
    Prepare WikiText-2 dataset with continuous tokenization and chunking.
    Matches HPU implementation exactly.

    Args:
        tokenizer: HuggingFace tokenizer
        split: "train", "validation", or "test"
        block_size: sequence length for chunks (default 512)

    Returns:
        Dataset with input_ids and labels
    """
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


def pick_device():
    if torch.cuda.is_available():
        print(f"CUDA: {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    total  = sum(p.nelement() * p.element_size() for p in model.parameters())
    total += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total / (1024 * 1024)


# ============================================================================
# BASELINE CACHE
# ============================================================================

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


def load_baseline_cache():
    if not os.path.exists(BASELINE_CACHE_FILE):
        return None
    with open(BASELINE_CACHE_FILE, "r") as f:
        return json.load(f)


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
# FAKE QUANTIZATION -- GROUP-WISE SIMULATED
# ============================================================================

class LinearSymmetricGroupQuant(nn.Module):
    """
    Symmetric MinMax Group-Wise Weight Quantization (PTQ).

    Computes a static per-group scale from the max absolute weight value,
    then quantizes via round-to-nearest and clamps to the target bit range.
    This is standard Post-Training Quantization (PTQ), not Learned Step Size
    Quantization (LSQ) which would require learnable scale parameters and
    gradient-based optimization.

    FAKE quantization (simulated):
    Weights: FP32 -> quantize -> dequantize -> stored back as FP32.
    No real memory savings. Measures perplexity impact only.
    Any bit-width accepted (2, 4, 6, 8, 12, 16, 32).

    Quantization of weights happens at construction time on CPU.
    The forward pass (F.linear) runs on whatever device the input tensor is on.
    """

    def __init__(self, linear: nn.Linear, nbits_w: int, group_size: int = 128):
        super().__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.nbits_w      = int(nbits_w)
        self.group_size   = group_size
        qw = self._quantize(linear.weight.detach().clone())
        self.register_buffer("weight", qw)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.register_buffer("bias", None)

    def _quantize(self, w):
        if self.nbits_w >= 32: return w
        out_f, in_f = w.shape; orig_in = in_f
        if in_f % self.group_size != 0:
            pad = self.group_size - (in_f % self.group_size)
            w = torch.nn.functional.pad(w, (0, pad), value=0)
            in_f = w.shape[1]
        ng   = in_f // self.group_size
        wg   = w.view(out_f, ng, self.group_size)
        qmin = -(2 ** (self.nbits_w - 1))
        qmax = (2 ** (self.nbits_w - 1)) - 1
        sc   = torch.clamp(wg.abs().amax(dim=-1, keepdim=True) / qmax, min=1e-8)
        q    = torch.clamp(torch.round(wg / sc), qmin, qmax)
        wq   = (q * sc).view(out_f, -1)
        return wq[:, :orig_in] if wq.shape[1] > orig_in else wq

    def calculate_weight_bits(self):
        nw    = self.weight.numel()
        ngrps = self.out_features * ((self.in_features + self.group_size - 1) // self.group_size)
        return int(nw * 32), int(nw * self.nbits_w + ngrps * 16)

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)


def _set_module(root, qualname, new_mod):
    parts = qualname.split(".")
    parent = root
    for p in parts[:-1]: parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def quantize_model_fake(model, layer_bits_map, group_size=128,
                        is_data_parallel=False):
    actual = model.module if is_data_parallel else model
    targets = []
    for name, mod in actual.named_modules():
        if isinstance(mod, nn.Linear):
            m = re.search(r"(?:model\.|decoder\.)layers\.(\d+)\.", name)
            if m:
                layer_idx = int(m.group(1))
                targets.append((name, mod, layer_bits_map.get(layer_idx, 8)))

    print(f"  Found {len(targets)} Linear layers to quantize (FAKE / simulated)")
    print(f"  Group size: {group_size}")
    bit_counts = {}
    for _, _, b in targets:
        bit_counts[b] = bit_counts.get(b, 0) + 1
    for b in sorted(bit_counts.keys(), reverse=True):
        print(f"  {b}-bit layers: {bit_counts[b]}")

    total_orig = total_quant = 0
    for qualname, lm, nb in tqdm(targets, desc="  Fake-quantizing layers"):
        wrapper = LinearSymmetricGroupQuant(lm, nb, group_size)
        _set_module(actual, qualname, wrapper)
        o, q = wrapper.calculate_weight_bits()
        total_orig += o; total_quant += q
    return model, total_orig, total_quant


# ============================================================================
# WIKITEXT-2 PERPLEXITY EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_perplexity_wikitext(model, eval_dataset, device, eval_name="Perplexity", batch_size=4):
    """
    Evaluate perplexity using pre-prepared dataset.
    Matches HPU and sensitivity implementations.

    Args:
        model: nn.Module (can be wrapped in DDP)
        eval_dataset: HuggingFace dataset with input_ids and labels
        device: torch.device
        eval_name: label for progress bar
        batch_size: number of samples per batch (matches HPU per_device_batch_size)

    Returns:
        perplexity, eval_time_s, total_tokens, throughput (tokens/s), n_samples
    """
    n_samples = len(eval_dataset)
    seqlen = len(eval_dataset[0]["input_ids"])
    total_tokens = n_samples * seqlen

    print(f"  Split: test  |  Samples: {n_samples}  |  Seqlen: {seqlen}  |  Total tokens: {total_tokens:,}  |  Batch size: {batch_size}")

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

            # FP32 inference (no autocast) - matches HPU implementation
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
    print_section("PHASE 2: PMPQ WikiText-2 Evaluation -- FAKE QUANTIZATION (Simulated)")
    print("""
  FAKE QUANTIZATION:
    Weights simulated: FP32 -> quantize -> dequantize -> stored back as FP32.
    Any bit-width accepted (2, 4, 6, 8, 12, 16, 32).
    No real memory savings. Used to measure perplexity impact only.

  Evaluation: WikiText-2 TEST split (perplexity + tokens/s throughput).

  Baseline caching:
    FP32 results saved to Models/baseline_wikitext_fp32.json.
    When baseline is skipped, results loaded from that file.

  Model saving:
    Saved to Models/<n>/ in HuggingFace .bin sharded format.
    """)

    timing_log          = {}
    pipeline_start_time = time.time()

    set_seed(42)

    device = pick_device()
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
    print(f"\nLoaded: {selected}  "
          f"({format_duration(timing_log['sensitivity_file_loading_time_s'])})")
    print(f"  Layers: {num_layers}  |  "
          f"Sensitivity: [{sv.min():.6f}, {sv.max():.6f}]")
    if p1_time:
        print(f"  Phase 1 sensitivity time: {format_duration(p1_time)}")
    if p1_ppl:
        print(f"  Phase 1 FP32 perplexity (train): {p1_ppl:.4f}")

    # Detect method from filename
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
    model = model.to(device)

    is_data_parallel = False
    if torch.cuda.device_count() > 1:
        print(f"  Wrapping with DataParallel across {torch.cuda.device_count()} GPUs...")
        model = nn.DataParallel(model)
        is_data_parallel = True

    fp32_size_mb = get_model_size_mb(model)
    timing_log["model_loading_time_s"] = time.time() - t0
    print(f"  Model loaded in {format_duration(timing_log['model_loading_time_s'])}")
    print(f"  FP32 model size: {fp32_size_mb:.2f} MB")

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
    # STEP 4: Bit-Width Allocation
    # ==========================================================================
    print_section("STEP 4: BIT-WIDTH ALLOCATION")
    print("  FAKE QUANTIZATION accepts any bit-width (2, 4, 6, 8, 12, 16, 32).")
    print("  These are simulated -- no real hardware constraint applies here.")

    if n_clusters == 3:
        bit_options = [
            "[16, 8, 8]  -- safe, recommended",
            "[16, 8, 4]  -- moderate compression",
            "[8,  8, 4]  -- aggressive",
            "[8,  4, 2]  -- extreme (not recommended)",
            "Custom"
        ]
        bit_choice = prompt_user(
            "Select bit allocation (high sensitivity -> low sensitivity):",
            bit_options, default=bit_options[0]
        )
        preset_map = {
            "16, 8, 8": ([16, 8, 8], "16-8-8"),
            "16, 8, 4": ([16, 8, 4], "16-8-4"),
            "8,  8, 4": ([8,  8, 4], "8-8-4"),
            "8,  4, 2": ([8,  4, 2], "8-4-2"),
        }
        matched = next((v for k, v in preset_map.items() if k in bit_choice), None)
        if matched:
            cluster_bits, alloc_name = matched
        else:
            raw = input("Enter 3 values comma-separated (e.g. 16,8,4): ").strip()
            cluster_bits = [int(b.strip()) for b in raw.split(",")]
            alloc_name = "custom"
    else:
        bit_options = [
            "[16, 8, 8, 4]  -- recommended",
            "[16, 8, 4, 2]  -- aggressive",
            "[8,  8, 4, 4]  -- moderate",
            "Custom"
        ]
        bit_choice = prompt_user(
            "Select bit allocation (high sensitivity -> low sensitivity):",
            bit_options, default=bit_options[0]
        )
        preset_map4 = {
            "16, 8, 8, 4": ([16, 8, 8, 4], "16-8-8-4"),
            "16, 8, 4, 2": ([16, 8, 4, 2], "16-8-4-2"),
            "8,  8, 4, 4": ([8,  8, 4, 4], "8-8-4-4"),
        }
        matched = next((v for k, v in preset_map4.items() if k in bit_choice), None)
        if matched:
            cluster_bits, alloc_name = matched
        else:
            raw = input("Enter 4 values comma-separated (e.g. 16,8,8,4): ").strip()
            cluster_bits = [int(b.strip()) for b in raw.split(",")]
            alloc_name = "custom"

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

    print(f"\n  Layer bit allocation (PMPQ sensitivity-guided):")
    print(f"  Group size: {group_size}")
    bit_dist = {}
    for b in cluster_bits:
        bit_dist[b] = sum(1 for v in layer_bits_map.values() if v == b)
    print(f"\n  Bit-width distribution:")
    for b in sorted(set(cluster_bits), reverse=True):
        print(f"    {b:2d}-bit (simulated): {bit_dist[b]}/{num_layers} layers "
              f"({bit_dist[b]/num_layers*100:.1f}%)")
    print(f"\n  Layer-by-layer:")
    for i in range(num_layers):
        print(f"    Layer {i:2d}: {layer_bits_map[i]:2d}-bit "
              f"(sensitivity: {sv[i]:.6f})")

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

    if run_bl:
        print_section("STEP 5: EVALUATING FP32 BASELINE ON WikiText-2 TEST")
        t0 = time.time()
        ppl_b, eval_time_b, total_tok_b, tp_b, n_samples_b = evaluate_perplexity_wikitext(
            model, eval_dataset, device,
            eval_name="FP32 Baseline"
        )
        throughput_before = total_tok_b / eval_time_b
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
            print("  No cached baseline found. Please run with baseline evaluation first.")
            print("  Proceeding without baseline comparison.")
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
    # STEP 6: Apply Fake Quantization
    # ==========================================================================
    print_section("STEP 6: APPLYING GROUP-WISE FAKE QUANTIZATION (Simulated)")
    print(f"  Mode: FAKE -- weights remain FP32 after simulate-quantize-dequantize")
    print(f"  Group size: {group_size}\n")

    tq0 = time.time()
    model, orig_bits, quant_bits = quantize_model_fake(
        model, layer_bits_map, group_size=group_size,
        is_data_parallel=is_data_parallel
    )
    quantize_time_s = time.time() - tq0
    timing_log["quantization_application_time_s"] = quantize_time_s

    compression_ratio = orig_bits / quant_bits if quant_bits > 0 else float("inf")
    reduction_pct     = 100.0 * (1.0 - quant_bits / orig_bits) if orig_bits > 0 else 0.0
    post_fake_size_mb = get_model_size_mb(model)

    print(f"\n  Fake quantization complete in {format_duration(quantize_time_s)}")
    print(f"  Model size (unchanged -- still FP32): {post_fake_size_mb:.2f} MB")
    print(f"  Simulated compression ratio         : {compression_ratio:.3f}x")
    print(f"  Simulated size reduction            : {reduction_pct:.2f}%")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ==========================================================================
    # STEP 7: Evaluate Fake-Quantized Model
    # ==========================================================================
    print_section("STEP 7: EVALUATING FAKE-QUANTIZED MODEL ON WikiText-2 TEST")
    print(f"  WikiText-2 test split  |  Metric: Perplexity + tokens/s\n")

    t0 = time.time()
    ppl_a, eval_time_a, total_tok_a, tp_a, n_samples_a = evaluate_perplexity_wikitext(
        model, eval_dataset, device,
        eval_name="Fake-Quantized Model"
    )
    throughput_after = total_tok_a / eval_time_a
    speedup = throughput_after / throughput_before if throughput_before > 0 else 1.0
    timing_log["quantized_evaluation_time_s"] = time.time() - t0

    print(f"\n  Fake-Quantized Model (WikiText-2 test):")
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
  RESULTS -- PMPQ FAKE QUANT GROUP-WISE -- WikiText-2 TEST
====================================================================
  Model         : {MODEL_KEY}
  Quant type    : FAKE (simulated, weights remain FP32)
  Sensitivity   : {sensitivity_method}
  Clustering    : {sn}   |   Bits: {cluster_bits}
  Group size    : {group_size}      |   GPUs: {num_gpus}
====================================================================
  BASELINE (FP32) [{baseline_source}]:
    Perplexity  : {f"{ppl_b:.4f}" if ppl_b is not None else "N/A"}
    Throughput  : {f"{tp_b:.2f} tokens/s" if tp_b is not None else "N/A"}
====================================================================
  FAKE-QUANTIZED:
    Perplexity  : {ppl_a:.4f}
    Throughput  : {tp_a:.2f} tokens/s
====================================================================
  DEGRADATION:
    Perplexity  : {f"{ppl_diff:+.4f} ({ppl_diff_pct:+.2f}%)" if ppl_diff is not None else "N/A"}
    Throughput  : {f"{tp_change:+.2f}%" if tp_change is not None else "N/A"}
    Quality     : {quality}
====================================================================
  COMPRESSION (simulated only -- no real memory change):
    Simulated ratio   : {compression_ratio:.3f}x
    Actual model size : {post_fake_size_mb:.2f} MB (unchanged FP32)
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
    # STEP 9: Save Fake-Quantized Model (.bin HuggingFace format)
    # ==========================================================================
    print_section("STEP 9: SAVING FAKE-QUANTIZED MODEL")
    os.makedirs("Models", exist_ok=True)
    alloc_str      = "-".join(str(b) for b in cluster_bits)
    model_dir_name = (
        f"fake_quant_TinyLlama_WikiText_{alloc_str}"
        f"_gs{group_size}_{sn}_{sensitivity_method}"
    )
    model_save_path = os.path.join("Models", model_dir_name)
    os.makedirs(model_save_path, exist_ok=True)

    actual_model = model.module if is_data_parallel else model
    hf_config    = AutoConfig.from_pretrained(MODEL_NAME)
    save_model   = AutoModelForCausalLM.from_config(hf_config)
    missing, _   = save_model.load_state_dict(actual_model.state_dict(), strict=False)
    if missing:
        print(f"  Note: {len(missing)} keys not copied "
              f"(custom buffer layers -- expected)")

    save_model.save_pretrained(
        model_save_path, max_shard_size="500MB", safe_serialization=False
    )
    tokenizer.save_pretrained(model_save_path)

    quant_config = {
        "quantization_type":  "FAKE_SIMULATED",
        "description":        "Weights are FP32. Quantization effect is simulated only.",
        "sensitivity_method": sensitivity_method,
        "cluster_bits":       cluster_bits,
        "allocation_name":    alloc_name,
        "group_size":         group_size,
        "layer_bits_map":     {str(k): v for k, v in layer_bits_map.items()},
        "strategy_name":      sn,
        "model_key":          MODEL_KEY,
        "model_name":         MODEL_NAME,
        "ppl_after":          float(ppl_a),
        "ppl_before":         float(ppl_b) if ppl_b is not None else None,
        "baseline_source":    baseline_source,
        "eval_split":         EVAL_SPLIT,
        "timestamp":          datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(os.path.join(model_save_path, "quant_config.json"), "w") as qf:
        json.dump(quant_config, qf, indent=2)

    print(f"  Fake-quantized model saved to: {model_save_path}/")
    print(f"    config.json              -- model architecture & hyperparameters")
    print(f"    tokenizer.json           -- tokenizer files")
    print(f"    pytorch_model-*.bin      -- sharded FP32 weights (500MB per shard)")
    print(f"    pytorch_model.bin.index.json -- shard index")
    print(f"    quant_config.json        -- PMPQ quantization metadata")

    # ==========================================================================
    # STEP 10: Save Results Log
    # ==========================================================================
    print_section("STEP 10: SAVING RESULTS LOG")
    os.makedirs("Evaluation", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_filename = (
        f"fake_quant_eval_TinyLlama_WikiText_{alloc_name}_"
        f"{sensitivity_method}_GroupWise_{timestamp}.txt"
    )
    log_path = os.path.join("Evaluation", log_filename)

    with open(log_path, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"FAKE QUANTIZATION EVALUATION RESULTS "
                f"({sensitivity_method} + GROUP-WISE)\n")
        f.write(f"TinyLlama on WikiText-2 Test Set\n")
        f.write("="*80 + "\n\n")

        # CONFIGURATION
        f.write("="*80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("="*80 + "\n")
        f.write(f"Quantization Type: FAKE (simulated -- weights remain FP32 throughout)\n")
        f.write(f"  Mechanism: FP32 -> quantize -> dequantize -> store back as FP32\n")
        f.write(f"  Purpose: Measure perplexity impact only. No real memory savings.\n")
        f.write(f"  Bit-widths accepted: any (2, 4, 6, 8, 12, 16, 32)\n")
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
        f.write(f"Baseline Source: {baseline_source}\n")
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
        f.write(f"{'Step':<52} {'Time':<20} {'Seconds':<12}\n")
        f.write("-"*84 + "\n")
        for lbl, key in [
            ("Sensitivity File Loading",
             "sensitivity_file_loading_time_s"),
            ("Model Loading",
             "model_loading_time_s"),
            ("Clustering",
             "clustering_time_s"),
            (f"FP32 Baseline Eval [{baseline_source}]",
             "fp32_evaluation_time_s"),
            ("Fake Quantization Application",
             "quantization_application_time_s"),
            ("Fake-Quantized Model Evaluation",
             "quantized_evaluation_time_s"),
        ]:
            v = timing_log[key]
            f.write(f"{lbl:<52} {format_duration(v):<20} {v:<12.4f}\n")
        f.write("-"*84 + "\n")
        tv = timing_log["total_pipeline_time_s"]
        f.write(f"{'TOTAL PIPELINE TIME':<52} {format_duration(tv):<20} {tv:<12.4f}\n")
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

        # METRICS AFTER QUANTIZATION (Group-Wise Fake Mixed-Precision PTQ)
        f.write("="*80 + "\n")
        f.write("METRICS AFTER QUANTIZATION (Group-Wise Fake Mixed-Precision PTQ)\n")
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
        f.write(f"Quantization Type: FAKE (no real memory change)\n")
        f.write(f"FP32 Model Size (measured): {fp32_size_mb:.2f} MB\n")
        f.write(f"Model Size After Fake Quant (measured): {post_fake_size_mb:.2f} MB\n")
        f.write(f"  (identical -- weights remain FP32)\n")
        f.write(f"Simulated Compression Ratio: {compression_ratio:.3f}x\n")
        f.write(f"  (bit-level calculation only, not real savings)\n")
        f.write(f"Simulated Size Reduction: {reduction_pct:.2f}%\n")
        f.write(f"Quantization Time: {quantize_time_s:.4f}s\n\n")

        # MACHINE-READABLE METRICS (KEY-VALUE)
        f.write("="*80 + "\n")
        f.write("MACHINE-READABLE METRICS (KEY-VALUE)\n")
        f.write("="*80 + "\n")
        for k, v in [
            ("quantization_type",  "FAKE_SIMULATED"),
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
        f.write(f"post_fake_quant_size_mb: {post_fake_size_mb:.2f}\n")
        f.write(f"simulated_compression_ratio: {compression_ratio:.3f}\n")
        f.write(f"simulated_reduction_pct: {reduction_pct:.2f}\n")
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
            b = layer_bits_map[i]
            f.write(f"  layer_{i:02d}: {b:2d}-bit (simulated)  "
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
        f.write(f"  -> needs higher bit-width to preserve accuracy.\n")
        f.write(f"  Lower score  -> smaller magnitude -> less sensitive.\n\n")
        f.write(f"Quantization: FAKE (Simulated)\n")
        f.write(f"  Each weight: FP32 -> round to {cluster_bits} levels -> dequantize\n")
        f.write(f"  -> stored back as FP32. Captures rounding error numerically\n")
        f.write(f"  but uses no fewer bytes. Standard approach for measuring\n")
        f.write(f"  accuracy impact without hardware deployment.\n\n")
        f.write(f"Group-Wise Quantization:\n")
        f.write(f"  group_size={group_size}: every {group_size} consecutive weights\n")
        f.write(f"  share one scale factor. Same approach as GPTQ.\n\n")
        f.write(f"Dataset: WikiText-2 Test Set\n")
        f.write(f"  Language modeling benchmark.\n")
        f.write(f"  Metric: Perplexity (lower is better).\n")
        f.write(f"  Additional metric: throughput in tokens/s.\n")
        f.write(f"  Note: Phase 1 calibration used train split.\n")
        f.write(f"        Phase 2 evaluation always uses test split.\n\n")
        f.write(f"Clustering: {sn} with {n_clusters} groups\n")
        for rank, (cid, cmean) in enumerate(cm):
            lids = [i for i in range(num_layers) if labels[i] == cid]
            f.write(f"  Cluster {cid} (rank {rank}, bits={cluster_bits[rank]}): "
                    f"{len(lids)} layers, mean sensitivity={cmean:.6f}\n")
        f.write(f"\nModel saved in HuggingFace .bin sharded format:\n")
        f.write(f"  Folder: {model_save_path}/\n")
        f.write(f"  config.json              -- model architecture & hyperparameters\n")
        f.write(f"  tokenizer.json           -- tokenizer vocabulary & config\n")
        f.write(f"  pytorch_model-*.bin      -- weight shards (500MB each, FP32)\n")
        f.write(f"  pytorch_model.bin.index.json -- shard index\n")
        f.write(f"  quant_config.json        -- PMPQ quantization configuration\n")

    print(f"  Results log saved to: {log_path}")
    print(f"\nPhase 2 (FAKE) complete!")


if __name__ == "__main__":
    main()