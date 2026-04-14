"""
Phase 2: Pruning-Based Mixed-Precision Quantization (PMPQ) for TinyLlama on HellaSwag
======================================================================================
** Intel Gaudi HPU Version — Optimum-Habana + Direct HPU Inference **

Uses Phase 1 sensitivity scores to assign mixed-precision bit-widths per layer,
then evaluates both baseline and quantized model accuracy on HellaSwag.

Dataset: HellaSwag (https://huggingface.co/datasets/Rowan/hellaswag)
Task: Commonsense Reasoning (4-choice completion)
Method: PMPQ (Symmetric MinMax Group-Wise PTQ)
Platform: Intel Gaudi HPU (SDSC Voyager) via optimum-habana

Split Usage: VALIDATION split (10,042 samples - FULL dataset)
  Note: Uses complete validation set for accurate final metrics.
  Test labels not public. Train (2,500 subset) for sensitivity, validation (full) for evaluation.

Compute device summary:
  - Model inference (forward pass)      → HPU (direct, lazy mode)
  - Weight quantization (scale+round)   → CPU (one-time, ~1s total)
  - Clustering (K-means, hierarchical)  → CPU (sklearn, ~<1ms)
  - Dataset loading                     → CPU
  - Log-likelihood computation          → CPU
  - I/O, logging, result saving         → CPU

Pipeline:
    STEP 1: Load sensitivity file from Phase 1
    STEP 2: Load tokenizer
    STEP 3: Clustering + bit allocation
    STEP 4: Configure quantization
    STEP 5: Evaluate FP32 baseline accuracy on HellaSwag (HPU)
    STEP 6: Apply group-wise mixed-precision quantization (CPU)
    STEP 7: Evaluate quantized model accuracy on HellaSwag (HPU)
    STEP 8-9: Report results + save
    STEP 10: Interactive text generation

Usage (single-card):
    PT_HPU_LAZY_MODE=1 python PMPQ_evaluation_hellaswag_hpu.py

Author: Mixed-Precision Quantization Team
Date: 2025
"""

                   


import os
import re
import sys
import logging


HF_HOME = "/voyager/ceph/users/ananda2/Quantization/.cache/huggingface"
os.makedirs(HF_HOME, exist_ok=True)

os.environ.update({
    "HF_HOME": HF_HOME,
    "HF_DATASETS_CACHE": os.path.join(HF_HOME, "datasets"),
    "HF_HUB_CACHE": os.path.join(HF_HOME, "hub"),
    "TRANSFORMERS_CACHE": os.path.join(HF_HOME, "hub"),
    "TOKENIZERS_PARALLELISM": "false",
})


import gc
import json
import math
import time
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import chain

import torch
import torch.nn as nn
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import habana_frameworks.torch.core as htcore
    import habana_frameworks.torch.hpu as hpu
    HAS_HPU = True
except ImportError:
    HAS_HPU = False

from sklearn.cluster import KMeans, AgglomerativeClustering
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

adapt_transformers_to_gaudi()

warnings.filterwarnings('ignore')

LOG_DIR_BASE = "/voyager/ceph/users/ananda2/Quantization/logs"
os.makedirs(LOG_DIR_BASE, exist_ok=True)


               


def setup_logging(rank=0, log_dir=None):
    """Set up dual logging: console + file.

    Log file is saved to log_dir with a timestamp so each run is preserved.
    Only rank 0 logs to file in distributed mode.
    """
    if log_dir is None:
        log_dir = LOG_DIR_BASE
    logger = logging.getLogger("PMPQ_HELLASWAG_EVALUATION")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                   datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"hellaswag_evaluation_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Log file: {log_file}")

    return logger


                      


TINYLLAMA_MODELS = {
    "TinyLlama-1.1B": {
        "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "num_layers": 22,
        "hidden_dim": 2048,
        "description": "Compact 1.1B model trained on 3T tokens"
    }
}

DEFAULT_GROUP_SIZE = 128


                   


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0

def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1

def broadcast_object(obj, rank):
    if not dist.is_initialized():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if HAS_HPU:
        import habana_frameworks.torch.hpu.random as hrand
        hrand.manual_seed_all(seed)

def prompt_user(prompt_text, options, default=None):
    """Display a prompt and get user input (rank 0 only)."""
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options, 1):
        marker = " ← default" if opt == default else ""
        print(f"  {i}. {opt}{marker}")
    while True:
        try:
            sel = input(f"Select (1-{len(options)})[Enter={options.index(default)+1 if default else 1}]: ").strip()
            if not sel and default:
                return default
            idx = int(sel) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except (ValueError, EOFError, KeyboardInterrupt):
            pass
        print("Invalid selection.")

def print_section(title, rank=0):
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

def bits_to_mb(bits: int):
    return bits / (8 * 1024 * 1024)

def get_hpu_memory_info():
    try:
        stats = torch.hpu.memory_stats()
        return {
            "allocated_mb": stats.get("InUse", 0) / (1024 * 1024),
            "max_allocated_mb": stats.get("MaxInUse", 0) / (1024 * 1024),
        }
    except Exception:
        return None

def free_hpu_memory():
    gc.collect()
    if HAS_HPU:
        try:
            torch.hpu.empty_cache()
        except Exception:
            pass

def get_device_name():
    try:
        import habana_frameworks.torch.hpu as hpu_mod
        return f"Gaudi HPU ({hpu_mod.device_count()} cards)"
    except Exception:
        return "HPU (unknown)"

def log_hpu_status(logger):
    try:
        import habana_frameworks.torch.hpu as hpu_mod
        logger.info(f"HPU device count: {hpu_mod.device_count()}")
        logger.info(f"HPU current device: {hpu_mod.current_device()}")
        mem = get_hpu_memory_info()
        if mem:
            logger.info(f"HPU memory: {mem['allocated_mb']:.1f} MB allocated, "
                         f"{mem['max_allocated_mb']:.1f} MB peak")
    except Exception as e:
        logger.info(f"HPU status check failed: {e}")



                                       


def kmeans_clustering(sensitivities, n_clusters=3):
    """K-means clustering on sensitivity values. Runs on CPU (sklearn)."""
    X = sensitivities.reshape(-1, 1)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
    labels = km.labels_
    cluster_means = sorted([(i, sensitivities[labels == i].mean()) for i in range(n_clusters)],
                            key=lambda x: x[1], reverse=True)
    return labels, cluster_means

def hierarchical_clustering(sensitivities, n_clusters=3):
    """Hierarchical clustering on sensitivity values. Runs on CPU (sklearn)."""
    X = sensitivities.reshape(-1, 1)
    hc = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
    labels = hc.labels_
    cluster_means = sorted([(i, sensitivities[labels == i].mean()) for i in range(n_clusters)],
                            key=lambda x: x[1], reverse=True)
    return labels, cluster_means

def percentile_clustering(sensitivities, n_clusters=3):
    """Percentile-based clustering on sensitivity values. Runs on CPU."""
    sorted_indices = np.argsort(sensitivities)[::-1]
    labels = np.zeros(len(sensitivities), dtype=int)
    chunk_size = len(sensitivities) // n_clusters
    remainder = len(sensitivities) % n_clusters

    start = 0
    for cluster_id in range(n_clusters):
        end = start + chunk_size + (1 if cluster_id < remainder else 0)
        labels[sorted_indices[start:end]] = cluster_id
        start = end

    cluster_means = sorted([(i, sensitivities[labels == i].mean()) for i in range(n_clusters)],
                            key=lambda x: x[1], reverse=True)
    return labels, cluster_means



                                                              


class LinearSymmetricGroupQuant(nn.Module):
    """
    Symmetric MinMax Group-Wise Weight Quantization (PTQ).

    Quantization of weights happens at construction time on CPU.
    The forward pass (F.linear) is a standard matrix multiply that
    runs on whatever device the input tensor is on (HPU during eval).

    This is SIMULATED (fake) quantization:
    FP32 weights → quantize (round/clamp to N-bit grid) → dequantize back to FP32
    All computation still happens in FP32.
    """

    def __init__(self, linear: nn.Linear, nbits_w: int, group_size: int = 128):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.nbits_w = int(nbits_w)
        self.group_size = group_size
        quantized_weight = self._quantize_weight_groupwise(linear.weight.detach().clone())
        self.register_buffer("weight", quantized_weight)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.register_buffer("bias", None)

    def _quantize_weight_groupwise(self, weight):
        """Quantize weights on CPU. Called once at construction time."""
        if self.nbits_w >= 32:
            return weight
        out_features, in_features = weight.shape
        original_in_features = in_features
        if in_features % self.group_size != 0:
            pad_size = self.group_size - (in_features % self.group_size)
            weight = torch.nn.functional.pad(weight, (0, pad_size), value=0)
            in_features = weight.shape[1]
        num_groups = in_features // self.group_size
        weight_grouped = weight.view(out_features, num_groups, self.group_size)
        qmin = -(2 ** (self.nbits_w - 1))
        qmax = (2 ** (self.nbits_w - 1)) - 1
        max_val = weight_grouped.abs().amax(dim=-1, keepdim=True)
        scale = max_val / qmax
        scale = torch.clamp(scale, min=1e-8)
        q = torch.round(weight_grouped / scale)
        q = torch.clamp(q, qmin, qmax)
        weight_quantized = q * scale
        weight_quantized = weight_quantized.view(out_features, -1)
        if weight_quantized.shape[1] > original_in_features:
            weight_quantized = weight_quantized[:, :original_in_features]
        return weight_quantized

    def calculate_weight_bits(self):
        num_weights = self.weight.numel()
        num_groups_per_row = (self.in_features + self.group_size - 1) // self.group_size
        num_groups = self.out_features * num_groups_per_row
        orig_bits = int(num_weights * 32)
        quant_bits = int(num_weights * self.nbits_w + num_groups * 16)
        return orig_bits, quant_bits

    def forward(self, x):
        """Forward pass — runs on HPU when model is placed there."""
        return nn.functional.linear(x, self.weight, self.bias)


def set_module_by_qualname(root: nn.Module, qualname: str, new_mod: nn.Module):
    parts = qualname.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def quantize_model_layers(model, layer_bits_map, group_size=128, rank=0, logger=None):
    """
    Quantize linear layers with GROUP-WISE quantization.

    Runs on CPU — weights are quantized (scale, round, clamp) once.
    The resulting model has LinearSymmetricGroupQuant modules with buffers
    that will be moved to HPU when model.to("hpu") is called.
    """
    total_orig, total_quant = 0, 0
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            m = re.search(r"(?:model\.|decoder\.)layers\.(\d+)\.", name)
            if m:
                layer_idx = int(m.group(1))
                nbits = layer_bits_map.get(layer_idx, 8)
                targets.append((name, module, nbits))

    if rank == 0 and logger:
        logger.info(f"  Found {len(targets)} Linear layers to quantize (CPU)")
        logger.info(f"  Group-wise quantization: group_size={group_size}")

    for qualname, linear_mod, nbits in targets:
        wrapper = LinearSymmetricGroupQuant(linear_mod, nbits, group_size=group_size)
        set_module_by_qualname(model, qualname, wrapper)
        o, q = wrapper.calculate_weight_bits()
        total_orig += o
        total_quant += q

    return model, total_orig, total_quant



                                


def preprocess_hellaswag_text(text):
    """Clean up HellaSwag text artifacts from WikiHow portion."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def load_hellaswag_dataset(split="validation"):
    """Load HellaSwag dataset. Validation has ~10042 samples."""
    dataset = load_dataset("Rowan/hellaswag", split=split, trust_remote_code=True)
    return dataset


def evaluate_hellaswag_accuracy(model, dataset, tokenizer, logger,
                                 eval_name="HellaSwag", max_samples=0, batch_size=64):
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

    Returns: (accuracy_percent, eval_time_seconds)
    """
    device = torch.device("hpu")
    model.to(device)
    model.eval()

    samples = dataset
    if max_samples > 0:
        n = min(max_samples, len(dataset))
        samples = dataset.select(range(n))

    correct = 0
    total = len(samples)
    rank = get_rank()

    if rank == 0:
        logger.info(f"[{eval_name}] Evaluating {total} samples on HPU (FP32, lazy mode, batch_size={batch_size})...")

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

                              
            if rank == 0 and (batch_end % 500 == 0 or batch_end == total):
                logger.info(f"  [{eval_name}] {batch_end}/{total}: "
                             f"acc={correct/batch_end*100:.2f}%")

    eval_time = time.time() - t0
    accuracy = correct / total * 100.0

    if rank == 0:
        logger.info(f"[{eval_name}] Final: {accuracy:.2f}% ({correct}/{total}), time={eval_time:.1f}s")

                                                       
    model.to("cpu")
    free_hpu_memory()

    if rank == 0:
        mem = get_hpu_memory_info()
        if mem:
            logger.info(f"[{eval_name}] HPU after cleanup: {mem['allocated_mb']:.1f} MB allocated, "
                         f"{mem['max_allocated_mb']:.1f} MB peak")

    return accuracy, eval_time



               


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate mixed-precision quantization on HellaSwag (PMPQ) - Gaudi HPU'
    )
    parser.add_argument('--gaudi_version', type=str, default=None,
                        choices=['Gaudi1', 'Gaudi2'],
                        help='Gaudi version for organizing output folders (default: prompt)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for HellaSwag evaluation inference (default: 64)')
    args = parser.parse_args()

    rank = get_rank()
    world_size = get_world_size()
    set_seed(42)

                                                   
    if rank == 0:
        if args.gaudi_version:
            gaudi_version = args.gaudi_version
        else:
            print("\nSelect Gaudi version for output organization:")
            print("  1. Gaudi1")
            print("  2. Gaudi2")
            while True:
                try:
                    sel = input("Select (1-2) [default=1]: ").strip()
                    if not sel or sel == "1":
                        gaudi_version = "Gaudi1"
                        break
                    if sel == "2":
                        gaudi_version = "Gaudi2"
                        break
                    print("Invalid selection. Enter 1 or 2.")
                except (EOFError, KeyboardInterrupt):
                    gaudi_version = "Gaudi1"
                    break
        print(f"  → Output will be saved under '{gaudi_version}' subfolders")
    else:
        gaudi_version = None
    gaudi_version = broadcast_object(gaudi_version, rank)

    LOG_DIR = os.path.join(LOG_DIR_BASE, gaudi_version)
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = setup_logging(rank, log_dir=LOG_DIR)

    if rank == 0:
        logger.info("=" * 80)
        logger.info("PMPQ PHASE 2: MIXED-PRECISION QUANTIZATION EVALUATION (HellaSwag)")
        logger.info("=" * 80)
        logger.info(f"Cache:     {HF_HOME}")
        logger.info(f"Log dir:   {LOG_DIR}")
        mode_str = f"Distributed: {world_size} Gaudi cards" if world_size > 1 else "Single-card"
        logger.info(f"Mode:      {mode_str}")
        logger.info(f"Metric:    Accuracy (acc_norm)")
        logger.info(f"Inference: Direct HPU (lazy mode + FP32)")
        logger.info("")

        logger.info("Compute Device Mapping:")
        logger.info("  Model inference (forward pass)     → HPU (direct, lazy mode)")
        logger.info("  Weight quantization (scale+round)  → CPU (one-time)")
        logger.info("  Clustering (K-means, etc.)         → CPU (sklearn)")
        logger.info("  Dataset loading                    → CPU")
        logger.info("  Log-likelihood computation         → CPU")
        logger.info("  I/O / logging / result saving      → CPU")
        logger.info("")

        log_hpu_status(logger)
        logger.info("")

        logger.info(f"PyTorch:        {torch.__version__}")
        try:
            import optimum.habana
            logger.info(f"optimum-habana: {optimum.habana.__version__}")
        except Exception:
            logger.info("optimum-habana: unknown version")
        try:
            import habana_frameworks.torch as ht
            logger.info(f"habana_frameworks available: Yes")
        except Exception:
            logger.info("habana_frameworks available: No")
        logger.info(f"PT_HPU_LAZY_MODE: {os.environ.get('PT_HPU_LAZY_MODE', 'not set')}")
        logger.info("")

    print_section("PHASE 2: PMPQ EVALUATION (HellaSwag) - Gaudi HPU", rank)

                                                                                  
    if rank == 0:
                                       
        print_section("STEP 1: LOAD PRUNING SENSITIVITY FILE", rank)
        sens_dir = Path(os.path.join("Sensitivities", gaudi_version))
        if not sens_dir.exists():
                                                                               
            sens_dir = Path("Sensitivities")
        if not sens_dir.exists():
            logger.error(f"'Sensitivities/{gaudi_version}' folder not found! Run Phase 1 first.")
            broadcast_object(None, rank)
            if dist.is_initialized():
                dist.destroy_process_group()
            return

                                                           
        sens_files = list(sens_dir.glob("sens_*HellaSwag*.json"))
        if not sens_files:
                                                
            sens_files = list(sens_dir.glob("sens_*.json"))
        if not sens_files:
            logger.error("No sensitivity files found!")
            broadcast_object(None, rank)
            if dist.is_initialized():
                dist.destroy_process_group()
            return

        sens_file_names = [f.name for f in sens_files]
        logger.info(f"Found {len(sens_file_names)} sensitivity file(s):")
        for i, fname in enumerate(sens_file_names, 1):
            logger.info(f"  {i}. {fname}")

        selected_file = prompt_user("Select sensitivity file:", sens_file_names, default=sens_file_names[0])

        sens_path = sens_dir / selected_file
        with open(sens_path, "r") as f:
            sensitivities = json.load(f)

        num_layers = max([int(k.split("_")[1]) for k in sensitivities.keys()]) + 1
        sens_values = np.array([sensitivities[f"layer_{i}"] for i in range(num_layers)], dtype=np.float32)

        logger.info(f"Loaded: {selected_file}")
        logger.info(f"  Layers: {num_layers}")
        logger.info(f"  Sensitivity range: [{sens_values.min():.4f}, {sens_values.max():.4f}]")

                      
        model_key = None
        for key, config in TINYLLAMA_MODELS.items():
            if config["num_layers"] == num_layers:
                model_key = key
                break
        if model_key is None:
            model_key = "TinyLlama-1.1B"
        model_name = TINYLLAMA_MODELS[model_key]["model_name"]

                            
        print_section("STEP 3: CLUSTERING CONFIGURATION", rank)
        clustering_choice = prompt_user(
            "Select clustering strategy:",
            ["K-means (recommended)", "Percentile bucketing", "Hierarchical clustering"],
            default="K-means (recommended)"
        )
        n_groups_choice = prompt_user(
            "Select number of groups:",
            ["3 groups (simpler)", "4 groups (finer control)"],
            default="3 groups (simpler)"
        )
        n_clusters = int(n_groups_choice.split()[0])

        logger.info(f"Clustering on CPU (sklearn): {clustering_choice}, {n_clusters} groups")

        if "K-means" in clustering_choice:
            labels, cluster_means = kmeans_clustering(sens_values, n_clusters=n_clusters)
            strategy_name = "kmeans"
        elif "Percentile" in clustering_choice:
            labels, cluster_means = percentile_clustering(sens_values, n_clusters=n_clusters)
            strategy_name = "percentile"
        else:
            labels, cluster_means = hierarchical_clustering(sens_values, n_clusters=n_clusters)
            strategy_name = "hierarchical"

                                
        print_section("STEP 4: BIT-WIDTH ALLOCATION", rank)
        if n_clusters == 3:
            bit_options = [
                "[16, 8, 8] (safe - recommended for LLMs)",
                "[16, 12, 8] (balanced quality & compression)",
                "[16, 8, 6] (moderate compression)",
                "[16, 8, 4] (aggressive - may degrade)",
                "[8, 6, 4] (very aggressive)",
                "[8, 4, 2] (extreme - not recommended)",
                "Custom"
            ]
        else:
            bit_options = [
                "[16, 12, 8, 8] (safe - recommended for LLMs)",
                "[16, 12, 8, 6] (balanced)",
                "[32, 16, 8, 4] (keep top cluster in FP32)",
                "[16, 8, 6, 4] (moderate)",
                "[16, 8, 4, 2] (aggressive)",
                "[8, 4, 2, 2] (extreme - not recommended)",
                "Custom"
            ]

        bit_choice = prompt_user(
            "Select bit allocation (high->low pruning sensitivity):",
            bit_options, default=bit_options[0]
        )

        if "Custom" in bit_choice:
            bits_str = input(f"Enter bits as comma-separated (e.g., '{','.join(str(b) for b in [16,8,8][:n_clusters])}'): ").strip()
            cluster_bits = [int(b.strip()) for b in bits_str.split(",")]
        else:
            import ast
            match = re.search(r'\[.*?\]', bit_choice)
            cluster_bits = ast.literal_eval(match.group())

                    
        group_size_choice = prompt_user(
            "Select quantization group size:",
            ["128 (standard, same as GPTQ)", "64 (finer granularity)", "32 (finest)"],
            default="128 (standard, same as GPTQ)"
        )
        if "64" in group_size_choice:
            group_size = 64
        elif "32" in group_size_choice:
            group_size = 32
        else:
            group_size = 128

                              
        layer_bits_map = {}
        for i in range(num_layers):
            cluster_id = labels[i]
            lbl_rank = next(j for j, (cid, _) in enumerate(cluster_means) if cid == cluster_id)
            layer_bits_map[i] = cluster_bits[lbl_rank]

        config = {
            'selected_file': selected_file,
            'sensitivities': sensitivities,
            'num_layers': num_layers,
            'sens_values': sens_values.tolist(),
            'model_key': model_key,
            'model_name': model_name,
            'strategy_name': strategy_name,
            'n_clusters': n_clusters,
            'cluster_bits': cluster_bits,
            'group_size': group_size,
            'layer_bits_map': layer_bits_map,
            'labels': labels.tolist(),
            'cluster_means': cluster_means,
        }

        logger.info(f"Layer bit allocation:")
        logger.info(f"  Group size: {group_size}")
        for i in range(num_layers):
            logger.info(f"  Layer {i:2d}: {layer_bits_map[i]:2d}-bit (sensitivity: {sens_values[i]:.4f})")
    else:
        config = None

    config = broadcast_object(config, rank)

    if config is None:
        if dist.is_initialized():
            dist.destroy_process_group()
        return

                   
    selected_file = config['selected_file']
    num_layers = config['num_layers']
    sens_values = np.array(config['sens_values'], dtype=np.float32)
    model_key = config['model_key']
    model_name = config['model_name']
    strategy_name = config['strategy_name']
    n_clusters = config['n_clusters']
    cluster_bits = config['cluster_bits']
    group_size = config['group_size']
    layer_bits_map = {int(k): v for k, v in config['layer_bits_map'].items()}
    labels = np.array(config['labels'])
    cluster_means = config['cluster_means']

                                                  
    print_section("STEP 2: LOADING TOKENIZER", rank)
    if rank == 0:
        logger.info(f"Loading tokenizer for {model_key} from {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        logger.info(f"  Tokenizer vocab: {tokenizer.vocab_size}")

                                                  
    if rank == 0:
        logger.info("Loading HellaSwag dataset...")
    dataset = load_hellaswag_dataset(split="validation")

    if rank == 0:
        logger.info(f"HellaSwag validation: {len(dataset)} samples")
        logger.info(f"  Official TinyLlama score: 59.20 (acc_norm)")

                                                          
    print_section("STEP 5: EVALUATING FP32 BASELINE (HPU)", rank)
    if rank == 0:
        logger.info("Loading BASELINE model (separate from quantized model)...")
        logger.info("  Using separate models to avoid Habana graph cache interference")

    baseline_model = AutoModelForCausalLM.from_pretrained(model_name)

    if rank == 0:
        logger.info(f"  Baseline model dtype: {next(baseline_model.parameters()).dtype}")
        logger.info("Computing FP32 baseline accuracy on HellaSwag (HPU)...")

    acc_before, eval_time_before = evaluate_hellaswag_accuracy(
        baseline_model, dataset, tokenizer, logger, eval_name="FP32 Baseline",
        batch_size=args.batch_size
    )

                                   
    num_samples = len(dataset)
    throughput_before = num_samples / eval_time_before                  

    if rank == 0:
        logger.info(f"FP32 Baseline: Accuracy = {acc_before:.2f}% ({eval_time_before:.1f}s)")
        logger.info(f"               Throughput = {throughput_before:.2f} samples/s")

                         
    MODELS_DIR = "/voyager/ceph/users/ananda2/Quantization/Models"
    baseline_save_dir = os.path.join(MODELS_DIR, f"{model_key}_baseline")
    if rank == 0:
        os.makedirs(baseline_save_dir, exist_ok=True)
        baseline_model.save_pretrained(baseline_save_dir)
        tokenizer.save_pretrained(baseline_save_dir)
        logger.info(f"Baseline model saved: {baseline_save_dir}")

    del baseline_model
    free_hpu_memory()

                                                                         
    print_section("STEP 6: APPLYING GROUP-WISE MIXED-PRECISION QUANTIZATION (CPU)", rank)

    if rank == 0:
        logger.info("Loading FRESH model for quantization (separate from baseline)...")

    quant_model = AutoModelForCausalLM.from_pretrained(model_name)

    if rank == 0:
        logger.info("Quantizing model weights on CPU...")

    t0 = time.time()
    quant_model, orig_bits, quant_bits = quantize_model_layers(
        quant_model, layer_bits_map, group_size=group_size, rank=rank, logger=logger
    )
    quantize_time_s = time.time() - t0

    compression_ratio = orig_bits / quant_bits if quant_bits > 0 else float("inf")
    reduction_pct = 100.0 * (1.0 - quant_bits / orig_bits) if orig_bits > 0 else 0.0

    if rank == 0:
        logger.info(f"Quantization complete in {quantize_time_s:.2f}s (CPU)")
        logger.info(f"  Linear layers: {bits_to_mb(orig_bits):.2f} MB → {bits_to_mb(quant_bits):.2f} MB ({compression_ratio:.2f}x)")
        logger.info(f"  NOTE: Compression ratio covers nn.Linear weights only (the quantized scope),")
        logger.info(f"        not embeddings, LayerNorm, or the LM head.")

                                                            
    print_section("STEP 7: EVALUATING QUANTIZED MODEL (HPU)", rank)

    if rank == 0:
        logger.info("Evaluating quantized model accuracy on HellaSwag (HPU)...")

    acc_after, eval_time_after = evaluate_hellaswag_accuracy(
        quant_model, dataset, tokenizer, logger, eval_name="Quantized Model",
        batch_size=args.batch_size
    )

                                    
    throughput_after = num_samples / eval_time_after                  

                          
    bits_str = '_'.join(str(b) for b in cluster_bits)
    quant_save_dir = os.path.join(MODELS_DIR, f"{model_key}_quantized_{strategy_name}_bits{bits_str}_g{group_size}")
    if rank == 0:
        os.makedirs(quant_save_dir, exist_ok=True)
        quant_model.save_pretrained(quant_save_dir)
        tokenizer.save_pretrained(quant_save_dir)
        quant_config = {
            'model_name': model_name,
            'strategy': strategy_name,
            'cluster_bits': cluster_bits,
            'group_size': group_size,
            'layer_bits_map': {str(k): v for k, v in layer_bits_map.items()},
            'n_clusters': n_clusters,
            'benchmark': 'hellaswag',
            'acc_baseline': float(acc_before),
            'acc_quantized': float(acc_after),
            'throughput_baseline_samples_per_s': float(throughput_before),
            'throughput_quantized_samples_per_s': float(throughput_after),
            'compression_ratio': float(compression_ratio),
        }
        with open(os.path.join(quant_save_dir, 'quant_config.json'), 'w') as f:
            json.dump(quant_config, f, indent=2)
        logger.info(f"Quantized model saved: {quant_save_dir}")
        logger.info(f"  NOTE: Saved weights are FP32 dequantized (quantize→round→dequantize). ")
        logger.info(f"        The file is NOT smaller on disk — this is simulated (fake) quantization.")

                                                    
    if rank == 0:
        acc_drop = acc_before - acc_after
        acc_drop_pct = (acc_drop / acc_before * 100.0) if acc_before > 0 else 0.0

        logger.info("=" * 70)
        logger.info("EVALUATION RESULTS (PMPQ + GROUP-WISE) — HellaSwag")
        logger.info("=" * 70)
        logger.info(f"  Model:           {model_key}")
        logger.info(f"  Benchmark:       HellaSwag (acc_norm)")
        logger.info(f"  Clustering:      {strategy_name} | Bits: {cluster_bits} | Group: {group_size}")
        logger.info(f"  Cards:           {world_size}")
        logger.info(f"  Platform:        Intel Gaudi HPU (direct inference, FP32)")
        logger.info(f"  Official Score:  59.20%")
        logger.info(f"  FP32 Accuracy:   {acc_before:.2f}%")
        logger.info(f"  Quant Accuracy:  {acc_after:.2f}%")
        logger.info(f"  Degradation:     {acc_drop:+.2f}pp ({acc_drop_pct:+.2f}%)")
        logger.info(f"  Compression:     {compression_ratio:.2f}x ({bits_to_mb(orig_bits):.2f} → {bits_to_mb(quant_bits):.2f} MB)")
        logger.info(f"  Baseline Throughput:  {throughput_before:.2f} samples/s")
        logger.info(f"  Quantized Throughput: {throughput_after:.2f} samples/s")
        speedup = throughput_after / throughput_before if throughput_before > 0 else 1.0
        logger.info(f"  Speedup:         {speedup:.2f}x")

        mem = get_hpu_memory_info()
        if mem:
            logger.info(f"  HPU peak:        {mem['max_allocated_mb']:.1f} MB")

        if acc_drop <= 1.0:
            quality = "EXCELLENT - Minimal degradation"
        elif acc_drop <= 3.0:
            quality = "GOOD - Acceptable degradation"
        elif acc_drop <= 5.0:
            quality = "MODERATE - Noticeable degradation"
        else:
            quality = "POOR - Significant degradation"
        logger.info(f"  Quality:         {quality}")

                      
        logger.info("=" * 70)
        logger.info("SAVING RESULTS")
        logger.info("=" * 70)

        eval_dir = os.path.join("Evaluation", gaudi_version)
        os.makedirs(eval_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"ptq_eval_{model_key}_HellaSwag_PMPQ_GroupWise_{strategy_name}_{timestamp}.txt"
        log_path = os.path.join(eval_dir, log_filename)

        with open(log_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("MIXED-PRECISION PTQ EVALUATION RESULTS (PMPQ + GROUP-WISE) — HellaSwag\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Model: {model_key}\n")
            f.write(f"HuggingFace ID: {model_name}\n")
            f.write(f"Task: Commonsense Reasoning (HellaSwag, 4-choice)\n")
            f.write(f"Dataset: HellaSwag (Rowan/hellaswag), validation\n")
            f.write(f"Metric: Accuracy (acc_norm)\n")
            f.write(f"Method: PMPQ with Group-Wise Quantization (Simulated/Fake)\n")
            f.write(f"Sensitivity File: {selected_file}\n")
            f.write(f"Clustering Strategy: {strategy_name}\n")
            f.write(f"Number of Clusters: {n_clusters}\n")
            f.write(f"Bit Allocation: {cluster_bits}\n")
            f.write(f"Group Size: {group_size}\n")
            f.write(f"Device: {get_device_name()}\n")
            f.write(f"Platform: Intel Gaudi HPU (direct inference, lazy mode, FP32)\n")
            f.write(f"Num Accelerators: {world_size}\n")
            f.write(f"Samples: {len(dataset)}\n")
            f.write(f"Timestamp: {timestamp}\n")
            if mem:
                f.write(f"HPU Peak Memory: {mem['max_allocated_mb']:.1f} MB\n")
            f.write("\n")

            f.write("-" * 70 + "\n")
            f.write("BASELINE (FP32)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy: {acc_before:.4f}%\n")
            f.write(f"Official TinyLlama: 59.20%\n")
            f.write(f"Eval Time: {eval_time_before:.2f}s\n")
            f.write(f"Throughput: {throughput_before:.2f} samples/s\n\n")

            f.write("-" * 70 + "\n")
            f.write("QUANTIZED (Mixed-Precision PTQ via PMPQ + Group-Wise)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy: {acc_after:.4f}%\n")
            f.write(f"Eval Time: {eval_time_after:.2f}s\n")
            f.write(f"Throughput: {throughput_after:.2f} samples/s\n")
            f.write(f"Speedup: {speedup:.2f}x\n\n")

            f.write("-" * 70 + "\n")
            f.write("DEGRADATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy Drop: {acc_drop:+.4f}pp ({acc_drop_pct:+.2f}%)\n")
            f.write(f"Quality: {quality}\n\n")

            f.write("-" * 70 + "\n")
            f.write("COMPRESSION (nn.Linear layers only)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Original Size: {bits_to_mb(orig_bits):.2f} MB (Linear weights only)\n")
            f.write(f"Quantized Size: {bits_to_mb(quant_bits):.2f} MB (Linear weights only)\n")
            f.write(f"Compression Ratio: {compression_ratio:.2f}x\n")
            f.write(f"Size Reduction: {reduction_pct:.1f}%\n")
            f.write(f"Note: Excludes embeddings, LayerNorm, and LM head (not quantized).\n")
            f.write(f"Note: Saved model is FP32 dequantized (simulated quantization).\n\n")

            f.write("-" * 70 + "\n")
            f.write("LAYER BIT ALLOCATION\n")
            f.write("-" * 70 + "\n")
            for i in range(num_layers):
                f.write(f"Layer {i:2d}: {layer_bits_map[i]:2d}-bit (sensitivity: {sens_values[i]:.4f})\n")

        logger.info(f"Results saved: {log_path}")

                      
        summary_log = os.path.join(LOG_DIR, f"hellaswag_evaluation_results_{timestamp}.txt")
        with open(summary_log, "w") as f:
            f.write(f"PMPQ HellaSwag Evaluation Results — {timestamp}\n")
            f.write(f"Model: {model_key}\n")
            f.write(f"Clustering: {strategy_name} | Bits: {cluster_bits} | Group: {group_size}\n")
            f.write(f"Cards: {world_size}\n")
            f.write(f"Official:      59.20%\n")
            f.write(f"FP32 Accuracy: {acc_before:.2f}%\n")
            f.write(f"Quant Accuracy:{acc_after:.2f}%\n")
            f.write(f"Drop: {acc_drop:+.2f}pp\n")
            f.write(f"Compression: {compression_ratio:.2f}x\n")
            f.write(f"Quality: {quality}\n")

        logger.info(f"Summary saved: {summary_log}")
        logger.info(f"Baseline model: {baseline_save_dir}")
        logger.info(f"Quantized model: {quant_save_dir}")
        logger.info("")
        logger.info("Phase 2 HellaSwag evaluation complete!")

                                                                
    if rank == 0 and not dist.is_initialized():
        print_section("STEP 10: INTERACTIVE TEXT GENERATION", rank)

        MODELS_DIR = "/voyager/ceph/users/ananda2/Quantization/Models"
        available_models = {}
        if os.path.isdir(MODELS_DIR):
            for name in sorted(os.listdir(MODELS_DIR)):
                model_path = os.path.join(MODELS_DIR, name)
                if os.path.isdir(model_path):
                    qconfig_path = os.path.join(model_path, "quant_config.json")
                    if os.path.exists(qconfig_path):
                        with open(qconfig_path) as f:
                            qcfg = json.load(f)
                        acc_key = qcfg.get('acc_quantized', qcfg.get('ppl_quantized', '?'))
                        label = f"{name} (Acc={acc_key}, {qcfg.get('compression_ratio', '?'):.1f}x compression)"
                    elif "baseline" in name.lower():
                        label = f"{name} (FP32 baseline)"
                    else:
                        label = name
                    available_models[name] = {"path": model_path, "label": label}

        if not available_models:
            logger.info("No saved models found in Models/ directory.")
        else:
            model_list = list(available_models.items())

            print(f"\n{'='*70}")
            print("  AVAILABLE MODELS")
            print(f"{'='*70}")
            for i, (name, info) in enumerate(model_list, 1):
                print(f"  {i}. {info['label']}")
            print(f"{'='*70}")

            loaded_models = {}
            model_names = []

            while True:
                try:
                    sel = input(f"\nSelect model to load (1-{len(model_list)}), or 'done' to start generating: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if sel.lower() == 'done':
                    break
                try:
                    idx = int(sel) - 1
                    if 0 <= idx < len(model_list):
                        name, info = model_list[idx]
                        if name not in loaded_models:
                            print(f"  Loading {name} → HPU...")
                            mdl = AutoModelForCausalLM.from_pretrained(info['path'])
                            mdl = mdl.eval().to("hpu")
                            loaded_models[name] = mdl
                            model_names = list(loaded_models.keys())
                            print(f"  ✓ Loaded: {info['label']}")
                        else:
                            print(f"  Already loaded: {name}")
                    else:
                        print("Invalid selection.")
                except (ValueError, EOFError, KeyboardInterrupt):
                    break

            if loaded_models:
                active_name = model_names[0]
                active_model = loaded_models[active_name]

                print(f"\n{'='*70}")
                print("  INTERACTIVE GENERATION")
                print(f"{'='*70}")
                print(f"  Active model: {active_name}")
                print(f"  Commands: 'switch', 'compare', 'list', 'quit'")
                print(f"{'='*70}")

                while True:
                    try:
                        prompt = input(f"\n[{active_name}] Enter prompt (or 'quit'): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        break

                    if not prompt or prompt.lower() in ('quit', 'exit', 'q'):
                        break

                    if prompt.lower() == 'list':
                        print("\nLoaded models:")
                        for n in model_names:
                            marker = " ← active" if n == active_name else ""
                            print(f"  • {n}{marker}")
                        continue

                    if prompt.lower() == 'switch':
                        print("\nLoaded models:")
                        for i, n in enumerate(model_names, 1):
                            marker = " ← active" if n == active_name else ""
                            print(f"  {i}. {n}{marker}")
                        try:
                            sel = input(f"Select (1-{len(model_names)}): ").strip()
                            idx = int(sel) - 1
                            if 0 <= idx < len(model_names):
                                active_name = model_names[idx]
                                active_model = loaded_models[active_name]
                                print(f"Switched to: {active_name}")
                        except (ValueError, EOFError, KeyboardInterrupt):
                            print("Cancelled.")
                        continue

                    if prompt.lower() == 'compare':
                        prompt = input("Enter prompt for comparison: ").strip()
                        if not prompt:
                            continue
                        inputs = tokenizer(prompt, return_tensors="pt")
                        input_ids = inputs["input_ids"].to("hpu")
                        for n, mdl in loaded_models.items():
                            with torch.no_grad():
                                htcore.mark_step()
                                output_ids = mdl.generate(input_ids, max_new_tokens=100,
                                                          do_sample=True, temperature=0.7, top_p=0.9)
                                htcore.mark_step()
                            text = tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=True)
                            print(f"\n[{n} OUTPUT]:")
                            print(text)
                        continue

                                               
                    inputs = tokenizer(prompt, return_tensors="pt")
                    input_ids = inputs["input_ids"].to("hpu")
                    with torch.no_grad():
                        htcore.mark_step()
                        output_ids = active_model.generate(input_ids, max_new_tokens=100,
                                                            do_sample=True, temperature=0.7, top_p=0.9)
                        htcore.mark_step()
                    text = tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=True)
                    print(f"\n[{active_name} OUTPUT]:")
                    print(text)

             
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
