"""
Phase 2: Pruning-Based Mixed-Precision Quantization (PMPQ) for TinyLlama on WikiText
=====================================================================================
** Intel Gaudi HPU Version — Optimum-Habana + GaudiTrainer **

This script performs mixed-precision quantization evaluation:
1. Load pruning sensitivity results from Phase 1
2. Cluster layers by sensitivity
3. Assign bit-widths (higher bits for sensitive layers)
4. Apply group-wise quantization to each layer
5. Compare FP32 vs quantized perplexity on WikiText-2 TEST set

Dataset: WikiText-2 (https://huggingface.co/datasets/wikitext)
Task: Language Modeling (Perplexity)
Method: PMPQ (Symmetric MinMax Group-Wise PTQ)
Platform: Intel Gaudi HPU (SDSC Voyager) via optimum-habana

Split Usage: TEST split
  Note: Final evaluation on test split, validation used in Phase 1 for sensitivity.

Compute device summary:
  - Model inference (forward pass, loss)     → HPU via GaudiTrainer
  - Weight quantization (scale+round)        → CPU (one-time, ~1s total)
  - Clustering (K-means, hierarchical, etc.) → CPU (sklearn, ~<1ms)
  - Dataset tokenization                     → CPU
  - I/O, logging, result saving              → CPU

Launch (single-card):
    PT_HPU_LAZY_MODE=1 python PMPQ_evaluation_wikitext_hpu.py --gaudi_version Gaudi2

Launch (multi-card, 8 HPUs):
    PT_HPU_LAZY_MODE=1 python ../gaudi_spawn.py --world_size 8 --use_mpi PMPQ_evaluation_wikitext_hpu.py

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
    "TOKENIZERS_PARALLELISM": "false"
})

                                                        
LOG_DIR_BASE = "/voyager/ceph/users/ananda2/Quantization/logs"
os.makedirs(LOG_DIR_BASE, exist_ok=True)


             


import gc
import json
import math
import time
import random
import warnings
from pathlib import Path
from datetime import datetime
from itertools import chain
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from datasets import load_dataset
from sklearn.cluster import KMeans, AgglomerativeClustering
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator,
)

                                                                 
from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

adapt_transformers_to_gaudi()

warnings.filterwarnings('ignore')

               


def setup_logging(rank=0, log_dir=None):
    """Set up dual logging: console + file.
    
    Log file is saved to log_dir with a timestamp so each run is preserved.
    Only rank 0 logs to file in distributed mode.
    """
    if log_dir is None:
        log_dir = LOG_DIR_BASE
    logger = logging.getLogger("PMPQ_WIKITEXT_EVALUATION")
    logger.setLevel(logging.INFO)
    logger.handlers = []                               

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

                              
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

                                
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"evaluation_{timestamp}.log")
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




                   


def init_distributed():
    """
    Initialize torch.distributed if running under a multi-card launcher.

    Supports two launcher types:
      1. gaudi_spawn.py (MPI) — sets OMPI_COMM_WORLD_RANK, OMPI_COMM_WORLD_SIZE
      2. torchrun — sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT

    Must be called BEFORE get_rank(), broadcast_object(), or any interactive prompts.
    Without this, every process thinks it is rank 0 and tries to call input(),
    causing all non-rank-0 processes to hang (no stdin in MPI/torchrun).
    """
    if dist.is_initialized():
        return

                                                                                  
    mpi_rank = os.environ.get("OMPI_COMM_WORLD_RANK") or os.environ.get("PMI_RANK")
    mpi_size = os.environ.get("OMPI_COMM_WORLD_SIZE") or os.environ.get("PMI_SIZE")
    if mpi_rank is not None and mpi_size is not None:
        os.environ.setdefault("RANK", mpi_rank)
        os.environ.setdefault("WORLD_SIZE", mpi_size)
        os.environ.setdefault("LOCAL_RANK",
                              os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", mpi_rank))

                                                
    rank_env = os.environ.get("RANK")
    size_env = os.environ.get("WORLD_SIZE")

    if rank_env is not None and size_env is not None and int(size_env) > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="hccl")


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

def broadcast_object(obj, rank):
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return obj
    container = [obj]
    dist.broadcast_object_list(container, src=0)
    return container[0]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        from optimum.habana.utils import set_seed as gaudi_set_seed
        gaudi_set_seed(seed)
    except Exception:
        pass

def prompt_user(prompt_text, options, default=None):
    """Display a prompt and get user input (rank 0 only)."""
    print(f"\n{prompt_text}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        try:
            choice = input(f"Select (1-{len(options)}) [default: {options.index(default)+1 if default else 1}]: ").strip()
            if not choice:
                return default if default else options[0]
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print(f"[ERROR] Please enter 1-{len(options)}")
        except ValueError:
            print("[ERROR] Invalid input.")

def print_section(title, rank=0):
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

def bits_to_mb(bits: int):
    return bits / 8.0 / (1024.0 * 1024.0)

def get_hpu_memory_info():
    """Get HPU memory usage. Returns dict with used/total in MB, or None if unavailable."""
    try:
        import habana_frameworks.torch.hpu as hthpu
        if hthpu.is_available():
            allocated = hthpu.memory_allocated() / (1024 ** 2)      
            max_allocated = hthpu.max_memory_allocated() / (1024 ** 2)      
            return {
                "allocated_mb": allocated,
                "max_allocated_mb": max_allocated,
            }
    except Exception:
        pass
    return None


def free_hpu_memory():
    """
    Force-free HPU memory after model deletion.
    
    GaudiTrainer moves the model to HPU. When we `del model` on Python side,
    the HPU tensors may not be freed immediately. This function forces cleanup.
    """
    gc.collect()  
    try:
        if hasattr(torch, 'hpu') and hasattr(torch.hpu, 'empty_cache'):
            torch.hpu.empty_cache()
    except Exception:
        pass
    try:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step() 
    except Exception:
        pass

def get_device_name():
    """Return device name for logging."""
    try:
        import habana_frameworks.torch.hpu as hthpu
        if hthpu.is_available():
            return "Intel Gaudi HPU"
    except Exception:
        pass
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"

def log_hpu_status(logger):
    """Log HPU device and memory status."""
    try:
        import habana_frameworks.torch.hpu as hthpu
        if hthpu.is_available():
            logger.info(f"HPU available: Yes")
            logger.info(f"HPU device count: {hthpu.device_count()}")
            mem = get_hpu_memory_info()
            if mem:
                logger.info(f"HPU memory allocated: {mem['allocated_mb']:.1f} MB")
                logger.info(f"HPU peak memory:      {mem['max_allocated_mb']:.1f} MB")
            return True
    except Exception:
        pass
    logger.warning("HPU not available — running on CPU only!")
    return False



                                       


def kmeans_clustering(sensitivities, n_clusters=3):
    """K-means clustering on sensitivity values. Runs on CPU (sklearn)."""
    values = sensitivities.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(values)
    cluster_means = [(c, float(values[labels == c].mean())) for c in range(n_clusters)]
    cluster_means.sort(key=lambda x: x[1], reverse=True)
    return labels, cluster_means


def hierarchical_clustering(sensitivities, n_clusters=3):
    """Hierarchical clustering on sensitivity values. Runs on CPU (sklearn)."""
    values = sensitivities.reshape(-1, 1)
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg.fit_predict(values)
    cluster_means = [(c, float(values[labels == c].mean())) for c in range(n_clusters)]
    cluster_means.sort(key=lambda x: x[1], reverse=True)
    return labels, cluster_means


def percentile_clustering(sensitivities, n_clusters=3):
    """Percentile-based clustering on sensitivity values. Runs on CPU."""
    num_layers = len(sensitivities)
    layer_sens_pairs = [(i, sensitivities[i]) for i in range(num_layers)]
    layer_sens_pairs.sort(key=lambda x: x[1], reverse=True)
    cluster_size = num_layers // n_clusters
    labels = np.zeros(num_layers, dtype=int)
    cluster_means = []
    for cluster_id in range(n_clusters):
        start_idx = cluster_id * cluster_size
        end_idx = start_idx + cluster_size if cluster_id < n_clusters - 1 else num_layers
        cluster_layer_indices = [layer_sens_pairs[i][0] for i in range(start_idx, end_idx)]
        cluster_vals = [sensitivities[idx] for idx in cluster_layer_indices]
        for layer_idx in cluster_layer_indices:
            labels[layer_idx] = cluster_id
        cluster_means.append((cluster_id, float(np.mean(cluster_vals))))
    cluster_means.sort(key=lambda x: x[1], reverse=True)
    return labels, cluster_means



                                                              


class LinearSymmetricGroupQuant(nn.Module):
    """
    Symmetric MinMax Group-Wise Weight Quantization (PTQ).

    Computes a static per-group scale from the max absolute weight value,
    then quantizes via round-to-nearest and clamps to the target bit range.
    This is standard Post-Training Quantization (PTQ), not Learned Step Size
    Quantization (LSQ) which would require learnable scale parameters and
    gradient-based optimization.

    Quantization of weights happens at construction time on CPU.
    The forward pass (F.linear) is a standard matrix multiply that
    runs on whatever device the input tensor is on (HPU during eval).
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
        """Forward pass — runs on HPU when GaudiTrainer places model there."""
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
    that will be moved to HPU automatically by GaudiTrainer.
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



                                              


def prepare_wikitext_dataset(tokenizer, split="test", block_size=512):
    """
    Load WikiText-2 and prepare it as a chunked dataset for GaudiTrainer.
    
    Runs on CPU — tokenization is text processing.
    """
    try:
        raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception:
        raw_data = load_dataset("wikitext", "wikitext-2-v1", split=split)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized = raw_data.map(
        tokenize_function,
        batched=True,
        remove_columns=raw_data.column_names,
        desc="Tokenizing",
    )

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_dataset = tokenized.map(
        group_texts,
        batched=True,
        desc="Chunking",
    )

    return lm_dataset



                         


def gaudi_evaluate_perplexity(model, eval_dataset, gaudi_config, logger,
                               eval_name="Evaluation", per_device_batch_size=4,
                               use_hpu_graphs=True):
    """
    Evaluate perplexity using GaudiTrainer — runs on HPU.

    GaudiTrainer handles:
    - Moving model to HPU device
    - Lazy mode + mark_step()
    - HPU graph caching (faster repeated inference)
    - Distributed evaluation (multi-card)

    The model is provided on CPU; GaudiTrainer moves it to HPU internally.
    After evaluation, we explicitly move model back to CPU and delete the
    trainer to free HPU memory (prevents the ~4.6 GB/iteration leak).

    NOTE: use_hpu_graphs=False is used for quantized models with custom
    LinearSymmetricGroupQuant modules, since HPU graph capture may fail on
    non-standard modules.

    Returns: (perplexity, eval_time_seconds)
    """
    training_args = GaudiTrainingArguments(
        output_dir="/tmp/pmpq_eval",
        per_device_eval_batch_size=per_device_batch_size,
        use_habana=True,
        use_lazy_mode=True,
        use_hpu_graphs_for_inference=use_hpu_graphs,
        bf16=False,                                          
        report_to="none",
        dataloader_drop_last=False,
        remove_unused_columns=False,
        do_train=False,
        do_eval=True,
    )

    trainer = GaudiTrainer(
        model=model,
        gaudi_config=gaudi_config,
        args=training_args,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    rank = get_rank()
    if rank == 0:
        logger.info(f"[{eval_name}] Evaluating on HPU "
                     f"(lazy_mode=True, FP32, hpu_graphs={use_hpu_graphs})...")

    t0 = time.time()
    metrics = trainer.evaluate()
    eval_time = time.time() - t0

    eval_loss = metrics.get("eval_loss", float("inf"))
    ppl = math.exp(eval_loss) if eval_loss < 100 else float("inf")

    if rank == 0:
        logger.info(f"[{eval_name}] eval_loss={eval_loss:.4f}, PPL={ppl:.2f}, time={eval_time:.1f}s")

                                
                                                                     
                                                                        
                                                  
    model.to("cpu")
    del trainer
    free_hpu_memory()

    if rank == 0:
        mem = get_hpu_memory_info()
        if mem:
            logger.info(f"[{eval_name}] HPU after cleanup: {mem['allocated_mb']:.1f} MB allocated, "
                         f"{mem['max_allocated_mb']:.1f} MB peak")

    return ppl, eval_time



               


def main():
    init_distributed()                                                            

    rank = get_rank()
    world_size = get_world_size()
    set_seed(42)

                                                   
    if rank == 0:
        print("\nSelect Gaudi version for output organization:")
        print("  1. Gaudi1")
        print("  2. Gaudi2")
        while True:
            try:
                sel = input("Select (1-2) [default=1]: ").strip()
                if not sel or sel == '1':
                    gaudi_version = "Gaudi1"
                    break
                elif sel == '2':
                    gaudi_version = "Gaudi2"
                    break
                else:
                    print("Invalid selection. Enter 1 or 2.")
            except (EOFError, KeyboardInterrupt):
                gaudi_version = "Gaudi1"
                break
        print(f"  → Output will be saved under '{gaudi_version}' subfolders")
    else:
        gaudi_version = None
    gaudi_version = broadcast_object(gaudi_version, rank)

                                                     
    global LOG_DIR_BASE
    LOG_DIR = os.path.join(LOG_DIR_BASE, gaudi_version)
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = setup_logging(rank, log_dir=LOG_DIR)

    if rank == 0:
        logger.info("=" * 80)
        logger.info("PMPQ PHASE 2: MIXED-PRECISION QUANTIZATION EVALUATION (Gaudi HPU)")
        logger.info("=" * 80)
        logger.info(f"Cache:     {HF_HOME}")
        logger.info(f"Log dir:   {LOG_DIR}")
        mode_str = f"Distributed: {world_size} Gaudi cards" if world_size > 1 else "Single-card"
        logger.info(f"Mode:      {mode_str}")
        logger.info(f"Trainer:   GaudiTrainer (lazy mode + HPU graphs + FP32)")
        logger.info("")

                                    
        logger.info("Compute Device Mapping:")
        logger.info("  Model inference (forward/loss)     → HPU (via GaudiTrainer)")
        logger.info("  Weight quantization (scale+round)  → CPU (one-time)")
        logger.info("  Clustering (K-means, etc.)         → CPU (sklearn)")
        logger.info("  Dataset tokenization               → CPU")
        logger.info("  I/O, logging, result saving        → CPU")
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

    print_section("PHASE 2: PMPQ EVALUATION (GROUP-WISE QUANTIZATION) - Gaudi HPU", rank)

                                                                                       
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
        logger.info(f"  Models loaded separately in Steps 5 & 6 to avoid graph cache bugs")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        logger.info(f"  Tokenizer vocab: {tokenizer.vocab_size}")
        logger.info(f"  Device: {get_device_name()}")

                                           
    if rank == 0:
        logger.info("Preparing dataset (CPU)...")
    eval_dataset = prepare_wikitext_dataset(tokenizer, split="test", block_size=512)
    total_tokens = len(eval_dataset) * 512

    if rank == 0:
        logger.info(f"WikiText-2 test: {len(eval_dataset)} samples × 512 tokens = {total_tokens:,} total tokens")


    gaudi_config = GaudiConfig(
        use_fused_adam=False,                         
        use_fused_clip_norm=False,                    
        use_torch_autocast=False,                                
    )

                                                          

    print_section("STEP 5: EVALUATING FP32 BASELINE (HPU)", rank)
    if rank == 0:
        logger.info("Loading BASELINE model (separate from quantized model)...")
        logger.info("  Using separate models to avoid Habana graph cache interference")

    baseline_model = AutoModelForCausalLM.from_pretrained(model_name)

    if rank == 0:
        logger.info(f"  Baseline model dtype: {next(baseline_model.parameters()).dtype}")
        logger.info(f"  Baseline model device: {next(baseline_model.parameters()).device} (CPU → HPU at eval)")
        logger.info("Computing FP32 baseline perplexity on HPU...")

    ppl_before, eval_time_before = gaudi_evaluate_perplexity(
        baseline_model, eval_dataset, gaudi_config, logger,
        eval_name="FP32 Baseline",
        use_hpu_graphs=True                                      
    )

    if rank == 0:
        throughput_before = total_tokens / eval_time_before                 
        logger.info(f"FP32 Baseline: PPL = {ppl_before:.2f} ({eval_time_before:.1f}s)")
        logger.info(f"  Throughput: {throughput_before:.2f} tokens/s")

                         
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
        logger.info("Evaluating quantized model on HPU...")
        logger.info("  NOTE: HPU graphs disabled for quantized model (custom LinearSymmetricGroupQuant modules)")

                                                                                    
    ppl_after, eval_time_after = gaudi_evaluate_perplexity(
        quant_model, eval_dataset, gaudi_config, logger,
        eval_name="Quantized Model",
        use_hpu_graphs=False                                              
    )

                          
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
            'ppl_baseline': float(ppl_before),
            'ppl_quantized': float(ppl_after),
            'compression_ratio': float(compression_ratio),
        }
        with open(os.path.join(quant_save_dir, 'quant_config.json'), 'w') as f:
            json.dump(quant_config, f, indent=2)
        logger.info(f"Quantized model saved: {quant_save_dir}")
        logger.info(f"  NOTE: Saved weights are FP32 dequantized (quantize→round→dequantize). ")
        logger.info(f"        The file is NOT smaller on disk — this is simulated (fake) quantization.")

                                                    
                                                         

                                                                  
    if rank == 0:
        throughput_after = total_tokens / eval_time_after                 
        speedup = throughput_after / throughput_before if throughput_before > 0 else 1.0
        ppl_increase = ppl_after - ppl_before
        ppl_increase_pct = (ppl_increase / ppl_before * 100.0) if ppl_before > 0 else 0.0

        logger.info("=" * 70)
        logger.info("EVALUATION RESULTS (PMPQ + GROUP-WISE)")
        logger.info("=" * 70)
        logger.info(f"  Model:       {model_key}")
        logger.info(f"  Clustering:  {strategy_name} | Bits: {cluster_bits} | Group: {group_size}")
        logger.info(f"  Cards:       {world_size}")
        logger.info(f"  Platform:    Intel Gaudi HPU (optimum-habana)")
        logger.info(f"  FP32 PPL:    {ppl_before:.2f}")
        logger.info(f"  Quant PPL:   {ppl_after:.2f}")
        logger.info(f"  Degradation: {ppl_increase:+.2f} ({ppl_increase_pct:+.2f}%)")
        logger.info(f"  Compression: {compression_ratio:.2f}x ({bits_to_mb(orig_bits):.2f} → {bits_to_mb(quant_bits):.2f} MB)")
        logger.info(f"  Throughput:  {throughput_before:.0f} tokens/s (FP32) → {throughput_after:.0f} tokens/s (Quant)")
        logger.info(f"  Speedup:     {speedup:.2f}x")

        mem = get_hpu_memory_info()
        if mem:
            logger.info(f"  HPU peak:    {mem['max_allocated_mb']:.1f} MB")

        if ppl_increase_pct <= 5:
            quality = "EXCELLENT - Minimal degradation"
        elif ppl_increase_pct <= 10:
            quality = "GOOD - Acceptable degradation"
        elif ppl_increase_pct <= 20:
            quality = "MODERATE - Noticeable degradation"
        else:
            quality = "POOR - Significant degradation"
        logger.info(f"  Quality:     {quality}")

                              
        logger.info("=" * 70)
        logger.info("SAVING RESULTS")
        logger.info("=" * 70)

        eval_dir = os.path.join("Evaluation", gaudi_version)
        os.makedirs(eval_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"ptq_eval_{model_key}_WikiText_PMPQ_GroupWise_{strategy_name}_{timestamp}.txt"
        log_path = os.path.join(eval_dir, log_filename)

        fp32_model_size_mb = bits_to_mb(orig_bits)
        quant_model_size_mb_est = bits_to_mb(quant_bits)

        with open(log_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("MIXED-PRECISION PTQ EVALUATION RESULTS (PMPQ + GROUP-WISE)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Model: {model_key}\n")
            f.write(f"HuggingFace ID: {model_name}\n")
            f.write(f"Task: Language Modeling (Perplexity)\n")
            f.write(f"Dataset: WikiText-2\n")
            f.write(f"Method: PMPQ with Group-Wise Quantization\n")
            f.write(f"Evaluation Set: WikiText-2 test\n")
            f.write(f"Sensitivity File: {selected_file}\n")
            f.write(f"Clustering Strategy: {strategy_name}\n")
            f.write(f"Number of Clusters: {n_clusters}\n")
            f.write(f"Bit Allocation: {cluster_bits}\n")
            f.write(f"Group Size: {group_size}\n")
            f.write(f"Device: {get_device_name()}\n")
            f.write(f"Platform: optimum-habana (lazy mode + FP32)\n")
            f.write(f"Num Accelerators: {world_size}\n")
            f.write(f"Sequence Length: 512\n")
            f.write(f"Total Tokens: {total_tokens}\n")
            f.write(f"Timestamp: {timestamp}\n")
            if mem:
                f.write(f"HPU Peak Memory: {mem['max_allocated_mb']:.1f} MB\n")
            f.write("\n")

            f.write("-" * 70 + "\n")
            f.write("COMPUTE DEVICE MAPPING\n")
            f.write("-" * 70 + "\n")
            f.write("Model inference (forward/loss)     → HPU (via GaudiTrainer)\n")
            f.write("Weight quantization (scale+round)  → CPU (one-time)\n")
            f.write("Clustering (K-means, etc.)         → CPU (sklearn)\n")
            f.write("Dataset tokenization               → CPU\n")
            f.write("I/O / logging / saving             → CPU\n\n")

            f.write("-" * 70 + "\n")
            f.write("BASELINE (FP32)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Perplexity: {ppl_before:.6f}\n")
            f.write(f"Eval Time: {eval_time_before:.2f}s\n")
            f.write(f"Throughput: {throughput_before:.2f} tokens/s\n\n")

            f.write("-" * 70 + "\n")
            f.write("QUANTIZED (Mixed-Precision PTQ via PMPQ + Group-Wise)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Perplexity: {ppl_after:.6f}\n")
            f.write(f"Eval Time: {eval_time_after:.2f}s\n")
            f.write(f"Throughput: {throughput_after:.2f} tokens/s\n")
            f.write(f"Speedup: {speedup:.2f}x\n\n")

            f.write("-" * 70 + "\n")
            f.write("DEGRADATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"Perplexity Increase: {ppl_increase:+.6f} ({ppl_increase_pct:+.2f}%)\n")
            f.write(f"Quality: {quality}\n\n")

            f.write("-" * 70 + "\n")
            f.write("COMPRESSION (nn.Linear layers only)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Original Size: {fp32_model_size_mb:.2f} MB (Linear weights only)\n")
            f.write(f"Quantized Size: {quant_model_size_mb_est:.2f} MB (Linear weights only)\n")
            f.write(f"Compression Ratio: {compression_ratio:.2f}\n")
            f.write(f"Size Reduction: {reduction_pct:.1f}%\n")
            f.write(f"Note: Excludes embeddings, LayerNorm, and LM head (not quantized).\n")
            f.write(f"Note: Saved model is FP32 dequantized (simulated quantization).\n\n")

            f.write("\n# Compression Metrics:\n")
            f.write(f"orig_bits: {orig_bits}\n")
            f.write(f"quant_bits: {quant_bits}\n")
            f.write(f"reduction_pct: {reduction_pct:.2f}\n")
            f.write(f"compression_ratio: {compression_ratio:.3f}\n")
            f.write(f"fp32_model_size_mb: {fp32_model_size_mb:.2f}\n")
            f.write(f"quant_model_size_mb: {quant_model_size_mb_est:.2f}\n")
            f.write(f"quantize_time_s: {quantize_time_s:.3f}\n")

            f.write("\n# Performance Metrics:\n")
            f.write(f"eval_time_fp32_s: {eval_time_before:.2f}\n")
            f.write(f"eval_time_quantized_s: {eval_time_after:.2f}\n")
            f.write(f"throughput_fp32_tokens_per_s: {throughput_before:.2f}\n")
            f.write(f"throughput_quantized_tokens_per_s: {throughput_after:.2f}\n")
            f.write(f"speedup: {speedup:.2f}\n")
            f.write(f"total_tokens: {total_tokens}\n")

            f.write("\n# Perplexity Metrics:\n")
            f.write(f"ppl_fp32: {ppl_before:.6f}\n")
            f.write(f"ppl_quantized: {ppl_after:.6f}\n")
            f.write(f"ppl_increase: {ppl_increase:.6f}\n")
            f.write(f"ppl_increase_pct: {ppl_increase_pct:.2f}\n")

            f.write("\n" + "-" * 70 + "\n")
            f.write("LAYER BIT ALLOCATION (BASED ON PRUNING SENSITIVITY)\n")
            f.write("-" * 70 + "\n")
            for i in range(num_layers):
                f.write(f"Layer {i:2d}: {layer_bits_map[i]:2d}-bit (sensitivity: {sens_values[i]:.4f})\n")

            f.write("\n" + "-" * 70 + "\n")
            f.write("METHOD NOTES\n")
            f.write("-" * 70 + "\n")
            f.write("PMPQ Assumption: Layers sensitive to weight pruning are also\n")
            f.write("sensitive to weight quantization.\n\n")
            f.write("GROUP-WISE: Each group of weights has its own scale factor,\n")
            f.write("providing much better precision than a single global scale.\n")

        logger.info(f"Results saved: {log_path}")

                                           
        summary_log = os.path.join(LOG_DIR, f"evaluation_results_{timestamp}.txt")
        with open(summary_log, "w") as f:
            f.write(f"PMPQ Evaluation Results — {timestamp}\n")
            f.write(f"Model: {model_key}\n")
            f.write(f"Clustering: {strategy_name} | Bits: {cluster_bits} | Group: {group_size}\n")
            f.write(f"Cards: {world_size}\n")
            f.write(f"FP32 PPL:    {ppl_before:.2f} ({eval_time_before:.1f}s, {throughput_before:.0f} tokens/s)\n")
            f.write(f"Quant PPL:   {ppl_after:.2f} ({eval_time_after:.1f}s, {throughput_after:.0f} tokens/s)\n")
            f.write(f"Degradation: {ppl_increase:+.2f} ({ppl_increase_pct:+.2f}%)\n")
            f.write(f"Compression: {compression_ratio:.2f}x ({fp32_model_size_mb:.2f} → {quant_model_size_mb_est:.2f} MB)\n")
            f.write(f"Speedup:     {speedup:.2f}x\n")
            f.write(f"Quality:     {quality}\n")
            if mem:
                f.write(f"HPU Peak:    {mem['max_allocated_mb']:.1f} MB\n")
            f.write(f"\nLayer allocations:\n")
            for i in range(num_layers):
                f.write(f"  Layer {i:2d}: {layer_bits_map[i]:2d}-bit\n")

        logger.info(f"Summary saved: {summary_log}")
        logger.info(f"Baseline model: {baseline_save_dir}")
        logger.info(f"Quantized model: {quant_save_dir}")
        logger.info("")
        logger.info("Phase 2 evaluation complete!")

                                                                
    if rank == 0:
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
                        label = f"{name} (PPL={qcfg.get('ppl_quantized', '?'):.2f}, {qcfg.get('compression_ratio', '?'):.1f}x compression)"
                    elif "baseline" in name.lower():
                        label = f"{name} (FP32 baseline)"
                    else:
                        label = name
                    available_models[name] = {"path": model_path, "label": label}

        if not available_models:
            logger.warning(f"No models found in {MODELS_DIR}")
        else:
            print(f"\n{'='*70}")
            print("  AVAILABLE MODELS")
            print(f"{'='*70}")
            model_list = list(available_models.items())
            for i, (name, info) in enumerate(model_list, 1):
                print(f"  {i}. {info['label']}")
            print(f"{'='*70}")

                                                  
            loaded_models = {}

            while True:
                try:
                    sel = input(f"\nSelect model to load (1-{len(model_list)}), or 'done' to start generating: ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if sel.lower() in ('done', 'd', ''):
                    if not loaded_models:
                        print("Please load at least one model first.")
                        continue
                    break

                if sel.lower() in ('quit', 'exit', 'q'):
                    loaded_models = {}
                    break

                try:
                    idx = int(sel) - 1
                    if 0 <= idx < len(model_list):
                        name, info = model_list[idx]
                        if name in loaded_models:
                            print(f"  Already loaded: {name}")
                            continue
                        print(f"  Loading {name} → HPU...")
                        mdl = AutoModelForCausalLM.from_pretrained(info['path'])
                        mdl = mdl.eval().to("hpu")
                        loaded_models[name] = mdl
                        print(f"  ✓ Loaded on HPU: {info['label']}")
                    else:
                        print(f"  Invalid selection. Enter 1-{len(model_list)}")
                except ValueError:
                    print("  Invalid input. Enter a number or 'done'")
                except Exception as e:
                    print(f"  Error loading model: {e}")

            if loaded_models:
                                                  
                model_names = list(loaded_models.keys())
                active_name = model_names[0]
                active_model = loaded_models[active_name]

                print(f"\n{'='*70}")
                print("  INTERACTIVE GENERATION")
                print(f"{'='*70}")
                print(f"  Active model: {active_name}")
                print(f"  Commands:")
                print(f"    'load'    — load another model from Models/ directory")
                print(f"    'switch'  — switch between loaded models")
                print(f"    'compare' — generate from ALL loaded models side-by-side")
                print(f"    'list'    — show loaded models")
                print(f"    'quit'    — exit")
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

                    if prompt.lower() == 'load':
                                                                     
                        unloaded = [(i, n, info) for i, (n, info) in enumerate(model_list, 1)
                                    if n not in loaded_models]
                        if not unloaded:
                            print("All available models are already loaded.")
                            continue
                        print("\nAvailable to load:")
                        for i, n, info in unloaded:
                            print(f"  {i}. {info['label']}")
                        try:
                            sel = input(f"Select model to load: ").strip()
                            idx = int(sel) - 1
                            if 0 <= idx < len(model_list):
                                name, info = model_list[idx]
                                if name in loaded_models:
                                    print(f"  Already loaded: {name}")
                                else:
                                    print(f"  Loading {name} → HPU...")
                                    mdl = AutoModelForCausalLM.from_pretrained(info['path'])
                                    mdl = mdl.eval().to("hpu")
                                    loaded_models[name] = mdl
                                    model_names = list(loaded_models.keys())
                                    print(f"  ✓ Loaded on HPU: {info['label']}")
                                    print(f"  Use 'switch' to switch to it.")
                            else:
                                print("Invalid selection.")
                        except (ValueError, EOFError, KeyboardInterrupt):
                            print("Cancelled.")
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
                            else:
                                print("Invalid selection.")
                        except (ValueError, EOFError, KeyboardInterrupt):
                            print("Cancelled.")
                        continue

                    if prompt.lower() == 'compare':
                        prompt = input("Enter prompt for comparison: ").strip()
                        if not prompt:
                            continue
                        inputs = tokenizer(prompt, return_tensors="pt")
                        input_ids = inputs["input_ids"].to("hpu")
                        print(f"\n{'='*70}")
                        print(f"Prompt: {prompt}")
                        print(f"{'='*70}")
                        for name, mdl in loaded_models.items():
                            with torch.no_grad():
                                try:
                                    import habana_frameworks.torch.core as htcore
                                    htcore.mark_step()
                                except Exception:
                                    pass
                                out = mdl.generate(
                                    input_ids,
                                    max_new_tokens=100,
                                    do_sample=True,
                                    temperature=0.7,
                                    top_p=0.9,
                                )
                                try:
                                    htcore.mark_step()
                                except Exception:
                                    pass
                            text = tokenizer.decode(out[0].cpu(), skip_special_tokens=True)
                            print(f"\n[{name}]:")
                            print(text)
                        print(f"{'='*70}")
                        continue

                                                       
                    inputs = tokenizer(prompt, return_tensors="pt")
                    input_ids = inputs["input_ids"].to("hpu")
                    with torch.no_grad():
                        try:
                            import habana_frameworks.torch.core as htcore
                            htcore.mark_step()
                        except Exception:
                            pass
                        output = active_model.generate(
                            input_ids,
                            max_new_tokens=100,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                        )
                        try:
                            htcore.mark_step()
                        except Exception:
                            pass
                    generated_text = tokenizer.decode(output[0].cpu(), skip_special_tokens=True)
                    print(f"\n[{active_name} OUTPUT]:")
                    print(generated_text)

                                       
                for mdl in loaded_models.values():
                    del mdl
                loaded_models.clear()

        logger.info("Interactive generation complete.")

             
    del quant_model
    free_hpu_memory()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
