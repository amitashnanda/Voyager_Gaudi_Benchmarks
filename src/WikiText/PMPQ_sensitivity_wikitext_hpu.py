"""
Phase 1: Pruning-Based Sensitivity Analysis for TinyLlama on WikiText (PMPQ Method)
=============================================================================
** Intel Gaudi HPU Version — Optimum-Habana + GaudiTrainer **

This script computes layer sensitivities using pruning-based approach:
1. For each layer, apply unstructured magnitude-based pruning (zero X% of smallest weights)
2. Measure perplexity increase after pruning each layer
3. Higher perplexity increase = more sensitive layer
4. Use pruning sensitivity to guide quantization bit allocation

Assumption: Layers sensitive to pruning are also sensitive to quantization

Dataset: WikiText-2 (https://huggingface.co/datasets/wikitext)
Task: Language Modeling (Perplexity)
Method: PMPQ (Pruning-based Mixed-Precision Quantization)
Platform: Intel Gaudi HPU (SDSC Voyager) via optimum-habana

Split Usage: VALIDATION split
  Note: Standard practice - validation for sensitivity analysis,
  test split reserved for final evaluation.

Compute device summary:
  - Model inference (forward pass, loss) → HPU via GaudiTrainer
  - Pruning (weight masking)            → CPU (lightweight, <1ms per layer)
  - Dataset tokenization                → CPU
  - I/O, logging, result saving         → CPU

Launch (single-card):
    PT_HPU_LAZY_MODE=1 python PMPQ_sensitivity_wikitext_hpu.py --sparsity 0.3 --gaudi_version Gaudi1

Launch (multi-card, 8 HPUs):
    PT_HPU_LAZY_MODE=1 python gaudi_spawn.py --world_size 2 --use_mpi PMPQ_sensitivity_wikitext_hpu.py --sparsity 0.3 --gaudi_version Gaudi1

Author: Mixed-Precision Quantization Team
Date: 2025
"""

                                        


import os
import sys
import logging


HF_HOME = os.environ.get("HF_HOME", "/voyager/ceph/users/ananda2/Quantization/.cache/huggingface")
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
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    default_data_collator,
)


from optimum.habana import GaudiConfig, GaudiTrainer, GaudiTrainingArguments
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

adapt_transformers_to_gaudi()

warnings.filterwarnings('ignore')


for p in (os.environ["HF_DATASETS_CACHE"], os.environ["HF_HUB_CACHE"]):
    Path(p).mkdir(parents=True, exist_ok=True)



               


def setup_logging(rank=0, log_dir=None):
    """Set up dual logging: console + file.
    
    Log file is saved to log_dir with a timestamp so each run is preserved.
    Only rank 0 logs to file in distributed mode.
    """
    if log_dir is None:
        log_dir = LOG_DIR_BASE
    logger = logging.getLogger("PMPQ_WIKITEXT_SENSITIVITY")
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
        log_file = os.path.join(log_dir, f"sensitivity_{timestamp}.log")
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


                   


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        from optimum.habana.utils import set_seed as gaudi_set_seed
        gaudi_set_seed(seed)
    except Exception:
        pass


def get_rank():
    """Get current distributed rank (0 if not distributed)."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get world size (1 if not distributed)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def broadcast_object(obj, rank):
    """Broadcast a Python object from rank 0 to all ranks."""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return obj
    container = [obj]
    dist.broadcast_object_list(container, src=0)
    return container[0]


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


def select_model(logger):
    """Interactive model selection (rank 0 only)."""
    logger.info("Available Models:")
    model_keys = list(TINYLLAMA_MODELS.keys())
    for i, key in enumerate(model_keys, 1):
        info = TINYLLAMA_MODELS[key]
        logger.info(f"  {i}. {key} — {info['description']} ({info['num_layers']} layers, {info['hidden_dim']} hidden)")

    while True:
        try:
            choice = input(f"Select model (1-{len(model_keys)}) [default: 1]: ").strip()
            if not choice:
                choice = "1"
            idx = int(choice) - 1
            if 0 <= idx < len(model_keys):
                selected_key = model_keys[idx]
                logger.info(f"Selected: {selected_key}")
                return selected_key, TINYLLAMA_MODELS[selected_key]
            print(f"[ERROR] Please enter a number between 1 and {len(model_keys)}")
        except ValueError:
            print("[ERROR] Invalid input. Please enter a number.")



                   


def apply_magnitude_pruning_to_layer(layer, sparsity_level):
    """
    Apply unstructured magnitude-based pruning to a specific transformer layer.

    Only prunes nn.Linear weight tensors (not biases, not LayerNorm/RMSNorm).
    This matches the scope of Phase 2 quantization, which only quantizes
    nn.Linear modules — ensuring the "pruning-sensitive ↔ quantization-sensitive"
    assumption holds.

    Uses a unified per-layer threshold: all Linear weight magnitudes in the
    layer are concatenated into a single vector, one threshold is computed,
    and masks are applied back to each weight tensor. This avoids distorting
    relative sensitivity by forcing each matrix to independent sparsity.

    Runs on CPU — lightweight weight masking (no matrix multiply).
    """
    with torch.no_grad():
                                                                              
        linear_weights = []
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                linear_weights.append(module.weight)

        if not linear_weights:
            return

                                                                               
        all_magnitudes = torch.cat([w.data.abs().view(-1) for w in linear_weights])
        k = int(all_magnitudes.numel() * (1 - sparsity_level))
        if k == 0:
            return

                                                            
        threshold = torch.topk(all_magnitudes, k, largest=True)[0][-1]

                                               
        for w in linear_weights:
            mask = (w.data.abs() >= threshold).float()
            w.data *= mask



                                        


def prepare_wikitext_dataset(tokenizer, split="validation", block_size=512):
    """
    Load WikiText-2 and prepare it as a chunked dataset for GaudiTrainer.
    
    Runs on CPU — tokenization is a text processing operation.

    The standard HuggingFace CLM evaluation approach:
    1. Load raw text
    2. Tokenize all text
    3. Chunk into fixed-length blocks (block_size)
    4. Each block is an independent sample with input_ids = labels
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
        logger.info(f"[{eval_name}] Evaluating with GaudiTrainer on HPU "
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

    return ppl



                         


def compute_pruning_sensitivity(model_name, num_layers, eval_dataset, gaudi_config,
                                 logger, sparsity_level=0.3, baseline_ppl=None):
    """
    Compute pruning-based sensitivity for each layer.

    For each layer:
    1. Load fresh model (CPU) — from_pretrained
    2. Prune that layer (CPU) — magnitude-based weight masking
    3. Evaluate perplexity (HPU) — GaudiTrainer handles device placement
    4. Sensitivity = pruned_ppl - baseline_ppl
    """
    rank = get_rank()

    if baseline_ppl is None:
        if rank == 0:
            logger.info("Computing baseline perplexity (no pruning)...")
            logger.info("  Model loaded on CPU → GaudiTrainer moves to HPU for eval")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if rank == 0:
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"  Model params: {param_count:,} ({param_count/1e6:.1f}M)")
            logger.info(f"  Model dtype:  {next(model.parameters()).dtype}")
            logger.info(f"  Model device: {next(model.parameters()).device} (will be moved to HPU)")
        baseline_ppl = gaudi_evaluate_perplexity(
            model, eval_dataset, gaudi_config, logger, eval_name="Baseline"
        )
                                                                                   
        del model
        free_hpu_memory()
        if rank == 0:
            mem = get_hpu_memory_info()
            mem_str = f" (HPU: {mem['allocated_mb']:.0f} MB after cleanup)" if mem else ""
            logger.info(f"Baseline Perplexity: {baseline_ppl:.2f}{mem_str}")

    if rank == 0:
        logger.info(f"Computing sensitivity for {num_layers} layers at {sparsity_level*100}% sparsity...")
        logger.info("-" * 80)

    layer_sensitivities = {}

    for layer_idx in range(num_layers):
        if rank == 0:
            logger.info(f"[Layer {layer_idx}/{num_layers-1}] Loading fresh model (CPU)...")

        pruned_model = AutoModelForCausalLM.from_pretrained(model_name)

        if hasattr(pruned_model, 'model'):
            target_layer = pruned_model.model.layers[layer_idx]
        else:
            target_layer = pruned_model.layers[layer_idx]

        if rank == 0:
            logger.info(f"[Layer {layer_idx}] Applying {sparsity_level*100}% pruning (CPU)...")
        apply_magnitude_pruning_to_layer(target_layer, sparsity_level)

        if rank == 0:
            logger.info(f"[Layer {layer_idx}] Evaluating pruned model (HPU)...")
        pruned_perplexity = gaudi_evaluate_perplexity(
            pruned_model, eval_dataset, gaudi_config, logger,
            eval_name=f"Layer {layer_idx} Pruned"
        )

        sensitivity = pruned_perplexity - baseline_ppl
        layer_sensitivities[f"layer_{layer_idx}"] = float(sensitivity)

                                                                                   
        del pruned_model
        free_hpu_memory()

        if rank == 0:
            mem = get_hpu_memory_info()
            mem_str = f" | HPU: {mem['allocated_mb']:.0f} MB" if mem else ""
            logger.info(f"[Layer {layer_idx}] Baseline: {baseline_ppl:.2f} | "
                         f"Pruned: {pruned_perplexity:.2f} | Sensitivity: {sensitivity:.4f}{mem_str}")

    if rank == 0:
        logger.info("=" * 80)
        logger.info("Pruning sensitivity computation complete!")

    return layer_sensitivities, baseline_ppl



               


def main():
    init_distributed()                                                            

    parser = argparse.ArgumentParser(
        description='Compute pruning-based sensitivity (PMPQ) - Gaudi HPU [Optimum-Habana]'
    )
    parser.add_argument('--sparsity', type=float, default=0.3,
                        help='Sparsity level for pruning (0-1, default: 0.3 = 30%%)')
    parser.add_argument('--gaudi_version', type=str, default=None,
                        choices=['Gaudi1', 'Gaudi2'],
                        help='Gaudi version for organizing output folders (default: prompt)')
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
        logger.info("PMPQ PHASE 1: PRUNING-BASED SENSITIVITY ANALYSIS (Gaudi HPU)")
        logger.info("=" * 80)
        logger.info(f"Cache:     {os.environ['HF_HOME']}")
        logger.info(f"Log dir:   {LOG_DIR}")
        mode_str = f"Distributed: {world_size} Gaudi cards" if world_size > 1 else "Single-card"
        logger.info(f"Mode:      {mode_str}")
        logger.info(f"Sparsity:  {args.sparsity*100}%")
        logger.info(f"Eval:      Full VALIDATION set, 512-token chunks")
        logger.info(f"Trainer:   GaudiTrainer (lazy mode + HPU graphs + FP32)")
        logger.info("")

                                    
        logger.info("Compute Device Mapping:")
        logger.info("  Model inference (forward/loss)  → HPU (via GaudiTrainer)")
        logger.info("  Weight pruning (masking)        → CPU (lightweight)")
        logger.info("  Dataset tokenization            → CPU")
        logger.info("  I/O, logging, result saving     → CPU")
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

                                                         
    if rank == 0:
        logger.info("=" * 80)
        logger.info("STEP 1: MODEL SELECTION & LOADING")
        logger.info("=" * 80)
        model_key, model_config = select_model(logger)
        selection = (model_key, model_config)
    else:
        selection = None
    selection = broadcast_object(selection, rank)
    model_key, model_config = selection

    model_name = model_config["model_name"]
    num_layers = model_config["num_layers"]

    if rank == 0:
        logger.info(f"Loading tokenizer for {model_key}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if rank == 0:
        logger.info(f"Tokenizer loaded (vocab_size={tokenizer.vocab_size})")

                                                   
    if rank == 0:
        logger.info("=" * 80)
        logger.info("STEP 2: PREPARING DATASET (CPU)")
        logger.info("=" * 80)

    eval_dataset = prepare_wikitext_dataset(tokenizer, split="validation", block_size=512)

    if rank == 0:
        logger.info(f"WikiText-2 validation: {len(eval_dataset)} samples × 512 tokens")

                                                       

    gaudi_config = GaudiConfig(
        use_fused_adam=False,          
        use_fused_clip_norm=False,      
        use_torch_autocast=False,       
    )

    if rank == 0:
        logger.info(f"Gaudi config: custom (use_torch_autocast=False for true FP32)")

                                                                 
    if rank == 0:
        logger.info("=" * 80)
        logger.info("STEP 3: COMPUTING PRUNING-BASED SENSITIVITIES")
        logger.info("=" * 80)

    t0 = time.time()
    layer_sensitivities, baseline_ppl = compute_pruning_sensitivity(
        model_name=model_name, num_layers=num_layers,
        eval_dataset=eval_dataset, gaudi_config=gaudi_config,
        logger=logger, sparsity_level=args.sparsity
    )
    sensitivity_time = time.time() - t0

                                                              
    if rank == 0:
        logger.info(f"Sensitivity computation complete in {sensitivity_time:.2f}s")

        logger.info("")
        logger.info("Pruning-Based Sensitivities (Perplexity Increase):")
        logger.info("-" * 80)
        for layer_name in sorted(layer_sensitivities, key=lambda x: int(x.split("_")[1])):
            logger.info(f"  {layer_name}: {layer_sensitivities[layer_name]:.4f}")

                             
        mem = get_hpu_memory_info()
        if mem:
            logger.info(f"HPU peak memory usage: {mem['max_allocated_mb']:.1f} MB")

        logger.info("=" * 80)
        logger.info("STEP 4: SAVING SENSITIVITY FILES")
        logger.info("=" * 80)
        sens_dir = os.path.join("Sensitivities", gaudi_version)
        os.makedirs(sens_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"sens_{model_key}_WikiText_fullval_pruning_s{int(args.sparsity*100)}_{timestamp}"

                   
        json_path = os.path.join(sens_dir, f"{filename_base}.json")
        with open(json_path, "w") as f:
            json.dump(layer_sensitivities, f, indent=2)

                                
        txt_path = os.path.join(sens_dir, f"{filename_base}.txt")
        with open(txt_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("LAYER SENSITIVITY FILE - PRUNING-BASED (PMPQ) ON WIKITEXT\n")
            f.write("=" * 80 + "\n\n")
            f.write("=" * 80 + "\n")
            f.write("CONFIGURATION\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model: {model_key}\n")
            f.write(f"Model HF Hub: {model_name}\n")
            f.write(f"Task: Language Modeling (Perplexity)\n")
            f.write(f"Dataset: WikiText-2\n")
            f.write(f"Method: PMPQ (Pruning-based Mixed-Precision Quantization)\n")
            f.write(f"Sparsity Level: {args.sparsity*100}%\n")
            f.write(f"Evaluation Set: VALIDATION (full, continuous text)\n")
            f.write(f"Tokenization: Continuous text, 512-token chunks, no padding\n")
            f.write(f"Num Layers: {num_layers}\n")
            f.write(f"Hidden Dim: {model_config['hidden_dim']}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Platform: Intel Gaudi HPU (optimum-habana)\n")
            f.write(f"Num Accelerators: {world_size}\n")
            f.write(f"Baseline PPL: {baseline_ppl:.2f}\n")
            if mem:
                f.write(f"HPU Peak Memory: {mem['max_allocated_mb']:.1f} MB\n")
            f.write(f"\n")
            f.write("=" * 80 + "\n")
            f.write("COMPUTE DEVICE MAPPING\n")
            f.write("=" * 80 + "\n")
            f.write("Model inference (forward/loss) → HPU (via GaudiTrainer)\n")
            f.write("Weight pruning (masking)       → CPU\n")
            f.write("Dataset tokenization           → CPU\n")
            f.write("I/O, logging, saving           → CPU\n\n")
            f.write("=" * 80 + "\n")
            f.write("COMPUTATION STATISTICS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Time: {sensitivity_time:.2f}s\n")
            f.write(f"Time per Layer: {sensitivity_time/num_layers:.2f}s\n\n")
            f.write("=" * 80 + "\n")
            f.write("LAYER SENSITIVITIES (PRUNING-BASED - PERPLEXITY INCREASE)\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Layer':<10} {'Sensitivity':<20} {'Rank':<10}\n")
            f.write("-" * 80 + "\n")
            sorted_layers = sorted(layer_sensitivities.items(),
                                   key=lambda x: float(x[1]), reverse=True)
            for ri, (ln, sv) in enumerate(sorted_layers, 1):
                f.write(f"{ln:<10} {sv:<20.4f} {ri:<10}\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("SENSITIVITY STATISTICS\n")
            f.write("=" * 80 + "\n")
            svs = np.array(list(layer_sensitivities.values()))
            f.write(f"Mean Sensitivity: {svs.mean():.4f}\n")
            f.write(f"Min Sensitivity: {svs.min():.4f}\n")
            f.write(f"Max Sensitivity: {svs.max():.4f}\n")
            f.write(f"Std Deviation: {svs.std():.4f}\n\n")
            f.write("=" * 80 + "\n")
            f.write("INTERPRETATION\n")
            f.write("=" * 80 + "\n")
            f.write("Higher sensitivity = Layer is more affected by pruning\n")
            f.write("Lower sensitivity = Layer can tolerate more weight removal\n")
            f.write("\nAssumption (PMPQ): Layers sensitive to pruning are also\n")
            f.write("sensitive to quantization. Use these sensitivities in Phase 2\n")
            f.write("to assign different bit-widths for mixed-precision quantization.\n\n")
            f.write("=" * 80 + "\n")
            f.write("METHOD DETAILS\n")
            f.write("=" * 80 + "\n")
            f.write("For each layer:\n")
            f.write(f"1. Load fresh model on CPU (patched by optimum-habana)\n")
            f.write(f"2. Apply {args.sparsity*100}% magnitude-based pruning to that layer (CPU)\n")
            f.write(f"3. Evaluate perplexity with GaudiTrainer on HPU (lazy mode + FP32)\n")
            f.write(f"4. Use continuous text tokenization (512-token chunks, no padding)\n")
            f.write(f"5. Sensitivity = Pruned_Perplexity - Baseline_Perplexity\n")
            f.write("\nWikiText-2 Dataset:\n")
            f.write("- Language modeling dataset from Wikipedia articles\n")
            f.write("- Evaluation metric: Perplexity (lower is better)\n")
            f.write("- Continuous tokenization matches Phase 2 evaluation\n")

        logger.info(f"Saved: {filename_base}.json / .txt")

                                            
        summary_log = os.path.join(LOG_DIR, f"sensitivity_results_{timestamp}.txt")
        with open(summary_log, "w") as f:
            f.write(f"PMPQ Sensitivity Results — {timestamp}\n")
            f.write(f"Model: {model_key}\n")
            f.write(f"Baseline PPL: {baseline_ppl:.2f}\n")
            f.write(f"Sparsity: {args.sparsity*100}%\n")
            f.write(f"Time: {sensitivity_time:.2f}s\n")
            f.write(f"Cards: {world_size}\n")
            if mem:
                f.write(f"HPU Peak Memory: {mem['max_allocated_mb']:.1f} MB\n")
            f.write(f"\nSensitivities:\n")
            for ln, sv in sorted_layers:
                f.write(f"  {ln}: {sv:.4f}\n")

        logger.info(f"Summary saved to: {summary_log}")

        logger.info("=" * 80)
        logger.info("PHASE 1 COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Model: {model_key} ({num_layers} layers)")
        logger.info(f"  Baseline PPL: {baseline_ppl:.2f}")
        logger.info(f"  Cards: {world_size}")
        logger.info(f"  Time:  {sensitivity_time:.2f}s")

             
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

