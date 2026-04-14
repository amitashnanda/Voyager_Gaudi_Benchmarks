"""
Phase 1: Pruning-Based Sensitivity Analysis for TinyLlama on BoolQ (PMPQ Method)
=================================================================================
** Intel Gaudi HPU Version — Optimum-Habana + Direct HPU Inference **

Computes per-layer sensitivity using pruning-based method.
Metric: Accuracy drop on BoolQ (Yes/No question answering)
Sensitivity = baseline_accuracy - pruned_accuracy

This matches TinyLlama's official benchmark evaluation.
TinyLlama-1.1B-3T official BoolQ score: 57.83 (accuracy)

Dataset: BoolQ (https://huggingface.co/datasets/google/boolq)
Task: Boolean Question Answering
Method: PMPQ (Pruning-based Mixed-Precision Quantization)
Platform: Intel Gaudi HPU (SDSC Voyager) via optimum-habana

Split Usage: TRAIN split (default: 2,500 samples from 9,427 available)
  Note: Uses representative subset for faster sensitivity calculation.
  BoolQ has no test split. Train (subset) for sensitivity, validation (full 3,270) for evaluation.
  Use --max_samples 0 for full train dataset (slower but more data).

Compute device summary:
  - Model inference (forward pass)  → HPU (direct, lazy mode)
  - Weight pruning (masking)        → CPU (lightweight, <1ms per layer)
  - Dataset loading                 → CPU
  - Log-likelihood computation      → CPU
  - I/O, logging, result saving     → CPU

Pipeline:
    STEP 1: Select model
    STEP 2: Load BoolQ train dataset
    STEP 3: Compute baseline accuracy (HPU)
    STEP 4: For each layer: prune → evaluate accuracy → compute sensitivity
    STEP 5: Save sensitivity scores

Usage (single-card, default 2500 samples):
    PT_HPU_LAZY_MODE=1 python PMPQ_sensitivity_boolq_hpu.py --sparsity 0.3 --gaudi_version Gaudi2

Usage (full train dataset):
    PT_HPU_LAZY_MODE=1 python PMPQ_sensitivity_boolq_hpu.py --sparsity 0.3 --max_samples 0

Usage (multi-card, 8 HPUs):
    PT_HPU_LAZY_MODE=1 python ../gaudi_spawn.py \
        --world_size 8 --use_mpi PMPQ_sensitivity_boolq_hpu.py --sparsity 0.3

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
    "TRANSFORMERS_CACHE": os.path.join(HF_HOME, "hub"),
    "TOKENIZERS_PARALLELISM": "false",
})


import re
import gc
import json
import math
import time
import argparse
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

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

from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

adapt_transformers_to_gaudi()

warnings.filterwarnings('ignore')


for p in (os.environ["HF_DATASETS_CACHE"], os.environ["HF_HUB_CACHE"]):
    Path(p).mkdir(parents=True, exist_ok=True)


LOG_DIR_BASE = "/voyager/ceph/users/ananda2/Quantization/logs"
os.makedirs(LOG_DIR_BASE, exist_ok=True)

                                
MAX_LENGTH = 512                                            

               


def setup_logging(rank=0, log_dir=None):
    """Set up dual logging: console + file.

    Log file is saved to log_dir with a timestamp so each run is preserved.
    Only rank 0 logs to file in distributed mode.
    """
    if log_dir is None:
        log_dir = LOG_DIR_BASE
    logger = logging.getLogger("PMPQ_BOOLQ_SENSITIVITY")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                   datefmt="%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"boolq_sensitivity_{timestamp}.log")
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
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if HAS_HPU:
        import habana_frameworks.torch.hpu.random as hrand
        hrand.manual_seed_all(seed)


def get_rank():
    """Get current distributed rank (0 if not distributed)."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    """Get world size (1 if not distributed)."""
    return dist.get_world_size() if dist.is_initialized() else 1


def broadcast_object(obj, rank):
    """Broadcast a Python object from rank 0 to all ranks."""
    if not dist.is_initialized():
        return obj
    obj_list = [obj]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def get_hpu_memory_info():
    """Get HPU memory usage. Returns dict with used/total in MB, or None if unavailable."""
    try:
        stats = torch.hpu.memory_stats()
        return {
            "allocated_mb": stats.get("InUse", 0) / (1024 * 1024),
            "max_allocated_mb": stats.get("MaxInUse", 0) / (1024 * 1024),
        }
    except Exception:
        return None


def free_hpu_memory():
    """Force-free HPU memory after model deletion."""
    gc.collect()
    if HAS_HPU:
        try:
            torch.hpu.empty_cache()
        except Exception:
            pass


def log_hpu_status(logger):
    """Log HPU device and memory status."""
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


def select_model(logger):
    """Interactive model selection (rank 0 only)."""
    models = list(TINYLLAMA_MODELS.items())
    if len(models) == 1:
        key, config = models[0]
        logger.info(f"Auto-selected: {key} — {config['description']}")
        return key, config

    print("\nAvailable models:")
    for i, (key, config) in enumerate(models, 1):
        print(f"  {i}. {key}: {config['description']}")

    while True:
        try:
            sel = int(input(f"Select (1-{len(models)}): "))
            if 1 <= sel <= len(models):
                key, config = models[sel - 1]
                logger.info(f"Selected: {key}")
                return key, config
        except (ValueError, EOFError, KeyboardInterrupt):
            pass
        print("Invalid selection.")



                   


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



               


def load_boolq_dataset(split="train"):
    """
    Load BoolQ dataset from HuggingFace.

    BoolQ train has ~9427 samples, validation has ~3270 samples.
    No test split available.
    Fields: question (str), passage (str), answer (bool).
    """
    try:
        dataset = load_dataset("google/boolq", split=split, trust_remote_code=True)
    except Exception:
        dataset = load_dataset("boolq", split=split, trust_remote_code=True)
    return dataset



                                                    


def evaluate_boolq_accuracy(model, dataset, tokenizer, logger,
                             eval_name="BoolQ", max_samples=0, batch_size=64):
    """
    Evaluate accuracy on BoolQ using log-likelihood scoring on HPU with batched inference.

    For each sample:
      1. Format prompt: "Passage: {passage}\\nQuestion: {question}\\nAnswer:"
      2. Compute log-likelihood of " yes" vs " no" continuation
      3. Predict True if P(" yes") > P(" no"), else False
      4. Compare to gold answer

    Batched processing: Processes batch_size samples in parallel for faster inference.

    This matches the lm-evaluation-harness methodology.
    TinyLlama-1.1B-3T official score: 57.83

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
            prompt_lens = []
            answers = []

            for sample in batch_samples:
                passage = sample["passage"]
                question = sample["question"]
                answer = sample["answer"]                 
                answers.append(answer)

                                                  
                prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"

                                   
                prompt_enc = tokenizer(prompt, add_special_tokens=True)
                prompt_len = len(prompt_enc["input_ids"])
                prompt_lens.append(prompt_len)

                                                                 
                all_texts.append(prompt + " yes")
                all_texts.append(prompt + " no")

                                                                 
            batch = tokenizer(all_texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=MAX_LENGTH)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

                                                             
            outputs = model(input_ids, attention_mask=attention_mask)
            htcore.mark_step()

                                                        
            logits = outputs.logits.float().cpu()
            input_ids_cpu = batch["input_ids"]
            attn_cpu = batch["attention_mask"]

                                              
            for idx in range(actual_batch_size):
                yes_idx = idx * 2
                no_idx = idx * 2 + 1
                prompt_len = prompt_lens[idx]

                                                  
                seq_len_yes = int(attn_cpu[yes_idx].sum().item())
                ans_len_yes = seq_len_yes - prompt_len
                if ans_len_yes > 0:
                    shift_logits = logits[yes_idx, prompt_len - 1 : seq_len_yes - 1, :]
                    shift_labels = input_ids_cpu[yes_idx, prompt_len : seq_len_yes]
                    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                    token_lps = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                    ll_yes = token_lps.sum().item() / ans_len_yes
                else:
                    ll_yes = float("-inf")

                                                 
                seq_len_no = int(attn_cpu[no_idx].sum().item())
                ans_len_no = seq_len_no - prompt_len
                if ans_len_no > 0:
                    shift_logits = logits[no_idx, prompt_len - 1 : seq_len_no - 1, :]
                    shift_labels = input_ids_cpu[no_idx, prompt_len : seq_len_no]
                    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                    token_lps = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
                    ll_no = token_lps.sum().item() / ans_len_no
                else:
                    ll_no = float("-inf")

                                                                               
                predicted = (ll_yes > ll_no)
                if predicted == answers[idx]:
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



                         


def compute_pruning_sensitivity(model_name, num_layers, dataset, tokenizer,
                                 logger, sparsity_level=0.3, baseline_acc=None,
                                 max_samples=0, batch_size=64):
    """
    Compute pruning-based sensitivity for each layer using accuracy drop.

    Sensitivity = baseline_accuracy - pruned_accuracy
    Positive sensitivity = layer is important.
    """
    rank = get_rank()

    if baseline_acc is None:
        if rank == 0:
            logger.info("Computing baseline accuracy (no pruning)...")
            logger.info("  Model loaded on CPU → moved to HPU for eval")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if rank == 0:
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"  Model params: {param_count:,} ({param_count/1e6:.1f}M)")
            logger.info(f"  Model dtype:  {next(model.parameters()).dtype}")
        baseline_acc, base_time = evaluate_boolq_accuracy(
            model, dataset, tokenizer, logger,
            eval_name="Baseline", max_samples=max_samples, batch_size=batch_size
        )
        del model
        free_hpu_memory()
        if rank == 0:
            mem = get_hpu_memory_info()
            mem_str = f" (HPU: {mem['allocated_mb']:.0f} MB after cleanup)" if mem else ""
            logger.info(f"Baseline Accuracy: {baseline_acc:.2f}%{mem_str}")

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
        pruned_acc, prune_time = evaluate_boolq_accuracy(
            pruned_model, dataset, tokenizer, logger,
            eval_name=f"Layer {layer_idx} Pruned", max_samples=max_samples, batch_size=batch_size
        )

        sensitivity = baseline_acc - pruned_acc
        layer_sensitivities[f"layer_{layer_idx}"] = float(sensitivity)

        del pruned_model
        free_hpu_memory()

        if rank == 0:
            mem = get_hpu_memory_info()
            mem_str = f" | HPU: {mem['allocated_mb']:.0f} MB" if mem else ""
            logger.info(f"[Layer {layer_idx}] Baseline: {baseline_acc:.2f}% | "
                         f"Pruned: {pruned_acc:.2f}% | Sensitivity: {sensitivity:.4f}{mem_str}")

    if rank == 0:
        logger.info("=" * 80)
        logger.info("Pruning sensitivity computation complete!")

    return layer_sensitivities, baseline_acc



               


def main():
    parser = argparse.ArgumentParser(
        description='Compute pruning-based sensitivity on BoolQ (PMPQ) - Gaudi HPU'
    )
    parser.add_argument('--sparsity', type=float, default=0.3,
                        help='Sparsity level for pruning (0-1, default: 0.3 = 30%%)')
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='Max BoolQ samples per evaluation (0 = full 9427 train samples, default: 2500)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference (default: 16, higher=faster but more memory)')
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
                    if sel == '2':
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
        logger.info("PMPQ PHASE 1: PRUNING-BASED SENSITIVITY ANALYSIS (BoolQ)")
        logger.info("=" * 80)
        logger.info(f"Cache:     {os.environ['HF_HOME']}")
        logger.info(f"Log dir:   {LOG_DIR}")
        mode_str = f"Distributed: {world_size} Gaudi cards" if world_size > 1 else "Single-card"
        logger.info(f"Mode:      {mode_str}")
        logger.info(f"Sparsity:  {args.sparsity*100}%")
        samples_str = "FULL" if args.max_samples == 0 else f"{args.max_samples}"
        logger.info(f"Eval:      BoolQ train ({samples_str} samples)")
        logger.info(f"Metric:    Accuracy (Yes/No log-likelihood)")
        logger.info(f"Inference: Direct HPU (lazy mode + FP32)")
        logger.info("")

                                    
        logger.info("Compute Device Mapping:")
        logger.info("  Model inference (forward pass)  → HPU (direct, lazy mode)")
        logger.info("  Weight pruning (masking)        → CPU (lightweight)")
        logger.info("  Dataset loading                 → CPU")
        logger.info("  Log-likelihood computation      → CPU")
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
        logger.info("STEP 2: LOADING BOOLQ TRAIN DATASET")
        logger.info("=" * 80)

    dataset = load_boolq_dataset(split="train")

    if rank == 0:
        logger.info(f"BoolQ train: {len(dataset)} samples")
        if args.max_samples > 0:
            logger.info(f"  Using subset: {min(args.max_samples, len(dataset))} samples")
        logger.info(f"  Official TinyLlama-1.1B score: 57.83 (accuracy)")

                                                                   
    if rank == 0:
        logger.info("=" * 80)
        logger.info("STEP 3: COMPUTING PRUNING-BASED SENSITIVITIES")
        logger.info("=" * 80)

    t0 = time.time()
    layer_sensitivities, baseline_acc = compute_pruning_sensitivity(
        model_name=model_name, num_layers=num_layers,
        dataset=dataset, tokenizer=tokenizer,
        logger=logger, sparsity_level=args.sparsity,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    sensitivity_time = time.time() - t0

                                                              
    if rank == 0:
        logger.info(f"Sensitivity computation complete in {sensitivity_time:.2f}s")

        logger.info("")
        logger.info("Pruning-Based Sensitivities (Accuracy Drop %):")
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
        samples_tag = f"_n{args.max_samples}" if args.max_samples > 0 else "_full"
        filename_base = f"sens_{model_key}_BoolQ{samples_tag}_pruning_s{int(args.sparsity*100)}_{timestamp}"

                   
        json_path = os.path.join(sens_dir, f"{filename_base}.json")
        with open(json_path, "w") as f:
            json.dump(layer_sensitivities, f, indent=2)

                                
        txt_path = os.path.join(sens_dir, f"{filename_base}.txt")
        samples_used = min(args.max_samples, len(dataset)) if args.max_samples > 0 else len(dataset)
        with open(txt_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("LAYER SENSITIVITY FILE - PRUNING-BASED (PMPQ) ON BOOLQ\n")
            f.write("=" * 80 + "\n\n")
            f.write("=" * 80 + "\n")
            f.write("CONFIGURATION\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model: {model_key}\n")
            f.write(f"Model HF Hub: {model_name}\n")
            f.write(f"Task: Question Answering (BoolQ, Yes/No)\n")
            f.write(f"Dataset: BoolQ (google/boolq)\n")
            f.write(f"Method: PMPQ (Pruning-based Mixed-Precision Quantization)\n")
            f.write(f"Metric: Accuracy (Yes/No log-likelihood)\n")
            f.write(f"Sparsity Level: {args.sparsity*100}%\n")
            f.write(f"Evaluation Set: VALIDATION ({len(dataset)} samples)\n")
            f.write(f"Samples Used: {samples_used}\n")
            f.write(f"Num Layers: {num_layers}\n")
            f.write(f"Hidden Dim: {model_config['hidden_dim']}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Platform: Intel Gaudi HPU (direct inference, lazy mode, FP32)\n")
            f.write(f"Num Accelerators: {world_size}\n")
            f.write(f"Baseline Accuracy: {baseline_acc:.2f}%\n")
            f.write(f"Official TinyLlama Score: 57.83 (accuracy)\n")
            if mem:
                f.write(f"HPU Peak Memory: {mem['max_allocated_mb']:.1f} MB\n")
            f.write(f"\n")
            f.write("=" * 80 + "\n")
            f.write("COMPUTE DEVICE MAPPING\n")
            f.write("=" * 80 + "\n")
            f.write("Model inference (forward pass) → HPU (direct, lazy mode)\n")
            f.write("Weight pruning (masking)       → CPU\n")
            f.write("Dataset loading                → CPU\n")
            f.write("Log-likelihood computation     → CPU\n")
            f.write("I/O, logging, saving           → CPU\n\n")
            f.write("=" * 80 + "\n")
            f.write("COMPUTATION STATISTICS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total Time: {sensitivity_time:.2f}s\n")
            f.write(f"Time per Layer: {sensitivity_time/num_layers:.2f}s\n\n")
            f.write("=" * 80 + "\n")
            f.write("LAYER SENSITIVITIES (PRUNING-BASED - ACCURACY DROP %)\n")
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
            f.write(f"3. Move model to HPU, evaluate accuracy on BoolQ (FP32, lazy mode)\n")
            f.write(f"4. Sensitivity = Baseline_Accuracy - Pruned_Accuracy\n")
            f.write("\nBoolQ Dataset:\n")
            f.write("- Yes/No question answering task\n")
            f.write("- Given passage + question, predict True/False\n")
            f.write("- Metric: accuracy (log-likelihood of Yes vs No)\n")
            f.write("- Matches TinyLlama official benchmark evaluation\n")

        logger.info(f"Saved: {filename_base}.json / .txt")

                                    
        summary_log = os.path.join(LOG_DIR, f"boolq_sensitivity_results_{timestamp}.txt")
        with open(summary_log, "w") as f:
            f.write(f"PMPQ BoolQ Sensitivity Results — {timestamp}\n")
            f.write(f"Model: {model_key}\n")
            f.write(f"Baseline Accuracy: {baseline_acc:.2f}%\n")
            f.write(f"Official TinyLlama: 57.83%\n")
            f.write(f"Sparsity: {args.sparsity*100}%\n")
            f.write(f"Samples: {samples_used}\n")
            f.write(f"Time: {sensitivity_time:.2f}s\n")
            f.write(f"Cards: {world_size}\n")
            if mem:
                f.write(f"HPU Peak Memory: {mem['max_allocated_mb']:.1f} MB\n")
            f.write(f"\nSensitivities (accuracy drop %):\n")
            for ln, sv in sorted_layers:
                f.write(f"  {ln}: {sv:.4f}\n")

        logger.info(f"Summary saved to: {summary_log}")

        logger.info("=" * 80)
        logger.info("PHASE 1 COMPLETE (BoolQ)")
        logger.info("=" * 80)
        logger.info(f"  Model: {model_key} ({num_layers} layers)")
        logger.info(f"  Baseline Accuracy: {baseline_acc:.2f}%")
        logger.info(f"  Official Score: 57.83%")
        logger.info(f"  Cards: {world_size}")
        logger.info(f"  Time:  {sensitivity_time:.2f}s")

             
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
