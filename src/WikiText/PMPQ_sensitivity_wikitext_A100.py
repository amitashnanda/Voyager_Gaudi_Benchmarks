                                                
"""
Phase 1: Pruning-Based Sensitivity Analysis for TinyLlama on WikiText-2 (PMPQ)
================================================================================
PMPQ -- Pruning-based Mixed-Precision Quantization

Algorithm (unchanged from working implementation):
  For each transformer layer i (0 to 21):
    1. Load a FRESH copy of the full FP32 model from HuggingFace
    2. Apply magnitude-based pruning to ONLY layer i:
         - For every parameter in that layer, flatten it
         - Keep top (1 - sparsity_level)% of weights by absolute value
         - Zero out the rest
         - All other layers remain completely unchanged
    3. Evaluate perplexity on WikiText-2 VALIDATION split
         (full split, 512-token chunks, continuous text, no padding)
    4. sensitivity[i] = pruned_perplexity - baseline_perplexity

  baseline_perplexity = perplexity of original unmodified model on same
  WikiText-2 VALIDATION split (full split, matches HPU).

  Higher perplexity increase -> more sensitive layer -> needs higher bits.
  Lower perplexity increase  -> less sensitive layer -> can use lower bits.

PMPQ Assumption:
  Layers sensitive to magnitude pruning are also sensitive to quantization.

Calibration: WikiText-2 VALIDATION split, full split (matches HPU).
Sparsity:    0.3 (30%) default. Controlled via --sparsity argument.

Usage:
  python Phase_1_PMPQ_TinyLlama_WikiText_Sensitivity.py
  python Phase_1_PMPQ_TinyLlama_WikiText_Sensitivity.py --sparsity 0.3

Output:
  Sensitivities/sens_PMPQ_TinyLlama_<timestamp>.json   -- for Phase 2
  Evaluation/phase1_PMPQ_sensitivity_TinyLlama_<timestamp>.txt -- full log

Author: Mixed-Precision Quantization Team
Date: 2025-2026
"""

                                                                              
                   
                                                                              
import os

HF_HOME = os.environ.get("HF_HOME", "/pscratch/sd/s/sreeb12/.cache/huggingface")
os.makedirs(HF_HOME, exist_ok=True)

os.environ.update({
    "HF_HOME":               HF_HOME,
    "HF_DATASETS_CACHE":     os.path.join(HF_HOME, "datasets"),
    "HF_HUB_CACHE":          os.path.join(HF_HOME, "hub"),
    "TOKENIZERS_PARALLELISM": "false",
    "CUDA_VISIBLE_DEVICES":  os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
})

from pathlib import Path
for p in (os.environ["HF_DATASETS_CACHE"], os.environ["HF_HUB_CACHE"]):
    Path(p).mkdir(parents=True, exist_ok=True)

print("Environment setup - cache:", HF_HOME)

                                                                              
         
                                                                              
import json, time, random, argparse, warnings
import numpy as np
from datetime import datetime
from tqdm import tqdm
from itertools import chain

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings('ignore')

print("All imports complete")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    num_gpus_available = torch.cuda.device_count()
    print(f"GPUs detected: {num_gpus_available}")
    for i in range(num_gpus_available):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  |  {props.total_memory / 1024**3:.1f} GB")

                                                                              
               
                                                                              

TINYLLAMA_MODELS = {
    "TinyLlama-1.1B": {
        "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "num_layers":  22,
        "hidden_dim":  2048,
        "description": "Compact 1.1B model trained on 3T tokens"
    }
}

CALIBRATION_SPLIT   = "validation"                                               
SEQUENCE_LENGTH     = 512
DEFAULT_GROUP_SIZE  = 128                                                  


                                                                              
           
                                                                              

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
        print("MPS (Apple Silicon)")
        return torch.device("mps")
    print("CPU (slow)")
    return torch.device("cpu")


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


def get_cuda_memory_info():
    """Get CUDA memory usage across all GPUs."""
    if not torch.cuda.is_available():
        return None

    num_gpus = torch.cuda.device_count()
    total_allocated = 0
    total_reserved = 0
    total_max_allocated = 0

    for i in range(num_gpus):
        total_allocated += torch.cuda.memory_allocated(i) / (1024 ** 2)      
        total_reserved += torch.cuda.memory_reserved(i) / (1024 ** 2)      
        total_max_allocated += torch.cuda.max_memory_allocated(i) / (1024 ** 2)      

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


                                                                              
                                                  
                                                                              

def prepare_wikitext_dataset(tokenizer, split="validation", block_size=512):
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

                       
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc=f"Tokenizing {split}"
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

    chunked = tokenized.map(
        group_texts,
        batched=True,
        desc=f"Chunking {split} into {block_size}-token blocks"
    )

    return chunked


                                                                              
                                                 
                                                                              

def apply_magnitude_pruning_to_layer(layer, sparsity_level):
    """
    Apply magnitude-based pruning to a specific layer.
    Sets the smallest magnitude weights to zero.

    Args:
        layer:         nn.Module -- the single transformer layer to prune
        sparsity_level: float (0-1) -- fraction of weights to zero out
    """
    with torch.no_grad():
        for param in layer.parameters():
            if param.requires_grad:
                flat_param = param.data.view(-1)

                                                                        
                k = int(flat_param.numel() * (1 - sparsity_level))
                if k == 0:
                    continue

                                                        
                threshold = torch.topk(
                    torch.abs(flat_param), k, largest=True
                )[0][-1]

                                                  
                mask = torch.abs(flat_param) >= threshold
                param.data *= mask.float().view(param.data.shape)


                                                                              
                                                                           
                                                                              

def evaluate_perplexity(model, eval_dataset, device, eval_name="Evaluation"):
    """
    Evaluate perplexity using pre-prepared dataset.
    Matches HPU implementation.

    Args:
        model: nn.Module (can be wrapped in DDP)
        eval_dataset: HuggingFace dataset with input_ids and labels
        device: torch.device
        eval_name: label for progress bar

    Returns:
        perplexity (float), eval_time_s (float),
        total_tokens (int), throughput (tokens/s float)
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    loss_fct = nn.CrossEntropyLoss()

    t_start = time.time()

    with torch.no_grad():
        for sample in tqdm(eval_dataset, desc=eval_name, leave=False):
            input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
            labels = torch.tensor(sample["labels"]).unsqueeze(0).to(device)

            outputs = model(input_ids)
            logits = outputs.logits

                                 
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

                                                                                
            seq_len = input_ids.size(1)
            total_loss += loss.item() * seq_len
            total_tokens += seq_len

    eval_time = time.time() - t_start
    ppl = np.exp(total_loss / total_tokens)
    throughput = total_tokens / eval_time if eval_time > 0 else 0.0

    return ppl, eval_time, total_tokens, throughput


                                                                              
                              
                                                                              

def compute_pruning_sensitivity(model, model_name, num_layers,
                                tokenizer, device, eval_dataset,
                                sparsity_level=0.3):
    """
    Compute pruning-based sensitivity for every transformer layer.

    Steps:
      1. Evaluate baseline perplexity on validation split (full dataset).
      2. For each layer i:
           a. Load a FRESH full model from HuggingFace.
           b. Prune ONLY layer i with magnitude pruning (sparsity_level).
           c. Evaluate perplexity on same validation split.
           d. sensitivity[i] = pruned_ppl - baseline_ppl.
           e. Record per-layer time.
           f. Delete pruned model, empty GPU cache.

    Returns:
        layer_sensitivities : dict  {"layer_0": float, ..., "layer_21": float}
        layer_times         : dict  {"layer_0": float_seconds, ...}
        baseline_ppl        : float
        baseline_time       : float
    """
    num_samples = len(eval_dataset)
    total_tokens = num_samples * SEQUENCE_LENGTH

    print(f"\nEvaluating FP32 baseline on WikiText-2 {CALIBRATION_SPLIT} split")
    print(f"  Samples: {num_samples} × {SEQUENCE_LENGTH} tokens = {total_tokens:,} tokens")

    baseline_ppl, baseline_time, _, _ = evaluate_perplexity(
        model, eval_dataset, device,
        eval_name="Baseline"
    )
    print(f"  Baseline Perplexity : {baseline_ppl:.4f}")
    print(f"  Baseline Eval Time  : {format_duration(baseline_time)}")

    layer_sensitivities = {}
    layer_times         = {}

    print(f"\nComputing per-layer pruning sensitivity ({num_layers} layers)...")
    print(f"Sparsity level : {sparsity_level*100:.0f}%")
    print(f"Calibration    : {CALIBRATION_SPLIT} split, {num_samples} samples")
    print("-"*80)

    for layer_idx in range(num_layers):
        print(f"\n  [Layer {layer_idx:2d}/{num_layers-1}] Loading fresh model...")
        layer_t0 = time.time()

                                  
        pruned_model = AutoModelForCausalLM.from_pretrained(model_name)

        if torch.cuda.device_count() > 1:
            pruned_model = nn.DataParallel(pruned_model)

        pruned_model = pruned_model.to(device)

                                       
        if hasattr(pruned_model, 'module'):
            base = pruned_model.module
        else:
            base = pruned_model

        if hasattr(base, 'model'):
            target_layer = base.model.layers[layer_idx]
        else:
            target_layer = base.layers[layer_idx]

                                            
        print(f"  [Layer {layer_idx:2d}] Applying {sparsity_level*100:.0f}% pruning...")
        apply_magnitude_pruning_to_layer(target_layer, sparsity_level)

                                            
        print(f"  [Layer {layer_idx:2d}] Evaluating perplexity...")
        pruned_ppl, _, _, _ = evaluate_perplexity(
            pruned_model, eval_dataset, device,
            eval_name=f"Layer {layer_idx:2d} Pruned"
        )

                                                        
        sensitivity = pruned_ppl - baseline_ppl
        layer_time  = time.time() - layer_t0

        layer_sensitivities[f"layer_{layer_idx}"] = float(sensitivity)
        layer_times[f"layer_{layer_idx}"]         = float(layer_time)

        print(f"  [Layer {layer_idx:2d}] Baseline: {baseline_ppl:.4f} | "
              f"Pruned: {pruned_ppl:.4f} | "
              f"Sensitivity: {sensitivity:.4f} | "
              f"Time: {format_duration(layer_time)}")

                               
        del pruned_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("Pruning sensitivity computation complete!")
    print("="*80)

    return layer_sensitivities, layer_times, baseline_ppl, baseline_time


                                                                              
               
                                                                              

def main():
    parser = argparse.ArgumentParser(
        description="PMPQ Phase 1 -- Pruning-based sensitivity for TinyLlama")
    parser.add_argument(
        '--sparsity', type=float, default=0.3,
        help="Sparsity level for pruning (0-1, default: 0.3 = 30%%)")
    args = parser.parse_args()

    print_section("PHASE 1: PMPQ SENSITIVITY ANALYSIS -- TinyLlama on WikiText-2")
    print(f"""
  Method : PMPQ (Pruning-based Mixed-Precision Quantization)
  Algorithm:
    For each layer i:
      1. Load fresh FP32 model
      2. Apply {args.sparsity*100:.0f}% magnitude pruning to layer i ONLY
      4. sensitivity[i] = pruned_ppl - baseline_ppl


  Sparsity    : {args.sparsity*100:.0f}%
  Seq length  : {SEQUENCE_LENGTH} tokens
  Group size  : {DEFAULT_GROUP_SIZE} (stored for Phase 2 reference)
    """)

    pipeline_t0 = time.time()
    set_seed(42)
    device   = pick_device()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

                                                                                
                        
                                                                                
    print_section("STEP 1: LOADING MODEL")
    model_key    = "TinyLlama-1.1B"
    model_config = TINYLLAMA_MODELS[model_key]
    model_name   = model_config["model_name"]
    num_layers   = model_config["num_layers"]

    t0        = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if torch.cuda.device_count() > 1:
        print(f"  Wrapping with DataParallel across {torch.cuda.device_count()} GPUs...")
        model = nn.DataParallel(model)

    model = model.to(device)
    model_loading_time = time.time() - t0
    fp32_size_mb       = get_model_size_mb(model)

    print(f"  Model     : {model_key}")
    print(f"  Layers    : {num_layers}  |  Hidden: {model_config['hidden_dim']}")
    print(f"  Loaded in : {format_duration(model_loading_time)}")
    print(f"  FP32 size : {fp32_size_mb:.2f} MB")

                                                                                
                             
                                                                                
    print_section("STEP 2: PREPARING DATASET")
    eval_dataset = prepare_wikitext_dataset(tokenizer, split=CALIBRATION_SPLIT, block_size=SEQUENCE_LENGTH)
    num_samples = len(eval_dataset)
    total_tokens = num_samples * SEQUENCE_LENGTH
    print(f"WikiText-2 {CALIBRATION_SPLIT}: {num_samples} samples × {SEQUENCE_LENGTH} tokens")
    print(f"Total tokens: {total_tokens:,}")

                                                                                
                                           
                                                                                
    print_section("STEP 3: COMPUTING PRUNING-BASED SENSITIVITIES")

    sensitivity_t0 = time.time()
    layer_sensitivities, layer_times, baseline_ppl, baseline_time =\
        compute_pruning_sensitivity(
            model=model,
            model_name=model_name,
            num_layers=num_layers,
            tokenizer=tokenizer,
            device=device,
            eval_dataset=eval_dataset,
            sparsity_level=args.sparsity
        )
    sensitivity_total_time = time.time() - sensitivity_t0

    total_pipeline_time = time.time() - pipeline_t0

                   
    print(f"\n  Sensitivity computation: {format_duration(sensitivity_total_time)}")
    print(f"  Avg time per layer    : {format_duration(sensitivity_total_time / num_layers)}")
    print(f"  Min time per layer    : {format_duration(min(layer_times.values()))}")
    print(f"  Max time per layer    : {format_duration(max(layer_times.values()))}")
    print(f"\n  Per-layer sensitivities (perplexity increase):")
    print(f"  {'Layer':<10} {'Sensitivity':<16}")
    print(f"  {'-'*26}")
    for i in range(num_layers):
        print(f"  layer_{i:<4}  {layer_sensitivities[f'layer_{i}']:.6f}")

                                                                                
                                                 
                                                                                
    print_section("STEP 3: SAVING SENSITIVITY FILES")
    os.makedirs("Sensitivities", exist_ok=True)
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"sens_PMPQ_TinyLlama_{timestamp}.json"
    json_path     = os.path.join("Sensitivities", json_filename)

                                              
    sv = np.array([layer_sensitivities[f"layer_{i}"] for i in range(num_layers)],
                  dtype=np.float32)
    ranked = sorted(range(num_layers), key=lambda i: sv[i], reverse=True)

                                                                                    
    json_data = {
        "sensitivities": layer_sensitivities,
        "metadata": {
            "method":               "PMPQ",
            "sensitivity_type":     "pruning_perplexity_increase",
            "formula":              "sensitivity[i] = pruned_ppl - baseline_ppl",
            "model_key":            model_key,
            "model_name":           model_name,
            "num_layers":           num_layers,
            "hidden_dim":           model_config["hidden_dim"],
            "group_size":           DEFAULT_GROUP_SIZE,
            "sparsity_level":       args.sparsity,
            "calibration_split":    CALIBRATION_SPLIT,
            "calibration_samples":  num_samples,
            "total_tokens":         total_tokens,
            "sequence_length":      SEQUENCE_LENGTH,
            "baseline_ppl_train":   float(baseline_ppl),
            "fp32_model_size_mb":   float(fp32_size_mb),
            "device":               str(device),
            "num_gpus":             num_gpus,
            "timestamp":            timestamp,
            "note": (
                "Higher sensitivity = more perplexity increase when pruned "
                "= more sensitive to quantization = assign higher bit-width."
            )
        },
        "timing": {
            "model_loading_time_s":         float(model_loading_time),
            "baseline_evaluation_time_s":   float(baseline_time),
            "sensitivity_computation_time_s": float(sensitivity_total_time),
            "total_pipeline_time_s":        float(total_pipeline_time),
            "per_layer_times_s":            {k: float(v)
                                             for k, v in layer_times.items()},
        },
        "ranked_layers_most_to_least_sensitive": [int(i) for i in ranked],
    }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"  Sensitivity JSON saved: {json_path}")

                                                                                
                                                                             
                                                                                
    print_section("STEP 4: SAVING RESULTS LOG")
    os.makedirs("Evaluation", exist_ok=True)
    log_filename = f"phase1_PMPQ_sensitivity_TinyLlama_{timestamp}.txt"
    log_path     = os.path.join("Evaluation", log_filename)

    with open(log_path, "w") as f:

        f.write("="*80 + "\n")
        f.write("LAYER SENSITIVITY FILE - PRUNING-BASED (PMPQ) ON WIKITEXT\n")
        f.write("="*80 + "\n\n")

                                                                                 
        f.write("="*80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {model_key}\n")
        f.write(f"Model HF Hub: {model_name}\n")
        f.write(f"Task: Language Modeling (Perplexity)\n")
        f.write(f"Dataset: WikiText-2\n")
        f.write(f"Method: PMPQ (Pruning-based Mixed-Precision Quantization)\n")
        f.write(f"Sparsity Level: {args.sparsity * 100:.0f}%\n")
        f.write(f"Calibration Split: {CALIBRATION_SPLIT} (full split)\n")
        f.write(f"Calibration Samples: {num_samples} × {SEQUENCE_LENGTH} tokens\n")
        f.write(f"Total Tokens: {total_tokens:,}\n")
        f.write(f"Tokenization: HuggingFace .map() with continuous chunking, "
                f"no padding\n")
        f.write(f"Num Layers: {num_layers}\n")
        f.write(f"Hidden Dim: {model_config['hidden_dim']}\n")
        f.write(f"Group Size (for Phase 2): {DEFAULT_GROUP_SIZE}\n")
        f.write(f"FP32 Model Size: {fp32_size_mb:.2f} MB\n")
        f.write(f"Baseline Perplexity ({CALIBRATION_SPLIT}): {baseline_ppl:.6f}\n")
        f.write(f"Timestamp: {timestamp}\n")
        if torch.cuda.is_available():
            mem = get_cuda_memory_info()
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            if mem:
                f.write(f"Peak CUDA Memory: {mem['max_allocated_mb']:.1f} MB across {mem['num_gpus']} GPUs\n")
        else:
            f.write("GPU: CPU\n")
        f.write(f"Num GPUs: {num_gpus}\n\n")

                                                                                 
        f.write("="*80 + "\n")
        f.write("DETAILED TIMING LOG\n")
        f.write("="*80 + "\n")
        f.write(f"Model Loading Time:              "
                f"{format_duration(model_loading_time)} "
                f"({model_loading_time:.4f}s)\n")
        f.write(f"Baseline Evaluation Time:        "
                f"{format_duration(baseline_time)} "
                f"({baseline_time:.4f}s)\n")
        f.write(f"Sensitivity Computation Time:    "
                f"{format_duration(sensitivity_total_time)} "
                f"({sensitivity_total_time:.4f}s)\n")
        f.write(f"  - Avg per layer:               "
                f"{format_duration(sensitivity_total_time / num_layers)} "
                f"({sensitivity_total_time / num_layers:.4f}s)\n")
        f.write(f"  - Min per layer:               "
                f"{format_duration(min(layer_times.values()))} "
                f"({min(layer_times.values()):.4f}s)\n")
        f.write(f"  - Max per layer:               "
                f"{format_duration(max(layer_times.values()))} "
                f"({max(layer_times.values()):.4f}s)\n")
        f.write(f"Total Pipeline Time:             "
                f"{format_duration(total_pipeline_time)} "
                f"({total_pipeline_time:.4f}s)\n\n")

                                                                                 
        f.write("="*80 + "\n")
        f.write("PER-LAYER PRUNING SENSITIVITY COMPUTATION TIMES\n")
        f.write("="*80 + "\n")
        f.write(f"{'Layer':<10} {'Time (s)':<16} {'% of Total':<14}\n")
        f.write("-"*40 + "\n")
        for i in range(num_layers):
            lt  = layer_times[f"layer_{i}"]
            pct = (lt / sensitivity_total_time * 100.0) if sensitivity_total_time > 0 else 0.0
            f.write(f"layer_{i:<4} {lt:<16.4f} {pct:<14.2f}\n")
        f.write("\n")

                                                                                 
        f.write("="*80 + "\n")
        f.write("LAYER SENSITIVITIES (PRUNING-BASED -- PERPLEXITY INCREASE)\n")
        f.write("="*80 + "\n")
        f.write(f"{'Layer':<10} {'Sensitivity':<22} {'Rank':<10}\n")
        f.write("-"*80 + "\n")
        for rank_idx, layer_i in enumerate(ranked, 1):
            lname = f"layer_{layer_i}"
            f.write(f"{lname:<10} {layer_sensitivities[lname]:<22.6f} {rank_idx:<10}\n")
        f.write("\n")

                                                                                 
        f.write("="*80 + "\n")
        f.write("SENSITIVITY STATISTICS\n")
        f.write("="*80 + "\n")
        f.write(f"Mean Sensitivity: {sv.mean():.6f}\n")
        f.write(f"Min Sensitivity:  {sv.min():.6f}\n")
        f.write(f"Max Sensitivity:  {sv.max():.6f}\n")
        f.write(f"Std Deviation:    {sv.std():.6f}\n\n")

                                                                                 
        f.write("="*80 + "\n")
        f.write("MACHINE-READABLE METRICS (KEY-VALUE)\n")
        f.write("="*80 + "\n")
        f.write(f"method: PMPQ\n")
        f.write(f"sensitivity_type: pruning_perplexity_increase\n")
        f.write(f"model_key: {model_key}\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"num_layers: {num_layers}\n")
        f.write(f"hidden_dim: {model_config['hidden_dim']}\n")
        f.write(f"sparsity_level: {args.sparsity}\n")
        f.write(f"group_size: {DEFAULT_GROUP_SIZE}\n")
        f.write(f"calibration_split: {CALIBRATION_SPLIT}\n")
        f.write(f"calibration_samples: {num_samples}\n")
        f.write(f"total_tokens: {total_tokens}\n")
        f.write(f"sequence_length: {SEQUENCE_LENGTH}\n")
        f.write(f"baseline_ppl_train: {baseline_ppl:.6f}\n")
        f.write(f"fp32_model_size_mb: {fp32_size_mb:.2f}\n")
        f.write(f"sensitivity_min: {sv.min():.6f}\n")
        f.write(f"sensitivity_max: {sv.max():.6f}\n")
        f.write(f"sensitivity_mean: {sv.mean():.6f}\n")
        f.write(f"sensitivity_std: {sv.std():.6f}\n")
        f.write(f"device: {device}\n")
        f.write(f"num_gpus: {num_gpus}\n")
        f.write(f"output_sensitivity_file: {json_path}\n")
        f.write(f"\n# Timing (seconds):\n")
        f.write(f"model_loading_time_s: {model_loading_time:.4f}\n")
        f.write(f"baseline_evaluation_time_s: {baseline_time:.4f}\n")
        f.write(f"sensitivity_computation_time_s: {sensitivity_total_time:.4f}\n")
        f.write(f"total_pipeline_time_s: {total_pipeline_time:.4f}\n")
        f.write(f"avg_per_layer_time_s: {sensitivity_total_time / num_layers:.4f}\n")
        f.write(f"\n# Per-layer times (seconds):\n")
        for i in range(num_layers):
            f.write(f"layer_{i}_time_s: {layer_times[f'layer_{i}']:.4f}\n")

                                                                                 
        f.write("\n" + "="*80 + "\n")
        f.write("LAYER BIT ALLOCATION\n")
        f.write("="*80 + "\n")
        f.write("Note: Bit allocation assigned in Phase 2 based on these scores.\n")
        f.write("      Higher sensitivity -> assigned higher bit-width.\n\n")
        for i in range(num_layers):
            f.write(f"  layer_{i:02d}: sensitivity={layer_sensitivities[f'layer_{i}']:.8f}\n")

                                                                                 
        f.write("\n" + "="*80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("="*80 + "\n")
        f.write("Higher sensitivity = Layer is more affected by weight pruning\n")
        f.write("Lower sensitivity  = Layer can tolerate more weight removal\n\n")
        f.write("PMPQ Assumption: Layers sensitive to pruning are also\n")
        f.write("sensitive to quantization. Use these sensitivities in Phase 2\n")
        f.write("to assign different bit-widths for mixed-precision quantization.\n\n")

                                                                                 
        f.write("="*80 + "\n")
        f.write("METHOD NOTES\n")
        f.write("="*80 + "\n")
        f.write(f"Sensitivity Analysis Method: PMPQ\n")
        f.write(f"  PMPQ (Pruning-based Mixed-Precision Quantization) computes\n")
        f.write(f"  layer sensitivity by measuring how much perplexity increases\n")
        f.write(f"  when {args.sparsity*100:.0f}% of the smallest-magnitude weights in that\n")
        f.write(f"  layer alone are set to zero.\n\n")
        f.write(f"  For each layer i:\n")
        f.write(f"    1. Load fresh FP32 model from HuggingFace\n")
        f.write(f"    2. Apply {args.sparsity*100:.0f}% magnitude pruning to layer i ONLY\n")
        f.write(f"       (all other layers unchanged)\n")
        f.write(f"    3. Evaluate perplexity on WikiText-2 {CALIBRATION_SPLIT.upper()} split\n")
        f.write(f"       ({num_samples} samples, {SEQUENCE_LENGTH} tokens each)\n")
        f.write(f"    4. sensitivity[i] = pruned_ppl - baseline_ppl\n\n")
        f.write(f"Calibration Data: WikiText-2 {CALIBRATION_SPLIT.upper()} split\n")
        f.write(f"  Full split: {num_samples} samples of {SEQUENCE_LENGTH} tokens each.\n")
        f.write(f"  Total tokens: {total_tokens:,}\n")
        f.write(f"  Used for baseline and all per-layer pruned evaluations.\n")
        f.write(f"  {CALIBRATION_SPLIT.capitalize()} split is used for Phase 1 calibration.\n")
        f.write(f"  Phase 2 evaluates on WikiText-2 TEST split.\n\n")
        f.write(f"Group-Wise Quantization Reference:\n")
        f.write(f"  group_size={DEFAULT_GROUP_SIZE} stored in metadata for Phase 2.\n")

    print(f"  Results log saved: {log_path}")

                                                                                
                   
                                                                                
    print_section("PHASE 1 COMPLETE -- PMPQ SENSITIVITY ANALYSIS FINISHED")
    print(f"""
  Model         : {model_key} ({num_layers} layers)
  Method        : PMPQ -- magnitude pruning + perplexity increase
  Sparsity      : {args.sparsity*100:.0f}%
  Calibration   : WikiText-2 {CALIBRATION_SPLIT.upper()} ({num_samples} samples)
  Baseline PPL  : {baseline_ppl:.4f}
  Sensitivity   : [{sv.min():.4f}, {sv.max():.4f}]  mean={sv.mean():.4f}
  Total time    : {format_duration(total_pipeline_time)}
  Avg per layer : {format_duration(sensitivity_total_time / num_layers)}

  Saved:
    JSON  : {json_path}
    Log   : {log_path}

  Next step: Run Phase 2 (FAKE or REAL) using this sensitivity file.
    """)


if __name__ == "__main__":
    main()