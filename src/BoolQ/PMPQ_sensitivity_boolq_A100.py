                                             
"""
Phase 1: Pruning-Based Sensitivity Analysis for TinyLlama on BoolQ (PMPQ)
================================================================================
PMPQ -- Pruning-based Mixed-Precision Quantization

Algorithm:
  For each transformer layer i (0 to 21):
    1. Load a FRESH copy of the full FP32 model from HuggingFace
    2. Apply magnitude-based pruning to ONLY layer i:
         - For every parameter in that layer, flatten it
         - Calculate threshold = quantile(|W_flat|, sparsity)
         - Create mask = (|W_flat| > threshold)
         - Apply: W_pruned = W * mask  (zero out small weights)
         - All other layers remain completely unchanged
    3. Evaluate accuracy on BoolQ TRAIN split
         (first 2000 samples, binary yes/no log-likelihood scoring)
    4. sensitivity[i] = baseline_accuracy - pruned_accuracy
         (Accuracy Drop = higher drop -> more sensitive layer)

  baseline_accuracy = accuracy of original unmodified model on same
  BoolQ TRAIN split (first 2000 samples).

  Higher accuracy drop -> more sensitive layer -> needs higher bits.
  Lower accuracy drop  -> less sensitive layer -> can use lower bits.

PMPQ Assumption:
  Layers sensitive to magnitude pruning are also sensitive to quantization.

Calibration: BoolQ TRAIN split, first 2000 samples, fixed.
Sparsity:    0.3 (30%) default. Controlled via --sparsity argument.

Usage:
  python Phase_1_PMPQ_TinyLlama_BoolQ_Sensitivity.py
  python Phase_1_PMPQ_TinyLlama_BoolQ_Sensitivity.py --sparsity 0.3 --batch_size 32

Output:
  Sensitivities/sens_PMPQ_TinyLlama_BoolQ_<timestamp>.json  -- for Phase 2
  Sensitivities/phase1_PMPQ_BoolQ_sensitivity_TinyLlama_<timestamp>.txt -- full log

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

CALIBRATION_SPLIT   = "train"                                    
MAX_LENGTH          = 512
DEFAULT_GROUP_SIZE  = 128                                                 

                                                     
BOOLQ_YES_TOKENS = [" yes", "yes", " Yes", "Yes"]
BOOLQ_NO_TOKENS  = [" no",  "no",  " No",  "No"]


                                                                              
           
                                                                              

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
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

    Args:
        layer:          nn.Module -- the single transformer layer to prune
        sparsity_level: float (0-1) -- fraction of weights to zero out
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


                                                                              
                           
                                                                              

@torch.no_grad()
def evaluate_boolq_accuracy(model, tokenizer, device,
                             split="train",
                             max_samples=None,
                             max_length=512,
                             batch_size=64,
                             eval_name="Evaluation"):
    """
    Evaluate accuracy on BoolQ using log-likelihood scoring with batched inference.

    For each example:
      - Build prompt = "Passage: {passage}\\nQuestion: {question}\\nAnswer:"
      - Score "yes" and "no" continuations via avg token log-likelihood
      - Pick the answer with the higher score
      - Compare to ground truth label (True -> yes, False -> no)

    Args:
        model:       TinyLlama model (original or pruned)
        tokenizer:   Tokenizer
        device:      torch.device
        split:       dataset split (train for Phase 1 calibration)
        max_samples: cap on number of examples (None = use all)
        max_length:  max tokenization length
        batch_size:  number of samples to process at once
        eval_name:   label shown on progress bar

    Returns:
        accuracy (float), num_correct (int), num_total (int),
        eval_time_s (float), throughput (examples/s float)
    """
    ds = load_dataset("google/boolq", split=split)

    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    num_total    = len(ds)
    num_correct  = 0
    all_examples = list(ds)

    model.eval()
    actual_model = model.module if hasattr(model, 'module') else model
    t_start      = time.time()

    print(f"[{eval_name}] Evaluating {num_total} samples on {device} (batch_size={batch_size})...")

                        
    for batch_idx in range(0, num_total, batch_size):
        batch_end = min(batch_idx + batch_size, num_total)
        batch_samples = all_examples[batch_idx:batch_end]
        actual_batch_size = len(batch_samples)

                            
        all_texts = []
        prompt_lens = []
        answers = []

        for sample in batch_samples:
            passage = sample.get("passage", "")
            question = sample.get("question", "")
            answer = sample.get("answer", False)                 
            answers.append(answer)

                                              
            prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"

                               
            prompt_enc = tokenizer(prompt, add_special_tokens=True)
            prompt_len = len(prompt_enc["input_ids"])
            prompt_lens.append(prompt_len)

                                                             
            all_texts.append(prompt + " yes")
            all_texts.append(prompt + " no")

                                                             
        batch = tokenizer(all_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_length)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

                                                                
        outputs = actual_model(input_ids=input_ids, attention_mask=attention_mask)

                                                             
        logits = outputs.logits.float().cpu()
        input_ids_cpu = batch["input_ids"].cpu()
        attention_mask_cpu = batch["attention_mask"].cpu()

                                        
        for sample_idx in range(actual_batch_size):
            yes_idx = sample_idx * 2
            no_idx = sample_idx * 2 + 1
            prompt_len = prompt_lens[sample_idx]

            candidate_scores = []
            for cand_idx in [yes_idx, no_idx]:
                if input_ids_cpu.shape[1] <= prompt_len:
                    candidate_scores.append(float('-inf'))
                    continue

                shift_logits = logits[cand_idx, prompt_len-1:-1, :]
                shift_labels = input_ids_cpu[cand_idx, prompt_len:]
                ending_mask = attention_mask_cpu[cand_idx, prompt_len:]

                log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs[range(len(shift_labels)), shift_labels]
                token_log_probs = token_log_probs * ending_mask.float()

                valid_tokens = ending_mask.sum().item()
                if valid_tokens == 0:
                    candidate_scores.append(float('-inf'))
                else:
                                                      
                    candidate_scores.append(token_log_probs.sum().item() / valid_tokens)

                                                     
            pred_yes = candidate_scores[0] > candidate_scores[1]

                                                                  
            if pred_yes == bool(answers[sample_idx]):
                num_correct += 1

                         
        if (batch_end % (batch_size * 10)) == 0 or batch_end == num_total:
            current_acc = num_correct / batch_end if batch_end > 0 else 0.0
            print(f"  Progress: {batch_end}/{num_total} samples | Accuracy: {current_acc:.4f}")

    eval_time  = time.time() - t_start
    accuracy   = num_correct / num_total if num_total > 0 else 0.0
    throughput = num_total / eval_time if eval_time > 0 else 0.0

    print(f"[{eval_name}] Completed: {num_correct}/{num_total} correct | Accuracy: {accuracy:.4f}")

    return accuracy, num_correct, num_total, eval_time, throughput


                                                                              
                              
                                                                              

def compute_pruning_sensitivity(model, model_name, num_layers,
                                tokenizer, device,
                                sparsity_level=0.3,
                                max_samples=2000,
                                batch_size=64):
    """
    Compute pruning-based sensitivity for every transformer layer.

    Steps:
      1. Evaluate baseline accuracy on BoolQ train split (max_samples examples).
      2. For each layer i:
           a. Load a FRESH full model from HuggingFace.
           b. Prune ONLY layer i with magnitude pruning (sparsity_level).
           c. Evaluate accuracy on same train split.
           d. sensitivity[i] = baseline_accuracy - pruned_accuracy
              (Accuracy Drop = Baseline Accuracy - Pruned Accuracy)
           e. Record per-layer time.
           f. Delete pruned model, empty GPU cache.

    Returns:
        layer_sensitivities : dict  {"layer_0": float, ..., "layer_21": float}
        layer_times         : dict  {"layer_0": float_seconds, ...}
        baseline_accuracy   : float
        baseline_correct    : int
        baseline_total      : int
        baseline_time       : float
        baseline_throughput : float
    """
    print(f"\nEvaluating FP32 baseline on BoolQ {CALIBRATION_SPLIT} split "
          f"(first {max_samples} samples)...")

    baseline_acc, baseline_correct, baseline_total, baseline_time, baseline_tp =\
        evaluate_boolq_accuracy(
            model, tokenizer, device,
            split=CALIBRATION_SPLIT,
            max_samples=max_samples,
            max_length=MAX_LENGTH,
            batch_size=batch_size,
            eval_name="Baseline"
        )

    print(f"  Baseline Accuracy   : {baseline_acc:.6f} ({baseline_acc*100:.2f}%)")
    print(f"  Correct             : {baseline_correct}/{baseline_total}")
    print(f"  Baseline Eval Time  : {format_duration(baseline_time)}")
    print(f"  Throughput          : {baseline_tp:.2f} examples/s")

    layer_sensitivities = {}
    layer_times         = {}
    layer_pruned_accs   = {}

    print(f"\nComputing per-layer pruning sensitivity ({num_layers} layers)...")
    print(f"Sparsity level : {sparsity_level*100:.0f}%")
    print(f"Calibration    : {CALIBRATION_SPLIT} split, {max_samples} samples")
    print(f"Sensitivity    : Accuracy Drop = Baseline Accuracy - Pruned Accuracy")
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

                                            
        print(f"  [Layer {layer_idx:2d}] Evaluating accuracy...")
        pruned_acc, pruned_correct, pruned_total, _, _ = evaluate_boolq_accuracy(
            pruned_model, tokenizer, device,
            split=CALIBRATION_SPLIT,
            max_samples=max_samples,
            max_length=MAX_LENGTH,
            batch_size=batch_size,
            eval_name=f"Layer {layer_idx:2d} Pruned"
        )

                                                  
        sensitivity = baseline_acc - pruned_acc
        layer_time  = time.time() - layer_t0

        layer_sensitivities[f"layer_{layer_idx}"] = float(sensitivity)
        layer_times[f"layer_{layer_idx}"]         = float(layer_time)
        layer_pruned_accs[f"layer_{layer_idx}"]   = float(pruned_acc)

        print(f"  [Layer {layer_idx:2d}] Baseline: {baseline_acc:.6f} | "
              f"Pruned: {pruned_acc:.6f} | "
              f"Sensitivity (drop): {sensitivity:.6f} | "
              f"Time: {format_duration(layer_time)}")

                               
        del pruned_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("Pruning sensitivity computation complete!")
    print("="*80)

    return (layer_sensitivities, layer_times, layer_pruned_accs,
            baseline_acc, baseline_correct, baseline_total,
            baseline_time, baseline_tp)


                                                                              
               
                                                                              

def main():
    parser = argparse.ArgumentParser(
        description="PMPQ Phase 1 -- Pruning-based sensitivity for TinyLlama on BoolQ")
    parser.add_argument(
        '--sparsity', type=float, default=0.3,
        help="Sparsity level for pruning (0-1, default: 0.3 = 30%%)")
    parser.add_argument(
        '--max_samples', type=int, default=2000,
        help='Max BoolQ samples per evaluation (0 = full dataset, default: 2000)')
    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size for inference (default: 64, reduce if OOM)')
    args = parser.parse_args()

    print_section("PHASE 1: PMPQ SENSITIVITY ANALYSIS -- TinyLlama on BoolQ")
    samples_str = "FULL" if args.max_samples == 0 else f"{args.max_samples}"
    print(f"""
  Method : PMPQ (Pruning-based Mixed-Precision Quantization)
  Algorithm:
    For each layer i:
      1. Load fresh FP32 model
      2. Apply {args.sparsity*100:.0f}% magnitude pruning to layer i ONLY
         (threshold = quantile(|W_flat|, sparsity), mask = |W| > threshold)
      3. Evaluate accuracy on BoolQ TRAIN split ({samples_str} samples)
      4. sensitivity[i] = baseline_accuracy - pruned_accuracy
         (Accuracy Drop)

  Calibration : BoolQ TRAIN split ({samples_str} samples)
  Sparsity    : {args.sparsity*100:.0f}%
  Max length  : {MAX_LENGTH} tokens
  Batch size  : {args.batch_size}
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

                                                                                
                                           
                                                                                
    print_section("STEP 2: COMPUTING PRUNING-BASED SENSITIVITIES")

    sensitivity_t0 = time.time()
    (layer_sensitivities, layer_times, layer_pruned_accs,
     baseline_acc, baseline_correct, baseline_total,
     baseline_time, baseline_tp) = compute_pruning_sensitivity(
        model=model,
        model_name=model_name,
        num_layers=num_layers,
        tokenizer=tokenizer,
        device=device,
        sparsity_level=args.sparsity,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    sensitivity_total_time = time.time() - sensitivity_t0
    total_pipeline_time    = time.time() - pipeline_t0

                                              
    sv = np.array([layer_sensitivities[f"layer_{i}"] for i in range(num_layers)],
                  dtype=np.float32)
    ranked = sorted(range(num_layers), key=lambda i: sv[i], reverse=True)

    print(f"\n  Sensitivity computation: {format_duration(sensitivity_total_time)}")
    print(f"  Avg time per layer    : {format_duration(sensitivity_total_time / num_layers)}")
    print(f"  Min time per layer    : {format_duration(min(layer_times.values()))}")
    print(f"  Max time per layer    : {format_duration(max(layer_times.values()))}")
    print(f"\n  Per-layer sensitivities (accuracy drop):")
    print(f"  {'Layer':<10} {'Sensitivity':<16} {'Pruned Acc':<16}")
    print(f"  {'-'*42}")
    for i in range(num_layers):
        print(f"  layer_{i:<4}  "
              f"{layer_sensitivities[f'layer_{i}']:.6f}       "
              f"{layer_pruned_accs[f'layer_{i}']:.6f}")

                                                                                
                                                 
                                                                                
    print_section("STEP 3: SAVING SENSITIVITY FILES")
    os.makedirs("Sensitivities", exist_ok=True)
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_filename = f"sens_PMPQ_TinyLlama_BoolQ_{timestamp}.json"
    json_path     = os.path.join("Sensitivities", json_filename)

    json_data = {
        "sensitivities": layer_sensitivities,
        "metadata": {
            "method":               "PMPQ",
            "sensitivity_type":     "pruning_accuracy_drop",
            "formula":              "sensitivity[i] = baseline_accuracy - pruned_accuracy",
            "model_key":            model_key,
            "model_name":           model_name,
            "num_layers":           num_layers,
            "hidden_dim":           model_config["hidden_dim"],
            "group_size":           DEFAULT_GROUP_SIZE,
            "sparsity_level":       args.sparsity,
            "calibration_split":    CALIBRATION_SPLIT,
            "calibration_samples":  args.max_samples,
            "max_length":           MAX_LENGTH,
            "batch_size":           args.batch_size,
            "baseline_accuracy":    float(baseline_acc),
            "baseline_correct":     int(baseline_correct),
            "baseline_total":       int(baseline_total),
            "fp32_model_size_mb":   float(fp32_size_mb),
            "device":               str(device),
            "num_gpus":             num_gpus,
            "timestamp":            timestamp,
            "note": (
                "Higher sensitivity = more accuracy drop when pruned "
                "= more sensitive to quantization = assign higher bit-width."
            )
        },
        "timing": {
            "model_loading_time_s":           float(model_loading_time),
            "baseline_evaluation_time_s":     float(baseline_time),
            "sensitivity_computation_time_s": float(sensitivity_total_time),
            "total_pipeline_time_s":          float(total_pipeline_time),
            "per_layer_times_s":              {k: float(v) for k, v in layer_times.items()},
        },
        "per_layer_pruned_accuracies": layer_pruned_accs,
        "ranked_layers_most_to_least_sensitive": [int(i) for i in ranked],
    }

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"  Sensitivity JSON saved: {json_path}")

                                                                                
                                   
                                                                                
    print_section("STEP 4: SAVING RESULTS LOG")
    os.makedirs("Sensitivities", exist_ok=True)
    log_filename = f"phase1_PMPQ_BoolQ_sensitivity_TinyLlama_{timestamp}.txt"
    log_path     = os.path.join("Sensitivities", log_filename)

    with open(log_path, "w") as f:

        f.write("="*80 + "\n")
        f.write("LAYER SENSITIVITY FILE - PRUNING-BASED (PMPQ) ON BOOLQ\n")
        f.write("="*80 + "\n\n")

                       
        f.write("="*80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {model_key}\n")
        f.write(f"Model HF Hub: {model_name}\n")
        f.write(f"Task: Boolean Question Answering (BoolQ -- binary yes/no)\n")
        f.write(f"Dataset: BoolQ\n")
        f.write(f"Method: PMPQ (Pruning-based Mixed-Precision Quantization)\n")
        f.write(f"Sparsity Level: {args.sparsity * 100:.0f}%\n")
        samples_used_str = "FULL dataset" if args.max_samples == 0 else f"first {args.max_samples} samples"
        f.write(f"Calibration Split: {CALIBRATION_SPLIT} ({samples_used_str})\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Tokenization: Max length {MAX_LENGTH} tokens, padding=True\n")
        f.write(f"Sensitivity Type: Accuracy Drop "
                f"(Baseline Accuracy - Pruned Accuracy)\n")
        f.write(f"Num Layers: {num_layers}\n")
        f.write(f"Hidden Dim: {model_config['hidden_dim']}\n")
        f.write(f"Group Size (for Phase 2): {DEFAULT_GROUP_SIZE}\n")
        f.write(f"FP32 Model Size: {fp32_size_mb:.2f} MB\n")
        f.write(f"Baseline Accuracy (train): {baseline_acc:.6f} "
                f"({baseline_acc*100:.2f}%)\n")
        f.write(f"Baseline Correct: {baseline_correct}/{baseline_total}\n")
        f.write(f"Timestamp: {timestamp}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
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
        f.write("LAYER SENSITIVITIES (PRUNING-BASED -- ACCURACY DROP)\n")
        f.write("="*80 + "\n")
        f.write(f"{'Layer':<10} {'Sensitivity':<22} {'Rank':<10} "
                f"{'Pruned Acc':<16} {'Baseline Acc':<16}\n")
        f.write("-"*80 + "\n")
        for rank_idx, layer_i in enumerate(ranked, 1):
            lname = f"layer_{layer_i}"
            f.write(f"{lname:<10} "
                    f"{layer_sensitivities[lname]:<22.6f} "
                    f"{rank_idx:<10} "
                    f"{layer_pruned_accs[lname]:<16.6f} "
                    f"{baseline_acc:<16.6f}\n")
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
        f.write(f"sensitivity_type: pruning_accuracy_drop\n")
        f.write(f"model_key: {model_key}\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"num_layers: {num_layers}\n")
        f.write(f"hidden_dim: {model_config['hidden_dim']}\n")
        f.write(f"sparsity_level: {args.sparsity}\n")
        f.write(f"group_size: {DEFAULT_GROUP_SIZE}\n")
        f.write(f"calibration_split: {CALIBRATION_SPLIT}\n")
        f.write(f"calibration_samples: {args.max_samples}\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"max_length: {MAX_LENGTH}\n")
        f.write(f"baseline_accuracy: {baseline_acc:.6f}\n")
        f.write(f"baseline_correct: {baseline_correct}\n")
        f.write(f"baseline_total: {baseline_total}\n")
        f.write(f"baseline_throughput: {baseline_tp:.2f}\n")
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
            f.write(f"  layer_{i:02d}: sensitivity={layer_sensitivities[f'layer_{i}']:.8f}  "
                    f"pruned_acc={layer_pruned_accs[f'layer_{i}']:.6f}\n")

                        
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
        f.write(f"  layer sensitivity by measuring how much accuracy drops\n")
        f.write(f"  when {args.sparsity*100:.0f}% of the smallest-magnitude weights in that\n")
        f.write(f"  layer alone are set to zero.\n\n")
        f.write(f"  For each layer i:\n")
        f.write(f"    1. Load fresh FP32 model from HuggingFace\n")
        f.write(f"    2. Flatten weights: W_flat = flatten(layer.weight)\n")
        f.write(f"    3. Compute threshold: threshold = quantile(|W_flat|, {args.sparsity})\n")
        f.write(f"    4. Create mask: mask = (|W_flat| > threshold)\n")
        f.write(f"    5. Apply pruning to layer i ONLY: W_pruned = W * mask\n")
        f.write(f"       (all other layers unchanged)\n")
        f.write(f"    6. Evaluate accuracy on BoolQ TRAIN split\n")
        samples_desc = "FULL dataset" if args.max_samples == 0 else f"first {args.max_samples} samples"
        f.write(f"       ({samples_desc}, {MAX_LENGTH}-token max length, batch_size={args.batch_size})\n")
        f.write(f"    7. sensitivity[i] = baseline_accuracy - pruned_accuracy\n\n")
        f.write(f"BoolQ Accuracy Evaluation:\n")
        f.write(f"  For each example: build prompt as\n")
        f.write(f"  \"Passage: {{passage}}\\nQuestion: {{question}}\\nAnswer:\"\n")
        f.write(f"  Score \"yes\" and \"no\" continuations via avg token log-likelihood\n")
        f.write(f"  given the prompt. Pick the higher-scoring answer and compare\n")
        f.write(f"  to ground truth label (True=yes, False=no).\n")
        f.write(f"  Metric: Accuracy (fraction of correctly answered questions).\n\n")
        f.write(f"Calibration Data: BoolQ TRAIN split\n")
        samples_cal_desc = "FULL dataset" if args.max_samples == 0 else f"First {args.max_samples} samples"
        f.write(f"  {samples_cal_desc}. Fixed for all layers.\n")
        f.write(f"  Used for baseline and all per-layer pruned evaluations.\n")
        f.write(f"  Phase 2 evaluates on BoolQ VALIDATION split.\n\n")
        f.write(f"Group-Wise Quantization Reference:\n")
        f.write(f"  group_size={DEFAULT_GROUP_SIZE} stored in metadata for Phase 2.\n")

    print(f"  Results log saved: {log_path}")

                                                                                
                   
                                                                                
    print_section("PHASE 1 COMPLETE -- PMPQ SENSITIVITY ANALYSIS FINISHED")
    print(f"""
  Model         : {model_key} ({num_layers} layers)
  Method        : PMPQ -- magnitude pruning + accuracy drop
  Sparsity      : {args.sparsity*100:.0f}%
  Calibration   : BoolQ TRAIN ({samples_str} samples)
  Batch size    : {args.batch_size}
  Baseline Acc  : {baseline_acc:.6f} ({baseline_acc*100:.2f}%)
  Sensitivity   : [{sv.min():.6f}, {sv.max():.6f}]  mean={sv.mean():.6f}
  Total time    : {format_duration(total_pipeline_time)}
  Avg per layer : {format_duration(sensitivity_total_time / num_layers)}

  Saved:
    JSON  : {json_path}
    Log   : {log_path}

  Next step: Run Phase 2 (FAKE or REAL) using this sensitivity file.
    """)


if __name__ == "__main__":
    main()