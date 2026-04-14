                                                        
"""
Phase 2: Group-Wise Quantization & Evaluation for TinyLlama on HellaSwag
=========================================================================
QUANTIZATION TYPE: FAKE (Simulated)
  Weights remain FP32 throughout. Quantization is simulated:
  FP32 -> quantize -> dequantize -> store back as FP32.
  No real memory savings. Used to measure accuracy impact of quantization.

Supports sensitivity files from:
  - CMPQ Phase 1 (SVCCA, PWCCA, CKA)
  - PMPQ Phase 1 (magnitude pruning accuracy drop)

Uses Phase 1 sensitivity files for mixed-precision PTQ quantization with
GROUP-WISE scales (like GPTQ). Evaluates on HellaSwag validation set.

HellaSwag Evaluation:
- For each example, compute log-likelihood of each of the 4 candidate endings
  given the context using the causal LM
- Pick the ending with the highest log-likelihood
- Report accuracy (% of correct picks)

Baseline Caching:
- FP32 baseline results are saved to Models/baseline_hellaswag_fp32.json
- If baseline skip is selected, results are loaded from that file
- This ensures reliable comparisons every time

Model Saving (.bin HuggingFace format):
- Saved to Models/fake_quant_TinyLlama_HellaSwag_<bits>_gs<gs>_<strategy>_<method>/
  containing: config.json, tokenizer.json, pytorch_model-*.bin, quant_config.json

Comprehensive timing logs are recorded for EVERY step.

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
    "CUDA_VISIBLE_DEVICES":  os.environ.get("CUDA_VISIBLE_DEVICES", "0"),                                     
})

print("Environment setup - cache:", HF_HOME)

                                                                              
         
                                                                              
import json, time, random, warnings, re
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

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
    num_gpus_available = torch.cuda.device_count()
    print(f"GPUs detected: {num_gpus_available}")
    for i in range(num_gpus_available):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}  |  {props.total_memory / 1024**3:.1f} GB")

                                                                              
               
                                                                              

DEFAULT_GROUP_SIZE  = 128
MAX_LENGTH          = 2048                                         
BATCH_SIZE          = 64                                             
BASELINE_CACHE_DIR  = "Models"
BASELINE_CACHE_FILE = os.path.join(BASELINE_CACHE_DIR, "baseline_hellaswag_fp32.json")

TINYLLAMA_MODELS = {
    "TinyLlama-1.1B": {
        "model_name": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "num_layers":  22,
        "hidden_dim":  2048,
        "description": "Compact 1.1B model trained on 3T tokens"
    }
}


                                                                              
           
                                                                              

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
            if not raw and default:
                return default
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print(f"Enter 1-{len(options)}")
        except ValueError:
            print("Invalid input.")


def prompt_yes_no(prompt_text, default="yes"):
    d = "[Y/n]" if default == "yes" else "[y/N]"
    while True:
        inp = input(f"\n{prompt_text} {d}: ").strip().lower()
        if not inp:            return default == "yes"
        if inp in ("y","yes"): return True
        if inp in ("n","no"):  return False
        print("Enter 'y' or 'n'")


def print_section(title):
    print(f"\n{'='*80}\n  {title}\n{'='*80}")


def format_duration(s):
    if s < 60:    return f"{s:.2f}s"
    if s < 3600:  return f"{int(s//60)}m {s%60:.2f}s"
    h = int(s//3600); m = int((s%3600)//60)
    return f"{h}h {m}m {s%60:.2f}s"


def bits_to_mb(bits):
    return bits / 8.0 / (1024.0 * 1024.0)


def get_model_size_mb(model):
    total  = sum(p.nelement() * p.element_size() for p in model.parameters())
    total += sum(b.nelement() * b.element_size() for b in model.buffers())
    return total / (1024 * 1024)


                                                                              
                
                                                                              

def save_baseline_cache(acc, correct, total, eval_time, throughput,
                        model_key, model_name, device_str, num_gpus, timestamp):
    os.makedirs(BASELINE_CACHE_DIR, exist_ok=True)
    data = {
        "accuracy":     acc,
        "correct":      correct,
        "total":        total,
        "eval_time_s":  eval_time,
        "throughput":   throughput,
        "model_key":    model_key,
        "model_name":   model_name,
        "dataset":      "HellaSwag",
        "split":        "validation",
        "device":       device_str,
        "num_gpus":     num_gpus,
        "quantization": "FP32_baseline",
        "timestamp":    timestamp,
        "note":         "FP32 baseline stored for future skip-baseline comparisons"
    }
    with open(BASELINE_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Baseline results saved to: {BASELINE_CACHE_FILE}")


def load_baseline_cache():
    if not os.path.exists(BASELINE_CACHE_FILE):
        return None
    with open(BASELINE_CACHE_FILE, "r") as f:
        return json.load(f)


def extract_baseline_from_log(log_file_path):
    """
    Extract FP32 baseline metrics from a previous evaluation log file.

    Looks for patterns like:
    - Accuracy: 56.8612% or Accuracy  : 0.5686 (56.86%)
    - Correct   : 5710/10042
    - Eval Time: 460.58s or Time      : 12m 31.21s
    - Throughput: 21.80 samples/s or Throughput: 13.37 examples/s

    Returns: dict with baseline metrics or None if not found
    """
    if not os.path.exists(log_file_path):
        print(f"  Log file not found: {log_file_path}")
        return None

    try:
        with open(log_file_path, "r") as f:
            content = f.read()

                                            
        acc_match = re.search(r"Accuracy\s*[:]\s*(\d+\.?\d*)[%]?", content) or\
                   re.search(r"Accuracy\s*[:]\s*0\.(\d+)", content)
        if acc_match:
            if "%" in acc_match.group(0):
                accuracy = float(acc_match.group(1)) / 100.0
            else:
                accuracy = float(acc_match.group(0).split(":")[-1].strip().split()[0])
        else:
            print("  Could not extract accuracy from log")
            return None

                               
        correct_match = re.search(r"Correct\s*[:]\s*(\d+)/(\d+)", content)
        if correct_match:
            correct = int(correct_match.group(1))
            total = int(correct_match.group(2))
        else:
            print("  Could not extract correct/total from log")
            return None

                                     
        time_match = re.search(r"Eval Time\s*[:]\s*(\d+\.?\d*)s", content) or\
                    re.search(r"Time\s*[:]\s*(\d+)m\s*(\d+\.?\d*)s", content)
        if time_match:
            if "m" in time_match.group(0):
                minutes = int(time_match.group(1))
                seconds = float(time_match.group(2))
                eval_time = minutes * 60 + seconds
            else:
                eval_time = float(time_match.group(1))
        else:
            print("  Could not extract eval time from log")
            eval_time = 0.0

                            
        throughput_match = re.search(r"Throughput\s*[:]\s*(\d+\.?\d*)\s*(?:samples|examples)/s", content)
        if throughput_match:
            throughput = float(throughput_match.group(1))
        else:
            throughput = total / eval_time if eval_time > 0 else 0.0

                                         
        model_match = re.search(r"Model\s*[:]\s*([^\n]+)", content)
        model_name = model_match.group(1).strip() if model_match else "TinyLlama-1.1B"

                                                    
        timestamp_match = re.search(r"(\d{8}_\d{6})", log_file_path)
        timestamp = timestamp_match.group(1) if timestamp_match else datetime.now().strftime("%Y%m%d_%H%M%S")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "eval_time_s": eval_time,
            "throughput": throughput,
            "model_name": model_name,
            "timestamp": timestamp,
            "source_file": os.path.basename(log_file_path)
        }

    except Exception as e:
        print(f"  Error reading log file: {e}")
        return None


def find_latest_evaluation_log(eval_dir="Evaluation"):
    """
    Find the most recent evaluation log file that contains FP32 baseline results.

    Looks in the Evaluation directory for files matching patterns like:
    - ptq_eval_*.txt
    - hellaswag_eval_*.txt

    Returns: path to the latest log file or None
    """
    if not os.path.exists(eval_dir):
        return None

                                   
    patterns = [
        "ptq_eval_*.txt",
        "hellaswag_eval_*.txt",
        "*_eval_*.txt"
    ]

    log_files = []
    for pattern in patterns:
        log_files.extend(Path(eval_dir).rglob(pattern))

    if not log_files:
        return None

                                                 
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                                               
    for log_file in log_files[:10]:                                 
        try:
            with open(log_file, "r") as f:
                content = f.read(2000)                  
                if "BASELINE" in content or "FP32" in content:
                    return str(log_file)
        except:
            continue

    return None


                                                                              
            
                                                                              

def kmeans_clustering(sensitivities, n_clusters=3):
    values = sensitivities.reshape(-1, 1)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(values)
    means = [(c, float(values[labels==c].mean())) for c in range(n_clusters)]
    means.sort(key=lambda x: x[1], reverse=True)
    return labels, means


def hierarchical_clustering(sensitivities, n_clusters=3):
    values = sensitivities.reshape(-1, 1)
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = agg.fit_predict(values)
    means = [(c, float(values[labels==c].mean())) for c in range(n_clusters)]
    means.sort(key=lambda x: x[1], reverse=True)
    return labels, means


def percentile_clustering(sensitivities, n_clusters=3):
    n = len(sensitivities)
    pairs = sorted(enumerate(sensitivities), key=lambda x: x[1], reverse=True)
    cs = n // n_clusters
    labels = np.zeros(n, dtype=int)
    means = []
    for cid in range(n_clusters):
        s = cid * cs
        e = s + cs if cid < n_clusters - 1 else n
        idxs = [pairs[i][0] for i in range(s, e)]
        for i in idxs:
            labels[i] = cid
        means.append((cid, float(np.mean([sensitivities[i] for i in idxs]))))
    means.sort(key=lambda x: x[1], reverse=True)
    return labels, means


                                                                              
                                           
                                                                              

class LinearLSQGroupWise(nn.Module):
    """
    FAKE group-wise quantization (simulated).
    Weights: FP32 -> quantize -> dequantize -> stored back as FP32.
    No real memory savings. Measures accuracy impact only.
    Any bit-width accepted (2, 4, 6, 8, 12, 16, 32).
    """

    def __init__(self, linear: nn.Linear, nbits_w: int, group_size: int = 128):
        super().__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.nbits_w      = int(nbits_w)
        self.group_size   = group_size

        qw = self._quantize_weight_groupwise(linear.weight.detach().clone())
        self.register_buffer("weight", qw)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.register_buffer("bias", None)

    def _quantize_weight_groupwise(self, weight):
        if self.nbits_w >= 32:
            return weight
        out_f, in_f = weight.shape
        orig_in = in_f
        if in_f % self.group_size != 0:
            pad = self.group_size - (in_f % self.group_size)
            weight = torch.nn.functional.pad(weight, (0, pad), value=0)
            in_f = weight.shape[1]
        num_groups = in_f // self.group_size
        wg   = weight.view(out_f, num_groups, self.group_size)
        qmin = -(2 ** (self.nbits_w - 1))
        qmax = (2 ** (self.nbits_w - 1)) - 1
        scale = torch.clamp(wg.abs().amax(dim=-1, keepdim=True) / qmax, min=1e-8)
        q    = torch.clamp(torch.round(wg / scale), qmin, qmax)
        wq   = (q * scale).view(out_f, -1)
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
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def quantize_model_fake(model, layer_bits_map, group_size=128):
    actual_model = model
    targets = []
    for name, module in actual_model.named_modules():
        if isinstance(module, nn.Linear):
            m = re.search(r"(?:model\.|decoder\.)layers\.(\d+)\.", name)
            if m:
                layer_idx = int(m.group(1))
                nbits = layer_bits_map.get(layer_idx, 8)
                targets.append((name, module, nbits))

    print(f"  Found {len(targets)} Linear layers to quantize (FAKE / simulated)")
    print(f"  Group size: {group_size}")
    bit_counts = {}
    for _, _, b in targets:
        bit_counts[b] = bit_counts.get(b, 0) + 1
    for b in sorted(bit_counts.keys(), reverse=True):
        print(f"  {b}-bit layers: {bit_counts[b]}")

    total_orig = total_quant = 0
    for qualname, linear_mod, nbits in tqdm(targets, desc="  Fake-quantizing layers"):
        wrapper = LinearLSQGroupWise(linear_mod, nbits, group_size)
        _set_module(actual_model, qualname, wrapper)
        o, q = wrapper.calculate_weight_bits()
        total_orig  += o
        total_quant += q
    return model, total_orig, total_quant


                                                                              
                                             
                                                                              

def preprocess_hellaswag_text(text):
    """Clean up HellaSwag text artifacts from WikiHow portion."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


@torch.no_grad()
def evaluate_hellaswag(model, tokenizer, device,
                       eval_name="Evaluation",
                       split="validation",
                       max_length=2048,
                       batch_size=64,
                       max_examples=None):
    """
    Evaluate accuracy on HellaSwag using log-likelihood scoring with batched inference.

    For each sample:
      1. Context = ctx (preprocessed)
      2. For each of 4 candidate endings:
         - Tokenize context + " " + ending
         - Forward pass → logits
         - Compute length-normalized log-likelihood of ending tokens
      3. Pick ending with highest normalized log-likelihood (acc_norm)

    Batched processing: Processes batch_size samples in parallel for faster inference.
    This matches the HPU implementation and lm-evaluation-harness methodology.

    Args:
        model:        TinyLlama model (baseline or quantized)
        tokenizer:    Tokenizer
        device:       torch.device
        eval_name:    label for progress display
        split:        dataset split (validation for evaluation)
        max_length:   max tokenization length (default: 2048 to match HPU)
        batch_size:   number of samples to process in parallel (default: 32)
        max_examples: cap on number of examples (None = use all)

    Returns:
        accuracy (float), num_correct (int), num_total (int), eval_time_s (float)
    """
    ds = load_dataset("Rowan/hellaswag", split=split, trust_remote_code=True)
    if max_examples is not None and max_examples > 0:
        ds = ds.select(range(min(max_examples, len(ds))))

    samples = ds
    total = len(samples)
    correct = 0

    model.eval()
    actual_model = model.module if hasattr(model, 'module') else model
    start_time = time.time()

    print(f"[{eval_name}] Evaluating {total} samples on {device} (FP32, batch_size={batch_size})...")

                        
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
                        truncation=True, max_length=max_length)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

                                                                
        outputs = actual_model(input_ids, attention_mask=attention_mask)

                                                    
        logits = outputs.logits.float().cpu()
        input_ids_cpu = batch["input_ids"]
        attn_cpu = batch["attention_mask"]

                                                       
        for sample_idx in range(actual_batch_size):
            ctx_len = ctx_lens[sample_idx]
            label = labels[sample_idx]
            scores = []

            for ending_idx in range(4):
                idx = sample_idx * 4 + ending_idx
                seq_logits = logits[idx]
                seq_ids = input_ids_cpu[idx]
                seq_mask = attn_cpu[idx]

                                                               
                                        
                ending_start = ctx_len
                ending_logits = seq_logits[ending_start - 1 : -1]                
                ending_ids = seq_ids[ending_start:]
                ending_mask = seq_mask[ending_start:]

                                                  
                log_probs = torch.nn.functional.log_softmax(ending_logits, dim=-1)
                token_log_probs = log_probs[range(len(ending_ids)), ending_ids]
                token_log_probs = token_log_probs * ending_mask.float()

                valid_tokens = ending_mask.sum().item()
                if valid_tokens > 0:
                    score = token_log_probs.sum().item() / valid_tokens
                else:
                    score = float('-inf')

                scores.append(score)

                                            
            pred = int(np.argmax(scores))
            if pred == label:
                correct += 1

                         
        if (batch_end % (batch_size * 10)) == 0 or batch_end == total:
            current_acc = correct / batch_end if batch_end > 0 else 0.0
            print(f"  Progress: {batch_end}/{total} samples | Accuracy: {current_acc:.4f}")

    eval_time = time.time() - start_time
    accuracy = correct / total if total > 0 else 0.0

    print(f"[{eval_name}] Completed: {correct}/{total} correct | Accuracy: {accuracy:.4f}")

    return accuracy, correct, total, eval_time


                                                                              
               
                                                                              

def main():
    print_section("PHASE 2: HellaSwag Evaluation -- FAKE QUANTIZATION (Simulated)")
    print("""
  FAKE QUANTIZATION:
    Weights simulated: FP32 -> quantize -> dequantize -> stored back as FP32.
    Any bit-width accepted (2, 4, 6, 8, 12, 16, 32).
    No real memory savings. Used to measure accuracy impact of quantization.

  Supports CMPQ sensitivity files (SVCCA, PWCCA, CKA) and
  PMPQ sensitivity files (magnitude pruning accuracy drop).

  Baseline caching:
    FP32 results saved to Models/baseline_hellaswag_fp32.json.
    When baseline is skipped, results are loaded from that file.

  Model saving:
    Saved to Models/<name>/ in HuggingFace .bin sharded format.
    """)

    timing_log          = {}
    pipeline_start_time = time.time()

    set_seed(42)
    device   = pick_device()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"  Device: {device}  |  GPUs: {num_gpus}")

                                                                                
                                   
                                                                                
    print_section("STEP 1: LOAD SENSITIVITY FILE")
    t0 = time.time()

    sens_dir = Path("Sensitivities")
    if not sens_dir.exists():
        print("'Sensitivities' folder not found. Run Phase 1 first.")
        return

    sens_files = list(sens_dir.glob("sens_*.json"))
    if not sens_files:
        print("No sensitivity files found.")
        return

    sens_file_names = [f.name for f in sens_files]
    print(f"Found {len(sens_file_names)} sensitivity file(s):")
    for i, fn in enumerate(sens_file_names, 1):
        print(f"  {i}. {fn}")

    selected_file = prompt_user("Select sensitivity file:", sens_file_names,
                                default=sens_file_names[0])
    sens_path = sens_dir / selected_file

    with open(sens_path, "r") as f:
        sens_data = json.load(f)

    if "sensitivities" in sens_data:
        sensitivities           = sens_data["sensitivities"]
        sens_metadata           = sens_data.get("metadata", {})
        phase1_group_size       = sens_metadata.get("group_size", DEFAULT_GROUP_SIZE)
        phase1_sensitivity_time = sens_data.get("timing", {}).get(
            "sensitivity_computation_time_s", None)
    else:
        sensitivities           = sens_data
        phase1_group_size       = DEFAULT_GROUP_SIZE
        phase1_sensitivity_time = None

    num_layers  = max([int(k.split("_")[1]) for k in sensitivities.keys()]) + 1
    sens_values = np.array(
        [sensitivities[f"layer_{i}"] for i in range(num_layers)], dtype=np.float32
    )

    timing_log["sensitivity_file_loading_time_s"] = time.time() - t0
    print(f"\nLoaded: {selected_file}  "
          f"({format_duration(timing_log['sensitivity_file_loading_time_s'])})")
    print(f"  Layers: {num_layers}  |  "
          f"Sensitivity: [{sens_values.min():.4f}, {sens_values.max():.4f}]")
    if phase1_sensitivity_time:
        print(f"  Phase 1 sensitivity time: {format_duration(phase1_sensitivity_time)}")

    model_key = next(
        (k for k, c in TINYLLAMA_MODELS.items() if c["num_layers"] == num_layers),
        "TinyLlama-1.1B"
    )
    model_name = TINYLLAMA_MODELS[model_key]["model_name"]
    print(f"  Detected model: {model_key}")

    sf_lower = selected_file.lower()
    sensitivity_method = (
        "PMPQ"  if "pmpq"  in sf_lower else
        "PWCCA" if "pwcca" in sf_lower else
        "SVCCA" if "svcca" in sf_lower else
        "CKA"   if "cka"   in sf_lower else
        "CMPQ"
    )
    print(f"  Sensitivity method detected: {sensitivity_method}")

                                                                                
                        
                                                                                
    print_section("STEP 2: LOADING MODEL")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32,
        device_map=None, low_cpu_mem_usage=True
    )

                       
                                                                                           
                                                                                              
    model = model.to(device)
    timing_log["model_loading_time_s"] = time.time() - t0
    print(f"  Model loaded in {format_duration(timing_log['model_loading_time_s'])}")

                                                                                
                        
                                                                                
    print_section("STEP 3: CLUSTERING CONFIGURATION")
    t0 = time.time()

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

    if "K-means" in clustering_choice:
        labels, cluster_means = kmeans_clustering(sens_values, n_clusters=n_clusters)
        strategy_name = "kmeans"
    elif "Percentile" in clustering_choice:
        labels, cluster_means = percentile_clustering(sens_values, n_clusters=n_clusters)
        strategy_name = "percentile"
    else:
        labels, cluster_means = hierarchical_clustering(sens_values, n_clusters=n_clusters)
        strategy_name = "hierarchical"

    timing_log["clustering_time_s"] = time.time() - t0
    print(f"\n  Clustering ({strategy_name}) in "
          f"{format_duration(timing_log['clustering_time_s'])}")
    for cid, cmean in cluster_means:
        lids = [i for i in range(num_layers) if labels[i] == cid]
        print(f"  Cluster {cid}: {len(lids)} layers (mean sensitivity: {cmean:.4f})")

                                                                                
                                  
                                                                                
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
            cluster_bits, allocation_name = matched
        else:
            raw = input("Enter 3 values comma-separated (e.g. 16,8,4): ").strip()
            cluster_bits = [int(b.strip()) for b in raw.split(",")]
            allocation_name = "custom"
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
            cluster_bits, allocation_name = matched
        else:
            raw = input("Enter 4 values comma-separated (e.g. 16,8,8,4): ").strip()
            cluster_bits = [int(b.strip()) for b in raw.split(",")]
            allocation_name = "custom"

    gsc = prompt_user(
        f"Select quantization group size (Phase 1 used: {phase1_group_size}):",
        ["128 (standard, GPTQ default)", "64 (finer granularity)", "32 (finest)", "Custom"],
        default="128 (standard, GPTQ default)"
    )
    if "64" in gsc and "128" not in gsc:   group_size = 64
    elif "32" in gsc and "128" not in gsc: group_size = 32
    elif "Custom" in gsc: group_size = int(input("Enter group size: ").strip())
    else: group_size = 128

    layer_bits_map = {}
    for i in range(num_layers):
        rank = next(j for j, (cid, _) in enumerate(cluster_means) if cid == labels[i])
        layer_bits_map[i] = cluster_bits[rank]

    print(f"\n  Layer bit allocation ({sensitivity_method} sensitivity-guided):")
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
              f"(sensitivity: {sens_values[i]:.4f})")

                                                                                
                                                                    
                                                                                
    cached_baseline = load_baseline_cache()
    latest_log_file = find_latest_evaluation_log()

    if cached_baseline is not None:
        print(f"\n  Found cached baseline: {BASELINE_CACHE_FILE}")
        print(f"  Cached accuracy: {cached_baseline['accuracy']:.4f} "
              f"({cached_baseline['timestamp']})")

    if latest_log_file:
        print(f"\n  Found recent evaluation log: {latest_log_file}")
        log_baseline = extract_baseline_from_log(latest_log_file)
        if log_baseline:
            print(f"  Log accuracy: {log_baseline['accuracy']:.4f} "
                  f"({log_baseline['timestamp']})")

    print("\nBaseline options:")
    print("  1. Run new FP32 evaluation")
    print("  2. Use cached baseline (if available)")
    print("  3. Extract from recent log file (if available)")

    baseline_choice = input("Select option [1/2/3] (default=1): ").strip()
    if not baseline_choice or baseline_choice == "1":
        run_bl = True
        use_log = False
    elif baseline_choice == "2":
        run_bl = False
        use_log = False
    elif baseline_choice == "3":
        run_bl = False
        use_log = True
    else:
        run_bl = True
        use_log = False

    acc_before = correct_before = total_before = fp32_throughput = None
    baseline_source = "computed"

    if run_bl:
        print_section("STEP 5: EVALUATING FP32 BASELINE ON HELLASWAG")
        t0 = time.time()
        acc_before, correct_before, total_before, eval_time_before = evaluate_hellaswag(
            model, tokenizer, device, "FP32 Baseline",
            split="validation", max_length=MAX_LENGTH, batch_size=BATCH_SIZE
        )
        timing_log["fp32_evaluation_time_s"] = time.time() - t0
        fp32_throughput = total_before / timing_log["fp32_evaluation_time_s"]

        timestamp_bl = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_baseline_cache(
            acc_before, correct_before, total_before,
            timing_log["fp32_evaluation_time_s"], fp32_throughput,
            model_key, model_name, str(device), num_gpus, timestamp_bl
        )

        print(f"\n  FP32 Baseline:")
        print(f"  Accuracy  : {acc_before:.4f} ({acc_before*100:.2f}%)")
        print(f"  Correct   : {correct_before}/{total_before}")
        print(f"  Time      : {format_duration(timing_log['fp32_evaluation_time_s'])}")
        print(f"  Throughput: {fp32_throughput:.2f} examples/s")

    else:
        if use_log and latest_log_file and 'log_baseline' in locals() and log_baseline:
            print_section("STEP 5: LOADING FP32 BASELINE FROM LOG FILE")
            acc_before      = log_baseline["accuracy"]
            correct_before  = log_baseline["correct"]
            total_before    = log_baseline["total"]
            fp32_throughput = log_baseline["throughput"]
            timing_log["fp32_evaluation_time_s"] = log_baseline["eval_time_s"]
            baseline_source = f"log ({log_baseline['source_file']})"
            print(f"  Loaded from: {latest_log_file}")
            print(f"  Accuracy  : {acc_before:.4f} ({acc_before*100:.2f}%)")
            print(f"  Correct   : {correct_before}/{total_before}")
            print(f"  Time      : {format_duration(log_baseline['eval_time_s'])}")
            print(f"  Throughput: {fp32_throughput:.2f} samples/s")
            print(f"  Source    : {log_baseline['source_file']}")

                                          
            save_baseline_cache(
                acc_before, correct_before, total_before,
                log_baseline["eval_time_s"], fp32_throughput,
                model_key, model_name, str(device), num_gpus,
                log_baseline["timestamp"]
            )
        elif cached_baseline is not None:
            print_section("STEP 5: LOADING FP32 BASELINE FROM CACHE")
            acc_before      = cached_baseline["accuracy"]
            correct_before  = cached_baseline["correct"]
            total_before    = cached_baseline["total"]
            fp32_throughput = cached_baseline["throughput"]
            timing_log["fp32_evaluation_time_s"] = cached_baseline["eval_time_s"]
            baseline_source = f"cached ({cached_baseline['timestamp']})"
            print(f"  Loaded from: {BASELINE_CACHE_FILE}")
            print(f"  Accuracy  : {acc_before:.4f} ({acc_before*100:.2f}%)")
            print(f"  Correct   : {correct_before}/{total_before}")
            print(f"  Cached on : {cached_baseline['timestamp']}")
        else:
            print_section("STEP 5: NO BASELINE AVAILABLE")
            print("  No cached baseline or log file found.")
            print("  Please run with baseline evaluation first.")
            print("  Proceeding without baseline comparison.")
            timing_log["fp32_evaluation_time_s"] = 0.0

                                                                                
                                     
                                                                                
    print_section("STEP 6: APPLYING GROUP-WISE FAKE QUANTIZATION (Simulated)")
    print(f"  Mode: FAKE -- weights remain FP32 after simulate-quantize-dequantize")
    print(f"  Group size: {group_size}\n")

    tq0 = time.time()
    model, orig_bits, quant_bits = quantize_model_fake(
        model, layer_bits_map, group_size=group_size
    )
    quantize_time_s = time.time() - tq0
    timing_log["quantization_application_time_s"] = quantize_time_s

    compression_ratio  = orig_bits / quant_bits if quant_bits > 0 else float("inf")
    reduction_pct      = 100.0 * (1.0 - quant_bits / orig_bits) if orig_bits > 0 else 0.0
    fp32_mb            = bits_to_mb(orig_bits)
    quant_mb_simulated = bits_to_mb(quant_bits)

    print(f"\n  Fake quantization complete in {format_duration(quantize_time_s)}")
    print(f"  NOTE: Actual GPU memory is UNCHANGED (weights still FP32)")
    print(f"  FP32 weight size (bit-level count): {fp32_mb:.2f} MB")
    print(f"  Simulated quantized size           : {quant_mb_simulated:.2f} MB")
    print(f"  Simulated compression ratio        : {compression_ratio:.3f}x")
    print(f"  Simulated size reduction           : {reduction_pct:.2f}%")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

                                                                                
                                           
                                                                                
    print_section("STEP 7: EVALUATING FAKE-QUANTIZED MODEL ON HELLASWAG")
    print(f"  Batched evaluation (batch_size={BATCH_SIZE}) -- single GPU mode\n")

    t0 = time.time()
    acc_after, correct_after, total_after, eval_time_after = evaluate_hellaswag(
        model, tokenizer, device, "Fake-Quantized Model",
        split="validation", max_length=MAX_LENGTH, batch_size=BATCH_SIZE
    )
    timing_log["quantized_evaluation_time_s"] = time.time() - t0
    quant_throughput = total_after / timing_log["quantized_evaluation_time_s"]

    print(f"\n  Fake-Quantized Model:")
    print(f"  Accuracy  : {acc_after:.4f} ({acc_after*100:.2f}%)")
    print(f"  Correct   : {correct_after}/{total_after}")
    print(f"  Time      : {format_duration(timing_log['quantized_evaluation_time_s'])}")
    print(f"  Throughput: {quant_throughput:.2f} examples/s")

                                                                                
                                  
                                                                                
    print_section("STEP 8: PERFORMANCE COMPARISON")
    timing_log["total_pipeline_time_s"] = time.time() - pipeline_start_time

    if acc_before is not None:
        acc_drop     = acc_before - acc_after
        acc_drop_pct = (acc_drop / acc_before * 100.0) if acc_before > 0 else 0.0
        if acc_drop_pct <= 1.0:    quality = "EXCELLENT -- Minimal degradation"
        elif acc_drop_pct <= 3.0:  quality = "GOOD -- Acceptable degradation"
        elif acc_drop_pct <= 5.0:  quality = "MODERATE -- Noticeable degradation"
        elif acc_drop_pct <= 10.0: quality = "FAIR -- Significant degradation"
        else:                      quality = "POOR -- Severe degradation"
    else:
        acc_drop = acc_drop_pct = None
        quality = "N/A (no baseline available)"

    print(f"""
====================================================================
  RESULTS -- {sensitivity_method} FAKE QUANT GROUP-WISE -- HellaSwag
====================================================================
  Model         : {model_key}
  Quant type    : FAKE (simulated, weights remain FP32)
  Sensitivity   : {sensitivity_method}
  Clustering    : {strategy_name}   |   Bits: {cluster_bits}
  Group size    : {group_size}      |   GPUs: {num_gpus}
====================================================================
  BASELINE (FP32) [{baseline_source}]:
    Accuracy  : {f"{acc_before:.4f} ({acc_before*100:.2f}%)" if acc_before is not None else "N/A"}
    Correct   : {f"{correct_before}/{total_before}" if correct_before is not None else "N/A"}
====================================================================
  FAKE-QUANTIZED:
    Accuracy  : {acc_after:.4f} ({acc_after*100:.2f}%)
    Correct   : {correct_after}/{total_after}
====================================================================
  DEGRADATION:
    Acc drop  : {f"{acc_drop:+.4f} ({acc_drop_pct:+.2f}%)" if acc_drop is not None else "N/A"}
    Quality   : {quality}
====================================================================
  COMPRESSION (simulated -- bit-level math, actual memory unchanged):
    Ratio     : {compression_ratio:.3f}x
    Sizes     : {fp32_mb:.2f} MB -> {quant_mb_simulated:.2f} MB (simulated)
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

                                                                                
                                                                 
                                                                                
    print_section("STEP 9: SAVING FAKE-QUANTIZED MODEL")
    os.makedirs("Models", exist_ok=True)
    alloc_str      = "-".join(str(b) for b in cluster_bits)
    model_dir_name = (
        f"fake_quant_TinyLlama_HellaSwag_{alloc_str}"
        f"_gs{group_size}_{strategy_name}_{sensitivity_method}"
    )
    model_save_path = os.path.join("Models", model_dir_name)
    os.makedirs(model_save_path, exist_ok=True)

    actual_model = model

    hf_config  = AutoConfig.from_pretrained(model_name)
    save_model = AutoModelForCausalLM.from_config(hf_config)

    missing, _ = save_model.load_state_dict(
        actual_model.state_dict(), strict=False
    )
    if missing:
        print(f"  Note: {len(missing)} keys not copied "
              f"(custom buffer layers -- expected for fake-quant wrappers)")

    save_model.save_pretrained(
        model_save_path,
        max_shard_size="500MB",
        safe_serialization=False
    )
    tokenizer.save_pretrained(model_save_path)

    quant_config = {
        "quantization_type":  "FAKE_SIMULATED",
        "description":        "Weights are FP32. Quantization effect is simulated only.",
        "cluster_bits":       cluster_bits,
        "allocation_name":    allocation_name,
        "group_size":         group_size,
        "layer_bits_map":     {str(k): v for k, v in layer_bits_map.items()},
        "strategy_name":      strategy_name,
        "sensitivity_method": sensitivity_method,
        "model_key":          model_key,
        "model_name":         model_name,
        "acc_after":          float(acc_after) if acc_after is not None else None,
        "acc_before":         float(acc_before) if acc_before is not None else None,
        "baseline_source":    baseline_source,
        "timestamp":          datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(os.path.join(model_save_path, "quant_config.json"), "w") as qf:
        json.dump(quant_config, qf, indent=2)

    print(f"  Fake-quantized model saved to: {model_save_path}/")
    print(f"    config.json              -- model architecture & hyperparameters")
    print(f"    tokenizer.json           -- tokenizer files")
    print(f"    pytorch_model-*.bin      -- sharded FP32 weights (500MB per shard)")
    print(f"    pytorch_model.bin.index.json -- shard index")
    print(f"    quant_config.json        -- quantization metadata")

                                                                                
                               
                                                                                
    print_section("STEP 10: SAVING RESULTS LOG")
    os.makedirs("Evaluation", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_filename = (
        f"fake_quant_eval_TinyLlama_HellaSwag_{allocation_name}_"
        f"{sensitivity_method}_GroupWise_{timestamp}.txt"
    )
    log_path = os.path.join("Evaluation", log_filename)

    with open(log_path, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"FAKE QUANTIZATION EVALUATION RESULTS "
                f"({sensitivity_method} + GROUP-WISE)\n")
        f.write(f"TinyLlama on HellaSwag Validation Set\n")
        f.write("="*80 + "\n\n")

                       
        f.write("="*80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("="*80 + "\n")
        f.write(f"Quantization Type: FAKE (simulated -- weights remain FP32 throughout)\n")
        f.write(f"  Mechanism: FP32 -> quantize -> dequantize -> store back as FP32\n")
        f.write(f"  Purpose: Measure accuracy impact only. No real memory savings.\n")
        f.write(f"  Bit-widths accepted: any (2, 4, 6, 8, 12, 16, 32)\n")
        f.write(f"Model: {model_key}\n")
        f.write(f"Model HF Hub: {model_name}\n")
        f.write(f"Task: Commonsense Reasoning (HellaSwag -- 4-choice)\n")
        f.write(f"Dataset: HellaSwag\n")
        f.write(f"Evaluation Split: Validation set\n")
        f.write(f"Sensitivity Method: {sensitivity_method}\n")
        f.write(f"Sensitivity File: {selected_file}\n")
        f.write(f"Clustering Strategy: {strategy_name}\n")
        f.write(f"Number of Clusters: {n_clusters}\n")
        f.write(f"Bit Allocation: {cluster_bits}\n")
        f.write(f"Allocation Name: {allocation_name}\n")
        f.write(f"Group Size: {group_size}\n")
        f.write(f"Baseline Source: {baseline_source}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Num GPUs: {num_gpus}\n")
        if torch.cuda.is_available():
            for gi in range(num_gpus):
                f.write(f"  GPU {gi}: {torch.cuda.get_device_name(gi)}\n")
        f.write(f"Eval Batch Size: {BATCH_SIZE}\n")
        f.write(f"Max Sequence Length: {MAX_LENGTH}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Saved Model Folder: {model_save_path}\n\n")

                                  
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE TIMING LOG\n")
        f.write("="*80 + "\n")
        f.write(f"{'Step':<52} {'Time':<20} {'Seconds':<12}\n")
        f.write("-"*84 + "\n")
        for lbl, key in [
            ("Sensitivity File Loading",   "sensitivity_file_loading_time_s"),
            ("Model Loading",              "model_loading_time_s"),
            ("Clustering",                 "clustering_time_s"),
            (f"FP32 Baseline Eval [{baseline_source}]", "fp32_evaluation_time_s"),
            ("Fake Quantization Application",            "quantization_application_time_s"),
            ("Fake-Quantized Model Evaluation",          "quantized_evaluation_time_s"),
        ]:
            v = timing_log[key]
            f.write(f"{lbl:<52} {format_duration(v):<20} {v:<12.4f}\n")
        f.write("-"*84 + "\n")
        tv = timing_log["total_pipeline_time_s"]
        f.write(f"{'TOTAL PIPELINE TIME':<52} {format_duration(tv):<20} {tv:<12.4f}\n")
        if phase1_sensitivity_time:
            f.write(f"\nPhase 1 Sensitivity Computation (from metadata):\n")
            f.write(f"  {format_duration(phase1_sensitivity_time)}  "
                    f"({phase1_sensitivity_time:.4f}s)\n")
        f.write("\n")

                                                     
        f.write("="*80 + "\n")
        f.write("METRICS BEFORE QUANTIZATION (FP32 Baseline)\n")
        f.write("="*80 + "\n")
        if acc_before is not None:
            f.write(f"HellaSwag Accuracy: {acc_before:.6f} ({acc_before*100:.2f}%)\n")
            f.write(f"Correct: {correct_before}/{total_before}\n")
            f.write(f"Eval Time: {format_duration(timing_log['fp32_evaluation_time_s'])}\n")
            f.write(f"Throughput: {fp32_throughput:.2f} examples/s\n")
            f.write(f"Baseline Source: {baseline_source}\n")
            f.write(f"Baseline Cache File: {BASELINE_CACHE_FILE}\n\n")
        else:
            f.write("FP32 BASELINE -- NOT AVAILABLE (no cache, skip selected)\n\n")

                                    
        f.write("="*80 + "\n")
        f.write("METRICS AFTER QUANTIZATION (Group-Wise Fake Mixed-Precision PTQ)\n")
        f.write("="*80 + "\n")
        f.write(f"HellaSwag Accuracy: {acc_after:.6f} ({acc_after*100:.2f}%)\n")
        f.write(f"Correct: {correct_after}/{total_after}\n")
        f.write(f"Eval Time: {format_duration(timing_log['quantized_evaluation_time_s'])}\n")
        f.write(f"Throughput: {quant_throughput:.2f} examples/s\n\n")

                                
        f.write("="*80 + "\n")
        f.write("PERFORMANCE COMPARISON\n")
        f.write("="*80 + "\n")
        if acc_drop is not None:
            f.write(f"Accuracy Drop: {acc_drop:+.6f} ({acc_drop_pct:+.2f}%)\n")
            f.write(f"Quality Assessment: {quality}\n\n")
        else:
            f.write("Not available (no baseline)\n\n")

                             
        f.write("="*80 + "\n")
        f.write("COMPRESSION METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Quantization Type: FAKE (simulated -- no real memory savings)\n")
        f.write(f"  Compression ratio is computed from bit-level arithmetic only.\n")
        f.write(f"  Actual model size in memory is identical to FP32 baseline.\n")
        f.write(f"FP32 Weight Size (bit-level count): {fp32_mb:.2f} MB\n")
        f.write(f"Simulated Quantized Size           : {quant_mb_simulated:.2f} MB\n")
        f.write(f"  (includes simulated group-scale overhead)\n")
        f.write(f"Simulated Compression Ratio: {compression_ratio:.3f}x\n")
        f.write(f"Simulated Size Reduction   : {reduction_pct:.2f}%\n")
        f.write(f"Quantization Time          : {quantize_time_s:.4f}s\n\n")

                                              
        f.write("="*80 + "\n")
        f.write("MACHINE-READABLE METRICS (KEY-VALUE)\n")
        f.write("="*80 + "\n")
        for k, v in [
            ("quantization_type",   "FAKE_SIMULATED"),
            ("model_key",            model_key),
            ("model_name",           model_name),
            ("task",                 "HellaSwag"),
            ("eval_split",           "validation"),
            ("device",               str(device)),
            ("num_gpus",             num_gpus),
            ("eval_examples",        total_after),
            ("sensitivity_method",   sensitivity_method),
            ("clustering_strategy",  strategy_name),
            ("n_clusters",           n_clusters),
            ("allocation_name",      allocation_name),
            ("cluster_bits",         str(cluster_bits)),
            ("group_size",           group_size),
            ("baseline_source",      baseline_source),
        ]:
            f.write(f"{k}: {v}\n")
        if acc_before is not None:
            f.write(f"acc_before: {acc_before:.6f}\n")
            f.write(f"correct_before: {correct_before}\n")
            f.write(f"total_before: {total_before}\n")
            f.write(f"fp32_throughput: {fp32_throughput:.2f}\n")
        f.write(f"acc_after: {acc_after:.6f}\n")
        f.write(f"correct_after: {correct_after}\n")
        f.write(f"total_after: {total_after}\n")
        f.write(f"quant_throughput: {quant_throughput:.2f}\n")
        if acc_drop is not None:
            f.write(f"acc_drop: {acc_drop:.6f}\n")
            f.write(f"acc_drop_pct: {acc_drop_pct:.2f}\n")
        f.write(f"fp32_mb_bit_level: {fp32_mb:.2f}\n")
        f.write(f"quant_mb_simulated: {quant_mb_simulated:.2f}\n")
        f.write(f"compression_ratio: {compression_ratio:.3f}\n")
        f.write(f"reduction_pct: {reduction_pct:.2f}\n")
        f.write(f"saved_model_folder: {model_save_path}\n")
        f.write(f"\n# Timing (seconds):\n")
        for k, v in timing_log.items():
            f.write(f"{k}: {v:.4f}\n")
        if phase1_sensitivity_time:
            f.write(f"phase1_sensitivity_time_s: {phase1_sensitivity_time:.4f}\n")

                              
        f.write("\n" + "="*80 + "\n")
        f.write("LAYER BIT ALLOCATION\n")
        f.write("="*80 + "\n")
        for i in range(num_layers):
            b = layer_bits_map[i]
            f.write(f"  layer_{i:02d}: {b:2d}-bit (simulated)  "
                    f"sensitivity: {sens_values[i]:.6f}\n")

                      
        f.write("\n" + "="*80 + "\n")
        f.write("METHOD NOTES\n")
        f.write("="*80 + "\n")
        f.write(f"Sensitivity Analysis Method: {sensitivity_method}\n")
        if sensitivity_method == "PMPQ":
            f.write(f"  PMPQ (Pruning-based Mixed-Precision Quantization) computes\n")
            f.write(f"  layer sensitivity as the accuracy drop on HellaSwag TRAIN split\n")
            f.write(f"  when a fixed percentage of the smallest-magnitude weights in that\n")
            f.write(f"  layer are set to zero. All other layers remain unchanged.\n")
            f.write(f"  sensitivity[i] = baseline_accuracy - pruned_accuracy.\n")
            f.write(f"  Layers with larger accuracy drops are more sensitive and receive\n")
            f.write(f"  higher bit-widths in mixed-precision quantization.\n")
            f.write(f"  Calibration data: Phase 1 training split of HellaSwag.\n\n")
        else:
            f.write(f"  SVCCA measures representational similarity between layer activations\n")
            f.write(f"  of the original model vs a perturbed version. High SVCCA score means\n")
            f.write(f"  the layer is insensitive to quantization (can use aggressive bit-widths).\n")
            f.write(f"  Low score means the layer is sensitive (needs higher precision).\n")
            f.write(f"  Calibration data: Phase 1 training split of HellaSwag.\n\n")
        f.write(f"Quantization: FAKE (Simulated)\n")
        f.write(f"  Each weight: FP32 -> round to {cluster_bits} levels -> dequantize\n")
        f.write(f"  -> stored back as FP32. Captures rounding error numerically\n")
        f.write(f"  but uses no fewer bytes in memory. Standard approach used by\n")
        f.write(f"  OmniQuant, AffineQuant, and similar papers for measuring accuracy\n")
        f.write(f"  impact without hardware deployment.\n\n")
        f.write(f"Group-Wise Quantization:\n")
        f.write(f"  group_size={group_size}: every {group_size} consecutive weights\n")
        f.write(f"  share one scale factor. Same approach as GPTQ. Provides finer\n")
        f.write(f"  precision than a single global scale per layer.\n\n")
        f.write(f"Dataset: HellaSwag Validation Set\n")
        f.write(f"  4-choice commonsense reasoning benchmark.\n")
        f.write(f"  Evaluation: log-likelihood scoring of 4 candidate endings.\n")
        f.write(f"  Model picks ending with highest avg log-likelihood.\n")
        f.write(f"  Metric: Accuracy (fraction of correctly chosen endings).\n\n")
        f.write(f"Clustering: {strategy_name} with {n_clusters} groups\n")
        f.write(f"  Sensitivity scores clustered to assign bit-widths:\n")
        for rank, (cid, cmean) in enumerate(cluster_means):
            lids = [i for i in range(num_layers) if labels[i] == cid]
            f.write(f"  Cluster {cid} (rank {rank}, bits={cluster_bits[rank]}): "
                    f"{len(lids)} layers, mean sensitivity={cmean:.4f}\n")
        f.write(f"\nModel saved in HuggingFace .bin sharded format:\n")
        f.write(f"  Folder: {model_save_path}/\n")
        f.write(f"  config.json              -- model architecture & hyperparameters\n")
        f.write(f"  tokenizer.json           -- tokenizer vocabulary & config\n")
        f.write(f"  pytorch_model-*.bin      -- weight shards (500MB each, FP32)\n")
        f.write(f"  pytorch_model.bin.index.json -- maps weight names to shard files\n")
        f.write(f"  quant_config.json        -- quantization configuration\n")

    print(f"  Results log saved to: {log_path}")
    print(f"\nPhase 2 (FAKE) complete!")


if __name__ == "__main__":
    main()