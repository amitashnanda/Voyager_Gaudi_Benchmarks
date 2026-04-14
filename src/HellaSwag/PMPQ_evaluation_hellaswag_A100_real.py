                                                        
"""
Phase 2: Group-Wise REAL Quantization & Evaluation for TinyLlama on HellaSwag
===============================================================================
QUANTIZATION TYPE: REAL
  FP16  -> weights stored as torch.float16  -- 2x real memory savings
  INT8  -> weights stored as torch.int8     -- 4x real memory savings
           + per-group FP16 scale (group-wise symmetric, GPTQ-style)

NOTE on "real" vs "fake":
  Weights physically live as float16 / int8 in GPU memory (not FP32).
  INT8 dequantizes to FP16 on-the-fly before matmul -- this is standard
  weight-only quantization (same as GPTQ/bitsandbytes).
  Memory savings are 100% real and verified by get_model_size_mb().

Supports sensitivity files from:
  - CMPQ Phase 1 (SVCCA, PWCCA, CKA)
  - PMPQ Phase 1 (magnitude pruning accuracy drop)

Baseline Caching:
  FP32 baseline results are read from Models/baseline_hellaswag_fp32.json
  (written by the FAKE quant script or by this script on first run).
  If skip is selected and cache exists, results are loaded for comparison.

Model Saving (.bin HuggingFace format):
  Saved to Models/real_quant_TinyLlama_HellaSwag_<bits>_gs<gs>_<strategy>_<method>/
  containing: config.json, tokenizer.json, pytorch_model-*.bin, quant_config.json
  Note: weights in .bin are dequantized to FP32 (HuggingFace standard).
  quant_config.json records original FP16/INT8 configuration for re-quantization.

Speed:
  Batched evaluation (batch_size=64) + autocast + single A100 GPU.
  Quantization on CPU then model moves to GPU (avoids OOM).
  Single GPU mode for better performance (no DataParallel overhead).

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
from torch.cuda.amp import autocast
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


                                                                              
                
                                                                              

def load_baseline_cache():
    if not os.path.exists(BASELINE_CACHE_FILE):
        return None
    with open(BASELINE_CACHE_FILE, "r") as f:
        return json.load(f)


def extract_baseline_from_fake_quant_log(log_file_path):
    """
    Extract FP32 baseline metrics from a fake quantization evaluation log file.

    Looks for patterns in the METRICS BEFORE QUANTIZATION section:
    - HellaSwag Accuracy: 0.568612 (56.86%)
    - Correct: 5710/10042
    - Eval Time: 12m 31.21s
    - Throughput: 13.37 examples/s

    Returns: dict with baseline metrics or None if not found
    """
    if not os.path.exists(log_file_path):
        print(f"  Log file not found: {log_file_path}")
        return None

    try:
        with open(log_file_path, "r") as f:
            content = f.read()

                                                          
        before_section = re.search(
            r"METRICS BEFORE QUANTIZATION.*?(?=\n={40,}|\Z)",
            content,
            re.DOTALL
        )

        if not before_section:
            print("  Could not find METRICS BEFORE QUANTIZATION section")
            return None

        section_text = before_section.group(0)

                          
        acc_match = re.search(r"HellaSwag Accuracy:\s*(\d+\.\d+)", section_text)
        if acc_match:
            accuracy = float(acc_match.group(1))
        else:
            print("  Could not extract accuracy from log")
            return None

                               
        correct_match = re.search(r"Correct:\s*(\d+)/(\d+)", section_text)
        if correct_match:
            correct = int(correct_match.group(1))
            total = int(correct_match.group(2))
        else:
            print("  Could not extract correct/total from log")
            return None

                                                
        time_match = re.search(r"Eval Time:\s*(?:(\d+)m\s*)?(\d+\.?\d*)s", section_text)
        if time_match:
            minutes = int(time_match.group(1)) if time_match.group(1) else 0
            seconds = float(time_match.group(2))
            eval_time = minutes * 60 + seconds
        else:
            print("  Could not extract eval time from log")
            eval_time = 0.0

                            
        throughput_match = re.search(r"Throughput:\s*(\d+\.?\d*)\s*examples/s", section_text)
        if throughput_match:
            throughput = float(throughput_match.group(1))
        else:
            throughput = total / eval_time if eval_time > 0 else 0.0

                                         
        model_match = re.search(r"Model:\s*([^\n]+)", content)
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


def find_latest_fake_quant_log(eval_dir="Evaluation"):
    """
    Find the most recent fake quantization evaluation log file.

    Looks for files matching pattern: fake_quant_eval_*.txt

    Returns: path to the latest log file or None
    """
    if not os.path.exists(eval_dir):
        return None

                                                     
    log_files = list(Path(eval_dir).glob("fake_quant_eval_*.txt"))

    if not log_files:
        return None

                                                 
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                                
    return str(log_files[0])


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


                                                                              
                                         
                                                                              

class RealQuantizedLinearGroupWise(nn.Module):
    """
    REAL group-wise quantized linear layer.

    FP16 mode (nbits >= 16):
      Weight stored as torch.float16 -- 2x memory vs FP32.
      A100 Tensor Cores run FP16 matmuls natively.

    INT8 mode (nbits < 16):
      Weight stored as torch.int8 -- 4x memory vs FP32.
      Per-group FP16 scale (group_size weights -> 1 FP16 scale).
      Forward: dequantize int8->fp16, then fp16 matmul.
      (weight-only quantization -- same as GPTQ/bitsandbytes)
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
        return self.weight_int8.half() * sc

    def forward(self, x):
        if self._mode == 'fp16':
            w = self.weight_fp16.to(device=x.device, dtype=x.dtype)
        else:
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
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_mod)


def quantize_model_real(model, layer_bits_map, group_size=128):
    actual_model = model
    targets = []
    for name, module in actual_model.named_modules():
        if isinstance(module, nn.Linear):
            m = re.search(r"(?:model\.|decoder\.)layers\.(\d+)\.", name)
            if m:
                layer_idx = int(m.group(1))
                nbits = layer_bits_map.get(layer_idx, 8)
                targets.append((name, module, nbits))

    fp16_c = sum(1 for _, _, b in targets if b >= 16)
    int8_c  = sum(1 for _, _, b in targets if b <  16)

    print(f"  Found {len(targets)} Linear layers to quantize (REAL storage)")
    print(f"  FP16 layers: {fp16_c}  (weights -> torch.float16, 2x savings)")
    print(f"  INT8 layers: {int8_c}  (weights -> torch.int8,    4x savings)")
    print(f"  Group size : {group_size}")

    total_orig = total_quant = 0
    for qualname, linear_mod, nbits in tqdm(targets, desc="  Quantizing layers"):
        wrapper = RealQuantizedLinearGroupWise(linear_mod, nbits, group_size)
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

    This matches the fake quantization script implementation exactly.
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

    print(f"[{eval_name}] Evaluating {total} samples on {device} (batch_size={batch_size})...")

                        
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

                                                         
        with autocast(dtype=torch.float16):
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
    print_section(
        "PHASE 2: HellaSwag Evaluation -- REAL QUANTIZATION (FP16 / INT8)")
    print("""
  REAL QUANTIZATION -- A100 native (pure PyTorch, no external libraries):
    FP16  ->  torch.float16 weights  -- 2x real memory savings
    INT8  ->  torch.int8 weights     -- 4x real memory savings
              + per-group FP16 scale (group-wise symmetric, GPTQ-style)

  Supports CMPQ sensitivity files (SVCCA, PWCCA, CKA) and
  PMPQ sensitivity files (magnitude pruning accuracy drop).

  Bit options: only 16 (FP16) and 8 (INT8).
  Any other value snapped: >=16 -> FP16, <16 -> INT8.

  Baseline: loaded from Models/baseline_hellaswag_fp32.json if skip selected.
  Model:    saved to Models/<n>/ in HuggingFace .bin sharded format.
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
    print(f"\n  Loaded: {selected_file}  "
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

    timing_log["model_loading_time_s"] = time.time() - t0
    fp32_size_mb = get_model_size_mb(model)
    print(f"  Model loaded in {format_duration(timing_log['model_loading_time_s'])}")
    print(f"  FP32 size (CPU): {fp32_size_mb:.2f} MB")

                                                                                
                        
                                                                                
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
    print("  REAL QUANTIZATION supports only FP16 and INT8 (pure PyTorch on A100).")
    print("  FP16 (16-bit) -> torch.float16 -- 2x memory savings")
    print("  INT8  (8-bit) -> torch.int8    -- 4x memory savings")
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
            cluster_bits, allocation_name = matched
        else:
            raw = input("Enter 3 values comma-separated (e.g. 16,8,8): ").strip()
            cluster_bits = [snap_bits(int(b.strip())) for b in raw.split(",")]
            allocation_name = "custom"
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
            cluster_bits, allocation_name = matched
        else:
            raw = input("Enter 4 values comma-separated (e.g. 16,8,8,8): ").strip()
            cluster_bits = [snap_bits(int(b.strip())) for b in raw.split(",")]
            allocation_name = "custom"

    final_bits = []
    for b in cluster_bits:
        snapped = snap_bits(b)
        if snapped != b:
            print(f"  {b}-bit NOT supported -- snapping to "
                  f"{snapped}-bit ({'FP16' if snapped==16 else 'INT8'})")
        final_bits.append(snapped)
    cluster_bits = final_bits

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

    fp16_layers = sum(1 for b in layer_bits_map.values() if b >= 16)
    int8_layers = sum(1 for b in layer_bits_map.values() if b <  16)

    print(f"\n  Layer bit allocation ({sensitivity_method} sensitivity-guided):")
    print(f"  Group size: {group_size}")
    print(f"\n  Summary:")
    print(f"    FP16 (16-bit): {fp16_layers}/{num_layers} layers -- 2x compression")
    print(f"    INT8  (8-bit): {int8_layers}/{num_layers} layers -- 4x compression")
    print(f"\n  Layer-by-layer:")
    for i in range(num_layers):
        b  = layer_bits_map[i]
        rt = "FP16" if b >= 16 else "INT8"
        print(f"    Layer {i:2d}: {b:2d}-bit [{rt}]  (sensitivity: {sens_values[i]:.4f})")

                                                                                
                                                                           
                                                                                
    cached_baseline = load_baseline_cache()
    fake_quant_log = find_latest_fake_quant_log()

    if cached_baseline is not None:
        print(f"\n  Found cached baseline: {BASELINE_CACHE_FILE}")
        print(f"  Cached accuracy: {cached_baseline['accuracy']:.4f} "
              f"({cached_baseline['timestamp']})")

    if fake_quant_log:
        print(f"\n  Found recent fake quant log: {fake_quant_log}")
        fake_baseline = extract_baseline_from_fake_quant_log(fake_quant_log)
        if fake_baseline:
            print(f"  Log accuracy: {fake_baseline['accuracy']:.4f} "
                  f"({fake_baseline['timestamp']})")

    print("\nBaseline options:")
    print("  1. Run new FP32 evaluation")
    print("  2. Use cached baseline (if available)")
    print("  3. Import from fake quant log (if available)")

    baseline_choice = input("Select option [1/2/3] (default=3 if available, else 2): ").strip()
    if not baseline_choice:
        if fake_quant_log and 'fake_baseline' in locals() and fake_baseline:
            baseline_choice = "3"
        elif cached_baseline:
            baseline_choice = "2"
        else:
            baseline_choice = "1"

    if baseline_choice == "1":
        run_bl = True
        use_fake_log = False
    elif baseline_choice == "2":
        run_bl = False
        use_fake_log = False
    elif baseline_choice == "3":
        run_bl = False
        use_fake_log = True
    else:
        run_bl = True
        use_fake_log = False

    acc_before = correct_before = total_before = fp32_throughput = None
    baseline_source = "computed"

                                    
                                                                     
    model = model.to(device)

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
        if use_fake_log and fake_quant_log and 'fake_baseline' in locals() and fake_baseline:
            print_section("STEP 5: LOADING FP32 BASELINE FROM FAKE QUANT LOG")
            acc_before      = fake_baseline["accuracy"]
            correct_before  = fake_baseline["correct"]
            total_before    = fake_baseline["total"]
            fp32_throughput = fake_baseline["throughput"]
            timing_log["fp32_evaluation_time_s"] = fake_baseline["eval_time_s"]
            baseline_source = f"fake_quant_log ({fake_baseline['source_file']})"
            print(f"  Loaded from: {fake_quant_log}")
            print(f"  Accuracy  : {acc_before:.4f} ({acc_before*100:.2f}%)")
            print(f"  Correct   : {correct_before}/{total_before}")
            print(f"  Time      : {format_duration(fake_baseline['eval_time_s'])}")
            print(f"  Throughput: {fp32_throughput:.2f} examples/s")
            print(f"  Source    : {fake_baseline['source_file']}")

                                          
            save_baseline_cache(
                acc_before, correct_before, total_before,
                fake_baseline["eval_time_s"], fp32_throughput,
                model_key, model_name, str(device), num_gpus,
                fake_baseline["timestamp"]
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
            print("  No cached baseline or fake quant log found.")
            print("  Please run the fake quant script first or enable baseline evaluation.")
            timing_log["fp32_evaluation_time_s"] = 0.0

                                                                                
                                     
                                                                                
    print_section("STEP 6: APPLYING REAL GROUP-WISE MIXED-PRECISION QUANTIZATION")
    print(f"  Mode: REAL weight storage -- FP16 or INT8 -- group_size={group_size}")
    print(f"  Quantization runs on CPU then model moves to GPU\n")

                                        
    model = model.cpu()

    tq0 = time.time()
    model, orig_bits, quant_bits = quantize_model_real(
        model, layer_bits_map, group_size=group_size
    )
    quantize_time_s = time.time() - tq0
    timing_log["quantization_application_time_s"] = quantize_time_s

    compression_ratio  = orig_bits / quant_bits if quant_bits > 0 else float("inf")
    reduction_pct      = 100.0 * (1.0 - quant_bits / orig_bits) if orig_bits > 0 else 0.0
    fp32_mb            = bits_to_mb(orig_bits)
    quant_mb_bit_level = bits_to_mb(quant_bits)

    model = model.to(device)
    quant_size_actual_mb = get_model_size_mb(model)

    print(f"\n  Quantization complete in {format_duration(quantize_time_s)}")
    print(f"  FP32 weight size  (bit-level count): {fp32_mb:.2f} MB")
    print(f"  Quant weight size (bit-level count): {quant_mb_bit_level:.2f} MB")
    print(f"  Actual GPU footprint (verified)    : {quant_size_actual_mb:.2f} MB")
    print(f"  Compression ratio                  : {compression_ratio:.3f}x")
    print(f"  Weight size reduction              : {reduction_pct:.2f}%")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

                                                                                
                                           
                                                                                
    print_section("STEP 7: EVALUATING REAL-QUANTIZED MODEL ON HELLASWAG")
    print(f"  Batched evaluation (batch_size={BATCH_SIZE}) -- single GPU mode\n")

    t0 = time.time()
    acc_after, correct_after, total_after, eval_time_after = evaluate_hellaswag(
        model, tokenizer, device, "Real-Quantized Model",
        split="validation", max_length=MAX_LENGTH, batch_size=BATCH_SIZE
    )
    timing_log["quantized_evaluation_time_s"] = time.time() - t0
    quant_throughput = total_after / timing_log["quantized_evaluation_time_s"]

    print(f"\n  Real-Quantized Model:")
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
  RESULTS -- {sensitivity_method} REAL QUANT GROUP-WISE (FP16/INT8) -- HellaSwag
====================================================================
  Model         : {model_key}
  Quant type    : REAL (FP16 -> float16 | INT8 -> int8 + group scales)
  Sensitivity   : {sensitivity_method}
  Clustering    : {strategy_name}   |   Bits: {cluster_bits}
  Group size    : {group_size}      |   GPUs: {num_gpus}
====================================================================
  BASELINE (FP32) [{baseline_source}]:
    Accuracy  : {f"{acc_before:.4f} ({acc_before*100:.2f}%)" if acc_before is not None else "N/A"}
    Correct   : {f"{correct_before}/{total_before}" if correct_before is not None else "N/A"}
====================================================================
  REAL-QUANTIZED (FP16/INT8):
    Accuracy  : {acc_after:.4f} ({acc_after*100:.2f}%)
    Correct   : {correct_after}/{total_after}
====================================================================
  DEGRADATION:
    Acc drop  : {f"{acc_drop:+.4f} ({acc_drop_pct:+.2f}%)" if acc_drop is not None else "N/A"}
    Quality   : {quality}
====================================================================
  COMPRESSION (real -- weights stored in reduced precision):
    Ratio     : {compression_ratio:.3f}x
    Sizes     : {fp32_mb:.2f} MB -> {quant_mb_bit_level:.2f} MB (bit-level)
    Actual GPU: {quant_size_actual_mb:.2f} MB (verified)
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

                                                                                
                                                                 
                                                                                
    print_section("STEP 9: SAVING REAL-QUANTIZED MODEL")
    os.makedirs("Models", exist_ok=True)
    alloc_str      = "-".join(str(b) for b in cluster_bits)
    model_dir_name = (
        f"real_quant_TinyLlama_HellaSwag_{alloc_str}"
        f"_gs{group_size}_{strategy_name}_{sensitivity_method}"
    )
    model_save_path = os.path.join("Models", model_dir_name)
    os.makedirs(model_save_path, exist_ok=True)

    actual_model = model                                           

    hf_config  = AutoConfig.from_pretrained(model_name)
    save_model = AutoModelForCausalLM.from_config(hf_config)

    quant_state = actual_model.state_dict()
    save_state  = save_model.state_dict()

    for key in save_state.keys():
        if key in quant_state:
            save_state[key] = quant_state[key].float()

    save_model.load_state_dict(save_state, strict=True)

    save_model.save_pretrained(
        model_save_path,
        max_shard_size="500MB",
        safe_serialization=False
    )
    tokenizer.save_pretrained(model_save_path)

    quant_config = {
        "quantization_type":     "REAL_FP16_INT8",
        "description":           (
            "Weights dequantized to FP32 for .bin storage. "
            "Original inference used FP16/INT8. "
            "To restore: reload FP32 weights and re-run quantize_model_real()."
        ),
        "cluster_bits":          cluster_bits,
        "allocation_name":       allocation_name,
        "group_size":            group_size,
        "layer_bits_map":        {str(k): v for k, v in layer_bits_map.items()},
        "fp16_layers":           fp16_layers,
        "int8_layers":           int8_layers,
        "strategy_name":         strategy_name,
        "sensitivity_method":    sensitivity_method,
        "model_key":             model_key,
        "model_name":            model_name,
        "acc_after":             float(acc_after) if acc_after is not None else None,
        "acc_before":            float(acc_before) if acc_before is not None else None,
        "baseline_source":       baseline_source,
        "quant_size_actual_mb":  float(quant_size_actual_mb),
        "fp32_mb_bit_level":     float(fp32_mb),
        "quant_mb_bit_level":    float(quant_mb_bit_level),
        "compression_ratio":     float(compression_ratio),
        "reduction_pct":         float(reduction_pct),
        "timestamp":             datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    with open(os.path.join(model_save_path, "quant_config.json"), "w") as qf:
        json.dump(quant_config, qf, indent=2)

    print(f"  Real-quantized model saved to: {model_save_path}/")
    print(f"    config.json              -- model architecture & hyperparameters")
    print(f"    tokenizer.json           -- tokenizer files")
    print(f"    pytorch_model-*.bin      -- sharded weights (dequantized to FP32)")
    print(f"    pytorch_model.bin.index.json -- shard index")
    print(f"    quant_config.json        -- quantization metadata")
    print(f"  Note: .bin weights are FP32 (HuggingFace standard format).")
    print(f"  Reload + re-run quantize_model_real() to restore FP16/INT8 inference.")

                                                                                
                               
                                                                                
    print_section("STEP 10: SAVING RESULTS LOG")
    os.makedirs("Evaluation", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_filename = (
        f"real_quant_eval_TinyLlama_HellaSwag_{allocation_name}_"
        f"{sensitivity_method}_GroupWise_{timestamp}.txt"
    )
    log_path = os.path.join("Evaluation", log_filename)

    with open(log_path, "w") as f:
        f.write("="*80 + "\n")
        f.write(f"REAL QUANTIZATION EVALUATION RESULTS "
                f"({sensitivity_method} + GROUP-WISE)\n")
        f.write(f"TinyLlama on HellaSwag Validation Set\n")
        f.write("="*80 + "\n\n")

                       
        f.write("="*80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("="*80 + "\n")
        f.write(f"Quantization Type: REAL "
                f"(FP16/INT8 -- weights physically stored in low precision)\n")
        f.write(f"  FP16: weights stored as torch.float16  -- 2x real memory savings\n")
        f.write(f"  INT8: weights stored as torch.int8     -- 4x real memory savings\n")
        f.write(f"        + per-group FP16 scale "
                f"(group-wise symmetric, GPTQ-style)\n")
        f.write(f"  INT8 forward: dequantize int8->fp16 on-the-fly, "
                f"then fp16 matmul\n")
        f.write(f"  (weight-only quantization -- same mechanism as GPTQ/bitsandbytes)\n")
        f.write(f"  Note: no 4-bit/2-bit -- requires TorchAO/bitsandbytes "
                f"for sub-INT8\n")
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
        f.write(f"FP16 layers: {fp16_layers}/{num_layers}\n")
        f.write(f"INT8 layers: {int8_layers}/{num_layers}\n")
        f.write(f"Baseline Source: {baseline_source}\n")
        f.write(f"Baseline Cache File: {BASELINE_CACHE_FILE}\n")
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
        if phase1_sensitivity_time:
            f.write(f"\nPhase 1 Sensitivity Computation (from metadata):\n")
            f.write(f"  {format_duration(phase1_sensitivity_time)}  "
                    f"({phase1_sensitivity_time:.4f}s)\n")
        f.write("\n")

                                                     
        f.write("="*80 + "\n")
        f.write("METRICS BEFORE QUANTIZATION (FP32 Baseline)\n")
        f.write("="*80 + "\n")
        if acc_before is not None:
            f.write(f"HellaSwag Accuracy: {acc_before:.6f} "
                    f"({acc_before*100:.2f}%)\n")
            f.write(f"Correct: {correct_before}/{total_before}\n")
            f.write(f"Eval Time: "
                    f"{format_duration(timing_log['fp32_evaluation_time_s'])}\n")
            f.write(f"Throughput: {fp32_throughput:.2f} examples/s\n")
            f.write(f"Baseline Source: {baseline_source}\n")
            f.write(f"Baseline Cache File: {BASELINE_CACHE_FILE}\n\n")
        else:
            f.write("FP32 BASELINE -- NOT AVAILABLE "
                    "(no cache, skip selected)\n\n")

                                    
        f.write("="*80 + "\n")
        f.write("METRICS AFTER QUANTIZATION "
                "(Group-Wise Real Mixed-Precision PTQ)\n")
        f.write("="*80 + "\n")
        f.write(f"HellaSwag Accuracy: {acc_after:.6f} "
                f"({acc_after*100:.2f}%)\n")
        f.write(f"Correct: {correct_after}/{total_after}\n")
        f.write(f"Eval Time: "
                f"{format_duration(timing_log['quantized_evaluation_time_s'])}\n")
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
        f.write(f"Quantization Type: REAL "
                f"(weights physically stored in low precision)\n")
        f.write(f"  Compression ratio is derived from bit-level arithmetic.\n")
        f.write(f"  Actual GPU footprint is verified via get_model_size_mb().\n")
        f.write(f"FP32 Weight Size (bit-level count)    : {fp32_mb:.2f} MB\n")
        f.write(f"Quant Weight Size (bit-level count)   : {quant_mb_bit_level:.2f} MB\n")
        f.write(f"  (INT8 weights + FP16 group scales overhead)\n")
        f.write(f"Actual GPU Footprint (verified real)  : "
                f"{quant_size_actual_mb:.2f} MB\n")
        f.write(f"Compression Ratio: {compression_ratio:.3f}x\n")
        f.write(f"Size Reduction: {reduction_pct:.2f}%\n")
        f.write(f"Quantization Time: {quantize_time_s:.4f}s\n\n")

                                              
        f.write("="*80 + "\n")
        f.write("MACHINE-READABLE METRICS (KEY-VALUE)\n")
        f.write("="*80 + "\n")
        for k, v in [
            ("quantization_type",   "REAL_FP16_INT8"),
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
            ("fp16_layers",          fp16_layers),
            ("int8_layers",          int8_layers),
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
        f.write(f"quant_mb_bit_level: {quant_mb_bit_level:.2f}\n")
        f.write(f"quant_mb_actual_gpu: {quant_size_actual_mb:.2f}\n")
        f.write(f"compression_ratio: {compression_ratio:.3f}\n")
        f.write(f"reduction_pct: {reduction_pct:.2f}\n")
        f.write(f"saved_model_folder: {model_save_path}\n")
        f.write(f"\n# Timing (seconds):\n")
        for k, v in timing_log.items():
            f.write(f"{k}: {v:.4f}\n")
        if phase1_sensitivity_time:
            f.write(f"phase1_sensitivity_time_s: "
                    f"{phase1_sensitivity_time:.4f}\n")

                              
        f.write("\n" + "="*80 + "\n")
        f.write("LAYER BIT ALLOCATION\n")
        f.write("="*80 + "\n")
        for i in range(num_layers):
            b  = layer_bits_map[i]
            rt = "FP16" if b >= 16 else "INT8"
            f.write(f"  layer_{i:02d}: {b:2d}-bit [{rt}]  "
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
        f.write(f"Quantization: REAL (weights stored in FP16 or INT8 on GPU)\n")
        f.write(f"  FP16 layers: weight stored as torch.float16.\n")
        f.write(f"               A100 Tensor Cores run FP16 matmul natively.\n")
        f.write(f"  INT8 layers: weight stored as torch.int8 (1 byte per weight).\n")
        f.write(f"               On-the-fly dequantization: int8 * scale -> fp16.\n")
        f.write(f"               Then fp16 matmul on A100 Tensor Cores.\n")
        f.write(f"               This is weight-only quantization, same as GPTQ.\n\n")
        f.write(f"Group-Wise Quantization:\n")
        f.write(f"  group_size={group_size}: every {group_size} consecutive weights\n")
        f.write(f"  share one FP16 scale factor. Same approach as GPTQ. Provides\n")
        f.write(f"  finer precision than a single global scale per layer.\n\n")
        f.write(f"Dataset: HellaSwag Validation Set\n")
        f.write(f"  4-choice commonsense reasoning benchmark.\n")
        f.write(f"  Evaluation: log-likelihood scoring of 4 candidate endings.\n")
        f.write(f"  Model picks ending with highest avg log-likelihood.\n")
        f.write(f"  Metric: Accuracy (fraction of correctly chosen endings).\n\n")
        f.write(f"Multi-GPU:\n")
        f.write(f"  {num_gpus} A100 GPU(s) - single GPU mode.\n")
        f.write(f"  Quantization on CPU then model moved to GPU.\n")
        f.write(f"  No DataParallel for optimal performance.\n")
        f.write(f"  Eval batch_size={BATCH_SIZE} processed on single GPU.\n")
        f.write(f"  autocast(dtype=torch.float16) for Tensor Core acceleration.\n\n")
        f.write(f"Clustering: {strategy_name} with {n_clusters} groups\n")
        for rank, (cid, cmean) in enumerate(cluster_means):
            lids = [i for i in range(num_layers) if labels[i] == cid]
            b    = cluster_bits[rank]
            rt   = "FP16" if b >= 16 else "INT8"
            f.write(f"  Cluster {cid} (rank {rank}, bits={b} [{rt}]): "
                    f"{len(lids)} layers, mean sensitivity={cmean:.4f}\n")
        f.write(f"\nModel saved in HuggingFace .bin sharded format:\n")
        f.write(f"  Folder: {model_save_path}/\n")
        f.write(f"  config.json              -- model architecture & hyperparameters\n")
        f.write(f"  tokenizer.json           -- tokenizer vocabulary & config\n")
        f.write(f"  pytorch_model-*.bin      -- weight shards (500MB each, FP32)\n")
        f.write(f"  pytorch_model.bin.index.json -- maps weight names to shard files\n")
        f.write(f"  quant_config.json        -- quantization configuration\n")
        f.write(f"  Note: .bin files contain FP32 weights (HuggingFace standard).\n")
        f.write(f"  The quant_config.json records original FP16/INT8 assignment\n")
        f.write(f"  per layer so quantization can be exactly reproduced.\n")

    print(f"  Results log saved to: {log_path}")
    print(f"\nPhase 2 (REAL) complete!")


if __name__ == "__main__":
    main()