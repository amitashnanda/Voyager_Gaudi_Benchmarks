# Perfomance optimization and Comparative Analysis of Generative AI models on Advanced AI Accelerators

## Overview

This repository implements a comprehensive Perfomance optimization and Comparative Analysis of Generative AI models on Advanced AI Accelerators (Intel Gaudi1, Gaudi2, Gaudi3, Nvidia V100, Nvidia A100, Nvidia H100).In this section we provide the tools to reproduce the results shown on the PEARC26 paper.

### Mixed-Precision Quantization
- **Multi-Platform Support**: Optimized implementations for Gaudi 2 (FP8/BF16) and A100 (INT8/FP16)
- **Three Evaluation Tasks**: WikiText-2 (perplexity), HellaSwag (reasoning), BoolQ (QA)
- **Two-Phase Approach**: Sensitivity analysis → Mixed-precision quantization
- **Real & Fake Quantization**: Test with simulated (fake) or actual memory-reduced (real) quantization
- **GroupWise Quantization**: Fine-grained 128-weight groups for better accuracy preservation
- **Submodule-Level Analysis**: Per-component sensitivity (attention vs MLP layers)

### LLM Fine-Tuning
- **Fine-Tuning LLMs**: Fine-tuning runtime of the Llama3.1-8B-Instruct model across three generations of Gaudi nodes, with varying numbers of HPU devices and using Flash Attention.Also fine-tuning of a medium and a large model on the Gaudi2 node for Gemma2:27B-Instruct and Llama3.3:70B-Instruct using sharding mechanisms like ZeRO3 implementation of DeepSpeed, FSDP and DDP. 

### Diffusion Model
- **Diffusion Model**: Scaling of a Diffusion model on different accelerators.

## Repository Structure

```
Voyager_Gaudi_Benchmarks/
│
├── src/                          # Source code for quantization
│   ├── WikiText/                 # WikiText-2 evaluation scripts
│   ├── HellaSwag/                # HellaSwag evaluation scripts
│   ├── BoolQ/                    # BoolQ evaluation scripts
│   ├── plot_code/plot_gaudi2_eval_clusters.py    # Visualization for cluster analysis
│   ├── plot_code/plot_submodule_sensitivity.py   # Submodule sensitivity plots
│   └── gaudi_spawn.py                            # gaudi spawn script
│
├── Sensitivities/                # Sensitivity analysis results
│   ├── A100/                     # A100 sensitivity scores
│   │   ├── WikiText/
│   │   ├── HellaSwag/
│   │   └── BoolQ/
│   └── Gaudi2/                   # Gaudi sensitivity scores
│       ├── WikiText/
│       ├── HellaSwag/
│       └── BoolQ/
│
├── Evaluation/                   # Evaluation results
│   ├── A100/                     # A100 quantization results
│   └── Gaudi2/                   # Gaudi quantization results
│
├── Models/                       # Saved quantized models
│   ├── baseline_*.json           # Cached FP32 baselines
│   └── real_quant_*/             # Quantized model checkpoints
│
├── plots/                        # Generated visualizations
├── logs/                         # Execution logs
├── launch/                       # Kubernetes deployment configs
│   └── run_quantization.yaml     # Gaudi pod configuration
│
├── scaling_diffusion/            # Diffusion model experiments
├── fine-tuning_llm/              # LLM fine-tuning experiments
└── requirements.txt              # environment setup
```

## Installation for Mixed-Precision Quantization

### Prerequisites
- **For Gaudi**: Intel Gaudi 2 HPU, Habana SynapseAI 1.21+
- **For A100**: NVIDIA A100 GPU, CUDA 11.8+
- Python 3.10+

### Environment Setup

#### For A100 (Local/Cluster)
```bash
# Clone repository
git clone https://github.com/amitashnanda/Voyager_Gaudi_Benchmarks.git
cd Voyager_Gaudi_Benchmarks

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### For Gaudi (Kubernetes)
```bash
# Deploy the Gaudi pod
kubectl apply -f launch/run_quantization.yaml

# Access the pod
kubectl exec -it gaudi-quant-pod -n user-name -- /bin/bash

# Inside pod - packages auto-install on first run
cd /voyager/ceph/users/user-name/Quantization
```

## Quick Start

### Phase 1: Sensitivity Analysis

#### A100 Execution
```bash
# WikiText-2 sensitivity
python src/WikiText/PMPQ_sensitivity_wikitext_A100.py --sparsity 0.3

# HellaSwag sensitivity
python src/HellaSwag/PMPQ_sensitivity_hellaswag_A100.py --sparsity 0.3

# BoolQ sensitivity
python src/BoolQ/PMPQ_sensitivity_boolq_A100.py --sparsity 0.3
```

#### Gaudi Execution (Single Card)
```bash
# WikiText-2 sensitivity
PT_HPU_LAZY_MODE=1 python src/WikiText/PMPQ_sensitivity_wikitext_hpu.py --sparsity 0.3

# HellaSwag sensitivity
PT_HPU_LAZY_MODE=1 python src/HellaSwag/PMPQ_sensitivity_hellaswag_hpu.py --sparsity 0.3

# BoolQ sensitivity
PT_HPU_LAZY_MODE=1 python src/BoolQ/PMPQ_sensitivity_boolq_hpu.py --sparsity 0.3
```

#### Gaudi Execution (Multi-Card)
```bash
# Using 2 Gaudi cards with MPI
PT_HPU_LAZY_MODE=1 python gaudi_spawn.py --world_size 2 --use_mpi src/WikiText/PMPQ_sensitivity_wikitext_hpu.py
```

### Phase 2: Mixed-Precision Quantization Evaluation

#### Fake Quantization (Testing)
```bash
# A100
python src/WikiText/PMPQ_evaluation_wikitext_A100.py 

# Gaudi
python src/WikiText/PMPQ_evaluation_wikitext_hpu.py 
```

#### Real Quantization (Deployment)
```bash
# A100 - INT8/FP16
python src/WikiText/PMPQ_evaluation_wikitext_A100_real.py \
    --sensitivity_file Sensitivities/A100/WikiText/sens_*.json \
    --save_model

# Gaudi - FP8/BF16
# Example usage (MEASURE phase):
export LOG_LEVEL_INC=1
PT_HPU_LAZY_MODE=1 python src/WikiText/PMPQ_evaluation_wikitext_hpu_real.py \
    --sensitivity_file Sensitivities/Gaudi2/WikiText/sens_*.json \
    --run_mode measure \
    --target_family mlp_only \
    --clustering percentile \
    --baseline_dtype fp32 \
    --hp_dtype bf16 \
    --fp8_format E4M3 \
    --scale_method maxabs_hw_opt_weight\
    --calib_samples 512 \
    --use_mark_step

# Example usage (Quantize phase):
export LOG_LEVEL_INC=1
export PT_HPU_WEIGHT_SHARING=0
PT_HPU_LAZY_MODE=1 python src/WikiText/PMPQ_evaluation_wikitext_hpu_real.py \
    --sensitivity_file Sensitivities/Gaudi2/WikiText/sens_*.json \
    --run_mode quantize \
    --target_family mlp_only \
    --clustering percentile \
    --baseline_dtype fp32 \
    --hp_dtype bf16 \
    --fp8_format E4M3 \
    --scale_method maxabs_hw_opt_weight\
    --use_mark_step

```

## Configuration Options

### Common Parameters
- `--model_name`: Model to quantize (default: TinyLlama-1.1B)
- `--sparsity`: Pruning sparsity for sensitivity (default: 0.3)
- `--batch_size`: Evaluation batch size (default: 64)
- `--max_samples`: Limit evaluation samples (default: 2000)
- `--group_size`: GroupWise quantization group size (default: 128)

### Platform-Specific
#### A100
- `--num_bits`: Target bit-width (4, 8, 16)
- `--quantization_mode`: symmetric/asymmetric

#### Gaudi
- `--fp8_format`: E4M3 or E5M2
- `--hp_dtype`: bf16, fp16, or fp32
- `--use_habana_mixed`: Enable Habana mixed precision

## Results & Visualizations

### Sensitivity Analysis Results

![Layer Sensitivity ](plots/PMPQ_TinyLlama_WikiText_sensitivity.png)
*Figure 1: Layer-wise sensitivity*

![Submodule Sensitivity](plots/submodule_sensitivity_mlp_fp8_bf16_hellaswag.png)
*Figure 2: Fine-grained submodule sensitivity (attention vs MLP components)*

### Fine-tuning LLMs results

![LLM Fine-tuning](plots/Fig2.png)
*Figure 3: LLM Fine-tuning results

<!-- ![Memory Savings](plots/memory_savings_placeholder.png)
*Figure 4: Memory reduction achieved with mixed-precision quantization*

### Platform Comparison

![Gaudi vs A100 Performance](plots/platform_comparison_placeholder.png)
*Figure 5: Performance comparison between Gaudi FP8 and A100 INT8*

### Cluster Analysis

![Bit Allocation Clusters](plots/cluster_analysis_placeholder.png)
*Figure 6: Layer clustering and bit allocation strategy* -->

## Generating Plots -->

```bash
# Generate sensitivity heatmaps
python src/plot_submodule_sensitivity.py \
    --sensitivity_dir Sensitivities/ \
    --output_dir plots/

# Generate cluster analysis plots
python src/plot_gaudi2_eval_clusters.py \
    --eval_dir Evaluation/Gaudi2/ \
    --output_dir plots/

# Custom plotting example
python -c "
import matplotlib.pyplot as plt
import json

# Load sensitivity data
with open('Sensitivities/A100/WikiText/sens_*.json') as f:
    data = json.load(f)

# Create your custom plot
plt.figure(figsize=(12, 6))
plt.bar(range(len(data['sensitivities'])), data['sensitivities'])
plt.xlabel('Layer Index')
plt.ylabel('Sensitivity Score')
plt.title('Layer-wise Sensitivity Analysis')
plt.savefig('plots/custom_sensitivity.png')
"
```

## Performance Benchmarks

### Model: TinyLlama-1.1B (1.1B parameters)

#### Comprehensive Quantization Results

| Dataset | Metric | FP32 Baseline |  | Fake Quantization |  |  |  |  |  | Real Quantization |  |
|---------|--------|---------------|--|-------------------|--|--|--|--|--|-------------------|--|
| | |  |  | **Fake [16,8,4]** |  | **Fake [8,8,4]** |  | **Fake [8,4,4]** |  | **Real [BF16,FP8]** | **Real [FP16,INT8]** |
| | | Gaudi2 |A100  | Gaudi2 | A100 | Gaudi2 | A100 | Gaudi2 | A100 | Gaudi2 | A100 |
| **WikiText** | Perplexity ↓ | 12.02 | 12.02 | 12.09 | 12.26 | 12.09 | 12.80 | 12.39 | 12.94 | 12.02 | 12.02 |
| | Compression Ratio | 1× | 1× | 3.47× | 3.47× | 4.80× | 4.80× | 5.93× | 5.93× | 2.42× | 3.77× |
| | Throughput (tok/s) | 32397 | 17024 | 34235 | 17224 | 33695 | 17203 | 34081 | 17232 | 23484 | 13547 |
| | Eval Time (sec) | 10.45 | 19.88 | 9.89 | 19.65 | 10.04 | 19.67 | 9.93 | 19.64 | 14.41 | 24.98 |
| **HellaSwag** | Accuracy (%) ↑ | 56.86 | 56.97 | 57.02 | 56.86 | 57.04 | 56.70 | 56.39 | 56.45 | 56.92 | 56.92 |
| | Compression Ratio | 1× | 1× | 3.28× | 3.47× | 4.67× | 4.80× | 5.74× | 5.93× | 2.3× | 3.01× |
| | Throughput (samp/s) | 22 | 13 | 31 | 13 | 32 | 13 | 29 | 13 | 14 | 50 |
| | Eval Time (sec) | 460.58 | 750.42 | 325.62 | 748 | 312.47 | 746 | 351.42 | 745.14 | 707 | 199.37 |
| **BoolQ** | Accuracy (%) ↑ | 57.89 | 58.07 | 54.43 | 54.31 | 54.43 | 53.98 | 52.91 | 51.87 | 57.19 | 57.77 |
| | Compression Ratio | 1× | 1× | 3.28× | 3.47× | 4.67× | 4.80× | 5.74× | 5.93× | 2.40× | 3.01× |
| | Throughput (samp/s) | 14 | 8 | 22 | 9 | 19 | 9 | 24 | 9 | 7 | 8 |
| | Eval Time (sec) | 236.43 | 410.69 | 150.86 | 364.95 | 174.61 | 367.83 | 135.21 | 368.93 | 453.31 | 405.95 |


<!-- 
## Advanced Usage

### Custom Sensitivity Metrics

```python
# Implement custom sensitivity calculation
from src.WikiText.PMPQ_sensitivity_wikitext_A100 import compute_sensitivity

def custom_sensitivity(model, layer_idx):
    # Your custom metric (e.g., gradient-based, Fisher information)
    return sensitivity_score

# Use in evaluation
sensitivities = [custom_sensitivity(model, i) for i in range(22)]
```

### Multi-Model Comparison

```bash
# Run on multiple models
for model in "TinyLlama-1.1B" "Llama-2-7B" "Mistral-7B"; do
    python src/WikiText/PMPQ_sensitivity_wikitext_A100.py \
        --model_name $model \
        --output_dir Sensitivities/A100/$model/
done
```

### Hyperparameter Sweep

```bash
# Test different configurations
for sparsity in 0.1 0.2 0.3 0.4; do
    for group_size in 64 128 256; do
        python src/WikiText/PMPQ_evaluation_wikitext_A100.py \
            --sparsity $sparsity \
            --group_size $group_size \
            --output_tag "s${sparsity}_g${group_size}"
    done
done
``` -->

<!-- ## Troubleshooting

### Common Issues

1. **OOM Errors on A100**
   - Reduce batch_size (e.g., 32 → 16)
   - Move logits to CPU after forward pass
   - Use gradient checkpointing if available

2. **Gaudi HPU Graph Errors**
   - Ensure PT_HPU_LAZY_MODE=1 is set
   - Clear HPU cache: `torch.hpu.empty_cache()`
   - Restart pod if persistent issues

3. **Sensitivity File Not Found**
   - Check path: `ls Sensitivities/*/`
   - Ensure Phase 1 completed successfully
   - Verify file naming convention

4. **Low Accuracy After Quantization**
   - Increase n_clusters (e.g., 4 → 8)
   - Use asymmetric quantization
   - Adjust sensitivity threshold -->

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{voyager_gaudi_benchmarks2024,
  title={Performance Optimization and Comparative Analysis of Generative AI Models on Advanced Accelerators},
  author={Nanda, Amitash, Hernandez Nicolau, Javier, Gujral, Madhusudan, Tatineni, Mahidhar, Majumdar, Amitava, Sahoo, Debashis},
  year={2026},
  publisher={GitHub},
  url={https://github.com/amitashnanda/Voyager_Gaudi_Benchmarks}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

- **Author**: Amitash Ananda, Javier Hernandez Nicolau, Madhusudan Gujral, Mahidhar Tatineni, Amitava Majumdar, Debashis Sahoo 
- **Email**: ananda@ucsd.edu
- **GitHub**: [@amitashnanda](https://github.com/amitashnanda)

## Acknowledgments

- Intel Habana team for Gaudi support
- Voyager team for Gaudi support
- NERSC for providing the computational resources (Perlmutter)
- NVIDIA for A100 optimization guides
- HuggingFace for model hosting
- The open-source community

---

**Note**: This is an active research project. Results and configurations may be updated as experiments progress.