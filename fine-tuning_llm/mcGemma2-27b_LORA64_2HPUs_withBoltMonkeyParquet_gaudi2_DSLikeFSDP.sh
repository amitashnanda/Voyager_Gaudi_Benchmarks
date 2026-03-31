#!/bin/bash
  
### "gradient_clipping": 1.0,
### has been replaced with
### "gradient_clipping": "auto",
### in deepspeed_config_zero_2.json file
### else it interfers with  --max_grad_norm  0.3

### This is the same input as fsdp and deepspeed elements have substituted them.  

PT_HPU_LAZY_MODE=0 python3 ../gaudi_spawn.py \
    --world_size 2 --use_deepspeed  run_lora_clm.py \
    --model_name_or_path google/gemma-2-27b-it \
    --train_file /home/mgujral/parquetDatasets/BoltMonkey_psychology-instruction-output.parquet \
    --bf16 True \
    --output_dir ./gemma2_27b_lora64_2HPUs_withBoltMonkeyParquet_gaudi2_February19_2026_dslikefsdp \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --save_strategy "no" \
    --learning_rate 3e-4 \
    --warmup_ratio  0.03 \
    --lr_scheduler_type "constant" \
    --max_grad_norm  0.3 \
    --logging_steps 1 \
    --do_train \
    --use_habana \
    --throughput_warmup_steps 3 \
    --lora_rank=64 \
    --lora_alpha=64 \
    --lora_dropout=0.05 \
    --lora_target_modules "q_proj" "v_proj"  \
    --dataset_concatenation \
    --use_lazy_mode False \
    --deepspeed ./deepspeed_zero3_config_grad_0.3.json \
    --pipelining_fwd_bwd False \
    --use_fused_rope False \
    --torch_compile_backend hpu_backend \
    --torch_compile \
    --max_seq_length 512 \
    --use_flash_attention True \
    --flash_attention_causal_mask True
