#!/bin/bash

### "gradient_clipping": 1.0,
### has been replaced with
### "gradient_clipping": "auto",
### in deepspeed_config_zero_2.json file
### else it interfers with  --max_grad_norm  0.3


python3 ../gaudi_spawn.py \
    --world_size 1 --use_mpi run_lora_clm.py \
    --model_name_or_path google/gemma-2-27b-it \
    --train_file /home/mgujral/BoltMonkey_psychology-instruction-output.parquet \
    --bf16 True \
    --output_dir ./gemma2_27b_lora64_1HPU_withBoltMonkeyParquet_gaudi2_August4_2025 \
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
    --use_lazy_mode \
    --throughput_warmup_steps 3 \
    --lora_rank=64 \
    --lora_alpha=64 \
    --lora_dropout=0.05 \
    --lora_target_modules "q_proj" "v_proj"  \
    --dataset_concatenation \
    --max_seq_length 512
