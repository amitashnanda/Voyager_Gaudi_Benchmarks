### "gradient_clipping": 1.0,
### has been replaced with
### "gradient_clipping": "auto",
### in deepspeed_config_zero_2.json file
### else it interfers with  --max_grad_norm  0.3


PT_HPU_LAZY_MODE=1 PT_HPU_MAX_COMPOUND_OP_SIZE=10 DEEPSPEED_HPU_ZERO3_SYNC_MARK_STEP_REQUIRED=1 \
python3 ../gaudi_spawn.py \
    --world_size 4 --use_deepspeed  run_lora_clm.py \
    --model_name_or_path meta-llama/Llama-3.3-70B-Instruct \
    --train_file /home/mgujral/BoltMonkey_psychology-instruction-output.parquet \
    --bf16 True \
    --output_dir ./llama3.3_70b_lora64_4HPUs_withDSBoltMonkeyParquet_gaudi2_November19_2025 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 3e-4 \
    --warmup_ratio  0.03 \
    --lr_scheduler_type "constant" \
    --max_grad_norm  1.0 \
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
    --max_seq_length 1024 \
    --validation_split_percentage 5 \
    --attn_softmax_bf16 True \
    --pipelining_fwd_bwd \
    --use_flash_attention True \
    --flash_attention_causal_mask True \
    --deepspeed ./llama3_ds_zero3_4k.json
###    --deepspeed deepspeed_zero3_config.json
