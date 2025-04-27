set -x

GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-256}
#PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-128}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
#GRADIENT_ACC=1

#export PYTHONPATH="${PYTHONPATH}:$(pwd)"
#export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH="${PYTHONPATH}:/root/autodl-tmp/workspace/InternVL/internvl_chat"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/targets/x86_64-linux/lib
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='/root/autodl-tmp/workspace/InternVL/internvl_chat/shell/internvl2.0/2nd_finetune/4b-10140/outputdir-4b-Q1'

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 8
# batch size per gpu: 4
# gradient accumulation steps: 4
# total batch size: 128
# epoch: 1
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  /root/autodl-tmp/workspace/InternVL/internvl_chat/internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "/root/autodl-tmp/install/InternVL2-4B/" \
  --conv_style "phi3-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/root/autodl-tmp/workspace/InternVL/internvl_chat/shell/data/chaoyi_finetune_Q1_10140.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 2 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone True \
  --vision_select_layer -1 \
  --dataloader_num_workers 1 \
  --bf16 True \
  --num_train_epochs 2 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 2048 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "/root/autodl-tmp/workspace/InternVL/internvl_g/zero_stage4_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
