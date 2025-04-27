set -x

GPUS=${GPUS:-2}
BATCH_SIZE=${BATCH_SIZE:-32}
#PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))
#GRADIENT_ACC=1

#export PYTHONPATH="${PYTHONPATH}:$(pwd)"
#export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONPATH="${PYTHONPATH}:/root/autodl-tmp/workspace/InternVL/internvl_chat"
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.4/targets/x86_64-linux/lib
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

OUTPUT_DIR='/root/autodl-tmp/workspace/InternVL/internvl_chat/shell/internvl2.0/2nd_finetune/8b-10140/outputdir-8b-Q1-Q6-MPO'

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
  /root/autodl-tmp/workspace/InternVL/internvl_chat/internvl/train/internvl_chat_dpo.py\
  --model_name_or_path "/root/autodl-tmp/workspace/InternVL/internvl_chat/shell/internvl2.0/2nd_finetune/8b-10140/outputdir-8b-Q1-Q6" \
    --conv_style "internlm2-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/root/autodl-tmp/workspace/InternVL/internvl_chat/shell/data/chaoyi_finetune_Q1_Q6_DPO_10140.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --pad2square False \
  --freeze_llm False \
  --freeze_mlp False \
  --freeze_backbone False \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 8 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "no" \
  --save_steps 550 \
  --save_total_limit 100 \
  --learning_rate 5e-6 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 6144 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length False \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "/root/autodl-tmp/workspace/InternVL/internvl_g/zero_stage4_config.json" \
  --report_to "tensorboard" \
  --loss_type sigmoid,bco_pair \
  --sigmoid_loss_weight 0.8 \
  --bco_pair_loss_weight 0.2 \
  --rpo_alpha 1 \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"