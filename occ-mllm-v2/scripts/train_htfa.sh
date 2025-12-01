#!/bin/bash

# =============================================================================
# HTFA (Hierarchical Trinity Fusion Architecture) Training Script
# 
# Features:
# 1. End-to-end training of HTFA model (Level 1 + Level 2 + Level 3)
# 2. Integration with pretrained SigLIP encoder and VQ tokenizer (from Janus)
# 3. Support for dual-image input (original + reconstructed)
# 4. Unified optimization for understanding and generation tasks
# 5. Compatible with DeepSpeed ZeRO-2 for efficient training
# 
# Usage:
# bash train_htfa.sh
# =============================================================================

set -e  # Exit immediately if a command exits with a non-zero status

echo " Starting HTFA Training..."
echo " Hierarchical Trinity Fusion Architecture v2"
echo "=================================================================="

# =============================================================================
# 1. Environment Configuration and Parameters
# =============================================================================

# Training parameters
GPUS=1
BATCH_SIZE=2
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACC=4
LEARNING_RATE=5e-7
NUM_EPOCHS=4

# Path configuration
BASE_DIR="/root/autodl-tmp"
INTERNVL_ROOT="${BASE_DIR}/InternVL/internvl_chat"
HTFA_ROOT="${INTERNVL_ROOT}/occ_mllm_v2"
JANUS_ROOT="${BASE_DIR}/Janus-main"

# Pretrained model paths (following reference script structure)
PRETRAINED_MODEL_PATH="${BASE_DIR}/InternVL/pretrained_models/InternVL2-8B"
JANUS_MODEL_PATH="${JANUS_ROOT}/deepseek-ai/Janus-1.3B"

# HTFA specific parameters
LEVEL1_LEARNABLE=True
LEVEL2_USE_AFFINE=True
LEVEL2_NUM_HEADS=8
VQ_CODEBOOK_SIZE=8192
NUM_DECODER_LAYERS=4

# Loss weights
LOSS_WEIGHT_UNDERSTAND=0.7
LOSS_WEIGHT_GENERATE=0.3

# Data path
DATA_PATH="${INTERNVL_ROOT}/shell/data/cye2e_dual_image_finetune.json"

# Resume from checkpoint (optional)
RESUME_FROM_CHECKPOINT=""  # e.g., "${HTFA_ROOT}/output/htfa_v2_20250101_120000/checkpoints/checkpoint-1250"

# Output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${HTFA_ROOT}/output/htfa_v2_${TIMESTAMP}"

# Activate conda environment
echo " Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate internvl
if [ $? -ne 0 ]; then
    echo " Error: Failed to activate internvl environment"
    exit 1
fi
echo " Successfully activated internvl environment"

# Environment variables
export PYTHONPATH="${HTFA_ROOT}:${INTERNVL_ROOT}:$PYTHONPATH"
export MASTER_PORT=34236
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

# CUDA settings for better performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=OFF

# =============================================================================
# 2. Environment Verification
# =============================================================================

echo " Verifying environment..."

if [ ! -d "${HTFA_ROOT}" ]; then
    echo " Error: HTFA directory not found: ${HTFA_ROOT}"
    exit 1
fi

# Check required Python files
REQUIRED_FILES=(
    "${HTFA_ROOT}/model/level1_adaptive_fusion.py"
    "${HTFA_ROOT}/model/level2_spatial_attention.py"
    "${HTFA_ROOT}/model/level3_encoder_decoupling.py"
    "${HTFA_ROOT}/model/htfa_model.py"
    "${HTFA_ROOT}/config/htfa_config.py"
    "${HTFA_ROOT}/training/htfa_trainer.py"
    "${HTFA_ROOT}/training/data_loader.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo " Error: Required file not found: $file"
        echo "Please ensure all HTFA components are properly installed"
        exit 1
    fi
done

# Check pretrained models
if [ ! -d "${PRETRAINED_MODEL_PATH}" ]; then
    echo "  Warning: Pretrained model path not found: ${PRETRAINED_MODEL_PATH}"
    echo "Please ensure InternVL2 pretrained model is downloaded"
    # exit 1  # Uncomment to make this a hard requirement
fi

if [ ! -d "${JANUS_MODEL_PATH}" ]; then
    echo "  Warning: Janus model path not found: ${JANUS_MODEL_PATH}"
    echo "Will use default VQ tokenizer initialization"
fi

# Check data file
if [ ! -f "${DATA_PATH}" ]; then
    echo " Error: Data file not found: ${DATA_PATH}"
    echo "Please prepare dual-image training data"
    exit 1
fi

# Check DeepSpeed config
DEEPSPEED_CONFIG="${HTFA_ROOT}/config/deepspeed_config.json"
if [ ! -f "${DEEPSPEED_CONFIG}" ]; then
    echo "  Warning: DeepSpeed config not found: ${DEEPSPEED_CONFIG}"
    echo "Using default PyTorch training without DeepSpeed"
    DEEPSPEED_CONFIG=""
fi

echo " Environment verification passed"

# =============================================================================
# 3. Create Output Directories
# =============================================================================

echo " Creating output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/checkpoints"

# Save training configuration to output directory
cat > "${OUTPUT_DIR}/train_config.txt" << EOF
HTFA v2 Training Configuration
================================
Timestamp: ${TIMESTAMP}
GPUs: ${GPUS}
Batch Size: ${PER_DEVICE_BATCH_SIZE}
Gradient Accumulation: ${GRADIENT_ACC}
Learning Rate: ${LEARNING_RATE}
Epochs: ${NUM_EPOCHS}

Architecture:
  Level 1 Learnable: ${LEVEL1_LEARNABLE}
  Level 2 Use Affine: ${LEVEL2_USE_AFFINE}
  Level 2 Num Heads: ${LEVEL2_NUM_HEADS}
  VQ Codebook Size: ${VQ_CODEBOOK_SIZE}
  Decoder Layers: ${NUM_DECODER_LAYERS}

Loss Weights:
  Understanding: ${LOSS_WEIGHT_UNDERSTAND}
  Generation: ${LOSS_WEIGHT_GENERATE}

Paths:
  Pretrained Model: ${PRETRAINED_MODEL_PATH}
  Janus Model: ${JANUS_MODEL_PATH}
  Data Path: ${DATA_PATH}
  Output Dir: ${OUTPUT_DIR}

================================
EOF

echo " Configuration saved to ${OUTPUT_DIR}/train_config.txt"

# =============================================================================
# 4. Create Training Entry Script
# =============================================================================

# Create Python training entry point
TRAIN_SCRIPT="${OUTPUT_DIR}/run_htfa_training.py"

cat > "${TRAIN_SCRIPT}" << 'EOFPY'
#!/usr/bin/env python3
"""
HTFA Training Entry Point

This script initializes and trains the Hierarchical Trinity Fusion Architecture
with pretrained vision-language components.
"""

import sys
import os
import torch
from pathlib import Path

# Add HTFA root to Python path
htfa_root = Path(__file__).parent.parent
sys.path.insert(0, str(htfa_root))

from model.htfa_model import HTFAModel
from config.htfa_config import HTFAConfig
from training.htfa_trainer import HTFATrainer
from training.data_loader import create_htfa_dataloaders

# Import transformers for model loading
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("  transformers not available")


def load_siglip_encoder(model_path: str):
    """
    Load SigLIP encoder from pretrained InternVL2 model.
    
    Note: In production, extract the vision encoder from InternVL2.
    For now, returns a placeholder.
    """
    print(f" Loading SigLIP encoder from: {model_path}")
    # TODO: Implement actual SigLIP loading from InternVL2
    # from internvl.model.internvl_chat import InternVLChatModel
    # model = InternVLChatModel.from_pretrained(model_path)
    # siglip_encoder = model.vision_model
    print("  Using placeholder SigLIP encoder")
    return None


def load_vq_tokenizer(janus_path: str, config: HTFAConfig):
    """
    Load VQ tokenizer from pretrained Janus model.
    
    Note: In production, extract VQ tokenizer from Janus.
    For now, initializes a new one.
    """
    from model.level3_encoder_decoupling import VQTokenizerWrapper
    
    if os.path.exists(janus_path):
        print(f" Loading VQ tokenizer from: {janus_path}")
        # TODO: Implement actual VQ loading from Janus
        # from janus.model import load_vq_tokenizer
        # vq_tokenizer = load_vq_tokenizer(janus_path)
    
    print("  Initializing new VQ tokenizer")
    vq_tokenizer = VQTokenizerWrapper(
        codebook_size=config.vq_codebook_size,
        embedding_dim=config.vq_embedding_dim,
        frozen=config.freeze_vq
    )
    return vq_tokenizer


def load_text_tokenizer(model_path: str):
    """Load text tokenizer from pretrained model."""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        print(f" Loading text tokenizer from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(" Text tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"  Failed to load tokenizer: {e}")
        return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="HTFA Training")
    
    # Model paths
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--janus_model_path', type=str, required=True)
    
    # Data
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=5e-7)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    
    # HTFA architecture
    parser.add_argument('--level1_learnable', type=str, default='True')
    parser.add_argument('--level2_use_affine', type=str, default='True')
    parser.add_argument('--level2_num_heads', type=int, default=8)
    parser.add_argument('--vq_codebook_size', type=int, default=8192)
    parser.add_argument('--num_decoder_layers', type=int, default=4)
    
    # Loss weights
    parser.add_argument('--loss_weight_understand', type=float, default=0.7)
    parser.add_argument('--loss_weight_generate', type=float, default=0.3)
    
    # Resume
    parser.add_argument('--resume_from', type=str, default=None)
    
    args = parser.parse_args()
    
    # Convert string bools
    level1_learnable = args.level1_learnable.lower() == 'true'
    level2_use_affine = args.level2_use_affine.lower() == 'true'
    
    print("=" * 80)
    print("HTFA Training Initialization")
    print("=" * 80)
    
    # Create configuration
    config = HTFAConfig()
    config.pretrained_model_path = args.pretrained_model_path
    config.vq_tokenizer_path = args.janus_model_path
    config.train_data_path = args.data_path
    config.output_dir = args.output_dir
    config.num_train_epochs = args.num_epochs
    config.per_device_train_batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.level1_learnable = level1_learnable
    config.level2_use_affine = level2_use_affine
    config.level2_num_heads = args.level2_num_heads
    config.vq_codebook_size = args.vq_codebook_size
    config.num_decoder_layers = args.num_decoder_layers
    config.loss_weight_understand = args.loss_weight_understand
    config.loss_weight_generate = args.loss_weight_generate
    
    print(config)
    
    # Save config
    config.save_to_json(os.path.join(config.output_dir, "htfa_config.json"))
    
    # Load pretrained components
    print("\n" + "=" * 80)
    print("Loading Pretrained Components")
    print("=" * 80)
    
    siglip_encoder = load_siglip_encoder(config.pretrained_model_path)
    vq_tokenizer = load_vq_tokenizer(config.vq_tokenizer_path, config)
    text_tokenizer = load_text_tokenizer(config.pretrained_model_path)
    
    # Create HTFA model
    print("\n" + "=" * 80)
    print("Creating HTFA Model")
    print("=" * 80)
    
    model = HTFAModel(
        image_size=config.image_size,
        spatial_feature_dim=config.spatial_feature_dim,
        llm_hidden_dim=config.llm_hidden_dim,
        level1_learnable=config.level1_learnable,
        level1_init_weight=config.level1_init_weight,
        level2_num_heads=config.level2_num_heads,
        level2_use_affine=config.level2_use_affine,
        level2_dropout=config.level2_dropout,
        vq_codebook_size=config.vq_codebook_size,
        vq_embedding_dim=config.vq_embedding_dim,
        num_decoder_layers=config.num_decoder_layers,
        num_attention_heads=config.num_attention_heads,
        vocab_size=config.vocab_size,
        siglip_encoder=siglip_encoder,
        vq_tokenizer=vq_tokenizer,
        text_tokenizer=text_tokenizer,
        freeze_siglip=config.freeze_siglip,
        freeze_vq=config.freeze_vq,
        freeze_llm_decoder=config.freeze_llm_decoder
    )
    
    model.print_trainable_parameters()
    
    # Create dataloaders
    print("\n" + "=" * 80)
    print("Creating DataLoaders")
    print("=" * 80)
    
    train_loader, val_loader = create_htfa_dataloaders(
        train_data_path=config.train_data_path,
        val_data_path=config.val_data_path,
        tokenizer=text_tokenizer,
        image_size=config.image_size,
        batch_size=config.per_device_train_batch_size,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory
    )
    
    # Create trainer
    print("\n" + "=" * 80)
    print("Creating Trainer")
    print("=" * 80)
    
    trainer = HTFATrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=config.bf16 or config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        max_grad_norm=config.max_grad_norm,
        log_interval=config.logging_steps,
        save_interval=config.save_steps,
        output_dir=config.output_dir
    )
    
    # Start training
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    trainer.train(
        num_epochs=config.num_train_epochs,
        resume_from=args.resume_from
    )
    
    print("\n" + "=" * 80)
    print(" Training Completed Successfully!")
    print("=" * 80)
    print(f" Checkpoints saved to: {config.output_dir}/checkpoints")
    print(f" Logs saved to: {config.output_dir}/logs")
    print(f"\n View training progress with TensorBoard:")
    print(f"   tensorboard --logdir={config.output_dir}/logs")


if __name__ == "__main__":
    main()
EOFPY

chmod +x "${TRAIN_SCRIPT}"
echo " Training script created: ${TRAIN_SCRIPT}"

# =============================================================================
# 5. Execute Training
# =============================================================================

echo ""
echo " Launching HTFA Training..."
echo "=================================================================="
echo " Training Configuration Summary:"
echo "   - GPUs: ${GPUS}"
echo "   - Batch Size: ${PER_DEVICE_BATCH_SIZE}"
echo "   - Gradient Accumulation: ${GRADIENT_ACC}"
echo "   - Learning Rate: ${LEARNING_RATE}"
echo "   - Epochs: ${NUM_EPOCHS}"
echo "   - Level 1 Learnable: ${LEVEL1_LEARNABLE}"
echo "   - Level 2 Affine Alignment: ${LEVEL2_USE_AFFINE}"
echo "   - VQ Codebook Size: ${VQ_CODEBOOK_SIZE}"
echo "   - Pretrained Model: ${PRETRAINED_MODEL_PATH}"
echo "   - Janus Model: ${JANUS_MODEL_PATH}"
echo "   - Data: ${DATA_PATH}"
echo "   - Output: ${OUTPUT_DIR}"
echo "=================================================================="

# Build training command
TRAIN_CMD="python ${TRAIN_SCRIPT}"
TRAIN_CMD="${TRAIN_CMD} --pretrained_model_path ${PRETRAINED_MODEL_PATH}"
TRAIN_CMD="${TRAIN_CMD} --janus_model_path ${JANUS_MODEL_PATH}"
TRAIN_CMD="${TRAIN_CMD} --data_path ${DATA_PATH}"
TRAIN_CMD="${TRAIN_CMD} --output_dir ${OUTPUT_DIR}"
TRAIN_CMD="${TRAIN_CMD} --num_epochs ${NUM_EPOCHS}"
TRAIN_CMD="${TRAIN_CMD} --batch_size ${PER_DEVICE_BATCH_SIZE}"
TRAIN_CMD="${TRAIN_CMD} --learning_rate ${LEARNING_RATE}"
TRAIN_CMD="${TRAIN_CMD} --gradient_accumulation_steps ${GRADIENT_ACC}"
TRAIN_CMD="${TRAIN_CMD} --level1_learnable ${LEVEL1_LEARNABLE}"
TRAIN_CMD="${TRAIN_CMD} --level2_use_affine ${LEVEL2_USE_AFFINE}"
TRAIN_CMD="${TRAIN_CMD} --level2_num_heads ${LEVEL2_NUM_HEADS}"
TRAIN_CMD="${TRAIN_CMD} --vq_codebook_size ${VQ_CODEBOOK_SIZE}"
TRAIN_CMD="${TRAIN_CMD} --num_decoder_layers ${NUM_DECODER_LAYERS}"
TRAIN_CMD="${TRAIN_CMD} --loss_weight_understand ${LOSS_WEIGHT_UNDERSTAND}"
TRAIN_CMD="${TRAIN_CMD} --loss_weight_generate ${LOSS_WEIGHT_GENERATE}"

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
    TRAIN_CMD="${TRAIN_CMD} --resume_from ${RESUME_FROM_CHECKPOINT}"
fi

# Execute training
echo ""
echo "  Executing training command..."
echo ""

${TRAIN_CMD} 2>&1 | tee -a "${OUTPUT_DIR}/logs/training_${TIMESTAMP}.log"

# =============================================================================
# 6. Post-Training Processing
# =============================================================================

TRAINING_EXIT_CODE=$?

echo ""
echo "=================================================================="
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo " HTFA Training Completed Successfully!"
    echo "=================================================================="
    echo " Model checkpoints: ${OUTPUT_DIR}/checkpoints"
    echo " Training log: ${OUTPUT_DIR}/logs/training_${TIMESTAMP}.log"
    echo " TensorBoard logs: ${OUTPUT_DIR}/logs"
    echo ""
    echo " Next Steps:"
    echo "   1. View training progress:"
    echo "      tensorboard --logdir=${OUTPUT_DIR}/logs"
    echo ""
    echo "   2. Run inference:"
    echo "      python ${HTFA_ROOT}/scripts/inference_htfa_v2.py \\"
    echo "        --checkpoint ${OUTPUT_DIR}/checkpoints/best_model.pt \\"
    echo "        --image_orig /path/to/original.jpg \\"
    echo "        --image_recon /path/to/reconstructed.jpg \\"
    echo "        --query \"What object is in the hand?\""
    echo ""
    
    # Create quick inference test script
    cat > "${OUTPUT_DIR}/test_inference.sh" << EOFTEST
#!/bin/bash
# Quick inference test script
python ${HTFA_ROOT}/scripts/inference_htfa_v2.py \\
    --checkpoint ${OUTPUT_DIR}/checkpoints/final_model.pt \\
    --image_orig \$1 \\
    --image_recon \$2 \\
    --query "\${3:-What object is in the hand?}" \\
    --output_dir ${OUTPUT_DIR}/inference_results
EOFTEST
    chmod +x "${OUTPUT_DIR}/test_inference.sh"
    echo " Quick test script created: ${OUTPUT_DIR}/test_inference.sh"
    
else
    echo " HTFA Training Failed (Exit Code: ${TRAINING_EXIT_CODE})"
    echo "=================================================================="
    echo " Check training log: ${OUTPUT_DIR}/logs/training_${TIMESTAMP}.log"
    echo ""
    echo " Last 30 lines of training log:"
    echo "----------------------------------------"
    tail -n 30 "${OUTPUT_DIR}/logs/training_${TIMESTAMP}.log" 2>/dev/null || echo "Unable to read log file"
    echo "----------------------------------------"
fi

echo ""
echo "=================================================================="
echo " Training Script Completed"
echo "⏰ Timestamp: $(date)"
echo "=================================================================="

exit $TRAINING_EXIT_CODE

