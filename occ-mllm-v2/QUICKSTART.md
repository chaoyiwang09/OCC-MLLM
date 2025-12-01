# HTFA Quick Start Guide

This guide will get you training and running inference with HTFA in under 10 minutes.

## 📦 Prerequisites

```bash
# Ensure you have these installed
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- transformers
- deepspeed (optional, for efficient training)
```

## 🚀 5-Minute Setup

### Step 1: Verify Installation

```bash
cd /root/autodl-tmp/InternVL/internvl_chat/occ_mllm_v2

# Check all files are present
ls -la
# You should see: model/, training/, config/, scripts/, train_htfa.sh, README.md
```

### Step 2: Prepare Your Data

Create a JSON file with dual-image pairs:
See in OCC-MLLM-V1

Save as: `/root/autodl-tmp/InternVL/internvl_chat/shell/data/your_data.json`

### Step 3: Configure Training

Edit `train_htfa.sh` (lines 34-50):

```bash
# Update these paths
DATA_PATH="/path/to/your_data.json"
PRETRAINED_MODEL_PATH="/path/to/InternVL2-8B"
JANUS_MODEL_PATH="/path/to/Janus-1.3B"
```

### Step 4: Start Training

```bash
# Quick debug run (1 epoch, 2 decoder layers)
bash train_htfa.sh

# OR edit train_htfa.sh to set:
# NUM_EPOCHS=4
# NUM_DECODER_LAYERS=4
# for full training
```

### Step 5: Monitor Training

```bash
# View real-time logs
tail -f ./output/htfa_v2_*/logs/training_*.log

# Start TensorBoard
tensorboard --logdir=./output/htfa_v2_*/logs
# Open: http://localhost:6006
```

## 🔍 Run Inference

After training completes:

```bash
python scripts/inference_htfa_v2.py \
    --checkpoint ./output/htfa_v2_20250101_120000/checkpoints/best_model.pt \
    --image_orig /path/to/test_original.jpg \
    --image_recon /path/to/test_reconstructed.jpg \
    --query "What's the object in the hand?" \
    --output_dir ./inference_results
```

## ⚡ Pre-configured Training Modes

Use pre-configured settings for common scenarios:

### Debug Mode (Fast Testing)
```bash
# Edit train_htfa.sh, set:
NUM_EPOCHS=1
NUM_DECODER_LAYERS=2
BATCH_SIZE=1
```

### Standard Training (Recommended)
```bash
# Default settings in train_htfa.sh
NUM_EPOCHS=4
NUM_DECODER_LAYERS=4
BATCH_SIZE=2
LEARNING_RATE=5e-7
```

### Memory-Efficient (16GB GPU)
```bash
# Edit train_htfa.sh, set:
BATCH_SIZE=1
GRADIENT_ACC=8
# In Python config, set:
# config.freeze_llm_decoder = True
# config.gradient_checkpointing = True
```

## 🎯 Understanding vs Generation Tasks

### Understanding Only (VQA, Recognition)
```bash
# Edit train_htfa.sh, set:
LOSS_WEIGHT_UNDERSTAND=1.0
LOSS_WEIGHT_GENERATE=0.0
```

### Generation Only (Image Completion)
```bash
# Edit train_htfa.sh, set:
LOSS_WEIGHT_UNDERSTAND=0.0
LOSS_WEIGHT_GENERATE=1.0
```

### Balanced (Default)
```bash
LOSS_WEIGHT_UNDERSTAND=0.7
LOSS_WEIGHT_GENERATE=0.3
```

## 📊 Expected Training Time

| Configuration | Hardware | Time per Epoch | Total Time (4 epochs) |
|---------------|----------|----------------|----------------------|
| Debug (2 layers) | 1x A100 (40GB) | ~30 min | ~2 hours |
| Standard (4 layers) | 1x A100 (40GB) | ~1 hour | ~4 hours |
| Large (8 layers) | 4x A100 (40GB) | ~45 min | ~3 hours |

*Based on ~10K training samples*

## 🐛 Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```bash
# Solution: Reduce batch size
BATCH_SIZE=1
GRADIENT_ACC=8  # Keep effective batch size = 8
```

### Issue 2: Pretrained Model Not Found
```bash
# Check paths exist
ls /root/autodl-tmp/InternVL/pretrained_models/InternVL2-8B
ls /root/autodl-tmp/Janus-main/deepseek-ai/Janus-1.3B

# If missing, download or update paths in train_htfa.sh
```

### Issue 3: Data Loading Error
```bash
# Validate your JSON format
python -c "import json; data=json.load(open('your_data.json')); print(f'Loaded {len(data)} samples')"
```

### Issue 4: Training Too Slow
```bash
# Enable optimizations:
# 1. Increase dataloader workers
dataloader_num_workers=4

# 2. Enable pin memory
dataloader_pin_memory=True

# 3. Use larger batch size if memory allows
BATCH_SIZE=4
```

## 📈 Monitoring Training Progress

### Key Metrics to Watch

1. **Total Loss** (`train/total_loss`): Should decrease steadily
   - Good: Drops from ~5.0 to ~1.0 within first epoch
   - Bad: Stays flat or increases

2. **Understanding Loss** (`train/loss_understand`): Text prediction quality
   - Target: < 1.5 for good performance

3. **Generation Loss** (`train/loss_generate`): VQ code prediction quality
   - Target: < 2.0 for good performance

4. **Learning Rate** (`train/lr`): Should follow warmup → cosine decay
   - Starts low, increases during warmup
   - Gradually decreases to min_lr

### When to Stop Training

**Stop if:**
- Validation loss stops decreasing for 3+ epochs (overfitting)
- Training loss becomes unstable (learning rate too high)
- Generation loss drops but understanding loss increases (imbalanced)

**Good training:**
- Both losses decrease steadily
- Validation loss tracks training loss
- Model checkpoints improve on test queries

## 🎓 Next Steps

After successful training:

1. **Evaluate Model**
   ```bash
   python scripts/inference_htfa_v2.py --checkpoint best_model.pt ...
   ```

2. **Run Ablation Studies**
   ```bash
   # Disable Level 1
   LEVEL1_LEARNABLE=False
   
   # Disable Level 2 Affine
   LEVEL2_USE_AFFINE=False
   ```

3. **Fine-tune on Your Domain**
   ```bash
   # Resume from checkpoint with new data
   RESUME_FROM_CHECKPOINT="path/to/checkpoint-1250.pt"
   DATA_PATH="path/to/domain_specific_data.json"
   ```

4. **Deploy for Production**
   ```python
   # See scripts/inference_htfa_v2.py for deployment examples
   from scripts.inference_htfa_v2 import HTFAInference
   engine = HTFAInference(checkpoint_path="best_model.pt")
   result = engine.infer(img_orig, img_recon, query)
   ```

## 💡 Pro Tips

1. **Start Small**: Always test with debug config before full training
2. **Monitor Early**: Watch first 100 steps closely for issues
3. **Save Often**: Use `--save_steps 100` during experimentation
4. **Validate Frequently**: Set `--evaluation_strategy steps --eval_steps 500`
5. **Use Checkpoints**: Resume from best checkpoint for fine-tuning

## 📚 Additional Resources

- **Full Documentation**: See [README.md](README.md)
- **Configuration Reference**: See [config/htfa_config.py](config/htfa_config.py)
- **Example Configs**: See [config/training_config_examples.py](config/training_config_examples.py)
- **Architecture Details**: See paper and model docstrings

## 🆘 Getting Help

If you encounter issues:

1. Check training logs: `cat output/htfa_v2_*/logs/training_*.log`
2. Verify data format: `python -c "import json; json.load(open('data.json'))"`
3. Test with debug config first
4. Open an issue with full error trace

---

**Ready to train?** Run `bash train_htfa.sh` and watch the magic happen! 🚀

