# HTFA: Hierarchical Trinity Fusion Architecture

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

**HTFA** (Hierarchical Trinity Fusion Architecture) is a novel end-to-end architecture for occluded object understanding and generation tasks. It integrates three hierarchical fusion levels to achieve unified visual understanding and generation through learnable fusion, spatial attention, and encoder decoupling.

## 🏗️ Architecture Overview

The HTFA architecture consists of three hierarchical fusion levels:

### Level 1: Adaptive Weighted Image Fusion
- **Purpose**: Pixel-level blending of original and reconstructed images
- **Method**: Learnable fusion weights α optimized via backpropagation
- **Formula**: `I_fused = α ⊙ I_orig + (1-α) ⊙ I_recon`
- **Key Feature**: Gradient flow through both understanding and generation pathways

### Level 2: Spatial Attention Affine Fusion
- **Purpose**: Position-aware cross-attention with spatial alignment
- **Method**: Affine transformation + multi-head cross-attention
- **Components**:
  - Query: from original image features
  - Key: from spatially-aligned reconstructed features
  - Value: from fused image features
- **Key Feature**: Automatic object alignment via bounding box computation

### Level 3: Visual Encoder Decoupling Fusion
- **Purpose**: Resolve conflicting requirements between understanding and generation
- **Method**: Dual-encoder architecture with unified autoregressive integration
- **Branches**:
  - **Understanding Branch**: SigLIP features → MLP adapter → LLM
  - **Generation Branch**: VQ tokenization → MLP adapter → LLM
- **Key Feature**: Dual prediction heads for text and visual tokens

## 📁 Project Structure

```
occ_mllm_v2/
├── model/                              # Core model components
│   ├── level1_adaptive_fusion.py       # Level 1: Adaptive weighted fusion
│   ├── level2_spatial_attention.py     # Level 2: Spatial attention fusion
│   ├── level3_encoder_decoupling.py    # Level 3: Encoder decoupling fusion
│   └── htfa_model.py                   # Complete HTFA model integration
│
├── training/                           # Training utilities
│   ├── htfa_trainer.py                 # End-to-end trainer with DeepSpeed support
│   └── data_loader.py                  # Dual-image data loader
│
├── config/                             # Configuration files
│   ├── htfa_config.py                  # Model configuration class
│   └── deepspeed_config.json          # DeepSpeed ZeRO-2 configuration
│
├── scripts/                            # Executable scripts
│   └── inference_htfa_v2.py           # Inference script
│
├── train_htfa.sh                       # Main training script
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd /root/autodl-tmp/InternVL/internvl_chat/occ_mllm_v2

# Install dependencies (if not already installed)
pip install torch torchvision transformers deepspeed tensorboard
```

### 2. Prepare Data

Prepare your dual-image dataset in JSON format:

```json
[
  {
    "image_orig": "/path/to/original_image.jpg",
    "image_recon": "/path/to/reconstructed_image.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nWhat's the object in the hand?"},
      {"from": "gpt", "value": "The object is banana."}
    ],
    "occlusion_mask": "/path/to/mask.png"  // optional
  }
]
```

### 3. Download Pretrained Models

```bash
# Download InternVL2 pretrained model (for SigLIP encoder + LLM)
# Place at: /root/autodl-tmp/InternVL/pretrained_models/InternVL2-8B

# Download Janus pretrained model (for VQ tokenizer)
# Place at: /root/autodl-tmp/Janus-main/deepseek-ai/Janus-1.3B
```

### 4. Training

#### Basic Training
```bash
bash train_htfa.sh
```

#### Custom Configuration
Edit the parameters in `train_htfa.sh`:
```bash
# Training parameters
GPUS=1                          # Number of GPUs
BATCH_SIZE=2                    # Batch size per device
GRADIENT_ACC=4                  # Gradient accumulation steps
LEARNING_RATE=5e-7              # Learning rate
NUM_EPOCHS=4                    # Training epochs

# Architecture parameters
LEVEL1_LEARNABLE=True           # Enable learnable fusion weights
LEVEL2_USE_AFFINE=True          # Enable affine alignment
LEVEL2_NUM_HEADS=8              # Number of attention heads
VQ_CODEBOOK_SIZE=8192           # VQ codebook size
NUM_DECODER_LAYERS=4            # Number of decoder layers

# Loss weights
LOSS_WEIGHT_UNDERSTAND=0.7      # Weight for understanding loss
LOSS_WEIGHT_GENERATE=0.3        # Weight for generation loss
```

#### Resume Training
```bash
# Set checkpoint path in train_htfa.sh
RESUME_FROM_CHECKPOINT="/path/to/checkpoint-1250.pt"
bash train_htfa.sh
```

### 5. Inference

```bash
python scripts/inference_htfa_v2.py \
    --checkpoint ./output/htfa_v2_20250101_120000/checkpoints/best_model.pt \
    --image_orig /path/to/original_image.jpg \
    --image_recon /path/to/reconstructed_image.jpg \
    --query "What object is in the hand?" \
    --output_dir ./inference_results
```

#### Batch Inference
```python
from scripts.inference_htfa_v2 import HTFAInference

# Initialize inference engine
engine = HTFAInference(
    checkpoint_path="./checkpoints/best_model.pt",
    device="cuda"
)

# Batch inference
image_pairs = [
    ("orig1.jpg", "recon1.jpg"),
    ("orig2.jpg", "recon2.jpg"),
]
queries = [
    "What object is this?",
    "Describe the occluded object."
]

results = engine.batch_infer(image_pairs, queries, batch_size=4)
```

## 📊 Monitoring Training

### TensorBoard
```bash
# View training progress
tensorboard --logdir=./output/htfa_v2_20250101_120000/logs

# Open browser at: http://localhost:6006
```

### Training Logs
```bash
# View real-time logs
tail -f ./output/htfa_v2_20250101_120000/logs/training_20250101_120000.log
```

## ⚙️ Configuration

### Model Configuration (`config/htfa_config.py`)

Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | 448 | Input image resolution |
| `spatial_feature_dim` | 1152 | SigLIP feature dimension |
| `llm_hidden_dim` | 2048 | LLM hidden dimension |
| `vq_codebook_size` | 8192 | VQ codebook size |
| `num_decoder_layers` | 4 | Number of autoregressive decoder layers |
| `learning_rate` | 5e-7 | Base learning rate |
| `warmup_ratio` | 0.03 | Warmup ratio for learning rate scheduler |

### DeepSpeed Configuration (`config/deepspeed_config.json`)

Features:
- **ZeRO Stage 2**: Parameter and gradient partitioning
- **BF16 Training**: Mixed precision for faster training
- **Gradient Clipping**: Max norm of 1.0
- **Optimizer**: AdamW with cosine learning rate decay

## 🎯 Key Features

### 1. End-to-End Gradient Flow
- All three hierarchical levels are jointly optimized
- Unified backpropagation through understanding and generation tasks
- Learnable fusion weights at each level

### 2. Dual-Task Optimization
- **Understanding**: Visual question answering, object recognition
- **Generation**: Visual token prediction for image completion
- Weighted loss combining both objectives

### 3. Efficient Training
- DeepSpeed ZeRO-2 for reduced memory footprint
- Gradient checkpointing for large models
- Mixed precision (BF16) training
- Automatic gradient accumulation

### 4. Flexible Architecture
- Configurable number of decoder layers
- Adjustable attention heads
- Optional multi-scale fusion
- Freezeable pretrained components

## 📈 Performance Tips

### Memory Optimization
```python
# Reduce batch size
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACC=8

# Enable gradient checkpointing
gradient_checkpointing=True

# Freeze more components
freeze_siglip=True
freeze_vq=True
freeze_llm_decoder=True  # Freeze LLM decoder if memory limited
```

### Speed Optimization
```bash
# Increase dataloader workers
dataloader_num_workers=4

# Enable pin memory
dataloader_pin_memory=True

# Use larger batch size if memory allows
PER_DEVICE_BATCH_SIZE=4
GRADIENT_ACC=2
```

## 🔬 Ablation Studies

To conduct ablation studies, modify the configuration:

### Disable Level 1 Learnable Fusion
```bash
LEVEL1_LEARNABLE=False  # Use fixed α=0.5
```

### Disable Level 2 Affine Alignment
```bash
LEVEL2_USE_AFFINE=False  # Skip spatial alignment
```

### Adjust Loss Weights
```bash
# Understanding-only
LOSS_WEIGHT_UNDERSTAND=1.0
LOSS_WEIGHT_GENERATE=0.0

# Generation-only
LOSS_WEIGHT_UNDERSTAND=0.0
LOSS_WEIGHT_GENERATE=1.0
```

## 🐛 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Solution: Reduce batch size or enable gradient checkpointing
PER_DEVICE_BATCH_SIZE=1
gradient_checkpointing=True
```

#### 2. Pretrained Model Not Found
```bash
# Solution: Check model paths in train_htfa.sh
ls /root/autodl-tmp/InternVL/pretrained_models/InternVL2-8B
ls /root/autodl-tmp/Janus-main/deepseek-ai/Janus-1.3B
```

#### 3. Data Loading Error
```bash
# Solution: Verify data format and paths
python -c "import json; print(json.load(open('data.json'))[:1])"
```

#### 4. DeepSpeed Error
```bash
# Solution: Disable DeepSpeed and use standard PyTorch
# Comment out or remove deepspeed_config.json reference
```

## 📝 Citation

If you use HTFA in your research, please cite:

```bibtex
@article{wang5702186occ,
  title={OCC-MLLM-V1: Commonsense-Guided Multi-Modal LLM Based Agent for Occlusion Reasoning With Internal Chain-of-Thoughts (CoTs) Guidance},
  author={Wang, Chaoyi and He, Qingdong and Pei, Jun and Xia, Lijie and Liu, Jianpo and Li, Baoqing and Di, Xinhan},
  journal={Available at SSRN 5702186}
}

@article{wangocc,
  title={Occ-Mllm-Cot: Self-Correction Enhanced Occlusion Recognition with Large Language Models Via 3d-Aware Supervision, Chain-of-Thoughts Guidance},
  author={Wang, Chaoyi and Meng, Fangzhou and Pei, Jun and Xia, Lijie and Liu, Jianpo and Yuan, Xiaobing and Di, Xinhan},
  journal={Chain-of-Thoughts Guidance}
}

@article{wang2025occ,
  title={OCC-MLLM-CoT-Alpha: Towards Multi-stage Occlusion Recognition Based on Large Language Models via 3D-Aware Supervision and Chain-of-Thoughts Guidance},
  author={Wang, Chaoyi and Li, Baoqing and Di, Xinhan},
  journal={arXiv preprint arXiv:2504.04781},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: chaoyiwang@mail.sim.ac.com

## 🙏 Acknowledgments

- **InternVL**: For the pretrained vision-language model
- **Janus**: For the VQ tokenizer architecture
- **DeepSpeed**: For efficient distributed training
- **PyTorch**: For the deep learning framework

---

**Built with ❤️ for advancing occluded object understanding**

