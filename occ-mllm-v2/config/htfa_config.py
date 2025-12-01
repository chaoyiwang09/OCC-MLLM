"""
HTFA Model Configuration

Centralized configuration for the Hierarchical Trinity Fusion Architecture.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json


@dataclass
class HTFAConfig:
    """
    Configuration class for HTFA model.
    
    This class holds all hyperparameters and settings for the complete
    Hierarchical Trinity Fusion Architecture.
    """
    
    # =========================================================================
    # Model Architecture
    # =========================================================================
    
    # Basic dimensions
    image_size: int = 448
    spatial_feature_dim: int = 1152  # SigLIP-SO400M dimension
    llm_hidden_dim: int = 2048  # InternLM2-7B dimension
    vocab_size: int = 92544  # InternLM2 vocabulary size
    
    # =========================================================================
    # Level 1: Adaptive Weighted Image Fusion
    # =========================================================================
    
    level1_learnable: bool = True
    level1_init_weight: float = 0.0  # Corresponds to α=0.5 after sigmoid
    level1_use_multi_scale: bool = False
    level1_num_scales: int = 3
    
    # =========================================================================
    # Level 2: Spatial Attention Affine Fusion
    # =========================================================================
    
    level2_num_heads: int = 8
    level2_use_affine: bool = True
    level2_dropout: float = 0.1
    
    # =========================================================================
    # Level 3: Visual Encoder Decoupling Fusion
    # =========================================================================
    
    # VQ Tokenizer
    vq_codebook_size: int = 8192
    vq_embedding_dim: int = 256
    vq_num_tokens_h: int = 32
    vq_num_tokens_w: int = 32
    
    # Autoregressive Decoder
    num_decoder_layers: int = 4
    num_attention_heads: int = 16
    decoder_dropout: float = 0.1
    
    # MLP Adapters
    adapter_hidden_dim: Optional[int] = None  # Default: 4x output_dim
    adapter_dropout: float = 0.1
    adapter_use_layer_norm: bool = True
    
    # =========================================================================
    # Prediction Heads
    # =========================================================================
    
    tie_prediction_weights: bool = False
    
    # =========================================================================
    # Pre-trained Model Paths
    # =========================================================================
    
    # Vision-Language Model (for SigLIP encoder + LLM)
    pretrained_model_path: str = "/root/autodl-tmp/InternVL/pretrained_models/InternVL2-8B"
    
    # VQ Tokenizer (from Janus or similar model)
    vq_tokenizer_path: str = "/root/autodl-tmp/Janus-main/deepseek-ai/Janus-1.3B"
    
    # Text Tokenizer
    text_tokenizer_path: Optional[str] = None  # Will use pretrained_model_path if None
    
    # =========================================================================
    # Freezing Configuration
    # =========================================================================
    
    freeze_siglip: bool = True
    freeze_vq: bool = True
    freeze_llm_decoder: bool = False
    freeze_mlp_adapters: bool = False
    freeze_level1: bool = False
    freeze_level2: bool = False
    
    # =========================================================================
    # Loss Configuration
    # =========================================================================
    
    loss_weight_understand: float = 0.7
    loss_weight_generate: float = 0.3
    label_smoothing: float = 0.0
    
    # =========================================================================
    # Training Configuration
    # =========================================================================
    
    # Data
    train_data_path: str = "/root/autodl-tmp/InternVL/internvl_chat/shell/data/cye2e_dual_image_finetune.json"
    val_data_path: Optional[str] = None
    max_seq_length: int = 4096
    
    # Optimizer
    learning_rate: float = 5e-7  # Base learning rate (following InternVL2 fine-tuning)
    weight_decay: float = 0.01  # Weight decay for regularization
    adam_beta1: float = 0.9  # Adam beta1 parameter
    adam_beta2: float = 0.999  # Adam beta2 parameter
    adam_epsilon: float = 1e-8  # Adam epsilon for numerical stability
    max_grad_norm: float = 1.0  # Maximum gradient norm for clipping
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"  # Options: "cosine", "linear", "constant", "cosine_with_restarts"
    warmup_ratio: float = 0.03  # Ratio of total steps for warmup (3% of training)
    warmup_steps: int = 0  # Explicit warmup steps (overrides warmup_ratio if set)
    min_lr_ratio: float = 0.1  # Minimum LR as ratio of initial LR for cosine decay
    
    # Advanced optimizer settings
    use_8bit_adam: bool = False  # Use 8-bit Adam for memory efficiency
    use_lion: bool = False  # Use Lion optimizer instead of AdamW
    
    # Training loop
    num_train_epochs: int = 4  # Total number of training epochs
    per_device_train_batch_size: int = 2  # Batch size per GPU for training
    per_device_eval_batch_size: int = 2  # Batch size per GPU for evaluation
    gradient_accumulation_steps: int = 4  # Number of steps to accumulate gradients
    max_steps: int = -1  # Maximum number of training steps (overrides epochs if set)
    
    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 250
    save_total_limit: int = 16
    output_dir: str = "./htfa_checkpoints"
    
    # Evaluation
    evaluation_strategy: str = "no"
    eval_steps: int = 500
    
    # Mixed precision training
    bf16: bool = True  # Use BF16 mixed precision (recommended for A100/H100)
    fp16: bool = False  # Use FP16 mixed precision (for V100 or older)
    fp16_opt_level: str = "O1"  # FP16 optimization level (O0, O1, O2, O3)
    
    # Gradient checkpointing for memory efficiency
    gradient_checkpointing: bool = True  # Enable gradient checkpointing to reduce memory
    gradient_checkpointing_kwargs: Dict[str, Any] = None  # Additional kwargs for checkpointing
    
    # Logging
    logging_steps: int = 1
    logging_dir: Optional[str] = None
    report_to: str = "tensorboard"
    
    # =========================================================================
    # DeepSpeed Configuration
    # =========================================================================
    
    use_deepspeed: bool = True
    deepspeed_config_path: str = "./config/deepspeed_config.json"
    
    # =========================================================================
    # Inference Configuration
    # =========================================================================
    
    generation_max_length: int = 512
    generation_temperature: float = 1.0
    generation_top_p: float = 0.9
    generation_top_k: int = 50
    
    # =========================================================================
    # Distributed Training
    # =========================================================================
    
    local_rank: int = -1  # Local rank for distributed training
    ddp_backend: str = "nccl"  # Backend for distributed training (nccl, gloo)
    ddp_find_unused_parameters: bool = False  # Find unused parameters in DDP
    
    # =========================================================================
    # Data Loading Optimization
    # =========================================================================
    
    dataloader_num_workers: int = 2  # Number of worker processes for data loading
    dataloader_pin_memory: bool = False  # Pin memory for faster GPU transfer
    dataloader_prefetch_factor: int = 2  # Number of batches to prefetch
    ignore_data_skip: bool = True  # Skip data loading errors and continue training
    
    # =========================================================================
    # Reproducibility & Debugging
    # =========================================================================
    
    seed: int = 42  # Random seed for reproducibility
    deterministic: bool = False  # Use deterministic algorithms (slower but reproducible)
    debug_mode: bool = False  # Enable debug mode with extra logging
    profile: bool = False  # Enable profiling for performance analysis
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def save_to_json(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f" Configuration saved to {filepath}")
    
    @classmethod
    def load_from_json(cls, filepath: str) -> 'HTFAConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def __repr__(self) -> str:
        """Pretty print configuration."""
        lines = ["=" * 80, "HTFA Configuration", "=" * 80]
        
        sections = {
            "Model Architecture": [
                "image_size", "spatial_feature_dim", "llm_hidden_dim", "vocab_size"
            ],
            "Level 1 - Adaptive Fusion": [
                "level1_learnable", "level1_init_weight", "level1_use_multi_scale"
            ],
            "Level 2 - Spatial Attention": [
                "level2_num_heads", "level2_use_affine", "level2_dropout"
            ],
            "Level 3 - Encoder Decoupling": [
                "vq_codebook_size", "vq_embedding_dim", "num_decoder_layers", "num_attention_heads"
            ],
            "Pre-trained Models": [
                "pretrained_model_path", "vq_tokenizer_path"
            ],
            "Freezing Configuration": [
                "freeze_siglip", "freeze_vq", "freeze_llm_decoder", "freeze_level1", "freeze_level2"
            ],
            "Loss Configuration": [
                "loss_weight_understand", "loss_weight_generate"
            ],
            "Training Configuration": [
                "learning_rate", "num_train_epochs", "per_device_train_batch_size",
                "gradient_accumulation_steps", "bf16"
            ]
        }
        
        for section_name, keys in sections.items():
            lines.append(f"\n{section_name}:")
            for key in keys:
                value = getattr(self, key, "N/A")
                if isinstance(value, str) and len(value) > 60:
                    value = "..." + value[-57:]
                lines.append(f"  {key:30s}: {value}")
        
        lines.append("=" * 80)
        return "\n".join(lines)


# Default configurations for different scenarios
def get_default_config() -> HTFAConfig:
    """Get default configuration for standard training."""
    return HTFAConfig()


def get_debug_config() -> HTFAConfig:
    """Get configuration for debugging (smaller model, faster iterations)."""
    config = HTFAConfig()
    config.num_decoder_layers = 2
    config.per_device_train_batch_size = 1
    config.gradient_accumulation_steps = 2
    config.save_steps = 10
    config.logging_steps = 1
    config.num_train_epochs = 1
    return config


def get_large_scale_config() -> HTFAConfig:
    """Get configuration for large-scale training."""
    config = HTFAConfig()
    config.num_decoder_layers = 8
    config.num_attention_heads = 32
    config.per_device_train_batch_size = 4
    config.gradient_accumulation_steps = 8
    config.num_train_epochs = 10
    config.save_steps = 500
    return config


def get_inference_config(checkpoint_path: str) -> HTFAConfig:
    """Get configuration for inference."""
    config = HTFAConfig()
    config.pretrained_model_path = checkpoint_path
    config.freeze_siglip = True
    config.freeze_vq = True
    config.freeze_llm_decoder = True
    config.freeze_level1 = True
    config.freeze_level2 = True
    return config


if __name__ == "__main__":
    print("Testing HTFA Configuration")
    print("=" * 80)
    
    # Test default config
    config = get_default_config()
    print(config)
    
    # Test saving/loading
    print("\n[Test] Saving and loading configuration")
    config.save_to_json("test_htfa_config.json")
    
    loaded_config = HTFAConfig.load_from_json("test_htfa_config.json")
    print(" Configuration saved and loaded successfully")
    
    # Test different configs
    print("\n[Debug Config]")
    debug_config = get_debug_config()
    print(f"Decoder layers: {debug_config.num_decoder_layers}")
    print(f"Batch size: {debug_config.per_device_train_batch_size}")
    
    print("\n[Large Scale Config]")
    large_config = get_large_scale_config()
    print(f"Decoder layers: {large_config.num_decoder_layers}")
    print(f"Attention heads: {large_config.num_attention_heads}")
    
    print("\n All configuration tests passed!")

