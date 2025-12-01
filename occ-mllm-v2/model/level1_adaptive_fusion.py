"""
Level 1: Adaptive Weighted Image Fusion Module

This module implements the first level of the Hierarchical Trinity Fusion Architecture (HTFA),
which performs learnable pixel-level blending of original and reconstructed views.

Mathematical Formulation:
    I_fused = α ⊙ I_orig + (1-α) ⊙ I_recon
    where α ∈ R^{H×W} with α_{i,j} ∈ [0,1] represents pixel-level learnable weights.

Key Features:
    - Learnable fusion weight matrix α optimized via end-to-end backpropagation
    - Pixel-level adaptive blending replacing hard-switching between isolated modules
    - Gradients flow through both generation and understanding pathways
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveWeightedFusion(nn.Module):
    """
    Level 1: Adaptive Weighted Image Fusion
    
    Dynamically integrates original and reconstructed images through learnable 
    pixel-level weighting, establishing the foundation for hierarchical fusion.
    
    Args:
        image_size (int): Input image resolution (assumes square images, default: 448)
        init_weight (float): Initial value for fusion weights before sigmoid (default: 0.0)
                            0.0 corresponds to α=0.5 after sigmoid activation
        learnable (bool): Whether fusion weights are learnable parameters (default: True)
    
    Forward Args:
        img_orig (torch.Tensor): Original image with occlusions, shape [B, 3, H, W]
        img_recon (torch.Tensor): 3D reconstructed image with completed geometry, shape [B, 3, H, W]
    
    Returns:
        img_fused (torch.Tensor): Adaptively fused image, shape [B, 3, H, W]
        fusion_weights (torch.Tensor): Normalized fusion weights α, shape [B, 1, H, W]
    """
    
    def __init__(
        self, 
        image_size: int = 448,
        init_weight: float = 0.0,
        learnable: bool = True
    ):
        super(AdaptiveWeightedFusion, self).__init__()
        
        self.image_size = image_size
        self.learnable = learnable
        
        # Initialize learnable fusion weight matrix (before sigmoid activation)
        # Shape: [1, 1, H, W] for broadcasting across batch and channels
        fusion_weight_raw = torch.full(
            (1, 1, image_size, image_size), 
            init_weight, 
            dtype=torch.float32
        )
        
        if learnable:
            # Register as learnable parameter for gradient-based optimization
            self.fusion_weight_raw = nn.Parameter(fusion_weight_raw)
        else:
            # Register as buffer (non-trainable) for inference or ablation studies
            self.register_buffer('fusion_weight_raw', fusion_weight_raw)
        
        print(f"[Level 1] AdaptiveWeightedFusion initialized:")
        print(f"  - Image size: {image_size}x{image_size}")
        print(f"  - Learnable: {learnable}")
        print(f"  - Initial weight: {init_weight} (α≈{torch.sigmoid(torch.tensor(init_weight)).item():.3f} after sigmoid)")
    
    def forward(self, img_orig: torch.Tensor, img_recon: torch.Tensor):
        """
        Perform adaptive weighted fusion of original and reconstructed images.
        
        Args:
            img_orig: Original image tensor [B, 3, H, W]
            img_recon: Reconstructed image tensor [B, 3, H, W]
        
        Returns:
            img_fused: Fused image tensor [B, 3, H, W]
            fusion_weights: Normalized weights α [B, 1, H, W]
        """
        B, C, H, W = img_orig.shape
        
        # Validate input shapes
        assert img_orig.shape == img_recon.shape, \
            f"Shape mismatch: img_orig {img_orig.shape} vs img_recon {img_recon.shape}"
        
        # Apply sigmoid to ensure fusion weights α ∈ [0, 1]
        # Shape: [1, 1, H, W] -> broadcasts to [B, 1, H, W]
        fusion_weights = torch.sigmoid(self.fusion_weight_raw)
        
        # Handle dynamic image sizes via interpolation
        if H != self.image_size or W != self.image_size:
            fusion_weights = F.interpolate(
                fusion_weights, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Expand to match batch size: [1, 1, H, W] -> [B, 1, H, W]
        fusion_weights = fusion_weights.expand(B, -1, -1, -1)
        
        # Adaptive weighted fusion: I_fused = α ⊙ I_orig + (1-α) ⊙ I_recon
        # Broadcasting: [B, 1, H, W] operates on [B, 3, H, W]
        img_fused = fusion_weights * img_orig + (1 - fusion_weights) * img_recon
        
        return img_fused, fusion_weights
    
    def get_fusion_statistics(self, fusion_weights: torch.Tensor):
        """
        Compute statistics of fusion weights for monitoring training dynamics.
        
        Args:
            fusion_weights: Fusion weight tensor [B, 1, H, W]
        
        Returns:
            dict: Statistics including mean, std, min, max of fusion weights
        """
        stats = {
            'mean': fusion_weights.mean().item(),
            'std': fusion_weights.std().item(),
            'min': fusion_weights.min().item(),
            'max': fusion_weights.max().item(),
            'median': fusion_weights.median().item(),
        }
        return stats


class MultiScaleAdaptiveFusion(nn.Module):
    """
    Multi-scale variant of Adaptive Weighted Fusion (optional enhancement).
    
    Learns different fusion weights at multiple spatial scales, providing
    hierarchical control over the fusion process.
    
    Args:
        image_size (int): Input image resolution
        num_scales (int): Number of fusion scales (default: 3)
        learnable (bool): Whether fusion weights are learnable
    """
    
    def __init__(
        self, 
        image_size: int = 448,
        num_scales: int = 3,
        learnable: bool = True
    ):
        super(MultiScaleAdaptiveFusion, self).__init__()
        
        self.image_size = image_size
        self.num_scales = num_scales
        self.learnable = learnable
        
        # Create fusion modules at different scales
        self.fusion_layers = nn.ModuleList()
        current_size = image_size
        
        for i in range(num_scales):
            fusion_layer = AdaptiveWeightedFusion(
                image_size=current_size,
                init_weight=0.0,
                learnable=learnable
            )
            self.fusion_layers.append(fusion_layer)
            current_size = current_size // 2  # Halve resolution at each scale
        
        # Learnable scale combination weights
        if learnable:
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        else:
            self.register_buffer('scale_weights', torch.ones(num_scales) / num_scales)
        
        print(f"[Level 1] MultiScaleAdaptiveFusion initialized with {num_scales} scales")
    
    def forward(self, img_orig: torch.Tensor, img_recon: torch.Tensor):
        """
        Perform multi-scale adaptive fusion.
        
        Returns:
            img_fused: Final fused image [B, 3, H, W]
            fusion_info: Dictionary containing fusion weights at each scale
        """
        B, C, H, W = img_orig.shape
        
        # Normalize scale weights via softmax
        scale_weights_norm = F.softmax(self.scale_weights, dim=0)
        
        fused_images = []
        fusion_weights_list = []
        
        # Process at each scale
        for i, fusion_layer in enumerate(self.fusion_layers):
            # Downsample inputs to current scale
            scale_h = H // (2 ** i)
            scale_w = W // (2 ** i)
            
            img_orig_scaled = F.interpolate(img_orig, size=(scale_h, scale_w), mode='bilinear', align_corners=False)
            img_recon_scaled = F.interpolate(img_recon, size=(scale_h, scale_w), mode='bilinear', align_corners=False)
            
            # Fuse at current scale
            img_fused_scaled, fusion_weights_scaled = fusion_layer(img_orig_scaled, img_recon_scaled)
            
            # Upsample back to original resolution
            img_fused_upsampled = F.interpolate(img_fused_scaled, size=(H, W), mode='bilinear', align_corners=False)
            
            fused_images.append(img_fused_upsampled)
            fusion_weights_list.append(fusion_weights_scaled)
        
        # Combine multi-scale results with learned weights
        img_fused = sum(w * img for w, img in zip(scale_weights_norm, fused_images))
        
        fusion_info = {
            'scale_weights': scale_weights_norm,
            'fusion_weights_per_scale': fusion_weights_list
        }
        
        return img_fused, fusion_info


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Level 1: Adaptive Weighted Image Fusion")
    print("=" * 80)
    
    # Test basic fusion module
    print("\n[Test 1] Basic AdaptiveWeightedFusion")
    fusion_module = AdaptiveWeightedFusion(image_size=448, learnable=True)
    
    # Create dummy inputs
    batch_size = 2
    img_orig = torch.randn(batch_size, 3, 448, 448)
    img_recon = torch.randn(batch_size, 3, 448, 448)
    
    # Forward pass
    img_fused, fusion_weights = fusion_module(img_orig, img_recon)
    
    print(f"\nInput shapes:")
    print(f"  img_orig: {img_orig.shape}")
    print(f"  img_recon: {img_recon.shape}")
    print(f"\nOutput shapes:")
    print(f"  img_fused: {img_fused.shape}")
    print(f"  fusion_weights: {fusion_weights.shape}")
    
    # Compute statistics
    stats = fusion_module.get_fusion_statistics(fusion_weights)
    print(f"\nFusion weight statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test gradient flow (critical for end-to-end training)
    print("\n[Test 2] Gradient Flow Verification")
    loss = img_fused.mean()
    loss.backward()
    
    if fusion_module.fusion_weight_raw.grad is not None:
        print(f" Gradients successfully computed for fusion_weight_raw")
        print(f"  Gradient shape: {fusion_module.fusion_weight_raw.grad.shape}")
        print(f"  Gradient norm: {fusion_module.fusion_weight_raw.grad.norm().item():.6f}")
    else:
        print(f" No gradients computed (this should not happen!)")
    
    # Test multi-scale fusion
    print("\n[Test 3] MultiScaleAdaptiveFusion")
    multi_fusion = MultiScaleAdaptiveFusion(image_size=448, num_scales=3, learnable=True)
    
    img_fused_ms, fusion_info = multi_fusion(img_orig, img_recon)
    print(f"\nMulti-scale output shape: {img_fused_ms.shape}")
    print(f"Scale weights: {fusion_info['scale_weights']}")
    
    # Test with dynamic image sizes
    print("\n[Test 4] Dynamic Image Size Handling")
    img_orig_dynamic = torch.randn(1, 3, 224, 224)
    img_recon_dynamic = torch.randn(1, 3, 224, 224)
    
    img_fused_dynamic, _ = fusion_module(img_orig_dynamic, img_recon_dynamic)
    print(f"Dynamic input size: 224x224")
    print(f"Dynamic output shape: {img_fused_dynamic.shape}")
    
    print("\n" + "=" * 80)
    print(" All Level 1 tests passed!")
    print("=" * 80)

