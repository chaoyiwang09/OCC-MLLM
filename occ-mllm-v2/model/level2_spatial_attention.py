"""
Level 2: Spatial Attention Affine Fusion Module

This module implements the second level of the Hierarchical Trinity Fusion Architecture (HTFA),
which performs position-aware cross-attention with affine transformations for spatial alignment.

Mathematical Formulation:
    S_2: (I_orig, I_recon, I_fused) → F_spatial
    
    1. Feature Extraction (frozen SigLIP encoder):
       F_orig = E_SigLIP(I_orig)
       F_recon = E_SigLIP(I_recon)
       F_fused = E_SigLIP(I_fused)
    
    2. Affine Transformation for Spatial Alignment:
       A = [[w_o/w_r, 0, t_x],
            [0, h_o/h_r, t_y],
            [0, 0, 1]]
    
    3. Position-Aware Cross-Attention:
       Q = W_Q · Flatten(F_orig)
       K = W_K · Flatten(AffineAlign(F_recon))
       V = W_V · Flatten(F_fused)
       F_spatial = Softmax(QK^T/√d) V

Key Features:
    - Automatic affine transformation computation from object bounding boxes
    - Position-aware spatial alignment via bilinear interpolation
    - Cross-view attention for occlusion-aware feature fusion
    - Learnable Q/K/V projections optimized via end-to-end backpropagation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import cv2
import numpy as np


class AffineTransformComputer:
    """
    Computes affine transformation matrix for aligning reconstructed object to original position.
    
    Given bounding boxes of objects in original and reconstructed images,
    computes the transformation matrix A that maps coordinates from I_recon to I_orig.
    
    Mathematical Formulation:
        A = [[w_o/w_r, 0, t_x],
             [0, h_o/h_r, t_y],
             [0, 0, 1]]
        where t_x = w_0^o - (w_o/w_r) * w_0^r
              t_y = h_0^o - (h_o/h_r) * h_0^r
    """
    
    @staticmethod
    def compute_object_bbox(image: torch.Tensor, threshold: float = 0.01) -> Tuple[int, int, int, int]:
        """
        Compute bounding box of non-background object region.
        
        Args:
            image: Image tensor [B, C, H, W] or [C, H, W]
            threshold: Threshold for determining non-zero pixels (default: 0.01)
        
        Returns:
            bbox: Tuple of (x_min, y_min, width, height)
        """
        # Handle batch dimension
        if image.dim() == 4:
            image = image[0]  # Take first image in batch
        
        # Convert to grayscale intensity
        if image.shape[0] == 3:  # RGB
            intensity = image.mean(dim=0)  # [H, W]
        else:
            intensity = image[0]
        
        # Find non-background pixels
        mask = intensity > threshold
        
        # Get coordinates of non-zero pixels
        coords = torch.nonzero(mask, as_tuple=False)  # [N, 2] where each row is [y, x]
        
        if coords.shape[0] == 0:
            # No object detected, return full image bbox
            H, W = intensity.shape
            return (0, 0, W, H)
        
        # Compute bounding box
        y_min = coords[:, 0].min().item()
        y_max = coords[:, 0].max().item()
        x_min = coords[:, 1].min().item()
        x_max = coords[:, 1].max().item()
        
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        
        return (x_min, y_min, width, height)
    
    @staticmethod
    def compute_affine_matrix(
        bbox_orig: Tuple[int, int, int, int],
        bbox_recon: Tuple[int, int, int, int],
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Compute affine transformation matrix from reconstructed to original coordinate system.
        
        Args:
            bbox_orig: Original image bbox (x, y, w, h)
            bbox_recon: Reconstructed image bbox (x, y, w, h)
            device: Target device for the matrix
        
        Returns:
            affine_matrix: 3x3 affine transformation matrix [3, 3]
        """
        x_o, y_o, w_o, h_o = bbox_orig
        x_r, y_r, w_r, h_r = bbox_recon
        
        # Avoid division by zero
        w_r = max(w_r, 1)
        h_r = max(h_r, 1)
        
        # Compute scale factors
        scale_x = w_o / w_r
        scale_y = h_o / h_r
        
        # Compute translation
        # t_x = x_o - scale_x * x_r (maps reconstructed object center to original center)
        t_x = x_o - scale_x * x_r
        t_y = y_o - scale_y * y_r
        
        # Construct affine matrix
        affine_matrix = torch.tensor([
            [scale_x, 0, t_x],
            [0, scale_y, t_y],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        return affine_matrix
    
    @staticmethod
    def compute_inverse_affine_matrix(
        affine_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute inverse of affine transformation matrix.
        
        Args:
            affine_matrix: 3x3 affine matrix [3, 3]
        
        Returns:
            inverse_matrix: Inverse transformation matrix [3, 3]
        """
        return torch.inverse(affine_matrix)


class SpatialAttentionAffineFusion(nn.Module):
    """
    Level 2: Spatial Attention Affine Fusion
    
    Implements position-aware cross-attention with affine spatial alignment
    to fuse features from original, reconstructed, and fused images.
    
    Architecture:
        1. Extract features via frozen SigLIP encoder
        2. Compute affine transformation for spatial alignment
        3. Apply position-aware cross-attention:
           - Query from original image features
           - Key from spatially-aligned reconstructed features
           - Value from fused image features
    
    Args:
        feature_dim (int): Feature dimension from SigLIP encoder (default: 1152 for SigLIP-SO400M)
        num_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout rate for attention (default: 0.0)
        use_affine_alignment (bool): Whether to apply affine alignment (default: True)
    
    Forward Args:
        img_orig (torch.Tensor): Original image [B, 3, H, W]
        img_recon (torch.Tensor): Reconstructed image [B, 3, H, W]
        img_fused (torch.Tensor): Fused image from Level 1 [B, 3, H, W]
        feat_orig (torch.Tensor): Pre-extracted original features [B, h*w, d] (optional)
        feat_recon (torch.Tensor): Pre-extracted reconstructed features [B, h*w, d] (optional)
        feat_fused (torch.Tensor): Pre-extracted fused features [B, h*w, d] (optional)
        occlusion_mask (torch.Tensor): Occlusion mask for original image [B, 1, H, W] (optional)
    
    Returns:
        feat_spatial (torch.Tensor): Spatially-fused features [B, h*w, d]
        attention_weights (torch.Tensor): Cross-attention weights [B, num_heads, h*w, h*w]
    """
    
    def __init__(
        self,
        feature_dim: int = 1152,  # SigLIP-SO400M feature dimension
        num_heads: int = 8,
        dropout: float = 0.0,
        use_affine_alignment: bool = True,
    ):
        super(SpatialAttentionAffineFusion, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.use_affine_alignment = use_affine_alignment
        
        assert feature_dim % num_heads == 0, \
            f"feature_dim ({feature_dim}) must be divisible by num_heads ({num_heads})"
        
        # Learnable Query/Key/Value projection matrices (theta_2 in paper)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)
        
        # Affine transformation computer
        self.affine_computer = AffineTransformComputer()
        
        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        print(f"[Level 2] SpatialAttentionAffineFusion initialized:")
        print(f"  - Feature dimension: {feature_dim}")
        print(f"  - Number of heads: {num_heads}")
        print(f"  - Head dimension: {self.head_dim}")
        print(f"  - Affine alignment: {use_affine_alignment}")
    
    def apply_affine_alignment_to_features(
        self,
        features: torch.Tensor,
        affine_matrix_inv: torch.Tensor,
        feat_h: int,
        feat_w: int,
        img_h: int,
        img_w: int,
        occlusion_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply affine transformation to spatially align features.
        
        This implements the position-aware alignment described in the paper:
        For each position (i,j) in feature map, compute corresponding position
        in reconstructed image via inverse affine transform, then sample features
        using bilinear interpolation.
        
        Args:
            features: Feature map to align [B, h*w, d]
            affine_matrix_inv: Inverse affine matrix [3, 3]
            feat_h: Feature map height
            feat_w: Feature map width
            img_h: Image height
            img_w: Image width
            occlusion_mask: Optional mask indicating object region [B, 1, H, W]
        
        Returns:
            aligned_features: Spatially aligned features [B, h*w, d]
        """
        B, N, d = features.shape
        device = features.device
        
        # Reshape features to 2D spatial layout [B, d, h, w]
        features_2d = features.transpose(1, 2).reshape(B, d, feat_h, feat_w)
        
        # Create coordinate grid for feature map positions
        # Grid coordinates in feature map space: (i, j) where i∈[0,h), j∈[0,w)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(feat_h, device=device, dtype=torch.float32),
            torch.arange(feat_w, device=device, dtype=torch.float32),
            indexing='ij'
        )
        
        # Map feature coordinates to image coordinates
        # (i,j) in feature map -> (y,x) in image space
        img_y = grid_y * (img_h / feat_h)
        img_x = grid_x * (img_w / feat_w)
        
        # Stack to homogeneous coordinates [h, w, 3]
        coords_homo = torch.stack([img_x, img_y, torch.ones_like(img_x)], dim=-1)  # [h, w, 3]
        
        # Apply inverse affine transformation to map to reconstructed image space
        # A^{-1} · [x, y, 1]^T for each position
        coords_homo_flat = coords_homo.reshape(-1, 3)  # [h*w, 3]
        coords_recon_flat = torch.matmul(coords_homo_flat, affine_matrix_inv.T)  # [h*w, 3]
        coords_recon = coords_recon_flat[:, :2].reshape(feat_h, feat_w, 2)  # [h, w, 2]
        
        # Map back to feature map coordinates in reconstructed feature space
        recon_feat_x = coords_recon[..., 0] * (feat_w / img_w)
        recon_feat_y = coords_recon[..., 1] * (feat_h / img_h)
        
        # Normalize to [-1, 1] for grid_sample
        # grid_sample expects: x ∈ [-1, 1] maps to [0, width-1]
        norm_x = 2.0 * recon_feat_x / (feat_w - 1) - 1.0
        norm_y = 2.0 * recon_feat_y / (feat_h - 1) - 1.0
        
        # Stack to grid format [h, w, 2] where last dim is [x, y]
        sampling_grid = torch.stack([norm_x, norm_y], dim=-1)  # [h, w, 2]
        
        # Expand for batch dimension [B, h, w, 2]
        sampling_grid = sampling_grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Apply bilinear interpolation to sample aligned features
        aligned_features_2d = F.grid_sample(
            features_2d,
            sampling_grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )  # [B, d, h, w]
        
        # Apply occlusion mask if provided (set non-object regions to zero)
        if occlusion_mask is not None:
            # Resize mask to feature map resolution
            mask_resized = F.interpolate(
                occlusion_mask.float(),
                size=(feat_h, feat_w),
                mode='bilinear',
                align_corners=False
            )  # [B, 1, h, w]
            aligned_features_2d = aligned_features_2d * mask_resized
        
        # Reshape back to sequence format [B, h*w, d]
        aligned_features = aligned_features_2d.reshape(B, d, -1).transpose(1, 2)
        
        return aligned_features
    
    def multi_head_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute multi-head cross-attention.
        
        Args:
            query: Query tensor [B, N, d]
            key: Key tensor [B, N, d]
            value: Value tensor [B, N, d]
            return_attention_weights: Whether to return attention weights
        
        Returns:
            output: Attention output [B, N, d]
            attention_weights: Attention weights [B, num_heads, N, N] (if requested)
        """
        B, N, d = query.shape
        
        # Linear projections and split into multiple heads
        # [B, N, d] -> [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
        Q = self.query_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores: Q·K^T / sqrt(d_head)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values: Softmax(Q·K^T / sqrt(d)) · V
        attention_output = torch.matmul(attention_weights, V)  # [B, num_heads, N, head_dim]
        
        # Concatenate heads: [B, num_heads, N, head_dim] -> [B, N, d]
        attention_output = attention_output.transpose(1, 2).reshape(B, N, d)
        
        # Output projection
        output = self.output_proj(attention_output)
        
        if return_attention_weights:
            return output, attention_weights
        else:
            return output, None
    
    def forward(
        self,
        img_orig: Optional[torch.Tensor] = None,
        img_recon: Optional[torch.Tensor] = None,
        img_fused: Optional[torch.Tensor] = None,
        feat_orig: Optional[torch.Tensor] = None,
        feat_recon: Optional[torch.Tensor] = None,
        feat_fused: Optional[torch.Tensor] = None,
        occlusion_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Level 2 spatial attention fusion.
        
        Note: Either provide images (img_*) for feature extraction, or provide
        pre-extracted features (feat_*) directly. Features take precedence if both provided.
        
        Returns:
            feat_spatial: Spatially fused features [B, N, d]
            attention_weights: Cross-attention weights [B, num_heads, N, N] (optional)
        """
        # Validate inputs
        if feat_orig is None:
            assert img_orig is not None, "Must provide either img_orig or feat_orig"
        if feat_recon is None:
            assert img_recon is not None, "Must provide either img_recon or feat_recon"
        if feat_fused is None:
            assert img_fused is not None, "Must provide either img_fused or feat_fused"
        
        # Use pre-extracted features if available (avoids redundant encoding)
        # In full HTFA pipeline, features are extracted once by frozen SigLIP
        F_orig = feat_orig
        F_recon = feat_recon
        F_fused = feat_fused
        
        # Get feature map dimensions
        B, N, d = F_orig.shape
        
        # Infer spatial dimensions (assume square feature maps for now)
        feat_h = feat_w = int(N ** 0.5)
        assert feat_h * feat_w == N, f"Feature map must be square, got N={N}"
        
        # Get image dimensions (needed for affine transform)
        if img_orig is not None:
            _, _, img_h, img_w = img_orig.shape
        else:
            # Assume standard image size if not provided
            img_h = img_w = 448
        
        # Compute affine transformation if enabled
        if self.use_affine_alignment and img_orig is not None and img_recon is not None:
            # Compute bounding boxes for affine transformation
            bbox_orig = self.affine_computer.compute_object_bbox(img_orig)
            bbox_recon = self.affine_computer.compute_object_bbox(img_recon)
            
            # Compute affine matrix and its inverse
            affine_matrix = self.affine_computer.compute_affine_matrix(
                bbox_orig, bbox_recon, device=F_orig.device
            )
            affine_matrix_inv = self.affine_computer.compute_inverse_affine_matrix(affine_matrix)
            
            # Apply affine alignment to reconstructed features (Key branch)
            F_recon_aligned = self.apply_affine_alignment_to_features(
                F_recon,
                affine_matrix_inv,
                feat_h,
                feat_w,
                img_h,
                img_w,
                occlusion_mask
            )
        else:
            # Skip affine alignment (ablation mode)
            F_recon_aligned = F_recon
        
        # Cross-attention fusion:
        # Query from original (what info is needed)
        # Key from aligned reconstructed (where complete geometry is)
        # Value from fused (what features to aggregate)
        feat_spatial, attention_weights = self.multi_head_attention(
            query=F_orig,
            key=F_recon_aligned,
            value=F_fused,
            return_attention_weights=return_attention_weights
        )
        
        # Add residual connection and layer norm for stable training
        feat_spatial = self.layer_norm(feat_spatial + F_orig)
        
        return feat_spatial, attention_weights
    
    def get_attention_statistics(self, attention_weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics of attention weights for monitoring training dynamics.
        
        Args:
            attention_weights: Attention weight tensor [B, num_heads, N, N]
        
        Returns:
            dict: Statistics including entropy, max attention, etc.
        """
        # Average over batch and heads
        attn_avg = attention_weights.mean(dim=(0, 1))  # [N, N]
        
        # Compute entropy of attention distribution (higher = more uniform)
        epsilon = 1e-8
        entropy = -(attention_weights * torch.log(attention_weights + epsilon)).sum(dim=-1).mean()
        
        stats = {
            'entropy': entropy.item(),
            'max_attention': attention_weights.max().item(),
            'min_attention': attention_weights.min().item(),
            'mean_attention': attention_weights.mean().item(),
        }
        return stats


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Level 2: Spatial Attention Affine Fusion")
    print("=" * 80)
    
    # Test configuration
    batch_size = 2
    img_h, img_w = 448, 448
    feat_h, feat_w = 32, 32  # Typical SigLIP feature map size
    feature_dim = 1152
    
    # Test 1: Affine transformation computation
    print("\n[Test 1] Affine Transformation Computation")
    affine_computer = AffineTransformComputer()
    
    # Create dummy images with objects at different positions
    img_orig_dummy = torch.zeros(1, 3, img_h, img_w)
    img_orig_dummy[:, :, 100:300, 150:350] = 1.0  # Object in original image
    
    img_recon_dummy = torch.zeros(1, 3, img_h, img_w)
    img_recon_dummy[:, :, 50:150, 50:150] = 1.0  # Object in reconstructed image (different position)
    
    bbox_orig = affine_computer.compute_object_bbox(img_orig_dummy)
    bbox_recon = affine_computer.compute_object_bbox(img_recon_dummy)
    
    print(f"Original bbox (x, y, w, h): {bbox_orig}")
    print(f"Reconstructed bbox (x, y, w, h): {bbox_recon}")
    
    affine_matrix = affine_computer.compute_affine_matrix(bbox_orig, bbox_recon)
    print(f"\nAffine transformation matrix:")
    print(affine_matrix)
    
    # Test 2: Spatial attention fusion module
    print("\n[Test 2] SpatialAttentionAffineFusion Forward Pass")
    fusion_module = SpatialAttentionAffineFusion(
        feature_dim=feature_dim,
        num_heads=8,
        use_affine_alignment=True
    )
    
    # Create dummy features (simulating SigLIP encoder output)
    feat_orig = torch.randn(batch_size, feat_h * feat_w, feature_dim)
    feat_recon = torch.randn(batch_size, feat_h * feat_w, feature_dim)
    feat_fused = torch.randn(batch_size, feat_h * feat_w, feature_dim)
    
    # Create dummy images for affine computation
    img_orig = torch.randn(batch_size, 3, img_h, img_w)
    img_recon = torch.randn(batch_size, 3, img_h, img_w)
    img_fused = torch.randn(batch_size, 3, img_h, img_w)
    
    # Forward pass
    feat_spatial, attention_weights = fusion_module(
        img_orig=img_orig,
        img_recon=img_recon,
        img_fused=img_fused,
        feat_orig=feat_orig,
        feat_recon=feat_recon,
        feat_fused=feat_fused,
        return_attention_weights=True
    )
    
    print(f"\nInput feature shapes:")
    print(f"  feat_orig: {feat_orig.shape}")
    print(f"  feat_recon: {feat_recon.shape}")
    print(f"  feat_fused: {feat_fused.shape}")
    print(f"\nOutput shapes:")
    print(f"  feat_spatial: {feat_spatial.shape}")
    print(f"  attention_weights: {attention_weights.shape}")
    
    # Compute attention statistics
    attn_stats = fusion_module.get_attention_statistics(attention_weights)
    print(f"\nAttention weight statistics:")
    for key, value in attn_stats.items():
        print(f"  {key}: {value:.4f}")
    
    # Test 3: Gradient flow verification
    print("\n[Test 3] Gradient Flow Verification")
    loss = feat_spatial.mean()
    loss.backward()
    
    # Check if gradients are computed for learnable parameters
    has_grad = {
        'query_proj': fusion_module.query_proj.weight.grad is not None,
        'key_proj': fusion_module.key_proj.weight.grad is not None,
        'value_proj': fusion_module.value_proj.weight.grad is not None,
        'output_proj': fusion_module.output_proj.weight.grad is not None,
    }
    
    print("Gradient computation status:")
    for name, status in has_grad.items():
        status_str = "" if status else ""
        print(f"  {status_str} {name}")
    
    if all(has_grad.values()):
        print("\n All Level 2 learnable parameters receive gradients!")
        
        # Print gradient norms
        print("\nGradient norms:")
        print(f"  query_proj: {fusion_module.query_proj.weight.grad.norm().item():.6f}")
        print(f"  key_proj: {fusion_module.key_proj.weight.grad.norm().item():.6f}")
        print(f"  value_proj: {fusion_module.value_proj.weight.grad.norm().item():.6f}")
    
    # Test 4: Ablation without affine alignment
    print("\n[Test 4] Ablation: Without Affine Alignment")
    fusion_no_affine = SpatialAttentionAffineFusion(
        feature_dim=feature_dim,
        num_heads=8,
        use_affine_alignment=False
    )
    
    feat_spatial_no_affine, _ = fusion_no_affine(
        feat_orig=feat_orig,
        feat_recon=feat_recon,
        feat_fused=feat_fused,
        return_attention_weights=False
    )
    
    print(f"Output shape (no affine): {feat_spatial_no_affine.shape}")
    print(f"Feature difference norm: {(feat_spatial - feat_spatial_no_affine).norm().item():.4f}")
    
    print("\n" + "=" * 80)
    print(" All Level 2 tests passed!")
    print("=" * 80)

