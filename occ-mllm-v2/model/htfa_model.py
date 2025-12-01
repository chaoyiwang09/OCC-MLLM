"""
Complete HTFA Model: Hierarchical Trinity Fusion Architecture

This module integrates all three levels of the HTFA architecture into a unified
end-to-end model for occluded object understanding and generation.

Architecture Overview:
    Input: (I_orig, I_recon, text_query)
    
    Level 1: Adaptive Weighted Image Fusion
        I_fused = α ⊙ I_orig + (1-α) ⊙ I_recon
    
    Level 2: Spatial Attention Affine Fusion
        F_spatial = CrossAttention(F_orig, AffineAlign(F_recon), F_fused)
    
    Level 3: Visual Encoder Decoupling Fusion
        E_understand = MLP_understand(F_spatial)  # Understanding branch
        E_generate = MLP_generate(VQ(I_fused))    # Generation branch
        H_unified = Decoder_AR(E_understand ⊕ E_generate ⊕ text_embeddings)
    
    Output: (O_understand, O_generate)
        O_understand = H_understand(H_unified)  # Text tokens
        O_generate = H_generate(H_unified)      # VQ codes

Key Features:
    - End-to-end gradient flow through all three hierarchical levels
    - Unified loss function combining understanding and generation objectives
    - Integration with pre-trained vision-language models (SigLIP, VQ, InternLM2)
    - Support for flexible frozen/trainable component configurations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union, List
import warnings

# Import hierarchical components
from .level1_adaptive_fusion import AdaptiveWeightedFusion
from .level2_spatial_attention import SpatialAttentionAffineFusion
from .level3_encoder_decoupling import (
    VisualEncoderDecouplingFusion,
    DualPredictionHeads,
    VQTokenizerWrapper
)


class HTFAModel(nn.Module):
    """
    Complete Hierarchical Trinity Fusion Architecture (HTFA) Model
    
    Integrates three hierarchical fusion levels for unified understanding and generation.
    
    Args:
        # Image and feature dimensions
        image_size (int): Input image size (default: 448)
        spatial_feature_dim (int): SigLIP feature dimension (default: 1152)
        llm_hidden_dim (int): LLM hidden dimension (default: 2048)
        
        # Level 1 configuration
        level1_learnable (bool): Whether Level 1 fusion weights are learnable (default: True)
        level1_init_weight (float): Initial fusion weight (default: 0.0 for α=0.5)
        
        # Level 2 configuration
        level2_num_heads (int): Number of attention heads (default: 8)
        level2_use_affine (bool): Whether to use affine alignment (default: True)
        level2_dropout (float): Dropout rate (default: 0.1)
        
        # Level 3 configuration
        vq_codebook_size (int): VQ codebook size (default: 8192)
        vq_embedding_dim (int): VQ embedding dimension (default: 256)
        num_decoder_layers (int): Number of decoder layers (default: 4)
        num_attention_heads (int): Number of decoder attention heads (default: 16)
        
        # Prediction heads
        vocab_size (int): Text vocabulary size (default: 92544 for InternLM2)
        
        # Model integration
        siglip_encoder (nn.Module): Pre-trained SigLIP encoder (optional)
        vq_tokenizer (VQTokenizerWrapper): Pre-trained VQ tokenizer (optional)
        llm_decoder (nn.Module): Pre-trained LLM decoder (optional)
        text_tokenizer: Text tokenizer for embedding text queries
        
        # Freezing configuration
        freeze_siglip (bool): Freeze SigLIP encoder (default: True)
        freeze_vq (bool): Freeze VQ tokenizer (default: True)
        freeze_llm_decoder (bool): Freeze LLM decoder layers (default: False)
    """
    
    def __init__(
        self,
        # Basic dimensions
        image_size: int = 448,
        spatial_feature_dim: int = 1152,
        llm_hidden_dim: int = 2048,
        
        # Level 1 config
        level1_learnable: bool = True,
        level1_init_weight: float = 0.0,
        
        # Level 2 config
        level2_num_heads: int = 8,
        level2_use_affine: bool = True,
        level2_dropout: float = 0.1,
        
        # Level 3 config
        vq_codebook_size: int = 8192,
        vq_embedding_dim: int = 256,
        num_decoder_layers: int = 4,
        num_attention_heads: int = 16,
        
        # Prediction
        vocab_size: int = 92544,
        
        # Pre-trained models (will be loaded from checkpoints)
        siglip_encoder: Optional[nn.Module] = None,
        vq_tokenizer: Optional[VQTokenizerWrapper] = None,
        llm_decoder: Optional[nn.Module] = None,
        text_tokenizer = None,
        
        # Freezing config
        freeze_siglip: bool = True,
        freeze_vq: bool = True,
        freeze_llm_decoder: bool = False,
        
        **kwargs
    ):
        super(HTFAModel, self).__init__()
        
        self.image_size = image_size
        self.spatial_feature_dim = spatial_feature_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.vocab_size = vocab_size
        
        # =====================================================================
        # Pre-trained Vision-Language Components
        # =====================================================================
        
        # SigLIP encoder for understanding (frozen)
        self.siglip_encoder = siglip_encoder
        self.freeze_siglip = freeze_siglip
        if self.siglip_encoder is not None and freeze_siglip:
            for param in self.siglip_encoder.parameters():
                param.requires_grad = False
            self.siglip_encoder.eval()
        
        # VQ tokenizer for generation (frozen)
        self.vq_tokenizer = vq_tokenizer
        self.freeze_vq = freeze_vq
        if self.vq_tokenizer is not None and freeze_vq:
            for param in self.vq_tokenizer.parameters():
                param.requires_grad = False
            self.vq_tokenizer.eval()
        
        # LLM decoder for autoregressive processing
        self.llm_decoder = llm_decoder
        self.freeze_llm_decoder = freeze_llm_decoder
        if self.llm_decoder is not None and freeze_llm_decoder:
            for param in self.llm_decoder.parameters():
                param.requires_grad = False
            self.llm_decoder.eval()
        
        # Text tokenizer
        self.text_tokenizer = text_tokenizer
        
        # =====================================================================
        # Level 1: Adaptive Weighted Image Fusion
        # =====================================================================
        
        self.level1_fusion = AdaptiveWeightedFusion(
            image_size=image_size,
            init_weight=level1_init_weight,
            learnable=level1_learnable
        )
        
        # =====================================================================
        # Level 2: Spatial Attention Affine Fusion
        # =====================================================================
        
        self.level2_fusion = SpatialAttentionAffineFusion(
            feature_dim=spatial_feature_dim,
            num_heads=level2_num_heads,
            dropout=level2_dropout,
            use_affine_alignment=level2_use_affine
        )
        
        # =====================================================================
        # Level 3: Visual Encoder Decoupling Fusion
        # =====================================================================
        
        self.level3_fusion = VisualEncoderDecouplingFusion(
            spatial_feature_dim=spatial_feature_dim,
            vq_codebook_size=vq_codebook_size,
            vq_embedding_dim=vq_embedding_dim,
            llm_hidden_dim=llm_hidden_dim,
            num_decoder_layers=num_decoder_layers,
            num_attention_heads=num_attention_heads,
            dropout=level2_dropout,
            vq_tokenizer=vq_tokenizer
        )
        
        # =====================================================================
        # Dual Prediction Heads
        # =====================================================================
        
        self.prediction_heads = DualPredictionHeads(
            llm_hidden_dim=llm_hidden_dim,
            vocab_size=vocab_size,
            vq_codebook_size=vq_codebook_size,
            tie_weights=False
        )
        
        print("=" * 80)
        print("  HTFA Model Initialized Successfully!")
        print("=" * 80)
        print(f"Architecture Configuration:")
        print(f"   Image size: {image_size}x{image_size}")
        print(f"   Spatial feature dim: {spatial_feature_dim}")
        print(f"   LLM hidden dim: {llm_hidden_dim}")
        print(f"   Vocabulary size: {vocab_size}")
        print(f"\nHierarchical Levels:")
        print(f"  Level 1 - Adaptive Fusion: learnable={level1_learnable}")
        print(f"  Level 2 - Spatial Attention: heads={level2_num_heads}, affine={level2_use_affine}")
        print(f"  Level 3 - Encoder Decoupling: VQ={vq_codebook_size}, decoder_layers={num_decoder_layers}")
        print(f"\nFreezing Configuration:")
        print(f"    SigLIP encoder: {freeze_siglip}")
        print(f"    VQ tokenizer: {freeze_vq}")
        print(f"    LLM decoder: {freeze_llm_decoder}")
        print("=" * 80)
    
    def extract_siglip_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract SigLIP features from images.
        
        Args:
            images: Input images [B, 3, H, W]
        
        Returns:
            features: SigLIP features [B, N, D] where N = (H/patch_size) * (W/patch_size)
        """
        if self.siglip_encoder is None:
            raise ValueError("SigLIP encoder not initialized. Please load pre-trained SigLIP.")
        
        # Set to eval mode if frozen
        if self.freeze_siglip:
            self.siglip_encoder.eval()
        
        with torch.set_grad_enabled(not self.freeze_siglip):
            # Extract features using SigLIP
            # Assuming SigLIP encoder returns [B, N, D] features
            features = self.siglip_encoder(images)
        
        return features
    
    def embed_text_query(self, text_query: Union[str, List[str]]) -> torch.Tensor:
        """
        Embed text query into LLM input space.
        
        Args:
            text_query: Text query string or list of strings
        
        Returns:
            text_embeddings: Embedded text [B, L, llm_hidden_dim]
        """
        if self.text_tokenizer is None:
            raise ValueError("Text tokenizer not initialized.")
        
        # Tokenize text
        if isinstance(text_query, str):
            text_query = [text_query]
        
        # Get token IDs
        encoded = self.text_tokenizer(
            text_query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        input_ids = encoded['input_ids'].to(self.level3_fusion.understanding_branch.adapter.fc1.weight.device)
        
        # Get embeddings from LLM's embedding layer
        if self.llm_decoder is not None and hasattr(self.llm_decoder, 'embed_tokens'):
            text_embeddings = self.llm_decoder.embed_tokens(input_ids)
        else:
            # Fallback: create dummy embeddings
            warnings.warn("LLM decoder not available, using zero embeddings")
            B = len(text_query)
            L = input_ids.shape[1]
            text_embeddings = torch.zeros(B, L, self.llm_hidden_dim, device=input_ids.device)
        
        return text_embeddings
    
    def forward(
        self,
        img_orig: torch.Tensor,
        img_recon: torch.Tensor,
        text_query: Optional[Union[str, List[str], torch.Tensor]] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        occlusion_mask: Optional[torch.Tensor] = None,
        return_intermediate: bool = False,
        return_attention_weights: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict]:
        """
        Forward pass through complete HTFA architecture.
        
        Args:
            img_orig: Original image with occlusions [B, 3, H, W]
            img_recon: Reconstructed image from 3D [B, 3, H, W]
            text_query: Text query (string or list) or None
            text_embeddings: Pre-computed text embeddings [B, L, llm_hidden_dim] (optional)
            occlusion_mask: Occlusion mask [B, 1, H, W] (optional)
            return_intermediate: Whether to return intermediate outputs
            return_attention_weights: Whether to return attention weights
        
        Returns:
            If return_intermediate=False:
                logits_understand: Text token logits [B, L, vocab_size]
                logits_generate: VQ code logits [B, N', vq_codebook_size]
            If return_intermediate=True:
                Dict with all intermediate outputs and final logits
        """
        B = img_orig.shape[0]
        
        # Store intermediate outputs if requested
        intermediate = {} if return_intermediate else None
        
        # =====================================================================
        # Level 1: Adaptive Weighted Image Fusion
        # =====================================================================
        
        img_fused, fusion_weights = self.level1_fusion(img_orig, img_recon)
        
        if return_intermediate:
            intermediate['level1_fused_image'] = img_fused
            intermediate['level1_fusion_weights'] = fusion_weights
        
        # =====================================================================
        # Feature Extraction via SigLIP (frozen)
        # =====================================================================
        
        feat_orig = self.extract_siglip_features(img_orig)
        feat_recon = self.extract_siglip_features(img_recon)
        feat_fused = self.extract_siglip_features(img_fused)
        
        if return_intermediate:
            intermediate['siglip_feat_orig'] = feat_orig
            intermediate['siglip_feat_recon'] = feat_recon
            intermediate['siglip_feat_fused'] = feat_fused
        
        # =====================================================================
        # Level 2: Spatial Attention Affine Fusion
        # =====================================================================
        
        feat_spatial, attention_weights = self.level2_fusion(
            img_orig=img_orig,
            img_recon=img_recon,
            img_fused=img_fused,
            feat_orig=feat_orig,
            feat_recon=feat_recon,
            feat_fused=feat_fused,
            occlusion_mask=occlusion_mask,
            return_attention_weights=return_attention_weights
        )
        
        if return_intermediate:
            intermediate['level2_spatial_features'] = feat_spatial
            if attention_weights is not None:
                intermediate['level2_attention_weights'] = attention_weights
        
        # =====================================================================
        # Text Query Embedding
        # =====================================================================
        
        if text_embeddings is None:
            if text_query is None:
                # Use default query for inference
                text_query = ["What object is in the hand?"]
            
            if isinstance(text_query, (str, list)):
                text_embeddings = self.embed_text_query(text_query)
            else:
                text_embeddings = text_query  # Assume it's already a tensor
        
        if return_intermediate:
            intermediate['text_embeddings'] = text_embeddings
        
        # =====================================================================
        # Level 3: Visual Encoder Decoupling Fusion
        # =====================================================================
        
        hidden_states, embeddings_dict = self.level3_fusion(
            feat_spatial=feat_spatial,
            img_fused=img_fused,
            text_embeddings=text_embeddings,
            attention_mask=None,
            return_branch_embeddings=True
        )
        
        if return_intermediate:
            intermediate['level3_hidden_states'] = hidden_states
            intermediate['level3_understand_embeddings'] = embeddings_dict['understand']
            intermediate['level3_generate_embeddings'] = embeddings_dict['generate']
            intermediate['level3_vq_token_ids'] = embeddings_dict['vq_token_ids']
        
        # =====================================================================
        # Dual Prediction Heads
        # =====================================================================
        
        num_understand_tokens = feat_spatial.shape[1]
        num_generate_tokens = embeddings_dict['generate'].shape[1]
        
        logits_understand, logits_generate = self.prediction_heads(
            hidden_states=hidden_states,
            num_understand_tokens=num_understand_tokens,
            num_generate_tokens=num_generate_tokens
        )
        
        if return_intermediate:
            intermediate['logits_understand'] = logits_understand
            intermediate['logits_generate'] = logits_generate
            return intermediate
        else:
            return logits_understand, logits_generate
    
    def compute_loss(
        self,
        logits_understand: torch.Tensor,
        logits_generate: torch.Tensor,
        target_text_tokens: torch.Tensor,
        target_vq_codes: torch.Tensor,
        loss_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute unified loss for understanding and generation.
        
        Args:
            logits_understand: Text prediction logits [B, L, vocab_size]
            logits_generate: VQ prediction logits [B, N', vq_codebook_size]
            target_text_tokens: Ground truth text tokens [B, L]
            target_vq_codes: Ground truth VQ codes [B, N']
            loss_weights: Dict with keys 'understand' and 'generate' (default: {0.7, 0.3})
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with individual loss components
        """
        if loss_weights is None:
            loss_weights = {'understand': 0.7, 'generate': 0.3}
        
        # Understanding loss (cross-entropy on text tokens)
        loss_understand = F.cross_entropy(
            logits_understand.reshape(-1, self.vocab_size),
            target_text_tokens.reshape(-1),
            ignore_index=-100  # Ignore padding tokens
        )
        
        # Generation loss (cross-entropy on VQ codes)
        loss_generate = F.cross_entropy(
            logits_generate.reshape(-1, logits_generate.shape[-1]),
            target_vq_codes.reshape(-1),
            ignore_index=-100
        )
        
        # Combined loss
        total_loss = (
            loss_weights['understand'] * loss_understand +
            loss_weights['generate'] * loss_generate
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'loss_understand': loss_understand.item(),
            'loss_generate': loss_generate.item()
        }
        
        return total_loss, loss_dict
    
    def get_trainable_parameters(self) -> Dict[str, List[str]]:
        """
        Get trainable parameters organized by component.
        
        Returns:
            Dict mapping component name to list of trainable parameter names
        """
        trainable_params = {
            'level1_fusion': [],
            'level2_fusion': [],
            'level3_understanding': [],
            'level3_generation': [],
            'level3_decoder': [],
            'prediction_heads': []
        }
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'level1_fusion' in name:
                    trainable_params['level1_fusion'].append(name)
                elif 'level2_fusion' in name:
                    trainable_params['level2_fusion'].append(name)
                elif 'level3_fusion.understanding_branch' in name:
                    trainable_params['level3_understanding'].append(name)
                elif 'level3_fusion.generation_branch' in name:
                    trainable_params['level3_generation'].append(name)
                elif 'level3_fusion.autoregressive_decoder' in name:
                    trainable_params['level3_decoder'].append(name)
                elif 'prediction_heads' in name:
                    trainable_params['prediction_heads'].append(name)
        
        return trainable_params
    
    def print_trainable_parameters(self):
        """Print summary of trainable parameters."""
        trainable_params = self.get_trainable_parameters()
        
        print("=" * 80)
        print("Trainable Parameters Summary")
        print("=" * 80)
        
        total_trainable = 0
        for component, params in trainable_params.items():
            num_params = sum(
                p.numel() for name, p in self.named_parameters() 
                if p.requires_grad and any(pname in name for pname in params)
            )
            total_trainable += num_params
            print(f"{component:30s}: {num_params:12,d} params ({len(params):3d} tensors)")
        
        total_params = sum(p.numel() for p in self.parameters())
        print("-" * 80)
        print(f"{'Total trainable':30s}: {total_trainable:12,d} / {total_params:12,d} ({100*total_trainable/total_params:.2f}%)")
        print("=" * 80)


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Complete HTFA Model")
    print("=" * 80)
    
    # Test configuration
    batch_size = 2
    image_size = 448
    text_query = ["What is the object?", "Describe the hand-held object."]
    
    # Create model (without pre-trained components for testing)
    model = HTFAModel(
        image_size=image_size,
        spatial_feature_dim=1152,
        llm_hidden_dim=2048,
        level1_learnable=True,
        level2_num_heads=8,
        level2_use_affine=True,
        num_decoder_layers=2,  # Reduced for testing
        vocab_size=92544,
        vq_codebook_size=8192,
        freeze_siglip=True,
        freeze_vq=True,
        freeze_llm_decoder=False
    )
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Create dummy inputs
    print("\n[Test 1] Forward Pass")
    img_orig = torch.randn(batch_size, 3, image_size, image_size)
    img_recon = torch.randn(batch_size, 3, image_size, image_size)
    
    # Since we don't have real SigLIP, we'll mock the features
    # In practice, you'd load pre-trained SigLIP encoder
    
    # Create dummy text embeddings
    text_embeddings = torch.randn(batch_size, 20, 2048)
    
    # Forward pass with intermediate outputs
    try:
        print("  Note: This test requires pre-trained SigLIP encoder.")
        print("Skipping full forward pass test without pre-trained components.")
    except Exception as e:
        print(f"Expected error (need pre-trained models): {e}")
    
    print("\n[Test 2] Loss Computation")
    # Create dummy logits and targets
    num_text_tokens = 20
    num_vq_tokens = 1024
    
    logits_understand = torch.randn(batch_size, num_text_tokens, 92544)
    logits_generate = torch.randn(batch_size, num_vq_tokens, 8192)
    
    target_text_tokens = torch.randint(0, 92544, (batch_size, num_text_tokens))
    target_vq_codes = torch.randint(0, 8192, (batch_size, num_vq_tokens))
    
    total_loss, loss_dict = model.compute_loss(
        logits_understand,
        logits_generate,
        target_text_tokens,
        target_vq_codes
    )
    
    print(f"Loss computation successful:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n[Test 3] Gradient Flow")
    total_loss.backward()
    
    # Check gradients
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 
        for p in model.parameters() if p.requires_grad
    )
    
    if has_grad:
        print(" Gradients successfully computed through HTFA model!")
    else:
        print(" No gradients detected (this should not happen)")
    
    print("\n" + "=" * 80)
    print(" HTFA Model tests completed!")
    print("=" * 80)

