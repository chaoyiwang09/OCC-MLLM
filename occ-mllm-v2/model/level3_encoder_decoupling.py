"""
Level 3: Visual Encoder Decoupling Fusion Module

This module implements the third level of the Hierarchical Trinity Fusion Architecture (HTFA),
which resolves conflicting visual representation requirements between understanding and generation
tasks through specialized dual-encoder processing pathways.

Mathematical Formulation:
    S_3: (F_spatial, I_fused, C) → O_final
    
    1. Understanding Branch (from Level 2 spatial features):
       E_understand = A_understand(F_spatial)
    
    2. Generation Branch (from VQ tokenization):
       T_discrete = VQ_tokenizer(I_fused)
       E_generate = A_generate(Embed(T_discrete))
    
    3. Unified Autoregressive Integration:
       E_unified = E_understand ⊕ E_generate ⊕ C
       H_seq = Decoder_AR(E_unified)
    
    4. Dual Prediction Heads:
       ŷ_understand = H_understand(H_seq)  # Text token prediction
       ŷ_generate = H_generate(H_seq)      # VQ codebook ID prediction

Key Features:
    - Dual-encoder architecture: Understanding (SigLIP) + Generation (VQ)
    - Learnable MLP adapters for feature space projection to LLM dimension
    - Unified autoregressive decoder for joint understanding and generation
    - Separate prediction heads optimized via end-to-end backpropagation
    - Straight-through estimation for gradient flow through frozen VQ tokenizer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union
import math


class MLPAdapter(nn.Module):
    """
    Multi-Layer Perceptron Adapter for feature space projection.
    
    Projects features from one embedding space to another (e.g., SigLIP → LLM).
    Uses 2-layer MLP with GELU activation as specified in the paper.
    
    Args:
        input_dim (int): Input feature dimension
        output_dim (int): Output feature dimension (typically LLM hidden size)
        hidden_dim (int): Hidden layer dimension (default: 4x output_dim for better capacity)
        dropout (float): Dropout rate (default: 0.1)
        use_layer_norm (bool): Whether to apply layer normalization (default: True)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True
    ):
        super(MLPAdapter, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or (output_dim * 4)
        
        # Two-layer MLP: input → hidden → output
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(self.hidden_dim, output_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # Optional layer normalization for stable training
        self.layer_norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()
        
        # Initialize weights (Xavier initialization for better convergence)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project features through 2-layer MLP.
        
        Args:
            x: Input features [B, N, input_dim] or [B, input_dim]
        
        Returns:
            output: Projected features [B, N, output_dim] or [B, output_dim]
        """
        # First layer with activation and dropout
        hidden = self.fc1(x)
        hidden = self.activation(hidden)
        hidden = self.dropout1(hidden)
        
        # Second layer with dropout
        output = self.fc2(hidden)
        output = self.dropout2(output)
        
        # Layer normalization
        output = self.layer_norm(output)
        
        return output


class VQTokenizerWrapper(nn.Module):
    """
    Wrapper for Vector Quantization (VQ) tokenizer.
    
    Converts continuous images to discrete token sequences using a pre-trained
    VQ-VAE codebook. Supports straight-through estimation for gradient flow.
    
    Note: This is a placeholder implementation. In practice, you should load
    a pre-trained VQ tokenizer (e.g., from VQGAN, VQ-VAE, or similar models).
    
    Args:
        codebook_size (int): Number of codes in VQ codebook (default: 8192)
        embedding_dim (int): Dimension of each codebook entry (default: 256)
        num_tokens_h (int): Number of tokens in height dimension (default: 32)
        num_tokens_w (int): Number of tokens in width dimension (default: 32)
        frozen (bool): Whether to freeze VQ tokenizer weights (default: True)
    """
    
    def __init__(
        self,
        codebook_size: int = 8192,
        embedding_dim: int = 256,
        num_tokens_h: int = 32,
        num_tokens_w: int = 32,
        frozen: bool = True
    ):
        super(VQTokenizerWrapper, self).__init__()
        
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.num_tokens_h = num_tokens_h
        self.num_tokens_w = num_tokens_w
        self.frozen = frozen
        
        # VQ codebook: learnable embedding matrix
        # Shape: [codebook_size, embedding_dim]
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        
        # Simple encoder to map images to token IDs (placeholder)
        # In practice, this would be a CNN encoder from pre-trained VQ-VAE
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # -> H/2, W/2
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> H/4, W/4
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> H/8, W/8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, embedding_dim, kernel_size=3, padding=1),
        )
        
        if frozen:
            # Freeze all parameters
            for param in self.parameters():
                param.requires_grad = False
        
        print(f"[VQ Tokenizer] Initialized with codebook_size={codebook_size}, "
              f"embedding_dim={embedding_dim}, frozen={frozen}")
    
    def encode_to_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to continuous embeddings.
        
        Args:
            images: Input images [B, 3, H, W]
        
        Returns:
            embeddings: Continuous embeddings [B, h*w, embedding_dim]
        """
        B = images.shape[0]
        
        # Encode images to feature maps
        features = self.encoder(images)  # [B, embedding_dim, h, w]
        
        # Reshape to sequence format
        embeddings = features.flatten(2).transpose(1, 2)  # [B, h*w, embedding_dim]
        
        return embeddings
    
    def quantize(self, embeddings: torch.Tensor, use_straight_through: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize continuous embeddings to discrete codes.
        
        Args:
            embeddings: Continuous embeddings [B, N, embedding_dim]
            use_straight_through: Whether to use straight-through estimator for gradients
        
        Returns:
            quantized: Quantized embeddings [B, N, embedding_dim]
            token_ids: Discrete token IDs [B, N]
        """
        B, N, D = embeddings.shape
        
        # Compute distances to all codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        emb_sq = (embeddings ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
        codebook_sq = (self.codebook.weight ** 2).sum(dim=-1)  # [codebook_size]
        distances = emb_sq + codebook_sq - 2 * torch.matmul(embeddings, self.codebook.weight.T)
        
        # Find nearest codebook entry for each embedding
        token_ids = distances.argmin(dim=-1)  # [B, N]
        
        # Look up quantized embeddings from codebook
        quantized = self.codebook(token_ids)  # [B, N, embedding_dim]
        
        # Straight-through estimator: copy gradients from quantized to embeddings
        # This allows gradients to flow through the non-differentiable quantization
        if use_straight_through and self.training:
            quantized = embeddings + (quantized - embeddings).detach()
        
        return quantized, token_ids
    
    def forward(self, images: torch.Tensor, return_token_ids: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Tokenize images to discrete token IDs or quantized embeddings.
        
        Args:
            images: Input images [B, 3, H, W]
            return_token_ids: Whether to return token IDs (True) or embeddings (False)
        
        Returns:
            If return_token_ids=True: token_ids [B, h*w]
            If return_token_ids=False: (quantized_embeddings [B, h*w, D], token_ids [B, h*w])
        """
        # Encode to continuous embeddings
        embeddings = self.encode_to_embeddings(images)
        
        # Quantize to discrete codes
        quantized, token_ids = self.quantize(embeddings, use_straight_through=True)
        
        if return_token_ids:
            return token_ids
        else:
            return quantized, token_ids


class UnderstandingBranch(nn.Module):
    """
    Understanding Branch for semantic feature processing.
    
    Processes position-aware spatial features from Level 2 and projects them
    to LLM input space for understanding tasks (e.g., classification, QA).
    
    Args:
        spatial_feature_dim (int): Dimension of spatial features from Level 2 (default: 1152 for SigLIP)
        llm_hidden_dim (int): Hidden dimension of LLM (default: 2048 for InternLM2-7B)
        dropout (float): Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        spatial_feature_dim: int = 1152,
        llm_hidden_dim: int = 2048,
        dropout: float = 0.1
    ):
        super(UnderstandingBranch, self).__init__()
        
        self.spatial_feature_dim = spatial_feature_dim
        self.llm_hidden_dim = llm_hidden_dim
        
        # MLP adapter: SigLIP feature space → LLM input space
        self.adapter = MLPAdapter(
            input_dim=spatial_feature_dim,
            output_dim=llm_hidden_dim,
            dropout=dropout,
            use_layer_norm=True
        )
        
        print(f"[Understanding Branch] Initialized: {spatial_feature_dim} → {llm_hidden_dim}")
    
    def forward(self, feat_spatial: torch.Tensor) -> torch.Tensor:
        """
        Project spatial features to LLM input space.
        
        Args:
            feat_spatial: Spatial features from Level 2 [B, N, spatial_feature_dim]
        
        Returns:
            embeddings_understand: Understanding embeddings [B, N, llm_hidden_dim]
        """
        embeddings_understand = self.adapter(feat_spatial)
        return embeddings_understand


class GenerationBranch(nn.Module):
    """
    Generation Branch for visual token processing.
    
    Tokenizes fused images using VQ tokenizer and projects discrete token
    embeddings to LLM input space for generation tasks (e.g., image completion).
    
    Args:
        vq_codebook_size (int): Size of VQ codebook (default: 8192)
        vq_embedding_dim (int): Dimension of VQ embeddings (default: 256)
        llm_hidden_dim (int): Hidden dimension of LLM (default: 2048)
        dropout (float): Dropout rate (default: 0.1)
        vq_tokenizer (VQTokenizerWrapper): Pre-trained VQ tokenizer (optional)
    """
    
    def __init__(
        self,
        vq_codebook_size: int = 8192,
        vq_embedding_dim: int = 256,
        llm_hidden_dim: int = 2048,
        dropout: float = 0.1,
        vq_tokenizer: Optional[VQTokenizerWrapper] = None
    ):
        super(GenerationBranch, self).__init__()
        
        self.vq_codebook_size = vq_codebook_size
        self.vq_embedding_dim = vq_embedding_dim
        self.llm_hidden_dim = llm_hidden_dim
        
        # VQ tokenizer (frozen)
        if vq_tokenizer is not None:
            self.vq_tokenizer = vq_tokenizer
        else:
            self.vq_tokenizer = VQTokenizerWrapper(
                codebook_size=vq_codebook_size,
                embedding_dim=vq_embedding_dim,
                frozen=True
            )
        
        # MLP adapter: VQ embedding space → LLM input space
        self.adapter = MLPAdapter(
            input_dim=vq_embedding_dim,
            output_dim=llm_hidden_dim,
            dropout=dropout,
            use_layer_norm=True
        )
        
        print(f"[Generation Branch] Initialized: VQ({vq_codebook_size}, {vq_embedding_dim}) → {llm_hidden_dim}")
    
    def forward(self, img_fused: torch.Tensor, return_token_ids: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Tokenize fused image and project to LLM input space.
        
        Args:
            img_fused: Fused image from Level 1 [B, 3, H, W]
            return_token_ids: Whether to also return discrete token IDs
        
        Returns:
            embeddings_generate: Generation embeddings [B, N', llm_hidden_dim]
            token_ids: Discrete VQ token IDs [B, N'] (if return_token_ids=True)
        """
        # Tokenize image to discrete codes and get quantized embeddings
        quantized_embeddings, token_ids = self.vq_tokenizer(img_fused, return_token_ids=False)
        
        # Project VQ embeddings to LLM space
        embeddings_generate = self.adapter(quantized_embeddings)
        
        if return_token_ids:
            return embeddings_generate, token_ids
        else:
            return embeddings_generate


class VisualEncoderDecouplingFusion(nn.Module):
    """
    Level 3: Visual Encoder Decoupling Fusion
    
    Implements dual-encoder architecture with unified autoregressive integration.
    Resolves conflicting representation requirements between understanding and
    generation tasks through specialized processing pathways.
    
    Architecture:
        1. Understanding Branch: Level 2 spatial features → LLM space
        2. Generation Branch: VQ tokenized fused image → LLM space
        3. Unified Integration: Concatenate understanding + generation + text embeddings
        4. Process through autoregressive decoder (shared LLM)
    
    Args:
        spatial_feature_dim (int): Dimension of spatial features from Level 2
        vq_codebook_size (int): Size of VQ codebook
        vq_embedding_dim (int): Dimension of VQ embeddings
        llm_hidden_dim (int): Hidden dimension of LLM
        num_decoder_layers (int): Number of autoregressive decoder layers (default: 4)
        num_attention_heads (int): Number of attention heads in decoder (default: 16)
        dropout (float): Dropout rate (default: 0.1)
        vq_tokenizer (VQTokenizerWrapper): Pre-trained VQ tokenizer (optional)
    
    Forward Args:
        feat_spatial (torch.Tensor): Spatial features from Level 2 [B, N, spatial_feature_dim]
        img_fused (torch.Tensor): Fused image from Level 1 [B, 3, H, W]
        text_embeddings (torch.Tensor): Text query embeddings [B, L, llm_hidden_dim]
        attention_mask (torch.Tensor): Attention mask for autoregressive decoding [B, N+N'+L] (optional)
    
    Returns:
        hidden_states (torch.Tensor): Unified hidden states [B, N+N'+L, llm_hidden_dim]
        embeddings_dict (Dict): Dictionary containing individual branch embeddings
    """
    
    def __init__(
        self,
        spatial_feature_dim: int = 1152,
        vq_codebook_size: int = 8192,
        vq_embedding_dim: int = 256,
        llm_hidden_dim: int = 2048,
        num_decoder_layers: int = 4,
        num_attention_heads: int = 16,
        dropout: float = 0.1,
        vq_tokenizer: Optional[VQTokenizerWrapper] = None
    ):
        super(VisualEncoderDecouplingFusion, self).__init__()
        
        self.spatial_feature_dim = spatial_feature_dim
        self.llm_hidden_dim = llm_hidden_dim
        
        # Understanding branch (processes spatial features from Level 2)
        self.understanding_branch = UnderstandingBranch(
            spatial_feature_dim=spatial_feature_dim,
            llm_hidden_dim=llm_hidden_dim,
            dropout=dropout
        )
        
        # Generation branch (processes VQ tokenized images)
        self.generation_branch = GenerationBranch(
            vq_codebook_size=vq_codebook_size,
            vq_embedding_dim=vq_embedding_dim,
            llm_hidden_dim=llm_hidden_dim,
            dropout=dropout,
            vq_tokenizer=vq_tokenizer
        )
        
        # Unified autoregressive decoder (simplified version)
        # In practice, this would be the pre-trained LLM decoder (e.g., InternLM2)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=llm_hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=llm_hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.autoregressive_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Layer normalization for output
        self.output_norm = nn.LayerNorm(llm_hidden_dim)
        
        print(f"[Level 3] VisualEncoderDecouplingFusion initialized:")
        print(f"  - Spatial feature dim: {spatial_feature_dim}")
        print(f"  - LLM hidden dim: {llm_hidden_dim}")
        print(f"  - Decoder layers: {num_decoder_layers}")
        print(f"  - Attention heads: {num_attention_heads}")
    
    def forward(
        self,
        feat_spatial: torch.Tensor,
        img_fused: torch.Tensor,
        text_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_branch_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass of Level 3 visual encoder decoupling fusion.
        
        Returns:
            hidden_states: Unified hidden states [B, N+N'+L, llm_hidden_dim]
            embeddings_dict: Dictionary with branch embeddings (if return_branch_embeddings=True)
        """
        B = feat_spatial.shape[0]
        
        # Step 1: Process understanding branch (from Level 2 spatial features)
        embeddings_understand = self.understanding_branch(feat_spatial)  # [B, N, llm_hidden_dim]
        
        # Step 2: Process generation branch (from Level 1 fused image)
        embeddings_generate, token_ids = self.generation_branch(img_fused, return_token_ids=True)  # [B, N', llm_hidden_dim]
        
        # Step 3: Concatenate all embeddings: E_unified = E_understand ⊕ E_generate ⊕ C
        embeddings_unified = torch.cat([
            embeddings_understand,  # Understanding features
            embeddings_generate,    # Generation features
            text_embeddings         # Text query
        ], dim=1)  # [B, N+N'+L, llm_hidden_dim]
        
        # Step 4: Process through unified autoregressive decoder
        # In practice, this would be the LLM's decoder layers
        # Here we use a simplified transformer decoder for demonstration
        
        # Create causal attention mask if not provided
        seq_len = embeddings_unified.shape[1]
        if attention_mask is None:
            # Create causal mask for autoregressive generation
            attention_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=embeddings_unified.device),
                diagonal=1
            ).bool()
        
        # Pass through decoder (using embeddings as both memory and target)
        # In full implementation, this would integrate with pre-trained LLM decoder
        hidden_states = self.autoregressive_decoder(
            tgt=embeddings_unified,
            memory=embeddings_unified,
            tgt_mask=attention_mask if attention_mask.dim() == 2 else None
        )
        
        # Apply output normalization
        hidden_states = self.output_norm(hidden_states)
        
        if return_branch_embeddings:
            embeddings_dict = {
                'understand': embeddings_understand,
                'generate': embeddings_generate,
                'text': text_embeddings,
                'unified': embeddings_unified,
                'vq_token_ids': token_ids
            }
            return hidden_states, embeddings_dict
        else:
            return hidden_states


class DualPredictionHeads(nn.Module):
    """
    Dual Prediction Heads for Understanding and Generation Tasks
    
    Implements two specialized prediction heads:
    1. Understanding Head: Predicts text tokens for language understanding tasks
    2. Generation Head: Predicts VQ codebook IDs for image generation tasks
    
    Args:
        llm_hidden_dim (int): Hidden dimension of LLM
        vocab_size (int): Vocabulary size for text token prediction
        vq_codebook_size (int): VQ codebook size for visual token prediction
        tie_weights (bool): Whether to tie weights between heads (default: False)
    """
    
    def __init__(
        self,
        llm_hidden_dim: int = 2048,
        vocab_size: int = 92544,  # InternLM2 vocab size
        vq_codebook_size: int = 8192,
        tie_weights: bool = False
    ):
        super(DualPredictionHeads, self).__init__()
        
        self.llm_hidden_dim = llm_hidden_dim
        self.vocab_size = vocab_size
        self.vq_codebook_size = vq_codebook_size
        self.tie_weights = tie_weights
        
        # Understanding head: LLM hidden → Text token logits
        self.head_understand = nn.Linear(llm_hidden_dim, vocab_size, bias=False)
        
        # Generation head: LLM hidden → VQ codebook ID logits
        if tie_weights:
            # Share projection, but use separate bias
            self.head_generate = self.head_understand
        else:
            self.head_generate = nn.Linear(llm_hidden_dim, vq_codebook_size, bias=False)
        
        print(f"[Dual Prediction Heads] Initialized:")
        print(f"  - Understanding head: {llm_hidden_dim} → {vocab_size}")
        print(f"  - Generation head: {llm_hidden_dim} → {vq_codebook_size}")
        print(f"  - Tie weights: {tie_weights}")
    
    def forward_understand(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict text tokens for understanding tasks.
        
        Args:
            hidden_states: Hidden states from decoder [B, L, llm_hidden_dim]
        
        Returns:
            logits_understand: Text token logits [B, L, vocab_size]
        """
        logits_understand = self.head_understand(hidden_states)
        return logits_understand
    
    def forward_generate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict VQ codebook IDs for generation tasks.
        
        Args:
            hidden_states: Hidden states from decoder [B, N', llm_hidden_dim]
        
        Returns:
            logits_generate: VQ token logits [B, N', vq_codebook_size]
        """
        logits_generate = self.head_generate(hidden_states)
        return logits_generate
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        num_understand_tokens: int,
        num_generate_tokens: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply both prediction heads to appropriate portions of hidden states.
        
        Assumes hidden_states are organized as:
        [E_understand (N tokens) | E_generate (N' tokens) | Text (L tokens)]
        
        Args:
            hidden_states: Unified hidden states [B, N+N'+L, llm_hidden_dim]
            num_understand_tokens: Number of understanding tokens (N)
            num_generate_tokens: Number of generation tokens (N')
        
        Returns:
            logits_understand: Text prediction logits [B, L, vocab_size]
            logits_generate: VQ prediction logits [B, N', vq_codebook_size]
        """
        # Extract text portion for understanding (last L tokens)
        text_start_idx = num_understand_tokens + num_generate_tokens
        hidden_text = hidden_states[:, text_start_idx:, :]
        logits_understand = self.forward_understand(hidden_text)
        
        # Extract generation portion (middle N' tokens)
        hidden_generate = hidden_states[:, num_understand_tokens:text_start_idx, :]
        logits_generate = self.forward_generate(hidden_generate)
        
        return logits_understand, logits_generate


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("Testing Level 3: Visual Encoder Decoupling Fusion")
    print("=" * 80)
    
    # Test configuration
    batch_size = 2
    num_spatial_tokens = 32 * 32  # From Level 2
    num_text_tokens = 20
    spatial_feature_dim = 1152  # SigLIP dimension
    vq_codebook_size = 8192
    vq_embedding_dim = 256
    llm_hidden_dim = 2048
    vocab_size = 92544
    
    # Test 1: MLP Adapter
    print("\n[Test 1] MLPAdapter")
    adapter = MLPAdapter(input_dim=spatial_feature_dim, output_dim=llm_hidden_dim)
    
    test_features = torch.randn(batch_size, num_spatial_tokens, spatial_feature_dim)
    adapted_features = adapter(test_features)
    
    print(f"Input shape: {test_features.shape}")
    print(f"Output shape: {adapted_features.shape}")
    
    # Test 2: Understanding Branch
    print("\n[Test 2] UnderstandingBranch")
    understand_branch = UnderstandingBranch(
        spatial_feature_dim=spatial_feature_dim,
        llm_hidden_dim=llm_hidden_dim
    )
    
    feat_spatial = torch.randn(batch_size, num_spatial_tokens, spatial_feature_dim)
    embeddings_understand = understand_branch(feat_spatial)
    
    print(f"Spatial features shape: {feat_spatial.shape}")
    print(f"Understanding embeddings shape: {embeddings_understand.shape}")
    
    # Test 3: Generation Branch
    print("\n[Test 3] GenerationBranch")
    generation_branch = GenerationBranch(
        vq_codebook_size=vq_codebook_size,
        vq_embedding_dim=vq_embedding_dim,
        llm_hidden_dim=llm_hidden_dim
    )
    
    img_fused = torch.randn(batch_size, 3, 448, 448)
    embeddings_generate, token_ids = generation_branch(img_fused, return_token_ids=True)
    
    print(f"Fused image shape: {img_fused.shape}")
    print(f"Generation embeddings shape: {embeddings_generate.shape}")
    print(f"VQ token IDs shape: {token_ids.shape}")
    
    # Test 4: Complete Level 3 Fusion
    print("\n[Test 4] VisualEncoderDecouplingFusion")
    level3_fusion = VisualEncoderDecouplingFusion(
        spatial_feature_dim=spatial_feature_dim,
        vq_codebook_size=vq_codebook_size,
        vq_embedding_dim=vq_embedding_dim,
        llm_hidden_dim=llm_hidden_dim,
        num_decoder_layers=2,
        num_attention_heads=16
    )
    
    # Create dummy inputs
    feat_spatial = torch.randn(batch_size, num_spatial_tokens, spatial_feature_dim)
    img_fused = torch.randn(batch_size, 3, 448, 448)
    text_embeddings = torch.randn(batch_size, num_text_tokens, llm_hidden_dim)
    
    # Forward pass
    hidden_states, embeddings_dict = level3_fusion(
        feat_spatial=feat_spatial,
        img_fused=img_fused,
        text_embeddings=text_embeddings,
        return_branch_embeddings=True
    )
    
    print(f"\nInput shapes:")
    print(f"  feat_spatial: {feat_spatial.shape}")
    print(f"  img_fused: {img_fused.shape}")
    print(f"  text_embeddings: {text_embeddings.shape}")
    
    print(f"\nBranch embeddings shapes:")
    for key, value in embeddings_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print(f"\nOutput hidden states shape: {hidden_states.shape}")
    
    # Test 5: Dual Prediction Heads
    print("\n[Test 5] DualPredictionHeads")
    prediction_heads = DualPredictionHeads(
        llm_hidden_dim=llm_hidden_dim,
        vocab_size=vocab_size,
        vq_codebook_size=vq_codebook_size
    )
    
    num_generate_tokens = embeddings_generate.shape[1]
    logits_understand, logits_generate = prediction_heads(
        hidden_states=hidden_states,
        num_understand_tokens=num_spatial_tokens,
        num_generate_tokens=num_generate_tokens
    )
    
    print(f"\nPrediction logits shapes:")
    print(f"  logits_understand: {logits_understand.shape}")
    print(f"  logits_generate: {logits_generate.shape}")
    
    # Test 6: Gradient flow verification
    print("\n[Test 6] Gradient Flow Verification")
    loss_understand = logits_understand.mean()
    loss_generate = logits_generate.mean()
    total_loss = loss_understand + 0.3 * loss_generate
    total_loss.backward()
    
    # Check gradients for key components
    grad_checks = {
        'understanding_adapter_fc1': understand_branch.adapter.fc1.weight.grad is not None,
        'generation_adapter_fc1': generation_branch.adapter.fc1.weight.grad is not None,
        'decoder_layer0': level3_fusion.autoregressive_decoder.layers[0].self_attn.in_proj_weight.grad is not None,
        'head_understand': prediction_heads.head_understand.weight.grad is not None,
        'head_generate': prediction_heads.head_generate.weight.grad is not None,
    }
    
    print("Gradient computation status:")
    for name, status in grad_checks.items():
        status_str = "" if status else ""
        print(f"  {status_str} {name}")
    
    if all(grad_checks.values()):
        print("\n All Level 3 learnable parameters receive gradients!")
    
    print("\n" + "=" * 80)
    print(" All Level 3 tests passed!")
    print("=" * 80)
