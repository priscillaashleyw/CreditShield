"""
TabTransformer - Transformer architecture for tabular data
Based on: https://arxiv.org/abs/2012.06678
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat
import numpy as np

class Embeddings(nn.Module):
    """Embedding layer for categorical features"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, output_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)
    
    def forward(self, x):
        return self.embedding(x)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    """Feed-forward network with GLU"""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                MultiHeadAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))
    
    def forward(self, x):
        for ln1, attn, ln2, ff in self.layers:
            x = attn(ln1(x)) + x
            x = ff(ln2(x)) + x
        return x

class TabTransformer(nn.Module):
    """TabTransformer: Transformer for tabular data with numerical bypass"""
    def __init__(
        self,
        num_numerical_features,
        num_categorical_features,
        categorical_dims,
        embedding_dim=32,
        depth=6,
        heads=8,
        dim_head=64,
        mlp_dim=512,
        num_classes=1,  # Changed: single output for probability
        dropout=0.1,
        debug=False,
        **kwargs
    ):
        super().__init__()
        
        self.num_numerical = num_numerical_features
        self.num_categorical = num_categorical_features
        self.embedding_dim = embedding_dim
        self.debug = debug
        
        # Validate heads parameter
        if heads > num_categorical_features:
            raise ValueError(
                f"heads ({heads}) must be <= num_categorical_features ({num_categorical_features}). "
                f"Automatically adjusted to {num_categorical_features}."
            )
        
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            Embeddings(cat_dim + 1, embedding_dim)
            for cat_dim in categorical_dims
        ])
        
        # Project each embedding to embedding_dim if needed
        self.embedding_projection = nn.Identity()
        
        # Transformer processes each categorical feature as a token
        # Input shape: (batch_size, num_categorical_features, embedding_dim)
        transformer_dim = embedding_dim
        self.transformer = TransformerBlock(
            dim=transformer_dim,
            depth=depth,
            heads=min(heads, num_categorical_features),  # heads <= sequence length
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout
        )
        
        # Pool transformer output
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Final MLP - outputs single probability
        final_dim = transformer_dim + num_numerical_features
        self.mlp = nn.Sequential(
            nn.Linear(final_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, 1)  # Single output
        )
    
    def forward(self, numerical_features, categorical_features):
        """
        Forward pass
        Args:
            numerical_features: (batch_size, num_numerical_features)
            categorical_features: (batch_size, num_categorical_features) - integer indices
        Returns:
            probabilities: (batch_size,) - probability of default (0 to 1)
        """
        if self.debug:
            print(f"[DEBUG] Input shapes - Numerical: {numerical_features.shape}, Categorical: {categorical_features.shape}")
        
        # Embed categorical features
        # Output: list of (batch_size, embedding_dim)
        embedded_list = [emb(categorical_features[:, i]) for i, emb in enumerate(self.embeddings)]
        
        # Stack to (batch_size, num_categorical_features, embedding_dim)
        embedded = torch.stack(embedded_list, dim=1)
        
        if self.debug:
            print(f"[DEBUG] Embedded shape: {embedded.shape}")
        
        # Apply transformer
        # Expects (batch_size, sequence_length, feature_dim)
        transformed = self.transformer(embedded)
        
        if self.debug:
            print(f"[DEBUG] Transformed shape: {transformed.shape}")
        
        # Pool over sequence dimension: (batch_size, num_categorical_features, embedding_dim) -> (batch_size, embedding_dim)
        # Permute to (batch_size, embedding_dim, num_categorical_features) for pooling
        pooled = self.pool(transformed.permute(0, 2, 1))  # (batch_size, embedding_dim, 1)
        pooled = pooled.squeeze(-1)  # (batch_size, embedding_dim)
        
        if self.debug:
            print(f"[DEBUG] Pooled shape: {pooled.shape}")
            print(f"[DEBUG] Numerical features shape: {numerical_features.shape}")
        
        # Concatenate numerical features (bypass)
        combined = torch.cat([pooled, numerical_features], dim=-1)
        
        if self.debug:
            print(f"[DEBUG] Combined shape: {combined.shape}")
        
        # Output probability via sigmoid
        logit = self.mlp(combined)
        prob = torch.sigmoid(logit).squeeze(-1)  # (batch_size,)
        
        if self.debug:
            print(f"[DEBUG] Probability shape: {prob.shape}")
        
        return prob
