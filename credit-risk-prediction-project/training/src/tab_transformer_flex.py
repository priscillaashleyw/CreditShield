"""
tab_transformer_flex.py
=======================
Flexible TabTransformer that supports multiple output modes:

  1. BINARY CLASSIFICATION (default — current credit risk use case)
     Input:  loan features → Output: P(default) ∈ [0, 1]

  2. MULTI-CLASS CLASSIFICATION
     Input:  loan features → Output: P(class_k) for k = 1..K

  3. REGRESSION
     Input:  loan features → Output: continuous value (e.g. loss-given-default)

  4. MULTI-OUTPUT (sequence-to-output)
     Input:  loan features → Output: multiple targets simultaneously
     e.g. (P(default), expected_loss, risk_grade)

Architecture
------------
  Encoder-only transformer (no decoder needed for tabular input → output).

  Categorical features → per-feature Embedding → Transformer self-attention
  Numerical features   → optional MLP projection → bypass concatenation
  Fused representation → task-specific output head(s)

This file extends tab_transformer.py without modifying it.
Import the original components and build on top.

Usage examples are in the docstring of FlexTabTransformer.
"""

import torch
import torch.nn as nn
import numpy as np

from tab_transformer import (
    Embeddings,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock,
)


class NumericalProjection(nn.Module):
    """
    Optional learnable projection for numerical features.

    Instead of raw bypass, this projects each numerical feature into
    the same embedding_dim space as categorical tokens, then adds them
    as extra tokens to the transformer sequence.

    This is the "FT-Transformer" variant from Gorishniy et al. (2021):
    https://arxiv.org/abs/2106.11959
    """

    def __init__(self, num_features: int, embedding_dim: int):
        super().__init__()
        # One learned linear projection per numerical feature
        self.projections = nn.ModuleList([
            nn.Linear(1, embedding_dim) for _ in range(num_features)
        ])
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features)
        Returns:
            (batch, num_features, embedding_dim)
        """
        tokens = [
            proj(x[:, i:i+1])          # (batch, 1) → (batch, embedding_dim)
            for i, proj in enumerate(self.projections)
        ]
        return self.norm(torch.stack(tokens, dim=1))


class FlexTabTransformer(nn.Module):
    """
    Flexible TabTransformer with configurable output mode.

    Parameters
    ----------
    num_numerical_features : int
    num_categorical_features : int
    categorical_dims : list[int]
        Vocabulary size per categorical column.
    embedding_dim : int
    depth : int
    heads : int
    dim_head : int
    mlp_dim : int
    dropout : float

    output_mode : str
        'binary'      → sigmoid, single scalar P(default)
        'multiclass'  → softmax over num_classes
        'regression'  → raw linear output (no activation)
        'multi_output'→ multiple heads, one per target

    num_classes : int
        For 'multiclass' mode only.

    output_dims : list[tuple(str, int)]
        For 'multi_output' mode.  Each entry is (name, dim).
        Example: [('default_prob', 1), ('loss_amount', 1), ('risk_grade', 5)]

    project_numerical : bool
        If True, project numerical features into embedding space and
        feed them as extra tokens to the transformer (FT-Transformer style).
        If False, use the original bypass concatenation.

    Example usage
    -------------
    >>> # Binary classification (drop-in replacement for TabTransformer)
    >>> model = FlexTabTransformer(
    ...     num_numerical_features=33, num_categorical_features=7,
    ...     categorical_dims=[2,12,5,3,14,50,2],
    ...     output_mode='binary',
    ... )
    >>> probs = model(numerical, categorical)  # (batch,)

    >>> # Multi-output: predict default + loss + grade simultaneously
    >>> model = FlexTabTransformer(
    ...     num_numerical_features=33, num_categorical_features=7,
    ...     categorical_dims=[2,12,5,3,14,50,2],
    ...     output_mode='multi_output',
    ...     output_dims=[
    ...         ('default_prob', 1),   # binary: sigmoid
    ...         ('loss_amount', 1),    # regression: no activation
    ...         ('risk_grade', 5),     # 5-class: softmax
    ...     ],
    ... )
    >>> outputs = model(numerical, categorical)
    >>> # outputs = {'default_prob': (batch,1), 'loss_amount': (batch,1), 'risk_grade': (batch,5)}

    >>> # Regression: predict continuous loss-given-default
    >>> model = FlexTabTransformer(
    ...     ..., output_mode='regression',
    ... )
    >>> loss_amount = model(numerical, categorical)  # (batch, 1)
    """

    def __init__(
        self,
        num_numerical_features: int,
        num_categorical_features: int,
        categorical_dims: list,
        embedding_dim: int = 32,
        depth: int = 4,
        heads: int = 4,
        dim_head: int = 32,
        mlp_dim: int = 256,
        dropout: float = 0.2,
        # ── Output configuration ──
        output_mode: str = 'binary',
        num_classes: int = 2,
        output_dims: list = None,
        # ── Numerical projection ──
        project_numerical: bool = False,
    ):
        super().__init__()

        assert output_mode in ('binary', 'multiclass', 'regression', 'multi_output'), \
            f"Unknown output_mode: {output_mode}"

        self.output_mode = output_mode
        self.num_numerical = num_numerical_features
        self.num_categorical = num_categorical_features
        self.embedding_dim = embedding_dim
        self.project_numerical = project_numerical

        # ── Categorical embeddings ────────────────────────────────────
        self.embeddings = nn.ModuleList([
            Embeddings(cat_dim + 1, embedding_dim)
            for cat_dim in categorical_dims
        ])

        # ── Optional numerical projection ─────────────────────────────
        if project_numerical and num_numerical_features > 0:
            self.num_projection = NumericalProjection(
                num_numerical_features, embedding_dim
            )
            # Transformer sees categorical + numerical tokens
            seq_len = num_categorical_features + num_numerical_features
        else:
            self.num_projection = None
            seq_len = num_categorical_features

        # ── Transformer encoder ───────────────────────────────────────
        heads = min(heads, seq_len)   # heads <= sequence length
        self.transformer = TransformerBlock(
            dim=embedding_dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        # ── Pooling ──────────────────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool1d(1)

        # ── Fusion dimension ─────────────────────────────────────────
        if project_numerical:
            # Everything goes through the transformer, pooled to embedding_dim
            fusion_dim = embedding_dim
        else:
            # Bypass: transformer output + raw numerical features
            fusion_dim = embedding_dim + num_numerical_features

        # ── Output head(s) ───────────────────────────────────────────
        if output_mode == 'binary':
            self.head = nn.Sequential(
                nn.Linear(fusion_dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, mlp_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim // 2, 1),
            )

        elif output_mode == 'multiclass':
            self.head = nn.Sequential(
                nn.Linear(fusion_dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, num_classes),
            )

        elif output_mode == 'regression':
            self.head = nn.Sequential(
                nn.Linear(fusion_dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, mlp_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim // 2, 1),
            )

        elif output_mode == 'multi_output':
            assert output_dims is not None, \
                "output_dims required for multi_output mode"
            self.output_dims = output_dims
            self.heads_dict = nn.ModuleDict()
            for name, dim in output_dims:
                self.heads_dict[name] = nn.Sequential(
                    nn.Linear(fusion_dim, mlp_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim // 2, dim),
                )

    def _encode(self, numerical_features, categorical_features):
        """
        Shared encoder: embed → transform → pool → fuse.

        Returns the fused representation vector (batch, fusion_dim).
        """
        # ── Embed categorical features ────────────────────────────────
        embedded = torch.stack([
            emb(categorical_features[:, i])
            for i, emb in enumerate(self.embeddings)
        ], dim=1)   # (batch, num_cat, embedding_dim)

        # ── Optionally project + concat numerical tokens ──────────────
        if self.project_numerical and self.num_projection is not None:
            num_tokens = self.num_projection(numerical_features)
            # (batch, num_num, embedding_dim)
            tokens = torch.cat([embedded, num_tokens], dim=1)
        else:
            tokens = embedded

        # ── Transformer ───────────────────────────────────────────────
        transformed = self.transformer(tokens)

        # ── Pool → (batch, embedding_dim) ─────────────────────────────
        pooled = self.pool(transformed.permute(0, 2, 1)).squeeze(-1)

        # ── Fuse with numerical bypass (if not projected) ─────────────
        if self.project_numerical:
            return pooled
        else:
            return torch.cat([pooled, numerical_features], dim=-1)

    def forward(self, numerical_features, categorical_features):
        """
        Forward pass.

        Returns depend on output_mode:
            'binary'      → (batch,)      probabilities
            'multiclass'  → (batch, K)    logits (apply softmax yourself)
            'regression'  → (batch, 1)    continuous predictions
            'multi_output'→ dict[str, Tensor]  one tensor per output head
        """
        h = self._encode(numerical_features, categorical_features)

        if self.output_mode == 'binary':
            return torch.sigmoid(self.head(h)).squeeze(-1)

        elif self.output_mode == 'multiclass':
            return self.head(h)   # raw logits; use CrossEntropyLoss

        elif self.output_mode == 'regression':
            return self.head(h)

        elif self.output_mode == 'multi_output':
            return {
                name: head(h)
                for name, head in self.heads_dict.items()
            }
