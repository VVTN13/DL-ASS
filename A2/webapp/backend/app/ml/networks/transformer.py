"""Temporal Transformer evaluated in A2, ported from ``A2/model_lib.py`` for serving.

Module and attribute names are unchanged so the research checkpoints load with
``strict=True``. Only the attention-recording branch, unused at inference, is removed.
Input: ``[B, in_channels, 30, 200]`` (30 one-second patches at 200 Hz, microvolts / 100).
Output: ``[B, num_classes]`` logits.
"""

import torch
import torch.nn.functional as F
from torch import nn


class TemporalConvEmbedding(nn.Module):
    """Encode each one-second, multichannel waveform patch as one time token."""

    def __init__(self, in_channels: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.in_channels = in_channels
        self.short_branch = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=25, stride=2, padding=12, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.long_branch = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=101, stride=4, padding=50, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.projection = nn.Sequential(
            nn.Linear(2 * 64 * 4, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, 31, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (self.in_channels, 30, 200):
            raise ValueError("Expected [B, in_channels, 30, 200]: 30-second EEG at 200 Hz.")
        batch_size, channels, seconds, samples = x.shape
        # Both channels of the same second stay together, in chronological order.
        patches = x.float().permute(0, 2, 1, 3).reshape(batch_size * seconds, channels, samples)
        features = torch.cat((self.short_branch(patches), self.long_branch(patches)), dim=1)
        tokens = self.projection(features.flatten(1)).reshape(batch_size, seconds, -1)
        tokens = torch.cat((self.cls_token.expand(batch_size, -1, -1), tokens), dim=1)
        return self.dropout(tokens + self.position_embedding)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(emb_dim, emb_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        dropout_p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        x = x.transpose(1, 2).reshape(batch_size, tokens, channels)
        return self.proj_drop(self.proj(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, mlp_ratio: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = MultiHeadSelfAttention(emb_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_ratio * emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self, in_channels: int = 2, emb_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 3, mlp_ratio: int = 3, num_classes: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        if emb_dim % num_heads:
            raise ValueError("emb_dim must be divisible by num_heads.")
        self.patch_embed = TemporalConvEmbedding(in_channels, emb_dim, dropout)
        self.encoder = nn.ModuleList(
            [TransformerEncoderLayer(emb_dim, num_heads, mlp_ratio, dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(emb_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for layer in self.encoder:
            x = layer(x)
        return self.head(self.norm(x[:, 0]))  # CLS token only
