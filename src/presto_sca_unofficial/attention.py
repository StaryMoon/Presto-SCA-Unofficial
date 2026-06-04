from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SegmentedCrossAttention(nn.Module):
    """Segment video tokens along time and attend to matching sub-captions."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(
        self,
        video_tokens: torch.Tensor,
        subcaption_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if video_tokens.ndim != 4:
            raise ValueError("video_tokens must have shape [B, T, P, C].")
        if subcaption_tokens.ndim != 4:
            raise ValueError("subcaption_tokens must have shape [B, S, L, C].")

        batch, frames, patches, channels = video_tokens.shape
        _, segments, _, text_channels = subcaption_tokens.shape
        if channels != text_channels:
            raise ValueError("video and text channels must match.")

        outputs = []
        for index in range(segments):
            start = round(index * frames / segments)
            end = round((index + 1) * frames / segments)
            segment = video_tokens[:, start:end]
            segment_frames = segment.shape[1]
            query = segment.reshape(batch, segment_frames * patches, channels)
            key_value = subcaption_tokens[:, index]
            attended, _ = self.attn(query, key_value, key_value)
            outputs.append(attended.reshape(batch, segment_frames, patches, channels))
        return torch.cat(outputs, dim=1)


class TinyPrestoBlock(nn.Module):
    """Tiny DiT-style wrapper around segmented cross-attention."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, mlp_ratio: int = 4):
        super().__init__()
        self.norm_video = nn.LayerNorm(embed_dim)
        self.norm_text = nn.LayerNorm(embed_dim)
        self.sca = SegmentedCrossAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(
        self,
        video_tokens: torch.Tensor,
        subcaption_tokens: torch.Tensor,
    ) -> torch.Tensor:
        attended = self.sca(
            self.norm_video(video_tokens),
            self.norm_text(subcaption_tokens),
        )
        x = video_tokens + attended
        return x + self.mlp(self.norm_mlp(x))

    def toy_denoising_loss(
        self,
        noisy_tokens: torch.Tensor,
        clean_tokens: torch.Tensor,
        subcaption_tokens: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.forward(noisy_tokens, subcaption_tokens)
        return F.mse_loss(prediction, clean_tokens)
