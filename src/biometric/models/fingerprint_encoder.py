"""CNN encoder for fingerprint feature extraction."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FingerprintEncoder(nn.Module):
    """CNN encoder for grayscale fingerprint images.

    Input:  [B, 1, H, W]  (grayscale, typically 128x128)
    Output: [B, embedding_dim]  (default 256)

    Architecture: N ConvBlocks with doubling channels, followed by
    AdaptiveAvgPool2d and a linear projection head.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        enc_cfg = cfg.model.fingerprint_encoder

        blocks: list[nn.Module] = []
        in_ch = enc_cfg.in_channels
        for i in range(enc_cfg.num_blocks):
            out_ch = enc_cfg.base_filters * (2 ** i)
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, enc_cfg.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(enc_cfg.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)
