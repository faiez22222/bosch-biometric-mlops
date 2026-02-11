"""CNN encoder for iris feature extraction."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from biometric.models.fingerprint_encoder import ConvBlock


class IrisEncoder(nn.Module):
    """CNN encoder for RGB iris images.

    Input:  [B, 3, H, W]  (RGB, typically 224x224)
    Output: [B, embedding_dim]  (default 256)

    Same ConvBlock architecture as FingerprintEncoder but with 3 input
    channels. Kept as a separate class to allow independent configuration
    and future modality-specific layers (e.g., Gabor filters for iris).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        enc_cfg = cfg.model.iris_encoder

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
