"""Top-level multimodal biometric network."""

from __future__ import annotations

from typing import Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from biometric.models.fingerprint_encoder import FingerprintEncoder
from biometric.models.fusion import MultimodalFusion
from biometric.models.iris_encoder import IrisEncoder


class MultimodalBiometricNet(nn.Module):
    """Composes modality-specific encoders with a fusion classifier.

    Supports:
    - Full multimodal (fingerprint + iris)
    - Fingerprint-only (set iris_encoder to null in config)
    - Iris-only (set fingerprint_encoder to null in config)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.fp_encoder: Optional[FingerprintEncoder] = None
        self.iris_encoder: Optional[IrisEncoder] = None

        if cfg.model.get("fingerprint_encoder") is not None:
            self.fp_encoder = FingerprintEncoder(cfg)
        if cfg.model.get("iris_encoder") is not None:
            self.iris_encoder = IrisEncoder(cfg)

        self.fusion = MultimodalFusion(cfg)

    def forward(
        self,
        fingerprint: Optional[Tensor] = None,
        iris: Optional[Tensor] = None,
    ) -> Tensor:
        fp_features = None
        iris_features = None

        if self.fp_encoder is not None and fingerprint is not None:
            fp_features = self.fp_encoder(fingerprint)
        if self.iris_encoder is not None and iris is not None:
            iris_features = self.iris_encoder(iris)

        return self.fusion(fp_features, iris_features)

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts per component."""
        counts: dict[str, int] = {}
        if self.fp_encoder is not None:
            counts["fingerprint_encoder"] = sum(
                p.numel() for p in self.fp_encoder.parameters()
            )
        if self.iris_encoder is not None:
            counts["iris_encoder"] = sum(
                p.numel() for p in self.iris_encoder.parameters()
            )
        counts["fusion"] = sum(p.numel() for p in self.fusion.parameters())
        counts["total"] = sum(counts.values())
        return counts
