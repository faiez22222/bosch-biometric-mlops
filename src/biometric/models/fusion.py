"""Multimodal fusion head for combining fingerprint and iris features."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


class MultimodalFusion(nn.Module):
    """Fuses fingerprint and iris embeddings into classification logits.

    Supports two strategies:
    - 'concatenation': Concatenate both feature vectors -> classifier.
    - 'single_modality': Use one modality's features -> classifier.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        fusion_cfg = cfg.model.fusion
        num_classes = cfg.model.num_classes
        self.strategy = fusion_cfg.strategy

        if self.strategy == "concatenation":
            fp_dim = cfg.model.fingerprint_encoder.embedding_dim
            iris_dim = cfg.model.iris_encoder.embedding_dim
            input_dim = fp_dim + iris_dim
        elif self.strategy == "single_modality":
            if cfg.model.get("iris_encoder") is not None:
                input_dim = cfg.model.iris_encoder.embedding_dim
            else:
                input_dim = cfg.model.fingerprint_encoder.embedding_dim
        else:
            raise ValueError(f"Unknown fusion strategy: {self.strategy}")

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, fusion_cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_cfg.dropout),
            nn.Linear(fusion_cfg.hidden_dim, num_classes),
        )

    def forward(
        self,
        fp_features: Optional[Tensor] = None,
        iris_features: Optional[Tensor] = None,
    ) -> Tensor:
        if self.strategy == "concatenation":
            assert fp_features is not None and iris_features is not None, (
                "Concatenation fusion requires both modalities."
            )
            x = torch.cat([fp_features, iris_features], dim=1)
        else:
            x = fp_features if fp_features is not None else iris_features
            assert x is not None, "At least one modality must be provided."
        return self.classifier(x)
