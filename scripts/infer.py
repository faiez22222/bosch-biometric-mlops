"""CLI entry point for inference.

Usage:
    python scripts/infer.py \
        checkpoint_path=./checkpoints/best.pt \
        fingerprint_path=./data/1/Fingerprint/1__M_Left_index_finger.BMP \
        iris_path=./data/1/left/aeval1.bmp
"""

from __future__ import annotations

import json
import logging

import hydra
from omegaconf import DictConfig

from biometric.inference.predictor import BiometricPredictor

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run inference on a single fingerprint + iris pair."""
    predictor = BiometricPredictor.from_checkpoint(cfg.checkpoint_path)

    result = predictor.predict(
        fingerprint=cfg.fingerprint_path,
        iris=cfg.iris_path,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
