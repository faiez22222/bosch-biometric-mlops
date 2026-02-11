"""CLI entry point for inference.

Usage:
    python scripts/infer.py checkpoint=./checkpoints/best.pt
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run inference on provided samples."""
    # Phase 6 implementation will import and use:
    # from biometric.inference.predictor import BiometricPredictor
    raise NotImplementedError(
        "Inference pipeline not yet implemented. See docs/ARCHITECTURE.md for the plan."
    )


if __name__ == "__main__":
    main()
