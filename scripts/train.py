"""CLI entry point for model training.

Usage:
    python scripts/train.py
    python scripts/train.py training=debug data=fast_dev
    python scripts/train.py training.epochs=100 training.learning_rate=0.0005
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the training pipeline."""
    # Phase 5 implementation will import and use:
    # from biometric.training.trainer import Trainer
    # from biometric.utils import set_seed, configure_deterministic
    raise NotImplementedError(
        "Training pipeline not yet implemented. See docs/ARCHITECTURE.md for the plan."
    )


if __name__ == "__main__":
    main()
