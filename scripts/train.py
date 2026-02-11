"""CLI entry point for model training.

Usage:
    python scripts/train.py
    python scripts/train.py training=debug data=fast_dev
    python scripts/train.py training.epochs=100 training.learning_rate=0.0005
"""

from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from biometric.training.trainer import Trainer
from biometric.utils import configure_deterministic, set_seed

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the training pipeline."""
    set_seed(cfg.seed)
    configure_deterministic(True)

    trainer = Trainer(cfg)
    results = trainer.fit()

    logger.info("Training complete.")
    logger.info("Best val accuracy: %.4f", results["training"]["best_val_accuracy"])
    logger.info("Test accuracy: %.4f", results["training"]["final_test_accuracy"])


if __name__ == "__main__":
    main()
