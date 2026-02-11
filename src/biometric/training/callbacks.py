"""Training callbacks: checkpointing and early stopping."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class CheckpointCallback:
    """Saves model checkpoints when the monitored metric improves.

    Keeps the top-K best checkpoints plus a 'last.pt' that is always saved.
    Each checkpoint stores the full config for reproducibility.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._dir = Path(cfg.training.checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._save_top_k = cfg.training.save_top_k
        self._best_metric: float | None = None
        self._best_path: Path | None = None
        # List of (metric_value, path) sorted worst-first
        self._saved: list[tuple[float, Path]] = []

    @property
    def best_metric(self) -> float | None:
        return self._best_metric

    @property
    def best_checkpoint_path(self) -> Path | None:
        return self._best_path

    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any,
        metrics: dict[str, float],
        cfg: DictConfig,
        label_map: dict[int, int],
    ) -> None:
        """Save checkpoint if this epoch's val_loss is among the best."""
        val_loss = metrics.get("loss", float("inf"))

        # Always save last.pt
        last_path = self._dir / "last.pt"
        self._save(last_path, epoch, model, optimizer, scheduler, metrics, cfg, label_map)

        # Check if this is a top-K checkpoint
        if len(self._saved) < self._save_top_k or val_loss < self._saved[-1][0]:
            ckpt_path = self._dir / f"epoch_{epoch:03d}_loss_{val_loss:.4f}.pt"
            self._save(ckpt_path, epoch, model, optimizer, scheduler, metrics, cfg, label_map)
            self._saved.append((val_loss, ckpt_path))
            self._saved.sort(key=lambda x: x[0])

            # Remove worst if over limit
            while len(self._saved) > self._save_top_k:
                _, worst_path = self._saved.pop()
                if worst_path.exists():
                    worst_path.unlink()
                    logger.info("Removed checkpoint: %s", worst_path.name)

            # Update best
            self._best_metric = self._saved[0][0]
            self._best_path = self._saved[0][1]
            logger.info(
                "Saved checkpoint: %s (val_loss=%.4f, best=%.4f)",
                ckpt_path.name,
                val_loss,
                self._best_metric,
            )

    def _save(
        self,
        path: Path,
        epoch: int,
        model: nn.Module,
        optimizer: Any,
        scheduler: Any,
        metrics: dict[str, float],
        cfg: DictConfig,
        label_map: dict[int, int],
    ) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "label_map": label_map,
        }
        torch.save(checkpoint, path)


class EarlyStoppingCallback:
    """Stops training when the monitored metric stops improving."""

    def __init__(self, cfg: DictConfig) -> None:
        es_cfg = cfg.training.early_stopping
        self._enabled = es_cfg.enabled
        self._patience = es_cfg.patience
        self._mode = es_cfg.mode  # 'min' or 'max'
        self._monitor = es_cfg.monitor  # e.g. 'val_loss'
        self._counter = 0
        self._best: float | None = None

    @property
    def counter(self) -> int:
        return self._counter

    def on_epoch_end(self, metrics: dict[str, float]) -> bool:
        """Check if training should stop.

        Args:
            metrics: Dict with the monitored metric key.

        Returns:
            True if training should stop.
        """
        if not self._enabled:
            return False

        # Map monitor name to metric key
        if self._monitor == "val_loss":
            current = metrics.get("loss", float("inf"))
        elif self._monitor == "val_accuracy":
            current = metrics.get("accuracy", 0.0)
        else:
            current = metrics.get(self._monitor, float("inf"))

        if self._best is None:
            self._best = current
            return False

        improved = (
            current < self._best if self._mode == "min" else current > self._best
        )

        if improved:
            self._best = current
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self._patience:
                logger.info(
                    "Early stopping triggered after %d epochs without improvement.",
                    self._patience,
                )
                return True

        return False
