"""Training metrics tracking and aggregation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks loss and accuracy across steps and epochs.

    Usage:
        tracker = MetricsTracker()
        tracker.reset_epoch()
        for batch in loader:
            tracker.update(loss_value, logits, labels)
        epoch_metrics = tracker.compute_epoch_metrics()
    """

    def __init__(self) -> None:
        self._step_losses: list[float] = []
        self._step_correct: int = 0
        self._step_total: int = 0
        self._epoch_history: list[dict[str, Any]] = []

    def reset_epoch(self) -> None:
        """Reset step-level accumulators for a new epoch."""
        self._step_losses = []
        self._step_correct = 0
        self._step_total = 0

    def update(self, loss: float, logits: Tensor, labels: Tensor) -> None:
        """Update with a single batch result.

        Args:
            loss: Scalar loss value for this batch.
            logits: Model output logits [B, num_classes].
            labels: Ground truth labels [B].
        """
        self._step_losses.append(loss)
        predictions = torch.argmax(logits, dim=1)
        self._step_correct += (predictions == labels).sum().item()
        self._step_total += labels.size(0)

    def compute_epoch_metrics(self) -> dict[str, float]:
        """Compute epoch-level averages and store in history.

        Returns:
            Dict with 'loss' and 'accuracy' for the epoch.
        """
        avg_loss = sum(self._step_losses) / max(len(self._step_losses), 1)
        accuracy = self._step_correct / max(self._step_total, 1)
        metrics = {"loss": avg_loss, "accuracy": accuracy}
        self._epoch_history.append(metrics)
        return metrics

    def get_history(self) -> list[dict[str, Any]]:
        """Return full epoch history."""
        return list(self._epoch_history)

    def to_json(self, path: Path) -> None:
        """Save epoch history to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._epoch_history, f, indent=2)
        logger.info("Saved metrics history to %s", path)
