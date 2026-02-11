"""Tests for the training pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from biometric.training.callbacks import CheckpointCallback, EarlyStoppingCallback
from biometric.training.metrics import MetricsTracker
from biometric.training.trainer import Trainer


class TestMetricsTracker:
    def test_single_epoch(self):
        tracker = MetricsTracker()
        tracker.reset_epoch()
        tracker.update(0.5, torch.randn(4, 3), torch.tensor([0, 1, 2, 0]))
        tracker.update(0.3, torch.randn(4, 3), torch.tensor([1, 2, 0, 1]))
        metrics = tracker.compute_epoch_metrics()

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] == pytest.approx(0.4, abs=0.01)

    def test_history_accumulates(self):
        tracker = MetricsTracker()
        for _ in range(3):
            tracker.reset_epoch()
            tracker.update(0.5, torch.randn(2, 3), torch.tensor([0, 1]))
            tracker.compute_epoch_metrics()

        assert len(tracker.get_history()) == 3

    def test_to_json(self, tmp_path):
        tracker = MetricsTracker()
        tracker.reset_epoch()
        tracker.update(0.5, torch.randn(2, 3), torch.tensor([0, 1]))
        tracker.compute_epoch_metrics()

        path = tmp_path / "metrics.json"
        tracker.to_json(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1


class TestEarlyStopping:
    def _make_cfg(self, patience: int = 3):
        from omegaconf import OmegaConf

        return OmegaConf.create({
            "training": {
                "early_stopping": {
                    "enabled": True,
                    "patience": patience,
                    "monitor": "val_loss",
                    "mode": "min",
                }
            }
        })

    def test_no_stop_when_improving(self):
        cb = EarlyStoppingCallback(self._make_cfg(patience=3))
        assert not cb.on_epoch_end({"loss": 1.0})
        assert not cb.on_epoch_end({"loss": 0.9})
        assert not cb.on_epoch_end({"loss": 0.8})
        assert cb.counter == 0

    def test_stop_after_patience(self):
        cb = EarlyStoppingCallback(self._make_cfg(patience=2))
        cb.on_epoch_end({"loss": 1.0})
        cb.on_epoch_end({"loss": 1.1})  # no improvement
        assert cb.counter == 1
        result = cb.on_epoch_end({"loss": 1.2})  # still no improvement
        assert result is True

    def test_disabled(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "training": {
                "early_stopping": {
                    "enabled": False,
                    "patience": 1,
                    "monitor": "val_loss",
                    "mode": "min",
                }
            }
        })
        cb = EarlyStoppingCallback(cfg)
        # Should never stop
        for _ in range(10):
            assert not cb.on_epoch_end({"loss": 999.0})


class TestTrainer:
    def test_fit_completes(self, sample_config, tiny_dataset):
        """2-epoch training on tiny data completes without errors."""
        trainer = Trainer(sample_config)
        results = trainer.fit()

        assert results["training"]["epochs_run"] == 2
        assert "final_test_accuracy" in results["training"]
        assert results["training"]["final_test_accuracy"] >= 0.0

    def test_checkpoint_created(self, sample_config, tiny_dataset):
        """After fit(), checkpoint files exist."""
        trainer = Trainer(sample_config)
        trainer.fit()

        ckpt_dir = Path(sample_config.training.checkpoint_dir)
        assert ckpt_dir.exists()
        assert (ckpt_dir / "last.pt").exists()
        assert (ckpt_dir / "training_summary.json").exists()

    def test_summary_format(self, sample_config, tiny_dataset):
        """Training summary contains expected keys."""
        trainer = Trainer(sample_config)
        results = trainer.fit()

        assert "experiment_name" in results
        assert "model_parameters" in results
        assert "dataset" in results
        assert "training" in results
        assert "timing" in results
        assert "config" in results

    def test_reproducibility(self, sample_config, tiny_dataset):
        """Two runs with the same seed produce identical val losses."""
        trainer1 = Trainer(sample_config)
        results1 = trainer1.fit()

        trainer2 = Trainer(sample_config)
        results2 = trainer2.fit()

        assert results1["training"]["best_val_accuracy"] == results2["training"]["best_val_accuracy"]
