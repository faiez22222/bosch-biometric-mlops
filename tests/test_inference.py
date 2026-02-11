"""Tests for the inference pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from biometric.inference.predictor import BiometricPredictor
from biometric.training.trainer import Trainer


@pytest.fixture
def trained_checkpoint(sample_config, tiny_dataset, tmp_path):
    """Train a model for 2 epochs and return the best checkpoint path."""
    ckpt_dir = tmp_path / "ckpts"
    sample_config.training.checkpoint_dir = str(ckpt_dir)
    trainer = Trainer(sample_config)
    trainer.fit()
    return ckpt_dir / "last.pt"


class TestBiometricPredictor:
    def test_from_checkpoint(self, trained_checkpoint):
        """Load a predictor from a training checkpoint."""
        predictor = BiometricPredictor.from_checkpoint(trained_checkpoint)
        assert predictor.model.training is False  # eval mode

    def test_predict_returns_expected_format(self, trained_checkpoint, tiny_dataset):
        """predict() returns dict with the expected keys."""
        predictor = BiometricPredictor.from_checkpoint(trained_checkpoint)

        # Pick a fingerprint and iris from the tiny dataset
        fp_path = list((tiny_dataset / "1" / "Fingerprint").glob("*.BMP"))[0]
        iris_path = list((tiny_dataset / "1" / "left").glob("*.bmp"))[0]

        result = predictor.predict(fp_path, iris_path)

        assert "predicted_subject_id" in result
        assert "confidence" in result
        assert "top_k" in result
        assert isinstance(result["top_k"], list)
        assert len(result["top_k"]) > 0
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_batch(self, trained_checkpoint, tiny_dataset):
        """predict_batch processes multiple samples."""
        predictor = BiometricPredictor.from_checkpoint(trained_checkpoint)

        fp_path = list((tiny_dataset / "1" / "Fingerprint").glob("*.BMP"))[0]
        iris_path = list((tiny_dataset / "1" / "left").glob("*.bmp"))[0]

        samples = [
            {"fingerprint": fp_path, "iris": iris_path},
            {"fingerprint": fp_path, "iris": iris_path},
        ]
        results = predictor.predict_batch(samples)

        assert len(results) == 2
        for r in results:
            assert "predicted_subject_id" in r
            assert "confidence" in r

    def test_confidence_sums_to_approximately_one(self, trained_checkpoint, tiny_dataset):
        """Softmax probabilities in top_k should sum to ~1.0 when k=num_classes."""
        predictor = BiometricPredictor.from_checkpoint(trained_checkpoint)

        fp_path = list((tiny_dataset / "1" / "Fingerprint").glob("*.BMP"))[0]
        iris_path = list((tiny_dataset / "1" / "left").glob("*.bmp"))[0]

        # Request all classes
        result = predictor.predict(fp_path, iris_path, top_k=100)
        total_conf = sum(item["confidence"] for item in result["top_k"])
        assert abs(total_conf - 1.0) < 0.01

    def test_checkpoint_not_found_raises(self):
        """from_checkpoint raises FileNotFoundError for missing path."""
        with pytest.raises(FileNotFoundError):
            BiometricPredictor.from_checkpoint("/nonexistent/checkpoint.pt")

    def test_empty_batch(self, trained_checkpoint):
        """predict_batch with empty list returns empty list."""
        predictor = BiometricPredictor.from_checkpoint(trained_checkpoint)
        results = predictor.predict_batch([])
        assert results == []
