"""Shared test fixtures for biometric tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from PIL import Image


def _create_bmp(path: Path, width: int, height: int, mode: str = "L") -> None:
    """Create a synthetic BMP image for testing."""
    if mode == "L":
        arr = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    else:
        arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    Image.fromarray(arr, mode=mode).save(str(path))


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    """Create a minimal 3-subject dataset for testing.

    Structure mirrors the real dataset:
        {tmp_path}/
            1/
                Fingerprint/ -> 2 fingerprint BMPs
                left/        -> 2 iris BMPs
                right/       -> 2 iris BMPs
            2/ ...
            3/ ...
    """
    for subject_id in range(1, 4):
        gender = "M" if subject_id <= 2 else "F"
        base = tmp_path / str(subject_id)

        # Fingerprints
        fp_dir = base / "Fingerprint"
        fp_dir.mkdir(parents=True)
        for hand in ["Left", "Right"]:
            fname = f"{subject_id}__{gender}_{hand}_index_finger.BMP"
            _create_bmp(fp_dir / fname, 96, 103, mode="L")

        # Iris
        for eye in ["left", "right"]:
            eye_dir = base / eye
            eye_dir.mkdir(parents=True)
            suffix = "l" if eye == "left" else "r"
            for i in range(1, 3):
                fname = f"test{suffix}{i}.bmp"
                _create_bmp(eye_dir / fname, 320, 240, mode="RGB")

    return tmp_path


@pytest.fixture
def sample_config(tiny_dataset: Path):
    """Return a DictConfig pointing to the tiny_dataset."""
    return OmegaConf.create({
        "seed": 42,
        "experiment_name": "test",
        "data": {
            "root_dir": str(tiny_dataset),
            "preprocessed_dir": str(tiny_dataset / "preprocessed"),
            "train_split": 0.7,
            "val_split": 0.15,
            "test_split": 0.15,
            "num_workers": 0,
            "batch_size": 2,
            "pin_memory": False,
            "prefetch_factor": 2,
            "fingerprint": {
                "resize": [64, 64],
                "normalize_mean": [0.5],
                "normalize_std": [0.5],
            },
            "iris": {
                "resize": [64, 64],
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
            "augmentation": {
                "enabled": False,
                "rotation_degrees": 0,
                "horizontal_flip_prob": 0.0,
                "color_jitter_brightness": 0.0,
                "color_jitter_contrast": 0.0,
            },
            "preprocessing": {
                "num_ray_workers": 2,
                "compute_quality_metrics": True,
                "output_format": "parquet",
            },
        },
        "model": {
            "name": "multimodal_cnn",
            "num_classes": 3,
            "fingerprint_encoder": {
                "in_channels": 1,
                "base_filters": 16,
                "num_blocks": 2,
                "embedding_dim": 64,
                "dropout": 0.1,
            },
            "iris_encoder": {
                "in_channels": 3,
                "base_filters": 16,
                "num_blocks": 2,
                "embedding_dim": 64,
                "dropout": 0.1,
            },
            "fusion": {
                "strategy": "concatenation",
                "hidden_dim": 64,
                "dropout": 0.1,
            },
        },
        "training": {
            "epochs": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "optimizer": "adam",
            "scheduler": "cosine",
            "warmup_epochs": 0,
            "grad_clip_norm": 1.0,
            "checkpoint_dir": str(tiny_dataset / "checkpoints"),
            "save_top_k": 1,
            "log_every_n_steps": 1,
            "early_stopping": {
                "enabled": False,
                "patience": 2,
                "monitor": "val_loss",
                "mode": "min",
            },
        },
    })
