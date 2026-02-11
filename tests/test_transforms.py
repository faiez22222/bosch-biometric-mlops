"""Tests for image transform pipelines."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from biometric.data.transforms import get_fingerprint_transforms, get_iris_transforms


def _make_cfg(augmentation_enabled: bool = False):
    return OmegaConf.create({
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
            "enabled": augmentation_enabled,
            "rotation_degrees": 10,
            "horizontal_flip_prob": 0.5,
            "color_jitter_brightness": 0.2,
            "color_jitter_contrast": 0.2,
        },
    })


class TestFingerprintTransforms:
    def test_eval_output_shape(self):
        cfg = _make_cfg(augmentation_enabled=False)
        tfm = get_fingerprint_transforms(cfg, is_train=False)
        img = Image.fromarray(np.random.randint(0, 255, (103, 96), dtype=np.uint8))
        result = tfm(img)
        assert result.shape == (1, 64, 64)

    def test_train_output_shape(self):
        cfg = _make_cfg(augmentation_enabled=True)
        tfm = get_fingerprint_transforms(cfg, is_train=True)
        img = Image.fromarray(np.random.randint(0, 255, (103, 96), dtype=np.uint8))
        result = tfm(img)
        assert result.shape == (1, 64, 64)

    def test_output_is_tensor(self):
        cfg = _make_cfg()
        tfm = get_fingerprint_transforms(cfg, is_train=False)
        img = Image.fromarray(np.random.randint(0, 255, (103, 96), dtype=np.uint8))
        result = tfm(img)
        assert isinstance(result, torch.Tensor)


class TestIrisTransforms:
    def test_eval_output_shape(self):
        cfg = _make_cfg(augmentation_enabled=False)
        tfm = get_iris_transforms(cfg, is_train=False)
        img = Image.fromarray(
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        )
        result = tfm(img)
        assert result.shape == (3, 64, 64)

    def test_train_output_shape(self):
        cfg = _make_cfg(augmentation_enabled=True)
        tfm = get_iris_transforms(cfg, is_train=True)
        img = Image.fromarray(
            np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        )
        result = tfm(img)
        assert result.shape == (3, 64, 64)
