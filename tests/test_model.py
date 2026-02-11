"""Tests for model architecture: encoders, fusion, and full network."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from biometric.models.fingerprint_encoder import FingerprintEncoder
from biometric.models.fusion import MultimodalFusion
from biometric.models.iris_encoder import IrisEncoder
from biometric.models.multimodal_net import MultimodalBiometricNet


def _make_model_cfg(num_classes: int = 3):
    return OmegaConf.create({
        "model": {
            "name": "multimodal_cnn",
            "num_classes": num_classes,
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
        }
    })


class TestFingerprintEncoder:
    def test_output_shape(self):
        cfg = _make_model_cfg()
        encoder = FingerprintEncoder(cfg)
        x = torch.randn(4, 1, 64, 64)
        out = encoder(x)
        assert out.shape == (4, 64)

    def test_different_input_sizes(self):
        cfg = _make_model_cfg()
        encoder = FingerprintEncoder(cfg)
        # AdaptiveAvgPool makes it input-size agnostic
        for size in [(32, 32), (64, 64), (128, 128)]:
            x = torch.randn(2, 1, *size)
            out = encoder(x)
            assert out.shape == (2, 64)

    def test_batch_size_one(self):
        cfg = _make_model_cfg()
        encoder = FingerprintEncoder(cfg)
        x = torch.randn(1, 1, 64, 64)
        out = encoder(x)
        assert out.shape == (1, 64)


class TestIrisEncoder:
    def test_output_shape(self):
        cfg = _make_model_cfg()
        encoder = IrisEncoder(cfg)
        x = torch.randn(4, 3, 64, 64)
        out = encoder(x)
        assert out.shape == (4, 64)

    def test_rgb_input(self):
        cfg = _make_model_cfg()
        encoder = IrisEncoder(cfg)
        x = torch.randn(2, 3, 112, 112)
        out = encoder(x)
        assert out.shape == (2, 64)


class TestMultimodalFusion:
    def test_concatenation(self):
        cfg = _make_model_cfg(num_classes=5)
        fusion = MultimodalFusion(cfg)
        fp = torch.randn(4, 64)
        iris = torch.randn(4, 64)
        out = fusion(fp, iris)
        assert out.shape == (4, 5)

    def test_single_modality(self):
        cfg = OmegaConf.create({
            "model": {
                "num_classes": 5,
                "fingerprint_encoder": {
                    "embedding_dim": 64,
                },
                "iris_encoder": None,
                "fusion": {
                    "strategy": "single_modality",
                    "hidden_dim": 32,
                    "dropout": 0.1,
                },
            }
        })
        fusion = MultimodalFusion(cfg)
        fp = torch.randn(4, 64)
        out = fusion(fp_features=fp)
        assert out.shape == (4, 5)


class TestMultimodalBiometricNet:
    def test_forward_multimodal(self):
        cfg = _make_model_cfg(num_classes=3)
        model = MultimodalBiometricNet(cfg)
        fp = torch.randn(4, 1, 64, 64)
        iris = torch.randn(4, 3, 64, 64)
        logits = model(fp, iris)
        assert logits.shape == (4, 3)

    def test_count_parameters(self):
        cfg = _make_model_cfg()
        model = MultimodalBiometricNet(cfg)
        counts = model.count_parameters()
        assert "fingerprint_encoder" in counts
        assert "iris_encoder" in counts
        assert "fusion" in counts
        assert "total" in counts
        assert counts["total"] > 0
        assert counts["total"] == (
            counts["fingerprint_encoder"] + counts["iris_encoder"] + counts["fusion"]
        )

    def test_gradient_flow(self):
        cfg = _make_model_cfg(num_classes=3)
        model = MultimodalBiometricNet(cfg)
        fp = torch.randn(2, 1, 64, 64)
        iris = torch.randn(2, 3, 64, 64)
        labels = torch.tensor([0, 1])

        logits = model(fp, iris)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_eval_mode(self):
        cfg = _make_model_cfg()
        model = MultimodalBiometricNet(cfg)
        model.eval()
        fp = torch.randn(2, 1, 64, 64)
        iris = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            logits = model(fp, iris)
        assert logits.shape == (2, 3)
