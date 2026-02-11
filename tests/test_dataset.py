"""Tests for dataset discovery, parsing, splitting, and Dataset classes."""

from __future__ import annotations

from pathlib import Path

import pytest

from biometric.data.dataset import (
    FingerprintDataset,
    IrisDataset,
    MultimodalBiometricDataset,
    build_label_map,
)
from biometric.data.transforms import get_fingerprint_transforms, get_iris_transforms
from biometric.data.utils import (
    create_splits,
    discover_dataset,
    discover_subject,
    parse_fingerprint_filename,
)


class TestParseFingerprint:
    def test_valid_male(self):
        result = parse_fingerprint_filename("1__M_Left_index_finger.BMP")
        assert result == {
            "subject_id": 1,
            "gender": "M",
            "hand": "Left",
            "finger_type": "index",
        }

    def test_valid_female(self):
        result = parse_fingerprint_filename("25__F_Right_thumb_finger.BMP")
        assert result is not None
        assert result["gender"] == "F"
        assert result["hand"] == "Right"

    def test_invalid_filename(self):
        assert parse_fingerprint_filename("desktop.ini") is None
        assert parse_fingerprint_filename("random.txt") is None


class TestDiscoverDataset:
    def test_discover_tiny(self, tiny_dataset: Path):
        subjects = discover_dataset(tiny_dataset)
        assert len(subjects) == 3

    def test_subject_structure(self, tiny_dataset: Path):
        subjects = discover_dataset(tiny_dataset)
        s1 = next(s for s in subjects if s["subject_id"] == 1)

        assert s1["gender"] == "M"
        assert len(s1["fingerprints"]) == 2  # Left + Right index
        assert len(s1["iris"]["left"]) == 2
        assert len(s1["iris"]["right"]) == 2

    def test_nonexistent_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            discover_dataset("/nonexistent/path")


class TestCreateSplits:
    def test_split_ratios(self, tiny_dataset: Path):
        subjects = discover_dataset(tiny_dataset)
        splits = create_splits(subjects, 0.7, 0.15, 0.15, seed=42)

        # With 3 subjects, at least 1 in train, and the rest split
        assert len(splits["train"]) >= 1
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 3

    def test_no_subject_overlap(self, tiny_dataset: Path):
        subjects = discover_dataset(tiny_dataset)
        splits = create_splits(subjects, 0.7, 0.15, 0.15, seed=42)

        train_ids = {s["subject_id"] for s in splits["train"]}
        val_ids = {s["subject_id"] for s in splits["val"]}
        test_ids = {s["subject_id"] for s in splits["test"]}

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_deterministic(self, tiny_dataset: Path):
        subjects = discover_dataset(tiny_dataset)
        splits1 = create_splits(subjects, 0.7, 0.15, 0.15, seed=42)
        splits2 = create_splits(subjects, 0.7, 0.15, 0.15, seed=42)

        ids1 = [s["subject_id"] for s in splits1["train"]]
        ids2 = [s["subject_id"] for s in splits2["train"]]
        assert ids1 == ids2


class TestLabelMap:
    def test_contiguous_labels(self, tiny_dataset: Path):
        subjects = discover_dataset(tiny_dataset)
        label_map = build_label_map(subjects)

        assert set(label_map.values()) == {0, 1, 2}
        assert len(label_map) == 3


class TestFingerprintDataset:
    def test_length(self, tiny_dataset: Path, sample_config):
        subjects = discover_dataset(tiny_dataset)
        label_map = build_label_map(subjects)
        tfm = get_fingerprint_transforms(sample_config.data, is_train=False)
        ds = FingerprintDataset(subjects, label_map, transform=tfm)

        # 3 subjects x 2 fingerprints each = 6
        assert len(ds) == 6

    def test_getitem_shape(self, tiny_dataset: Path, sample_config):
        subjects = discover_dataset(tiny_dataset)
        label_map = build_label_map(subjects)
        tfm = get_fingerprint_transforms(sample_config.data, is_train=False)
        ds = FingerprintDataset(subjects, label_map, transform=tfm)

        sample = ds[0]
        assert sample["fingerprint"].shape == (1, 64, 64)
        assert isinstance(sample["label"], int)
        assert "subject_id" in sample["metadata"]


class TestIrisDataset:
    def test_length(self, tiny_dataset: Path, sample_config):
        subjects = discover_dataset(tiny_dataset)
        label_map = build_label_map(subjects)
        tfm = get_iris_transforms(sample_config.data, is_train=False)
        ds = IrisDataset(subjects, label_map, transform=tfm)

        # 3 subjects x 4 iris images each = 12
        assert len(ds) == 12

    def test_getitem_shape(self, tiny_dataset: Path, sample_config):
        subjects = discover_dataset(tiny_dataset)
        label_map = build_label_map(subjects)
        tfm = get_iris_transforms(sample_config.data, is_train=False)
        ds = IrisDataset(subjects, label_map, transform=tfm)

        sample = ds[0]
        assert sample["iris"].shape == (3, 64, 64)


class TestMultimodalDataset:
    def test_cross_product_length(self, tiny_dataset: Path, sample_config):
        subjects = discover_dataset(tiny_dataset)
        label_map = build_label_map(subjects)
        fp_tfm = get_fingerprint_transforms(sample_config.data, is_train=False)
        iris_tfm = get_iris_transforms(sample_config.data, is_train=False)
        ds = MultimodalBiometricDataset(
            subjects, label_map, fp_tfm, iris_tfm
        )

        # 3 subjects x 2 fingerprints x 4 iris = 24
        assert len(ds) == 24

    def test_getitem_multimodal(self, tiny_dataset: Path, sample_config):
        subjects = discover_dataset(tiny_dataset)
        label_map = build_label_map(subjects)
        fp_tfm = get_fingerprint_transforms(sample_config.data, is_train=False)
        iris_tfm = get_iris_transforms(sample_config.data, is_train=False)
        ds = MultimodalBiometricDataset(
            subjects, label_map, fp_tfm, iris_tfm
        )

        sample = ds[0]
        assert sample["fingerprint"].shape == (1, 64, 64)
        assert sample["iris"].shape == (3, 64, 64)
        assert isinstance(sample["label"], int)
        assert "finger_type" in sample["metadata"]
        assert "eye_side" in sample["metadata"]
