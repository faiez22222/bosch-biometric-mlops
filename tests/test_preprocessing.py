"""Tests for Ray preprocessing and PyArrow storage."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from biometric.data.arrow_store import CATALOG_SCHEMA, ArrowBiometricStore
from biometric.data.preprocessing import _compute_image_quality, _process_single_subject
from biometric.data.utils import discover_dataset

# Mark all tests in this module; Ray tests can be slow.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class TestImageQuality:
    def test_returns_expected_keys(self, tiny_dataset: Path):
        from PIL import Image

        subjects = discover_dataset(tiny_dataset)
        fp_path = list(subjects[0]["fingerprints"].values())[0]
        img = Image.open(fp_path)
        metrics = _compute_image_quality(img)

        assert "brightness_mean" in metrics
        assert "contrast_std" in metrics
        assert "blur_score" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_brightness_range(self, tiny_dataset: Path):
        from PIL import Image

        subjects = discover_dataset(tiny_dataset)
        fp_path = list(subjects[0]["fingerprints"].values())[0]
        img = Image.open(fp_path)
        metrics = _compute_image_quality(img)

        assert 0 <= metrics["brightness_mean"] <= 255


class TestProcessSingleSubject:
    def test_record_count(self, tiny_dataset: Path):
        subjects = discover_dataset(tiny_dataset)
        records = _process_single_subject(subjects[0], compute_quality=True)

        # Subject 1: 2 fingerprints + 2 left iris + 2 right iris = 6
        assert len(records) == 6

    def test_record_fields(self, tiny_dataset: Path):
        subjects = discover_dataset(tiny_dataset)
        records = _process_single_subject(subjects[0], compute_quality=False)

        for record in records:
            assert "subject_id" in record
            assert "modality" in record
            assert record["modality"] in ("fingerprint", "iris")
            assert "image_path" in record
            assert "width" in record
            assert "height" in record


class TestArrowStore:
    def test_write_read_roundtrip(self, tiny_dataset: Path):
        subjects = discover_dataset(tiny_dataset)
        records = _process_single_subject(subjects[0], compute_quality=True)

        store = ArrowBiometricStore()
        out_path = tiny_dataset / "test_catalog.parquet"
        store.write(records, out_path)

        table = store.read(out_path)
        assert table.num_rows == len(records)
        assert table.schema.equals(CATALOG_SCHEMA)

    def test_filtered_read(self, tiny_dataset: Path):
        subjects = discover_dataset(tiny_dataset)
        all_records = []
        for s in subjects:
            all_records.extend(_process_single_subject(s, compute_quality=False))

        store = ArrowBiometricStore()
        out_path = tiny_dataset / "test_catalog.parquet"
        store.write(all_records, out_path)

        # Filter to iris only
        iris_table = store.read(
            out_path, filters=[("modality", "=", "iris")]
        )
        assert iris_table.num_rows > 0
        modalities = iris_table.column("modality").to_pylist()
        assert all(m == "iris" for m in modalities)

    def test_column_pruning(self, tiny_dataset: Path):
        subjects = discover_dataset(tiny_dataset)
        records = _process_single_subject(subjects[0], compute_quality=False)

        store = ArrowBiometricStore()
        out_path = tiny_dataset / "test_catalog.parquet"
        store.write(records, out_path)

        table = store.read(out_path, columns=["subject_id", "modality"])
        assert table.num_columns == 2
        assert "subject_id" in table.column_names
        assert "modality" in table.column_names

    def test_summary(self, tiny_dataset: Path):
        subjects = discover_dataset(tiny_dataset)
        all_records = []
        for s in subjects:
            all_records.extend(_process_single_subject(s, compute_quality=True))

        store = ArrowBiometricStore()
        out_path = tiny_dataset / "test_catalog.parquet"
        store.write(all_records, out_path)

        summary = store.summary(out_path)
        assert summary["total_records"] == len(all_records)
        assert summary["num_subjects"] == 3
        assert "fingerprint" in summary["modality_counts"]
        assert "iris" in summary["modality_counts"]

    def test_empty_records_warning(self, tiny_dataset: Path, caplog):
        store = ArrowBiometricStore()
        out_path = tiny_dataset / "empty.parquet"
        store.write([], out_path)
        assert "No records" in caplog.text
