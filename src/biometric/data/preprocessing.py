"""Ray-based parallel preprocessing pipeline for biometric images.

This module provides distributed preprocessing of the raw dataset using Ray.
Each subject is processed independently as a Ray remote task, enabling linear
speedup on multi-core machines. Results are collected and written to a PyArrow
Parquet catalog for efficient downstream access.

Typical usage:
    python scripts/preprocess.py              # Uses default config
    python scripts/preprocess.py data.preprocessing.num_ray_workers=8
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import ray
from omegaconf import DictConfig
from PIL import Image

from biometric.data.arrow_store import ArrowBiometricStore
from biometric.data.utils import discover_dataset

logger = logging.getLogger(__name__)


def _compute_image_quality(image: Image.Image) -> dict[str, float]:
    """Compute basic quality metrics for an image.

    These metrics are useful for filtering low-quality samples in a
    production pipeline (e.g., reject blurry or underexposed captures).

    Args:
        image: PIL Image to analyze.

    Returns:
        Dict with brightness_mean, contrast_std, and blur_score.
    """
    arr = np.array(image, dtype=np.float32)

    # Mean brightness (average pixel intensity across all channels)
    brightness_mean = float(np.mean(arr))

    # Contrast (standard deviation of pixel values)
    contrast_std = float(np.std(arr))

    # Blur score via Laplacian variance (higher = sharper)
    # Simplified: use variance of second-order differences as a proxy
    if arr.ndim == 3:
        gray = np.mean(arr, axis=2)
    else:
        gray = arr
    laplacian = (
        gray[2:, 1:-1]
        + gray[:-2, 1:-1]
        + gray[1:-1, 2:]
        + gray[1:-1, :-2]
        - 4 * gray[1:-1, 1:-1]
    )
    blur_score = float(np.var(laplacian))

    return {
        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
        "blur_score": blur_score,
    }


def _process_single_subject(
    subject: dict[str, Any],
    compute_quality: bool,
) -> list[dict[str, Any]]:
    """Process all images for a single subject (runs inside a Ray worker).

    Args:
        subject: Subject record from discover_dataset.
        compute_quality: Whether to compute quality metrics.

    Returns:
        List of per-image metadata records.
    """
    records: list[dict[str, Any]] = []
    sid = subject["subject_id"]
    gender = subject["gender"]

    # Process fingerprints
    for key, fp_path in sorted(subject["fingerprints"].items()):
        hand, finger = key.split("_", 1)
        image = Image.open(fp_path)
        w, h = image.size

        record: dict[str, Any] = {
            "subject_id": sid,
            "gender": gender,
            "modality": "fingerprint",
            "side": hand,
            "detail": finger,
            "sample_idx": 0,
            "image_path": str(fp_path),
            "width": w,
            "height": h,
        }

        if compute_quality:
            record.update(_compute_image_quality(image))
        else:
            record.update(
                {"brightness_mean": 0.0, "contrast_std": 0.0, "blur_score": 0.0}
            )

        records.append(record)

    # Process iris images
    for eye_side in ["left", "right"]:
        for sample_idx, iris_path in enumerate(subject["iris"][eye_side]):
            image = Image.open(iris_path).convert("RGB")
            w, h = image.size

            record = {
                "subject_id": sid,
                "gender": gender,
                "modality": "iris",
                "side": eye_side,
                "detail": "eye",
                "sample_idx": sample_idx,
                "image_path": str(iris_path),
                "width": w,
                "height": h,
            }

            if compute_quality:
                record.update(_compute_image_quality(image))
            else:
                record.update(
                    {"brightness_mean": 0.0, "contrast_std": 0.0, "blur_score": 0.0}
                )

            records.append(record)

    return records


# Define the Ray remote function at module level for clean serialization.
@ray.remote
def _process_subject_remote(
    subject: dict[str, Any],
    compute_quality: bool,
) -> list[dict[str, Any]]:
    """Ray remote wrapper around _process_single_subject."""
    return _process_single_subject(subject, compute_quality)


def _process_serial(
    subjects: list[dict[str, Any]],
    compute_quality: bool,
) -> tuple[list[dict[str, Any]], float]:
    """Process all subjects serially (baseline for benchmarking).

    Returns:
        Tuple of (all_records, elapsed_seconds).
    """
    start = time.perf_counter()
    all_records: list[dict[str, Any]] = []
    for subject in subjects:
        records = _process_single_subject(subject, compute_quality)
        all_records.extend(records)
    elapsed = time.perf_counter() - start
    return all_records, elapsed


def _process_parallel(
    subjects: list[dict[str, Any]],
    compute_quality: bool,
    num_workers: int,
) -> tuple[list[dict[str, Any]], float]:
    """Process all subjects in parallel using Ray.

    Returns:
        Tuple of (all_records, elapsed_seconds).
    """
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init(
            num_cpus=num_workers,
            log_to_driver=False,
            ignore_reinit_error=True,
        )

    start = time.perf_counter()

    # Submit all tasks
    futures = [
        _process_subject_remote.remote(subject, compute_quality)
        for subject in subjects
    ]

    # Collect results
    all_records: list[dict[str, Any]] = []
    results = ray.get(futures)
    for records in results:
        all_records.extend(records)

    elapsed = time.perf_counter() - start
    return all_records, elapsed


def run_parallel_preprocessing(cfg: DictConfig) -> dict[str, Any]:
    """Execute the full preprocessing pipeline.

    Steps:
    1. Discover all subjects from the raw dataset.
    2. Run serial baseline (for performance comparison).
    3. Run Ray parallel preprocessing.
    4. Write results to PyArrow Parquet catalog.
    5. Return timing report.

    Args:
        cfg: Full Hydra config.

    Returns:
        Dict with timing stats and record counts.
    """
    data_cfg = cfg.data
    preproc_cfg = data_cfg.preprocessing
    compute_quality = preproc_cfg.compute_quality_metrics
    num_workers = preproc_cfg.num_ray_workers

    # Discover dataset
    subjects = discover_dataset(data_cfg.root_dir)
    max_subjects = getattr(data_cfg, "max_subjects", None)
    if max_subjects is not None:
        subjects = subjects[:max_subjects]

    logger.info("Preprocessing %d subjects...", len(subjects))

    # --- Serial baseline ---
    logger.info("Running serial baseline...")
    serial_records, serial_time = _process_serial(subjects, compute_quality)
    logger.info(
        "Serial: %d records in %.2fs (%.1f records/s)",
        len(serial_records),
        serial_time,
        len(serial_records) / serial_time if serial_time > 0 else 0,
    )

    # --- Parallel with Ray ---
    logger.info("Running Ray parallel with %d workers...", num_workers)
    parallel_records, parallel_time = _process_parallel(
        subjects, compute_quality, num_workers
    )
    logger.info(
        "Parallel: %d records in %.2fs (%.1f records/s)",
        len(parallel_records),
        parallel_time,
        len(parallel_records) / parallel_time if parallel_time > 0 else 0,
    )

    # Speedup analysis
    speedup = serial_time / parallel_time if parallel_time > 0 else float("inf")
    efficiency = speedup / num_workers * 100
    logger.info(
        "Speedup: %.2fx with %d workers (efficiency: %.1f%%)",
        speedup,
        num_workers,
        efficiency,
    )

    # --- Write to Arrow/Parquet ---
    output_dir = Path(data_cfg.preprocessed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "biometric_catalog.parquet"

    store = ArrowBiometricStore()
    store.write(parallel_records, output_path)
    logger.info("Wrote catalog to %s", output_path)

    # Cleanup Ray
    if ray.is_initialized():
        ray.shutdown()

    report = {
        "num_subjects": len(subjects),
        "num_records": len(parallel_records),
        "serial_time_s": round(serial_time, 3),
        "parallel_time_s": round(parallel_time, 3),
        "speedup": round(speedup, 2),
        "efficiency_pct": round(efficiency, 1),
        "num_workers": num_workers,
        "output_path": str(output_path),
    }

    logger.info("Preprocessing report: %s", report)
    return report
