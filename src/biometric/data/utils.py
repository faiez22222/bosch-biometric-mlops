"""Utilities for dataset discovery, path parsing, and split creation."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import random as _random

from sklearn.model_selection import StratifiedShuffleSplit

logger = logging.getLogger(__name__)

# Fingerprint filename pattern: {id}__{M/F}_{Left/Right}_{finger}_finger.BMP
_FINGERPRINT_PATTERN = re.compile(
    r"^(\d+)__([MF])_(Left|Right)_(\w+)_finger\.BMP$",
    re.IGNORECASE,
)

# Canonical finger ordering for deterministic iteration
FINGER_TYPES = ["thumb", "index", "middle", "ring", "little"]
HANDS = ["Left", "Right"]
EYE_SIDES = ["left", "right"]
SAMPLES_PER_EYE = 5


def parse_fingerprint_filename(filename: str) -> dict[str, Any] | None:
    """Extract metadata from a fingerprint filename.

    Args:
        filename: Filename like '1__M_Left_index_finger.BMP'.

    Returns:
        Dict with subject_id, gender, hand, finger_type, or None if no match.
    """
    match = _FINGERPRINT_PATTERN.match(filename)
    if not match:
        return None
    return {
        "subject_id": int(match.group(1)),
        "gender": match.group(2),
        "hand": match.group(3),
        "finger_type": match.group(4),
    }


def discover_subject(subject_dir: Path) -> dict[str, Any] | None:
    """Discover all biometric data for a single subject directory.

    Expected structure:
        {subject_id}/
            Fingerprint/  -> {id}__{M/F}_{Hand}_{finger}_finger.BMP
            left/         -> {name}l{1-5}.bmp  (left iris images)
            right/        -> {name}r{1-5}.bmp  (right iris images)

    Args:
        subject_dir: Path to a numbered subject directory.

    Returns:
        Subject record dict, or None if the directory is invalid.
    """
    if not subject_dir.is_dir():
        return None

    # Parse subject_id from directory name
    try:
        subject_id = int(subject_dir.name)
    except ValueError:
        return None

    # --- Fingerprints ---
    fp_dir = subject_dir / "Fingerprint"
    fingerprints: dict[str, Path] = {}
    gender: str | None = None

    if fp_dir.is_dir():
        for fp_file in sorted(fp_dir.iterdir()):
            meta = parse_fingerprint_filename(fp_file.name)
            if meta is None:
                continue
            key = f"{meta['hand'].lower()}_{meta['finger_type'].lower()}"
            fingerprints[key] = fp_file
            if gender is None:
                gender = meta["gender"]

    # --- Iris images ---
    iris: dict[str, list[Path]] = {"left": [], "right": []}
    for eye_side in EYE_SIDES:
        eye_dir = subject_dir / eye_side
        if not eye_dir.is_dir():
            continue
        # Collect all .bmp files (excluding desktop.ini, etc.)
        bmp_files = sorted(
            f for f in eye_dir.iterdir() if f.suffix.lower() == ".bmp"
        )
        iris[eye_side] = bmp_files

    # Validate minimum data
    if not fingerprints and not iris["left"] and not iris["right"]:
        logger.warning("Subject %d has no biometric data, skipping.", subject_id)
        return None

    return {
        "subject_id": subject_id,
        "gender": gender or "U",  # U = unknown if no fingerprints
        "fingerprints": fingerprints,
        "iris": iris,
    }


def discover_dataset(root_dir: str | Path) -> list[dict[str, Any]]:
    """Walk the entire dataset tree and return per-subject records.

    Args:
        root_dir: Path to the root dataset directory containing numbered folders.

    Returns:
        Sorted list of subject record dicts.

    Raises:
        FileNotFoundError: If root_dir does not exist.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    subjects: list[dict[str, Any]] = []
    for child in sorted(root.iterdir(), key=lambda p: p.name.zfill(5)):
        record = discover_subject(child)
        if record is not None:
            subjects.append(record)

    logger.info(
        "Discovered %d subjects with %d fingerprints and %d iris images.",
        len(subjects),
        sum(len(s["fingerprints"]) for s in subjects),
        sum(len(s["iris"]["left"]) + len(s["iris"]["right"]) for s in subjects),
    )
    return subjects


def create_splits(
    subjects: list[dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split subjects into train/val/test sets, stratified by gender.

    Splitting is done at the *subject* level to prevent data leakage
    (no images from the same person appear in both train and test).

    Args:
        subjects: List of subject records from discover_dataset.
        train_ratio: Fraction of subjects for training.
        val_ratio: Fraction of subjects for validation.
        test_ratio: Fraction of subjects for testing.
        seed: Random seed for reproducible splits.

    Returns:
        Dict with keys 'train', 'val', 'test', each a list of subject records.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    n = len(subjects)
    genders = [s["gender"] for s in subjects]

    # Check if stratified split is feasible (each class needs >=2 members).
    from collections import Counter

    gender_counts = Counter(genders)
    can_stratify = all(c >= 2 for c in gender_counts.values()) and n >= 4

    if can_stratify:
        subject_ids = list(range(n))

        # First split: train vs (val + test)
        holdout_ratio = val_ratio + test_ratio
        splitter1 = StratifiedShuffleSplit(
            n_splits=1, test_size=holdout_ratio, random_state=seed
        )
        train_idx, holdout_idx = next(splitter1.split(subject_ids, genders))

        # Second split: val vs test (within holdout)
        holdout_genders = [genders[i] for i in holdout_idx]
        relative_test_ratio = test_ratio / holdout_ratio

        if len(holdout_idx) >= 2:
            try:
                splitter2 = StratifiedShuffleSplit(
                    n_splits=1, test_size=relative_test_ratio, random_state=seed
                )
                val_idx_local, test_idx_local = next(
                    splitter2.split(list(range(len(holdout_idx))), holdout_genders)
                )
                val_idx = holdout_idx[val_idx_local]
                test_idx = holdout_idx[test_idx_local]
            except ValueError:
                # Fallback: split holdout in half
                mid = len(holdout_idx) // 2
                val_idx = holdout_idx[:mid] if mid > 0 else holdout_idx[:1]
                test_idx = holdout_idx[mid:] if mid > 0 else holdout_idx[1:]
        else:
            val_idx = holdout_idx
            test_idx = np.array([], dtype=int)
    else:
        # Fallback: simple random shuffle split for very small datasets.
        logger.warning(
            "Dataset too small for stratified split (%d subjects). "
            "Using simple random split.",
            n,
        )
        indices = list(range(n))
        rng = _random.Random(seed)
        rng.shuffle(indices)

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio)) if n - n_train >= 2 else 0
        # Ensure at least 1 in test if possible
        n_test = n - n_train - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

    splits = {
        "train": [subjects[i] for i in train_idx],
        "val": [subjects[i] for i in val_idx],
        "test": [subjects[i] for i in test_idx],
    }

    for split_name, split_subjects in splits.items():
        logger.info(
            "Split '%s': %d subjects (M=%d, F=%d)",
            split_name,
            len(split_subjects),
            sum(1 for s in split_subjects if s["gender"] == "M"),
            sum(1 for s in split_subjects if s["gender"] == "F"),
        )

    return splits
