"""PyTorch Dataset classes for fingerprint, iris, and multimodal biometric data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class FingerprintDataset(Dataset):
    """Single-modality dataset for fingerprint images.

    Each sample is a single fingerprint image paired with its subject label.
    """

    def __init__(
        self,
        subjects: list[dict[str, Any]],
        label_map: dict[int, int],
        transform: transforms.Compose | None = None,
    ) -> None:
        """
        Args:
            subjects: List of subject records from discover_dataset.
            label_map: Mapping from subject_id -> contiguous class index.
            transform: Torchvision transforms to apply.
        """
        self.transform = transform
        self.samples: list[tuple[Path, int, dict[str, str]]] = []

        for subject in subjects:
            sid = subject["subject_id"]
            label = label_map[sid]
            for key, fp_path in sorted(subject["fingerprints"].items()):
                hand, finger = key.split("_", 1)
                meta = {
                    "subject_id": str(sid),
                    "gender": subject["gender"],
                    "hand": hand,
                    "finger_type": finger,
                }
                self.samples.append((fp_path, label, meta))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, label, meta = self.samples[idx]
        image = Image.open(path)

        if self.transform is not None:
            image = self.transform(image)

        return {"fingerprint": image, "label": label, "metadata": meta}


class IrisDataset(Dataset):
    """Single-modality dataset for iris images.

    Each sample is a single iris image paired with its subject label.
    """

    def __init__(
        self,
        subjects: list[dict[str, Any]],
        label_map: dict[int, int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.transform = transform
        self.samples: list[tuple[Path, int, dict[str, str]]] = []

        for subject in subjects:
            sid = subject["subject_id"]
            label = label_map[sid]
            for eye_side in ["left", "right"]:
                for iris_path in subject["iris"][eye_side]:
                    meta = {
                        "subject_id": str(sid),
                        "gender": subject["gender"],
                        "eye_side": eye_side,
                    }
                    self.samples.append((iris_path, label, meta))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, label, meta = self.samples[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {"iris": image, "label": label, "metadata": meta}


class MultimodalBiometricDataset(Dataset):
    """Multimodal dataset that pairs fingerprint and iris samples per subject.

    For each subject, every fingerprint image is paired with every iris image
    to form (fingerprint, iris, label) triplets. This maximizes the number of
    training samples: 45 subjects x 10 fingerprints x 10 iris = 4,500 pairs.

    The pairing is deterministic given the same subject records and ordering.
    """

    def __init__(
        self,
        subjects: list[dict[str, Any]],
        label_map: dict[int, int],
        fingerprint_transform: transforms.Compose | None = None,
        iris_transform: transforms.Compose | None = None,
    ) -> None:
        """
        Args:
            subjects: Subject records from discover_dataset.
            label_map: subject_id -> contiguous class index.
            fingerprint_transform: Transforms for fingerprint images.
            iris_transform: Transforms for iris images.
        """
        self.fingerprint_transform = fingerprint_transform
        self.iris_transform = iris_transform
        self.pairs: list[tuple[Path, Path, int, dict[str, str]]] = []

        for subject in subjects:
            sid = subject["subject_id"]
            label = label_map[sid]

            # Collect all fingerprint paths
            fp_items = sorted(subject["fingerprints"].items())
            # Collect all iris paths
            iris_paths: list[tuple[str, Path]] = []
            for eye_side in ["left", "right"]:
                for iris_path in subject["iris"][eye_side]:
                    iris_paths.append((eye_side, iris_path))

            # Cross-product pairing
            for fp_key, fp_path in fp_items:
                hand, finger = fp_key.split("_", 1)
                for eye_side, iris_path in iris_paths:
                    meta = {
                        "subject_id": str(sid),
                        "gender": subject["gender"],
                        "hand": hand,
                        "finger_type": finger,
                        "eye_side": eye_side,
                    }
                    self.pairs.append((fp_path, iris_path, label, meta))

        logger.info(
            "MultimodalBiometricDataset created with %d pairs from %d subjects.",
            len(self.pairs),
            len(subjects),
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        fp_path, iris_path, label, meta = self.pairs[idx]

        # Load fingerprint (may be grayscale or RGB depending on source)
        fp_image = Image.open(fp_path)
        if self.fingerprint_transform is not None:
            fp_image = self.fingerprint_transform(fp_image)

        # Load iris (always convert to RGB)
        iris_image = Image.open(iris_path).convert("RGB")
        if self.iris_transform is not None:
            iris_image = self.iris_transform(iris_image)

        return {
            "fingerprint": fp_image,
            "iris": iris_image,
            "label": label,
            "metadata": meta,
        }


def build_label_map(subjects: list[dict[str, Any]]) -> dict[int, int]:
    """Create a mapping from subject_id to a contiguous class index [0, N).

    Args:
        subjects: All subjects (across all splits) to ensure consistent mapping.

    Returns:
        Dict mapping subject_id -> class_index.
    """
    sorted_ids = sorted(s["subject_id"] for s in subjects)
    return {sid: idx for idx, sid in enumerate(sorted_ids)}


def collate_multimodal(
    batch: list[dict[str, Any]],
) -> dict[str, torch.Tensor | list[dict[str, str]]]:
    """Custom collate function for MultimodalBiometricDataset.

    Stacks tensors and collects metadata into a list.

    Args:
        batch: List of sample dicts from __getitem__.

    Returns:
        Collated batch dict with stacked tensors.
    """
    return {
        "fingerprint": torch.stack([s["fingerprint"] for s in batch]),
        "iris": torch.stack([s["iris"] for s in batch]),
        "label": torch.tensor([s["label"] for s in batch], dtype=torch.long),
        "metadata": [s["metadata"] for s in batch],
    }
