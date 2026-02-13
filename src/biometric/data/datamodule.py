"""DataModule: orchestrates dataset creation and DataLoader instantiation."""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from biometric.data.dataset import (
    MultimodalBiometricDataset,
    build_label_map,
    collate_multimodal,
)
from biometric.data.transforms import get_fingerprint_transforms, get_iris_transforms
from biometric.data.utils import create_splits, discover_dataset

logger = logging.getLogger(__name__)


class BiometricDataModule:
    """Orchestrates data discovery, splitting, and DataLoader creation.

    This is the single entry point for all data access in the training and
    evaluation pipelines. It ensures that:
    - Splits are done at the subject level (no data leakage).
    - Label maps are consistent across all splits.
    - Transforms are appropriate for train vs. eval modes.
    - DataLoader configuration is centralized and config-driven.

    Usage:
        dm = BiometricDataModule(cfg)
        dm.setup()
        for batch in dm.train_dataloader():
            ...
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self._all_subjects: list[dict[str, Any]] = []
        self._splits: dict[str, list[dict[str, Any]]] = {}
        self._label_map: dict[int, int] = {}
        self._train_dataset: MultimodalBiometricDataset | None = None
        self._val_dataset: MultimodalBiometricDataset | None = None
        self._test_dataset: MultimodalBiometricDataset | None = None

    @property
    def num_classes(self) -> int:
        """Number of unique identity classes."""
        return len(self._label_map)

    @property
    def label_map(self) -> dict[int, int]:
        """Subject ID -> contiguous class index mapping."""
        return self._label_map

    def setup(self) -> None:
        """Discover data, create splits, and instantiate datasets.

        Must be called before accessing dataloaders.
        """
        data_cfg = self.cfg.data

        # 1. Discover all subjects
        self._all_subjects = discover_dataset(data_cfg.root_dir)

        # Optional: limit subjects for fast development
        max_subjects = getattr(data_cfg, "max_subjects", None)
        if max_subjects is not None:
            logger.info("Limiting to first %d subjects (fast_dev mode).", max_subjects)
            self._all_subjects = self._all_subjects[:max_subjects]

        # 2. Build label map from ALL subjects before splitting
        self._label_map = build_label_map(self._all_subjects)

        # 3. Create stratified splits
        self._splits = create_splits(
            self._all_subjects,
            train_ratio=data_cfg.train_split,
            val_ratio=data_cfg.val_split,
            test_ratio=data_cfg.test_split,
            seed=self.cfg.seed,
        )

        # 4. Build transforms
        fp_train_tfm = get_fingerprint_transforms(data_cfg, is_train=True)
        fp_eval_tfm = get_fingerprint_transforms(data_cfg, is_train=False)
        iris_train_tfm = get_iris_transforms(data_cfg, is_train=True)
        iris_eval_tfm = get_iris_transforms(data_cfg, is_train=False)

        # 5. Create datasets
        self._train_dataset = MultimodalBiometricDataset(
            subjects=self._splits["train"],
            label_map=self._label_map,
            fingerprint_transform=fp_train_tfm,
            iris_transform=iris_train_tfm,
        )
        self._val_dataset = MultimodalBiometricDataset(
            subjects=self._splits["val"],
            label_map=self._label_map,
            fingerprint_transform=fp_eval_tfm,
            iris_transform=iris_eval_tfm,
        )
        self._test_dataset = MultimodalBiometricDataset(
            subjects=self._splits["test"],
            label_map=self._label_map,
            fingerprint_transform=fp_eval_tfm,
            iris_transform=iris_eval_tfm,
        )

        logger.info(
            "Datasets ready — train: %d, val: %d, test: %d pairs.",
            len(self._train_dataset),
            len(self._val_dataset),
            len(self._test_dataset),
        )

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader with shuffling."""
        assert self._train_dataset is not None, "Call setup() first."
        return self._build_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader (no shuffling)."""
        assert self._val_dataset is not None, "Call setup() first."
        return self._build_dataloader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader (no shuffling)."""
        assert self._test_dataset is not None, "Call setup() first."
        return self._build_dataloader(self._test_dataset, shuffle=False)

    def _build_dataloader(
        self,
        dataset: MultimodalBiometricDataset,
        shuffle: bool,
    ) -> DataLoader:
        """Construct a DataLoader with config-driven parameters."""
        data_cfg = self.cfg.data
        return DataLoader(
            dataset=dataset,
            batch_size=data_cfg.batch_size,
            shuffle=shuffle,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            prefetch_factor=data_cfg.prefetch_factor if data_cfg.num_workers > 0 else None,
            collate_fn=collate_multimodal,
            drop_last=shuffle,  # Drop incomplete last batch only during training
            persistent_workers=data_cfg.num_workers > 0,
        )
