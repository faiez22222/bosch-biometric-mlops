"""Image preprocessing and augmentation pipelines for each modality."""

from __future__ import annotations

from typing import Any

from torchvision import transforms


def get_fingerprint_transforms(
    cfg: Any,
    is_train: bool = True,
) -> transforms.Compose:
    """Build transform pipeline for fingerprint images.

    Fingerprint images are grayscale, 96x103 pixels. We resize to a
    configurable square (default 128x128), optionally augment, and
    normalize to [-1, 1].

    Args:
        cfg: Hydra data config node (expects cfg.fingerprint and cfg.augmentation).
        is_train: Whether to include data augmentation transforms.

    Returns:
        Composed torchvision transforms.
    """
    fp_cfg = cfg.fingerprint
    aug_cfg = cfg.augmentation
    target_size = tuple(fp_cfg.resize)

    pipeline: list[transforms.Transform] = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(target_size),
    ]

    if is_train and aug_cfg.enabled:
        pipeline.extend([
            transforms.RandomRotation(degrees=aug_cfg.rotation_degrees),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
            ),
        ])

    pipeline.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=list(fp_cfg.normalize_mean),
            std=list(fp_cfg.normalize_std),
        ),
    ])

    return transforms.Compose(pipeline)


def get_iris_transforms(
    cfg: Any,
    is_train: bool = True,
) -> transforms.Compose:
    """Build transform pipeline for iris images.

    Iris images are RGB, 320x240 pixels. We resize to a configurable
    square (default 224x224), optionally augment, and normalize using
    ImageNet-style statistics.

    Args:
        cfg: Hydra data config node (expects cfg.iris and cfg.augmentation).
        is_train: Whether to include data augmentation transforms.

    Returns:
        Composed torchvision transforms.
    """
    iris_cfg = cfg.iris
    aug_cfg = cfg.augmentation
    target_size = tuple(iris_cfg.resize)

    pipeline: list[transforms.Transform] = [
        transforms.Resize(target_size),
    ]

    if is_train and aug_cfg.enabled:
        pipeline.extend([
            transforms.RandomHorizontalFlip(p=aug_cfg.horizontal_flip_prob),
            transforms.ColorJitter(
                brightness=aug_cfg.color_jitter_brightness,
                contrast=aug_cfg.color_jitter_contrast,
            ),
            transforms.RandomRotation(degrees=aug_cfg.rotation_degrees),
        ])

    pipeline.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=list(iris_cfg.normalize_mean),
            std=list(iris_cfg.normalize_std),
        ),
    ])

    return transforms.Compose(pipeline)
