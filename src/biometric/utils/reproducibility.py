"""Reproducibility utilities for deterministic training."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds across all libraries for reproducible results.

    Args:
        seed: Integer seed value to use across all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_deterministic(enabled: bool = True) -> None:
    """Configure PyTorch for deterministic execution.

    Note: Deterministic mode may reduce performance. Disable for production
    training where exact reproducibility is not required.

    Args:
        enabled: Whether to enforce deterministic algorithms.
    """
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled
    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=True)
