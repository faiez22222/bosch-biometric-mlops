

from __future__ import annotations

import json
import logging

import hydra
from omegaconf import DictConfig

from biometric.data.preprocessing import run_parallel_preprocessing

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the preprocessing pipeline and print the timing report."""
    report = run_parallel_preprocessing(cfg)

    print("\n" + "=" * 60)
    print("PREPROCESSING REPORT")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
