"""Benchmark data loading performance across different configurations.

Measures:
1. Raw BMP loading throughput (PIL.Image.open)
2. DataLoader throughput vs. num_workers (0, 1, 2, 4, 8)
3. Impact of pin_memory on overall batch preparation time
4. PyArrow catalog metadata lookup vs. filesystem walk
5. Ray preprocessing speedup (serial vs. parallel)

Usage:
    python scripts/benchmark_dataloader.py
    python scripts/benchmark_dataloader.py data=fast_dev
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from biometric.data.datamodule import BiometricDataModule
from biometric.data.utils import discover_dataset

logger = logging.getLogger(__name__)


def benchmark_raw_loading(root_dir: str, max_images: int = 100) -> dict:
    """Measure raw image loading speed with PIL."""
    subjects = discover_dataset(root_dir)
    image_paths: list[Path] = []
    for s in subjects:
        image_paths.extend(s["fingerprints"].values())
        for eye in ["left", "right"]:
            image_paths.extend(s["iris"][eye])
    image_paths = image_paths[:max_images]

    start = time.perf_counter()
    for p in image_paths:
        img = Image.open(p)
        img.load()  # Force decode
    elapsed = time.perf_counter() - start

    return {
        "test": "raw_pil_loading",
        "num_images": len(image_paths),
        "total_time_s": round(elapsed, 4),
        "images_per_second": round(len(image_paths) / elapsed, 1),
    }


def benchmark_dataloader_workers(cfg: DictConfig) -> list[dict]:
    """Measure DataLoader throughput for different num_workers values."""
    results = []
    worker_counts = [0, 1, 2, 4]

    for nw in worker_counts:
        # Override num_workers
        test_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        test_cfg.data.num_workers = nw
        if nw == 0:
            test_cfg.data.pin_memory = False

        dm = BiometricDataModule(test_cfg)
        dm.setup()
        loader = dm.train_dataloader()

        # Warm up
        batch_iter = iter(loader)
        try:
            next(batch_iter)
        except StopIteration:
            pass

        # Measure
        num_batches = 0
        num_samples = 0
        start = time.perf_counter()
        for batch in loader:
            num_batches += 1
            num_samples += batch["fingerprint"].shape[0]
        elapsed = time.perf_counter() - start

        results.append({
            "test": "dataloader_throughput",
            "num_workers": nw,
            "num_batches": num_batches,
            "num_samples": num_samples,
            "total_time_s": round(elapsed, 4),
            "samples_per_second": round(num_samples / elapsed, 1) if elapsed > 0 else 0,
            "batches_per_second": round(num_batches / elapsed, 1) if elapsed > 0 else 0,
        })

        logger.info(
            "Workers=%d: %d samples in %.2fs (%.1f samples/s)",
            nw,
            num_samples,
            elapsed,
            num_samples / elapsed if elapsed > 0 else 0,
        )

    return results


def benchmark_pin_memory(cfg: DictConfig) -> list[dict]:
    """Compare pin_memory=True vs False for GPU transfer readiness."""
    if not torch.cuda.is_available():
        logger.info("CUDA not available, skipping pin_memory benchmark.")
        return [{"test": "pin_memory", "skipped": True, "reason": "no CUDA"}]

    results = []
    for pin in [False, True]:
        test_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        test_cfg.data.pin_memory = pin
        test_cfg.data.num_workers = 2

        dm = BiometricDataModule(test_cfg)
        dm.setup()
        loader = dm.train_dataloader()

        num_samples = 0
        start = time.perf_counter()
        for batch in loader:
            # Simulate GPU transfer
            batch["fingerprint"].cuda(non_blocking=pin)
            batch["iris"].cuda(non_blocking=pin)
            num_samples += batch["fingerprint"].shape[0]
        elapsed = time.perf_counter() - start

        results.append({
            "test": "pin_memory",
            "pin_memory": pin,
            "num_samples": num_samples,
            "total_time_s": round(elapsed, 4),
            "samples_per_second": round(num_samples / elapsed, 1),
        })

    return results


def benchmark_catalog_vs_walk(cfg: DictConfig) -> dict:
    """Compare filesystem walk vs PyArrow catalog read time."""
    from biometric.data.arrow_store import ArrowBiometricStore

    data_cfg = cfg.data
    catalog_path = Path(data_cfg.preprocessed_dir) / "biometric_catalog.parquet"

    # Filesystem walk
    start = time.perf_counter()
    subjects = discover_dataset(data_cfg.root_dir)
    walk_time = time.perf_counter() - start
    num_records_walk = sum(
        len(s["fingerprints"]) + len(s["iris"]["left"]) + len(s["iris"]["right"])
        for s in subjects
    )

    # PyArrow catalog read
    catalog_time = -1.0
    num_records_arrow = 0
    if catalog_path.exists():
        store = ArrowBiometricStore()
        start = time.perf_counter()
        table = store.read(catalog_path)
        catalog_time = time.perf_counter() - start
        num_records_arrow = table.num_rows

    return {
        "test": "catalog_vs_walk",
        "walk_time_s": round(walk_time, 4),
        "walk_records": num_records_walk,
        "catalog_time_s": round(catalog_time, 4) if catalog_time >= 0 else "N/A",
        "catalog_records": num_records_arrow,
        "speedup": (
            round(walk_time / catalog_time, 2)
            if catalog_time > 0
            else "catalog not found"
        ),
    }


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run all benchmarks and print a consolidated report."""
    all_results: list[dict] = []

    print("\n" + "=" * 70)
    print("DATA LOADING PERFORMANCE BENCHMARK")
    print("=" * 70)

    # 1. Raw PIL loading
    print("\n[1/4] Raw PIL image loading...")
    raw_result = benchmark_raw_loading(cfg.data.root_dir)
    all_results.append(raw_result)

    # 2. DataLoader workers
    print("\n[2/4] DataLoader worker scaling...")
    worker_results = benchmark_dataloader_workers(cfg)
    all_results.extend(worker_results)

    # 3. Pin memory
    print("\n[3/4] Pin memory benchmark...")
    pin_results = benchmark_pin_memory(cfg)
    all_results.extend(pin_results)

    # 4. Catalog vs walk
    print("\n[4/4] Arrow catalog vs filesystem walk...")
    catalog_result = benchmark_catalog_vs_walk(cfg)
    all_results.append(catalog_result)

    # Print summary
    print("\n" + "=" * 70)
    print("FULL BENCHMARK REPORT")
    print("=" * 70)
    print(json.dumps(all_results, indent=2, default=str))
    print("=" * 70)


if __name__ == "__main__":
    main()
