"""Training loop with reproducibility, checkpointing, and metrics tracking."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from biometric.data.datamodule import BiometricDataModule
from biometric.models.multimodal_net import MultimodalBiometricNet
from biometric.training.callbacks import CheckpointCallback, EarlyStoppingCallback
from biometric.training.metrics import MetricsTracker
from biometric.utils.reproducibility import configure_deterministic, set_seed

logger = logging.getLogger(__name__)


class Trainer:
    """Production training loop for the multimodal biometric model.

    Handles the full lifecycle: data setup, model creation, training,
    validation, checkpointing, early stopping, and final evaluation.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.device = self._detect_device()

    def fit(self) -> dict[str, Any]:
        """Execute the full training pipeline.

        Returns:
            Training summary dict with metrics, timing, and config.
        """
        cfg = self.cfg
        t_start = time.perf_counter()

        # Reproducibility
        set_seed(cfg.seed)
        configure_deterministic(True)

        # Data
        dm = BiometricDataModule(cfg)
        dm.setup()
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        # Model
        model = MultimodalBiometricNet(cfg).to(self.device)
        param_counts = model.count_parameters()
        logger.info("Model parameters: %s", param_counts)

        # Optimizer, scheduler, loss
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss()

        # Callbacks
        ckpt_cb = CheckpointCallback(cfg)
        es_cb = EarlyStoppingCallback(cfg)

        # Metrics
        train_tracker = MetricsTracker()
        val_tracker = MetricsTracker()

        # Training loop
        best_val_acc = 0.0
        best_epoch = 0
        stopped_early = False
        epochs_run = 0

        for epoch in range(cfg.training.epochs):
            epochs_run = epoch + 1

            # --- Train ---
            train_metrics = self._train_one_epoch(
                epoch, model, train_loader, optimizer, criterion, train_tracker
            )

            # --- Validate ---
            val_metrics = self._validate(epoch, model, val_loader, criterion, val_tracker)

            # --- Scheduler step ---
            if scheduler is not None:
                scheduler.step()

            # --- Logging ---
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %d/%d | "
                "train_loss=%.4f train_acc=%.4f | "
                "val_loss=%.4f val_acc=%.4f | "
                "lr=%.6f",
                epoch + 1,
                cfg.training.epochs,
                train_metrics["loss"],
                train_metrics["accuracy"],
                val_metrics["loss"],
                val_metrics["accuracy"],
                lr,
            )

            # --- Checkpoint ---
            ckpt_cb.on_epoch_end(
                epoch, model, optimizer, scheduler, val_metrics, cfg, dm.label_map
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch + 1

            # --- Early stopping ---
            if es_cb.on_epoch_end(val_metrics):
                stopped_early = True
                break

        # --- Load best and evaluate on test set ---
        best_ckpt = ckpt_cb.best_checkpoint_path
        if best_ckpt is not None and best_ckpt.exists():
            checkpoint = torch.load(best_ckpt, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Loaded best checkpoint from %s", best_ckpt.name)

        test_metrics = self._evaluate(model, test_loader, criterion)
        logger.info(
            "Test results | loss=%.4f accuracy=%.4f",
            test_metrics["loss"],
            test_metrics["accuracy"],
        )

        # --- Save training summary ---
        t_total = time.perf_counter() - t_start
        summary = {
            "experiment_name": cfg.experiment_name,
            "seed": cfg.seed,
            "device": str(self.device),
            "model_parameters": param_counts,
            "dataset": {
                "num_classes": dm.num_classes,
                "train_pairs": len(train_loader.dataset),
                "val_pairs": len(val_loader.dataset),
                "test_pairs": len(test_loader.dataset),
            },
            "training": {
                "epochs_run": epochs_run,
                "stopped_early": stopped_early,
                "best_epoch": best_epoch,
                "best_val_accuracy": round(best_val_acc, 4),
                "best_val_loss": round(
                    ckpt_cb.best_metric if ckpt_cb.best_metric is not None else 0.0, 4
                ),
                "final_test_accuracy": round(test_metrics["accuracy"], 4),
                "final_test_loss": round(test_metrics["loss"], 4),
            },
            "timing": {
                "total_seconds": round(t_total, 1),
                "seconds_per_epoch": round(t_total / max(epochs_run, 1), 1),
            },
            "config": OmegaConf.to_container(cfg, resolve=True),
        }

        summary_path = Path(cfg.training.checkpoint_dir) / "training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Training summary saved to %s", summary_path)

        # Save metric histories
        train_tracker.to_json(Path(cfg.training.checkpoint_dir) / "train_metrics.json")
        val_tracker.to_json(Path(cfg.training.checkpoint_dir) / "val_metrics.json")

        return summary

    def _train_one_epoch(
        self,
        epoch: int,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        tracker: MetricsTracker,
    ) -> dict[str, float]:
        model.train()
        tracker.reset_epoch()
        log_interval = self.cfg.training.log_every_n_steps

        for step, batch in enumerate(loader):
            fingerprint = batch["fingerprint"].to(self.device)
            iris = batch["iris"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()
            logits = model(fingerprint, iris)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.cfg.training.grad_clip_norm
            )
            optimizer.step()

            tracker.update(loss.item(), logits.detach(), labels)

            if log_interval > 0 and (step + 1) % log_interval == 0:
                logger.debug(
                    "  Step %d | loss=%.4f", step + 1, loss.item()
                )

        return tracker.compute_epoch_metrics()

    def _validate(
        self,
        epoch: int,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        tracker: MetricsTracker,
    ) -> dict[str, float]:
        return self._evaluate(model, loader, criterion, tracker)

    def _evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        tracker: MetricsTracker | None = None,
    ) -> dict[str, float]:
        if tracker is None:
            tracker = MetricsTracker()

        model.eval()
        tracker.reset_epoch()

        with torch.no_grad():
            for batch in loader:
                fingerprint = batch["fingerprint"].to(self.device)
                iris = batch["iris"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = model(fingerprint, iris)
                loss = criterion(logits, labels)
                tracker.update(loss.item(), logits, labels)

        return tracker.compute_epoch_metrics()

    def _detect_device(self) -> torch.device:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Silicon MPS")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        cfg = self.cfg.training
        params = model.parameters()

        if cfg.optimizer == "adam":
            return torch.optim.Adam(
                params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay
            )
        elif cfg.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler | None:
        cfg = self.cfg.training

        if cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs
            )
        elif cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(cfg.epochs // 3, 1)
            )
        elif cfg.scheduler == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")
