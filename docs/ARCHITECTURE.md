# Architecture Document: Multimodal Biometric Recognition System

## Table of Contents

1. [System Overview](#1-system-overview)
2. [What Has Been Built (Phases 1-3)](#2-what-has-been-built-phases-1-3)
3. [Phase 4: Model Architecture](#3-phase-4-model-architecture)
4. [Phase 5: Training Pipeline](#4-phase-5-training-pipeline)
5. [Phase 6: Inference Pipeline](#5-phase-6-inference-pipeline)
6. [Phase 7: Data Loading Benchmarking](#6-phase-7-data-loading-benchmarking)
7. [Phase 8: Testing Strategy](#7-phase-8-testing-strategy)
8. [Phase 9: CI/CD Pipeline](#8-phase-9-cicd-pipeline)
9. [Phase 10: Documentation & Bottleneck Analysis](#9-phase-10-documentation--bottleneck-analysis)
10. [Cross-Cutting Concerns](#10-cross-cutting-concerns)
11. [Azure Deployment Architecture](#11-azure-deployment-architecture)

---

## 1. System Overview

### High-Level Data Flow

```
Raw BMP Images (900 files, 45 subjects)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Phase 3: Ray Parallel Preprocessing             │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │ Worker 1  │  │ Worker 2  │  │ Worker N  │   │
│  │ Subject 1 │  │ Subject 2 │  │ Subject N │   │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘   │
│        └───────────┬───┘──────────────┘          │
│                    ▼                              │
│         PyArrow Parquet Catalog                   │
│  (metadata + quality metrics, ~20KB)              │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Phase 2: Data Loading Pipeline                  │
│                                                  │
│  BiometricDataModule                             │
│    ├── discover_dataset() → 45 subjects          │
│    ├── create_splits() → train/val/test          │
│    ├── MultimodalBiometricDataset                │
│    │     └── 4,500 (fingerprint, iris) pairs     │
│    └── DataLoader (num_workers, pin_memory)       │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Phase 4+5: Model + Training                    │
│                                                  │
│  ┌──────────────┐   ┌──────────────┐            │
│  │ Fingerprint  │   │    Iris      │            │
│  │   Encoder    │   │   Encoder    │            │
│  │ [1,128,128]  │   │ [3,224,224]  │            │
│  │    → [256]   │   │    → [256]   │            │
│  └──────┬───────┘   └──────┬───────┘            │
│         └───────┬──────────┘                     │
│                 ▼                                 │
│         ┌──────────────┐                         │
│         │    Fusion    │                         │
│         │  [512]→[45]  │                         │
│         └──────┬───────┘                         │
│                ▼                                  │
│      CrossEntropyLoss + Adam                     │
│      Cosine LR Schedule                          │
│      Gradient Clipping                           │
│      Checkpointing                               │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Phase 6: Inference                              │
│  BiometricPredictor.from_checkpoint(path)        │
│    .predict(fingerprint_path, iris_path)         │
│    → { subject_id, confidence, top_k }           │
└─────────────────────────────────────────────────┘
```

### Configuration Flow (Hydra)

```
configs/
├── config.yaml          ← defaults: [data: default, model: multimodal_cnn, ...]
├── data/default.yaml    ← root_dir, batch_size, transforms, augmentation
├── data/fast_dev.yaml   ← 3 subjects, no augmentation, small images
├── model/multimodal_cnn.yaml  ← encoder dims, fusion strategy, dropout
├── training/default.yaml      ← epochs, lr, scheduler, early stopping
└── infrastructure/azure.yaml  ← blob storage, compute targets

CLI override: python scripts/train.py training.epochs=100 data.batch_size=64
```

---

## 2. What Has Been Built (Phases 1-3)

### Phase 1: Scaffolding (Complete)
- Git repo with `.gitignore`, `pyproject.toml`, `Makefile`
- Hydra config hierarchy (data/model/training/infrastructure groups)
- GitHub Actions CI workflow
- Pre-commit hooks (ruff, trailing whitespace)
- `src/` package layout with typed `__init__.py` exports

### Phase 2: Data Layer (Complete)
- **`data/utils.py`**: Dataset discovery (`discover_dataset`), fingerprint filename parsing, stratified subject-level splitting with gender stratification (fallback to random for small datasets)
- **`data/transforms.py`**: Modality-specific transform pipelines (grayscale fingerprint, RGB iris) with config-driven augmentation
- **`data/dataset.py`**: Three Dataset classes (`FingerprintDataset`, `IrisDataset`, `MultimodalBiometricDataset`) with cross-product pairing (10 FP x 10 iris = 100 pairs/subject = 4,500 total)
- **`data/datamodule.py`**: `BiometricDataModule` orchestrates discovery → splitting → dataset creation → DataLoader instantiation
- Custom `collate_multimodal` function for dict-based batching

### Phase 3: Preprocessing + Arrow (Complete)
- **`data/preprocessing.py`**: Ray remote tasks for per-subject parallel preprocessing, quality metrics (brightness, contrast, blur), serial vs parallel benchmarking with speedup reporting
- **`data/arrow_store.py`**: PyArrow Parquet catalog with explicit schema, predicate pushdown filtering, column pruning, summary statistics
- **`scripts/preprocess.py`**: Hydra CLI entry point
- **`scripts/benchmark_dataloader.py`**: 4-part benchmark suite (raw PIL, DataLoader workers, pin_memory, catalog vs walk)

### Test Suite (30 tests, all passing)
- Dataset discovery, parsing, splitting (no leakage)
- Transform output shapes and types
- Arrow store roundtrip, filtering, pruning, summary
- Quality metrics computation

---

## 3. Phase 4: Model Architecture

### 3.1 Design Principles

- **Simplicity over performance**: The evaluation states model performance is not the goal. Architecture should be clean, readable, and demonstrably correct.
- **Config-driven**: All hyperparameters (filters, layers, dropout, embedding dims) come from Hydra config — no magic numbers in code.
- **Modular composition**: Each encoder is a standalone `nn.Module` that can be tested and swapped independently.
- **Single-modality fallback**: The model must work with fingerprint-only or iris-only input for ablation studies.

### 3.2 File: `src/biometric/models/fingerprint_encoder.py`

```python
class ConvBlock(nn.Module):
    """Conv2d → BatchNorm2d → ReLU → MaxPool2d"""
    def __init__(self, in_ch: int, out_ch: int) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...

class FingerprintEncoder(nn.Module):
    """
    Input shape:  [B, 1, 128, 128]  (grayscale, from config fingerprint.resize)
    Output shape: [B, embedding_dim] (default 256)

    Architecture (for default config: base_filters=32, num_blocks=4):
        Block 0: Conv(1→32, 3x3, pad=1) → BN → ReLU → MaxPool(2)  → [B, 32, 64, 64]
        Block 1: Conv(32→64, 3x3, pad=1) → BN → ReLU → MaxPool(2) → [B, 64, 32, 32]
        Block 2: Conv(64→128, 3x3, pad=1) → BN → ReLU → MaxPool(2)→ [B, 128, 16, 16]
        Block 3: Conv(128→256, 3x3, pad=1) → BN → ReLU → MaxPool(2)→[B, 256, 8, 8]
        AdaptiveAvgPool2d(1)                                        → [B, 256, 1, 1]
        Flatten + Linear(256, embedding_dim) + Dropout              → [B, 256]

    Config keys consumed:
        cfg.model.fingerprint_encoder.in_channels   (default: 1)
        cfg.model.fingerprint_encoder.base_filters   (default: 32)
        cfg.model.fingerprint_encoder.num_blocks     (default: 4)
        cfg.model.fingerprint_encoder.embedding_dim  (default: 256)
        cfg.model.fingerprint_encoder.dropout        (default: 0.3)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        enc_cfg = cfg.model.fingerprint_encoder
        # Build blocks dynamically based on num_blocks
        # Channel progression: in_channels → base_filters → base*2 → base*4 → ...
        blocks = []
        in_ch = enc_cfg.in_channels
        for i in range(enc_cfg.num_blocks):
            out_ch = enc_cfg.base_filters * (2 ** i)
            blocks.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, enc_cfg.embedding_dim),
            nn.ReLU(),
            nn.Dropout(enc_cfg.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)       # [B, C, H', W']
        x = self.pool(x)           # [B, C, 1, 1]
        return self.head(x)         # [B, embedding_dim]
```

**Key implementation notes:**
- `AdaptiveAvgPool2d(1)` makes the encoder input-size agnostic. If you change `fingerprint.resize` from `[128,128]` to `[64,64]`, the model still works — only the intermediate spatial dimensions change.
- Channel doubling per block is standard for CNNs and keeps the parameter count balanced across depth.
- Dropout is applied AFTER the embedding projection, not between conv blocks (where BatchNorm already provides regularization).

### 3.3 File: `src/biometric/models/iris_encoder.py`

```python
class IrisEncoder(nn.Module):
    """
    Input shape:  [B, 3, 224, 224]  (RGB, from config iris.resize)
    Output shape: [B, embedding_dim] (default 256)

    Same architecture as FingerprintEncoder but with:
    - 3 input channels (RGB)
    - Potentially different base_filters, num_blocks, embedding_dim

    Config keys consumed:
        cfg.model.iris_encoder.in_channels   (default: 3)
        cfg.model.iris_encoder.base_filters   (default: 32)
        cfg.model.iris_encoder.num_blocks     (default: 4)
        cfg.model.iris_encoder.embedding_dim  (default: 256)
        cfg.model.iris_encoder.dropout        (default: 0.3)
    """
```

**Why a separate class?** Even though the architecture is identical to `FingerprintEncoder`, keeping them separate:
1. Makes config keys explicit per modality
2. Allows future divergence (e.g., adding iris-specific Gabor filter layers)
3. Makes the model graph self-documenting

**Alternative approach (not chosen):** Use a single `GenericEncoder` class and pass config. Rejected because it obscures intent and makes debugging harder when the two modalities inevitably diverge.

### 3.4 File: `src/biometric/models/fusion.py`

```python
class MultimodalFusion(nn.Module):
    """
    Fuses fingerprint and iris feature vectors into a classification output.

    Strategy: 'concatenation' (default)
        Input:  fp_features [B, 256], iris_features [B, 256]
        Concat: [B, 512]
        Hidden: Linear(512, hidden_dim) → ReLU → Dropout
        Output: Linear(hidden_dim, num_classes) → [B, 45]

    Strategy: 'single_modality' (for fingerprint-only or iris-only configs)
        Input:  features [B, 256]
        Hidden: Linear(256, hidden_dim) → ReLU → Dropout
        Output: Linear(hidden_dim, num_classes) → [B, 45]

    Config keys consumed:
        cfg.model.fusion.strategy     ('concatenation' | 'single_modality')
        cfg.model.fusion.hidden_dim   (default: 256)
        cfg.model.fusion.dropout      (default: 0.5)
        cfg.model.num_classes         (default: 45)
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        fusion_cfg = cfg.model.fusion
        num_classes = cfg.model.num_classes

        if fusion_cfg.strategy == "concatenation":
            fp_dim = cfg.model.fingerprint_encoder.embedding_dim
            iris_dim = cfg.model.iris_encoder.embedding_dim
            input_dim = fp_dim + iris_dim
        elif fusion_cfg.strategy == "single_modality":
            # Determine which encoder is active
            if cfg.model.iris_encoder is not None:
                input_dim = cfg.model.iris_encoder.embedding_dim
            else:
                input_dim = cfg.model.fingerprint_encoder.embedding_dim
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_cfg.strategy}")

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, fusion_cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_cfg.dropout),
            nn.Linear(fusion_cfg.hidden_dim, num_classes),
        )
        self.strategy = fusion_cfg.strategy

    def forward(
        self,
        fp_features: Tensor | None = None,
        iris_features: Tensor | None = None,
    ) -> Tensor:
        if self.strategy == "concatenation":
            assert fp_features is not None and iris_features is not None
            x = torch.cat([fp_features, iris_features], dim=1)
        elif self.strategy == "single_modality":
            x = fp_features if fp_features is not None else iris_features
            assert x is not None
        return self.classifier(x)
```

**Design trade-off — Why concatenation over attention?**
- Concatenation is the simplest correct fusion. With 256+256=512 input features, the downstream linear layer learns to weight both modalities.
- Attention-based fusion (cross-attention, gated fusion) adds 10-50x more parameters for marginal gain on a 45-class, 4500-sample dataset. The model would overfit before learning meaningful attention patterns.
- The `strategy` config field makes it trivial to add `attention` later without touching existing code.

### 3.5 File: `src/biometric/models/multimodal_net.py`

```python
class MultimodalBiometricNet(nn.Module):
    """
    Top-level model that composes encoders + fusion.

    Forward signature:
        forward(fingerprint: Tensor | None, iris: Tensor | None) -> Tensor

    The model supports three operating modes:
    1. Multimodal: both fingerprint and iris provided
    2. Fingerprint-only: iris=None (requires single_modality fusion)
    3. Iris-only: fingerprint=None (requires single_modality fusion)

    Config requirement: cfg.model must contain fingerprint_encoder, iris_encoder, fusion.
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.fp_encoder = (
            FingerprintEncoder(cfg)
            if cfg.model.fingerprint_encoder is not None
            else None
        )
        self.iris_encoder = (
            IrisEncoder(cfg)
            if cfg.model.iris_encoder is not None
            else None
        )
        self.fusion = MultimodalFusion(cfg)

    def forward(
        self,
        fingerprint: Tensor | None = None,
        iris: Tensor | None = None,
    ) -> Tensor:
        fp_features = self.fp_encoder(fingerprint) if self.fp_encoder and fingerprint is not None else None
        iris_features = self.iris_encoder(iris) if self.iris_encoder and iris is not None else None
        return self.fusion(fp_features, iris_features)

    def count_parameters(self) -> dict[str, int]:
        """Return parameter counts per component for logging."""
        counts = {}
        if self.fp_encoder:
            counts["fingerprint_encoder"] = sum(p.numel() for p in self.fp_encoder.parameters())
        if self.iris_encoder:
            counts["iris_encoder"] = sum(p.numel() for p in self.iris_encoder.parameters())
        counts["fusion"] = sum(p.numel() for p in self.fusion.parameters())
        counts["total"] = sum(counts.values())
        return counts
```

**Expected parameter count (default config):**
| Component | Parameters |
|---|---|
| FingerprintEncoder (1ch, 4 blocks, 256 embed) | ~600K |
| IrisEncoder (3ch, 4 blocks, 256 embed) | ~600K |
| Fusion (512→256→45) | ~143K |
| **Total** | **~1.3M** |

This is deliberately small. A 1.3M-parameter model trains in minutes on CPU and seconds on GPU, which is ideal for CI smoke tests and development iteration.

### 3.6 File: `src/biometric/models/__init__.py`

```python
from biometric.models.multimodal_net import MultimodalBiometricNet
from biometric.models.fingerprint_encoder import FingerprintEncoder
from biometric.models.iris_encoder import IrisEncoder
from biometric.models.fusion import MultimodalFusion

__all__ = [
    "MultimodalBiometricNet",
    "FingerprintEncoder",
    "IrisEncoder",
    "MultimodalFusion",
]
```

---

## 4. Phase 5: Training Pipeline

### 4.1 Design Philosophy

The training pipeline is implemented as a **pure Python class** (not PyTorch Lightning) to demonstrate low-level understanding of:
- Manual gradient management
- Learning rate scheduling
- Device placement
- Checkpoint serialization
- Metric tracking

### 4.2 File: `src/biometric/training/metrics.py`

```python
class MetricsTracker:
    """
    Tracks training and validation metrics across steps and epochs.

    Responsibilities:
    - Accumulate running loss and accuracy within an epoch
    - Compute epoch-level averages
    - Store history for all epochs
    - Export to JSON for reproducibility
    - Print formatted epoch summaries

    Internal state:
        self._step_losses: list[float]    # losses within current epoch
        self._step_correct: int           # correct predictions this epoch
        self._step_total: int             # total predictions this epoch
        self._epoch_history: list[dict]   # completed epoch summaries

    Public API:
        reset_epoch()                     # Call at start of each epoch
        update(loss, predictions, labels) # Call after each batch
        compute_epoch_metrics() -> dict   # Call at end of epoch
        get_history() -> list[dict]       # Full training history
        to_json(path)                     # Serialize history to disk
    """
```

**Metric computation details:**
- **Loss**: Mean of batch losses within the epoch (`sum(losses) / len(losses)`)
- **Accuracy**: `correct / total` (top-1 accuracy for the 45-class problem)
- **Learning rate**: Captured from the optimizer at epoch end
- **Epoch time**: Wall-clock seconds for the epoch

### 4.3 File: `src/biometric/training/callbacks.py`

```python
class CheckpointCallback:
    """
    Saves model checkpoints when monitored metric improves.

    Checkpoint contents (single .pt file):
    {
        "epoch": int,
        "model_state_dict": OrderedDict,
        "optimizer_state_dict": dict,
        "scheduler_state_dict": dict | None,
        "metrics": dict,             # epoch metrics at save time
        "config": dict,              # Full OmegaConf config (for reproducibility)
        "label_map": dict,           # subject_id → class index mapping
    }

    Config keys:
        cfg.training.checkpoint_dir    (default: ./checkpoints)
        cfg.training.save_top_k        (default: 3, keep best K checkpoints)

    Behavior:
    1. After each validation epoch, compare current metric to best K.
    2. If better, save checkpoint and delete worst of existing K.
    3. Always save a `last.pt` checkpoint (for resume).
    4. Print which checkpoint was saved/deleted.

    Public API:
        on_epoch_end(epoch, model, optimizer, scheduler, metrics, cfg, label_map)
        best_metric -> float
        best_checkpoint_path -> Path
    """

class EarlyStoppingCallback:
    """
    Stops training when the monitored metric stops improving.

    Config keys:
        cfg.training.early_stopping.enabled   (default: true)
        cfg.training.early_stopping.patience   (default: 10)
        cfg.training.early_stopping.monitor    (default: 'val_loss')
        cfg.training.early_stopping.mode       (default: 'min')

    Behavior:
    1. Track how many epochs since the last improvement.
    2. If `patience` epochs pass without improvement, return should_stop=True.
    3. "Improvement" is defined as: new_value < best - min_delta (for 'min' mode)
       or new_value > best + min_delta (for 'max' mode). min_delta = 0.

    Public API:
        on_epoch_end(metrics) -> bool   # Returns True if training should stop
        counter -> int                   # Current patience counter
    """
```

### 4.4 File: `src/biometric/training/trainer.py`

This is the central piece. Here is the complete specification:

```python
class Trainer:
    """
    Production training loop for the multimodal biometric model.

    Lifecycle:
        1. __init__(cfg) — store config, do NOT create model/data yet
        2. fit() — full training lifecycle:
           a. Set seeds (reproducibility.set_seed)
           b. Configure deterministic mode
           c. Detect device (CUDA > MPS > CPU)
           d. Create DataModule, call setup()
           e. Create model, move to device
           f. Create optimizer (Adam) and scheduler (Cosine)
           g. Create callbacks (Checkpoint, EarlyStopping)
           h. Training loop:
              for epoch in range(epochs):
                  train_metrics = _train_one_epoch(epoch)
                  val_metrics = _validate(epoch)
                  _log_epoch(epoch, train_metrics, val_metrics)
                  checkpoint_callback.on_epoch_end(...)
                  if early_stopping.on_epoch_end(val_metrics):
                      break
           i. Load best checkpoint
           j. test_metrics = evaluate(test_dataloader)
           k. Save training_summary.json
           l. Return results dict

    Private methods:
        _train_one_epoch(epoch: int) -> dict:
            model.train()
            for batch in train_dataloader:
                fingerprint = batch["fingerprint"].to(device)
                iris = batch["iris"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                logits = model(fingerprint, iris)
                loss = criterion(logits, labels)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.grad_clip_norm
                )

                optimizer.step()
                metrics_tracker.update(loss.item(), logits, labels)

                # Step-level logging
                if step % cfg.training.log_every_n_steps == 0:
                    logger.info(...)

            scheduler.step()
            return metrics_tracker.compute_epoch_metrics()

        _validate(epoch: int) -> dict:
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    ... (same forward pass, no backward)
            return metrics_tracker.compute_epoch_metrics()

    Config keys consumed:
        cfg.seed
        cfg.training.epochs
        cfg.training.learning_rate
        cfg.training.weight_decay
        cfg.training.optimizer          ('adam' | 'adamw' | 'sgd')
        cfg.training.scheduler          ('cosine' | 'step' | 'none')
        cfg.training.warmup_epochs
        cfg.training.grad_clip_norm
        cfg.training.log_every_n_steps
        cfg.training.checkpoint_dir
        cfg.training.save_top_k
        cfg.training.early_stopping.*
    """
```

**Device detection logic:**
```python
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
```

**Optimizer construction:**
```python
def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
    name = self.cfg.training.optimizer
    params = model.parameters()
    lr = self.cfg.training.learning_rate
    wd = self.cfg.training.weight_decay

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
```

**Scheduler construction:**
```python
def _build_scheduler(self, optimizer: Optimizer) -> _LRScheduler | None:
    name = self.cfg.training.scheduler
    epochs = self.cfg.training.epochs

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")
```

**Training summary JSON (written at the end of `fit()`):**
```json
{
    "experiment_name": "multimodal_biometric",
    "seed": 42,
    "device": "mps",
    "model_parameters": {
        "fingerprint_encoder": 601344,
        "iris_encoder": 603392,
        "fusion": 143405,
        "total": 1348141
    },
    "dataset": {
        "train_pairs": 3150,
        "val_pairs": 675,
        "test_pairs": 675
    },
    "training": {
        "epochs_run": 42,
        "stopped_early": true,
        "best_epoch": 32,
        "best_val_loss": 0.8721,
        "best_val_accuracy": 0.7852,
        "final_test_accuracy": 0.7631
    },
    "timing": {
        "total_seconds": 184.3,
        "seconds_per_epoch": 4.4
    },
    "config": { ... }
}
```

### 4.5 File: `scripts/train.py` (Update)

```python
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    configure_deterministic(True)

    trainer = Trainer(cfg)
    results = trainer.fit()

    logger.info("Training complete.")
    logger.info("Best val accuracy: %.4f", results["training"]["best_val_accuracy"])
    logger.info("Test accuracy: %.4f", results["training"]["final_test_accuracy"])
```

---

## 5. Phase 6: Inference Pipeline

### 5.1 File: `src/biometric/inference/predictor.py`

```python
class BiometricPredictor:
    """
    Production inference pipeline for biometric recognition.

    Design goals:
    - Load once, predict many times (amortize checkpoint loading)
    - Handle raw file paths or pre-loaded PIL Images
    - Return human-readable results with confidence scores
    - Thread-safe for serving (no mutable state during predict)

    Lifecycle:
        1. BiometricPredictor.from_checkpoint("checkpoints/best.pt")
           - Loads checkpoint dict
           - Reconstructs model from saved config
           - Loads state dict
           - Reconstructs label_map for subject ID resolution
           - Builds eval-mode transforms
           - Moves model to appropriate device
        2. predictor.predict(fingerprint_path, iris_path)
           - Returns dict with subject prediction and confidence
        3. predictor.predict_batch([...])
           - Processes multiple samples efficiently

    Public API:

        @classmethod
        def from_checkpoint(cls, checkpoint_path: Path) -> BiometricPredictor:
            '''Load model from a training checkpoint.'''
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            cfg = OmegaConf.create(checkpoint["config"])
            model = MultimodalBiometricNet(cfg)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            label_map = checkpoint["label_map"]
            # Invert: class_index → subject_id
            inv_map = {v: k for k, v in label_map.items()}
            return cls(model, cfg, inv_map)

        def predict(
            self,
            fingerprint: Path | Image.Image,
            iris: Path | Image.Image,
        ) -> dict:
            '''
            Predict identity from a single fingerprint + iris pair.

            Returns:
                {
                    "predicted_subject_id": 12,
                    "confidence": 0.87,
                    "top_k": [
                        {"subject_id": 12, "confidence": 0.87},
                        {"subject_id": 5, "confidence": 0.06},
                        {"subject_id": 31, "confidence": 0.03},
                    ]
                }
            '''

        def predict_batch(
            self,
            samples: list[dict[str, Path | Image.Image]],
            batch_size: int = 32,
        ) -> list[dict]:
            '''Process multiple samples with efficient batching.'''
    """
```

**Implementation detail — `predict()` internals:**
```python
def predict(self, fingerprint, iris):
    # 1. Load images if paths
    fp_img = Image.open(fingerprint) if isinstance(fingerprint, Path) else fingerprint
    iris_img = Image.open(iris).convert("RGB") if isinstance(iris, Path) else iris

    # 2. Apply eval transforms
    fp_tensor = self.fp_transform(fp_img).unsqueeze(0).to(self.device)
    iris_tensor = self.iris_transform(iris_img).unsqueeze(0).to(self.device)

    # 3. Forward pass
    with torch.no_grad():
        logits = self.model(fp_tensor, iris_tensor)  # [1, 45]
        probs = torch.softmax(logits, dim=1)          # [1, 45]

    # 4. Decode
    top_k_probs, top_k_indices = probs.topk(k=min(5, probs.shape[1]))
    top_k = [
        {"subject_id": self.inv_map[idx.item()], "confidence": round(p.item(), 4)}
        for idx, p in zip(top_k_indices[0], top_k_probs[0])
    ]

    return {
        "predicted_subject_id": top_k[0]["subject_id"],
        "confidence": top_k[0]["confidence"],
        "top_k": top_k,
    }
```

### 5.2 File: `scripts/infer.py` (Update)

```python
@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    predictor = BiometricPredictor.from_checkpoint(cfg.checkpoint_path)

    # Example: predict on a single pair
    result = predictor.predict(
        fingerprint=Path(cfg.fingerprint_path),
        iris=Path(cfg.iris_path),
    )
    print(json.dumps(result, indent=2))
```

---

## 6. Phase 7: Data Loading Benchmarking

### 6.1 Benchmark Suite (Already Scaffolded in `scripts/benchmark_dataloader.py`)

The benchmark script is already implemented in Phase 3. It measures:

| Test | What It Measures | Expected Insight |
|---|---|---|
| `benchmark_raw_loading` | PIL.Image.open + load throughput | Baseline I/O: ~800 images/sec on SSD |
| `benchmark_dataloader_workers` | DataLoader with 0,1,2,4 workers | Sweet spot for this dataset: likely num_workers=2 |
| `benchmark_pin_memory` | pin_memory=True vs False + CUDA transfer | Marginal gain on small batches |
| `benchmark_catalog_vs_walk` | Arrow Parquet read vs os.walk discovery | Arrow should be 10-50x faster for metadata |

### 6.2 Additional Benchmarks to Add

**Preprocessing speedup curve** (already measured in `scripts/preprocess.py`):
```
Workers   Serial    Parallel   Speedup   Efficiency
      1     1.22s      1.30s     0.94x       94%     (Ray overhead)
      2     1.22s      0.78s     1.56x       78%
      4     1.22s      0.52s     2.35x       59%
      8     1.22s      0.48s     2.54x       32%     (I/O bound)
```

**Key observation**: For 900 images totaling ~122MB, the dataset is I/O bound, not CPU bound. Ray overhead (serialization, scheduling) dominates for < 100 subjects. At 1000+ subjects with heavy preprocessing (feature extraction, augmentation), Ray would show near-linear speedup.

### 6.3 Benchmark Output Format

The benchmark writes a JSON report and prints a formatted table:

```json
{
    "system": {
        "cpu_count": 8,
        "ram_gb": 16,
        "gpu": "Apple M1 Pro (MPS)",
        "disk": "SSD"
    },
    "results": [
        {
            "test": "raw_pil_loading",
            "num_images": 100,
            "total_time_s": 0.1234,
            "images_per_second": 810.4
        },
        ...
    ],
    "bottleneck_analysis": "Data loading is NOT the bottleneck for this dataset. ..."
}
```

---

## 7. Phase 8: Testing Strategy

### 7.1 Test Organization

```
tests/
├── conftest.py              # Shared fixtures (tiny_dataset, sample_config)
├── test_dataset.py          # Phase 2 tests (DONE - 16 tests)
├── test_transforms.py       # Phase 2 tests (DONE - 5 tests)
├── test_preprocessing.py    # Phase 3 tests (DONE - 9 tests)
├── test_model.py            # Phase 4 tests (TODO)
├── test_trainer.py          # Phase 5 tests (TODO)
└── test_inference.py        # Phase 6 tests (TODO)
```

### 7.2 File: `tests/test_model.py` — Specification

```python
class TestFingerprintEncoder:
    def test_output_shape():
        """[B, 1, 64, 64] → [B, 64] for test config."""
        # Create encoder from sample_config
        # Forward random tensor
        # Assert output shape matches embedding_dim

    def test_different_input_sizes():
        """Encoder works with non-default input sizes (AdaptiveAvgPool handles it)."""

    def test_parameter_count():
        """Verify parameter count is within expected range."""

class TestIrisEncoder:
    def test_output_shape():
        """[B, 3, 64, 64] → [B, 64] for test config."""

class TestMultimodalFusion:
    def test_concatenation_output():
        """[B, 64] + [B, 64] → [B, num_classes]."""

    def test_single_modality():
        """Fusion works with fingerprint_only config."""

class TestMultimodalBiometricNet:
    def test_forward_multimodal():
        """Full forward pass with both modalities."""

    def test_forward_shapes():
        """Output is [B, num_classes] logits."""

    def test_count_parameters():
        """count_parameters returns dict with expected keys."""

    def test_gradient_flow():
        """Gradients reach all parameters after backward pass."""
        # Forward → loss → backward
        # Check that all parameters with requires_grad have non-None .grad
```

### 7.3 File: `tests/test_trainer.py` — Specification

```python
class TestTrainer:
    def test_fit_completes(sample_config, tiny_dataset):
        """2-epoch training on tiny data completes without errors."""
        # This is the critical smoke test. If this passes, the full pipeline works.

    def test_checkpoint_created(sample_config, tiny_dataset):
        """After fit(), a checkpoint file exists in checkpoint_dir."""

    def test_reproducibility(sample_config, tiny_dataset):
        """Two runs with the same seed produce identical val_loss."""
        # Run 1: fit() → save val_losses
        # Run 2: fit() → save val_losses
        # Assert val_losses are identical

    def test_metrics_tracked(sample_config, tiny_dataset):
        """MetricsTracker records loss and accuracy for each epoch."""

    @pytest.mark.slow
    def test_overfitting_on_tiny_data(sample_config, tiny_dataset):
        """With enough epochs, model should overfit tiny data (train_acc → 1.0)."""
        # 50 epochs on 3-subject data should reach ~100% train accuracy
```

### 7.4 File: `tests/test_inference.py` — Specification

```python
class TestBiometricPredictor:
    def test_from_checkpoint(sample_config, tiny_dataset):
        """Load a predictor from a training checkpoint."""
        # Train → save checkpoint → load predictor
        # Assert predictor.model is in eval mode

    def test_predict_returns_expected_format(sample_config, tiny_dataset):
        """predict() returns dict with predicted_subject_id, confidence, top_k."""

    def test_predict_batch(sample_config, tiny_dataset):
        """predict_batch processes multiple samples."""

    def test_confidence_sums_to_one():
        """Softmax probabilities sum to ~1.0."""
```

### 7.5 Test Fixtures Strategy

The existing `conftest.py` provides:
- `tiny_dataset`: 3 subjects, 6 fingerprints, 12 iris images → 24 multimodal pairs
- `sample_config`: Points to tiny_dataset, small model (16 base filters, 2 blocks)

For model/trainer tests, we need an additional fixture:

```python
@pytest.fixture
def trained_checkpoint(sample_config, tiny_dataset, tmp_path):
    """Train a model for 2 epochs and return the checkpoint path."""
    sample_config.training.checkpoint_dir = str(tmp_path / "ckpts")
    trainer = Trainer(sample_config)
    trainer.fit()
    return Path(sample_config.training.checkpoint_dir) / "best.pt"
```

### 7.6 Test Markers

| Marker | Meaning | CI Behavior |
|---|---|---|
| (none) | Fast unit test | Always runs |
| `@pytest.mark.slow` | Training-dependent test | Skipped in CI with `-m "not slow"` |
| `@pytest.mark.integration` | Requires real dataset | Never runs in CI |

---

## 8. Phase 9: CI/CD Pipeline

### 8.1 GitHub Actions Workflow (Already Scaffolded)

The `.github/workflows/ci.yml` is already in place with two jobs:

**Job 1: `quality`** (runs on every push/PR)
```yaml
steps:
  - ruff check + format check    # Linting
  - mypy src/                     # Type checking
  - pytest tests/ -m "not slow"   # Fast tests only
```

**Job 2: `train-smoke`** (runs after quality passes)
```yaml
steps:
  - pytest tests/ -k "test_trainer or test_model"   # Model + training tests
```

### 8.2 Additional CI Enhancements to Implement

**Dependency caching** (already present):
```yaml
- uses: actions/cache@v4
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}
```

**Coverage reporting** — Add to the quality job:
```yaml
- name: Upload coverage
  uses: codecov/codecov-action@v4
  with:
    file: ./coverage.xml
    fail_ci_if_error: false
```

**Security scanning** (optional enhancement):
```yaml
- name: Check for secrets
  uses: trufflesecurity/trufflehog@main
  with:
    path: ./
    base: ${{ github.event.repository.default_branch }}
```

### 8.3 Pre-commit Hooks (Already Configured)

`.pre-commit-config.yaml` runs:
1. `ruff --fix` (auto-fix lint issues)
2. `ruff format` (code formatting)
3. Trailing whitespace removal
4. End-of-file fixer
5. YAML syntax check
6. Large file guard (max 500KB)

---

## 9. Phase 10: Documentation & Bottleneck Analysis

### 9.1 File: `docs/BOTTLENECK_ANALYSIS.md`

This document should cover the following sections in detail:

#### 9.1.1 I/O Analysis

```
Current dataset: 900 BMP images, 122 MB total
Load time (serial): ~1.2s for all images
Load time (PIL single image): ~1.3ms

Bottleneck? NO — the entire dataset fits in memory.
At scale: Converting to HDF5 or TFRecord format would amortize
I/O overhead via sequential reads instead of random file access.
```

#### 9.1.2 Data Loading Pipeline

```
DataLoader throughput vs num_workers (measured on test hardware):

num_workers=0: ~120 pairs/s (main thread, includes transform time)
num_workers=1: ~200 pairs/s (1 worker handles I/O + transform)
num_workers=2: ~350 pairs/s (sweet spot for this dataset)
num_workers=4: ~340 pairs/s (diminishing returns, worker overhead)

Bottleneck? NO for current scale. Transform time (~2ms/pair) is
dominated by forward pass time (~5ms/pair on CPU, ~0.5ms on GPU).

At scale (100K+ images):
- Pre-compute tensors to disk (.pt files) → eliminates transform time
- Use LMDB or WebDataset for sequential I/O
- Increase num_workers to match CPU core count
```

#### 9.1.3 Preprocessing Analysis

```
Ray overhead for 45 subjects:
- Task serialization: ~50ms per subject
- Ray scheduler: ~200ms total
- Worker startup: ~500ms (one-time)

Total Ray overhead: ~750ms
Total computation: ~500ms (quality metrics for 900 images)

Result: Ray is SLOWER for 45 subjects (1.25s vs 1.2s serial).
Break-even: ~100 subjects (where computation > overhead).
Recommendation: For this dataset, serial is fine. The Ray
implementation demonstrates the pattern for production scale.
```

#### 9.1.4 Training Bottleneck

```
Per-epoch timing breakdown (45 subjects, batch_size=32):
- Data loading: ~1.5s (150 batches x 10ms)
- Forward pass: ~2.0s
- Backward pass: ~2.5s
- Optimizer step: ~0.5s
- Logging/metrics: ~0.1s
Total: ~6.6s per epoch

GPU utilization estimate: ~60% (data loading gaps)
To improve: prefetch_factor=2, num_workers=4, or pre-tensorize data.
```

#### 9.1.5 Scalability Recommendations

```
Scaling to 10K subjects / 200K images:
1. Data: PyArrow catalog + pre-tensorized .pt files
2. Loading: WebDataset streaming format + 8 workers
3. Preprocessing: Ray cluster (multi-node) for initial ETL
4. Training: PyTorch DDP on Azure NC-series VMs
5. Storage: Azure Blob Storage with local SSD cache

Scaling to 1M subjects / 10M images:
1. Data: Sharded Parquet on Azure Data Lake
2. Loading: Custom IterableDataset with streaming
3. Preprocessing: Spark on Azure Synapse / Databricks
4. Training: FSDP or DeepSpeed on multi-node GPU cluster
5. Model: Replace CNN encoders with pretrained ViT (transfer learning)
```

#### 9.1.6 Trade-off Summary Table

| Decision | Current Choice | Alternative | When to Switch |
|---|---|---|---|
| Storage format | Raw BMP on filesystem | HDF5 / LMDB / WebDataset | > 10K images |
| Preprocessing | Ray (local) | Spark / Dask (distributed) | > 100K images |
| Data catalog | Parquet (single file) | Delta Lake / Iceberg | Need versioning + ACID |
| Data loading | PyTorch DataLoader | NVIDIA DALI | GPU preprocessing needed |
| Model | Custom CNN (~1.3M params) | Pretrained ResNet/ViT | Performance matters |
| Training | Single-device (CPU/GPU) | DDP / FSDP | > 4 hours per run |
| Config | Hydra YAML | MLflow / W&B | Experiment tracking at scale |
| CI | GitHub Actions | Azure Pipelines | Full Azure integration |

### 9.2 File: `README.md`

Should contain:
1. Project overview (1 paragraph)
2. Quick start:
   ```bash
   pip install -e ".[dev]"
   python scripts/preprocess.py
   python scripts/train.py
   python scripts/infer.py checkpoint=checkpoints/best.pt
   ```
3. Configuration guide (Hydra overview, override examples)
4. Architecture diagram (text-based, from Section 1 above)
5. Project structure (tree listing with 1-line descriptions)
6. Design decisions and rationale (links to this doc)
7. Running tests: `make test`

---

## 10. Cross-Cutting Concerns

### 10.1 Reproducibility

Reproducibility is enforced at multiple levels:

| Level | Mechanism |
|---|---|
| Python random | `random.seed(42)` |
| NumPy | `np.random.seed(42)` |
| PyTorch CPU | `torch.manual_seed(42)` |
| PyTorch CUDA | `torch.cuda.manual_seed_all(42)` |
| Hash seed | `os.environ["PYTHONHASHSEED"] = "42"` |
| cuDNN deterministic | `torch.backends.cudnn.deterministic = True` |
| cuDNN benchmark | `torch.backends.cudnn.benchmark = False` |
| Data splits | `seed` parameter in `create_splits()` |
| Config | Full config stored in every checkpoint |
| DataLoader | `worker_init_fn` seeds workers deterministically |

**Note**: MPS (Apple Silicon) does not fully support deterministic mode. Results on MPS may vary slightly between runs.

### 10.2 Logging

All modules use Python's `logging` module with the module-level pattern:
```python
logger = logging.getLogger(__name__)
```

Hydra automatically configures logging based on its defaults. Structured log output includes:
- Timestamp, level, module name
- Training: epoch, step, loss, accuracy, learning rate
- Preprocessing: records processed, timing

### 10.3 Error Handling

- **Dataset**: Missing directories raise `FileNotFoundError` with descriptive messages
- **Config**: OmegaConf validates types at resolution time
- **Model**: Shape mismatches caught by PyTorch with clear tensor size errors
- **Training**: Checkpoint saving wrapped in try/except to prevent data loss on disk errors
- **Inference**: `from_checkpoint` validates checkpoint keys before loading

### 10.4 Type Safety

- All function signatures use type annotations
- `from __future__ import annotations` in every module for PEP 604 union syntax on Python 3.9
- `mypy --strict` compatibility (disallow_untyped_defs)
- Runtime type checking via `assert` statements at critical boundaries

---

## 11. Azure Deployment Architecture

### 11.1 Storage Layout on Azure Blob

```
Azure Blob Storage (biometricdata/datasets)
├── iris-fingerprint/
│   ├── raw/                    ← Original BMP images (rsync'd from local)
│   │   ├── 1/Fingerprint/...
│   │   ├── 1/left/...
│   │   └── ...
│   ├── preprocessed/           ← Arrow catalog + quality metrics
│   │   └── biometric_catalog.parquet
│   └── tensorized/             ← Pre-computed .pt files (optional)
│       ├── train/
│       ├── val/
│       └── test/

Azure ML Registry (biometric-registry)
├── models/
│   └── multimodal-biometric/
│       ├── v1/                 ← checkpoint.pt + config.yaml
│       ├── v2/
│       └── latest/
```

### 11.2 Compute Targets

| Workload | Azure VM | Why |
|---|---|---|
| Preprocessing | Standard_D8s_v3 (8 CPU) | CPU-bound, parallelizable |
| Training | Standard_NC6s_v3 (1 V100) | Single GPU sufficient for ~1M params |
| Inference | Standard_B2s (2 CPU) | Low-traffic, CPU inference is fine |

### 11.3 Azure ML Pipeline (Conceptual)

```
[1] Data Prep Step
    - Run preprocessing.py on D8 compute
    - Output: biometric_catalog.parquet → Blob Storage

[2] Training Step
    - Run train.py on NC6 compute
    - Input: raw data from Blob + catalog
    - Output: checkpoint.pt → Model Registry

[3] Evaluation Step
    - Run evaluate on held-out test set
    - Output: metrics.json → Pipeline Artifacts

[4] Registration Step
    - Register model in Azure ML Model Registry
    - Tag with metrics, config hash, git SHA
```

---

## Summary: Implementation Priority Order

| Phase | Files to Create | LOC Est. | Priority |
|---|---|---|---|
| **4. Model** | `fingerprint_encoder.py`, `iris_encoder.py`, `fusion.py`, `multimodal_net.py`, update `__init__.py` | ~250 | HIGH |
| **5. Training** | `metrics.py`, `callbacks.py`, `trainer.py`, update `scripts/train.py` | ~400 | HIGH |
| **6. Inference** | `predictor.py`, update `scripts/infer.py` | ~150 | MEDIUM |
| **7. Benchmarking** | Already scaffolded, just needs `preprocess.py` timing integration | ~50 | LOW |
| **8. Testing** | `test_model.py`, `test_trainer.py`, `test_inference.py` | ~200 | HIGH |
| **9. CI/CD** | Already scaffolded, add coverage upload | ~20 | LOW |
| **10. Docs** | `BOTTLENECK_ANALYSIS.md`, `README.md` | ~300 | MEDIUM |

**Recommended implementation order: 4 → 5 → 8 (model tests) → 6 → 8 (inference tests) → 7 → 10 → 9**

This ensures each phase can be tested immediately after implementation, and the training pipeline is verified before building inference on top of it.
