# Multimodal Biometric Recognition System

A production-quality MLOps pipeline for multimodal (fingerprint + iris) biometric identity recognition, built with PyTorch, Hydra, Ray, and PyArrow.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Preprocess dataset (Ray parallel + PyArrow catalog)
DATA_ROOT="/path/to/IRIS and FINGERPRINT DATASET" python scripts/preprocess.py

# Train model
DATA_ROOT="/path/to/IRIS and FINGERPRINT DATASET" python scripts/train.py

# Train (fast debug mode)
DATA_ROOT="/path/to/IRIS and FINGERPRINT DATASET" python scripts/train.py data=fast_dev training=debug

# Inference
python scripts/infer.py checkpoint_path=./checkpoints/best.pt \
    fingerprint_path=./data/1/Fingerprint/1__M_Left_index_finger.BMP \
    iris_path=./data/1/left/aeval1.bmp

# Benchmark data loading
DATA_ROOT="/path/to/IRIS and FINGERPRINT DATASET" python scripts/benchmark_dataloader.py

# Run tests
make test
```

## Architecture

```
Raw BMP Images (900 files, 45 subjects)
        │
        ▼
┌────────────────────────────────────┐
│  Ray Parallel Preprocessing        │
│  45 subjects → parallel workers    │
│  Quality metrics → PyArrow catalog │
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│  PyTorch Data Pipeline             │
│  MultimodalBiometricDataset        │
│  4,500 (fingerprint, iris) pairs   │
│  Subject-level stratified splits   │
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│  Multimodal CNN                    │
│  FingerprintEncoder → [256]        │
│  IrisEncoder → [256]               │
│  Concatenation Fusion → [45]       │
│  ~1.3M parameters                  │
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│  Training Pipeline                 │
│  Adam + Cosine LR + Grad Clipping  │
│  Checkpointing + Early Stopping    │
│  Full Reproducibility (seed=42)    │
└──────────────┬─────────────────────┘
               ▼
┌────────────────────────────────────┐
│  Inference Pipeline                │
│  BiometricPredictor.from_checkpoint│
│  → Top-K predictions + confidence  │
└────────────────────────────────────┘
```

## Project Structure

```
bosch-biometric-mlops/
├── configs/                     # Hydra config hierarchy
│   ├── config.yaml              #   Main config (composes groups)
│   ├── data/                    #   Dataset, transforms, preprocessing
│   ├── model/                   #   Model architecture hyperparameters
│   ├── training/                #   Training loop settings
│   └── infrastructure/          #   Azure deployment hints
├── src/biometric/               # Main Python package
│   ├── data/                    #   Dataset, transforms, DataModule, Arrow, Ray
│   ├── models/                  #   CNN encoders, fusion, top-level network
│   ├── training/                #   Trainer, metrics, callbacks
│   ├── inference/               #   Predictor (checkpoint → predictions)
│   └── utils/                   #   Reproducibility, logging
├── scripts/                     # CLI entry points
│   ├── preprocess.py            #   Ray preprocessing + Arrow catalog
│   ├── train.py                 #   Model training
│   ├── infer.py                 #   Inference from checkpoint
│   └── benchmark_dataloader.py  #   Data loading benchmarks
├── tests/                       # 57 tests (pytest)
├── docs/                        # Architecture + bottleneck analysis
├── .github/workflows/ci.yml     # GitHub Actions CI
├── pyproject.toml               # Dependencies + tool config
└── Makefile                     # Dev commands
```

## Configuration

The project uses [Hydra](https://hydra.cc/) for hierarchical, composable configuration. Override any parameter from the CLI:

```bash
# Change batch size and learning rate
python scripts/train.py data.batch_size=64 training.learning_rate=0.0005

# Use debug training with fast_dev data
python scripts/train.py data=fast_dev training=debug

# Change model architecture
python scripts/train.py model=fingerprint_only

# Set number of Ray preprocessing workers
python scripts/preprocess.py data.preprocessing.num_ray_workers=8
```

## Dataset

| Modality | Images/Subject | Resolution | Format |
|---|---|---|---|
| Fingerprint | 10 (5 left + 5 right hand) | 96x103 px | Grayscale BMP |
| Iris | 10 (5 left + 5 right eye) | 320x240 px | RGB BMP |
| **Total** | **20 per subject, 900 total** | | |

Subjects: 45 (34 Male, 11 Female)
Training pairs: 4,500 (cross-product of 10 fingerprints x 10 iris per subject)

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Split strategy | By subject | Prevents data leakage — same person never in train+test |
| Config system | Hydra | Composable, CLI overrides, experiment management |
| Preprocessing | Ray | Demonstrates distributed pattern; scales to clusters |
| Data catalog | PyArrow Parquet | Columnar, fast metadata queries, no DB server |
| Model framework | Pure PyTorch | Shows low-level understanding; no Lightning abstraction |
| Fusion | Concatenation | Simplest correct approach for 45-class, 4,500-sample task |
| Reproducibility | Full seed control | Config in checkpoints, deterministic mode |

## Testing

```bash
make test        # Run all 57 tests with coverage
make test-fast   # Skip slow training tests
make lint        # Ruff linting
make typecheck   # mypy type checking
make all         # lint + typecheck + test
```

## Documentation

- [Architecture Document](docs/ARCHITECTURE.md) — System design, component specs, Azure deployment
- [Bottleneck Analysis](docs/BOTTLENECK_ANALYSIS.md) — Performance analysis, scalability roadmap, trade-offs
