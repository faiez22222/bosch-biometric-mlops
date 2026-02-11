.PHONY: install lint typecheck test preprocess train benchmark all clean

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src/ tests/ scripts/
	ruff format --check src/ tests/ scripts/

format:
	ruff check --fix src/ tests/ scripts/
	ruff format src/ tests/ scripts/

typecheck:
	mypy src/

test:
	pytest tests/ -v --cov=src/biometric --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow" --cov=src/biometric

preprocess:
	python scripts/preprocess.py

train:
	python scripts/train.py

train-debug:
	python scripts/train.py data=fast_dev training=debug

infer:
	python scripts/infer.py

benchmark:
	python scripts/benchmark_dataloader.py

all: lint typecheck test

clean:
	rm -rf outputs/ checkpoints/ preprocessed/ .mypy_cache/ .pytest_cache/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
