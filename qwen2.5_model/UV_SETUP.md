# UV Setup Guide

All scripts have been converted to use `uv` for dependency management.

## Quick Start

### 1. Install Dependencies

From the `qwen2.5_model` directory:
```bash
cd qwen2.5_model
uv sync
```

Or from repository root:
```bash
uv sync --project qwen2.5_model
```

**Note**: Python 3.13 is not yet fully supported by PyTorch. The project is configured to use Python 3.10-3.12. If you have Python 3.13, uv will automatically use a compatible version (3.12 if available).

### 2. Run Scripts

All scripts now use `uv run`:

```bash
# Generate dataset
uv run python qwen2.5_model/generate_dataset.py

# Train model
uv run python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/train.jsonl \
    --eval-path qwen2.5_model/eval.jsonl \
    --output-dir qwen2.5-rego-policy-lora

# Inference helper
uv run python qwen2.5_model/inference_helper.py <package> <import1> [import2 ...]
```

## Python Version Compatibility

**Issue**: Python 3.13 is not yet fully supported by PyTorch.

**Solution**: The `pyproject.toml` specifies `requires-python = ">=3.10,<3.13"` to ensure compatibility.

If you have Python 3.13 installed, uv will automatically use Python 3.12 (or 3.11/3.10 if available) when you run `uv sync`.

To explicitly use a specific Python version:
```bash
uv sync --python 3.12
```

## What Changed

1. **Created `pyproject.toml`**: Defines Python dependencies (transformers, peft, torch, accelerate)
2. **Updated all documentation**: All examples now use `uv run python` instead of `python`
3. **Updated `requirements.txt`**: Added training dependencies (still works with pip if needed)
4. **Python version constraint**: Limited to Python 3.10-3.12 for PyTorch compatibility

## Dependencies

The `pyproject.toml` includes:
- `transformers>=4.40.0` - For model loading and training
- `peft>=0.10.0` - For LoRA fine-tuning
- `torch>=2.1.0` - PyTorch backend (updated for better compatibility)
- `accelerate>=0.30.0` - For distributed training support

## Alternative: Using pip

If you prefer pip, you can still use:
```bash
pip install -r requirements.txt
python qwen2.5_model/generate_dataset.py
```

But `uv` is recommended for:
- Faster dependency resolution
- Better dependency management
- Reproducible environments
- Automatic virtual environment management
- Automatic Python version selection
