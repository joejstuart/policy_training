# Quick Setup Guide

## Python Version Issue

**Problem**: Python 3.13 is not yet fully supported by PyTorch.

**Solution**: Use Python 3.10-3.12. The project is configured to automatically use a compatible version.

## Setup Steps

### Option 1: From qwen2.5_model directory (Recommended)

```bash
cd qwen2.5_model
uv sync --python 3.12
```

Then run scripts:
```bash
uv run python generate_dataset.py
uv run python train_policy.py --train-path train.jsonl --eval-path eval.jsonl --output-dir ../qwen2.5-rego-policy-lora
```

### Option 2: From repository root

```bash
# First time setup
cd qwen2.5_model
uv sync --python 3.12
cd ..

# Run scripts with --project flag
uv run --project qwen2.5_model python qwen2.5_model/generate_dataset.py
uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/train.jsonl \
    --eval-path qwen2.5_model/eval.jsonl \
    --output-dir qwen2.5-rego-policy-lora
```

## Verify Setup

Test that PyTorch works:
```bash
cd qwen2.5_model
uv run python -c "import torch; print(f'PyTorch {torch.__version__} works!'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

You should see:
```
PyTorch 2.9.1 works!
MPS available: True
```

## Troubleshooting

### If you get Python 3.13 errors:

1. Make sure you have Python 3.12 installed:
   ```bash
   which python3.12
   ```

2. If not installed, install it:
   ```bash
   brew install python@3.12
   ```

3. Then sync with explicit version:
   ```bash
   cd qwen2.5_model
   uv sync --python 3.12
   ```

### If uv can't find the project:

Make sure you're either:
- In the `qwen2.5_model` directory, OR
- Using `--project qwen2.5_model` flag from root

