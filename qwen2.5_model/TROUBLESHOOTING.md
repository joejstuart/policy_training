# Troubleshooting Guide

## Training Hangs at 3% (MPS/Apple Silicon)

**Symptom**: Training starts, gets to ~3% progress, then hangs indefinitely.

**Cause**: Common issue with MPS (Metal Performance Shaders) on Apple Silicon. The first evaluation or first few batches can hang due to MPS synchronization issues.

### Solutions (try in order):

#### 1. Disable Evaluation During Training (Fastest Fix)

```bash
uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/train.jsonl \
    --eval-path qwen2.5_model/eval.jsonl \
    --output-dir qwen2.5-rego-policy-lora \
    --eval-strategy no \
    --disable-gradient-checkpointing \
    --batch-size 4 \
    --grad-accum-steps 2
```

This skips evaluation during training. You can evaluate separately after training completes.

#### 2. Use Epoch-Based Evaluation (Less Frequent)

```bash
uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/train.jsonl \
    --eval-path qwen2.5_model/eval.jsonl \
    --output-dir qwen2.5-rego-policy-lora \
    --eval-strategy epoch \
    --disable-gradient-checkpointing \
    --batch-size 4 \
    --grad-accum-steps 2
```

This only evaluates at the end of each epoch, reducing MPS synchronization points.

#### 3. Reduce Batch Size

If the above don't work, try smaller batches:

```bash
uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/train.jsonl \
    --eval-path qwen2.5_model/eval.jsonl \
    --output-dir qwen2.5-rego-policy-lora \
    --eval-strategy no \
    --batch-size 2 \
    --grad-accum-steps 4
```

#### 4. Enable Gradient Checkpointing

If you disabled it, try re-enabling (trades speed for stability):

```bash
uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/train.jsonl \
    --eval-path qwen2.5_model/eval.jsonl \
    --output-dir qwen2.5-rego-policy-lora \
    --eval-strategy no \
    --batch-size 2 \
    --grad-accum-steps 4
# Note: No --disable-gradient-checkpointing flag
```

#### 5. Use CPU Instead (Slow but Stable)

If MPS continues to cause issues:

```bash
# Set environment variable to force CPU
export PYTORCH_ENABLE_MPS_FALLBACK=1
uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/train.jsonl \
    --eval-path qwen2.5_model/eval.jsonl \
    --output-dir qwen2.5-rego-policy-lora \
    --eval-strategy no
```

## Other Common Issues

### Out of Memory

**Symptom**: `RuntimeError: MPS backend out of memory`

**Solutions**:
1. Reduce batch size: `--batch-size 1`
2. Enable gradient checkpointing (remove `--disable-gradient-checkpointing`)
3. Reduce max sequence length: `--max-seq-len 512`

### Slow Training

**Symptom**: Training is very slow

**Solutions**:
1. Disable gradient checkpointing: `--disable-gradient-checkpointing`
2. Increase batch size (if memory allows): `--batch-size 4` or `8`
3. Reduce gradient accumulation: `--grad-accum-steps 2` or `1`
4. Disable evaluation: `--eval-strategy no`

### Model Loading Errors

**Symptom**: `Error loading model` or authentication errors

**Solutions**:
1. Authenticate with HuggingFace: `huggingface-cli login`
2. Check model name: Should be `Qwen/Qwen2.5-1.5B-Instruct`
3. Check internet connection (model downloads on first run)

### Python 3.13 Errors

**Symptom**: `ImportError: cannot import name 'Library' from 'torch.library'`

**Solution**: Use Python 3.10-3.12:
```bash
cd qwen2.5_model
uv sync --python 3.12
```

## Monitoring Training

Watch for these signs:

- ✅ **Good**: Train loss decreasing steadily
- ✅ **Good**: Eval loss decreasing (if evaluation enabled)
- ⚠️ **Warning**: Eval loss increasing while train loss decreases (overfitting - stop early)
- ❌ **Bad**: Loss not changing (learning rate too low or model not training)
- ❌ **Bad**: Loss exploding (learning rate too high - reduce `--learning-rate`)

## Getting Help

If issues persist:
1. Check PyTorch version: `uv run python -c "import torch; print(torch.__version__)"`
2. Check MPS availability: `uv run python -c "import torch; print(torch.backends.mps.is_available())"`
3. Try the most conservative settings:
   ```bash
   uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
       --train-path qwen2.5_model/train.jsonl \
       --eval-path qwen2.5_model/eval.jsonl \
       --output-dir qwen2.5-rego-policy-lora \
       --eval-strategy no \
       --batch-size 1 \
       --grad-accum-steps 8
   ```

