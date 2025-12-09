# Memory Optimization Guide for Qwen3-Coder-30B

If you're getting "CUDA out of memory" errors, try these optimizations in order:

## Quick Fix: Set Environment Variable

Before running training, set this environment variable:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Or add it to your training command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python qwen2.5_model/train_policy.py ...
```

This reduces memory fragmentation and can free up significant memory.

## Memory Optimization Steps (Try in Order)

### Step 1: Minimal Settings (Maximum Memory Savings)

```bash
python qwen2.5_model/train_policy.py \
    --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --train-path data/compiler_errors/combined_train.jsonl \
    --eval-path data/compiler_errors/combined_eval.jsonl \
    --output-dir qwen3-coder-30b-rego-policy-lora \
    --use-4bit \
    --batch-size 1 \
    --grad-accum-steps 16 \
    --learning-rate 2e-4 \
    --lora-r 4 \
    --lora-alpha 8 \
    --num-epochs 3
```

**Changes:**
- `--lora-r 4` (minimum rank, reduces LoRA memory)
- `--grad-accum-steps 16` (larger accumulation to compensate for small batch)

### Step 2: Clear GPU Cache Before Training

```bash
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"
```

Then immediately run training.

### Step 3: Reduce Sequence Length

If your examples are shorter, reduce max sequence length:

```bash
--max-seq-len 512  # Instead of 1024
```

### Step 4: Disable Gradient Checkpointing (Uses More Memory, But Faster)

Only if you have a bit more headroom:

```bash
--disable-gradient-checkpointing
```

**Warning**: This uses MORE memory, but if you're close to the limit and checkpointing is causing issues, try it.

### Step 5: Use CPU Offloading (Last Resort)

If nothing else works, you may need to use DeepSpeed ZeRO or reduce model size.

## Memory Usage Breakdown

With QLoRA (4-bit) on 22GB GPU:

| Component | Memory Usage |
|-----------|--------------|
| Quantized model (4-bit) | ~15-16 GB |
| LoRA adapters (r=4) | ~50-100 MB |
| LoRA adapters (r=8) | ~100-200 MB |
| LoRA adapters (r=16) | ~200-400 MB |
| Training overhead | ~3-5 GB |
| **Total (r=4)** | **~18-21 GB** |
| **Total (r=8)** | **~18-21 GB** |
| **Total (r=16)** | **~19-22 GB** |

## Recommended Settings for 22GB GPU

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python qwen2.5_model/train_policy.py \
    --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --train-path data/compiler_errors/combined_train.jsonl \
    --eval-path data/compiler_errors/combined_eval.jsonl \
    --output-dir qwen3-coder-30b-rego-policy-lora \
    --use-4bit \
    --batch-size 1 \
    --grad-accum-steps 16 \
    --learning-rate 2e-4 \
    --lora-r 4 \
    --lora-alpha 8 \
    --max-seq-len 512 \
    --num-epochs 3
```

## What Changed in the Code

1. **LoRA target modules**: For 4-bit models, only attention layers are used (not MLP) - saves ~200-400MB
2. **Eval batch size**: Automatically set to 1 for 4-bit models
3. **Environment variable**: Set to `expandable_segments:True` by default
4. **Memory optimizations**: Added DDP and other memory-saving flags

## Monitoring Memory

During training, you can monitor memory usage:

```bash
# In another terminal
watch -n 1 nvidia-smi
```

If memory usage is still too high, try:
1. Lower LoRA rank to 4
2. Increase gradient accumulation to 16 or 32
3. Reduce max sequence length to 512
4. Clear cache between runs

## Alternative: Use Smaller Model

If memory is still an issue, consider:
- `Qwen/Qwen2.5-1.5B-Instruct` (your current model) - ~3-4GB VRAM
- `Qwen/Qwen2.5-3B-Instruct` - ~6-8GB VRAM
- `Qwen/Qwen2.5-7B-Instruct` - ~14-16GB VRAM (with QLoRA)

