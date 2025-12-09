# Training Guide: Qwen3-Coder-30B-A3B-Instruct

This guide explains how to fine-tune Qwen3-Coder-30B-A3B-Instruct with QLoRA (4-bit quantization) for memory efficiency.

## Prerequisites

### 1. Install Required Dependencies

```bash
pip install bitsandbytes
```

BitsAndBytes is required for 4-bit quantization (QLoRA).

### 2. Hardware Requirements

- **Minimum**: 17.5 GB VRAM (with QLoRA)
- **Recommended**: 24+ GB VRAM
- **GPU**: CUDA-compatible GPU (NVIDIA)
- **Note**: MPS (Apple Silicon) and CPU are not supported for 4-bit quantization

## Training Command

### Basic Training (Memory-Optimized)

```bash
python qwen2.5_model/train_policy.py \
    --model-name Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --train-path data/compiler_errors/combined_train.jsonl \
    --eval-path data/compiler_errors/combined_eval.jsonl \
    --output-dir qwen3-coder-30b-rego-policy-lora \
    --use-4bit \
    --batch-size 1 \
    --grad-accum-steps 8 \
    --learning-rate 2e-4 \
    --lora-r 8 \
    --lora-alpha 16 \
    --num-epochs 3
```

### Recommended Settings for Qwen3-Coder-30B

```python
# Memory-optimized configuration
--use-4bit                    # Enable 4-bit quantization (required)
--batch-size 1                # Minimal batch size
--grad-accum-steps 8          # Effective batch size = 8
--learning-rate 2e-4          # Slightly higher for QLoRA
--lora-r 8                    # Lower rank = less memory (can increase to 16 if memory allows)
--lora-alpha 16               # Alpha = 2 * r
--num-epochs 3                # Standard epochs
```

### If You Have More Memory (24GB+ VRAM)

You can increase:
- `--batch-size 2` (instead of 1)
- `--grad-accum-steps 4` (instead of 8)
- `--lora-r 16` (instead of 8)
- `--lora-alpha 32` (instead of 16)

## Memory Usage Breakdown

With QLoRA (4-bit quantization):
- **Model weights**: ~15-18 GB (4-bit quantized from ~60GB)
- **LoRA adapters**: ~100-200 MB (depends on rank)
- **Training overhead**: ~2-4 GB (gradients, optimizer states)
- **Total**: ~17.5-22 GB VRAM

## Troubleshooting

### Out of Memory Errors

1. **Reduce batch size**: `--batch-size 1`
2. **Increase gradient accumulation**: `--grad-accum-steps 16`
3. **Lower LoRA rank**: `--lora-r 4` (minimum)
4. **Enable gradient checkpointing**: Already enabled by default

### "BitsAndBytes not available" Error

```bash
pip install bitsandbytes
```

### "4-bit quantization requires CUDA" Warning

QLoRA only works with CUDA (NVIDIA GPUs). If you see this warning:
- You're using MPS (Apple Silicon) or CPU
- Switch to a CUDA-enabled machine
- Or use a smaller model without quantization

### Model Loading Fails

1. **Check model name**: `Qwen/Qwen3-Coder-30B-A3B-Instruct`
2. **Authenticate**: `huggingface-cli login` (if model is gated)
3. **Check disk space**: Model is ~60GB (downloads in FP16, then quantized)

## Expected Training Time

With recommended settings on a 24GB GPU:
- **Per epoch**: ~2-4 hours (depends on dataset size)
- **Total (3 epochs)**: ~6-12 hours

## Monitoring Training

Watch for:
- **Loss decreasing**: Should drop steadily
- **Memory usage**: Should stay under VRAM limit
- **Validation loss**: Should track training loss

## After Training

Test the model:

```bash
python qwen2.5_model/infer_policy.py \
    --model-dir qwen3-coder-30b-rego-policy-lora \
    --base-model Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --instruction "Write a Rego rule that checks all tasks succeeded"
```

## Comparison with Qwen2.5-1.5B

| Aspect | Qwen2.5-1.5B | Qwen3-Coder-30B (QLoRA) |
|--------|--------------|-------------------------|
| VRAM Required | ~3-4 GB | ~17.5-22 GB |
| Model Size | 1.5B params | 30B params |
| Code Quality | Good | Excellent |
| Training Speed | Fast | Slower |
| Best For | Quick iterations | Production quality |

## Next Steps

1. **Start training** with the recommended settings
2. **Monitor** memory usage and loss
3. **Adjust** batch size/rank if needed
4. **Evaluate** on test set after training

