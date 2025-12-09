# CUDA GPU Training Recommendations

## Dataset Size
- **Total:** 3,120 examples (2,808 train, 312 eval)
- This is a **larger dataset** than the default settings were designed for (~260 examples)

## Training Data Characteristics

Based on analysis of 2,808 training examples:
- **Average sequence length**: ~755 tokens
- **Median sequence length**: ~540 tokens
- **95th percentile**: ~1,453 tokens
- **99th percentile**: ~6,340 tokens
- **Max sequence length**: ~10,458 tokens (outlier)
- **14.1% of examples exceed 1024 tokens**
- **4.8% of examples exceed 1536 tokens**

### Component Breakdown
- **Context (trimmed attestation JSON)**: Average ~643 tokens, max ~10,389 tokens
- **Output Code (Rego)**: Average ~87 tokens, max ~227 tokens
- **Instruction**: Average ~25 tokens, max ~100 tokens

### Coverage by max_seq_len
| max_seq_len | Coverage | Truncated |
|-------------|----------|-----------|
| 1024 | 85.9% | 397 examples |
| 1280 | 92.3% | 216 examples |
| **1536** | **95.2%** | **134 examples** |
| 2048 | 96.2% | 106 examples |
| 4096 | 96.9% | 87 examples |

**Note**: Even 4096 doesn't cover all examples - some are extreme outliers.

## Recommended CUDA Settings

### Option 1: Balanced (Recommended for Most GPUs)

Good balance of speed and memory usage. Works well on GPUs with 8GB+ VRAM (e.g., RTX 3060, RTX 3070, A10G).

**Choose one based on your priority:**

#### Option 1a: Maximum Coverage (Recommended if you want 95% coverage)

```bash
python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/attestation_train.jsonl \
    --eval-path qwen2.5_model/attestation_eval.jsonl \
    --output-dir qwen2.5-attestation-parse \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --learning-rate 5e-5 \
    --num-epochs 3 \
    --eval-strategy steps \
    --max-seq-len 1536 \
    --dataloader-num-workers 2
```

**Settings:**
- `--batch-size 4`: Reduced to fit 1536 sequence length
- `--grad-accum-steps 4`: Maintains effective batch size = 16
- `--max-seq-len 1536`: Covers 95.2% of examples (vs 85.9% with 1024)
- `--dataloader-num-workers 2`: Speeds up data loading on CUDA
- **Gradient checkpointing enabled** (default): Required for CUDA stability
- Effective batch size: 4 × 4 = **16**
- Steps per epoch: ~175, Total steps: ~525

**Trade-off**: Slightly slower training (smaller batch) but better coverage of training examples.

#### Option 1b: Faster Training (Stable default)

```bash
python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/attestation_train.jsonl \
    --eval-path qwen2.5_model/attestation_eval.jsonl \
    --output-dir qwen2.5-attestation-parse \
    --batch-size 8 \
    --grad-accum-steps 2 \
    --learning-rate 5e-5 \
    --num-epochs 3 \
    --eval-strategy steps \
    --max-seq-len 1024 \
    --dataloader-num-workers 2
```

**Settings:**
- `--batch-size 8`: Larger batch for faster training
- `--grad-accum-steps 2`: Effective batch size = 16
- `--max-seq-len 1024`: Stable, covers 85.9% of examples
- `--dataloader-num-workers 2`: Speeds up data loading on CUDA
- **Gradient checkpointing enabled** (default): Required for CUDA stability
- Effective batch size: 8 × 2 = **16**
- Steps per epoch: ~175, Total steps: ~525

**Trade-off**: Faster training but 14.1% of examples are truncated.

**Note:** Keep gradient checkpointing enabled (don't use `--disable-gradient-checkpointing`) - it's required for CUDA stability on many systems.

### Option 2: Fast Training (High-End GPUs)

For GPUs with 16GB+ VRAM (e.g., RTX 3090, RTX 4090, A100, A10G with more memory).

```bash
python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/attestation_train.jsonl \
    --eval-path qwen2.5_model/attestation_eval.jsonl \
    --output-dir qwen2.5-attestation-parse \
    --batch-size 16 \
    --grad-accum-steps 1 \
    --learning-rate 5e-5 \
    --num-epochs 3 \
    --eval-strategy steps \
    --max-seq-len 1024 \
    --dataloader-num-workers 4
```

**Note**: If you have 16GB+ VRAM and want better coverage, you can try `--max-seq-len 1536` or `2048` with this configuration.

**Settings:**
- `--batch-size 16`: Large batch for fast training
- `--grad-accum-steps 1`: No accumulation needed with large batch
- **Gradient checkpointing enabled** (default): Keep enabled for stability
- Effective batch size: 16 × 1 = **16**
- **Warning:** Monitor GPU memory usage - reduce batch size if OOM errors occur
- **Note:** Even on high-end GPUs, gradient checkpointing may be required for CUDA stability

### Option 3: Memory-Constrained (Smaller GPUs)

For GPUs with 6-8GB VRAM (e.g., RTX 2060, RTX 3050).

```bash
python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/attestation_train.jsonl \
    --eval-path qwen2.5_model/attestation_eval.jsonl \
    --output-dir qwen2.5-attestation-parse \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --learning-rate 5e-5 \
    --num-epochs 3 \
    --eval-strategy steps \
    --max-seq-len 1024 \
    --dataloader-num-workers 2
```

**Settings:**
- `--batch-size 4`: Smaller batch for memory safety
- `--grad-accum-steps 4`: Maintain effective batch size
- `--max-seq-len 1024`: Safe for memory-constrained GPUs
- **Keep gradient checkpointing enabled** (don't disable)
- Effective batch size: 4 × 4 = **16**

### Option 4: Alternative Configurations

If you want to experiment with different coverage/speed trade-offs:

```bash
# Option 4a: 1280 tokens with batch 8 (covers 92.3% vs 85.9%)
python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/attestation_train.jsonl \
    --eval-path qwen2.5_model/attestation_eval.jsonl \
    --output-dir qwen2.5-attestation-parse \
    --batch-size 8 \
    --grad-accum-steps 2 \
    --learning-rate 5e-5 \
    --num-epochs 3 \
    --eval-strategy steps \
    --max-seq-len 1280 \
    --dataloader-num-workers 2
```

**Note**: Option 1a above (1536 with batch_size=4) is the recommended approach for maximum coverage.

## Key Differences from MPS Defaults

| Setting | MPS Default | CUDA Recommended | Reason |
|---------|-------------|-------------------|--------|
| `--batch-size` | 2 | 8-16 | CUDA GPUs have more VRAM |
| `--grad-accum-steps` | 4 | 1-2 | Larger batches need less accumulation |
| `--disable-gradient-checkpointing` | No | **No (keep enabled)** | Required for CUDA stability on many systems |
| Effective batch size | 8 | 16 | Larger dataset benefits from larger effective batch |

**Important:** Keep gradient checkpointing **enabled** (don't use `--disable-gradient-checkpointing`) - it's required for CUDA stability and prevents NVML errors.

## Learning Rate Considerations

With 3,120 examples (vs ~260 default):
- **Keep `5e-5`**: Still appropriate for fine-tuning (standard for Qwen2.5)
- **Monitor eval loss**: Stop early if it increases (sign of overfitting)
- **Consider 2-3 epochs**: Dataset is larger, may need fewer epochs

## Evaluation Strategy

```bash
--eval-strategy steps
```

With 2,808 training examples and effective batch size 16:
- **Steps per epoch**: ~175 steps
- **Total steps**: ~525 steps (3 epochs)
- **Optimal eval frequency**: Every 40-50 steps (~4 evals per epoch)
- Current default (50 steps) is good - gives ~11 evals per epoch

## Max Sequence Length Optimization

**Analysis of training data shows:**
- **14.1% of examples exceed 1024 tokens** and are being truncated
- **95th percentile**: ~1,453 tokens
- **Ideal**: Use `--max-seq-len 1536` to cover 95% of examples

### Memory vs Coverage Trade-off

| max_seq_len | batch_size | Coverage | Memory Impact | Status |
|-------------|------------|----------|---------------|--------|
| 1024 | 8 | 85.9% | Baseline | ✅ **Stable & Fast** |
| 1280 | 8 | 92.3% | +15-20% | ⚠️ May work, test it |
| 1536 | 8 | ~95% | +20-30% | ❌ Causes NVML errors |
| 1536 | 4 | ~95% | Similar to 1024/8 | ✅ **Stable with Better Coverage** |

**Recommendation**: 
- **For maximum coverage**: Use `--max-seq-len 1536` with `--batch-size 4` and `--grad-accum-steps 4` (Option 1a)
- **For faster training**: Use `--max-seq-len 1024` with `--batch-size 8` and `--grad-accum-steps 2` (Option 1b)

Both maintain the same effective batch size of 16, so training quality is similar - the choice is between coverage and speed.

### Why This Matters

Truncated examples lose important context, which can:
- Reduce model's ability to learn from complex attestations
- Cause incomplete Rego code generation
- Lead to poor performance on longer attestations

**However**, stability is more important than perfect coverage. 85.9% coverage with 1024 is still good, and the model can learn from the truncated examples (they're just shorter).

## Additional CUDA Optimizations

### Data Loading Performance

For CUDA, you can speed up data loading:

```bash
--dataloader-num-workers 2
```

This uses 2 worker processes for data loading (default is 0). Benefits:
- Faster data loading
- Better GPU utilization
- Recommended for CUDA (not needed for MPS)

### Warmup Steps

Current default: 50 steps (9.5% of total steps)
- This is optimal (10% is ideal)
- No change needed

### Weight Decay

Default: 0.01 (standard for fine-tuning)
- Good for preventing overfitting
- Can reduce to 0.0 if overfitting isn't an issue (Qwen2.5 docs suggest 0.0)
- Current default is fine

### Mixed Precision

The model is already loaded in bfloat16, but you can enable bf16 in TrainingArguments for additional optimizations (though it's already set via model dtype).

## Full Fine-Tuning vs LoRA

### LoRA (Recommended)
- Faster training
- Less memory usage
- Good for fine-tuning
- Use default LoRA settings: `--lora-r 16 --lora-alpha 32`

### Full Fine-Tuning
If you want to train all parameters:

```bash
python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/attestation_train.jsonl \
    --eval-path qwen2.5_model/attestation_eval.jsonl \
    --output-dir qwen2.5-attestation-parse \
    --no-lora \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --learning-rate 2e-5 \
    --num-epochs 2 \
    --eval-strategy steps
```

**Note:** Full fine-tuning uses more memory and typically needs:
- Smaller batch size (4 instead of 8)
- Lower learning rate (2e-5 instead of 5e-5)
- Fewer epochs (2 instead of 3)

## Monitoring Training

Watch for:
1. **GPU Memory Usage**: Use `nvidia-smi` to monitor
2. **Eval Loss**: Should decrease, not increase
3. **Training Speed**: Should be faster than MPS

## Example: Complete Training Command

For a typical CUDA GPU (8GB+ VRAM), choose based on your priority:

**Maximum Coverage (Recommended):**
```bash
python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/attestation_train.jsonl \
    --eval-path qwen2.5_model/attestation_eval.jsonl \
    --output-dir qwen2.5-attestation-parse \
    --batch-size 4 \
    --grad-accum-steps 4 \
    --learning-rate 5e-5 \
    --num-epochs 3 \
    --eval-strategy steps \
    --max-seq-len 1536 \
    --dataloader-num-workers 2
```

**Faster Training:**
```bash
python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/attestation_train.jsonl \
    --eval-path qwen2.5_model/attestation_eval.jsonl \
    --output-dir qwen2.5-attestation-parse \
    --batch-size 8 \
    --grad-accum-steps 2 \
    --learning-rate 5e-5 \
    --num-epochs 3 \
    --eval-strategy steps \
    --max-seq-len 1024 \
    --dataloader-num-workers 2
```

**Expected:**
- Faster training than MPS
- Effective batch size: 16 (both configurations)
- Coverage: 95.2% (1536) vs 85.9% (1024)
- ~2-3x faster than MPS defaults
- Monitor GPU memory with `nvidia-smi`

**Important:** Do NOT use `--disable-gradient-checkpointing` - gradient checkpointing is required for CUDA stability on many systems and prevents NVML initialization errors.

## Troubleshooting

### NVML Errors or Out of Memory (OOM) Errors

**Error**: `RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED`

This occurs when CUDA memory allocation fails. Solutions:

1. **If `max_seq_len=1536` with `batch_size=8` causes NVML errors**: 
   - Use `--max-seq-len 1024` with `--batch-size 8` (Option 1b - stable)
   - Or use `--max-seq-len 1536` with `--batch-size 4` (Option 1a - better coverage)

2. **General memory fixes**:
   - Reduce `--batch-size` (try 4 or 2)
   - Increase `--grad-accum-steps` to maintain effective batch size
   - Keep gradient checkpointing enabled (don't use `--disable-gradient-checkpointing`)
   - Set environment variables before training:
     ```bash
     export CUDA_VISIBLE_DEVICES=0
     export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
     ```

3. **Why this happens**:
   - Memory scales quadratically with sequence length (attention mechanism)
   - 1536 requires ~20-30% more memory than 1024
   - GPU memory fragmentation can cause allocation failures

### Training Too Slow
- Increase `--batch-size` (if memory allows)
- Reduce `--grad-accum-steps` (but maintain effective batch size)
- **Note:** Don't disable gradient checkpointing - it's required for CUDA stability

### Overfitting (Eval Loss Increases)
- Reduce `--num-epochs` (try 2 instead of 3)
- Reduce `--learning-rate` (try 3e-5)
- Stop training early when eval loss plateaus

