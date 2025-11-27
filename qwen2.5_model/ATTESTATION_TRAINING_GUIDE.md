# Attestation Parsing Training & Inference Guide

## Quick Start

### 1. Generate Dataset (if not already done)

```bash
cd /Users/jstuart/Documents/repos/policy-training
python3 qwen2.5_model/generate_attestation_dataset.py
```

This creates:
- `qwen2.5_model/attestation_train.jsonl` (972 examples)
- `qwen2.5_model/attestation_eval.jsonl` (108 examples)

### 2. Train with LoRA (Recommended)

**Basic training:**
```bash
uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/attestation_train.jsonl \
    --eval-path qwen2.5_model/attestation_eval.jsonl \
    --output-dir qwen2.5-attestation-parse-lora
```

**Faster training (if you have enough GPU memory):**
```bash
uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/attestation_train.jsonl \
    --eval-path qwen2.5_model/attestation_eval.jsonl \
    --output-dir qwen2.5-attestation-parse-lora \
    --disable-gradient-checkpointing \
    --batch-size 4 \
    --grad-accum-steps 2
```

**Full fine-tuning (without LoRA):**
```bash
uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/attestation_train.jsonl \
    --eval-path qwen2.5_model/attestation_eval.jsonl \
    --output-dir qwen2.5-attestation-parse-full \
    --no-lora
```

### 3. Run Inference

**With LoRA adapters:**
```bash
uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \
    --model-dir qwen2.5-attestation-parse-lora \
    --instruction "In an attestation, check all tasks for a task named 'init'"
```

**Interactive chat mode:**
```bash
uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \
    --model-dir qwen2.5-attestation-parse-lora
```

Then you can type instructions like:
- "Find a task named 'buildah' in the attestation"
- "Get the status of task 'init'"
- "List all task names"

**With full fine-tuned model:**
```bash
uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \
    --model-dir qwen2.5-attestation-parse-full \
    --instruction "In an attestation, check all tasks for a task named 'init'"
```

## Training Options

### LoRA vs Full Fine-tuning

**LoRA (default, recommended):**
- ✅ Faster training
- ✅ Less memory usage
- ✅ Smaller model files (just adapters)
- ✅ Can merge adapters later if needed
- Output: LoRA adapters in `adapter_model.safetensors` + `adapter_config.json`

**Full Fine-tuning (`--no-lora`):**
- ✅ All parameters trainable
- ✅ Potentially better quality (but needs more data)
- ❌ Slower training
- ❌ More memory usage
- ❌ Larger model files
- Output: Full model in `pytorch_model.bin` + `config.json`

### Training Parameters

- `--batch-size`: Batch size per device (default: 2, increase if you have memory)
- `--grad-accum-steps`: Gradient accumulation (default: 4, effective batch = batch_size × grad_accum_steps)
- `--learning-rate`: Learning rate (default: 5e-5)
- `--num-epochs`: Number of epochs (default: 3)
- `--max-seq-len`: Maximum sequence length (default: 1024)
- `--disable-gradient-checkpointing`: Faster but uses more memory

### Recommended Settings

**For 1080 examples (attestation dataset):**
- Use LoRA (default)
- `--batch-size 2` (or 4 if you have memory)
- `--grad-accum-steps 4` (effective batch = 8)
- `--learning-rate 5e-5`
- `--num-epochs 3` (watch for overfitting)

## Inference Options

The inference script auto-detects:
- **LoRA adapters**: If `adapter_config.json` exists
- **Full model**: If `config.json` exists (but no `adapter_config.json`)

You can also:
- `--no-lora`: Skip LoRA loading even if adapters exist
- `--base-model`: Specify different base model
- `--device`: Choose device (auto/mps/cuda/cpu)

## Example Workflow

```bash
# 1. Generate dataset
python3 qwen2.5_model/generate_attestation_dataset.py

# 2. Train with LoRA
uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/attestation_train.jsonl \
    --eval-path qwen2.5_model/attestation_eval.jsonl \
    --output-dir qwen2.5-attestation-parse-lora

# 3. Test inference
uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \
    --model-dir qwen2.5-attestation-parse-lora \
    --instruction "Find a task named 'init' in the attestation"
```

## Output Files

**After training with LoRA:**
```
qwen2.5-attestation-parse-lora/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer_config.json
├── tokenizer.json
└── ...
```

**After training with full fine-tuning:**
```
qwen2.5-attestation-parse-full/
├── config.json
├── pytorch_model.bin (or .safetensors)
├── tokenizer_config.json
└── ...
```

## Tips

1. **Monitor eval loss**: Stop early if it starts increasing (sign of overfitting)
2. **Start with LoRA**: It's faster and uses less memory
3. **Use gradient checkpointing**: Enabled by default, saves memory
4. **Watch memory**: If you get OOM errors, reduce batch-size or enable gradient checkpointing
5. **Test incrementally**: Try inference after a few epochs to see if model is learning

