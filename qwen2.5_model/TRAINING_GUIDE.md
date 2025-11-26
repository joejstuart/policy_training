# Training Guide

## Quick Start

### 1. Setup (First Time Only)

```bash
cd qwen2.5_model
uv sync --python 3.12  # Python 3.13 not yet supported by PyTorch
cd ..
```

### 2. Generate Dataset (if not already done)

```bash
# From repository root
uv run --project qwen2.5_model python qwen2.5_model/generate_dataset.py

# Or from qwen2.5_model directory
cd qwen2.5_model
uv run python generate_dataset.py
```

This creates `train.jsonl` and `eval.jsonl` with 263 validated examples.

### 3. Train the Model

**Basic training (recommended for first run):**

```bash
# From repository root
uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/train.jsonl \
    --eval-path qwen2.5_model/eval.jsonl \
    --output-dir qwen2.5-rego-policy-lora

# Or from qwen2.5_model directory
cd qwen2.5_model
uv run python train_policy.py \
    --train-path train.jsonl \
    --eval-path eval.jsonl \
    --output-dir ../qwen2.5-rego-policy-lora
```

**Faster training (if you have enough GPU memory):**

```bash
uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/train.jsonl \
    --eval-path qwen2.5_model/eval.jsonl \
    --output-dir qwen2.5-rego-policy-lora \
    --disable-gradient-checkpointing \
    --batch-size 4 \
    --grad-accum-steps 2
```

## Training Settings Explained

### Conservative Settings (Default)

- **max-seq-len**: 1024 (covers most examples, avg ~300 tokens)
- **batch-size**: 2 (safe for MPS memory)
- **grad-accum-steps**: 4 (effective batch = 8)
- **learning-rate**: 5e-5 (conservative for small datasets)
- **num-epochs**: 3 (max for small datasets to avoid overfitting)
- **gradient-checkpointing**: Enabled (trades compute for memory)

### Performance Optimizations

**For faster training:**
- `--disable-gradient-checkpointing`: 2-3x faster, uses more memory
- `--batch-size 4`: Better GPU utilization (if memory allows)
- `--grad-accum-steps 2`: Reduce if batch-size increased

**For better quality:**
- `--max-seq-len 1536`: Covers more examples fully (slower)
- `--num-epochs 4`: More training (risk of overfitting)
- `--learning-rate 1e-4`: Higher learning rate (for larger datasets)

## Monitoring Training

Watch for signs of overfitting:
- ✅ Train loss decreasing
- ✅ Eval loss decreasing (or stable)
- ❌ Eval loss increasing while train loss decreases → **Stop early!**

**Early stopping:** If eval loss starts increasing, stop training (Ctrl+C) and use the last checkpoint before the increase.

## Expected Training Time

On Apple Silicon (M1/M2/M3):
- **With gradient checkpointing**: ~2-3 hours for 3 epochs
- **Without gradient checkpointing**: ~1-1.5 hours for 3 epochs

On CUDA GPU:
- Similar or faster depending on GPU

## Output

After training, you'll have:
- `qwen2.5-rego-policy-lora/`: Directory containing LoRA adapters
  - `adapter_config.json`: LoRA configuration
  - `adapter_model.safetensors`: Trained weights
  - `tokenizer.json`, `tokenizer_config.json`: Tokenizer files

## Using the Trained Model

The trained model can be loaded with:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "qwen2.5-rego-policy-lora")
tokenizer = AutoTokenizer.from_pretrained("qwen2.5-rego-policy-lora")
```

## Troubleshooting

### Out of Memory

- Reduce `--batch-size` to 1
- Enable gradient checkpointing (remove `--disable-gradient-checkpointing`)
- Reduce `--max-seq-len` to 512

### Model Loading Errors

- Check HuggingFace authentication: `huggingface-cli login`
- Verify model name: `Qwen/Qwen2.5-1.5B-Instruct`

### Training Too Slow

- Add `--disable-gradient-checkpointing` (if memory allows)
- Increase `--batch-size` (if memory allows)
- Reduce `--max-seq-len` (if most examples fit)

## Next Steps

After training:
1. Evaluate the model on the eval set
2. Test on new rules not in training data
3. Monitor for hallucination (inventing helpers)
4. Iterate with more training data if needed

