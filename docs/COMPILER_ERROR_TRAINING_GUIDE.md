# Training Guide: Compiler Error Correction

This guide explains how to train the model to fix Rego compiler errors using the generated mutation dataset.

## Step 1: Convert Dataset to Training Format

The mutation dataset needs to be converted to the format expected by `train_policy.py`:

```bash
python qwen2.5_model/format_compiler_errors_for_training.py \
    --input data/compiler_errors/mutated_errors.jsonl \
    --output data/compiler_errors/compiler_error_train.jsonl \
    --split-train-val \
    --train-ratio 0.9
```

This will:
- Convert compiler error examples to training format
- Include error messages in the instruction
- Split into train (90%) and validation (10%) sets
- Create `compiler_error_train.jsonl` and `compiler_error_val.jsonl`

## Step 2: Train the Model

### Option A: Train Only on Compiler Errors (Fine-tune existing model)

```bash
python qwen2.5_model/train_policy.py \
    --train-path data/compiler_errors/compiler_error_train.jsonl \
    --eval-path data/compiler_errors/compiler_error_val.jsonl \
    --output-dir qwen2.5-rego-policy-compiler-errors-lora \
    --num-epochs 3 \
    --learning-rate 5e-5 \
    --batch-size 2 \
    --grad-accum-steps 4
```

### Option B: Combine with Existing Training Data

First, combine the datasets:

```bash
# Combine compiler errors with existing training data
cat qwen2.5_model/attestation_train.jsonl \
    data/compiler_errors/compiler_error_train.jsonl > \
    data/compiler_errors/combined_train.jsonl

cat qwen2.5_model/attestation_eval.jsonl \
    data/compiler_errors/compiler_error_val.jsonl > \
    data/compiler_errors/combined_eval.jsonl
```

Then train:

```bash
python qwen2.5_model/train_policy.py \
    --train-path data/compiler_errors/combined_train.jsonl \
    --eval-path data/compiler_errors/combined_eval.jsonl \
    --output-dir qwen2.5-rego-policy-combined-lora \
    --num-epochs 3 \
    --learning-rate 5e-5 \
    --batch-size 2 \
    --grad-accum-steps 4
```

## Step 3: Test the Model

After training, test the model's ability to fix compiler errors:

```bash
python qwen2.5_model/infer_policy.py \
    --model-dir qwen2.5-rego-policy-compiler-errors-lora \
    --instruction "Fix this Rego code: package attestation_check\nimport rego.v1\n\ndeny contains result if {\n    some result in input.attestations\n    result := {\"msg\": \"error\"}\n}"
```

## Training Configuration Recommendations

### For Compiler Error Dataset Only (~500 examples)

```python
num_epochs = 5          # More epochs for smaller dataset
learning_rate = 3e-5    # Slightly lower learning rate
batch_size = 2
grad_accum_steps = 8    # Effective batch size = 16
warmup_steps = 20
```

### For Combined Dataset (~500 compiler errors + existing data)

```python
num_epochs = 3          # Standard epochs
learning_rate = 5e-5    # Standard learning rate
batch_size = 2
grad_accum_steps = 4    # Effective batch size = 8
warmup_steps = 50
```

## Expected Results

After training, the model should:
- ✅ Recognize common compiler error patterns
- ✅ Generate fixes for variable redeclaration errors
- ✅ Fix unsafe variable errors
- ✅ Correct type errors
- ✅ Fix syntax errors

## Monitoring Training

Watch for:
- **Loss decreasing**: Should drop steadily
- **Validation loss**: Should track training loss (not diverging)
- **Error fix rate**: Test on validation set to see % of errors fixed

## Troubleshooting

### Model doesn't fix errors
- Try more epochs (5-7)
- Lower learning rate (1e-5 to 3e-5)
- Increase dataset size (generate more mutations)

### Overfitting
- Reduce epochs
- Increase dropout
- Add more training data

### Out of memory
- Reduce batch size
- Increase gradient accumulation steps
- Use smaller LoRA rank (r=8 instead of 16)

## Next Steps

1. **Evaluate on test set**: Create a test set of compiler errors
2. **Iterate**: Generate more mutations for underrepresented error types
3. **Combine approaches**: Use both mutation-based and real error collection

