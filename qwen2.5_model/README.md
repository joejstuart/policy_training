# Qwen2.5-1.5B Policy Training Dataset

This directory contains tools to generate a training dataset for fine-tuning the qwen2.5-1.5B model on Rego policy rules.

## Overview

The dataset focuses on:
- **Scope**: Rules from `policy/release/**` with helpers from `policy/lib/**` and `policy/release/lib/**`
- **Task Types**: 
  - "Implement rule from instruction" - Generate Rego code from natural language description
  - "Refactor rule to correct style" - Fix style issues in existing Rego code
- **Quality Control**: All examples are validated with:
  - `opa parse` (syntax validation)
  - `opa fmt` (formatting validation)
  - `regal` (linting)
  - `opa test` (if tests exist)

## Requirements

- Python 3.8+
- OPA (Open Policy Agent) - must be in PATH
- Regal (optional, for linting) - must be in PATH

Install Python dependencies:
```bash
# Using uv (recommended)
# From qwen2.5_model directory:
cd qwen2.5_model
uv sync --python 3.12  # Python 3.13 not yet supported by PyTorch

# Or from repository root:
uv sync --project qwen2.5_model --python 3.12

# Or using pip (requires Python 3.10-3.12)
pip install -r requirements.txt
```

**Note**: Python 3.13 is not yet fully supported by PyTorch. Use Python 3.10-3.12.

## Usage

### 1. Generate the Dataset

Generate the dataset:
```bash
# From repository root (recommended)
cd qwen2.5_model
uv sync --python 3.12  # First time setup
cd ..
uv run --project qwen2.5_model python qwen2.5_model/generate_dataset.py

# Or from qwen2.5_model directory
cd qwen2.5_model
uv sync --python 3.12
uv run python generate_dataset.py
```

This will:
1. Parse all `.rego` files in `policy/release/**`
2. Extract rules with METADATA blocks
3. Generate training examples (implement + refactor)
4. Validate each example
5. Split into train/eval sets (90/10)
6. Write `train.jsonl` and `eval.jsonl`

### 2. Train the Model

Train the model using LoRA fine-tuning:

```bash
# From repository root (after running: cd qwen2.5_model && uv sync --python 3.12)
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

**Training Options:**

- `--model-name`: HuggingFace model name (default: `Qwen/Qwen2.5-1.5B-Instruct`)
- `--max-seq-len`: Maximum sequence length (default: `1024`, covers most examples)
- `--batch-size`: Batch size per device (default: `2` for MPS memory safety)
- `--grad-accum-steps`: Gradient accumulation steps (default: `4`, effective batch = 8)
- `--learning-rate`: Learning rate (default: `5e-5` for small datasets)
- `--num-epochs`: Number of epochs (default: `3` max for small datasets)
- `--disable-gradient-checkpointing`: Disable for faster training (uses more memory)
- `--eval-strategy`: Evaluation strategy - `no`, `steps`, or `epoch` (default: `steps`)

**Example with faster settings (if you have enough memory):**

```bash
# From repository root
uv run --project qwen2.5_model python qwen2.5_model/train_policy.py \
    --train-path qwen2.5_model/train.jsonl \
    --eval-path qwen2.5_model/eval.jsonl \
    --output-dir qwen2.5-rego-policy-lora \
    --disable-gradient-checkpointing \
    --batch-size 4 \
    --grad-accum-steps 2
```

**Prerequisites:**

```bash
# Install dependencies (using uv)
# From qwen2.5_model directory:
cd qwen2.5_model
uv sync --python 3.12  # Python 3.13 not yet supported by PyTorch

# Or from repository root:
uv sync --project qwen2.5_model --python 3.12

# Or using pip (requires Python 3.10-3.12)
pip install transformers peft torch accelerate

# Authenticate with HuggingFace (if model is gated)
huggingface-cli login
```

**Note**: Python 3.13 is not yet fully supported by PyTorch. The project requires Python 3.10-3.12. If you have Python 3.13, uv will automatically use Python 3.12 (or 3.11/3.10 if available).

## Output Format

Each line in the JSONL files is a JSON object:

```json
{
  "instruction": "Rule title and description",
  "context": "package declaration + imports + helper cheat sheet",
  "input_code": "Only for refactor tasks - the code to refactor",
  "output_code": "The correct Rego code",
  "task_type": "implement" or "refactor"
}
```

## Dataset Statistics

After generation, check `dataset_summary.json` for:
- Total examples
- Train/eval split
- Task type distribution
- Invalid examples count

## Training Features

- **LoRA Fine-tuning**: Parameter-efficient training (only ~1% of parameters trainable)
- **Pre-tokenization**: All examples tokenized once for faster training
- **Dynamic Padding**: Batches padded to longest example in batch (not max_seq_len)
- **Conservative Settings**: Optimized for small datasets (~260 examples) to avoid overfitting
- **Task Types**: Supports both "implement" and "refactor" tasks

## Running Inference

The inference script now includes **dynamic context building** that automatically finds relevant helpers from your library files. This is enabled by default and works with both the fine-tuned LoRA model and base models.

### Interactive Chatbot Mode (Recommended)

Start an interactive chat session with the fine-tuned model:

```bash
# From repository root (recommended - uses fine-tuned LoRA model)
uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \
    --model-dir qwen2.5-rego-policy-lora \
    --package tasks

# Or from qwen2.5_model directory
cd qwen2.5_model
uv run python infer_policy.py --model-dir ../qwen2.5-rego-policy-lora --package tasks
```

In chat mode, you can:
- Ask questions about Rego/OPA policies
- Request rule implementations
- Ask to refactor existing code
- Get explanations of how rules work
- Specify package in your message: `"tasks: write a rule that checks if all tasks succeeded"`

Type `quit` or `exit` to end the conversation, or `clear` to start a new conversation.

### Single Inference

Generate a response from a single instruction:

```bash
# With fine-tuned model (recommended)
uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \
    --model-dir qwen2.5-rego-policy-lora \
    --package tasks \
    --instruction "Write a rule that checks if all tasks in a PipelineRun succeeded"

# Different package example
uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \
    --model-dir qwen2.5-rego-policy-lora \
    --package sbom_spdx \
    --instruction "Write a rule that validates SPDX SBOM format"
```

### Dynamic Context Building

The inference script automatically:
1. **Indexes all library functions** from `policy/lib/**` and `policy/release/lib/**`
2. **Finds relevant helpers** based on your instruction keywords
3. **Builds minimal context** (signatures + usage examples, not full implementations)
4. **Stays within token budget** (~500 tokens for context)

**Key Features:**
- Shows helper signatures and usage examples (not full code)
- Filters out irrelevant helpers automatically
- Respects strict token budgets for small models
- Works in both interactive and single-inference modes

**Package Specification:**
- Use `--package tasks` to specify the target package
- Or in interactive mode, use `"tasks: your instruction"` format
- If not specified, package is inferred from instruction keywords

**Disable Dynamic Context:**
```bash
# Use base model without library context
uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \
    --base-model Qwen/Qwen2.5-1.5B-Instruct \
    --no-lora \
    --no-context \
    --instruction "Write a simple Rego rule"
```

### Legacy: Manual Context Building

If you want to manually build context (not recommended with dynamic context enabled):

```bash
# From repository root
uv run --project qwen2.5_model python qwen2.5_model/inference_helper.py <package> <import1> [import2 ...]

# Example
uv run --project qwen2.5_model python qwen2.5_model/inference_helper.py base_image_registries data.lib data.lib.image data.lib.sbom
```

