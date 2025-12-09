# Policy Training

Fine-tuning LLMs (Qwen) to generate Rego/OPA policy code for attestation parsing and enterprise contract validation.

## Project Structure

```
policy-training/
├── src/                    # Core Python modules
│   ├── paths.py           # Central path definitions
│   ├── train_policy.py    # Training script
│   ├── infer_policy.py    # Inference/chat script
│   ├── library_mapper.py  # Maps Rego imports to files
│   ├── library_indexer.py # Indexes helper functions
│   └── ...
│
├── scripts/               # Data generation utilities
│   ├── generate_dataset.py
│   ├── generate_attestation_dataset.py
│   └── ...
│
├── data/
│   ├── attestations/      # Source attestation JSON files
│   └── training/          # Training datasets (JSONL)
│       ├── attestation/   # Attestation parsing examples
│       ├── policy_rules/  # Policy rule examples
│       ├── compiler_errors/ # Error correction examples
│       └── combined/      # Merged datasets
│
├── policy/                # Reference Rego policies
│   ├── lib/              # Helper library
│   └── release/          # Release policies
│
├── docs/                  # Documentation
├── tests/                 # Test suite
└── notebooks/             # Jupyter notebooks
```

## Quick Start

### Installation

```bash
# Using uv (recommended)
uv sync --python 3.12

# Or using pip
pip install -r requirements.txt
```

### Generate Training Data

```bash
# Generate attestation parsing dataset
python scripts/generate_attestation_dataset.py

# Generate policy rules dataset
python scripts/generate_dataset.py
```

### Train a Model

```bash
python src/train_policy.py \
    --train-path data/training/attestation/train.jsonl \
    --eval-path data/training/attestation/eval.jsonl \
    --output-dir models/qwen2.5-attestation-parse
```

### Run Inference

```bash
# Interactive chat mode
python src/infer_policy.py \
    --model-dir models/qwen2.5-attestation-parse

# Single instruction
python src/infer_policy.py \
    --model-dir models/qwen2.5-attestation-parse \
    --instruction "Write a rule that checks if all tasks succeeded"
```

## Training Data Types

1. **Attestation Parsing** (`data/training/attestation/`)
   - Parse SLSA attestation JSON structures
   - Generate Rego code to extract/validate attestation data

2. **Policy Rules** (`data/training/policy_rules/`)
   - Generate Rego policy rules from natural language
   - Based on the `policy/release/` reference implementation

3. **Compiler Errors** (`data/training/compiler_errors/`)
   - Fix Rego syntax and semantic errors
   - Learn from error-correction pairs

## Documentation

- [Training Output Guidelines](docs/training.md) - How to structure training data
- [Rego Style Guide](docs/rego_style_guide.md) - Best practices for Rego code

## Requirements

- Python 3.10-3.12
- OPA (Open Policy Agent) CLI
- Optional: Regal (Rego linter)
- GPU recommended for training

