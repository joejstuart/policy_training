#!/usr/bin/env bash
set -e

# =============================================================================
# Full Two-Stage Training Data Generation Pipeline
# =============================================================================
# This script generates complete training data for the two-stage Rego model:
#   Stage 1: Context inference (requirements → schema + helpers)
#   Stage 2: Rule generation (context + requirements → Rego code)
#
# Steps:
#   1. Generate base dataset from existing rules (with LLM augmentation)
#   2. Generate synthetic rules for code diversity
#   3. Combine into final training dataset
#   4. Report statistics
# =============================================================================

echo "=============================================="
echo "Full Two-Stage Training Data Generation"
echo "=============================================="
echo ""

# Ensure output directories exist
mkdir -p data/training/two_stage
mkdir -p data/training/synthetic
mkdir -p data/training/combined
mkdir -p data/training/combined_augmented

# -----------------------------------------------------------------------------
# Step 1: Generate base dataset from existing rules
# -----------------------------------------------------------------------------
echo "Step 1: Generating two-stage dataset from existing rules..."
echo "  (This includes LLM-based instruction variations)"
echo ""

# Check if generate_two_stage_dataset.py exists
if [[ -f "scripts/generate_two_stage_dataset.py" ]]; then
    python3 scripts/generate_two_stage_dataset.py --augment --variations 4
else
    echo "  Warning: generate_two_stage_dataset.py not found, skipping..."
    echo "  Using existing data in data/training/two_stage/"
fi

echo ""
echo "  Base dataset generated:"
if [[ -f "data/training/two_stage/stage1_train.jsonl" ]]; then
    echo "    Stage 1 train: $(wc -l < data/training/two_stage/stage1_train.jsonl) examples"
    echo "    Stage 1 eval:  $(wc -l < data/training/two_stage/stage1_eval.jsonl) examples"
    echo "    Stage 2 train: $(wc -l < data/training/two_stage/stage2_train.jsonl) examples"
    echo "    Stage 2 eval:  $(wc -l < data/training/two_stage/stage2_eval.jsonl) examples"
fi
echo ""

# -----------------------------------------------------------------------------
# Step 2: Generate synthetic rules for code diversity
# -----------------------------------------------------------------------------
echo "Step 2: Generating synthetic rules for code diversity..."
echo "  (Composing unique rules from iteration patterns + conditions)"
echo ""

# Generate synthetic rules with validation
python3 scripts/generate_synthetic_rules.py \
    --count 200 \
    --validate \
    --seed 42 \
    --output-dir data/training/synthetic

echo ""
echo "  Synthetic rules generated:"
if [[ -f "data/training/synthetic/stage2_synthetic_train.jsonl" ]]; then
    echo "    Stage 2 synthetic train: $(wc -l < data/training/synthetic/stage2_synthetic_train.jsonl) examples"
    echo "    Stage 2 synthetic eval:  $(wc -l < data/training/synthetic/stage2_synthetic_eval.jsonl) examples"
fi
echo ""

# -----------------------------------------------------------------------------
# Step 3: Combine into final training dataset
# -----------------------------------------------------------------------------
echo "Step 3: Combining datasets..."
echo ""

# Stage 1: Combine base + synthetic (synthetic adds schema patterns for new iteration sources)
echo "  Stage 1: Merging two_stage + synthetic"
cat data/training/two_stage/stage1_train.jsonl \
    data/training/synthetic/stage1_synthetic_train.jsonl \
    > data/training/combined/stage1_train.jsonl

cat data/training/two_stage/stage1_eval.jsonl \
    data/training/synthetic/stage1_synthetic_eval.jsonl \
    > data/training/combined/stage1_eval.jsonl

# Stage 2: Combine base + synthetic for training
echo "  Stage 2: Merging two_stage + synthetic"
cat data/training/two_stage/stage2_train.jsonl \
    data/training/synthetic/stage2_synthetic_train.jsonl \
    > data/training/combined/stage2_train.jsonl

# Stage 2 eval: Include some synthetic for eval diversity
cat data/training/two_stage/stage2_eval.jsonl \
    data/training/synthetic/stage2_synthetic_eval.jsonl \
    > data/training/combined/stage2_eval.jsonl

# Also create augmented directory (same as combined for now)
cp data/training/combined/stage1_train.jsonl data/training/combined_augmented/stage1_train.jsonl
cp data/training/combined/stage1_eval.jsonl data/training/combined_augmented/stage1_eval.jsonl
cp data/training/combined/stage2_train.jsonl data/training/combined_augmented/stage2_train.jsonl
cp data/training/combined/stage2_eval.jsonl data/training/combined_augmented/stage2_eval.jsonl

echo ""

# -----------------------------------------------------------------------------
# Step 4: Calculate statistics
# -----------------------------------------------------------------------------
echo "Step 4: Calculating dataset statistics..."
echo ""

# Count examples
STAGE1_TRAIN=$(wc -l < data/training/combined/stage1_train.jsonl)
STAGE1_EVAL=$(wc -l < data/training/combined/stage1_eval.jsonl)
STAGE2_TRAIN=$(wc -l < data/training/combined/stage2_train.jsonl)
STAGE2_EVAL=$(wc -l < data/training/combined/stage2_eval.jsonl)

# Calculate uniqueness for Stage 2
UNIQUE_RULES=$(cat data/training/combined/stage2_train.jsonl | python3 -c "
import sys, json, re
rules = []
for line in sys.stdin:
    if not line.strip():
        continue
    d = json.loads(line)
    output = d.get('output', '')
    rule_start = output.find('RULE:')
    tests_start = output.find('TESTS:')
    if rule_start >= 0:
        if tests_start >= 0:
            rule = output[rule_start:tests_start]
        else:
            rule = output[rule_start:]
        normalized = re.sub(r'\s+', ' ', rule.strip())
        rules.append(normalized)
print(len(set(rules)))
")

UNIQUENESS_RATE=$(python3 -c "print(f'{100*${UNIQUE_RULES}/${STAGE2_TRAIN}:.1f}')")

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Generation Complete!"
echo "=============================================="
echo ""
echo "=== Final Dataset Statistics ==="
echo ""
echo "Stage 1 (Context Inference):"
echo "  Train: ${STAGE1_TRAIN} examples"
echo "  Eval:  ${STAGE1_EVAL} examples"
echo ""
echo "Stage 2 (Rule Generation):"
echo "  Train: ${STAGE2_TRAIN} examples"
echo "  Eval:  ${STAGE2_EVAL} examples"
echo "  Unique rules: ${UNIQUE_RULES} (${UNIQUENESS_RATE}% uniqueness)"
echo ""
echo "=== Output Directories ==="
echo ""
echo "  data/training/combined/         <- Use for training"
echo "  data/training/combined_augmented/ <- Same (alias)"
echo "  data/training/two_stage/        <- Base dataset only"
echo "  data/training/synthetic/        <- Synthetic rules only"
echo ""
echo "=== Next Steps ==="
echo ""
echo "Train Stage 1 model:"
echo "  python3 src/train_policy.py \\"
echo "      --train-path data/training/combined/stage1_train.jsonl \\"
echo "      --eval-path data/training/combined/stage1_eval.jsonl \\"
echo "      --output-dir models/stage1-context-inference \\"
echo "      --num-epochs 4 --learning-rate 2e-4 --max-seq-len 1024 --use-4bit"
echo ""
echo "Train Stage 2 model:"
echo "  python3 src/train_policy.py \\"
echo "      --train-path data/training/combined/stage2_train.jsonl \\"
echo "      --eval-path data/training/combined/stage2_eval.jsonl \\"
echo "      --output-dir models/stage2-rule-generation \\"
echo "      --num-epochs 4 --learning-rate 2e-4 --max-seq-len 3072 --use-4bit"
echo ""
