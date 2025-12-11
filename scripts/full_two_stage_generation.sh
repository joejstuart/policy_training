#!/usr/bin/env bash
set -e

# Ensure output directory exists
mkdir -p data/training/combined

# Step 1: Generate with LLM variations
echo "Step 1: Generating two-stage dataset with LLM variations..."
python3 scripts/generate_two_stage_dataset.py --augment --variations 4
# (slow but high quality)

# Step 2: Add synthetic rules for more coverage
echo "Step 2: Generating synthetic rules..."
python3 scripts/generate_synthetic_rules.py

# Step 3: Combine train files
echo "Step 3: Combining train files..."
cat data/training/two_stage/stage1_train.jsonl \
    data/training/synthetic/stage1_synthetic.jsonl \
    > data/training/combined/stage1_train.jsonl

cat data/training/two_stage/stage2_train.jsonl \
    data/training/synthetic/stage2_synthetic.jsonl \
    > data/training/combined/stage2_train.jsonl

# Step 4: Copy eval files (eval comes from real rules only, not synthetic)
echo "Step 4: Copying eval files..."
cp data/training/two_stage/stage1_eval.jsonl data/training/combined/stage1_eval.jsonl
cp data/training/two_stage/stage2_eval.jsonl data/training/combined/stage2_eval.jsonl

# Summary
echo ""
echo "=== Generation Complete ==="
echo "Train files:"
wc -l data/training/combined/stage1_train.jsonl data/training/combined/stage2_train.jsonl
echo ""
echo "Eval files:"
wc -l data/training/combined/stage1_eval.jsonl data/training/combined/stage2_eval.jsonl
echo ""
echo "Output directory: data/training/combined/"
