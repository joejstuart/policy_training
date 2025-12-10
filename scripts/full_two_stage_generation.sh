#!/usr/bin/env bash

# Step 1: Generate with LLM variations
python3 scripts/generate_two_stage_dataset.py --augment --variations 4
# (slow but high quality)

# Step 2: Add synthetic rules for more coverage
python3 scripts/generate_synthetic_rules.py

# Step 3: Combine
cat data/training/two_stage/stage1_train.jsonl \
    data/training/synthetic/stage1_synthetic.jsonl \
    > data/training/combined/stage1_train.jsonl

cat data/training/two_stage/stage2_train.jsonl \
    data/training/synthetic/stage2_synthetic.jsonl \
    > data/training/combined/stage2_train.jsonl



# Result: ~700+ examples with diverse patterns AND natural phrasing
