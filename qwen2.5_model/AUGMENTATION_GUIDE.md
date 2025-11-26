# Dataset Augmentation Guide

## Current Status

After improvements, we now have **268 examples** (up from 202). This is still below the target of 500-3000 examples, but it's a solid foundation.

## Is 268 Examples Enough?

For a **1.5B parameter model**, the answer depends on your goals:

### ✅ **268 examples MAY be sufficient if:**
- You're doing **few-shot fine-tuning** (using the model as a code assistant with examples in context)
- You have **high-quality, diverse examples** (which we do - all validated)
- You're fine-tuning with **strong regularization** to prevent overfitting
- You're using **transfer learning** from a model already trained on code

### ❌ **You likely need more if:**
- You want the model to **generalize to new rule patterns** not in the training set
- You're doing **full fine-tuning** (not LoRA/QLoRA)
- You need the model to work **without examples in context**

## Recommended: Get to 500+ Examples

For better results, aim for **at least 500 examples**. Here are strategies to get there:

## Augmentation Strategies

### 1. ✅ Already Implemented
- **Increased refactor rate**: 30% → 60% (doubles refactor examples)
- **Instruction variations**: 30% chance for paraphrased instructions
- **Larger rule support**: Can now refactor rules up to 800 chars (was 500)

### 2. Additional Strategies You Can Try

#### A. Extract Helper Rules (Not Yet Implemented)
Many files have helper rules (starting with `_`) that don't have METADATA but could be training examples:

```python
# In generate_dataset.py, add after line 603:
# Extract helper rules without metadata
for parsed_file in parsed_files:
    content = parsed_file.full_content
    # Find helper rules with substantial logic
    # Create synthetic examples for them
```

#### B. Create Rule Combinations
Combine patterns from multiple rules to create synthetic examples:

```python
# Take rule structure from one rule, apply to different domain
# Example: "Check X is in allowed list" pattern applied to different data types
```

#### C. Split Complex Rules
For very large rules (>30 lines), create multiple examples:
- First half as "implement rule fragment"
- Second half as "complete the rule"

#### D. Create Negative Examples
Add examples of **incorrect** code that the model should avoid:
- Common mistakes (wrong operator, missing checks)
- Style violations (formatting, naming)

#### E. Increase Variation Rate
In `generate_dataset.py`, line 600:
```python
if random.random() < 0.8:  # Increase from 0.6 to 0.8
```

And line 607:
```python
if random.random() < 0.5:  # Increase from 0.3 to 0.5
```

## Quick Win: Increase Rates

The easiest way to get more examples is to increase the random rates in `generate_dataset.py`:

```python
# Line ~600: Refactor examples
if random.random() < 0.8:  # Was 0.6, try 0.8 or 0.9

# Line ~607: Instruction variations  
if random.random() < 0.5:  # Was 0.3, try 0.5 or 0.7
```

This should get you to **400-500 examples** easily.

## Quality vs Quantity Trade-off

**Important**: More examples is good, but quality matters more for small models.

- ✅ **268 high-quality, validated examples** > 1000 low-quality examples
- ✅ All examples pass `opa parse`, `opa fmt`, `regal`
- ✅ Examples are from real, working code

## Recommendation

1. **Try the current 268 examples first** - they might be sufficient for your use case
2. **If you need more**, increase the rates in `generate_dataset.py` (see above)
3. **Monitor performance** - if the model overfits, you have too many similar examples
4. **If still not enough**, implement helper rule extraction (strategy 2A above)

## Testing Your Dataset

After fine-tuning, test on:
- Rules similar to training set (should work well)
- Rules with different patterns (tests generalization)
- Rules requiring different helpers (tests context understanding)

If the model fails on similar rules → need more examples
If the model fails on different patterns → need more diversity
If the model invents APIs → need better context/grounding

