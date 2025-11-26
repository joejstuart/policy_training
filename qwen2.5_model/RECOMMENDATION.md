# Do You Need More Training Examples?

## Current Status: 268 Examples ✅

You currently have **268 high-quality, validated examples**:
- 164 implement examples
- 104 refactor examples  
- 100% validation pass rate (all pass opa parse, opa fmt, regal)

## Answer: It Depends, But Probably Yes

### For a 1.5B Model:

**268 examples is on the low side** for full fine-tuning, but **may be sufficient** if:
- You're using **LoRA/QLoRA** (parameter-efficient fine-tuning)
- You're doing **few-shot learning** (providing examples in context)
- You have **strong regularization** to prevent overfitting

**You should aim for 500+ examples** if:
- You want better generalization
- You're doing full fine-tuning
- You need the model to work without examples in context

## Quick Solution: Increase Rates

The easiest way to get to **400-500 examples**:

1. Edit `generate_dataset.py`:
   - Line ~600: Change `0.6` to `0.9` (refactor examples)
   - Line ~606: Change `0.3` to `0.7` (instruction variations)

2. Or run:
   ```bash
   ./qwen2.5_model/quick_augment.sh --apply
   uv run python qwen2.5_model/generate_dataset.py
   ```

This should give you **400-500 examples** without any code changes.

## Quality Over Quantity

**Important**: Your current 268 examples are all:
- ✅ From real, working code
- ✅ Validated with OPA tools
- ✅ Properly formatted
- ✅ Include correct context (package, imports, helpers)

This quality is more valuable than having 1000+ low-quality examples.

## Recommendation

1. **Try 268 first** - Test if it works for your use case
2. **If you need more** - Increase the rates (see above) to get 400-500
3. **Monitor performance** - If model overfits, you have too many similar examples
4. **Iterate** - Fine-tune, test, and adjust dataset size based on results

## Expected Results by Dataset Size

| Examples | Best For | Notes |
|----------|----------|-------|
| 200-300 | Few-shot, LoRA | Minimum viable, may overfit |
| 400-600 | **Recommended** | Good balance for 1.5B model |
| 600-1000 | Full fine-tuning | Better generalization |
| 1000+ | Overkill | Diminishing returns, risk of overfitting |

## Bottom Line

**268 is a good start, but aim for 400-500 for best results.** The quick augmentation (increasing rates) will get you there easily.

