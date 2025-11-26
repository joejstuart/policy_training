# Fixes Applied to Training Dataset Generator

All 7 issues have been addressed in `generate_dataset.py`:

## ✅ Issue 1: Context is HUGE and noisy

**Fixed**: Context now only includes:
- Package name
- Imports actually used in the output code (extracted via `extract_used_imports()`)
- Only relevant helper descriptions (8 max, only those used in code)

**Before**: 400+ tokens of imports, test functions, unrelated helpers
**After**: ~50-100 tokens of focused, relevant context

## ✅ Issue 2: No helper descriptions

**Fixed**: Added `HELPER_DESCRIPTIONS` dictionary with descriptions of common helpers:
- `lib.result_helper(chain, params): Creates a result object...`
- `lib.tekton.tasks(obj): Returns set of tasks...`
- etc.

Context now includes only helpers actually used in the code, with descriptions.

## ✅ Issue 3: No clear distinction between task types

**Fixed**: 
- All examples explicitly labeled as `"implement"` or `"refactor"`
- Refactor examples have clear instruction: "Refactor the following rule to match correct Rego style"
- Implement examples have clear instruction: "Write a {package} rule: {title}"

## ✅ Issue 4: Ambiguous instructions

**Fixed**: `make_instruction_specific()` function now:
- Adds helper references: "Required helpers: - lib.result_helper..."
- Adds rule type guidance: "The rule should use 'deny contains result if'..."
- Makes instructions concrete and actionable

**Before**: "Ensure that all tasks in the provenance have succeeded."
**After**: "Write a tasks rule: Task success check. Verify all TaskRuns have Succeeded status. Required helpers: - lib.tekton.status(task)"

## ✅ Issue 5: output_code sometimes contains artifacts

**Fixed**: Enhanced validation:
- All code validated with `opa parse`, `opa fmt`, `regal`
- Additional check for suspicious patterns (e.g., all result_helper calls with empty arrays)
- Formatted code used as output (ensures consistency)

## ✅ Issue 6: No negative or contrastive examples

**Fixed**: Added negative example generation:
- ~20 examples with explicit instruction: "IMPORTANT: Only use existing helpers from the context. If a helper does not exist, write a TODO comment instead of creating a new helper function."
- Teaches model not to invent APIs

## ✅ Issue 7: No evaluation set yet

**Fixed**: Improved eval set splitting:
- Separates by task type (implement vs refactor)
- Ensures eval set has both types
- Better distribution across packages

## Usage

Run the fixed generator:

```bash
uv run python qwen2.5_model/generate_dataset.py
```

The output will have:
- Smaller, focused contexts (~50-100 tokens vs 400+)
- Helper descriptions for grounding
- Clear task type labels
- Specific, unambiguous instructions
- Clean, validated output code
- Negative examples for anti-hallucination
- Proper train/eval split

## Expected Improvements

1. **Reduced hallucination**: Model sees only relevant helpers, can't mix-and-match from bloated context
2. **Better grounding**: Helper descriptions tell model what functions do
3. **Clearer learning**: Task types are explicit, not implicit
4. **More specific**: Instructions include helper references and guidance
5. **Cleaner output**: All code validated and formatted
6. **Anti-hallucination**: Negative examples teach model not to invent APIs
7. **Better evaluation**: Proper eval set for honest metrics

