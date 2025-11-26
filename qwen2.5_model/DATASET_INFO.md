# Dataset Information

## Current Dataset Statistics

- **Total Examples**: 202
- **Train Examples**: 181 (90%)
- **Eval Examples**: 21 (10%)
- **Task Types**:
  - Implement: 161 examples
  - Refactor: 41 examples
- **Validation**: 100% pass rate (all examples validated with opa parse, opa fmt, regal)

## Dataset Size

The current dataset has 202 examples, which is below the target range of 500-3000 examples. To increase the dataset size, you can:

1. **Increase refactor examples**: Currently only 30% of rules get refactor examples. Increase this percentage in `generate_dataset.py`:
   ```python
   if random.random() < 0.5:  # Change from 0.3 to 0.5 or higher
   ```

2. **Include rules without metadata**: Modify the parser to include helper rules and other rules that don't have METADATA blocks.

3. **Create synthetic variations**: Add code to create variations of existing examples with different phrasings or styles.

4. **Include more files**: Currently only parsing `policy/release/**/*.rego` (excluding test files). You could include additional rule files.

## Example Format

Each example in the JSONL files contains:

```json
{
  "instruction": "Rule title and description from METADATA",
  "context": "package declaration + imports + helper cheat sheet",
  "input_code": "Only for refactor tasks - code with style issues",
  "output_code": "The correct, validated Rego code",
  "task_type": "implement" or "refactor"
}
```

## Token Limits

Examples are designed to stay under 1024 tokens. Current examples average around 300-400 tokens, leaving room for growth.

## Quality Assurance

All examples are validated with:
- ✅ `opa parse` - Syntax validation
- ✅ `opa fmt` - Formatting validation (code is auto-formatted)
- ✅ `regal` - Linting (errors only, warnings ignored)
- ✅ `opa test` - Test execution (if tests exist, failures are warnings)

## Next Steps

1. **Fine-tune the model**: Use the generated `train.jsonl` and `eval.jsonl` files with your fine-tuning pipeline.

2. **Evaluate**: Test the fine-tuned model on the eval set to measure performance.

3. **Iterate**: If the model needs more examples or different task types, regenerate the dataset with modified parameters.

4. **Inference**: Use `inference_helper.py` to build proper context when generating new rules with the fine-tuned model.

