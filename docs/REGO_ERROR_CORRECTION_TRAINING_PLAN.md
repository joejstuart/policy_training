# Plan: Training Data for Rego Parse Error Correction

## Overview
Create training data that teaches the model to recognize and fix common Rego compilation/parse errors, improving the agentic workflow's repair phase.

## Goals
1. Model learns to recognize common Rego error patterns
2. Model can generate corrected code from error messages
3. Reduces iteration count in agentic workflow
4. Improves success rate of code generation

## Error Categories to Cover

### 1. Variable Redeclaration Errors
**Error Pattern:** `var <name> declared above`
**Common Causes:**
- Using same variable name in `some` and assignment
- Redeclaring variables in same scope
- Shadowing variables in nested blocks

**Example:**
```rego
# WRONG
deny contains result if {
    some result in task.results
    result := {"msg": "error"}
}

# CORRECT
deny contains result if {
    some res in task.results
    result := {"msg": "error"}
}
```

### 2. Unsafe Variable Errors
**Error Pattern:** `var <name> is unsafe`
**Common Causes:**
- Using variables not bound to input
- Missing input path navigation
- Variables not properly scoped

**Example:**
```rego
# WRONG
deny contains result if {
    some task in predicate.buildConfig.tasks
}

# CORRECT
deny contains result if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
}
```

### 3. Type Errors
**Error Pattern:** `type error: <details>`
**Common Causes:**
- Accessing array fields as objects
- Wrong type comparisons
- Missing null checks

**Example:**
```rego
# WRONG
deny contains result if {
    task.results.commit  # results is array, not object
}

# CORRECT
deny contains result if {
    some result in task.results
    result.name == "commit"
}
```

### 4. Syntax Errors
**Error Pattern:** `non-terminated string`, `unexpected token`
**Common Causes:**
- Invalid Rego keywords (rule, match, then, for, break)
- Missing braces
- Invalid string delimiters

**Example:**
```rego
# WRONG
rule check_task {
    if task.status == "Failed" then
        deny
}

# CORRECT
deny contains result if {
    task.status == "Failed"
    result := {"msg": "task failed"}
}
```

### 5. Undefined Reference Errors
**Error Pattern:** `undefined ref: <path>`
**Common Causes:**
- Wrong JSON path navigation
- Missing array iteration
- Typos in field names

**Example:**
```rego
# WRONG
deny contains result if {
    task.status == "Succeeded"  # task not bound
}

# CORRECT
deny contains result if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.status == "Succeeded"
}
```

## Training Data Structure

### Format: Instruction + Error + Correction

Each training example should include:

1. **Original Instruction** - What the user asked for
2. **Generated Code (with error)** - Code that has a parse/compile error
3. **Error Message** - **ACTUAL OPA error output** from running `opa eval` or `opa parse` (JSON format)
4. **Corrected Code** - The fixed version
5. **Error Category** - For categorization and filtering

### Important: Use Real Error Messages

**The error messages MUST be actual OPA output**, not synthetic or paraphrased. This ensures the model learns to parse and understand real OPA error formats, including:
- Exact error codes (`rego_compile_error`, `rego_unsafe_var_error`, etc.)
- Location information (file, row, column)
- Error message phrasing that OPA actually uses
- Full JSON structure that OPA returns

### Example Training Entry

```json
{
  "instruction": "Write a rule that checks all tasks for a result named 'commit'",
  "incorrect_code": "deny contains result if {\n    some task in predicate.buildConfig.tasks\n    some result in task.results\n    result.name == \"commit\"\n    result := {\"msg\": \"error\"}\n}",
  "error_message": "{\"errors\": [{\"message\": \"var result declared above\", \"code\": \"rego_compile_error\", \"location\": {\"file\": \"/tmp/rego_exec_123456.rego\", \"row\": 5, \"col\": 5}}]}",
  "corrected_code": "deny contains result if {\n    some att in input.attestations\n    some task in att.statement.predicate.buildConfig.tasks\n    some res in task.results\n    res.name == \"commit\"\n    result := {\"msg\": \"error\"}\n}",
  "error_category": "variable_redeclaration",
  "error_explanation": "Variable 'result' was declared in 'some result in task.results' and then redeclared with 'result := {...}'. Changed iteration variable to 'res' and kept 'result' for the deny output.",
  "error_source": "actual_opa_output",
  "opa_command": "opa eval --data <file> --input <input> data.deny --format json"
}
```

**Note:** The `error_message` field contains the exact JSON output from OPA, captured from `result.stderr` or `result.stdout` when `returncode != 0`.

## Data Generation Strategy

### How Incorrect Code Will Be Generated

We'll use multiple methods to generate incorrect code, ensuring diversity and realism:

#### Method 1: Collect from Real Model Outputs (Primary - Most Realistic)
**Source:** Code generated by the current fine-tuned model that fails execution check

**Process:**
1. Run inference on diverse instructions using current model
2. Capture all generated code that fails `check_execution_against_attestations()`
3. This gives us **realistic errors** that the model actually makes
4. Errors match the model's current failure patterns

**Advantages:**
- Most realistic - these are actual errors the model produces
- Natural error patterns based on model's current weaknesses
- Includes context (instruction + generated code)

**Example:**
```python
# From actual inference run
instruction = "Write a rule that checks all tasks for a result named 'commit'"
generated_code = """deny contains result if {
    some task in predicate.buildConfig.tasks  # Missing input.attestations
    some result in task.results
    result.name == "commit"
    result := {"msg": "error"}  # Variable redeclaration
}"""
# This code fails execution check → capture it
```

#### Method 2: Mutate Correct Code (Systematic Coverage)
**Source:** Take correct code and introduce specific error patterns

**Process:**
1. Start with correct Rego code (from training data or manually written)
2. Apply error mutation templates:
   - Variable redeclaration: Change iteration variable to match assignment variable
   - Unsafe variables: Remove `input.attestations` iteration
   - Type errors: Access array fields as objects
   - Missing iterations: Remove `some item in array` declarations
3. Validate that mutation produces a real OPA error

**Advantages:**
- Systematic coverage of all error categories
- Can target specific error patterns
- Ensures we have examples of each error type

**Example Template:**
```python
# Correct code
correct = """deny contains result if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    some res in task.results
    res.name == "commit"
    result := {"msg": "error"}
}"""

# Mutation: Variable redeclaration
incorrect = """deny contains result if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    some result in task.results  # Changed 'res' to 'result'
    result.name == "commit"
    result := {"msg": "error"}  # Now conflicts with above
}"""
```

#### Method 3: Template-Based Generation (Scale)
**Source:** Code templates with error patterns built-in

**Process:**
1. Create templates for each error category
2. Fill templates with variations:
   - Different variable names
   - Different contexts (tasks, materials, subjects)
   - Different rule types (deny, warn, allow)
3. Generate incorrect code from templates
4. **Always validate with OPA** to get real error message

**Advantages:**
- Can generate large numbers of examples quickly
- Ensures coverage across contexts
- Systematic variation

**Example Template:**
```python
variable_redeclaration_template = """deny contains result if {{
    some {context_var} in {context_path}
    some {iter_var} in {array_path}
    {iter_var}.{field} == "{value}"
    {iter_var} := {{"msg": "error"}}  # Error: redeclaration
}}"""

# Generate variations
for context in ["task", "material", "subject"]:
    for iter_var in ["result", "item", "obj"]:
        code = template.format(...)
        # Run OPA, capture error
```

#### Method 4: LLM-Assisted Generation (Quality)
**Source:** Use GPT-4/Claude to generate incorrect code with specific errors

**Process:**
1. Prompt: "Generate Rego code that has a [specific error type]"
2. Request code that compiles but has the target error
3. Validate with OPA to ensure error is real
4. Use for complex error patterns

**Advantages:**
- Can generate sophisticated error patterns
- Natural-looking incorrect code
- Good for edge cases

**Example Prompt:**
```
Generate Rego code that checks tasks for results, but has a variable 
redeclaration error. The code should:
- Use 'some result in task.results' to iterate
- Then try to assign 'result := {...}' 
- This should cause "var result declared above" error
```

### Phase 1: Collect Real Errors (Primary Method)
1. Run existing inference on diverse instructions
2. Capture all code that fails execution check
3. **Extract actual error messages from OPA** - Run `opa eval` or `opa parse` and capture the exact JSON output
4. Store the complete error response (not just the message text)
5. Manually or semi-automatically create corrections
6. Build initial dataset of 100-200 examples

**Error Collection Process:**
```python
# Pseudo-code for error collection from real inference
for instruction in diverse_instructions:
    generated_code, _ = agentic_inference(...)  # Current model
    errors, _ = check_execution_against_attestations(generated_code, ...)
    if errors:
        # Get actual OPA error
        opa_result = run_opa_eval(generated_code, attestation_file)
        error_json = opa_result.stderr or opa_result.stdout
        # Store: instruction, generated_code, error_json, corrected_code
```

### Phase 2: Synthesize Variations (Scale Up)
1. Use Method 2 (mutations) and Method 3 (templates) to generate variations:
   - Different variable names
   - Different contexts (tasks, materials, subjects)
   - Different error locations
   - Different rule types (deny, warn, allow)
2. **For each variation, run OPA to get actual error message**
   - Don't synthesize error messages
   - Generate incorrect code, run OPA, capture real error
3. Validate all generated code produces real OPA errors
4. Target 500-1000 examples per error category

**Important:** 
- Even for synthetic variations, we must run OPA to get the actual error message
- All incorrect code must produce real OPA errors (not just look wrong)
- This ensures the model sees real OPA error formats and phrasing

### Recommended Mix of Generation Methods

**Phase 1 (Initial Dataset - 100-200 examples):**
- 70% Method 1 (Real model outputs) - Most realistic
- 20% Method 2 (Mutated correct code) - Systematic coverage
- 10% Method 4 (LLM-assisted) - Complex patterns

**Phase 2 (Scale Up - 500-1000 per category):**
- 40% Method 1 (Real model outputs) - Keep realistic base
- 40% Method 3 (Template-based) - Scale efficiently
- 15% Method 2 (Mutations) - Fill gaps
- 5% Method 4 (LLM-assisted) - Edge cases

**Rationale:**
- Method 1 ensures we capture what the model actually does wrong
- Methods 2 & 3 ensure systematic coverage of all error types
- Method 4 handles complex/edge cases
- All methods validated with real OPA errors

### Validation Requirements

**Every incorrect code example MUST:**
1. **Produce a real OPA error** when run through `opa eval` or `opa parse`
   - Code that looks wrong but compiles is NOT useful
   - Code that has syntax errors but OPA doesn't catch them is NOT useful
2. **Have the error message captured from actual OPA output**
   - Not paraphrased or synthesized
   - Full JSON error structure preserved
3. **Have a verified correction** that:
   - Compiles successfully
   - Executes without errors
   - Addresses the specific error

**Validation Pipeline:**
```python
def validate_incorrect_code(incorrect_code, attestation_file):
    # 1. Run OPA - must produce error
    result = run_opa_eval(incorrect_code, attestation_file)
    if result.returncode == 0:
        return None  # Code compiles - not useful for training
    
    # 2. Capture actual error
    error_json = result.stderr or result.stdout
    if not error_json:
        return None  # No error captured
    
    # 3. Verify error is parseable
    try:
        error_data = json.loads(error_json)
        if "errors" not in error_data:
            return None  # Invalid error format
    except:
        return None
    
    return error_json  # Valid incorrect code with real error
```

### Phase 3: Generate Repair Prompts
1. Create training examples in chat format:
   ```
   System: [System prompt about Rego/OPA]
   User: [Original instruction]
   Assistant: [Incorrect code with error]
   User: [Error message + repair request]
   Assistant: [Corrected code]
   ```
2. Include error explanation in repair response
3. Show multiple repair attempts if needed

## Training Data Format

### Option A: Chat Format (Recommended)
```jsonl
{"messages": [
  {"role": "system", "content": "You are an expert Rego/OPA policy assistant..."},
  {"role": "user", "content": "Write a rule that checks all tasks for a result named 'commit'"},
  {"role": "assistant", "content": "```rego\n[incorrect code]\n```"},
  {"role": "user", "content": "The generated Rego code has validation errors. Please fix them.\n\nError from opa eval:\n{\"errors\": [{\"message\": \"var result declared above\", \"code\": \"rego_compile_error\", \"location\": {\"file\": \"/tmp/rego_exec_123.rego\", \"row\": 5, \"col\": 5}}]}\n\nGenerated code:\n```rego\n[incorrect code]\n```\n\nPlease provide the corrected Rego code that fixes these errors."},
  {"role": "assistant", "content": "```rego\n[corrected code]\n```"}
]}
```

**Note:** The error message in the user's repair request is the **actual JSON output from OPA**, exactly as returned by `opa eval` or `opa parse`. This matches what the model will see during the agentic workflow's repair phase.

### Option B: Instruction-Response Pairs
```jsonl
{"instruction": "[original instruction]", "response": "[incorrect code]", "error": "[error message]", "corrected_response": "[corrected code]"}
```

## Integration with Training Pipeline

### 1. Dataset Augmentation
- Add error correction examples to existing training dataset
- Mix with existing instruction-response pairs
- Maintain ratio: 70% normal examples, 30% error correction

### 2. Fine-tuning Strategy
- Option A: Single fine-tune with mixed dataset
- Option B: Two-stage fine-tune:
  1. Fine-tune on normal examples (existing)
  2. Continue fine-tuning on error correction examples
- Option C: LoRA adapter specifically for error correction

### 3. Validation
- Create validation set with known error patterns
- Measure:
  - Error detection rate
  - Correction success rate
  - Iteration reduction in agentic workflow

## Implementation Steps

### Step 1: Error Collection Script
**File:** `collect_rego_errors.py`

**Purpose:** Collect real errors from model inference runs

**Functionality:**
- Load current fine-tuned model
- Run inference on diverse instruction set (100-500 instructions)
- For each generated code:
  - Run `check_execution_against_attestations()`
  - If errors found, capture:
    - Original instruction
    - Generated code (with error)
    - Full OPA error JSON output
    - Attestation file used
- Store in structured format (JSON/JSONL)

**Output Format:**
```json
{
  "instruction": "...",
  "incorrect_code": "...",
  "error_json": "{...}",
  "error_category": "variable_redeclaration",
  "attestation_file": "attestation.json",
  "timestamp": "...",
  "model_version": "..."
}
```

**Implementation Details:**
- Reuse `infer_policy.py` functions (`agentic_inference`, `check_execution_against_attestations`)
- Batch process instructions
- Save progress incrementally (don't lose data on crash)
- Log statistics (error rate, error types distribution)

### Step 2: Error Correction Generator
**File:** `generate_error_corrections.py`

**Purpose:** Generate corrected code for collected errors

**Functionality:**
- Load collected errors from Step 1
- For each error:
  - Parse OPA error JSON to extract:
    - Error message
    - Error code
    - Location (row, column)
  - Generate correction using one of:
    - **Manual templates** (for common patterns)
    - **LLM-assisted** (GPT-4/Claude for complex cases)
    - **Rule-based fixes** (variable renaming, path fixes)
- Validate correction:
  - Run through OPA to ensure it compiles
  - Test against attestation file
  - Verify error is fixed

**Output Format:**
```json
{
  "instruction": "...",
  "incorrect_code": "...",
  "error_json": "{...}",
  "corrected_code": "...",
  "error_explanation": "...",
  "correction_method": "manual_template|llm_assisted|rule_based",
  "validation_status": "passed|failed"
}
```

**Implementation Details:**
- Create correction templates for each error category
- Use OpenAI/Anthropic API for LLM-assisted corrections
- Implement validation pipeline (OPA + execution check)
- Track correction success rate

### Step 3: Mutation & Template Generator
**File:** `generate_error_variations.py`

**Purpose:** Generate synthetic error variations

**Functionality:**
- Load correct Rego code examples (from existing training data)
- Apply mutation templates:
  - Variable redeclaration mutations
  - Unsafe variable mutations
  - Type error mutations
  - Path error mutations
- Generate template-based variations:
  - Different variable names
  - Different contexts (tasks, materials, subjects)
  - Different rule types
- For each variation:
  - Run OPA to get actual error
  - Generate correction
  - Validate correction

**Output:** Same format as Step 2

**Implementation Details:**
- Mutation engine with pattern matching
- Template system with variable substitution
- OPA validation for all generated code
- Ensure diversity (no duplicate patterns)

### Step 4: Dataset Builder
**File:** `build_error_correction_dataset.py`

**Purpose:** Combine all sources and format for training

**Functionality:**
- Load:
  - Collected errors (Step 1)
  - Corrected errors (Step 2)
  - Synthetic variations (Step 3)
- Format as chat conversations:
  - System prompt
  - User: Original instruction
  - Assistant: Incorrect code
  - User: Error message + repair request
  - Assistant: Corrected code
- Apply filtering:
  - Remove duplicates
  - Remove invalid corrections
  - Balance error categories
- Split dataset:
  - Train: 80%
  - Val: 10%
  - Test: 10%
- Export in training format (JSONL)

**Output Format:**
```jsonl
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

**Implementation Details:**
- Deduplication logic (similar code/errors)
- Category balancing (ensure all error types represented)
- Quality filtering (only validated corrections)
- Dataset statistics reporting

### Step 5: Training Integration
**File:** `train_policy.py` (modify existing)

**Purpose:** Integrate error correction dataset into training

**Changes Needed:**
1. **Dataset Loading:**
   - Load normal training dataset
   - Load error correction dataset
   - Mix with configurable ratio (default: 70% normal, 30% error correction)

2. **Data Mixing Logic:**
   ```python
   def mix_datasets(normal_dataset, error_correction_dataset, ratio=0.7):
       normal_size = int(len(normal_dataset) * ratio / (1 - ratio) * len(error_correction_dataset))
       # Interleave or concatenate based on ratio
       return mixed_dataset
   ```

3. **Training Configuration:**
   - Add `--error-correction-dataset` argument
   - Add `--error-correction-ratio` argument (default 0.3)
   - Update data loading to handle mixed datasets

**Implementation Details:**
- Backward compatible (works without error correction dataset)
- Configurable mixing ratios
- Logging of dataset composition
- Support for two-stage fine-tuning

### Step 6: Evaluation Framework
**File:** `evaluate_error_correction.py`

**Purpose:** Measure improvement from error correction training

**Functionality:**
- Test on held-out error patterns
- Measure:
  - **Error Detection Rate**: Can model identify errors?
  - **Correction Success Rate**: Do corrections compile?
  - **First-Try Success**: Corrections that work on first attempt
  - **Iteration Reduction**: Compare iterations before/after training
- Run agentic workflow on test set
- Compare metrics:
  - Before training (baseline)
  - After training (improved)

**Metrics to Track:**
```python
metrics = {
    "error_detection_rate": 0.85,  # % of errors correctly identified
    "correction_success_rate": 0.80,  # % of corrections that compile
    "first_try_success": 0.75,  # % that work on first repair attempt
    "avg_iterations_before": 3.2,  # Before training
    "avg_iterations_after": 1.8,  # After training
    "iteration_reduction": 0.44  # 44% reduction
}
```

**Implementation Details:**
- Reuse `infer_policy.py` agentic workflow
- Test on curated error patterns
- Statistical significance testing
- Report generation

### Step 7: Iteration & Refinement
**Process:**
1. Review evaluation results
2. Identify weak areas (error types model struggles with)
3. Generate more training data for those areas
4. Re-train and re-evaluate
5. Iterate until success criteria met

## Implementation Timeline

### Week 1: Foundation
- **Day 1-2:** Implement `collect_rego_errors.py`
  - Set up inference pipeline
  - Implement error capture
  - Test on small instruction set
- **Day 3-4:** Implement `generate_error_corrections.py`
  - Create correction templates
  - Set up LLM API integration
  - Implement validation
- **Day 5:** Initial data collection
  - Run on 100+ instructions
  - Generate 50-100 error examples
  - Manual review of corrections

### Week 2: Scale Up
- **Day 1-2:** Implement `generate_error_variations.py`
  - Mutation engine
  - Template system
  - OPA validation
- **Day 3-4:** Implement `build_error_correction_dataset.py`
  - Dataset formatting
  - Deduplication
  - Train/val/test split
- **Day 5:** Dataset building
  - Combine all sources
  - Target 500-1000 examples
  - Quality review

### Week 3: Training
- **Day 1-2:** Update `train_policy.py`
  - Dataset mixing logic
  - Configuration updates
  - Testing
- **Day 3-4:** Fine-tuning
  - Run training with error correction dataset
  - Monitor training metrics
  - Save checkpoints
- **Day 5:** Initial evaluation
  - Run evaluation script
  - Compare before/after
  - Identify improvements

### Week 4: Evaluation & Iteration
- **Day 1-2:** Comprehensive evaluation
  - Test on diverse error patterns
  - Measure all metrics
  - Generate report
- **Day 3-4:** Refinement
  - Identify weak areas
  - Generate additional training data
  - Re-train if needed
- **Day 5:** Final validation
  - Production readiness check
  - Documentation
  - Integration testing

## File Structure

```
qwen2.5_model/
├── collect_rego_errors.py          # Step 1: Error collection
├── generate_error_corrections.py    # Step 2: Correction generation
├── generate_error_variations.py     # Step 3: Synthetic variations
├── build_error_correction_dataset.py # Step 4: Dataset building
├── evaluate_error_correction.py    # Step 6: Evaluation
├── train_policy.py                 # Step 5: (modify existing)
├── infer_policy.py                 # (reuse existing)
├── data/
│   ├── error_correction/
│   │   ├── collected_errors.jsonl  # Raw collected errors
│   │   ├── corrected_errors.jsonl  # With corrections
│   │   ├── synthetic_variations.jsonl
│   │   └── final_dataset.jsonl     # Combined, formatted
│   └── ...
└── ...
```

## Dependencies

**New Python packages needed:**
- `openai` or `anthropic` (for LLM-assisted corrections)
- No other major dependencies (reuse existing OPA integration)

**Existing dependencies:**
- All existing dependencies from `infer_policy.py` and `train_policy.py`

## Success Criteria Checkpoints

**After Step 1:**
- ✓ Collected 100+ real error examples
- ✓ Error categories distributed
- ✓ All errors have actual OPA JSON output

**After Step 2:**
- ✓ 80%+ correction success rate
- ✓ All corrections validated with OPA
- ✓ Error explanations included

**After Step 3:**
- ✓ 500+ synthetic variations generated
- ✓ All variations produce real OPA errors
- ✓ Good coverage of error categories

**After Step 4:**
- ✓ Dataset formatted correctly
- ✓ Train/val/test split done
- ✓ Quality checks passed

**After Step 5:**
- ✓ Training runs successfully
- ✓ Model converges
- ✓ No degradation on normal examples

**After Step 6:**
- ✓ 30%+ iteration reduction
- ✓ 80%+ correction success rate
- ✓ Meets all success criteria

## Metrics to Track

1. **Error Detection Rate**: % of errors correctly identified
2. **Correction Success Rate**: % of corrections that compile
3. **Iteration Reduction**: Average iterations before success
4. **Error Category Coverage**: Distribution across error types
5. **False Positive Rate**: Incorrect "corrections" that break valid code

## Challenges & Solutions

### Challenge 1: Error Message Parsing
- **Solution**: 
  - **Use actual OPA JSON error output** - Don't synthesize or paraphrase
  - Standardize on OPA JSON error format (as returned by `opa eval --format json`)
  - Use structured error parsing in collection script to extract:
    - Error message text
    - Error code (`rego_compile_error`, `rego_unsafe_var_error`, etc.)
    - Location (file, row, column)
  - Preserve full JSON structure so model learns OPA's exact error format

### Challenge 2: Generating Correct Corrections
- **Solution**: 
  - Manual review for initial dataset
  - Use GPT-4/Claude for synthetic examples
  - Validate all corrections with OPA

### Challenge 3: Context Preservation
- **Solution**: Include full instruction context in repair prompts
- Show original plan/context in error correction examples

### Challenge 4: Overfitting to Error Patterns
- **Solution**: 
  - Diverse error variations
  - Mix with normal examples
  - Regular validation on unseen patterns

## Timeline Estimate

1. **Week 1**: Error collection and initial dataset (100-200 examples)
2. **Week 2**: Synthetic generation and dataset building (500-1000 examples)
3. **Week 3**: Training integration and fine-tuning
4. **Week 4**: Evaluation and iteration

## Success Criteria

- Model can fix 80%+ of common error patterns in first repair attempt
- Average iterations in agentic workflow reduced by 30-50%
- No degradation in normal code generation quality
- Error correction examples don't negatively impact base capabilities

## Next Steps

1. Review and approve plan
2. Start error collection from existing inference runs
3. Create initial dataset of 50-100 examples
4. Test fine-tuning on small subset
5. Iterate based on results

