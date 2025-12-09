# Plan: Training Model to Fix Rego Compiler Errors

## Overview
Focus on training the model to recognize and fix Rego compilation errors (parse/compile errors from OPA), which are the most critical errors that prevent code from running.

## Goals
1. Model can identify compiler error types from OPA error messages
2. Model can generate correct fixes for common compiler error patterns
3. Reduce repair iterations by 50%+ for compiler errors
4. Achieve 90%+ first-attempt fix rate for common compiler errors

## Compiler Error Categories

### 1. Variable Redeclaration (`var <name> declared above`)
**Frequency:** Very common
**Root Cause:** Using same variable name in multiple declarations
**Example:**
```rego
# WRONG
deny contains result if {
    some result in task.results
    result := {"msg": "error"}  # Error: result already declared
}

# CORRECT
deny contains result if {
    some res in task.results
    result := {"msg": "error"}
}
```

### 2. Unsafe Variable (`var <name> is unsafe`)
**Frequency:** Very common
**Root Cause:** Variable not bound to input or missing iteration
**Example:**
```rego
# WRONG
deny contains result if {
    task.status == "Failed"  # Error: task is unsafe
}

# CORRECT
deny contains result if {
    some att in input.attestations
    some task in att.predicate.buildConfig.tasks
    task.status == "Failed"
}
```

### 3. Type Errors (`type error: ...`)
**Frequency:** Common
**Root Cause:** Wrong data structure access (array vs object)
**Example:**
```rego
# WRONG
deny contains result if {
    task.results.commit  # Error: results is array, not object
}

# CORRECT
deny contains result if {
    some result in task.results
    result.name == "commit"
}
```

### 4. Syntax Errors (`non-terminated string`, `unexpected token`)
**Frequency:** Less common
**Root Cause:** Invalid keywords, missing braces, string issues
**Example:**
```rego
# WRONG
rule check_task {
    if task.status == "Failed" then deny
}

# CORRECT
deny contains result if {
    task.status == "Failed"
    result := {"msg": "task failed"}
}
```

## Training Data Strategy

### Phase 1: Collect Real Compiler Errors (Week 1)

**Source:** Current inference runs that fail execution check

**Process:**
1. Run inference on 200-500 diverse instructions
2. Capture all code that fails `check_execution_against_attestations()`
3. Filter to only compiler errors (syntax/compile errors, not just undefined rules)
4. Extract full OPA error JSON for each failure
5. Target: 100-200 real compiler error examples

**Script:** `collect_compiler_errors.py`
```python
# Pseudo-code
for instruction in instructions:
    code, _ = agentic_inference(...)
    errors, _ = check_execution_against_attestations(code, ...)
    
    for error in errors:
        # Check if it's a compiler error (not just undefined)
        if is_compiler_error(error):
            opa_error = get_opa_error_json(code, attestation_file)
            store_error_example(instruction, code, opa_error)
```

**Error Classification:**
- Parse errors (syntax)
- Compile errors (variable redeclaration, unsafe variables)
- Type errors
- Other runtime errors (exclude these - focus on compile-time)

### Phase 2: Generate Corrections (Week 1-2)

**Process:**
1. For each collected error, generate correction
2. Use correction templates for common patterns
3. Validate all corrections with OPA
4. Ensure corrections compile and execute

**Correction Methods:**
- **Template-based** (80%): Pre-written fixes for common patterns
- **LLM-assisted** (20%): GPT-4/Claude for complex cases

**Script:** `generate_compiler_error_corrections.py`

**Correction Templates:**
```python
CORRECTION_TEMPLATES = {
    "variable_redeclaration": {
        "pattern": r"some (\w+) in .*;\s*\1 :=",
        "fix": "Change iteration variable name",
        "example": "some res in ...; result := ..."
    },
    "unsafe_variable": {
        "pattern": r"(\w+)\.\w+",  # Variable used before declaration
        "fix": "Add iteration: some var in input.attestations...",
        "example": "some att in input.attestations; some task in ..."
    },
    "type_error_array_access": {
        "pattern": r"(\w+)\.(\w+)\.(\w+)",  # Array accessed as object
        "fix": "Iterate array: some item in array; item.field",
        "example": "some result in task.results; result.name"
    }
}
```

### Phase 3: Synthesize Variations (Week 2)

**Goal:** Scale to 1000+ examples per error category

**Process:**
1. For each error pattern, create variations:
   - Different variable names
   - Different contexts (tasks, materials, subjects, results)
   - Different rule types (deny, warn, allow)
   - Different error locations in code
2. Generate incorrect code with error
3. Run OPA to get actual error message
4. Generate correction
5. Validate correction

**Script:** `generate_compiler_error_variations.py`

**Variation Strategy:**
- Variable redeclaration: 500+ examples
- Unsafe variable: 500+ examples
- Type errors: 300+ examples
- Syntax errors: 200+ examples

### Phase 4: Format Training Data (Week 2)

**Format:** Chat conversation with error correction

**Structure:**
```
System: [System prompt]
User: [Original instruction]
Assistant: [Incorrect code with compiler error]
User: [OPA error message + repair request]
Assistant: [Corrected code]
```

**Script:** `build_compiler_error_dataset.py`

**Output:** JSONL file with chat format

### Phase 5: Fine-tuning (Week 3)

**Dataset Mix:**
- 60% Normal examples (existing training data)
- 40% Compiler error correction examples

**Training Strategy:**
- Option A: Single fine-tune with mixed dataset
- Option B: Continue fine-tuning from existing model
- Option C: LoRA adapter for error correction

**Validation:**
- Test on held-out compiler error patterns
- Measure first-attempt fix rate
- Measure iteration reduction

## Implementation Scripts

### 1. `collect_compiler_errors.py`

**Purpose:** Collect real compiler errors from inference runs

**Features:**
- Run inference on instruction set
- Capture execution failures
- Filter to compiler errors only
- Extract OPA error JSON
- Store in structured format

**Output:** `data/compiler_errors/collected_errors.jsonl`

### 2. `generate_compiler_error_corrections.py`

**Purpose:** Generate corrections for collected errors

**Features:**
- Load collected errors
- Apply correction templates
- Use LLM for complex cases
- Validate corrections with OPA
- Store corrected examples

**Output:** `data/compiler_errors/corrected_errors.jsonl`

### 3. `generate_compiler_error_variations.py`

**Purpose:** Generate synthetic variations of error patterns

**Features:**
- Mutation engine for common patterns
- Template-based generation
- OPA validation for all generated code
- Ensure diversity

**Output:** `data/compiler_errors/variations.jsonl`

### 4. `build_compiler_error_dataset.py`

**Purpose:** Combine and format for training

**Features:**
- Load all error sources
- Format as chat conversations
- Deduplicate
- Balance error categories
- Split train/val/test (80/10/10)

**Output:** `data/compiler_errors/final_dataset.jsonl`

## Training Data Format

### Example Entry

```jsonl
{"messages": [
  {"role": "system", "content": "You are an expert Rego/OPA policy assistant..."},
  {"role": "user", "content": "Write a rule that checks all tasks for a result named 'commit'"},
  {"role": "assistant", "content": "```rego\npackage attestation_check\n\nimport rego.v1\n\ndeny contains result if {\n    some task in predicate.buildConfig.tasks\n    some result in task.results\n    result.name == \"commit\"\n    result := {\"msg\": \"error\"}\n}\n```"},
  {"role": "user", "content": "The generated Rego code has validation errors. Please fix them.\n\nError from opa eval:\n{\"errors\": [{\"message\": \"var result declared above\", \"code\": \"rego_compile_error\", \"location\": {\"file\": \"/tmp/rego_exec_123.rego\", \"row\": 8, \"col\": 5}}]}\n\nGenerated code:\n```rego\npackage attestation_check\n\nimport rego.v1\n\ndeny contains result if {\n    some task in predicate.buildConfig.tasks\n    some result in task.results\n    result.name == \"commit\"\n    result := {\"msg\": \"error\"}\n}\n```\n\nPlease provide the corrected Rego code that fixes these errors."},
  {"role": "assistant", "content": "```rego\npackage attestation_check\n\nimport rego.v1\n\ndeny contains result if {\n    some att in input.attestations\n    some task in att.predicate.buildConfig.tasks\n    some res in task.results\n    res.name == \"commit\"\n    result := {\"msg\": \"error\"}\n}\n```"}
]}
```

## Success Metrics

### Primary Metrics
1. **First-Attempt Fix Rate**: % of compiler errors fixed on first repair attempt
   - Target: 90%+ for common errors
   - Target: 70%+ for all compiler errors

2. **Iteration Reduction**: Average iterations before success
   - Baseline: ~3 iterations for compiler errors
   - Target: ~1.5 iterations (50% reduction)

3. **Error Detection Accuracy**: Can model identify error type?
   - Target: 95%+ correct error type identification

### Secondary Metrics
- Correction compile rate: 100% (all corrections must compile)
- Correction execution rate: 95%+ (corrections should execute)
- No degradation on normal code generation

## Timeline

### Week 1: Collection & Initial Corrections
- **Day 1-2:** Implement `collect_compiler_errors.py`
  - Run inference on 200+ instructions
  - Collect compiler errors
  - Target: 100-200 examples
- **Day 3-4:** Implement `generate_compiler_error_corrections.py`
  - Create correction templates
  - Generate corrections
  - Validate all corrections
- **Day 5:** Review and refine initial dataset

### Week 2: Scale Up
- **Day 1-2:** Implement `generate_compiler_error_variations.py`
  - Mutation engine
  - Template system
  - Generate 1000+ variations
- **Day 3:** Implement `build_compiler_error_dataset.py`
  - Format as chat conversations
  - Deduplicate and balance
  - Create train/val/test splits
- **Day 4-5:** Quality review and dataset refinement

### Week 3: Training & Evaluation
- **Day 1-2:** Integrate with training pipeline
  - Update `train_policy.py`
  - Mix with normal examples (60/40 split)
  - Configure training
- **Day 3-4:** Fine-tuning
  - Run training
  - Monitor metrics
  - Save checkpoints
- **Day 5:** Evaluation
  - Test on held-out compiler errors
  - Measure improvements
  - Generate report

## Priority Error Patterns

### High Priority (Focus First)
1. **Variable redeclaration** - Most common, easy to fix
2. **Unsafe variables** - Very common, clear fix pattern
3. **Type errors (array access)** - Common, specific fix needed

### Medium Priority
4. **Type errors (other)** - Less common but important
5. **Syntax errors** - Less common, but critical when they occur

### Low Priority
6. **Other compile errors** - Rare cases

## Validation Requirements

**Every training example MUST:**
1. Have a real OPA compiler error (not just undefined rule)
2. Include full OPA error JSON (not paraphrased)
3. Have a validated correction that:
   - Compiles successfully
   - Executes without errors
   - Fixes the specific error

**Quality Checks:**
- All corrections validated with `opa eval`
- Error messages are actual OPA output
- Corrections address the root cause, not just symptoms

## Next Steps

1. **Immediate:** Start collecting compiler errors from current inference runs
2. **This Week:** Build initial dataset of 100-200 examples
3. **Next Week:** Scale to 1000+ examples with variations
4. **Week 3:** Fine-tune and evaluate

## File Structure

```
qwen2.5_model/
├── collect_compiler_errors.py
├── generate_compiler_error_corrections.py
├── generate_compiler_error_variations.py
├── build_compiler_error_dataset.py
├── data/
│   └── compiler_errors/
│       ├── collected_errors.jsonl
│       ├── corrected_errors.jsonl
│       ├── variations.jsonl
│       └── final_dataset.jsonl
└── ...
```

## Dependencies

- Existing: All dependencies from `infer_policy.py`
- New: `openai` or `anthropic` (for LLM-assisted corrections, optional)

