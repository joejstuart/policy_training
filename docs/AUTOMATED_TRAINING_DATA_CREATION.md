# Automated Training Data Creation Plan

## Overview
Use AI assistance to automate 90%+ of training data creation, minimizing manual work to just review and validation.

## Two Critical Questions

### Q1: How are instructions generated?
**Answer:** Multiple automated sources, no manual work needed.

### Q2: Who fixes scripts when they generate compiler errors?
**Answer:** Iterative improvement - scripts fix themselves, or you ask me to fix them.

## Instruction Generation (Fully Automated)

### Option 1: Extract from Existing Training Dataset (Recommended)
**Source:** Your existing training dataset (e.g., `attestation_train.jsonl`)

**Process:**
```python
# Script automatically extracts instructions
def extract_instructions_from_dataset(dataset_file):
    instructions = []
    with open(dataset_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Extract instruction from training examples
            if "messages" in data:
                for msg in data["messages"]:
                    if msg["role"] == "user" and "Instruction:" in msg["content"]:
                        instruction = extract_instruction(msg["content"])
                        instructions.append(instruction)
    return instructions
```

**Advantages:**
- ✅ No manual work
- ✅ Instructions already validated
- ✅ Diverse and realistic
- ✅ Reuses existing data

### Option 2: Generate from Existing Dataset Patterns
**Source:** Analyze existing dataset to extract patterns, then generate variations

**Process:**
```python
# Script analyzes patterns and generates variations
def generate_instruction_variations(dataset_file, count=500):
    # Analyze common patterns
    patterns = analyze_instruction_patterns(dataset_file)
    
    # Generate variations
    variations = []
    for pattern in patterns:
        variations.extend(generate_variations(pattern, count // len(patterns)))
    
    return variations
```

**Advantages:**
- ✅ Automated generation
- ✅ Ensures diversity
- ✅ Can target specific error patterns

### Option 3: Reuse from Previous Inference Runs
**Source:** Instructions you've already tested

**Process:**
```python
# Script reads from inference logs or stored instructions
def load_instructions_from_logs(log_dir):
    instructions = []
    for log_file in Path(log_dir).glob("*.log"):
        # Extract instructions from logs
        instructions.extend(parse_instructions_from_log(log_file))
    return instructions
```

### Option 4: Template-Based Generation
**Source:** Templates that generate instructions targeting specific error patterns

**Process:**
```python
# Script generates instructions designed to trigger specific errors
TEMPLATES = [
    "Write a rule that checks all {context} for a result named '{name}'",
    "Write a rule that verifies {context} has {field} equal to '{value}'",
    # ... more templates
]

def generate_targeted_instructions(templates, count=200):
    instructions = []
    for template in templates:
        for context in ["tasks", "materials", "subjects"]:
            for name in ["commit", "url", "digest"]:
                instructions.append(template.format(context=context, name=name))
    return instructions[:count]
```

### Recommended Approach: Hybrid
**Use all sources combined:**
1. Extract 200-300 from existing training dataset (Option 1)
2. Generate 100-200 variations (Option 2)
3. Use 50-100 from previous runs if available (Option 3)
4. Generate 50-100 targeted for error patterns (Option 4)

**Total: 400-700 instructions, all automated**

**Script includes:**
- Automatic extraction from existing dataset
- Pattern analysis and variation generation
- Deduplication
- Statistics reporting

## Script Error Handling (Iterative Improvement)

### Problem: Scripts May Have Bugs
**Solution:** Multi-layered approach with automatic recovery

### Layer 1: Robust Error Handling (Built into Scripts)

**AI writes scripts with:**
```python
# All scripts include comprehensive error handling
try:
    # Main logic
    result = process_item(item)
except SpecificError as e:
    # Log error, skip item, continue
    log_error(item, e)
    continue
except Exception as e:
    # Unexpected error - save state and report
    save_progress()  # Don't lose work
    log_critical_error(e)
    # Continue with next item
    continue
```

**Features:**
- ✅ Graceful error handling
- ✅ Progress saving (resume capability)
- ✅ Detailed error logging
- ✅ Continues processing even if some items fail

### Layer 2: Self-Validation (Scripts Check Themselves)

**AI includes validation:**
```python
# Scripts validate their own output
def validate_output(output_file):
    """Validate that output is correct format."""
    errors = []
    with open(output_file) as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                # Validate structure
                if not validate_structure(data):
                    errors.append(f"Line {i}: Invalid structure")
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: JSON error - {e}")
    
    if errors:
        print(f"Validation found {len(errors)} errors")
        return False
    return True
```

### Layer 3: Iterative Fix Process

**When scripts have errors:**

**Scenario A: Script crashes on specific input**
1. Script logs error with full context
2. You show me the error: "Script failed on line 42 with error X"
3. I fix the script
4. Script resumes from saved progress (no data loss)

**Scenario B: Script produces incorrect output**
1. Validation catches it
2. You show me: "Validation failed - output has wrong format"
3. I fix the script
4. Re-run on failed items only

**Scenario C: Script logic is wrong**
1. You notice during review: "Corrections don't look right"
2. You show me examples: "These corrections are wrong because..."
3. I fix the logic
4. Re-run script

### Layer 4: Test Mode (Catch Errors Early)

**AI includes test mode:**
```python
# All scripts have --test mode
if args.test:
    # Run on small subset first
    test_items = items[:10]
    results = process_items(test_items)
    
    # Validate results
    if validate_results(results):
        print("✓ Test passed, proceeding with full run")
    else:
        print("✗ Test failed, fix issues before full run")
        sys.exit(1)
```

**Workflow:**
```bash
# 1. Test first
python collect_compiler_errors.py --test --instructions-file instructions.txt

# 2. If test passes, run full
python collect_compiler_errors.py --instructions-file instructions.txt
```

### Layer 5: Incremental Processing (No Data Loss)

**AI includes:**
```python
# Scripts process incrementally and save progress
def process_with_checkpoint(items, checkpoint_file):
    # Load existing progress
    processed = load_checkpoint(checkpoint_file)
    
    for item in items:
        if item.id in processed:
            continue  # Skip already processed
        
        try:
            result = process_item(item)
            save_result(result)
            mark_processed(item.id, checkpoint_file)
        except Exception as e:
            log_error(item, e)
            # Continue with next item
            continue
```

**Benefits:**
- ✅ Can stop and resume anytime
- ✅ No data loss on crash
- ✅ Can re-run on failed items only

## Complete Error Handling Workflow

### Example: Collection Script Has Bug

**Step 1: Script runs, hits error**
```python
# Script logs:
ERROR: Failed to process instruction "Write a rule..."
Traceback: ...
Item saved to failed_items.jsonl
Continuing with next item...
```

**Step 2: You review**
```bash
# Check what failed
cat failed_items.jsonl
# See: 5 items failed out of 200
```

**Step 3: You ask me to fix**
> "The collection script failed on 5 items with error: 'KeyError: instruction'. Here's the traceback: ..."

**Step 4: I fix and provide updated script**

**Step 5: Re-run on failed items only**
```bash
python collect_compiler_errors.py --resume --failed-items failed_items.jsonl
```

## Automation Strategy

### What AI Can Do (Automated)
1. ✅ Write collection scripts
2. ✅ Generate corrections using templates/logic
3. ✅ Create variations programmatically
4. ✅ Format data for training
5. ✅ Validate corrections with OPA
6. ✅ Generate synthetic error patterns
7. ✅ Parse and structure OPA errors

### What Requires Human (Minimal)
1. ⚠️ Review initial dataset quality (spot check)
2. ⚠️ Approve correction templates
3. ⚠️ Run scripts and monitor progress
4. ⚠️ Final validation before training

## Step-by-Step Automation Workflow

### Step 1: AI Writes Collection Script (5 minutes)

**You ask me:**
> "Create `collect_compiler_errors.py` that runs inference on instructions, captures compiler errors, and stores them in JSONL format"

**I provide:**
- Complete script with error filtering
- OPA error extraction
- Structured storage format
- Progress tracking

**You do:**
- Review script (2 min)
- Run it: `python collect_compiler_errors.py --instructions-file instructions.txt`

### Step 2: AI Writes Correction Generator (10 minutes)

**You ask me:**
> "Create `generate_compiler_error_corrections.py` with correction templates for variable redeclaration, unsafe variables, and type errors. Use template-based fixes for 80% and LLM API for 20%."

**I provide:**
- Complete script with correction templates
- Template matching logic
- LLM integration (optional)
- OPA validation
- Progress reporting

**You do:**
- Review templates (5 min)
- Run it: `python generate_compiler_error_corrections.py --input collected_errors.jsonl`

### Step 3: AI Writes Variation Generator (15 minutes)

**You ask me:**
> "Create `generate_compiler_error_variations.py` that generates 1000+ variations of compiler errors using mutation and templates. Ensure all generated code produces real OPA errors."

**I provide:**
- Complete script with mutation engine
- Template system
- Variation generation
- OPA validation
- Deduplication

**You do:**
- Review mutation patterns (5 min)
- Run it: `python generate_compiler_error_variations.py --target 1000`

### Step 4: AI Writes Dataset Builder (10 minutes)

**You ask me:**
> "Create `build_compiler_error_dataset.py` that combines all sources, formats as chat conversations, deduplicates, and splits train/val/test."

**I provide:**
- Complete script
- Chat format conversion
- Deduplication logic
- Train/val/test split
- Statistics reporting

**You do:**
- Run it: `python build_compiler_error_dataset.py`

### Step 5: Automated Quality Checks (Built-in)

**AI includes in scripts:**
- Automatic OPA validation
- Error message verification
- Correction validation
- Statistics and reports

**You do:**
- Review statistics output
- Spot check a few examples

## Detailed Automation Plan

### Phase 1: Collection (Automated 95%)

**Script Features (AI writes):**
```python
# collect_compiler_errors.py
- Loads instruction file or generates from existing dataset
- Runs inference using existing infer_policy.py functions
- Captures all execution failures
- Filters to compiler errors only (not undefined rules)
- Extracts full OPA error JSON
- Stores in structured format
- Progress tracking and resume capability
- Statistics reporting
```

**Your Work:**
1. Provide instruction file (or use existing)
2. Run script
3. Review statistics (5 min)

**Time: 5 minutes manual work**

### Phase 2: Corrections (Automated 90%)

**Script Features (AI writes):**
```python
# generate_compiler_error_corrections.py
- Loads collected errors
- Matches errors to correction templates
- Applies template-based fixes automatically
- For complex cases, uses LLM API (optional)
- Validates all corrections with OPA
- Reports success rate
- Stores corrected examples
```

**Correction Templates (AI creates):**
```python
TEMPLATES = {
    "variable_redeclaration": {
        "detect": lambda error: "var" in error and "declared above" in error,
        "fix": lambda code, error: rename_iteration_variable(code, error),
        "validate": lambda fixed: opa_validate(fixed)
    },
    "unsafe_variable": {
        "detect": lambda error: "unsafe" in error.lower(),
        "fix": lambda code, error: add_input_iteration(code, error),
        "validate": lambda fixed: opa_validate(fixed)
    },
    # ... more templates
}
```

**Your Work:**
1. Review correction templates (10 min)
2. Run script
3. Review correction success rate (2 min)
4. Manually fix any failed corrections (if < 10% failure rate)

**Time: 15 minutes manual work**

### Phase 3: Variations (Automated 98%)

**Script Features (AI writes):**
```python
# generate_compiler_error_variations.py
- Loads correct Rego code examples
- Applies mutation patterns programmatically
- Generates variations with different:
  * Variable names
  * Contexts (tasks, materials, subjects)
  * Rule types (deny, warn, allow)
  * Error locations
- Runs OPA on each to get real error
- Generates correction automatically
- Validates correction
- Ensures diversity (no duplicates)
```

**Mutation Patterns (AI creates):**
```python
MUTATIONS = {
    "variable_redeclaration": [
        lambda code: change_iteration_var_to_match_assignment(code),
        lambda code: change_assignment_var_to_match_iteration(code),
    ],
    "unsafe_variable": [
        lambda code: remove_input_iteration(code),
        lambda code: remove_some_declaration(code),
    ],
    # ... more mutations
}
```

**Your Work:**
1. Review mutation patterns (5 min)
2. Run script
3. Review statistics (2 min)

**Time: 7 minutes manual work**

### Phase 4: Dataset Building (Automated 100%)

**Script Features (AI writes):**
```python
# build_compiler_error_dataset.py
- Loads all error sources
- Formats as chat conversations automatically
- Deduplicates using code similarity
- Balances error categories
- Splits train/val/test (80/10/10)
- Generates statistics report
- Exports in training format
```

**Your Work:**
1. Run script
2. Review statistics (3 min)

**Time: 3 minutes manual work**

## Complete Workflow Example

### Day 1: Setup (30 min total)

**You:**
```bash
# 1. Ask AI to create collection script
"Create collect_compiler_errors.py that collects compiler errors from inference runs"

# 2. Review and run
python collect_compiler_errors.py --instructions-file instructions.txt --output collected.jsonl
# Output: 150 compiler errors collected
```

**AI provides:** Complete script ready to run

**Manual time: 10 min**

### Day 2: Corrections (20 min total)

**You:**
```bash
# 1. Ask AI to create correction generator
"Create generate_compiler_error_corrections.py with templates for common errors"

# 2. Review templates, run script
python generate_compiler_error_corrections.py --input collected.jsonl --output corrected.jsonl
# Output: 145 corrections generated (96% success rate)
```

**AI provides:** Script + correction templates

**Manual time: 15 min (review templates) + 5 min (fix 5 failed corrections)**

### Day 3: Variations (15 min total)

**You:**
```bash
# 1. Ask AI to create variation generator
"Create generate_compiler_error_variations.py to generate 1000 variations"

# 2. Run script
python generate_compiler_error_variations.py --target 1000 --output variations.jsonl
# Output: 1050 variations generated, all validated
```

**AI provides:** Complete script with mutation engine

**Manual time: 5 min (review) + 10 min (run time, automated)**

### Day 4: Dataset (10 min total)

**You:**
```bash
# 1. Ask AI to create dataset builder
"Create build_compiler_error_dataset.py to format and split dataset"

# 2. Run script
python build_compiler_error_dataset.py \
    --sources corrected.jsonl variations.jsonl \
    --output final_dataset.jsonl
# Output: 1195 examples, train/val/test split, statistics report
```

**AI provides:** Complete script

**Manual time: 3 min (review statistics)**

## AI-Assisted Quality Review

### Automated Quality Checks (Built into scripts)

**AI includes:**
- OPA validation for all corrections
- Error message verification (must be real OPA output)
- Correction success rate tracking
- Duplicate detection
- Category balancing
- Statistics generation

**You review:**
- Statistics report (2 min)
- Spot check 10-20 random examples (5 min)

## Template-Based Corrections (No Manual Work)

### Variable Redeclaration Template

**AI creates:**
```python
def fix_variable_redeclaration(code, error):
    """Automatically fix variable redeclaration errors."""
    # Parse error to find variable name
    var_name = extract_variable_name(error)
    
    # Find iteration declaration
    iteration_pattern = rf"some {var_name} in"
    
    # Find assignment
    assignment_pattern = rf"{var_name} :="
    
    # Generate new variable name for iteration
    new_iter_var = f"{var_name}_iter" if not var_name.endswith("_iter") else f"{var_name}_item"
    
    # Replace iteration variable
    fixed = re.sub(iteration_pattern, f"some {new_iter_var} in", code)
    fixed = re.sub(rf"{var_name}\.", f"{new_iter_var}.", fixed)
    
    return fixed
```

**No manual work needed** - AI writes the logic

### Unsafe Variable Template

**AI creates:**
```python
def fix_unsafe_variable(code, error):
    """Automatically fix unsafe variable errors."""
    # Extract variable name from error
    var_name = extract_variable_name(error)
    
    # Determine context (task, material, subject, etc.)
    context = determine_context(code, var_name)
    
    # Generate input iteration
    if context == "task":
        input_iter = "some att in input.attestations\n    some task in att.predicate.buildConfig.tasks"
    elif context == "material":
        input_iter = "some att in input.attestations\n    some material in att.predicate.materials"
    # ... more contexts
    
    # Insert before first use of variable
    fixed = insert_before_first_use(code, var_name, input_iter)
    
    return fixed
```

**No manual work needed** - AI writes the logic

## LLM-Assisted Corrections (Optional, Automated)

**For complex cases, AI can:**
- Use OpenAI/Anthropic API automatically
- Generate corrections for edge cases
- Validate with OPA
- Store results

**You:**
- Provide API key (one-time)
- Script handles everything else

## Validation Automation

**AI includes in all scripts:**
```python
def validate_correction(original_code, corrected_code, error):
    """Automatically validate correction."""
    # 1. Check correction compiles
    compile_result = opa_parse(corrected_code)
    if not compile_result.success:
        return False, "Correction doesn't compile"
    
    # 2. Check error is fixed
    execution_result = opa_eval(corrected_code, attestation_file)
    if execution_result.has_error(error):
        return False, "Error not fixed"
    
    # 3. Check no new errors introduced
    if execution_result.has_other_errors():
        return False, "New errors introduced"
    
    return True, "Valid correction"
```

**No manual validation needed** - all automated

## Statistics & Reporting (Automated)

**AI generates reports:**
```
Dataset Statistics:
- Total examples: 1,195
- Variable redeclaration: 485 (40.6%)
- Unsafe variable: 420 (35.1%)
- Type errors: 210 (17.6%)
- Syntax errors: 80 (6.7%)

Correction Success Rate: 96.2%
Validation Success Rate: 98.5%

Train/Val/Test Split:
- Train: 956 (80%)
- Val: 120 (10%)
- Test: 119 (10%)
```

**You review:** 2 minutes

## Total Manual Work Estimate

| Phase | Automated | Manual Work |
|-------|-----------|-------------|
| Collection | 95% | 5 min |
| Corrections | 90% | 15 min |
| Variations | 98% | 7 min |
| Dataset Building | 100% | 3 min |
| Quality Review | 80% | 7 min |
| **Total** | **~95%** | **~37 minutes** |

## Getting Started

### Immediate Next Steps

1. **Ask me to create the first script:**
   > "Create `collect_compiler_errors.py` that collects compiler errors from inference runs. It should use the existing `infer_policy.py` functions, filter to compiler errors only, and store in JSONL format."

2. **I'll provide:**
   - Complete, runnable script
   - Documentation
   - Example usage

3. **You:**
   - Review (2 min)
   - Run it
   - Review results

4. **Repeat for each script in sequence**

## Benefits of This Approach

✅ **Minimal manual work** - ~95% automated
✅ **Fast iteration** - Scripts run in minutes/hours
✅ **Quality assurance** - Built-in validation
✅ **Scalable** - Easy to generate 1000+ examples
✅ **Reproducible** - Scripts can be re-run
✅ **Flexible** - Easy to adjust templates/patterns

## What You Need to Provide

1. **Instruction file** (or use existing dataset)
2. **API key** (optional, for LLM-assisted corrections)
3. **Review time** (~30-40 minutes total)
4. **Run scripts** (automated, just execute)

Everything else is automated by AI!

## Summary: Answers to Key Questions

### Q1: How are instructions generated?

**Answer: Fully automated - no manual work**

**Sources (all automated):**
1. **Extract from existing training dataset** - Script reads your `attestation_train.jsonl` and extracts instructions
2. **Generate variations** - Script analyzes patterns and creates variations
3. **Reuse from logs** - Script reads from previous inference runs
4. **Template-based** - Script generates instructions targeting specific error patterns

**You provide:** Nothing - script handles it all

**Example:**
```bash
# Script automatically extracts from your existing dataset
python collect_compiler_errors.py \
    --source-dataset qwen2.5_model/attestation_train.jsonl \
    --extract-instructions \
    --output instructions.txt
# Output: 400 instructions extracted automatically
```

### Q2: Who fixes scripts when they generate compiler errors?

**Answer: Iterative process - scripts are self-healing, or you ask me to fix**

**Process:**
1. **Scripts have robust error handling** - They catch errors, log them, and continue
2. **Progress is saved** - No data loss if script crashes
3. **Test mode catches issues early** - Run `--test` first on small subset
4. **If script has bug:** You show me the error, I fix it, script resumes
5. **If output is wrong:** Validation catches it, you show me, I fix logic

**You provide:** Error reports (just copy/paste the error)

**Example workflow:**
```bash
# 1. Test first (catches errors early)
python collect_compiler_errors.py --test
# Output: ✓ Test passed

# 2. Run full (if test passes)
python collect_compiler_errors.py
# Error: KeyError on line 42

# 3. You: "Script failed with KeyError on line 42"
# 4. Me: [Fixes script, provides updated version]
# 5. You: Resume from checkpoint
python collect_compiler_errors.py --resume
# Output: ✓ Completed, processed 200 items
```

**Key Points:**
- ✅ Scripts save progress (can resume)
- ✅ Test mode catches errors early
- ✅ Error logging is detailed (easy to debug)
- ✅ I fix bugs when you report them
- ✅ No manual debugging needed from you

