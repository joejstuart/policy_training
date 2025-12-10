# Two-Stage Training Implementation Plan

This document outlines the implementation plan for the two-stage Rego policy training system. It references existing code that should be reused and identifies new components to build.

---

## Implementation Status

| Component | Status | File |
|-----------|--------|------|
| Data Generation | ✅ Complete | `scripts/generate_two_stage_dataset.py` |
| Training Script | ✅ Complete | `src/train_policy.py` (updated for two-stage format) |
| Inference Script | ✅ Complete | `src/infer_two_stage.py` |
| Training Data | ✅ Generated | `data/training/two_stage/*.jsonl` |

### Quick Commands

```bash
# 1. Generate training data (uses LLM for ANALYSIS sections)
python scripts/generate_two_stage_dataset.py

# 2. Train Stage 1 (context inference)
python src/train_policy.py \
    --train-path data/training/two_stage/stage1_train.jsonl \
    --eval-path data/training/two_stage/stage1_eval.jsonl \
    --output-dir models/stage1-context-inference

# 3. Train Stage 2 (rule generation)
python src/train_policy.py \
    --train-path data/training/two_stage/stage2_train.jsonl \
    --eval-path data/training/two_stage/stage2_eval.jsonl \
    --output-dir models/stage2-rule-generation \
    --max-seq-len 2048

# 4. Run inference
python src/infer_two_stage.py \
    --stage1-model models/stage1-context-inference \
    --stage2-model models/stage2-rule-generation \
    --instruction "Check that all pipeline tasks succeeded"
```

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TWO-STAGE TRAINING PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   [Existing Rules]  ──►  [Data Generator]  ──►  [Stage 1 + Stage 2]    │
│   policy/release/         scripts/new/          data/training/          │
│                                                                         │
│   [Stage 1 Model]   ◄──  [Train Stage 1]   ◄──  [stage1_train.jsonl]   │
│   models/stage1/          src/new/                                      │
│                                                                         │
│   [Stage 2 Model]   ◄──  [Train Stage 2]   ◄──  [stage2_train.jsonl]   │
│   models/stage2/          src/new/                                      │
│                                                                         │
│   [Two-Stage Infer] ◄──  [Stage 1] + [Stage 2]  ──►  [Rego Output]     │
│   src/new/                                                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Generation Scripts

### 1.1 New Script: `scripts/generate_two_stage_dataset.py`

**Purpose:** Generate training data for both Stage 1 and Stage 2 from existing Rego policies.

**Reuse from existing code:**

| Component | Source File | What to Reuse |
|-----------|-------------|---------------|
| Rule parsing | `scripts/generate_dataset.py` | `parse_rego_file()`, `extract_metadata_block()`, rule extraction logic (lines 62-272) |
| Helper extraction | `scripts/generate_dataset.py` | `extract_used_helpers()`, `HELPER_DESCRIPTIONS` dict (lines 276-342) |
| Import extraction | `scripts/generate_dataset.py` | `extract_used_imports()` (lines 295-318) |
| Library indexing | `src/library_indexer.py` | `LibraryIndexer` class - indexes all helpers with signatures, docs, usage examples |
| Library mapping | `src/library_mapper.py` | `LibraryMapper` class - maps import prefixes to directories |
| Context extraction | `src/context_extractor.py` | `extract_signature()`, `scan_usage_sites_in_rules()` |
| Validation | `scripts/generate_dataset.py` | `validate_rego_code()` using opa parse/regal (lines 369-459) |

**Key enhancement needed:**

The current `parse_rego_file()` in `generate_dataset.py` only extracts the main deny/warn block. However, the actual `train.jsonl` data includes **complete rule files** with private helpers. The two-stage generator needs to:

1. **Extract associated private helpers** — Functions starting with `_` that are called by the main rule
2. **Include package + imports + helpers + rule** — Assemble into complete, compilable output
3. **Order correctly** — Package → imports → private helpers → main rule

```python
def extract_associated_helpers(content: str, rule_code: str) -> List[str]:
    """Find private helpers (_name) that the rule depends on.
    
    1. Parse rule_code to find all function calls starting with _
    2. Search content for those function definitions
    3. Recursively find helpers that helpers depend on
    4. Return in dependency order (deps first)
    """
    # Pattern to find _function_name calls
    helper_calls = set(re.findall(r'\b(_\w+)\s*\(', rule_code))
    
    # Find definitions in the file
    helpers = []
    for helper_name in helper_calls:
        # Find: _helper_name := ... or _helper_name(...) := ...
        pattern = rf'^{re.escape(helper_name)}(?:\([^)]*\))?\s*:='
        # Extract the complete helper function
        ...
    
    return helpers
```

**New functionality to build:**

```python
# Pseudo-code structure

class TwoStageDataGenerator:
    """Generates Stage 1 and Stage 2 training examples from existing policies."""
    
    def __init__(self, repo_root: Path):
        # Reuse existing components
        self.mapper = LibraryMapper(repo_root)
        self.mapper.build_mappings()
        self.indexer = LibraryIndexer(repo_root, self.mapper)
        self.indexer.index_all_libraries()
    
    def generate_from_policy(self, rego_file: Path) -> Tuple[Stage1Example, Stage2Example]:
        """Generate both Stage 1 and Stage 2 examples from a policy file."""
        
        # 1. Parse the rule (reuse from generate_dataset.py)
        parsed = parse_rego_file(rego_file)
        
        for rule in parsed.rules:
            # 2. Extract REQUIREMENTS from metadata
            requirements = self.extract_requirements(rule)
            
            # 3. Generate Stage 1 output (what we want the model to infer)
            stage1_output = self.generate_stage1_context(rule, parsed)
            
            # 4. Generate Stage 2 output (ANALYSIS + RULE + TESTS)
            stage2_output = self.generate_stage2_output(rule, parsed)
            
            yield Stage1Example(requirements, stage1_output)
            yield Stage2Example(requirements + stage1_output, stage2_output)
    
    def extract_requirements(self, rule: Dict) -> str:
        """Convert METADATA to natural language requirements."""
        # NEW: Convert structured metadata to requirements format
        # - Package: from parsed.package
        # - Rule type: from rule name (deny/warn/allow)
        # - Short name: from custom.short_name
        # - Purpose: from description
        # - Behavioral requirements: from title + description
        pass
    
    def generate_stage1_context(self, rule: Dict, parsed: RegoFile) -> str:
        """Generate ATTESTATION_SCHEMA + AVAILABLE_HELPERS + RULE_DATA_KEYS."""
        
        # ATTESTATION_SCHEMA: Analyze rule code to identify accessed fields
        schema = self.infer_attestation_schema(rule["code"])
        
        # AVAILABLE_HELPERS: Reuse extract_used_helpers() then get full info from indexer
        helper_names = extract_used_helpers(rule["code"])
        helpers = [self.indexer.get_helper_context(h) for h in helper_names]
        
        # RULE_DATA_KEYS: Extract from lib.rule_data() calls
        rule_data_keys = self.extract_rule_data_keys(rule["code"])
        
        return format_stage1_output(schema, helpers, rule_data_keys)
    
    def infer_attestation_schema(self, code: str) -> List[SchemaEntry]:
        """NEW: Analyze code to infer which attestation fields are accessed."""
        # Pattern matching for:
        # - att.statement.predicate.buildConfig.tasks[]
        # - input.attestations[].statement...
        # - lib.pipelinerun_attestations patterns
        pass
    
    def extract_rule_data_keys(self, code: str) -> List[RuleDataKey]:
        """NEW: Extract rule_data keys from lib.rule_data() calls."""
        # Pattern: lib.rule_data("key_name")
        # Look up documentation from existing rule_data patterns
        pass
    
    def generate_expected_behavior(self, rule: Dict, rule_path: Path) -> str:
        """NEW: Generate expected behavior from test file or metadata."""
        test_path = rule_path.parent / rule_path.name.replace(".rego", "_test.rego")
        if test_path.exists():
            # Parse test file to extract scenarios
            return self.extract_test_scenarios(test_path)
        else:
            # Generate from metadata
            return self.infer_scenarios_from_metadata(rule)
    
    def generate_stage2_output(self, rule: Dict, parsed: RegoFile) -> str:
        """Generate ANALYSIS + RULE + TESTS."""
        
        # ANALYSIS: Field-to-logic mapping (NEW)
        analysis = self.generate_analysis(rule)
        
        # RULE: The COMPLETE code including:
        #   - Package declaration
        #   - All imports
        #   - Private helper functions (_status, _task_info, etc.)
        #   - METADATA comment block
        #   - Main deny/warn rule
        # IMPORTANT: Existing training data includes private helpers inline!
        # See: generate_dataset.py extracts complete rule blocks
        rule_code = self.extract_complete_rule_block(rule, parsed)
        
        # TESTS: From test file if exists (NEW)
        tests = self.extract_tests(parsed.path, rule)
        
        return format_stage2_output(analysis, rule_code, tests)
    
    def extract_complete_rule_block(self, rule: Dict, parsed: RegoFile) -> str:
        """Extract complete rule including private helpers.
        
        CRITICAL: The existing training data (train.jsonl) includes:
        1. Package declaration
        2. All imports
        3. Private helper functions (e.g., _status, _task_info, _builder_id)
        4. The main rule with METADATA
        
        This pattern is what makes the model good at creating well-structured rules.
        
        Example from existing data:
        ```rego
        package tasks
        
        import rego.v1
        import data.lib
        import data.lib.tekton
        
        _status(task) := status if {
            task.status
            ...
        }
        
        deny contains result if {
            some att in lib.pipelinerun_attestations
            some task in tekton.tasks(att)
            some status in _status(task)
            ...
        }
        ```
        
        The model learns to:
        - Create appropriate private helpers (prefixed with _)
        - Structure code: imports → helpers → main rule
        - Break complex logic into helpers
        """
        # Reuse logic from generate_dataset.py that extracts:
        # - Full content from rule start to end
        # - Associated private helpers (functions starting with _)
        # - Package and imports
        pass
```

**Output files:**
- `data/training/two_stage/stage1_train.jsonl`
- `data/training/two_stage/stage1_eval.jsonl`
- `data/training/two_stage/stage2_train.jsonl`
- `data/training/two_stage/stage2_eval.jsonl`

---

### 1.2 LLM-Assisted Data Generation

**Purpose:** Use an LLM to synthesize high-quality training examples from existing rules.

This is a key technique already used successfully in the current training data. An LLM reads the rule metadata and code, then generates:

1. **Natural language instructions** — What a user would ask to produce this rule
2. **Requirement summaries** — Converting structured METADATA to natural language
3. **ANALYSIS sections** — Explaining the field-to-logic mapping

**Script: `scripts/llm_generate_training_data.py`**

```python
import openai  # or anthropic, etc.

class LLMDataGenerator:
    """Uses an LLM to generate training examples from existing rules."""
    
    def __init__(self, model: str = "gpt-4o"):
        self.client = openai.OpenAI()
        self.model = model
    
    def generate_instruction(self, rule_code: str, metadata: dict) -> str:
        """Generate a natural language instruction that would produce this rule.
        
        The LLM reads the rule and metadata, then synthesizes an instruction
        that a user might write to request this rule.
        """
        prompt = f"""You are helping create training data for a Rego policy model.

Given this Rego rule and its metadata, write a natural language instruction 
that a user would give to request this rule be written.

METADATA:
- Title: {metadata.get('title', '')}
- Description: {metadata.get('description', '')}
- Short name: {metadata.get('short_name', '')}
- Failure message: {metadata.get('failure_msg', '')}

RULE CODE:
```rego
{rule_code}
```

Write a clear, specific instruction (1-3 sentences) that would lead to this rule.
Vary your phrasing - don't always use "Write a Rego deny rule that..."
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Some variation
        )
        return response.choices[0].message.content.strip()
    
    def generate_requirements(self, rule_code: str, metadata: dict, package: str) -> str:
        """Convert structured metadata to REQUIREMENTS format for Stage 1 input."""
        prompt = f"""Convert this Rego rule metadata into a REQUIREMENTS section.

METADATA:
- Package: {package}
- Title: {metadata.get('title', '')}
- Description: {metadata.get('description', '')}
- Short name: {metadata.get('short_name', '')}
- Failure message: {metadata.get('failure_msg', '')}
- Solution: {metadata.get('solution', '')}

RULE CODE (for understanding the logic):
```rego
{rule_code}
```

Output format:
REQUIREMENTS:
- Package: [package name]
- Rule type: [deny/warn/allow]
- Short name: [short_name]
- Purpose: [one sentence summary]
- [Bullet points describing the behavioral requirements]
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # More deterministic
        )
        return response.choices[0].message.content.strip()
    
    def generate_analysis(self, rule_code: str, helpers_used: list) -> str:
        """Generate ANALYSIS section explaining field-to-logic mapping."""
        prompt = f"""Analyze this Rego rule and explain the field-to-logic mapping.

RULE CODE:
```rego
{rule_code}
```

HELPERS USED: {helpers_used}

Generate an ANALYSIS section that explains:
1. Which attestation fields are accessed and how
2. Why each helper was chosen
3. The message template and its arguments

Format each field analysis as:
- Field: [field path or description]
  Access: [how it's accessed - via which helper or direct]
  Role: [what it's used for in the rule logic]
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
```

**Benefits of LLM-assisted generation:**

1. **Natural variation** — Generates diverse instruction phrasings, not templated
2. **Understanding transfer** — LLM "reads" complex rule logic and explains it
3. **Quality ANALYSIS sections** — Explains reasoning a human would use
4. **Scalable** — Can process hundreds of rules quickly
5. **Consistent format** — Follows the target schema reliably

**Cost considerations:**

| Model | Cost per rule (est.) | Quality |
|-------|---------------------|---------|
| GPT-4o | ~$0.02 | Highest |
| GPT-4o-mini | ~$0.002 | Good |
| Claude 3.5 Sonnet | ~$0.01 | High |
| Local LLM (Qwen-72B) | Free | Moderate |

For ~50 rules: GPT-4o = ~$1, GPT-4o-mini = ~$0.10

---

### 1.3 New Script: `scripts/augment_two_stage_data.py`

**Purpose:** Create additional variations of training examples for robustness.

**Augmentation strategies:**

1. **Requirements paraphrasing** — LLM generates 2-3 different ways to phrase each instruction
2. **Partial context** — Some examples with fewer helpers listed
3. **Order shuffling** — Randomize order of helpers, schema paths
4. **Optional sections** — Some Stage 2 examples without TESTS

---

## Phase 2: Training Scripts

### 2.1 New Script: `src/train_stage1.py`

**Purpose:** Train the context inference model (Stage 1).

**Reuse from existing code:**

| Component | Source File | What to Reuse |
|-----------|-------------|---------------|
| Model loading | `src/train_policy.py` | `load_qwen_model()` (lines 288-423) |
| LoRA application | `src/train_policy.py` | `apply_lora()` (lines 426-498) |
| Dataset class | `src/train_policy.py` | `PolicyDataset` pattern (lines 212-285) |
| Training loop | `src/train_policy.py` | `train_with_trainer()` (lines 501-617) |
| Dynamic padding | `src/train_policy.py` | `rego_collate_fn()` (lines 180-209) |

**New functionality:**

```python
# Stage 1 specific system prompt
STAGE1_SYSTEM_PROMPT = """You are a Rego policy context inference assistant.
Given natural language requirements for a policy rule, infer:
1. ATTESTATION_SCHEMA - Which attestation fields the rule needs to access
2. AVAILABLE_HELPERS - Which library functions should be used
3. RULE_DATA_KEYS - Which configurable parameters are needed

Choose from known attestation families (SLSA Provenance, Tekton PipelineRun, CycloneDX SBOM, SPDX SBOM) and known helper modules (lib.*, tekton.*, sbom.*, image.*)."""

class Stage1Dataset(PolicyDataset):
    """Dataset for Stage 1 training (requirements → context)."""
    
    def build_messages_from_example(self, example):
        # Input: REQUIREMENTS only
        # Output: ATTESTATION_SCHEMA + AVAILABLE_HELPERS + RULE_DATA_KEYS
        pass
```

**CLI:**
```bash
python src/train_stage1.py \
    --train-path data/training/two_stage/stage1_train.jsonl \
    --eval-path data/training/two_stage/stage1_eval.jsonl \
    --output-dir models/rego-stage1-lora \
    --model-name Qwen/Qwen2.5-1.5B-Instruct
```

---

### 2.2 New Script: `src/train_stage2.py`

**Purpose:** Train the rule generation model (Stage 2).

**Reuse:** Nearly identical to `train_stage1.py`, with different:
- System prompt (focused on rule generation)
- Dataset format (requirements + context → analysis + rule + tests)

```python
STAGE2_SYSTEM_PROMPT = """You are a Rego policy rule writer.
Given requirements and context (schema, helpers, rule data keys, expected behavior), generate:
1. ANALYSIS - How to combine fields and helpers into rule logic
2. RULE - Complete Rego code with METADATA annotations
3. TESTS - (Optional) Executable Rego test code with fixtures

The ANALYSIS explains your reasoning. The RULE must be valid Rego that compiles. METADATA (title, description, short_name, failure_msg, solution, collections, effective_on) is part of the RULE output."""
```

---

## Phase 3: Inference Pipeline

### 3.1 New Script: `src/infer_two_stage.py`

**Purpose:** Two-stage inference for generating Rego policies.

**Reuse from existing code:**

| Component | Source File | What to Reuse |
|-----------|-------------|---------------|
| Model loading | `src/infer_policy.py` | `load_policy_model()` (lines 149+) |
| OPA validation | `src/infer_policy.py` | `validate_rego_syntax()`, `validate_rego_semantic()` |
| Regal linting | `src/infer_policy.py` | `validate_rego_style()` |
| Library context | `src/smart_context_builder.py` | `SmartContextBuilder` for fallback/validation |

**New functionality:**

```python
class TwoStageRegoGenerator:
    """Generates Rego policies using two-stage inference."""
    
    def __init__(self, stage1_model_path: str, stage2_model_path: str, base_model: str):
        self.stage1_model, self.stage1_tokenizer = load_model(stage1_model_path, base_model)
        self.stage2_model, self.stage2_tokenizer = load_model(stage2_model_path, base_model)
        
        # For validation and fallback
        self.mapper = LibraryMapper(REPO_ROOT)
        self.indexer = LibraryIndexer(REPO_ROOT, self.mapper)
    
    def generate(self, requirements: str, validate: bool = True) -> Tuple[str, str, str]:
        """
        Generate Rego rule from requirements.
        
        Returns:
            (context, analysis, rule_with_tests)
        """
        # Stage 1: Infer context
        context = self._run_stage1(requirements)
        
        # Validate context (check helpers exist)
        if validate:
            context = self._validate_and_fix_context(context)
        
        # Stage 2: Generate rule
        stage2_input = f"REQUIREMENTS:\n{requirements}\n\n{context}"
        output = self._run_stage2(stage2_input)
        
        # Parse output into components
        analysis, rule, tests = self._parse_stage2_output(output)
        
        # Validate generated Rego
        if validate:
            rule = self._validate_and_fix_rule(rule)
        
        return context, analysis, rule + "\n\n" + tests
    
    def _validate_and_fix_context(self, context: str) -> str:
        """Validate that helpers exist, fix or warn if not."""
        # Parse AVAILABLE_HELPERS section
        # Check each helper against self.indexer.index
        # Replace hallucinated helpers with closest match or remove
        pass
    
    def _run_stage1(self, requirements: str) -> str:
        """Run Stage 1 model to infer context."""
        messages = [
            {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
            {"role": "user", "content": f"REQUIREMENTS:\n{requirements}"}
        ]
        
        input_ids = self.stage1_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.stage1_model.device)
        
        with torch.no_grad():
            outputs = self.stage1_model.generate(
                input_ids,
                max_new_tokens=1024,
                temperature=0.3,  # Low temp for consistent context
                do_sample=True,
            )
        
        # Slice off input tokens
        return self.stage1_tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
```

**CLI Interface:**

```bash
# Interactive mode
python src/infer_two_stage.py \
    --stage1-model models/rego-stage1-lora \
    --stage2-model models/rego-stage2-lora \
    --base-model Qwen/Qwen2.5-1.5B-Instruct

# Single query mode
python src/infer_two_stage.py \
    --stage1-model models/rego-stage1-lora \
    --stage2-model models/rego-stage2-lora \
    --requirements "Write a deny rule that checks if all tasks succeeded"

# Stage 1 only (for context inference)
python src/infer_two_stage.py \
    --stage1-model models/rego-stage1-lora \
    --stage1-only \
    --requirements "Write a deny rule that checks if all tasks succeeded"
```

---

## Phase 4: Validation & Evaluation

### 4.1 New Script: `scripts/evaluate_two_stage.py`

**Purpose:** Evaluate model quality for both stages.

**Stage 1 Metrics:**
- Schema path validity (do paths exist in known attestation families?)
- Helper existence (do helpers exist in library?)
- Rule data key validity
- Coverage (did model identify all needed components?)

**Stage 2 Metrics:**
- Compilation success (`opa parse`)
- Style compliance (`regal lint`)
- Test pass rate (`opa test`)
- Semantic correctness (manual review sample)

---

## Directory Structure

```
policy-training/
├── scripts/
│   ├── generate_dataset.py              # EXISTING - keep for reference
│   ├── generate_two_stage_dataset.py    # NEW - Phase 1.1
│   ├── augment_two_stage_data.py        # NEW - Phase 1.2
│   └── evaluate_two_stage.py            # NEW - Phase 4.1
├── src/
│   ├── train_policy.py                  # EXISTING - keep for reference
│   ├── infer_policy.py                  # EXISTING - keep for reference
│   ├── library_indexer.py               # EXISTING - reuse heavily
│   ├── library_mapper.py                # EXISTING - reuse heavily
│   ├── context_extractor.py             # EXISTING - reuse heavily
│   ├── smart_context_builder.py         # EXISTING - reuse for validation
│   ├── train_stage1.py                  # NEW - Phase 2.1
│   ├── train_stage2.py                  # NEW - Phase 2.2
│   └── infer_two_stage.py               # NEW - Phase 3.1
├── data/training/
│   ├── policy_rules/                    # EXISTING - keep
│   ├── attestation/                     # EXISTING - keep
│   └── two_stage/                       # NEW
│       ├── stage1_train.jsonl
│       ├── stage1_eval.jsonl
│       ├── stage2_train.jsonl
│       └── stage2_eval.jsonl
└── models/
    ├── rego-stage1-lora/                # NEW
    └── rego-stage2-lora/                # NEW
```

---

## Implementation Order

### Week 1: Data Generation
1. ✅ Document training format (`TWO_STAGE_INFERENCE.md`)
2. ✅ Create prototype example (`prototype_example.json`)
3. [ ] Implement `generate_two_stage_dataset.py`
   - Extract complete rule blocks with private helpers
   - Extract package, imports, and associated code
   - Parse rule metadata and test files
4. [ ] Implement `llm_generate_training_data.py`
   - Generate natural language instructions from rule metadata
   - Generate REQUIREMENTS sections
   - Generate ANALYSIS sections (field-to-logic mapping)
5. [ ] Generate initial dataset from `policy/release/`
   - Process all rules through extraction + LLM generation
   - Validate generated examples compile (`opa parse`)

### Week 2: Training Pipeline
5. [ ] Implement `train_stage1.py` (copy from `train_policy.py`, modify prompts)
6. [ ] Train Stage 1 model
7. [ ] Implement `train_stage2.py`
8. [ ] Train Stage 2 model

### Week 3: Inference & Evaluation
9. [ ] Implement `infer_two_stage.py`
10. [ ] Implement `evaluate_two_stage.py`
11. [ ] Test end-to-end pipeline
12. [ ] Iterate on training data quality

---

## Key Dependencies to Reuse

### From `library_indexer.py` (critical for Stage 1)

```python
# This indexes ALL helpers with signatures, docs, and usage examples
indexer = LibraryIndexer(repo_root, mapper)
indexer.index_all_libraries()

# Get helper info for Stage 1 output
helper_info = indexer.index["task_ref"]  # HelperInfo object
# - helper_info.signature: "task_ref(task) := ref if { ... }"
# - helper_info.doc: "Parses task reference into normalized form"
# - helper_info.usage_examples: ["ref := tekton.task_ref(task)"]
```

### From `generate_dataset.py` (critical for parsing)

```python
# Parse Rego files
parsed = parse_rego_file(rego_file)
# - parsed.package: "trusted_task"
# - parsed.imports: ["rego.v1", "data.lib", "data.lib.tekton"]
# - parsed.rules: [{name: "warn", metadata: {...}, code: "..."}]

# Extract helpers used in code
helpers = extract_used_helpers(rule["code"])
# Returns: ["lib.tasks_from_pipelinerun", "tekton.expiry_of", ...]

# Validate Rego code
is_valid, formatted, error = validate_rego_code(code, package, imports)
```

### Private Helper Extraction Pattern (critical for Stage 2 output)

The existing `train.jsonl` includes **complete rule files with inline private helpers**. This is a key pattern that makes the model effective at creating well-structured rules.

**Example from existing training data (line 1):**
```rego
package tasks

import rego.v1
import data.lib
import data.lib.json as j
import data.lib.tekton

_status(task) := status if {
    # Handle SLSA Provenance v0.2
    task.status
    not task.status.conditions
    status := [s |
        s := task.status
    ]
}

deny contains result if {
    some att in lib.pipelinerun_attestations
    some task in tekton.tasks(att)
    some status in _status(task)
    status != "Succeeded"
    result := lib.result_helper_with_term(...)
}
```

**What the model learns from this pattern:**
1. When to create private helpers (complex logic, reusable computations)
2. Naming convention: `_helper_name` (underscore prefix)
3. Code structure: package → imports → helpers → main rule
4. Breaking complex logic into testable, readable pieces

**Implementation note:** When extracting rules for Stage 2 training, include ALL private helpers that the rule depends on. Look for functions starting with `_` that are called within the rule.

---

## Notes

1. **Keep existing scripts working** - Don't modify `generate_dataset.py` or `train_policy.py`. Create new files.

2. **Reuse the library indexer heavily** - It already does the hard work of:
   - Scanning all library files
   - Extracting signatures and docstrings
   - Finding usage examples from tests
   - Keyword-based search for relevant helpers

3. **Start with a small dataset** - Generate from 10-20 rules first, validate quality, then scale up.

4. **Validate at every step** - Use `opa parse`, `regal lint`, and helper existence checks throughout.

