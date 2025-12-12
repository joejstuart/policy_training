# Training Accuracy Improvement Plan

This document analyzes the strengths and weaknesses of both training approaches (Attestation Training vs. Two-Stage Training) and provides a concrete plan to improve Stage 1 and Stage 2 accuracy.

---

## Executive Summary

| Aspect | Attestation Training | Two-Stage Training |
|--------|---------------------|-------------------|
| **Rule Accuracy** | ✅ Higher accuracy | ⚠️ Lower accuracy |
| **Metadata Generation** | ❌ Minimal | ✅ Excellent |
| **Helper Creation** | ❌ None | ✅ Good |
| **Test Generation** | ❌ None | ✅ Good |
| **Structure** | ❌ Flat | ✅ Well-organized (ANALYSIS/RULE/TESTS) |
| **Context Provision** | ✅ Rich (trimmed JSON) | ⚠️ Abstract (schema paths) |
| **Grounding** | ✅ Concrete examples | ⚠️ More abstract |

**Key Insight:** The attestation training produces more accurate *rule logic* because it provides **concrete, grounded context** (actual trimmed JSON attestations with schema headers). The two-stage approach produces better *structure and metadata* but the rule logic can drift because the context is more abstract.

---

## Root Cause Analysis

### Why Attestation Training Produces More Accurate Rules

1. **Concrete Context Grounding**
   - Attestation training provides actual trimmed JSON from real attestations
   - The model sees the exact structure it needs to navigate
   - Example: Shows `{"tasks": [{"name": "build", "status": "Succeeded"}]}` directly

2. **Query-Specific Schema Headers**
   - Generates headers like:
     ```
     # Attestation Structure:
     # input.attestations[] → statement → predicate → buildConfig.tasks[]
     # Task fields: name, status, ref.bundle, ref.params[]...
     ```
   - This directly tells the model the navigation path

3. **Validated Code Templates**
   - Uses `RegoCodeGenerator` class with ~30 pre-validated patterns
   - Each pattern is syntactically correct and follows Rego idioms
   - Patterns cover: task checks, status checks, bundle validation, material queries, etc.

4. **Multiple Instruction Variations → Same Output**
   - Generates 3-5 variations of instructions that all map to identical code
   - Teaches the model that different phrasings should produce the same rule
   - Example: "Check if task X has status Y" / "Verify task X status equals Y" / "Deny if task X status is Y" → all produce same deny rule

### Why Two-Stage Rules Are Less Accurate

1. **Abstract Context**
   - Stage 1 outputs abstract schema paths like `.statement.predicate.buildConfig.tasks[]`
   - Model must mentally translate these to actual navigation code
   - No concrete JSON to ground understanding

2. **Inferred (Not Verified) Helpers**
   - `_generate_available_helpers()` uses regex to find `lib.X` patterns
   - Descriptions come from indexer but may be incomplete
   - Model doesn't see actual helper signatures/usage

3. **LLM-Generated Analysis Can Drift**
   - `_generate_analysis_llm()` asks an LLM to explain the rule
   - This can introduce variations that don't match the actual code logic
   - Rule-based fallback is more reliable but less descriptive

4. **One Output Per Rule** (until augmentation)
   - Without `--augment`, each rule produces only one training example
   - Less instruction variation means weaker generalization

---

## Improvement Plan

### Phase 1: Incorporate Attestation Training Strengths into Two-Stage

#### 1.1 Add Concrete Schema Examples to Stage 1 Output

**Current Stage 1 Output:**
```
ATTESTATION_SCHEMA:
- .statement.predicate.buildConfig.tasks[]
- .statement.predicate.buildConfig.tasks[].results[]
```

**Improved Stage 1 Output:**
```
ATTESTATION_SCHEMA:
- path: .statement.predicate.buildConfig.tasks[]
  example: {"name": "build-container", "status": "Succeeded", "ref": {...}}
  
- path: .statement.predicate.buildConfig.tasks[].results[]
  example: {"name": "IMAGE_DIGEST", "value": "sha256:abc123...", "type": "string"}

NAVIGATION_PATTERN:
  some att in input.attestations
  some task in att.statement.predicate.buildConfig.tasks
  some r in task.results
```

**Implementation:** Modify `_infer_attestation_schema()` to include:
- Mini JSON examples for each path
- A suggested navigation snippet (the `some X in` pattern)

#### 1.2 Add Helper Signatures to Stage 1 Output

**Current:**
```
AVAILABLE_HELPERS:
- name: lib.result_helper
```

**Improved:**
```
AVAILABLE_HELPERS:
- name: lib.result_helper(chain, terms)
  signature: result_helper(chain, terms) := {...}
  usage: result := lib.result_helper(rego.metadata.chain(), [task.name])
  description: Generates violation result with metadata from chain
```

**Implementation:** Enhance `_generate_available_helpers()` to pull:
- Full signature from `LibraryIndexer`
- A usage example from actual rules (store in indexer)
- Brief description

#### 1.3 Generate Navigation Code Hints

Add a new section to Stage 1 output:

```
NAVIGATION_HINTS:
- To iterate tasks: `some task in att.statement.predicate.buildConfig.tasks`
- To get results: `some r in task.results` (use 'r' not 'result' to avoid shadowing)
- To check membership: `x in {val1, val2, val3}`
- To use every (FOR ALL): `every task in tasks { task.status == "Succeeded" }`
```

**Implementation:** Create a `_generate_navigation_hints()` function that:
- Analyzes the rule code for patterns
- Extracts idiom examples
- Provides 3-5 key navigation patterns

### Phase 2: Improve Stage 2 Training Data Quality

#### 2.1 Use Validated Code Templates for Core Patterns

**Current:** Extracts rule code directly from files.

**Improved:** For common patterns, verify the extracted code matches known templates:

```python
VALIDATED_PATTERNS = {
    "task_iteration": "some att in lib.pipelinerun_attestations\nsome task in tekton.tasks(att)",
    "deny_result": "result := lib.result_helper(rego.metadata.chain(), [...])",
    "sbom_iteration": "some sbom in sbom.cyclonedx_sboms\nsome component in sbom.components",
}
```

Use these to:
1. Validate extracted code uses correct patterns
2. Provide consistent training signal
3. Catch drift in source policy files

#### 2.2 Multi-Variation Training for Stage 2

**Current:** With `--augment`, generates 4 instruction variations but same output.

**Improved:** Also vary the ANALYSIS section slightly while keeping code identical:

```python
ANALYSIS_VARIATIONS = [
    "- Iterates over pipeline tasks to find...",  # Technical
    "- Walks through each task and checks...",     # Conversational  
    "- For each task, validates that...",          # Formal
]
```

This teaches the model that different ways of explaining lead to same code.

#### 2.3 Instruction-to-Code Consistency Validation

Add validation that instruction complexity matches code complexity:

```python
def validate_instruction_code_match(instruction: str, code: str) -> bool:
    """Verify instruction and code complexity align."""
    # Count key elements
    instruction_entities = count_entities_in_instruction(instruction)
    code_iterations = count_some_clauses(code)
    
    # If instruction mentions 3 things, code should iterate ~3 things
    return abs(instruction_entities - code_iterations) <= 1
```

### Phase 3: Enhanced Training Signal

#### 3.1 Add Negative Examples

Train the model what NOT to do:

```json
{
  "instruction": "Check if task succeeded",
  "input": "...",
  "output": "INCORRECT - Missing 'some' iteration:\n```rego\n# WRONG: task.status == \"Succeeded\"\n```\n\nCORRECT:\n```rego\nsome att in input.attestations\nsome task in att.statement.predicate.buildConfig.tasks\ntask.status == \"Succeeded\"\n```"
}
```

**Caution:** Use sparingly (5-10% of dataset) to avoid confusing the model.

#### 3.2 Chain-of-Thought in Analysis

Make ANALYSIS more structured:

```
ANALYSIS:
1. DATA_SOURCE: PipelineRun attestations via lib.pipelinerun_attestations
2. ITERATION: For each attestation → for each task in buildConfig.tasks
3. CONDITION: task.status != "Succeeded" 
4. OUTPUT: lib.result_helper with task name as term
```

This explicitly shows the logic chain.

#### 3.3 Test-Driven Examples

For rules with tests, format as:

```
Given this test:
```rego
test_task_succeeded if {
    lib.assert_empty(deny) with input as _good_attestation
}
```

The rule must pass when task.status == "Succeeded".
```

### Phase 4: Training Process Improvements

#### 4.1 Curriculum Learning

Train in order of complexity:
1. **Epoch 1-2:** Simple rules (1 iteration, 1 condition)
2. **Epoch 3-4:** Medium rules (nested iteration, helpers)
3. **Epoch 5+:** Complex rules (multiple conditions, time-based, rule_data)

**Implementation:** Add complexity scoring to training data generator.

#### 4.2 Contrastive Learning

For each correct example, generate a "near miss":
- Correct: `some task in tekton.tasks(att)`
- Near miss: `some task in att.tasks` (wrong path)

Train the model to distinguish.

#### 4.3 Eval Split by Complexity

Don't just random split train/eval. Ensure:
- Eval has examples at all complexity levels
- Eval includes patterns not heavily represented in train
- Eval includes edge cases (empty arrays, missing fields)

---

## Implementation Priority

### High Priority (Do First)
1. **Add concrete examples to schema paths** - Biggest accuracy win
2. **Add navigation hints section** - Reduces navigation errors
3. **Multi-instruction variations (attestation style)** - Better generalization

### Medium Priority
4. **Enhanced helper signatures** - Reduces helper usage errors
5. **Chain-of-thought ANALYSIS** - Better reasoning
6. **Complexity-based eval split** - Better evaluation

### Lower Priority (Research)
7. Negative examples
8. Contrastive learning
9. Curriculum learning

---

## Concrete Code Changes

### File: `scripts/generate_two_stage_dataset.py`

#### Change 1: Enhance `_infer_attestation_schema()`

```python
def _infer_attestation_schema(self, rule: ExtractedRule) -> str:
    """Infer attestation schema paths from rule code WITH examples."""
    code = rule.get_complete_code()
    paths = []
    
    # ... existing pattern matching ...
    
    # NEW: Add concrete examples for each path
    SCHEMA_EXAMPLES = {
        ".statement.predicate.buildConfig.tasks[]": {
            "example": '{"name": "build", "status": "Succeeded", "ref": {...}}',
            "navigation": "some task in att.statement.predicate.buildConfig.tasks",
        },
        ".statement.predicate.buildConfig.tasks[].results[]": {
            "example": '{"name": "IMAGE_DIGEST", "value": "sha256:...", "type": "string"}',
            "navigation": "some r in task.results  # use 'r' to avoid shadowing 'result'",
        },
        # ... more examples ...
    }
    
    for path_desc in detected_paths:
        base_path = path_desc.split('(')[0].strip()  # Remove parenthetical
        if base_path in SCHEMA_EXAMPLES:
            ex = SCHEMA_EXAMPLES[base_path]
            paths.append(f"- path: {path_desc}")
            paths.append(f"  example: {ex['example']}")
            paths.append(f"  navigation: {ex['navigation']}")
        else:
            paths.append(f"- {path_desc}")
    
    return '\n'.join(paths)
```

#### Change 2: Add Navigation Hints

```python
def _generate_navigation_hints(self, rule: ExtractedRule) -> str:
    """Generate navigation pattern hints from rule code."""
    code = rule.get_complete_code()
    hints = []
    
    # Detect patterns used
    if 'lib.pipelinerun_attestations' in code:
        hints.append("- Attestation iteration: `some att in lib.pipelinerun_attestations`")
    
    if 'tekton.tasks(' in code:
        hints.append("- Task iteration: `some task in tekton.tasks(att)`")
    elif 'buildConfig.tasks' in code:
        hints.append("- Task iteration: `some task in att.statement.predicate.buildConfig.tasks`")
    
    if re.search(r'some\s+r\s+in.*results', code):
        hints.append("- Result iteration: `some r in task.results` (use 'r' not 'result')")
    
    if 'every ' in code:
        hints.append("- Universal check: `every item in collection { condition }`")
    
    if ' in {' in code:
        hints.append("- Membership: `value in {opt1, opt2, opt3}`")
    
    return '\n'.join(hints) if hints else ""
```

#### Change 3: Enhanced Stage1Example format

```python
def format_output(self) -> str:
    parts = []
    parts.append(f"ATTESTATION_SCHEMA:\n{self.attestation_schema}")
    parts.append(f"\nAVAILABLE_HELPERS:\n{self.available_helpers}")
    if self.rule_data_keys:
        parts.append(f"\nRULE_DATA_KEYS:\n{self.rule_data_keys}")
    
    # NEW: Add navigation hints
    if self.navigation_hints:
        parts.append(f"\nNAVIGATION_HINTS:\n{self.navigation_hints}")
    
    parts.append(f"\nSUGGESTED_PACKAGE: {self.suggested_package}")
    parts.append(f"SUGGESTED_RULE_TYPE: {self.suggested_rule_type}")
    return "\n".join(parts)
```

---

## Success Metrics

After implementing improvements, measure:

1. **Syntax Validity:** % of generated rules that pass `opa parse`
2. **Semantic Correctness:** % of rules that pass manual review
3. **Helper Accuracy:** % of rules that use correct helper signatures
4. **Navigation Correctness:** % of rules with correct path traversal
5. **Metadata Completeness:** % of rules with all required METADATA fields

**Target:** Achieve 90%+ syntax validity and 80%+ semantic correctness.

---

## Appendix: Key Patterns from Attestation Training to Preserve

These patterns from `generate_attestation_dataset.py` should be incorporated:

1. **RegoCodeGenerator patterns:** Pre-validated templates for common operations
2. **Query-specific schema headers:** Navigation hints based on query type
3. **Multiple instruction → same output:** 3-5 phrasings per rule
4. **Concrete JSON examples:** Trimmed attestations as context
5. **Validation checks:** `_validate_attestation_parsing()` ensures rules access correct paths

---

## Next Steps

1. Implement Phase 1 changes in `generate_two_stage_dataset.py`
2. Regenerate training data with `python scripts/generate_two_stage_dataset.py --augment`
3. Train new models and compare eval loss
4. Run manual accuracy assessment on 20 random prompts
5. Iterate based on error patterns

