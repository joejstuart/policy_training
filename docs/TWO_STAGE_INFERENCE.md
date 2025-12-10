# Two-Stage Inference for Rego Policy Generation

This document describes a two-stage approach to training and using a model for Rego policy rule generation. This approach separates **context inference** from **rule generation**, providing better transparency, debuggability, and flexibility.

---

## Overview

### The Problem

When given only natural language requirements, a model must:
1. Infer which attestation fields are relevant
2. Determine which library helpers to use
3. Identify configurable rule data keys
4. Generate correct Rego code

Doing all of this in a single step is error-prone and hard to debug.

### The Solution

Split the task into two stages:

```
STAGE 1: Context Inference
┌─────────────────────────────────────────────────────────────────┐
│  Input:  REQUIREMENTS (natural language)                        │
│  Output: ATTESTATION_SCHEMA + AVAILABLE_HELPERS + RULE_DATA_KEYS│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
STAGE 2: Rule Generation
┌─────────────────────────────────────────────────────────────────┐
│  Input:  REQUIREMENTS (repeated) + CONTEXT (from Stage 1)       │
│  Output: ANALYSIS + RULE (with METADATA) + TESTS (optional)     │
└─────────────────────────────────────────────────────────────────┘
```

**Key Point:** Stage 2 input = original requirements + Stage 1 output (**verbatim, not modified**). The requirements are repeated so the model has both the "what" (requirements) and the "how" (context).

### Training Data Consistency Rule

```
Stage 1 Training Example:
  input:  REQUIREMENTS
  output: ATTESTATION_SCHEMA + AVAILABLE_HELPERS + RULE_DATA_KEYS

Stage 2 Training Example:
  input:  REQUIREMENTS + CONTEXT  ← CONTEXT must be IDENTICAL to Stage 1 output
  output: ANALYSIS + RULE + TESTS (optional)
```

If Stage 1 outputs 5 schema paths, Stage 2 input must have those same 5 paths. Never truncate or modify.

### Important Constraints

> ⚠️ **Stage 1 must choose from known vocabularies:**
> - **ATTESTATION_SCHEMA** paths should come from known attestation families (SLSA Provenance, Tekton PipelineRun, CycloneDX SBOM, SPDX SBOM, etc.) — not invented JSON paths.
> - **AVAILABLE_HELPERS** must reference actual library functions (`lib.*`, `tekton.*`, `sbom.*`, `image.*`) — not hallucinated APIs.
> - **RULE_DATA_KEYS** correspond to entries accessed via `lib.rule_data("key")` in policy configuration.
>
> Stage 1 is selecting from a finite, known set — not inventing new libraries or arbitrary paths. If the model outputs a helper that doesn't exist, Stage 2 will generate broken code.

---

## Quick Start

### 1. Generate Training Data

```bash
# Generate Stage 1 and Stage 2 training data from existing policies
python scripts/generate_two_stage_dataset.py
```

Output:
- `data/training/two_stage/stage1_train.jsonl` (122 examples)
- `data/training/two_stage/stage1_eval.jsonl` (14 examples)
- `data/training/two_stage/stage2_train.jsonl` (122 examples)
- `data/training/two_stage/stage2_eval.jsonl` (14 examples)

### 2. Train Models

```bash
# Stage 1: Context Inference Model
python src/train_policy.py \
    --train-path data/training/two_stage/stage1_train.jsonl \
    --eval-path data/training/two_stage/stage1_eval.jsonl \
    --output-dir models/stage1-context-inference \
    --max-seq-len 1024 \
    --num-epochs 3

# Stage 2: Rule Generation Model
python src/train_policy.py \
    --train-path data/training/two_stage/stage2_train.jsonl \
    --eval-path data/training/two_stage/stage2_eval.jsonl \
    --output-dir models/stage2-rule-generation \
    --max-seq-len 2048 \
    --num-epochs 3
```

### 3. Run Inference

```bash
# Full two-stage pipeline
python src/infer_two_stage.py \
    --stage1-model models/stage1-context-inference \
    --stage2-model models/stage2-rule-generation \
    --instruction "Check that all pipeline tasks succeeded"

# Stage 1 only (get context)
python src/infer_two_stage.py \
    --stage1-model models/stage1-context-inference \
    --stage 1 \
    --instruction "Verify SBOM contains required packages"

# Interactive mode
python src/infer_two_stage.py \
    --stage1-model models/stage1-context-inference \
    --stage2-model models/stage2-rule-generation \
    --interactive
```

### 4. Larger Models (QLoRA on CUDA)

```bash
# For larger models like Qwen3-4B with 4-bit quantization
python src/train_policy.py \
    --train-path data/training/two_stage/stage1_train.jsonl \
    --eval-path data/training/two_stage/stage1_eval.jsonl \
    --output-dir models/stage1-context-inference-4b \
    --model-name Qwen/Qwen3-4B-Instruct \
    --use-4bit \
    --max-seq-len 1024
```

---

## Stage 1: Context Inference

### Purpose

Given natural language requirements, infer:
- **ATTESTATION_SCHEMA** — Which attestation fields the rule needs to access
- **AVAILABLE_HELPERS** — Which library functions should be used
- **RULE_DATA_KEYS** — Which configurable parameters are needed (accessed via `lib.rule_data()`)

### Section Naming Reference

| Section | Stage | Content |
|---------|-------|---------|
| `ATTESTATION_SCHEMA` | Stage 1 output | JSON paths from known attestation families |
| `AVAILABLE_HELPERS` | Stage 1 output | Library functions from known modules (`lib.*`, `tekton.*`, etc.) |
| `RULE_DATA_KEYS` | Stage 1 output | Keys for `lib.rule_data()` configuration |
| `ANALYSIS` | Stage 2 output | How to combine fields + helpers into logic |
| `RULE` | Stage 2 output | Rego code with METADATA comment block |
| `TESTS` | Stage 2 output | (Optional) Executable Rego test code with fixtures |

### Training Data Format

**Instruction:**
```
Analyze the requirements and identify the attestation schema, available helpers, 
and rule data keys needed to implement this Rego rule.
```

**Input:**
```
REQUIREMENTS:
- Package: trusted_task
- Rule type: warn
- Short name: current
- Purpose: Warn when trusted tasks are approaching their expiry date
- For each Tekton Task in the pipeline, check if it uses a trusted but 
  soon-to-expire reference.
- Use configurable warning window (default 30 days before expiry).
- Only warn if the task IS trusted but will expire within the warning window.
- Include the expiry date and latest available reference in the warning message.
```

**Output:**
```
ATTESTATION_SCHEMA:
- path: .statement.predicate.buildType
  description: Identifies attestation type. Filter for Tekton PipelineRun.
  type: string

- path: .statement.predicate.buildConfig.tasks[]
  description: Array of tasks executed in the pipeline.
  type: array

- path: .statement.predicate.buildConfig.tasks[].name
  description: The PipelineTask name from the Pipeline definition.
  type: string

- path: .statement.predicate.buildConfig.tasks[].ref.resolver
  description: Resolver type ("bundles" for OCI, "git" for git resolver).
  type: string

- path: .statement.predicate.buildConfig.tasks[].ref.params[]
  description: Resolver parameters including bundle reference or git revision.
  type: array

AVAILABLE_HELPERS:
- lib.tasks_from_pipelinerun: Returns all tasks from PipelineRun attestations
- tekton.task_ref(task): Parses task reference into {key, pinned_ref, pinned, kind}
- tekton.expiry_of(task): Returns expiry timestamp if within warning window
- tekton.pipeline_task_name(task): Returns the PipelineTask name
- tekton.task_name(task): Returns the Task name from labels
- tekton.latest_trusted_ref(task): Returns latest trusted ref for upgrade suggestion
- lib.result_helper_with_term(chain, args, term): Creates result with searchable term
- time.format(ns): Formats epoch nanoseconds as RFC3339 string

RULE_DATA_KEYS:
- task_expiry_warning_days:
    description: Days before expiry to start warning
    type: integer
    default: 30

- trusted_tasks:
    description: Map of task refs to trusted versions with expiry dates
    type: object
    schema: {"ref_key": [{"ref": "...", "effective_on": "...", "expires_on": "..."}]}
```

---

## Stage 2: Rule Generation

### Purpose

Given requirements plus inferred context, generate:
- **ANALYSIS** — How to combine fields and helpers into rule logic (not re-documenting what helpers do)
- **RULE** — The actual Rego code, including METADATA annotations (title, description, failure_msg, etc.)
- **TESTS** — (Optional) Executable Rego test code with fixtures

> **ANALYSIS vs Stage 1 context:** Stage 1's `AVAILABLE_HELPERS` documents what each helper *does*. Stage 2's `ANALYSIS` explains how to *combine* those helpers to implement the requirements — the field-to-logic mapping, conditional flow, and why certain helpers are chosen. Don't repeat helper documentation; focus on the synthesis.

> **TESTS are optional:** In early training phases, you may not have test code for all examples. The model should always emit `ANALYSIS:` and `RULE:`. Include `TESTS:` when available. Indicate optionality in training data by having some examples with tests and some without.

> **METADATA is part of RULE:** The Rego METADATA comment block (title, description, short_name, failure_msg, solution, collections, effective_on) is generated as part of Stage 2's `RULE:` output, not inferred by Stage 1.

> **Private helpers are part of RULE:** The Stage 2 `RULE:` output should include the **complete rule file**: package, imports, private helper functions (prefixed with `_`), and the main deny/warn rule. The existing training data (`train.jsonl`) follows this pattern and the model learns to create well-structured rules with appropriate helper functions.

### Training Data Format

**Instruction:**
```
Write a Rego rule that enforces the requirements below using the provided context.
```

**Input:**

> ⚠️ **Important:** Stage 2 input = original REQUIREMENTS + **exact** Stage 1 output. Do not modify or truncate the Stage 1 output.

```
REQUIREMENTS:
- Package: trusted_task
- Rule type: warn
- Short name: current
- Purpose: Warn when trusted tasks are approaching their expiry date
- For each Tekton Task in the pipeline, check if it uses a trusted but 
  soon-to-expire reference.
- Use configurable warning window (default 30 days before expiry).
- Only warn if the task IS trusted but will expire within the warning window.
- Include the expiry date and latest available reference in the warning message.

ATTESTATION_SCHEMA:
- path: .statement.predicate.buildType
  description: Identifies attestation type. Filter for Tekton PipelineRun.
  type: string

- path: .statement.predicate.buildConfig.tasks[]
  description: Array of tasks executed in the pipeline.
  type: array

- path: .statement.predicate.buildConfig.tasks[].name
  description: The PipelineTask name from the Pipeline definition.
  type: string

- path: .statement.predicate.buildConfig.tasks[].ref.resolver
  description: Resolver type ("bundles" for OCI, "git" for git resolver).
  type: string

- path: .statement.predicate.buildConfig.tasks[].ref.params[]
  description: Resolver parameters including bundle reference or git revision.
  type: array

AVAILABLE_HELPERS:
- lib.tasks_from_pipelinerun: Returns all tasks from PipelineRun attestations
- tekton.task_ref(task): Parses task reference into {key, pinned_ref, pinned, kind}
- tekton.expiry_of(task): Returns expiry timestamp if within warning window
- tekton.pipeline_task_name(task): Returns the PipelineTask name
- tekton.task_name(task): Returns the Task name from labels
- tekton.latest_trusted_ref(task): Returns latest trusted ref for upgrade suggestion
- lib.result_helper_with_term(chain, args, term): Creates result with searchable term
- time.format(ns): Formats epoch nanoseconds as RFC3339 string

RULE_DATA_KEYS:
- task_expiry_warning_days:
    description: Days before expiry to start warning
    type: integer
    default: 30

- trusted_tasks:
    description: Map of task refs to trusted versions with expiry dates
    type: object
    schema: {"ref_key": [{"ref": "...", "effective_on": "...", "expires_on": "..."}]}
```

Notice that the context sections (ATTESTATION_SCHEMA, AVAILABLE_HELPERS, RULE_DATA_KEYS) are **identical** to the Stage 1 output above. This is intentional — Stage 2 input is formed by concatenating:

```python
stage2_input = f"REQUIREMENTS:\n{requirements}\n\n{stage1_output}"
```

**Output:**
```
ANALYSIS:
- Field: .statement.predicate.buildConfig.tasks[]
  Access: via lib.tasks_from_pipelinerun
  Role: Iterate all pipeline tasks to check for expiring trusted references.

- Field: task reference
  Access: via tekton.task_ref(task) → {key, pinned_ref}
  Role: Provides the canonical key for looking up in trusted_tasks data.

- Field: trusted_tasks data
  Access: via tekton.expiry_of(task)
  Role: Determines if the task's current ref has an upcoming expiry.
  Logic: Returns expiry timestamp only if within warning window.

- Helper Selection:
  - lib.tasks_from_pipelinerun: Iterate all tasks
  - tekton.expiry_of: Check expiry window (encapsulates warning logic)
  - tekton.pipeline_task_name: For error message
  - tekton.task_name: For searchable term
  - tekton.latest_trusted_ref: Suggest upgrade path
  - time.format: Human-readable date
  - lib.result_helper_with_term: Generate result with metadata

RULE:
```rego
package trusted_task

import rego.v1

import data.lib
import data.lib.tekton

# METADATA
# title: Tasks using the latest versions
# description: >-
#   Check if all Tekton Tasks use the latest known Task reference.
# custom:
#   short_name: current
#   failure_msg: >-
#     A newer version of task %q exists. Please update before %s.
#     The current bundle is %q and the latest bundle ref is %q
#   solution: Update the Task reference to a newer version.
#   collections:
#   - redhat
#   - redhat_rpms
#   effective_on: 2024-05-07T00:00:00Z
#
warn contains result if {
    some task in lib.tasks_from_pipelinerun
    expiry := tekton.expiry_of(task)
    result := lib.result_helper_with_term(
        rego.metadata.chain(),
        [tekton.pipeline_task_name(task), time.format(expiry), _task_info(task), tekton.latest_trusted_ref(task)],
        tekton.task_name(task),
    )
}

_task_info(task) := info if {
    ref := tekton.task_ref(task)
    info := sprintf("%s@%s", [object.get(ref, "key", ""), object.get(ref, "pinned_ref", "")])
}
```

TESTS:
```rego
package trusted_task_test

import rego.v1

import data.lib
import data.trusted_task

test_no_warning_latest_version if {
    att := {"statement": {"predicate": {
        "buildType": lib.tekton_pipeline_run,
        "buildConfig": {"tasks": [trusted_bundle_pipeline_task]},
    }}}

    lib.assert_empty(trusted_task.warn) with input.attestations as [att]
        with data.trusted_tasks as trusted_tasks_data
}

test_outdated_warning if {
    att := {"statement": {"predicate": {
        "buildType": lib.tekton_pipeline_run,
        "buildConfig": {"tasks": [outdated_bundle_pipeline_task]},
    }}}

    expected := {{
        "code": "trusted_task.current",
        "msg": `A newer version of task "outdated-trusty-p" exists...`,
        "term": "trusty",
    }}

    lib.assert_equal_results(trusted_task.warn, expected) with input.attestations as [att]
        with data.trusted_tasks as trusted_tasks_data
}

# Fixtures
trusted_bundle_pipeline_task := {
    "name": "trusty-p",
    "ref": {"resolver": "bundles", "params": [
        {"name": "bundle", "value": "registry.local/trusty:1.0@sha256:digest"},
        {"name": "name", "value": "trusty"},
    ]},
}

outdated_bundle_pipeline_task := {
    "name": "outdated-trusty-p",
    "ref": {"resolver": "bundles", "params": [
        {"name": "bundle", "value": "registry.local/trusty:1.0@sha256:outdated-digest"},
        {"name": "name", "value": "trusty"},
    ]},
}

trusted_tasks_data := {
    "oci://registry.local/trusty:1.0": [
        {"ref": "sha256:digest", "effective_on": "2099-01-01T00:00:00Z"},
        {"ref": "sha256:outdated-digest", "effective_on": "2024-01-01T00:00:00Z", "expires_on": "2099-01-01T00:00:00Z"},
    ],
}
```
```

---

## Training Data Organization

```
data/training/
├── stage1_context_inference/
│   ├── train.jsonl          # Requirements → Context
│   └── eval.jsonl
├── stage2_rule_generation/
│   ├── train.jsonl          # Requirements + Context → Rule
│   └── eval.jsonl
└── combined/                 # Optional: for single-model training
    ├── train.jsonl
    └── eval.jsonl
```

### JSONL Format

**stage1_context_inference/train.jsonl:**
```jsonl
{"instruction": "Analyze the requirements...", "input": "REQUIREMENTS:\n- Package: trusted_task\n...", "output": "ATTESTATION_SCHEMA:\n..."}
{"instruction": "Analyze the requirements...", "input": "REQUIREMENTS:\n- Package: sbom_cyclonedx\n...", "output": "ATTESTATION_SCHEMA:\n..."}
```

**stage2_rule_generation/train.jsonl:**
```jsonl
{"instruction": "Write a Rego rule...", "input": "REQUIREMENTS:\n...\n\nATTESTATION_SCHEMA:\n...", "output": "ANALYSIS:\n...\n\nRULE:\n..."}
```

---

## Inference Pipeline

### Critical: Prompt Format Must Match Training

The inference prompt format **must exactly match** how the model was trained. If you train with:

```json
{"instruction": "Analyze the requirements...", "input": "REQUIREMENTS:\n...", "output": "..."}
```

Then at inference, you must format the prompt the same way. Common formats:

**Alpaca-style:**
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
```

**ChatML-style:**
```
<|im_start|>system
You are a Rego policy expert.<|im_end|>
<|im_start|>user
{instruction}

{input}<|im_end|>
<|im_start|>assistant
```

**Llama-style:**
```
[INST] {instruction}

{input} [/INST]
```

### Python Implementation

```python
from typing import Optional
import torch

class TwoStageRegoGenerator:
    """Two-stage Rego policy generator."""
    
    # Instruction templates
    STAGE1_INSTRUCTION = "Analyze the requirements and identify the attestation schema, available helpers, and rule data keys needed to implement this Rego rule."
    STAGE2_INSTRUCTION = "Write a Rego rule that enforces the requirements below using the provided context."
    
    def __init__(self, model, tokenizer, prompt_template: str = "alpaca"):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def format_prompt(self, instruction: str, input_text: str) -> str:
        """Format prompt to match training format."""
        if self.prompt_template == "alpaca":
            return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
        elif self.prompt_template == "chatml":
            return f"""<|im_start|>user
{instruction}

{input_text}<|im_end|>
<|im_start|>assistant
"""
        elif self.prompt_template == "llama":
            return f"[INST] {instruction}\n\n{input_text} [/INST]"
        else:
            # Simple concatenation (use if model was trained without special format)
            return f"{instruction}\n\n{input_text}\n\n"
    
    def generate(self, instruction: str, input_text: str, max_tokens: int = 2048) -> str:
        """Generate text from instruction + input."""
        prompt = self.format_prompt(instruction, input_text)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        # Track input length to slice it off from output
        input_length = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode only the NEW tokens (exclude input)
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def infer_context(self, requirements: str) -> str:
        """Stage 1: Infer context from requirements."""
        input_text = f"REQUIREMENTS:\n{requirements}"
        return self.generate(self.STAGE1_INSTRUCTION, input_text)
    
    def generate_rule(
        self, 
        requirements: str, 
        context: Optional[str] = None,
        test_cases: Optional[str] = None
    ) -> dict:
        """
        Generate a Rego rule.
        
        Args:
            requirements: Natural language requirements
            context: Optional pre-defined context. If None, Stage 1 runs first.
            test_cases: Optional specific test cases to include
        
        Returns:
            dict with 'context' (inferred or provided) and 'output' (rule)
        """
        # Stage 1: Infer context if not provided
        if context is None:
            context = self.infer_context(requirements)
            
            # Validate Stage 1 output has expected sections
            if not self._validate_context(context):
                raise ValueError(f"Stage 1 produced invalid context:\n{context[:500]}...")
        
        # Build Stage 2 input: Requirements + Context
        input_text = f"REQUIREMENTS:\n{requirements}\n\n{context}"
        
        if test_cases:
            input_text += f"\n\nTEST_CASES:\n{test_cases}"
        
        # Stage 2: Generate rule
        output = self.generate(self.STAGE2_INSTRUCTION, input_text)
        
        return {
            "context": context,
            "output": output
        }
    
    def _validate_context(self, context: str) -> bool:
        """Validate that Stage 1 output contains expected sections."""
        required_sections = ["ATTESTATION_SCHEMA:", "AVAILABLE_HELPERS:"]
        return all(section in context for section in required_sections)


# Usage Example
generator = TwoStageRegoGenerator(model, tokenizer, prompt_template="alpaca")

# Option 1: Full two-stage (requirements only)
result = generator.generate_rule("""
- Package: trusted_task
- Rule type: warn
- Purpose: Warn when tasks are expiring soon
- Check all pipeline tasks for expiring references
- Use 30-day warning window
""")

print("=== Inferred Context ===")
print(result["context"])
print("\n=== Generated Rule ===")
print(result["output"])

# Option 2: Provide context, skip Stage 1
result = generator.generate_rule(
    requirements="- Package: labels\n- Deny if required labels are missing",
    context="""ATTESTATION_SCHEMA:
- path: .image.config.Labels
  description: Image labels

AVAILABLE_HELPERS:
- lib.result_helper: Creates result object
- lib.rule_data: Gets rule data""",
)
```

### Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| **Prompt format mismatch** | Use exact same format as training data |
| **Including input in output** | Slice off input tokens before decoding |
| **Missing EOS token** | Set `eos_token_id` in generate() |
| **GPU memory issues** | Use `torch.no_grad()` during inference |
| **Long context truncation** | Set appropriate `max_length` in tokenizer |

### CLI Tool

```python
#!/usr/bin/env python3
"""CLI for two-stage Rego generation."""

import argparse
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path: str, prompt_template: str = "alpaca"):
    """Load model and create generator."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
    )
    return TwoStageRegoGenerator(model, tokenizer, prompt_template)

def main():
    parser = argparse.ArgumentParser(description="Generate Rego policies")
    parser.add_argument("--model", "-m", type=str, required=True, help="Model path")
    parser.add_argument("--requirements", "-r", type=str, required=True, 
                        help="Requirements text or file path")
    parser.add_argument("--context", "-c", type=str, help="Optional context file")
    parser.add_argument("--stage", choices=["1", "2", "both"], default="both",
                        help="Which stage(s) to run")
    parser.add_argument("--output", "-o", type=str, help="Output file")
    parser.add_argument("--prompt-template", type=str, default="alpaca",
                        choices=["alpaca", "chatml", "llama", "simple"],
                        help="Prompt template format")
    
    args = parser.parse_args()
    
    # Load model
    generator = load_model(args.model, args.prompt_template)
    
    # Load requirements (from file or direct text)
    req_path = Path(args.requirements)
    requirements = req_path.read_text() if req_path.exists() else args.requirements
    
    # Load context if provided
    context = None
    if args.context:
        ctx_path = Path(args.context)
        context = ctx_path.read_text() if ctx_path.exists() else args.context
    
    # Execute requested stage(s)
    if args.stage == "1":
        output = generator.infer_context(requirements)
    elif args.stage == "2":
        if not context:
            raise ValueError("Stage 2 requires --context")
        result = generator.generate_rule(requirements, context=context)
        output = result["output"]
    else:
        result = generator.generate_rule(requirements, context=context)
        output = json.dumps({
            "context": result["context"],
            "output": result["output"]
        }, indent=2)
    
    # Output results
    if args.output:
        Path(args.output).write_text(output)
        print(f"Written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# Full two-stage generation
python generate_rego.py -m ./my-model -r requirements.txt -o output.json

# Stage 1 only (infer context)
python generate_rego.py -m ./my-model -r requirements.txt --stage 1 -o context.txt

# Stage 2 only (with provided context)
python generate_rego.py -m ./my-model -r requirements.txt -c context.txt --stage 2
```

---

## Generating Training Data

### Important: Avoiding Data Leakage

When generating Stage 1 and Stage 2 training data from the same source examples:

1. **Keep paired examples in the same split** — If example X produces Stage1_X and Stage2_X, both should be in train or both in eval. Never split them.

2. **Shuffle before splitting** — Shuffle the source examples first, then generate both stages from each example.

```python
# CORRECT: Split first, then generate stages
train_sources, eval_sources = train_test_split(all_sources, test_size=0.1)

train_stage1 = [create_stage1(ex) for ex in train_sources]
train_stage2 = [create_stage2(ex) for ex in train_sources]
eval_stage1 = [create_stage1(ex) for ex in eval_sources]
eval_stage2 = [create_stage2(ex) for ex in eval_sources]

# WRONG: Generate then split (causes leakage)
# all_stage1 = [create_stage1(ex) for ex in all_sources]
# train_stage1, eval_stage1 = train_test_split(all_stage1)  # BAD!
```

### Deriving Stage 1 Data from Full Examples

If you have complete training examples (with all sections), you can automatically generate Stage 1 training data:

```python
import json
import re
from pathlib import Path

def parse_sections(text: str) -> dict:
    """Parse structured text into sections."""
    sections = {}
    current = None
    buffer = []
    
    section_headers = [
        "REQUIREMENTS:", "ATTESTATION_SCHEMA:", "AVAILABLE_HELPERS:",
        "RULE_DATA_KEYS:", "CONVENTIONS:",
        "ANALYSIS:", "RULE:", "TESTS:"
    ]
    
    for line in text.split('\n'):
        header_match = None
        for header in section_headers:
            if line.strip().startswith(header):
                header_match = header.rstrip(':').lower().replace(' ', '_')
                break
        
        if header_match:
            if current:
                sections[current] = '\n'.join(buffer).strip()
            current = header_match
            buffer = []
        else:
            buffer.append(line)
    
    if current:
        sections[current] = '\n'.join(buffer).strip()
    
    return sections


def create_stage1_example(full_example: dict) -> dict:
    """Create Stage 1 training example from a full example."""
    
    input_sections = parse_sections(full_example["input"])
    
    # Stage 1 input: just requirements
    stage1_input = f"REQUIREMENTS:\n{input_sections.get('requirements', '')}"
    
    # Stage 1 output: inferred context
    output_parts = []
    
    if "attestation_schema" in input_sections:
        output_parts.append(f"ATTESTATION_SCHEMA:\n{input_sections['attestation_schema']}")
    
    if "available_helpers" in input_sections:
        output_parts.append(f"AVAILABLE_HELPERS:\n{input_sections['available_helpers']}")
    
    if "rule_data_keys" in input_sections:
        output_parts.append(f"RULE_DATA_KEYS:\n{input_sections['rule_data_keys']}")
    
    return {
        "instruction": "Analyze the requirements and identify the attestation schema, available helpers, and rule data keys needed to implement this Rego rule.",
        "input": stage1_input,
        "output": "\n\n".join(output_parts)
    }


def create_stage2_example(full_example: dict) -> dict:
    """Create Stage 2 training example (same as original format)."""
    return {
        "instruction": "Write a Rego rule that enforces the requirements below using the provided context.",
        "input": full_example["input"],
        "output": full_example["output"]
    }


def summarize_test_cases(test_cases: str) -> str:
    """Convert detailed test cases to summaries."""
    lines = []
    current_name = None
    current_result = None
    
    for line in test_cases.split('\n'):
        if 'name:' in line.lower():
            if current_name and current_result:
                lines.append(f"- {current_name} → {current_result}")
            current_name = line.split(':', 1)[1].strip()
            current_result = None
        elif 'expected_result' in line.lower():
            current_result = line.split('=')[1].strip()
    
    if current_name and current_result:
        lines.append(f"- {current_name} → {current_result}")
    
    return '\n'.join(lines) if lines else "1) valid input → pass\n2) invalid input → deny"


def generate_training_data(full_examples_path: str, output_dir: str):
    """Generate Stage 1 and Stage 2 training data from full examples."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    stage1_examples = []
    stage2_examples = []
    
    with open(full_examples_path, 'r') as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                stage1_examples.append(create_stage1_example(example))
                stage2_examples.append(create_stage2_example(example))
    
    # Write Stage 1 data
    stage1_path = output_path / "stage1_context_inference"
    stage1_path.mkdir(exist_ok=True)
    with open(stage1_path / "train.jsonl", 'w') as f:
        for ex in stage1_examples:
            f.write(json.dumps(ex) + '\n')
    
    # Write Stage 2 data
    stage2_path = output_path / "stage2_rule_generation"
    stage2_path.mkdir(exist_ok=True)
    with open(stage2_path / "train.jsonl", 'w') as f:
        for ex in stage2_examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"Generated {len(stage1_examples)} Stage 1 examples")
    print(f"Generated {len(stage2_examples)} Stage 2 examples")


# Usage
if __name__ == "__main__":
    generate_training_data(
        "data/training/combined/train.jsonl",
        "data/training/"
    )
```

---

## Benefits of Two-Stage Approach

| Benefit | Description |
|---------|-------------|
| **Transparency** | You can inspect the inferred context before rule generation |
| **Debuggability** | If the rule is wrong, you can identify if it's a context or generation problem |
| **Flexibility** | Users can provide partial context and let the model fill in gaps |
| **Overrides** | You can correct the inferred context before Stage 2 |
| **Modularity** | Can use different models for each stage |
| **Smaller Models** | Each stage is simpler, potentially works with smaller models |

---

## Trade-offs

| Consideration | Impact |
|---------------|--------|
| **Latency** | Two model calls instead of one |
| **Token Usage** | Context appears in both stages (some duplication) |
| **Complexity** | More infrastructure to maintain |
| **Error Propagation** | Stage 1 errors affect Stage 2 |

---

## Training Approaches

### Option A: Two Separate Models

Train two specialized models:
- **Model A:** Fine-tuned for context inference (Stage 1)
- **Model B:** Fine-tuned for rule generation (Stage 2)

**Pros:** Each model is specialized, potentially better at its task.  
**Cons:** Two models to maintain and deploy.

### Option B: Single Model with Task Prefixes

Train one model on both tasks using instruction prefixes:

```jsonl
{"instruction": "[CONTEXT] Analyze the requirements...", "input": "...", "output": "..."}
{"instruction": "[GENERATE] Write a Rego rule...", "input": "...", "output": "..."}
```

**Pros:** Single model to deploy.  
**Cons:** Model capacity split between tasks.

### Option C: Single Model, Natural Instructions

Train on both tasks without prefixes, relying on instruction text:

```jsonl
{"instruction": "Analyze the requirements and identify...", "input": "...", "output": "..."}
{"instruction": "Write a Rego rule that enforces...", "input": "...", "output": "..."}
```

**Pros:** More natural, flexible prompting.  
**Cons:** May need more training data for reliable task switching.

---

## Example: Complete Workflow

```python
# 1. User provides requirements
requirements = """
- Package: labels
- Rule type: deny
- Purpose: Ensure required labels are present on container images
- Check that all labels in required_labels rule data exist
- Include label description in error message
"""

# 2. Stage 1: Infer context
context = generator.infer_context(requirements)
print("Inferred Context:")
print(context)

# Output:
# ATTESTATION_SCHEMA:
# - path: .image.config.Labels
#   description: Map of label names to values from image config
# 
# AVAILABLE_HELPERS:
# - lib.rule_data(key): Gets rule data by key
# - lib.result_helper_with_term(chain, args, term): Creates result
# - ec.oci.image_manifest(ref): Gets image manifest
# - ec.oci.blob(ref): Gets blob content
# 
# RULE_DATA_KEYS:
# - required_labels:
#     description: List of required label objects with name and description
#     type: array

# 3. (Optional) User reviews/modifies context
context = context.replace("ec.oci.blob", "json.unmarshal(ec.oci.blob(ref))")

# 4. Stage 2: Generate rule
result = generator.generate_rule(requirements, context=context)
print("Generated Rule:")
print(result["output"])
```

---

## Evaluation Metrics

### Stage 1 Evaluation

| Metric | Description |
|--------|-------------|
| **Schema Recall** | % of required attestation fields correctly identified |
| **Schema Precision** | % of identified fields that are actually needed |
| **Helper Accuracy** | % of helpers correctly identified |
| **Hallucination Rate** | % of outputs containing non-existent helpers or fields |

```python
def evaluate_stage1(predicted: str, expected: str) -> dict:
    """Evaluate Stage 1 output."""
    pred_sections = parse_sections(predicted)
    exp_sections = parse_sections(expected)
    
    # Extract schema paths
    pred_paths = extract_paths(pred_sections.get("attestation_schema", ""))
    exp_paths = extract_paths(exp_sections.get("attestation_schema", ""))
    
    # Calculate metrics
    recall = len(pred_paths & exp_paths) / len(exp_paths) if exp_paths else 0
    precision = len(pred_paths & exp_paths) / len(pred_paths) if pred_paths else 0
    
    return {"schema_recall": recall, "schema_precision": precision}
```

### Stage 2 Evaluation

| Metric | Description |
|--------|-------------|
| **Syntax Valid** | Does the generated Rego compile without errors? |
| **Tests Pass** | Do generated tests pass when run? |
| **Rule Coverage** | Are all test cases handled by the rule? |
| **Style Compliance** | Does code follow project conventions? |

```python
def evaluate_stage2(generated_rule: str, generated_tests: str) -> dict:
    """Evaluate Stage 2 output."""
    import subprocess
    
    # Write to temp files
    # ... 
    
    # Check syntax
    syntax_result = subprocess.run(
        ["opa", "check", rule_file],
        capture_output=True
    )
    syntax_valid = syntax_result.returncode == 0
    
    # Run tests
    test_result = subprocess.run(
        ["opa", "test", rule_file, test_file, "-v"],
        capture_output=True
    )
    tests_pass = test_result.returncode == 0
    
    return {"syntax_valid": syntax_valid, "tests_pass": tests_pass}
```

### End-to-End Evaluation

For the full pipeline, measure:
1. **Stage 1 accuracy** (context quality)
2. **Stage 2 accuracy given perfect context** (generation quality)
3. **End-to-end accuracy** (full pipeline quality)

This helps identify whether failures are due to context inference or rule generation.

---

## Summary

The two-stage approach provides:

1. **Stage 1** — Context inference from requirements
2. **Stage 2** — Rule generation from requirements + context

This separation enables inspection, debugging, and human-in-the-loop workflows while maintaining the ability to run fully automatically when context inference is reliable.

### Recommended Adoption Path

1. **Start with Stage 2 only** — Manually provide context, validate rule generation quality
2. **Add Stage 1 with human review** — Generate context, let humans verify before Stage 2
3. **Full automation** — Once both stages are reliable, run end-to-end

### Key Engineering Considerations

- **Prompt format must match training exactly**
- **Slice input tokens from generation output**
- **Validate Stage 1 output before Stage 2**
- **Keep paired examples in same train/eval split**
- **Measure each stage independently for debugging**

