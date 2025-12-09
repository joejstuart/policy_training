# Attestation Parsing Training Data Generation Plan

## Overview

This document outlines the plan for generating training data that teaches a model to write Rego policy rules that evaluate attestation JSON documents. Rego is a declarative policy language (inspired by Datalog) designed for policy evaluation and reasoning about structured data like JSON.

The model will learn to:
- Write Rego queries/assertions that navigate attestation JSON structures
- Create rules that evaluate conditions and make policy decisions about attestations
- Use Rego's declarative syntax to express "what should be true" rather than "how to check"
- Identify and access specific fields (e.g., task names, statuses, digests) using Rego
- Write proper Rego expressions that check for values, conditions, and violations in attestations

## Goals

1. **Rego Policy Rules**: Teach the model to write Rego rules/queries that evaluate attestations and make policy decisions
2. **Declarative Syntax**: Teach Rego's declarative approach - expressing "what should be true" rather than "how to check"
3. **Generalization**: Create examples that work for various instruction types, not just task names
4. **Attestation Navigation**: Teach Rego patterns for navigating nested JSON structures (arrays, objects, nested paths)
5. **Field Access & Evaluation**: Help the model write Rego that accesses fields, evaluates conditions, and makes assertions
6. **Multiple Formats**: Handle different attestation structures (SLSA v0.2, SLSA v1, in-toto)

## Training Data Structure

Each training example will follow this format (similar to existing `generate_dataset.py`):

```json
{
  "instruction": "In an attestation, check all tasks for a task named 'init'",
  "context": "<trimmed attestation JSON structure>",
  "output_code": "<Rego code that parses the attestation>",
  "task_type": "rego_attestation_parse",
  "source_file": "att1.json"
}
```

**Key Points:**
- `instruction`: Natural language description of what to find/check
- `context`: Trimmed attestation JSON (only relevant parts)
- `output_code`: Rego code that implements the parsing logic
- `task_type`: "rego_attestation_parse" (distinguishes from "implement" and "refactor")

## Output Format: Rego Code

**Key Decision**: The output format focuses on **Rego code generation** that parses attestations.

### Why Rego Code Output
1. **Primary Goal**: Teach the model to write Rego rules that parse attestations
2. **Practical Application**: Model learns to create actual policy rules
3. **Rego Patterns**: Teaches proper Rego syntax for JSON navigation
4. **Real-World Usage**: Matches how Rego is actually used for attestation parsing

### Output Format: Rego Code

The `output_code` field contains Rego code that implements the parsing logic:

**Example 1: Simple Task Check**
```rego
some att in input.attestations
some task in att.statement.predicate.buildConfig.tasks
task.name == "init"
```

**Example 2: Task Status Check**
```rego
some att in input.attestations
some task in att.statement.predicate.buildConfig.tasks
task.name == "init"
task.status == "Succeeded"
```

**Example 3: Subject Digest Access**
```rego
some att in input.attestations
some subject in att.statement.subject
subject.digest.sha256 == "f45b04f41083e21be9ecd4ae2e8e601d9b280eaaef10e78d48949088928c6de6"
```

### Rego Patterns to Teach

1. **Declarative Assertions**: Expressing what should be true, not how to check it
2. **Array Iteration**: `some task in att.statement.predicate.buildConfig.tasks` (using `some` keyword)
3. **Field Access**: `task.name`, `task.status`, `task.ref.bundle` (nested JSON navigation)
4. **Conditional Checks**: `task.name == "init"`, `task.status == "Succeeded"` (equality and comparison)
5. **Rule Bodies**: Multiple expressions combined with AND logic
6. **Set Comprehensions**: `task_names := {name | ...}` for collecting values
7. **Variable Bindings**: `name := task.name` for extracting values
8. **Multiple Conditions**: Combining checks (implicit AND in rule bodies)
9. **Optional Helper Libraries**: Can use `lib.task_in_pipelinerun(name)` but not required

**Key Rego Concepts** (from [OPA Policy Language docs](https://www.openpolicyagent.org/docs/policy-language)):
- Rego is declarative - focus on what queries should return, not how to execute
- Rule bodies are AND expressions: `expression-1 AND expression-2 AND ...`
- Rules can be understood as: `rule-name IS value IF body`
- Use `some` keyword for existential quantification (finding items in collections)
- Rego queries are assertions on data for making policy decisions

## Attestation Structure Patterns (from test files)

### SLSA v0.2 Format (Most Common)
- Tasks: `statement.predicate.buildConfig.tasks[]`
- Task fields: `name`, `status`, `ref`, `invocation`, `results`, `steps`, `finishedOn`, `startedOn`
- Materials: `statement.predicate.materials[]`
- Invocation: `statement.predicate.buildConfig.invocation.environment.annotations`, `labels`
- Build type: `statement.predicate.buildType` (e.g., "tekton.dev/v1/PipelineRun")

### SLSA v1 Format  
- Tasks: `statement.predicate.buildDefinition.resolvedDependencies[]` (base64 encoded TaskRun objects)
- Predicate type: `statement.predicateType == "https://slsa.dev/provenance/v1"`
- Build type: `statement.predicate.buildDefinition.buildType` (e.g., "https://tekton.dev/chains/v2/slsa-tekton")
- External parameters: `statement.predicate.buildDefinition.externalParameters.runSpec`
- Internal parameters: `statement.predicate.buildDefinition.internalParameters.labels`
- Tasks are base64-encoded JSON objects that need to be decoded to access fields

### Key Differences
- **SLSA v0.2**: Direct access to tasks array, simpler structure
- **SLSA v1**: Tasks are base64-encoded in `resolvedDependencies`, need decoding
- Both formats can appear in the same attestation set
- Helper functions in `lib/attestations.rego` handle both formats

### Common Paths
- `statement.subject[]` - subjects with `name` and `digest.sha256`
- `statement.predicate.buildConfig.tasks[i].name` - task name
- `statement.predicate.buildConfig.tasks[i].status` - task status
- `statement.predicate.buildConfig.tasks[i].ref.bundle` - task bundle reference
- `statement.predicate.buildConfig.tasks[i].invocation.parameters.X` - task parameters
- `statement.predicate.buildConfig.tasks[i].results[j].name` - result name
- `statement.predicate.buildConfig.tasks[i].results[j].value` - result value
- `statement.predicate.materials[i].uri` - material URI
- `statement.predicate.materials[i].digest.sha1` - material digest

## Instruction Categories

### 1. Task-Related Queries (Most Common)
- "Find all tasks in the attestation"
- "Check all tasks for a task named 'X'"
- "Get the status of task 'X'"
- "List all task names"
- "Find tasks that have status 'Succeeded'"
- "Get the bundle reference for task 'X'"
- "Find tasks with a specific bundle"
- "Get all task results"
- "Get the result named 'X' from task 'Y'"
- "Find tasks that finished before/after timestamp 'X'"

### 2. Subject-Related Queries
- "Get all subjects from the attestation"
- "Find the digest for subject 'X'"
- "List all subject names"
- "Get the SHA256 digest of the first subject"
- "Find subjects with name matching 'X'"

### 3. Predicate-Related Queries
- "Get the buildConfig from the predicate"
- "Find the invocation configSource for task 'X'"
- "Get the environment annotations"
- "Extract the repository URL from annotations"
- "Get all annotation keys that start with 'pipelinesascode'"
- "Find the value of annotation 'X'"
- "Get all labels from the environment"

### 4. Materials-Related Queries
- "Get all materials from the predicate"
- "Find materials with URI matching 'X'"
- "Get the digest for material 'X'"
- "Check if material exists for git repo 'X' and commit 'Y'"

### 5. Task Results Queries
- "Get all results from task 'X'"
- "Find the value of result 'Y' from task 'X'"
- "List all result names from all tasks"
- "Find tasks that produced result 'X'"

### 6. Task Reference Queries
- "Get the bundle reference for task 'X'"
- "Find tasks using bundle 'X'"
- "Check if task 'X' uses a trusted bundle"
- "Get the resolver type for task 'X'"

### 7. Conditional/Filtering Queries
- "Find tasks that finished after timestamp 'X'"
- "Get tasks with specific labels"
- "Find subjects with digest matching 'X'"
- "Find tasks with status 'Failed'"
- "Get tasks with parameter 'X' set to 'Y'"

## Implementation Plan

### Phase 1: JSON Structure Analysis
1. **Scan all JSON files** in the root directory
2. **Extract common patterns**:
   - Task structures and their locations (SLSA v0.2: `buildConfig.tasks`, SLSA v1: `buildDefinition.resolvedDependencies`)
   - Subject structures (`subject[]` with `name` and `digest.sha256`)
   - Predicate structures (`buildConfig`, `buildDefinition`, `materials`)
   - Common field names and their locations:
     - Task: `name`, `status`, `ref.bundle`, `ref.name`, `ref.kind`, `ref.resolver`, `ref.params`, `invocation.parameters`, `invocation.environment.annotations`, `invocation.environment.labels`, `results[]`, `steps[]`, `finishedOn`, `startedOn`
     - Subject: `name`, `digest.sha256`
     - Materials: `uri`, `digest.sha1`, `digest.sha256`
3. **Build a schema map** of different attestation types (SLSA v0.2, SLSA v1, in-toto)
4. **Identify helper patterns** from `lib/attestations.rego`:
   - `lib.tasks_from_pipelinerun` - how tasks are extracted
   - `lib.task_in_pipelinerun(name)` - finding tasks by name
   - `lib.task_results(task)` - getting results from tasks

### Phase 2: Instruction Template Generation
1. **Create instruction templates** for each category above
2. **Generate variations** of each template:
   - Different phrasings
   - Different field names
   - Different query types
3. **Parameterize templates** with actual values from JSON files

### Phase 3: Example Generation
For each JSON file and instruction template:
1. **Extract relevant data** from the JSON based on the instruction
2. **Trim the attestation** to only include relevant structure (see Trimming Strategy below)
3. **Generate the output** showing:
   - The JSON path to the data (e.g., `predicate.buildConfig.tasks[0].name`)
   - The field name that contains the requested information
   - The actual value(s) found
   - A brief explanation of the path
4. **Create multiple examples** per JSON file with different instructions

### Trimming Strategy (Critical for Large Attestations)

Attestations can be 7000+ lines. We need to trim them to only relevant parts while preserving structural context.

**Principles:**
1. **Keep structural path**: Always keep the full path from root to target (e.g., `statement.predicate.buildConfig.tasks`)
2. **Keep relevant data**: Include the target field(s) and a few related fields for context
3. **Remove irrelevant siblings**: Remove other tasks, subjects, materials if not needed
4. **Truncate large fields**: For very large fields (like `entryPoint` scripts), truncate or remove
5. **Preserve array structure**: Keep array structure even if removing most elements

**Trimming Rules:**

1. **For task queries:**
   - Keep: `statement.predicate.buildConfig.tasks` array structure
   - Keep: Relevant task(s) with key fields (`name`, `status`, `ref`, `results` if needed)
   - Remove: Other tasks in the array (or keep 1-2 for context)
   - Remove: Large fields like `steps[].entryPoint` (full scripts)
   - Remove: Verbose annotations (keep structure, remove most values)
   - Remove: `subject` array if not needed
   - Remove: `materials` if not needed

2. **For subject queries:**
   - Keep: `statement.subject` array
   - Keep: Relevant subject(s) with `name` and `digest`
   - Remove: Other subjects if not needed
   - Remove: `predicate.buildConfig.tasks` if not needed

3. **For material queries:**
   - Keep: `statement.predicate.materials` array
   - Keep: Relevant material(s) with `uri` and `digest`
   - Remove: Other materials if not needed
   - Remove: Tasks if not needed

4. **General trimming:**
   - Remove: `_type` and `predicateType` if not relevant to query
   - Truncate: Large string values (keep first 100 chars + "...")
   - Keep: At least one example of each structural level for context

**Example: Query "Find task named 'init'"**

**Before (7000+ lines):**
```json
{
  "_type": "...",
  "subject": [/* 4 subjects */],
  "predicate": {
    "buildConfig": {
      "tasks": [
        {/* task 0: init - huge with steps, annotations, etc */},
        {/* task 1: clone-repository - huge */},
        {/* task 2: prefetch-dependencies - huge */},
        {/* ... many more tasks ... */}
      ]
    }
  }
}
```

**After (trimmed, ~50 lines):**
```json
{
  "predicate": {
    "buildConfig": {
      "tasks": [
        {
          "name": "init",
          "status": "Succeeded",
          "ref": {
            "resolver": "bundles",
            "params": [
              {"name": "name", "value": "init"},
              {"name": "bundle", "value": "quay.io/konflux-ci/tekton-catalog/task-init:0.2@sha256:..."}
            ]
          },
          "results": [
            {"name": "build", "value": "true", "type": "string"}
          ],
          "startedOn": "2025-11-19T18:00:56Z",
          "finishedOn": "2025-11-19T18:01:02Z"
        }
      ]
    }
  }
}
```

**Benefits:**
- Reduces context size by 99%+ (7000 lines → 50 lines)
- Focuses model attention on relevant structure
- Faster training (fewer tokens to process)
- Lower costs
- Clearer learning signal

### Phase 4: Rego Code Generation

**Output Format: Rego Code**

For each instruction, generate Rego code that implements the parsing logic:

**Example: "Find task named 'init'"**
```rego
some att in input.attestations
some task in att.statement.predicate.buildConfig.tasks
task.name == "init"
```

**Example: "Get status of task 'init'"**
```rego
some att in input.attestations
some task in att.statement.predicate.buildConfig.tasks
task.name == "init"
task.status == "Succeeded"
```

**Example: "List all task names"**
```rego
task_names := {name |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    name := task.name
}
```

**Rego Code Guidelines:**
1. **Declarative style**: Express "what should be true" not "how to check"
2. **Use proper Rego syntax**: `some`, `in`, `==`, `:=`, etc.
3. **Iterate over arrays**: `some task in array` (existential quantification)
4. **Access nested fields**: `att.statement.predicate.buildConfig.tasks` (JSON navigation)
5. **Rule bodies as AND**: Multiple expressions implicitly combined with AND
6. **Handle both SLSA formats**: Check for v0.2 and v1 structures
7. **Optional helpers**: Can reference `lib.*` functions but not required
8. **Keep it simple**: Focus on clear, readable policy logic
9. **Validate**: Ensure generated Rego is syntactically correct (use `opa parse`)

### Phase 5: Training Data Format
Convert to the same format as existing training data:
- Use Qwen chat template format
- Include system prompt explaining the task (focus on Rego code generation)
- Format as instruction-response pairs
- Response contains Rego code that parses the attestation
- Save as JSONL for training

**System Prompt Example**:
```
You are an expert Rego policy rule writer. Rego is a declarative policy language for evaluating structured data like JSON. Given an instruction about what to find or check in an attestation, write Rego code that evaluates the attestation JSON structure and makes the requested policy decision. Use proper Rego syntax: declarative expressions, array iteration with 'some', field access, and condition checking. Express "what should be true" rather than "how to check it".
```

## Script Structure

### `generate_attestation_dataset.py`

**Main Components:**

1. **AttestationAnalyzer**
   - Scans JSON files
   - Extracts common structures
   - Builds schema knowledge

2. **InstructionGenerator**
   - Creates instruction templates
   - Generates variations
   - Parameterizes with real data

3. **PathResolver**
   - Given an instruction and JSON, finds the path
   - Identifies relevant fields
   - Extracts values

4. **AttestationTrimmer**
   - Trims large attestations to only relevant parts
   - Preserves structural context needed for path understanding
   - Removes irrelevant siblings and large fields
   - Applies trimming rules based on query type

5. **ExampleBuilder**
   - Combines instruction + trimmed JSON + output
   - Formats as training example
   - Validates examples
   - Ensures trimmed JSON still contains necessary structural context

6. **DatasetWriter**
   - Writes JSONL format
   - Splits train/eval
   - Generates summary statistics
   - Tracks average context size (before/after trimming)

## Example Training Examples

### Example 1: Simple Task Name Query
```
Instruction: "In an attestation, check all tasks for a task named 'init'"

Context: [Trimmed attestation JSON - statement.predicate.buildConfig.tasks array]

Output Code:
```rego
some att in input.attestations
some task in att.statement.predicate.buildConfig.tasks
task.name == "init"
```
```

### Example 2: List All Task Names
```
Instruction: "List all task names in the attestation"

Context: [Trimmed attestation JSON]

Output Code:
```rego
task_names := {name |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    name := task.name
}
```
```

### Example 3: Task Status Query
```
Instruction: "Get the status of task 'init'"

Context: [Trimmed attestation JSON]

Output Code:
```rego
some att in input.attestations
some task in att.statement.predicate.buildConfig.tasks
task.name == "init"
task.status == "Succeeded"
```
```

### Example 4: Task Results Query
```
Instruction: "Get all results from task 'git-clone'"

Context: [Trimmed attestation JSON]

Output Code:
```rego
task_results := {result |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "git-clone"
    some result in task.results
}
```
```

### Example 5: Subject Digest Query
```
Instruction: "Get the SHA256 digest of the first subject"

Context: [Trimmed attestation JSON]

Output Code:
```rego
first_subject_digest := digest if {
    some att in input.attestations
    att.statement.subject[0].digest.sha256 == digest
}
```
```

### Example 6: Task Bundle Reference Query
```
Instruction: "Get the bundle reference for task 'buildah'"

Context: [Trimmed attestation JSON]

Output Code:
```rego
bundle_ref := ref if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "buildah"
    task.ref.bundle == ref
}
```
```

### Example 7: Materials Query
```
Instruction: "Check if material exists for git repo 'git+https://gitforge/repo.git' and commit '9d25f3b6ab8cfba5d2d68dc8d062988534a63e87'"

Context: [Trimmed attestation JSON]

Output Code:
```rego
material_found if {
    some att in input.attestations
    some material in att.statement.predicate.materials
    material.uri == "git+https://gitforge/repo.git"
    material.digest.sha1 == "9d25f3b6ab8cfba5d2d68dc8d062988534a63e87"
}
```
```

## Configuration

- **Input**: All `*.json` files in repo root (attestation JSON documents)
- **Output**: `attestation_train.jsonl` and `attestation_eval.jsonl`
- **Train/Eval Split**: 90/10
- **Examples per JSON**: 5-15 (depending on complexity and available data)
- **Instruction Variations**: 3-5 per template
- **Format Support**: Both SLSA v0.2 and SLSA v1 formats
- **Context Size**: Trim attestations to only relevant parts (target: <200 lines per example, typically 20-100 lines)
- **Trimming**: Aggressive trimming while preserving structural context for path navigation

## Validation

1. **Path Validation**: Verify all paths exist in the trimmed JSON
2. **Value Validation**: Check that extracted values match the instruction
3. **Format Validation**: Ensure trimmed JSON and output are valid JSON
4. **Structural Validation**: Ensure trimmed JSON preserves necessary structural context
   - Path from root to target is complete
   - Array structures are preserved (even if empty or single-element)
   - Required parent objects exist
5. **Trimming Validation**: 
   - Verify trimmed JSON is significantly smaller (<10% of original size)
   - Ensure target data is still accessible
   - Check that structural context is sufficient for path understanding
6. **Coverage**: Ensure all instruction categories are represented

## Next Steps

1. Review and approve this plan
2. Implement `generate_attestation_dataset.py`
3. Test on a few JSON files
4. Generate full dataset
5. Validate dataset quality
6. Integrate with existing training pipeline

