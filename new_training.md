# Model Training Design: Rego Policy Rule Generation from Attestations

This document describes how to structure training data for a model that can generate Rego policy rules based on:

- Natural-language requirements
- Relevant fields extracted from SLSA/Tekton attestations
- Canonical test cases demonstrating the rule's expected behavior

It also covers when and why to use analysis sections, and how to design inputs when attestations are very large.

---

## 🧭 1. Overview of the Task

When writing a Rego policy rule, the model must:

1. Interpret natural-language requirements
2. Select the relevant fields from an attestation
3. Understand how those fields determine pass/warn/deny behavior
4. Translate that logic into correct Rego code
5. Use appropriate library helpers rather than reimplementing common patterns
6. Optionally produce test rules that cover all relevant cases

This is a multi-step reasoning task, and the training format should reflect that.

---

## 🔑 2. Training Example Structure

Each training example will consist of:

| Field | Description |
|-------|-------------|
| `instruction` | High-level request |
| `input` | Requirements, schema, helpers, rule data keys, and test cases |
| `output` | Structured into: **ANALYSIS**, **RULE**, and optionally **TESTS** |

### Output Sections

- **ANALYSIS:** — mapping of attestation fields → validation logic
- **RULE:** — the actual Rego rule
- **TESTS:** (optional) — full test coverage for the rule

### Example High-Level Schema

```json
{
  "instruction": "...",
  "input": "...",
  "output": "ANALYSIS:\n...\n\nRULE:\n...\n\nTESTS:\n...\n"
}
```

> **Note:** Only the content of `output` is structured; the dataset still uses the standard single-target-field format.

---

## 🧱 3. Input Structure (Critical)

The model's input should contain up to six major parts:

### A. REQUIREMENTS (Natural Language)

Describe exactly what the rule must validate, including metadata about the rule itself.

**Example:**

```
REQUIREMENTS:
- Package: trusted_task
- Rule type: warn
- Purpose: Check for tasks expiring soon
- Verify each Tekton Task used in a build is from the trusted list.
- If a task is within N days of expiry → emit warn.
- If expiry date is in the past → emit deny (handled by separate rule).
- Ignore tasks with type "source_image".
```

This is the **specification**. Key metadata to include:

| Field | Description |
|-------|-------------|
| Package | The Rego package name (e.g., `trusted_task`, `sbom_spdx`) |
| Rule type | `deny`, `warn`, or helper function |
| Purpose | One-line description for documentation |

### B. ATTESTATION_SCHEMA (Relevant Fields Only)

> ⚠️ Attestations are huge.  
> **Do NOT feed raw attestations.**  
> Feed only the fields the rule cares about, with paths and short descriptions.

**Example:**

```yaml
ATTESTATION_SCHEMA:
- path: .statement.predicate.buildConfig.tasks[].ref.name
  description: The Tekton Task name used in the pipeline step.
  type: string

- path: .statement.predicate.buildConfig.tasks[].ref.bundle
  description: The OCI bundle reference for the task (e.g., oci://registry/task:tag@sha256:...).
  type: string

- path: .statement.predicate.buildConfig.tasks[].ref.kind
  description: Reference type - "task" for OCI bundles, "git" for git resolvers.
  type: string
  values: ["task", "git"]

- path: .statement.predicate.buildType
  description: Identifies attestation type. Filter for "https://tekton.dev/chains/v2/slsa".
  type: string
```

This teaches the model **what the data looks like**, without noise.

### C. AVAILABLE_HELPERS (Library Functions)

Rego rules should use existing library abstractions rather than reimplementing common patterns. List the helpers relevant to this rule.

**Example:**

```yaml
AVAILABLE_HELPERS:
- name: lib.pipelinerun_attestations
  description: Returns attestations matching Tekton PipelineRun buildType.
  returns: set of attestation objects

- name: tekton.tasks(attestation)
  description: Extracts all tasks from a PipelineRun attestation.
  returns: array of task objects

- name: tekton.task_ref(task)
  description: Returns task reference info.
  returns: object with {key, pinned_ref, pinned, kind}

- name: tekton.pipeline_task_name(task)
  description: Returns the pipeline task name (the name in the Pipeline definition).
  returns: string

- name: lib.result_helper_with_term(chain, args, term)
  description: Creates a policy result with a searchable term for filtering.
  params:
    - chain: rego.metadata.chain() - provides rule metadata
    - args: array of values to interpolate into the error message
    - term: searchable identifier for this specific violation
  returns: result object
```

> **Why this matters:** The model must learn to use `tekton.tasks()` rather than manually navigating `.statement.predicate.buildConfig.tasks[]`. This reduces errors and maintains consistency with existing policies.

### D. RULE_DATA_KEYS (Configurable Parameters)

Many rules use `lib.rule_data("key")` for configurable thresholds or lists. Document which keys this rule uses.

**Example:**

```yaml
RULE_DATA_KEYS:
- key: task_expiry_warning_days
  description: Number of days before task expiry to start warning.
  type: integer
  default: 30

- key: trusted_tasks
  description: List of trusted task definitions with refs and expiry dates.
  type: array
  schema: |
    [
      {
        "ref": "oci://registry.local/task@sha256:...",
        "effective_on": "2024-01-01T00:00:00Z",
        "expires_on": "2025-01-01T00:00:00Z"
      }
    ]
```

### E. TEST_CASES (Canonical Examples)

Demonstrate how fields → decisions. **Include both positive and negative cases.**

**Example:**

```
TEST_CASES:

1) name: trusted task - no warning
   description: A trusted task with expiry far in the future should pass.
   .statement.predicate.buildConfig.tasks[0].ref.name = "buildah"
   .statement.predicate.buildConfig.tasks[0].ref.bundle = "oci://registry.local/buildah:1.0@sha256:abc123"
   trusted_tasks contains matching ref with expires_on = "2099-01-01T00:00:00Z"
   expected_result = pass
   expected_violations = 0

2) name: untrusted task
   description: A task not in the trusted list should be denied.
   .statement.predicate.buildConfig.tasks[0].ref.name = "unknown-task"
   .statement.predicate.buildConfig.tasks[0].ref.bundle = "oci://registry.local/unknown:1.0@sha256:def456"
   trusted_tasks does NOT contain matching ref
   expected_result = deny
   expected_code = "trusted_task.untrusted"
   expected_msg_contains = "not found in trusted task list"

3) name: expiring soon
   description: A trusted task within warning window should warn.
   .statement.predicate.buildConfig.tasks[0].ref.name = "buildah"
   .statement.predicate.buildConfig.tasks[0].ref.bundle = "oci://registry.local/buildah:1.0@sha256:abc123"
   trusted_tasks contains matching ref with expires_on = "2025-02-01T00:00:00Z"
   current_time = "2025-01-15T00:00:00Z"
   task_expiry_warning_days = 30
   expected_result = warn
   expected_code = "trusted_task.current"

4) name: empty tasks array
   description: Edge case - attestation with no tasks should not error.
   .statement.predicate.buildConfig.tasks = []
   expected_result = pass
   expected_violations = 0

5) name: missing ref field
   description: Edge case - task without ref should be handled gracefully.
   .statement.predicate.buildConfig.tasks[0].ref = null
   expected_result = deny
   expected_code = "trusted_task.untrusted"
```

This is where the model learns **how the logic should behave**, including:
- ✅ Pass cases (what does NOT trigger the rule)
- ❌ Deny/warn cases (what triggers violations)
- ⚠️ Edge cases (empty arrays, missing fields, malformed data)

### F. CONVENTIONS (Optional)

For complex rules, explicitly state Rego conventions to follow.

```
CONVENTIONS:
- Use rego.v1 import for modern Rego syntax
- Use `some x in collection` for iteration (not `x := collection[_]`)
- Prefix private helpers with underscore (e.g., `_task_info`)
- Use rego.metadata.chain() for all result helpers
- Follow existing package patterns for deny/warn rule naming
```

---

## 🧠 4. Why Include an ANALYSIS Section in the Output?

This particular task benefits heavily from a structured `ANALYSIS:` section because:

### ✔️ It Forces Deliberate Reasoning

The model must explicitly identify:

- Which attestation fields matter
- How each field is used in validation
- Why certain conditions lead to warn/deny/pass
- Which library helpers to use and why

This dramatically reduces hallucination and incorrect rule logic.

### ✔️ It Creates an Inspectable Artifact

You (or an agent) can read the analysis and verify:

- Did the model pick the correct fields?
- Does the logic match the requirements?
- Are the cases fully covered?
- Are the right helpers being used?

This gives **transparency** and **debuggability**.

### ✔️ It Teaches Field-to-Logic Mapping

The model learns:

> "Given this attestation structure, here's how policy logic should be built."

This becomes reusable for future policy rules.

### ✔️ It Supports Optional Automated Pipelines

Agents or validators can parse `ANALYSIS:` to:

- Automatically generate documentation
- Compare rule behavior against requirements
- Confirm the rule is complete

---

## 🧩 5. Output Structure

A recommended output structure for this task:

```
ANALYSIS:
- Field: .statement.predicate.buildConfig.tasks[]
  Access: via tekton.tasks(attestation)
  Role: Iterates over all pipeline tasks to validate.

- Field: task ref bundle
  Access: via tekton.task_ref(task).key and tekton.task_ref(task).pinned_ref
  Role: Identifies the specific task version used.
  Validation:
    - Must match an entry in trusted_tasks rule data.
    - If no match → deny with code "trusted_task.untrusted".

- Field: trusted_tasks[].expires_on
  Access: via lib.rule_data("trusted_tasks")
  Role: Determines expiration behavior.
  Validation:
    - If expires_on < now → deny (expired).
    - If now <= expires_on < now + warning_days → warn.

- Helper Selection:
  - Use tekton.tasks() to iterate tasks (handles attestation structure).
  - Use lib.result_helper_with_term() to include task name as searchable term.

RULE:
```

```rego
package trusted_task

import rego.v1

import data.lib
import data.lib.tekton

# Warn if a trusted task is approaching expiry
warn contains result if {
    some attestation in lib.pipelinerun_attestations
    some task in tekton.tasks(attestation)

    ref := tekton.task_ref(task)
    trusted := _matching_trusted_task(ref)

    _is_expiring_soon(trusted)

    result := lib.result_helper_with_term(
        rego.metadata.chain(),
        [tekton.pipeline_task_name(task), trusted.expires_on],
        tekton.task_name(task),
    )
}

_matching_trusted_task(ref) := trusted if {
    some trusted in lib.rule_data("trusted_tasks")
    trusted.ref == sprintf("%s@%s", [ref.key, ref.pinned_ref])
}

_is_expiring_soon(trusted) if {
    warning_days := lib.rule_data("task_expiry_warning_days")
    expires := time.parse_rfc3339_ns(trusted.expires_on)
    warning_threshold := lib.time.effective_current_time_ns + (warning_days * 24 * 60 * 60 * 1000000000)
    expires < warning_threshold
    expires > lib.time.effective_current_time_ns
}
```

```
TESTS:
```

```rego
package trusted_task_test

import rego.v1

import data.lib
import data.trusted_task

test_trusted_task_no_warning if {
    attestation := {"statement": {"predicate": {
        "buildType": lib.tekton_pipeline_run,
        "buildConfig": {"tasks": [_valid_task]},
    }}}

    lib.assert_empty(trusted_task.warn) with input.attestations as [attestation]
        with data.trusted_tasks as [_trusted_task_far_expiry]
}

test_trusted_task_expiring_soon if {
    attestation := {"statement": {"predicate": {
        "buildType": lib.tekton_pipeline_run,
        "buildConfig": {"tasks": [_valid_task]},
    }}}

    expected := {{
        "code": "trusted_task.current",
        "msg": `Task "my-task" expires on 2025-02-01T00:00:00Z`,
    }}

    lib.assert_equal_results(trusted_task.warn, expected) with input.attestations as [attestation]
        with data.trusted_tasks as [_trusted_task_expiring_soon]
        with data.rule_data.task_expiry_warning_days as 30
}

# Test fixtures
_valid_task := {
    "name": "my-task",
    "ref": {"name": "buildah", "bundle": "oci://registry.local/buildah:1.0@sha256:abc123"},
}

_trusted_task_far_expiry := {
    "ref": "oci://registry.local/buildah:1.0@sha256:abc123",
    "expires_on": "2099-01-01T00:00:00Z",
}

_trusted_task_expiring_soon := {
    "ref": "oci://registry.local/buildah:1.0@sha256:abc123",
    "expires_on": "2025-02-01T00:00:00Z",
}
```

The model learns a **consistent decomposition pattern**:

1. **Analyze** → Understand the requirements, map fields, select helpers
2. **Implement** → Write the Rego rule using appropriate abstractions
3. **Validate** → Create tests matching the test infrastructure (optional)

---

## 🧭 6. Full Example Training Row

```json
{
  "instruction": "Write a Rego rule that enforces the requirements below using the provided attestation schema, helpers, and test cases.",
  "input": "REQUIREMENTS:\n- Package: trusted_task\n- Rule type: warn\n- Purpose: Warn when trusted tasks are approaching expiry\n- Check each Tekton Task in the pipeline.\n- If task is trusted but expires within task_expiry_warning_days → warn.\n- Use lib.result_helper_with_term for searchable results.\n\nATTESTATION_SCHEMA:\n- path: .statement.predicate.buildConfig.tasks[]\n  description: Array of pipeline tasks\n- path: .statement.predicate.buildConfig.tasks[].ref.bundle\n  description: OCI bundle reference for the task\n\nAVAILABLE_HELPERS:\n- tekton.tasks(attestation): Returns all tasks from attestation\n- tekton.task_ref(task): Returns {key, pinned_ref, pinned, kind}\n- lib.result_helper_with_term(chain, args, term): Creates searchable result\n- lib.time.effective_current_time_ns: Current time for policy evaluation\n\nRULE_DATA_KEYS:\n- task_expiry_warning_days: Days before expiry to warn (default: 30)\n- trusted_tasks: Array of {ref, expires_on} objects\n\nTEST_CASES:\n1) name: trusted task - no warning\n   task matches trusted_tasks, expires_on far in future\n   expected_result = pass\n\n2) name: expiring soon\n   task matches trusted_tasks, expires_on within warning window\n   expected_result = warn\n   expected_code = \"trusted_task.current\"\n\n3) name: empty tasks\n   tasks array is empty\n   expected_result = pass\n",
  "output": "ANALYSIS:\n- Field: .statement.predicate.buildConfig.tasks[]\n  Access: via tekton.tasks(attestation)\n  Role: Iterate all pipeline tasks.\n\n- Field: task ref bundle\n  Access: via tekton.task_ref(task)\n  Validation: Match against trusted_tasks rule data.\n\n- Field: trusted_tasks[].expires_on\n  Role: Determines warn behavior.\n  Logic: If within warning_days of now → warn.\n\n- Helper Selection:\n  - tekton.tasks() for iteration\n  - lib.result_helper_with_term() for searchable output\n\nRULE:\n```rego\npackage trusted_task\n\nimport rego.v1\n\nimport data.lib\nimport data.lib.tekton\n\nwarn contains result if {\n    some attestation in lib.pipelinerun_attestations\n    some task in tekton.tasks(attestation)\n    ref := tekton.task_ref(task)\n    trusted := _matching_trusted_task(ref)\n    _is_expiring_soon(trusted)\n    result := lib.result_helper_with_term(\n        rego.metadata.chain(),\n        [tekton.pipeline_task_name(task), trusted.expires_on],\n        tekton.task_name(task),\n    )\n}\n\n_matching_trusted_task(ref) := trusted if {\n    some trusted in lib.rule_data(\"trusted_tasks\")\n    trusted.ref == sprintf(\"%s@%s\", [ref.key, ref.pinned_ref])\n}\n\n_is_expiring_soon(trusted) if {\n    warning_days := lib.rule_data(\"task_expiry_warning_days\")\n    expires := time.parse_rfc3339_ns(trusted.expires_on)\n    warning_threshold := lib.time.effective_current_time_ns + (warning_days * 24 * 60 * 60 * 1000000000)\n    expires < warning_threshold\n    expires > lib.time.effective_current_time_ns\n}\n```\n\nTESTS:\n```rego\npackage trusted_task_test\n\nimport rego.v1\nimport data.lib\nimport data.trusted_task\n\ntest_no_warning_far_expiry if {\n    # ... test implementation\n}\n\ntest_warn_expiring_soon if {\n    # ... test implementation\n}\n```\n"
}
```

---

## 📊 7. Test Infrastructure Patterns

When generating tests, follow these patterns used in the existing test suite:

### Assertion Helpers

```rego
# Assert no violations
lib.assert_empty(trusted_task.warn)

# Assert specific violations
lib.assert_equal_results(trusted_task.deny, expected_set)

# Expected format
expected := {{
    "code": "package_name.rule_name",
    "msg": "Expected error message",
    "term": "searchable_term",  # optional
}}
```

### Mock Data Injection

```rego
# Inject attestations
with input.attestations as [attestation]

# Inject rule data
with data.trusted_tasks as trusted_tasks_data
with data.rule_data.task_expiry_warning_days as 30

# Inject time for deterministic tests
with lib.time.effective_current_time_ns as time.parse_rfc3339_ns("2025-01-15T00:00:00Z")
```

### Test Fixture Naming

```rego
# Prefix fixtures with underscore
_valid_attestation := {...}
_trusted_task_data := [...]

# Use descriptive names
_expired_task := {...}
_task_missing_ref := {...}
```

---

## 🏁 8. Summary

| Principle | Description |
|-----------|-------------|
| ✔️ Input Structure | Use **REQUIREMENTS** + **ATTESTATION_SCHEMA** + **AVAILABLE_HELPERS** + **RULE_DATA_KEYS** + **TEST_CASES** |
| ✔️ Output Structure | Use **ANALYSIS** + **RULE** (+ **TESTS**) as output |
| ✔️ No Raw Attestations | Feed only relevant field slices, never entire attestations |
| ✔️ Library Awareness | Document available helpers so the model uses existing abstractions |
| ✔️ Rule Data Keys | Explicitly list configurable parameters and their schemas |
| ✔️ Complete Test Cases | Include pass cases, fail cases, and edge cases |
| ✔️ Analysis Section | Forces reasoning about fields → helpers → logic |
| ✔️ Test Patterns | Match existing test infrastructure for runnable tests |
| ✔️ Consistency | Structure should be consistent across all training examples |

---

## 🚀 9. Data Generation Strategy

### Recommended Approach

1. **Select representative packages** (start with 5-10):
   - `trusted_task` — complex multi-rule package with expiry logic
   - `sbom_spdx` / `sbom_cyclonedx` — SBOM validation patterns
   - `slsa_build_build_service` — simple attestation field checks
   - `tasks` — required task validation
   - `labels` — image label validation (different input type)

2. **Extract schema automatically** from existing attestations:
   - Parse the JSON attestations in `data/attestations/`
   - Identify which fields each policy package accesses
   - Generate `ATTESTATION_SCHEMA` entries

3. **Derive TEST_CASES from existing tests**:
   - Parse `*_test.rego` files
   - Extract test names, mock data, and assertions
   - Convert to the canonical TEST_CASES format

4. **Identify helpers per package**:
   - Grep for `import data.lib.*` patterns
   - Document which helpers each package uses

5. **Validate before training**:
   - Run generated rules through OPA
   - Ensure tests pass
   - Check for linter errors with Regal

### Quality Checklist

For each training example:
- [ ] REQUIREMENTS include package name and rule type
- [ ] ATTESTATION_SCHEMA covers all accessed fields
- [ ] AVAILABLE_HELPERS lists all used library functions
- [ ] TEST_CASES include at least one pass case
- [ ] TEST_CASES include relevant edge cases
- [ ] Generated RULE compiles without errors
- [ ] Generated TESTS pass when run
