# Prototype Training Example: trusted_task.current

This is a human-readable version of the training example for the "Tasks using the latest versions" warning rule.

---

## Instruction

> Write a Rego rule that enforces the requirements below using the provided context.

---

## Input

### REQUIREMENTS

- **Package:** `trusted_task`
- **Rule type:** `warn`
- **Short name:** `current`
- **Purpose:** Warn when trusted tasks are approaching their expiry date

**Behavioral Requirements:**
- For each Tekton Task in the pipeline, check if it uses a trusted but soon-to-expire reference.
- Use the `task_expiry_warning_days` rule data to determine the warning window (default: 30 days).
- Only warn if the task IS trusted but will expire within the warning window.
- Do not warn if the task has already expired (that's handled by the `trusted` deny rule).
- Include the expiry date and latest available reference in the warning message.

### ATTESTATION_SCHEMA

| Path | Description | Type |
|------|-------------|------|
| `.statement.predicate.buildType` | Identifies the attestation type. Filter for Tekton PipelineRun attestations. | string |
| `.statement.predicate.buildConfig.tasks[]` | Array of tasks executed in the pipeline. | array |
| `.statement.predicate.buildConfig.tasks[].name` | The PipelineTask name (how the task is named in the Pipeline definition). | string |
| `.statement.predicate.buildConfig.tasks[].ref.resolver` | The resolver type used ("bundles" for OCI, "git" for git resolver). | string |
| `.statement.predicate.buildConfig.tasks[].ref.params[]` | Resolver parameters including bundle reference or git revision. | array |
| `.statement.predicate.buildConfig.tasks[].invocation.environment.labels` | Labels from the TaskRun, including "tekton.dev/task" with the Task name. | object |

### AVAILABLE_HELPERS

| Helper | Description | Returns |
|--------|-------------|---------|
| `lib.tasks_from_pipelinerun` | Returns all tasks from all PipelineRun attestations in the input. | set of task objects |
| `tekton.task_ref(task)` | Parses task reference into normalized form. | `{key, pinned_ref, pinned, kind, bundle?, tagged?, tagged_ref?}` |
| `tekton.expiry_of(task)` | Returns the expiry time in nanoseconds if the task is expiring within the warning window. | epoch nanoseconds or undefined |
| `tekton.pipeline_task_name(task)` | Returns the PipelineTask name (the name in the Pipeline definition). | string |
| `tekton.task_name(task)` | Returns the Task name (from tekton.dev/task label or ref params). | string |
| `tekton.latest_trusted_ref(task)` | Returns the latest trusted reference for this task from trusted_tasks data. | string |
| `lib.result_helper_with_term(chain, args, term)` | Creates a policy result with a searchable term. | result object |
| `time.format(ns)` | Formats epoch nanoseconds as RFC3339 string. | string |

### RULE_DATA_KEYS

| Key | Type | Description |
|-----|------|-------------|
| `task_expiry_warning_days` | integer | Number of days before task expiry to start warning. Default: 30 |
| `trusted_tasks` | object | Map of task references to arrays of trusted versions with expiry dates |

**trusted_tasks schema:**
```json
{
  "oci://registry.local/task:tag": [
    {
      "ref": "sha256:abc123...",
      "effective_on": "2024-01-01T00:00:00Z",
      "expires_on": "2025-06-01T00:00:00Z"
    }
  ]
}
```

### EXPECTED_BEHAVIOR

> **Note:** This section describes pass/fail scenarios (not actual test code). Stage 2 will generate the actual `TESTS` from these.

| # | Name | Description | Expected |
|---|------|-------------|----------|
| 1 | trusted task with far expiry | A trusted task that expires far in the future should not trigger a warning. | pass |
| 2 | outdated task expiring soon | A trusted task with an expiry date within warning window should warn. | warn, code=`trusted_task.current` |
| 3 | git resolver outdated task | Git resolver tasks should also warn when expiring. | warn, code=`trusted_task.current` |
| 4 | untrusted task | An untrusted task should not trigger this warn rule (handled by trusted deny rule). | pass |
| 5 | expired task | A task that has already expired should not warn (it's denied by trusted rule). | pass |

---

## Output

### ANALYSIS

**Field-to-Logic Mapping:**

| Field | Access | Role |
|-------|--------|------|
| `.statement.predicate.buildConfig.tasks[]` | via `lib.tasks_from_pipelinerun` | Iterate all pipeline tasks to check for expiring trusted references |
| task reference (bundle or git) | via `tekton.task_ref(task) → {key, pinned_ref}` | Provides the canonical key for looking up in trusted_tasks data |
| trusted_tasks data | via `tekton.expiry_of(task)` | Determines if the task's current ref has an upcoming expiry |

**Helper Selection:**
- `lib.tasks_from_pipelinerun`: Convenient accessor for all tasks across attestations
- `tekton.expiry_of`: Encapsulates expiry window logic, respects task_expiry_warning_days
- `tekton.pipeline_task_name`: For error message (the name users see in Pipeline)
- `tekton.task_name`: For the searchable term (the actual Task name)
- `tekton.latest_trusted_ref`: To suggest upgrade path in message
- `time.format`: Convert nanoseconds to readable date
- `lib.result_helper_with_term`: Generate result with metadata and searchable term

**Message Template:**
```
A newer version of task %q exists. Please update before %s.
The current bundle is %q and the latest bundle ref is %q
```
Args: `[pipeline_task_name, expiry_date, current_bundle_info, latest_ref]`

### RULE

```rego
package trusted_task

import rego.v1

import data.lib
import data.lib.tekton

# METADATA
# title: Tasks using the latest versions
# description: >-
#   Check if all Tekton Tasks use the latest known Task reference. When warnings
#   will be reported can be configured using the `task_expiry_warning_days` rule
#   data setting. It holds the number of days before the task is to expire within
#   which the warnings will be reported.
# custom:
#   short_name: current
#   failure_msg: >-
#     A newer version of task %q exists. Please update before %s.
#     The current bundle is %q and the latest bundle ref is %q
#   solution: >-
#     Update the Task reference to a newer version.
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

### TESTS

```rego
package trusted_task_test

import rego.v1

import data.lib
import data.trusted_task

# Test: trusted task with far expiry - no warning
test_no_warning_latest_version if {
	att := {"statement": {"predicate": {
		"buildType": lib.tekton_pipeline_run,
		"buildConfig": {"tasks": [trusted_bundle_pipeline_task]},
	}}}

	lib.assert_empty(trusted_task.warn) with input.attestations as [att]
		with data.trusted_tasks as trusted_tasks_data
}

# Test: outdated task expiring soon - warn
test_outdated_warning if {
	att := {"statement": {"predicate": {
		"buildType": lib.tekton_pipeline_run,
		"buildConfig": {"tasks": [
			trusted_bundle_pipeline_task,
			outdated_bundle_pipeline_task,
		]},
	}}}

	expected := {{
		"code": "trusted_task.current",
		"msg": `A newer version of task "outdated-trusty-p" exists...`,
		"term": "trusty",
	}}

	lib.assert_equal_results(trusted_task.warn, expected) with input.attestations as [att]
		with data.trusted_tasks as trusted_tasks_data
}

# Test fixtures
trusted_bundle_pipeline_task := {
	"name": "trusty-p",
	"ref": {"resolver": "bundles", "params": [
		{"name": "bundle", "value": "registry.local/trusty:1.0@sha256:digest"},
		{"name": "name", "value": "trusty"},
		{"name": "kind", "value": "task"},
	]},
}

outdated_bundle_pipeline_task := {
	"name": "outdated-trusty-p",
	"ref": {"resolver": "bundles", "params": [
		{"name": "bundle", "value": "registry.local/trusty:1.0@sha256:outdated-digest"},
		{"name": "name", "value": "trusty"},
		{"name": "kind", "value": "task"},
	]},
}

trusted_tasks_data := {
	"oci://registry.local/trusty:1.0": [
		{
			"ref": "sha256:digest",
			"effective_on": "2099-01-01T00:00:00Z",
		},
		{
			"ref": "sha256:outdated-digest",
			"effective_on": "2024-01-01T00:00:00Z",
			"expires_on": "2099-01-01T00:00:00Z",
		},
	],
}
```

---

## Key Observations

### What Makes This Example Good

1. **ANALYSIS explicitly maps fields → helpers → logic**
   - Shows the model exactly which library function to use for each field access
   - Explains *why* each helper is chosen

2. **EXPECTED_BEHAVIOR covers multiple scenarios**
   - Pass case (trusted, not expiring)
   - Warn case (trusted, expiring soon)
   - Different resolver types (bundles vs git)
   - Boundary cases (already expired, untrusted)

3. **AVAILABLE_HELPERS documents the API**
   - Return types are specified
   - Params are explained
   - Notes clarify non-obvious behavior

4. **RULE_DATA_KEYS shows configuration**
   - Documents the schema for complex data structures
   - Shows defaults and how they're used

5. **Generated code matches existing patterns**
   - Uses same METADATA format as real policies
   - Same helper functions
   - Same test infrastructure patterns

### Complexity Level

This is a **medium complexity** example:
- Single rule (not a package with multiple rules)
- Uses 6-7 library helpers
- Depends on rule data
- Has both OCI and git resolver cases
- Requires understanding of expiry window logic

