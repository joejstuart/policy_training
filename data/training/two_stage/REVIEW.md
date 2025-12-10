# Training Data Review

Sampled 5 examples for human review.

For each example, check:
- [ ] Requirements are clear and specific
- [ ] ATTESTATION_SCHEMA paths are correct
- [ ] AVAILABLE_HELPERS are appropriate
- [ ] ANALYSIS explains the logic well
- [ ] RULE code matches requirements

---

## Example 1 (Stage 2) - ✅ Valid

**File:** `stage2_eval.jsonl` index 4

### Instruction

```
Write a Rego rule that enforces the requirements below using the provided context.
```

### Input

```
REQUIREMENTS:
- Package: github_certificate
- Rule type: deny
- Short name: gh_workflow_ref
- Purpose: GitHub Workflow Repository
- Check if the value of the GitHub Workflow Ref extension in the image signature certificate matches one of the allowed values. Use the rule data key `allowed_gh_workflow_refs` to specify the list of allowed values. An empty allow list, which is the default value, causes this check to succeeded.
- Error message: 'Ref %q not in allowed list: %v'

ATTESTATION_SCHEMA:
- (Unable to infer schema - review manually)

AVAILABLE_HELPERS:
- name: lib.json
- name: image.signatures
- name: lib.rule_data
  description: Returns the "first found" of the following: data.rule_data__configuration__[key_name]
- name: lib.result_helper
- name: j.validate_schema
  description: Validates schema reporting the error message as well as the severity
- name: lib.result_helper_with_severity
- name: rego.metadata.chain()
  description: Returns metadata chain for current rule

EXPECTED_B
```

### Output

```
ANALYSIS:
- Iterates: sig in input.image.signatures
- Iterates: cert in _certs
- Iterates: extension in cert.Extensions
- Uses negation (not) for condition checking
- Uses lib.result_helper_with_severity for severity levels

Helper Selection:
  - lib.json: Used for processing
  - lib.result_helper: Used for processing
  - lib.result_helper_with_severity: Used for processing
  - lib.rule_data: Used for processing
  - image.signatures: Used for processing

RULE:
```rego
package github_certificate

import rego.v1
import data.lib
import data.lib.json as j

_NAME := {"id": 4, "name": "GitHub Workflow Name"}

_REF := {"id": 6, "name": "GitHub Workflow Ref"}

_TRIGGER := {"id": 2, "name": "GitHub Workflow Trigger"}

_certs contains cert if {
	some sig in input.image.signatures
	cert := crypto.x509.parse_certificates(sig.certificate)[0]
	cert.KeyUsage == 1
	cert.ExtKeyUsage == [3]
}

_check_extension(chain, key, extension) := result if {
	value := _fulcio_extension_value(extension)
	allowed := lib.rule_data(key)
	count(allowed) > 0
	not value in allowed
	result := lib.result_helper(chain, [value, allowed])
}

_fulcio_extension_value(ext) := value if {
	id := [1, 3, 6, 1, 4, 1, 57264, 1, ext.id]
	some cert in _certs
	some extension in cert.Extensions
	extension.Id == id
	value := base64.decode(extension.Value)
}

_rule_data_errors contains error if {
	keys := [
		"allowed_gh_workflow_repos",
		"allowed_gh_workflow_refs",
		"allowed_gh_workflow_names",
		"allowed_gh_workflow_triggers",
	]
	some key in keys

	some e in j.validate_schema(
		lib.rule_data(key),
		{
			"$schema": "http://json-schema.org/draft-07/schema#",
			"type": "array",
			"items": {"type": "string"},
			"uniqueItems": true,
		},
	)
	error := {
		"message": sprintf("Rule data %s has unexpected format: %s", [key, e.message]),
		"severity": e.severity,
	}
}

# METADATA
# title: GitHub Workflow Repository
# description: >-
#   Check if the value of the GitHub Workflow Ref extension in the image
#   signature ce
```


---

## Example 2 (Stage 1) - ✅ Valid

**File:** `stage1_eval.jsonl` index 3

### Instruction

```
Analyze the requirements and identify the attestation schema, available helpers, and rule data keys needed to implement this Rego rule.
```

### Input

```
REQUIREMENTS:
- Package: trusted_task
- Rule type: warn
- Short name: tagged
- Purpose: Task references are tagged
- Check if all Tekton Tasks defined with the bundle format contain a tag reference.
- Error message: Pipeline task %q uses an untagged task reference, %s
```

### Output

```
ATTESTATION_SCHEMA:
- .statement.predicate.buildConfig.tasks[].ref

AVAILABLE_HELPERS:
- name: lib.image
- name: lib.tekton
- name: tekton.task_ref
  description: Return an object that represents the task "name", "kind", and "bundle". "bundle" is omitted if a bundle is not used.
- name: tekton.untagged_task_references
  description: Returns a subset of tasks that use untagged bundle Task references.
- name: lib.tasks_from_pipelinerun
- name: lib.result_helper_with_term
- name: tekton.pipeline_task_name
  description: returns a slsav0.2 pipeline task name the name field (which is the taskRun name) for slsav1.0 is metadata.name
- name: tekton.task_name
  description: task name from a v0.2 and v1.0 attestation
- name: rego.metadata.chain()
  description: Returns metadata chain for current rule

EXPECTED_BEHAVIOR:
1) Optimal input → pass (no warning)
2) Suboptimal input → warn
  - success
  - pinned warning
  - tagged warning
```


---

## Example 3 (Stage 2) - ✅ Valid

**File:** `stage2_eval.jsonl` index 3

### Instruction

```
Write a Rego rule that enforces the requirements below using the provided context.
```

### Input

```
REQUIREMENTS:
- Package: trusted_task
- Rule type: warn
- Short name: tagged
- Purpose: Task references are tagged
- Check if all Tekton Tasks defined with the bundle format contain a tag reference.
- Error message: Pipeline task %q uses an untagged task reference, %s

ATTESTATION_SCHEMA:
- .statement.predicate.buildConfig.tasks[].ref

AVAILABLE_HELPERS:
- name: lib.image
- name: lib.tekton
- name: tekton.task_ref
  description: Return an object that represents the task "name", "kind", and "bundle". "bundle" is omitted if a bundle is not used.
- name: tekton.untagged_task_references
  description: Returns a subset of tasks that use untagged bundle Task references.
- name: lib.tasks_from_pipelinerun
- name: lib.result_helper_with_term
- name: tekton.pipeline_task_name
  description: returns a slsav0.2 pipeline task name the name field (which is the taskRun name) for slsav1.0 is metadata.name
- name: tekton.task_name
  description: task name from a v0.2 and v1.0 attestation
- name: rego.
```

### Output

```
ANALYSIS:
- Iterates: task in tekton.untagged_task_references(lib.tasks_from_pipelinerun)
- Uses lib.result_helper_with_term for searchable results

Helper Selection:
  - lib.tasks_from_pipelinerun: Used for data access
  - lib.result_helper_with_term: Used for processing
  - tekton.task_name: Used for processing
  - lib.tekton: Used for processing
  - tekton.pipeline_task_name: Used for processing

RULE:
```rego
package trusted_task

import rego.v1
import data.lib
import data.lib.image
import data.lib.tekton

_task_info(task) := info if {
	ref := tekton.task_ref(task)
	info := sprintf("%s@%s", [object.get(ref, "key", ""), object.get(ref, "pinned_ref", "")])
}

# METADATA
# title: Task references are tagged
# description: >-
#   Check if all Tekton Tasks defined with the bundle format contain a tag reference.
# custom:
#   short_name: tagged
#   failure_msg: Pipeline task %q uses an untagged task reference, %s
#   solution: >-
#     Update the Pipeline definition so that all Task references have a tagged value as mentioned
#     in the description.
#   collections:
#   - redhat
#   - redhat_rpms
#   effective_on: 2024-05-07T00:00:00Z
#
warn contains result if {
	some task in tekton.untagged_task_references(lib.tasks_from_pipelinerun)
	result := lib.result_helper_with_term(
		rego.metadata.chain(),
		[tekton.pipeline_task_name(task), _task_info(task)],
		tekton.task_name(task),
	)
}
```

TESTS:
```rego
package trusted_task_test

import rego.v1

test_success if {
	att_no_ta := {"statement": {"predicate": {
		"buildType": lib.tekton_pipeline_run,
		"buildConfig": {"tasks": [
			newest_bundle_pipeline_task,
			newest_git_pipeline_task,
		]},
	}}}

	lib.assert_empty(trusted_task.warn | trusted_task.deny, expected) with data.trusted_tasks as trusted_tasks_data
		with input.attestations as [att_no_ta, attestation_ta]
}

test_pinned_warning if {
	att := {"statement": {"predicate": {
		"buildType": lib.tekton_pipeline_run,
		"buildConfig": {"tasks": [
			trusted_bundle_pipel
```


---

## Example 4 (Stage 1) - ✅ Valid

**File:** `stage1_eval.jsonl` index 2

### Instruction

```
Analyze the requirements and identify the attestation schema, available helpers, and rule data keys needed to implement this Rego rule.
```

### Input

```
REQUIREMENTS:
- Package: sbom_spdx
- Rule type: deny
- Short name: disallowed_package_external_references
- Purpose: Disallowed package external references
- Confirm the SPDX SBOM contains only packages without disallowed external references. By default all external references are allowed. Use the "disallowed_external_references" rule data key to provide a list of type-pattern pairs that forbid the use of an external reference of the given type where the reference url matches the given pattern.
- Error message: Package %s has reference %q of type %q which is disallowed%s
```

### Output

```
ATTESTATION_SCHEMA:
- SPDX SBOM structure

AVAILABLE_HELPERS:
- name: lib.image
- name: lib.sbom
- name: sbom.spdx_sboms
- name: lib.rule_data
  description: Returns the "first found" of the following: data.rule_data__configuration__[key_name]
- name: sbom.rule_data_disallowed_external_references_key
- name: lib.result_helper
- name: rego.metadata.chain()
  description: Returns metadata chain for current rule

EXPECTED_BEHAVIOR:
1) Valid input → pass (no denial)
2) Invalid input → deny with error
  - all good
  - all good marshaled
  - missing packages
```


---

## Example 5 (Stage 2) - ✅ Valid

**File:** `stage2_train.jsonl` index 4

### Instruction

```
Write a Rego rule that enforces the requirements below using the provided context.
```

### Input

```
REQUIREMENTS:
- Package: rpm_ostree_task
- Rule type: deny
- Short name: builder_image_param
- Purpose: Builder image parameter
- Verify the BUILDER_IMAGE parameter of the rpm-ostree Task uses an image reference that is both pinned to a digest and starts with a pre-defined list of prefixes. By default, the list of prefixes is empty allowing any pinned image reference to be used. This is customizable via the `allowed_rpm_ostree_builder_image_prefixes` rule data.
- Error message: "%s"

ATTESTATION_SCHEMA:
- (Unable to infer schema - review manually)

AVAILABLE_HELPERS:
- name: lib.json
- name: lib.tekton
- name: lib.result_helper
- name: rego.metadata.chain()
  description: Returns metadata chain for current rule

EXPECTED_BEHAVIOR:
1) Valid input → pass (no denial)
2) Invalid input → deny with error
  - success
  - builder image param failures
  - rule data failures
```

### Output

```
ANALYSIS:
- Iterates: error in builder_image_param_errors
- Uses lib.result_helper for result generation

Helper Selection:
  - lib.json: Used for processing
  - lib.result_helper: Used for processing
  - lib.tekton: Used for processing

RULE:
```rego
package rpm_ostree_task

import rego.v1
import data.lib
import data.lib.json as j
import data.lib.tekton

_with_effective_on(obj, record) := new_obj if {
	new_obj := object.union(obj, {"effective_on": record.effective_on})
} else := obj

# METADATA
# title: Builder image parameter
# description: >-
#   Verify the BUILDER_IMAGE parameter of the rpm-ostree Task uses an image reference that is both
#   pinned to a digest and starts with a pre-defined list of prefixes. By default, the list of
#   prefixes is empty allowing any pinned image reference to be used. This is customizable via the
#   `allowed_rpm_ostree_builder_image_prefixes` rule data.
# custom:
#   short_name: builder_image_param
#   failure_msg: "%s"
#   solution: >-
#     Make sure the rpm-ostree Task uses a pinned image reference from a pre-approved location.
#   collections:
#   - redhat
#   effective_on: 2024-03-20T00:00:00Z
#
deny contains result if {
	some error in builder_image_param_errors
	result := _with_effective_on(lib.result_helper(rego.metadata.chain(), [error.msg]), error)
}
```

TESTS:
```rego
package rpm_ostree_task_test

import rego.v1

test_success if {
	slsa_v02_attestation := {"statement": {"predicate": {
		"buildType": lib.tekton_pipeline_run,
		"buildConfig": {"tasks": [{
			"name": "rpm-ostree-p",
			"ref": {"kind": "Task", "name": "rpm-ostree"},
			"invocation": {"parameters": {"BUILDER_IMAGE": "registry.local/builder:v0.2@sha256:abc"}},
		}]},
	}}}

	slsa_v1_attestation := {"statement": {
		"predicateType": "https://slsa.dev/provenance/v1",
		"predicate": {"buildDefinition": {
			"buildType": "https://tekton.dev/chains/v2/slsa-tekton",
			"externalParameters": {"runSpec": {"pipelineSpec": {}}},
			"resolvedDependencies": [{
				"name
```


---

