#!/usr/bin/env python3
"""
Generate synthetic Rego rules for training data augmentation.

This script creates diverse rule variations from templates, ensuring:
- Correct Rego syntax
- Proper Conforma patterns (deny contains result, METADATA, result_helper)
- Accurate helper function usage
- Varied task names, labels, thresholds, etc.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

# Output directory
REPO_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / "data" / "training" / "synthetic"


@dataclass
class SyntheticRule:
    """A synthetic rule with all required training data."""
    instruction: str
    package: str
    rule_type: str  # deny or warn
    attestation_schema: str
    available_helpers: str
    rule_data_keys: str
    analysis: str
    rule_code: str
    test_code: str


# =============================================================================
# TEMPLATE 1: Task Existence Check
# Verifies a specific task exists in the pipeline
# =============================================================================

TASK_EXISTENCE_TEMPLATE = '''package {package}

import rego.v1
import data.lib
import data.lib.tekton

# METADATA
# title: {title}
# description: {description}
# custom:
#   short_name: {short_name}
#   failure_msg: {failure_msg}
#
{rule_type} contains result if {{
	some att in lib.pipelinerun_attestations
	tasks := tekton.tasks(att)
	not _has_required_task(tasks)
	result := lib.result_helper(rego.metadata.chain(), [])
}}

_has_required_task(tasks) if {{
	some task in tasks
	task.name == "{task_name}"
}}'''

TASK_EXISTENCE_TEST_TEMPLATE = '''package {package}_test

import rego.v1
import data.lib

test_{short_name}_pass if {{
	lib.assert_empty({package}.{rule_type}) with input.attestations as [_attestation_with_task]
}}

test_{short_name}_fail if {{
	lib.assert_not_empty({package}.{rule_type}) with input.attestations as [_attestation_without_task]
}}

_attestation_with_task := {{"statement": {{"predicate": {{"buildConfig": {{"tasks": [
	{{"name": "{task_name}", "ref": {{"name": "{task_name}", "kind": "Task"}}}}
]}}}}}}}}

_attestation_without_task := {{"statement": {{"predicate": {{"buildConfig": {{"tasks": [
	{{"name": "other-task", "ref": {{"name": "other-task", "kind": "Task"}}}}
]}}}}}}}}'''

TASK_EXISTENCE_VARIATIONS = [
    {
        "task_name": "buildah",
        "title": "Buildah task present",
        "description": "Verify the pipeline includes a buildah build task.",
        "instructions": [
            "Check that the pipeline has a buildah task",
            "Verify buildah is used for container builds",
            "Ensure the build uses buildah",
        ]
    },
    {
        "task_name": "git-clone",
        "title": "Git clone task present",
        "description": "Verify the pipeline includes a git-clone task for source checkout.",
        "instructions": [
            "Check that the pipeline clones the git repository",
            "Verify git-clone task exists in the pipeline",
            "Ensure source code is cloned via git-clone task",
        ]
    },
    {
        "task_name": "clair-scan",
        "title": "Clair scan task present",
        "description": "Verify the pipeline includes a clair-scan task for vulnerability scanning.",
        "instructions": [
            "Check that vulnerability scanning is performed",
            "Verify clair-scan task runs in the pipeline",
            "Ensure CVE scanning with clair-scan is included",
        ]
    },
    {
        "task_name": "source-build",
        "title": "Source build task present",
        "description": "Verify the pipeline includes a source-build task.",
        "instructions": [
            "Check that source-build task is present",
            "Verify the pipeline builds source containers",
            "Ensure source-build runs in the pipeline",
        ]
    },
    {
        "task_name": "sbom-json-check",
        "title": "SBOM generation task present",
        "description": "Verify the pipeline includes an SBOM generation task.",
        "instructions": [
            "Check that SBOM is generated in the pipeline",
            "Verify sbom-json-check task exists",
            "Ensure the build produces an SBOM",
        ]
    },
]


# =============================================================================
# TEMPLATE 2: Task Parameter Value Check
# Verifies a task parameter has a specific value or matches a pattern
# =============================================================================

TASK_PARAM_TEMPLATE = '''package {package}

import rego.v1
import data.lib
import data.lib.tekton

# METADATA
# title: {title}
# description: {description}
# custom:
#   short_name: {short_name}
#   failure_msg: {failure_msg}
#
{rule_type} contains result if {{
	some att in lib.pipelinerun_attestations
	some task in tekton.tasks(att)
	task.name == "{task_name}"
	param_value := tekton.task_param(task, "{param_name}")
	{check_condition}
	result := lib.result_helper_with_term(
		rego.metadata.chain(),
		[task.name, param_value],
		task.name,
	)
}}'''

TASK_PARAM_TEST_TEMPLATE = '''package {package}_test

import rego.v1
import data.lib

test_{short_name}_pass if {{
	lib.assert_empty({package}.{rule_type}) with input.attestations as [_good_attestation]
}}

test_{short_name}_fail if {{
	lib.assert_not_empty({package}.{rule_type}) with input.attestations as [_bad_attestation]
}}

_good_attestation := {{"statement": {{"predicate": {{"buildConfig": {{"tasks": [
	{{"name": "{task_name}", "params": [{{"name": "{param_name}", "value": "{good_value}"}}]}}
]}}}}}}}}

_bad_attestation := {{"statement": {{"predicate": {{"buildConfig": {{"tasks": [
	{{"name": "{task_name}", "params": [{{"name": "{param_name}", "value": "{bad_value}"}}]}}
]}}}}}}}}'''

TASK_PARAM_VARIATIONS = [
    {
        "task_name": "buildah",
        "param_name": "HERMETIC",
        "check_condition": 'param_value != "true"',
        "good_value": "true",
        "bad_value": "false",
        "title": "Buildah hermetic build",
        "description": "Verify the buildah task uses hermetic build mode.",
        "instructions": [
            "Check that buildah uses hermetic builds",
            "Verify HERMETIC parameter is true for buildah",
            "Ensure buildah builds are hermetic",
        ]
    },
    {
        "task_name": "buildah",
        "param_name": "DOCKERFILE",
        "check_condition": 'startswith(param_value, "http")',
        "good_value": "./Dockerfile",
        "bad_value": "https://example.com/Dockerfile",
        "title": "Local Dockerfile used",
        "description": "Verify the buildah task uses a local Dockerfile, not fetched from external source.",
        "instructions": [
            "Check that Dockerfile is local, not from URL",
            "Verify DOCKERFILE param doesn't use http/https",
            "Ensure Dockerfile is not fetched externally",
        ]
    },
    {
        "task_name": "git-clone",
        "param_name": "DEPTH",
        "check_condition": 'to_number(param_value) > 1',
        "good_value": "1",
        "bad_value": "100",
        "title": "Shallow git clone",
        "description": "Verify the git-clone task uses shallow clone (depth=1).",
        "instructions": [
            "Check that git clone uses shallow depth",
            "Verify DEPTH parameter is 1 for git-clone",
            "Ensure git-clone uses depth 1",
        ]
    },
]


# =============================================================================
# TEMPLATE 3: Task Result Validation
# Verifies a task produced expected results
# =============================================================================

TASK_RESULT_TEMPLATE = '''package {package}

import rego.v1
import data.lib
import data.lib.tekton

# METADATA
# title: {title}
# description: {description}
# custom:
#   short_name: {short_name}
#   failure_msg: {failure_msg}
#
{rule_type} contains result if {{
	some att in lib.pipelinerun_attestations
	some task in tekton.tasks(att)
	task.name == "{task_name}"
	not _has_result(task, "{result_name}")
	result := lib.result_helper_with_term(
		rego.metadata.chain(),
		[task.name, "{result_name}"],
		task.name,
	)
}}

_has_result(task, name) if {{
	some result in tekton.task_results(task)
	result.name == name
}}'''

TASK_RESULT_TEST_TEMPLATE = '''package {package}_test

import rego.v1
import data.lib

test_{short_name}_pass if {{
	lib.assert_empty({package}.{rule_type}) with input.attestations as [_good_attestation]
}}

test_{short_name}_fail if {{
	lib.assert_not_empty({package}.{rule_type}) with input.attestations as [_bad_attestation]
}}

_good_attestation := {{"statement": {{"predicate": {{"buildConfig": {{"tasks": [
	{{"name": "{task_name}", "results": [{{"name": "{result_name}", "value": "test-value"}}]}}
]}}}}}}}}

_bad_attestation := {{"statement": {{"predicate": {{"buildConfig": {{"tasks": [
	{{"name": "{task_name}", "results": [{{"name": "OTHER_RESULT", "value": "test-value"}}]}}
]}}}}}}}}'''

TASK_RESULT_VARIATIONS = [
    {
        "task_name": "buildah",
        "result_name": "IMAGE_DIGEST",
        "title": "Image digest produced",
        "description": "Verify the buildah task produces an IMAGE_DIGEST result.",
        "instructions": [
            "Check that buildah outputs IMAGE_DIGEST",
            "Verify IMAGE_DIGEST result exists from buildah",
            "Ensure build task produces image digest",
        ]
    },
    {
        "task_name": "buildah",
        "result_name": "IMAGE_URL",
        "title": "Image URL produced",
        "description": "Verify the buildah task produces an IMAGE_URL result.",
        "instructions": [
            "Check that buildah outputs IMAGE_URL",
            "Verify IMAGE_URL result exists from build task",
            "Ensure build produces image URL",
        ]
    },
    {
        "task_name": "git-clone",
        "result_name": "COMMIT",
        "title": "Git commit recorded",
        "description": "Verify the git-clone task records the COMMIT result.",
        "instructions": [
            "Check that git-clone records the commit SHA",
            "Verify COMMIT result from git-clone exists",
            "Ensure source commit is captured",
        ]
    },
    {
        "task_name": "clair-scan",
        "result_name": "REPORTS",
        "title": "CVE reports produced",
        "description": "Verify the clair-scan task produces vulnerability REPORTS.",
        "instructions": [
            "Check that clair-scan produces REPORTS",
            "Verify vulnerability scan results exist",
            "Ensure CVE scan outputs reports",
        ]
    },
]


# =============================================================================
# TEMPLATE 4: SBOM Package Check
# Verifies packages in SBOM meet criteria
# =============================================================================

SBOM_PACKAGE_TEMPLATE = '''package {package}

import rego.v1
import data.lib
import data.lib.sbom

# METADATA
# title: {title}
# description: {description}
# custom:
#   short_name: {short_name}
#   failure_msg: {failure_msg}
#
{rule_type} contains result if {{
	some s in sbom.{sbom_type}_sboms
	some pkg in s.{package_path}
	{check_condition}
	result := lib.result_helper_with_term(
		rego.metadata.chain(),
		[pkg.name],
		pkg.name,
	)
}}'''

SBOM_PACKAGE_TEST_TEMPLATE = '''package {package}_test

import rego.v1
import data.lib

test_{short_name}_pass if {{
	lib.assert_empty({package}.{rule_type}) with data.sbom.{sbom_type}_sboms as [_good_sbom]
}}

test_{short_name}_fail if {{
	lib.assert_not_empty({package}.{rule_type}) with data.sbom.{sbom_type}_sboms as [_bad_sbom]
}}

_good_sbom := {{{good_sbom}}}

_bad_sbom := {{{bad_sbom}}}'''

SBOM_PACKAGE_VARIATIONS = [
    {
        "sbom_type": "spdx",
        "package_path": "packages",
        "check_condition": 'not pkg.versionInfo',
        "title": "SPDX packages have versions",
        "description": "Verify all packages in SPDX SBOM have version information.",
        "good_sbom": '"packages": [{"name": "pkg1", "versionInfo": "1.0.0"}]',
        "bad_sbom": '"packages": [{"name": "pkg1"}]',
        "instructions": [
            "Check that SPDX packages have version info",
            "Verify all SBOM packages include versions",
            "Ensure package versions are present in SPDX",
        ]
    },
    {
        "sbom_type": "cyclonedx",
        "package_path": "components",
        "check_condition": 'not pkg.purl',
        "title": "CycloneDX components have purl",
        "description": "Verify all components in CycloneDX SBOM have Package URL.",
        "good_sbom": '"components": [{"name": "pkg1", "purl": "pkg:npm/pkg1@1.0.0"}]',
        "bad_sbom": '"components": [{"name": "pkg1"}]',
        "instructions": [
            "Check that CycloneDX components have purl",
            "Verify all SBOM components include package URL",
            "Ensure purl is present for all components",
        ]
    },
    {
        "sbom_type": "spdx",
        "package_path": "packages",
        "check_condition": 'pkg.name == ""',
        "title": "SPDX packages have names",
        "description": "Verify all packages in SPDX SBOM have non-empty names.",
        "good_sbom": '"packages": [{"name": "valid-package", "versionInfo": "1.0.0"}]',
        "bad_sbom": '"packages": [{"name": "", "versionInfo": "1.0.0"}]',
        "instructions": [
            "Check that SPDX packages have names",
            "Verify all SBOM packages have non-empty names",
            "Ensure package names exist in SPDX",
        ]
    },
]


# =============================================================================
# TEMPLATE 5: Image Label Check
# Verifies image has required labels
# =============================================================================

IMAGE_LABEL_TEMPLATE = '''package {package}

import rego.v1
import data.lib
import data.lib.image

# METADATA
# title: {title}
# description: {description}
# custom:
#   short_name: {short_name}
#   failure_msg: {failure_msg}
#
{rule_type} contains result if {{
	config := image.config(input.image.ref)
	labels := object.get(config, ["config", "Labels"], {{}})
	not labels["{label_name}"]
	result := lib.result_helper(rego.metadata.chain(), ["{label_name}"])
}}'''

IMAGE_LABEL_TEST_TEMPLATE = '''package {package}_test

import rego.v1
import data.lib

test_{short_name}_pass if {{
	lib.assert_empty({package}.{rule_type}) with input.image as _image_with_label
		with data.image.config as _config_with_label
}}

test_{short_name}_fail if {{
	lib.assert_not_empty({package}.{rule_type}) with input.image as _image_without_label
		with data.image.config as _config_without_label
}}

_image_with_label := {{"ref": "registry.example.com/image@sha256:abc123"}}

_image_without_label := {{"ref": "registry.example.com/image@sha256:abc123"}}

_config_with_label := {{"registry.example.com/image@sha256:abc123": {{"config": {{"Labels": {{"{label_name}": "{label_value}"}}}}}}}}

_config_without_label := {{"registry.example.com/image@sha256:abc123": {{"config": {{"Labels": {{}}}}}}}}'''

IMAGE_LABEL_VARIATIONS = [
    {
        "label_name": "com.redhat.component",
        "label_value": "my-component",
        "title": "Red Hat component label",
        "description": "Verify image has the com.redhat.component label.",
        "instructions": [
            "Check that image has Red Hat component label",
            "Verify com.redhat.component label exists",
            "Ensure image is labeled with component name",
        ]
    },
    {
        "label_name": "version",
        "label_value": "1.0.0",
        "title": "Version label present",
        "description": "Verify image has a version label.",
        "instructions": [
            "Check that image has version label",
            "Verify the version label is set",
            "Ensure image version is labeled",
        ]
    },
    {
        "label_name": "maintainer",
        "label_value": "team@example.com",
        "title": "Maintainer label present",
        "description": "Verify image has a maintainer label.",
        "instructions": [
            "Check that image has maintainer label",
            "Verify maintainer contact is labeled",
            "Ensure image has maintainer information",
        ]
    },
    {
        "label_name": "description",
        "label_value": "My application",
        "title": "Description label present",
        "description": "Verify image has a description label.",
        "instructions": [
            "Check that image has description label",
            "Verify the image description is set",
            "Ensure image includes description",
        ]
    },
]


# =============================================================================
# Generator Functions
# =============================================================================

def generate_task_existence_rules() -> List[SyntheticRule]:
    """Generate task existence check rules."""
    rules = []
    
    for var in TASK_EXISTENCE_VARIATIONS:
        for rule_type in ["deny", "warn"]:
            for instruction in var["instructions"]:
                package = f"check_{var['task_name'].replace('-', '_')}"
                short_name = f"has_{var['task_name'].replace('-', '_')}"
                
                rule_code = TASK_EXISTENCE_TEMPLATE.format(
                    package=package,
                    title=var["title"],
                    description=var["description"],
                    short_name=short_name,
                    failure_msg=f"Required task '{var['task_name']}' not found in pipeline",
                    rule_type=rule_type,
                    task_name=var["task_name"],
                )
                
                test_code = TASK_EXISTENCE_TEST_TEMPLATE.format(
                    package=package,
                    short_name=short_name,
                    rule_type=rule_type,
                    task_name=var["task_name"],
                )
                
                rules.append(SyntheticRule(
                    instruction=instruction + (" (warn only)" if rule_type == "warn" else ""),
                    package=package,
                    rule_type=rule_type,
                    attestation_schema="""- .statement.predicate.buildConfig.tasks[]
  - predicate.buildConfig.tasks[].name
  - predicate.buildConfig.tasks[].ref""",
                    available_helpers="""- name: lib.pipelinerun_attestations
  description: Returns all PipelineRun attestations
- name: tekton.tasks
  description: Returns all tasks from attestation
- name: lib.result_helper
  description: Creates result with metadata chain""",
                    rule_data_keys="",
                    analysis=f"""- Data source: PipelineRun attestations
- Iterates over all tasks in the pipeline
- Check: Verifies task with name '{var["task_name"]}' exists
- Output: {rule_type.capitalize()} result if task not found""",
                    rule_code=rule_code,
                    test_code=test_code,
                ))
    
    return rules


def generate_task_param_rules() -> List[SyntheticRule]:
    """Generate task parameter validation rules."""
    rules = []
    
    for var in TASK_PARAM_VARIATIONS:
        for rule_type in ["deny", "warn"]:
            for instruction in var["instructions"]:
                package = f"check_{var['task_name'].replace('-', '_')}_{var['param_name'].lower()}"
                short_name = f"{var['task_name'].replace('-', '_')}_{var['param_name'].lower()}"
                
                rule_code = TASK_PARAM_TEMPLATE.format(
                    package=package,
                    title=var["title"],
                    description=var["description"],
                    short_name=short_name,
                    failure_msg=f"Task '{var['task_name']}' has invalid {var['param_name']} value: %s",
                    rule_type=rule_type,
                    task_name=var["task_name"],
                    param_name=var["param_name"],
                    check_condition=var["check_condition"],
                )
                
                test_code = TASK_PARAM_TEST_TEMPLATE.format(
                    package=package,
                    short_name=short_name,
                    rule_type=rule_type,
                    task_name=var["task_name"],
                    param_name=var["param_name"],
                    good_value=var["good_value"],
                    bad_value=var["bad_value"],
                )
                
                rules.append(SyntheticRule(
                    instruction=instruction + (" (warn only)" if rule_type == "warn" else ""),
                    package=package,
                    rule_type=rule_type,
                    attestation_schema=f"""- .statement.predicate.buildConfig.tasks[]
  - predicate.buildConfig.tasks[].name
  - predicate.buildConfig.tasks[].params[]
  - predicate.buildConfig.tasks[].params[].name
  - predicate.buildConfig.tasks[].params[].value""",
                    available_helpers="""- name: lib.pipelinerun_attestations
  description: Returns all PipelineRun attestations
- name: tekton.tasks
  description: Returns all tasks from attestation
- name: tekton.task_param
  description: Returns value of a task parameter
- name: lib.result_helper_with_term
  description: Creates result with searchable term""",
                    rule_data_keys="",
                    analysis=f"""- Data source: PipelineRun attestations
- Iterates over tasks to find '{var["task_name"]}'
- Check: Validates {var["param_name"]} parameter value
- Output: {rule_type.capitalize()} result with task name as term""",
                    rule_code=rule_code,
                    test_code=test_code,
                ))
    
    return rules


def generate_task_result_rules() -> List[SyntheticRule]:
    """Generate task result validation rules."""
    rules = []
    
    for var in TASK_RESULT_VARIATIONS:
        for rule_type in ["deny", "warn"]:
            for instruction in var["instructions"]:
                package = f"check_{var['task_name'].replace('-', '_')}_{var['result_name'].lower()}"
                short_name = f"has_{var['result_name'].lower()}"
                
                rule_code = TASK_RESULT_TEMPLATE.format(
                    package=package,
                    title=var["title"],
                    description=var["description"],
                    short_name=short_name,
                    failure_msg=f"Task '{var['task_name']}' missing {var['result_name']} result",
                    rule_type=rule_type,
                    task_name=var["task_name"],
                    result_name=var["result_name"],
                )
                
                test_code = TASK_RESULT_TEST_TEMPLATE.format(
                    package=package,
                    short_name=short_name,
                    rule_type=rule_type,
                    task_name=var["task_name"],
                    result_name=var["result_name"],
                )
                
                rules.append(SyntheticRule(
                    instruction=instruction + (" (warn only)" if rule_type == "warn" else ""),
                    package=package,
                    rule_type=rule_type,
                    attestation_schema=f"""- .statement.predicate.buildConfig.tasks[]
  - predicate.buildConfig.tasks[].name
  - predicate.buildConfig.tasks[].results[]
  - predicate.buildConfig.tasks[].results[].name
  - predicate.buildConfig.tasks[].results[].value""",
                    available_helpers="""- name: lib.pipelinerun_attestations
  description: Returns all PipelineRun attestations
- name: tekton.tasks
  description: Returns all tasks from attestation
- name: tekton.task_results
  description: Returns results from a task
- name: lib.result_helper_with_term
  description: Creates result with searchable term""",
                    rule_data_keys="",
                    analysis=f"""- Data source: PipelineRun attestations
- Iterates over tasks to find '{var["task_name"]}'
- Check: Verifies '{var["result_name"]}' result exists
- Output: {rule_type.capitalize()} result with task name as term""",
                    rule_code=rule_code,
                    test_code=test_code,
                ))
    
    return rules


def generate_sbom_package_rules() -> List[SyntheticRule]:
    """Generate SBOM package validation rules."""
    rules = []
    
    for var in SBOM_PACKAGE_VARIATIONS:
        for rule_type in ["deny", "warn"]:
            for instruction in var["instructions"]:
                package = f"sbom_{var['sbom_type']}_{var['title'].lower().replace(' ', '_')[:20]}"
                short_name = var['title'].lower().replace(' ', '_')[:20]
                
                rule_code = SBOM_PACKAGE_TEMPLATE.format(
                    package=package,
                    title=var["title"],
                    description=var["description"],
                    short_name=short_name,
                    failure_msg="Package '%s' missing required field",
                    rule_type=rule_type,
                    sbom_type=var["sbom_type"],
                    package_path=var["package_path"],
                    check_condition=var["check_condition"],
                )
                
                test_code = SBOM_PACKAGE_TEST_TEMPLATE.format(
                    package=package,
                    short_name=short_name,
                    rule_type=rule_type,
                    sbom_type=var["sbom_type"],
                    good_sbom=var["good_sbom"],
                    bad_sbom=var["bad_sbom"],
                )
                
                schema_desc = "SPDX SBOM structure" if var["sbom_type"] == "spdx" else "CycloneDX SBOM structure"
                
                rules.append(SyntheticRule(
                    instruction=instruction + (" (warn only)" if rule_type == "warn" else ""),
                    package=package,
                    rule_type=rule_type,
                    attestation_schema=f"""- {schema_desc}
  - {var['package_path']}[]
  - {var['package_path']}[].name""",
                    available_helpers=f"""- name: sbom.{var['sbom_type']}_sboms
  description: Returns all {var['sbom_type'].upper()} SBOMs
- name: lib.result_helper_with_term
  description: Creates result with searchable term""",
                    rule_data_keys="",
                    analysis=f"""- Data source: {var['sbom_type'].upper()} SBOMs
- Iterates over {var['package_path']} in SBOM
- Check: {var['check_condition']}
- Output: {rule_type.capitalize()} result with package name""",
                    rule_code=rule_code,
                    test_code=test_code,
                ))
    
    return rules


def generate_image_label_rules() -> List[SyntheticRule]:
    """Generate image label validation rules."""
    rules = []
    
    for var in IMAGE_LABEL_VARIATIONS:
        for rule_type in ["deny", "warn"]:
            for instruction in var["instructions"]:
                package = f"image_label_{var['label_name'].replace('.', '_').replace('-', '_')[:20]}"
                short_name = f"has_{var['label_name'].replace('.', '_').replace('-', '_')[:15]}"
                
                rule_code = IMAGE_LABEL_TEMPLATE.format(
                    package=package,
                    title=var["title"],
                    description=var["description"],
                    short_name=short_name,
                    failure_msg=f"Image missing required label: {var['label_name']}",
                    rule_type=rule_type,
                    label_name=var["label_name"],
                )
                
                test_code = IMAGE_LABEL_TEST_TEMPLATE.format(
                    package=package,
                    short_name=short_name,
                    rule_type=rule_type,
                    label_name=var["label_name"],
                    label_value=var["label_value"],
                )
                
                rules.append(SyntheticRule(
                    instruction=instruction + (" (warn only)" if rule_type == "warn" else ""),
                    package=package,
                    rule_type=rule_type,
                    attestation_schema="""- input.image.ref
- input.image.config.config.Labels""",
                    available_helpers="""- name: image.config
  description: Returns image configuration
- name: lib.result_helper
  description: Creates result with metadata chain""",
                    rule_data_keys="",
                    analysis=f"""- Data source: Image configuration
- Check: Verifies '{var["label_name"]}' label exists
- Output: {rule_type.capitalize()} result if label missing""",
                    rule_code=rule_code,
                    test_code=test_code,
                ))
    
    return rules


def format_stage1_example(rule: SyntheticRule) -> dict:
    """Format a synthetic rule as Stage 1 training example."""
    output_parts = [
        f"ATTESTATION_SCHEMA:\n{rule.attestation_schema}",
        f"\nAVAILABLE_HELPERS:\n{rule.available_helpers}",
    ]
    if rule.rule_data_keys:
        output_parts.append(f"\nRULE_DATA_KEYS:\n{rule.rule_data_keys}")
    
    # Add suggested metadata for Stage 2 requirements
    output_parts.append(f"\nSUGGESTED_PACKAGE: {rule.package}")
    output_parts.append(f"SUGGESTED_RULE_TYPE: {rule.rule_type}")
    
    return {
        "instruction": rule.instruction,
        "input": "Analyze the requirements and identify the attestation schema, available helpers, rule data keys, and suggest an appropriate package name and rule type (deny/warn) for this Rego rule.",
        "output": "\n".join(output_parts),
    }


def format_stage2_example(rule: SyntheticRule) -> dict:
    """Format a synthetic rule as Stage 2 training example."""
    # Build input from Stage 1 output
    context_parts = [
        f"ATTESTATION_SCHEMA:\n{rule.attestation_schema}",
        f"\nAVAILABLE_HELPERS:\n{rule.available_helpers}",
    ]
    if rule.rule_data_keys:
        context_parts.append(f"\nRULE_DATA_KEYS:\n{rule.rule_data_keys}")
    context = "\n".join(context_parts)
    
    input_text = f"REQUIREMENTS:\n- {rule.instruction}\n- Package: {rule.package}\n- Rule type: {rule.rule_type}\n\n{context}"
    
    output_text = f"""ANALYSIS:
{rule.analysis}

RULE:
```rego
{rule.rule_code}
```

TESTS:
```rego
{rule.test_code}
```"""
    
    return {
        "instruction": "Write a Rego rule that enforces the requirements below using the provided context.",
        "input": input_text,
        "output": output_text,
    }


def main():
    """Generate all synthetic rules and save to JSONL files."""
    print("=" * 60)
    print("Synthetic Rule Generator")
    print("=" * 60)
    
    # Generate all rules
    all_rules = []
    
    print("\nGenerating task existence rules...")
    task_existence = generate_task_existence_rules()
    print(f"  Generated {len(task_existence)} rules")
    all_rules.extend(task_existence)
    
    print("Generating task parameter rules...")
    task_param = generate_task_param_rules()
    print(f"  Generated {len(task_param)} rules")
    all_rules.extend(task_param)
    
    print("Generating task result rules...")
    task_result = generate_task_result_rules()
    print(f"  Generated {len(task_result)} rules")
    all_rules.extend(task_result)
    
    print("Generating SBOM package rules...")
    sbom_package = generate_sbom_package_rules()
    print(f"  Generated {len(sbom_package)} rules")
    all_rules.extend(sbom_package)
    
    print("Generating image label rules...")
    image_label = generate_image_label_rules()
    print(f"  Generated {len(image_label)} rules")
    all_rules.extend(image_label)
    
    print(f"\nTotal synthetic rules: {len(all_rules)}")
    
    # Shuffle for variety
    random.seed(42)
    random.shuffle(all_rules)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save Stage 1 examples
    stage1_path = OUTPUT_DIR / "stage1_synthetic.jsonl"
    with open(stage1_path, 'w') as f:
        for rule in all_rules:
            example = format_stage1_example(rule)
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"\nSaved Stage 1: {stage1_path}")
    
    # Save Stage 2 examples
    stage2_path = OUTPUT_DIR / "stage2_synthetic.jsonl"
    with open(stage2_path, 'w') as f:
        for rule in all_rules:
            example = format_stage2_example(rule)
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"Saved Stage 2: {stage2_path}")
    
    print("\nDone!")
    print(f"\nTo combine with existing training data:")
    print(f"  cat data/training/two_stage/stage1_train.jsonl {stage1_path} > combined_stage1.jsonl")
    print(f"  cat data/training/two_stage/stage2_train.jsonl {stage2_path} > combined_stage2.jsonl")


if __name__ == "__main__":
    main()

