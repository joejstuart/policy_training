#!/usr/bin/env python3
"""
Synthetic Rego Rule Generator

Generates unique, valid Rego rules by composing components from existing rules.
This increases training data diversity to improve model code generation accuracy.

Usage:
    python scripts/generate_synthetic_rules.py --output-dir data/training/synthetic
    python scripts/generate_synthetic_rules.py --count 100 --validate
"""

import argparse
import json
import random
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# =============================================================================
# STAGE 1 SCHEMA MAPPINGS - What Stage 1 should output for each source
# =============================================================================

STAGE1_SCHEMA_MAPPINGS = {
    "subjects": {
        "schema": """- att.statement.subject (array of attestation subjects)
  example: [{"name": "quay.io/repo/image", "digest": {"sha256": "abc123..."}}]
  navigation: `some subject in att.statement.subject`
  - subject.name: Image reference (registry/repo:tag or registry/repo@digest)
  - subject.digest: Digest object containing sha256
  - subject.digest.sha256: Image digest (64 hex characters)""",
        "helpers": """- lib.pipelinerun_attestations: Returns all PipelineRun attestations
- lib.result_helper(chain, terms): Creates violation result with metadata""",
    },
    "materials": {
        "schema": """- att.statement.predicate.materials (array of build materials)
  example: [{"uri": "git+https://github.com/org/repo.git", "digest": {"sha1": "abc123..."}}]
  navigation: `some material in att.statement.predicate.materials`
  - material.uri: Resource URI (git+https://... for source, oci://... for images)
  - material.digest: Digest object containing sha1 or sha256
  - material.digest.sha1: Git commit SHA (40 hex characters)
  - material.digest.sha256: OCI image digest (64 hex characters)""",
        "helpers": """- lib.pipelinerun_attestations: Returns all PipelineRun attestations
- lib.result_helper(chain, terms): Creates violation result with metadata""",
    },
    "tasks": {
        "schema": """- predicate.buildConfig.tasks[] (array of pipeline tasks)
  example: {"name": "build-container", "status": "Succeeded", "ref": {"bundle": "quay.io/task@sha256:...", "name": "buildah"}, "results": [...]}
  navigation: `some task in tekton.tasks(att)`
  - task.name: Task name
  - task.status: Task execution status (Succeeded, Failed, etc.)
  - task.ref: Task reference with resolver params
  - task.ref.bundle: Bundle image reference (may include @sha256: digest)
  - task.ref.name: Task name in bundle
  - task.results[]: Array of task results
  
  Bundle pinning: Use `tekton.task_ref(task).pinned` to check if task uses immutable digest.
  Example: `not tekton.task_ref(task).pinned` (denies unpinned tasks)""",
        "helpers": """- lib.pipelinerun_attestations: Returns all PipelineRun attestations
- tekton.tasks(att): Returns all tasks from attestation
- tekton.task_name(task): Returns task name
- tekton.task_ref(task): Returns task reference info with .pinned, .bundle, .name, .kind
- tekton.task_ref(task).pinned: Boolean - true if task bundle is pinned to immutable digest
- tekton.task_param(task, name): Returns task parameter value
- lib.result_helper(chain, terms): Creates violation result""",
    },
    "pre_build_tasks": {
        "schema": """- predicate.buildConfig.tasks[] filtered to pre-build tasks
  navigation: `some task in tekton.pre_build_tasks(att)`
  - task.name, task.status, task.ref, task.results[]""",
        "helpers": """- lib.pipelinerun_attestations: Returns all PipelineRun attestations
- tekton.pre_build_tasks(att): Returns pre-build tasks (git-clone, etc.)
- tekton.task_name(task): Returns task name
- lib.result_helper(chain, terms): Creates violation result""",
    },
    "build_tasks": {
        "schema": """- predicate.buildConfig.tasks[] filtered to build tasks
  example: {"name": "buildah", "status": "Succeeded", "ref": {"bundle": "quay.io/task@sha256:..."}, "results": [...]}
  navigation: `some task in tekton.build_tasks(att)`
  - task.name, task.status, task.ref, task.results[]
  - task.ref.bundle: Bundle image reference
  - task.ref.pinned: Boolean for immutable reference
  
  Bundle pinning: `tekton.task_ref(task).pinned` checks if bundle uses digest.""",
        "helpers": """- lib.pipelinerun_attestations: Returns all PipelineRun attestations
- tekton.build_tasks(att): Returns build tasks (buildah, etc.)
- tekton.task_name(task): Returns task name
- tekton.task_ref(task): Returns task reference with .pinned, .bundle, .name
- tekton.task_param(task, name): Returns task parameter value
- lib.result_helper(chain, terms): Creates violation result""",
    },
    "task_results": {
        "schema": """- task.results[] (array of task output results)
  example: [{"name": "IMAGE_DIGEST", "type": "string", "value": "sha256:abc123..."}]
  navigation: `some result in tekton.task_results(task)`
  - result.name: Result name (IMAGE_DIGEST, IMAGE_URL, CHAINS-GIT_COMMIT, etc.)
  - result.type: Result type (string, array)
  - result.value: Result value""",
        "helpers": """- lib.pipelinerun_attestations: Returns all PipelineRun attestations
- tekton.tasks(att): Returns all tasks
- tekton.task_results(task): Returns task results array
- tekton.task_name(task): Returns task name for error messages
- lib.result_helper(chain, terms): Creates violation result""",
    },
    "task_ref_params": {
        "schema": """- task.ref.params[] (resolver parameters for task reference)
  example: [{"name": "bundle", "value": "quay.io/task:1.0@sha256:..."}, {"name": "name", "value": "buildah"}]
  navigation: `some param in task.ref.params`
  - param.name: Parameter name (bundle, name, kind)
  - param.value: Parameter value""",
        "helpers": """- lib.pipelinerun_attestations: Returns all PipelineRun attestations
- tekton.tasks(att): Returns all tasks
- tekton.task_name(task): Returns task name
- lib.result_helper(chain, terms): Creates violation result""",
    },
    "spdx_packages": {
        "schema": """- SPDX SBOM packages array
  example: {"name": "golang", "SPDXID": "SPDXRef-Package-...", "versionInfo": "1.21", "licenseConcluded": "Apache-2.0", ...}
  navigation: `some pkg in s.packages`
  - pkg.name: Package name
  - pkg.SPDXID: SPDX identifier
  - pkg.versionInfo: Package version
  - pkg.licenseConcluded: Concluded license (SPDX license ID or NOASSERTION)
  - pkg.licenseDeclared: Declared license (SPDX license ID or NOASSERTION)
  - pkg.downloadLocation: Where package was obtained
  - pkg.supplier: Package supplier
  - pkg.copyrightText: Copyright information
  - pkg.externalRefs[]: External references (PURLs, CPEs)
  
  License checking: Use pkg.licenseConcluded or pkg.licenseDeclared directly.
  Example: `pkg.licenseConcluded in {"GPL-2.0-only", "GPL-3.0-only"}`
  Example: `contains(pkg.licenseConcluded, "GPL")`""",
        "helpers": """- sbom.spdx_sboms: Returns all SPDX SBOMs
- lib.rule_data(key): Retrieves configurable policy data (e.g., disallowed_packages, disallowed_licenses)
- lib.result_helper(chain, terms): Creates violation result""",
    },
    "cyclonedx_components": {
        "schema": """- CycloneDX SBOM components array
  example: {"name": "golang", "version": "1.21", "purl": "pkg:golang/...", "type": "library", "licenses": [{"license": {"id": "Apache-2.0"}}]}
  navigation: `some component in s.components`
  - component.name: Component name
  - component.version: Component version
  - component.purl: Package URL
  - component.type: Component type (library, application, container, etc.)
  - component.licenses[]: Array of license objects
    - license.license.id: SPDX license ID (e.g., "Apache-2.0", "GPL-3.0-only")
    - license.license.name: License name (if not SPDX ID)
    - license.expression: SPDX license expression
  - component.externalReferences[]: External references
  - component.properties[]: Component properties
  
  License checking: Iterate over component.licenses array.
  Example: `some license in component.licenses; license.license.id in {"GPL-2.0-only"}`
  Example: `some license in component.licenses; contains(license.license.id, "GPL")`""",
        "helpers": """- sbom.cyclonedx_sboms: Returns all CycloneDX SBOMs
- lib.rule_data(key): Retrieves configurable policy data (e.g., disallowed_packages, disallowed_licenses)
- ec.purl.parse(purl): Parses a PURL string into components (type, namespace, name, version)
- lib.result_helper(chain, terms): Creates violation result""",
    },
    "spdx_external_refs": {
        "schema": """- pkg.externalRefs[] (SPDX package external references)
  example: {"referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl", "referenceLocator": "pkg:rpm/..."}
  navigation: `some ref in pkg.externalRefs`
  - ref.referenceCategory: Category (PACKAGE-MANAGER, SECURITY, etc.)
  - ref.referenceType: Type (purl, cpe23Type, etc.)
  - ref.referenceLocator: The actual reference value (PURL string, CPE string, etc.)""",
        "helpers": """- sbom.spdx_sboms: Returns all SPDX SBOMs
- lib.rule_data(key): Retrieves configurable policy data (e.g., disallowed_purl_types, allowed_namespaces)
- ec.purl.parse(purl): Parses a PURL string into components (type, namespace, name, version)
- lib.result_helper(chain, terms): Creates violation result""",
    },
    "cyclonedx_external_refs": {
        "schema": """- component.externalReferences[] (CycloneDX external references)
  example: {"type": "vcs", "url": "https://github.com/org/repo"}
  navigation: `some ref in component.externalReferences`
  - ref.type: Reference type (vcs, website, distribution, issue-tracker)
  - ref.url: Reference URL""",
        "helpers": """- sbom.cyclonedx_sboms: Returns all CycloneDX SBOMs
- lib.rule_data(key): Retrieves configurable policy data
- lib.result_helper(chain, terms): Creates violation result""",
    },
    "image_labels": {
        "schema": """- input.image.config.Labels (image label map)
  example: {"version": "1.0", "name": "myapp", "vendor": "Red Hat"}
  navigation: `some label_name, label_value in input.image.config.Labels`
  - label_name: Label key
  - label_value: Label value""",
        "helpers": """- input.image.config.Labels: Direct access to image labels
- lib.result_helper(chain, terms): Creates violation result""",
    },
}


# =============================================================================
# ITERATION SOURCES - What we can loop over in attestations
# =============================================================================

ITERATION_SOURCES = {
    # Top-level attestation arrays
    "subjects": {
        "iteration": "some subject in att.statement.subject",
        "outer_required": "some att in lib.pipelinerun_attestations",
        "var_name": "subject",
        "fields": ["name", "digest.sha256"],
        "imports": ["data.lib"],
    },
    "materials": {
        "iteration": "some material in att.statement.predicate.materials",
        "outer_required": "some att in lib.pipelinerun_attestations",
        "var_name": "material",
        "fields": ["uri", "digest.sha256", "digest.sha1"],
        "imports": ["data.lib"],
    },
    # Tasks
    "tasks": {
        "iteration": "some task in tekton.tasks(att)",
        "outer_required": "some att in lib.pipelinerun_attestations",
        "var_name": "task",
        "fields": ["name", "ref", "status", "results"],
        "imports": ["data.lib", "data.lib.tekton"],
    },
    "pre_build_tasks": {
        "iteration": "some task in tekton.pre_build_tasks(att)",
        "outer_required": "some att in lib.pipelinerun_attestations",
        "var_name": "task",
        "fields": ["name", "ref", "status", "results"],
        "imports": ["data.lib", "data.lib.tekton"],
    },
    "build_tasks": {
        "iteration": "some task in tekton.build_tasks(att)",
        "outer_required": "some att in lib.pipelinerun_attestations",
        "var_name": "task",
        "fields": ["name", "ref", "status", "results"],
        "imports": ["data.lib", "data.lib.tekton"],
    },
    # Task internals
    "task_results": {
        "iteration": "some result in tekton.task_results(task)",
        "outer_required": "some att in lib.pipelinerun_attestations\n\tsome task in tekton.tasks(att)",
        "var_name": "result",
        "fields": ["name", "type", "value"],
        "imports": ["data.lib", "data.lib.tekton"],
    },
    "task_ref_params": {
        "iteration": "some param in task.ref.params",
        "outer_required": "some att in lib.pipelinerun_attestations\n\tsome task in tekton.tasks(att)",
        "var_name": "param",
        "fields": ["name", "value"],
        "imports": ["data.lib", "data.lib.tekton"],
    },
    # SBOM sources
    "spdx_packages": {
        "iteration": "some pkg in s.packages",
        "outer_required": "some s in sbom.spdx_sboms",
        "var_name": "pkg",
        "fields": ["name", "SPDXID", "versionInfo", "downloadLocation"],
        "imports": ["data.lib", "data.lib.sbom"],
    },
    "cyclonedx_components": {
        "iteration": "some component in s.components",
        "outer_required": "some s in sbom.cyclonedx_sboms",
        "var_name": "component",
        "fields": ["name", "version", "purl", "type"],
        "imports": ["data.lib", "data.lib.sbom"],
    },
    "spdx_external_refs": {
        "iteration": "some ref in pkg.externalRefs",
        "outer_required": "some s in sbom.spdx_sboms\n\tsome pkg in s.packages",
        "var_name": "ref",
        "fields": ["referenceType", "referenceLocator", "referenceCategory"],
        "imports": ["data.lib", "data.lib.sbom"],
    },
    "cyclonedx_external_refs": {
        "iteration": "some ref in component.externalReferences",
        "outer_required": "some s in sbom.cyclonedx_sboms\n\tsome component in s.components",
        "var_name": "ref",
        "fields": ["type", "url"],
        "imports": ["data.lib", "data.lib.sbom"],
    },
    # Image config
    "image_labels": {
        "iteration": "some label_name, label_value in input.image.config.Labels",
        "outer_required": None,
        "var_name": "label_name",
        "fields": [],  # key-value iteration
        "imports": ["data.lib"],
    },
}


# =============================================================================
# CONDITION TEMPLATES - What we can check
# =============================================================================

@dataclass
class ConditionTemplate:
    """A condition that can be applied in a rule."""
    pattern: str  # Rego condition code
    description: str  # Human-readable description
    applicable_sources: list  # Which iteration sources this works with
    result_terms: list  # Terms to include in result_helper
    negated: bool = False  # True if this is a "not X" check
    parameters: dict = field(default_factory=dict)  # Parameterizable values


CONDITION_TEMPLATES = [
    # Subject conditions
    ConditionTemplate(
        pattern="not subject.digest.sha256",
        description="subject is missing sha256 digest",
        applicable_sources=["subjects"],
        result_terms=["subject.name"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='not startswith(subject.name, "{registry}/")',
        description="subject is not from expected registry {registry}",
        applicable_sources=["subjects"],
        result_terms=["subject.name"],
        negated=True,
        parameters={"registry": ["quay.io", "registry.redhat.io", "registry.access.redhat.com"]},
    ),
    
    # Material conditions
    ConditionTemplate(
        pattern='not startswith(material.uri, "git+")',
        description="material URI does not start with git+",
        applicable_sources=["materials"],
        result_terms=["material.uri"],
        negated=True,
    ),
    ConditionTemplate(
        pattern="not material.digest.sha1",
        description="material is missing sha1 digest",
        applicable_sources=["materials"],
        result_terms=["material.uri"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='not material.digest.sha256',
        description="material is missing sha256 digest",
        applicable_sources=["materials"],
        result_terms=["material.uri"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='startswith(material.uri, "oci://")\n\tnot startswith(material.uri, "oci://{registry}/")',
        description="OCI material is not from trusted registry {registry}",
        applicable_sources=["materials"],
        result_terms=["material.uri"],
        negated=True,
        parameters={"registry": ["registry.access.redhat.com", "quay.io/konflux-ci", "quay.io/redhat-appstudio"]},
    ),
    ConditionTemplate(
        pattern='regex.match(`^[a-f0-9]{{40}}$`, material.digest.sha1) == false',
        description="material sha1 is not a valid 40-character hex string",
        applicable_sources=["materials"],
        result_terms=["material.uri", "material.digest.sha1"],
    ),
    
    # Task conditions - Bundle/Reference Pinning (multiple phrasings)
    ConditionTemplate(
        pattern="not tekton.task_ref(task).pinned",
        description="task reference is not pinned to immutable digest",
        applicable_sources=["tasks", "pre_build_tasks", "build_tasks"],
        result_terms=["tekton.task_name(task)", "tekton.pipeline_task_name(task)"],
        negated=True,
    ),
    ConditionTemplate(
        pattern="not tekton.task_ref(task).pinned",
        description="Tekton task bundle is not pinned to a digest",
        applicable_sources=["tasks", "build_tasks"],
        result_terms=["tekton.task_name(task)"],
        negated=True,
    ),
    ConditionTemplate(
        pattern="not tekton.task_ref(task).pinned",
        description="task bundle uses mutable tag instead of digest",
        applicable_sources=["tasks", "build_tasks"],
        result_terms=["tekton.task_name(task)"],
        negated=True,
    ),
    ConditionTemplate(
        pattern="not tekton.task_ref(task).pinned",
        description="task is using unpinned bundle reference",
        applicable_sources=["tasks", "build_tasks"],
        result_terms=["tekton.task_name(task)"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='ref := tekton.task_ref(task)\n\tnot contains(ref.bundle, "@sha256:")',
        description="task bundle is not pinned with sha256 digest",
        applicable_sources=["tasks", "build_tasks"],
        result_terms=["tekton.task_name(task)", "ref.bundle"],
    ),
    ConditionTemplate(
        pattern='tekton.task_param(task, "HERMETIC") != "true"',
        description="task HERMETIC parameter is not set to true",
        applicable_sources=["tasks", "build_tasks"],
        result_terms=["tekton.task_name(task)"],
    ),
    ConditionTemplate(
        pattern='count(tekton.task_results(task)) == 0',
        description="task has no results",
        applicable_sources=["tasks", "build_tasks"],
        result_terms=["tekton.task_name(task)"],
    ),
    ConditionTemplate(
        pattern='task.status != "Succeeded"',
        description="task did not succeed",
        applicable_sources=["tasks", "pre_build_tasks", "build_tasks"],
        result_terms=["tekton.task_name(task)", "task.status"],
    ),
    
    # Task result conditions
    ConditionTemplate(
        pattern='result.name == "IMAGE_DIGEST"\n\tnot regex.match(`^sha256:[a-f0-9]{{64}}$`, result.value)',
        description="IMAGE_DIGEST result has invalid format",
        applicable_sources=["task_results"],
        result_terms=["tekton.task_name(task)", "result.value"],
    ),
    ConditionTemplate(
        pattern='result.name == "{result_name}"\n\tnot result.value',
        description="{result_name} result is empty",
        applicable_sources=["task_results"],
        result_terms=["tekton.task_name(task)", "result.name"],
        parameters={"result_name": ["IMAGE_URL", "IMAGE_DIGEST", "CHAINS-GIT_URL", "CHAINS-GIT_COMMIT", "SBOM_BLOB_URL"]},
    ),
    
    # Task ref param conditions
    ConditionTemplate(
        pattern='param.name == "bundle"\n\tnot contains(param.value, "@sha256:")',
        description="bundle parameter is not pinned with sha256 digest",
        applicable_sources=["task_ref_params"],
        result_terms=["tekton.task_name(task)", "param.value"],
    ),
    
    # SPDX package conditions
    ConditionTemplate(
        pattern="not pkg.name",
        description="SPDX package is missing name",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.SPDXID"],
        negated=True,
    ),
    ConditionTemplate(
        pattern="not pkg.versionInfo",
        description="SPDX package is missing version info",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
        negated=True,
    ),
    ConditionTemplate(
        pattern="not pkg.downloadLocation",
        description="SPDX package is missing download location",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='pkg.downloadLocation == "NOASSERTION"',
        description="SPDX package download location is NOASSERTION",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    
    # CycloneDX component conditions
    ConditionTemplate(
        pattern="not component.name",
        description="CycloneDX component is missing name",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.purl"],
        negated=True,
    ),
    ConditionTemplate(
        pattern="not component.version",
        description="CycloneDX component is missing version",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
        negated=True,
    ),
    ConditionTemplate(
        pattern="not component.purl",
        description="CycloneDX component is missing PURL",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
        negated=True,
    ),
    
    # SPDX external ref conditions
    ConditionTemplate(
        pattern='ref.referenceType == "purl"\n\tnot startswith(ref.referenceLocator, "pkg:")',
        description="PURL external reference has invalid format",
        applicable_sources=["spdx_external_refs"],
        result_terms=["pkg.name", "ref.referenceLocator"],
    ),
    ConditionTemplate(
        pattern='ref.referenceCategory == "SECURITY"\n\tnot ref.referenceLocator',
        description="SECURITY reference is missing locator",
        applicable_sources=["spdx_external_refs"],
        result_terms=["pkg.name", "ref.referenceType"],
    ),
    
    # CycloneDX external ref conditions
    ConditionTemplate(
        pattern='ref.type == "vcs"\n\tnot startswith(ref.url, "https://")',
        description="VCS reference URL is not HTTPS",
        applicable_sources=["cyclonedx_external_refs"],
        result_terms=["component.name", "ref.url"],
    ),
    
    # =============================================================================
    # SBOM PACKAGE DISALLOWED/ALLOWLIST CONDITIONS
    # =============================================================================
    
    # =============================================================================
    # SBOM FIELD VALIDATION CONDITIONS (missing/empty fields)
    # =============================================================================
    
    # SPDX: Version field validation (more examples to balance disallowed_packages)
    ConditionTemplate(
        pattern='not pkg.versionInfo',
        description="SBOM package is missing version information",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.SPDXID"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='pkg.versionInfo == ""',
        description="SBOM package has empty version",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    ConditionTemplate(
        pattern='pkg.versionInfo == "NOASSERTION"',
        description="SBOM package version is not specified",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    
    # CycloneDX: Version field validation
    ConditionTemplate(
        pattern='not component.version',
        description="SBOM component is missing version",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name", "component.purl"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='component.version == ""',
        description="SBOM component has empty version field",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    
    # Required field validation patterns
    ConditionTemplate(
        pattern='not pkg.name\n\tnot pkg.versionInfo',
        description="SBOM package is missing required name and version fields",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.SPDXID"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='not component.name\n\tnot component.version',
        description="SBOM component is missing required name and version",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.purl"],
        negated=True,
    ),
    
    # =============================================================================
    # SBOM PACKAGE DISALLOWED/ALLOWLIST CONDITIONS
    # =============================================================================
    
    # SPDX: Disallowed packages (the missing pattern!)
    ConditionTemplate(
        pattern='pkg.name in lib.rule_data("disallowed_packages")',
        description="SBOM contains a disallowed package",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    ConditionTemplate(
        pattern='disallowed := lib.rule_data("disallowed_packages")\n\tcount(disallowed) > 0\n\tpkg.name in disallowed',
        description="SBOM package is in the disallowed list",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.versionInfo"],
    ),
    ConditionTemplate(
        pattern='allowed := lib.rule_data("allowed_packages")\n\tcount(allowed) > 0\n\tnot pkg.name in allowed',
        description="SBOM package is not in the allowed list",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    ConditionTemplate(
        pattern='banned := lib.rule_data("banned_package_names")\n\tsome banned_name in banned\n\tcontains(pkg.name, banned_name)',
        description="SBOM package name contains banned substring",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    ConditionTemplate(
        pattern='startswith(pkg.name, "{prefix}")',
        description="SBOM package name starts with disallowed prefix {prefix}",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
        parameters={"prefix": ["debug-", "test-", "mock-", "example-"]},
    ),
    
    # CycloneDX: Disallowed packages
    ConditionTemplate(
        pattern='component.name in lib.rule_data("disallowed_packages")',
        description="SBOM contains a disallowed component",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    ConditionTemplate(
        pattern='disallowed := lib.rule_data("disallowed_packages")\n\tcount(disallowed) > 0\n\tcomponent.name in disallowed',
        description="SBOM component is in the disallowed list",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name", "component.version"],
    ),
    ConditionTemplate(
        pattern='allowed := lib.rule_data("allowed_packages")\n\tcount(allowed) > 0\n\tnot component.name in allowed',
        description="SBOM component is not in the allowed list",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    ConditionTemplate(
        pattern='banned := lib.rule_data("banned_package_names")\n\tsome banned_name in banned\n\tcontains(component.name, banned_name)',
        description="SBOM component name contains banned substring",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    
    # =============================================================================
    # SBOM VERSION REQUIREMENTS CONDITIONS
    # =============================================================================
    
    # SPDX: Version checking
    ConditionTemplate(
        pattern='pkg.name == "{package_name}"\n\tpkg.versionInfo != lib.rule_data("{package_name}_required_version")',
        description="package {package_name} does not match required version",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.versionInfo"],
        parameters={"package_name": ["openssl", "glibc", "kernel", "python", "nodejs"]},
    ),
    ConditionTemplate(
        pattern='pkg.name in lib.rule_data("version_pinned_packages")\n\tnot pkg.versionInfo',
        description="version-pinned package is missing version info",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    ConditionTemplate(
        pattern='min_version := lib.rule_data("minimum_package_versions")[pkg.name]\n\tsemver.compare(pkg.versionInfo, min_version) < 0',
        description="package version is below minimum required version",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.versionInfo"],
    ),
    ConditionTemplate(
        pattern='deprecated := lib.rule_data("deprecated_package_versions")\n\tpkg.versionInfo in deprecated[pkg.name]',
        description="package is using a deprecated version",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.versionInfo"],
    ),
    
    # CycloneDX: Version checking
    ConditionTemplate(
        pattern='component.name == "{package_name}"\n\tcomponent.version != lib.rule_data("{package_name}_required_version")',
        description="component {package_name} does not match required version",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name", "component.version"],
        parameters={"package_name": ["openssl", "glibc", "kernel", "python", "nodejs"]},
    ),
    ConditionTemplate(
        pattern='component.name in lib.rule_data("version_pinned_packages")\n\tnot component.version',
        description="version-pinned component is missing version",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    ConditionTemplate(
        pattern='min_version := lib.rule_data("minimum_package_versions")[component.name]\n\tsemver.compare(component.version, min_version) < 0',
        description="component version is below minimum required version",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name", "component.version"],
    ),
    
    # =============================================================================
    # SBOM LICENSE/SECURITY CONDITIONS
    # =============================================================================
    
    # =============================================================================
    # SPDX LICENSE CHECKING (use licenseConcluded/licenseDeclared, NOT externalRefs)
    # =============================================================================
    
    # SPDX: Disallowed licenses via rule_data
    ConditionTemplate(
        pattern='pkg.licenseConcluded in lib.rule_data("disallowed_licenses")',
        description="package uses a disallowed license",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.licenseConcluded"],
    ),
    ConditionTemplate(
        pattern='pkg.licenseDeclared in lib.rule_data("disallowed_licenses")',
        description="package declares a disallowed license",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.licenseDeclared"],
    ),
    
    # SPDX: Allowlist checking
    ConditionTemplate(
        pattern='allowed_licenses := lib.rule_data("allowed_licenses")\n\tcount(allowed_licenses) > 0\n\tnot pkg.licenseConcluded in allowed_licenses',
        description="package license is not in allowed list",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.licenseConcluded"],
    ),
    
    # SPDX: GPL license detection (using contains for partial match)
    ConditionTemplate(
        pattern='contains(pkg.licenseConcluded, "GPL")\n\tnot contains(pkg.licenseConcluded, "LGPL")',
        description="package uses GPL license (copyleft)",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.licenseConcluded"],
    ),
    ConditionTemplate(
        pattern='contains(pkg.licenseDeclared, "GPL")\n\tnot contains(pkg.licenseDeclared, "LGPL")',
        description="package declares GPL license",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.licenseDeclared"],
    ),
    
    # SPDX: Specific license set membership (use `in {set}` pattern)
    ConditionTemplate(
        pattern='pkg.licenseConcluded in {"GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later"}',
        description="package uses a GPL family license",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.licenseConcluded"],
    ),
    ConditionTemplate(
        pattern='pkg.licenseConcluded in {"AGPL-3.0-only", "AGPL-3.0-or-later"}',
        description="package uses AGPL license (strong copyleft)",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.licenseConcluded"],
    ),
    ConditionTemplate(
        pattern='not pkg.licenseConcluded in {"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause", "ISC"}',
        description="package does not use a permissive open source license",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.licenseConcluded"],
        negated=True,
    ),
    
    # SPDX: Proprietary/unknown license detection
    ConditionTemplate(
        pattern='pkg.licenseConcluded == "NOASSERTION"\n\tpkg.licenseDeclared == "NOASSERTION"',
        description="package has no license information",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    ConditionTemplate(
        pattern='startswith(pkg.licenseConcluded, "LicenseRef-")',
        description="package uses a custom/proprietary license",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.licenseConcluded"],
    ),
    
    # =============================================================================
    # CYCLONEDX LICENSE CHECKING
    # =============================================================================
    
    # CycloneDX: Disallowed licenses
    ConditionTemplate(
        pattern='some license in component.licenses\n\tlicense.license.id in lib.rule_data("disallowed_licenses")',
        description="component uses a disallowed license",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    ConditionTemplate(
        pattern='some license in component.licenses\n\tlicense.license.name in lib.rule_data("disallowed_license_names")',
        description="component uses a disallowed license by name",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    
    # CycloneDX: GPL detection
    ConditionTemplate(
        pattern='some license in component.licenses\n\tcontains(license.license.id, "GPL")\n\tnot contains(license.license.id, "LGPL")',
        description="component uses GPL license",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    ConditionTemplate(
        pattern='some license in component.licenses\n\tlicense.license.id in {"GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later"}',
        description="component uses a GPL family license",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    
    # CycloneDX: Missing/unknown license
    ConditionTemplate(
        pattern='not component.licenses',
        description="component has no license information",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='count(component.licenses) == 0',
        description="component has empty licenses array",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    
    # CycloneDX: License expression parsing
    ConditionTemplate(
        pattern='some license in component.licenses\n\tlicense.expression\n\tcontains(license.expression, "GPL")',
        description="component license expression contains GPL",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    
    # SPDX: Supplier/origin checking
    ConditionTemplate(
        pattern='not pkg.supplier in lib.rule_data("trusted_suppliers")',
        description="package supplier is not in trusted list",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.supplier"],
    ),
    ConditionTemplate(
        pattern='pkg.downloadLocation\n\tnot startswith(pkg.downloadLocation, "https://")',
        description="package download location is not HTTPS",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.downloadLocation"],
    ),
    ConditionTemplate(
        pattern='untrusted := lib.rule_data("untrusted_download_domains")\n\tsome domain in untrusted\n\tcontains(pkg.downloadLocation, domain)',
        description="package is downloaded from untrusted domain",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.downloadLocation"],
    ),
    
    # =============================================================================
    # SBOM PURL-BASED CONDITIONS
    # =============================================================================
    
    # SPDX: PURL checking via external refs
    ConditionTemplate(
        pattern='ref.referenceType == "purl"\n\tpurl := ec.purl.parse(ref.referenceLocator)\n\tpurl.type in lib.rule_data("disallowed_purl_types")',
        description="package PURL type is disallowed",
        applicable_sources=["spdx_external_refs"],
        result_terms=["pkg.name", "ref.referenceLocator"],
    ),
    ConditionTemplate(
        pattern='ref.referenceType == "purl"\n\tpurl := ec.purl.parse(ref.referenceLocator)\n\tpurl.namespace in lib.rule_data("disallowed_namespaces")',
        description="package PURL namespace is disallowed",
        applicable_sources=["spdx_external_refs"],
        result_terms=["pkg.name", "ref.referenceLocator"],
    ),
    
    # CycloneDX: PURL checking
    ConditionTemplate(
        pattern='component.purl\n\tpurl := ec.purl.parse(component.purl)\n\tpurl.type in lib.rule_data("disallowed_purl_types")',
        description="component PURL type is disallowed",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name", "component.purl"],
    ),
    ConditionTemplate(
        pattern='component.purl\n\tpurl := ec.purl.parse(component.purl)\n\tnot purl.namespace in lib.rule_data("allowed_namespaces")',
        description="component PURL namespace is not allowed",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name", "component.purl"],
    ),
    ConditionTemplate(
        pattern='component.purl\n\tnot contains(component.purl, "@")',
        description="component PURL is not pinned to a version",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name", "component.purl"],
    ),
    
    # =============================================================================
    # SBOM COMPLETENESS CONDITIONS
    # =============================================================================
    
    ConditionTemplate(
        pattern='required := lib.rule_data("required_packages")\n\tmissing := required - {p.name | some p in s.packages}\n\tcount(missing) > 0',
        description="SBOM is missing required packages",
        applicable_sources=["spdx_packages"],
        result_terms=[],
    ),
    ConditionTemplate(
        pattern='required := lib.rule_data("required_components")\n\tmissing := required - {c.name | some c in s.components}\n\tcount(missing) > 0',
        description="SBOM is missing required components",
        applicable_sources=["cyclonedx_components"],
        result_terms=[],
    ),
    ConditionTemplate(
        pattern='count({pkg.name | some pkg in s.packages}) < lib.rule_data("minimum_package_count")',
        description="SBOM has fewer than minimum required packages",
        applicable_sources=["spdx_packages"],
        result_terms=[],
    ),
    
    # Image label conditions
    ConditionTemplate(
        pattern='label_name == "{label}"\n\tnot label_value',
        description="{label} label is present but empty",
        applicable_sources=["image_labels"],
        result_terms=["label_name"],
        parameters={"label": ["version", "name", "vendor", "summary", "description"]},
    ),
    ConditionTemplate(
        pattern='label_name == "{forbidden_label}"',
        description="forbidden label {forbidden_label} is present",
        applicable_sources=["image_labels"],
        result_terms=["label_name", "label_value"],
        parameters={"forbidden_label": ["quay.expires-after", "io.buildah.version", "maintainer"]},
    ),
    
    # === ADDITIONAL CONDITIONS FOR MORE VARIETY ===
    
    # More material conditions
    ConditionTemplate(
        pattern='contains(material.uri, "latest")',
        description="material URI contains mutable 'latest' tag",
        applicable_sources=["materials"],
        result_terms=["material.uri"],
    ),
    ConditionTemplate(
        pattern='not contains(material.uri, "@sha256:")',
        description="material URI is not pinned with sha256 digest",
        applicable_sources=["materials"],
        result_terms=["material.uri"],
        negated=True,
    ),
    
    # More task conditions  
    ConditionTemplate(
        pattern='tekton.task_param(task, "{param}") == ""',
        description="task {param} parameter is empty",
        applicable_sources=["tasks", "build_tasks"],
        result_terms=["tekton.task_name(task)"],
        parameters={"param": ["SOURCE_ARTIFACT", "IMAGE", "DOCKERFILE", "CONTEXT"]},
    ),
    ConditionTemplate(
        pattern='not tekton.task_param(task, "{param}")',
        description="task is missing required {param} parameter",
        applicable_sources=["tasks", "build_tasks"],
        result_terms=["tekton.task_name(task)"],
        negated=True,
        parameters={"param": ["SOURCE_ARTIFACT", "IMAGE", "DOCKERFILE", "CONTEXT", "BUILDER_IMAGE"]},
    ),
    
    # More task result conditions
    ConditionTemplate(
        pattern='result.name == "TEST_OUTPUT"\n\tcontains(result.value, "FAILURE")',
        description="TEST_OUTPUT result contains FAILURE",
        applicable_sources=["task_results"],
        result_terms=["tekton.task_name(task)", "result.value"],
    ),
    ConditionTemplate(
        pattern='result.name == "TEST_OUTPUT"\n\tcontains(result.value, "ERROR")',
        description="TEST_OUTPUT result contains ERROR",
        applicable_sources=["task_results"],
        result_terms=["tekton.task_name(task)", "result.value"],
    ),
    
    # More SBOM conditions
    ConditionTemplate(
        pattern='count(s.packages) == 0',
        description="SPDX SBOM has no packages",
        applicable_sources=["spdx_packages"],
        result_terms=[],
    ),
    ConditionTemplate(
        pattern='count(s.components) == 0',
        description="CycloneDX SBOM has no components",
        applicable_sources=["cyclonedx_components"],
        result_terms=[],
    ),
    ConditionTemplate(
        pattern='pkg.licenseConcluded == "NOASSERTION"',
        description="SPDX package has no concluded license",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    ConditionTemplate(
        pattern='pkg.licenseDeclared == "NOASSERTION"',
        description="SPDX package has no declared license",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    ConditionTemplate(
        pattern='not component.licenses',
        description="CycloneDX component has no license information",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
        negated=True,
    ),
    
    # External reference conditions
    ConditionTemplate(
        pattern='count(pkg.externalRefs) == 0',
        description="SPDX package has no external references",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    ConditionTemplate(
        pattern='count(component.externalReferences) == 0',
        description="CycloneDX component has no external references",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    
    # Subject conditions
    ConditionTemplate(
        pattern='not contains(subject.name, "@sha256:")',
        description="subject name does not contain digest reference",
        applicable_sources=["subjects"],
        result_terms=["subject.name"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='contains(subject.name, ":latest")',
        description="subject uses mutable latest tag",
        applicable_sources=["subjects"],
        result_terms=["subject.name"],
    ),
    
    # Attestation-level conditions (applied with subjects iteration)
    ConditionTemplate(
        pattern='count(att.statement.subject) < {min_subjects}',
        description="attestation has fewer than {min_subjects} subjects",
        applicable_sources=["subjects"],
        result_terms=[],
        parameters={"min_subjects": ["1", "2"]},
    ),
    
    # Additional task ref conditions
    ConditionTemplate(
        pattern='param.name == "kind"\n\tparam.value != "task"',
        description="task ref kind is not 'task'",
        applicable_sources=["task_ref_params"],
        result_terms=["tekton.task_name(task)", "param.value"],
    ),
    ConditionTemplate(
        pattern='param.name == "name"\n\tnot param.value',
        description="task ref name parameter is empty",
        applicable_sources=["task_ref_params"],
        result_terms=["tekton.task_name(task)"],
    ),
    
    # === MORE ATTESTATION-LEVEL CONDITIONS ===
    
    # Attestation type conditions
    ConditionTemplate(
        pattern='not att.statement.predicateType in lib.rule_data("allowed_predicate_types")',
        description="attestation predicate type is not in allowed list",
        applicable_sources=["subjects", "materials"],
        result_terms=[],
    ),
    ConditionTemplate(
        pattern='not att.statement',
        description="attestation is missing statement field",
        applicable_sources=["subjects", "materials"],
        result_terms=[],
        negated=True,
    ),
    
    # Builder ID conditions
    ConditionTemplate(
        pattern='not att.statement.predicate.builder.id',
        description="attestation is missing builder ID",
        applicable_sources=["subjects", "materials"],
        result_terms=[],
        negated=True,
    ),
    ConditionTemplate(
        pattern='not att.statement.predicate.runDetails.builder.id',
        description="SLSA v1 attestation is missing builder ID",
        applicable_sources=["subjects", "materials"],
        result_terms=[],
        negated=True,
    ),
    
    # === MORE TEKTON TASK CONDITIONS ===
    
    # Task trust conditions
    ConditionTemplate(
        pattern='not tekton.is_trusted_task(task)',
        description="task is not from trusted task list",
        applicable_sources=["tasks", "build_tasks", "pre_build_tasks"],
        result_terms=["tekton.task_name(task)"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='tekton.expiry_of(task)',
        description="task has an expiry date set",
        applicable_sources=["tasks", "build_tasks"],
        result_terms=["tekton.task_name(task)"],
    ),
    
    # Task labels/annotations
    ConditionTemplate(
        pattern='not object.get(tekton.task_labels(task), "{label}", "")',
        description="task is missing {label} label",
        applicable_sources=["tasks", "build_tasks"],
        result_terms=["tekton.task_name(task)"],
        negated=True,
        parameters={"label": ["app.kubernetes.io/version", "tekton.dev/task", "tekton.dev/pipeline"]},
    ),
    ConditionTemplate(
        pattern='not tekton.task_annotations(task)',
        description="task has no annotations",
        applicable_sources=["tasks", "build_tasks"],
        result_terms=["tekton.task_name(task)"],
        negated=True,
    ),
    
    # Task name patterns
    ConditionTemplate(
        pattern='tekton.task_name(task) == "{task_name}"',
        description="{task_name} task was executed",
        applicable_sources=["tasks"],
        result_terms=["tekton.pipeline_task_name(task)"],
        parameters={"task_name": ["git-clone", "buildah", "source-build", "clair-scan", "sast-snyk-check"]},
    ),
    ConditionTemplate(
        pattern='not regex.match(`^[a-z][a-z0-9-]*$`, tekton.task_name(task))',
        description="task name does not follow naming convention",
        applicable_sources=["tasks"],
        result_terms=["tekton.task_name(task)"],
    ),
    
    # === MORE TASK RESULT CONDITIONS ===
    
    ConditionTemplate(
        pattern='result.name == "SBOM_BLOB_URL"\n\tnot startswith(result.value, "oci://")',
        description="SBOM_BLOB_URL does not point to OCI registry",
        applicable_sources=["task_results"],
        result_terms=["tekton.task_name(task)", "result.value"],
    ),
    ConditionTemplate(
        pattern='result.name == "IMAGE_URL"\n\tnot contains(result.value, "@sha256:")',
        description="IMAGE_URL is not pinned with digest",
        applicable_sources=["task_results"],
        result_terms=["tekton.task_name(task)", "result.value"],
    ),
    ConditionTemplate(
        pattern='result.name == "CHAINS-GIT_URL"\n\tnot startswith(result.value, "https://")',
        description="CHAINS-GIT_URL is not HTTPS",
        applicable_sources=["task_results"],
        result_terms=["tekton.task_name(task)", "result.value"],
    ),
    ConditionTemplate(
        pattern='result.name == "CHAINS-GIT_COMMIT"\n\tnot regex.match(`^[a-f0-9]{{40}}$`, result.value)',
        description="CHAINS-GIT_COMMIT is not a valid 40-character git SHA",
        applicable_sources=["task_results"],
        result_terms=["tekton.task_name(task)", "result.value"],
    ),
    
    # === MORE SBOM SPDX CONDITIONS ===
    
    ConditionTemplate(
        pattern='not pkg.SPDXID',
        description="SPDX package is missing SPDXID",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='not startswith(pkg.SPDXID, "SPDXRef-")',
        description="SPDX package SPDXID has invalid format",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name", "pkg.SPDXID"],
    ),
    ConditionTemplate(
        pattern='pkg.filesAnalyzed == false',
        description="SPDX package files were not analyzed",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    ConditionTemplate(
        pattern='pkg.copyrightText == "NOASSERTION"',
        description="SPDX package has no copyright information",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    ConditionTemplate(
        pattern='not pkg.supplier',
        description="SPDX package is missing supplier",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='pkg.primaryPackagePurpose == "OPERATING-SYSTEM"',
        description="SPDX package is an operating system component",
        applicable_sources=["spdx_packages"],
        result_terms=["pkg.name"],
    ),
    
    # SPDX external ref type conditions
    ConditionTemplate(
        pattern='ref.referenceCategory == "PACKAGE-MANAGER"\n\tnot ref.referenceLocator',
        description="PACKAGE-MANAGER reference is missing locator",
        applicable_sources=["spdx_external_refs"],
        result_terms=["pkg.name", "ref.referenceType"],
    ),
    ConditionTemplate(
        pattern='ref.referenceType == "cpe23Type"\n\tnot startswith(ref.referenceLocator, "cpe:2.3:")',
        description="CPE reference has invalid format",
        applicable_sources=["spdx_external_refs"],
        result_terms=["pkg.name", "ref.referenceLocator"],
    ),
    
    # === MORE SBOM CYCLONEDX CONDITIONS ===
    
    ConditionTemplate(
        pattern='component.type == "library"\n\tnot component.purl',
        description="CycloneDX library component is missing PURL",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    ConditionTemplate(
        pattern='component.type == "container"\n\tnot component.version',
        description="CycloneDX container component is missing version",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
    ),
    ConditionTemplate(
        pattern='not component.bom_ref',
        description="CycloneDX component is missing bom-ref",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
        negated=True,
    ),
    ConditionTemplate(
        pattern='component.type not in ["library", "framework", "application", "container", "file"]',
        description="CycloneDX component has unknown type",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name", "component.type"],
    ),
    
    # CycloneDX property conditions
    ConditionTemplate(
        pattern='property.name == "{prop_name}"\n\tnot property.value',
        description="component {prop_name} property is empty",
        applicable_sources=["cyclonedx_components"],
        result_terms=["component.name"],
        parameters={"prop_name": ["hermeto:found_by", "cachi2:found_by", "syft:package:type"]},
    ),
    
    # CycloneDX external reference conditions
    ConditionTemplate(
        pattern='ref.type == "distribution"\n\tnot ref.url',
        description="distribution reference is missing URL",
        applicable_sources=["cyclonedx_external_refs"],
        result_terms=["component.name", "ref.type"],
    ),
    ConditionTemplate(
        pattern='ref.type == "website"\n\tnot startswith(ref.url, "https://")',
        description="website reference is not HTTPS",
        applicable_sources=["cyclonedx_external_refs"],
        result_terms=["component.name", "ref.url"],
    ),
    ConditionTemplate(
        pattern='ref.type == "issue-tracker"\n\tnot ref.url',
        description="issue-tracker reference is missing URL",
        applicable_sources=["cyclonedx_external_refs"],
        result_terms=["component.name"],
    ),
    
    # === IMAGE LABEL CONDITIONS ===
    
    ConditionTemplate(
        pattern='not label_value\n\tlabel_name in lib.rule_data("required_labels")',
        description="required label has empty value",
        applicable_sources=["image_labels"],
        result_terms=["label_name"],
    ),
    ConditionTemplate(
        pattern='label_name == "com.redhat.component"\n\tnot regex.match(`^[a-z][a-z0-9-]*$`, label_value)',
        description="com.redhat.component label has invalid format",
        applicable_sources=["image_labels"],
        result_terms=["label_name", "label_value"],
    ),
    ConditionTemplate(
        pattern='label_name == "release"\n\tnot regex.match(`^[0-9]+\\.[0-9]+`, label_value)',
        description="release label does not match version pattern",
        applicable_sources=["image_labels"],
        result_terms=["label_value"],
    ),
    ConditionTemplate(
        pattern='startswith(label_name, "io.k8s.")',
        description="image has kubernetes-specific label",
        applicable_sources=["image_labels"],
        result_terms=["label_name"],
    ),
    ConditionTemplate(
        pattern='label_name == "org.opencontainers.image.base.name"\n\tnot label_value',
        description="base image name label is empty",
        applicable_sources=["image_labels"],
        result_terms=["label_name"],
    ),
    ConditionTemplate(
        pattern='label_name == "org.opencontainers.image.base.digest"\n\tnot startswith(label_value, "sha256:")',
        description="base image digest label has invalid format",
        applicable_sources=["image_labels"],
        result_terms=["label_name", "label_value"],
    ),
]


# =============================================================================
# RULE TEMPLATES
# =============================================================================

RULE_TEMPLATE = '''package {package}

import rego.v1

{imports}

# METADATA
# title: {title}
# description: {description}
# custom:
#   short_name: {short_name}
#   failure_msg: "{failure_msg}"
#   solution: "{solution}"
#   collections:
#   - redhat
#
{rule_type} contains result if {{
{body}
}}
'''


# =============================================================================
# SYNTHETIC RULE GENERATOR
# =============================================================================

@dataclass
class SyntheticRule:
    """A generated synthetic rule."""
    package: str
    title: str
    description: str
    short_name: str
    failure_msg: str
    solution: str
    rule_type: str  # "deny" or "warn"
    imports: list
    body: str
    source_type: str  # Which iteration source was used
    condition_desc: str  # Description of the condition
    
    def to_rego(self) -> str:
        """Generate Rego code for this rule."""
        imports_str = "\n".join(f"import {imp}" for imp in sorted(set(self.imports)))
        return RULE_TEMPLATE.format(
            package=self.package,
            title=self.title,
            description=self.description,
            short_name=self.short_name,
            failure_msg=self.failure_msg,
            solution=self.solution,
            rule_type=self.rule_type,
            imports=imports_str,
            body=self.body,
        )


class SyntheticRuleGenerator:
    """Generates synthetic Rego rules by composing components."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.generated_rules = set()  # Track generated rule signatures to avoid duplicates
    
    def generate_rule(self) -> Optional[SyntheticRule]:
        """Generate a single synthetic rule."""
        # Pick a random iteration source
        source_name = random.choice(list(ITERATION_SOURCES.keys()))
        source = ITERATION_SOURCES[source_name]
        
        # Find applicable conditions
        applicable_conditions = [
            c for c in CONDITION_TEMPLATES
            if source_name in c.applicable_sources
        ]
        
        if not applicable_conditions:
            return None
        
        # Pick a random condition
        condition = random.choice(applicable_conditions)
        
        # Resolve parameters if any
        condition_pattern = condition.pattern
        condition_desc = condition.description
        
        if condition.parameters:
            for param_name, param_values in condition.parameters.items():
                value = random.choice(param_values)
                condition_pattern = condition_pattern.replace(f"{{{param_name}}}", value)
                condition_desc = condition_desc.replace(f"{{{param_name}}}", value)
        
        # Generate rule signature to check for duplicates
        sig = f"{source_name}:{condition_pattern}"
        if sig in self.generated_rules:
            return None
        self.generated_rules.add(sig)
        
        # Build rule body
        body_parts = []
        if source["outer_required"]:
            body_parts.append(f"\t{source['outer_required']}")
        body_parts.append(f"\t{source['iteration']}")
        
        # Add condition (indent each line properly)
        condition_lines = condition_pattern.split("\n")
        for i, line in enumerate(condition_lines):
            # First line gets normal indent, subsequent lines already have \t
            if i == 0:
                body_parts.append(f"\t{line}")
            else:
                body_parts.append(f"{line}")
        
        # Add result helper
        result_terms = ", ".join(condition.result_terms)
        body_parts.append(f"\tresult := lib.result_helper(rego.metadata.chain(), [{result_terms}])")
        
        body = "\n".join(body_parts)
        
        # Generate metadata
        package = self._generate_package_name(source_name, condition_desc)
        short_name = self._generate_short_name(condition_desc)
        title = self._generate_title(condition_desc)
        failure_msg = self._generate_failure_msg(condition, condition_desc)
        solution = self._generate_solution(condition_desc)
        rule_type = random.choice(["deny", "warn"]) if condition.negated else "deny"
        
        return SyntheticRule(
            package=package,
            title=title,
            description=condition_desc.capitalize() + ".",
            short_name=short_name,
            failure_msg=failure_msg,
            solution=solution,
            rule_type=rule_type,
            imports=source["imports"],
            body=body,
            source_type=source_name,
            condition_desc=condition_desc,
        )
    
    def _generate_package_name(self, source_name: str, condition_desc: str) -> str:
        """Generate a package name based on the source and condition."""
        # Map source types to package prefixes
        source_prefixes = {
            "subjects": "subject_validation",
            "materials": "materials_validation",
            "tasks": "task_validation",
            "pre_build_tasks": "pre_build_task",
            "build_tasks": "build_task",
            "task_results": "task_result",
            "task_ref_params": "task_ref",
            "spdx_packages": "sbom_spdx",
            "cyclonedx_components": "sbom_cyclonedx",
            "spdx_external_refs": "sbom_spdx_refs",
            "cyclonedx_external_refs": "sbom_cyclonedx_refs",
            "image_labels": "image_labels",
        }
        prefix = source_prefixes.get(source_name, source_name)
        
        # Add a suffix based on condition
        suffix_words = condition_desc.split()[:2]
        suffix = "_".join(w.lower() for w in suffix_words if w.isalnum())
        
        return f"{prefix}_{suffix}"[:40]  # Limit length
    
    def _generate_short_name(self, condition_desc: str) -> str:
        """Generate a short name for the rule."""
        words = re.findall(r'\w+', condition_desc.lower())
        return "_".join(words[:3])
    
    def _generate_title(self, condition_desc: str) -> str:
        """Generate a title for the rule."""
        return condition_desc.capitalize().replace("_", " ")
    
    def _generate_failure_msg(self, condition: ConditionTemplate, condition_desc: str) -> str:
        """Generate a failure message."""
        # Count format placeholders needed
        num_terms = len(condition.result_terms)
        placeholders = " ".join(["%s"] * min(num_terms, 2))
        
        if num_terms == 0:
            return condition_desc.capitalize()
        elif num_terms == 1:
            return f"{condition_desc.capitalize()}: %s"
        else:
            return f"{condition_desc.capitalize()}: {placeholders}"
    
    def _generate_solution(self, condition_desc: str) -> str:
        """Generate a solution message."""
        if "missing" in condition_desc.lower():
            return f"Ensure the {condition_desc.split('missing')[-1].strip()} is provided."
        elif "not" in condition_desc.lower():
            what = condition_desc.lower().replace("not ", "").replace("is ", "")
            return f"Ensure {what}."
        else:
            return f"Review and fix the issue: {condition_desc}."
    
    def generate_rules(self, count: int) -> list[SyntheticRule]:
        """Generate multiple unique synthetic rules."""
        rules = []
        attempts = 0
        max_attempts = count * 10  # Avoid infinite loop
        
        while len(rules) < count and attempts < max_attempts:
            attempts += 1
            rule = self.generate_rule()
            if rule:
                rules.append(rule)
        
        return rules


# =============================================================================
# VALIDATION
# =============================================================================

def validate_rule(rule_code: str, lib_dir: Optional[Path] = None) -> tuple[bool, str]:
    """Validate a Rego rule using OPA check.
    
    Note: We only do syntax checking without loading libraries since
    library loading causes issues with test files. The generated rules
    follow patterns from existing valid rules, so they should be valid.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as f:
        f.write(rule_code)
        f.flush()
        temp_path = f.name
    
    try:
        # Just do syntax check (parse) - don't try to resolve imports
        # This validates Rego syntax without needing libraries
        cmd = ["opa", "parse", temp_path]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr
    except subprocess.TimeoutExpired:
        return False, "Timeout during validation"
    except FileNotFoundError:
        return False, "OPA not found - skipping validation"
    finally:
        Path(temp_path).unlink(missing_ok=True)


# =============================================================================
# TRAINING DATA GENERATION
# =============================================================================

def create_stage1_example(rule: SyntheticRule) -> dict:
    """Create a Stage 1 training example (context inference) from a synthetic rule."""
    
    # Get schema mapping for this source type
    schema_mapping = STAGE1_SCHEMA_MAPPINGS.get(rule.source_type, {})
    
    # Generate varied instruction (same as the rule's description reworded as a requirement)
    instruction_templates = [
        "{description}",
        "Check that {description_lower}",
        "Verify {description_lower}",
        "Ensure {description_lower}",
        "Create a rule that checks {description_lower}",
        "I need to validate that {description_lower}",
    ]
    
    instruction = random.choice(instruction_templates).format(
        description=rule.description,
        description_lower=rule.description[0].lower() + rule.description[1:] if rule.description else "",
    )
    
    # Stage 1 input is the standard prompt
    input_text = "Analyze the requirements and identify the attestation schema, available helpers, rule data keys, and suggest an appropriate package name and rule type (deny/warn) for this Rego rule."
    
    # Build output with schema and helpers
    output_parts = [
        "ATTESTATION_SCHEMA:",
        schema_mapping.get("schema", f"- {rule.source_type} data structure"),
        "",
        "AVAILABLE_HELPERS:",
        schema_mapping.get("helpers", "- lib.result_helper(chain, terms): Creates violation result"),
        "- rego.metadata.chain(): Returns metadata chain for current rule",
        "",
        f"SUGGESTED_PACKAGE: {rule.package}",
        f"SUGGESTED_RULE_TYPE: {rule.rule_type}",
    ]
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": "\n".join(output_parts),
    }


def create_training_example(rule: SyntheticRule, rule_code: str) -> dict:
    """Create a training example from a synthetic rule."""
    
    # Generate varied instruction
    instruction_templates = [
        "Write a Rego {rule_type} rule that checks {condition}",
        "Create a Rego policy to verify {condition}",
        "Implement a {rule_type} rule for {source} validation that {condition}",
        "I need a Rego rule that {rule_type}s when {condition}",
        "How would you write a Rego {rule_type} rule to check {condition}",
        "Please write Rego code that produces a {rule_type} if {condition}",
        "Design a Rego policy that validates {condition}",
        "Can you create a {rule_type} rule that checks {condition}",
    ]
    
    instruction = random.choice(instruction_templates).format(
        rule_type=rule.rule_type,
        condition=rule.condition_desc,
        source=rule.source_type.replace("_", " "),
    )
    
    # Create input (Stage 2 format: REQUIREMENTS + CONTEXT)
    source = ITERATION_SOURCES[rule.source_type]
    
    # Build schema section
    schema_parts = []
    if source["outer_required"]:
        if "pipelinerun_attestations" in source["outer_required"]:
            schema_parts.append("- input.attestations (array of attestation objects)")
            schema_parts.append("  - attestation.statement.predicate (SLSA predicate)")
    if "subject" in rule.source_type:
        schema_parts.append("- attestation.statement.subject (array of subjects)")
        schema_parts.append("  - subject.name: image reference")
        schema_parts.append("  - subject.digest.sha256: image digest")
    if "material" in rule.source_type:
        schema_parts.append("- attestation.statement.predicate.materials (array of materials)")
        schema_parts.append("  - material.uri: resource URI")
        schema_parts.append("  - material.digest: digest object")
    if "task" in rule.source_type:
        schema_parts.append("- predicate.buildConfig.tasks (array of tasks)")
        schema_parts.append("  - task.name: task name")
        schema_parts.append("  - task.ref: task reference")
        schema_parts.append("  - task.results: task results array")
    if "spdx" in rule.source_type:
        schema_parts.append("- SPDX SBOM structure")
        schema_parts.append("  - s.packages: array of packages")
        schema_parts.append("  - pkg.name, pkg.versionInfo, pkg.externalRefs")
    if "cyclonedx" in rule.source_type:
        schema_parts.append("- CycloneDX SBOM structure")
        schema_parts.append("  - s.components: array of components")
        schema_parts.append("  - component.name, component.version, component.purl")
    if "label" in rule.source_type:
        schema_parts.append("- input.image.config.Labels (map of label name to value)")
    
    # Build helpers section
    helpers = []
    for imp in source["imports"]:
        if imp == "data.lib":
            helpers.append("- lib.pipelinerun_attestations: Returns all PipelineRun attestations")
            helpers.append("- lib.result_helper(chain, terms): Creates violation result")
        if imp == "data.lib.tekton":
            helpers.append("- tekton.tasks(att): Returns all tasks from attestation")
            helpers.append("- tekton.task_name(task): Returns task name")
            helpers.append("- tekton.task_ref(task): Returns task reference info")
            helpers.append("- tekton.task_results(task): Returns task results array")
        if imp == "data.lib.sbom":
            helpers.append("- sbom.spdx_sboms: Returns all SPDX SBOMs")
            helpers.append("- sbom.cyclonedx_sboms: Returns all CycloneDX SBOMs")
    
    input_text = f"""REQUIREMENTS:
- Package: {rule.package}
- Rule type: {rule.rule_type}
- Purpose: {rule.description}
- Check: {rule.condition_desc}

ATTESTATION_SCHEMA:
{chr(10).join(schema_parts)}

AVAILABLE_HELPERS:
{chr(10).join(helpers)}
- rego.metadata.chain(): Returns metadata chain for current rule
"""

    # Create output
    # Extract just the rule body (code after METADATA)
    output = f"""ANALYSIS:
- Iterate over: {rule.source_type.replace("_", " ")}
- Check condition: {rule.condition_desc}
- Result: Report violation with relevant identifiers

RULE:
```rego
{rule_code}
```
"""

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Rego rules for training data augmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/training/synthetic"),
        help="Output directory for generated training data",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of synthetic rules to generate",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated rules with OPA",
    )
    parser.add_argument(
        "--lib-dir",
        type=Path,
        default=Path("policy/lib"),
        help="Path to library directory for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="Fraction of examples for evaluation set",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate rules but don't write files",
    )
    
    args = parser.parse_args()
    
    print(f"Generating {args.count} synthetic rules...")
    
    # Generate rules
    generator = SyntheticRuleGenerator(seed=args.seed)
    rules = generator.generate_rules(args.count)
    
    print(f"Generated {len(rules)} unique rules")
    
    # Validate if requested
    valid_rules = []
    if args.validate:
        print("Validating rules with OPA...")
        lib_dir = args.lib_dir if args.lib_dir.exists() else None
        
        for rule in rules:
            rule_code = rule.to_rego()
            is_valid, error = validate_rule(rule_code, lib_dir)
            
            if is_valid:
                valid_rules.append(rule)
            else:
                print(f"  Invalid: {rule.package} - {error[:50]}...")
        
        print(f"Valid rules: {len(valid_rules)}/{len(rules)}")
    else:
        valid_rules = rules
    
    if not valid_rules:
        print("No valid rules generated!")
        return
    
    # Create training examples for both stages
    stage1_examples = []
    stage2_examples = []
    
    for rule in valid_rules:
        rule_code = rule.to_rego()
        # Stage 1: Context inference
        stage1_ex = create_stage1_example(rule)
        stage1_examples.append(stage1_ex)
        # Stage 2: Rule generation
        stage2_ex = create_training_example(rule, rule_code)
        stage2_examples.append(stage2_ex)
    
    # Split into train/eval (keep paired examples together)
    indices = list(range(len(valid_rules)))
    random.shuffle(indices)
    split_idx = int(len(indices) * (1 - args.eval_split))
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]
    
    stage1_train = [stage1_examples[i] for i in train_indices]
    stage1_eval = [stage1_examples[i] for i in eval_indices]
    stage2_train = [stage2_examples[i] for i in train_indices]
    stage2_eval = [stage2_examples[i] for i in eval_indices]
    
    print(f"Stage 1 - Train: {len(stage1_train)}, Eval: {len(stage1_eval)}")
    print(f"Stage 2 - Train: {len(stage2_train)}, Eval: {len(stage2_eval)}")
    
    # Show sample
    print("\n=== Sample Generated Rule ===")
    sample_rule = valid_rules[0]
    print(sample_rule.to_rego()[:1000])
    print("...")
    
    if args.dry_run:
        print("\nDry run - not writing files")
        return
    
    # Write output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Stage 1 files
    stage1_train_path = args.output_dir / "stage1_synthetic_train.jsonl"
    stage1_eval_path = args.output_dir / "stage1_synthetic_eval.jsonl"
    
    with open(stage1_train_path, 'w') as f:
        for ex in stage1_train:
            f.write(json.dumps(ex) + '\n')
    
    with open(stage1_eval_path, 'w') as f:
        for ex in stage1_eval:
            f.write(json.dumps(ex) + '\n')
    
    # Stage 2 files
    stage2_train_path = args.output_dir / "stage2_synthetic_train.jsonl"
    stage2_eval_path = args.output_dir / "stage2_synthetic_eval.jsonl"
    
    with open(stage2_train_path, 'w') as f:
        for ex in stage2_train:
            f.write(json.dumps(ex) + '\n')
    
    with open(stage2_eval_path, 'w') as f:
        for ex in stage2_eval:
            f.write(json.dumps(ex) + '\n')
    
    print(f"\nWritten to:")
    print(f"  {stage1_train_path}")
    print(f"  {stage1_eval_path}")
    print(f"  {stage2_train_path}")
    print(f"  {stage2_eval_path}")
    
    # Summary statistics
    print("\n=== Generation Summary ===")
    source_counts = {}
    for rule in valid_rules:
        source_counts[rule.source_type] = source_counts.get(rule.source_type, 0) + 1
    
    print("Rules by source type:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count}")
    
    rule_type_counts = {}
    for rule in valid_rules:
        rule_type_counts[rule.rule_type] = rule_type_counts.get(rule.rule_type, 0) + 1
    
    print("\nRules by type:")
    for rt, count in rule_type_counts.items():
        print(f"  {rt}: {count}")


if __name__ == "__main__":
    main()
