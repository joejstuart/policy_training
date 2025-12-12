#!/usr/bin/env python3
"""
Extract field documentation from tektoncd-chains source code.

This script parses the Go source code from tektoncd-chains to extract
accurate descriptions for each field in the SLSA provenance attestation.
This provides ground-truth documentation for schema enrichment.
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FieldDoc:
    """Documentation for an attestation field."""
    json_path: str
    go_type: str
    description: str
    source_file: str
    example_values: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


# Manually curated documentation from tektoncd-chains source code
# Each entry maps a JSONPath pattern to its documentation
FIELD_DOCS: dict[str, FieldDoc] = {
    # Meta fields
    "$._type": FieldDoc(
        json_path="$._type",
        go_type="string",
        description="In-toto statement type URI. Always 'https://in-toto.io/Statement/v0.1' for SLSA v0.2",
        source_file="pkg/chains/formats/slsa/v1/pipelinerun/pipelinerun.go",
        example_values=["https://in-toto.io/Statement/v0.1"],
        keywords=["statement type", "in-toto", "attestation format"]
    ),
    "$.predicateType": FieldDoc(
        json_path="$.predicateType",
        go_type="string",
        description="SLSA provenance predicate type URI. 'https://slsa.dev/provenance/v0.2' for SLSA v0.2",
        source_file="pkg/chains/formats/slsa/v1/pipelinerun/pipelinerun.go",
        example_values=["https://slsa.dev/provenance/v0.2"],
        keywords=["predicate type", "slsa version", "provenance format"]
    ),
    
    # Statement-level fields (in-toto attestation)
    "$.subject[*].name": FieldDoc(
        json_path="$.subject[*].name",
        go_type="string",
        description="OCI image reference (URI) of the built artifact. Format: registry/repository:tag or registry/repository@sha256:digest",
        source_file="vendor/github.com/in-toto/in-toto-golang/in_toto/attestations.go",
        example_values=["quay.io/org/image@sha256:abc123...", "gcr.io/project/app:v1.0"],
        keywords=["image", "artifact", "subject", "OCI", "container", "built image", "output"]
    ),
    "$.subject[*].digest": FieldDoc(
        json_path="$.subject[*].digest",
        go_type="map[string]string",
        description="Cryptographic digest(s) of the built artifact. Key is algorithm (e.g., 'sha256'), value is hex-encoded digest",
        source_file="vendor/github.com/in-toto/in-toto-golang/in_toto/attestations.go",
        example_values=['{"sha256": "abc123..."}'],
        keywords=["digest", "sha256", "hash", "checksum", "verification", "integrity"]
    ),
    "$.subject[*].digest.sha256": FieldDoc(
        json_path="$.subject[*].digest.sha256",
        go_type="string",
        description="SHA256 digest of the built artifact (hex-encoded, 64 chars). Used to verify artifact integrity",
        source_file="vendor/github.com/in-toto/in-toto-golang/in_toto/attestations.go",
        example_values=["abc123def456..."],
        keywords=["sha256", "digest", "hash", "artifact integrity", "verification", "image digest"]
    ),
    
    # Predicate-level fields
    "$.predicate.builder.id": FieldDoc(
        json_path="$.predicate.builder.id",
        go_type="string",
        description="URI identifying the build system/platform that created the attestation (e.g., Tekton Chains)",
        source_file="pkg/chains/formats/slsa/v1/pipelinerun/pipelinerun.go",
        example_values=["https://tekton.dev/chains/v2"],
        keywords=["builder", "build system", "tekton", "chains", "provenance"]
    ),
    "$.predicate.buildType": FieldDoc(
        json_path="$.predicate.buildType",
        go_type="string",
        description="URI identifying the attestation build type (PipelineRun vs TaskRun). Rarely needed in policies",
        source_file="pkg/chains/formats/slsa/v1/pipelinerun/pipelinerun.go",
        example_values=["tekton.dev/v1beta1/PipelineRun", "tekton.dev/v1beta1/TaskRun"],
        keywords=["build type uri", "pipelinerun type", "attestation type"]
    ),
    
    # Materials - source inputs
    "$.predicate.materials[*].uri": FieldDoc(
        json_path="$.predicate.materials[*].uri",
        go_type="string",
        description="URI of a material (input) that influenced the build. Includes: git repos, step/sidecar container images, task bundle references",
        source_file="pkg/chains/formats/slsa/internal/material/material.go",
        example_values=[
            "git+https://github.com/org/repo.git",
            "oci://quay.io/image@sha256:...",
            "oci://registry.redhat.io/task@sha256:..."
        ],
        keywords=["material", "input", "source", "git", "image", "dependency", "base image"]
    ),
    "$.predicate.materials[*].digest": FieldDoc(
        json_path="$.predicate.materials[*].digest",
        go_type="map[string]string",
        description="Cryptographic digest of the material. For git: sha1 commit. For images: sha256 digest",
        source_file="pkg/chains/formats/slsa/internal/material/material.go",
        example_values=['{"sha1": "abc123..."}', '{"sha256": "def456..."}'],
        keywords=["digest", "sha256", "sha1", "commit", "pinned", "verification"]
    ),
    "$.predicate.materials[*].digest.sha256": FieldDoc(
        json_path="$.predicate.materials[*].digest.sha256",
        go_type="string",
        description="SHA256 digest of a material/input artifact (container image, OCI bundle). Used to verify input integrity",
        source_file="pkg/chains/formats/slsa/internal/material/material.go",
        example_values=["abc123def456..."],
        keywords=["sha256", "digest", "material", "input", "pinned", "image digest", "bundle digest"]
    ),
    "$.predicate.materials[*].digest.sha1": FieldDoc(
        json_path="$.predicate.materials[*].digest.sha1",
        go_type="string",
        description="SHA1 digest, typically a git commit hash for source code materials",
        source_file="pkg/chains/formats/slsa/internal/material/material.go",
        example_values=["abc123def456..."],
        keywords=["sha1", "git commit", "commit hash", "source", "revision"]
    ),
    
    # BuildConfig.Tasks - main task attestation structure
    "$.predicate.buildConfig.tasks[*].name": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].name",
        go_type="string",
        description="Name of the task within the pipeline (the PipelineTask name, not the Task definition name)",
        source_file="pkg/chains/formats/slsa/v1/pipelinerun/pipelinerun.go",
        example_values=["build-image", "git-clone", "push-image", "verify-signature"],
        keywords=["task name", "pipeline task", "step", "stage"]
    ),
    "$.predicate.buildConfig.tasks[*].after": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].after",
        go_type="[]string",
        description="List of task names that must complete before this task runs (runAfter dependencies)",
        source_file="pkg/chains/formats/slsa/v1/pipelinerun/pipelinerun.go",
        example_values=['["git-clone"]', '["build-image", "test"]'],
        keywords=["after", "dependency", "order", "sequence", "runAfter"]
    ),
    "$.predicate.buildConfig.tasks[*].status": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].status",
        go_type="string",
        description="Execution status of the task: 'Succeeded', 'Failed', or 'Running'",
        source_file="pkg/chains/formats/slsa/v1/pipelinerun/pipelinerun.go",
        example_values=["Succeeded", "Failed"],
        keywords=["status", "success", "failure", "execution", "result"]
    ),
    "$.predicate.buildConfig.tasks[*].startedOn": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].startedOn",
        go_type="time.Time",
        description="UTC timestamp when the task started execution",
        source_file="pkg/chains/formats/slsa/v1/pipelinerun/pipelinerun.go",
        example_values=["2024-01-15T10:30:00Z"],
        keywords=["start time", "timestamp", "execution", "timing"]
    ),
    "$.predicate.buildConfig.tasks[*].finishedOn": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].finishedOn",
        go_type="time.Time",
        description="UTC timestamp when the task completed execution",
        source_file="pkg/chains/formats/slsa/v1/pipelinerun/pipelinerun.go",
        example_values=["2024-01-15T10:35:00Z"],
        keywords=["finish time", "end time", "timestamp", "execution", "timing", "duration"]
    ),
    "$.predicate.buildConfig.tasks[*].serviceAccountName": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].serviceAccountName",
        go_type="string",
        description="Kubernetes ServiceAccount used to run the task (from PipelineRun.Spec.TaskRunTemplate)",
        source_file="pkg/chains/formats/slsa/v1/pipelinerun/pipelinerun.go",
        example_values=["pipeline", "build-bot", "default"],
        keywords=["service account", "identity", "rbac", "kubernetes", "permissions"]
    ),
    
    # Task Ref - reference to the task definition
    "$.predicate.buildConfig.tasks[*].ref.name": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].ref.name",
        go_type="string",
        description="Name of the referenced Task resource (only for cluster/namespace-scoped Tasks, not remote resolvers)",
        source_file="vendor/github.com/tektoncd/pipeline/pkg/apis/pipeline/v1/taskref_types.go",
        example_values=["buildah", "git-clone"],
        keywords=["task ref", "task name", "cluster task"]
    ),
    "$.predicate.buildConfig.tasks[*].ref.kind": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].ref.kind",
        go_type="string",
        description="Kind of the referenced task: 'Task' (namespaced) or custom task kind",
        source_file="vendor/github.com/tektoncd/pipeline/pkg/apis/pipeline/v1/taskref_types.go",
        example_values=["Task"],
        keywords=["kind", "task kind", "namespaced"]
    ),
    "$.predicate.buildConfig.tasks[*].ref.resolver": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].ref.resolver",
        go_type="string",
        description="Name of the resolver to fetch remote tasks: 'bundles' for OCI bundles, 'git' for git repos, 'hub' for Tekton Hub",
        source_file="vendor/github.com/tektoncd/pipeline/pkg/apis/pipeline/v1/resolver_types.go",
        example_values=["bundles", "git", "hub"],
        keywords=["resolver", "bundle", "git", "remote task", "OCI", "tekton hub"]
    ),
    "$.predicate.buildConfig.tasks[*].ref.params[*].name": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].ref.params[*].name",
        go_type="string",
        description="Parameter name for the resolver. For 'bundles': 'bundle' (OCI ref) and 'name' (task name). For 'git': 'url', 'revision', 'pathInRepo'",
        source_file="vendor/github.com/tektoncd/pipeline/pkg/apis/pipeline/v1/resolver_types.go",
        example_values=["bundle", "name", "url", "revision", "pathInRepo"],
        keywords=["param name", "resolver param", "bundle", "name", "url", "git"]
    ),
    "$.predicate.buildConfig.tasks[*].ref.params[*].value": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].ref.params[*].value",
        go_type="string|[]string|map[string]string",
        description="Parameter value for the resolver. When param.name='bundle': OCI bundle reference (e.g., quay.io/org/task@sha256:...). When param.name='name': task name within the bundle",
        source_file="vendor/github.com/tektoncd/pipeline/pkg/apis/pipeline/v1/param_types.go",
        example_values=[
            "quay.io/konflux-ci/buildah-oci-ta@sha256:abc123...",
            "buildah-oci-ta",
            "https://github.com/tektoncd/catalog.git",
            "main"
        ],
        keywords=["bundle reference", "OCI reference", "digest", "pinned", "task bundle", "git url", "revision"]
    ),
    
    # Task Results - outputs from task execution
    "$.predicate.buildConfig.tasks[*].results[*].name": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].results[*].name",
        go_type="string",
        description="Name of a result produced by the task. Check this field to verify specific task results exist or match expected names",
        source_file="vendor/github.com/tektoncd/pipeline/pkg/apis/pipeline/v1/result_types.go",
        example_values=["IMAGE_URL", "IMAGE_DIGEST", "CHAINS-GIT_URL", "CHAINS-GIT_COMMIT", "SBOM_BLOB_URL", "REPORTS"],
        keywords=["task result", "result name", "output name", "task output", "check result", "verify result", "result exists"]
    ),
    "$.predicate.buildConfig.tasks[*].results[*].type": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].results[*].type",
        go_type="string",
        description="Type of the result value: 'string', 'array', or 'object'",
        source_file="vendor/github.com/tektoncd/pipeline/pkg/apis/pipeline/v1/result_types.go",
        example_values=["string", "array", "object"],
        keywords=["result type", "string", "array", "object"]
    ),
    "$.predicate.buildConfig.tasks[*].results[*].value": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].results[*].value",
        go_type="string|[]string|map[string]string",
        description="Value of the result. For IMAGE_URL: full image reference. For IMAGE_DIGEST: sha256:... digest. For CHAINS-GIT_COMMIT: git commit SHA. For REPORTS: mapping of image digests to CVE scan report digests",
        source_file="vendor/github.com/tektoncd/pipeline/pkg/apis/pipeline/v1/result_types.go",
        example_values=[
            "quay.io/org/app:tag",
            "sha256:abc123...",
            "abc123def456...",
            '{"sha256:image_digest": "sha256:report_digest"}'
        ],
        keywords=["result value", "image url", "digest", "commit sha", "output value", "CVE", "vulnerability", "clair", "REPORTS", "scan"]
    ),
    
    # Steps - individual step execution details
    "$.predicate.buildConfig.tasks[*].steps[*].entryPoint": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].steps[*].entryPoint",
        go_type="string",
        description="The command or script executed by the step. Either the joined command array or the script content",
        source_file="pkg/chains/formats/slsa/attest/attest.go",
        example_values=["#!/bin/bash\nbuildah push...", "/ko-app/git-init"],
        keywords=["entrypoint", "command", "script", "execution"]
    ),
    "$.predicate.buildConfig.tasks[*].steps[*].arguments": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].steps[*].arguments",
        go_type="[]string",
        description="Arguments passed to the step command (step.Args from the Task definition)",
        source_file="pkg/chains/formats/slsa/attest/attest.go",
        example_values=['["--format", "oci"]'],
        keywords=["arguments", "args", "parameters", "command args"]
    ),
    "$.predicate.buildConfig.tasks[*].steps[*].environment.image": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].steps[*].environment.image",
        go_type="string",
        description="Full OCI reference of the container image used for this step (with digest from stepState.ImageID)",
        source_file="pkg/chains/formats/slsa/attest/attest.go",
        example_values=["oci://quay.io/buildah/buildah@sha256:abc..."],
        keywords=["step image", "container image", "execution environment", "base image"]
    ),
    "$.predicate.buildConfig.tasks[*].steps[*].environment.container": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].steps[*].environment.container",
        go_type="string",
        description="Name of the container/step within the pod",
        source_file="pkg/chains/formats/slsa/attest/attest.go",
        example_values=["build", "push", "clone"],
        keywords=["container name", "step name"]
    ),
    
    # Invocation - build invocation parameters
    "$.predicate.buildConfig.tasks[*].invocation.parameters": FieldDoc(
        json_path="$.predicate.buildConfig.tasks[*].invocation.parameters",
        go_type="map[string]interface{}",
        description="Input parameters passed to the task, filtered to only include non-sensitive params defined in the task spec",
        source_file="pkg/chains/formats/slsa/attest/attest.go",
        example_values=['{"IMAGE": "quay.io/org/app", "DOCKERFILE": "./Dockerfile"}'],
        keywords=["parameters", "inputs", "task params", "build inputs"]
    ),
    "$.predicate.invocation.parameters": FieldDoc(
        json_path="$.predicate.invocation.parameters",
        go_type="map[string]interface{}",
        description="Input parameters passed to the pipeline, filtered to only include non-sensitive params defined in the pipeline spec",
        source_file="pkg/chains/formats/slsa/attest/attest.go",
        example_values=['{"git-url": "https://github.com/...", "output-image": "quay.io/..."}'],
        keywords=["pipeline parameters", "inputs", "build inputs"]
    ),
    
    # Metadata
    "$.predicate.metadata.buildStartedOn": FieldDoc(
        json_path="$.predicate.metadata.buildStartedOn",
        go_type="*time.Time",
        description="UTC timestamp when the pipeline/task run started",
        source_file="pkg/chains/formats/slsa/v1/pipelinerun/pipelinerun.go",
        example_values=["2024-01-15T10:00:00Z"],
        keywords=["build start", "timestamp", "timing"]
    ),
    "$.predicate.metadata.buildFinishedOn": FieldDoc(
        json_path="$.predicate.metadata.buildFinishedOn",
        go_type="*time.Time",
        description="UTC timestamp when the pipeline/task run completed",
        source_file="pkg/chains/formats/slsa/v1/pipelinerun/pipelinerun.go",
        example_values=["2024-01-15T10:45:00Z"],
        keywords=["build finish", "timestamp", "timing", "duration"]
    ),
    "$.predicate.metadata.reproducible": FieldDoc(
        json_path="$.predicate.metadata.reproducible",
        go_type="bool",
        description="Whether the build is reproducible (set via chains.tekton.dev/reproducible annotation)",
        source_file="pkg/chains/formats/slsa/v1/pipelinerun/pipelinerun.go",
        example_values=["true", "false"],
        keywords=["reproducible", "deterministic", "hermetic"]
    ),
}


def normalize_path(path: str) -> str:
    """Normalize a JSONPath for matching."""
    # Remove array indices like [0], [1], etc. and replace with [*]
    normalized = re.sub(r'\[\d+\]', '[*]', path)
    return normalized


def find_matching_doc(json_path: str) -> Optional[FieldDoc]:
    """Find documentation for a given JSONPath."""
    normalized = normalize_path(json_path)
    
    # Direct match
    if normalized in FIELD_DOCS:
        return FIELD_DOCS[normalized]
    
    # Try without $ prefix
    if normalized.startswith("$."):
        without_prefix = normalized[2:]
        for pattern, doc in FIELD_DOCS.items():
            if pattern.endswith(without_prefix) or pattern.endswith("." + without_prefix.split(".")[-1]):
                return doc
    
    return None


def generate_schema_docs(output_path: Path):
    """Generate a JSON file with all schema field documentation."""
    docs = []
    for path, doc in FIELD_DOCS.items():
        docs.append({
            "json_path": doc.json_path,
            "go_type": doc.go_type,
            "description": doc.description,
            "source_file": doc.source_file,
            "example_values": doc.example_values,
            "keywords": doc.keywords,
        })
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(docs, f, indent=2)
    
    print(f"Generated {len(docs)} field documentation entries")
    print(f"Saved to: {output_path}")


def enrich_kb_schemas(kb_path: Path, docs_path: Path):
    """Enrich KB schemas with documentation from chains source.
    
    Only enriches SLSA provenance schemas - SBOM schemas have their own docs.
    """
    # Load docs
    with open(docs_path) as f:
        docs = json.load(f)
    
    # Build lookup by normalized path (full path match only)
    docs_by_path = {}
    for doc in docs:
        normalized = normalize_path(doc["json_path"])
        docs_by_path[normalized] = doc
    
    # Load schemas
    schemas_file = kb_path / "schemas.jsonl"
    if not schemas_file.exists():
        print(f"No schemas file found at {schemas_file}")
        return
    
    schemas = []
    with open(schemas_file) as f:
        for line in f:
            if line.strip():
                schemas.append(json.loads(line))
    
    enriched_count = 0
    skipped_count = 0
    for schema in schemas:
        att_type = schema.get("attestation_type", "")
        
        # Only enrich SLSA provenance schemas, not SBOM
        if att_type not in ("slsa_provenance_v02", "slsa_provenance"):
            skipped_count += 1
            continue
        
        path = schema.get("canonical_path", "")
        normalized = normalize_path(path)
        
        # Try to find matching doc by full path
        doc = docs_by_path.get(normalized)
        
        if doc:
            # Enrich with source documentation
            schema["description"] = doc["description"]
            schema["keywords"] = doc["keywords"]
            schema["example_values"] = doc.get("example_values", [])
            schema["use_when"] = [
                f"Check {kw}" for kw in doc["keywords"][:3]
            ]
            schema["source_doc"] = doc["source_file"]
            enriched_count += 1
    
    # Save enriched schemas
    with open(schemas_file, 'w') as f:
        for schema in schemas:
            f.write(json.dumps(schema) + "\n")
    
    print(f"Enriched {enriched_count}/{len(schemas)} schemas with source documentation")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract chains documentation for schema enrichment")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/knowledge_base/chains_field_docs.json"),
                       help="Output path for field documentation JSON")
    parser.add_argument("--enrich-kb", action="store_true",
                       help="Directly enrich KB schemas with documentation")
    parser.add_argument("--kb-path", type=Path, default=Path("data/knowledge_base"),
                       help="Path to knowledge base directory")
    
    args = parser.parse_args()
    
    # Always generate docs
    generate_schema_docs(args.output)
    
    if args.enrich_kb:
        enrich_kb_schemas(args.kb_path, args.output)


if __name__ == "__main__":
    main()

