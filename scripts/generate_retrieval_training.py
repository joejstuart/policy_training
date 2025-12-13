#!/usr/bin/env python3
"""Generate training data for fine-tuning the retrieval embedding model.

Creates high-quality triplets (query, positive, hard_negative) for:
1. Query-to-Schema matching
2. Query-to-Helper matching

Sources:
- Existing Stage 1 training data (instruction → expected schemas/helpers)
- Production rules (actual schema/helper usage)
- Manual curation for hard cases

Output format: JSONL compatible with sentence-transformers training.
"""

import json
import re
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from usage_miner import UsageMiner


@dataclass
class RetrievalExample:
    """A single retrieval training example."""
    query: str
    positive: str  # Text that should be retrieved
    positive_id: str  # ID of positive item (schema_id or helper_id)
    negative: str  # Hard negative - similar but wrong
    negative_id: str
    example_type: str  # "schema" or "helper"
    source: str  # Where this example came from
    domain: str  # "slsa", "sbom", "image", etc.


@dataclass
class SchemaInfo:
    """Schema field information."""
    schema_id: str
    canonical_path: str
    description: str
    keywords: List[str]
    attestation_type: str
    text: str  # Combined searchable text


@dataclass
class HelperInfo:
    """Helper function information."""
    helper_id: str
    signature: str
    description: str
    module: str
    text: str  # Combined searchable text


class RetrievalTrainingGenerator:
    """Generate training data for retrieval model fine-tuning."""
    
    # Manual query→schema mappings for hard cases
    CURATED_SCHEMA_MAPPINGS = [
        # Task bundle pinning - CRITICAL CASE
        {
            "queries": [
                "check if task bundle is pinned",
                "verify task bundles have digests",
                "task bundle pinned to sha256",
                "ensure all task bundles are pinned",
                "check task bundle digest",
                "bundle reference pinned",
                "OCI task bundle verification",
                "tekton task bundle pinned",
                "write a rule that checks if all task bundles are pinned",
                "loops over all pipelineRun attestations then checks if all task bundles are pinned",
                "policy to verify pinned task bundles",
                "ensure tekton tasks use pinned bundles",
                "validate bundle has sha256 digest",
                "check bundle immutability",
                "task bundle contains digest",
                "bundle reference includes sha256",
                "verify OCI bundle is pinned to digest",
                "check if task uses pinned bundle reference",
            ],
            "positive_path": "ref.params[*].value",
            "negative_paths": ["tasks[*].name", "tasks[*].status", "tasks[*].serviceAccountName", "buildType"],
            "domain": "slsa",
        },
        {
            "queries": [
                "task bundle reference",
                "get bundle URL from task",
                "extract bundle reference",
                "bundle resolver parameter",
                "where is the task bundle defined",
                "OCI bundle URL",
                "quay.io bundle reference",
                "bundle parameter value",
                "resolver bundle value",
            ],
            "positive_path": "ref.params[*].value",
            "negative_paths": ["ref.resolver", "tasks[*].name", "ref.params[*].name"],
            "domain": "slsa",
        },
        # Bundle resolver name parameter
        {
            "queries": [
                "bundle parameter name",
                "resolver parameter named bundle",
                "param name equals bundle",
            ],
            "positive_path": "ref.params[*].name",
            "negative_paths": ["ref.params[*].value", "tasks[*].name"],
            "domain": "slsa",
        },
        # Task results (CVE, etc.)
        {
            "queries": [
                "check task result for CVE",
                "task results contain vulnerability",
                "get CVE scan results",
                "image scan CVE results",
                "vulnerability scan output",
                "clair scan results",
                "block images with CVEs",
                "critical severity CVE",
                "high severity vulnerability",
                "REPORTS result value",
                "vulnerability scan reports",
                "clair vulnerability data",
                "image security scan",
                "CVE severity check",
                "patch available for CVE",
                "fixable vulnerability",
                "scan result contains CVE",
            ],
            "positive_path": "results[*].value",
            "negative_paths": ["materials[*].uri", "subject[*].name", "tasks[*].status"],
            "domain": "slsa",
        },
        {
            "queries": [
                "task result name",
                "result named IMAGE_DIGEST",
                "check if result exists",
                "task output name",
                "IMAGE_URL result",
                "CHAINS-GIT_COMMIT result",
                "SBOM_BLOB_URL result",
                "find result by name",
                "task produces result named",
                "check buildah result name",
            ],
            "positive_path": "results[*].name",
            "negative_paths": ["tasks[*].name", "ref.params[*].name", "buildType"],
            "domain": "slsa",
        },
        # Task status
        {
            "queries": [
                "task succeeded",
                "check task status",
                "task completed successfully",
                "task failure status",
                "verify task ran",
            ],
            "positive_path": "tasks[*].status",
            "negative_paths": ["tasks[*].name", "results[*].value"],
            "domain": "slsa",
        },
        # Task name
        {
            "queries": [
                "find task by name",
                "task named buildah",
                "specific task exists",
                "check task name",
            ],
            "positive_path": "tasks[*].name",
            "negative_paths": ["tasks[*].status", "ref.params[*].name"],
            "domain": "slsa",
        },
        # Resolver type
        {
            "queries": [
                "task resolver type",
                "bundles resolver",
                "git resolver task",
                "http resolver",
            ],
            "positive_path": "ref.resolver",
            "negative_paths": ["ref.params[*].value", "tasks[*].name"],
            "domain": "slsa",
        },
        # Materials/source
        {
            "queries": [
                "git source URI",
                "materials source",
                "source code location",
                "repository URL",
            ],
            "positive_path": "materials[*].uri",
            "negative_paths": ["subject[*].name", "results[*].value"],
            "domain": "slsa",
        },
        {
            "queries": [
                "git commit hash",
                "source commit digest",
                "materials digest",
            ],
            "positive_path": "materials[*].digest",
            "negative_paths": ["subject[*].digest", "materials[*].uri"],
            "domain": "slsa",
        },
        # Subject (built artifacts)
        {
            "queries": [
                "built image reference",
                "artifact name",
                "output image URI",
                "subject name",
            ],
            "positive_path": "subject[*].name",
            "negative_paths": ["materials[*].uri", "tasks[*].name"],
            "domain": "slsa",
        },
        {
            "queries": [
                "built image digest",
                "artifact sha256",
                "output image digest",
            ],
            "positive_path": "subject[*].digest",
            "negative_paths": ["materials[*].digest", "tasks[*].status"],
            "domain": "slsa",
        },
        # SBOM - packages
        {
            "queries": [
                "SBOM package name",
                "software package in SBOM",
                "SPDX package name",
                "component name SBOM",
            ],
            "positive_path": "packages[*].name",
            "negative_paths": ["tasks[*].name", "components[*].name"],
            "domain": "sbom",
        },
        {
            "queries": [
                "package version SBOM",
                "software version check",
                "SPDX version info",
            ],
            "positive_path": "packages[*].versionInfo",
            "negative_paths": ["packages[*].name", "components[*].version"],
            "domain": "sbom",
        },
        {
            "queries": [
                "package license SBOM",
                "SPDX license declared",
                "software license check",
            ],
            "positive_path": "packages[*].licenseDeclared",
            "negative_paths": ["packages[*].name", "components[*].licenses"],
            "domain": "sbom",
        },
        {
            "queries": [
                "package URL purl",
                "PURL reference",
                "package identifier",
            ],
            "positive_path": "externalRefs[*].referenceLocator",
            "negative_paths": ["packages[*].name", "components[*].purl"],
            "domain": "sbom",
        },
        # CycloneDX SBOM
        {
            "queries": [
                "CycloneDX component name",
                "component in SBOM",
                "software component",
            ],
            "positive_path": "components[*].name",
            "negative_paths": ["packages[*].name", "tasks[*].name"],
            "domain": "sbom",
        },
        {
            "queries": [
                "CycloneDX component version",
                "component version check",
            ],
            "positive_path": "components[*].version",
            "negative_paths": ["packages[*].versionInfo", "components[*].name"],
            "domain": "sbom",
        },
        # Statement type and predicate type
        {
            "queries": [
                "attestation type",
                "in-toto statement type",
                "predicate type",
                "SLSA attestation type",
            ],
            "positive_path": "_type",
            "negative_paths": ["predicateType", "buildType"],
            "domain": "slsa",
        },
        {
            "queries": [
                "SLSA predicate type",
                "provenance predicate URI",
                "predicate type URI",
            ],
            "positive_path": "predicateType",
            "negative_paths": ["_type", "buildType"],
            "domain": "slsa",
        },
        # Build invocation
        {
            "queries": [
                "build type",
                "pipeline or taskrun type",
                "tekton build type",
            ],
            "positive_path": "buildType",
            "negative_paths": ["predicateType", "tasks[*].name"],
            "domain": "slsa",
        },
        # Task timing
        {
            "queries": [
                "task start time",
                "when task started",
                "task startedOn timestamp",
            ],
            "positive_path": "tasks[*].startedOn",
            "negative_paths": ["tasks[*].finishedOn", "tasks[*].status"],
            "domain": "slsa",
        },
        {
            "queries": [
                "task end time",
                "when task finished",
                "task finishedOn timestamp",
            ],
            "positive_path": "tasks[*].finishedOn",
            "negative_paths": ["tasks[*].startedOn", "tasks[*].status"],
            "domain": "slsa",
        },
        # Invocation parameters
        {
            "queries": [
                "pipeline parameters",
                "build invocation parameters",
                "input parameters to build",
            ],
            "positive_path": "invocation.parameters",
            "negative_paths": ["ref.params[*].value", "tasks[*].name"],
            "domain": "slsa",
        },
        # Config source
        {
            "queries": [
                "pipeline config source",
                "where pipeline definition came from",
                "pipeline git source",
            ],
            "positive_path": "invocation.configSource",
            "negative_paths": ["materials[*].uri", "subject[*].name"],
            "domain": "slsa",
        },
    ]
    
    # Manual query→helper mappings for hard cases
    CURATED_HELPER_MAPPINGS = [
        # Attestation access - CRITICAL
        {
            "queries": [
                "loop over attestations",
                "iterate pipelinerun attestations",
                "access SLSA attestations",
                "get all attestations",
                "for each attestation",
                "some att in attestations",
                "pipelinerun attestation access",
                "slsa provenance attestations",
                "write a rule that loops over all pipelineRun attestations",
                "access attestation predicate",
            ],
            "positive_helper": "lib.pipelinerun_attestations",
            "negative_helpers": ["lib.taskrun_attestations", "lib.results_from_tests"],
            "domain": "slsa",
        },
        # Bundle helpers - CRITICAL
        {
            "queries": [
                "get task bundle reference",
                "extract bundle from task",
                "task bundle URL",
                "bundle from task object",
                "OCI bundle reference",
            ],
            "positive_helper": "tekton.bundle",
            "negative_helpers": ["tekton.task_ref", "tekton.tasks"],
            "domain": "slsa",
        },
        {
            "queries": [
                "check unpinned bundle",
                "find unpinned task bundles",
                "bundle not pinned",
                "detect unpinned bundles",
                "tasks without pinned bundles",
                "bundles missing digest",
                "unpinned bundle references",
                "check if task bundle is pinned",
                "verify all task bundles are pinned",
            ],
            "positive_helper": "tekton.unpinned_task_bundle",
            "negative_helpers": ["tekton.bundle", "tekton.is_trusted_task"],
            "domain": "slsa",
        },
        {
            "queries": [
                "trusted task check",
                "verify task is trusted",
                "task in trusted list",
                "allowed task reference",
                "check if task is in allowed list",
            ],
            "positive_helper": "tekton.is_trusted_task",
            "negative_helpers": ["tekton.bundle", "lib.task_in_pipelinerun"],
            "domain": "slsa",
        },
        {
            "queries": [
                "empty bundle reference",
                "task has no bundle",
                "missing bundle",
                "blank bundle reference",
            ],
            "positive_helper": "tekton.empty_task_bundle_reference",
            "negative_helpers": ["tekton.bundle", "tekton.unpinned_task_bundle"],
            "domain": "slsa",
        },
        {
            "queries": [
                "unpinned task references",
                "references without digest",
                "task refs not pinned",
            ],
            "positive_helper": "tekton.unpinned_task_references",
            "negative_helpers": ["tekton.unpinned_task_bundle", "tekton.bundle"],
            "domain": "slsa",
        },
        # Task helpers
        {
            "queries": [
                "get tasks from attestation",
                "iterate over tasks",
                "access tekton tasks",
            ],
            "positive_helper": "tekton.tasks",
            "negative_helpers": ["tekton.bundle", "lib.pipelinerun_attestations"],
            "domain": "slsa",
        },
        {
            "queries": [
                "build tasks with images",
                "tasks that produce images",
                "image building tasks",
            ],
            "positive_helper": "tekton.build_tasks",
            "negative_helpers": ["tekton.tasks", "tekton.bundle"],
            "domain": "slsa",
        },
        {
            "queries": [
                "task succeeded status",
                "check task success",
                "task completed OK",
            ],
            "positive_helper": "lib.task_succeeded",
            "negative_helpers": ["lib.task_in_pipelinerun", "tekton.tasks"],
            "domain": "slsa",
        },
        # Result helpers
        {
            "queries": [
                "create violation result",
                "result with message",
                "deny result helper",
            ],
            "positive_helper": "lib.result_helper",
            "negative_helpers": ["lib.result_helper_with_severity", "lib.results_from_tests"],
            "domain": "any",
        },
        {
            "queries": [
                "result with severity",
                "violation with severity level",
                "warn result critical",
            ],
            "positive_helper": "lib.result_helper_with_severity",
            "negative_helpers": ["lib.result_helper", "lib.results_from_tests"],
            "domain": "any",
        },
        # Rule data
        {
            "queries": [
                "get rule data",
                "allowed values from config",
                "configurable rule data",
            ],
            "positive_helper": "lib.rule_data",
            "negative_helpers": ["lib.result_helper", "lib.pipelinerun_attestations"],
            "domain": "any",
        },
        # SBOM helpers
        {
            "queries": [
                "SPDX SBOM attestations",
                "access SPDX data",
                "SBOM packages",
                "loop over SPDX SBOMs",
                "iterate SPDX packages",
            ],
            "positive_helper": "sbom.spdx_sboms",
            "negative_helpers": ["sbom.cyclonedx_sboms", "lib.pipelinerun_attestations"],
            "domain": "sbom",
        },
        {
            "queries": [
                "CycloneDX SBOM",
                "access CycloneDX",
                "components in SBOM",
                "loop over CycloneDX",
                "iterate CycloneDX components",
            ],
            "positive_helper": "sbom.cyclonedx_sboms",
            "negative_helpers": ["sbom.spdx_sboms", "lib.pipelinerun_attestations"],
            "domain": "sbom",
        },
        {
            "queries": [
                "check package URL allowed",
                "PURL pattern match",
                "allowed package patterns",
            ],
            "positive_helper": "sbom.purl_allowed_patterns",
            "negative_helpers": ["sbom.spdx_sboms", "lib.rule_data"],
            "domain": "sbom",
        },
        {
            "queries": [
                "disallowed packages in SBOM",
                "check if package is blocked",
                "package not allowed",
                "SBOM disallowed packages list",
            ],
            "positive_helper": "sbom.disallowed_packages_provided",
            "negative_helpers": ["sbom.spdx_sboms", "sbom.purl_allowed_patterns"],
            "domain": "sbom",
        },
        # Image helpers - HEAVILY USED
        {
            "queries": [
                "parse image reference",
                "extract image registry",
                "get image repo",
                "parse container image URL",
                "image reference parsing",
            ],
            "positive_helper": "lib.image.parse",
            "negative_helpers": ["lib.image.str", "lib.image.equal_ref"],
            "domain": "image",
        },
        {
            "queries": [
                "image reference string",
                "full image URL",
                "image to string",
                "container image reference",
                "image ref as string",
            ],
            "positive_helper": "lib.image.str",
            "negative_helpers": ["lib.image.parse", "lib.image.equal_ref"],
            "domain": "image",
        },
        {
            "queries": [
                "compare image references",
                "check if images are equal",
                "image ref equality",
                "same image check",
            ],
            "positive_helper": "lib.image.equal_ref",
            "negative_helpers": ["lib.image.parse", "lib.image.str"],
            "domain": "image",
        },
        {
            "queries": [
                "is image index",
                "check multi-arch image",
                "image manifest list",
            ],
            "positive_helper": "lib.image.is_image_index",
            "negative_helpers": ["lib.image.parse", "lib.image.str"],
            "domain": "image",
        },
        {
            "queries": [
                "image ref from PURL",
                "convert PURL to image",
                "SBOM PURL to image reference",
            ],
            "positive_helper": "lib.sbom.image_ref_from_purl",
            "negative_helpers": ["lib.image.parse", "sbom.spdx_sboms"],
            "domain": "sbom",
        },
        {
            "queries": [
                "images with digests from tasks",
                "build task image digests",
                "get image digest from tekton task",
            ],
            "positive_helper": "lib.tekton.images_with_digests",
            "negative_helpers": ["tekton.build_tasks", "tekton.bundle"],
            "domain": "slsa",
        },
        {
            "queries": [
                "task step container image",
                "image used in task step",
                "step image reference",
            ],
            "positive_helper": "lib.tekton.task_step_image_ref",
            "negative_helpers": ["lib.tekton.images_with_digests", "tekton.bundle"],
            "domain": "slsa",
        },
        # Additional Tekton helpers
        {
            "queries": [
                "get task result value",
                "access task result",
                "task output result",
            ],
            "positive_helper": "tekton.task_result",
            "negative_helpers": ["tekton.task_param", "tekton.tasks"],
            "domain": "slsa",
        },
        {
            "queries": [
                "get task parameter",
                "task input parameter",
                "access task param",
            ],
            "positive_helper": "tekton.task_param",
            "negative_helpers": ["tekton.task_result", "tekton.bundle"],
            "domain": "slsa",
        },
        {
            "queries": [
                "pre-build tasks",
                "tasks before build",
                "pre build task list",
            ],
            "positive_helper": "tekton.pre_build_tasks",
            "negative_helpers": ["tekton.build_tasks", "tekton.tasks"],
            "domain": "slsa",
        },
        {
            "queries": [
                "pipeline task name",
                "get task name from pipeline",
                "tekton pipeline task",
            ],
            "positive_helper": "tekton.pipeline_task_name",
            "negative_helpers": ["tekton.task_name", "lib.task_in_pipelinerun"],
            "domain": "slsa",
        },
        # Additional lib helpers
        {
            "queries": [
                "result with term",
                "result helper with term",
                "violation with specific term",
            ],
            "positive_helper": "lib.result_helper_with_term",
            "negative_helpers": ["lib.result_helper", "lib.result_helper_with_severity"],
            "domain": "any",
        },
        {
            "queries": [
                "get results by name",
                "find result named",
                "results with specific name",
            ],
            "positive_helper": "lib.results_named",
            "negative_helpers": ["lib.results_from_tests", "tekton.task_result"],
            "domain": "slsa",
        },
        {
            "queries": [
                "tasks from pipelinerun",
                "get all tasks in pipeline",
                "extract tasks from pipeline run",
            ],
            "positive_helper": "lib.tasks_from_pipelinerun",
            "negative_helpers": ["tekton.tasks", "lib.pipelinerun_attestations"],
            "domain": "slsa",
        },
        {
            "queries": [
                "pipeline intention",
                "check pipeline type",
                "production vs staging pipeline",
                "release pipeline intention",
            ],
            "positive_helper": "lib.pipeline_intention_match",
            "negative_helpers": ["lib.rule_data", "lib.pipelinerun_attestations"],
            "domain": "slsa",
        },
        {
            "queries": [
                "test results",
                "results from test tasks",
                "test task output",
            ],
            "positive_helper": "lib.results_from_tests",
            "negative_helpers": ["lib.results_named", "tekton.task_result"],
            "domain": "slsa",
        },
    ]
    
    def __init__(self, kb_dir: Path, policy_dir: Path, training_dir: Path):
        """Initialize generator.
        
        Args:
            kb_dir: Path to knowledge base
            policy_dir: Path to policy directory
            training_dir: Path to existing training data
        """
        self.kb_dir = Path(kb_dir)
        self.policy_dir = Path(policy_dir)
        self.training_dir = Path(training_dir)
        
        self.schemas: Dict[str, SchemaInfo] = {}
        self.helpers: Dict[str, HelperInfo] = {}
        self.examples: List[RetrievalExample] = []
        
        self.miner = UsageMiner(policy_dir)
    
    def load_knowledge_base(self):
        """Load schemas and helpers from KB."""
        # Load schemas
        schemas_file = self.kb_dir / "schemas.jsonl"
        if schemas_file.exists():
            for line in schemas_file.read_text().strip().split('\n'):
                if not line:
                    continue
                data = json.loads(line)
                schema_id = data.get('schema_id', '')
                path = data.get('canonical_path', '')
                desc = data.get('description', '')
                keywords = data.get('keywords', [])
                att_type = data.get('attestation_type', '')
                
                # Create searchable text
                text_parts = [
                    f"Path: {path}",
                    f"Description: {desc}",
                ]
                if keywords:
                    text_parts.append(f"Keywords: {', '.join(keywords)}")
                if att_type:
                    text_parts.append(f"Attestation: {att_type}")
                
                self.schemas[schema_id] = SchemaInfo(
                    schema_id=schema_id,
                    canonical_path=path,
                    description=desc,
                    keywords=keywords,
                    attestation_type=att_type,
                    text='\n'.join(text_parts),
                )
        
        print(f"Loaded {len(self.schemas)} schemas")
        
        # Load helpers
        helpers_file = self.kb_dir / "helpers.jsonl"
        if helpers_file.exists():
            for line in helpers_file.read_text().strip().split('\n'):
                if not line:
                    continue
                data = json.loads(line)
                helper_id = data.get('id', '')
                sig = data.get('signature', '')
                desc = data.get('description', '')
                
                # Derive module from ID
                parts = helper_id.split('.')
                module = '.'.join(parts[:-1]) if len(parts) > 1 else ''
                
                # Create searchable text
                text_parts = [
                    f"Helper: {helper_id}",
                    f"Signature: {sig}",
                    f"Description: {desc}",
                ]
                
                self.helpers[helper_id] = HelperInfo(
                    helper_id=helper_id,
                    signature=sig,
                    description=desc,
                    module=module,
                    text='\n'.join(text_parts),
                )
        
        print(f"Loaded {len(self.helpers)} helpers")
    
    def _find_schema_by_path_fragment(self, fragment: str) -> Optional[SchemaInfo]:
        """Find schema by path fragment."""
        fragment_lower = fragment.lower().replace('[*]', '').replace('[]', '')
        
        for schema in self.schemas.values():
            path_lower = schema.canonical_path.lower().replace('[*]', '').replace('[]', '')
            if fragment_lower in path_lower:
                return schema
        return None
    
    def _find_helper_by_name(self, name: str) -> Optional[HelperInfo]:
        """Find helper by name or partial match."""
        # Exact match
        if name in self.helpers:
            return self.helpers[name]
        
        # Try with common prefixes
        for prefix in ['lib.', 'tekton.', 'sbom.', 'image.']:
            full_name = f"{prefix}{name}"
            if full_name in self.helpers:
                return self.helpers[full_name]
        
        # Partial match
        name_lower = name.lower()
        for helper in self.helpers.values():
            if name_lower in helper.helper_id.lower():
                return helper
        
        return None
    
    def _get_hard_negatives_for_schema(
        self, 
        positive: SchemaInfo, 
        negative_paths: List[str],
        domain: str,
    ) -> List[SchemaInfo]:
        """Get hard negative schemas."""
        negatives = []
        
        # First, try specified negative paths
        for neg_path in negative_paths:
            neg = self._find_schema_by_path_fragment(neg_path)
            if neg and neg.schema_id != positive.schema_id:
                negatives.append(neg)
        
        # If not enough, find similar schemas (same attestation type, different field)
        if len(negatives) < 2:
            for schema in self.schemas.values():
                if schema.schema_id == positive.schema_id:
                    continue
                if schema.attestation_type == positive.attestation_type:
                    if schema not in negatives:
                        negatives.append(schema)
                        if len(negatives) >= 3:
                            break
        
        return negatives
    
    def _get_hard_negatives_for_helper(
        self,
        positive: HelperInfo,
        negative_helpers: List[str],
        domain: str,
    ) -> List[HelperInfo]:
        """Get hard negative helpers."""
        negatives = []
        
        # First, try specified negatives
        for neg_name in negative_helpers:
            neg = self._find_helper_by_name(neg_name)
            if neg and neg.helper_id != positive.helper_id:
                negatives.append(neg)
        
        # If not enough, find similar helpers (same module, different function)
        if len(negatives) < 2:
            for helper in self.helpers.values():
                if helper.helper_id == positive.helper_id:
                    continue
                if helper.module == positive.module:
                    if helper not in negatives:
                        negatives.append(helper)
                        if len(negatives) >= 3:
                            break
        
        return negatives
    
    def generate_from_curated_mappings(self):
        """Generate examples from curated query→schema/helper mappings."""
        print("\nGenerating from curated mappings...")
        
        # Schema mappings
        for mapping in self.CURATED_SCHEMA_MAPPINGS:
            positive = self._find_schema_by_path_fragment(mapping["positive_path"])
            if not positive:
                print(f"  Warning: Could not find schema for {mapping['positive_path']}")
                continue
            
            negatives = self._get_hard_negatives_for_schema(
                positive, 
                mapping["negative_paths"],
                mapping["domain"],
            )
            
            if not negatives:
                print(f"  Warning: No negatives for {mapping['positive_path']}")
                continue
            
            for query in mapping["queries"]:
                for negative in negatives:
                    self.examples.append(RetrievalExample(
                        query=query,
                        positive=positive.text,
                        positive_id=positive.schema_id,
                        negative=negative.text,
                        negative_id=negative.schema_id,
                        example_type="schema",
                        source="curated",
                        domain=mapping["domain"],
                    ))
        
        print(f"  Generated {len([e for e in self.examples if e.source == 'curated' and e.example_type == 'schema'])} schema examples")
        
        # Helper mappings
        helper_count_before = len(self.examples)
        for mapping in self.CURATED_HELPER_MAPPINGS:
            positive = self._find_helper_by_name(mapping["positive_helper"])
            if not positive:
                print(f"  Warning: Could not find helper {mapping['positive_helper']}")
                continue
            
            negatives = self._get_hard_negatives_for_helper(
                positive,
                mapping["negative_helpers"],
                mapping["domain"],
            )
            
            if not negatives:
                print(f"  Warning: No negatives for {mapping['positive_helper']}")
                continue
            
            for query in mapping["queries"]:
                for negative in negatives:
                    self.examples.append(RetrievalExample(
                        query=query,
                        positive=positive.text,
                        positive_id=positive.helper_id,
                        negative=negative.text,
                        negative_id=negative.helper_id,
                        example_type="helper",
                        source="curated",
                        domain=mapping["domain"],
                    ))
        
        print(f"  Generated {len(self.examples) - helper_count_before} helper examples")
    
    def generate_from_training_data(self):
        """Extract query→schema/helper mappings from existing training data."""
        print("\nExtracting from training data...")
        
        training_files = [
            self.training_dir / "combined" / "stage1_train.jsonl",
            self.training_dir / "combined_augmented" / "stage1_train.jsonl",
        ]
        
        examples_before = len(self.examples)
        
        for train_file in training_files:
            if not train_file.exists():
                continue
            
            print(f"  Processing {train_file.name}...")
            
            for line in train_file.read_text().strip().split('\n'):
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                instruction = data.get('instruction', '')
                output = data.get('output', '')
                
                if not instruction or not output:
                    continue
                
                # Extract schema references from output
                schemas_mentioned = self._extract_schemas_from_text(output)
                helpers_mentioned = self._extract_helpers_from_text(output)
                
                # Determine domain
                domain = self._detect_domain(instruction + output)
                
                # Create schema examples
                for positive_id in schemas_mentioned:
                    positive = self.schemas.get(positive_id)
                    if not positive:
                        continue
                    
                    # Find hard negatives
                    negatives = self._get_hard_negatives_for_schema(positive, [], domain)
                    
                    for negative in negatives[:2]:  # Limit negatives per positive
                        self.examples.append(RetrievalExample(
                            query=instruction,
                            positive=positive.text,
                            positive_id=positive.schema_id,
                            negative=negative.text,
                            negative_id=negative.schema_id,
                            example_type="schema",
                            source="training_data",
                            domain=domain,
                        ))
                
                # Create helper examples
                for positive_id in helpers_mentioned:
                    positive = self.helpers.get(positive_id) or self._find_helper_by_name(positive_id)
                    if not positive:
                        continue
                    
                    negatives = self._get_hard_negatives_for_helper(positive, [], domain)
                    
                    for negative in negatives[:2]:
                        self.examples.append(RetrievalExample(
                            query=instruction,
                            positive=positive.text,
                            positive_id=positive.helper_id,
                            negative=negative.text,
                            negative_id=negative.helper_id,
                            example_type="helper",
                            source="training_data",
                            domain=domain,
                        ))
        
        print(f"  Generated {len(self.examples) - examples_before} examples from training data")
    
    def _extract_schemas_from_text(self, text: str) -> List[str]:
        """Extract schema IDs mentioned in text."""
        found = []
        
        # Look for schema paths in text
        for schema in self.schemas.values():
            # Check if canonical path appears
            path_pattern = schema.canonical_path.replace('[*]', r'\[\*\]').replace('.', r'\.')
            if re.search(path_pattern, text, re.IGNORECASE):
                found.append(schema.schema_id)
                continue
            
            # Check if simplified path appears (last 2-3 segments)
            parts = schema.canonical_path.replace('$', '').strip('.').split('.')
            if len(parts) >= 2:
                suffix = '.'.join(parts[-2:])
                if suffix.replace('[*]', '') in text:
                    found.append(schema.schema_id)
        
        return list(set(found))
    
    def _extract_helpers_from_text(self, text: str) -> List[str]:
        """Extract helper IDs mentioned in text."""
        found = []
        
        # Look for helper names
        for helper in self.helpers.values():
            # Full ID match
            if helper.helper_id in text:
                found.append(helper.helper_id)
                continue
            
            # Just function name
            func_name = helper.helper_id.split('.')[-1]
            if func_name in text and len(func_name) > 3:  # Avoid false matches
                found.append(helper.helper_id)
        
        return list(set(found))
    
    def _detect_domain(self, text: str) -> str:
        """Detect domain from text."""
        text_lower = text.lower()
        
        # SBOM-specific keywords (be more strict to avoid false matches)
        sbom_keywords = ['sbom', 'spdx', 'cyclonedx', 'software bill of materials']
        if any(kw in text_lower for kw in sbom_keywords):
            return 'sbom'
        
        # SLSA/attestation keywords (this is the main domain)
        slsa_keywords = [
            'task', 'bundle', 'attestation', 'pipeline', 'tekton', 
            'cve', 'vulnerability', 'scan', 'provenance', 'slsa',
            'pipelinerun', 'taskrun', 'predicate', 'buildconfig',
            'materials', 'subject', 'digest', 'result', 'resolver',
        ]
        if any(kw in text_lower for kw in slsa_keywords):
            return 'slsa'
        
        # Image config keywords
        if any(kw in text_lower for kw in ['image.config', 'labels', 'annotation', 'oci config']):
            return 'image'
        
        # Default to SLSA since that's the most common use case
        return 'slsa'
    
    def generate_from_production_rules(self):
        """Mine query→schema/helper mappings from production rules."""
        print("\nMining from production rules...")
        
        self.miner.scan_all_rules()
        examples_before = len(self.examples)
        
        # For each helper usage, create training examples
        for helper_name, usages in self.miner._helper_usages.items():
            helper = self._find_helper_by_name(helper_name)
            if not helper:
                continue
            
            for usage in usages[:3]:  # Limit per helper
                # Generate a query from the rule context
                query = self._generate_query_from_rule(usage.rule_name, usage.context)
                if not query:
                    continue
                
                domain = usage.attestation_type or 'any'
                if domain == 'slsa_provenance':
                    domain = 'slsa'
                
                negatives = self._get_hard_negatives_for_helper(helper, [], domain)
                
                for negative in negatives[:2]:
                    self.examples.append(RetrievalExample(
                        query=query,
                        positive=helper.text,
                        positive_id=helper.helper_id,
                        negative=negative.text,
                        negative_id=negative.helper_id,
                        example_type="helper",
                        source="production_rules",
                        domain=domain,
                    ))
        
        print(f"  Generated {len(self.examples) - examples_before} examples from production rules")
    
    def _generate_query_from_rule(self, rule_name: str, context: str) -> Optional[str]:
        """Generate a natural language query from a rule context."""
        # Extract metadata comments if present
        metadata_match = re.search(r'# description:\s*>-?\s*(.*?)(?=\n#|\npackage|\ndeny)', context, re.DOTALL)
        if metadata_match:
            desc = metadata_match.group(1).strip()
            desc = re.sub(r'\s+', ' ', desc)  # Normalize whitespace
            if len(desc) > 20:
                return desc[:200]
        
        # Fall back to rule name
        query = rule_name.replace('_', ' ')
        return f"check {query}" if query else None
    
    def generate_query_variations(self):
        """Generate variations of existing queries to improve robustness."""
        print("\nGenerating query variations...")
        
        variations_added = 0
        original_examples = list(self.examples)
        
        for example in original_examples:
            # Generate variations
            new_queries = self._create_query_variations(example.query)
            
            for new_query in new_queries:
                # Check if this variation is different enough
                if new_query.lower() != example.query.lower():
                    self.examples.append(RetrievalExample(
                        query=new_query,
                        positive=example.positive,
                        positive_id=example.positive_id,
                        negative=example.negative,
                        negative_id=example.negative_id,
                        example_type=example.example_type,
                        source=f"{example.source}_variation",
                        domain=example.domain,
                    ))
                    variations_added += 1
        
        print(f"  Added {variations_added} variations")
    
    def _create_query_variations(self, query: str) -> List[str]:
        """Create variations of a query."""
        variations = []
        
        # Prefix variations
        prefixes = [
            "check if", "verify that", "ensure", "validate",
            "write a rule to", "create a policy that",
            "how to check", "policy for",
        ]
        
        # Remove existing prefix if present
        query_core = query.lower()
        for prefix in prefixes:
            if query_core.startswith(prefix):
                query_core = query_core[len(prefix):].strip()
                break
        
        # Add new prefixes
        if query_core:
            for prefix in random.sample(prefixes, min(2, len(prefixes))):
                variations.append(f"{prefix} {query_core}")
        
        # Synonym replacements
        synonyms = {
            "check": ["verify", "ensure", "validate"],
            "task": ["tekton task", "pipeline task"],
            "bundle": ["OCI bundle", "task bundle"],
            "pinned": ["has digest", "is immutable", "has sha256"],
            "cve": ["vulnerability", "security issue"],
            "image": ["container image", "artifact"],
        }
        
        for word, replacements in synonyms.items():
            if word in query.lower():
                for replacement in replacements[:1]:
                    new_query = re.sub(rf'\b{word}\b', replacement, query, flags=re.IGNORECASE)
                    if new_query != query:
                        variations.append(new_query)
        
        return variations[:3]  # Limit variations per query
    
    def save(self, output_dir: Path):
        """Save training data in sentence-transformers format."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Deduplicate examples
        seen = set()
        unique_examples = []
        for ex in self.examples:
            key = (ex.query.lower(), ex.positive_id, ex.negative_id)
            if key not in seen:
                seen.add(key)
                unique_examples.append(ex)
        
        print(f"\nDeduplicating: {len(self.examples)} -> {len(unique_examples)} examples")
        
        # Shuffle
        random.shuffle(unique_examples)
        
        # Split into train/eval
        split_idx = int(len(unique_examples) * 0.9)
        train_examples = unique_examples[:split_idx]
        eval_examples = unique_examples[split_idx:]
        
        # Save in triplet format for sentence-transformers
        train_file = output_dir / "retrieval_train.jsonl"
        eval_file = output_dir / "retrieval_eval.jsonl"
        
        def save_examples(examples: List[RetrievalExample], filepath: Path):
            with open(filepath, 'w') as f:
                for ex in examples:
                    # Format: {"query": ..., "positive": ..., "negative": ...}
                    record = {
                        "query": ex.query,
                        "positive": ex.positive,
                        "negative": ex.negative,
                        # Metadata for analysis
                        "_positive_id": ex.positive_id,
                        "_negative_id": ex.negative_id,
                        "_type": ex.example_type,
                        "_source": ex.source,
                        "_domain": ex.domain,
                    }
                    f.write(json.dumps(record) + '\n')
        
        save_examples(train_examples, train_file)
        save_examples(eval_examples, eval_file)
        
        print(f"Saved {len(train_examples)} training examples to {train_file}")
        print(f"Saved {len(eval_examples)} eval examples to {eval_file}")
        
        # Save statistics
        stats = {
            "total_examples": len(unique_examples),
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "by_type": {
                "schema": len([e for e in unique_examples if e.example_type == "schema"]),
                "helper": len([e for e in unique_examples if e.example_type == "helper"]),
            },
            "by_source": {},
            "by_domain": {},
        }
        
        for ex in unique_examples:
            stats["by_source"][ex.source] = stats["by_source"].get(ex.source, 0) + 1
            stats["by_domain"][ex.domain] = stats["by_domain"].get(ex.domain, 0) + 1
        
        stats_file = output_dir / "stats.json"
        stats_file.write_text(json.dumps(stats, indent=2))
        print(f"Saved statistics to {stats_file}")
        
        return stats


def main():
    """Generate retrieval training data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate retrieval training data")
    parser.add_argument(
        "--kb-dir",
        default="data/knowledge_base",
        help="Path to knowledge base",
    )
    parser.add_argument(
        "--policy-dir", 
        default="policy",
        help="Path to policy directory",
    )
    parser.add_argument(
        "--training-dir",
        default="data/training",
        help="Path to existing training data",
    )
    parser.add_argument(
        "--output-dir",
        default="data/training/retrieval",
        help="Output directory for retrieval training data",
    )
    
    args = parser.parse_args()
    
    generator = RetrievalTrainingGenerator(
        kb_dir=Path(args.kb_dir),
        policy_dir=Path(args.policy_dir),
        training_dir=Path(args.training_dir),
    )
    
    # Load KB
    generator.load_knowledge_base()
    
    # Generate from multiple sources
    generator.generate_from_curated_mappings()
    generator.generate_from_training_data()
    generator.generate_from_production_rules()
    generator.generate_query_variations()
    
    # Save
    stats = generator.save(Path(args.output_dir))
    
    print("\n=== Summary ===")
    print(f"Total examples: {stats['total_examples']}")
    print(f"By type: {stats['by_type']}")
    print(f"By source: {stats['by_source']}")
    print(f"By domain: {stats['by_domain']}")


if __name__ == "__main__":
    main()

