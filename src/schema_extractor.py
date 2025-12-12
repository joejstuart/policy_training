"""Extract schemas from attestation files with canonical IDs.

This module extracts schema information from attestation JSON files,
normalizing paths and generating stable identifiers for use in retrieval.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class SchemaField:
    """A single schema field extracted from attestations."""
    
    schema_id: str              # Stable ID, e.g., "slsa_v02_task_bundle"
    canonical_path: str         # JSONPath-like, e.g., "$.predicate.buildConfig.tasks[*].ref.bundle"
    attestation_type: str       # "slsa_provenance_v02", "spdx_sbom", etc.
    field_type: str             # "string", "boolean", "array", "object", "number"
    description: str            # Human-readable description
    use_when: List[str]         # When to use this field
    example_value: Any          # Real example from fixtures
    source_file: str            # Which attestation file this came from
    aliases: List[str] = field(default_factory=list)  # Path variations
    
    def to_card(self) -> str:
        """Compact representation for retrieval (~100 tokens)."""
        parts = [
            f"Schema: {self.schema_id}",
            f"Path: {self.canonical_path}",
            f"Type: {self.field_type}",
            f"Attestation: {self.attestation_type}",
        ]
        if self.description:
            parts.append(f"Description: {self.description}")
        if self.example_value is not None:
            example_str = json.dumps(self.example_value) if not isinstance(self.example_value, str) else self.example_value
            if len(example_str) > 100:
                example_str = example_str[:97] + "..."
            parts.append(f"Example: {example_str}")
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "schema_id": self.schema_id,
            "canonical_path": self.canonical_path,
            "attestation_type": self.attestation_type,
            "field_type": self.field_type,
            "description": self.description,
            "use_when": self.use_when,
            "example_value": self.example_value,
            "source_file": self.source_file,
            "aliases": self.aliases,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaField":
        """Create from dictionary."""
        return cls(
            schema_id=data["schema_id"],
            canonical_path=data["canonical_path"],
            attestation_type=data["attestation_type"],
            field_type=data["field_type"],
            description=data.get("description", ""),
            use_when=data.get("use_when", []),
            example_value=data.get("example_value"),
            source_file=data.get("source_file", ""),
            aliases=data.get("aliases", []),
        )


class SchemaExtractor:
    """Extract schemas from attestation JSON files."""
    
    # Known attestation type patterns
    ATTESTATION_TYPES = {
        "https://slsa.dev/provenance/v0.2": "slsa_provenance_v02",
        "https://slsa.dev/provenance/v1": "slsa_provenance_v1",
        "https://in-toto.io/Statement/v0.1": "in_toto_v01",
        "https://in-toto.io/Statement/v1": "in_toto_v1",
        "https://spdx.dev/Document": "spdx_sbom",
        "https://cyclonedx.org/bom": "cyclonedx_sbom",
    }
    
    # Important paths to extract (focused on policy-relevant fields)
    PRIORITY_PATHS = {
        "slsa_provenance_v02": [
            "$.predicate.buildConfig.tasks[*].name",
            "$.predicate.buildConfig.tasks[*].ref.bundle",
            "$.predicate.buildConfig.tasks[*].ref.name",
            "$.predicate.buildConfig.tasks[*].ref.kind",
            "$.predicate.buildConfig.tasks[*].ref.resolver",
            "$.predicate.buildConfig.tasks[*].ref.params[*].name",
            "$.predicate.buildConfig.tasks[*].ref.params[*].value",
            "$.predicate.buildConfig.tasks[*].results[*].name",
            "$.predicate.buildConfig.tasks[*].results[*].value",
            "$.predicate.buildConfig.tasks[*].results[*].type",
            "$.predicate.buildConfig.tasks[*].status",
            "$.predicate.buildConfig.tasks[*].startedOn",
            "$.predicate.buildConfig.tasks[*].finishedOn",
            "$.predicate.materials[*].uri",
            "$.predicate.materials[*].digest.sha256",
            "$.subject[*].name",
            "$.subject[*].digest.sha256",
            "$.predicateType",
        ],
        "slsa_provenance_v1": [
            "$.predicate.buildDefinition.buildType",
            "$.predicate.buildDefinition.externalParameters",
            "$.predicate.buildDefinition.resolvedDependencies[*].uri",
            "$.predicate.buildDefinition.resolvedDependencies[*].digest.sha256",
            "$.predicate.runDetails.builder.id",
            "$.predicate.runDetails.metadata.invocationId",
            "$.subject[*].name",
            "$.subject[*].digest.sha256",
        ],
    }
    
    def __init__(self, attestation_dir: Path):
        """Initialize with path to attestation directory.
        
        Args:
            attestation_dir: Directory containing attestation JSON files
        """
        self.attestation_dir = Path(attestation_dir)
        self.schemas: Dict[str, SchemaField] = {}  # schema_id -> SchemaField
    
    def extract_all(self) -> Dict[str, SchemaField]:
        """Extract schemas from all attestation files.
        
        Returns:
            Dictionary mapping schema_id to SchemaField
        """
        if not self.attestation_dir.exists():
            return {}
        
        for json_file in self.attestation_dir.glob("*.json"):
            try:
                self._extract_from_file(json_file)
            except Exception as e:
                print(f"Warning: Could not extract from {json_file.name}: {e}")
        
        return self.schemas
    
    def _extract_from_file(self, json_file: Path):
        """Extract schemas from a single attestation file."""
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                return
        
        # Handle different formats
        attestations = self._normalize_attestation_format(data)
        
        for att in attestations:
            att_type = self.detect_attestation_type(att)
            if att_type:
                self._extract_paths(att, att_type, json_file.name)
    
    def _normalize_attestation_format(self, data: Any) -> List[Dict]:
        """Normalize various attestation formats to a list of attestations."""
        if isinstance(data, list):
            return data
        
        # Check for wrapped format: {"attestations": [...]}
        if isinstance(data, dict):
            if "attestations" in data:
                atts = data["attestations"]
                if isinstance(atts, list):
                    # Each may be {"statement": {...}} or direct
                    result = []
                    for a in atts:
                        if isinstance(a, dict) and "statement" in a:
                            result.append(a["statement"])
                        else:
                            result.append(a)
                    return result
            # Direct attestation
            return [data]
        
        return []
    
    def detect_attestation_type(self, data: Dict) -> Optional[str]:
        """Detect attestation type from data.
        
        Args:
            data: Attestation data dictionary
            
        Returns:
            Attestation type string or None
        """
        # Check predicateType
        predicate_type = data.get("predicateType", "")
        if predicate_type in self.ATTESTATION_TYPES:
            return self.ATTESTATION_TYPES[predicate_type]
        
        # Check _type
        type_field = data.get("_type", "")
        if type_field in self.ATTESTATION_TYPES:
            # Need to also check predicate structure
            if "predicate" in data:
                predicate = data["predicate"]
                if "buildConfig" in predicate and "tasks" in predicate.get("buildConfig", {}):
                    return "slsa_provenance_v02"
                if "buildDefinition" in predicate:
                    return "slsa_provenance_v1"
        
        # Fallback: check predicate structure
        if "predicate" in data:
            predicate = data["predicate"]
            if "buildConfig" in predicate:
                return "slsa_provenance_v02"
            if "buildDefinition" in predicate:
                return "slsa_provenance_v1"
        
        return None
    
    def _extract_paths(self, data: Dict, att_type: str, source_file: str):
        """Extract schema paths from attestation data."""
        # Get priority paths for this attestation type
        priority_paths = self.PRIORITY_PATHS.get(att_type, [])
        
        # Always extract from actual data structure
        self._traverse_and_extract(data, "$", att_type, source_file, set(priority_paths))
    
    def _traverse_and_extract(
        self, 
        data: Any, 
        current_path: str, 
        att_type: str, 
        source_file: str,
        priority_paths: Set[str],
        depth: int = 0,
        max_depth: int = 10
    ):
        """Recursively traverse data and extract schema fields."""
        if depth > max_depth:
            return
        
        # Check if this path is a priority path or close to one
        is_priority = self._is_priority_path(current_path, priority_paths)
        
        if isinstance(data, dict):
            for key, value in data.items():
                child_path = f"{current_path}.{key}"
                
                # For arrays, use [*] notation
                if isinstance(value, list) and len(value) > 0:
                    array_path = f"{child_path}[*]"
                    # Extract from first element as example
                    if isinstance(value[0], dict):
                        self._traverse_and_extract(
                            value[0], array_path, att_type, source_file, 
                            priority_paths, depth + 1, max_depth
                        )
                    else:
                        # Leaf array
                        self._add_schema_field(
                            array_path, "array", value[0] if value else None,
                            att_type, source_file, is_priority
                        )
                elif isinstance(value, dict):
                    self._traverse_and_extract(
                        value, child_path, att_type, source_file,
                        priority_paths, depth + 1, max_depth
                    )
                else:
                    # Leaf value
                    field_type = self._infer_type(value)
                    if is_priority or self._is_priority_path(child_path, priority_paths):
                        self._add_schema_field(
                            child_path, field_type, value,
                            att_type, source_file, True
                        )
    
    def _is_priority_path(self, path: str, priority_paths: Set[str]) -> bool:
        """Check if path matches any priority path pattern."""
        # Normalize path for comparison
        normalized = path.replace("[0]", "[*]").replace("[*][*]", "[*]")
        
        for priority in priority_paths:
            # Exact match
            if normalized == priority:
                return True
            # Prefix match (path is parent of priority)
            if priority.startswith(normalized + ".") or priority.startswith(normalized + "["):
                return True
            # Path is child of priority
            if normalized.startswith(priority + ".") or normalized.startswith(priority + "["):
                return True
        
        return False
    
    def _add_schema_field(
        self,
        path: str,
        field_type: str,
        example_value: Any,
        att_type: str,
        source_file: str,
        is_priority: bool
    ):
        """Add a schema field to the collection."""
        # Generate stable ID
        schema_id = self._generate_schema_id(path, att_type)
        
        # Skip if we already have this schema (keep first example)
        if schema_id in self.schemas:
            return
        
        # Only add priority paths or important-looking fields
        if not is_priority:
            # Skip internal/metadata fields
            skip_patterns = [
                "annotations", "labels", "environment", "invocation",
                "configSource", "parameters", "digest", "sha1", "sha256"
            ]
            path_lower = path.lower()
            if any(skip in path_lower for skip in skip_patterns):
                return
        
        # Generate description based on path
        description = self._generate_description(path, field_type, att_type)
        
        # Generate use_when hints
        use_when = self._generate_use_when(path, att_type)
        
        self.schemas[schema_id] = SchemaField(
            schema_id=schema_id,
            canonical_path=path,
            attestation_type=att_type,
            field_type=field_type,
            description=description,
            use_when=use_when,
            example_value=example_value,
            source_file=source_file,
        )
    
    def _generate_schema_id(self, path: str, att_type: str) -> str:
        """Generate a stable schema ID from path and type."""
        # Simplify path for ID
        simplified = path.replace("$.", "").replace("[*]", "")
        simplified = re.sub(r'[^a-zA-Z0-9]', '_', simplified)
        simplified = re.sub(r'_+', '_', simplified).strip('_').lower()
        
        # Shorten common prefixes
        simplified = simplified.replace("predicate_buildconfig_", "")
        simplified = simplified.replace("predicate_builddefinition_", "")
        
        # Prefix with attestation type
        type_prefix = att_type.replace("_", "")[:8]
        
        # Ensure uniqueness with short hash
        full_key = f"{att_type}:{path}"
        short_hash = hashlib.md5(full_key.encode()).hexdigest()[:4]
        
        return f"{type_prefix}_{simplified}_{short_hash}"
    
    def _infer_type(self, value: Any) -> str:
        """Infer JSON type from value."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "unknown"
    
    def _generate_description(self, path: str, field_type: str, att_type: str) -> str:
        """Generate a description for a schema field."""
        # Extract field name from path
        parts = path.replace("[*]", "").split(".")
        field_name = parts[-1] if parts else "field"
        
        # Common descriptions
        descriptions = {
            "bundle": "OCI bundle reference for the task. May include @sha256: digest if pinned.",
            "name": "Name identifier for this element.",
            "resolver": "Resolution strategy (e.g., 'bundles', 'git').",
            "results": "Task execution results.",
            "status": "Execution status of the task.",
            "uri": "Resource URI or reference.",
            "digest": "Cryptographic digest for verification.",
            "sha256": "SHA-256 hash value.",
            "value": "Parameter or result value.",
            "type": "Type classification.",
        }
        
        if field_name in descriptions:
            return descriptions[field_name]
        
        # Default description
        return f"{field_name.replace('_', ' ').title()} ({field_type})"
    
    def _generate_use_when(self, path: str, att_type: str) -> List[str]:
        """Generate use_when hints for a schema field."""
        hints = []
        path_lower = path.lower()
        
        if "bundle" in path_lower:
            hints.extend(["task bundle validation", "bundle pinning checks"])
        if "ref" in path_lower:
            hints.extend(["task reference validation"])
        if "results" in path_lower:
            hints.extend(["checking task results", "result verification"])
        if "status" in path_lower:
            hints.extend(["task status checks", "success verification"])
        if "name" in path_lower:
            hints.extend(["identifying elements", "name matching"])
        if "digest" in path_lower or "sha256" in path_lower:
            hints.extend(["integrity verification", "pinning checks"])
        if "materials" in path_lower:
            hints.extend(["dependency tracking", "provenance verification"])
        
        return hints
    
    def save_schemas(self, output_path: Path):
        """Save schemas to JSONL file.
        
        Args:
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for schema in self.schemas.values():
                f.write(json.dumps(schema.to_dict()) + "\n")
    
    @classmethod
    def load_schemas(cls, input_path: Path) -> Dict[str, SchemaField]:
        """Load schemas from JSONL file.
        
        Args:
            input_path: Path to input file
            
        Returns:
            Dictionary mapping schema_id to SchemaField
        """
        schemas = {}
        
        if not input_path.exists():
            return schemas
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    schema = SchemaField.from_dict(data)
                    schemas[schema.schema_id] = schema
        
        return schemas

