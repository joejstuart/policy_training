#!/usr/bin/env python3
"""
Add SBOM schema documentation to the Knowledge Base.

SBOMs (Software Bill of Materials) use CycloneDX or SPDX format,
which is completely different from SLSA Provenance attestations.
"""

import json
from pathlib import Path

# SPDX SBOM schema documentation
SPDX_SCHEMAS = [
    {
        "schema_id": "spdx_packages_name",
        "canonical_path": "$.packages[*].name",
        "attestation_type": "spdx_sbom",
        "field_type": "string",
        "description": "Package name in the SBOM. Used to identify specific packages for allow/deny lists",
        "keywords": ["package name", "sbom", "spdx", "component", "dependency", "disallow", "allow"],
        "use_when": ["Check if package is allowed", "Block disallowed packages", "Filter by package name"],
        "example_values": ["acl", "bash", "openssl", "python3"],
        "source_doc": "SPDX specification",
    },
    {
        "schema_id": "spdx_packages_purl",
        "canonical_path": "$.packages[*].externalRefs[*].referenceLocator",
        "attestation_type": "spdx_sbom",
        "field_type": "string",
        "description": "Package URL (purl) uniquely identifying the package. Format: pkg:type/namespace/name@version",
        "keywords": ["purl", "package url", "sbom", "spdx", "identifier", "rpm", "npm", "pypi"],
        "use_when": ["Match package by purl", "Check package type (rpm, npm, etc.)", "Verify package version"],
        "example_values": ["pkg:rpm/fedora/bash@5.1.8-1.fc36", "pkg:npm/lodash@4.17.21"],
        "source_doc": "SPDX specification",
    },
    {
        "schema_id": "spdx_packages_version",
        "canonical_path": "$.packages[*].versionInfo",
        "attestation_type": "spdx_sbom",
        "field_type": "string",
        "description": "Package version string from the SBOM",
        "keywords": ["version", "sbom", "spdx", "package version"],
        "use_when": ["Check minimum version", "Verify specific version", "Block vulnerable versions"],
        "example_values": ["5.1.8-1.fc36", "4.17.21", "2.3.2-4.fc43"],
        "source_doc": "SPDX specification",
    },
    {
        "schema_id": "spdx_packages_license",
        "canonical_path": "$.packages[*].licenseDeclared",
        "attestation_type": "spdx_sbom",
        "field_type": "string",
        "description": "Declared license for the package in SPDX format",
        "keywords": ["license", "sbom", "spdx", "licensing", "compliance"],
        "use_when": ["Check license compliance", "Block disallowed licenses", "Verify open source licenses"],
        "example_values": ["MIT", "Apache-2.0", "GPL-3.0-only", "NOASSERTION"],
        "source_doc": "SPDX specification",
    },
    {
        "schema_id": "spdx_packages_supplier",
        "canonical_path": "$.packages[*].supplier",
        "attestation_type": "spdx_sbom",
        "field_type": "string",
        "description": "Supplier/vendor of the package",
        "keywords": ["supplier", "vendor", "sbom", "spdx", "origin"],
        "use_when": ["Verify package supplier", "Check trusted vendors"],
        "example_values": ["Organization: Red Hat", "Organization: Fedora Project"],
        "source_doc": "SPDX specification",
    },
]

# CycloneDX SBOM schema documentation
CYCLONEDX_SCHEMAS = [
    {
        "schema_id": "cyclonedx_components_name",
        "canonical_path": "$.components[*].name",
        "attestation_type": "cyclonedx_sbom",
        "field_type": "string",
        "description": "Component name in CycloneDX SBOM. Used to identify specific packages",
        "keywords": ["component name", "sbom", "cyclonedx", "package", "dependency", "disallow", "allow"],
        "use_when": ["Check if component is allowed", "Block disallowed components", "Filter by name"],
        "example_values": ["lodash", "express", "openssl"],
        "source_doc": "CycloneDX specification",
    },
    {
        "schema_id": "cyclonedx_components_purl",
        "canonical_path": "$.components[*].purl",
        "attestation_type": "cyclonedx_sbom",
        "field_type": "string",
        "description": "Package URL (purl) for the component. Unique identifier in purl format",
        "keywords": ["purl", "package url", "sbom", "cyclonedx", "identifier"],
        "use_when": ["Match component by purl", "Check component type", "Verify exact package"],
        "example_values": ["pkg:npm/lodash@4.17.21", "pkg:maven/org.apache/log4j@2.14.1"],
        "source_doc": "CycloneDX specification",
    },
    {
        "schema_id": "cyclonedx_components_version",
        "canonical_path": "$.components[*].version",
        "attestation_type": "cyclonedx_sbom",
        "field_type": "string",
        "description": "Component version in CycloneDX SBOM",
        "keywords": ["version", "sbom", "cyclonedx", "component version"],
        "use_when": ["Check version requirements", "Block vulnerable versions"],
        "example_values": ["4.17.21", "2.14.1", "1.0.0"],
        "source_doc": "CycloneDX specification",
    },
    {
        "schema_id": "cyclonedx_components_type",
        "canonical_path": "$.components[*].type",
        "attestation_type": "cyclonedx_sbom",
        "field_type": "string",
        "description": "Component type (library, framework, application, etc.)",
        "keywords": ["type", "sbom", "cyclonedx", "component type", "library", "framework"],
        "use_when": ["Filter by component type", "Check for specific types"],
        "example_values": ["library", "framework", "application", "operating-system"],
        "source_doc": "CycloneDX specification",
    },
    {
        "schema_id": "cyclonedx_components_licenses",
        "canonical_path": "$.components[*].licenses[*].license.id",
        "attestation_type": "cyclonedx_sbom",
        "field_type": "string",
        "description": "SPDX license ID for the component",
        "keywords": ["license", "sbom", "cyclonedx", "licensing", "compliance", "spdx"],
        "use_when": ["Check license compliance", "Block disallowed licenses"],
        "example_values": ["MIT", "Apache-2.0", "GPL-3.0-only"],
        "source_doc": "CycloneDX specification",
    },
]


def add_sbom_schemas(kb_dir: Path):
    """Add SBOM schemas to the knowledge base."""
    schemas_path = kb_dir / "schemas.jsonl"
    
    # Load existing schemas
    existing_schemas = []
    existing_ids = set()
    if schemas_path.exists():
        with open(schemas_path) as f:
            for line in f:
                if line.strip():
                    schema = json.loads(line)
                    existing_schemas.append(schema)
                    existing_ids.add(schema.get("schema_id", ""))
    
    # Add new SBOM schemas
    added = 0
    for schema in SPDX_SCHEMAS + CYCLONEDX_SCHEMAS:
        if schema["schema_id"] not in existing_ids:
            existing_schemas.append(schema)
            added += 1
            print(f"  Added: {schema['schema_id']}")
    
    # Save
    with open(schemas_path, "w") as f:
        for schema in existing_schemas:
            f.write(json.dumps(schema) + "\n")
    
    print(f"\nAdded {added} SBOM schemas to {schemas_path}")
    print(f"Total schemas: {len(existing_schemas)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Add SBOM schemas to KB")
    parser.add_argument("--kb-dir", type=Path, default=Path("data/knowledge_base"),
                       help="Knowledge base directory")
    args = parser.parse_args()
    
    print("Adding SBOM schemas...")
    add_sbom_schemas(args.kb_dir)


if __name__ == "__main__":
    main()

