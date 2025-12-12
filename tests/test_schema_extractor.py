"""Tests for schema_extractor module.

Tests follow the architecture spec requirements:
- Extract schemas from attestations with canonical IDs
- Normalize into canonical representation
- Generate stable schema_id
- Include grounding (source file, example values)
"""

import json
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from schema_extractor import SchemaExtractor, SchemaField


class TestSchemaField:
    """Tests for SchemaField dataclass."""
    
    def test_to_card_basic(self):
        """Test card generation with basic fields."""
        field = SchemaField(
            schema_id="slsa_task_bundle_abc1",
            canonical_path="$.predicate.buildConfig.tasks[*].ref.bundle",
            attestation_type="slsa_provenance_v02",
            field_type="string",
            description="OCI bundle reference",
            use_when=["bundle pinning checks"],
            example_value="quay.io/org/task@sha256:abc123",
            source_file="test.json",
        )
        
        card = field.to_card()
        
        assert "slsa_task_bundle_abc1" in card
        assert "$.predicate.buildConfig.tasks[*].ref.bundle" in card
        assert "string" in card
        assert "slsa_provenance_v02" in card
        assert "OCI bundle reference" in card
        assert "quay.io/org/task@sha256:abc123" in card
    
    def test_to_card_truncates_long_examples(self):
        """Test that long example values are truncated."""
        long_value = "x" * 200
        field = SchemaField(
            schema_id="test_id",
            canonical_path="$.test",
            attestation_type="test",
            field_type="string",
            description="",
            use_when=[],
            example_value=long_value,
            source_file="test.json",
        )
        
        card = field.to_card()
        
        # Should be truncated with ...
        assert "..." in card
        assert len(card) < len(long_value) + 200  # Card should be shorter
    
    def test_to_dict_and_from_dict_roundtrip(self):
        """Test serialization/deserialization roundtrip."""
        original = SchemaField(
            schema_id="test_id_123",
            canonical_path="$.predicate.tasks[*].name",
            attestation_type="slsa_provenance_v02",
            field_type="string",
            description="Task name",
            use_when=["identifying tasks", "name matching"],
            example_value="build-task",
            source_file="attestation.json",
            aliases=["$.predicate.buildConfig.tasks[*].name"],
        )
        
        data = original.to_dict()
        restored = SchemaField.from_dict(data)
        
        assert restored.schema_id == original.schema_id
        assert restored.canonical_path == original.canonical_path
        assert restored.attestation_type == original.attestation_type
        assert restored.field_type == original.field_type
        assert restored.description == original.description
        assert restored.use_when == original.use_when
        assert restored.example_value == original.example_value
        assert restored.source_file == original.source_file
        assert restored.aliases == original.aliases


class TestSchemaExtractor:
    """Tests for SchemaExtractor class."""
    
    @pytest.fixture
    def sample_slsa_v02_attestation(self):
        """Create a sample SLSA v0.2 attestation."""
        return {
            "_type": "https://in-toto.io/Statement/v0.1",
            "predicateType": "https://slsa.dev/provenance/v0.2",
            "subject": [
                {
                    "name": "quay.io/org/image",
                    "digest": {"sha256": "abc123"}
                }
            ],
            "predicate": {
                "buildConfig": {
                    "tasks": [
                        {
                            "name": "build-task",
                            "ref": {
                                "bundle": "quay.io/org/task@sha256:def456",
                                "name": "buildah",
                                "kind": "Task",
                                "resolver": "bundles",
                                "params": [
                                    {"name": "bundle", "value": "quay.io/org/task@sha256:def456"}
                                ]
                            },
                            "results": [
                                {"name": "IMAGE_DIGEST", "type": "string", "value": "sha256:abc123"}
                            ],
                            "status": "Succeeded",
                            "startedOn": "2025-01-01T00:00:00Z",
                            "finishedOn": "2025-01-01T00:05:00Z"
                        }
                    ]
                },
                "materials": [
                    {
                        "uri": "git+https://github.com/org/repo",
                        "digest": {"sha256": "commit123"}
                    }
                ]
            }
        }
    
    @pytest.fixture
    def temp_attestation_dir(self, sample_slsa_v02_attestation):
        """Create a temporary directory with attestation files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Write sample attestation
            att_file = tmppath / "sample_attestation.json"
            with open(att_file, 'w') as f:
                json.dump(sample_slsa_v02_attestation, f)
            
            yield tmppath
    
    def test_detect_attestation_type_slsa_v02(self, sample_slsa_v02_attestation):
        """Test SLSA v0.2 attestation type detection."""
        extractor = SchemaExtractor(Path("."))
        
        att_type = extractor.detect_attestation_type(sample_slsa_v02_attestation)
        
        assert att_type == "slsa_provenance_v02"
    
    def test_detect_attestation_type_by_predicate_structure(self):
        """Test detection by predicate structure when predicateType is missing."""
        extractor = SchemaExtractor(Path("."))
        
        # SLSA v0.2 style (buildConfig)
        v02_style = {
            "_type": "https://in-toto.io/Statement/v0.1",
            "predicate": {
                "buildConfig": {
                    "tasks": []
                }
            }
        }
        assert extractor.detect_attestation_type(v02_style) == "slsa_provenance_v02"
        
        # SLSA v1 style (buildDefinition)
        v1_style = {
            "_type": "https://in-toto.io/Statement/v0.1",
            "predicate": {
                "buildDefinition": {
                    "buildType": "https://tekton.dev/chains/v2/slsa"
                }
            }
        }
        assert extractor.detect_attestation_type(v1_style) == "slsa_provenance_v1"
    
    def test_extract_all_from_directory(self, temp_attestation_dir):
        """Test extracting schemas from all files in directory."""
        extractor = SchemaExtractor(temp_attestation_dir)
        
        schemas = extractor.extract_all()
        
        # Should have extracted some schemas
        assert len(schemas) > 0
        
        # Check that priority fields were extracted
        schema_ids = list(schemas.keys())
        paths = [s.canonical_path for s in schemas.values()]
        
        # Should have task-related paths
        assert any("tasks" in p for p in paths)
        assert any("bundle" in p for p in paths)
    
    def test_schema_id_is_stable(self, sample_slsa_v02_attestation):
        """Test that schema IDs are stable across extractions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            att_file = tmppath / "test.json"
            
            with open(att_file, 'w') as f:
                json.dump(sample_slsa_v02_attestation, f)
            
            # Extract twice
            extractor1 = SchemaExtractor(tmppath)
            schemas1 = extractor1.extract_all()
            
            extractor2 = SchemaExtractor(tmppath)
            schemas2 = extractor2.extract_all()
            
            # Schema IDs should be identical
            assert set(schemas1.keys()) == set(schemas2.keys())
    
    def test_extract_includes_example_values(self, temp_attestation_dir):
        """Test that extracted schemas include real example values."""
        extractor = SchemaExtractor(temp_attestation_dir)
        schemas = extractor.extract_all()
        
        # Find a schema with an example value
        schemas_with_examples = [s for s in schemas.values() if s.example_value is not None]
        
        assert len(schemas_with_examples) > 0
        
        # Check that example values are from the attestation
        for schema in schemas_with_examples:
            # Example should be a real value, not None or empty
            assert schema.example_value is not None
    
    def test_extract_includes_source_file(self, temp_attestation_dir):
        """Test that extracted schemas include source file info (grounding)."""
        extractor = SchemaExtractor(temp_attestation_dir)
        schemas = extractor.extract_all()
        
        for schema in schemas.values():
            assert schema.source_file != ""
            assert schema.source_file.endswith(".json")
    
    def test_save_and_load_schemas(self, temp_attestation_dir):
        """Test saving and loading schemas to/from JSONL."""
        extractor = SchemaExtractor(temp_attestation_dir)
        schemas = extractor.extract_all()
        
        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "schemas.jsonl"
            extractor.save_schemas(output_path)
            
            # Verify file was created
            assert output_path.exists()
            
            # Load and verify
            loaded = SchemaExtractor.load_schemas(output_path)
            
            assert len(loaded) == len(schemas)
            assert set(loaded.keys()) == set(schemas.keys())
    
    def test_normalize_wrapped_attestation_format(self):
        """Test handling of wrapped attestation format."""
        extractor = SchemaExtractor(Path("."))
        
        # Format: {"attestations": [{"statement": {...}}]}
        wrapped = {
            "attestations": [
                {
                    "statement": {
                        "_type": "https://in-toto.io/Statement/v0.1",
                        "predicateType": "https://slsa.dev/provenance/v0.2",
                        "predicate": {"buildConfig": {"tasks": []}}
                    }
                }
            ]
        }
        
        result = extractor._normalize_attestation_format(wrapped)
        
        assert len(result) == 1
        assert "_type" in result[0]
    
    def test_generate_use_when_for_bundle_path(self):
        """Test that use_when is generated correctly for bundle paths."""
        extractor = SchemaExtractor(Path("."))
        
        hints = extractor._generate_use_when(
            "$.predicate.buildConfig.tasks[*].ref.bundle",
            "slsa_provenance_v02"
        )
        
        assert any("bundle" in h.lower() for h in hints)
        assert any("pinning" in h.lower() for h in hints)
    
    def test_generate_description_for_known_fields(self):
        """Test that descriptions are generated for known fields."""
        extractor = SchemaExtractor(Path("."))
        
        # bundle field
        desc = extractor._generate_description(
            "$.predicate.buildConfig.tasks[*].ref.bundle",
            "string",
            "slsa_provenance_v02"
        )
        assert "bundle" in desc.lower()
        
        # name field
        desc = extractor._generate_description(
            "$.predicate.buildConfig.tasks[*].name",
            "string",
            "slsa_provenance_v02"
        )
        assert "name" in desc.lower()


class TestSchemaExtractorWithRealData:
    """Tests using real attestation data from the repository."""
    
    @pytest.fixture
    def real_attestation_dir(self):
        """Get the real attestation directory if it exists."""
        repo_root = Path(__file__).parent.parent
        att_dir = repo_root / "data" / "attestations"
        
        if att_dir.exists() and any(att_dir.glob("*.json")):
            return att_dir
        
        pytest.skip("No real attestation data available")
    
    def test_extract_from_real_attestations(self, real_attestation_dir):
        """Test extraction from real attestation files."""
        extractor = SchemaExtractor(real_attestation_dir)
        schemas = extractor.extract_all()
        
        # Should extract many schemas from real data
        assert len(schemas) > 0
        
        # Print summary for debugging
        print(f"\nExtracted {len(schemas)} schemas from real data")
        for schema in list(schemas.values())[:5]:
            print(f"  - {schema.schema_id}: {schema.canonical_path}")
    
    def test_real_schemas_have_valid_structure(self, real_attestation_dir):
        """Test that schemas from real data have valid structure."""
        extractor = SchemaExtractor(real_attestation_dir)
        schemas = extractor.extract_all()
        
        for schema in schemas.values():
            # Required fields
            assert schema.schema_id, "schema_id should not be empty"
            assert schema.canonical_path, "canonical_path should not be empty"
            assert schema.canonical_path.startswith("$"), "Path should start with $"
            assert schema.attestation_type, "attestation_type should not be empty"
            assert schema.field_type in [
                "string", "integer", "number", "boolean", 
                "array", "object", "null", "unknown"
            ]
            assert schema.source_file, "source_file should not be empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

