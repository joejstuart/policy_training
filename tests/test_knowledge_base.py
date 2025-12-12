"""Tests for knowledge_base module.

Tests follow the architecture spec requirements:
- Unified access to helpers and schemas
- Two-tier chunks (card/full)
- Existence checks for validation gate
- Grounding (source files, line numbers)
"""

import json
import pytest
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_base import KnowledgeBase, HelperCard, HelperFull
from schema_extractor import SchemaField
from kb_manifest import KBManifest


class TestHelperCard:
    """Tests for HelperCard dataclass."""
    
    def test_to_text_includes_key_fields(self):
        """Test text representation includes all key fields."""
        card = HelperCard(
            id="lib.tekton.task_ref",
            name="task_ref",
            module_path="data.lib.tekton",
            signature="task_ref(task)",
            description="Returns task reference info with .pinned field",
            use_when=["bundle pinning checks", "task validation"],
            source_file="policy/lib/tekton/tekton.rego",
            source_lines=(142, 158),
        )
        
        text = card.to_text()
        
        assert "lib.tekton.task_ref" in text
        assert "task_ref(task)" in text
        assert "pinned" in text
        assert "bundle pinning" in text
    
    def test_to_dict_and_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = HelperCard(
            id="lib.result_helper",
            name="result_helper",
            module_path="data.lib",
            signature="result_helper(chain, args)",
            description="Creates formatted result object",
            use_when=["error messages", "deny results"],
            source_file="policy/lib/result_helper.rego",
            source_lines=(10, 25),
        )
        
        data = original.to_dict()
        restored = HelperCard.from_dict(data)
        
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.signature == original.signature
        assert restored.source_lines == original.source_lines


class TestHelperFull:
    """Tests for HelperFull dataclass."""
    
    def test_to_card_creates_compact_form(self):
        """Test conversion to card drops body and examples."""
        full = HelperFull(
            id="lib.tekton.tasks",
            name="tasks",
            module_path="data.lib.tekton",
            signature="tasks(attestation)",
            description="Returns all tasks from attestation",
            use_when=["iterating tasks"],
            source_file="policy/lib/tekton/tekton.rego",
            source_lines=(50, 70),
            body="tasks(attestation) := result if {\n    result := attestation.predicate.buildConfig.tasks\n}",
            usage_examples=["some task in tekton.tasks(att)"],
            imports_required=["data.lib.tekton"],
        )
        
        card = full.to_card()
        
        assert isinstance(card, HelperCard)
        assert card.id == full.id
        assert card.signature == full.signature
        # Card should not have body or usage_examples (those are on HelperFull only)
        assert not hasattr(card, 'body')
        assert not hasattr(card, 'usage_examples')
    
    def test_to_text_includes_body(self):
        """Test full text representation includes body."""
        full = HelperFull(
            id="lib.test",
            name="test",
            module_path="data.lib",
            signature="test(x)",
            description="Test helper",
            use_when=[],
            source_file="test.rego",
            source_lines=(1, 5),
            body="test(x) := x > 0",
            usage_examples=["test(5)"],
            imports_required=[],
        )
        
        text = full.to_text()
        
        assert "test(x) := x > 0" in text
        assert "test(5)" in text


class TestKnowledgeBase:
    """Tests for KnowledgeBase class."""
    
    @pytest.fixture
    def sample_kb(self):
        """Create a sample knowledge base with test data."""
        kb = KnowledgeBase()
        
        # Add helpers
        kb.add_helper(HelperFull(
            id="lib.tekton.task_ref",
            name="task_ref",
            module_path="data.lib.tekton",
            signature="task_ref(task)",
            description="Returns task reference info",
            use_when=["bundle pinning"],
            source_file="policy/lib/tekton/tekton.rego",
            source_lines=(142, 158),
            body="task_ref(task) := ref if {...}",
            usage_examples=["ref := tekton.task_ref(task)"],
            imports_required=["data.lib.tekton"],
        ))
        
        kb.add_helper(HelperFull(
            id="lib.tekton.tasks",
            name="tasks",
            module_path="data.lib.tekton",
            signature="tasks(attestation)",
            description="Returns all tasks from attestation",
            use_when=["iterating tasks"],
            source_file="policy/lib/tekton/tekton.rego",
            source_lines=(50, 70),
            body="tasks(att) := att.predicate.buildConfig.tasks",
            usage_examples=["some task in tekton.tasks(att)"],
            imports_required=["data.lib.tekton"],
        ))
        
        kb.add_helper(HelperFull(
            id="lib.pipelinerun_attestations",
            name="pipelinerun_attestations",
            module_path="data.lib",
            signature="pipelinerun_attestations",
            description="Returns all pipeline run attestations",
            use_when=["iterating attestations"],
            source_file="policy/lib/lib.rego",
            source_lines=(100, 110),
            body="pipelinerun_attestations := attestations if {...}",
            usage_examples=["some att in lib.pipelinerun_attestations"],
            imports_required=["data.lib"],
        ))
        
        # Add schemas
        kb.add_schema(SchemaField(
            schema_id="slsa_task_bundle_abc1",
            canonical_path="$.predicate.buildConfig.tasks[*].ref.bundle",
            attestation_type="slsa_provenance_v02",
            field_type="string",
            description="OCI bundle reference",
            use_when=["bundle pinning checks"],
            example_value="quay.io/org/task@sha256:abc123",
            source_file="test.json",
        ))
        
        kb.add_schema(SchemaField(
            schema_id="slsa_task_name_def2",
            canonical_path="$.predicate.buildConfig.tasks[*].name",
            attestation_type="slsa_provenance_v02",
            field_type="string",
            description="Task name",
            use_when=["identifying tasks"],
            example_value="build-task",
            source_file="test.json",
        ))
        
        return kb
    
    def test_helper_exists_by_id(self, sample_kb):
        """Test helper existence check by full ID."""
        assert sample_kb.helper_exists("lib.tekton.task_ref") is True
        assert sample_kb.helper_exists("lib.tekton.nonexistent") is False
    
    def test_helper_exists_by_name(self, sample_kb):
        """Test helper existence check by name only."""
        assert sample_kb.helper_exists("task_ref") is True
        assert sample_kb.helper_exists("tasks") is True
        assert sample_kb.helper_exists("nonexistent") is False
    
    def test_get_helper_card(self, sample_kb):
        """Test getting helper card."""
        card = sample_kb.get_helper_card("lib.tekton.task_ref")
        
        assert card is not None
        assert card.id == "lib.tekton.task_ref"
        assert card.signature == "task_ref(task)"
    
    def test_get_helper_full(self, sample_kb):
        """Test getting full helper with body."""
        full = sample_kb.get_helper_full("lib.tekton.task_ref")
        
        assert full is not None
        assert full.body is not None
        assert len(full.body) > 0
    
    def test_schema_exists_by_id(self, sample_kb):
        """Test schema existence check by ID."""
        assert sample_kb.schema_exists("slsa_task_bundle_abc1") is True
        assert sample_kb.schema_exists("nonexistent_schema") is False
    
    def test_schema_exists_with_attestation_type_filter(self, sample_kb):
        """Test schema existence check with attestation type."""
        # Correct type
        assert sample_kb.schema_exists("slsa_task_bundle_abc1", "slsa_provenance_v02") is True
        
        # Wrong type
        assert sample_kb.schema_exists("slsa_task_bundle_abc1", "spdx_sbom") is False
    
    def test_get_schema(self, sample_kb):
        """Test getting schema by ID."""
        schema = sample_kb.get_schema("slsa_task_bundle_abc1")
        
        assert schema is not None
        assert schema.canonical_path == "$.predicate.buildConfig.tasks[*].ref.bundle"
        assert schema.example_value == "quay.io/org/task@sha256:abc123"
    
    def test_get_schemas_for_type(self, sample_kb):
        """Test getting all schemas for an attestation type."""
        schemas = sample_kb.get_schemas_for_type("slsa_provenance_v02")
        
        assert len(schemas) == 2
        assert all(s.attestation_type == "slsa_provenance_v02" for s in schemas)
    
    def test_save_and_load(self, sample_kb):
        """Test saving and loading KB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb_dir = Path(tmpdir) / "kb"
            
            # Save
            sample_kb.save(kb_dir)
            
            # Verify files created
            assert (kb_dir / "helpers.jsonl").exists()
            assert (kb_dir / "helpers_full.jsonl").exists()
            assert (kb_dir / "schemas.jsonl").exists()
            
            # Load into new KB
            loaded_kb = KnowledgeBase(kb_dir)
            
            # Verify data
            assert len(loaded_kb.helper_cards) == len(sample_kb.helper_cards)
            assert len(loaded_kb.schemas) == len(sample_kb.schemas)
            
            # Verify helper exists after load
            assert loaded_kb.helper_exists("lib.tekton.task_ref")
            assert loaded_kb.schema_exists("slsa_task_bundle_abc1")
    
    def test_stats(self, sample_kb):
        """Test KB statistics."""
        stats = sample_kb.stats()
        
        assert stats["helper_count"] == 3
        assert stats["schema_count"] == 2
        assert "slsa_provenance_v02" in stats["attestation_types"]
    
    def test_get_all_helper_cards(self, sample_kb):
        """Test getting all helper cards."""
        cards = sample_kb.get_all_helper_cards()
        
        assert len(cards) == 3
        assert all(isinstance(c, HelperCard) for c in cards)
    
    def test_get_helper_ids(self, sample_kb):
        """Test getting all helper IDs."""
        ids = sample_kb.get_helper_ids()
        
        assert "lib.tekton.task_ref" in ids
        assert "lib.tekton.tasks" in ids
        assert "lib.pipelinerun_attestations" in ids


class TestKBManifest:
    """Tests for KBManifest."""
    
    def test_create_manifest(self):
        """Test creating manifest from repo state."""
        repo_root = Path(__file__).parent.parent
        
        manifest = KBManifest.create(repo_root, helper_count=10, schema_count=5)
        
        assert manifest.helper_count == 10
        assert manifest.schema_count == 5
        assert manifest.git_ref != ""  # Should have some ref (or "unknown")
        assert manifest.built_at != ""
    
    def test_save_and_load_manifest(self):
        """Test manifest save/load roundtrip."""
        manifest = KBManifest(
            git_ref="abc123",
            built_at="2025-01-01T00:00:00Z",
            policy_lib_hash="hash123",
            helper_count=50,
            schema_count=25,
            attestation_types=["slsa_provenance_v02"],
            attestation_files_count=10,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.yaml"
            
            manifest.save(path)
            assert path.exists()
            
            loaded = KBManifest.load(path)
            
            assert loaded.git_ref == "abc123"
            assert loaded.helper_count == 50
            assert loaded.schema_count == 25
            assert "slsa_provenance_v02" in loaded.attestation_types
    
    def test_summary(self):
        """Test human-readable summary."""
        manifest = KBManifest(
            git_ref="abc123",
            built_at="2025-01-01T00:00:00Z",
            policy_lib_hash="hash123",
            helper_count=50,
            schema_count=25,
        )
        
        summary = manifest.summary()
        
        assert "abc123" in summary
        assert "50" in summary
        assert "25" in summary


class TestKnowledgeBaseIntegration:
    """Integration tests with real data if available."""
    
    @pytest.fixture
    def real_kb_dir(self):
        """Get real KB directory if it exists."""
        repo_root = Path(__file__).parent.parent
        kb_dir = repo_root / "data" / "knowledge_base"
        
        if kb_dir.exists() and (kb_dir / "helpers.jsonl").exists():
            return kb_dir
        
        pytest.skip("No built KB available")
    
    def test_load_real_kb(self, real_kb_dir):
        """Test loading a real KB."""
        kb = KnowledgeBase(real_kb_dir)
        
        assert len(kb.helper_cards) > 0
        print(f"\nLoaded {len(kb.helper_cards)} helpers from real KB")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

