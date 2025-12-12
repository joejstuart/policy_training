"""Tests for retrieval modules (vector_index, bm25_index, hybrid_retriever).

Tests follow the architecture spec requirements:
- Vector search for semantic matching
- BM25 for exact symbol matching
- Hybrid retrieval with per-type caps
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestVectorIndex:
    """Tests for VectorIndex class."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {
                "id": "lib.tekton.task_ref",
                "type": "helper",
                "text": "Helper: lib.tekton.task_ref\nSignature: task_ref(task)\nDescription: Returns task reference info with .pinned field for bundle pinning checks",
                "metadata": {"signature": "task_ref(task)"},
            },
            {
                "id": "lib.tekton.tasks",
                "type": "helper",
                "text": "Helper: lib.tekton.tasks\nSignature: tasks(attestation)\nDescription: Returns all tasks from a PipelineRun attestation",
                "metadata": {"signature": "tasks(attestation)"},
            },
            {
                "id": "lib.sbom.spdx_sboms",
                "type": "helper",
                "text": "Helper: lib.sbom.spdx_sboms\nSignature: spdx_sboms\nDescription: Returns all SPDX format SBOMs from attestations",
                "metadata": {"signature": "spdx_sboms"},
            },
        ]
    
    def test_import_available(self):
        """Test that vector_index module can be imported."""
        from vector_index import VectorIndex, SENTENCE_TRANSFORMERS_AVAILABLE, FAISS_AVAILABLE
        
        # Module should import even if dependencies are missing
        assert VectorIndex is not None
    
    @pytest.mark.skipif(
        not pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed"),
        reason="sentence-transformers required"
    )
    @pytest.mark.skipif(
        not pytest.importorskip("faiss", reason="faiss not installed"),
        reason="faiss required"
    )
    def test_build_and_search(self, sample_chunks):
        """Test building index and searching."""
        from vector_index import VectorIndex
        
        index = VectorIndex()
        index.build(sample_chunks)
        
        assert index.size == 3
        
        # Search for task-related content
        results = index.search("check if task bundle is pinned", top_k=2)
        
        assert len(results) > 0
        # task_ref should be highly relevant
        assert any("task_ref" in r.chunk_id for r in results)
    
    @pytest.mark.skipif(
        not pytest.importorskip("sentence_transformers", reason="sentence-transformers not installed"),
        reason="sentence-transformers required"
    )
    @pytest.mark.skipif(
        not pytest.importorskip("faiss", reason="faiss not installed"),
        reason="faiss required"
    )
    def test_save_and_load(self, sample_chunks):
        """Test saving and loading index."""
        from vector_index import VectorIndex
        
        with tempfile.TemporaryDirectory() as tmpdir:
            index_dir = Path(tmpdir) / "index"
            
            # Build and save
            index = VectorIndex()
            index.build(sample_chunks)
            index.save(index_dir)
            
            # Load into new index
            loaded = VectorIndex()
            loaded.load(index_dir)
            
            assert loaded.size == index.size
            
            # Search should work on loaded index
            results = loaded.search("task", top_k=2)
            assert len(results) > 0


class TestBM25Index:
    """Tests for BM25Index class."""
    
    @pytest.fixture
    def sample_chunks(self):
        """Sample chunks for testing."""
        return [
            {
                "id": "lib.tekton.task_ref",
                "type": "helper",
                "text": "lib.tekton.task_ref task_ref(task) Returns task reference info with pinned field for bundle pinning checks",
                "metadata": {},
            },
            {
                "id": "lib.tekton.tasks",
                "type": "helper",
                "text": "lib.tekton.tasks tasks(attestation) Returns all tasks from a PipelineRun attestation",
                "metadata": {},
            },
            {
                "id": "lib.sbom.spdx_sboms",
                "type": "helper",
                "text": "lib.sbom.spdx_sboms spdx_sboms Returns all SPDX format SBOMs from attestations for license checking",
                "metadata": {},
            },
        ]
    
    def test_import_available(self):
        """Test that bm25_index module can be imported."""
        from bm25_index import BM25Index, BM25_AVAILABLE
        
        assert BM25Index is not None
    
    @pytest.mark.skipif(
        not pytest.importorskip("rank_bm25", reason="rank-bm25 not installed"),
        reason="rank-bm25 required"
    )
    def test_build_and_search(self, sample_chunks):
        """Test building index and searching."""
        from bm25_index import BM25Index
        
        index = BM25Index()
        index.build(sample_chunks)
        
        assert index.size == 3
        
        # Search for exact symbol
        results = index.search("tekton.task_ref", top_k=2)
        
        assert len(results) > 0
        # Exact match should be first
        assert results[0].chunk_id == "lib.tekton.task_ref"
    
    @pytest.mark.skipif(
        not pytest.importorskip("rank_bm25", reason="rank-bm25 not installed"),
        reason="rank-bm25 required"
    )
    def test_tokenization_preserves_dotted_names(self):
        """Test that tokenization preserves dotted identifiers."""
        from bm25_index import BM25Index
        
        index = BM25Index()
        tokens = index._tokenize("Use lib.tekton.task_ref for bundle checks")
        
        # Should have the full dotted name
        assert "lib.tekton.task_ref" in tokens
        # And also the parts
        assert "tekton" in tokens
        assert "task_ref" in tokens
    
    @pytest.mark.skipif(
        not pytest.importorskip("rank_bm25", reason="rank-bm25 not installed"),
        reason="rank-bm25 required"
    )
    def test_save_and_load(self, sample_chunks):
        """Test saving and loading index."""
        from bm25_index import BM25Index
        
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "bm25.pkl"
            
            # Build and save
            index = BM25Index()
            index.build(sample_chunks)
            index.save(index_path)
            
            # Load into new index
            loaded = BM25Index()
            loaded.load(index_path)
            
            assert loaded.size == index.size
            
            # Search should work
            results = loaded.search("task_ref", top_k=1)
            assert len(results) > 0


class TestHybridRetriever:
    """Tests for HybridRetriever class."""
    
    def test_import_available(self):
        """Test that hybrid_retriever module can be imported."""
        from hybrid_retriever import HybridRetriever, RetrievalResult
        
        assert HybridRetriever is not None
        assert RetrievalResult is not None
    
    def test_retrieval_result_format_for_prompt(self):
        """Test formatting retrieval results for prompts."""
        from hybrid_retriever import RetrievalResult
        
        result = RetrievalResult(
            helpers=[
                {
                    "id": "lib.tekton.task_ref",
                    "signature": "task_ref(task)",
                    "description": "Returns task reference info",
                    "use_when": ["bundle pinning"],
                }
            ],
            schemas=[
                {
                    "schema_id": "slsa_task_bundle",
                    "canonical_path": "$.predicate.buildConfig.tasks[*].ref.bundle",
                    "field_type": "string",
                    "attestation_type": "slsa_provenance_v02",
                    "description": "OCI bundle reference",
                    "example_value": "quay.io/org/task@sha256:abc",
                }
            ],
        )
        
        prompt = result.format_for_prompt()
        
        assert "AVAILABLE HELPERS" in prompt
        assert "lib.tekton.task_ref" in prompt
        assert "task_ref(task)" in prompt
        assert "AVAILABLE SCHEMAS" in prompt
        assert "slsa_task_bundle" in prompt
        assert "$.predicate.buildConfig" in prompt
    
    def test_retrieval_result_all_chunks(self):
        """Test getting all chunks from result."""
        from hybrid_retriever import RetrievalResult
        
        result = RetrievalResult(
            helpers=[{"id": "h1"}, {"id": "h2"}],
            schemas=[{"id": "s1"}],
        )
        
        all_chunks = result.all_chunks
        
        assert len(all_chunks) == 3
    
    @pytest.mark.skipif(
        not pytest.importorskip("rank_bm25", reason="rank-bm25 not installed"),
        reason="rank-bm25 required"
    )
    def test_hybrid_retrieve_with_bm25_only(self):
        """Test hybrid retrieval with BM25 only (no vector)."""
        from hybrid_retriever import HybridRetriever
        from bm25_index import BM25Index
        
        # Create BM25 index for helpers
        chunks = [
            {"id": "h1", "type": "helper", "text": "tekton task_ref bundle pinning", "metadata": {"id": "h1"}},
            {"id": "h2", "type": "helper", "text": "tekton tasks attestation", "metadata": {"id": "h2"}},
        ]
        bm25 = BM25Index()
        bm25.build(chunks)
        
        # Create retriever with BM25 only
        retriever = HybridRetriever(helpers_bm25=bm25)
        
        result = retriever.retrieve("task bundle pinning", helper_k=2)
        
        assert len(result.helpers) > 0
        # task_ref should be more relevant for "bundle pinning"
        assert result.helpers[0]["id"] == "h1"


class TestCreateChunkFunctions:
    """Tests for chunk creation helper functions."""
    
    def test_create_helper_chunks(self):
        """Test creating helper chunks for indexing."""
        from vector_index import create_helper_chunks
        
        helpers = [
            {
                "id": "lib.tekton.task_ref",
                "signature": "task_ref(task)",
                "description": "Returns task reference",
                "use_when": ["bundle pinning"],
            }
        ]
        
        chunks = create_helper_chunks(helpers)
        
        assert len(chunks) == 1
        assert chunks[0]["id"] == "lib.tekton.task_ref"
        assert chunks[0]["type"] == "helper"
        assert "task_ref" in chunks[0]["text"]
        assert "bundle pinning" in chunks[0]["text"]
    
    def test_create_schema_chunks(self):
        """Test creating schema chunks for indexing."""
        from vector_index import create_schema_chunks
        
        schemas = [
            {
                "schema_id": "slsa_task_bundle",
                "canonical_path": "$.predicate.buildConfig.tasks[*].ref.bundle",
                "field_type": "string",
                "attestation_type": "slsa_provenance_v02",
                "description": "OCI bundle reference",
                "use_when": ["bundle validation"],
            }
        ]
        
        chunks = create_schema_chunks(schemas)
        
        assert len(chunks) == 1
        assert chunks[0]["id"] == "slsa_task_bundle"
        assert chunks[0]["type"] == "schema"
        assert "bundle" in chunks[0]["text"]
        assert "slsa_provenance" in chunks[0]["text"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

