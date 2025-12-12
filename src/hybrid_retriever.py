"""Hybrid retriever combining vector and BM25 search.

Per architecture spec:
- BM25 keyword search + Vector search (semantic)
- Rerank top 30 → return top 7–10
- Cap results by type (max 4 helpers + 2-3 schemas)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    from vector_index import VectorIndex, SearchResult
    from bm25_index import BM25Index, BM25SearchResult
except ImportError:
    from .vector_index import VectorIndex, SearchResult
    from .bm25_index import BM25Index, BM25SearchResult


@dataclass
class RetrievalResult:
    """Result of hybrid retrieval.
    
    Per architecture spec: separate helpers and schemas with caps.
    """
    helpers: List[Dict[str, Any]]
    schemas: List[Dict[str, Any]]
    
    @property
    def all_chunks(self) -> List[Dict[str, Any]]:
        """All retrieved chunks."""
        return self.helpers + self.schemas
    
    def format_for_prompt(self, include_full: bool = False) -> str:
        """Format for inclusion in LLM prompt.
        
        Args:
            include_full: Include full helper bodies (for codegen)
            
        Returns:
            Formatted string
        """
        lines = []
        
        if self.helpers:
            lines.append("AVAILABLE HELPERS:")
            for h in self.helpers:
                lines.append(f"\n--- Helper: {h.get('id', '')} ---")
                lines.append(f"Signature: {h.get('signature', '')}")
                lines.append(f"Description: {h.get('description', '')}")
                if h.get('use_when'):
                    lines.append(f"Use when: {', '.join(h['use_when'])}")
                if include_full and h.get('body'):
                    lines.append(f"Body:\n{h['body']}")
        
        if self.schemas:
            lines.append("\n\nAVAILABLE SCHEMAS:")
            for s in self.schemas:
                lines.append(f"\n--- Schema: {s.get('schema_id', '')} ---")
                lines.append(f"Path: {s.get('canonical_path', '')}")
                lines.append(f"Type: {s.get('field_type', '')}")
                lines.append(f"Attestation: {s.get('attestation_type', '')}")
                if s.get('description'):
                    lines.append(f"Description: {s['description']}")
                if s.get('example_value') is not None:
                    lines.append(f"Example: {s['example_value']}")
        
        return "\n".join(lines)


class HybridRetriever:
    """Hybrid retriever combining vector and BM25 search.
    
    Per architecture spec:
    - Query BM25 (top 20) + Vector (top 20)
    - Merge and dedupe (top 30)
    - Rerank with cross-encoder (optional)
    - Apply per-type caps
    """
    
    def __init__(
        self,
        helpers_vector: Optional[VectorIndex] = None,
        helpers_bm25: Optional[BM25Index] = None,
        schemas_vector: Optional[VectorIndex] = None,
        schemas_bm25: Optional[BM25Index] = None,
    ):
        """Initialize retriever with indexes.
        
        Args:
            helpers_vector: Vector index for helpers
            helpers_bm25: BM25 index for helpers
            schemas_vector: Vector index for schemas
            schemas_bm25: BM25 index for schemas
        """
        self.helpers_vector = helpers_vector
        self.helpers_bm25 = helpers_bm25
        self.schemas_vector = schemas_vector
        self.schemas_bm25 = schemas_bm25
    
    @classmethod
    def from_kb_dir(cls, kb_dir: Path) -> "HybridRetriever":
        """Load retriever from KB directory.
        
        Args:
            kb_dir: Knowledge base directory with index/ subdirectory
            
        Returns:
            HybridRetriever instance
        """
        index_dir = Path(kb_dir) / "index"
        
        helpers_vector = None
        helpers_bm25 = None
        schemas_vector = None
        schemas_bm25 = None
        
        # Load helpers vector index
        helpers_vector_dir = index_dir / "helpers_vector"
        if helpers_vector_dir.exists():
            helpers_vector = VectorIndex()
            helpers_vector.load(helpers_vector_dir)
        
        # Load helpers BM25 index
        helpers_bm25_path = index_dir / "helpers_bm25.pkl"
        if helpers_bm25_path.exists():
            helpers_bm25 = BM25Index()
            helpers_bm25.load(helpers_bm25_path)
        
        # Load schemas vector index
        schemas_vector_dir = index_dir / "schemas_vector"
        if schemas_vector_dir.exists():
            schemas_vector = VectorIndex()
            schemas_vector.load(schemas_vector_dir)
        
        # Load schemas BM25 index
        schemas_bm25_path = index_dir / "schemas_bm25.pkl"
        if schemas_bm25_path.exists():
            schemas_bm25 = BM25Index()
            schemas_bm25.load(schemas_bm25_path)
        
        return cls(
            helpers_vector=helpers_vector,
            helpers_bm25=helpers_bm25,
            schemas_vector=schemas_vector,
            schemas_bm25=schemas_bm25,
        )
    
    def retrieve(
        self,
        query: str,
        helper_k: int = 4,
        schema_k: int = 2,
        bm25_weight: float = 0.1,
        vector_weight: float = 0.9,
    ) -> RetrievalResult:
        """Retrieve relevant helpers and schemas.
        
        Per architecture spec:
        - Cap results by type
        - Combine BM25 and vector scores
        
        Args:
            query: Query text
            helper_k: Maximum helpers to return
            schema_k: Maximum schemas to return
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores
            
        Returns:
            RetrievalResult with helpers and schemas
        """
        # Retrieve helpers
        helpers = self._retrieve_type(
            query=query,
            vector_index=self.helpers_vector,
            bm25_index=self.helpers_bm25,
            top_k=helper_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )
        
        # Retrieve schemas
        schemas = self._retrieve_type(
            query=query,
            vector_index=self.schemas_vector,
            bm25_index=self.schemas_bm25,
            top_k=schema_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
        )
        
        return RetrievalResult(helpers=helpers, schemas=schemas)
    
    def _retrieve_type(
        self,
        query: str,
        vector_index: Optional[VectorIndex],
        bm25_index: Optional[BM25Index],
        top_k: int,
        bm25_weight: float,
        vector_weight: float,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """Retrieve from a single type (helpers or schemas).
        
        Uses Reciprocal Rank Fusion (RRF) to combine results from vector
        and BM25 search. RRF is more robust than weighted averages because
        it uses rank positions rather than raw scores.
        
        RRF formula: score(d) = sum over rankings: weight / (k + rank(d))
        
        Args:
            query: Query text
            vector_index: Vector index (optional)
            bm25_index: BM25 index (optional)
            top_k: Maximum results
            bm25_weight: Weight for BM25 ranking contribution
            vector_weight: Weight for vector ranking contribution
            rrf_k: RRF constant (default 60, higher = more weight to lower ranks)
            
        Returns:
            List of chunk metadata
        """
        # Collect RRF scores and metadata
        rrf_scores: Dict[str, float] = {}  # chunk_id -> RRF score
        chunks: Dict[str, Dict] = {}   # chunk_id -> chunk metadata
        
        # Query vector index and add RRF scores
        if vector_index is not None:
            vector_results = vector_index.search(query, top_k=20)
            for rank, r in enumerate(vector_results, start=1):
                chunk_id = r.chunk_id
                # RRF contribution: weight / (k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (vector_weight / (rrf_k + rank))
                if chunk_id not in chunks:
                    chunks[chunk_id] = r.metadata
        
        # Query BM25 index and add RRF scores
        if bm25_index is not None:
            bm25_results = bm25_index.search(query, top_k=20)
            for rank, r in enumerate(bm25_results, start=1):
                chunk_id = r.chunk_id
                # RRF contribution: weight / (k + rank)
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (bm25_weight / (rrf_k + rank))
                if chunk_id not in chunks:
                    chunks[chunk_id] = r.metadata
        
        # Sort by RRF score and take top_k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        # Return metadata for top results
        return [chunks[chunk_id] for chunk_id in sorted_ids if chunk_id in chunks]
    
    def retrieve_with_full(
        self,
        query: str,
        kb,  # KnowledgeBase
        helper_k: int = 4,
        schema_k: int = 2,
    ) -> RetrievalResult:
        """Retrieve and enrich with full helper bodies.
        
        Per architecture spec: Codegen gets full helper bodies.
        
        Args:
            query: Query text
            kb: KnowledgeBase for enrichment
            helper_k: Maximum helpers
            schema_k: Maximum schemas
            
        Returns:
            RetrievalResult with full helper bodies
        """
        result = self.retrieve(query, helper_k, schema_k)
        
        # Enrich helpers with full bodies
        enriched_helpers = []
        for h in result.helpers:
            helper_id = h.get('id', '')
            full = kb.get_helper_full(helper_id)
            if full:
                enriched = h.copy()
                enriched['body'] = full.body
                enriched['usage_examples'] = full.usage_examples
                enriched_helpers.append(enriched)
            else:
                enriched_helpers.append(h)
        
        return RetrievalResult(helpers=enriched_helpers, schemas=result.schemas)

