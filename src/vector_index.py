"""FAISS vector search index for semantic retrieval.

Per architecture spec:
- Embedding model: sentence-transformers/all-MiniLM-L6-v2
- Vector store: FAISS (simple, no external deps)
"""

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional imports - graceful degradation
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None


@dataclass
class SearchResult:
    """A single search result."""
    chunk_id: str
    chunk_type: str  # "helper" or "schema"
    text: str
    score: float
    metadata: Dict[str, Any]


class VectorIndex:
    """FAISS-based vector search index.
    
    Per architecture spec:
    - Semantic search over helpers and schemas
    - Uses sentence-transformers for embeddings
    """
    
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(self, embedding_model: str = None):
        """Initialize vector index.
        
        Args:
            embedding_model: Model name for sentence-transformers
        """
        self.model_name = embedding_model or self.DEFAULT_MODEL
        self.model = None
        self.index = None
        self.chunks: List[Dict[str, Any]] = []  # chunk_id -> chunk data
        self.id_to_idx: Dict[str, int] = {}  # chunk_id -> index position
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
        
        if not FAISS_AVAILABLE:
            print("Warning: faiss not available. Install with: pip install faiss-cpu")
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
            self.model = SentenceTransformer(self.model_name)
    
    def _embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using the model.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        self._load_model()
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 100)
        return embeddings.astype('float32')
    
    def build(self, chunks: List[Dict[str, Any]], text_field: str = "text"):
        """Build index from chunks.
        
        Args:
            chunks: List of chunks with at least 'id', 'type', and text_field
            text_field: Field to use for embedding text
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")
        
        if not chunks:
            print("Warning: No chunks to index")
            return
        
        print(f"Building vector index with {len(chunks)} chunks...")
        
        # Store chunks
        self.chunks = chunks
        self.id_to_idx = {chunk['id']: i for i, chunk in enumerate(chunks)}
        
        # Extract texts
        texts = [chunk.get(text_field, "") for chunk in chunks]
        
        # Embed
        embeddings = self._embed(texts)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        print(f"Built index with {self.index.ntotal} vectors of dimension {dimension}")
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search for similar chunks.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Embed query
        query_embedding = self._embed([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for not found
                continue
            
            chunk = self.chunks[idx]
            results.append(SearchResult(
                chunk_id=chunk.get('id', str(idx)),
                chunk_type=chunk.get('type', 'unknown'),
                text=chunk.get('text', ''),
                score=float(score),
                metadata=chunk.get('metadata', {}),
            ))
        
        return results
    
    def save(self, path: Path):
        """Save index and chunks to disk.
        
        Args:
            path: Directory to save to
        """
        if self.index is None:
            print("Warning: No index to save")
            return
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save chunks
        with open(path / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f)
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "num_chunks": len(self.chunks),
        }
        with open(path / "metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f)
        
        print(f"Saved vector index to {path}")
    
    def load(self, path: Path):
        """Load index and chunks from disk.
        
        Args:
            path: Directory to load from
        """
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")
        
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Index directory not found: {path}")
        
        # Load FAISS index
        index_path = path / "index.faiss"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        # Load chunks
        chunks_path = path / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            self.id_to_idx = {chunk['id']: i for i, chunk in enumerate(self.chunks)}
        
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.model_name = metadata.get("model_name", self.DEFAULT_MODEL)
        
        print(f"Loaded vector index from {path} ({self.index.ntotal} vectors)")
    
    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self.chunks)


def create_helper_chunks(helpers: List[Dict]) -> List[Dict]:
    """Create indexable chunks from helper data.
    
    Args:
        helpers: List of helper dictionaries (from helpers.jsonl)
        
    Returns:
        List of chunks ready for indexing
    """
    chunks = []
    for helper in helpers:
        # Build searchable text
        text_parts = [
            f"Helper: {helper.get('id', '')}",
            f"Signature: {helper.get('signature', '')}",
            f"Description: {helper.get('description', '')}",
        ]
        use_when = helper.get('use_when', [])
        if use_when:
            text_parts.append(f"Use when: {', '.join(use_when)}")
        
        chunks.append({
            'id': helper.get('id', ''),
            'type': 'helper',
            'text': '\n'.join(text_parts),
            'metadata': helper,
        })
    
    return chunks


def create_schema_chunks(schemas: List[Dict]) -> List[Dict]:
    """Create indexable chunks from schema data.
    
    Args:
        schemas: List of schema dictionaries (from schemas.jsonl)
        
    Returns:
        List of chunks ready for indexing
    """
    chunks = []
    for schema in schemas:
        # Build searchable text
        text_parts = [
            f"Schema: {schema.get('schema_id', '')}",
            f"Path: {schema.get('canonical_path', '')}",
            f"Type: {schema.get('field_type', '')}",
            f"Attestation: {schema.get('attestation_type', '')}",
            f"Description: {schema.get('description', '')}",
        ]
        use_when = schema.get('use_when', [])
        if use_when:
            text_parts.append(f"Use when: {', '.join(use_when)}")
        
        chunks.append({
            'id': schema.get('schema_id', ''),
            'type': 'schema',
            'text': '\n'.join(text_parts),
            'metadata': schema,
        })
    
    return chunks

