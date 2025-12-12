"""BM25 keyword search index.

Per architecture spec:
- BM25 keyword search for exact symbol matching
- Symbols matter: tekton.task_ref, pipelinerun_attestations
"""

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional import
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    BM25Okapi = None


@dataclass
class BM25SearchResult:
    """A single BM25 search result."""
    chunk_id: str
    chunk_type: str
    text: str
    score: float
    metadata: Dict[str, Any]


class BM25Index:
    """BM25 keyword search index.
    
    Per architecture spec:
    - Keyword search for exact symbol matches
    - Complements vector search for hybrid retrieval
    """
    
    def __init__(self):
        """Initialize BM25 index."""
        self.bm25 = None
        self.chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        
        if not BM25_AVAILABLE:
            print("Warning: rank-bm25 not available. Install with: pip install rank-bm25")
    
    def _simple_stem(self, word: str) -> str:
        """Simple stemmer for common English suffixes.
        
        Handles: -s, -es, -ed, -ing, -ies
        This is lightweight and doesn't require additional dependencies.
        Goal: CVEs->cve, images->image, patches->patch, vulnerabilities->vulnerability
        """
        if len(word) <= 3:
            return word
        
        # ies -> y (e.g., vulnerabilities -> vulnerability)
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'
        # ches, shes, xes, sses, zes -> remove es (e.g., patches -> patch)
        if word.endswith(('ches', 'shes', 'xes', 'sses', 'zes')) and len(word) > 4:
            return word[:-2]
        # s -> remove for simple plurals (e.g., images -> image, cves -> cve)
        # Skip words ending in ss, us, is
        if word.endswith('s') and not word.endswith(('ss', 'us', 'is')) and len(word) > 3:
            return word[:-1]
        # ed -> remove for past tense (e.g., pinned -> pin)
        if word.endswith('ed') and len(word) > 4:
            if word[-3] == word[-4]:  # doubled consonant: pinned -> pin
                return word[:-3]
            return word[:-2]
        # ing -> remove (e.g., running -> run)
        if word.endswith('ing') and len(word) > 5:
            if word[-4] == word[-5]:  # doubled consonant: running -> run
                return word[:-4]
            return word[:-3]
        
        return word
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 with simple stemming.
        
        Preserves important patterns like:
        - tekton.task_ref
        - lib.pipelinerun_attestations
        - $.predicate.buildConfig
        
        Uses simple stemming so "CVEs" matches "CVE", "patches" matches "patch".
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens (stemmed)
        """
        # Lowercase
        text = text.lower()
        
        # Split on whitespace and punctuation, but preserve dots in identifiers
        # First, protect dotted identifiers
        protected = re.findall(r'[a-z_][a-z0-9_.]+', text)
        
        # Also get individual words
        words = re.findall(r'[a-z]{2,}', text)
        
        # Combine, keeping dotted patterns and their parts (with stemming)
        tokens = []
        for p in protected:
            tokens.append(p)
            # Also add parts split by dots and underscores (stemmed)
            parts = re.split(r'[._]', p)
            tokens.extend(self._simple_stem(part) for part in parts if len(part) >= 2)
        
        # Stem individual words too
        tokens.extend(self._simple_stem(w) for w in words)
        
        # Dedupe while preserving order
        seen = set()
        unique_tokens = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)
        
        return unique_tokens
    
    def build(self, chunks: List[Dict[str, Any]], text_field: str = "text"):
        """Build BM25 index from chunks.
        
        Args:
            chunks: List of chunks with 'id', 'type', and text_field
            text_field: Field to use for indexing
        """
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 is required. Install with: pip install rank-bm25")
        
        if not chunks:
            print("Warning: No chunks to index")
            return
        
        print(f"Building BM25 index with {len(chunks)} chunks...")
        
        self.chunks = chunks
        
        # Tokenize corpus
        self.tokenized_corpus = [
            self._tokenize(chunk.get(text_field, ""))
            for chunk in chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"Built BM25 index with {len(chunks)} documents")
    
    def search(self, query: str, top_k: int = 20) -> List[BM25SearchResult]:
        """Search for chunks matching query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of BM25SearchResult objects
        """
        if self.bm25 is None or not self.chunks:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score <= 0:
                continue
            
            chunk = self.chunks[idx]
            results.append(BM25SearchResult(
                chunk_id=chunk.get('id', str(idx)),
                chunk_type=chunk.get('type', 'unknown'),
                text=chunk.get('text', ''),
                score=float(score),
                metadata=chunk.get('metadata', {}),
            ))
        
        return results
    
    def save(self, path: Path):
        """Save index to disk.
        
        Args:
            path: Path to save pickle file
        """
        if self.bm25 is None:
            print("Warning: No index to save")
            return
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'chunks': self.chunks,
            'tokenized_corpus': self.tokenized_corpus,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved BM25 index to {path}")
    
    def load(self, path: Path):
        """Load index from disk.
        
        Args:
            path: Path to pickle file
        """
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 is required. Install with: pip install rank-bm25")
        
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.tokenized_corpus = data['tokenized_corpus']
        
        # Rebuild BM25 from tokenized corpus
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"Loaded BM25 index from {path} ({len(self.chunks)} documents)")
    
    @property
    def size(self) -> int:
        """Number of indexed chunks."""
        return len(self.chunks)

