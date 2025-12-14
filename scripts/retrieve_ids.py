#!/usr/bin/env python3
"""Use the trained ID selector for retrieval.

Usage:
    # Single query
    python scripts/retrieve_ids.py --query "check if task bundle is pinned"
    
    # Interactive mode
    python scripts/retrieve_ids.py --interactive
    
    # In Python code
    from scripts.retrieve_ids import IDRetriever
    retriever = IDRetriever.load()
    results = retriever.retrieve("check if task bundle is pinned")
    print(results.helper_ids)
    print(results.schema_ids)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss
import numpy as np


@dataclass
class RetrievalResult:
    """Result from ID retrieval."""
    query: str
    helper_ids: List[str]
    schema_ids: List[str]
    all_results: List[Tuple[str, float]]  # (id, score) pairs
    
    def format_context(self, corpus: Dict[str, str], max_items: int = 7) -> str:
        """Format results as context for downstream model."""
        lines = []
        count = 0
        
        for doc_id, score in self.all_results:
            if count >= max_items:
                break
            if doc_id in corpus:
                lines.append(f"--- {doc_id} ---")
                lines.append(corpus[doc_id])
                lines.append("")
                count += 1
        
        return "\n".join(lines)


class IDRetriever:
    """Retrieve relevant helper and schema IDs for a query."""
    
    def __init__(
        self,
        corpus: Dict[str, str],
        embedding_model: SentenceTransformer,
        reranker: CrossEncoder,
        bm25: BM25Okapi,
        doc_ids: List[str],
        faiss_index,
    ):
        self.corpus = corpus
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.bm25 = bm25
        self.doc_ids = doc_ids
        self.faiss_index = faiss_index
    
    @classmethod
    def load(
        cls,
        kb_dir: str = "data/knowledge_base",
        reranker_path: str = "models/id-selector-curated-v2",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ) -> "IDRetriever":
        """Load the retriever with trained reranker."""
        kb_dir = Path(kb_dir)
        
        # Load corpus
        print("Loading knowledge base...")
        corpus = {}
        doc_types = {}
        
        helpers_file = kb_dir / "helpers.jsonl"
        if helpers_file.exists():
            for line in helpers_file.read_text().strip().split('\n'):
                if not line:
                    continue
                data = json.loads(line)
                doc_id = data.get('id', '')
                
                # Enriched text for retrieval
                parts = [
                    doc_id,
                    doc_id.replace('.', ' ').replace('_', ' '),
                    data.get('signature', ''),
                    data.get('description', ''),
                ]
                id_parts = doc_id.split('.')
                parts.extend(id_parts)
                
                if 'task' in doc_id.lower():
                    parts.extend(['tasks', 'pipeline', 'tekton', 'pipelinerun'])
                if 'attestation' in doc_id.lower():
                    parts.extend(['attestation', 'provenance', 'slsa'])
                if 'result' in doc_id.lower():
                    parts.extend(['result', 'output', 'test', 'report'])
                if 'sbom' in doc_id.lower() or 'spdx' in doc_id.lower():
                    parts.extend(['sbom', 'spdx', 'cyclonedx', 'package', 'license'])
                if 'bundle' in doc_id.lower():
                    parts.extend(['bundle', 'pinned', 'digest', 'oci'])
                
                corpus[doc_id] = ' '.join(filter(None, parts))
                doc_types[doc_id] = 'helper'
        
        schemas_file = kb_dir / "schemas.jsonl"
        if schemas_file.exists():
            for line in schemas_file.read_text().strip().split('\n'):
                if not line:
                    continue
                data = json.loads(line)
                doc_id = data.get('schema_id', '')
                
                parts = [
                    data.get('canonical_path', ''),
                    data.get('description', ''),
                    data.get('attestation_type', ''),
                ]
                keywords = data.get('keywords', [])
                parts.extend(keywords)
                
                corpus[doc_id] = ' '.join(filter(None, parts))
                doc_types[doc_id] = 'schema'
        
        print(f"  Loaded {len(corpus)} documents")
        
        # Build BM25 index
        print("Building BM25 index...")
        doc_ids = list(corpus.keys())
        texts = [corpus[did] for did in doc_ids]
        tokenized = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized)
        
        # Build embedding index
        print(f"Loading embedding model: {embedding_model}")
        encoder = SentenceTransformer(embedding_model)
        embeddings = encoder.encode(texts, show_progress_bar=False)
        
        dim = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        faiss_index.add(embeddings)
        
        # Load reranker
        print(f"Loading reranker: {reranker_path}")
        reranker = CrossEncoder(reranker_path, max_length=512)
        
        instance = cls(
            corpus=corpus,
            embedding_model=encoder,
            reranker=reranker,
            bm25=bm25,
            doc_ids=doc_ids,
            faiss_index=faiss_index,
        )
        instance.doc_types = doc_types
        
        print("Retriever ready!")
        return instance
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        candidate_k: int = 100,
    ) -> RetrievalResult:
        """
        Retrieve relevant IDs for a query.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            candidate_k: Number of candidates to rerank
        
        Returns:
            RetrievalResult with helper_ids, schema_ids, and scores
        """
        # Step 1: Generate candidates with BM25 + embeddings
        candidates = self._get_candidates(query, candidate_k)
        
        # Step 2: Rerank with cross-encoder
        reranked = self._rerank(query, candidates, top_k)
        
        # Step 3: Split by type
        helper_ids = []
        schema_ids = []
        
        for doc_id, score in reranked:
            doc_type = self.doc_types.get(doc_id, 'helper')
            if doc_type == 'helper':
                helper_ids.append(doc_id)
            else:
                schema_ids.append(doc_id)
        
        return RetrievalResult(
            query=query,
            helper_ids=helper_ids,
            schema_ids=schema_ids,
            all_results=reranked,
        )
    
    def _get_candidates(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Get candidates using hybrid BM25 + embedding search."""
        results = {}
        
        bm25_k = min(top_k, len(self.doc_ids))
        emb_k = min(top_k, len(self.doc_ids))
        
        # BM25
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        
        for idx in np.argsort(bm25_scores)[-bm25_k:][::-1]:
            doc_id = self.doc_ids[idx]
            results[doc_id] = float(bm25_scores[idx]) / max_bm25
        
        # Embeddings
        query_emb = self.embedding_model.encode([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self.faiss_index.search(query_emb, emb_k)
        
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                doc_id = self.doc_ids[idx]
                results[doc_id] = results.get(doc_id, 0) + float(scores[0][i])
        
        # Sort by combined score
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    
    def _rerank(
        self, 
        query: str, 
        candidates: List[Tuple[str, float]], 
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Rerank candidates with cross-encoder."""
        if not candidates:
            return []
        
        # Create query-document pairs
        pairs = [(query, self.corpus[doc_id]) for doc_id, _ in candidates if doc_id in self.corpus]
        valid_ids = [doc_id for doc_id, _ in candidates if doc_id in self.corpus]
        
        if not pairs:
            return []
        
        # Score with cross-encoder
        scores = self.reranker.predict(pairs)
        
        # Combine and sort
        results = list(zip(valid_ids, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


def main():
    parser = argparse.ArgumentParser(description="Retrieve relevant IDs for a query")
    parser.add_argument("--query", help="Query to retrieve IDs for")
    parser.add_argument("--kb-dir", default="data/knowledge_base")
    parser.add_argument("--reranker", default="models/id-selector-curated-v2")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--show-context", action="store_true", help="Show formatted context")
    
    args = parser.parse_args()
    
    # Load retriever
    retriever = IDRetriever.load(
        kb_dir=args.kb_dir,
        reranker_path=args.reranker,
    )
    
    def process_query(query: str):
        result = retriever.retrieve(query, top_k=args.top_k)
        
        print(f"\n=== Query: {query} ===\n")
        
        print("Helpers:")
        for i, h in enumerate(result.helper_ids[:5], 1):
            print(f"  {i}. {h}")
        
        if result.schema_ids:
            print("\nSchemas:")
            for i, s in enumerate(result.schema_ids[:3], 1):
                print(f"  {i}. {s}")
        
        print(f"\nAll results (top {args.top_k}):")
        for doc_id, score in result.all_results:
            print(f"  {doc_id}: {score:.4f}")
        
        if args.show_context:
            print("\n=== Context ===")
            print(result.format_context(retriever.corpus))
    
    if args.query:
        process_query(args.query)
    
    elif args.interactive:
        print("\n=== ID Retriever Interactive Mode ===")
        print("Enter queries to find relevant helpers/schemas. Type 'quit' to exit.\n")
        
        while True:
            try:
                query = input("Query: ").strip()
                if query.lower() in ('quit', 'exit', 'q'):
                    break
                if query:
                    process_query(query)
            except KeyboardInterrupt:
                break
        
        print("\nGoodbye!")
    
    else:
        # Demo queries
        demo_queries = [
            "check if task bundle is pinned",
            "iterate over pipelinerun attestations",
            "verify SBOM packages have valid licenses",
        ]
        
        print("\n=== Demo Queries ===")
        for query in demo_queries:
            process_query(query)


if __name__ == "__main__":
    main()

