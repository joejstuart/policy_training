#!/usr/bin/env python3
"""Inference with trained agentic retrieval model.

This script demonstrates how to use the trained agentic retrieval model
for actual document retrieval at inference time.

Key features:
- Multi-turn iterative search
- Adaptive query refinement
- Configurable compute budget (turns)
- Can be used as a sub-agent in larger systems (SID-1 composability)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: Install sentence-transformers and faiss-cpu for full functionality")


@dataclass
class RetrievalResult:
    """Result from agentic retrieval."""
    query: str
    retrieved_docs: List[Dict]  # [{id, text, score}]
    turns_used: int
    search_history: List[Dict]
    
    def get_ids(self) -> List[str]:
        return [doc["id"] for doc in self.retrieved_docs]
    
    def format_for_context(self, max_chars: int = 5000) -> str:
        """Format retrieved docs as context for downstream model."""
        lines = []
        chars_used = 0
        
        for doc in self.retrieved_docs:
            doc_text = f"--- {doc['id']} ---\n{doc['text']}\n"
            if chars_used + len(doc_text) > max_chars:
                break
            lines.append(doc_text)
            chars_used += len(doc_text)
        
        return "\n".join(lines)


class AgenticRetriever:
    """
    Multi-turn agentic retriever for production use.
    
    Usage:
        retriever = AgenticRetriever.load("models/agentic-retrieval")
        result = retriever.retrieve("check if task bundle is pinned")
        context = result.format_for_context()
    """
    
    SYSTEM_PROMPT = """You are a retrieval agent. Your task is to find all relevant documents for a query by searching iteratively.

Actions:
1. SEARCH: <query> - Execute a search
2. REPORT: <doc_ids> - Submit final results

Be concise. Find relevant helpers AND schemas."""

    def __init__(
        self,
        model,
        tokenizer,
        corpus: Dict[str, str],
        embedding_model: Optional[SentenceTransformer] = None,
        max_turns: int = 3,  # Fewer turns at inference for speed
        top_k: int = 7,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.corpus = corpus  # {doc_id: text}
        self.max_turns = max_turns
        self.top_k = top_k
        
        # Build search index
        if embedding_model and HAS_DEPS:
            self._build_index(embedding_model)
        else:
            self.index = None
            print("Warning: No embedding model, using keyword search")
    
    def _build_index(self, model: SentenceTransformer):
        """Build FAISS index."""
        self.doc_ids = list(self.corpus.keys())
        texts = [self.corpus[did] for did in self.doc_ids]
        
        embeddings = model.encode(texts, show_progress_bar=False)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.embedding_model = model
    
    @classmethod
    def load(
        cls,
        model_path: str,
        kb_dir: str = "data/knowledge_base",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        device: str = "auto",
        **kwargs
    ) -> "AgenticRetriever":
        """Load a trained agentic retriever."""
        # Load LLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device if device != "auto" else "auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load corpus
        corpus = cls._load_corpus(Path(kb_dir))
        
        # Load embedding model
        emb_model = None
        if HAS_DEPS:
            emb_model = SentenceTransformer(embedding_model)
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            corpus=corpus,
            embedding_model=emb_model,
            **kwargs
        )
    
    @staticmethod
    def _load_corpus(kb_dir: Path) -> Dict[str, str]:
        """Load corpus from KB."""
        corpus = {}
        
        helpers_file = kb_dir / "helpers.jsonl"
        if helpers_file.exists():
            for line in helpers_file.read_text().strip().split('\n'):
                if not line:
                    continue
                data = json.loads(line)
                doc_id = data.get('id', '')
                text = f"Helper: {doc_id}\nSignature: {data.get('signature', '')}\nDescription: {data.get('description', '')}"
                corpus[doc_id] = text
        
        schemas_file = kb_dir / "schemas.jsonl"
        if schemas_file.exists():
            for line in schemas_file.read_text().strip().split('\n'):
                if not line:
                    continue
                data = json.loads(line)
                doc_id = data.get('schema_id', '')
                text = f"Schema: {data.get('canonical_path', '')}\nType: {data.get('attestation_type', '')}\nDescription: {data.get('description', '')}"
                corpus[doc_id] = text
        
        return corpus
    
    def retrieve(self, query: str, verbose: bool = False) -> RetrievalResult:
        """
        Retrieve relevant documents for a query.
        
        The agent will iteratively search and refine until it's satisfied
        or max_turns is reached.
        """
        search_history = []
        retrieved_so_far = set()
        turn = 0
        
        while turn < self.max_turns:
            turn += 1
            
            # Generate action
            action, action_query = self._generate_action(
                query, search_history, retrieved_so_far, turn
            )
            
            if verbose:
                print(f"Turn {turn}: {action} - {action_query[:50] if action_query else ''}")
            
            if action == "report":
                break
            
            elif action == "search":
                search_query = action_query or query
                results = self._search(search_query)
                
                search_history.append({
                    "turn": turn,
                    "query": search_query,
                    "results": results[:5],  # Top 5 for history
                })
                
                for doc in results:
                    retrieved_so_far.add(doc["id"])
        
        # Rank final results by score
        final_docs = []
        for doc_id in retrieved_so_far:
            if doc_id in self.corpus:
                final_docs.append({
                    "id": doc_id,
                    "text": self.corpus[doc_id],
                    "score": 1.0,  # TODO: track actual scores
                })
        
        # Limit to top_k
        final_docs = final_docs[:self.top_k]
        
        return RetrievalResult(
            query=query,
            retrieved_docs=final_docs,
            turns_used=turn,
            search_history=search_history,
        )
    
    def _generate_action(
        self,
        original_query: str,
        search_history: List[Dict],
        retrieved_so_far: Set[str],
        turn: int,
    ) -> tuple:
        """Generate next action using the LLM."""
        # Format prompt
        prompt = self._format_prompt(
            original_query, search_history, retrieved_so_far, turn
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.3,  # Lower temperature at inference
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return self._parse_action(response, original_query)
    
    def _format_prompt(
        self,
        original_query: str,
        search_history: List[Dict],
        retrieved_so_far: Set[str],
        turn: int,
    ) -> str:
        """Format state into prompt."""
        # Format history
        if search_history:
            history_lines = []
            for h in search_history[-2:]:  # Last 2
                result_ids = [r["id"][:30] for r in h["results"][:3]]
                history_lines.append(f"  T{h['turn']}: '{h['query']}' → {result_ids}")
            history_str = "\n".join(history_lines)
        else:
            history_str = "  (none)"
        
        user_content = f"""Turn {turn}/{self.max_turns}
Query: {original_query}

History:
{history_str}

Found: {len(retrieved_so_far)} docs

Action (SEARCH: <query> or REPORT):"""
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def _parse_action(self, response: str, default_query: str) -> tuple:
        """Parse action from response."""
        response_upper = response.upper()
        
        if "REPORT" in response_upper:
            return "report", None
        
        if "SEARCH:" in response_upper:
            # Extract query
            import re
            match = re.search(r'SEARCH:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            if match:
                return "search", match.group(1).strip()
        
        # Default: search with original query (turn 1) or report (later)
        if not any(s for s in []):
            return "search", default_query
        return "report", None
    
    def _search(self, query: str) -> List[Dict]:
        """Execute a search query."""
        if self.index is not None and HAS_DEPS:
            query_emb = self.embedding_model.encode([query])
            faiss.normalize_L2(query_emb)
            
            scores, indices = self.index.search(query_emb, self.top_k * 2)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx >= 0:
                    doc_id = self.doc_ids[idx]
                    results.append({
                        "id": doc_id,
                        "text": self.corpus[doc_id],
                        "score": float(scores[0][i]),
                    })
            return results
        else:
            # Keyword fallback
            query_words = query.lower().split()
            scored = []
            for doc_id, text in self.corpus.items():
                text_lower = text.lower()
                score = sum(1 for w in query_words if w in text_lower)
                if score > 0:
                    scored.append({
                        "id": doc_id,
                        "text": text,
                        "score": score,
                    })
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:self.top_k * 2]


class SimpleRetriever:
    """
    Simple single-step retriever (baseline for comparison).
    
    No multi-turn iteration, just embed and search.
    """
    
    def __init__(
        self,
        corpus: Dict[str, str],
        embedding_model: SentenceTransformer,
        top_k: int = 7,
    ):
        self.corpus = corpus
        self.top_k = top_k
        
        # Build index
        self.doc_ids = list(corpus.keys())
        texts = [corpus[did] for did in self.doc_ids]
        
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.embedding_model = embedding_model
    
    def retrieve(self, query: str) -> RetrievalResult:
        """Single-step retrieval."""
        query_emb = self.embedding_model.encode([query])
        faiss.normalize_L2(query_emb)
        
        scores, indices = self.index.search(query_emb, self.top_k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                doc_id = self.doc_ids[idx]
                results.append({
                    "id": doc_id,
                    "text": self.corpus[doc_id],
                    "score": float(scores[0][i]),
                })
        
        return RetrievalResult(
            query=query,
            retrieved_docs=results,
            turns_used=1,
            search_history=[{"turn": 1, "query": query, "results": results}],
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run agentic retrieval inference"
    )
    parser.add_argument(
        "--model-path",
        default="models/agentic-retrieval",
        help="Path to trained model",
    )
    parser.add_argument(
        "--kb-dir",
        default="data/knowledge_base",
        help="Knowledge base directory",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model for search",
    )
    parser.add_argument(
        "--query",
        help="Query to retrieve documents for",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=3,
        help="Maximum turns for retrieval",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=7,
        help="Number of documents to return",
    )
    parser.add_argument(
        "--compare-simple",
        action="store_true",
        help="Compare with simple single-step retrieval",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        print("Using untrained base model for demo...")
        args.model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # Load retriever
    print("Loading agentic retriever...")
    retriever = AgenticRetriever.load(
        args.model_path,
        kb_dir=args.kb_dir,
        embedding_model=args.embedding_model,
        max_turns=args.max_turns,
        top_k=args.top_k,
    )
    
    # Optional: load simple retriever for comparison
    simple_retriever = None
    if args.compare_simple and HAS_DEPS:
        print("Loading simple retriever for comparison...")
        simple_retriever = SimpleRetriever(
            corpus=retriever.corpus,
            embedding_model=retriever.embedding_model,
            top_k=args.top_k,
        )
    
    def process_query(query: str):
        """Process a single query."""
        print(f"\n=== Query: {query} ===\n")
        
        # Agentic retrieval
        result = retriever.retrieve(query, verbose=args.verbose)
        
        print(f"Agentic Retrieval (turns: {result.turns_used}):")
        for i, doc in enumerate(result.retrieved_docs[:5]):
            print(f"  {i+1}. {doc['id']}")
            if args.verbose:
                preview = doc['text'][:100].replace('\n', ' ')
                print(f"      {preview}...")
        
        # Simple comparison
        if simple_retriever:
            simple_result = simple_retriever.retrieve(query)
            print(f"\nSimple Retrieval (single-step):")
            for i, doc in enumerate(simple_result.retrieved_docs[:5]):
                print(f"  {i+1}. {doc['id']}")
        
        return result
    
    # Process query or run interactive mode
    if args.query:
        process_query(args.query)
    
    elif args.interactive:
        print("\n=== Agentic Retrieval Interactive Mode ===")
        print("Enter queries to retrieve documents. Type 'quit' to exit.\n")
        
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
            "How do I check if a task bundle is pinned?",
            "What helpers are available for iterating over attestations?",
            "Check all SBOM packages have valid licenses",
        ]
        
        print("\n=== Demo Queries ===")
        for query in demo_queries:
            process_query(query)


if __name__ == "__main__":
    main()

