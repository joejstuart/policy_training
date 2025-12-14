#!/usr/bin/env python3
"""Simple ID Selector Training - Fast Practical Approach.

This is a simplified version that:
1. Uses cross-encoder reranking (no LLM needed for training)
2. Trains quickly on your existing data
3. Gets you to a working baseline fast

For more advanced DPO training, see train_id_selector.py
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse
import numpy as np

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
    from torch.utils.data import DataLoader
    HAS_ST = True
except ImportError:
    HAS_ST = False
    print("Error: sentence-transformers not installed")
    print("Run: pip install sentence-transformers")

try:
    from rank_bm25 import BM25Okapi
    import faiss
    HAS_RETRIEVAL = True
except ImportError:
    HAS_RETRIEVAL = False
    print("Warning: rank-bm25 or faiss not installed")


@dataclass
class Document:
    id: str
    text: str
    doc_type: str


@dataclass
class TrainingExample:
    query: str
    positive_id: str
    negative_id: str


def load_corpus(kb_dir: Path) -> Dict[str, Document]:
    """Load documents from knowledge base with enriched text for better retrieval."""
    corpus = {}
    
    helpers_file = kb_dir / "helpers.jsonl"
    if helpers_file.exists():
        for line in helpers_file.read_text().strip().split('\n'):
            if not line:
                continue
            data = json.loads(line)
            doc_id = data.get('id', '')
            
            # Enrich text with multiple representations for better matching
            parts = [
                doc_id,  # Full ID
                doc_id.replace('.', ' ').replace('_', ' '),  # ID as words
                data.get('signature', ''),
                data.get('description', ''),
            ]
            
            # Add keywords from ID
            id_parts = doc_id.split('.')
            parts.extend(id_parts)  # e.g., "lib", "tekton", "tasks"
            
            # Add common aliases/related terms
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
            
            text = ' '.join(filter(None, parts))
            corpus[doc_id] = Document(id=doc_id, text=text, doc_type="helper")
    
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
            
            # Add keywords
            keywords = data.get('keywords', [])
            parts.extend(keywords)
            
            text = ' '.join(filter(None, parts))
            corpus[doc_id] = Document(id=doc_id, text=text, doc_type="schema")
    
    print(f"Loaded {len(corpus)} documents")
    return corpus


def load_training_data(train_file: Path) -> List[TrainingExample]:
    """Load training examples."""
    examples = []
    
    for line in train_file.read_text().strip().split('\n'):
        if not line:
            continue
        data = json.loads(line)
        
        query = data.get('query', '')
        pos_id = data.get('_positive_id', '')
        neg_id = data.get('_negative_id', '')
        
        if query and pos_id:
            examples.append(TrainingExample(
                query=query,
                positive_id=pos_id,
                negative_id=neg_id,
            ))
    
    print(f"Loaded {len(examples)} training examples")
    return examples


class HybridRetriever:
    """BM25 + Embedding hybrid retrieval."""
    
    def __init__(self, corpus: Dict[str, Document], embedding_model: str):
        self.corpus = corpus
        self.doc_ids = list(corpus.keys())
        texts = [corpus[did].text for did in self.doc_ids]
        
        # BM25
        print("Building BM25 index...")
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        
        # Embeddings
        print(f"Building embedding index with {embedding_model}...")
        self.encoder = SentenceTransformer(embedding_model)
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
    
    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """Retrieve candidates using hybrid BM25 + embedding search."""
        results = {}
        
        # For small corpus (151 docs), retrieve more to ensure high recall
        bm25_k = min(top_k, len(self.doc_ids))
        emb_k = min(top_k, len(self.doc_ids))
        
        # BM25
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        for idx in np.argsort(bm25_scores)[-bm25_k:][::-1]:
            doc_id = self.doc_ids[idx]
            results[doc_id] = float(bm25_scores[idx]) / max_bm25  # Normalize
        
        # Embeddings
        query_emb = self.encoder.encode([query])
        faiss.normalize_L2(query_emb)
        scores, indices = self.index.search(query_emb, emb_k)
        
        for i, idx in enumerate(indices[0]):
            if idx >= 0:
                doc_id = self.doc_ids[idx]
                results[doc_id] = results.get(doc_id, 0) + float(scores[0][i])
        
        # Sort by combined score
        ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class CrossEncoderReranker:
    """Train a cross-encoder to rerank candidates."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
    
    def train(
        self,
        examples: List[TrainingExample],
        corpus: Dict[str, Document],
        output_dir: str,
        epochs: int = 2,
        batch_size: int = 16,
    ):
        """Train the reranker."""
        # Create training pairs
        train_samples = []
        
        for ex in examples:
            if ex.positive_id not in corpus:
                continue
            
            pos_text = corpus[ex.positive_id].text
            
            # Positive pair
            train_samples.append(InputExample(
                texts=[ex.query, pos_text],
                label=1.0
            ))
            
            # Negative pair
            if ex.negative_id and ex.negative_id in corpus:
                neg_text = corpus[ex.negative_id].text
                train_samples.append(InputExample(
                    texts=[ex.query, neg_text],
                    label=0.0
                ))
            else:
                # Random negative
                rand_id = random.choice(list(corpus.keys()))
                if rand_id != ex.positive_id:
                    train_samples.append(InputExample(
                        texts=[ex.query, corpus[rand_id].text],
                        label=0.0
                    ))
        
        print(f"Training on {len(train_samples)} pairs for {epochs} epochs...")
        
        # Train
        train_dataloader = DataLoader(
            train_samples, 
            shuffle=True, 
            batch_size=batch_size
        )
        
        self.model.fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            warmup_steps=100,
            output_path=output_dir,
            show_progress_bar=True,
        )
        
        print(f"Model saved to {output_dir}")
    
    def rerank(
        self, 
        query: str, 
        candidates: List[Tuple[str, float]], 
        corpus: Dict[str, Document],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Rerank candidates."""
        if not candidates:
            return []
        
        # Create pairs
        pairs = [(query, corpus[doc_id].text) for doc_id, _ in candidates if doc_id in corpus]
        valid_ids = [doc_id for doc_id, _ in candidates if doc_id in corpus]
        
        if not pairs:
            return []
        
        # Score
        scores = self.model.predict(pairs)
        
        # Combine with original scores (optional)
        results = [(doc_id, float(score)) for doc_id, score in zip(valid_ids, scores)]
        
        # Sort by reranker score
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]


def evaluate(
    retriever: HybridRetriever,
    reranker: CrossEncoderReranker,
    examples: List[TrainingExample],
    corpus: Dict[str, Document],
    k_values: List[int] = [1, 3, 5, 10],
    verbose: bool = False,
) -> Dict[str, float]:
    """Evaluate retrieval + reranking."""
    
    recalls = {k: [] for k in k_values}
    retrieval_only_recalls = {k: [] for k in k_values}
    mrrs = []
    retrieval_mrrs = []
    
    # Track candidate generation quality
    candidate_recalls = []
    
    # Sample for speed
    sample = random.sample(examples, min(500, len(examples)))
    
    misses = []
    
    for ex in sample:
        # Check if target exists in corpus
        if ex.positive_id not in corpus:
            continue
        
        # Retrieve - use 100 candidates for small corpus
        candidates = retriever.retrieve(ex.query, top_k=100)
        candidate_ids = [doc_id for doc_id, _ in candidates]
        
        # Track if target is in candidates at all
        in_candidates = ex.positive_id in candidate_ids
        candidate_recalls.append(1.0 if in_candidates else 0.0)
        
        # Evaluate RETRIEVAL ONLY (before reranking)
        for k in k_values:
            if ex.positive_id in candidate_ids[:k]:
                retrieval_only_recalls[k].append(1.0)
            else:
                retrieval_only_recalls[k].append(0.0)
        
        for i, doc_id in enumerate(candidate_ids):
            if doc_id == ex.positive_id:
                retrieval_mrrs.append(1.0 / (i + 1))
                break
        else:
            retrieval_mrrs.append(0.0)
        
        # Rerank
        reranked = reranker.rerank(ex.query, candidates, corpus, top_k=max(k_values))
        
        # Evaluate AFTER RERANKING
        retrieved_ids = [doc_id for doc_id, _ in reranked]
        
        for k in k_values:
            if ex.positive_id in retrieved_ids[:k]:
                recalls[k].append(1.0)
            else:
                recalls[k].append(0.0)
        
        # MRR
        found = False
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id == ex.positive_id:
                mrrs.append(1.0 / (i + 1))
                found = True
                break
        if not found:
            mrrs.append(0.0)
            if len(misses) < 5:
                misses.append((ex.query[:80], ex.positive_id, in_candidates))
    
    # Results with both retrieval-only and reranked
    results = {}
    results["candidate_recall@100"] = np.mean(candidate_recalls)
    
    # Print comparison
    print(f"\n  Candidate generation recall@100: {results['candidate_recall@100']:.4f}")
    print(f"\n  Comparison (retrieval-only vs reranked):")
    print(f"  {'Metric':<15} {'Retrieval':>12} {'Reranked':>12} {'Delta':>10}")
    print(f"  {'-'*50}")
    
    for k in k_values:
        ret_val = np.mean(retrieval_only_recalls[k])
        rer_val = np.mean(recalls[k])
        delta = rer_val - ret_val
        print(f"  {'recall@'+str(k):<15} {ret_val:>12.4f} {rer_val:>12.4f} {delta:>+10.4f}")
        results[f"retrieval_recall@{k}"] = ret_val
        results[f"reranked_recall@{k}"] = rer_val
    
    ret_mrr = np.mean(retrieval_mrrs)
    rer_mrr = np.mean(mrrs)
    print(f"  {'MRR':<15} {ret_mrr:>12.4f} {rer_mrr:>12.4f} {rer_mrr - ret_mrr:>+10.4f}")
    results["retrieval_mrr"] = ret_mrr
    results["reranked_mrr"] = rer_mrr
    
    if misses and verbose:
        print("\n  Sample misses (after reranking):")
        for query, target, in_cand in misses:
            status = "in candidates" if in_cand else "NOT in candidates"
            print(f"    Query: {query}...")
            print(f"    Target: {target} ({status})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Simple ID Selector Training")
    parser.add_argument("--kb-dir", default="data/knowledge_base")
    parser.add_argument("--train-file", default="data/training/retrieval/retrieval_train.jsonl")
    parser.add_argument("--eval-file", default="data/training/retrieval/retrieval_eval.jsonl")
    parser.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--output-dir", default="models/id-selector-simple")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-train", type=int, default=10000, help="Max training examples")
    parser.add_argument("--verbose", action="store_true", help="Show detailed diagnostics")
    
    args = parser.parse_args()
    
    if not HAS_ST or not HAS_RETRIEVAL:
        print("Missing dependencies. Install with:")
        print("pip install sentence-transformers rank-bm25 faiss-cpu")
        return
    
    # Load data
    corpus = load_corpus(Path(args.kb_dir))
    
    train_file = Path(args.train_file)
    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        return
    
    examples = load_training_data(train_file)
    
    # Limit training data for speed
    if len(examples) > args.max_train:
        print(f"Sampling {args.max_train} examples for training")
        examples = random.sample(examples, args.max_train)
    
    # Build retriever
    print("\n=== Building Retriever ===")
    retriever = HybridRetriever(corpus, args.embedding_model)
    
    # Load or train reranker
    output_dir = Path(args.output_dir)
    
    if args.eval_only and output_dir.exists():
        print(f"\n=== Loading Reranker from {output_dir} ===")
        # CrossEncoder saves config.json - check if it's a valid model
        config_file = output_dir / "config.json"
        if config_file.exists():
            try:
                reranker = CrossEncoderReranker(str(output_dir))
            except ValueError:
                print("  Saved model format incompatible, using base model for eval")
                reranker = CrossEncoderReranker(args.reranker_model)
        else:
            print("  No saved model found, using base model for eval")
            reranker = CrossEncoderReranker(args.reranker_model)
    else:
        print("\n=== Training Reranker ===")
        reranker = CrossEncoderReranker(args.reranker_model)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        reranker.train(
            examples=examples,
            corpus=corpus,
            output_dir=str(output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    
    # Evaluate
    print("\n=== Evaluation ===")
    
    # Load eval data if available
    eval_file = Path(args.eval_file)
    if eval_file.exists():
        eval_examples = load_training_data(eval_file)
    else:
        print("Using train split for eval")
        eval_examples = examples
    
    results = evaluate(retriever, reranker, eval_examples, corpus, verbose=args.verbose)
    
    print("\nResults:")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")
    
    # Save results
    results_file = output_dir / "eval_results.json"
    results_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()

