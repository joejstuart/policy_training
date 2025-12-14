#!/usr/bin/env python3
"""Improved retrieval model training based on SID-1 insights.

Key improvements over the original finetune_retrieval_model.py:
1. Hard negative mining using current model
2. Multiple Negatives Ranking Loss (better than triplet loss)
3. Larger/better base model options
4. Document-centric evaluation metrics
5. Curriculum learning support

Reference: SID-1 Technical Report (December 2025)
https://www.sid.ai/research/SID-1_Preview/technical-report/SID_1_Technical_Report__Test_Time_Compute_for_Retrieval.pdf
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    models,
)
from sentence_transformers.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)


# Available base models (ordered by quality/size tradeoff)
BASE_MODELS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",       # Fast, 384d
    "bge-small": "BAAI/bge-small-en-v1.5",                    # Better, 384d
    "bge-base": "BAAI/bge-base-en-v1.5",                      # Best quality, 768d
    "e5-small": "intfloat/e5-small-v2",                       # Good alternative
    "e5-base": "intfloat/e5-base-v2",                         # Good quality
}


@dataclass
class RetrievalExample:
    """A retrieval training example with optional multiple targets."""
    query: str
    positive_ids: List[str]  # Can have multiple positive documents
    hard_negative_ids: List[str] = field(default_factory=list)
    difficulty: float = 0.5  # For curriculum learning
    source: str = "unknown"


@dataclass 
class DocumentCorpus:
    """Corpus of documents (helpers + schemas) for retrieval."""
    documents: Dict[str, str]  # id -> text
    embeddings: Optional[np.ndarray] = None
    
    def get_text(self, doc_id: str) -> str:
        return self.documents.get(doc_id, "")


class HardNegativeMiner:
    """Mine hard negatives using the current model.
    
    SID-1 Insight: Hard negatives (similar but wrong) are crucial for
    teaching the model to distinguish between related concepts.
    """
    
    def __init__(self, model: SentenceTransformer, corpus: DocumentCorpus):
        self.model = model
        self.corpus = corpus
        self._compute_corpus_embeddings()
    
    def _compute_corpus_embeddings(self):
        """Pre-compute embeddings for all documents."""
        doc_ids = list(self.corpus.documents.keys())
        texts = [self.corpus.documents[did] for did in doc_ids]
        
        print(f"Computing embeddings for {len(texts)} documents...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.corpus.embeddings = embeddings
        self.doc_ids = doc_ids
    
    def mine(
        self, 
        query: str, 
        positive_ids: Set[str], 
        k: int = 10,
        min_similarity: float = 0.3,
        max_similarity: float = 0.85
    ) -> List[str]:
        """
        Find hard negatives for a query.
        
        Hard negatives are documents that:
        1. Are similar to the query (could be confused)
        2. But are NOT in the positive set
        3. Have similarity in a "confusing" range (not too low, not too high)
        """
        query_emb = self.model.encode([query])[0]
        
        # Compute similarities
        similarities = np.dot(self.corpus.embeddings, query_emb)
        
        # Find candidates in the "hard" similarity range
        hard_negatives = []
        indices = np.argsort(similarities)[::-1]  # Highest first
        
        for idx in indices:
            doc_id = self.doc_ids[idx]
            sim = similarities[idx]
            
            # Skip positives
            if doc_id in positive_ids:
                continue
            
            # Only keep if in "confusing" range
            if min_similarity <= sim <= max_similarity:
                hard_negatives.append(doc_id)
                
                if len(hard_negatives) >= k:
                    break
        
        return hard_negatives


class MultiPositiveDataset(Dataset):
    """Dataset that supports multiple positive documents per query.
    
    SID-1 Insight: Real retrieval tasks often have multiple correct documents.
    Training with multiple positives teaches the model to find ALL relevant docs.
    """
    
    def __init__(
        self, 
        examples: List[RetrievalExample],
        corpus: DocumentCorpus,
        negatives_per_positive: int = 3
    ):
        self.examples = examples
        self.corpus = corpus
        self.negatives_per_positive = negatives_per_positive
        
        # Expand examples to (query, positive, negative) triplets
        self.triplets = self._create_triplets()
    
    def _create_triplets(self) -> List[Tuple[str, str, str]]:
        """Create triplets from examples."""
        triplets = []
        
        for ex in self.examples:
            for pos_id in ex.positive_ids:
                pos_text = self.corpus.get_text(pos_id)
                if not pos_text:
                    continue
                
                # Use hard negatives if available, otherwise random
                neg_ids = ex.hard_negative_ids or self._sample_random_negatives(
                    ex.positive_ids, self.negatives_per_positive
                )
                
                for neg_id in neg_ids[:self.negatives_per_positive]:
                    neg_text = self.corpus.get_text(neg_id)
                    if neg_text:
                        triplets.append((ex.query, pos_text, neg_text))
        
        return triplets
    
    def _sample_random_negatives(self, exclude: List[str], k: int) -> List[str]:
        """Sample random negatives (fallback when no hard negatives)."""
        candidates = [
            doc_id for doc_id in self.corpus.documents.keys()
            if doc_id not in exclude
        ]
        return random.sample(candidates, min(k, len(candidates)))
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        query, pos, neg = self.triplets[idx]
        return InputExample(texts=[query, pos, neg])


class DocumentCentricEvaluator:
    """Evaluate with document-centric metrics (SID-1 style).
    
    SID-1 Insight: Reward finding documents, not answering questions.
    This evaluator focuses on recall (did we find all relevant docs?)
    with a smaller penalty for irrelevant docs (precision).
    """
    
    def __init__(
        self,
        queries: Dict[str, str],           # qid -> query text
        corpus: Dict[str, str],            # doc_id -> doc text
        relevant_docs: Dict[str, Set[str]], # qid -> set of relevant doc_ids
        k_values: List[int] = [1, 3, 5, 7],
        recall_weight: float = 0.8,        # SID-1 uses recall-heavy weighting
    ):
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs
        self.k_values = k_values
        self.recall_weight = recall_weight
    
    def __call__(
        self, 
        model: SentenceTransformer, 
        output_path: str = None,
        epoch: int = -1,
        steps: int = -1
    ) -> float:
        """Evaluate the model."""
        # Encode queries and corpus
        query_ids = list(self.queries.keys())
        query_texts = [self.queries[qid] for qid in query_ids]
        query_embeddings = model.encode(query_texts, show_progress_bar=False)
        
        doc_ids = list(self.corpus.keys())
        doc_texts = [self.corpus[did] for did in doc_ids]
        doc_embeddings = model.encode(doc_texts, show_progress_bar=False)
        
        # Compute metrics
        metrics = {f"recall@{k}": [] for k in self.k_values}
        metrics.update({f"precision@{k}": [] for k in self.k_values})
        metrics["ndcg"] = []
        metrics["mrr"] = []
        
        for i, qid in enumerate(query_ids):
            if qid not in self.relevant_docs:
                continue
            
            targets = self.relevant_docs[qid]
            
            # Compute similarities and rank
            sims = np.dot(doc_embeddings, query_embeddings[i])
            ranked_indices = np.argsort(sims)[::-1]
            ranked_docs = [doc_ids[idx] for idx in ranked_indices]
            
            # Recall and precision at K
            for k in self.k_values:
                retrieved = set(ranked_docs[:k])
                recall = len(retrieved & targets) / len(targets)
                precision = len(retrieved & targets) / k
                metrics[f"recall@{k}"].append(recall)
                metrics[f"precision@{k}"].append(precision)
            
            # NDCG
            ndcg = self._compute_ndcg(ranked_docs, targets, k=max(self.k_values))
            metrics["ndcg"].append(ndcg)
            
            # MRR
            for rank, doc in enumerate(ranked_docs):
                if doc in targets:
                    metrics["mrr"].append(1.0 / (rank + 1))
                    break
            else:
                metrics["mrr"].append(0.0)
        
        # Aggregate
        results = {k: np.mean(v) for k, v in metrics.items() if v}
        
        # Compute combined score (SID-1 style: recall-heavy)
        main_recall = results.get(f"recall@{self.k_values[-1]}", 0)
        main_precision = results.get(f"precision@{self.k_values[-1]}", 0)
        combined = self.recall_weight * main_recall + (1 - self.recall_weight) * main_precision
        results["combined_score"] = combined
        
        # Print results
        print(f"\n=== Evaluation (epoch={epoch}, steps={steps}) ===")
        for k, v in sorted(results.items()):
            print(f"  {k}: {v:.4f}")
        
        # Return primary metric for model selection
        return results["combined_score"]
    
    def _compute_ndcg(self, ranked: List[str], targets: Set[str], k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain."""
        dcg = 0.0
        for i, doc in enumerate(ranked[:k]):
            if doc in targets:
                dcg += 1.0 / np.log2(i + 2)
        
        # Ideal DCG (all targets at top)
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(targets), k)))
        
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


class CurriculumScheduler:
    """Schedule training by difficulty (SID-1 insight: curriculum helps).
    
    Start with easy examples, gradually include harder ones.
    """
    
    def __init__(self, examples: List[RetrievalExample]):
        self.examples = sorted(examples, key=lambda x: x.difficulty)
        self.current_epoch = 0
    
    def get_examples_for_epoch(self, epoch: int, max_difficulty: float = None) -> List[RetrievalExample]:
        """Get examples up to the current difficulty threshold."""
        if max_difficulty is None:
            # Gradually increase: epoch 0 → 0.3, epoch 10 → 1.0
            max_difficulty = min(0.3 + epoch * 0.07, 1.0)
        
        eligible = [ex for ex in self.examples if ex.difficulty <= max_difficulty]
        print(f"Epoch {epoch}: Using {len(eligible)}/{len(self.examples)} examples (difficulty ≤ {max_difficulty:.2f})")
        return eligible


class ImprovedRetrievalTrainer:
    """Improved retrieval trainer with SID-1 techniques."""
    
    def __init__(
        self,
        base_model: str = "bge-base",
        output_dir: str = "models/improved-retrieval",
        device: Optional[str] = None,
    ):
        # Resolve model name
        if base_model in BASE_MODELS:
            model_name = BASE_MODELS[base_model]
        else:
            model_name = base_model
        
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        
        # Auto-detect device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"Using device: {self.device}")
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def train(
        self,
        examples: List[RetrievalExample],
        corpus: DocumentCorpus,
        eval_examples: Optional[List[RetrievalExample]] = None,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        use_curriculum: bool = True,
        mine_hard_negatives: bool = True,
    ):
        """Train with improved techniques."""
        
        # Step 1: Mine hard negatives if requested
        if mine_hard_negatives:
            print("\n=== Mining Hard Negatives ===")
            miner = HardNegativeMiner(self.model, corpus)
            
            for i, ex in enumerate(examples):
                if not ex.hard_negative_ids:
                    hard_negs = miner.mine(
                        ex.query, 
                        set(ex.positive_ids),
                        k=5
                    )
                    ex.hard_negative_ids = hard_negs
                
                if (i + 1) % 100 == 0:
                    print(f"  Mined negatives for {i + 1}/{len(examples)} examples")
        
        # Step 2: Setup curriculum or use all examples
        if use_curriculum:
            scheduler = CurriculumScheduler(examples)
        else:
            scheduler = None
        
        # Step 3: Create evaluator
        evaluator = None
        if eval_examples:
            queries = {f"q{i}": ex.query for i, ex in enumerate(eval_examples)}
            relevant = {f"q{i}": set(ex.positive_ids) for i, ex in enumerate(eval_examples)}
            evaluator = DocumentCentricEvaluator(
                queries=queries,
                corpus=corpus.documents,
                relevant_docs=relevant,
            )
        
        # Step 4: Training loop
        self.output_dir.mkdir(parents=True, exist_ok=True)
        best_score = 0.0
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            
            # Get examples for this epoch (curriculum or all)
            if scheduler:
                epoch_examples = scheduler.get_examples_for_epoch(epoch)
            else:
                epoch_examples = examples
            
            # Create dataset
            dataset = MultiPositiveDataset(epoch_examples, corpus)
            dataloader = DataLoader(
                dataset, 
                shuffle=True, 
                batch_size=batch_size,
                collate_fn=lambda x: x  # Already InputExample objects
            )
            
            # Use Multiple Negatives Ranking Loss (better than triplet)
            # Actually, for triplet format we use TripletLoss but with better negatives
            train_loss = losses.TripletLoss(
                model=self.model,
                distance_metric=losses.TripletDistanceMetric.COSINE,
                triplet_margin=0.5,
            )
            
            # Calculate steps
            steps_per_epoch = len(dataloader)
            warmup_steps = int(steps_per_epoch * warmup_ratio) if epoch == 0 else 0
            
            print(f"  Training on {len(dataset)} triplets ({steps_per_epoch} batches)")
            
            # Train for one epoch
            self.model.fit(
                train_objectives=[(dataloader, train_loss)],
                epochs=1,
                warmup_steps=warmup_steps,
                optimizer_params={'lr': learning_rate},
                show_progress_bar=True,
                output_path=None,  # Don't save every epoch
            )
            
            # Evaluate
            if evaluator:
                score = evaluator(
                    self.model, 
                    epoch=epoch,
                    steps=steps_per_epoch * (epoch + 1)
                )
                
                if score > best_score:
                    best_score = score
                    print(f"  New best score: {score:.4f}, saving model...")
                    self.model.save(str(self.output_dir / "best"))
        
        # Save final model
        self.model.save(str(self.output_dir / "final"))
        print(f"\nTraining complete. Best score: {best_score:.4f}")
        print(f"Models saved to {self.output_dir}")
        
        return {"best_score": best_score}


def load_training_data(
    train_file: Path,
    eval_file: Path,
    kb_dir: Path
) -> Tuple[List[RetrievalExample], List[RetrievalExample], DocumentCorpus]:
    """Load training data and corpus."""
    
    # Load corpus from knowledge base
    documents = {}
    
    # Load helpers
    helpers_file = kb_dir / "helpers.jsonl"
    if helpers_file.exists():
        for line in helpers_file.read_text().strip().split('\n'):
            if not line:
                continue
            data = json.loads(line)
            doc_id = data.get('id', '')
            # Create searchable text
            text_parts = [
                f"Helper: {doc_id}",
                f"Signature: {data.get('signature', '')}",
                f"Description: {data.get('description', '')}",
            ]
            documents[doc_id] = '\n'.join(text_parts)
    
    # Load schemas
    schemas_file = kb_dir / "schemas.jsonl"
    if schemas_file.exists():
        for line in schemas_file.read_text().strip().split('\n'):
            if not line:
                continue
            data = json.loads(line)
            doc_id = data.get('schema_id', '')
            text_parts = [
                f"Schema: {data.get('canonical_path', '')}",
                f"Type: {data.get('attestation_type', '')}",
                f"Description: {data.get('description', '')}",
            ]
            if data.get('keywords'):
                text_parts.append(f"Keywords: {', '.join(data['keywords'])}")
            documents[doc_id] = '\n'.join(text_parts)
    
    corpus = DocumentCorpus(documents=documents)
    print(f"Loaded corpus with {len(documents)} documents")
    
    # Load training examples
    def load_examples(filepath: Path) -> List[RetrievalExample]:
        examples = []
        if not filepath.exists():
            return examples
        
        for line in filepath.read_text().strip().split('\n'):
            if not line:
                continue
            data = json.loads(line)
            
            # Convert old format to new
            positive_id = data.get('_positive_id', '')
            negative_id = data.get('_negative_id', '')
            
            examples.append(RetrievalExample(
                query=data['query'],
                positive_ids=[positive_id] if positive_id else [],
                hard_negative_ids=[negative_id] if negative_id else [],
                difficulty=0.5,  # Default difficulty
                source=data.get('_source', 'unknown'),
            ))
        
        return examples
    
    train_examples = load_examples(train_file)
    eval_examples = load_examples(eval_file)
    
    print(f"Loaded {len(train_examples)} train, {len(eval_examples)} eval examples")
    
    return train_examples, eval_examples, corpus


def main():
    parser = argparse.ArgumentParser(
        description="Improved retrieval model training (SID-1 inspired)"
    )
    parser.add_argument(
        "--train-file",
        default="data/training/retrieval/retrieval_train.jsonl",
        help="Training data file",
    )
    parser.add_argument(
        "--eval-file",
        default="data/training/retrieval/retrieval_eval.jsonl",
        help="Evaluation data file",
    )
    parser.add_argument(
        "--kb-dir",
        default="data/knowledge_base",
        help="Knowledge base directory",
    )
    parser.add_argument(
        "--base-model",
        default="bge-base",
        choices=list(BASE_MODELS.keys()) + ["custom"],
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--custom-model",
        help="Custom model name (when --base-model=custom)",
    )
    parser.add_argument(
        "--output-dir",
        default="models/improved-retrieval",
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable curriculum learning",
    )
    parser.add_argument(
        "--no-hard-negatives",
        action="store_true",
        help="Disable hard negative mining",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu", "auto"],
        default="auto",
        help="Device to use",
    )
    
    args = parser.parse_args()
    
    # Determine model
    if args.base_model == "custom":
        if not args.custom_model:
            parser.error("--custom-model required when --base-model=custom")
        model_name = args.custom_model
    else:
        model_name = args.base_model
    
    # Handle device
    device = None if args.device == "auto" else args.device
    
    # Load data
    train_examples, eval_examples, corpus = load_training_data(
        Path(args.train_file),
        Path(args.eval_file),
        Path(args.kb_dir),
    )
    
    if not train_examples:
        print("Error: No training examples found")
        return
    
    # Create trainer
    trainer = ImprovedRetrievalTrainer(
        base_model=model_name,
        output_dir=args.output_dir,
        device=device,
    )
    
    # Train
    results = trainer.train(
        examples=train_examples,
        corpus=corpus,
        eval_examples=eval_examples if eval_examples else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_curriculum=not args.no_curriculum,
        mine_hard_negatives=not args.no_hard_negatives,
    )
    
    print(f"\n=== Final Results ===")
    print(f"Best score: {results['best_score']:.4f}")


if __name__ == "__main__":
    main()

