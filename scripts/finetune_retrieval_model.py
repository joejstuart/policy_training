#!/usr/bin/env python3
"""Fine-tune a sentence transformer model for Rego retrieval.

Uses triplet loss with hard negatives to train the model to:
1. Rank relevant schemas higher for policy queries
2. Rank relevant helpers higher for policy queries

Output: A fine-tuned model that can be used in place of the base model.
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

import torch
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
    models,
)
from sentence_transformers.evaluation import TripletEvaluator


class RetrievalDataLoader:
    """Load retrieval training data for sentence-transformers."""
    
    def __init__(self, train_file: Path, eval_file: Optional[Path] = None):
        self.train_file = Path(train_file)
        self.eval_file = Path(eval_file) if eval_file else None
    
    def load_triplets(self, filepath: Path) -> List[InputExample]:
        """Load triplets from JSONL file."""
        examples = []
        
        for line in filepath.read_text().strip().split('\n'):
            if not line:
                continue
            
            data = json.loads(line)
            
            # Create InputExample with anchor, positive, negative
            example = InputExample(
                texts=[
                    data["query"],      # anchor
                    data["positive"],   # positive
                    data["negative"],   # negative
                ]
            )
            examples.append(example)
        
        return examples
    
    def get_train_examples(self) -> List[InputExample]:
        """Get training examples."""
        return self.load_triplets(self.train_file)
    
    def get_eval_examples(self) -> List[InputExample]:
        """Get evaluation examples."""
        if self.eval_file and self.eval_file.exists():
            return self.load_triplets(self.eval_file)
        return []


class RetrievalModelTrainer:
    """Train a retrieval model using sentence-transformers."""
    
    def __init__(
        self,
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dir: str = "models/retrieval-model",
        device: Optional[str] = None,
    ):
        self.base_model_name = base_model
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
        
        # Load model
        print(f"Loading base model: {base_model}")
        self.model = SentenceTransformer(base_model, device=self.device)
    
    def train(
        self,
        train_examples: List[InputExample],
        eval_examples: List[InputExample],
        epochs: int = 3,
        batch_size: int = 16,
        warmup_ratio: float = 0.1,
        learning_rate: float = 2e-5,
        margin: float = 0.5,
    ):
        """Train the model.
        
        Args:
            train_examples: Training triplets
            eval_examples: Evaluation triplets
            epochs: Number of training epochs
            batch_size: Batch size
            warmup_ratio: Warmup steps as ratio of total
            learning_rate: Learning rate
            margin: Triplet loss margin
        """
        print(f"\nTraining on {len(train_examples)} examples for {epochs} epochs")
        print(f"Batch size: {batch_size}, LR: {learning_rate}, Margin: {margin}")
        
        # Create data loader
        train_dataloader = DataLoader(
            train_examples, 
            shuffle=True, 
            batch_size=batch_size,
        )
        
        # Calculate training steps
        total_steps = len(train_dataloader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
        
        # Define loss function
        # TripletLoss with hard margin
        train_loss = losses.TripletLoss(
            model=self.model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=margin,
        )
        
        # Create evaluator if we have eval examples
        evaluator = None
        if eval_examples:
            # Split eval examples into components
            anchors = [ex.texts[0] for ex in eval_examples]
            positives = [ex.texts[1] for ex in eval_examples]
            negatives = [ex.texts[2] for ex in eval_examples]
            
            evaluator = TripletEvaluator(
                anchors=anchors,
                positives=positives,
                negatives=negatives,
                name="retrieval-eval",
            )
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={'lr': learning_rate},
            evaluator=evaluator,
            evaluation_steps=max(100, len(train_dataloader) // 2),
            output_path=str(self.output_dir),
            show_progress_bar=True,
            checkpoint_path=str(self.output_dir / "checkpoints"),
            checkpoint_save_steps=max(100, len(train_dataloader)),
            checkpoint_save_total_limit=2,
        )
        
        print(f"\nModel saved to {self.output_dir}")
        
        # Save training info
        info = {
            "base_model": self.base_model_name,
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "margin": margin,
        }
        (self.output_dir / "training_info.json").write_text(json.dumps(info, indent=2))
    
    def evaluate(self, eval_examples: List[InputExample]) -> Dict[str, float]:
        """Evaluate the model on triplets."""
        if not eval_examples:
            return {}
        
        print(f"\nEvaluating on {len(eval_examples)} examples...")
        
        # Calculate accuracy: positive should be closer than negative
        correct = 0
        total = 0
        
        for example in eval_examples:
            query, positive, negative = example.texts
            
            # Encode all three
            embeddings = self.model.encode([query, positive, negative])
            
            # Calculate cosine similarities
            query_emb = embeddings[0]
            pos_emb = embeddings[1]
            neg_emb = embeddings[2]
            
            pos_sim = self._cosine_similarity(query_emb, pos_emb)
            neg_sim = self._cosine_similarity(query_emb, neg_emb)
            
            if pos_sim > neg_sim:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"Triplet accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return {"accuracy": accuracy, "correct": correct, "total": total}
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        return float(torch.nn.functional.cosine_similarity(
            torch.tensor(a).unsqueeze(0),
            torch.tensor(b).unsqueeze(0),
        ))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune retrieval model")
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
        "--base-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--output-dir",
        default="models/rego-retrieval",
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Triplet loss margin",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "mps", "cpu", "auto"],
        default="auto",
        help="Device to use",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on existing model",
    )
    
    args = parser.parse_args()
    
    # Handle device
    device = None if args.device == "auto" else args.device
    
    # Load data
    print("Loading training data...")
    data_loader = RetrievalDataLoader(
        train_file=Path(args.train_file),
        eval_file=Path(args.eval_file),
    )
    
    train_examples = data_loader.get_train_examples()
    eval_examples = data_loader.get_eval_examples()
    
    print(f"Loaded {len(train_examples)} train, {len(eval_examples)} eval examples")
    
    # Create trainer
    if args.eval_only:
        # Load existing model
        trainer = RetrievalModelTrainer(
            base_model=args.output_dir,
            output_dir=args.output_dir,
            device=device,
        )
    else:
        trainer = RetrievalModelTrainer(
            base_model=args.base_model,
            output_dir=args.output_dir,
            device=device,
        )
    
    if args.eval_only:
        # Just evaluate
        results = trainer.evaluate(eval_examples)
        print(f"\nResults: {results}")
    else:
        # Train
        trainer.train(
            train_examples=train_examples,
            eval_examples=eval_examples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            margin=args.margin,
        )
        
        # Final evaluation
        print("\n=== Final Evaluation ===")
        results = trainer.evaluate(eval_examples)
        print(f"Final results: {results}")


if __name__ == "__main__":
    main()

