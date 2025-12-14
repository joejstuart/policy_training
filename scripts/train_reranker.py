#!/usr/bin/env python3
"""Train a reranker that can actually be saved and reloaded.

Uses transformers directly instead of sentence-transformers CrossEncoder
to ensure the model can be saved and loaded properly.
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import numpy as np


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


class RerankerDataset(Dataset):
    """Dataset for training a reranker."""
    
    def __init__(
        self, 
        examples: List[TrainingExample], 
        corpus: Dict[str, Document],
        tokenizer,
        max_length: int = 256,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create pairs
        self.pairs = []
        for ex in examples:
            if ex.positive_id in corpus:
                # Positive pair (label=1)
                self.pairs.append((ex.query, corpus[ex.positive_id].text, 1))
                
                # Negative pair (label=0)
                if ex.negative_id and ex.negative_id in corpus:
                    self.pairs.append((ex.query, corpus[ex.negative_id].text, 0))
                else:
                    # Random negative
                    neg_id = random.choice(list(corpus.keys()))
                    if neg_id != ex.positive_id:
                        self.pairs.append((ex.query, corpus[neg_id].text, 0))
        
        print(f"Created {len(self.pairs)} training pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        query, doc, label = self.pairs[idx]
        
        # Tokenize as a pair
        encoding = self.tokenizer(
            query,
            doc,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_corpus(kb_dir: Path) -> Dict[str, Document]:
    """Load documents with enriched text."""
    corpus = {}
    
    helpers_file = kb_dir / "helpers.jsonl"
    if helpers_file.exists():
        for line in helpers_file.read_text().strip().split('\n'):
            if not line:
                continue
            data = json.loads(line)
            doc_id = data.get('id', '')
            
            parts = [
                doc_id,
                doc_id.replace('.', ' ').replace('_', ' '),
                data.get('signature', ''),
                data.get('description', ''),
            ]
            parts.extend(doc_id.split('.'))
            
            if 'task' in doc_id.lower():
                parts.extend(['tasks', 'pipeline', 'tekton', 'pipelinerun'])
            if 'attestation' in doc_id.lower():
                parts.extend(['attestation', 'provenance', 'slsa'])
            if 'bundle' in doc_id.lower():
                parts.extend(['bundle', 'pinned', 'digest', 'oci'])
            
            corpus[doc_id] = Document(id=doc_id, text=' '.join(filter(None, parts)), doc_type='helper')
    
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
            parts.extend(data.get('keywords', []))
            
            corpus[doc_id] = Document(id=doc_id, text=' '.join(filter(None, parts)), doc_type='schema')
    
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
            examples.append(TrainingExample(query=query, positive_id=pos_id, negative_id=neg_id))
    
    print(f"Loaded {len(examples)} training examples")
    return examples


def main():
    parser = argparse.ArgumentParser(description="Train a reranker model")
    parser.add_argument("--kb-dir", default="data/knowledge_base")
    parser.add_argument("--train-file", default="data/training/retrieval/curated_only.jsonl")
    parser.add_argument("--base-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--output-dir", default="models/reranker")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    
    args = parser.parse_args()
    
    # Load data
    corpus = load_corpus(Path(args.kb_dir))
    examples = load_training_data(Path(args.train_file))
    
    # Load tokenizer and model
    print(f"\nLoading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,  # Binary classification: relevant or not
    )
    
    # Create dataset
    dataset = RerankerDataset(examples, corpus, tokenizer, args.max_length)
    
    # Split into train/eval
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Training arguments
    output_dir = Path(args.output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=torch.cuda.is_available(),
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    print("\n=== Training ===")
    trainer.train()
    
    # Save the final model properly
    print(f"\n=== Saving to {output_dir} ===")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Done! Model saved and can be loaded with:")
    print(f"  model = AutoModelForSequenceClassification.from_pretrained('{output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")


if __name__ == "__main__":
    main()

