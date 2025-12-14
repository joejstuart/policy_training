#!/usr/bin/env python3
"""ID Selector Training - The Practical Approach.

Based on feedback: The task is fundamentally "pick IDs from a catalog",
not "train better embeddings". This is a multi-label selection problem.

Architecture:
1. Candidate Generation: BM25 + off-the-shelf embeddings → top 50 candidates
2. Selection Model: Small LM that outputs IDs from the candidate set
3. Training: DPO (Direct Preference Optimization) - much more stable than PPO

Key insight: Constrain the model to select from candidates, not generate IDs
from scratch. This eliminates hallucination.

References:
- SID-1 Technical Report (document-centric optimization)
- DPO Paper: https://arxiv.org/abs/2305.18290
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import argparse
import numpy as np

import torch
from torch import nn

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers/peft not installed")

try:
    from trl import DPOTrainer, DPOConfig
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("Warning: trl not installed. Install with: pip install trl>=0.7.0")

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    from rank_bm25 import BM25Okapi
    import faiss
    HAS_RETRIEVAL = True
except ImportError:
    HAS_RETRIEVAL = False
    print("Warning: retrieval deps not installed")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Document:
    """A document in the catalog."""
    id: str
    text: str
    doc_type: str  # "helper" or "schema"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelectionTask:
    """A selection task with ground truth labels."""
    instruction: str
    
    # Multi-level ground truth (as recommended)
    required_ids: Set[str]     # Must be selected
    helpful_ids: Set[str]      # Good to select
    distractor_ids: Set[str]   # Should NOT be selected (penalties)
    
    # Metadata
    task_id: str = ""
    difficulty: float = 0.5


@dataclass
class Candidate:
    """A candidate document for selection."""
    id: str
    title: str      # Short identifier
    snippet: str    # Preview text
    score: float    # Retrieval score


# ============================================================================
# Candidate Generation (BM25 + Embeddings)
# ============================================================================

class HybridCandidateGenerator:
    """
    Generate candidates using BM25 + embeddings.
    
    This is the "pragmatic" approach - use off-the-shelf models for
    candidate generation, then train selection.
    """
    
    def __init__(
        self,
        corpus: Dict[str, Document],
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        top_k_bm25: int = 30,
        top_k_embedding: int = 30,
    ):
        self.corpus = corpus
        self.top_k_bm25 = top_k_bm25
        self.top_k_embedding = top_k_embedding
        
        self._build_indexes(embedding_model)
    
    def _build_indexes(self, embedding_model: str):
        """Build BM25 and embedding indexes."""
        self.doc_ids = list(self.corpus.keys())
        texts = [self.corpus[did].text for did in self.doc_ids]
        
        # BM25 index
        print("Building BM25 index...")
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)
        
        # Embedding index
        if HAS_RETRIEVAL:
            print(f"Building embedding index with {embedding_model}...")
            self.embedding_model = SentenceTransformer(embedding_model)
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            dim = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings)
        else:
            self.embedding_model = None
            self.faiss_index = None
        
        print(f"Indexed {len(texts)} documents")
    
    def generate_candidates(self, query: str) -> List[Candidate]:
        """Generate candidates using hybrid retrieval."""
        candidates = {}
        
        # BM25 candidates
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        top_bm25 = np.argsort(bm25_scores)[-self.top_k_bm25:][::-1]
        
        for idx in top_bm25:
            doc_id = self.doc_ids[idx]
            doc = self.corpus[doc_id]
            candidates[doc_id] = Candidate(
                id=doc_id,
                title=doc_id,
                snippet=doc.text[:200],
                score=float(bm25_scores[idx]),
            )
        
        # Embedding candidates
        if self.embedding_model and self.faiss_index:
            query_emb = self.embedding_model.encode([query])
            faiss.normalize_L2(query_emb)
            
            scores, indices = self.faiss_index.search(query_emb, self.top_k_embedding)
            
            for i, idx in enumerate(indices[0]):
                if idx >= 0:
                    doc_id = self.doc_ids[idx]
                    doc = self.corpus[doc_id]
                    
                    if doc_id in candidates:
                        # Boost score if found by both
                        candidates[doc_id].score += float(scores[0][i])
                    else:
                        candidates[doc_id] = Candidate(
                            id=doc_id,
                            title=doc_id,
                            snippet=doc.text[:200],
                            score=float(scores[0][i]),
                        )
        
        # Sort by combined score
        result = sorted(candidates.values(), key=lambda x: x.score, reverse=True)
        return result


# ============================================================================
# ID Selector Model
# ============================================================================

class IDSelectorModel:
    """
    Model that selects IDs from a candidate set.
    
    Key design choices:
    1. Candidates are provided in the prompt - no hallucination possible
    2. Output is constrained to valid IDs
    3. Separate treatment for helpers vs schemas
    """
    
    SYSTEM_PROMPT = """You are an ID selector for a Rego policy system.
Given an instruction and a list of candidate documents, select the IDs that are relevant.

Rules:
- Select ALL required helpers/schemas for the task
- Prefer specific helpers over general ones
- Include both helpers AND schemas when needed
- Output ONLY valid IDs from the candidate list"""

    SELECTION_PROMPT = """Instruction: {instruction}

Candidates:
{candidates}

Select the relevant IDs. Output format:
HELPERS: id1, id2, ...
SCHEMAS: id1, id2, ..."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = True,
    ):
        self.model_name = model_name
        
        print(f"Loading model: {model_name}")
        
        # Quantization for efficiency
        if load_in_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            bnb_config = None
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device if device != "auto" else "auto",
            torch_dtype=torch.bfloat16,
        )
    
    def select(
        self,
        instruction: str,
        candidates: List[Candidate],
        valid_ids: Set[str] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Select IDs from candidates.
        
        Returns: (helper_ids, schema_ids)
        """
        prompt = self._format_prompt(instruction, candidates)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        # Parse and validate
        helper_ids, schema_ids = self._parse_response(response)
        
        # Enforce ID whitelist
        if valid_ids:
            helper_ids = [h for h in helper_ids if h in valid_ids]
            schema_ids = [s for s in schema_ids if s in valid_ids]
        
        return helper_ids, schema_ids
    
    def _format_prompt(self, instruction: str, candidates: List[Candidate]) -> str:
        """Format candidates into prompt."""
        # Group by type
        helpers = []
        schemas = []
        
        for c in candidates:
            if "helper" in c.id.lower() or "." in c.id:
                helpers.append(f"  [{c.id}] {c.snippet[:100]}")
            else:
                schemas.append(f"  [{c.id}] {c.snippet[:100]}")
        
        candidates_str = "HELPERS:\n" + "\n".join(helpers[:20])
        if schemas:
            candidates_str += "\n\nSCHEMAS:\n" + "\n".join(schemas[:15])
        
        user_content = self.SELECTION_PROMPT.format(
            instruction=instruction,
            candidates=candidates_str,
        )
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def _parse_response(self, response: str) -> Tuple[List[str], List[str]]:
        """Parse model response into ID lists."""
        import re
        
        helper_ids = []
        schema_ids = []
        
        # Extract HELPERS line
        helpers_match = re.search(r'HELPERS:\s*(.+?)(?:\n|SCHEMAS:|$)', response, re.IGNORECASE | re.DOTALL)
        if helpers_match:
            ids_str = helpers_match.group(1).strip()
            helper_ids = [id.strip() for id in ids_str.split(',') if id.strip()]
        
        # Extract SCHEMAS line
        schemas_match = re.search(r'SCHEMAS:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.DOTALL)
        if schemas_match:
            ids_str = schemas_match.group(1).strip()
            schema_ids = [id.strip() for id in ids_str.split(',') if id.strip()]
        
        return helper_ids, schema_ids


# ============================================================================
# Reward Function (Document-Centric, as recommended)
# ============================================================================

def compute_ndcg(selected: List[str], relevant: Set[str], k: int = 10) -> float:
    """
    Compute NDCG@K - rewards correct items being higher in the list.
    
    Better than mixing recall+precision because it directly rewards ranking.
    """
    dcg = 0.0
    for i, item in enumerate(selected[:k]):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # 1-indexed position
    
    # Ideal DCG
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_selection_reward(
    selected_helpers: List[str],
    selected_schemas: List[str],
    task: SelectionTask,
    k: int = 10,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute reward for a selection.
    
    Based on feedback:
    - Use NDCG for ranking quality
    - Penalize invalid IDs heavily
    - Separate rewards for helpers vs schemas
    - Penalize distractors
    
    Formula:
    R = 0.5*NDCG_helpers + 0.5*NDCG_schemas - 2*invalid - 0.5*distractors
    """
    all_relevant = task.required_ids | task.helpful_ids
    
    # Split by type
    relevant_helpers = {id for id in all_relevant if "." in id or "lib" in id.lower()}
    relevant_schemas = all_relevant - relevant_helpers
    
    # NDCG for helpers
    ndcg_helpers = compute_ndcg(selected_helpers, relevant_helpers, k)
    
    # NDCG for schemas
    ndcg_schemas = compute_ndcg(selected_schemas, relevant_schemas, k)
    
    # Count invalid IDs
    all_selected = set(selected_helpers) | set(selected_schemas)
    all_valid = task.required_ids | task.helpful_ids | task.distractor_ids
    invalid_count = len(all_selected - all_valid)
    
    # Count distractor selections
    distractor_count = len(all_selected & task.distractor_ids)
    
    # Count duplicates
    all_selected_list = selected_helpers + selected_schemas
    dupe_count = len(all_selected_list) - len(set(all_selected_list))
    
    # Required coverage bonus
    required_found = len(all_selected & task.required_ids)
    required_bonus = required_found / max(len(task.required_ids), 1)
    
    # Compute final reward
    reward = (
        0.4 * ndcg_helpers +
        0.4 * ndcg_schemas +
        0.3 * required_bonus -
        2.0 * invalid_count -
        0.5 * distractor_count -
        0.1 * dupe_count
    )
    
    breakdown = {
        "ndcg_helpers": ndcg_helpers,
        "ndcg_schemas": ndcg_schemas,
        "required_bonus": required_bonus,
        "invalid_penalty": -2.0 * invalid_count,
        "distractor_penalty": -0.5 * distractor_count,
        "dupe_penalty": -0.1 * dupe_count,
        "total": reward,
    }
    
    return reward, breakdown


# ============================================================================
# DPO Training (More stable than PPO for selection tasks)
# ============================================================================

class DPOSelectionTrainer:
    """
    Train selection model using DPO (Direct Preference Optimization).
    
    DPO is much more stable than PPO for "text → structured selection":
    1. Sample N candidate outputs for each instruction
    2. Score them with reward function
    3. Train DPO on (best, worst) pairs
    """
    
    def __init__(
        self,
        model: IDSelectorModel,
        candidate_generator: HybridCandidateGenerator,
        learning_rate: float = 5e-6,
        beta: float = 0.1,  # DPO temperature
    ):
        self.model = model
        self.candidate_gen = candidate_generator
        self.learning_rate = learning_rate
        self.beta = beta
    
    def create_preference_pairs(
        self,
        tasks: List[SelectionTask],
        samples_per_task: int = 4,
    ) -> List[Dict]:
        """
        Create preference pairs for DPO training.
        
        For each task:
        1. Generate candidates
        2. Sample multiple outputs from model (with temperature)
        3. Score each output
        4. Create (best, worst) pairs
        """
        pairs = []
        
        for task in tasks:
            # Generate candidates
            candidates = self.candidate_gen.generate_candidates(task.instruction)
            valid_ids = {c.id for c in candidates}
            
            # Sample multiple outputs
            outputs_with_scores = []
            
            for _ in range(samples_per_task):
                helpers, schemas = self.model.select(
                    task.instruction,
                    candidates,
                    valid_ids=valid_ids,
                )
                
                reward, breakdown = compute_selection_reward(
                    helpers, schemas, task
                )
                
                # Format as text output
                output_text = f"HELPERS: {', '.join(helpers)}\nSCHEMAS: {', '.join(schemas)}"
                
                outputs_with_scores.append({
                    "output": output_text,
                    "reward": reward,
                    "breakdown": breakdown,
                })
            
            # Sort by reward
            outputs_with_scores.sort(key=lambda x: x["reward"], reverse=True)
            
            # Create pairs: best vs each worse output
            best = outputs_with_scores[0]
            for worse in outputs_with_scores[1:]:
                if best["reward"] > worse["reward"]:  # Must be strictly better
                    prompt = self.model._format_prompt(task.instruction, candidates)
                    
                    pairs.append({
                        "prompt": prompt,
                        "chosen": best["output"],
                        "rejected": worse["output"],
                        "chosen_reward": best["reward"],
                        "rejected_reward": worse["reward"],
                    })
        
        print(f"Created {len(pairs)} preference pairs from {len(tasks)} tasks")
        return pairs
    
    def train(
        self,
        tasks: List[SelectionTask],
        epochs: int = 3,
        samples_per_task: int = 4,
        output_dir: str = "models/id-selector",
    ):
        """Train with DPO."""
        if not HAS_TRL:
            print("Error: trl not installed. Cannot train with DPO.")
            return
        
        # Create preference pairs
        print("Creating preference pairs...")
        pairs = self.create_preference_pairs(tasks, samples_per_task)
        
        if not pairs:
            print("Error: No valid preference pairs created")
            return
        
        # Setup LoRA for efficient training
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        
        model = prepare_model_for_kbit_training(self.model.model)
        model = get_peft_model(model, lora_config)
        
        # Create dataset
        from datasets import Dataset
        train_dataset = Dataset.from_list(pairs)
        
        # DPO config
        dpo_config = DPOConfig(
            output_dir=output_dir,
            learning_rate=self.learning_rate,
            beta=self.beta,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            logging_steps=10,
            save_steps=100,
            bf16=True,
        )
        
        # Create trainer
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=train_dataset,
            tokenizer=self.model.tokenizer,
        )
        
        # Train!
        print("Starting DPO training...")
        trainer.train()
        
        # Save
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")


# ============================================================================
# Constrained Decoding (Enforce valid outputs at inference)
# ============================================================================

def validate_and_repair_selection(
    selected_helpers: List[str],
    selected_schemas: List[str],
    valid_ids: Set[str],
    model: IDSelectorModel = None,
    instruction: str = None,
    candidates: List[Candidate] = None,
    max_repairs: int = 2,
) -> Tuple[List[str], List[str]]:
    """
    Validate and optionally repair selections.
    
    Even with training, enforce this at inference:
    1. Remove invalid IDs
    2. Optionally re-ask model with explicit candidate list
    """
    # Filter to valid IDs only
    valid_helpers = [h for h in selected_helpers if h in valid_ids]
    valid_schemas = [s for s in selected_schemas if s in valid_ids]
    
    # If too many were removed, try repair
    removed = (len(selected_helpers) - len(valid_helpers)) + (len(selected_schemas) - len(valid_schemas))
    
    if removed > 2 and model and instruction and candidates and max_repairs > 0:
        print(f"  Repair: {removed} invalid IDs removed, re-asking model...")
        
        # Add explicit instruction to select from list
        repair_instruction = f"{instruction}\n\n[IMPORTANT: Select ONLY from the provided candidate IDs]"
        
        repaired_helpers, repaired_schemas = model.select(
            repair_instruction,
            candidates,
            valid_ids=valid_ids,
        )
        
        # Validate again (no recursion this time)
        repaired_helpers = [h for h in repaired_helpers if h in valid_ids]
        repaired_schemas = [s for s in repaired_schemas if s in valid_ids]
        
        # Use repaired if better
        if len(repaired_helpers) + len(repaired_schemas) > len(valid_helpers) + len(valid_schemas):
            return repaired_helpers, repaired_schemas
    
    return valid_helpers, valid_schemas


# ============================================================================
# Data Loading
# ============================================================================

def load_corpus(kb_dir: Path) -> Dict[str, Document]:
    """Load document corpus."""
    corpus = {}
    
    helpers_file = kb_dir / "helpers.jsonl"
    if helpers_file.exists():
        for line in helpers_file.read_text().strip().split('\n'):
            if not line:
                continue
            data = json.loads(line)
            doc_id = data.get('id', '')
            text = f"Helper: {doc_id}\nSignature: {data.get('signature', '')}\nDescription: {data.get('description', '')}"
            corpus[doc_id] = Document(
                id=doc_id,
                text=text,
                doc_type="helper",
                metadata=data,
            )
    
    schemas_file = kb_dir / "schemas.jsonl"
    if schemas_file.exists():
        for line in schemas_file.read_text().strip().split('\n'):
            if not line:
                continue
            data = json.loads(line)
            doc_id = data.get('schema_id', '')
            text = f"Schema: {data.get('canonical_path', '')}\nType: {data.get('attestation_type', '')}\nDescription: {data.get('description', '')}"
            corpus[doc_id] = Document(
                id=doc_id,
                text=text,
                doc_type="schema",
                metadata=data,
            )
    
    print(f"Loaded {len(corpus)} documents")
    return corpus


def load_tasks_with_labels(train_file: Path) -> List[SelectionTask]:
    """
    Load tasks with multi-level ground truth.
    
    Expected format:
    {
        "instruction": "...",
        "required": ["id1", "id2"],
        "helpful": ["id3"],
        "distractors": ["id4"]
    }
    
    Falls back to simple format if multi-level not available.
    """
    tasks = []
    
    for line in train_file.read_text().strip().split('\n'):
        if not line:
            continue
        
        data = json.loads(line)
        
        # Try multi-level format
        if "required" in data:
            tasks.append(SelectionTask(
                instruction=data.get("instruction", data.get("query", "")),
                required_ids=set(data.get("required", [])),
                helpful_ids=set(data.get("helpful", [])),
                distractor_ids=set(data.get("distractors", [])),
                task_id=data.get("task_id", ""),
            ))
        else:
            # Fall back to simple format
            positive_id = data.get("_positive_id", "")
            negative_id = data.get("_negative_id", "")
            
            if positive_id:
                tasks.append(SelectionTask(
                    instruction=data.get("query", data.get("instruction", "")),
                    required_ids={positive_id},
                    helpful_ids=set(),
                    distractor_ids={negative_id} if negative_id else set(),
                ))
    
    print(f"Loaded {len(tasks)} tasks")
    return tasks


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train ID Selector with DPO (practical approach)"
    )
    parser.add_argument(
        "--kb-dir",
        default="data/knowledge_base",
        help="Knowledge base directory",
    )
    parser.add_argument(
        "--train-file",
        default="data/training/retrieval/retrieval_train.jsonl",
        help="Training data file",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base LM for selection",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model for candidate generation",
    )
    parser.add_argument(
        "--output-dir",
        default="models/id-selector",
        help="Output directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Training epochs",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=4,
        help="Number of samples per task for preference pairs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cuda, mps, cpu)",
    )
    
    args = parser.parse_args()
    
    # Load corpus
    corpus = load_corpus(Path(args.kb_dir))
    
    # Create candidate generator
    print("\nBuilding candidate generator...")
    candidate_gen = HybridCandidateGenerator(
        corpus=corpus,
        embedding_model=args.embedding_model,
    )
    
    # Load or create model
    print("\nLoading selection model...")
    model = IDSelectorModel(
        model_name=args.model_name,
        device=args.device,
        load_in_4bit=not args.no_4bit,
    )
    
    # Load tasks
    train_file = Path(args.train_file)
    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        return
    
    tasks = load_tasks_with_labels(train_file)
    
    if args.eval_only:
        # Evaluate
        print("\n=== Evaluation ===")
        
        total_reward = 0
        for task in tasks[:50]:  # Sample
            candidates = candidate_gen.generate_candidates(task.instruction)
            valid_ids = {c.id for c in candidates}
            
            helpers, schemas = model.select(task.instruction, candidates, valid_ids)
            reward, breakdown = compute_selection_reward(helpers, schemas, task)
            
            total_reward += reward
            print(f"  Task: {task.instruction[:50]}...")
            print(f"    Selected: {len(helpers)} helpers, {len(schemas)} schemas")
            print(f"    Reward: {reward:.3f}")
        
        print(f"\nAverage reward: {total_reward / 50:.3f}")
    else:
        # Train with DPO
        trainer = DPOSelectionTrainer(
            model=model,
            candidate_generator=candidate_gen,
            learning_rate=args.learning_rate,
        )
        
        trainer.train(
            tasks=tasks,
            epochs=args.epochs,
            samples_per_task=args.samples_per_task,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()

