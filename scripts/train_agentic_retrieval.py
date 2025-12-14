#!/usr/bin/env python3
"""Agentic Retrieval Training with Reinforcement Learning.

Phase 3 implementation based on SID-1 Technical Report.

Key Innovation: Multi-turn retrieval where the model can:
1. SEARCH: Execute a query against the index
2. READ: Examine retrieved documents
3. REFINE: Generate a new query based on what it learned
4. REPORT: Submit final document set

This replaces single-step embedding retrieval with an iterative agent
that adapts its search strategy based on results.

Reference: SID-1 Technical Report (December 2025)
https://www.sid.ai/research/SID-1_Preview/technical-report/SID_1_Technical_Report__Test_Time_Compute_for_Retrieval.pdf
"""

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
import argparse
import numpy as np

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("Warning: trl not installed. Install with: pip install trl")

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss/sentence-transformers not installed")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Document:
    """A document in the corpus."""
    id: str
    text: str
    doc_type: str  # "helper" or "schema"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalTask:
    """A retrieval task for training."""
    query: str
    target_doc_ids: Set[str]  # Documents that should be found
    difficulty: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """An action taken by the retrieval agent."""
    action_type: str  # "search", "read", "report"
    query: Optional[str] = None  # For search actions
    doc_ids: Optional[List[str]] = None  # For report actions
    

@dataclass
class AgentObservation:
    """What the agent observes after an action."""
    original_query: str
    turn: int
    max_turns: int
    search_history: List[Dict]  # Previous searches and results
    current_results: List[Document]  # Most recent search results
    retrieved_so_far: Set[str]  # All docs retrieved across turns


# ============================================================================
# Retrieval Environment
# ============================================================================

class RetrievalEnvironment:
    """
    Multi-turn retrieval environment for RL training.
    
    The agent interacts with this environment by:
    1. Receiving an observation (query + previous results)
    2. Taking an action (search with new query, or report final docs)
    3. Receiving a reward (based on recall of target documents)
    """
    
    def __init__(
        self,
        corpus: Dict[str, Document],
        embedding_model: Optional[SentenceTransformer] = None,
        max_turns: int = 5,
        results_per_search: int = 10,
    ):
        self.corpus = corpus
        self.max_turns = max_turns
        self.results_per_search = results_per_search
        
        # Build search index
        if embedding_model and HAS_FAISS:
            self._build_index(embedding_model)
        else:
            self.index = None
            self.doc_ids = list(corpus.keys())
    
    def _build_index(self, model: SentenceTransformer):
        """Build FAISS index for semantic search."""
        self.doc_ids = list(self.corpus.keys())
        texts = [self.corpus[did].text for did in self.doc_ids]
        
        print(f"Building FAISS index for {len(texts)} documents...")
        embeddings = model.encode(texts, show_progress_bar=True)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        self.embedding_model = model
        print(f"Index built with dimension {dim}")
    
    def reset(self, task: RetrievalTask) -> AgentObservation:
        """Start a new episode."""
        self.task = task
        self.turn = 0
        self.search_history = []
        self.retrieved_so_far = set()
        
        return AgentObservation(
            original_query=task.query,
            turn=0,
            max_turns=self.max_turns,
            search_history=[],
            current_results=[],
            retrieved_so_far=set(),
        )
    
    def step(self, action: AgentAction) -> Tuple[AgentObservation, float, bool, Dict]:
        """
        Execute an action and return (observation, reward, done, info).
        """
        self.turn += 1
        info = {"action": action.action_type}
        
        if action.action_type == "search":
            return self._handle_search(action.query, info)
        
        elif action.action_type == "report":
            return self._handle_report(action.doc_ids, info)
        
        else:
            # Invalid action - small penalty
            obs = self._get_observation([])
            return obs, -0.1, True, {"error": "invalid_action"}
    
    def _handle_search(self, query: str, info: Dict) -> Tuple[AgentObservation, float, bool, Dict]:
        """Handle a search action."""
        # Execute search
        results = self._search(query)
        
        # Track what we've found
        result_ids = set(doc.id for doc in results)
        
        # BUG FIX: Compute new_relevant BEFORE updating retrieved_so_far
        new_ids = result_ids - self.retrieved_so_far
        new_relevant = new_ids & self.task.target_doc_ids
        self.retrieved_so_far.update(result_ids)
        
        # Record in history
        self.search_history.append({
            "turn": self.turn,
            "query": query,
            "result_ids": list(result_ids),
            "new_relevant_count": len(new_relevant),
        })
        
        # Small intermediate reward for finding new relevant docs
        # Cap at 0.3 to prevent reward farming via spamming searches
        intermediate_reward = min(len(new_relevant) * 0.1, 0.3)
        
        # Check if done (max turns reached)
        done = self.turn >= self.max_turns
        
        # If max turns and not reported, auto-report
        if done:
            final_reward = self._compute_final_reward(self.retrieved_so_far)
            info["auto_report"] = True
            info["final_recall"] = final_reward
            return self._get_observation(results), intermediate_reward + final_reward, True, info
        
        return self._get_observation(results), intermediate_reward, False, info
    
    def _handle_report(self, doc_ids: List[str], info: Dict) -> Tuple[AgentObservation, float, bool, Dict]:
        """Handle a report action (submit final answer)."""
        reported = set(doc_ids) if doc_ids else self.retrieved_so_far
        
        reward = self._compute_final_reward(reported)
        info["reported_docs"] = list(reported)
        info["target_docs"] = list(self.task.target_doc_ids)
        info["final_recall"] = reward
        
        return self._get_observation([]), reward, True, info
    
    def _search(self, query: str) -> List[Document]:
        """Execute a search query."""
        if self.index is not None:
            # Semantic search with FAISS
            query_emb = self.embedding_model.encode([query])
            faiss.normalize_L2(query_emb)
            
            scores, indices = self.index.search(query_emb, self.results_per_search)
            
            results = []
            for idx in indices[0]:
                if idx >= 0:  # FAISS returns -1 for missing results
                    doc_id = self.doc_ids[idx]
                    results.append(self.corpus[doc_id])
            return results
        else:
            # Fallback: keyword matching
            query_lower = query.lower()
            scored = []
            for doc_id, doc in self.corpus.items():
                # Simple keyword matching
                text_lower = doc.text.lower()
                score = sum(1 for word in query_lower.split() if word in text_lower)
                if score > 0:
                    scored.append((score, doc))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored[:self.results_per_search]]
    
    def _get_observation(self, current_results: List[Document]) -> AgentObservation:
        """Create observation from current state."""
        return AgentObservation(
            original_query=self.task.query,
            turn=self.turn,
            max_turns=self.max_turns,
            search_history=self.search_history.copy(),
            current_results=current_results,
            retrieved_so_far=self.retrieved_so_far.copy(),
        )
    
    def _compute_final_reward(self, retrieved: Set[str]) -> float:
        """
        Compute document-centric reward (SID-1 style).
        
        Key insight from SID-1:
        "Overreporting by a few documents is preferable to 
        underreporting by one crucial document."
        
        So we weight recall more heavily than precision.
        """
        targets = self.task.target_doc_ids
        
        if not targets:
            return 1.0  # No targets = success
        
        # Recall: did we find all target documents?
        true_positives = len(retrieved & targets)
        recall = true_positives / len(targets)
        
        # Precision: what fraction of retrieved are relevant?
        precision = true_positives / max(len(retrieved), 1)
        
        # SID-1 uses recall-heavy weighting (α = 0.8)
        alpha = 0.8
        reward = alpha * recall + (1 - alpha) * precision
        
        return reward


# ============================================================================
# Agentic Retrieval Model
# ============================================================================

class AgenticRetrievalModel:
    """
    LLM-based agent for multi-turn retrieval.
    
    The model receives observations and generates actions:
    - Observe: query + previous search results
    - Think: what information is still missing?
    - Act: search with refined query OR report final docs
    """
    
    SYSTEM_PROMPT = """You are a retrieval agent. Your task is to find all relevant documents for a query by searching iteratively.

Available actions:
1. SEARCH: <query> - Search with a new or refined query
2. REPORT: <doc_ids> - Submit your final list of relevant document IDs

Guidelines:
- Start with the original query, then refine based on results
- Look for helpers AND schemas that are relevant
- If you find partial matches, search for related concepts
- Report when you've found all relevant documents or can't find more"""

    ACTION_PROMPT = """Turn {turn}/{max_turns}

Original Query: {original_query}

Search History:
{search_history}

Current Results:
{current_results}

Documents Found So Far: {retrieved_count}

What action should you take? Respond with either:
- SEARCH: <your refined query>
- REPORT: <comma-separated doc IDs to report>"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = True,
    ):
        self.device = device
        self.model_name = model_name
        
        print(f"Loading model: {model_name}")
        
        # Quantization config for efficiency
        if load_in_4bit and torch.cuda.is_available():
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
        
        if HAS_TRL:
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device if device != "auto" else "auto",
                torch_dtype=torch.bfloat16,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device if device != "auto" else "auto",
                torch_dtype=torch.bfloat16,
            )
    
    def get_action(self, observation: AgentObservation) -> AgentAction:
        """Generate an action given an observation."""
        prompt = self._format_prompt(observation)
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return self._parse_action(response)
    
    def _format_prompt(self, obs: AgentObservation) -> str:
        """Format observation into prompt."""
        # Format search history
        if obs.search_history:
            history_lines = []
            for h in obs.search_history[-3:]:  # Last 3 searches
                history_lines.append(
                    f"  Turn {h['turn']}: Searched '{h['query']}' → "
                    f"{len(h['result_ids'])} results, {h['new_relevant_count']} new relevant"
                )
            history_str = "\n".join(history_lines)
        else:
            history_str = "  (No searches yet)"
        
        # Format current results
        if obs.current_results:
            results_lines = []
            for doc in obs.current_results[:5]:  # Top 5 results
                # Truncate text
                text_preview = doc.text[:150].replace('\n', ' ')
                results_lines.append(f"  [{doc.id}] {text_preview}...")
            results_str = "\n".join(results_lines)
        else:
            results_str = "  (No results yet - make your first search)"
        
        action_prompt = self.ACTION_PROMPT.format(
            turn=obs.turn,
            max_turns=obs.max_turns,
            original_query=obs.original_query,
            search_history=history_str,
            current_results=results_str,
            retrieved_count=len(obs.retrieved_so_far),
        )
        
        # Format as chat
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": action_prompt},
        ]
        
        return self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    
    def _parse_action(self, response: str) -> AgentAction:
        """Parse model response into an action."""
        response = response.strip().upper()
        
        # Try to parse SEARCH action
        search_match = re.search(r'SEARCH:\s*(.+)', response, re.IGNORECASE)
        if search_match:
            query = search_match.group(1).strip()
            return AgentAction(action_type="search", query=query)
        
        # Try to parse REPORT action
        report_match = re.search(r'REPORT:\s*(.+)', response, re.IGNORECASE)
        if report_match:
            doc_ids_str = report_match.group(1).strip()
            doc_ids = [d.strip() for d in doc_ids_str.split(',')]
            return AgentAction(action_type="report", doc_ids=doc_ids)
        
        # Default: search with original query (first turn) or report (later turns)
        return AgentAction(action_type="search", query="")


# ============================================================================
# RL Trainer
# ============================================================================

class AgenticRetrievalTrainer:
    """
    Train the agentic retrieval model with RL (PPO/GRPO).
    
    Training loop:
    1. Sample a retrieval task
    2. Agent interacts with environment over multiple turns
    3. Compute reward based on final recall
    4. Update model with PPO
    """
    
    def __init__(
        self,
        model: AgenticRetrievalModel,
        environment: RetrievalEnvironment,
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        mini_batch_size: int = 1,
    ):
        self.model = model
        self.env = environment
        
        if HAS_TRL:
            self.ppo_config = PPOConfig(
                learning_rate=learning_rate,
                batch_size=batch_size,
                mini_batch_size=mini_batch_size,
                gradient_accumulation_steps=batch_size // mini_batch_size,
                ppo_epochs=4,
                max_grad_norm=0.5,
                log_with=None,  # Disable wandb
            )
            
            self.trainer = PPOTrainer(
                config=self.ppo_config,
                model=self.model.model,
                tokenizer=self.model.tokenizer,
            )
        else:
            print("Warning: TRL not available, training will be simulated")
            self.trainer = None
    
    def train(
        self,
        tasks: List[RetrievalTask],
        num_episodes: int = 1000,
        eval_every: int = 100,
        eval_tasks: Optional[List[RetrievalTask]] = None,
    ):
        """Train the model on retrieval tasks."""
        print(f"\n=== Starting Agentic Retrieval Training ===")
        print(f"Tasks: {len(tasks)}, Episodes: {num_episodes}")
        
        episode_rewards = []
        episode_recalls = []
        
        for episode in range(num_episodes):
            # Sample a task
            task = random.choice(tasks)
            
            # Run episode
            reward, info = self._run_episode(task)
            episode_rewards.append(reward)
            episode_recalls.append(info.get("final_recall", 0))
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_recall = np.mean(episode_recalls[-10:])
                print(f"Episode {episode + 1}/{num_episodes}: "
                      f"Avg Reward = {avg_reward:.3f}, Avg Recall = {avg_recall:.3f}")
            
            # Evaluate
            if eval_tasks and (episode + 1) % eval_every == 0:
                eval_recall = self._evaluate(eval_tasks)
                print(f"  Eval Recall @ {episode + 1}: {eval_recall:.3f}")
        
        print("\n=== Training Complete ===")
        print(f"Final Avg Recall: {np.mean(episode_recalls[-100:]):.3f}")
    
    def _run_episode(self, task: RetrievalTask) -> Tuple[float, Dict]:
        """Run one episode and update model."""
        obs = self.env.reset(task)
        
        total_reward = 0
        queries = []
        responses = []
        rewards = []
        
        while True:
            # Get action from model
            prompt = self.model._format_prompt(obs)
            action = self.model.get_action(obs)
            
            # Format action as response
            if action.action_type == "search":
                response = f"SEARCH: {action.query}"
            else:
                response = f"REPORT: {','.join(action.doc_ids or [])}"
            
            queries.append(prompt)
            responses.append(response)
            
            # Execute action
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
            total_reward += reward
            
            if done:
                break
        
        # Update model with PPO (if available)
        if self.trainer and len(queries) > 0:
            # Tokenize
            query_tensors = [
                self.model.tokenizer(q, return_tensors="pt").input_ids[0]
                for q in queries
            ]
            response_tensors = [
                self.model.tokenizer(r, return_tensors="pt").input_ids[0]
                for r in responses
            ]
            reward_tensors = [torch.tensor([r]) for r in rewards]
            
            # PPO update
            try:
                self.trainer.step(query_tensors, response_tensors, reward_tensors)
            except Exception as e:
                print(f"PPO update failed: {e}")
        
        return total_reward, info
    
    def _evaluate(self, tasks: List[RetrievalTask]) -> float:
        """Evaluate on a set of tasks."""
        recalls = []
        
        for task in tasks[:50]:  # Limit evaluation size
            obs = self.env.reset(task)
            
            while True:
                action = self.model.get_action(obs)
                obs, reward, done, info = self.env.step(action)
                
                if done:
                    recalls.append(info.get("final_recall", 0))
                    break
        
        return np.mean(recalls)


# ============================================================================
# Data Loading
# ============================================================================

def load_corpus(kb_dir: Path) -> Dict[str, Document]:
    """Load document corpus from knowledge base."""
    corpus = {}
    
    # Load helpers
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
    
    # Load schemas
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


def load_tasks(train_file: Path) -> List[RetrievalTask]:
    """Load retrieval tasks from training data."""
    tasks = []
    
    # Group by query
    query_targets: Dict[str, Set[str]] = {}
    
    for line in train_file.read_text().strip().split('\n'):
        if not line:
            continue
        data = json.loads(line)
        query = data.get('query', '')
        positive_id = data.get('_positive_id', '')
        
        if query and positive_id:
            if query not in query_targets:
                query_targets[query] = set()
            query_targets[query].add(positive_id)
    
    # Convert to tasks
    for query, targets in query_targets.items():
        tasks.append(RetrievalTask(
            query=query,
            target_doc_ids=targets,
            difficulty=0.5,
        ))
    
    print(f"Loaded {len(tasks)} retrieval tasks")
    return tasks


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train agentic retrieval model with RL (SID-1 style)"
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
        "--eval-file",
        default="data/training/retrieval/retrieval_eval.jsonl",
        help="Evaluation data file",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base LLM for the agent",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-base-en-v1.5",
        help="Embedding model for search index",
    )
    parser.add_argument(
        "--output-dir",
        default="models/agentic-retrieval",
        help="Output directory",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum turns per episode",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for PPO",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device (auto, cuda, mps, cpu)",
    )
    
    args = parser.parse_args()
    
    # Load corpus
    corpus = load_corpus(Path(args.kb_dir))
    
    # Load embedding model for search
    if HAS_FAISS:
        print(f"\nLoading embedding model: {args.embedding_model}")
        embedding_model = SentenceTransformer(args.embedding_model)
    else:
        embedding_model = None
    
    # Create environment
    env = RetrievalEnvironment(
        corpus=corpus,
        embedding_model=embedding_model,
        max_turns=args.max_turns,
    )
    
    # Load agentic model
    agent = AgenticRetrievalModel(
        model_name=args.model_name,
        device=args.device,
        load_in_4bit=not args.no_4bit,
    )
    
    # Load tasks
    train_file = Path(args.train_file)
    eval_file = Path(args.eval_file)
    
    if not train_file.exists():
        print(f"Error: Training file not found: {train_file}")
        print("Run generate_retrieval_training.py first to create training data")
        return
    
    train_tasks = load_tasks(train_file)
    eval_tasks = load_tasks(eval_file) if eval_file.exists() else None
    
    # Create trainer
    trainer = AgenticRetrievalTrainer(
        model=agent,
        environment=env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )
    
    # Train!
    trainer.train(
        tasks=train_tasks,
        num_episodes=args.episodes,
        eval_every=100,
        eval_tasks=eval_tasks,
    )
    
    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if hasattr(agent.model, 'save_pretrained'):
        agent.model.save_pretrained(output_dir)
        agent.tokenizer.save_pretrained(output_dir)
        print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()

