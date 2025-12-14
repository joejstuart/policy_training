# Improved Retrieval Training Plan

Based on insights from the [SID-1 Technical Report](https://www.sid.ai/research/SID-1_Preview/technical-report/SID_1_Technical_Report__Test_Time_Compute_for_Retrieval.pdf) (December 2025) and practical refinements.

## Executive Summary

**Key Insight**: This task is fundamentally **"pick IDs from a catalog"** - a multi-label selection problem, not an embedding problem.

The practical approach:
1. **Candidate Generation**: BM25 + off-the-shelf embeddings (don't over-invest here)
2. **Selection Model**: Small LM that outputs IDs from the candidate set
3. **Training**: DPO (much more stable than PPO for structured selection)
4. **Constrained Decoding**: Enforce valid IDs at inference time

SID-1 insights that apply:
- **Document-centric rewards**: Use NDCG, penalize invalid IDs
- **Synthetic questions**: Train without human cold-start data
- **Agentic retrieval**: Optional - can simulate with two-stage inference

## Current State Analysis

### What We Have

```
Current Pipeline:
Query → Embed → Vector Search → Rerank → Top-K → Model

Training: Triplet loss on (query, positive_doc, hard_negative_doc)
Model: sentence-transformers/all-MiniLM-L6-v2
```

### Core Problem

The end product is a **multi-label selector** over:
- Library chunks (helper symbols / modules)
- Schema paths

This is closer to a **reranker/classifier** (query + candidate → score) than "train a better embedding model".

### Why Embeddings Aren't the Right Focus

| Investment | Payoff | Better Alternative |
|------------|--------|-------------------|
| Weeks on contrastive fine-tuning | Maybe +10-20% recall | BM25 + off-the-shelf embeddings |
| Training custom embedder | Domain fit | Train selection/reranking instead |
| Multi-positive learning | Better representation | DPO on ID selection |

**Key insight**: Embeddings are useful for candidate generation only. Don't over-invest unless candidate gen is proven to be the bottleneck.

---

## Recommended Approach: Fast, Reliable Baseline

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ID Selection Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Instruction ──┬──► BM25 ──────────┬──► Top 50 Candidates   │
│                │                    │                        │
│                └──► Embeddings ────┘                         │
│                      (off-the-shelf)                         │
│                                                              │
│  Candidates ──► Selection LM ──► Selected IDs               │
│                (trained with DPO)                            │
│                                                              │
│  Selected IDs ──► Validate ──► Repair if needed ──► Output  │
│                   (whitelist)                                │
└─────────────────────────────────────────────────────────────┘
```

### Why This Works

| Component | Purpose | Why it's better |
|-----------|---------|-----------------|
| BM25 + Embeddings | Candidate generation | Off-the-shelf is fine, no training needed |
| Selection LM | Pick IDs from candidates | Constrained output = no hallucination |
| DPO training | Learn preferences | Much more stable than PPO for text→selection |
| ID validation | Enforce whitelist | Catches any remaining errors |

---

## Critical: Ground Truth Structure

The hard part is defining "correct library/schema" for each instruction. You need multi-level labels:

```json
{
  "instruction": "Check if all task bundles are pinned",
  "required": [
    "lib.pipelinerun_attestations",
    "tekton.task_ref"
  ],
  "helpful": [
    "tekton.tasks",
    "tekton.unpinned_task_bundle",
    "slsa_v1_task_bundle"
  ],
  "distractors": [
    "tekton.task_param",
    "lib.taskrun_attestations"
  ]
}
```

### Label Types

| Type | Description | Reward |
|------|-------------|--------|
| **required** | Must be selected for correct solution | Heavy bonus |
| **helpful** | Useful but not strictly necessary | Moderate bonus |
| **distractors** | Similar but wrong (confusing cases) | Penalty |

### Why Distractors Matter

Hard negatives that teach the model to distinguish:
- `lib.pipelinerun_attestations` vs `lib.taskrun_attestations`
- "digest" for bundles vs "digest" for images
- "pinning" for bundles vs "pinning" for git refs

---

## Reward Function Design

Based on feedback, use **NDCG** (not recall+precision mix) and **separate rewards**:

```python
def compute_reward(selected_helpers, selected_schemas, task):
    """
    R = 0.4*NDCG_helpers + 0.4*NDCG_schemas 
        + 0.3*required_coverage
        - 2.0*invalid_count
        - 0.5*distractor_count
        - 0.1*duplicate_count
    """
```

### Why This Design

| Component | Purpose |
|-----------|---------|
| **NDCG per type** | Rewards correct items ranked higher, prevents "only emit helpers" |
| **Required coverage** | Ensures critical items are found |
| **Invalid penalty (-2)** | Hard penalty for hallucinated IDs |
| **Distractor penalty** | Teaches model to distinguish similar items |

---

## Phase 1: Enhanced Contrastive Learning (Quick Win)

Before moving to RL, improve the current contrastive approach.

### 1.1 Upgrade Base Model

Current `all-MiniLM-L6-v2` is general-purpose. Options:

| Model | Dim | Speed | Domain Fit |
|-------|-----|-------|------------|
| `all-MiniLM-L6-v2` (current) | 384 | Fast | Generic |
| `bge-small-en-v1.5` | 384 | Fast | Better retrieval |
| `bge-base-en-v1.5` | 768 | Medium | Best quality |
| `Qwen3-Embedding-0.6B` | 1024 | Slow | SoTA, matches SID-1 base |

**Recommendation**: Start with `bge-base-en-v1.5`, consider `Qwen3-Embedding` for max quality.

### 1.2 Improved Hard Negative Mining

Current negatives are hand-selected. Implement **in-batch hard negatives**:

```python
class HardNegativeMiner:
    """Mine hard negatives using the current model."""
    
    def __init__(self, model, corpus_embeddings):
        self.model = model
        self.corpus = corpus_embeddings
    
    def mine_hard_negatives(self, query: str, positive_id: str, k: int = 10):
        """Find documents that are similar but wrong."""
        query_emb = self.model.encode(query)
        
        # Get top-k similar documents
        similarities = cosine_similarity(query_emb, self.corpus)
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Filter out the positive, keep hard negatives
        hard_negatives = [
            idx for idx in top_indices 
            if idx != positive_id and similarities[idx] > 0.5  # Similar but wrong
        ]
        return hard_negatives
```

### 1.3 Multi-Positive Training

Instead of single positive, use **all relevant documents** as positives:

```python
# Before: Single positive per query
{"query": "check bundle pinning", "positive": doc_A, "negative": doc_B}

# After: Multiple positives (InfoNCE loss)
{
    "query": "check bundle pinning",
    "positives": [doc_A, doc_C, doc_D],  # All relevant
    "negatives": [doc_B, doc_E, doc_F]   # Hard negatives
}
```

Use **Multiple Negatives Ranking Loss (MNRL)** instead of triplet loss:

```python
from sentence_transformers import losses

# Better than triplet loss for retrieval
train_loss = losses.MultipleNegativesRankingLoss(model)
```

### 1.4 Synthetic Query Generation

SID-1 emphasizes synthetic data. Generate more queries using LLM:

```python
QUERY_GEN_PROMPT = '''
Given this Rego helper function:

{helper_source}

Generate 10 diverse natural language queries that a user might ask if they needed this helper.
Include variations:
- Direct ("check if bundle is pinned")
- Imperative ("write a rule to verify...")
- Question form ("how do I check...")
- Domain jargon ("OCI bundle digest validation")
- Typos/casual ("task bunle pinned?")

Output as JSON list of strings.
'''

def generate_synthetic_queries(helper: HelperInfo, llm) -> List[str]:
    prompt = QUERY_GEN_PROMPT.format(helper_source=helper.source)
    response = llm.generate(prompt)
    return json.loads(response)
```

---

## Phase 2: Document-Centric Rewards (SID-1 Core Insight)

The key insight from SID-1: **reward finding documents, not answering questions**.

### 2.1 Reward Design

Instead of binary correct/incorrect, use document-level metrics:

```python
@dataclass
class RetrievalReward:
    """SID-1 style document-centric reward."""
    
    def compute(
        self,
        retrieved_docs: List[str],
        ground_truth_docs: Set[str],
        max_docs: int = 7
    ) -> float:
        """
        Recall-focused reward with precision penalty.
        
        SID-1 insight: "Overreporting by a few documents is preferable
        to underreporting by one crucial document."
        """
        retrieved_set = set(retrieved_docs[:max_docs])
        
        # Recall: Did we find all the documents?
        recall = len(retrieved_set & ground_truth_docs) / len(ground_truth_docs)
        
        # Precision: Penalize too much noise
        precision = len(retrieved_set & ground_truth_docs) / len(retrieved_set)
        
        # SID-1 uses recall-heavy weighting
        # α = 0.8 favors recall over precision
        alpha = 0.8
        reward = alpha * recall + (1 - alpha) * precision
        
        return reward
```

### 2.2 Partial Credit

Unlike binary triplet loss, give partial credit:

```python
def partial_credit_reward(retrieved: List[str], targets: Set[str]) -> float:
    """
    Reward each correct document incrementally.
    
    If targets = {A, B, C} and retrieved = [A, D, B]:
    - Finding A: +0.33
    - Finding D: +0.0 (wrong)
    - Finding B: +0.33
    Total: 0.67 (not binary 0 or 1)
    """
    score = 0.0
    for doc in retrieved:
        if doc in targets:
            score += 1.0 / len(targets)
    return score
```

---

## Phase 3: Multi-Turn Agentic Retrieval (SID-1 Main Innovation)

The biggest improvement in SID-1 comes from **iterative retrieval**.

### 3.1 Agentic Retrieval Architecture

```
Traditional (Current):
Query → Search → Top-K → Done

Agentic (SID-1):
Query → Search → Read Results → Refine Query → Search → ... → Final Results
```

### 3.2 Multi-Turn Environment

Create an RL environment where the model can:

1. **Search**: Execute a query against the index
2. **Read**: Examine retrieved documents
3. **Refine**: Generate a new query based on what it learned
4. **Report**: Submit final document set

```python
class RegoRetrievalEnvironment:
    """Multi-turn retrieval environment for RL training."""
    
    def __init__(self, index, max_turns: int = 5):
        self.index = index
        self.max_turns = max_turns
        self.current_turn = 0
        self.retrieved_so_far = set()
    
    def reset(self, query: str, target_docs: Set[str]):
        """Start new episode."""
        self.original_query = query
        self.target_docs = target_docs
        self.current_turn = 0
        self.retrieved_so_far = set()
        return self._get_observation()
    
    def step(self, action: dict) -> Tuple[dict, float, bool]:
        """
        Take an action in the environment.
        
        Actions:
        - {"type": "search", "query": "..."}
        - {"type": "report", "docs": [...]}
        """
        self.current_turn += 1
        
        if action["type"] == "search":
            results = self.index.search(action["query"], k=10)
            self.retrieved_so_far.update(results)
            
            observation = {
                "original_query": self.original_query,
                "search_results": results,
                "turn": self.current_turn,
                "retrieved_so_far": list(self.retrieved_so_far)
            }
            
            # Intermediate reward: did we find new relevant docs?
            new_relevant = set(results) & self.target_docs - self.retrieved_so_far
            reward = len(new_relevant) * 0.1  # Small reward for progress
            
            done = self.current_turn >= self.max_turns
            return observation, reward, done
        
        elif action["type"] == "report":
            # Final reward based on reported documents
            final_docs = set(action["docs"])
            reward = self._compute_final_reward(final_docs)
            return None, reward, True
    
    def _compute_final_reward(self, reported_docs: Set[str]) -> float:
        """SID-1 style recall-focused reward."""
        recall = len(reported_docs & self.target_docs) / len(self.target_docs)
        precision = len(reported_docs & self.target_docs) / max(len(reported_docs), 1)
        
        # Heavy recall bias (SID-1 approach)
        return 0.8 * recall + 0.2 * precision
```

### 3.3 RL Training with GRPO

SID-1 uses Group Relative Policy Optimization (GRPO). Simplified implementation:

```python
from transformers import AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig

class AgenticRetrievalTrainer:
    """Train retrieval model with RL."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B"):
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        self.ppo_config = PPOConfig(
            model_name=model_name,
            learning_rate=1e-5,
            batch_size=16,
            mini_batch_size=4,
            gradient_accumulation_steps=4,
        )
        
        self.trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )
    
    def train_episode(self, env: RegoRetrievalEnvironment, query: str, targets: Set[str]):
        """Train on single episode."""
        obs = env.reset(query, targets)
        episode_reward = 0
        
        while True:
            # Model generates action
            prompt = self._format_observation(obs)
            action_text = self.model.generate(prompt)
            action = self._parse_action(action_text)
            
            obs, reward, done = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        # Update model with episode reward
        self.trainer.step(
            queries=[prompt],
            responses=[action_text],
            rewards=[episode_reward]
        )
```

---

## Phase 4: Training Data Pipeline

### 4.1 Synthetic Question Generation (SID-1 Key)

SID-1 emphasizes training without human cold-start data:

```python
class SyntheticQuestionGenerator:
    """Generate training questions without human labeling."""
    
    def __init__(self, llm, knowledge_base):
        self.llm = llm
        self.kb = knowledge_base
    
    def generate_for_helper(self, helper: HelperInfo) -> List[TrainingQuestion]:
        """Generate diverse questions for a helper."""
        questions = []
        
        # Type 1: Direct usage questions
        direct_prompt = f"""
        Given this helper function:
        {helper.source}
        
        Generate questions a developer might ask when they need this.
        """
        questions.extend(self.llm.generate(direct_prompt))
        
        # Type 2: Problem-based questions  
        problem_prompt = f"""
        This helper solves: {helper.description}
        
        Generate questions describing the PROBLEM, not the solution.
        Example: "My builds keep failing security checks" (not "use tekton.task_ref")
        """
        questions.extend(self.llm.generate(problem_prompt))
        
        # Type 3: Multi-hop questions (harder)
        multihop_prompt = f"""
        This helper is often used with: {helper.related_helpers}
        
        Generate questions that require COMBINING multiple helpers.
        Example: "Check all tasks in pipeline have pinned bundles and passed tests"
        """
        questions.extend(self.llm.generate(multihop_prompt))
        
        return questions
```

### 4.2 Ground Truth Document Labels

Create comprehensive ground truth for each question:

```python
@dataclass
class TrainingExample:
    """Training example with full ground truth."""
    
    question: str
    
    # All documents that could help answer this question
    # (SID-1 key insight: multi-document, not single-document)
    target_documents: List[str]  
    
    # Documents that are similar but wrong (for hard negative mining)
    confusing_documents: List[str]
    
    # Difficulty estimate (for curriculum learning)
    difficulty: float  # 0.0 = easy, 1.0 = hard
    
    # Multi-hop indicator
    requires_multi_hop: bool
```

### 4.3 Difficulty-Based Curriculum

SID-1 shows scaling with training compute. Implement curriculum learning:

```python
class CurriculumScheduler:
    """Schedule training examples by difficulty."""
    
    def __init__(self, examples: List[TrainingExample]):
        self.examples = sorted(examples, key=lambda x: x.difficulty)
    
    def get_batch(self, epoch: int, batch_size: int) -> List[TrainingExample]:
        """
        Start with easy examples, gradually include harder ones.
        
        Epoch 0: Only easiest 20%
        Epoch 5: Up to 50% difficulty
        Epoch 10: All examples
        """
        max_difficulty = min(0.2 + epoch * 0.08, 1.0)
        
        eligible = [ex for ex in self.examples if ex.difficulty <= max_difficulty]
        return random.sample(eligible, min(batch_size, len(eligible)))
```

---

## Phase 5: Evaluation Framework

### 5.1 Metrics (SID-1 Aligned)

```python
@dataclass
class RetrievalMetrics:
    """Comprehensive retrieval evaluation metrics."""
    
    # Primary metric (SID-1 focus)
    recall_at_k: Dict[int, float]  # R@1, R@3, R@5, R@7
    
    # Secondary metrics
    ndcg: float  # Normalized Discounted Cumulative Gain
    mrr: float   # Mean Reciprocal Rank
    precision_at_k: Dict[int, float]
    
    # Efficiency metrics
    avg_latency_ms: float
    avg_turns: float  # For agentic retrieval


def evaluate_retrieval(
    model,
    test_examples: List[TrainingExample],
    k_values: List[int] = [1, 3, 5, 7]
) -> RetrievalMetrics:
    """Evaluate retrieval performance."""
    
    recalls = {k: [] for k in k_values}
    ndcgs = []
    mrrs = []
    
    for example in test_examples:
        retrieved = model.retrieve(example.question, max_k=max(k_values))
        
        for k in k_values:
            recall = len(set(retrieved[:k]) & set(example.target_documents))
            recall /= len(example.target_documents)
            recalls[k].append(recall)
        
        # NDCG
        ndcg = compute_ndcg(retrieved, example.target_documents)
        ndcgs.append(ndcg)
        
        # MRR
        for i, doc in enumerate(retrieved):
            if doc in example.target_documents:
                mrrs.append(1.0 / (i + 1))
                break
        else:
            mrrs.append(0.0)
    
    return RetrievalMetrics(
        recall_at_k={k: np.mean(recalls[k]) for k in k_values},
        ndcg=np.mean(ndcgs),
        mrr=np.mean(mrrs),
        # ... other metrics
    )
```

### 5.2 Benchmark Questions

Create a held-out benchmark with diverse difficulty:

```python
BENCHMARK_QUESTIONS = [
    # Easy: Direct helper lookup
    {
        "question": "How to iterate over pipelinerun attestations",
        "targets": ["lib.pipelinerun_attestations"],
        "difficulty": 0.1
    },
    
    # Medium: Requires understanding
    {
        "question": "Check if task bundle is pinned to a digest",
        "targets": ["tekton.task_ref", "tekton.unpinned_task_bundle"],
        "difficulty": 0.5
    },
    
    # Hard: Multi-hop reasoning
    {
        "question": "Verify all build tasks in pipeline produce images with valid digests",
        "targets": [
            "lib.pipelinerun_attestations",
            "tekton.build_tasks",
            "tekton.task_result",
            "lib.image.parse"
        ],
        "difficulty": 0.9
    },
]
```

---

## Implementation Roadmap

### Week 1-2: Enhanced Contrastive (Quick Wins) ✅ IMPLEMENTED

| Task | Priority | Status | Script |
|------|----------|--------|--------|
| Upgrade to `bge-base-en-v1.5` | High | ✅ Done | `improved_retrieval_training.py` |
| Implement hard negative mining | High | ✅ Done | `improved_retrieval_training.py` |
| Switch to MNRL loss | High | ✅ Done | `improved_retrieval_training.py` |
| Add synthetic query generation | Medium | ✅ Done | `generate_synthetic_queries.py` |
| Evaluate on benchmark | High | ✅ Done | `improved_retrieval_training.py` |

**Expected improvement**: 10-20% better recall

### Week 3-4: Document-Centric Rewards ✅ IMPLEMENTED

| Task | Priority | Status | Script |
|------|----------|--------|--------|
| Implement partial credit rewards | High | ✅ Done | `train_agentic_retrieval.py` |
| Multi-positive training | High | ✅ Done | `improved_retrieval_training.py` |
| Create ground truth labels | High | ✅ Done | `generate_synthetic_queries.py` |
| Re-train with new rewards | High | ✅ Done | `train_agentic_retrieval.py` |

**Expected improvement**: Additional 15-25% recall improvement

### Week 5-8: Agentic Retrieval (Major Effort) ✅ IMPLEMENTED

| Task | Priority | Status | Script |
|------|----------|--------|--------|
| Design RL environment | High | ✅ Done | `train_agentic_retrieval.py` |
| Implement multi-turn search | High | ✅ Done | `train_agentic_retrieval.py` |
| GRPO/PPO training setup | High | ✅ Done | `train_agentic_retrieval.py` |
| Curriculum learning | Medium | ✅ Done | `improved_retrieval_training.py` |
| Inference pipeline | High | ✅ Done | `infer_agentic_retrieval.py` |

**Expected improvement**: ~1.5-2x recall improvement (SID-1 results)

---

## Resource Requirements

### Compute

| Phase | GPU Hours | Notes |
|-------|-----------|-------|
| Phase 1 (Contrastive) | 4-8 hours | A100 or similar |
| Phase 2 (Rewards) | 8-16 hours | Retraining |
| Phase 3 (RL) | 40-80 hours | Multi-turn RL expensive |

### Data

| Dataset | Size | Source |
|---------|------|--------|
| Curated queries | ~800 | Existing |
| Synthetic queries | ~5,000 | LLM generated |
| Hard negatives | ~20,000 | Mined from model |
| Ground truth labels | ~2,000 | Manual + LLM |

---

## Key Takeaways from SID-1

1. **Multi-turn beats single-step**: Allow the model to iterate and refine
2. **Recall > Precision**: Missing a crucial document is worse than including extras
3. **Synthetic data works**: No need for expensive human labeling
4. **RL is worth it**: 1.9x improvement over contrastive methods
5. **Scaling continues**: Log-linear improvement with more training compute
6. **Document-centric rewards**: Reward finding documents, not answers
7. **Composability**: Retrieval model works as sub-agent for larger systems

---

## Scripts Created

### Recommended Approach (Fast Baseline)

| Script | Purpose |
|--------|---------|
| `scripts/train_id_selector.py` | **THE MAIN SCRIPT** - BM25 + embeddings for candidates, DPO for selection |
| `scripts/generate_synthetic_queries.py` | Generate synthetic training queries without manual labeling |

### Alternative Approaches

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `scripts/improved_retrieval_training.py` | Enhanced contrastive embedding training | Only if candidate gen is proven bottleneck |
| `scripts/train_agentic_retrieval.py` | Multi-turn RL (PPO) | Only after baseline works |
| `scripts/infer_agentic_retrieval.py` | Agentic inference | Only if using multi-turn |

## Quick Start (Recommended Path)

```bash
# Step 1: Generate synthetic queries with ground truth labels
python scripts/generate_synthetic_queries.py --kb-dir data/knowledge_base

# Step 2: Train ID Selector with DPO (THE MAIN APPROACH)
python scripts/train_id_selector.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --embedding-model BAAI/bge-base-en-v1.5 \
    --epochs 3 \
    --output-dir models/id-selector

# Step 3: Evaluate
python scripts/train_id_selector.py \
    --model-name models/id-selector \
    --eval-only
```

### Alternative: Multi-Turn Agentic (Advanced)

Only pursue after the baseline is working:

```bash
# Train agentic retrieval model (requires GPU, more complex)
python scripts/train_agentic_retrieval.py \
    --model-name Qwen/Qwen2.5-1.5B-Instruct \
    --episodes 1000 \
    --output-dir models/agentic-retrieval
```

## Requirements

```bash
# Core
pip install sentence-transformers faiss-cpu torch transformers rank-bm25

# For DPO training (recommended)
pip install trl peft bitsandbytes accelerate datasets

# For RL training (advanced)
pip install trl>=0.7.0
```

---

## References

- [SID-1 Technical Report](https://www.sid.ai/research/SID-1_Preview/technical-report/SID_1_Technical_Report__Test_Time_Compute_for_Retrieval.pdf) - SID Research, December 2025
- [Qwen3-Embedding](https://arxiv.org/abs/2506.05176) - State-of-the-art embedding model
- [GRPO](https://arxiv.org/abs/2504.20571) - Group Relative Policy Optimization for RL

