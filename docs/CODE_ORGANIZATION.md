# Code Organization: RAG-Based Rego Assistant

## Directory Structure

```
policy-training/
├── src/                          # Core library modules (importable)
│   ├── __init__.py
│   │
│   │  ─────────────────────────────────────────────────────────
│   │  EXISTING (keep)
│   │  ─────────────────────────────────────────────────────────
│   ├── paths.py                  # Central path definitions
│   ├── logging_setup.py          # Structured logging
│   ├── library_mapper.py         # Maps imports to directories
│   ├── context_extractor.py      # Extracts signatures from Rego
│   ├── rego_validator.py         # opa fmt, opa test, code extraction
│   │
│   │  ─────────────────────────────────────────────────────────
│   │  ENHANCE (modify existing)
│   │  ─────────────────────────────────────────────────────────
│   ├── library_indexer.py        # Index helpers → ADD two-tier chunks
│   ├── smart_context_builder.py  # Token budgets → ADD progressive disclosure
│   │
│   │  ─────────────────────────────────────────────────────────
│   │  NEW (create)
│   │  ─────────────────────────────────────────────────────────
│   ├── schema_extractor.py       # Extract schemas from attestations
│   ├── knowledge_base.py         # Unified KB access (helpers + schemas)
│   ├── vector_index.py           # FAISS vector search
│   ├── bm25_index.py             # BM25 keyword search
│   ├── hybrid_retriever.py       # Combine vector + BM25 + rerank
│   ├── plan_validator.py         # Validate plans against KB
│   ├── context_allocator.py      # Token budget management
│   ├── provenance.py             # Track citations
│   └── kb_manifest.py            # KB versioning
│
├── scripts/                      # Executable scripts (run directly)
│   │
│   │  ─────────────────────────────────────────────────────────
│   │  BUILD SCRIPTS (run once to create artifacts)
│   │  ─────────────────────────────────────────────────────────
│   ├── build_kb.py               # Build knowledge base from policy/lib + attestations
│   ├── build_index.py            # Build vector + BM25 indexes
│   ├── enrich_kb.py              # LLM enrichment for descriptions (optional)
│   │
│   │  ─────────────────────────────────────────────────────────
│   │  TRAINING SCRIPTS (generate training data)
│   │  ─────────────────────────────────────────────────────────
│   ├── generate_rag_training.py  # Generate training data with real retrieval
│   ├── validate_training.py      # Validate training data quality
│   │
│   │  ─────────────────────────────────────────────────────────
│   │  INFERENCE SCRIPTS (run the assistant)
│   │  ─────────────────────────────────────────────────────────
│   ├── infer.py                  # Main inference CLI
│   ├── serve.py                  # HTTP API server (optional)
│   │
│   │  ─────────────────────────────────────────────────────────
│   │  UTILITY SCRIPTS
│   │  ─────────────────────────────────────────────────────────
│   ├── test_retrieval.py         # Test retrieval quality
│   ├── eval_pipeline.py          # Evaluate full pipeline
│   └── inspect_kb.py             # Inspect KB contents
│
├── data/
│   ├── attestations/             # Input: attestation JSON files
│   ├── knowledge_base/           # Output: built KB
│   │   ├── manifest.yaml         # KB version info
│   │   ├── helpers.jsonl         # Helper cards
│   │   ├── helpers_full.jsonl    # Helper full bodies
│   │   ├── schemas.jsonl         # Schema definitions
│   │   └── index/                # Vector + BM25 indexes
│   │       ├── helpers.faiss
│   │       ├── helpers_bm25.pkl
│   │       ├── schemas.faiss
│   │       └── schemas_bm25.pkl
│   └── training/                 # Training datasets
│
└── policy/                       # Input: Rego policy source
    ├── lib/                      # Helper libraries
    └── release/                  # Production rules
```

---

## Script Reference

### Build Scripts

These scripts build artifacts. Run them when policy/lib or attestations change.

---

#### `scripts/build_kb.py`

**Purpose:** Extract helpers and schemas into knowledge base files.

**When to run:**
- First time setup
- When `policy/lib/*.rego` files change
- When `data/attestations/*.json` files are added/changed

**Inputs:**
- `policy/lib/**/*.rego` — Rego helper libraries
- `data/attestations/*.json` — Attestation examples

**Outputs:**
- `data/knowledge_base/helpers.jsonl` — Helper metadata (cards)
- `data/knowledge_base/helpers_full.jsonl` — Full helper bodies
- `data/knowledge_base/schemas.jsonl` — Schema definitions
- `data/knowledge_base/manifest.yaml` — Version info

**Usage:**
```bash
# Build everything
python scripts/build_kb.py

# Build only helpers
python scripts/build_kb.py --helpers-only

# Build only schemas
python scripts/build_kb.py --schemas-only

# Specify output directory
python scripts/build_kb.py --output data/knowledge_base/
```

**Dependencies:** None (uses only stdlib + existing src modules)

---

#### `scripts/build_index.py`

**Purpose:** Build vector and BM25 search indexes from knowledge base.

**When to run:**
- After `build_kb.py` completes
- When switching embedding models

**Inputs:**
- `data/knowledge_base/helpers.jsonl`
- `data/knowledge_base/schemas.jsonl`

**Outputs:**
- `data/knowledge_base/index/helpers.faiss`
- `data/knowledge_base/index/helpers_bm25.pkl`
- `data/knowledge_base/index/schemas.faiss`
- `data/knowledge_base/index/schemas_bm25.pkl`

**Usage:**
```bash
# Build all indexes
python scripts/build_index.py

# Build only vector indexes
python scripts/build_index.py --vector-only

# Build only BM25 indexes
python scripts/build_index.py --bm25-only

# Use different embedding model
python scripts/build_index.py --embedding-model bge-small-en-v1.5
```

**Dependencies:** `sentence-transformers`, `faiss-cpu`, `rank-bm25`

---

#### `scripts/enrich_kb.py`

**Purpose:** Use LLM to generate descriptions and `use_when` tags for helpers/schemas.

**When to run:**
- After `build_kb.py` (optional enhancement)
- When you want better retrieval quality

**Inputs:**
- `data/knowledge_base/helpers.jsonl`
- `data/knowledge_base/schemas.jsonl`

**Outputs:**
- `data/knowledge_base/helpers.jsonl` (enriched in-place)
- `data/knowledge_base/schemas.jsonl` (enriched in-place)

**Usage:**
```bash
# Enrich using Ollama
python scripts/enrich_kb.py --model qwen3-coder:30b

# Enrich using OpenAI
python scripts/enrich_kb.py --provider openai --model gpt-4

# Enrich only unenriched entries
python scripts/enrich_kb.py --skip-existing
```

**Dependencies:** `ollama` or `openai`

---

### Training Scripts

These scripts generate training data.

---

#### `scripts/generate_rag_training.py`

**Purpose:** Generate training data using real retrieval (train/deploy parity).

**When to run:**
- After building KB and indexes
- Before fine-tuning

**Inputs:**
- `data/knowledge_base/` — Built KB with indexes
- `policy/release/**/*.rego` — Existing rules as ground truth

**Outputs:**
- `data/training/rag/stage1_train.jsonl` — Planner training data
- `data/training/rag/stage1_eval.jsonl` — Planner eval data
- `data/training/rag/stage2_train.jsonl` — Codegen training data
- `data/training/rag/stage2_eval.jsonl` — Codegen eval data

**Usage:**
```bash
# Generate all training data
python scripts/generate_rag_training.py

# Generate only Stage 1 (planner)
python scripts/generate_rag_training.py --stage 1

# Generate only Stage 2 (codegen)
python scripts/generate_rag_training.py --stage 2

# Specify train/eval split
python scripts/generate_rag_training.py --train-split 0.9
```

**Dependencies:** KB + indexes must exist

---

#### `scripts/validate_training.py`

**Purpose:** Validate training data quality and consistency.

**When to run:**
- After generating training data
- Before fine-tuning

**Inputs:**
- `data/training/rag/*.jsonl`

**Outputs:**
- Validation report (stdout)
- Errors written to `logs/validate_training.log`

**Usage:**
```bash
# Validate all training files
python scripts/validate_training.py

# Validate specific file
python scripts/validate_training.py --file data/training/rag/stage1_train.jsonl

# Check that all helpers in training data exist in KB
python scripts/validate_training.py --check-helpers
```

---

### Inference Scripts

These scripts run the assistant.

---

#### `scripts/infer.py`

**Purpose:** Main CLI for generating Rego rules.

**When to run:**
- User wants to generate a rule

**Inputs:**
- User instruction (CLI arg or stdin)
- `data/knowledge_base/` — Built KB with indexes
- Model path (fine-tuned or base)

**Outputs:**
- Generated rule + tests + metadata (stdout or file)
- Provenance/citations
- Verification status

**Usage:**
```bash
# Basic usage
python scripts/infer.py \
    --instruction "Deny if task bundle not pinned" \
    --kb data/knowledge_base/ \
    --model models/rego-assistant

# Output to file
python scripts/infer.py \
    --instruction "Check GPL licenses in SBOM" \
    --output generated_rule.rego \
    --output-format json

# Stage 1 only (get plan)
python scripts/infer.py \
    --instruction "Verify provenance" \
    --stage 1

# Verbose mode (show retrieval, plan, verification)
python scripts/infer.py \
    --instruction "Deny if task bundle not pinned" \
    --verbose

# Skip verification (faster, for testing)
python scripts/infer.py \
    --instruction "Check tasks" \
    --skip-verify
```

**Output format (JSON):**
```json
{
  "success": true,
  "rule": "package policy.task_bundle\n\ndeny contains result if {...}",
  "tests": "package policy.task_bundle_test\n\ntest_deny {...}",
  "metadata": {
    "rule_data_keys": ["allowed_bundles"]
  },
  "plan": {
    "package": "task_bundle",
    "helpers": ["tekton.task_ref", "tekton.tasks"],
    "schemas": ["slsa_v1_task_bundle"]
  },
  "provenance": {
    "kb_version": "git:abc123",
    "helpers_used": [
      {"id": "tekton.task_ref", "source": "policy/lib/tekton/tekton.rego:142-158"}
    ]
  },
  "verification": {
    "opa_fmt": "passed",
    "opa_test": "passed (3/3)"
  }
}
```

**Dependencies:** KB + indexes must exist; model must be loaded

---

#### `scripts/serve.py`

**Purpose:** HTTP API server for the assistant.

**When to run:**
- Deploy as a service

**Usage:**
```bash
# Start server
python scripts/serve.py \
    --kb data/knowledge_base/ \
    --model models/rego-assistant \
    --port 8080

# With CORS for web UI
python scripts/serve.py --cors-origins "*"
```

**API:**
```
POST /api/generate
{
  "instruction": "Deny if task bundle not pinned"
}

Response: Same as infer.py JSON output
```

---

### Utility Scripts

These scripts help with debugging and evaluation.

---

#### `scripts/test_retrieval.py`

**Purpose:** Test retrieval quality on sample queries.

**Usage:**
```bash
# Interactive mode
python scripts/test_retrieval.py --interactive

# Test specific query
python scripts/test_retrieval.py --query "check task bundle pinning"

# Run evaluation suite
python scripts/test_retrieval.py --eval-file eval/retrieval_queries.json
```

**Output:**
```
Query: "check task bundle pinning"
Retrieved (top 5):
  1. tekton.task_ref (score: 0.89) ✓ [expected]
  2. tekton.tasks (score: 0.82) ✓ [expected]
  3. lib.pipelinerun_attestations (score: 0.78) ✓ [expected]
  4. tekton.task_name (score: 0.71)
  5. sbom.spdx_sboms (score: 0.65) ✗ [irrelevant]

Recall@5: 3/3 = 100%
```

---

#### `scripts/eval_pipeline.py`

**Purpose:** Evaluate full pipeline on test cases.

**Usage:**
```bash
# Run full evaluation
python scripts/eval_pipeline.py --test-file eval/test_cases.json

# Run specific test
python scripts/eval_pipeline.py --test-id bundle_pinning_001

# Generate report
python scripts/eval_pipeline.py --output eval/report.json
```

**Metrics reported:**
- Compile rate
- Test pass rate
- Retrieval recall@K
- Plan validity rate
- Out-of-plan reference rate

---

#### `scripts/inspect_kb.py`

**Purpose:** Inspect knowledge base contents.

**Usage:**
```bash
# List all helpers
python scripts/inspect_kb.py --list helpers

# List all schemas
python scripts/inspect_kb.py --list schemas

# Show specific helper
python scripts/inspect_kb.py --show helper:tekton.task_ref

# Show specific schema
python scripts/inspect_kb.py --show schema:slsa_v1_task_bundle

# Search KB
python scripts/inspect_kb.py --search "bundle pinning"

# Show KB stats
python scripts/inspect_kb.py --stats
```

---

## Module Reference (src/)

### Existing Modules (No Changes)

| Module | Purpose | Used By |
|--------|---------|---------|
| `paths.py` | Central path definitions | All scripts |
| `logging_setup.py` | Structured logging | All scripts |
| `library_mapper.py` | Map `data.lib.*` to directories | `library_indexer.py` |
| `context_extractor.py` | Extract signatures, usage sites | `library_indexer.py` |
| `rego_validator.py` | `opa fmt`, `opa test`, code extraction | `infer.py`, `eval_pipeline.py` |

### Enhanced Modules

| Module | Changes | Used By |
|--------|---------|---------|
| `library_indexer.py` | Add source spans, two-tier chunks, grounding | `build_kb.py` |
| `smart_context_builder.py` | Add progressive disclosure, per-type caps | `context_allocator.py` |

### New Modules

| Module | Purpose | Used By |
|--------|---------|---------|
| `schema_extractor.py` | Extract schemas from attestations | `build_kb.py` |
| `knowledge_base.py` | Unified KB access (load, query helpers/schemas) | `infer.py`, `generate_rag_training.py` |
| `vector_index.py` | FAISS vector search | `hybrid_retriever.py` |
| `bm25_index.py` | BM25 keyword search | `hybrid_retriever.py` |
| `hybrid_retriever.py` | Combine vector + BM25 + rerank | `infer.py`, `generate_rag_training.py` |
| `plan_validator.py` | Validate plans against KB | `infer.py` |
| `context_allocator.py` | Token budget management | `infer.py` |
| `provenance.py` | Track citations | `infer.py` |
| `kb_manifest.py` | KB versioning | `build_kb.py`, `infer.py` |

---

## Workflow: What to Run When

### First Time Setup

```bash
# 1. Install dependencies
pip install sentence-transformers faiss-cpu rank-bm25

# 2. Build knowledge base
python scripts/build_kb.py

# 3. Build search indexes
python scripts/build_index.py

# 4. (Optional) Enrich with LLM descriptions
python scripts/enrich_kb.py --model qwen3-coder:30b

# 5. Test retrieval quality
python scripts/test_retrieval.py --interactive
```

### When Policy Libraries Change

```bash
# Rebuild KB and indexes
python scripts/build_kb.py --helpers-only
python scripts/build_index.py
```

### When Attestation Examples Change

```bash
# Rebuild schemas and indexes
python scripts/build_kb.py --schemas-only
python scripts/build_index.py
```

### Generate Training Data

```bash
# Generate training data with real retrieval
python scripts/generate_rag_training.py

# Validate quality
python scripts/validate_training.py
```

### Run Inference

```bash
# Generate a rule
python scripts/infer.py \
    --instruction "Deny if task bundle not pinned" \
    --kb data/knowledge_base/ \
    --model models/rego-assistant
```

### Evaluate Pipeline

```bash
# Full evaluation
python scripts/eval_pipeline.py --test-file eval/test_cases.json
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         BUILD PHASE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  policy/lib/*.rego ──► build_kb.py ──► knowledge_base/          │
│  data/attestations/ ─┘               │  ├── helpers.jsonl       │
│                                      │  ├── helpers_full.jsonl  │
│                                      │  ├── schemas.jsonl       │
│                                      │  └── manifest.yaml       │
│                                      │                           │
│                                      ▼                           │
│                                build_index.py                    │
│                                      │                           │
│                                      ▼                           │
│                                  index/                          │
│                                  ├── helpers.faiss               │
│                                  ├── helpers_bm25.pkl            │
│                                  ├── schemas.faiss               │
│                                  └── schemas_bm25.pkl            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       TRAINING PHASE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  knowledge_base/ ──► generate_rag_training.py ──► training/     │
│  policy/release/ ─┘                              │  ├── stage1_train.jsonl
│                                                  │  ├── stage1_eval.jsonl
│                                                  │  ├── stage2_train.jsonl
│                                                  │  └── stage2_eval.jsonl
│                                                  │
│                                                  ▼
│                                           train_policy.py
│                                                  │
│                                                  ▼
│                                            models/rego-assistant
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       INFERENCE PHASE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Instruction ──► infer.py                                  │
│                          │                                       │
│                          ├── 1. Load KB + indexes               │
│                          │                                       │
│                          ├── 2. Retrieve (hybrid_retriever)     │
│                          │      └── vector_index + bm25_index   │
│                          │                                       │
│                          ├── 3. Plan (Stage 1 model)            │
│                          │      └── context_allocator (cards)   │
│                          │                                       │
│                          ├── 4. Validate (plan_validator)       │
│                          │      └── check helpers/schemas exist │
│                          │                                       │
│                          ├── 5. Generate (Stage 2 model)        │
│                          │      └── context_allocator (full)    │
│                          │                                       │
│                          ├── 6. Verify (rego_validator)         │
│                          │      └── opa fmt, opa test           │
│                          │                                       │
│                          └── 7. Output with provenance          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference Card

| Task | Command |
|------|---------|
| **Build KB** | `python scripts/build_kb.py` |
| **Build indexes** | `python scripts/build_index.py` |
| **Test retrieval** | `python scripts/test_retrieval.py --interactive` |
| **Generate training** | `python scripts/generate_rag_training.py` |
| **Run inference** | `python scripts/infer.py --instruction "..."` |
| **Evaluate** | `python scripts/eval_pipeline.py` |
| **Inspect KB** | `python scripts/inspect_kb.py --stats` |

