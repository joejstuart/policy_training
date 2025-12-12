# RAG-Based Rego Assistant

This document describes how to build, update, and use the RAG-based context retrieval system for Rego policy generation.

## Quick Start

```bash
# Build everything from scratch
uv run python scripts/build_kb.py
uv run python scripts/enrich_kb.py
uv run python scripts/build_index.py

# Run inference with RAG + Ollama
uv run python src/infer_two_stage.py \
    --use-rag \
    --ollama qwen3-coder:30b \
    --instruction "check if task bundle is pinned"

# Run inference with RAG + fine-tuned model
uv run python src/infer_two_stage.py \
    --use-rag \
    --stage2-model models/stage2-rule-generation \
    --instruction "check if task bundle is pinned"
```

## Building the Knowledge Base

### Step 1: Extract Helpers and Schemas

```bash
uv run python scripts/build_kb.py
```

This extracts:
- **Helpers** from `policy/lib/` and `policy/release/lib/` using AST parser
- **Schemas** from `data/attestations/` JSON files

Output: `data/knowledge_base/helpers.jsonl`, `schemas.jsonl`, `manifest.yaml`

### Step 2: Enrich with LLM (Optional but Recommended)

```bash
# Requires Ollama running: ollama serve
uv run python scripts/enrich_kb.py

# Skip already-enriched items
uv run python scripts/enrich_kb.py --skip-existing

# Dry run (see context without calling LLM)
uv run python scripts/enrich_kb.py --dry-run --limit 5
```

This adds:
- **Descriptions** - LLM-generated from source code + usage examples
- **use_when** tags - When to use each helper

### Step 3: Build Search Indexes

```bash
uv run python scripts/build_index.py
```

This creates:
- **Vector index** (FAISS) - Semantic search
- **BM25 index** - Keyword search

Output: `data/knowledge_base/index/`

## Updating the Knowledge Base

### After Policy Changes

```bash
# Rebuild KB (extracts new/modified helpers)
uv run python scripts/build_kb.py

# Enrich only new helpers
uv run python scripts/enrich_kb.py --skip-existing

# Rebuild indexes
uv run python scripts/build_index.py
```

### Full Rebuild

```bash
uv run python scripts/build_kb.py
uv run python scripts/enrich_kb.py  # Re-enriches everything
uv run python scripts/build_index.py
```

## Running Inference

### RAG + Ollama (No Fine-tuning Required)

```bash
# Pull model first
ollama pull qwen3-coder:30b

# Run inference
uv run python src/infer_two_stage.py \
    --use-rag \
    --ollama qwen3-coder:30b \
    --instruction "verify SBOM contains required packages"
```

### RAG + Fine-tuned Model

```bash
uv run python src/infer_two_stage.py \
    --use-rag \
    --stage2-model models/stage2-rule-generation \
    --instruction "check if task bundle is pinned"
```

### Test Retrieval Only (Stage 1)

```bash
uv run python src/infer_two_stage.py \
    --use-rag \
    --stage 1 \
    --instruction "check if task bundle is pinned"
```

### Interactive Mode

```bash
uv run python src/infer_two_stage.py \
    --use-rag \
    --ollama qwen3-coder:30b \
    --interactive
```

## Command Reference

| Command | Description |
|---------|-------------|
| `scripts/build_kb.py` | Extract helpers/schemas into KB |
| `scripts/enrich_kb.py` | Add LLM-generated descriptions |
| `scripts/build_index.py` | Build FAISS + BM25 indexes |
| `scripts/test_retrieval.py` | Test retrieval quality |

## Options

### build_kb.py
```
--output DIR          Output directory (default: data/knowledge_base)
--helpers-only        Only extract helpers
--schemas-only        Only extract schemas
```

### enrich_kb.py
```
--model MODEL         Ollama model (default: qwen3-coder:30b)
--skip-existing       Skip items with descriptions
--dry-run             Show context without calling LLM
--limit N             Limit to N items
--helpers-only        Only enrich helpers
--schemas-only        Only enrich schemas
```

### infer_two_stage.py
```
--use-rag             Use RAG retrieval for context
--ollama MODEL        Use Ollama model for Stage 2
--stage2-model PATH   Use fine-tuned model for Stage 2
--stage {1,2}         Run only Stage 1 or 2
--kb-dir DIR          KB directory (default: data/knowledge_base)
--top-k-helpers N     Helpers to retrieve (default: 7)
--top-k-schemas N     Schemas to retrieve (default: 3)
--interactive         Interactive mode
```

## File Structure

```
data/knowledge_base/
├── helpers.jsonl          # Helper cards (compact)
├── helpers_full.jsonl     # Full helpers (with body)
├── schemas.jsonl          # Schema fields
├── manifest.yaml          # KB version info
└── index/
    ├── helpers_vector/    # FAISS index
    ├── schemas_vector/    # FAISS index
    ├── helpers_bm25.pkl   # BM25 index
    └── schemas_bm25.pkl   # BM25 index
```

## Troubleshooting

### "faiss not available"
```bash
uv pip install faiss-cpu sentence-transformers rank-bm25
```

### "Ollama not available"
```bash
ollama serve  # Start Ollama server
ollama pull qwen3-coder:30b  # Pull model
```

### Poor retrieval quality
1. Check if helpers are enriched: `grep '"description": ""' data/knowledge_base/helpers.jsonl | wc -l`
2. Re-enrich if needed: `uv run python scripts/enrich_kb.py`
3. Rebuild indexes: `uv run python scripts/build_index.py`

### Missing helpers
1. Check KB extraction: `wc -l data/knowledge_base/helpers.jsonl`
2. Verify source directories exist: `ls policy/lib/ policy/release/lib/`
3. Rebuild KB: `uv run python scripts/build_kb.py`

