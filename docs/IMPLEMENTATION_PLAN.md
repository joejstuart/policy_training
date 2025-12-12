# Implementation Plan: RAG-Based Rego Assistant

## Executive Summary

**Good news:** ~60% of the required components already exist and can be reused or enhanced. The existing codebase has solid foundations for helper extraction, context building, validation, and two-stage inference.

**What's needed:** Vector/BM25 indexing, schema extraction with canonical IDs, validation gate, and provenance tracking.

---

## Existing Code Assessment

### ✅ Can Reuse Directly

| Component | File | What It Does | Reuse Status |
|-----------|------|--------------|--------------|
| **Library Mapper** | `src/library_mapper.py` | Maps `data.lib.*` imports to directories | ✅ Use as-is |
| **Context Extractor** | `src/context_extractor.py` | Extracts signatures, usage sites | ✅ Use as-is |
| **Rego Validator** | `src/rego_validator.py` | `opa fmt`, `opa test`, code extraction | ✅ Use as-is |
| **Paths** | `src/paths.py` | Central path definitions | ✅ Use as-is |
| **Logging** | `src/logging_setup.py` | Structured logging | ✅ Use as-is |

### 🔧 Enhance Existing

| Component | File | Current State | Enhancement Needed |
|-----------|------|---------------|-------------------|
| **Library Indexer** | `src/library_indexer.py` | Indexes helpers with signatures, docs, keywords | Add two-tier chunks (card/full), grounding fields, source spans |
| **Smart Context Builder** | `src/smart_context_builder.py` | Token-budgeted context building | Add progressive disclosure, per-type caps |
| **Two-Stage Generator** | `src/infer_two_stage.py` | Planner → Codegen pipeline | Add retrieval integration, validation gate, provenance |
| **Training Data Generator** | `scripts/generate_two_stage_dataset.py` | Creates training examples | Add real retrieval during training |
| **Attestation Analyzer** | `scripts/generate_attestation_dataset.py` | Analyzes attestation structure | Extract to reusable schema extractor |

### 🆕 Build New

| Component | Purpose | Priority |
|-----------|---------|----------|
| **Schema Extractor** | Extract schemas from attestations with canonical IDs | P0 |
| **Vector Index** | Semantic search over helpers/schemas | P0 |
| **BM25 Index** | Keyword search for exact symbols | P1 |
| **Hybrid Retriever** | Combine BM25 + vector + reranking | P1 |
| **Plan Validator** | Validate helper/schema existence | P0 |
| **KB Manifest** | Version tracking (git ref, hashes) | P1 |
| **Provenance Tracker** | Track citations in output | P1 |
| **AST Verifier** | Post-codegen out-of-plan reference check | P2 |

---

## Implementation Phases

### Phase 1: Knowledge Base Foundation (Week 1-2)

**Goal:** Extract and store helpers + schemas in two-tier format with grounding.

#### 1.1 Enhance Library Indexer

**File:** `src/library_indexer.py`

**Changes:**

```python
# Current HelperInfo
@dataclass
class HelperInfo:
    name: str
    package: str
    file_path: Path
    import_prefix: str
    signature: str
    doc: str
    usage_examples: List[str]
    keywords: Set[str]

# Enhanced HelperInfo
@dataclass
class HelperInfo:
    # Existing fields
    name: str
    package: str
    file_path: Path
    import_prefix: str
    signature: str
    doc: str
    usage_examples: List[str]
    keywords: Set[str]
    
    # NEW: Grounding fields
    source_span: Tuple[int, int]  # (start_line, end_line)
    body: str  # Full function body
    
    # NEW: Two-tier support
    def to_card(self) -> str:
        """Compact representation (~100-150 tokens)"""
        pass
    
    def to_full(self) -> str:
        """Complete representation (~400-600 tokens)"""
        pass
```

**Tasks:**
- [ ] Add `source_span` extraction (line numbers)
- [ ] Add `body` extraction (full function source)
- [ ] Add `to_card()` method (signature + doc + 1 usage)
- [ ] Add `to_full()` method (body + all examples)
- [ ] Add `to_json()` for serialization
- [ ] Add grounding fields to output

**Estimated effort:** 4-6 hours

#### 1.2 Create Schema Extractor

**New file:** `src/schema_extractor.py`

**Purpose:** Extract schemas from attestation files with canonical IDs.

```python
@dataclass
class SchemaField:
    schema_id: str           # e.g., "slsa_v1_task_bundle"
    canonical_path: str      # e.g., "$.predicate.buildConfig.tasks[*].ref.bundle"
    attestation_type: str    # e.g., "slsa_provenance"
    field_type: str          # string, boolean, array, object
    description: str         # LLM-enriched
    use_when: List[str]      # When to use this field
    example_value: Any       # Real example from fixtures
    source_file: str         # Which attestation file this came from
    aliases: List[str]       # Variations seen
    
    def to_card(self) -> str:
        """Compact representation for retrieval"""
        pass

class SchemaExtractor:
    def __init__(self, attestation_dir: Path):
        pass
    
    def extract_all(self) -> Dict[str, SchemaField]:
        """Extract schemas from all attestation files"""
        pass
    
    def detect_attestation_type(self, data: dict) -> str:
        """Detect if SLSA, SPDX, CycloneDX, etc."""
        pass
    
    def normalize_path(self, path: str) -> str:
        """Convert to canonical JSONPath format"""
        pass
```

**Leverage existing:** The `AttestationAnalyzer` class in `generate_attestation_dataset.py` already does attestation type detection. Extract and reuse.

**Tasks:**
- [ ] Extract `AttestationAnalyzer.detect_attestation_type()` to reusable module
- [ ] Implement path traversal and canonicalization
- [ ] Generate stable `schema_id` from path + type
- [ ] Extract example values from attestations
- [ ] Add LLM enrichment for descriptions (optional, can do later)

**Estimated effort:** 8-12 hours

#### 1.3 KB Manifest and Versioning

**New file:** `src/kb_manifest.py`

```python
@dataclass
class KBManifest:
    git_ref: str
    built_at: str
    policy_lib_hash: str
    attestation_count: int
    helper_count: int
    schema_count: int
    
    @classmethod
    def create(cls, repo_root: Path) -> 'KBManifest':
        """Create manifest from current repo state"""
        pass
    
    def save(self, path: Path):
        pass
    
    @classmethod
    def load(cls, path: Path) -> 'KBManifest':
        pass
```

**Tasks:**
- [ ] Get git ref from repo
- [ ] Hash policy/lib directory
- [ ] Save/load manifest as YAML

**Estimated effort:** 2 hours

---

### Phase 2: Retrieval Infrastructure (Week 2-3)

**Goal:** Build vector + BM25 indexes with hybrid retrieval.

#### 2.1 Vector Index

**New file:** `src/vector_index.py`

```python
class VectorIndex:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embedding_model)
        self.index = None  # FAISS index
        self.chunks = []   # Original chunks for retrieval
    
    def build(self, chunks: List[dict]):
        """Build index from helper/schema chunks"""
        pass
    
    def search(self, query: str, top_k: int = 10) -> List[dict]:
        """Semantic search"""
        pass
    
    def save(self, path: Path):
        pass
    
    def load(self, path: Path):
        pass
```

**Dependencies:**
- `sentence-transformers`
- `faiss-cpu` (or `faiss-gpu`)

**Tasks:**
- [ ] Install dependencies
- [ ] Implement embedding generation
- [ ] Implement FAISS index build/search
- [ ] Implement save/load

**Estimated effort:** 4-6 hours

#### 2.2 BM25 Index

**New file:** `src/bm25_index.py`

```python
class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.chunks = []
    
    def build(self, chunks: List[dict], text_field: str = "text"):
        """Build BM25 index from chunks"""
        pass
    
    def search(self, query: str, top_k: int = 20) -> List[dict]:
        """Keyword search"""
        pass
```

**Dependencies:**
- `rank-bm25`

**Tasks:**
- [ ] Implement tokenization
- [ ] Implement BM25 build/search

**Estimated effort:** 2-3 hours

#### 2.3 Hybrid Retriever

**New file:** `src/hybrid_retriever.py`

```python
class HybridRetriever:
    def __init__(
        self,
        vector_index: VectorIndex,
        bm25_index: BM25Index,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        pass
    
    def retrieve(
        self,
        query: str,
        helper_k: int = 4,
        schema_k: int = 2,
        usage_k: int = 1
    ) -> RetrievalResult:
        """Hybrid retrieval with per-type caps"""
        # 1. Query BM25 (top 20)
        # 2. Query vector (top 20)
        # 3. Merge and dedupe (top 30)
        # 4. Rerank with cross-encoder
        # 5. Apply per-type caps
        pass
```

**Dependencies:**
- `sentence-transformers` (for cross-encoder)

**Tasks:**
- [ ] Implement merge + dedupe logic
- [ ] Implement reranking
- [ ] Implement per-type caps
- [ ] Add multi-query by intent (optional enhancement)

**Estimated effort:** 6-8 hours

---

### Phase 3: Validation Gate (Week 3)

**Goal:** Validate plans before codegen, route repairs correctly.

#### 3.1 Plan Validator

**New file:** `src/plan_validator.py`

```python
@dataclass
class PlanValidationResult:
    valid: bool
    errors: List[str]
    warnings: List[str]

class PlanValidator:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
    
    def validate(self, plan: dict) -> PlanValidationResult:
        """Validate plan against KB"""
        errors = []
        
        # Check helpers exist
        for helper in plan.get("helpers", []):
            if not self.kb.helper_exists(helper["id"]):
                errors.append(f"Helper '{helper['id']}' not found")
        
        # Check schemas exist for attestation type
        for schema in plan.get("schemas", []):
            if not self.kb.schema_exists(schema, plan.get("attestation_type")):
                errors.append(f"Schema '{schema}' not found")
        
        # Check importability
        # Check type compatibility (best effort)
        
        return PlanValidationResult(valid=len(errors) == 0, errors=errors)
```

**Tasks:**
- [ ] Implement helper existence check
- [ ] Implement schema existence check
- [ ] Implement importability check
- [ ] Add repair suggestion generation

**Estimated effort:** 4-6 hours

#### 3.2 Integrate Validation into Inference

**File:** `src/infer_two_stage.py`

**Changes:**
- Add retrieval call before planner
- Add validation gate between planner and codegen
- Add repair loop for invalid plans

```python
def generate_rule(self, instruction: str) -> GenerationResult:
    # NEW: Retrieve context
    retrieved = self.retriever.retrieve(instruction)
    
    # Stage 1: Planner
    plan = self.run_planner(instruction, retrieved)
    
    # NEW: Validate plan
    for attempt in range(self.max_repair_attempts):
        validation = self.validator.validate(plan)
        if validation.valid:
            break
        plan = self.repair_plan(plan, validation.errors, retrieved)
    
    if not validation.valid:
        return GenerationResult(success=False, errors=validation.errors)
    
    # Stage 2: Codegen (with selected helper bodies)
    selected_context = self.build_codegen_context(plan, retrieved)
    output = self.run_codegen(instruction, plan, selected_context)
    
    # NEW: Verify codegen used only plan helpers
    verify_result = self.verify_codegen(output, plan)
    
    return GenerationResult(
        success=True,
        rule=output.rule,
        tests=output.tests,
        plan=plan,
        provenance=self.build_provenance(retrieved, plan)
    )
```

**Estimated effort:** 6-8 hours

---

### Phase 4: Context Budget Allocator (Week 3-4)

**Goal:** Manage token budgets with progressive disclosure.

#### 4.1 Context Allocator

**Enhance file:** `src/smart_context_builder.py`

```python
class ContextBudgetAllocator:
    BUDGETS = {
        "small": {  # 8K context models
            "instruction": 600,
            "plan": 400,
            "helpers": 2500,
            "schemas": 1500,
            "idioms": 800,
        },
        "large": {  # 32K+ context models
            "instruction": 800,
            "plan": 600,
            "helpers": 8000,
            "schemas": 4000,
            "idioms": 2000,
        }
    }
    
    TRUNCATION_ORDER = [
        "extra_idioms",
        "extra_schema_examples",
        "helper_bodies_to_cards",
        "reduce_k",
    ]
    
    def allocate(
        self,
        retrieved: RetrievalResult,
        stage: str,  # "planner" or "codegen"
        budget_profile: str = "small"
    ) -> AllocatedContext:
        """Allocate context within budget"""
        pass
```

**Tasks:**
- [ ] Implement token counting (reuse existing `estimate_tokens`)
- [ ] Implement progressive disclosure (cards for planner, full for codegen)
- [ ] Implement truncation order
- [ ] Integrate with inference pipeline

**Estimated effort:** 4-6 hours

---

### Phase 5: Provenance and Citations (Week 4)

**Goal:** Track and output citations for trust.

#### 5.1 Provenance Tracker

**New file:** `src/provenance.py`

```python
@dataclass
class Provenance:
    kb_version: str
    helpers_used: List[HelperCitation]
    schemas_used: List[SchemaCitation]
    retrieved_chunks: RetrievalTrace
    
    def to_json(self) -> dict:
        pass

@dataclass
class HelperCitation:
    id: str
    source_file: str
    source_lines: Tuple[int, int]

@dataclass
class SchemaCitation:
    id: str
    example_source: str

@dataclass
class RetrievalTrace:
    selected: List[str]
    excluded: List[ExcludedChunk]

@dataclass
class ExcludedChunk:
    id: str
    reason: str
```

**Tasks:**
- [ ] Collect citations during inference
- [ ] Include in output JSON
- [ ] Add to CLI output

**Estimated effort:** 4 hours

---

### Phase 6: Training Data with Real Retrieval (Week 4-5)

**Goal:** Generate training data using actual retrieval (train/deploy parity).

#### 6.1 Enhance Training Generator

**File:** `scripts/generate_two_stage_dataset.py`

**Changes:**

```python
class RAGTrainingGenerator:
    def __init__(self, retriever: HybridRetriever, kb: KnowledgeBase):
        self.retriever = retriever
        self.kb = kb
    
    def generate_example(self, rule: ExistingRule) -> TrainingExample:
        instruction = rule.instruction
        
        # CRITICAL: Use real retrieval
        retrieved = self.retriever.retrieve(instruction)
        
        # Extract ground truth plan from existing rule
        expected_plan = self.extract_plan(rule)
        
        # Stage 1 example
        stage1 = {
            "instruction": instruction,
            "retrieved_context": format_as_cards(retrieved),
            "expected_output": expected_plan
        }
        
        # Stage 2 example
        stage2 = {
            "instruction": instruction,
            "plan": expected_plan,
            "retrieved_context": format_selected_full(retrieved, expected_plan),
            "expected_output": rule.code
        }
        
        return TrainingExample(stage1=stage1, stage2=stage2)
```

**Tasks:**
- [ ] Integrate retriever into training generation
- [ ] Generate validation failure examples
- [ ] Add `excluded_candidates` supervision
- [ ] Add confuser injection (optional)

**Estimated effort:** 8-12 hours

---

### Phase 7: Build Scripts and CLI (Week 5)

**Goal:** Orchestration scripts for building KB and running inference.

#### 7.1 KB Build Script

**New file:** `scripts/build_knowledge_base.py`

```bash
# Usage
python scripts/build_knowledge_base.py \
    --extract-helpers \
    --extract-schemas \
    --build-index \
    --output data/knowledge_base/
```

**Tasks:**
- [ ] Orchestrate helper extraction
- [ ] Orchestrate schema extraction
- [ ] Build vector + BM25 indexes
- [ ] Generate manifest

**Estimated effort:** 4 hours

#### 7.2 Inference CLI

**Enhance file:** `src/infer_two_stage.py` (or new `src/infer_rag.py`)

```bash
# Usage
python src/infer_rag.py \
    --instruction "Deny if task bundle not pinned" \
    --kb data/knowledge_base/ \
    --model models/rego-assistant \
    --output-format json
```

**Tasks:**
- [ ] Add KB loading
- [ ] Add retrieval
- [ ] Add provenance output
- [ ] Add verification output

**Estimated effort:** 4 hours

---

## Dependency Summary

### New Python Dependencies

```txt
# Add to requirements.txt
sentence-transformers>=2.2.0
faiss-cpu>=1.7.0  # or faiss-gpu
rank-bm25>=0.2.0
```

### File Changes Summary

| File | Action | Priority |
|------|--------|----------|
| `src/library_indexer.py` | Enhance | P0 |
| `src/schema_extractor.py` | **NEW** | P0 |
| `src/kb_manifest.py` | **NEW** | P1 |
| `src/vector_index.py` | **NEW** | P0 |
| `src/bm25_index.py` | **NEW** | P1 |
| `src/hybrid_retriever.py` | **NEW** | P1 |
| `src/plan_validator.py` | **NEW** | P0 |
| `src/smart_context_builder.py` | Enhance | P1 |
| `src/infer_two_stage.py` | Enhance | P0 |
| `src/provenance.py` | **NEW** | P1 |
| `scripts/generate_two_stage_dataset.py` | Enhance | P1 |
| `scripts/build_knowledge_base.py` | **NEW** | P0 |

---

## Estimated Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: KB Foundation | Week 1-2 | None |
| Phase 2: Retrieval | Week 2-3 | Phase 1 |
| Phase 3: Validation Gate | Week 3 | Phase 1, 2 |
| Phase 4: Context Budget | Week 3-4 | Phase 2 |
| Phase 5: Provenance | Week 4 | Phase 3 |
| Phase 6: Training Data | Week 4-5 | Phase 2, 3 |
| Phase 7: Scripts/CLI | Week 5 | All |

**Total estimated effort:** 60-80 hours (~2 weeks full-time, or 4-5 weeks part-time)

---

## Quick Start: MVP Path

If you want the fastest path to a working system:

### Week 1: Minimal Retrieval

1. **Enhance `library_indexer.py`** with source spans (~4 hours)
2. **Build simple vector index** with FAISS (~4 hours)
3. **Test retrieval quality** on 10 example queries (~2 hours)

### Week 2: Validation Gate

1. **Create `plan_validator.py`** with helper existence check (~4 hours)
2. **Integrate into `infer_two_stage.py`** (~4 hours)
3. **Test end-to-end** on bundle pinning, GPL license examples (~4 hours)

### Week 3: Polish

1. **Add schema extraction** (simplified, no LLM enrichment) (~6 hours)
2. **Add provenance tracking** (~4 hours)
3. **Create build script** (~2 hours)

**MVP total:** ~34 hours

This gives you:
- ✅ Vector retrieval for helpers
- ✅ Validation gate (no hallucinated helpers)
- ✅ Basic schema support
- ✅ Provenance in output

Defer to later:
- ❌ BM25 hybrid (vector-only is fine for MVP)
- ❌ Reranking (top-K is fine for MVP)
- ❌ LLM enrichment (manual descriptions work)
- ❌ Context budget allocator (hardcode for now)
- ❌ Training data with retrieval (use existing data first)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Retrieval quality poor | Start with higher K (20), add BM25 hybrid later |
| Context overflow | Use progressive disclosure, defer to full context stuffing if library is small |
| Validation too strict | Start with existence-only, add type checking later |
| Training data mismatch | Use existing training data for MVP, add retrieval-based training in Phase 6 |

---

## Next Steps

1. **Install dependencies**: `pip install sentence-transformers faiss-cpu rank-bm25`
2. **Start with Phase 1.1**: Enhance `library_indexer.py` with source spans
3. **Build vector index**: Test retrieval on known queries
4. **Integrate validation gate**: Prevent hallucinated helpers

Would you like me to start implementing any specific component?
