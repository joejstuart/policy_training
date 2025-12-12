# Rego Developer Assistant — Architecture Specification

A production-oriented architecture for shipping a Rego-expert assistant to developers. The system is accurate, grounded in real libraries/schemas, and resistant to hallucinations. It treats the LLM as a component inside a verified pipeline, not a trusted oracle.

---

## 0) Non-Negotiable Goals

| Goal | Description |
|------|-------------|
| **Grounding** | Every helper/schema used must be traceable to your repo/fixtures |
| **Constrained Generation** | The model can only use what it selected + what exists |
| **Executable Verification** | Compile + tests (and ideally eval on fixtures) gate output |
| **Deterministic Behavior** | Context allocation, truncation, and repair are rule-based, not random |

---

## 1) Knowledge Base (KB)

The KB is the **source of truth** for all helpers, schemas, and patterns. It must be versioned and tied to a specific git ref.

### 1.1 KB Versioning

```yaml
kb_manifest:
  git_ref: "abc123def"
  library_hash: "sha256:..."
  built_at: "2025-12-12T10:00:00Z"
  policy_lib_version: "v1.2.3"
  attestation_types: ["slsa_provenance", "spdx_sbom", "cyclonedx_sbom"]
```

All outputs must annotate which KB version was used. Plan validation checks against this exact ref.

### 1.2 Extract Helpers with AST Parser (not regex)

Store per helper with **two-tier chunks**:

| Tier | Content | Tokens | Use Case |
|------|---------|--------|----------|
| `helper_card` | Signature + purpose + gotchas + 1 idiom | ~100-150 | Planner stage, initial retrieval |
| `helper_full` | Full body + multiple examples | ~400-600 | Codegen for selected helpers |

**Card fields:**

- Fully-qualified name + module path
- Signature (arity)
- Short description (LLM-enriched, grounded)
- `use_when` tags
- 1 canonical usage snippet

**Full fields (in addition to card):**

- Complete function body
- Imports required
- Source location (file + line range)
- Multiple usage snippets mined from real rules/tests
- Related helpers

**Grounding requirement:**

```json
{
  "grounding": {
    "source_span": {"file": "policy/lib/tekton/tekton.rego", "lines": [142, 158]},
    "usage_spans": [
      {"file": "policy/release/attestation_task_bundle.rego", "lines": [45, 52]}
    ],
    "example_code": "ref := tekton.task_ref(task)\nnot ref.pinned"
  }
}
```

### 1.3 Extract Schemas from Attestations

Normalize into a canonical representation with stable IDs:

| Field | Description |
|-------|-------------|
| `schema_id` | Stable identifier (e.g., `slsa_v1_task_bundle`) |
| `canonical_path` | JSONPath-like (e.g., `$.predicate.buildConfig.tasks[*].ref.bundle`) |
| `attestation_type` | `slsa_provenance`, `spdx_sbom`, `cyclonedx_sbom`, etc. |
| `aliases` | Variations seen in practice |
| `type` | string, boolean, array, object |
| `presence` | When field exists vs. is absent |
| `example_fragments` | Small, real examples from fixtures |

**Planner uses IDs, not raw paths:**

```json
{
  "schemas": ["slsa_v1_task_bundle", "slsa_v1_task_name"]
}
```

### 1.4 Idioms Index

Mine common patterns (co-occurrence + minimal snippet):

- PipelineRun iteration skeletons
- `result_helper` patterns
- Typical error formatting
- SBOM traversal idioms

**Outcome:** Three KB collections: `helpers`, `schemas`, `idioms`.

---

## 2) Retrieval That Doesn't Miss

### 2.1 Hybrid Retrieval Architecture

```
User Instruction
       │
       ├──► BM25 Keyword Search (symbols matter)
       │
       └──► Vector Search (semantic)
               │
               ▼
         Merge Top 30
               │
               ▼
         Cross-Encoder Rerank
               │
               ▼
         Final Top 7-10
```

**Why hybrid?**

| Query Type | BM25 Wins | Vector Wins |
|------------|-----------|-------------|
| `tekton.task_ref` | ✅ Exact match | ❌ Similar names |
| "check if bundle is pinned" | ❌ No keywords | ✅ Semantic |
| "GPL license validation" | Mixed | ✅ Semantic |

### 2.2 Multi-Query by Intent

Query as 2-3 sub-queries against separate indexes:

```python
def retrieve_with_intent(instruction: str) -> RetrievalResult:
    helper_query = f"helper function for: {instruction}"
    schema_query = f"schema fields needed for: {instruction}"
    usage_query = f"example code for: {instruction}"
    
    helpers = helper_index.search(helper_query, k=6)
    schemas = schema_index.search(schema_query, k=4)
    usage = usage_index.search(usage_query, k=3)
    
    return RetrievalResult(
        helpers=helpers[:4],      # Max 4 helper chunks
        schemas=schemas[:2],      # Max 2 schema chunks
        usage_patterns=usage[:1]  # Max 1 usage pattern
    )
```

### 2.3 Caps by Type

Cap results to avoid drowning:

| Type | Max Chunks | Rationale |
|------|------------|-----------|
| Helpers | 4 | Core building blocks |
| Schemas | 2-3 | Field structure |
| Idioms/Usage | 1-2 | Pattern examples |

### 2.4 Handling Noisy Retrieval

**This is the normal operating condition.** The system must handle:

| Scenario | Problem | Mitigation |
|----------|---------|------------|
| Relevant not in top-K | Best helper wasn't retrieved | Higher K, better embeddings, hybrid search |
| Irrelevant in top-K | Unrelated helpers retrieved | Planner learns to exclude; `excluded_candidates` in plan |
| Similar but wrong | `taskrun_attestations` vs `pipelinerun_attestations` | Planner discrimination training; validation catches |

**Training must use real (noisy) retrieval:**

```python
# WRONG: Hand-pick "ideal" context
# context = hand_picked_helpers[instruction]

# RIGHT: Run actual retrieval (same index as inference)
retrieved = vector_index.search(embed(instruction), top_k=7)
```

---

## 3) Two-Stage Generation with Hard Constraints

### 3.1 Stage A — Planner (Structured JSON)

**Input:** Instruction + retrieved candidates (as cards)

**Output:** (JSON Schema enforced, versioned)

```json
{
  "$schema": "plan_v1.0",
  "package": "task_bundle_pinning",
  "rule_type": "deny",
  "attestation_type": "slsa_provenance",
  "imports": ["data.lib", "data.lib.tekton"],
  "helpers": [
    {"id": "lib.pipelinerun_attestations", "reason": "Iterate SLSA attestations"},
    {"id": "tekton.tasks", "reason": "Get tasks from attestation"},
    {"id": "tekton.task_ref", "reason": "Check .pinned field"}
  ],
  "schemas": ["slsa_v1_task_bundle", "slsa_v1_task_name"],
  "new_helpers": [],
  "iteration_skeleton": "some att in lib.pipelinerun_attestations; some task in tekton.tasks(att)",
  "condition": "not tekton.task_ref(task).pinned",
  "excluded_candidates": [
    {"id": "sbom.spdx_sboms", "reason": "Wrong attestation type (SBOM, not SLSA)"}
  ]
}
```

### 3.2 Validation Gate (Tooling, Not LLM Self-Check)

Reject plans if:

| Check | What It Catches |
|-------|-----------------|
| Helper existence | Hallucinated helper names |
| Importability | Helpers from wrong module paths |
| Type compatibility | Passing `task` to function expecting `attestation` |
| Schema existence | Non-existent field paths |
| Schema/attestation match | Schema incompatible with selected attestation type |

```python
def validate_plan(plan: dict, kb: KnowledgeBase) -> ValidationResult:
    errors = []
    
    for helper in plan["helpers"]:
        if helper["id"] not in kb.helpers:
            errors.append(f"Helper '{helper['id']}' not found")
            continue
        
        helper_info = kb.helpers[helper["id"]]
        
        if not is_importable(helper_info.module_path, plan["package"]):
            errors.append(f"Helper '{helper['id']}' not importable")
    
    for schema_ref in plan["schemas"]:
        if not kb.schema_exists(schema_ref, plan["attestation_type"]):
            errors.append(f"Schema '{schema_ref}' not found for {plan['attestation_type']}")
    
    return ValidationResult(valid=len(errors) == 0, errors=errors)
```

If invalid: repair loop reruns Planner with explicit alternatives.

### 3.3 Stage B — Codegen (Constrained)

**Input:** Instruction + validated plan + selected helper bodies (full tier)

**Output:**
- Rego rule
- Tests
- Metadata/ruleData keys

**Post-gen constraint enforcement:**

```python
def verify_codegen_compliance(code: str, plan: dict) -> bool:
    # Parse Rego AST
    imports = extract_imports(code)
    data_refs = extract_data_refs(code)
    
    allowed = set(plan["helpers"]) | set(h["name"] for h in plan.get("new_helpers", []))
    
    # Fail if any reference not in plan
    for ref in data_refs:
        if ref not in allowed:
            raise OutOfPlanReferenceError(f"'{ref}' used but not in plan")
    
    return True
```

### 3.4 Component Contracts

**Explicit interfaces between components:**

```
┌─────────────┐     Plan v1.0      ┌─────────────┐
│   Planner   │ ─────────────────► │  Validator  │
└─────────────┘                    └──────┬──────┘
                                          │
                                   Valid? │
                              ┌───────────┴───────────┐
                              │                       │
                              ▼                       ▼
                    ┌─────────────┐          ┌─────────────┐
                    │   Codegen   │          │   Repair    │
                    └──────┬──────┘          │  (Planner)  │
                           │                 └─────────────┘
                           ▼
                    ┌─────────────┐
                    │  Verifier   │
                    │ (AST Check) │
                    └─────────────┘
```

**Repair Boundaries (frozen vs. mutable):**

| Failure Type | Who Repairs | What's Frozen |
|--------------|-------------|---------------|
| `opa fmt` error | Codegen | Plan |
| `opa test` failure | Codegen | Plan |
| Type error in generated code | Codegen | Plan |
| Missing helper in KB | Planner | — |
| Wrong attestation type | Planner | — |
| Schema doesn't exist | Planner | — |
| Iteration produces wrong type | Planner | — |

This prevents repair loops from becoming random walks.

---

## 4) Context Budget Allocator

### 4.1 Token Budget by Component

```python
CONTEXT_BUDGETS = {
    "small_model": {  # 8K context
        "instruction_framing": 600,
        "plan_schema": 400,
        "helpers": 2500,
        "schemas": 1500,
        "idioms": 800,
        "reserve": 500,
    },
    "large_model": {  # 32K+ context
        "instruction_framing": 800,
        "plan_schema": 600,
        "helpers": 8000,
        "schemas": 4000,
        "idioms": 2000,
        "reserve": 1000,
    }
}
```

### 4.2 Progressive Disclosure Strategy

| Stage | Helper Tier | Why |
|-------|-------------|-----|
| Planner | `helper_card` (summaries) | Needs to see many options |
| Codegen | `helper_full` (selected only) | Needs implementation details |

```python
def build_planner_context(retrieved: List[Chunk], budget: int) -> str:
    # Use cards for all candidates
    return format_as_cards(retrieved, max_tokens=budget)

def build_codegen_context(plan: Plan, kb: KnowledgeBase, budget: int) -> str:
    # Use full bodies only for selected helpers
    selected = [kb.get_full(h["id"]) for h in plan["helpers"]]
    return format_full_helpers(selected, max_tokens=budget)
```

### 4.3 Deterministic Truncation Order

When over budget, drop in this order (least useful first):

```python
TRUNCATION_ORDER = [
    "extra_idioms",                    # 1. Drop additional idioms
    "extra_schema_examples",           # 2. Keep only 1 example per schema
    "helper_bodies_to_cards",          # 3. Degrade full → card
    "reduce_K",                        # 4. Fewer candidates, higher quality
    "header_only_chunks",              # 5. Last resort: signatures only
]
```

### 4.4 Full Context Stuffing Option

For small libraries with large context models:

```python
def can_stuff_full_context(kb: KnowledgeBase, model_context: int) -> bool:
    total = (
        len(kb.helpers) * 200 +      # Cards
        len(kb.schemas) * 100 +
        2000                          # Instruction + output
    )
    return total < model_context * 0.5  # Leave headroom

# If True, skip retrieval entirely — just include everything
```

---

## 5) Execution-Based Verification

### 5.1 Verification Pipeline

```python
def verify_output(rule: str, tests: str, fixtures: List[str]) -> VerificationResult:
    # Step 1: Style check
    fmt_result = run_command(f"opa fmt --check {rule_file}")
    if not fmt_result.success:
        return VerificationResult(stage="fmt", error=fmt_result.stderr)
    
    # Step 2: Test execution
    test_result = run_command(f"opa test {rule_file} {test_file}")
    if not test_result.success:
        return VerificationResult(stage="test", error=test_result.stderr)
    
    # Step 3: Optional fixture evaluation
    for fixture in fixtures:
        eval_result = run_command(f"opa eval -d {rule_file} -i {fixture} 'data.policy.deny'")
        # Golden check against expected
    
    return VerificationResult(stage="complete", success=True)
```

### 5.2 Repair Routing

```python
def route_repair(verification: VerificationResult, plan: Plan) -> RepairAction:
    if verification.stage in ["fmt", "test"]:
        # Codegen error → retry codegen with error context
        return RepairAction(
            target="codegen",
            context=verification.error,
            plan=plan,  # Frozen
            max_attempts=2
        )
    
    if "undefined" in verification.error and "helper" in verification.error:
        # Missing helper → rerun planner
        return RepairAction(
            target="planner",
            context=f"Helper not found: {verification.error}",
            alternatives=kb.suggest_alternatives(verification.error)
        )
```

---

## 6) Fine-Tuning Strategy

**Priority order** (tune for pipeline competence, not memorization):

### 6.1 Collect Data First (No Fine-Tuning)

1. Deploy pipeline with strong base model (Claude, GPT-4)
2. Collect real usage: instructions, retrieved context, outputs, failures
3. Build evaluation suite from real cases

### 6.2 Train Repair Model First (Highest ROI)

**Dataset:** `(plan + code + failing error + retrieved snippets) → patch`

This dramatically reduces "stuck" cases and makes the system feel dependable.

### 6.3 Train Planner Second

Dataset must include:

- Real retrieved context (noisy, from actual index)
- Injected confusers (similar-named wrong helpers)
- Supervision for `excluded_candidates` + reasons
- Supervision for "expand retrieval" signals

### 6.4 Style/Idiom Alignment Last

Teach consistent:
- Result formatting
- Metadata blocks
- Test layout
- Naming conventions

> **Key insight:** You don't need heavy "learn Rego from scratch" tuning if you have compile/test gates + idiom retrieval + grounded context.

---

## 7) Developer UX

### 7.1 Output Format

When shipping to other devs, return structured output:

```json
{
  "success": true,
  "rule": "package policy.task_bundle\n\ndeny contains result if {...}",
  "tests": "package policy.task_bundle_test\n\ntest_unpinned_denied {...}",
  "metadata": {
    "rule_data_keys": ["allowed_bundles"],
    "rule_data_example": {"allowed_bundles": ["quay.io/konflux-ci/*"]}
  },
  "plan": { /* structured plan JSON */ },
  "provenance": {
    "kb_version": "git:abc123",
    "helpers_used": [
      {"id": "tekton.task_ref", "source": "policy/lib/tekton/tekton.rego:142-158"}
    ],
    "schemas_used": [
      {"id": "slsa_v1_task_bundle", "example_from": "data/attestations/slsa-example.json"}
    ],
    "retrieved_chunks": {
      "selected": ["tekton.task_ref", "tekton.tasks", "lib.pipelinerun_attestations"],
      "excluded": [
        {"id": "sbom.spdx_sboms", "reason": "Wrong attestation type"}
      ]
    }
  },
  "verification": {
    "opa_fmt": "passed",
    "opa_test": "passed (3/3)",
    "fixtures_evaluated": ["slsa-provenance-pinned.json", "slsa-provenance-unpinned.json"]
  }
}
```

### 7.2 Citations Requirement

Every output must include:

| Citation Type | Content |
|---------------|---------|
| **Helper sources** | File + line range for each helper used |
| **Schema sources** | Schema ID + example fragment reference |
| **Chunk selection** | Which retrieved chunks were selected/excluded + reasons |
| **KB version** | Git ref used for validation |

This is a **trust feature** and debugging accelerant.

### 7.3 CLI/Service Interface

```bash
# CLI usage
rego-assist generate \
  --instruction "Deny if task bundle not pinned" \
  --kb-version main \
  --output-format json

# Service API
POST /api/generate
{
  "instruction": "Deny if task bundle not pinned",
  "attestation_type": "slsa_provenance",
  "fixtures": ["slsa-example.json"]
}
```

---

## 8) Continuous Evaluation

### 8.1 Metrics to Track

| Metric | Target | Description |
|--------|--------|-------------|
| **Compile rate** | >95% | Generated code passes `opa fmt` |
| **Test pass rate** | >85% | Generated tests pass |
| **Retrieval recall@10** | >90% | Right helper in top 10 |
| **Plan validity rate** | >95% | Plans pass validation gate |
| **Out-of-plan reference rate** | <2% | Hallucination metric |
| **Repair success rate** | >90% | Failures fixed within 2 retries |
| **Median repair iterations** | <1.5 | Efficiency metric |

### 8.2 Release Gates

Gate releases on:

- All metrics above threshold
- Regression suite pass rate (from real policies + edge cases)
- No new hallucinated helper patterns

---

## 9) Extensibility

### 9.1 Adding New Attestation Types

No model retraining required for basic retrieval:

```bash
# Step 1: Add example files
cp new-attestation.json data/attestations/sigstore-example-1.json

# Step 2: Extract schemas (auto-discovers new files)
python scripts/build_knowledge_base.py --extract-schemas

# Step 3: Enrich with LLM
python scripts/build_knowledge_base.py --enrich --attestation-type sigstore

# Step 4: Rebuild index
python scripts/build_vector_index.py --rebuild

# Step 5: (Optional) Generate training examples
python scripts/generate_rag_training.py --include-type sigstore
```

### 9.2 Adding New Helpers

```bash
# Helpers are auto-discovered from policy/lib/
# After adding new helper:
python scripts/build_knowledge_base.py --extract-helpers --enrich
python scripts/build_vector_index.py --rebuild
```

---

## 10) MVP Scope

For the tightest first release:

### 10.1 Policy Coverage

| Priority | Policies | Count |
|----------|----------|-------|
| P0 | Task bundle pinning, required tasks | 2 |
| P1 | Allowed licenses, banned packages, provenance constraints | 3 |
| P2 | Required labels, image signing, SBOM presence | 3 |
| P3 | Custom org policies | 2-10 |

### 10.2 Fixtures Per Policy

- 5-10 fixtures per policy
- Include adversarial variants (edge cases, malformed input)
- Golden expected outputs for each

### 10.3 MVP Architecture

```
Week 1-2: Foundation
├── Build evaluation suite (50+ test cases)
├── Extract helpers + schemas (simple JSON)
└── Test with base model + context stuffing

Week 3-4: Retrieval (if needed)
├── Vector search (start simple)
└── Measure retrieval recall@10

Week 5-6: Verification loop
├── opa fmt + opa test gates
├── Simple repair (2 retries)
└── Measure compile/test rates

Week 7-8: Production hardening
├── Add structured plans
├── Add provenance/citations
└── CLI wrapper

Post-MVP: Fine-tuning
├── Collect real failures
├── Fine-tune repair model
└── Add complexity based on data
```

---

## Appendix A: File Structure

```
scripts/
  build_knowledge_base.py     # Extract and enrich library code
  build_vector_index.py       # Create embeddings and index
  generate_rag_training.py    # Generate training data with real retrieval

data/
  attestations/               # Example attestation files
    slsa-provenance-*.json
    spdx-*.json
    cyclonedx-*.json
  knowledge_base/
    manifest.yaml             # KB version info
    helpers.jsonl             # Enriched helper metadata
    helpers_full.jsonl        # Full helper bodies
    schemas/
      slsa_provenance.jsonl
      spdx_sbom.jsonl
      cyclonedx_sbom.jsonl
    idioms.jsonl
    index/                    # Vector + BM25 indexes

src/
  infer_rag.py               # RAG-based inference
  context_allocator.py       # Token budget management
  plan_validator.py          # Validation gate
  codegen_verifier.py        # Post-gen AST checks
```

---

## Appendix B: Implementation Todos

### Phase 1: Knowledge Base Extraction

| ID | Task | Dependencies | Status |
|----|------|--------------|--------|
| `extract-helpers` | Extract helpers with AST parser | - | Pending |
| `two-tier-chunks` | Create card + full variants | `extract-helpers` | Pending |
| `mine-usage` | Mine usage spans from policy/release/ | `extract-helpers` | Pending |
| `extract-schemas` | Extract schemas from all attestation types | - | Pending |
| `canonical-ids` | Normalize schemas to canonical IDs | `extract-schemas` | Pending |
| `llm-enrich` | Build LLM enrichment with grounding | `extract-helpers`, `extract-schemas` | Pending |
| `kb-versioning` | Add manifest with git ref | `llm-enrich` | Pending |

### Phase 2: Indexing

| ID | Task | Dependencies | Status |
|----|------|--------------|--------|
| `index-helpers` | Build helper index (BM25 + vector) | `llm-enrich` | Pending |
| `index-schemas` | Build schema index | `llm-enrich` | Pending |
| `reranker` | Integrate cross-encoder reranker | `index-*` | Pending |
| `hybrid-retriever` | Build hybrid retriever with caps | `reranker` | Pending |

### Phase 3: Inference Pipeline

| ID | Task | Dependencies | Status |
|----|------|--------------|--------|
| `context-allocator` | Token budget + progressive disclosure | `hybrid-retriever` | Pending |
| `planner` | Stage 1 with structured JSON output | `context-allocator` | Pending |
| `validator` | Validation gate with repair routing | `planner` | Pending |
| `codegen` | Stage 2 with plan constraint | `validator` | Pending |
| `ast-verifier` | Post-gen out-of-plan reference check | `codegen` | Pending |
| `execution-verify` | opa fmt + opa test integration | `ast-verifier` | Pending |

### Phase 4: UX & Deployment

| ID | Task | Dependencies | Status |
|----|------|--------------|--------|
| `provenance` | Add citations to output | `execution-verify` | Pending |
| `cli` | CLI wrapper | `provenance` | Pending |
| `evaluation-suite` | Build test cases + metrics | `cli` | Pending |

---

## Appendix C: Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` or `bge-small-en-v1.5` | Fast, good accuracy |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast, good quality |
| Vector store | FAISS | Simple, no external deps |
| BM25 | `rank_bm25` or Elasticsearch | Depends on scale |
| Plan schema | JSON Schema, versioned | Explicit contracts |
| Chunk format | Two-tier (card/full) | Progressive disclosure |
| Top-K | 7-10 with per-type caps | Balanced context |
