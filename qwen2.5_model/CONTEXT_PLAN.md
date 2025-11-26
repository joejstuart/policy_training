# Dynamic Context Generation Plan

## Goal
Instead of hardcoding helper descriptions, provide the model with:
1. **Access to helper signatures and usage examples** - Show how helpers are actually used, not full implementations
2. **Dynamic context building** - Extract only relevant helpers based on instruction keywords
3. **Strict token budgets** - Keep context minimal (400-600 tokens) for MacBook-friendly inference
4. **Call-site centric** - Model learns to call helpers, not rewrite them

## Current Problem
- Hardcoded `HELPER_DESCRIPTIONS` dictionary
- Static context building that guesses helpers
- No access to actual helper signatures or usage patterns
- Context can grow unbounded, overwhelming small models
- Model may hallucinate helpers that don't exist

## Existing Foundation
The `pipelinerun_model/context_extractor.py` already has:
- `extract_function()` - Extracts actual function code from source
- `extract_sections()` - Extracts multiple functions + METADATA
- `get_tekton_context()` - Gets focused tekton library context
- `get_attestation_context()` - Gets attestation library context

**We'll build on this pattern but make it dynamic and general.**

## Proposed Solution

### Phase 1: Library Discovery System

#### 1.1 Library Structure Mapper
Create `library_mapper.py` that:
- Maps import prefixes to directory patterns (keep it simple):
  - `data.lib` → `policy/lib/*.rego`
  - `data.lib.tekton` → `policy/lib/tekton/*.rego`
  - `data.lib.image` → `policy/lib/image/*.rego`
  - `data.release.lib.*` → `policy/release/lib/*.rego`
- Provides reverse lookup: given a file path, what import prefix does it correspond to?
- **Note**: Map packages/import prefixes → directories, not every single file to every import
- Reverse mapping is primarily for indexer and debugging, not for model context

#### 1.2 Library Indexer
Create `library_indexer.py` that:
- Scans all library files in `policy/lib/**` and `policy/release/lib/**`
- Extracts all public functions (not starting with `_`)
- For each function, records:
  - Function name and package
  - File path
  - One-line signature string
  - First comment block above function (short docstring)
  - **Usage examples**: Scan `policy/release/**` for call sites (e.g., `tekton.tasks(...)`) and store 1-2 real usage snippets
- Builds searchable index: `{function_name: {package, file, signature, doc, usage_examples}}`
- Can be queried by:
  - Function name (exact match)
  - Keyword matching (e.g., "task" → finds `task_name`, `tasks`, `task_result`)
  - Package/module (e.g., all functions in `tekton` module)

#### 1.3 Smart Context Builder
Create `smart_context_builder.py` that:
- Takes instruction + package as input
- Uses keyword extraction from instruction to find relevant helpers
- **Strict token budget**: Enforces 400-600 token limit total
- Selects top K helpers (e.g., 3-5 most relevant) based on keyword matching
- Builds minimal context with:
  - Package declaration
  - Core imports (rego.v1, data.lib, etc.)
  - Only imports for libraries that have selected helpers
  - For each helper: signature + short doc + usage example (NOT full implementation)
- **No discovery instructions**: Model only sees helpers already selected by the system

### Phase 2: Implementation

#### 2.1 Library Mapper (`library_mapper.py`)
```python
class LibraryMapper:
    """Maps import prefixes to directory patterns and vice versa."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.lib_dir = repo_root / "policy" / "lib"
        self.release_lib_dir = repo_root / "policy" / "release" / "lib"
        self.import_to_dir = {}  # "data.lib.tekton" -> Path("policy/lib/tekton")
        self.dir_to_import = {}  # Reverse mapping
    
    def build_mappings(self):
        """Scan library directories and build import prefix ↔ directory mappings."""
        # Map packages and import prefixes → directories, not every file
        # Scan policy/lib/** and map to data.lib.* imports
        # Scan policy/release/lib/** and map to data.release.lib.* imports
        # Handle subdirectories (e.g., tekton/ → data.lib.tekton)
    
    def get_library_dir(self, import_path: str) -> Path:
        """Get directory for an import prefix."""
        # e.g., "data.lib.tekton" -> Path("policy/lib/tekton")
    
    def get_import_prefix(self, file_path: Path) -> str:
        """Get import prefix for a library file."""
        # e.g., Path("policy/lib/tekton/task.rego") -> "data.lib.tekton"
        # Used by indexer and for debugging, not shown to model
```

#### 2.2 Library Indexer (`library_indexer.py`)
```python
class LibraryIndexer:
    """Indexes all library functions for search."""
    
    def __init__(self, repo_root: Path, mapper: LibraryMapper):
        self.repo_root = repo_root
        self.mapper = mapper
        self.index = {}  # function_name -> {package, file, signature, doc, usage_examples}
    
    def index_all_libraries(self):
        """Scan all library files and build searchable index."""
        # For each .rego file in policy/lib/** and policy/release/lib/**
        #   - Extract all public functions (not starting with _)
        #   - Store: name, file path, package, one-line signature, first comment/doc
        #   - Scan policy/release/** for usage sites (e.g., tekton.tasks(...))
        #   - Store 1-2 usage examples per helper from real rules
        #   - Build keyword index for search
    
    def find_relevant_helpers(self, instruction: str, package: str = None, max_results: int = 5) -> List[HelperInfo]:
        """Find helpers relevant to instruction using keyword matching."""
        # Extract keywords from instruction
        # Match against function names, package names, keywords
        # Rank by relevance
        # Return top N most relevant helpers (default: 5)
    
    def get_helper_context(self, helper_name: str) -> str:
        """Get context for a helper: signature + doc + usage example."""
        # Return formatted string with:
        #   - Signature (one line)
        #   - Short docstring (if available)
        #   - One usage example from a real rule
        # NOT the full function implementation
```

#### 2.3 Smart Context Builder (`smart_context_builder.py`)
```python
class SmartContextBuilder:
    """Builds context dynamically from helper signatures and usage examples."""
    
    def __init__(self, indexer: LibraryIndexer, mapper: LibraryMapper, max_tokens: int = 500):
        self.indexer = indexer
        self.mapper = mapper
        self.max_tokens = max_tokens  # Strict token budget
    
    def build_context(self, instruction: str, package: str = None) -> str:
        """Build context dynamically with strict token budget."""
        # 1. Always start with:
        #    - package declaration
        #    - core imports (rego.v1, data.lib, etc.)
        # 2. Extract keywords from instruction
        # 3. Find top N helpers (default: 3-5) using indexer.find_relevant_helpers()
        # 4. Group helpers by import path
        # 5. For each helper, include:
        #    - One-line signature / doc
        #    - One usage example from a real rule (if available)
        # 6. Stop once token budget is reached
        # 7. NO instructions to "find more helpers" - model only sees what's selected
```

#### 2.4 Integration with Inference
Update `infer_policy.py` to:
- Initialize `LibraryMapper` and `LibraryIndexer` on startup
- Use `SmartContextBuilder` instead of hardcoded descriptions
- Pass helper signatures + usage examples in context (NOT full implementations)
- Enforce strict token budgets (400-600 tokens)
- System prompt: "Before writing Rego, briefly plan the approach in your head. Then output only the final Rego code."

### Phase 3: Context Format

#### New Context Format
```
package tasks

import rego.v1
import data.lib
import data.lib.tekton

# Relevant helpers (signatures & usage):

# data.lib.tekton.tasks(obj) -> set[task]
# Used like:
#   some task in tekton.tasks(pipelinerun)

# data.lib.tekton.task_name(task) -> string
# Used like:
#   name := tekton.task_name(task)

# data.lib.pipelinerun_attestations -> array[attestation]
# Used like:
#   some att in pipelinerun_attestations

# Use only the helpers and imports provided in this context.
# Do not invent new modules or helper functions; if a needed helper is missing,
# add a TODO comment instead of creating a new import.
```

### Phase 4: Benefits

1. **Generalization**: Model sees helper signatures and usage patterns, not hardcoded descriptions
2. **Call-site learning**: Model learns to call helpers correctly by seeing real usage examples
3. **Accuracy**: Signatures + usage examples prevent hallucination better than full implementations
4. **Maintainability**: No need to update descriptions when libraries change; indexer scans code automatically
5. **Token efficiency**: Strict budgets (400-600 tokens) keep context manageable for small models
6. **Hallucination resistance**: Model only sees pre-selected helpers; no "discover more" instructions

### Phase 5: Implementation Steps

1. ✅ Create `library_indexer.py` - Scan and index all library files
2. ✅ Create `smart_context_builder.py` - Build context from index
3. ✅ Update `infer_policy.py` - Use new system
4. ✅ Add library structure documentation to context
5. ✅ Test with various instructions to ensure it finds right helpers

### Phase 6: Advanced Features (Future)

- Semantic search (embedding-based) instead of keyword matching
- Caching frequently used helpers
- Learning from user corrections
- Multi-file helper extraction (when helpers span files)

## Example Usage

```python
# In infer_policy.py
mapper = LibraryMapper(repo_root)
mapper.build_mappings()

indexer = LibraryIndexer(repo_root, mapper)
indexer.index_all_libraries()  # Indexes signatures + usage examples

builder = SmartContextBuilder(indexer, mapper, max_tokens=500)
context = builder.build_context(
    "Write a rule that checks if all tasks succeeded",
    package="tasks"
)

# Context now contains:
# - Package + imports
# - Top 3-5 relevant helper signatures + usage examples
# - Total context size: ~400-600 tokens
# Model learns to call helpers correctly, not copy/paste implementations
```

## Files to Create

1. `qwen2.5_model/library_mapper.py` - Map import prefixes to directories
2. `qwen2.5_model/library_indexer.py` - Index function signatures + usage examples
3. `qwen2.5_model/smart_context_builder.py` - Build context with strict token budgets
4. `qwen2.5_model/context_extractor.py` - Extract usage examples from call sites (reuse pattern from pipelinerun_model)

## Files to Update

1. `qwen2.5_model/infer_policy.py` - Use new dynamic system
2. `qwen2.5_model/README.md` - Document new system
3. `qwen2.5_model/inference_helper.py` - Can be enhanced or replaced

## Implementation Order

1. **Step 1**: Create `context_extractor.py` (reuse/extend pattern from pipelinerun_model)
   - `extract_usage_sites()` - Extract call sites from rules (e.g., `tekton.tasks(...)`)
   - `extract_signature()` - Extract function signature + docstring (one-liner)
   - Handle Rego syntax properly
   - **Note**: NOT for extracting full function bodies - only signatures and usage examples

2. **Step 2**: Create `library_mapper.py`
   - Build import path ↔ file path mappings
   - Handle subdirectories and aliases

3. **Step 3**: Create `library_indexer.py`
   - Scan all library files
   - Extract all public functions (signatures + docstrings)
   - Scan policy/release/** for usage sites
   - Store 1-2 usage examples per helper
   - Build searchable index

4. **Step 4**: Create `smart_context_builder.py`
   - Keyword extraction from instructions
   - Helper selection using indexer (top 3-5 most relevant)
   - Context assembly with signatures + usage examples (NOT full implementations)
   - Token budget enforcement (400-600 tokens max)
   - No "discover more helpers" instructions

5. **Step 5**: Update `infer_policy.py`
   - Initialize mapper/indexer/builder
   - Replace hardcoded context with dynamic building
   - Add library structure docs to context

## Key Design Decisions

1. **Show signatures + usage, not full implementations** - Model learns to call helpers, not copy/paste logic
2. **Strict token budgets** - Enforce 400-600 token limit to keep context manageable
3. **Call-site centric** - Include real usage examples from actual rules, not just function definitions
4. **No discovery instructions** - Model only sees helpers pre-selected by the system; prevents hallucination
5. **Minimal context** - Only include top 3-5 most relevant helpers (not entire libraries)
6. **Keyword-based selection** - Simple but effective for finding relevant helpers
7. **Extensible** - Easy to add semantic search later if needed
8. **Internal reasoning** - Encourage model to think through the problem, but output only code by default

## Chain-of-Thought Reasoning

The model should think through the problem internally before generating code:

1. **Understand the instruction** - What is the user asking for?
2. **Identify relevant helpers** - Which helpers from the provided context are needed?
3. **Plan the rule structure** - What pattern should be used?
4. **Check helper availability** - Are the needed helpers in the context?
5. **Generate the rule** - Write the actual Rego code

**Implementation approach:**
- System prompt encourages internal reasoning: "Before writing Rego, briefly plan the approach in your head. Then output only the final Rego code."
- **No explicit "Thought:" or "Plan:" sections** in output by default (keeps token usage low)
- Focus on context quality first; can add light "plan in bullets first, then code" later if needed
- For a 1.5B model, implicit reasoning is more efficient than verbose CoT output

## Summary: Key Refinements

This plan has been refined to be MacBook-friendly and hallucination-resistant:

### ✅ What's Great (Keep)
- **Dynamic, code-driven context** - Uses actual .rego code as source of truth
- **Clear separation of responsibilities** - LibraryMapper, LibraryIndexer, SmartContextBuilder
- **Minimal-context intent** - Only relevant helpers, not entire libraries
- **Planning for future semantic search** - Keyword search is fine for v1

### 🔧 Key Refinements Made
1. **Signatures + usage examples, NOT full implementations**
   - Model learns to call helpers correctly by seeing real usage patterns
   - Prevents copy/paste of helper logic into generated rules
   - Much more token-efficient

2. **Strict token budgets (400-600 tokens)**
   - Hard limits prevent context from growing unbounded
   - Top 3-5 most relevant helpers only
   - MacBook-friendly for small models

3. **No "discover more helpers" instructions**
   - Model only sees helpers pre-selected by the system
   - Prevents hallucination of non-existent helpers
   - Clear instruction: "Use only helpers provided; add TODO if missing"

4. **Call-site centric approach**
   - Indexer scans `policy/release/**` for real usage examples
   - Shows how helpers are actually used in rules
   - Better than showing full function bodies

5. **Implicit chain-of-thought**
   - Internal reasoning encouraged, but no verbose output
   - Focus on context quality before reasoning verbosity
   - Token-efficient for 1.5B models

6. **Simplified LibraryMapper**
   - Maps import prefixes → directories, not every file
   - Reverse mapping for indexer/debugging, not model context

