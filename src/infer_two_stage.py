#!/usr/bin/env python3
"""
Two-stage inference for Rego policy generation.

Stage 1: Natural language → Context (schema, helpers, rule_data_keys)
Stage 2: Context + requirements → Rule + tests

Usage:
    # Full two-stage pipeline with fine-tuned Stage 1 model
    python src/infer_two_stage.py \
        --stage1-model models/stage1-context-inference \
        --stage2-model models/stage2-rule-generation \
        --instruction "Check that all pipeline tasks succeeded"

    # RAG mode: Use knowledge base retrieval instead of Stage 1 model
    python src/infer_two_stage.py \
        --use-rag \
        --stage2-model models/stage2-rule-generation \
        --instruction "Check that task bundles are pinned"

    # RAG mode with custom KB directory
    python src/infer_two_stage.py \
        --use-rag --kb-dir data/knowledge_base \
        --stage2-model models/stage2-rule-generation \
        --instruction "Verify SBOM contains required packages"

    # Stage 1 only (get context)
    python src/infer_two_stage.py \
        --stage1-model models/stage1-context-inference \
        --stage 1 \
        --instruction "Verify SBOM contains required packages"

    # Stage 2 only (provide context)
    python src/infer_two_stage.py \
        --stage2-model models/stage2-rule-generation \
        --stage 2 \
        --instruction "Check tasks succeeded" \
        --context "ATTESTATION_SCHEMA:\n- .statement.predicate..."
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Set environment before torch import
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try to import PEFT for LoRA support
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Try to import RAG components
try:
    from knowledge_base import KnowledgeBase
    from hybrid_retriever import HybridRetriever
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


class OllamaGenerator:
    """Generate responses using Ollama API."""
    
    def __init__(self, model: str = "qwen3-coder:30b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._available = None
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if self._available is not None:
            return self._available
        
        import urllib.request
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self._available = resp.status == 200
        except:
            self._available = False
        
        return self._available
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> str:
        """Generate completion from Ollama."""
        import json
        import urllib.request
        
        # Build messages for chat API
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        }
        
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(data).encode('utf-8'),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result.get("message", {}).get("content", "")
        except Exception as e:
            print(f"Ollama error: {e}")
            return ""


class RAGContextRetriever:
    """Retrieve context from Knowledge Base using hybrid search."""
    
    def __init__(self, kb_dir: Path):
        if not RAG_AVAILABLE:
            raise RuntimeError(
                "RAG components not available. "
                "Install with: uv pip install sentence-transformers faiss-cpu rank-bm25"
            )
        
        self.kb_dir = Path(kb_dir)
        self.kb = None
        self.retriever = None
        self._loaded = False
    
    def load(self):
        """Load KB and retriever indexes."""
        if self._loaded:
            return
        
        print(f"Loading Knowledge Base from {self.kb_dir}...")
        self.kb = KnowledgeBase(self.kb_dir)
        print(f"  Loaded {len(self.kb.helper_cards)} helpers, {len(self.kb.schemas)} schemas")
        
        print("Loading retriever indexes...")
        self.retriever = HybridRetriever.from_kb_dir(self.kb_dir)
        print("  ✓ Retriever ready")
        
        self._loaded = True
    
    def retrieve_context(
        self,
        query: str,
        top_k_helpers: int = 7,
        top_k_schemas: int = 3,
    ) -> str:
        """Retrieve context formatted for Stage 2.
        
        Uses multi-faceted retrieval:
        1. Domain helpers (task/sbom/image specific)
        2. Boilerplate helpers (result_helper, rule_data, attestations)
        """
        self.load()
        
        # Multi-faceted retrieval
        results = self._multi_facet_retrieve(query, top_k_helpers, top_k_schemas)
        
        # Format as Stage 1-compatible context
        return self._format_context(results, query)
    
    def _multi_facet_retrieve(self, query: str, helper_k: int, schema_k: int):
        """Retrieve helpers using multiple query facets.
        
        Splits retrieval into domain-specific and boilerplate queries
        to ensure we get both specialized and common helpers.
        
        Includes validation gate to filter helpers by detected attestation type.
        """
        from hybrid_retriever import RetrievalResult
        
        query_lower = query.lower()
        
        # Facet 1: Domain-specific helpers (main query)
        domain_results = self.retriever.retrieve(
            query=query,
            helper_k=helper_k - 2,  # Reserve space for boilerplate
            schema_k=schema_k,
        )
        
        # Detect attestation domain from schemas
        detected_domain = self._detect_domain(domain_results.schemas, query_lower)
        
        # Facet 2: Boilerplate helpers for deny rules
        boilerplate_query = "deny rule result helper create violation output metadata"
        boilerplate_results = self.retriever.retrieve(
            query=boilerplate_query,
            helper_k=3,
            schema_k=0,
        )
        
        # Facet 3: Data access helpers (attestations, rule_data)
        access_query = "attestations pipelinerun rule_data configuration access"
        access_results = self.retriever.retrieve(
            query=access_query,
            helper_k=2,
            schema_k=0,
        )
        
        # Merge and dedupe helpers, with domain filtering
        seen_ids = set()
        merged_helpers = []
        
        for h in domain_results.helpers:
            hid = h.get("id", "")
            if hid not in seen_ids and self._helper_matches_domain(hid, detected_domain):
                seen_ids.add(hid)
                merged_helpers.append(h)
        
        for h in boilerplate_results.helpers + access_results.helpers:
            hid = h.get("id", "")
            if hid not in seen_ids and self._helper_matches_domain(hid, detected_domain):
                seen_ids.add(hid)
                merged_helpers.append(h)
        
        # Cap to requested size
        merged_helpers = merged_helpers[:helper_k]
        
        return RetrievalResult(
            helpers=merged_helpers,
            schemas=domain_results.schemas,
        )
    
    def _detect_domain(self, schemas: list, query_lower: str) -> str:
        """Detect the attestation domain from retrieved schemas and query.
        
        Returns: 'slsa', 'sbom', or 'any'
        """
        # Check query keywords first
        sbom_keywords = ['sbom', 'package', 'component', 'license', 'purl', 'cyclonedx', 'spdx']
        if any(kw in query_lower for kw in sbom_keywords):
            return 'sbom'
        
        # SLSA/provenance keywords (includes CVE since scans are in pipeline results)
        slsa_keywords = ['task', 'pipeline', 'attestation', 'bundle', 'pinned', 'material', 
                        'result', 'cve', 'vulnerability', 'scan', 'image', 'digest', 'git']
        if any(kw in query_lower for kw in slsa_keywords):
            return 'slsa'
        
        # Fall back to schema types
        schema_types = set(s.get('attestation_type', '') for s in schemas)
        
        if all('sbom' in t for t in schema_types if t):
            return 'sbom'
        if all('slsa' in t or 'provenance' in t for t in schema_types if t):
            return 'slsa'
        
        return 'any'
    
    def _helper_matches_domain(self, helper_id: str, domain: str) -> bool:
        """Check if a helper matches the detected domain.
        
        Filters out cross-domain helpers (e.g., SBOM helpers for SLSA queries).
        """
        if domain == 'any':
            return True
        
        # SBOM helpers should only appear for SBOM queries
        sbom_modules = ['lib.sbom', 'sbom.']
        is_sbom_helper = any(helper_id.startswith(m) or f'.{m}' in helper_id for m in sbom_modules)
        
        if domain == 'slsa' and is_sbom_helper:
            return False
        
        # Tekton helpers should only appear for SLSA queries
        tekton_modules = ['lib.tekton', 'tekton.']
        is_tekton_helper = any(helper_id.startswith(m) or f'.{m}' in helper_id for m in tekton_modules)
        
        if domain == 'sbom' and is_tekton_helper:
            return False
        
        return True
    
    def _format_context(self, results, query: str) -> str:
        """Format retrieval results as Stage 1-style context.
        
        Includes ranking hints so the model knows to prioritize top results.
        
        Args:
            results: RetrievalResult from HybridRetriever
            query: Original query
        """
        parts = []
        
        # Add ranking guidance at the top
        parts.append("NOTE: Items are ranked by relevance. Prioritize items marked [1] over [2], etc.")
        parts.append("")
        
        # ATTESTATION_SCHEMA section (from retrieved schemas)
        schema_results = results.schemas
        if schema_results:
            parts.append("ATTESTATION_SCHEMA:")
            att_types = set()
            for rank, item in enumerate(schema_results, start=1):
                schema_id = item.get("schema_id") or item.get("id", "")
                schema = self.kb.get_schema(schema_id)
                if schema:
                    # Convert JSONPath [*] to Rego iteration hint
                    rego_path = self._jsonpath_to_rego_hint(schema.canonical_path)
                    # Add rank indicator - top item gets emphasis
                    if rank == 1:
                        parts.append(f"[{rank}] {rego_path}  <- MOST RELEVANT")
                    else:
                        parts.append(f"[{rank}] {rego_path}")
                    att_types.add(schema.attestation_type)
            parts.append("")
            
            # Note attestation type from schemas
            if att_types:
                parts.insert(2, f"- Attestation type: {', '.join(att_types)}")
        
        # AVAILABLE_HELPERS section
        helper_results = results.helpers
        if helper_results:
            # Derive imports from retrieved helpers
            imports_needed = set()
            
            for item in helper_results:
                helper_id = item.get("id", "")
                module_prefix = self._get_module_prefix(helper_id)
                
                # Build import from module prefix
                if module_prefix == "lib":
                    imports_needed.add("import data.lib")
                elif module_prefix:
                    imports_needed.add(f"import data.lib.{module_prefix}")
            
            if imports_needed:
                parts.append("REQUIRED_IMPORTS:")
                for imp in sorted(imports_needed):
                    parts.append(f"- {imp}")
                parts.append("")
            
            parts.append("AVAILABLE_HELPERS (ranked by relevance):")
            
            for rank, item in enumerate(helper_results, start=1):
                helper_id = item.get("id", "")
                helper = self.kb.get_helper_card(helper_id)
                if helper:
                    # Extract module prefix from helper_id
                    module_prefix = self._get_module_prefix(helper_id)
                    
                    # Format: module.name(args) -- description
                    base_sig = helper.signature or helper.name
                    if module_prefix and not base_sig.startswith(module_prefix):
                        full_sig = f"{module_prefix}.{base_sig}"
                    else:
                        full_sig = base_sig
                    
                    desc = helper.description if helper.description else ""
                    # Top 2 helpers get emphasis
                    if rank <= 2:
                        marker = f"[{rank}]"
                        if desc:
                            parts.append(f"{marker} {full_sig} -- {desc}")
                        else:
                            parts.append(f"{marker} {full_sig}")
                    else:
                        if desc:
                            parts.append(f"[{rank}] {full_sig} -- {desc}")
                        else:
                            parts.append(f"[{rank}] {full_sig}")
            parts.append("")
        
        # RULE_DATA_KEYS section (infer from query and helpers)
        rule_data_keys = self._infer_rule_data_keys(query, helper_results)
        if rule_data_keys:
            parts.append("RULE_DATA_KEYS:")
            for key in rule_data_keys:
                parts.append(f"- {key}")
            parts.append("")
        
        # SUGGESTED_PACKAGE and SUGGESTED_RULE_TYPE
        package, rule_type = self._infer_package_and_type(query, helper_results)
        parts.append(f"SUGGESTED_PACKAGE: {package}")
        parts.append(f"SUGGESTED_RULE_TYPE: {rule_type}")
        
        return "\n".join(parts)
    
    def _get_module_prefix(self, helper_id: str) -> str:
        """Extract module prefix from helper ID.
        
        Examples:
            "lib.tekton.bundle" -> "tekton"
            "lib.sbom.spdx_sboms" -> "sbom"
            "lib.result_helper" -> "lib"
            "lib.image.parse" -> "image"
        """
        parts = helper_id.split('.')
        
        # Pattern: lib.module.function or lib.function
        if len(parts) >= 3 and parts[0] == "lib":
            # lib.tekton.bundle -> tekton
            return parts[1]
        elif len(parts) == 2 and parts[0] == "lib":
            # lib.result_helper -> lib
            return "lib"
        
        return ""
    
    def _jsonpath_to_rego_hint(self, jsonpath: str) -> str:
        """Convert JSONPath notation to Rego-friendly path description.
        
        JSONPath uses [*] for array iteration, but Rego uses "some X in array".
        
        Example:
            $.predicate.buildConfig.tasks[*].ref.bundle
            → .predicate.buildConfig.tasks[].ref.bundle (iterate with: some task in tasks)
        """
        import re
        
        # Remove leading $. if present
        path = jsonpath.lstrip('$').lstrip('.')
        
        # Find array patterns and convert to Rego hints
        # Pattern: field[*] or field[0] etc.
        array_pattern = r'(\w+)\[\*\]'
        
        # Replace [*] with [] and note the iteration variable
        iteration_vars = []
        
        def replace_array(match):
            field_name = match.group(1)
            # Suggest singular form for iteration variable
            if field_name.endswith('ies'):
                var_name = field_name[:-3] + 'y'  # entries -> entry
            elif field_name.endswith('es'):
                var_name = field_name[:-2]  # boxes -> box
            elif field_name.endswith('s'):
                var_name = field_name[:-1]  # tasks -> task
            else:
                var_name = field_name
            iteration_vars.append((field_name, var_name))
            return f"{field_name}[]"
        
        converted_path = re.sub(array_pattern, replace_array, path)
        
        # Build Rego access pattern
        if iteration_vars:
            # Show how to iterate in Rego
            iterations = []
            for field, var in iteration_vars:
                iterations.append(f"some {var} in {field}")
            rego_hint = f".{converted_path} (Rego: {'; '.join(iterations)})"
            return rego_hint
        
        return f".{converted_path}"
    
    def _infer_rule_data_keys(self, query: str, helper_results: List[Dict]) -> List[str]:
        """Infer rule_data keys from retrieved helpers.
        
        Extracts rule_data keys mentioned in helper descriptions/use_when.
        """
        keys = set()
        
        # Extract from helper metadata
        for item in helper_results:
            # Check description for rule_data mentions
            desc = item.get("description", "").lower()
            use_when = item.get("use_when", [])
            combined = desc + " ".join(use_when).lower()
            
            # Look for rule_data key patterns
            if "allowed_" in combined or "restrict_" in combined or "warn_" in combined:
                # Extract potential key names
                import re
                matches = re.findall(r'(allowed_\w+|restrict_\w+|warn_\w+|required_\w+)', combined)
                keys.update(matches)
        
        return list(keys)[:5]
    
    def _infer_package_and_type(
        self, 
        query: str, 
        helper_results: List[Dict]
    ) -> Tuple[str, str]:
        """Infer package name and rule type from query and retrieved helpers."""
        query_lower = query.lower()
        
        # Detect rule type from query intent
        rule_type = "deny"
        if "warn" in query_lower:
            rule_type = "warn"
        
        # Infer package from retrieved helper modules
        modules = set()
        for item in helper_results:
            helper_id = item.get("id", "")
            prefix = self._get_module_prefix(helper_id)
            if prefix and prefix != "lib":
                modules.add(prefix)
        
        # Use most common module as package hint
        if modules:
            primary_module = list(modules)[0]
            package = f"policy.release.{primary_module}"
        else:
            package = "policy.release"
        
        return package, rule_type


class TwoStageGenerator:
    """Two-stage Rego policy generator."""
    
    # System prompt - MUST MATCH training exactly (train_policy.py)
    SYSTEM_PROMPT = (
        "You are an expert Rego policy assistant. "
        "Follow the instructions carefully and provide accurate, well-structured responses."
    )
    
    # Fixed instruction for Stage 1 (model trained with this)
    STAGE1_INPUT_PROMPT = "Analyze the requirements and identify the attestation schema, available helpers, rule data keys, and suggest an appropriate package name and rule type (deny/warn) for this Rego rule."
    
    # Fixed instruction for Stage 2 (model trained with this)
    STAGE2_INSTRUCTION = "Write a Rego rule that enforces the requirements below using the provided context."
    
    # Optional: Pattern reminder for Stage 2 (appended after context)
    STAGE2_PATTERN_REMINDER = """
Output format:
ANALYSIS: Brief explanation of approach
RULE: Complete Rego code (package, imports, helpers, METADATA, deny/warn rule)
TESTS: Test functions with _mock fixtures for pass/fail cases"""
    
    def __init__(
        self,
        stage1_model_path: Optional[str] = None,
        stage2_model_path: Optional[str] = None,
        base_model: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "auto",
        rag_retriever: Optional[RAGContextRetriever] = None,
        ollama_model: Optional[str] = None,
    ):
        self.device = self._detect_device(device)
        print(f"Using device: {self.device}")
        
        self.stage1_model = None
        self.stage1_tokenizer = None
        self.stage2_model = None
        self.stage2_tokenizer = None
        self.rag_retriever = rag_retriever
        self.ollama = None
        
        # Use Ollama for Stage 2 if specified
        if ollama_model:
            print(f"Using Ollama model: {ollama_model}")
            self.ollama = OllamaGenerator(model=ollama_model)
            if not self.ollama.is_available():
                raise RuntimeError(
                    f"Ollama not available. Make sure it's running: ollama serve"
                )
            print("  ✓ Ollama ready")
        
        # Load Stage 1 model if path provided
        if stage1_model_path:
            print(f"Loading Stage 1 model from {stage1_model_path}...")
            self.stage1_tokenizer, self.stage1_model = self._load_model(
                stage1_model_path, base_model
            )
        
        # Load Stage 2 model if path provided (skip if using Ollama)
        if stage2_model_path and not ollama_model:
            print(f"Loading Stage 2 model from {stage2_model_path}...")
            self.stage2_tokenizer, self.stage2_model = self._load_model(
                stage2_model_path, base_model
            )
    
    def _detect_device(self, device: str) -> torch.device:
        """Detect best available device."""
        if device != "auto":
            return torch.device(device)
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _load_model(
        self, 
        model_path: str, 
        base_model: str
    ) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Load model (full fine-tuned or LoRA adapter)."""
        model_path = Path(model_path)
        
        # Check if this is a LoRA adapter or full model
        is_lora = (model_path / "adapter_config.json").exists()
        
        if is_lora and PEFT_AVAILABLE:
            # Load base model first
            print(f"  Loading base model: {base_model}")
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                torch_dtype=torch.bfloat16,
                device_map={"": self.device},
                trust_remote_code=True,
            )
            
            # Load LoRA adapter
            print(f"  Loading LoRA adapter: {model_path}")
            model = PeftModel.from_pretrained(model, str(model_path))
            model = model.merge_and_unload()  # Merge for faster inference
        else:
            # Load full fine-tuned model
            print(f"  Loading full model: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.bfloat16,
                device_map={"": self.device},
                trust_remote_code=True,
            )
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        return tokenizer, model
    
    def _build_messages(
        self, 
        system_prompt: str, 
        user_content: str
    ) -> list:
        """Build chat messages for model input."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    
    def _generate(
        self,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        messages: list,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """Generate response from messages."""
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        
        input_length = inputs.shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only generated tokens
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
    
    def retrieve_context_rag(
        self,
        instruction: str,
        top_k_helpers: int = 5,
        top_k_schemas: int = 3,
    ) -> str:
        """
        Retrieve context from Knowledge Base using RAG.
        
        Input: "Check that all pipeline tasks succeeded"
        Output: ATTESTATION_SCHEMA + AVAILABLE_HELPERS + RULE_DATA_KEYS
        """
        if self.rag_retriever is None:
            raise RuntimeError("RAG retriever not initialized. Provide --use-rag and --kb-dir.")
        
        return self.rag_retriever.retrieve_context(
            query=instruction,
            top_k_helpers=top_k_helpers,
            top_k_schemas=top_k_schemas,
        )
    
    def infer_context(
        self, 
        instruction: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """
        Stage 1: Infer context from natural language instruction.
        
        Input: "Check that all pipeline tasks succeeded"
        Output: ATTESTATION_SCHEMA + AVAILABLE_HELPERS + RULE_DATA_KEYS
        """
        if self.stage1_model is None:
            raise RuntimeError("Stage 1 model not loaded. Provide --stage1-model path.")
        
        # Build user content: instruction + system prompt (matches training format)
        user_content = f"{instruction}\n{self.STAGE1_INPUT_PROMPT}"
        
        messages = self._build_messages(self.SYSTEM_PROMPT, user_content)
        
        return self._generate(
            self.stage1_tokenizer,
            self.stage1_model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    def generate_rule(
        self,
        requirements: str,
        context: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        use_pattern_reminder: bool = False,  # Disabled by default - not in training data
    ) -> str:
        """
        Stage 2: Generate rule from requirements + context.
        
        Input: REQUIREMENTS + ATTESTATION_SCHEMA + AVAILABLE_HELPERS + RULE_DATA_KEYS
        Output: ANALYSIS + RULE + TESTS
        
        Args:
            use_pattern_reminder: If True, append pattern hints to help model accuracy
        """
        # Build user content
        input_text = f"REQUIREMENTS:\n{requirements}\n\n{context}"
        
        # Optionally add pattern reminder for better accuracy
        if use_pattern_reminder:
            input_text += self.STAGE2_PATTERN_REMINDER
        
        user_content = f"{self.STAGE2_INSTRUCTION}\n{input_text}"
        
        # Use Ollama if available
        if self.ollama is not None:
            return self.ollama.generate(
                prompt=user_content,
                system_prompt=self.SYSTEM_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        
        # Use HuggingFace model
        if self.stage2_model is None:
            raise RuntimeError("Stage 2 model not loaded. Provide --stage2-model or --ollama.")
        
        messages = self._build_messages(self.SYSTEM_PROMPT, user_content)
        
        return self._generate(
            self.stage2_tokenizer,
            self.stage2_model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    def generate(
        self,
        instruction: str,
        context: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
        verbose: bool = True,
        use_hints: bool = False,  # Disabled by default - not in training data
        use_rag: bool = False,  # Use RAG retrieval instead of Stage 1 model
    ) -> dict:
        """
        Full two-stage pipeline.
        
        Args:
            instruction: Natural language instruction (what user types)
            context: Optional pre-computed context. If None, Stage 1 or RAG runs first.
            max_tokens: Max tokens for generation
            temperature: Sampling temperature
            verbose: Print progress
            use_rag: Use RAG retrieval instead of Stage 1 model
        
        Returns:
            dict with 'context' and 'output' keys
        """
        # Stage 1: Infer context if not provided
        if context is None:
            if use_rag and self.rag_retriever is not None:
                # Use RAG retrieval
                if verbose:
                    print("\n=== Stage 1: Retrieving Context (RAG) ===")
                    print(f"Query: {instruction[:100]}...")
                
                context = self.retrieve_context_rag(instruction)
                
                if verbose:
                    print(f"\nRetrieved context ({len(context)} chars):")
                    print(context)
            else:
                # Use Stage 1 model
                if verbose:
                    print("\n=== Stage 1: Inferring Context (Model) ===")
                    print(f"Instruction: {instruction[:100]}...")
                
                context = self.infer_context(instruction, max_tokens=1024, temperature=temperature)
                
                if verbose:
                    print(f"\nInferred context ({len(context)} chars):")
                    print(context)  # Show full context
            
            # Validate context
            if not self._validate_context(context):
                print("\nWarning: Context missing expected sections (ATTESTATION_SCHEMA, AVAILABLE_HELPERS)")
        
        # Build structured requirements from instruction + Stage 1 metadata
        requirements = self._build_structured_requirements(instruction, context)
        
        if verbose:
            print(f"\nStructured requirements:\n{requirements}")
        
        # Stage 2: Generate rule
        if verbose:
            print("\n=== Stage 2: Generating Rule ===")
        
        output = self.generate_rule(
            requirements, 
            context, 
            max_tokens=max_tokens, 
            temperature=temperature,
            use_pattern_reminder=use_hints,
        )
        
        if verbose:
            print(f"\nGenerated output ({len(output)} chars):")
            print(output)  # Show full output
        
        return {
            "context": context,
            "output": output,
        }
    
    def _validate_context(self, context: str) -> bool:
        """Validate Stage 1 output contains expected sections."""
        required = ["ATTESTATION_SCHEMA:", "AVAILABLE_HELPERS:"]
        return all(section in context for section in required)
    
    def _parse_stage1_metadata(self, context: str) -> dict:
        """Extract SUGGESTED_PACKAGE and SUGGESTED_RULE_TYPE from Stage 1 output."""
        metadata = {
            "package": "",
            "rule_type": "deny",  # Default
        }
        
        for line in context.split('\n'):
            line = line.strip()
            if line.startswith('SUGGESTED_PACKAGE:'):
                metadata["package"] = line.split(':', 1)[1].strip()
            elif line.startswith('SUGGESTED_RULE_TYPE:'):
                rule_type = line.split(':', 1)[1].strip().lower()
                if rule_type in ('deny', 'warn'):
                    metadata["rule_type"] = rule_type
        
        return metadata
    
    def _build_structured_requirements(self, instruction: str, context: str) -> str:
        """Build structured requirements for Stage 2 using Stage 1 metadata.
        
        This bridges the gap between:
        - Training data: rich structured requirements
        - Inference: just user's instruction
        """
        metadata = self._parse_stage1_metadata(context)
        
        parts = [f"- {instruction}"]
        
        if metadata["package"]:
            parts.append(f"- Package: {metadata['package']}")
        
        parts.append(f"- Rule type: {metadata['rule_type']}")
        
        return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage inference for Rego policy generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # RAG + Ollama (recommended): No fine-tuned model needed
  python src/infer_two_stage.py \\
      --use-rag \\
      --ollama qwen3-coder:30b \\
      --instruction "Check that task bundles are pinned"

  # RAG + Ollama with smaller model
  python src/infer_two_stage.py \\
      --use-rag \\
      --ollama qwen3-coder:8b \\
      --instruction "Verify SBOM contains required packages"

  # RAG mode with fine-tuned Stage 2 model
  python src/infer_two_stage.py \\
      --use-rag \\
      --stage2-model models/stage2-rule-generation \\
      --instruction "Check that task bundles are pinned"

  # Full two-stage pipeline (with Stage 1 model)
  python src/infer_two_stage.py \\
      --stage1-model models/stage1-context-inference \\
      --stage2-model models/stage2-rule-generation \\
      --instruction "Check that all pipeline tasks succeeded"

  # Stage 1 only with RAG (test retrieval)
  python src/infer_two_stage.py \\
      --use-rag --stage 1 \\
      --instruction "Verify SBOM contains packages"

  # Stage 2 only (provide context)
  python src/infer_two_stage.py \\
      --stage2-model models/stage2-rule-generation \\
      --stage 2 \\
      --instruction "Check tasks" \\
      --context-file context.txt

  # Interactive RAG mode
  python src/infer_two_stage.py \\
      --use-rag \\
      --stage2-model models/stage2-rule-generation \\
      --interactive
"""
    )
    
    parser.add_argument(
        "--stage1-model",
        type=str,
        help="Path to Stage 1 model (context inference)",
    )
    parser.add_argument(
        "--stage2-model",
        type=str,
        help="Path to Stage 2 model (rule generation)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base model name for LoRA adapters",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Natural language instruction",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Pre-computed context (skips Stage 1)",
    )
    parser.add_argument(
        "--context-file",
        type=str,
        help="File containing pre-computed context",
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        help="Run only Stage 1 or Stage 2 (default: both)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (lower = more deterministic)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode - enter instructions at prompt",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "--use-hints",
        action="store_true",
        help="Enable pattern reminder hints for Stage 2 (experimental, not in training data)",
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Use RAG retrieval instead of Stage 1 model for context",
    )
    parser.add_argument(
        "--kb-dir",
        type=str,
        default="data/knowledge_base",
        help="Knowledge base directory for RAG mode (default: data/knowledge_base)",
    )
    parser.add_argument(
        "--top-k-helpers",
        type=int,
        default=5,
        help="Number of helpers to retrieve in RAG mode (default: 5)",
    )
    parser.add_argument(
        "--top-k-schemas",
        type=int,
        default=3,
        help="Number of schemas to retrieve in RAG mode (default: 3)",
    )
    parser.add_argument(
        "--ollama",
        type=str,
        metavar="MODEL",
        help="Use Ollama model for Stage 2 (e.g., qwen3-coder:30b, qwen3-coder:8b)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.stage == 1 and not args.stage1_model and not args.use_rag:
        parser.error("--stage1-model required for Stage 1 (or use --use-rag)")
    if args.stage == 2 and not args.stage2_model and not args.ollama:
        parser.error("--stage2-model or --ollama required for Stage 2")
    if not args.stage and not args.stage1_model and not args.stage2_model and not args.use_rag and not args.ollama:
        parser.error("Provide at least one of --stage1-model, --stage2-model, --use-rag, or --ollama")
    
    # Load context from file if provided
    context = args.context
    if args.context_file:
        context = Path(args.context_file).read_text()
    
    # Initialize RAG retriever if requested
    rag_retriever = None
    if args.use_rag:
        if not RAG_AVAILABLE:
            parser.error(
                "RAG components not available. "
                "Install with: uv pip install sentence-transformers faiss-cpu rank-bm25"
            )
        rag_retriever = RAGContextRetriever(Path(args.kb_dir))
    
    # Initialize generator
    generator = TwoStageGenerator(
        stage1_model_path=args.stage1_model,
        stage2_model_path=args.stage2_model,
        base_model=args.base_model,
        device=args.device,
        rag_retriever=rag_retriever,
        ollama_model=args.ollama,
    )
    
    verbose = not args.quiet
    
    # Interactive mode
    if args.interactive:
        mode = "RAG" if args.use_rag else "Model"
        print(f"\n=== Two-Stage Rego Generator ({mode} mode) ===")
        print("Enter instructions (Ctrl+D to exit)\n")
        
        while True:
            try:
                instruction = input("Instruction: ").strip()
                if not instruction:
                    continue
                
                if args.stage == 1:
                    if args.use_rag:
                        result = generator.retrieve_context_rag(instruction)
                    else:
                        result = generator.infer_context(instruction)
                    print(f"\n{result}\n")
                elif args.stage == 2:
                    if not context:
                        print("Error: Provide --context or --context-file for Stage 2")
                        continue
                    # Build structured requirements from context metadata
                    requirements = generator._build_structured_requirements(instruction, context)
                    result = generator.generate_rule(requirements, context, use_pattern_reminder=args.use_hints)
                    print(f"\n{result}\n")
                else:
                    result = generator.generate(
                        instruction, 
                        context=context, 
                        verbose=verbose, 
                        use_hints=args.use_hints,
                        use_rag=args.use_rag,
                    )
                    print(f"\n=== Result ===\n{result['output']}\n")
                    
            except EOFError:
                print("\nGoodbye!")
                break
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
        return
    
    # Single instruction mode
    if not args.instruction:
        parser.error("Provide --instruction or use --interactive mode")
    
    if args.stage == 1:
        # Stage 1 only (context retrieval/inference)
        if args.use_rag:
            result = generator.retrieve_context_rag(args.instruction)
        else:
            result = generator.infer_context(
                args.instruction,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        print(result)
        
    elif args.stage == 2:
        # Stage 2 only
        if not context:
            parser.error("Stage 2 requires --context or --context-file")
        
        # Build structured requirements from context metadata
        requirements = generator._build_structured_requirements(args.instruction, context)
        
        result = generator.generate_rule(
            requirements,
            context,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_pattern_reminder=args.use_hints,
        )
        print(result)
        
    else:
        # Full pipeline
        result = generator.generate(
            args.instruction,
            context=context,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            verbose=verbose,
            use_hints=args.use_hints,
            use_rag=args.use_rag,
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("FINAL OUTPUT")
            print("=" * 60)
        print(result["output"])


if __name__ == "__main__":
    main()

