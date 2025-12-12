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
        top_k_helpers: int = 5,
        top_k_schemas: int = 3,
    ) -> str:
        """Retrieve context formatted for Stage 2.
        
        Returns context in same format as Stage 1 model output:
        ATTESTATION_SCHEMA, AVAILABLE_HELPERS, RULE_DATA_KEYS, etc.
        """
        self.load()
        
        results = self.retriever.retrieve(
            query=query,
            helper_k=top_k_helpers,
            schema_k=top_k_schemas,
        )
        
        # Format as Stage 1-compatible context
        return self._format_context(results, query)
    
    def _format_context(self, results, query: str) -> str:
        """Format retrieval results as Stage 1-style context.
        
        Args:
            results: RetrievalResult from HybridRetriever
            query: Original query
        """
        parts = []
        
        # ATTESTATION_SCHEMA section (from retrieved schemas)
        schema_results = results.schemas
        if schema_results:
            parts.append("ATTESTATION_SCHEMA:")
            att_types = set()
            for item in schema_results:
                schema_id = item.get("schema_id") or item.get("id", "")
                schema = self.kb.get_schema(schema_id)
                if schema:
                    parts.append(f"- {schema.canonical_path}")
                    att_types.add(schema.attestation_type)
            parts.append("")
            
            # Infer attestation type from schemas
            if att_types:
                att_type = list(att_types)[0]  # Primary type
                if "slsa" in att_type.lower():
                    parts.insert(1, "- Attestation type: SLSA Provenance")
                elif "spdx" in att_type.lower():
                    parts.insert(1, "- Attestation type: SPDX SBOM")
                elif "cyclonedx" in att_type.lower():
                    parts.insert(1, "- Attestation type: CycloneDX SBOM")
        
        # AVAILABLE_HELPERS section
        helper_results = results.helpers
        if helper_results:
            parts.append("AVAILABLE_HELPERS:")
            for item in helper_results:
                helper_id = item.get("id", "")
                helper = self.kb.get_helper_card(helper_id)
                if helper:
                    # Format: module.name(args) - description
                    sig = helper.signature or helper.name
                    desc = helper.description[:80] if helper.description else ""
                    if desc:
                        parts.append(f"- {sig} -- {desc}")
                    else:
                        parts.append(f"- {sig}")
            parts.append("")
        
        # RULE_DATA_KEYS section (infer from query and helpers)
        rule_data_keys = self._infer_rule_data_keys(query, helper_results)
        if rule_data_keys:
            parts.append("RULE_DATA_KEYS:")
            for key in rule_data_keys:
                parts.append(f"- {key}")
            parts.append("")
        
        # SUGGESTED_PACKAGE and SUGGESTED_RULE_TYPE
        package, rule_type = self._infer_package_and_type(query, schema_results)
        parts.append(f"SUGGESTED_PACKAGE: {package}")
        parts.append(f"SUGGESTED_RULE_TYPE: {rule_type}")
        
        return "\n".join(parts)
    
    def _infer_rule_data_keys(self, query: str, helper_results: List[Dict]) -> List[str]:
        """Infer rule_data keys from query and helpers."""
        keys = []
        query_lower = query.lower()
        
        # Common rule_data patterns
        if "allowed" in query_lower or "trusted" in query_lower:
            keys.append("allowed_values")
        if "bundle" in query_lower:
            keys.append("allowed_bundles")
        if "package" in query_lower or "sbom" in query_lower:
            keys.append("required_packages")
        if "task" in query_lower:
            keys.append("allowed_tasks")
        if "label" in query_lower:
            keys.append("required_labels")
        
        return keys[:3]  # Limit to top 3
    
    def _infer_package_and_type(
        self, 
        query: str, 
        schema_results: List[Dict]
    ) -> Tuple[str, str]:
        """Infer package name and rule type from query."""
        query_lower = query.lower()
        
        # Default rule type
        rule_type = "deny"
        if "warn" in query_lower:
            rule_type = "warn"
        
        # Infer package from query keywords
        if "bundle" in query_lower:
            package = "policy.release.attestation_task_bundle"
        elif "sbom" in query_lower or "package" in query_lower:
            package = "policy.release.sbom"
        elif "task" in query_lower:
            package = "policy.release.tasks"
        elif "source" in query_lower:
            package = "policy.release.source"
        elif "image" in query_lower:
            package = "policy.release.image"
        else:
            package = "policy.release.custom"
        
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
    ):
        self.device = self._detect_device(device)
        print(f"Using device: {self.device}")
        
        self.stage1_model = None
        self.stage1_tokenizer = None
        self.stage2_model = None
        self.stage2_tokenizer = None
        self.rag_retriever = rag_retriever
        
        # Load Stage 1 model if path provided
        if stage1_model_path:
            print(f"Loading Stage 1 model from {stage1_model_path}...")
            self.stage1_tokenizer, self.stage1_model = self._load_model(
                stage1_model_path, base_model
            )
        
        # Load Stage 2 model if path provided
        if stage2_model_path:
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
        if self.stage2_model is None:
            raise RuntimeError("Stage 2 model not loaded. Provide --stage2-model path.")
        
        # Build user content EXACTLY matching training format:
        # instruction + "\n" + input (where input = "REQUIREMENTS:\n..." + "\n\n" + context)
        # See: train_policy.py lines 169-177 and generate_two_stage_dataset.py Stage2Example.format_input()
        input_text = f"REQUIREMENTS:\n{requirements}\n\n{context}"
        
        # Optionally add pattern reminder for better accuracy
        if use_pattern_reminder:
            input_text += self.STAGE2_PATTERN_REMINDER
        
        user_content = f"{self.STAGE2_INSTRUCTION}\n{input_text}"
        
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
  # RAG mode (recommended): Use KB retrieval instead of Stage 1 model
  python src/infer_two_stage.py \\
      --use-rag \\
      --stage2-model models/stage2-rule-generation \\
      --instruction "Check that task bundles are pinned"

  # RAG mode with more helpers
  python src/infer_two_stage.py \\
      --use-rag --top-k-helpers 8 --top-k-schemas 5 \\
      --stage2-model models/stage2-rule-generation \\
      --instruction "Verify SBOM contains required packages"

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
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.stage == 1 and not args.stage1_model and not args.use_rag:
        parser.error("--stage1-model required for Stage 1 (or use --use-rag)")
    if args.stage == 2 and not args.stage2_model:
        parser.error("--stage2-model required for Stage 2")
    if not args.stage and not args.stage1_model and not args.stage2_model and not args.use_rag:
        parser.error("Provide at least one of --stage1-model, --stage2-model, or --use-rag")
    
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

