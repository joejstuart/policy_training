#!/usr/bin/env python3
"""
Two-stage inference for Rego policy generation.

Stage 1: Natural language → Context (schema, helpers, rule_data_keys)
Stage 2: Context + requirements → Rule + tests

Usage:
    # Full two-stage pipeline
    python src/infer_two_stage.py \
        --stage1-model models/stage1-context-inference \
        --stage2-model models/stage2-rule-generation \
        --instruction "Check that all pipeline tasks succeeded"

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
from typing import Optional, Tuple

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


class TwoStageGenerator:
    """Two-stage Rego policy generator."""
    
    # System prompt for Stage 1
    STAGE1_SYSTEM_PROMPT = (
        "You are an expert Rego policy assistant. "
        "Analyze requirements and identify the attestation schema, helpers, and rule data keys needed."
    )
    
    # System prompt for Stage 2
    STAGE2_SYSTEM_PROMPT = (
        "You are an expert Rego policy assistant. "
        "Write valid Rego code using the provided context. "
        "Follow Conforma patterns: deny contains result, METADATA blocks, result_helper."
    )
    
    # Fixed instruction for Stage 1 (model trained with this)
    STAGE1_INPUT_PROMPT = "Analyze the requirements and identify the attestation schema, available helpers, and rule data keys needed to implement this Rego rule."
    
    # Fixed instruction for Stage 2 (model trained with this)
    STAGE2_INSTRUCTION = "Write a Rego rule that enforces the requirements below using the provided context."
    
    def __init__(
        self,
        stage1_model_path: Optional[str] = None,
        stage2_model_path: Optional[str] = None,
        base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "auto",
    ):
        self.device = self._detect_device(device)
        print(f"Using device: {self.device}")
        
        self.stage1_model = None
        self.stage1_tokenizer = None
        self.stage2_model = None
        self.stage2_tokenizer = None
        
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
        
        messages = self._build_messages(self.STAGE1_SYSTEM_PROMPT, user_content)
        
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
    ) -> str:
        """
        Stage 2: Generate rule from requirements + context.
        
        Input: REQUIREMENTS + ATTESTATION_SCHEMA + AVAILABLE_HELPERS + RULE_DATA_KEYS
        Output: ANALYSIS + RULE + TESTS
        """
        if self.stage2_model is None:
            raise RuntimeError("Stage 2 model not loaded. Provide --stage2-model path.")
        
        # Build user content: instruction + requirements + context
        user_content = f"{self.STAGE2_INSTRUCTION}\n\nREQUIREMENTS:\n{requirements}\n\n{context}"
        
        messages = self._build_messages(self.STAGE2_SYSTEM_PROMPT, user_content)
        
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
    ) -> dict:
        """
        Full two-stage pipeline.
        
        Args:
            instruction: Natural language instruction (what user types)
            context: Optional pre-computed context. If None, Stage 1 runs first.
            max_tokens: Max tokens for generation
            temperature: Sampling temperature
            verbose: Print progress
        
        Returns:
            dict with 'context' and 'output' keys
        """
        # Stage 1: Infer context if not provided
        if context is None:
            if verbose:
                print("\n=== Stage 1: Inferring Context ===")
                print(f"Instruction: {instruction[:100]}...")
            
            context = self.infer_context(instruction, max_tokens=1024, temperature=temperature)
            
            if verbose:
                print(f"\nInferred context ({len(context)} chars):")
                print(context[:500] + "..." if len(context) > 500 else context)
            
            # Validate context
            if not self._validate_context(context):
                print("\nWarning: Context missing expected sections (ATTESTATION_SCHEMA, AVAILABLE_HELPERS)")
        
        # Build requirements from instruction for Stage 2
        requirements = f"- {instruction}"
        
        # Stage 2: Generate rule
        if verbose:
            print("\n=== Stage 2: Generating Rule ===")
        
        output = self.generate_rule(
            requirements, 
            context, 
            max_tokens=max_tokens, 
            temperature=temperature
        )
        
        if verbose:
            print(f"\nGenerated output ({len(output)} chars):")
            print(output[:1000] + "..." if len(output) > 1000 else output)
        
        return {
            "context": context,
            "output": output,
        }
    
    def _validate_context(self, context: str) -> bool:
        """Validate Stage 1 output contains expected sections."""
        required = ["ATTESTATION_SCHEMA:", "AVAILABLE_HELPERS:"]
        return all(section in context for section in required)


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage inference for Rego policy generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full two-stage pipeline
  python src/infer_two_stage.py \\
      --stage1-model models/stage1-context-inference \\
      --stage2-model models/stage2-rule-generation \\
      --instruction "Check that all pipeline tasks succeeded"

  # Stage 1 only (get context)
  python src/infer_two_stage.py \\
      --stage1-model models/stage1-context-inference \\
      --stage 1 \\
      --instruction "Verify SBOM contains packages"

  # Stage 2 only (provide context)
  python src/infer_two_stage.py \\
      --stage2-model models/stage2-rule-generation \\
      --stage 2 \\
      --instruction "Check tasks" \\
      --context-file context.txt

  # Interactive mode
  python src/infer_two_stage.py \\
      --stage1-model models/stage1-context-inference \\
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
        default="Qwen/Qwen2.5-1.5B-Instruct",
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
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.stage == 1 and not args.stage1_model:
        parser.error("--stage1-model required for Stage 1")
    if args.stage == 2 and not args.stage2_model:
        parser.error("--stage2-model required for Stage 2")
    if not args.stage and not args.stage1_model and not args.stage2_model:
        parser.error("Provide at least one of --stage1-model or --stage2-model")
    
    # Load context from file if provided
    context = args.context
    if args.context_file:
        context = Path(args.context_file).read_text()
    
    # Initialize generator
    generator = TwoStageGenerator(
        stage1_model_path=args.stage1_model,
        stage2_model_path=args.stage2_model,
        base_model=args.base_model,
        device=args.device,
    )
    
    verbose = not args.quiet
    
    # Interactive mode
    if args.interactive:
        print("\n=== Two-Stage Rego Generator ===")
        print("Enter instructions (Ctrl+D to exit)\n")
        
        while True:
            try:
                instruction = input("Instruction: ").strip()
                if not instruction:
                    continue
                
                if args.stage == 1:
                    result = generator.infer_context(instruction)
                    print(f"\n{result}\n")
                elif args.stage == 2:
                    if not context:
                        print("Error: Provide --context or --context-file for Stage 2")
                        continue
                    result = generator.generate_rule(f"- {instruction}", context)
                    print(f"\n{result}\n")
                else:
                    result = generator.generate(instruction, context=context, verbose=verbose)
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
        # Stage 1 only
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
        
        result = generator.generate_rule(
            f"- {args.instruction}",
            context,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
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
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("FINAL OUTPUT")
            print("=" * 60)
        print(result["output"])


if __name__ == "__main__":
    main()

