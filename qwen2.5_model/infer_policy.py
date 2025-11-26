#!/usr/bin/env python3
"""
Interactive chatbot for the policy rule fine-tuned model.

Supports:
- Interactive chat mode for asking questions about policies
- Generating rules from instructions
- Refactoring existing rules
- Explaining how rules work
"""

import os
import re
import sys
import torch
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Import dynamic context building components
try:
    # Try direct imports first (when running as script)
    from library_mapper import LibraryMapper
    from library_indexer import LibraryIndexer
    from smart_context_builder import SmartContextBuilder
except ImportError:
    # Try relative imports (when used as module)
    try:
        from .library_mapper import LibraryMapper
        from .library_indexer import LibraryIndexer
        from .smart_context_builder import SmartContextBuilder
    except (ImportError, ValueError):
        # Add current directory to path
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent))
        from library_mapper import LibraryMapper
        from library_indexer import LibraryIndexer
        from smart_context_builder import SmartContextBuilder

# Default paths
DEFAULT_MODEL_DIR = "qwen2.5-rego-policy-lora"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# System prompt with internal reasoning (not verbose CoT)
QWEN_SYSTEM_PROMPT = (
    "You are an expert Rego/OPA policy assistant. "
    "You follow instructions carefully and emit valid Rego code using "
    "Conforma's preferred patterns (deny contains result, METADATA, result_helper, etc).\n\n"
    "Prefer helpers that are provided in the context. If you cannot find an appropriate helper, "
    "it is better to write a TODO comment than to invent a new module or helper function.\n\n"
    "Before writing Rego, briefly plan the approach in your head:\n"
    "1. Understand what the instruction is asking for\n"
    "2. Identify which helpers from the context are relevant\n"
    "3. Plan the rule structure (deny/warn, what conditions to check)\n"
    "4. Use the provided helpers correctly based on their usage examples\n"
    "Then output only the final Rego code."
)


def load_policy_model(base_model: str, model_dir: str = None, device: str = "mps", no_lora: bool = False):
    """Load the policy model (base model with optional LoRA adapters).
    
    Args:
        base_model: Base model name (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
        model_dir: Optional path to LoRA adapter directory
        device: Device to load on (default: mps for Apple Silicon)
        no_lora: If True, skip loading LoRA adapters even if model_dir is provided
        
    Returns:
        (tokenizer, model, device) tuple
    """
    print(f"Loading base model: {base_model}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print("\nPossible solutions:")
        print("1. Check if the model name is correct on HuggingFace")
        print("2. If it's a gated model, authenticate with: huggingface-cli login")
        raise
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading base model weights...")
    try:
        # Detect device
        if device == "mps" and not torch.backends.mps.is_available():
            print("⚠ MPS not available, falling back to CPU")
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU")
            device = "cpu"
        
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        device_map_value = device if device != "cpu" else None
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map={"": device_map_value} if device_map_value else None,
            trust_remote_code=True,
        )
        
        if device == "cpu" or device_map_value is None:
            base = base.to(device)
    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        raise
    
    # Load LoRA adapters if requested
    if no_lora:
        print("Skipping LoRA adapters (--no-lora flag set)")
        model = base
    elif model_dir:
        print(f"Loading LoRA adapters from: {model_dir}...")
        try:
            adapter_path = Path(model_dir)
            if not adapter_path.exists():
                print(f"⚠ Model directory not found: {model_dir}")
                print("Using base model only (no fine-tuning)")
                model = base
            elif (adapter_path / "adapter_config.json").exists():
                # Check adapter config to see what base model it was trained on
                import json
                adapter_config_path = adapter_path / "adapter_config.json"
                if adapter_config_path.exists():
                    try:
                        with open(adapter_config_path) as f:
                            adapter_config = json.load(f)
                            trained_base = adapter_config.get("base_model_name_or_path", "unknown")
                            if trained_base != base_model:
                                print(f"⚠ Warning: LoRA adapters were trained on: {trained_base}")
                                print(f"  But you're loading with base model: {base_model}")
                                print("  This may cause size mismatch errors.")
                                print("  Consider using: --base-model " + trained_base)
                    except Exception:
                        pass  # Ignore config read errors
                
                model = PeftModel.from_pretrained(base, model_dir)
                print("✓ LoRA adapters loaded")
            else:
                print(f"⚠ No adapter config found in {model_dir}, using base model only")
                model = base
        except RuntimeError as e:
            error_msg = str(e)
            if "size mismatch" in error_msg or "shape" in error_msg.lower():
                print(f"❌ Error: LoRA adapter size mismatch!")
                print(f"   The adapters in {model_dir} were trained on a different base model.")
                print(f"   Current base model: {base_model}")
                print()
                print("   Solutions:")
                print("   1. Use the correct base model that the LoRA was trained on")
                print("   2. Or use --no-lora to skip adapters and use base model only")
                print()
                # Try to extract the trained model from adapter config
                try:
                    import json
                    adapter_config_path = Path(model_dir) / "adapter_config.json"
                    if adapter_config_path.exists():
                        with open(adapter_config_path) as f:
                            adapter_config = json.load(f)
                            trained_base = adapter_config.get("base_model_name_or_path")
                            if trained_base:
                                print(f"   The adapters were likely trained on: {trained_base}")
                                print(f"   Try: --base-model {trained_base}")
                except Exception:
                    pass
                print()
                raise
            else:
                print(f"⚠ Error loading LoRA adapters: {e}")
                print("Using base model only")
                model = base
        except Exception as e:
            print(f"⚠ Error loading LoRA adapters: {e}")
            print("Using base model only")
            model = base
    else:
        print("No LoRA adapter directory specified, using base model only")
        model = base
    
    model.eval()
    print(f"✓ Model loaded successfully on {device}")
    print()
    
    return tokenizer, model, device


def generate_response_with_validation(
    tokenizer, model, device, messages, package: str = None, imports: List[str] = None,
    max_tokens=512, temperature=0.7, max_iterations=5, validate=True
):
    """Generate a response and validate it, iterating on errors if needed.
    
    Args:
        tokenizer: Tokenizer
        model: Model
        device: Device
        messages: Chat messages
        package: Package name for validation
        imports: List of imports for validation
        max_tokens: Max tokens per generation
        temperature: Temperature
        max_iterations: Maximum correction attempts
        validate: If True, validate and iterate on errors
        
    Returns:
        (final_response, was_validated, iterations_used)
    """
    from rego_validator import extract_rego_code, validate_rego_syntax
    
    if not validate:
        # Skip validation, just generate
        response = generate_response(tokenizer, model, device, messages, max_tokens, temperature)
        return response, False, 1
    
    iterations = 0
    conversation_messages = messages.copy()
    
    while iterations < max_iterations:
        iterations += 1
        
        # Generate response
        response = generate_response(tokenizer, model, device, conversation_messages, max_tokens, temperature)
        
        # Extract Rego code
        rego_code = extract_rego_code(response)
        
        if not rego_code:
            # No Rego code found, return as-is
            return response, False, iterations
        
        # Validate the code
        is_valid, formatted_code, error_msg = validate_rego_syntax(
            rego_code,
            package=package or "",
            imports=imports or []
        )
        
        if is_valid:
            # Code is valid! Replace with formatted version if different
            if formatted_code != rego_code:
                # Replace the code in response with formatted version
                if "```" in response:
                    response = re.sub(
                        r'```(?:rego)?\s*\n.*?```',
                        f'```rego\n{formatted_code}\n```',
                        response,
                        flags=re.DOTALL,
                        count=1
                    )
                else:
                    response = formatted_code
            
            return response, True, iterations
        
        # Code has errors, ask model to fix it
        if iterations < max_iterations:
            correction_prompt = f"""The generated Rego code has validation errors. Please fix them.

Error from opa parse:
{error_msg}

Generated code:
```rego
{rego_code}
```

Please provide the corrected Rego code that fixes these errors."""
            
            # Add assistant's previous response and correction request
            conversation_messages.append({"role": "assistant", "content": response})
            conversation_messages.append({"role": "user", "content": correction_prompt})
        else:
            # Max iterations reached, return with error note
            error_note = f"\n\n⚠ Note: Code validation failed after {max_iterations} attempts. Error: {error_msg}"
            return response + error_note, False, iterations
    
    return response, False, iterations


def generate_response(tokenizer, model, device, messages, max_tokens=512, temperature=0.7):
    """Generate a response from the model."""
    # Build chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract just the assistant's response
    # The chat template includes the full conversation, we want just the new part
    assistant_text = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": ""}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    if generated_text.startswith(assistant_text):
        response = generated_text[len(assistant_text):].strip()
    else:
        # Fallback: try to find the assistant response
        if "<|im_start|>assistant" in generated_text:
            parts = generated_text.split("<|im_start|>assistant")
            if len(parts) > 1:
                response = parts[-1].split("<|im_end|>")[0].strip()
            else:
                response = generated_text
        else:
            response = generated_text
    
    # Clean up special tokens
    response = response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
    
    return response


def interactive_chat(tokenizer, model, device, builder=None, default_package=None, validate=True, max_corrections=3):
    """Run interactive chat mode with dynamic context building.
    
    Args:
        tokenizer: Tokenizer for the model
        model: Model instance
        device: Device to run on
        builder: SmartContextBuilder instance (optional, will build context if provided)
        default_package: Default package name to use if not specified in user input
    """
    print("=" * 60)
    print("Policy Rule Chatbot")
    print("=" * 60)
    print()
    print("You can:")
    print("  - Ask questions about Rego/OPA policies")
    print("  - Request rule implementations")
    print("  - Ask to refactor existing code")
    print("  - Get explanations of how rules work")
    print()
    if builder:
        print("Dynamic context building is enabled.")
        print("  - Specify package with 'package:instruction' (e.g., 'tasks: write a rule...')")
        if default_package:
            print(f"  - Default package: {default_package}")
    print()
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("=" * 60)
    print()
    
    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT}
    ]
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nGoodbye!")
                break
            
            if user_input.lower() == "clear":
                messages = [{"role": "system", "content": QWEN_SYSTEM_PROMPT}]
                print("Conversation cleared.")
                continue
            
            # Parse package prefix if present (format: "package:instruction")
            package = default_package
            instruction = user_input
            if ':' in user_input and not user_input.startswith('http'):
                parts = user_input.split(':', 1)
                if len(parts) == 2 and len(parts[0].strip()) < 50:  # Reasonable package name length
                    package = parts[0].strip()
                    instruction = parts[1].strip()
            
            # Build context if builder is available
            if builder:
                try:
                    built_context = builder.build_context(instruction, package=package)
                    user_content = f"{built_context}\n\nInstruction: {instruction}"
                except Exception as e:
                    print(f"⚠ Warning: Failed to build context: {e}")
                    user_content = instruction
            else:
                user_content = instruction
            
            # Add user message
            messages.append({"role": "user", "content": user_content})
            
            # Generate response with validation
            print("\nAssistant: ", end="", flush=True)
            
            # Extract package and imports from context if available
            package_from_context = None
            imports_from_context = []
            if builder and user_content:
                # Try to extract package from context
                package_match = re.search(r'package\s+(\S+)', user_content)
                if package_match:
                    package_from_context = package_match.group(1)
                # Extract imports
                import_matches = re.findall(r'import\s+([^\n]+)', user_content)
                imports_from_context = [imp.strip() for imp in import_matches if imp.strip()]
            
            response, was_validated, iterations = generate_response_with_validation(
                tokenizer, model, device, messages,
                package=package_from_context or default_package,
                imports=imports_from_context,
                max_tokens=1024,
                temperature=0.7,
                max_iterations=max_corrections,
                validate=validate
            )
            
            if was_validated:
                print(f"✓ Generated and validated code (after {iterations} iteration{'s' if iterations > 1 else ''})")
            else:
                print(f"Generated response (validation {'skipped' if iterations == 1 else f'failed after {iterations} attempts'})")
            
            print(response)
            
            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": response})
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue chatting.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("You can continue chatting or type 'quit' to exit.")


def single_inference(
    tokenizer, model, device, instruction, 
    context=None, builder=None, package=None, max_tokens=1024,
    validate=True, max_corrections=3
):
    """Run a single inference with dynamic context building.
    
    Args:
        tokenizer: Tokenizer for the model
        model: Model instance
        device: Device to run on
        instruction: User instruction text
        context: Optional static context (overrides dynamic context if provided)
        builder: SmartContextBuilder instance (optional, for dynamic context)
        package: Optional package name for context building
        max_tokens: Maximum tokens to generate
    """
    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT}
    ]
    
    # Build context dynamically if builder is available
    if builder and not context:
        print("Building dynamic context from libraries...")
        try:
            built_context = builder.build_context(instruction, package=package)
            
            print("Context built:")
            print("-" * 60)
            print(built_context)
            print("-" * 60)
            print()
            
            # Simple instruction format (no verbose thinking prompt)
            user_content = f"{built_context}\n\nInstruction: {instruction}"
        except Exception as e:
            print(f"Warning: Failed to build dynamic context: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to instruction only.")
            user_content = instruction
    elif context:
        # Use provided static context
        user_content = f"{context}\n\nInstruction: {instruction}"
    else:
        # No context available
        user_content = instruction
    
    messages.append({"role": "user", "content": user_content})
    
    # Extract package and imports from context if available
    package_from_context = None
    imports_from_context = []
    if builder and user_content:
        package_match = re.search(r'package\s+(\S+)', user_content)
        if package_match:
            package_from_context = package_match.group(1)
        import_matches = re.findall(r'import\s+([^\n]+)', user_content)
        imports_from_context = [imp.strip() for imp in import_matches if imp.strip()]
    
    print("Generating response...")
    response, was_validated, iterations = generate_response_with_validation(
        tokenizer, model, device, messages,
        package=package_from_context or package,
        imports=imports_from_context,
        max_tokens=max_tokens,
        temperature=0.7,
        max_iterations=max_corrections,
        validate=validate
    )
    
    if was_validated:
        print(f"✓ Code validated successfully (after {iterations} iteration{'s' if iterations > 1 else ''})")
    elif iterations > 1:
        print(f"⚠ Validation failed after {iterations} attempts")
    print("\n" + "=" * 60)
    print("Response:")
    print("=" * 60)
    print(response)
    print("=" * 60)


def find_repo_root() -> Path:
    """Find repository root by looking for policy/ directory."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "policy").exists():
            return current
        current = current.parent
    # Fallback: assume we're in repo root
    return Path.cwd()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run inference with policy rule model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (using fine-tuned LoRA model - recommended):
  # Interactive chat mode with fine-tuned model (best results)
  uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \\
      --model-dir qwen2.5-rego-policy-lora \\
      --package tasks

  # Single inference with explicit package
  uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \\
      --model-dir qwen2.5-rego-policy-lora \\
      --package tasks \\
      --instruction "Write a rule that checks if all tasks in a PipelineRun succeeded"

  # Interactive mode: specify package in prompt (tasks: write a rule...)
  uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \\
      --model-dir qwen2.5-rego-policy-lora

  # Different package (SBOM rules)
  uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \\
      --model-dir qwen2.5-rego-policy-lora \\
      --package sbom_spdx \\
      --instruction "Write a rule that validates SPDX SBOM format"

Examples (using base model only - for comparison):
  # Use base model without LoRA adapters
  uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \\
      --base-model Qwen/Qwen2.5-1.5B-Instruct \\
      --no-lora \\
      --package tasks \\
      --instruction "Write a rule that checks if all tasks succeeded"

  # Disable dynamic context entirely
  uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \\
      --base-model Qwen/Qwen2.5-1.5B-Instruct \\
      --no-lora \\
      --no-context \\
      --instruction "Write a simple Rego rule"
        """
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"Base model name from HuggingFace (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help=f"Path to LoRA adapter directory (optional, default: {DEFAULT_MODEL_DIR} if exists, otherwise base model only)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Skip loading LoRA adapters, use base model only",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["mps", "cpu", "cuda"],
        help="Device to run on (default: mps for Apple Silicon)",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        help="Instruction text (if not provided, will use interactive chat mode)",
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Context (package + imports + helpers) for the instruction",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7, lower = more deterministic)",
    )
    parser.add_argument(
        "--package",
        type=str,
        default=None,
        help="Package name for context building (e.g., 'tasks', 'sbom_spdx'). If not provided, will be inferred from instruction.",
    )
    parser.add_argument(
        "--no-context",
        action="store_true",
        help="Disable dynamic context building (use base model without library context)",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable code validation and self-correction (faster but may return invalid code)",
    )
    parser.add_argument(
        "--max-corrections",
        type=int,
        default=3,
        help="Maximum number of correction attempts when validation fails (default: 3)",
    )
    
    args = parser.parse_args()
    
    # Resolve model directory relative to repo root
    script_dir = Path(__file__).parent.resolve()
    current_dir = Path.cwd()
    
    # Find repo root
    repo_root = None
    for candidate in [script_dir, current_dir]:
        check_dir = candidate
        for _ in range(5):
            if (check_dir / ".git").exists() or (check_dir / "policy").exists():
                repo_root = check_dir
                break
            check_dir = check_dir.parent
        if repo_root:
            break
    
    if not repo_root:
        repo_root = current_dir
    
    # Resolve model directory if provided
    model_dir = None
    if args.model_dir:
        model_dir = args.model_dir if os.path.isabs(args.model_dir) else str(repo_root / args.model_dir)
    elif not args.no_lora:
        # If no model_dir specified and --no-lora not set, try default
        default_model_dir = repo_root / DEFAULT_MODEL_DIR
        if default_model_dir.exists():
            model_dir = str(default_model_dir)
    
    # Load model
    print("=" * 60)
    print("Loading Policy Rule Model")
    print("=" * 60)
    print()
    
    try:
        tokenizer, model, device = load_policy_model(
            base_model=args.base_model,
            model_dir=model_dir,
            device=args.device,
            no_lora=args.no_lora
        )
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Initialize context builder if not disabled
    builder = None
    if not args.no_context:
        print("Initializing library context system...")
        try:
            mapper = LibraryMapper(repo_root)
            mapper.build_mappings()
            
            indexer = LibraryIndexer(repo_root, mapper)
            # Scan for usage examples (slower but provides better context)
            indexer.index_all_libraries(scan_usage=True)
            
            builder = SmartContextBuilder(indexer, mapper, max_tokens=500)
            print(f"✓ Indexed {len(indexer.index)} helpers")
            print()
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize context system: {e}")
            print("Continuing without dynamic context...")
            print()
    
    # Run inference
    if args.instruction:
        # Single inference mode
        single_inference(
            tokenizer,
            model,
            device,
            args.instruction,
            context=args.context,
            builder=builder,
            package=args.package,
            max_tokens=args.max_tokens,
            validate=not args.no_validate,
            max_corrections=args.max_corrections
        )
    else:
        # Interactive chat mode
        interactive_chat(
            tokenizer,
            model,
            device,
            builder=builder,
            default_package=args.package,
            validate=not args.no_validate,
            max_corrections=args.max_corrections
        )


if __name__ == "__main__":
    main()

