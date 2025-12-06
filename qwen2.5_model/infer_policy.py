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
import json
import torch
import argparse
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

# Fix tokenizers parallelism warning when using subprocess (OPA calls)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

# System prompt for attestation parsing (with style guide awareness)
QWEN_SYSTEM_PROMPT_ATTESTATION = (
    "You are an expert Rego/OPA policy assistant for attestation parsing. "
    "You write Rego code that correctly parses attestation JSON structures.\n\n"
    "CRITICAL: Match instruction keywords to the CORRECT JSON paths:\n"
    "- If instruction mentions 'task' or 'tasks' → use predicate.buildConfig.tasks[]\n"
    "- If instruction mentions 'material' or 'materials' → use predicate.materials[]\n"
    "- If instruction mentions 'subject' → use subject[]\n"
    "- If instruction mentions 'builder' → use predicate.builder\n"
    "DO NOT assume all checks are about tasks - read the instruction carefully!\n\n"
    "CRITICAL: Follow the user's instructions precisely. "
    "If the instruction specifies a rule name, function name, variable name, return value, "
    "or any other specific requirement, you MUST use exactly what is requested. "
    "Do not substitute defaults (like 'deny') unless the instruction explicitly asks for them.\n\n"
    "Follow Rego style guide best practices:\n"
    "- Use 'in' for membership checks when checking multiple values\n"
    "- Use 'every' for FOR ALL queries (e.g., 'all tasks succeeded')\n"
    "- Use 'some ... in' for iteration (declarative pattern)\n"
    "- Prefer sets over arrays when order doesn't matter\n"
    "- Use unconditional assignment in rule head when possible\n"
    "- Use snake_case for all variable and rule names\n"
    "- Always include 'package attestation_check' and 'import rego.v1'\n\n"
    "Generate valid Rego code that correctly navigates attestation structures based on the instruction keywords."
)

# Condensed Rego style guide for attestation parsing (~310 tokens)
STYLE_GUIDE_CONDENSED = """# Rego Style Guide - Key Patterns for Attestation Parsing

## Critical Patterns:

1. **Use `in` for membership checks**: 
   - Prefer: `task.status in {"Succeeded", "Failed"}`
   - Avoid: `task.status == "Succeeded"` (when checking multiple values)

2. **Use `every` for FOR ALL queries**:
   - Prefer: `every task in att.predicate.buildConfig.tasks { task.status == "Succeeded" }`
   - Use for: "all tasks succeeded", "verify all tasks have X"

3. **Use `some ... in` for iteration**:
   - Prefer: `some task in att.predicate.buildConfig.tasks`
   - This is the modern, declarative pattern

4. **Prefer sets over arrays when order doesn't matter**:
   - Prefer: `task_names := {name | ...}`
   - Avoid: `task_names := [name | ...]` (unless order matters)

5. **Unconditional assignment in rule head**:
   - Prefer: `result := value if { ... }`
   - Avoid: `result if { value := ... }`

6. **Helper rules with leading underscore**:
   - Use: `_helper_name()` for internal helpers
   - Example: `_task_by_name(name) := task if { ... }`

7. **snake_case for all names**:
   - Use: `task_name`, `bundle_ref`, `subject_digest`
   - Avoid: `taskName`, `bundleRef`

8. **Package and import**:
   - Always include: `package attestation_check` and `import rego.v1`
"""


@dataclass
class AgentState:
    """Tracks the agent's state through the workflow."""
    iteration: int = 0
    plan: Optional[str] = None
    implementation: Optional[str] = None
    best_code: Optional[str] = None  # Track best code seen so far (even if invalid)
    best_score: int = 0  # Score: 4=all valid, 3=3 valid, 2=2 valid, 1=1 valid, 0=none
    syntax_valid: bool = False
    semantic_valid: bool = False
    execution_valid: bool = False
    style_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)  # Conversation history


def load_policy_model(base_model: str, model_dir: str = None, device: str = "mps", no_lora: bool = False):
    """Load the policy model (base model, full fine-tuned model, or base with LoRA adapters).
    
    Args:
        base_model: Base model name (e.g., "Qwen/Qwen2.5-1.5B-Instruct")
        model_dir: Optional path to model directory (full fine-tuned model or LoRA adapter)
        device: Device to load on (default: mps for Apple Silicon)
        no_lora: If True, skip loading LoRA adapters even if model_dir is provided
        
    Returns:
        (tokenizer, model, device) tuple
    """
    # Auto-detect device if requested
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"✓ Auto-detected CUDA (GPU: {torch.cuda.get_device_name(0)})")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("✓ Auto-detected MPS (Apple Silicon)")
        else:
            device = "cpu"
            print("⚠ No GPU available, using CPU")
    # Detect device availability for explicit choices
    elif device == "mps" and not torch.backends.mps.is_available():
        print("⚠ MPS not available, falling back to CPU")
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, falling back to CPU")
        device = "cpu"
    
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    device_map_value = device if device != "cpu" else None
    
    # Check if model_dir contains a full fine-tuned model or LoRA adapter
    is_full_model = False
    is_lora_adapter = False
    
    if model_dir and not no_lora:
        model_path = Path(model_dir)
        if model_path.exists():
            has_config = (model_path / "config.json").exists()
            has_adapter_config = (model_path / "adapter_config.json").exists()
            
            if has_config and not has_adapter_config:
                # This looks like a full fine-tuned model
                is_full_model = True
            elif has_adapter_config:
                # This is a LoRA adapter
                is_lora_adapter = True
    
    # Load full fine-tuned model directly
    if is_full_model:
        print(f"Loading full fine-tuned model from: {model_dir}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                device_map={"": device_map_value} if device_map_value else None,
                trust_remote_code=True,
            )
            
            if device == "cpu" or device_map_value is None:
                model = model.to(device)
            
            model.eval()
            print(f"✓ Full fine-tuned model loaded successfully on {device}")
            print()
            return tokenizer, model, device
        except Exception as e:
            print(f"❌ Error loading full model: {e}")
            raise
    
    # Otherwise, load base model first
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
    elif is_lora_adapter:
        print(f"Loading LoRA adapters from: {model_dir}...")
        try:
            # Import PEFT only when needed
            try:
                from peft import PeftModel
            except ImportError:
                print("❌ Error: PEFT library is required to load LoRA adapters")
                print("   Install it with: pip install peft")
                print("   Or use --no-lora to run without LoRA adapters")
                raise
            
            adapter_path = Path(model_dir)
            if (adapter_path / "adapter_config.json").exists():
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
        if model_dir:
            print(f"⚠ Model directory provided ({model_dir}) but doesn't appear to be a full model or LoRA adapter")
            print("   Expected: config.json (full model) or adapter_config.json (LoRA adapter)")
            print("   Using base model only")
        else:
            print("No model directory specified, using base model only")
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


def enhance_instruction_with_emphasis(instruction: str) -> str:
    """Enhance instruction to emphasize following specific requirements.
    
    Extracts and emphasizes any explicitly specified names, values, or requirements
    from the instruction to help the model follow them precisely.
    """
    # Look for quoted strings that might be specific names/values
    quoted_pattern = r"['\"]([^'\"]+)['\"]"
    quoted_matches = re.findall(quoted_pattern, instruction)
    
    if not quoted_matches:
        return instruction
    
    # Build emphasis text for quoted values that appear to be specifications
    emphasis_parts = []
    instruction_lower = instruction.lower()
    
    for match in quoted_matches:
        # Find the context around this quoted value
        match_pos = instruction.find(f"'{match}'")
        if match_pos == -1:
            match_pos = instruction.find(f'"{match}"')
        
        if match_pos > 0:
            # Look at context before the quoted value
            context_before = instruction_lower[max(0, match_pos-50):match_pos]
            
            # Check if this looks like a specification (named, called, return, etc.)
            if any(keyword in context_before for keyword in [
                'named', 'called', 'name', 'rule', 'function', 'variable',
                'return', 'value', 'status', 'result', 'create', 'write', 'make'
            ]):
                emphasis_parts.append(f"Use exactly '{match}' as specified in the instruction")
    
    if emphasis_parts:
        emphasis_text = "\n".join(emphasis_parts)
        return f"{instruction}\n\nIMPORTANT: {emphasis_text}. Do not substitute with defaults."
    
    return instruction


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
    
    # Generate with error handling for numerical instability
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        except RuntimeError as e:
            if "inf" in str(e) or "nan" in str(e) or "probability tensor" in str(e):
                # Numerical instability - try with different parameters
                # Use greedy decoding (temperature=0 or do_sample=False) as fallback
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,  # Greedy decoding
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                except RuntimeError as e2:
                    # If that also fails, try with very low temperature
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=0.1,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            else:
                raise  # Re-raise if it's a different error
    
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


def interactive_chat(
    tokenizer, model, device, builder=None, default_package=None, 
    validate=True, max_corrections=5, include_style_guide=False, 
    enhance_instruction=True, agentic=True, verbose=True,
    include_execution_check=True, attestation_files=None, include_planning=True
):
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
            
            # Auto-detect if this is an attestation parsing task
            is_attestation_task = any(keyword in instruction.lower() for keyword in [
                'attestation', 'task', 'subject', 'material', 'bundle', 'digest',
                'succeeded', 'failed', 'status', 'timestamp', 'finishedon', 'startedon'
            ])
            
            # Choose system prompt based on task type (update if needed)
            if is_attestation_task and messages[0]["content"] != QWEN_SYSTEM_PROMPT_ATTESTATION:
                messages[0]["content"] = QWEN_SYSTEM_PROMPT_ATTESTATION
            elif not is_attestation_task and messages[0]["content"] != QWEN_SYSTEM_PROMPT:
                messages[0]["content"] = QWEN_SYSTEM_PROMPT
            
            # Build context if builder is available
            context_parts = []
            
            # Add style guide if requested
            if include_style_guide:
                context_parts.append(STYLE_GUIDE_CONDENSED)
            
            if builder:
                try:
                    built_context = builder.build_context(instruction, package=package)
                    context_parts.append(built_context)
                except Exception as e:
                    print(f"⚠ Warning: Failed to build context: {e}")
            
            # Combine context parts
            combined_context = "\n\n".join(context_parts) if context_parts else None
            
            # Enhance instruction to emphasize specific requirements (if enabled)
            if enhance_instruction:
                enhanced_instruction = enhance_instruction_with_emphasis(instruction)
            else:
                enhanced_instruction = instruction
            
            # Build user content for non-agentic mode
            if combined_context:
                user_content = f"{combined_context}\n\nInstruction: {enhanced_instruction}"
            else:
                user_content = enhanced_instruction
            
            # Add user message (for non-agentic mode)
            if not agentic:
                messages.append({"role": "user", "content": user_content})
            
            # Generate response
            print("\nAssistant: ", end="", flush=True)
            
            if agentic:
                # Use agentic workflow
                # Extract package and imports from context if available
                package_from_context = default_package
                imports_from_context = []
                if builder and user_content:
                    package_match = re.search(r'package\s+(\S+)', user_content)
                    if package_match:
                        package_from_context = package_match.group(1)
                    import_matches = re.findall(r'import\s+([^\n]+)', user_content)
                    imports_from_context = [imp.strip() for imp in import_matches if imp.strip()]
                
                # Find attestation files if needed
                if include_execution_check and attestation_files is None:
                    is_attestation_task = any(kw in instruction.lower() for kw in [
                        'attestation', 'task', 'subject', 'material'
                    ])
                    if is_attestation_task:
                        repo_root = find_repo_root()
                        attestation_files = find_attestation_files(repo_root, max_files=3)
                
                final_code, state = agentic_inference(
                    tokenizer, model, device,
                    instruction,
                    context=combined_context,
                    package=package_from_context,
                    imports=imports_from_context,
                    max_iterations=max_corrections,
                    include_planning=include_planning,
                    include_style_check=include_style_guide,
                    include_execution_check=include_execution_check,
                    attestation_files=attestation_files,
                    verbose=verbose
                )
                
                print(final_code)
                
                if verbose and (state.errors or state.warnings):
                    if state.errors:
                        print("\nErrors:")
                        for error in state.errors:
                            print(f"  - {error}")
                    if state.warnings:
                        print("\nWarnings:")
                        for warning in state.warnings:
                            print(f"  - {warning}")
                
                # Add to conversation history
                messages.append({"role": "assistant", "content": final_code})
            else:
                # Use existing workflow
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
    validate=True, max_corrections=5, include_style_guide=False, 
    enhance_instruction=True, agentic=True, verbose=True,
    include_execution_check=True, attestation_files=None, include_planning=True
):
    """Run inference with optional agentic workflow.
    
    Args:
        tokenizer: Tokenizer for the model
        model: Model instance
        device: Device to run on
        instruction: User instruction text
        context: Optional static context (overrides dynamic context if provided)
        builder: SmartContextBuilder instance (optional, for dynamic context)
        package: Optional package name for context building
        max_tokens: Maximum tokens to generate
        include_style_guide: If True, include condensed style guide in context
        agentic: If True, use agentic workflow (Plan → Implement → Check → Repair)
        verbose: If True, show detailed workflow progress
        include_execution_check: If True, test code against real attestation files
        attestation_files: List of Path objects to attestation files (auto-discovered if None)
        include_planning: If True, include planning phase in agentic workflow
    """
    if agentic:
        # Use new agentic workflow
        # Build context if builder available
        context_parts = []
        if include_style_guide:
            context_parts.append(STYLE_GUIDE_CONDENSED)
        
        if builder and not context:
            built_context = builder.build_context(instruction, package=package)
            context_parts.append(built_context)
        elif context:
            context_parts.append(context)
        
        combined_context = "\n\n".join(context_parts) if context_parts else None
        
        # Extract package/imports from context
        package_from_context = package
        imports_from_context = []
        if combined_context:
            package_match = re.search(r'package\s+(\S+)', combined_context)
            if package_match:
                package_from_context = package_match.group(1)
            import_matches = re.findall(r'import\s+([^\n]+)', combined_context)
            imports_from_context = [imp.strip() for imp in import_matches if imp.strip()]
        
        # Find attestation files if needed
        if include_execution_check and attestation_files is None:
            is_attestation_task = any(kw in instruction.lower() for kw in [
                'attestation', 'task', 'subject', 'material'
            ])
            if is_attestation_task:
                repo_root = find_repo_root()
                attestation_files = find_attestation_files(repo_root, max_files=3)
                if verbose and attestation_files:
                    print(f"✓ Found {len(attestation_files)} attestation files for execution testing")
        
        final_code, state = agentic_inference(
            tokenizer, model, device,
            instruction,
            context=combined_context,
            package=package_from_context,
            imports=imports_from_context,
            max_iterations=max_corrections,
            include_planning=include_planning,
            include_style_check=include_style_guide,
            include_execution_check=include_execution_check,
            attestation_files=attestation_files,
            verbose=verbose
        )
        
        # Format output
        print("\n" + "=" * 60)
        print("Final Result:")
        print("=" * 60)
        if "```" not in final_code:
            print(f"```rego\n{final_code}\n```")
        else:
            print(final_code)
        print("=" * 60)
        
        if verbose and state.errors:
            print("\nErrors encountered:")
            for error in state.errors:
                print(f"  - {error}")
        if verbose and state.warnings:
            print("\nWarnings:")
            for warning in state.warnings:
                print(f"  - {warning}")
    else:
        # Use existing workflow (backward compatibility)
        # Auto-detect if this is an attestation parsing task
        is_attestation_task = any(keyword in instruction.lower() for keyword in [
            'attestation', 'task', 'subject', 'material', 'bundle', 'digest',
            'succeeded', 'failed', 'status', 'timestamp', 'finishedon', 'startedon'
        ])
        
        # Choose system prompt based on task type
        if is_attestation_task:
            system_prompt = QWEN_SYSTEM_PROMPT_ATTESTATION
        else:
            system_prompt = QWEN_SYSTEM_PROMPT
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Build context dynamically if builder is available
        context_parts = []
        
        # Add style guide if requested
        if include_style_guide:
            context_parts.append(STYLE_GUIDE_CONDENSED)
            print("✓ Including Rego style guide in context")
        
        if builder and not context:
            print("Building dynamic context from libraries...")
            try:
                built_context = builder.build_context(instruction, package=package)
                context_parts.append(built_context)
                
                print("Context built:")
                print("-" * 60)
                print(built_context)
                print("-" * 60)
                print()
            except Exception as e:
                print(f"Warning: Failed to build dynamic context: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to instruction only.")
        elif context:
            # Use provided static context
            context_parts.append(context)
        
        # Enhance instruction to emphasize specific requirements (if enabled)
        if enhance_instruction:
            enhanced_instruction = enhance_instruction_with_emphasis(instruction)
        else:
            enhanced_instruction = instruction
        
        # Combine context parts
        if context_parts:
            combined_context = "\n\n".join(context_parts)
            user_content = f"{combined_context}\n\nInstruction: {enhanced_instruction}"
        else:
            # No context available
            user_content = enhanced_instruction
        
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


# ============================================================================
# AGENTIC WORKFLOW FUNCTIONS
# ============================================================================

def _tree_structure(obj, indent: str = "", max_depth: int = 4) -> str:
    """Recursively generate tree structure from JSON object.
    
    Args:
        obj: JSON object (dict, list, or primitive)
        indent: Current indentation string
        max_depth: Maximum depth to recurse
    
    Returns:
        Tree structure string
    """
    if max_depth <= 0:
        return indent + "...\n"
    
    if isinstance(obj, dict):
        result = []
        for key in sorted(obj.keys()):
            result.append(indent + key + "\n")
            result.append(_tree_structure(obj[key], indent + "  ", max_depth - 1))
        return "".join(result)
    elif isinstance(obj, list):
        if len(obj) > 0:
            result = indent + "[]\n"
            result += _tree_structure(obj[0], indent + "  ", max_depth - 1)
            return result
        else:
            return indent + "[]\n"
    else:
        return ""


def _extract_example_values(obj, path_parts: List[str], max_examples: int = 3) -> str:
    """Extract example values for a specific path in the JSON.
    
    Args:
        obj: Current JSON object
        path_parts: List of path segments to navigate (e.g., ['predicate', 'buildConfig', 'tasks', 'results'])
        max_examples: Maximum number of examples to show
        
    Returns:
        String showing example values, or empty string if path not found
    """
    if not path_parts:
        # We've reached the target - show the value
        if isinstance(obj, dict):
            # Show key-value pairs (limit to first 5 keys)
            items = list(obj.items())[:5]
            examples = []
            for key, val in items:
                if isinstance(val, (str, int, float, bool)):
                    examples.append(f"  {key}: {repr(val)}")
                elif isinstance(val, list):
                    examples.append(f"  {key}: [array with {len(val)} items]")
                elif isinstance(val, dict):
                    examples.append(f"  {key}: {{object with {len(val)} keys}}")
            if examples:
                return "\n".join(examples)
        elif isinstance(obj, list) and len(obj) > 0:
            # Show first few items
            examples = []
            for i, item in enumerate(obj[:max_examples]):
                if isinstance(item, dict):
                    # Show key-value pairs of first item
                    items = list(item.items())[:5]
                    item_str = ", ".join(f"{k}: {repr(v) if isinstance(v, (str, int, float, bool)) else type(v).__name__}" for k, v in items)
                    examples.append(f"  [{i}]: {{{item_str}}}")
                else:
                    examples.append(f"  [{i}]: {repr(item)}")
            if len(obj) > max_examples:
                examples.append(f"  ... ({len(obj) - max_examples} more items)")
            return "\n".join(examples)
        return ""
    
    # Navigate deeper
    key = path_parts[0]
    if isinstance(obj, dict) and key in obj:
        next_obj = obj[key]
        # If the value is an array and we have more path parts, navigate into array items
        if isinstance(next_obj, list) and len(next_obj) > 0 and len(path_parts) > 1:
            # Try to navigate into first array item with remaining path
            for item in next_obj[:2]:  # Try first 2 items
                if isinstance(item, dict):
                    result = _extract_example_values(item, path_parts[1:], max_examples)
                    if result:
                        return result
            return ""  # Couldn't find path in array items
        else:
            # Normal navigation
            return _extract_example_values(next_obj, path_parts[1:], max_examples)
    elif isinstance(obj, list) and len(obj) > 0:
        # Already in an array - try to navigate into first item
        for item in obj[:2]:  # Try first 2 items
            if isinstance(item, dict):
                result = _extract_example_values(item, path_parts, max_examples)
                if result:
                    return result
        return ""  # Couldn't navigate into array items
    
    return ""


def _extract_keywords_from_instruction(instruction: str) -> List[str]:
    """Extract relevant keywords from instruction that might match JSON keys.
    
    Returns:
        List of keywords (normalized, singular/plural variants)
    """
    instruction_lower = instruction.lower()
    keywords = []
    
    # Common keywords to look for
    keyword_patterns = {
        'task': ['task', 'tasks'],
        'material': ['material', 'materials'],
        'subject': ['subject', 'subjects'],
        'builder': ['builder', 'builders'],
        'result': ['result', 'results'],
        'digest': ['digest', 'digests'],
        'bundle': ['bundle', 'bundles'],
        'artifact': ['artifact', 'artifacts'],
        'step': ['step', 'steps'],
        'command': ['command', 'commands'],
    }
    
    # Check which keywords are mentioned in instruction
    for base_keyword, variants in keyword_patterns.items():
        if any(variant in instruction_lower for variant in variants):
            keywords.extend(variants)
    
    # Also extract quoted strings (might be specific field names)
    import re
    quoted = re.findall(r'["\']([^"\']+)["\']', instruction)
    keywords.extend([q.lower() for q in quoted])
    
    return list(set(keywords))  # Remove duplicates


def _find_paths_to_keywords(obj, keywords: List[str], current_path: List[str] = None, max_depth: int = 6) -> List[List[str]]:
    """Recursively find all paths in JSON that lead to keys matching keywords.
    
    Args:
        obj: Current JSON object
        keywords: List of keywords to search for (normalized to lowercase)
        current_path: Current path being explored
        max_depth: Maximum depth to recurse
        
    Returns:
        List of paths (each path is a list of keys)
    """
    if current_path is None:
        current_path = []
    
    if max_depth <= 0:
        return []
    
    found_paths = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_lower = key.lower()
            # Check if key matches any keyword (exact match or contains keyword)
            if any(kw in key_lower or key_lower in kw for kw in keywords):
                found_paths.append(current_path + [key])
            
            # Recurse into nested structures
            new_path = current_path + [key]
            if isinstance(value, (dict, list)):
                found_paths.extend(_find_paths_to_keywords(value, keywords, new_path, max_depth - 1))
    
    elif isinstance(obj, list):
        # For arrays, check first few items
        for i, item in enumerate(obj[:3]):  # Check first 3 items
            if isinstance(item, (dict, list)):
                found_paths.extend(_find_paths_to_keywords(item, keywords, current_path, max_depth - 1))
    
    return found_paths


def _find_relevant_paths(instruction: str, data: dict) -> List[tuple]:
    """Dynamically find relevant paths in the data based on instruction keywords.
    
    Searches the JSON structure for keys that match keywords from the instruction,
    then extracts example values from those paths.
    
    Returns:
        List of (path_parts, example_values) tuples
    """
    # Extract keywords from instruction
    keywords = _extract_keywords_from_instruction(instruction)
    
    if not keywords:
        return []
    
    # Find all paths that match keywords
    all_paths = _find_paths_to_keywords(data, keywords, max_depth=6)
    
    # Remove duplicate paths and sort by relevance (shorter paths first, then by keyword match)
    unique_paths = []
    seen = set()
    for path in all_paths:
        path_str = ".".join(path)
        if path_str not in seen:
            seen.add(path_str)
            unique_paths.append(path)
    
    # Sort by length (shorter = more specific) and limit to most relevant
    unique_paths.sort(key=len)
    unique_paths = unique_paths[:10]  # Limit to top 10 paths
    
    # Extract examples for each path
    relevant_paths = []
    for path_parts in unique_paths:
        example_values = _extract_example_values(data, path_parts, max_examples=3)
        if example_values:
            relevant_paths.append((path_parts, example_values))
    
    return relevant_paths


def inspect_attestation_structure_tree(attestation_files: List[Path], instruction: str = "") -> str:
    """Generate a tree structure of the attestation JSON using Python.
    
    Recursively walks the JSON structure and shows all keys in a tree format.
    Also extracts example values for paths mentioned in the instruction.
    
    Args:
        attestation_files: List of attestation file paths
        instruction: Optional instruction text to extract relevant examples
    
    Returns:
        Tree structure string with examples, or empty string if file can't be read
    """
    if not attestation_files:
        return ""
    
    # Use the first attestation file
    att_file = attestation_files[0]
    if not att_file.exists():
        return ""
    
    try:
        import json
        with open(att_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle wrapped format (input.attestations)
        if isinstance(data, dict) and "attestations" in data:
            data = data["attestations"][0] if isinstance(data["attestations"], list) and len(data["attestations"]) > 0 else data
        elif isinstance(data, list) and len(data) > 0:
            data = data[0]
        
        tree_output = _tree_structure(data, max_depth=4).strip()
        
        # Extract example values for relevant paths mentioned in instruction
        example_sections = []
        relevant_paths = []
        if instruction:
            relevant_paths = _find_relevant_paths(instruction, data)
            for path_parts, example_values in relevant_paths:
                path_str = " → ".join(path_parts)
                example_sections.append(f"EXAMPLE VALUES for path '{path_str}':\n{example_values}")
        
        # Combine structure and examples
        output_parts = [f"ATTESTATION STRUCTURE (from {att_file.name}):\n{tree_output}"]
        
        # Build dynamic guidance based on discovered paths
        guidance_parts = []
        if instruction and relevant_paths:
            guidance_parts.append("IMPORTANT: Based on your instruction, the following paths were found:")
            for path_parts, example_values in relevant_paths:
                path_str = " → ".join(path_parts)
                # Check if this path ends in an array by examining the example
                is_array = example_values.strip().startswith("[") or "array" in example_values.lower()
                if is_array:
                    guidance_parts.append(f"- Path '{path_str}' → This is an ARRAY. See example values below.")
                else:
                    guidance_parts.append(f"- Path '{path_str}' → See example values below.")
            guidance_parts.append("")
        
        if example_sections:
            output_parts.append("\n" + "\n\n".join(example_sections))
        
        guidance_parts.append("""
This shows the JSON structure. Use this to understand:
- Which paths exist in the attestation
- Whether fields are arrays (shown as [])
- The nesting level of fields
- The actual structure of array items (see examples above)

For Rego code:
- If the file has an 'attestations' array at the top, use: some att in input.attestations
- If the file is a single attestation object, use: input directly
- Navigate using dot notation based on the structure above
- Iterate arrays with: some item in array; item
- If an array contains a list of objects, use: some item in array; item.field == value
- If an array contains a list of strings, use: some item in array; item == value
""")
        
        if guidance_parts:
            output_parts.append("\n" + "\n".join(guidance_parts))
        
        full_output = "\n".join(output_parts)
        
        # Limit output to avoid overwhelming the model
        if len(full_output) > 3000:
            full_output = full_output[:3000] + "\n... (truncated)"
        
        return full_output
    except (json.JSONDecodeError, IOError, Exception) as e:
        # File read or JSON parse failed, return empty
        import sys
        import os
        if os.getenv("VERBOSE_DEBUG", "false").lower() == "true":
            print(f"DEBUG: Failed to read/parse attestation file: {e}", file=sys.stderr)
        return ""


def generate_plan(
    tokenizer, model, device, instruction: str, context: str = None,
    attestation_files: List[Path] = None, verbose: bool = False
) -> str:
    """Generate a structured plan for implementing the instruction.
    
    Returns a plan that includes:
    - Understanding of requirements
    - Approach/strategy
    - Relevant helpers/patterns to use
    - Expected structure
    """
    # Inspect attestation structure if files are provided and instruction mentions attestations
    # Only use the first file for structure inspection (all attestations should have same structure)
    attestation_structure = ""
    if attestation_files and any(kw in instruction.lower() for kw in ['attestation', 'task', 'subject', 'material', 'build', 'result']):
        try:
            # Only use first file for structure (they should all have the same structure)
            structure_files = [attestation_files[0]] if attestation_files else []
            attestation_structure = inspect_attestation_structure_tree(structure_files, instruction=instruction)
            if verbose and attestation_structure:
                print(f"  ✓ Attestation structure extracted from 1 file ({len(attestation_structure)} chars)")
            elif verbose and not attestation_structure:
                print(f"  ⚠ Attestation structure extraction returned empty")
        except Exception as e:
            # If extraction fails, continue without structure info
            import sys
            import os
            if verbose:
                print(f"  ⚠ Failed to extract attestation structure: {e}")
            if os.getenv("VERBOSE_DEBUG", "false").lower() == "true":
                print(f"DEBUG: structure inspection failed: {e}", file=sys.stderr)
            pass
    
    planning_prompt = f"""Analyze this instruction and create a detailed implementation plan. DO NOT write code yet - only create a plan.

Instruction: {instruction}

{context if context else ""}

{attestation_structure if attestation_structure else ""}

Create a structured plan using this format:

## 1. Requirements Analysis
- What is the instruction asking for?
- What keywords are mentioned? (task, material, subject, result, etc.)
- What conditions need to be checked?

## 2. Data Structure Navigation
{f"- Based on the attestation structure above, identify the correct JSON paths" if attestation_structure else "- Identify the JSON paths needed"}
- Which arrays need to be iterated?
- What fields need to be accessed?
{f"- Pay attention to EXAMPLE VALUES: Are any fields arrays? If so, they must be iterated with 'some item in array'" if attestation_structure else ""}

## 3. Rego Patterns Needed
- What rule type? (deny/warn/allow)
- What iteration patterns? (some, every)
- What comparison operators?
- Any helper functions needed?

## 4. Implementation Strategy
- Step-by-step approach
- Order of conditions
- Variable naming strategy
- Error handling considerations

## 5. Potential Challenges
- What could go wrong?
- Common pitfalls to avoid
{f"- Array iteration: Remember arrays must use 'some item in array', not direct field access" if attestation_structure else ""}
- Variable scoping issues

{f"CRITICAL NOTES:" if attestation_structure else ""}
{f"- Match instruction keywords to the structure paths shown above" if attestation_structure else ""}
{f"- If you see arrays in the example values, you MUST iterate them" if attestation_structure else ""}
{f"- DO NOT write code - only describe the plan" if True else ""}

Provide ONLY the plan, no code."""
    
    # Example of a good plan (few-shot example)
    example_plan = """## 1. Requirements Analysis
- Instruction asks to check all tasks for a result named 'commit'
- Keywords: "task", "result", "commit"
- Need to verify that every task has a result with name 'commit'

## 2. Data Structure Navigation
- Path: input.attestations[].predicate.buildConfig.tasks[]
- tasks is an array - must iterate with 'some task in ...'
- Each task has a results array - must iterate with 'some result in task.results'
- Check result.name == "commit"

## 3. Rego Patterns Needed
- Rule type: deny (or warn, depending on requirement)
- Use 'every task in ...' for "all tasks" requirement
- Use 'some result in task.results' to iterate results array
- Comparison: result.name == "commit"

## 4. Implementation Strategy
- Iterate through all tasks using 'every'
- For each task, iterate through results
- Check if any result has name "commit"
- If missing, create deny result

## 5. Potential Challenges
- Variable scoping: Use different variable names for iteration vs assignment
- Array iteration: Must use 'some result in task.results', not 'task.results.commit'
- Input path: Must start with 'input.attestations'"""
    
    messages = [
        {"role": "system", "content": "You are a Rego policy planning assistant. Your job is to create detailed, structured plans for implementing Rego policies. You should analyze requirements, identify data structures, and plan the implementation approach. DO NOT write code - only create plans. Use clear sections and bullet points."},
        {"role": "user", "content": "Example instruction: Write a rule that checks all tasks for a result named 'commit'"},
        {"role": "assistant", "content": example_plan},
        {"role": "user", "content": planning_prompt}
    ]
    
    plan = generate_response(tokenizer, model, device, messages, max_tokens=1024, temperature=0.3)
    
    # Post-process: Remove code blocks if model outputs code instead of plan
    # Check if plan contains code blocks
    if "```" in plan:
        # Try to extract just the plan part (before code blocks)
        parts = plan.split("```")
        if len(parts) > 0:
            # Take the part before the first code block
            plan_text = parts[0].strip()
            # If there's substantial text before code, use it
            if len(plan_text) > 100:
                plan = plan_text
            else:
                # If code is the main content, warn but keep it (better than nothing)
                if verbose:
                    print("  ⚠ Warning: Plan contains code blocks, but keeping for now")
    
    return plan


def check_style_compliance(code: str) -> List[str]:
    """Check style guide compliance using Regal and custom checks."""
    violations = []
    
    # Try Regal lint (import from validate_and_improve_dataset if available)
    try:
        from validate_and_improve_dataset import validate_with_regal
        regal_ok, regal_issues = validate_with_regal(code)
        if not regal_ok and regal_issues:
            # Filter out violations we want to ignore
            ignored_patterns = [
                "Directory structure should mirror package",
                "directory-structure-should-mirror-package",  # Rule ID format
            ]
            for issue in regal_issues:
                # Skip if issue matches any ignored pattern
                if not any(pattern.lower() in issue.lower() for pattern in ignored_patterns):
                    violations.append(issue)
    except (ImportError, Exception):
        # Regal not available or import failed, skip
        pass
    
    
    
    return violations


def check_execution_against_attestations(
    code: str,
    attestation_files: List[Path],
    package: str = None,
    imports: List[str] = None,
    max_files: int = 3
) -> Tuple[List[str], List[str]]:
    """Test Rego code against real attestation JSON files.
    
    Args:
        code: Rego code to test
        attestation_files: List of paths to attestation JSON files
        package: Package name for code
        imports: List of imports
        max_files: Maximum number of files to test (for performance)
        
    Returns:
        (list_of_errors, list_of_tested_file_names)
        
    Uses `opa eval` to execute the code against wrapped attestation data.
    Checks for runtime errors (undefined references, type errors, etc.).
    """
    import json  # Import at function level to avoid UnboundLocalError
    errors = []
    tested_files = []
    
    # Limit number of files to test
    files_to_test = attestation_files[:max_files]
    
    # Determine OPA command (try 'ec opa' first, fall back to 'opa')
    opa_base = ["opa"]
    try:
        result = subprocess.run(
            ["ec", "opa", "--version"],
            capture_output=True,
            timeout=1,
            text=True
        )
        if result.returncode == 0:
            opa_base = ["ec", "opa"]
    except:
        pass
    
    # Build complete code with package/imports
    complete_code = code
    if package and f"package {package}" not in code:
        code_parts = [f"package {package}\n"]
        if imports:
            code_parts.append("import rego.v1\n")
            for imp in imports:
                if not imp.startswith("rego.v1") and f"import {imp}" not in code:
                    code_parts.append(f"import {imp}\n")
        code_parts.append("\n")
        code_parts.append(code)
        complete_code = "".join(code_parts)
    
    # Final safety check: remove ALL backticks from complete_code
    if '`' in complete_code:
        print(f"WARNING: Execution check - backticks found in complete_code! Removing them.")
        complete_code = complete_code.replace('`', '')
    
    # Write Rego code to temp file (preserve for debugging)
    import time
    timestamp = int(time.time() * 1000)
    rego_path = Path(tempfile.gettempdir()) / f"rego_exec_{timestamp}.rego"
    with open(rego_path, 'w', encoding='utf-8') as rego_file:
        rego_file.write(complete_code)
        rego_file.flush()
    
    # Debug: check for backticks
    if '`' in complete_code:
        print(f"DEBUG: Execution check - backtick found in code!")
        print(f"DEBUG: Temp file: {rego_path}")
        print(f"DEBUG: First 200 chars: {repr(complete_code[:200])}")
    
    try:
        # Test against each attestation file
        for att_file in files_to_test:
            if not att_file.exists():
                continue
            
            tested_files.append(att_file.name)
            
            try:
                # Read and wrap attestation in expected format
                with open(att_file, 'r') as f:
                    att_data = json.load(f)
                
                # Wrap in input.attestations format
                # If file already has attestations array, use it; otherwise wrap single object
                if isinstance(att_data, list):
                    wrapped_input = {"attestations": att_data}
                elif "attestations" in att_data:
                    wrapped_input = att_data
                else:
                    # Single attestation object, wrap it
                    wrapped_input = {"attestations": [att_data]}
                
                # Write input to temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
                    input_path = Path(input_file.name)
                    json.dump(wrapped_input, input_file)
                    input_file.flush()
                
                try:
                    # Evaluate code against input
                    # Query for 'deny' or 'warn' or any rule that might exist
                    # We'll try multiple queries to see if code executes
                    queries_to_try = [
                        "data.deny",
                        "data.warn", 
                        "data.allow",
                        "data"  # Catch-all to see if anything evaluates
                    ]
                    
                    execution_errors = []
                    any_query_worked = False
                    
                    for query in queries_to_try:
                        try:
                            result = subprocess.run(
                                opa_base + [
                                    "eval",
                                    "--data", str(rego_path),
                                    "--input", str(input_path),
                                    query,
                                    "--format", "json"
                                ],
                                capture_output=True,
                                timeout=5,
                                text=True
                            )
                            
                            if result.returncode == 0:
                                any_query_worked = True
                                break  # At least one query worked
                            else:
                                # Check if it's a "not found" error (acceptable)
                                error_output = result.stderr or result.stdout
                                if "undefined" not in error_output.lower():
                                    # Try to parse JSON error for better message
                                    error_msg = error_output
                                    try:
                                        error_json = json.loads(error_output)
                                        if isinstance(error_json, dict) and "errors" in error_json:
                                            errors_list = error_json["errors"]
                                            if errors_list:
                                                first_error = errors_list[0]
                                                msg = first_error.get("message", "")
                                                location = first_error.get("location", {})
                                                row = location.get("row", "")
                                                col = location.get("col", "")
                                                code_type = first_error.get("code", "")
                                                
                                                # Build clearer error message
                                                if row:
                                                    error_msg = f"{msg} (line {row}"
                                                    if col:
                                                        error_msg += f", col {col}"
                                                    error_msg += ")"
                                                else:
                                                    error_msg = msg
                                                
                                                if code_type:
                                                    error_msg = f"[{code_type}] {error_msg}"
                                    except (json.JSONDecodeError, KeyError, TypeError):
                                        # Not JSON or unexpected format, use truncated original
                                        error_msg = error_output[:500]  # Increased from 200
                                    
                                    # Real error, not just "rule not found"
                                    execution_errors.append(
                                        f"Query '{query}' failed: {error_msg}"
                                    )
                        except subprocess.TimeoutExpired:
                            execution_errors.append(f"Query '{query}' timed out")
                        except Exception as e:
                            execution_errors.append(f"Query '{query}' error: {e}")
                    
                    # If no queries worked and we have errors, report them
                    if not any_query_worked and execution_errors:
                        # Only report if it's a real execution error
                        # (undefined rules are okay - code might not define deny/warn)
                        real_errors = [e for e in execution_errors if "undefined" not in e.lower()]
                        if real_errors:
                            errors.append(
                                f"{att_file.name}: {real_errors[0]}"
                            )
                
                finally:
                    try:
                        input_path.unlink()
                    except:
                        pass
            
            except json.JSONDecodeError as e:
                errors.append(f"{att_file.name}: Invalid JSON - {e}")
            except Exception as e:
                errors.append(f"{att_file.name}: Error loading file - {e}")
    
    finally:
        # PRESERVE temp file for debugging
        print(f"DEBUG: Execution check temp file preserved at: {rego_path}")
        # Uncomment to auto-delete:
        # try:
        #     rego_path.unlink()
        # except:
        #     pass
    
    return errors, tested_files


def check_code_comprehensively(
    code: str,
    instruction: str,
    package: str = None,
    imports: List[str] = None,
    include_style: bool = True,
    include_execution: bool = True,
    attestation_files: List[Path] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Perform comprehensive validation across multiple layers.
    
    Returns:
        (is_valid, validation_results)
        validation_results contains:
        - syntax: {valid, error_msg, formatted_code}
        - style: {valid, violations: []}
        - execution: {valid, errors: [], tested_files: []}
    """
    from rego_validator import validate_rego_syntax
    
    results = {
        "syntax": {"valid": False, "error_msg": "", "formatted_code": code},
        "semantic": {"valid": True, "issues": []},  # Always valid (semantic check removed)
        "style": {"valid": True, "violations": []},
        "execution": {"valid": True, "errors": [], "tested_files": []}
    }
    
    # 1. Syntax check
    is_valid, formatted_code, error_msg = validate_rego_syntax(
        code, package=package or "", imports=imports or []
    )
    results["syntax"] = {
        "valid": is_valid,
        "error_msg": error_msg,
        "formatted_code": formatted_code
    }
    
    if not is_valid:
        return False, results
    
    # 2. Semantic check (removed - not helpful)
    results["semantic"] = {
        "valid": True,
        "issues": []
    }
    
    # 3. Style check (optional)
    if include_style:
        style_violations = check_style_compliance(code)
        results["style"] = {
            "valid": len(style_violations) == 0,
            "violations": style_violations
        }
    
    # 4. Execution check (optional, only for attestation-related code)
    if include_execution and attestation_files:
        execution_errors, tested_files = check_execution_against_attestations(
            code, attestation_files, package=package, imports=imports
        )
        results["execution"] = {
            "valid": len(execution_errors) == 0,
            "errors": execution_errors,
            "tested_files": tested_files
        }
    
    # Overall validity: syntax and execution must pass; style is optional
    overall_valid = (
        results["syntax"]["valid"] and 
        results["execution"]["valid"]
    )
    
    return overall_valid, results


def generate_repair(
    tokenizer, model, device,
    instruction: str,
    plan: str,
    current_code: str,
    validation_results: Dict[str, Any],
    iteration: int,
    max_iterations: int
) -> str:
    """Generate a repair for the code based on validation results.
    
    Uses the format the model was trained on for error correction:
    
    Error: [error_code] error message (line X, column Y)
    Error type: type
    
    Instruction: <original instruction>
    
    The following Rego code has compilation errors:
    ```rego
    <broken code>
    ```
    
    Please fix the compilation errors and provide the corrected code.
    """
    import json
    import re
    
    # Build error string in training data format
    error_lines = []
    error_type = "unknown"
    
    # 1. SYNTAX ERRORS - format exactly like training data
    if not validation_results["syntax"]["valid"]:
        error_msg = validation_results["syntax"]["error_msg"]
        
        # Try to parse JSON error format
        try:
            error_data = json.loads(error_msg)
            if isinstance(error_data, dict) and "errors" in error_data:
                for err in error_data["errors"]:
                    code = err.get("code", "rego_error")
                    msg = err.get("message", "")
                    location = err.get("location", {})
                    row = location.get("row", "")
                    col = location.get("col", "")
                    
                    if row and col:
                        error_lines.append(f"[{code}] {msg} (line {row}, column {col})")
                    elif row:
                        error_lines.append(f"[{code}] {msg} (line {row})")
                    else:
                        error_lines.append(f"[{code}] {msg}")
                    
                    # Determine error type from code
                    if "parse" in code:
                        error_type = "parse_error"
                    elif "unsafe" in code or "type" in code:
                        error_type = "type_error"
                    elif "compile" in code:
                        error_type = "compile_error"
        except (json.JSONDecodeError, KeyError, TypeError):
            # Not JSON, try to extract from raw error message
            error_lines.append(error_msg[:300])
            error_type = "syntax_error"
    
    # 2. EXECUTION ERRORS - format like training data
    if not validation_results["execution"]["valid"]:
        exec_errors = validation_results["execution"]["errors"]
        for err in exec_errors[:5]:
            # Extract error code and message from execution error
            # Format: "file.json: Query 'data.deny' failed: [rego_unsafe_var_error] var result is unsafe (line 5, col 1)"
            match = re.search(r'\[(\w+)\]\s*(.+?)(?:\s*\(line\s*(\d+)(?:,\s*col(?:umn)?\s*(\d+))?\))?$', err)
            if match:
                code = match.group(1)
                msg = match.group(2).strip()
                row = match.group(3)
                col = match.group(4)
                
                if row and col:
                    error_lines.append(f"[{code}] {msg} (line {row}, column {col})")
                elif row:
                    error_lines.append(f"[{code}] {msg} (line {row})")
                else:
                    error_lines.append(f"[{code}] {msg}")
                
                # Update error type
                if "unsafe" in code:
                    error_type = "type_error"
                elif "compile" in code:
                    error_type = "compile_error"
            else:
                # Fallback: include the full error
                error_lines.append(err)
    
    # Format errors exactly like training data
    if len(error_lines) == 1:
        error_string = f"Error: {error_lines[0]}"
    else:
        error_string = "Error: " + "\n".join(f"{i}. {e}" for i, e in enumerate(error_lines, 1))
    
    # Build the repair prompt in EXACT training data format
    repair_prompt = f"""{error_string}
Error type: {error_type}

Instruction: {instruction}

The following Rego code has compilation errors:
```rego
{current_code}
```

Please fix the compilation errors and provide the corrected code."""
    
    # Detect task type for system prompt
    is_attestation_task = any(kw in instruction.lower() for kw in [
        'attestation', 'task', 'subject', 'material'
    ])
    system_prompt = QWEN_SYSTEM_PROMPT_ATTESTATION if is_attestation_task else QWEN_SYSTEM_PROMPT
    
    # Single-turn format matching training data structure
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": repair_prompt}
    ]
    
    # Use slightly lower temperature for repair to reduce numerical instability
    # Also add retry logic if generation fails
    try:
        repair = generate_response(tokenizer, model, device, messages, max_tokens=1024, temperature=0.3)
    except RuntimeError as e:
        error_msg = str(e)
        if "inf" in error_msg or "nan" in error_msg or "probability tensor" in error_msg:
            # Try with even lower temperature
            try:
                repair = generate_response(tokenizer, model, device, messages, max_tokens=1024, temperature=0.1)
            except RuntimeError:
                # Last resort: use greedy decoding (temperature=0 with do_sample=False)
                # We need to call generate directly with do_sample=False
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        do_sample=False,  # Greedy decoding
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
                assistant_text = tokenizer.apply_chat_template(
                    messages + [{"role": "assistant", "content": ""}],
                    tokenize=False,
                    add_generation_prompt=True
                )
                if generated_text.startswith(assistant_text):
                    repair = generated_text[len(assistant_text):].strip()
                else:
                    if "<|im_start|>assistant" in generated_text:
                        parts = generated_text.split("<|im_start|>assistant")
                        if len(parts) > 1:
                            repair = parts[-1].split("<|im_end|>")[0].strip()
                        else:
                            repair = generated_text
                    else:
                        repair = generated_text
                repair = repair.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
        else:
            raise
    return repair




def find_attestation_files(repo_root: Path, max_files: int = 5) -> List[Path]:
    """Find attestation JSON files in the repository root.
    
    Only looks for attestation.json in the repo root.
    
    Returns:
        List of Path objects to attestation files (empty if file doesn't exist)
    """
    attestation_file = repo_root / "attestation.json"
    
    if attestation_file.exists():
        return [attestation_file]
    
    return []


def build_implementation_messages(
    instruction: str, context: str = None, plan: str = None, 
    previous_errors: List[str] = None, is_regeneration: bool = False
) -> List[Dict]:
    """Build messages for implementation phase.
    
    Args:
        instruction: The user's instruction
        context: Context (package, imports, helpers)
        plan: The generated plan
        previous_errors: Errors from previous attempts (for regeneration)
        is_regeneration: Whether this is a regeneration after failed repair
    """
    # Detect task type
    is_attestation_task = any(kw in instruction.lower() for kw in [
        'attestation', 'task', 'subject', 'material'
    ])
    
    system_prompt = (
        QWEN_SYSTEM_PROMPT_ATTESTATION if is_attestation_task 
        else QWEN_SYSTEM_PROMPT
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Build user content
    parts = []
    if context:
        parts.append(context)
    if plan:
        parts.append(f"Plan:\n{plan}")
    
    # If regenerating after failed repair, include previous errors
    if is_regeneration and previous_errors:
        error_summary = "\n".join(f"- {err}" for err in previous_errors[-5:])  # Last 5 errors
        parts.append(f"Previous attempt had these errors (please avoid them):\n{error_summary}")
    
    parts.append(f"Instruction: {instruction}")
    
    # Add reminder about matching keywords to paths
    if is_attestation_task:
        keyword_reminder = "IMPORTANT: Match instruction keywords to paths - "
        keywords_found = []
        if 'task' in instruction.lower():
            keywords_found.append("'task' → predicate.buildConfig.tasks[]")
        if 'material' in instruction.lower():
            keywords_found.append("'material' → predicate.materials[]")
        if 'subject' in instruction.lower():
            keywords_found.append("'subject' → subject[]")
        if keywords_found:
            parts.append(keyword_reminder + ", ".join(keywords_found))
    
    parts.append("\nPlease provide the Rego code in a code block (```rego ... ```).")
    
    user_content = "\n\n".join(parts)
    messages.append({"role": "user", "content": user_content})
    
    return messages


def agentic_inference(
    tokenizer, model, device,
    instruction: str,
    context: str = None,
    package: str = None,
    imports: List[str] = None,
    max_iterations: int = 5,
    include_planning: bool = True,
    include_style_check: bool = True,
    include_execution_check: bool = True,
    attestation_files: List[Path] = None,
    verbose: bool = True
) -> Tuple[str, AgentState]:
    """Run agentic inference with Plan → Implement → Check → Repair loop.
    
    Returns:
        (final_code, agent_state)
    """
    state = AgentState()
    
    if verbose:
        print("=" * 60)
        print("AGENTIC INFERENCE WORKFLOW")
        print("=" * 60)
        print()
    
    # PHASE 1: PLANNING
    if include_planning:
        if verbose:
            print("📋 Phase 1: Planning...")
            if attestation_files:
                print(f"  (Inspecting structure from 1 attestation file)")
        try:
            state.plan = generate_plan(tokenizer, model, device, instruction, context, attestation_files=attestation_files, verbose=verbose)
            if verbose:
                print("✓ Plan generated")
                print("-" * 60)
                print(state.plan)
                print("-" * 60)
                print()
        except Exception as e:
            if verbose:
                print(f"⚠ Planning failed: {e}, continuing without plan")
            state.plan = None
    
    # PHASE 2-5: IMPLEMENT → CHECK → REPAIR LOOP
    while state.iteration < max_iterations:
        state.iteration += 1
        
        if verbose:
            print(f"🔄 Iteration {state.iteration}/{max_iterations}")
        
        # PHASE 2: IMPLEMENTATION
        # Only generate new code on first iteration or if we don't have repaired code
        # On subsequent iterations, reuse the repaired code from previous iteration
        if state.iteration == 1 or not state.implementation:
            is_regeneration = (state.iteration > 1 and not state.implementation)
            if verbose:
                if is_regeneration:
                    print("  📝 Phase 2: Regenerating implementation (previous repair failed)...")
                else:
                    print("  📝 Phase 2: Implementation...")
            
            # Build messages for implementation
            # Include previous errors if regenerating after failed repair
            messages = build_implementation_messages(
                instruction, context, state.plan,
                previous_errors=state.errors if is_regeneration else None,
                is_regeneration=is_regeneration
            )
            implementation_response = generate_response(
                tokenizer, model, device, messages, max_tokens=1024, temperature=0.7
            )
            
            # Extract code
            from rego_validator import extract_rego_code
            code = extract_rego_code(implementation_response)
            
            if not code:
                if verbose:
                    print("  ⚠ No Rego code found in response")
                state.errors.append("No Rego code found in model response")
                if state.iteration == 1:
                    # First iteration, return the response as-is
                    return implementation_response, state
                continue
            
            state.implementation = code
            if verbose:
                print(f"  Generated code ({len(code)} chars):")
                print("  " + "-" * 56)
                # Print code with indentation, limit to first 500 chars if too long
                code_preview = code[:500] + "..." if len(code) > 500 else code
                for line in code_preview.split('\n'):
                    print(f"  {line}")
                if len(code) > 500:
                    print(f"  ... ({len(code) - 500} more characters)")
                print("  " + "-" * 56)
        else:
            # Reuse repaired code from previous iteration
            if verbose:
                print("  📝 Phase 2: Using repaired code from previous iteration...")
                print(f"  Reusing code from previous repair ({len(state.implementation)} chars):")
                print("  " + "-" * 56)
                code_preview = state.implementation[:500] + "..." if len(state.implementation) > 500 else state.implementation
                for line in code_preview.split('\n'):
                    print(f"  {line}")
                if len(state.implementation) > 500:
                    print(f"  ... ({len(state.implementation) - 500} more characters)")
                print("  " + "-" * 56)
            code = state.implementation
        
        # PHASE 3: CHECKING
        if verbose:
            print("  🔍 Phase 3: Checking...")
        
        is_valid, validation_results = check_code_comprehensively(
            code, instruction, package, imports, 
            include_style=include_style_check,
            include_execution=include_execution_check,
            attestation_files=attestation_files
        )
        
        state.syntax_valid = validation_results["syntax"]["valid"]
        state.semantic_valid = True  # Always true (semantic check removed)
        state.execution_valid = validation_results["execution"]["valid"]
        state.style_valid = validation_results["style"]["valid"]
        
        # Calculate score for this iteration (to track best code)
        # Prioritize syntax validity - code with valid syntax is always better
        # Score breakdown: syntax=100, execution=1, style=1 (max=102)
        # This ensures valid syntax always beats invalid syntax by a huge margin
        current_score = (
            (100 if state.syntax_valid else 0) +
            (1 if state.execution_valid else 0) +
            (1 if state.style_valid else 0)
        )
        
        # Track best code seen so far
        # Always prefer code with valid syntax
        if current_score > state.best_score:
            state.best_code = code
            state.best_score = current_score
            if verbose:
                valid_checks = []
                if state.syntax_valid:
                    valid_checks.append("syntax")
                if state.execution_valid:
                    valid_checks.append("execution")
                if state.style_valid:
                    valid_checks.append("style")
                checks_str = ", ".join(valid_checks) if valid_checks else "none"
                print(f"    📊 New best code (score: {current_score}/102, valid: {checks_str})")
        
        # Collect errors and warnings
        if not state.syntax_valid:
            state.errors.append(f"Syntax: {validation_results['syntax']['error_msg']}")
        if not state.execution_valid:
            state.errors.extend(validation_results["execution"]["errors"])
        if not state.style_valid:
            state.warnings.extend(validation_results["style"]["violations"])
        
        if verbose:
            print(f"    Syntax: {'✓' if state.syntax_valid else '✗'}")
            if not state.syntax_valid:
                error_msg = validation_results["syntax"]["error_msg"]
                # Try to extract a concise error message
                if error_msg:
                    # If it's JSON, try to parse it
                    try:
                        import json
                        error_data = json.loads(error_msg)
                        if isinstance(error_data, dict) and "errors" in error_data:
                            errors = error_data["errors"]
                            if errors:
                                first_error = errors[0]
                                msg = first_error.get("message", str(first_error))
                                location = first_error.get("location", {})
                                if location:
                                    row = location.get("row", "")
                                    col = location.get("col", "")
                                    if row:
                                        print(f"      Error: {msg} (line {row}, col {col})")
                                    else:
                                        print(f"      Error: {msg}")
                                else:
                                    print(f"      Error: {msg}")
                        else:
                            print(f"      Error: {error_msg[:200]}")
                    except:
                        # Not JSON, print as-is (truncated)
                        print(f"      Error: {error_msg[:200]}")
                else:
                    print(f"      Error: Unknown syntax error")
            
            print(f"    Execution: {'✓' if state.execution_valid else '✗'}")
            if not state.execution_valid:
                exec_errors = validation_results["execution"]["errors"]
                if exec_errors:
                    print(f"      Errors:")
                    for err in exec_errors[:3]:  # Show first 3 errors
                        print(f"        - {err}")
                    if len(exec_errors) > 3:
                        print(f"        ... and {len(exec_errors) - 3} more")
            if validation_results["execution"].get("tested_files"):
                print(f"      (tested against {len(validation_results['execution']['tested_files'])} attestation files)")
            
            print(f"    Style: {'✓' if state.style_valid else '⚠'}")
            if not state.style_valid:
                style_violations = validation_results["style"]["violations"]
                if style_violations:
                    print(f"      Violations:")
                    for violation in style_violations[:3]:  # Show first 3 violations
                        print(f"        - {violation}")
                    if len(style_violations) > 3:
                        print(f"        ... and {len(style_violations) - 3} more")
        
        # PHASE 4: SUCCESS
        if is_valid:
            if verbose:
                print("  ✓ All checks passed!")
                print()
            
            # Use formatted code if available
            final_code = validation_results["syntax"]["formatted_code"]
            return final_code, state
        
        # PHASE 5: REPAIR
        if state.iteration < max_iterations:
            if verbose:
                print("  🔧 Phase 5: Repairing...")
                # Show the prompt being sent to the model (matches training data format)
                print("  Repair prompt (training data format):")
                print("  " + "-" * 56)
                
                # Build preview of the repair prompt in training format
                import json
                import re as re_module
                error_lines = []
                error_type = "unknown"
                
                # Syntax errors
                if not validation_results["syntax"]["valid"]:
                    error_msg = validation_results["syntax"]["error_msg"]
                    try:
                        error_data = json.loads(error_msg)
                        if isinstance(error_data, dict) and "errors" in error_data:
                            for err in error_data["errors"]:
                                err_code = err.get("code", "rego_error")
                                msg = err.get("message", "")
                                location = err.get("location", {})
                                row = location.get("row", "")
                                col = location.get("col", "")
                                if row and col:
                                    error_lines.append(f"[{err_code}] {msg} (line {row}, column {col})")
                                elif row:
                                    error_lines.append(f"[{err_code}] {msg} (line {row})")
                                else:
                                    error_lines.append(f"[{err_code}] {msg}")
                                if "parse" in err_code:
                                    error_type = "parse_error"
                                elif "unsafe" in err_code or "type" in err_code:
                                    error_type = "type_error"
                    except:
                        error_lines.append(error_msg[:200])
                
                # Execution errors
                if not validation_results["execution"]["valid"]:
                    exec_errors = validation_results["execution"]["errors"]
                    for err in exec_errors[:3]:
                        match = re_module.search(r'\[(\w+)\]\s*(.+?)(?:\s*\(line\s*(\d+)(?:,\s*col(?:umn)?\s*(\d+))?\))?$', err)
                        if match:
                            err_code = match.group(1)
                            msg = match.group(2).strip()
                            row = match.group(3)
                            col = match.group(4)
                            if row and col:
                                error_lines.append(f"[{err_code}] {msg} (line {row}, column {col})")
                            elif row:
                                error_lines.append(f"[{err_code}] {msg} (line {row})")
                            else:
                                error_lines.append(f"[{err_code}] {msg}")
                            if "unsafe" in err_code:
                                error_type = "type_error"
                        else:
                            error_lines.append(err)
                
                # Show the formatted prompt
                if len(error_lines) == 1:
                    print(f"  Error: {error_lines[0]}")
                else:
                    print("  Error: " + "\n  ".join(f"{i}. {e}" for i, e in enumerate(error_lines, 1)))
                print(f"  Error type: {error_type}")
                print(f"  Instruction: {instruction[:100]}..." if len(instruction) > 100 else f"  Instruction: {instruction}")
                print("  [broken code block]")
                print("  Please fix the compilation errors and provide the corrected code.")
                print("  " + "-" * 56)
            
            try:
                repair_response = generate_repair(
                    tokenizer, model, device,
                    instruction, state.plan or "",
                    code, validation_results,
                    state.iteration, max_iterations
                )
                
                if verbose:
                    print(f"  Raw repair response ({len(repair_response)} chars):")
                    print("  " + "-" * 56)
                    response_preview = repair_response[:300] + "..." if len(repair_response) > 300 else repair_response
                    for line in response_preview.split('\n'):
                        print(f"  {line}")
                    if len(repair_response) > 300:
                        print(f"  ... ({len(repair_response) - 300} more characters)")
                    print("  " + "-" * 56)
                
                # Extract repaired code
                from rego_validator import has_meaningful_content
                repaired_code = extract_rego_code(repair_response)
                if repaired_code and has_meaningful_content(repaired_code):
                    state.implementation = repaired_code
                    if verbose:
                        print(f"  ✓ Repaired code extracted ({len(repaired_code)} chars):")
                        print("  " + "-" * 56)
                        code_preview = repaired_code[:500] + "..." if len(repaired_code) > 500 else repaired_code
                        for line in code_preview.split('\n'):
                            print(f"  {line}")
                        if len(repaired_code) > 500:
                            print(f"  ... ({len(repaired_code) - 500} more characters)")
                        print("  " + "-" * 56)
                    # Continue loop with repaired code
                    continue
                else:
                    if verbose:
                        print("  ⚠ No code found in repair response")
                        print(f"  Repair response preview: {repair_response[:200]}...")
                    state.errors.append("Repair response contained no code")
                    # Try to use the repair response as-is if it contains any Rego-like content
                    # This is a fallback for cases where extract_rego_code is too strict
                    if any(keyword in repair_response for keyword in ['package', 'deny', 'warn', 'allow', 'import']):
                        if verbose:
                            print("  Attempting to use repair response as-is (contains Rego keywords)")
                        # Try to extract anything that looks like code
                        # Look for content between code blocks or after "package"
                        code_candidates = []
                        # Try to find code block content even without proper markers
                        if '```' in repair_response:
                            parts = repair_response.split('```')
                            for i, part in enumerate(parts):
                                if i > 0 and i < len(parts) - 1:  # Content between ``` markers
                                    if 'package' in part or 'deny' in part or 'warn' in part:
                                        code_candidates.append(part.strip())
                        # If no code blocks, try to find package declaration onwards
                        if 'package' in repair_response and not code_candidates:
                            pkg_match = re.search(r'(package\s+\S+.*)', repair_response, re.DOTALL)
                            if pkg_match:
                                code_candidates.append(pkg_match.group(1).strip())
                        
                        if code_candidates:
                            # Use the longest candidate (most likely to be complete)
                            repaired_code = max(code_candidates, key=len)
                            state.implementation = repaired_code
                            if verbose:
                                print(f"  ✓ Using fallback extracted code ({len(repaired_code)} chars):")
                                print("  " + "-" * 56)
                                code_preview = repaired_code[:500] + "..." if len(repaired_code) > 500 else repaired_code
                                for line in code_preview.split('\n'):
                                    print(f"  {line}")
                                if len(repaired_code) > 500:
                                    print(f"  ... ({len(repaired_code) - 500} more characters)")
                                print("  " + "-" * 56)
                            continue
                    
                    # If all repair attempts failed, clear implementation to force regeneration on next iteration
                    # This prevents infinite loop of validating the same broken code
                    if verbose:
                        print("  ⚠ All repair attempts failed. Will try regenerating code on next iteration.")
                    state.implementation = None  # Clear to force regeneration
                    continue
            except RuntimeError as e:
                # Handle numerical instability errors during generation
                error_msg = str(e)
                if "inf" in error_msg or "nan" in error_msg or "probability tensor" in error_msg:
                    if verbose:
                        print(f"  ⚠ Repair generation failed due to numerical instability")
                        print(f"  Error: {error_msg}")
                        print(f"  Will try regenerating code on next iteration instead of reusing broken code.")
                    state.errors.append(f"Repair generation failed: numerical instability")
                    # Clear implementation to force regeneration on next iteration
                    # This prevents infinite loop of validating the same broken code
                    state.implementation = None
                    continue
                else:
                    if verbose:
                        print(f"  ⚠ Repair failed: {e}")
                    state.errors.append(f"Repair failed: {e}")
                    # Clear implementation to force regeneration on next iteration
                    state.implementation = None
                    continue
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Repair failed: {e}")
                state.errors.append(f"Repair failed: {e}")
                # Clear implementation to force regeneration on next iteration
                state.implementation = None
                continue
        else:
            if verbose:
                print("  ⚠ Max iterations reached")
            print()
    
    # Max iterations reached, return best attempt
    if verbose:
        print("=" * 60)
        print("⚠ Max iterations reached. Returning best attempt.")
        print("=" * 60)
        if state.best_code:
            # Determine which iteration had the best code
            best_has_syntax = state.best_score >= 100
            print(f"Returning best code seen (score: {state.best_score}/112, syntax: {'✓' if best_has_syntax else '✗'}, length: {len(state.best_code)} chars)")
            if state.best_code != state.implementation:
                print("Note: Best code differs from final iteration code")
        elif state.implementation:
            print(f"Returning code from iteration {state.iteration} (length: {len(state.implementation)} chars)")
        else:
            print("No code available to return")
    
    # Return the best code we've seen, or the last implementation, or empty string
    # Prefer best_code if it exists (even if invalid, it's better than nothing)
    # But filter out code that's only package/imports
    from rego_validator import has_meaningful_content
    
    final_code = None
    for candidate in [state.best_code, state.implementation]:
        if candidate and has_meaningful_content(candidate):
            final_code = candidate
            break
    
    # If no meaningful code found, return empty string with error
    if not final_code:
        if verbose:
            print("⚠ No meaningful code found (only package/imports). Returning empty result.")
        state.errors.append("No meaningful code generated (only package/imports, no actual rules)")
        return "", state
    
    # Format and validate the final code one more time
    # This ensures we return properly formatted code and only show errors from the final code
    final_validation_results = None
    if final_code:
        try:
            is_valid, validation_results = check_code_comprehensively(
                final_code, instruction, package, imports,
                include_style=include_style_check,
                include_execution=include_execution_check,
                attestation_files=attestation_files
            )
            final_validation_results = validation_results
            
            # Use formatted version if available (OPA fmt formats code properly)
            # The formatted_code from validate_rego_syntax uses opa fmt for proper indentation
            if validation_results["syntax"]["formatted_code"]:
                final_code = validation_results["syntax"]["formatted_code"]
            
            # Clear old errors and only keep errors from final code
            state.errors = []
            state.warnings = []
            
            # Only add errors if the final code is invalid
            if not validation_results["syntax"]["valid"]:
                error_msg = validation_results["syntax"]["error_msg"]
                # Try to extract a concise error message
                try:
                    import json
                    error_data = json.loads(error_msg)
                    if isinstance(error_data, dict) and "errors" in error_data:
                        for err in error_data["errors"]:
                            msg = err.get("message", "")
                            location = err.get("location", {})
                            row = location.get("row", "")
                            if row:
                                state.errors.append(f"Syntax: {msg} (line {row})")
                            else:
                                state.errors.append(f"Syntax: {msg}")
                    else:
                        state.errors.append(f"Syntax: {error_msg[:200]}")
                except:
                    state.errors.append(f"Syntax: {error_msg[:200]}")
            
            
            if not validation_results["execution"]["valid"]:
                state.errors.extend(validation_results["execution"]["errors"])
            
            if not validation_results["style"]["valid"]:
                state.warnings.extend(validation_results["style"]["violations"])
                
        except Exception as e:
            if verbose:
                print(f"Warning: Could not validate final code: {e}")
            # If validation fails, keep the code as-is
    
    return final_code, state


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run inference with policy rule model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (using fine-tuned model):
  # Full fine-tuned model (trained without PEFT/LoRA)
  uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \\
      --model-dir /path/to/full-fine-tuned-model \\
      --package tasks \\
      --instruction "Write a rule that checks if all tasks succeeded"

  # LoRA adapter model (trained with PEFT)
  uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \\
      --model-dir qwen2.5-rego-policy-lora \\
      --package tasks \\
      --instruction "Write a rule that checks if all tasks in a PipelineRun succeeded"

  # Interactive chat mode with fine-tuned model (best results)
  uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \\
      --model-dir qwen2.5-rego-policy-lora \\
      --package tasks

  # Interactive mode: specify package in prompt (tasks: write a rule...)
  uv run --project qwen2.5_model python qwen2.5_model/infer_policy.py \\
      --model-dir qwen2.5-rego-policy-lora

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
        help=f"Path to model directory (full fine-tuned model or LoRA adapter, optional, default: {DEFAULT_MODEL_DIR} if exists, otherwise base model only)",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Skip loading LoRA adapters, use base model only",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Device to run on (default: auto - detects CUDA, then MPS, then CPU)",
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
        default=5,
        help="Maximum number of correction attempts when validation fails (default: 5)",
    )
    parser.add_argument(
        "--include-style-guide",
        action="store_true",
        help="Include condensed Rego style guide in context (~310 tokens). Helps ensure style guide compliance.",
    )
    parser.add_argument(
        "--no-enhance-instruction",
        action="store_true",
        help="Disable instruction enhancement that emphasizes specific requirements (rule names, variable names, etc.).",
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        default=True,
        help="Use agentic workflow (Plan → Implement → Check → Repair) [default: True]",
    )
    parser.add_argument(
        "--no-agentic",
        action="store_false",
        dest="agentic",
        help="Disable agentic workflow, use simple validation loop",
    )
    parser.add_argument(
        "--no-planning",
        action="store_true",
        help="Skip planning phase in agentic workflow (faster but less structured)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show detailed workflow progress [default: True]",
    )
    parser.add_argument(
        "--quiet",
        action="store_false",
        dest="verbose",
        help="Minimal output",
    )
    parser.add_argument(
        "--no-execution-check",
        action="store_true",
        help="Disable execution validation against real attestation files (faster but less thorough).",
    )
    parser.add_argument(
        "--attestation-files",
        type=str,
        nargs="+",
        help="Specific attestation JSON files to use for execution testing (default: auto-discover from repo root).",
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
            
            # indexer = LibraryIndexer(repo_root, mapper)
            # # Scan for usage examples (slower but provides better context)
            # indexer.index_all_libraries(scan_usage=True)
            
            # builder = SmartContextBuilder(indexer, mapper, max_tokens=500)
            # print(f"✓ Indexed {len(indexer.index)} helpers")
            print()
        except Exception as e:
            print(f"⚠ Warning: Failed to initialize context system: {e}")
            print("Continuing without dynamic context...")
            print()
    
    # Resolve attestation files if provided
    attestation_files = None
    if args.attestation_files:
        attestation_files = [Path(f) if os.path.isabs(f) else repo_root / f for f in args.attestation_files]
    
    # Run inference
    enhance_instruction = not args.no_enhance_instruction
    include_execution_check = not args.no_execution_check
    include_planning = not args.no_planning
    
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
            max_corrections=args.max_corrections,
            include_style_guide=args.include_style_guide,
            enhance_instruction=enhance_instruction,
            agentic=args.agentic,
            verbose=args.verbose,
            include_execution_check=include_execution_check,
            attestation_files=attestation_files,
            include_planning=include_planning
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
            max_corrections=args.max_corrections,
            include_style_guide=args.include_style_guide,
            enhance_instruction=enhance_instruction,
            agentic=args.agentic,
            verbose=args.verbose,
            include_execution_check=include_execution_check,
            attestation_files=attestation_files,
            include_planning=include_planning
        )


if __name__ == "__main__":
    main()

