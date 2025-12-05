#!/usr/bin/env python3
"""
Collect compiler errors from inference runs for training data generation.

This script:
1. Runs inference on instructions (from file or existing dataset)
2. Captures code that fails execution check
3. Filters to compiler errors only (not just undefined rules)
4. Extracts full OPA error JSON
5. Saves in git-friendly JSONL format

Usage:
    # Extract instructions from existing dataset
    python collect_compiler_errors.py --source-dataset attestation_train.jsonl --output data/compiler_errors/collected_from_dataset.jsonl

    # Stop after collecting 200 errors (recommended for initial training)
    python collect_compiler_errors.py --source-dataset attestation_train.jsonl --output data/compiler_errors/collected.jsonl --max-errors 200

    # Collect 500 errors for more robust training
    python collect_compiler_errors.py --source-dataset attestation_train.jsonl --output data/compiler_errors/collected.jsonl --max-errors 500

    # Use instruction file
    python collect_compiler_errors.py --instructions-file instructions.txt --output data/compiler_errors/collected_from_file.jsonl

    # Test mode (small subset first)
    python collect_compiler_errors.py --test --instructions-file instructions.txt

How many errors do you need?
    - 100-200: Good starting point for initial fine-tuning
    - 300-500: Better coverage, more robust training
    - 1000+: Diminishing returns, but more diversity
    - Current run (1776): More than needed, but good for diversity
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Add current directory to path for imports (script is in qwen2.5_model/)
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    # Import from same directory
    from infer_policy import (
        load_policy_model,
        agentic_inference,
        find_repo_root,
        find_attestation_files,
        check_execution_against_attestations
    )
except ImportError as e:
    print("Error: Could not import infer_policy. Make sure you're running from the correct directory.")
    print(f"Script directory: {script_dir}")
    print(f"Error: {e}")
    print(f"\nTrying to import from: {script_dir / 'infer_policy.py'}")
    if not (script_dir / 'infer_policy.py').exists():
        print(f"ERROR: infer_policy.py not found in {script_dir}")
    sys.exit(1)


def is_compiler_error(error: str) -> bool:
    """Check if error is a compiler error (not just undefined rule).
    
    Compiler errors include:
    - rego_compile_error (variable redeclaration, etc.)
    - rego_unsafe_var_error (unsafe variables)
    - rego_type_error (type errors)
    - Parse errors (syntax errors)
    
    Excludes:
    - undefined (rule not found - this is okay)
    """
    error_lower = error.lower()
    
    # Compiler error indicators
    compiler_indicators = [
        "rego_compile_error",
        "rego_unsafe_var_error",
        "rego_type_error",
        "rego_parse_error",
        "var declared above",
        "var is unsafe",
        "type error",
        "non-terminated string",
        "unexpected token",
        "syntax error",
    ]
    
    # Exclude undefined errors (these are okay - rule might not exist)
    if "undefined" in error_lower and "unsafe" not in error_lower:
        return False
    
    # Check for compiler error indicators
    return any(indicator in error_lower for indicator in compiler_indicators)


def extract_instructions_from_dataset(dataset_file: Path) -> List[str]:
    """Extract instructions from existing training dataset.
    
    Args:
        dataset_file: Path to JSONL training dataset
        
    Returns:
        List of instruction strings
    """
    instructions = []
    
    if not dataset_file.exists():
        print(f"Warning: Dataset file {dataset_file} does not exist")
        return instructions
    
    print(f"Extracting instructions from {dataset_file}...")
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Try to extract instruction from messages format
                if "messages" in data:
                    for msg in data["messages"]:
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            # Look for "Instruction:" pattern
                            if "Instruction:" in content:
                                # Extract instruction text
                                parts = content.split("Instruction:", 1)
                                if len(parts) > 1:
                                    instruction = parts[1].strip()
                                    # Remove any trailing context/formatting
                                    if "\n" in instruction:
                                        instruction = instruction.split("\n")[0].strip()
                                    if instruction:
                                        instructions.append(instruction)
                            elif content and not content.startswith("Error") and not content.startswith("The generated"):
                                # Might be instruction without "Instruction:" prefix
                                if len(content) > 20 and len(content) < 500:
                                    instructions.append(content.strip())
                
                # Try to extract from instruction field
                elif "instruction" in data:
                    instructions.append(data["instruction"].strip())
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    # Deduplicate while preserving order
    seen = set()
    unique_instructions = []
    for inst in instructions:
        if inst not in seen:
            seen.add(inst)
            unique_instructions.append(inst)
    
    print(f"✓ Extracted {len(unique_instructions)} unique instructions")
    return unique_instructions


def get_opa_error_json(code: str, attestation_file: Path, package: str = "attestation_check") -> Optional[str]:
    """Get full OPA error JSON for code that fails to compile/execute.
    
    Args:
        code: Rego code to test
        attestation_file: Attestation file to test against
        package: Package name
        
    Returns:
        Full OPA error JSON string, or None if code compiles/executes
    """
    import json as json_lib
    import tempfile as tf
    
    # Determine OPA command
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
    
    # Build complete code
    complete_code = code
    if package and f"package {package}" not in code:
        code_parts = [f"package {package}\n"]
        code_parts.append("import rego.v1\n")
        code_parts.append("\n")
        code_parts.append(code)
        complete_code = "".join(code_parts)
    
    # Remove backticks
    complete_code = complete_code.replace('`', '')
    
    # Write Rego to temp file
    with tf.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as rego_file:
        rego_path = Path(rego_file.name)
        rego_file.write(complete_code)
        rego_file.flush()
    
    try:
        # Read attestation
        with open(attestation_file, 'r') as f:
            att_data = json_lib.load(f)
        
        # Wrap in expected format
        if isinstance(att_data, list):
            wrapped_input = {"attestations": att_data}
        elif "attestations" in att_data:
            wrapped_input = att_data
        else:
            wrapped_input = {"attestations": [att_data]}
        
        # Write input to temp file
        with tf.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
            input_path = Path(input_file.name)
            json_lib.dump(wrapped_input, input_file)
            input_file.flush()
        
        try:
            # Try to evaluate - if it fails, capture error
            result = subprocess.run(
                opa_base + [
                    "eval",
                    "--data", str(rego_path),
                    "--input", str(input_path),
                    "data.deny",
                    "--format", "json"
                ],
                capture_output=True,
                timeout=5,
                text=True
            )
            
            if result.returncode != 0:
                # Code has error - return full error JSON
                error_output = result.stderr or result.stdout
                # Try to parse as JSON, if not JSON return as-is
                try:
                    json_lib.loads(error_output)  # Validate it's JSON
                    return error_output
                except:
                    # Not JSON, wrap it
                    return json.dumps({"errors": [{"message": error_output[:500]}]})
            
            # Also try parse to catch compile errors
            parse_result = subprocess.run(
                opa_base + ["parse", str(rego_path)],
                capture_output=True,
                timeout=5,
                text=True
            )
            
            if parse_result.returncode != 0:
                # Parse error - return it
                error_output = parse_result.stderr or parse_result.stdout
                try:
                    json_lib.loads(error_output)
                    return error_output
                except:
                    return json.dumps({"errors": [{"message": error_output[:500]}]})
        
        finally:
            try:
                input_path.unlink()
            except:
                pass
    
    finally:
        try:
            rego_path.unlink()
        except:
            pass
    
    return None  # Code compiles and executes


def normalize_error_json(error_json_str: str) -> str:
    """Normalize error JSON by replacing temp file paths with placeholders.
    
    This ensures training data doesn't include variable temp file paths that
    would confuse the model.
    
    Args:
        error_json_str: Error JSON string from OPA
        
    Returns:
        Normalized error JSON string with temp paths replaced
    """
    import json as json_lib
    import re
    
    try:
        # Parse JSON
        error_data = json_lib.loads(error_json_str)
        
        # Normalize file paths in errors
        if isinstance(error_data, dict) and "errors" in error_data:
            for error in error_data["errors"]:
                if "location" in error and "file" in error["location"]:
                    file_path = error["location"]["file"]
                    # Replace temp file paths with placeholder
                    if "/tmp/" in file_path or file_path.startswith("/tmp"):
                        # Extract just the filename or use a placeholder
                        filename = Path(file_path).name
                        if filename.startswith("tmp") or filename.startswith("rego_"):
                            error["location"]["file"] = "<temp_file>.rego"
                        else:
                            error["location"]["file"] = filename
        
        # Return normalized JSON
        return json_lib.dumps(error_data, indent=2)
    except (json_lib.JSONDecodeError, KeyError, TypeError):
        # If parsing fails, try simple string replacement
        # Replace common temp file patterns
        normalized = re.sub(r'/tmp/tmp[a-zA-Z0-9_]+\.rego', '<temp_file>.rego', error_json_str)
        normalized = re.sub(r'/tmp/[^"]+\.rego', '<temp_file>.rego', normalized)
        return normalized


def collect_compiler_errors(
    instructions: List[str],
    tokenizer,
    model,
    device,
    attestation_file: Path,
    output_file: Path,
    test_mode: bool = False,
    max_instructions: Optional[int] = None,
    max_errors: Optional[int] = None,
    checkpoint_file: Optional[Path] = None
) -> Dict[str, int]:
    """Collect compiler errors from inference runs.
    
    Args:
        instructions: List of instructions to test
        tokenizer: Model tokenizer
        model: Model instance
        device: Device
        attestation_file: Attestation file for testing
        output_file: Output JSONL file
        test_mode: If True, only process first 10 instructions
        max_instructions: Maximum number of instructions to process
        checkpoint_file: Path to checkpoint file for resume
        
    Returns:
        Statistics dictionary
    """
    stats = {
        "total": 0,
        "processed": 0,
        "compiler_errors": 0,
        "other_errors": 0,
        "success": 0,
        "failed": 0
    }
    
    # Load checkpoint if exists
    processed_instructions = set()
    if checkpoint_file and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                processed_instructions = set(checkpoint_data.get("processed", []))
                print(f"Resuming: {len(processed_instructions)} instructions already processed")
        except:
            pass
    
    # Limit instructions for test mode
    if test_mode:
        instructions = instructions[:10]
        print("Test mode: Processing first 10 instructions only")
    
    if max_instructions:
        instructions = instructions[:max_instructions]
    
    stats["total"] = len(instructions)
    
    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Open output file in append mode (for resume)
    output_fp = open(output_file, 'a', encoding='utf-8')
    
    try:
        for i, instruction in enumerate(instructions, 1):
            # Check if we've reached max_errors limit (before processing)
            if max_errors and stats['compiler_errors'] >= max_errors:
                print(f"\n✓ Already have {stats['compiler_errors']} compiler errors (target: {max_errors}). Stopping.")
                break
            # Skip if already processed
            if instruction in processed_instructions:
                continue
            
            print(f"\n[{i}/{len(instructions)}] Processing: {instruction[:80]}...")
            
            try:
                # Run inference
                final_code, state = agentic_inference(
                    tokenizer, model, device,
                    instruction,
                    package="attestation_check",
                    imports=["rego.v1"],
                    max_iterations=3,  # Limit iterations for collection
                    include_planning=True,
                    include_style_check=False,  # Focus on compiler errors
                    include_execution_check=True,
                    attestation_files=[attestation_file],
                    verbose=False  # Less verbose for batch processing
                )
                
                stats["processed"] += 1
                
                # Check if code has compiler errors
                if state.errors or not state.syntax_valid or not state.execution_valid:
                    # Get OPA error JSON
                    opa_error = get_opa_error_json(final_code, attestation_file)
                    
                    if opa_error:
                        # Check if it's a compiler error
                        if is_compiler_error(opa_error):
                            stats["compiler_errors"] += 1
                            
                            # Extract error category
                            error_category = "unknown"
                            if "var declared above" in opa_error.lower() or "rego_compile_error" in opa_error.lower():
                                error_category = "variable_redeclaration"
                            elif "unsafe" in opa_error.lower() or "rego_unsafe_var_error" in opa_error.lower():
                                error_category = "unsafe_variable"
                            elif "type error" in opa_error.lower() or "rego_type_error" in opa_error.lower():
                                error_category = "type_error"
                            elif "parse" in opa_error.lower() or "syntax" in opa_error.lower():
                                error_category = "syntax_error"
                            
                            # Save error example
                            # If syntax is invalid, execution can't be valid
                            execution_valid = state.execution_valid if state.syntax_valid else False
                            
                            # Normalize error JSON (replace temp file paths)
                            normalized_error_json = normalize_error_json(opa_error)
                            
                            example = {
                                "instruction": instruction,
                                "incorrect_code": final_code,
                                "error_json": normalized_error_json,
                                "error_category": error_category,
                                "attestation_file": str(attestation_file.name),
                                "timestamp": datetime.utcnow().isoformat(),
                                "model_version": "current",  # Could be enhanced
                                "iteration": state.iteration,
                                "syntax_valid": state.syntax_valid,
                                "execution_valid": execution_valid,
                            }
                            
                            output_fp.write(json.dumps(example) + "\n")
                            output_fp.flush()
                            
                            print(f"  ✓ Compiler error captured ({error_category}) [{stats['compiler_errors']} total]")
                            
                            # Check if we've reached max_errors limit
                            if max_errors and stats['compiler_errors'] >= max_errors:
                                print(f"\n✓ Reached target of {max_errors} compiler errors. Stopping collection.")
                                break
                        else:
                            stats["other_errors"] += 1
                            print(f"  - Other error (not compiler error)")
                    else:
                        stats["other_errors"] += 1
                        print(f"  - Error but couldn't extract OPA error JSON")
                else:
                    stats["success"] += 1
                    print(f"  - Code compiled successfully")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                stats["failed"] += 1
                print(f"  ✗ Error processing: {e}")
                import traceback
                traceback.print_exc()
            
            # Update checkpoint
            processed_instructions.add(instruction)
            if checkpoint_file:
                checkpoint_data = {
                    "processed": list(processed_instructions),
                    "stats": stats,
                    "last_updated": datetime.utcnow().isoformat()
                }
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
    
    finally:
        output_fp.close()
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Collect compiler errors from inference runs",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--source-dataset",
        type=str,
        help="Extract instructions from existing training dataset (JSONL file)"
    )
    
    parser.add_argument(
        "--instructions-file",
        type=str,
        help="File with instructions (one per line)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/compiler_errors/collected_errors.jsonl",
        help="Output JSONL file (default: data/compiler_errors/collected_errors.jsonl)"
    )
    
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name"
    )
    
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Path to fine-tuned model or LoRA adapter"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Device to use"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only process first 10 instructions"
    )
    
    parser.add_argument(
        "--max-instructions",
        type=int,
        help="Maximum number of instructions to process"
    )
    
    parser.add_argument(
        "--max-errors",
        type=int,
        help="Stop after collecting this many compiler errors (recommended: 200-500 for initial training)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Checkpoint file for resume (default: <output>.checkpoint.json)"
    )
    
    args = parser.parse_args()
    
    # Find repo root
    repo_root = find_repo_root()
    print(f"Repository root: {repo_root}")
    
    # Resolve paths
    if args.source_dataset:
        source_dataset = repo_root / args.source_dataset
    else:
        source_dataset = None
    
    if args.instructions_file:
        instructions_file = repo_root / args.instructions_file
    else:
        instructions_file = None
    
    output_file = repo_root / args.output
    checkpoint_file = repo_root / (args.checkpoint or str(args.output) + ".checkpoint.json")
    
    # Get instructions
    instructions = []
    
    if source_dataset:
        instructions = extract_instructions_from_dataset(source_dataset)
    elif instructions_file:
        if instructions_file.exists():
            with open(instructions_file, 'r') as f:
                instructions = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(instructions)} instructions from {instructions_file}")
        else:
            print(f"Error: Instructions file {instructions_file} does not exist")
            sys.exit(1)
    else:
        print("Error: Must provide --source-dataset or --instructions-file")
        sys.exit(1)
    
    if not instructions:
        print("Error: No instructions found")
        sys.exit(1)
    
    # Find attestation file
    attestation_file = repo_root / "attestation.json"
    if not attestation_file.exists():
        print(f"Error: Attestation file {attestation_file} does not exist")
        sys.exit(1)
    
    print(f"Using attestation file: {attestation_file}")
    
    # Load model
    print("\n" + "=" * 60)
    print("Loading Model")
    print("=" * 60)
    
    model_dir = None
    if args.model_dir:
        model_dir = repo_root / args.model_dir if not os.path.isabs(args.model_dir) else Path(args.model_dir)
    
    try:
        tokenizer, model, device = load_policy_model(
            base_model=args.base_model,
            model_dir=str(model_dir) if model_dir else None,
            device=args.device,
            no_lora=False
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Collect errors
    print("\n" + "=" * 60)
    print("Collecting Compiler Errors")
    print("=" * 60)
    print(f"Output: {output_file}")
    print(f"Checkpoint: {checkpoint_file}")
    if args.max_errors:
        print(f"Target: {args.max_errors} compiler errors (will stop when reached)")
    print()
    
    stats = collect_compiler_errors(
        instructions,
        tokenizer,
        model,
        device,
        attestation_file,
        output_file,
        test_mode=args.test,
        max_instructions=args.max_instructions,
        max_errors=args.max_errors,
        checkpoint_file=checkpoint_file
    )
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Collection Statistics")
    print("=" * 60)
    print(f"Total instructions: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Compiler errors found: {stats['compiler_errors']}")
    print(f"Other errors: {stats['other_errors']}")
    print(f"Successful (no errors): {stats['success']}")
    print(f"Failed to process: {stats['failed']}")
    print()
    print(f"✓ Data saved to: {output_file}")
    print(f"  This file can be committed to git")
    print()


if __name__ == "__main__":
    main()

