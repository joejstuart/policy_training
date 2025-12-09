#!/usr/bin/env python3
"""
Generate compiler errors by mutating correct Rego code.

Since the model generates mostly correct code, we need to:
1. Take correct code examples
2. Apply mutations to introduce errors
3. Get OPA errors for those mutations
4. Create training data (incorrect_code + error + correction)

Usage:
    # Generate errors from existing correct code
    python generate_compiler_errors_from_mutations.py \
        --source-dataset attestation_train.jsonl \
        --output data/compiler_errors/mutated_errors.jsonl \
        --max-errors 500
"""

import os
import sys
import json
import argparse
import subprocess
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import random

# Add current directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    from infer_policy import find_repo_root, find_attestation_files
except ImportError:
    # Fallback: implement simple versions of these functions
    def find_repo_root() -> Path:
        """Find repository root by looking for policy/ directory."""
        current = Path(__file__).resolve()
        while current != current.parent:
            if (current / "policy").exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def find_attestation_files(repo_root: Path, max_files: int = 5) -> List[Path]:
        """Find attestation JSON files in the repository root."""
        attestation_file = repo_root / "attestation.json"
        if attestation_file.exists():
            return [attestation_file]
        return []


def mutate_variable_redeclaration(code: str) -> List[str]:
    """Mutate code to introduce variable redeclaration errors.
    
    Returns list of mutated code strings.
    """
    mutations = []
    
    # Find all 'some X in Y' patterns
    some_pattern = r'some\s+(\w+)\s+in\s+([^\n]+)'
    matches = list(re.finditer(some_pattern, code))
    
    if not matches:
        return []
    
    # For each 'some X in Y', try to find where X is used later
    for match in matches:
        var_name = match.group(1)
        var_context = match.group(2)
        
        # Look for assignments using the same variable name
        assignment_pattern = rf'{var_name}\s*:='
        assignment_matches = list(re.finditer(assignment_pattern, code))
        
        if assignment_matches:
            # Already has potential redeclaration - skip
            continue
        
        # Try mutation: change iteration variable to match a common assignment name
        # Find common assignment variable names in the code
        common_names = ['result', 'msg', 'output', 'value']
        
        for name in common_names:
            if name != var_name and name in code.lower():
                # Create mutation: change 'some X in Y' to 'some <name> in Y'
                # Replace just the variable name, keep rest of code
                mutated = code[:match.start(1)] + name + code[match.end(1):]
                # Also change references to var_name to name
                mutated = re.sub(rf'\b{var_name}\.', f'{name}.', mutated)
                mutations.append(mutated)
                break
    
    # Another mutation: duplicate variable in nested iteration
    if len(matches) > 1:
        # Use same variable name in nested iteration
        first_match = matches[0]
        second_match = matches[1]
        var_name = first_match.group(1)
        
        # Change second iteration to use same variable
        mutated = code[:second_match.start(1)] + var_name + code[second_match.end(1):]
        mutations.append(mutated)
    
    return mutations


def mutate_unsafe_variable(code: str) -> List[str]:
    """Mutate code to introduce unsafe variable errors.
    
    Returns list of mutated code strings.
    """
    mutations = []
    
    # Find all 'some X in input...' patterns
    input_iter_pattern = r'some\s+(\w+)\s+in\s+input\.[^\n]+'
    input_matches = list(re.finditer(input_iter_pattern, code))
    
    if not input_matches:
        return []
    
    # Mutation 1: Remove input iteration, keep variable usage
    for match in input_matches:
        var_name = match.group(1)
        # Remove the 'some X in input...' line
        line_start = code.rfind('\n', 0, match.start())
        line_end = code.find('\n', match.end())
        if line_end == -1:
            line_end = len(code)
        
        # Remove the iteration line
        mutated = code[:line_start+1] + code[line_end+1:]
        # But keep variable references (they'll be unsafe now)
        mutations.append(mutated)
    
    # Mutation 2: Change 'some X in input.Y' to 'some X in Y' (missing input)
    for match in input_matches:
        var_name = match.group(1)
        full_match = match.group(0)
        # Replace 'input.' with nothing
        mutated = full_match.replace('input.', '')
        mutations.append(code[:match.start()] + mutated + code[match.end():])
    
    return mutations


def mutate_type_error(code: str) -> List[str]:
    """Mutate code to introduce type errors (array access as object).
    
    Returns list of mutated code strings.
    """
    mutations = []
    
    # Find patterns like 'some X in Y.Z' where Y.Z might be an array
    # Then try to access it as object: Y.Z.field instead of iterating
    
    # Pattern: some result in task.results
    array_iter_pattern = r'some\s+(\w+)\s+in\s+(\w+(?:\.\w+)+)'
    matches = list(re.finditer(array_iter_pattern, code))
    
    for match in matches:
        iter_var = match.group(1)
        array_path = match.group(2)
        
        # Find where iter_var.field is used
        field_access_pattern = rf'{iter_var}\.(\w+)'
        field_matches = list(re.finditer(field_access_pattern, code))
        
        if field_matches:
            # Try mutation: replace iteration with direct access
            field_name = field_matches[0].group(1)
            # Change 'some X in Y.Z' to 'Y.Z.field' (wrong - treating array as object)
            mutated = code[:match.start()] + f'{array_path}.{field_name}' + code[match.end():]
            # Remove the field access that used iter_var
            if field_matches:
                field_match = field_matches[0]
                mutated = mutated[:field_match.start()] + mutated[field_match.end():]
            mutations.append(mutated)
    
    return mutations


def mutate_syntax_error(code: str) -> List[str]:
    """Mutate code to introduce syntax errors.
    
    Returns list of mutated code strings.
    """
    mutations = []
    
    # Mutation 1: Add invalid keywords
    invalid_keywords = ['for', 'if', 'then', 'else', 'rule', 'match']
    
    for keyword in invalid_keywords:
        if keyword not in code.lower():
            # Try to insert invalid keyword
            # Find a good insertion point (after 'deny' or 'warn')
            deny_match = re.search(r'(deny|warn)\s+', code)
            if deny_match:
                insert_pos = deny_match.end()
                mutated = code[:insert_pos] + f'{keyword} ' + code[insert_pos:]
                mutations.append(mutated)
    
    # Mutation 2: Wrong import syntax
    if 'import rego.v1' in code:
        # Change to invalid import
        mutated = code.replace('import rego.v1', 'import "rego/v1"')
        mutations.append(mutated)
    
    # Mutation 3: Add Python-like syntax
    if 'some' in code and 'in' in code:
        # Try to add colon after 'in' (Python style)
        some_in_pattern = r'some\s+\w+\s+in\s+[^\n:]+'
        match = re.search(some_in_pattern, code)
        if match:
            mutated = code[:match.end()] + ':' + code[match.end():]
            mutations.append(mutated)
    
    return mutations


def get_opa_error_for_code(code: str, attestation_file: Path, package: str = "attestation_check") -> Optional[str]:
    """Get OPA error JSON for code.
    
    Returns error JSON string if code has errors, None if valid.
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
    
    # Write Rego to temp file
    with tf.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as rego_file:
        rego_path = Path(rego_file.name)
        rego_file.write(complete_code)
        rego_file.flush()
    
    try:
        # Try parse first (catches syntax errors)
        parse_result = subprocess.run(
            opa_base + ["parse", str(rego_path)],
            capture_output=True,
            timeout=5,
            text=True
        )
        
        if parse_result.returncode != 0:
            error_output = parse_result.stderr or parse_result.stdout
            try:
                json_lib.loads(error_output)
                return error_output
            except:
                return json.dumps({"errors": [{"message": error_output[:500]}]})
        
        # If parse succeeds, try execution
        with open(attestation_file, 'r') as f:
            att_data = json_lib.load(f)
        
        if isinstance(att_data, list):
            wrapped_input = {"attestations": att_data}
        elif "attestations" in att_data:
            wrapped_input = att_data
        else:
            wrapped_input = {"attestations": [att_data]}
        
        with tf.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
            input_path = Path(input_file.name)
            json_lib.dump(wrapped_input, input_file)
            input_file.flush()
        
        try:
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
                error_output = result.stderr or result.stdout
                if "undefined" not in error_output.lower():
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
    
    return None  # Code is valid


def normalize_error_json(error_json_str: str) -> str:
    """Normalize error JSON by replacing temp file paths."""
    import json as json_lib
    import re
    
    try:
        error_data = json_lib.loads(error_json_str)
        if isinstance(error_data, dict) and "errors" in error_data:
            for error in error_data["errors"]:
                if "location" in error and "file" in error["location"]:
                    file_path = error["location"]["file"]
                    # Handle various temp file paths: /tmp/, /var/folders/, etc.
                    if any(path in file_path for path in ["/tmp/", "/var/folders/", "tmp"]) or file_path.startswith("/tmp"):
                        error["location"]["file"] = "<temp_file>.rego"
        return json_lib.dumps(error_data, indent=2)
    except:
        # Fallback: regex replacement for common temp file patterns
        normalized = re.sub(r'/tmp/tmp[a-zA-Z0-9_]+\.rego', '<temp_file>.rego', error_json_str)
        normalized = re.sub(r'/tmp/[^"]+\.rego', '<temp_file>.rego', normalized)
        normalized = re.sub(r'/var/folders/[^"]+\.rego', '<temp_file>.rego', normalized)
        normalized = re.sub(r'/[^"]*tmp[^"]*\.rego', '<temp_file>.rego', normalized)
        return normalized


def extract_correct_code_from_dataset(dataset_file: Path) -> List[Dict]:
    """Extract correct Rego code examples from training dataset.
    
    Returns list of dicts with 'instruction' and 'correct_code'.
    """
    examples = []
    
    if not dataset_file.exists():
        return examples
    
    print(f"Extracting correct code from {dataset_file}...")
    
    with open(dataset_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                instruction = None
                correct_code = None
                
                # Try direct format (instruction + output_code)
                if "instruction" in data and "output_code" in data:
                    instruction = data["instruction"]
                    correct_code = data["output_code"]
                
                # Try messages format (chat format)
                elif "messages" in data:
                    for msg in data["messages"]:
                        if msg.get("role") == "user":
                            content = msg.get("content", "")
                            if "Instruction:" in content:
                                parts = content.split("Instruction:", 1)
                                if len(parts) > 1:
                                    instruction = parts[1].strip().split("\n")[0]
                        elif msg.get("role") == "assistant":
                            # Extract Rego code from response
                            content = msg.get("content", "")
                            # Look for code blocks
                            if "```rego" in content:
                                code_match = re.search(r'```rego\n(.*?)```', content, re.DOTALL)
                                if code_match:
                                    correct_code = code_match.group(1).strip()
                            elif "package" in content and ("deny" in content or "warn" in content):
                                # Might be code without markdown
                                correct_code = content.strip()
                
                # Validate we have both instruction and code
                if instruction and correct_code:
                    # Make sure code looks like Rego
                    if "package" in correct_code or "deny" in correct_code or "warn" in correct_code:
                        examples.append({
                            "instruction": instruction,
                            "correct_code": correct_code
                        })
            
            except Exception as e:
                continue
    
    print(f"✓ Extracted {len(examples)} correct code examples")
    return examples


def generate_mutated_errors(
    correct_examples: List[Dict],
    attestation_file: Path,
    output_file: Path,
    max_errors: int = 500,
    mutations_per_example: int = 3
) -> Dict[str, int]:
    """Generate compiler errors by mutating correct code.
    
    Returns statistics.
    """
    stats = {
        "total_examples": len(correct_examples),
        "mutations_tried": 0,
        "errors_generated": 0,
        "by_category": {
            "variable_redeclaration": 0,
            "unsafe_variable": 0,
            "type_error": 0,
            "syntax_error": 0
        }
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    mutation_functions = [
        ("variable_redeclaration", mutate_variable_redeclaration),
        ("unsafe_variable", mutate_unsafe_variable),
        ("type_error", mutate_type_error),
        ("syntax_error", mutate_syntax_error),
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(correct_examples, 1):
            if stats["errors_generated"] >= max_errors:
                break
            
            instruction = example["instruction"]
            correct_code = example["correct_code"]
            
            print(f"\n[{i}/{len(correct_examples)}] Mutating: {instruction[:60]}...")
            
            # Try each mutation type
            for category, mutate_func in mutation_functions:
                if stats["errors_generated"] >= max_errors:
                    break
                
                mutations = mutate_func(correct_code)
                if not mutations:
                    continue
                
                # Try up to mutations_per_example mutations
                for mutated_code in mutations[:mutations_per_example]:
                    if stats["errors_generated"] >= max_errors:
                        break
                    
                    stats["mutations_tried"] += 1
                    
                    # Get error for mutated code
                    error_json = get_opa_error_for_code(mutated_code, attestation_file)
                    
                    if error_json:
                        # Check if it's a compiler error (not just undefined)
                        if "undefined" not in error_json.lower() or any(
                            kw in error_json.lower() for kw in [
                                "rego_compile_error", "rego_parse_error", "rego_type_error",
                                "unsafe", "declared above", "type error"
                            ]
                        ):
                            stats["errors_generated"] += 1
                            stats["by_category"][category] += 1
                            
                            # Normalize error
                            normalized_error = normalize_error_json(error_json)
                            
                            # Create training example
                            training_example = {
                                "instruction": instruction,
                                "incorrect_code": mutated_code,
                                "correct_code": correct_code,
                                "error_json": normalized_error,
                                "error_category": category,
                                "mutation_type": category,
                                "attestation_file": str(attestation_file.name),
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                            
                            f.write(json.dumps(training_example) + "\n")
                            f.flush()
                            
                            print(f"  ✓ Generated {category} error [{stats['errors_generated']}/{max_errors}]")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate compiler errors by mutating correct code"
    )
    
    parser.add_argument(
        "--source-dataset",
        type=str,
        required=True,
        help="Training dataset with correct code examples (JSONL)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/compiler_errors/mutated_errors.jsonl",
        help="Output JSONL file"
    )
    
    parser.add_argument(
        "--max-errors",
        type=int,
        default=500,
        help="Maximum number of errors to generate"
    )
    
    parser.add_argument(
        "--mutations-per-example",
        type=int,
        default=3,
        help="Number of mutations to try per example"
    )
    
    args = parser.parse_args()
    
    repo_root = find_repo_root()
    source_dataset = repo_root / args.source_dataset
    output_file = repo_root / args.output
    attestation_file = repo_root / "attestation.json"
    
    if not attestation_file.exists():
        print(f"Error: Attestation file {attestation_file} not found")
        sys.exit(1)
    
    # Extract correct code
    correct_examples = extract_correct_code_from_dataset(source_dataset)
    
    if not correct_examples:
        print("Error: No correct code examples found")
        sys.exit(1)
    
    print(f"\nGenerating {args.max_errors} compiler errors from {len(correct_examples)} examples...")
    
    # Generate mutations
    stats = generate_mutated_errors(
        correct_examples,
        attestation_file,
        output_file,
        max_errors=args.max_errors,
        mutations_per_example=args.mutations_per_example
    )
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Generation Statistics")
    print("=" * 60)
    print(f"Total examples processed: {stats['total_examples']}")
    print(f"Mutations tried: {stats['mutations_tried']}")
    print(f"Errors generated: {stats['errors_generated']}")
    print("\nBy category:")
    for category, count in stats['by_category'].items():
        if count > 0:
            print(f"  {category}: {count}")
    print(f"\n✓ Data saved to: {output_file}")


if __name__ == "__main__":
    main()

