#!/usr/bin/env python3
"""
Convert compiler error dataset to training format.

Takes the mutated_errors.jsonl and converts it to the format expected by train_policy.py
for training the model to fix compiler errors.

Usage:
    python format_compiler_errors_for_training.py \
        --input data/compiler_errors/mutated_errors.jsonl \
        --output data/compiler_errors/compiler_error_train.jsonl \
        --split-train-val --train-ratio 0.9
"""

import json
import argparse
import random
from pathlib import Path
from typing import List, Dict


def format_error_message(error_json_str: str) -> str:
    """Format OPA error JSON into a readable error message for the model."""
    try:
        error_data = json.loads(error_json_str)
        if isinstance(error_data, dict) and "errors" in error_data:
            error_messages = []
            for error in error_data["errors"][:3]:  # Limit to first 3 errors
                msg = error.get("message", "")
                location = error.get("location", {})
                row = location.get("row", "")
                col = location.get("col", "")
                code_type = error.get("code", "")
                
                error_str = msg
                if row:
                    error_str += f" (line {row}"
                    if col:
                        error_str += f", column {col}"
                    error_str += ")"
                
                if code_type:
                    error_str = f"[{code_type}] {error_str}"
                
                error_messages.append(error_str)
            
            if len(error_messages) == 1:
                return error_messages[0]
            else:
                return "\n".join(f"{i+1}. {msg}" for i, msg in enumerate(error_messages))
    except:
        pass
    
    # Fallback: return first 200 chars
    return error_json_str[:200]


def convert_to_training_format(
    input_file: Path,
    output_file: Path,
    include_error_details: bool = True
) -> List[Dict]:
    """Convert compiler error examples to training format.
    
    Args:
        input_file: Input JSONL with compiler errors
        output_file: Output JSONL for training
        include_error_details: If True, include full error JSON in context
        
    Returns:
        List of training examples
    """
    examples = []
    
    print(f"Reading from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract fields
                instruction = data.get("instruction", "")
                incorrect_code = data.get("incorrect_code", "")
                correct_code = data.get("correct_code", "")
                error_json = data.get("error_json", "")
                error_category = data.get("error_category", "unknown")
                
                if not instruction or not incorrect_code or not correct_code:
                    continue
                
                # Format error message
                error_message = format_error_message(error_json)
                
                # Build training example
                # For compiler error correction, we want the model to:
                # 1. See the instruction
                # 2. See the error message
                # 3. See the incorrect code
                # 4. Generate the correct code
                
                # Build user message with error context
                user_parts = []
                
                # Add error message as context
                if include_error_details:
                    user_parts.append(f"Error: {error_message}")
                    user_parts.append(f"Error type: {error_category}")
                    user_parts.append("")
                
                # Add instruction
                user_parts.append(f"Instruction: {instruction}")
                user_parts.append("")
                
                # Add incorrect code
                user_parts.append("The following Rego code has compilation errors:")
                user_parts.append("```rego")
                user_parts.append(incorrect_code)
                user_parts.append("```")
                user_parts.append("")
                user_parts.append("Please fix the compilation errors and provide the corrected code.")
                
                user_content = "\n".join(user_parts)
                
                # Create training example
                training_example = {
                    "instruction": user_content,
                    "output_code": correct_code,
                    "task_type": "refactor",  # We're fixing/refactoring code
                    "error_category": error_category,
                }
                
                # Optionally include full error JSON in context
                if include_error_details:
                    training_example["error_json"] = error_json
                
                examples.append(training_example)
                
            except Exception as e:
                print(f"Warning: Skipping line {line_num}: {e}")
                continue
    
    print(f"✓ Converted {len(examples)} examples")
    
    # Write to output file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"✓ Saved to {output_file}")
    
    return examples


def split_train_val(
    examples: List[Dict],
    train_ratio: float = 0.9,
    output_dir: Path = None
) -> tuple:
    """Split examples into train and validation sets.
    
    Returns:
        (train_examples, val_examples)
    """
    # Shuffle for random split
    random.seed(42)  # For reproducibility
    shuffled = examples.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train_examples = shuffled[:split_idx]
    val_examples = shuffled[split_idx:]
    
    print(f"\nSplit: {len(train_examples)} train, {len(val_examples)} validation")
    
    if output_dir:
        train_file = output_dir / "compiler_error_train.jsonl"
        val_file = output_dir / "compiler_error_val.jsonl"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for ex in train_examples:
                f.write(json.dumps(ex) + "\n")
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for ex in val_examples:
                f.write(json.dumps(ex) + "\n")
        
        print(f"✓ Saved train to {train_file}")
        print(f"✓ Saved val to {val_file}")
    
    return train_examples, val_examples


def main():
    parser = argparse.ArgumentParser(
        description="Convert compiler error dataset to training format"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with compiler errors"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/compiler_errors/compiler_error_train.jsonl",
        help="Output JSONL file for training"
    )
    
    parser.add_argument(
        "--split-train-val",
        action="store_true",
        help="Split into train and validation sets"
    )
    
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Ratio for train split (default: 0.9)"
    )
    
    parser.add_argument(
        "--no-error-details",
        action="store_true",
        help="Don't include error details in context (just instruction + code)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    input_file = repo_root / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_file = repo_root / args.output if not Path(args.output).is_absolute() else Path(args.output)
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return 1
    
    # Convert to training format
    examples = convert_to_training_format(
        input_file,
        output_file,
        include_error_details=not args.no_error_details
    )
    
    if not examples:
        print("Error: No examples converted")
        return 1
    
    # Print statistics
    categories = {}
    for ex in examples:
        cat = ex.get("error_category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\nStatistics:")
    print(f"  Total examples: {len(examples)}")
    print(f"  By category:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")
    
    # Split if requested
    if args.split_train_val:
        split_train_val(examples, args.train_ratio, output_file.parent)
    
    print("\n✓ Conversion complete!")
    print(f"\nTo train with this data:")
    print(f"  python qwen2.5_model/train_policy.py --train-path {output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())

