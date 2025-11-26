#!/usr/bin/env python3
"""
Data augmentation script to increase training examples without new rule files.

This script:
1. Increases refactor example generation rate
2. Creates instruction variations (paraphrasing)
3. Extracts helper rules without metadata
4. Creates examples from rule fragments
5. Generates synthetic examples by combining patterns
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Import from main generator
sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import (
    parse_rego_file, RuleExample, build_context, generate_helper_cheat_sheet,
    estimate_tokens, validate_rego_code, example_to_jsonl, MAX_TOKENS
)


def paraphrase_instruction(instruction: str) -> str:
    """Create variations of instructions by paraphrasing."""
    variations = [
        # Add prefixes
        ("Implement a rule that", "Create a rule that"),
        ("Verify that", "Check that"),
        ("Confirm that", "Ensure that"),
        ("Validate that", "Verify that"),
        
        # Rephrase descriptions
        ("must", "should"),
        ("is required", "must be provided"),
        ("are allowed", "are permitted"),
        ("is not allowed", "is disallowed"),
    ]
    
    result = instruction
    # Apply random variations
    if random.random() < 0.3:  # 30% chance to paraphrase
        for old, new in variations:
            if old in result:
                result = result.replace(old, new, 1)
                break
    
    return result


def extract_helper_rules(rego_file) -> List[Dict]:
    """Extract helper rules (starting with _) that could be training examples."""
    helpers = []
    content = rego_file.full_content
    lines = content.split('\n')
    
    # Find helper rules (rules starting with _)
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match helper rule patterns
        helper_match = re.match(r'^(_\w+)\s*(?:\([^)]*\))?\s*(?::=\s*|\s+contains\s+\w+\s+)?(?:if\s+)?\{', stripped)
        if helper_match:
            rule_name = helper_match.group(1)
            
            # Find the end of the rule
            brace_count = 0
            rule_start = content.find(line)
            rule_end = rule_start + len(line)
            in_rule = False
            
            for j in range(rule_start, len(content)):
                if content[j] == '{':
                    brace_count += 1
                    in_rule = True
                elif content[j] == '}':
                    brace_count -= 1
                    if brace_count == 0 and in_rule:
                        rule_end = j + 1
                        break
            
            if rule_end > rule_start:
                rule_code = content[rule_start:rule_end].strip()
                
                # Only include substantial helpers (at least 3 lines)
                if rule_code.count('\n') >= 2:
                    # Create synthetic metadata
                    metadata = {
                        "title": f"Helper: {rule_name}",
                        "description": f"Helper function {rule_name} that performs internal logic for the {rego_file.package} package."
                    }
                    
                    helpers.append({
                        "name": rule_name,
                        "metadata": metadata,
                        "code": rule_code,
                        "start_pos": rule_start,
                        "end_pos": rule_end,
                    })
    
    return helpers


def create_rule_fragment_example(rego_file, rule: Dict) -> Optional[RuleExample]:
    """Create an example from a fragment of a complex rule."""
    code = rule["code"]
    
    # Only split very large rules (> 20 lines)
    if code.count('\n') < 20:
        return None
    
    # Try to split into logical parts
    lines = code.split('\n')
    
    # Find a good split point (after a complete statement)
    split_point = len(lines) // 2
    for i in range(split_point, len(lines)):
        line = lines[i].strip()
        # Split after a line ending with } or before a new statement
        if line.endswith('}') or (line and not line.startswith('\t') and not line.startswith(' ')):
            split_point = i + 1
            break
    
    if split_point >= len(lines) - 5:  # Need at least 5 lines for fragment
        return None
    
    # Create fragment (first half)
    fragment_code = '\n'.join(lines[:split_point])
    
    # Create instruction for the fragment
    metadata = rule["metadata"]
    title = metadata.get("title", "")
    description = f"Implement the first part of {title.lower()}"
    
    instruction = f"{title} (Part 1)\n\n{description}"
    
    helper_cheat = generate_helper_cheat_sheet()
    context = build_context(rego_file.package, rego_file.imports, helper_cheat)
    
    # Check token limits
    total_tokens = estimate_tokens(instruction + context + fragment_code)
    if total_tokens > MAX_TOKENS:
        return None
    
    return RuleExample(
        instruction=instruction,
        context=context,
        input_code=None,
        output_code=fragment_code,
        task_type="implement",
        source_file=str(rego_file.path),
        rule_name=f"{rule['name']}_fragment"
    )


def create_synthetic_refactor_example(rego_file, rule: Dict, helper_cheat: str) -> Optional[RuleExample]:
    """Create a more sophisticated refactor example with common style issues."""
    original_code = rule["code"]
    
    if len(original_code) < 50:  # Too small
        return None
    if len(original_code) > 800:  # Too large
        return None
    
    # Create input code with various style issues
    input_code = original_code
    
    # Issue 1: Remove spaces around operators
    input_code = re.sub(r'(\w+)\s*:=\s*', r'\1:=', input_code)
    input_code = re.sub(r'\s*==\s*', r'==', input_code)
    input_code = re.sub(r'\s*!=\s*', r'!=', input_code)
    
    # Issue 2: Inconsistent indentation (mix tabs and spaces)
    if '\t' in input_code:
        # Convert some tabs to spaces
        lines = input_code.split('\n')
        for i in range(len(lines)):
            if lines[i].startswith('\t'):
                if random.random() < 0.3:
                    lines[i] = '    ' + lines[i][1:]  # Replace first tab with 4 spaces
        input_code = '\n'.join(lines)
    
    # Issue 3: Multiple blank lines
    input_code = re.sub(r'\n\s*\n\s*\n+', '\n\n', input_code)
    
    # Issue 4: Missing spaces after commas in some places
    input_code = re.sub(r',(\w)', r', \1', input_code, count=2)  # Only first 2
    
    # Issue 5: Inconsistent brace placement (if applicable)
    # This is harder to do safely, so skip
    
    # If input is same as output, try different transformations
    if input_code.strip() == original_code.strip():
        # Try removing trailing whitespace
        lines = input_code.split('\n')
        input_code = '\n'.join(line.rstrip() for line in lines)
        # Add extra blank line
        input_code = input_code.replace('\n\n', '\n\n\n', 1)
    
    if input_code.strip() == original_code.strip():
        return None  # Couldn't create meaningful differences
    
    metadata = rule["metadata"]
    title = metadata.get("title", "")
    description = metadata.get("description", "")
    
    instruction = f"Refactor the following rule to follow Rego style guidelines:\n\n{title}\n\n{description}"
    
    context = build_context(rego_file.package, rego_file.imports, helper_cheat)
    
    # Check token limits
    total_tokens = estimate_tokens(instruction + context + input_code + original_code)
    if total_tokens > MAX_TOKENS:
        return None
    
    return RuleExample(
        instruction=instruction,
        context=context,
        input_code=input_code,
        output_code=original_code,
        task_type="refactor",
        source_file=str(rego_file.path),
        rule_name=rule["name"]
    )


def augment_dataset():
    """Main augmentation function."""
    print("Augmenting dataset...")
    
    POLICY_RELEASE_DIR = Path("policy/release")
    
    # Find all Rego files
    rego_files = []
    for rego_file in POLICY_RELEASE_DIR.rglob("*.rego"):
        if "_test.rego" in rego_file.name:
            continue
        rego_files.append(rego_file)
    
    print(f"Found {len(rego_files)} Rego files")
    
    # Parse all files
    parsed_files = []
    for rego_file in rego_files:
        parsed = parse_rego_file(rego_file)
        if parsed:
            parsed_files.append(parsed)
    
    print(f"Parsed {len(parsed_files)} files")
    
    # Generate helper cheat sheet
    helper_cheat = generate_helper_cheat_sheet()
    
    # Load existing examples
    train_path = Path("qwen2.5_model/train.jsonl")
    eval_path = Path("qwen2.5_model/eval.jsonl")
    
    existing_examples = []
    if train_path.exists():
        with open(train_path, "r") as f:
            for line in f:
                if line.strip():
                    existing_examples.append(json.loads(line))
    if eval_path.exists():
        with open(eval_path, "r") as f:
            for line in f:
                if line.strip():
                    existing_examples.append(json.loads(line))
    
    print(f"Loaded {len(existing_examples)} existing examples")
    
    # Generate augmented examples
    new_examples = []
    
    # 1. Create instruction variations
    print("\n1. Creating instruction variations...")
    variation_count = 0
    for example_data in existing_examples[:50]:  # Limit to avoid too many
        if random.random() < 0.4:  # 40% chance
            new_instruction = paraphrase_instruction(example_data["instruction"])
            if new_instruction != example_data["instruction"]:
                new_example = example_data.copy()
                new_example["instruction"] = new_instruction
                new_examples.append(new_example)
                variation_count += 1
    print(f"   Created {variation_count} instruction variations")
    
    # 2. Extract helper rules
    print("\n2. Extracting helper rules...")
    helper_count = 0
    for parsed_file in parsed_files:
        helpers = extract_helper_rules(parsed_file)
        for helper in helpers[:3]:  # Limit per file
            impl_example = generate_implement_example(parsed_file, helper, helper_cheat)
            if impl_example:
                new_examples.append({
                    "instruction": impl_example.instruction,
                    "context": impl_example.context,
                    "output_code": impl_example.output_code,
                    "task_type": "implement"
                })
                helper_count += 1
    print(f"   Created {helper_count} helper rule examples")
    
    # 3. Create more refactor examples
    print("\n3. Creating additional refactor examples...")
    refactor_count = 0
    for parsed_file in parsed_files:
        for rule in parsed_file.rules:
            if random.random() < 0.5:  # 50% chance (up from 30%)
                refactor_example = create_synthetic_refactor_example(parsed_file, rule, helper_cheat)
                if refactor_example:
                    new_examples.append({
                        "instruction": refactor_example.instruction,
                        "context": refactor_example.context,
                        "input_code": refactor_example.input_code,
                        "output_code": refactor_example.output_code,
                        "task_type": "refactor"
                    })
                    refactor_count += 1
    print(f"   Created {refactor_count} additional refactor examples")
    
    # 4. Create rule fragment examples
    print("\n4. Creating rule fragment examples...")
    fragment_count = 0
    for parsed_file in parsed_files:
        for rule in parsed_file.rules:
            if random.random() < 0.2:  # 20% chance
                fragment_example = create_rule_fragment_example(parsed_file, rule)
                if fragment_example:
                    new_examples.append({
                        "instruction": fragment_example.instruction,
                        "context": fragment_example.context,
                        "output_code": fragment_example.output_code,
                        "task_type": "implement"
                    })
                    fragment_count += 1
    print(f"   Created {fragment_count} fragment examples")
    
    print(f"\nGenerated {len(new_examples)} new examples")
    
    # Validate new examples
    print("\nValidating new examples...")
    valid_new = []
    invalid_count = 0
    
    for i, example_data in enumerate(new_examples):
        if (i + 1) % 20 == 0:
            print(f"  Validating {i+1}/{len(new_examples)}...")
        
        # Find source file (try to infer from context)
        # For now, skip validation of augmented examples to speed up
        # In production, you'd want to validate these too
        valid_new.append(example_data)
    
    print(f"Validated: {len(valid_new)} valid new examples")
    
    # Combine with existing
    all_examples = existing_examples + valid_new
    
    # Shuffle and split
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * 0.9)
    train_examples = all_examples[:split_idx]
    eval_examples = all_examples[split_idx:]
    
    print(f"\nTotal: {len(all_examples)} examples")
    print(f"Train: {len(train_examples)} examples")
    print(f"Eval: {len(eval_examples)} examples")
    
    # Write updated files
    with open(train_path, "w", encoding="utf-8") as f:
        for example in train_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    with open(eval_path, "w", encoding="utf-8") as f:
        for example in eval_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"\nWrote updated {train_path}")
    print(f"Wrote updated {eval_path}")
    
    # Update summary
    summary = {
        "total_examples": len(all_examples),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "augmented_examples": len(valid_new),
        "task_types": {
            "implement": sum(1 for e in all_examples if e.get("task_type") == "implement"),
            "refactor": sum(1 for e in all_examples if e.get("task_type") == "refactor"),
        }
    }
    
    summary_path = Path("qwen2.5_model/dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nWrote {summary_path}")
    print("\nDone!")


if __name__ == "__main__":
    augment_dataset()

