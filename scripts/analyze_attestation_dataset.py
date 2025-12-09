#!/usr/bin/env python3
"""
Analyze the attestation training dataset to break down examples by attestation part.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List


def categorize_instruction(instruction: str, output_code: str) -> str:
    """Categorize an example based on instruction and output code."""
    instruction_lower = instruction.lower()
    output_lower = output_code.lower()
    
    # Task-related categories
    if "task" in instruction_lower:
        # Check for timestamp queries first (start, finish, time, timestamp)
        if any(word in instruction_lower for word in ["start", "finish", "timestamp", "time", "startedon", "finishedon", "when did"]):
            return "task_timestamp"  # Task timestamp queries
        elif "status" in instruction_lower and ("find all" in instruction_lower or "list" in instruction_lower or "get all" in instruction_lower or "show all" in instruction_lower):
            return "task_status_filter"  # Find tasks by status
        elif "status" in instruction_lower:
            return "task_status"  # Get status of specific task
        elif "result" in instruction_lower:
            return "task_results"  # Get task results
        elif "bundle" in instruction_lower:
            return "task_bundle"  # Get task bundle reference
        elif "list" in instruction_lower or ("all" in instruction_lower and "name" in instruction_lower):
            return "task_list"  # List all task names
        else:
            return "task_name"  # Find task by name
    
    # Subject-related categories
    elif "subject" in instruction_lower:
        if "digest" in instruction_lower and ("find" in instruction_lower or "which" in instruction_lower or "has digest" in instruction_lower):
            return "subject_by_digest"  # Find subject by digest
        elif "digest" in instruction_lower:
            return "subject_digest"  # Get subject digest
        elif "list" in instruction_lower or "all" in instruction_lower:
            return "subject_list"  # List all subject names
        else:
            return "subject_other"
    
    # Material-related categories
    elif "material" in instruction_lower:
        if "list" in instruction_lower or ("all" in instruction_lower and "uri" in instruction_lower):
            return "material_list"  # List all materials
        elif "uri" in instruction_lower and "commit" not in instruction_lower:
            return "material_uri_only"  # Material by URI only
        else:
            return "material_check"  # Check for material with URI/commit
    
    # Unknown
    else:
        return "other"


def analyze_dataset(jsonl_path: Path) -> Dict:
    """Analyze the dataset and return statistics."""
    categories = defaultdict(int)
    examples_by_category = defaultdict(list)
    
    total_examples = 0
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                example = json.loads(line)
                total_examples += 1
                
                instruction = example.get("instruction", "")
                output_code = example.get("output_code", "")
                
                category = categorize_instruction(instruction, output_code)
                categories[category] += 1
                examples_by_category[category].append({
                    "instruction": instruction[:80] + "..." if len(instruction) > 80 else instruction,
                    "source_file": example.get("source_file", "unknown")
                })
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    
    return {
        "total_examples": total_examples,
        "categories": dict(categories),
        "examples_by_category": {k: v[:3] for k, v in examples_by_category.items()}  # Sample 3 per category
    }


def print_breakdown(stats: Dict):
    """Print a formatted breakdown of the dataset."""
    print("=" * 80)
    print("ATTESTATION TRAINING DATASET BREAKDOWN")
    print("=" * 80)
    print()
    
    print(f"Total Examples: {stats['total_examples']}")
    print()
    
    # Define category descriptions
    category_descriptions = {
        "task_name": "Task Name Check - Find a task by name",
        "task_status": "Task Status - Get status of a specific task",
        "task_status_filter": "Task Status Filter - Find all tasks with a specific status",
        "task_results": "Task Results - Get results from a task",
        "task_bundle": "Task Bundle - Get bundle reference for a task",
        "task_list": "Task List - List all task names",
        "task_timestamp": "Task Timestamp - Get startedOn/finishedOn timestamps",
        "subject_digest": "Subject Digest - Get SHA256 digest of subject",
        "subject_list": "Subject List - List all subject names",
        "subject_other": "Subject Other - Other subject-related queries",
        "subject_by_digest": "Subject by Digest - Find subject by SHA256 digest",
        "material_check": "Material Check - Check for material with URI/commit",
        "material_uri_only": "Material URI Only - Check material by URI without commit",
        "material_list": "Material List - List all material URIs",
        "other": "Other - Unclassified examples"
    }
    
    # Group by main category
    task_categories = ["task_name", "task_status", "task_status_filter", "task_results", "task_bundle", "task_list", "task_timestamp"]
    subject_categories = ["subject_digest", "subject_list", "subject_other"]
    material_categories = ["material_check"]
    
    # Print breakdown by main section
    print("📋 BREAKDOWN BY ATTESTATION PART")
    print("-" * 80)
    print()
    
    # Tasks section
    task_total = sum(stats['categories'].get(cat, 0) for cat in task_categories)
    print(f"🔧 TASKS ({task_total} examples)")
    print()
    for cat in task_categories:
        count = stats['categories'].get(cat, 0)
        if count > 0:
            pct = (count / stats['total_examples']) * 100
            desc = category_descriptions.get(cat, cat)
            print(f"  • {desc:50s} {count:4d} ({pct:5.1f}%)")
    print()
    
    # Subjects section
    subject_total = sum(stats['categories'].get(cat, 0) for cat in subject_categories)
    print(f"📦 SUBJECTS ({subject_total} examples)")
    print()
    for cat in subject_categories:
        count = stats['categories'].get(cat, 0)
        if count > 0:
            pct = (count / stats['total_examples']) * 100
            desc = category_descriptions.get(cat, cat)
            print(f"  • {desc:50s} {count:4d} ({pct:5.1f}%)")
    print()
    
    # Materials section
    material_total = sum(stats['categories'].get(cat, 0) for cat in material_categories)
    print(f"📄 MATERIALS ({material_total} examples)")
    print()
    for cat in material_categories:
        count = stats['categories'].get(cat, 0)
        if count > 0:
            pct = (count / stats['total_examples']) * 100
            desc = category_descriptions.get(cat, cat)
            print(f"  • {desc:50s} {count:4d} ({pct:5.1f}%)")
    print()
    
    # Other
    other_count = stats['categories'].get("other", 0)
    if other_count > 0:
        print(f"❓ OTHER ({other_count} examples)")
        print()
        pct = (other_count / stats['total_examples']) * 100
        print(f"  • Unclassified examples: {other_count:4d} ({pct:5.1f}%)")
        print()
    
    # Sample examples
    print("=" * 80)
    print("SAMPLE EXAMPLES BY CATEGORY")
    print("=" * 80)
    print()
    
    for category in sorted(stats['examples_by_category'].keys()):
        examples = stats['examples_by_category'][category]
        count = stats['categories'].get(category, 0)
        print(f"{category.upper().replace('_', ' ')} ({count} total)")
        for i, ex in enumerate(examples[:2], 1):  # Show 2 examples
            print(f"  {i}. {ex['instruction']}")
        print()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze attestation training dataset")
    parser.add_argument(
        "--train-file",
        type=str,
        default="qwen2.5_model/attestation_train.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--eval-file",
        type=str,
        default="qwen2.5_model/attestation_eval.jsonl",
        help="Path to eval JSONL file (optional)",
    )
    
    args = parser.parse_args()
    
    train_path = Path(args.train_file)
    eval_path = Path(args.eval_file) if args.eval_file else None
    
    if not train_path.exists():
        print(f"Error: Training file not found: {train_path}")
        return
    
    print("Analyzing training dataset...")
    print()
    
    train_stats = analyze_dataset(train_path)
    
    if eval_path and eval_path.exists():
        print("Analyzing eval dataset...")
        print()
        eval_stats = analyze_dataset(eval_path)
        
        # Combine stats
        combined_categories = defaultdict(int)
        for cat in set(list(train_stats['categories'].keys()) + list(eval_stats['categories'].keys())):
            combined_categories[cat] = train_stats['categories'].get(cat, 0) + eval_stats['categories'].get(cat, 0)
        
        combined_stats = {
            "total_examples": train_stats['total_examples'] + eval_stats['total_examples'],
            "categories": dict(combined_categories),
            "examples_by_category": train_stats['examples_by_category']  # Use train examples for samples
        }
        
        print("=" * 80)
        print("COMBINED (TRAIN + EVAL)")
        print("=" * 80)
        print()
        print_breakdown(combined_stats)
        
        print("=" * 80)
        print("TRAINING SET ONLY")
        print("=" * 80)
        print()
        print_breakdown(train_stats)
        
        print("=" * 80)
        print("EVAL SET ONLY")
        print("=" * 80)
        print()
        print_breakdown(eval_stats)
    else:
        print_breakdown(train_stats)


if __name__ == "__main__":
    main()

