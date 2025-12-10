#!/usr/bin/env python3
"""
Validate two-stage training data quality.

This script checks:
1. Stage 1: Do helpers exist? Are schema paths valid?
2. Stage 2: Does the Rego code compile? Does it match the requirements?
3. General: Are all required sections present?

Usage:
    python scripts/validate_two_stage_data.py
    python scripts/validate_two_stage_data.py --sample 10  # Check 10 random examples
    python scripts/validate_two_stage_data.py --export-review review.md  # Export for human review
"""

import json
import re
import subprocess
import sys
import random
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from library_mapper import LibraryMapper
    from library_indexer import LibraryIndexer
except ImportError:
    LibraryMapper = None
    LibraryIndexer = None

REPO_ROOT = Path(__file__).parent.parent
DATA_DIR = REPO_ROOT / "data" / "training" / "two_stage"


@dataclass
class ValidationResult:
    """Result of validating a single example."""
    file_path: str
    index: int
    stage: int
    
    # Structural checks
    has_instruction: bool = False
    has_input: bool = False
    has_output: bool = False
    
    # Stage 1 checks
    has_attestation_schema: bool = False
    has_available_helpers: bool = False
    helpers_exist: List[str] = field(default_factory=list)  # Helpers that exist
    helpers_missing: List[str] = field(default_factory=list)  # Helpers that don't exist
    
    # Stage 2 checks
    has_analysis: bool = False
    has_rule: bool = False
    has_tests: bool = False
    rego_compiles: bool = False
    rego_error: str = ""
    
    # Quality metrics
    requirements_length: int = 0
    output_length: int = 0
    
    @property
    def is_valid(self) -> bool:
        """Basic validity check."""
        if self.stage == 1:
            return (self.has_instruction and self.has_input and self.has_output and
                    self.has_attestation_schema and self.has_available_helpers)
        else:
            return (self.has_instruction and self.has_input and self.has_output and
                    self.has_analysis and self.has_rule and self.rego_compiles)


def load_examples(file_path: Path) -> List[dict]:
    """Load JSONL examples."""
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def extract_helpers_from_output(output: str) -> List[str]:
    """Extract helper names from AVAILABLE_HELPERS section."""
    helpers = []
    in_helpers = False
    
    for line in output.split('\n'):
        if 'AVAILABLE_HELPERS:' in line:
            in_helpers = True
            continue
        if in_helpers:
            if line.startswith('RULE_DATA_KEYS:') or line.strip() == '':
                break
            # Match "- name: lib.something" or "- lib.something:"
            match = re.search(r'[-•]\s*(?:name:\s*)?(\w+\.\w+)', line)
            if match:
                helpers.append(match.group(1))
    
    return helpers


def extract_rego_code(output: str) -> Optional[str]:
    """Extract Rego code from RULE section."""
    # Match ```rego ... ``` blocks after RULE:
    # The format is: RULE:\n```rego\ncode\n```
    match = re.search(r'RULE:\s*\n```rego\n(.*?)\n```', output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def validate_rego_syntax(code: str) -> Tuple[bool, str]:
    """Validate Rego code compiles using ec opa."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = Path(f.name)
    
    try:
        # Use ec opa parse for Enterprise Contract compatibility
        result = subprocess.run(
            ["ec", "opa", "parse", str(tmp_path)],
            capture_output=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, ""
        else:
            error = result.stderr.decode()[:200]
            return False, error
    except FileNotFoundError:
        # Fallback to standard opa if ec not found
        try:
            result = subprocess.run(
                ["opa", "parse", str(tmp_path)],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return True, ""
            else:
                error = result.stderr.decode()[:200]
                return False, error
        except FileNotFoundError:
            return True, "(ec/opa not found, skipping syntax check)"
    except Exception as e:
        return False, str(e)
    finally:
        tmp_path.unlink()


def validate_stage1_example(example: dict, indexer: Optional[object], index: int, file_path: str) -> ValidationResult:
    """Validate a Stage 1 example."""
    result = ValidationResult(file_path=file_path, index=index, stage=1)
    
    # Structural checks
    result.has_instruction = bool(example.get('instruction'))
    result.has_input = bool(example.get('input'))
    result.has_output = bool(example.get('output'))
    
    output = example.get('output', '')
    input_text = example.get('input', '')
    
    result.requirements_length = len(input_text)
    result.output_length = len(output)
    
    # Section checks
    result.has_attestation_schema = 'ATTESTATION_SCHEMA:' in output
    result.has_available_helpers = 'AVAILABLE_HELPERS:' in output
    
    # Helper existence check
    if indexer:
        helpers = extract_helpers_from_output(output)
        for helper in helpers:
            # Strip module prefix for lookup
            func_name = helper.split('.')[-1] if '.' in helper else helper
            if func_name in indexer.index or helper == 'rego.metadata.chain()':
                result.helpers_exist.append(helper)
            else:
                result.helpers_missing.append(helper)
    
    return result


def validate_stage2_example(example: dict, index: int, file_path: str) -> ValidationResult:
    """Validate a Stage 2 example."""
    result = ValidationResult(file_path=file_path, index=index, stage=2)
    
    # Structural checks
    result.has_instruction = bool(example.get('instruction'))
    result.has_input = bool(example.get('input'))
    result.has_output = bool(example.get('output'))
    
    output = example.get('output', '')
    input_text = example.get('input', '')
    
    result.requirements_length = len(input_text)
    result.output_length = len(output)
    
    # Section checks
    result.has_analysis = 'ANALYSIS:' in output
    result.has_rule = 'RULE:' in output
    result.has_tests = 'TESTS:' in output
    
    # Rego compilation check
    rego_code = extract_rego_code(output)
    if rego_code:
        result.rego_compiles, result.rego_error = validate_rego_syntax(rego_code)
    
    return result


def print_summary(results: List[ValidationResult], stage: int):
    """Print validation summary."""
    total = len(results)
    valid = sum(1 for r in results if r.is_valid)
    
    print(f"\n{'='*60}")
    print(f"Stage {stage} Validation Summary")
    print(f"{'='*60}")
    print(f"Total examples: {total}")
    print(f"Valid examples: {valid} ({100*valid/total:.1f}%)")
    
    if stage == 1:
        has_schema = sum(1 for r in results if r.has_attestation_schema)
        has_helpers = sum(1 for r in results if r.has_available_helpers)
        
        all_missing = []
        for r in results:
            all_missing.extend(r.helpers_missing)
        
        print(f"\nSection presence:")
        print(f"  ATTESTATION_SCHEMA: {has_schema}/{total}")
        print(f"  AVAILABLE_HELPERS:  {has_helpers}/{total}")
        
        if all_missing:
            unique_missing = set(all_missing)
            print(f"\nMissing helpers (not found in library):")
            for h in sorted(unique_missing)[:10]:
                count = all_missing.count(h)
                print(f"  - {h} ({count} occurrences)")
    
    else:
        has_analysis = sum(1 for r in results if r.has_analysis)
        has_rule = sum(1 for r in results if r.has_rule)
        has_tests = sum(1 for r in results if r.has_tests)
        compiles = sum(1 for r in results if r.rego_compiles)
        
        print(f"\nSection presence:")
        print(f"  ANALYSIS: {has_analysis}/{total}")
        print(f"  RULE:     {has_rule}/{total}")
        print(f"  TESTS:    {has_tests}/{total}")
        print(f"\nRego compilation:")
        print(f"  Compiles: {compiles}/{total} ({100*compiles/total:.1f}%)")
        
        # Show compilation errors
        errors = [r for r in results if not r.rego_compiles and r.rego_error]
        if errors[:3]:
            print(f"\nSample compilation errors:")
            for r in errors[:3]:
                print(f"  Example {r.index}: {r.rego_error[:100]}...")


def export_for_review(results: List[ValidationResult], examples: List[dict], output_path: Path, sample_size: int = 10):
    """Export examples for human review."""
    # Sample diverse examples
    valid_examples = [(r, e) for r, e in zip(results, examples) if r.is_valid]
    invalid_examples = [(r, e) for r, e in zip(results, examples) if not r.is_valid]
    
    sampled = []
    sampled.extend(random.sample(valid_examples, min(sample_size // 2, len(valid_examples))))
    sampled.extend(random.sample(invalid_examples, min(sample_size // 2, len(invalid_examples))))
    
    with open(output_path, 'w') as f:
        f.write("# Training Data Review\n\n")
        f.write(f"Sampled {len(sampled)} examples for human review.\n\n")
        f.write("For each example, check:\n")
        f.write("- [ ] Requirements are clear and specific\n")
        f.write("- [ ] ATTESTATION_SCHEMA paths are correct\n")
        f.write("- [ ] AVAILABLE_HELPERS are appropriate\n")
        f.write("- [ ] ANALYSIS explains the logic well\n")
        f.write("- [ ] RULE code matches requirements\n\n")
        f.write("---\n\n")
        
        for i, (result, example) in enumerate(sampled):
            status = "✅ Valid" if result.is_valid else "❌ Invalid"
            f.write(f"## Example {i+1} (Stage {result.stage}) - {status}\n\n")
            f.write(f"**File:** `{result.file_path}` index {result.index}\n\n")
            
            f.write("### Instruction\n\n")
            f.write(f"```\n{example.get('instruction', '')}\n```\n\n")
            
            f.write("### Input\n\n")
            f.write(f"```\n{example.get('input', '')[:1000]}\n```\n\n")
            
            f.write("### Output\n\n")
            f.write(f"```\n{example.get('output', '')[:2000]}\n```\n\n")
            
            if not result.is_valid:
                f.write("### Issues\n\n")
                if result.stage == 2 and not result.rego_compiles:
                    f.write(f"- Rego compilation error: {result.rego_error}\n")
                if result.helpers_missing:
                    f.write(f"- Missing helpers: {', '.join(result.helpers_missing)}\n")
            
            f.write("\n---\n\n")
    
    print(f"\nExported {len(sampled)} examples for review to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate two-stage training data")
    parser.add_argument("--sample", type=int, help="Validate only N random examples")
    parser.add_argument("--export-review", type=str, help="Export examples for human review")
    parser.add_argument("--stage", type=int, choices=[1, 2], help="Validate only specific stage")
    args = parser.parse_args()
    
    print("="*60)
    print("Two-Stage Training Data Validator")
    print("="*60)
    
    # Initialize library indexer for helper validation
    indexer = None
    if LibraryIndexer and LibraryMapper:
        print("\nInitializing library indexer...")
        mapper = LibraryMapper(REPO_ROOT)
        mapper.build_mappings()
        indexer = LibraryIndexer(REPO_ROOT, mapper)
        indexer.index_all_libraries(scan_usage=False)
        print(f"  Indexed {len(indexer.index)} helper functions")
    
    all_results = []
    all_examples = []
    
    # Validate Stage 1
    if args.stage is None or args.stage == 1:
        for file_name in ["stage1_train.jsonl", "stage1_eval.jsonl"]:
            file_path = DATA_DIR / file_name
            if file_path.exists():
                print(f"\nValidating {file_name}...")
                examples = load_examples(file_path)
                
                if args.sample:
                    indices = random.sample(range(len(examples)), min(args.sample, len(examples)))
                    examples = [examples[i] for i in indices]
                
                results = [
                    validate_stage1_example(ex, indexer, i, file_name) 
                    for i, ex in enumerate(examples)
                ]
                print_summary(results, 1)
                all_results.extend(results)
                all_examples.extend(examples)
    
    # Validate Stage 2
    if args.stage is None or args.stage == 2:
        for file_name in ["stage2_train.jsonl", "stage2_eval.jsonl"]:
            file_path = DATA_DIR / file_name
            if file_path.exists():
                print(f"\nValidating {file_name}...")
                examples = load_examples(file_path)
                
                if args.sample:
                    indices = random.sample(range(len(examples)), min(args.sample, len(examples)))
                    examples = [examples[i] for i in indices]
                
                results = [
                    validate_stage2_example(ex, i, file_name) 
                    for i, ex in enumerate(examples)
                ]
                print_summary(results, 2)
                all_results.extend(results)
                all_examples.extend(examples)
    
    # Export for human review if requested
    if args.export_review:
        export_for_review(all_results, all_examples, Path(args.export_review))
    
    # Overall summary
    total_valid = sum(1 for r in all_results if r.is_valid)
    print(f"\n{'='*60}")
    print(f"Overall: {total_valid}/{len(all_results)} examples valid ({100*total_valid/len(all_results):.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()

