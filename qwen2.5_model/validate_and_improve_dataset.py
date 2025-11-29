#!/usr/bin/env python3
"""
Agentic data validation and improvement script for training data.

This script:
1. Reads training examples from JSONL files
2. Validates each example against:
   - Rego style guide compliance
   - Correct attestation structure parsing
   - Instruction-output alignment
   - Rego syntax correctness
3. Uses LLM to review and improve examples
4. Outputs improved training data
"""

import json
import re
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import logging setup
try:
    from logging_setup import setup_logging, log_exception
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    def setup_logging(name, **kwargs):
        return logging.getLogger(name)
    def log_exception(logger, exc, context=""):
        logger.error(f"{context}: {exc}" if context else f"Exception: {exc}", exc_info=True)

# Import validation utilities
try:
    from rego_validator import extract_rego_code, validate_rego_syntax
except ImportError:
    # Fallback validation
    def extract_rego_code(text: str) -> str:
        """Extract Rego code from text."""
        # Look for code blocks
        code_block = re.search(r'```(?:rego)?\s*\n(.*?)```', text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        # If no code block, assume entire text is code
        return text.strip()
    
    def validate_rego_syntax(code: str, package: str = "", imports: List[str] = None) -> Tuple[bool, str, str]:
        """Validate Rego syntax using opa parse."""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            result = subprocess.run(
                ['opa', 'parse', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            Path(temp_file).unlink()
            
            if result.returncode == 0:
                return True, code, ""
            else:
                return False, code, result.stderr
        except Exception as e:
            return False, code, str(e)


# Rego style guide patterns to check
STYLE_GUIDE_PATTERNS = {
    "use_in_for_membership": {
        "pattern": r'status\s*(==|!=)\s*["\']',
        "preferred": "Use 'in' for membership checks when checking multiple values",
        "example": "task.status in {\"Succeeded\", \"Failed\"}"
    },
    "use_every_for_all": {
        "pattern": r'not\s+some\s+.*\s+in\s+.*\s+\{.*status\s*!=',
        "preferred": "Use 'every' for FOR ALL queries",
        "example": "every task in att.statement.predicate.buildConfig.tasks { task.status == \"Succeeded\" }"
    },
    "use_some_in": {
        "pattern": r'for\s+.*\s+in\s+',
        "preferred": "Use 'some ... in' for iteration (declarative pattern)",
        "example": "some task in att.statement.predicate.buildConfig.tasks"
    },
    "snake_case": {
        "pattern": r'[a-z]+[A-Z]',
        "preferred": "Use snake_case for all variable and rule names",
        "example": "task_name instead of taskName"
    },
    "unconditional_assignment": {
        "pattern": r'(\w+)\s+if\s+\{\s*\1\s*:=\s*',
        "preferred": "Use unconditional assignment in rule head when possible",
        "example": "result := value if { ... } instead of result if { value := ... }"
    }
}


@dataclass
class ValidationResult:
    """Result of validating a training example."""
    is_valid: bool
    issues: List[str]
    improvements: List[str]
    improved_code: Optional[str] = None
    style_guide_violations: List[str] = None


def check_style_guide_compliance(code: str) -> List[str]:
    """Check if code follows Rego style guide patterns."""
    violations = []
    
    # Check for membership checks that should use 'in'
    if re.search(r'status\s*==\s*["\']Succeeded["\']', code) and 'in {' not in code:
        # This is a warning, not always a violation
        pass
    
    # Check for snake_case
    camel_case_vars = re.findall(r'\b[a-z]+[A-Z]\w*\b', code)
    if camel_case_vars:
        violations.append(f"Found camelCase variables: {camel_case_vars}. Use snake_case instead.")
    
    # Check for 'every' usage (should be used for FOR ALL)
    if 'not some' in code and 'every' not in code:
        # Check if it's a FOR ALL pattern
        if re.search(r'not\s+some\s+.*\s+in\s+.*\s+\{.*\s+!=', code):
            violations.append("Consider using 'every' instead of 'not some' for FOR ALL queries")
    
    return violations


def validate_rego_code(code: str) -> Tuple[bool, str]:
    """Validate Rego code syntax."""
    rego_code = extract_rego_code(code)
    is_valid, formatted_code, error_msg = validate_rego_syntax(
        rego_code,
        package="attestation_check",
        imports=["rego.v1"]
    )
    
    if not is_valid:
        return False, error_msg
    return True, ""


def check_attestation_parsing(code: str, instruction: str) -> List[str]:
    """Check if code correctly parses attestation structure."""
    issues = []
    
    # Check for correct navigation paths
    if 'task' in instruction.lower() and 'buildConfig.tasks' not in code:
        issues.append("Instruction mentions 'task' but code doesn't access buildConfig.tasks")
    
    if 'material' in instruction.lower() and 'materials' not in code:
        issues.append("Instruction mentions 'material' but code doesn't access materials")
    
    if 'subject' in instruction.lower() and 'subject' not in code:
        issues.append("Instruction mentions 'subject' but code doesn't access subject")
    
    # Check for proper array iteration
    if 'some' not in code and 'every' not in code and 'input.attestations' in code:
        issues.append("Code accesses attestations but doesn't iterate with 'some' or 'every'")
    
    # Check for proper package and import
    if 'package attestation_check' not in code:
        issues.append("Missing 'package attestation_check'")
    
    if 'import rego.v1' not in code:
        issues.append("Missing 'import rego.v1'")
    
    return issues


def create_improvement_prompt(example: Dict, issues: List[str], style_violations: List[str]) -> str:
    """Create a prompt for LLM to improve the example."""
    prompt = f"""You are reviewing a training example for Rego attestation parsing. Please improve the output_code to fix the following issues:

INSTRUCTION:
{example['instruction']}

CURRENT OUTPUT CODE:
```rego
{example['output_code']}
```

ISSUES FOUND:
"""
    
    if issues:
        prompt += "\n".join(f"- {issue}" for issue in issues)
    
    if style_violations:
        prompt += "\n\nSTYLE GUIDE VIOLATIONS:"
        prompt += "\n".join(f"- {violation}" for violation in style_violations)
    
    prompt += """

REGO STYLE GUIDE REQUIREMENTS:
1. Use 'in' for membership checks when checking multiple values
2. Use 'every' for FOR ALL queries (e.g., 'all tasks succeeded')
3. Use 'some ... in' for iteration (declarative pattern)
4. Prefer sets over arrays when order doesn't matter
5. Use unconditional assignment in rule head when possible
6. Use snake_case for all variable and rule names
7. Always include 'package attestation_check' and 'import rego.v1'

Please provide the improved Rego code that:
- Fixes all the issues listed above
- Follows the Rego style guide
- Correctly parses the attestation structure
- Matches the instruction requirements

Output only the improved Rego code, no explanations."""
    
    return prompt


def improve_example_with_llm(example: Dict, issues: List[str], style_violations: List[str], 
                             tokenizer, model, device) -> Optional[str]:
    """Use LLM to improve the example."""
    prompt = create_improvement_prompt(example, issues, style_violations)
    
    messages = [
        {
            "role": "system",
            "content": "You are an expert Rego/OPA policy assistant. You write correct, style-compliant Rego code."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    try:
        from infer_policy import generate_response
        response = generate_response(tokenizer, model, device, messages, max_tokens=1024, temperature=0.3)
        
        # Extract code from response
        improved_code = extract_rego_code(response)
        
        # Validate the improved code
        is_valid, _, error_msg = validate_rego_syntax(
            improved_code,
            package="attestation_check",
            imports=["rego.v1"]
        )
        
        if is_valid:
            return improved_code
        else:
            return None
    except Exception as e:
        return None


def validate_example(example: Dict, tokenizer=None, model=None, device=None, use_llm: bool = False) -> ValidationResult:
    """Validate a single training example."""
    issues = []
    style_violations = []
    improved_code = None
    
    # Extract output code
    output_code = example.get('output_code', '')
    if not output_code:
        return ValidationResult(
            is_valid=False,
            issues=["Missing output_code"],
            improvements=[],
            style_guide_violations=[]
        )
    
    # 1. Validate Rego syntax
    is_valid, error_msg = validate_rego_code(output_code)
    if not is_valid:
        issues.append(f"Rego syntax error: {error_msg}")
    
    # 2. Check style guide compliance
    style_violations = check_style_guide_compliance(output_code)
    
    # 3. Check attestation parsing
    parsing_issues = check_attestation_parsing(output_code, example.get('instruction', ''))
    issues.extend(parsing_issues)
    
    # 4. Try to improve if there are issues and LLM is available
    improvements = []
    if (issues or style_violations) and use_llm and tokenizer and model:
        improved_code = improve_example_with_llm(example, issues, style_violations, tokenizer, model, device)
        if improved_code:
            improvements.append("Code improved using LLM")
            # Re-validate improved code
            is_valid_improved, _ = validate_rego_code(improved_code)
            if is_valid_improved:
                issues = []  # Clear issues if improved code is valid
                style_violations = check_style_guide_compliance(improved_code)
    
    is_valid = len(issues) == 0 and len(style_violations) == 0
    
    return ValidationResult(
        is_valid=is_valid,
        issues=issues,
        improvements=improvements,
        improved_code=improved_code,
        style_guide_violations=style_violations
    )


def process_jsonl_file(input_path: Path, output_path: Path, use_llm: bool = False,
                      tokenizer=None, model=None, device=None, logger=None):
    """Process a JSONL file and validate/improve examples."""
    if logger is None:
        logger = setup_logging("validate_dataset")
    
    examples = []
    validation_results = []
    
    # Read examples
    logger.info(f"Reading examples from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    example = json.loads(line)
                    examples.append((line_num, example))
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Invalid JSON - {e}")
    
    logger.info(f"Loaded {len(examples)} examples")
    
    # Validate each example
    valid_count = 0
    improved_count = 0
    invalid_count = 0
    
    for line_num, example in examples:
        logger.info(f"Validating example {line_num}/{len(examples)}...")
        result = validate_example(example, tokenizer, model, device, use_llm)
        validation_results.append((line_num, example, result))
        
        if result.is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            if result.improved_code:
                improved_count += 1
                # Update example with improved code
                example['output_code'] = result.improved_code
                logger.info(f"  ✓ Improved example {line_num}")
            else:
                logger.warning(f"  ✗ Example {line_num} has issues:")
                for issue in result.issues:
                    logger.warning(f"    - {issue}")
                for violation in result.style_guide_violations:
                    logger.warning(f"    - Style: {violation}")
    
    # Write improved examples
    logger.info(f"Writing results to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for line_num, example, result in validation_results:
            # Use improved code if available, otherwise original
            if result.improved_code:
                example['output_code'] = result.improved_code
            
            # Add validation metadata
            if not result.is_valid and not result.improved_code:
                example['_validation_issues'] = result.issues + result.style_guide_violations
            
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Summary
    logger.info("=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)
    logger.info(f"Total examples: {len(examples)}")
    logger.info(f"Valid examples: {valid_count}")
    logger.info(f"Invalid examples: {invalid_count}")
    logger.info(f"Improved examples: {improved_count}")
    logger.info(f"Output written to: {output_path}")
    logger.info("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Validate and improve training data using agentic validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate without LLM improvement (faster, rule-based only)
  python qwen2.5_model/validate_and_improve_dataset.py \\
      --input qwen2.5_model/attestation_train.jsonl \\
      --output qwen2.5_model/attestation_train_improved.jsonl

  # Validate with LLM improvement (slower, but fixes issues)
  python qwen2.5_model/validate_and_improve_dataset.py \\
      --input qwen2.5_model/attestation_train.jsonl \\
      --output qwen2.5_model/attestation_train_improved.jsonl \\
      --use-llm \\
      --model-dir qwen2.5-attestation-parse
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file with training examples"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file for improved examples"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM to improve examples (requires --model-dir or --base-model)"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Path to fine-tuned model directory (for LLM improvement)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name (if not using --model-dir)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "mps", "cpu", "cuda"],
        help="Device to run on (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("validate_dataset")
    
    # Resolve paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Load model if using LLM
    tokenizer = None
    model = None
    device = None
    
    if args.use_llm:
        logger.info("Loading model for LLM-based improvement...")
        try:
            from infer_policy import load_policy_model
            tokenizer, model, device = load_policy_model(
                base_model=args.base_model,
                model_dir=args.model_dir,
                device=args.device,
                no_lora=False
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Falling back to rule-based validation only")
            args.use_llm = False
    
    # Process file
    process_jsonl_file(
        input_path,
        output_path,
        use_llm=args.use_llm,
        tokenizer=tokenizer,
        model=model,
        device=device,
        logger=logger
    )
    
    logger.info("Validation complete!")


if __name__ == "__main__":
    main()

