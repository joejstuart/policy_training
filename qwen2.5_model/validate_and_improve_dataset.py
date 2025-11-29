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
                ['./opa', 'parse', temp_file],
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


@dataclass
class ValidationResult:
    """Result of validating a training example."""
    is_valid: bool
    issues: List[str]
    improvements: List[str]
    improved_code: Optional[str] = None
    style_guide_violations: List[str] = None


def validate_with_regal(code: str) -> Tuple[bool, List[str]]:
    """
    Run Regal linter on the given Rego code.
    
    Returns (ok, issues), where:
      - ok = True if no errors (warnings are allowed)
      - issues = list of human-readable messages
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', delete=False, encoding='utf-8') as f:
            f.write(code)
            temp_file = f.name
        
        # Basic regal invocation; adjust flags/config paths as needed
        result = subprocess.run(
            ['./regal', 'lint', temp_file],
            capture_output=True,
            text=True,
            timeout=10,
        )
        
        Path(temp_file).unlink(missing_ok=True)
        
        if result.returncode == 0:
            # No findings
            return True, []
        
        # Regal prints findings to stdout; stderr is more for fatal errors
        output = result.stdout.strip() or result.stderr.strip()
        if not output:
            return False, ['regal lint failed with no output']
        
        # Parse output - Regal can output in different formats
        # Try JSON first, fall back to text
        issues = []
        try:
            # Try JSON format
            regal_output = json.loads(output)
            if 'violations' in regal_output:
                for violation in regal_output['violations']:
                    rule_name = violation.get('title', 'Unknown rule')
                    description = violation.get('description', '')
                    location = violation.get('location', {})
                    line = location.get('row', '?')
                    col = location.get('col', '?')
                    issues.append(f"{rule_name}: {description} (line {line}, col {col})")
            elif 'summary' in regal_output:
                # Alternative JSON format
                for finding in regal_output.get('findings', []):
                    issues.append(finding.get('message', str(finding)))
        except (json.JSONDecodeError, KeyError):
            # Fall back to text parsing
            for line in output.splitlines():
                if line.strip() and not line.startswith('#'):
                    issues.append(line.strip())
        
        return False, issues if issues else ['regal lint reported issues (unable to parse output)']
        
    except FileNotFoundError:
        # regal not installed; treat as no-op
        return True, []
    except Exception as e:
        return False, [f'regal lint error: {e}']


def check_style_guide_compliance(code: str, logger=None) -> List[str]:
    """Check custom style guide patterns (complements Regal)."""
    violations = []
    
    # Custom checks that complement Regal
    # Check for 'every' usage (should be used for FOR ALL)
    if 'not some' in code and 'every' not in code:
        # Check if it's a FOR ALL pattern
        if re.search(r'not\s+some\s+.*\s+in\s+.*\s+\{.*\s+!=', code):
            violations.append("Consider using 'every' instead of 'not some' for FOR ALL queries")
    
    return violations


def _format_validation_error(error_msg: str) -> str:
    """Format validation error message for better readability.
    
    Handles both JSON and plain text error formats from OPA and Regal.
    Both can output JSON with similar structures.
    """
    if not error_msg or not error_msg.strip():
        return "Unknown validation error"
    
    # Try to parse as JSON first (both OPA and Regal can output JSON)
    try:
        error_json = json.loads(error_msg)
        
        # OPA format: {"errors": [{"message": "...", "location": {"row": ..., "col": ...}}]}
        if 'errors' in error_json:
            formatted_errors = []
            for err in error_json['errors']:
                message = err.get('message', 'Unknown error')
                location = err.get('location', {})
                line = location.get('row', location.get('line', '?'))
                col = location.get('col', location.get('column', '?'))
                file_path = location.get('file', '')
                
                # Format: "Line X, col Y: message" or "file:line:col: message"
                if file_path and file_path != '<temp file>':
                    formatted_errors.append(f"{file_path}:{line}:{col}: {message}")
                else:
                    formatted_errors.append(f"Line {line}, col {col}: {message}")
            return "\n".join(formatted_errors)
        
        # Regal format might have different structure - handle if needed
        # Regal typically outputs to stdout in text format, but could be JSON
        if 'violations' in error_json or 'findings' in error_json:
            # This is likely Regal output, but we're here for syntax errors (OPA)
            # Regal violations are handled separately
            pass
            
    except (json.JSONDecodeError, KeyError, TypeError):
        # Not JSON, return as-is (might already be formatted text)
        pass
    
    # Return original error message (could be plain text from OPA stderr)
    return error_msg


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
    instruction_lower = instruction.lower()
    
    # Check for correct navigation paths (more lenient - allow helpers)
    if 'task' in instruction_lower:
        # Check if code accesses tasks in any way (direct or via helpers)
        if 'buildConfig.tasks' not in code and 'tasks' not in code and '.task' not in code:
            issues.append("Instruction mentions 'task' but code doesn't appear to iterate tasks")
    
    if 'material' in instruction_lower:
        # Check if code accesses materials
        if 'materials' not in code and '.material' not in code:
            issues.append("Instruction mentions 'material' but code doesn't appear to access materials")
    
    if 'subject' in instruction_lower:
        # Check if code accesses subject
        if 'subject' not in code and '.subject' not in code:
            issues.append("Instruction mentions 'subject' but code doesn't appear to access subject")
    
    # Check for proper array iteration (only if accessing attestations directly)
    if 'input.attestations' in code:
        if 'some' not in code and 'every' not in code:
            issues.append("Code accesses attestations but doesn't iterate with 'some' or 'every'")
    
    # Check for proper package and import (hard requirements)
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
"""
    
    # Include context (attestation JSON structure) if available
    if 'context' in example and example['context']:
        prompt += f"""
ATTESTATION CONTEXT (the JSON structure the code should parse):
{example['context']}
"""
    
    prompt += f"""
CURRENT OUTPUT CODE:
```rego
{example['output_code']}
```

ISSUES FOUND:
"""
    
    if issues:
        prompt += "\n".join(f"- {issue}" for issue in issues)
    
    if style_violations:
        prompt += "\n\nSTYLE / LINT FINDINGS (from Regal and custom checks):"
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
- Correctly parses the attestation structure shown in the context
- Matches the instruction requirements

Output only the improved Rego code, no explanations."""
    
    return prompt


def improve_example_with_llm(example: Dict, issues: List[str], style_violations: List[str], 
                             tokenizer, model, device, logger=None, max_corrections: int = 3) -> Optional[str]:
    """Use LLM to improve the example with iterative correction.
    
    If the improved code fails validation, feeds the errors back to the LLM
    for correction, up to max_corrections times.
    """
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
        
        for attempt in range(max_corrections):
            # Generate response
            response = generate_response(tokenizer, model, device, messages, max_tokens=1024, temperature=0.3)
            
            if logger:
                logger.debug(f"Raw LLM response (attempt {attempt + 1}):\n{response[:500]}...")  # First 500 chars
            
            # Extract code from response
            improved_code = extract_rego_code(response)
            
            if logger:
                logger.debug(f"Extracted code length: {len(improved_code) if improved_code else 0} characters")
                if improved_code:
                    logger.debug(f"Extracted code (full):\n{improved_code}")
                else:
                    logger.warning(f"Failed to extract code from response. Response preview:\n{response[:500]}")
            
            if not improved_code:
                if logger:
                    logger.warning(f"LLM improvement attempt {attempt + 1} returned no code")
                    logger.warning(f"Raw response was: {response[:200]}")
                continue
            
            # Log what we extracted before validation
            if logger:
                logger.debug(f"Code before validation ({len(improved_code)} chars):\n{improved_code}")
            
            # Validate the improved code
            is_valid, formatted_code, error_msg = validate_rego_syntax(
                improved_code,
                package="attestation_check",
                imports=["rego.v1"]
            )
            
            # Use formatted_code if validation modified it (added package/imports)
            if formatted_code and formatted_code != improved_code:
                if logger:
                    logger.debug(f"Validation added package/imports. Using formatted code.")
                improved_code = formatted_code
            
            if is_valid:
                if logger:
                    if attempt > 0:
                        logger.info(f"LLM corrected code after {attempt + 1} attempt(s)")
                    # Print before/after comparison
                    original_code = example.get('output_code', '')
                    logger.info("=" * 80)
                    logger.info("CODE IMPROVEMENT - BEFORE:")
                    logger.info("-" * 80)
                    for i, line in enumerate(original_code.split('\n'), 1):
                        logger.info(f"{i:4d} | {line}")
                    logger.info("-" * 80)
                    logger.info("CODE IMPROVEMENT - AFTER:")
                    logger.info("-" * 80)
                    if improved_code and improved_code.strip():
                        for i, line in enumerate(improved_code.split('\n'), 1):
                            logger.info(f"{i:4d} | {line}")
                    else:
                        logger.warning("WARNING: Improved code is empty or invalid!")
                        logger.warning(f"Raw response was: {response[:500]}")
                    logger.info("=" * 80)
                return improved_code
            
            # Code failed validation - feed error back to LLM
            if attempt < max_corrections - 1:
                if logger:
                    logger.debug(f"LLM code failed validation (attempt {attempt + 1}), feeding error back...")
                
                # Format error message for better readability
                formatted_error = _format_validation_error(error_msg)
                
                # Create correction prompt with the validation error
                # Include context if available
                context_section = ""
                if 'context' in example and example['context']:
                    context_section = f"""
ATTESTATION CONTEXT (for reference):
{example['context']}

"""
                
                correction_prompt = f"""The generated Rego code has syntax errors. Please fix them.
{context_section}VALIDATION ERRORS:
{formatted_error}

GENERATED CODE (with errors):
```rego
{improved_code}
```

ORIGINAL INSTRUCTION:
{example.get('instruction', 'N/A')}

Please provide the corrected Rego code that fixes these syntax errors. Ensure:
- All rules have proper 'if' keywords before rule bodies
- All syntax is valid Rego
- The code still addresses the original instruction and issues
- The code correctly parses the attestation structure shown in the context

Output only the corrected Rego code, no explanations."""
                
                # Add to conversation for next iteration
                messages.append({"role": "assistant", "content": improved_code})
                messages.append({"role": "user", "content": correction_prompt})
            else:
                # Max attempts reached
                if logger:
                    logger.warning(f"LLM improved code failed validation after {max_corrections} attempts: {error_msg}")
                return None
        
        return None
        
    except Exception as e:
        if logger:
            log_exception(logger, e, context="LLM improvement failed")
        return None


def validate_example(example: Dict, tokenizer=None, model=None, device=None, use_llm: bool = False, logger=None) -> ValidationResult:
    """Validate a single training example.
    
    Uses 3-layer validation:
    1. Syntax (opa parse) → hard error
    2. Structure / attestation parsing → hard error
    3. Regal + custom style checks → style violations (soft errors)
    
    Note: Style violations are treated as warnings, not hard failures.
    An example is valid if it has no syntax errors or parsing issues.
    """
    if logger is None:
        import logging
        logger = logging.getLogger("validate_example")
    
    issues: List[str] = []
    style_violations: List[str] = []
    improved_code: Optional[str] = None
    
    # Extract output code
    output_code = example.get('output_code', '')
    if not output_code:
        return ValidationResult(
            is_valid=False,
            issues=["Missing output_code"],
            improvements=[],
            style_guide_violations=[]
        )
    
    # 1. Syntax validation (hard requirement)
    is_valid_syntax, error_msg = validate_rego_code(output_code)
    if not is_valid_syntax:
        issues.append(f"Rego syntax error: {error_msg}")
    
    # 2. Structural / attestation checks (hard requirements)
    parsing_issues = check_attestation_parsing(output_code, example.get('instruction', ''))
    issues.extend(parsing_issues)
    
    # 3. Regal lint (style + best practices)
    regal_ok, regal_issues = validate_with_regal(output_code)
    if not regal_ok and regal_issues:
        style_violations.extend([f"Regal: {msg}" for msg in regal_issues])
    
    # 4. Custom style checks (complements Regal)
    custom_style_issues = check_style_guide_compliance(output_code, logger)
    style_violations.extend(custom_style_issues)
    
    improvements: List[str] = []
    
    # 5. Use LLM only if something is off
    if (issues or style_violations) and use_llm and tokenizer and model:
        improved = improve_example_with_llm(
            example,
            issues,
            style_violations,
            tokenizer,
            model,
            device,
            logger=logger,
            max_corrections=3
        )
        if improved:
            improved_code = improved
            improvements.append("Code improved using LLM")
            
            # Re-run validation on improved code
            is_valid_syntax_improved, _ = validate_rego_code(improved_code)
            if not is_valid_syntax_improved:
                logger.warning("Improved code is syntactically invalid; keeping original")
                improved_code = None
            else:
                # Recompute checks on improved code
                issues = check_attestation_parsing(improved_code, example.get('instruction', ''))
                regal_ok, regal_issues = validate_with_regal(improved_code)
                style_violations = []
                if not regal_ok and regal_issues:
                    style_violations.extend([f"Regal: {msg}" for msg in regal_issues])
                style_violations.extend(check_style_guide_compliance(improved_code, logger))
    
    # Treat only structural/syntax issues as "invalid"
    is_valid = len(issues) == 0
    
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
        result = validate_example(example, tokenizer, model, device, use_llm, logger)
        validation_results.append((line_num, example, result))
        
        if result.is_valid:
            valid_count += 1
            # Log style warnings even for valid examples
            if result.style_guide_violations:
                logger.debug(f"  ✓ Valid (with style warnings):")
                for violation in result.style_guide_violations:
                    logger.debug(f"    - Style: {violation}")
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
                # Style violations are warnings, not blockers
                if result.style_guide_violations:
                    logger.info(f"    Style warnings (non-blocking):")
                    for violation in result.style_guide_violations:
                        logger.info(f"      - {violation}")
    
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

