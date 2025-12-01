"""Validate and test Rego code using OPA tools.

This module uses the OPA CLI for validation. For better performance,
a Go-based validator could be created that uses OPA's Go libraries directly.
See: https://pkg.go.dev/github.com/open-policy-agent/opa/ast
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List


def has_meaningful_content(code: str) -> bool:
    """Check if code has actual rules/content, not just package/imports.
    
    Returns True if code contains at least one rule (deny/warn/allow) or function definition.
    """
    if not code:
        return False
    
    code_lower = code.lower()
    
    # Remove package and import lines to check what's left
    lines = code.split('\n')
    content_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip package, import, and empty lines
        if stripped.startswith('package ') or stripped.startswith('import ') or not stripped:
            continue
        content_lines.append(stripped)
    
    # If no content lines after removing package/imports, it's not meaningful
    if not content_lines:
        return False
    
    # Check for actual rules or functions
    # Look for: deny, warn, allow, rule definitions, function definitions
    meaningful_patterns = [
        r'\bdeny\s+',
        r'\bwarn\s+',
        r'\ballow\s+',
        r'\bcontains\s+',
        r'\bif\s+\{',  # Rule body
        r':=\s+',  # Assignment (likely in a rule)
        r'\bdefault\s+',  # Default rule
    ]
    
    content_text = '\n'.join(content_lines)
    return any(re.search(pattern, content_text, re.IGNORECASE) for pattern in meaningful_patterns)


def extract_rego_code(text: str) -> Optional[str]:
    """Extract Rego code from model response.
    
    Looks for code blocks marked with ```rego or ``` or just Rego code.
    Only returns code that has meaningful content (actual rules, not just package/imports).
    """
    # First, try to find markdown code blocks (most common format)
    # Pattern: ```rego\n...code...\n``` or ```\n...code...\n```
    # Use a more robust pattern that handles various whitespace
    code_block_patterns = [
        r'```rego\s*\n(.*?)```',  # ```rego ... ```
        r'```\s*\n(.*?)```',       # ``` ... ```
    ]
    
    for pattern in code_block_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            code = matches[0].strip()
            # Aggressively remove any backticks (markdown artifacts)
            # Remove all backticks - they shouldn't be in Rego code anyway
            code = code.replace('`', '')
            code = code.strip()
            if code and has_meaningful_content(code):
                return code
    
    # If no code block, look for package declaration (likely Rego code)
    # But be more careful - extract more content, not stopping at first double newline
    if 'package ' in text:
        # Try to find the full code block - look for package and then continue until we hit meaningful content
        # First, find where package starts
        package_match = re.search(r'package\s+\S+', text)
        if package_match:
            start_pos = package_match.start()
            # Extract from package to end of text (or next markdown block)
            # But we need to make sure we get the full code, not stop at first \n\n
            remaining = text[start_pos:]
            # Stop at ``` or end, but don't stop at \n\n if there's more content
            # Look for the end of the code block
            end_match = re.search(r'```', remaining)
            if end_match:
                code = remaining[:end_match.start()].strip()
            else:
                # No closing ```, take everything but stop at next markdown block or end
                code = remaining.split('```')[0].strip()
            
            # Remove any backticks
            code = code.replace('`', '')
            code = code.strip()
            if code and has_meaningful_content(code):
                return code
    
    # Last resort: return the whole text if it looks like Rego and has meaningful content
    if 'package ' in text or 'deny ' in text or 'warn ' in text or 'allow ' in text:
        code = text.strip()
        # Remove any backticks
        code = code.replace('`', '')
        code = code.strip()
        if code and has_meaningful_content(code):
            return code
    
    return None


def validate_rego_syntax(code: str, package: str = "", imports: List[str] = None) -> Tuple[bool, str, str]:
    """Validate Rego code syntax using opa parse.
    
    Args:
        code: Rego code to validate
        package: Package name (optional, will be added if not in code)
        imports: List of imports (optional, will be added if not in code)
        
    Returns:
        (is_valid, formatted_code, error_message)
    """
    if imports is None:
        imports = []
    
    # Clean the code - aggressively remove ALL backticks (markdown artifacts)
    # Rego doesn't use backticks, so any backticks are definitely markdown artifacts
    # Remove all backticks from the code
    code = code.replace('`', '')
    code = code.strip()
    
    # Also clean each line individually to catch any edge cases
    lines = code.split('\n')
    cleaned_lines = []
    for line in lines:
        # Remove any backticks from the line
        cleaned_line = line.replace('`', '').strip()
        if cleaned_line:  # Only add non-empty lines
            cleaned_lines.append(cleaned_line)
    code = '\n'.join(cleaned_lines)
    
    # Build complete code with package and imports if needed
    complete_code = code
    added_package = False
    added_imports = []
    if package and f"package {package}" not in code:
        complete_code_parts = [f"package {package}\n"]
        added_package = True
        complete_code_parts.append("import rego.v1\n")
        added_imports.append("rego.v1")
        for imp in imports:
            if not imp.startswith("rego.v1") and f"import {imp}" not in code:
                complete_code_parts.append(f"import {imp}\n")
                added_imports.append(imp)
        complete_code_parts.append("\n")
        complete_code_parts.append(code)
        complete_code = "".join(complete_code_parts)
    
    # Final safety check: remove ALL backticks from complete_code
    # This is the last chance to catch any backticks before writing to file
    if '`' in complete_code:
        print(f"WARNING: Backticks found in complete_code before writing! Removing them.")
        complete_code = complete_code.replace('`', '')
    
    # Write to temp file (preserve for debugging)
    # Use a more descriptive name and don't auto-delete
    import time
    timestamp = int(time.time() * 1000)
    tmp_path = Path(tempfile.gettempdir()) / f"rego_validate_{timestamp}.rego"
    with open(tmp_path, 'w', encoding='utf-8') as tmp_file:
        tmp_file.write(complete_code)
        tmp_file.flush()
    
    # Debug: print what we're writing (first 200 chars)
    print(f"DEBUG: Writing to {tmp_path}")
    print(f"DEBUG: First 200 chars of code: {repr(complete_code[:200])}")
    print(f"DEBUG: Code contains backticks: {'`' in complete_code}")
    if '`' in complete_code:
        # Find where backticks are
        for i, char in enumerate(complete_code[:100]):
            if char == '`':
                print(f"DEBUG: Backtick found at position {i}: {repr(complete_code[max(0,i-10):i+10])}")
    
    error_msg = ""
    formatted_code = code
    
    # Determine which OPA command to use
    # Try 'ec opa' first (custom OPA with EC functions), then fall back to 'opa'
    opa_base = ["opa"]  # Default
    try:
        # Check if 'ec' command exists and has 'opa' subcommand
        result = subprocess.run(
            ["ec", "opa", "--version"],
            capture_output=True,
            timeout=1,
            text=True
        )
        if result.returncode == 0:
            opa_base = ["ec", "opa"]
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass  # Use default 'opa'
    
    try:
        # 1. opa parse (syntax check)
        # Note: Could use OPA's Go library directly for better performance:
        # github.com/open-policy-agent/opa/ast.ParseModule()
        try:
            
            result = subprocess.run(
                opa_base + ["parse", "--format", "json", str(tmp_path)],
                capture_output=True,
                timeout=3,  # Reduced timeout for faster feedback
                text=True
            )
            if result.returncode != 0:
                # Parse error message - OPA errors are usually in stderr
                error_output = result.stderr.strip() if result.stderr else result.stdout.strip()
                if not error_output:
                    error_output = "Syntax error (unknown)"
                # Clean up error message - remove file paths for clarity
                error_msg = re.sub(r'/tmp/[^\s]+', '<temp file>', error_output)
                return False, code, error_msg
        except FileNotFoundError:
            return False, code, "opa command not found. Install OPA to validate code."
        except subprocess.TimeoutExpired:
            return False, code, "opa parse timed out (code may be too complex)"
        except Exception as e:
            return False, code, f"opa parse error: {e}"
        
        # 2. opa fmt (format code for proper indentation)
        # Format the code using opa fmt for proper indentation
        try:
            fmt_result = subprocess.run(
                opa_base + ["fmt", str(tmp_path)],
                capture_output=True,
                timeout=3,
                text=True
            )
            if fmt_result.returncode == 0 and fmt_result.stdout:
                # opa fmt outputs the formatted code to stdout
                formatted_complete = fmt_result.stdout.strip()
                
                # If we added package/imports, extract just the formatted original code
                if added_package:
                    # Remove the package and imports we added
                    lines = formatted_complete.split('\n')
                    # Skip package line
                    if lines and lines[0].startswith('package'):
                        lines = lines[1:]
                    # Skip import lines we added
                    for imp in added_imports:
                        while lines and (lines[0].strip().startswith('import') or not lines[0].strip()):
                            if lines[0].strip().startswith(f'import {imp}') or lines[0].strip() == '':
                                lines = lines[1:]
                            else:
                                break
                    # Skip empty lines at start
                    while lines and not lines[0].strip():
                        lines = lines[1:]
                    formatted_code = '\n'.join(lines).strip()
                else:
                    formatted_code = formatted_complete
            # If fmt fails, keep the original formatted_code (from parse)
        except Exception:
            # If fmt is not available or fails, keep original formatted_code
            pass
        
        return True, formatted_code, ""
    
    finally:
        # PRESERVE temp file for debugging - don't delete
        # User can inspect the file to see what was written
        print(f"DEBUG: Temp file preserved at: {tmp_path}")
        print(f"DEBUG: To inspect: cat {tmp_path}")
        # Uncomment the following line to auto-delete after debugging:
        # try:
        #     tmp_path.unlink()
        # except Exception:
        #     pass


def test_rego_code(code: str, test_dir: Optional[Path] = None) -> Tuple[bool, str]:
    """Test Rego code using opa test.
    
    Args:
        code: Rego code to test
        test_dir: Directory containing test files (optional)
        
    Returns:
        (tests_passed, error_or_output_message)
    """
    if not test_dir or not test_dir.exists():
        return True, ""  # No tests to run
    
    # Write code to temp file in test directory
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', dir=test_dir, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(code)
        tmp_file.flush()
    
    try:
        result = subprocess.run(
            ["opa", "test", str(test_dir), "--format", "json"],
            capture_output=True,
            timeout=10,
            text=True
        )
        
        if result.returncode == 0:
            return True, "All tests passed"
        else:
            # Parse JSON output for test results
            try:
                test_data = json.loads(result.stdout) if result.stdout else {}
                if "errors" in test_data:
                    error_summary = "\n".join(str(e) for e in test_data["errors"][:3])  # First 3 errors
                    return False, f"Test failures:\n{error_summary}"
            except:
                pass
            
            return False, result.stderr.strip() if result.stderr else "Tests failed"
    
    except FileNotFoundError:
        return True, ""  # OPA not available, skip tests
    except subprocess.TimeoutExpired:
        return False, "Test execution timed out"
    except Exception as e:
        return False, f"Test error: {e}"
    
    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass

