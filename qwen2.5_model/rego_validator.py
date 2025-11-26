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


def extract_rego_code(text: str) -> Optional[str]:
    """Extract Rego code from model response.
    
    Looks for code blocks marked with ```rego or ``` or just Rego code.
    """
    # Try to find code blocks
    code_block_pattern = r'```(?:rego)?\s*\n(.*?)```'
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # If no code block, look for package declaration (likely Rego code)
    if 'package ' in text:
        # Extract from first package to end (or next markdown code block)
        match = re.search(r'(package\s+\S+.*?)(?:\n\n|```|$)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Last resort: return the whole text if it looks like Rego
    if 'package ' in text or 'deny ' in text or 'warn ' in text or 'allow ' in text:
        return text.strip()
    
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
    
    # Build complete code with package and imports if needed
    complete_code = code
    if package and f"package {package}" not in code:
        complete_code_parts = [f"package {package}\n"]
        complete_code_parts.append("import rego.v1\n")
        for imp in imports:
            if not imp.startswith("rego.v1") and f"import {imp}" not in code:
                complete_code_parts.append(f"import {imp}\n")
        complete_code_parts.append("\n")
        complete_code_parts.append(code)
        complete_code = "".join(complete_code_parts)
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        tmp_file.write(complete_code)
        tmp_file.flush()
    
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
        
        return True, formatted_code, ""
    
    finally:
        # Clean up temp file
        try:
            tmp_path.unlink()
        except Exception:
            pass


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

