#!/usr/bin/env python3
"""
Helper script for inference-time grounding when using the fine-tuned model.

This script helps construct the proper context (package + imports + helper snippets)
when asking the model to generate new rules.
"""

import re
from pathlib import Path
from typing import List, Dict

POLICY_LIB_DIR = Path("policy/lib")
RELEASE_LIB_DIR = Path("policy/release/lib")


def extract_helper_functions(lib_file: Path, max_functions: int = 5) -> List[str]:
    """Extract function names and signatures from a lib file."""
    if not lib_file.exists():
        return []
    
    content = lib_file.read_text()
    functions = []
    
    # Find function-like rules
    pattern = r'^(\w+)\s*(?:\([^)]*\))?\s*(?:contains\s+\w+\s+)?(?:if\s+)?\{'
    matches = re.finditer(pattern, content, re.MULTILINE)
    
    for match in matches:
        func_name = match.group(1)
        if not func_name.startswith("_") and func_name not in ["package", "import"]:
            # Try to extract signature
            line = content[match.start():content.find('\n', match.start())]
            functions.append(line.strip())
            if len(functions) >= max_functions:
                break
    
    return functions


def get_helper_snippets(imports: List[str], max_per_file: int = 3) -> str:
    """Get code snippets for imported helpers."""
    snippets = []
    
    for imp in imports:
        # Parse import: "data.lib.image" or "data.lib.json as j"
        if imp.startswith("data.lib."):
            parts = imp.replace("data.lib.", "").split(" as ")
            module = parts[0]
            
            # Map to file path
            if "/" in module:
                lib_file = POLICY_LIB_DIR / f"{module}.rego"
            else:
                lib_file = POLICY_LIB_DIR / f"{module}/{module}.rego"
            
            if lib_file.exists():
                funcs = extract_helper_functions(lib_file, max_per_file)
                if funcs:
                    snippets.append(f"\n# Helpers from {imp}:")
                    for func in funcs:
                        snippets.append(f"# {func}")
        
        elif imp.startswith("data.") and "release" in imp:
            # Release lib import
            parts = imp.replace("data.", "").split(" as ")
            module = parts[0].replace(".", "/")
            lib_file = RELEASE_LIB_DIR / f"{module.split('/')[-1]}.rego"
            
            if lib_file.exists():
                funcs = extract_helper_functions(lib_file, max_per_file)
                if funcs:
                    snippets.append(f"\n# Helpers from {imp}:")
                    for func in funcs:
                        snippets.append(f"# {func}")
    
    return "\n".join(snippets) if snippets else ""


def build_inference_context(package: str, imports: List[str], instruction: str) -> str:
    """Build the full context for inference, including package, imports, and helper snippets."""
    context_parts = []
    
    # Package and imports
    context_parts.append(f"package {package}\n")
    context_parts.append("import rego.v1\n")
    for imp in imports:
        if not imp.startswith("rego.v1"):
            context_parts.append(f"import {imp}\n")
    context_parts.append("\n")
    
    # Helper snippets
    helper_snippets = get_helper_snippets(imports)
    if helper_snippets:
        context_parts.append("# Available helper functions:\n")
        context_parts.append(helper_snippets)
        context_parts.append("\n")
    
    # Instruction
    context_parts.append(f"# Task: {instruction}\n")
    
    return "".join(context_parts)


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: uv run python inference_helper.py <package> <import1> [import2 ...]")
        print("\nExample:")
        print("  uv run python inference_helper.py base_image_registries data.lib data.lib.image data.lib.sbom")
        sys.exit(1)
    
    package = sys.argv[1]
    imports = sys.argv[2:]
    
    instruction = "Generate a rule that validates base images come from allowed registries"
    
    context = build_inference_context(package, imports, instruction)
    print(context)


if __name__ == "__main__":
    main()

