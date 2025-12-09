"""Extract function signatures and usage examples from Rego files."""

import re
from pathlib import Path
from typing import Optional, List, Tuple


def extract_signature(source: str, function_name: str) -> Optional[str]:
    """Extract function signature (one-line) and docstring.
    
    Args:
        source: Full source code
        function_name: Name of function to extract
        
    Returns:
        One-line signature with docstring, or None if not found
    """
    # Find the function definition line
    # Pattern: function_name := ... or function_name if ...
    lines = source.split('\n')
    func_line_idx = None
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match: function_name := or function_name if
        # Also match: function_name(args) := or function_name(args) if
        patterns = [
            rf'^{re.escape(function_name)}\s*(?::=|if\s+\{{)',  # tasks := or tasks if {
            rf'^{re.escape(function_name)}\([^)]*\)\s*(?::=|if\s+\{{)',  # tasks(obj) := or tasks(obj) if {
        ]
        for pattern in patterns:
            if re.match(pattern, stripped):
                func_line_idx = i
                break
        if func_line_idx is not None:
            break
    
    if func_line_idx is None:
        return None
    
    # Get the comment block above the function (docstring)
    comment_lines = []
    for i in range(func_line_idx - 1, -1, -1):
        stripped = lines[i].strip()
        if stripped.startswith('#'):
            comment_text = re.sub(r'^#+\s*', '', stripped)
            if comment_text:  # Skip empty comment lines
                comment_lines.insert(0, comment_text)
        elif stripped == '':
            continue  # Skip blank lines
        else:
            break  # Hit non-comment, non-blank line
    
    # Get the function signature (first line only, truncated if needed)
    func_line = lines[func_line_idx].strip()
    
    # For simple assignments (no braces/brackets), show full line if reasonable
    if ':=' in func_line and '{' not in func_line and '[' not in func_line:
        # Simple assignment - show full line if under 100 chars
        if len(func_line) <= 100:
            pass  # Keep as is
        else:
            # Truncate long simple assignments
            func_line = func_line[:97] + '...'
    # For "if {" patterns, just show: function_name if {
    elif 'if {' in func_line:
        func_line = f"{function_name} if {{"
    # For comprehensions or complex assignments, show simplified version
    elif ':=' in func_line:
        if '{' in func_line:
            # Set or object comprehension
            func_line = func_line.split('{')[0].strip() + ' { ... }'
        elif '[' in func_line:
            # Array comprehension - ensure proper spacing
            prefix = func_line.split('[')[0].strip()
            func_line = prefix + ' [ ... ]'
        else:
            # Shouldn't happen, but keep as is
            pass
    
    # Build result: docstring + signature
    result_parts = []
    if comment_lines:
        # Join comment lines, but keep it short (max 2 lines)
        doc = ' '.join(comment_lines[:2])
        if len(comment_lines) > 2:
            doc += '...'
        result_parts.append(f"# {doc}")
    
    result_parts.append(func_line)
    
    return '\n'.join(result_parts)


def extract_usage_sites(source: str, function_name: str, max_examples: int = 2) -> List[str]:
    """Extract usage examples of a function from source code.
    
    Looks for patterns like:
    - tekton.tasks(...)
    - tasks(...)
    - some task in tasks(...)
    - name := task_name(...)
    
    Args:
        source: Full source code
        function_name: Name of function to find usages of
        max_examples: Maximum number of examples to return
        
    Returns:
        List of usage example strings (context lines around the usage)
    """
    examples = []
    
    # Pattern to match function calls
    # Matches: identifier.function_name(...) or function_name(...)
    patterns = [
        rf'\w+\.{re.escape(function_name)}\s*\(',
        rf'\b{re.escape(function_name)}\s*\(',
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, source):
            # Get context around the match (3 lines before, 4 lines after for better context)
            # Tests often have more context that helps understand the pattern
            start_line = source[:match.start()].count('\n')
            end_line = source[:match.end()].count('\n')
            
            lines = source.split('\n')
            context_start = max(0, start_line - 3)
            context_end = min(len(lines), end_line + 4)
            
            # Find the actual call (might span multiple lines)
            call_lines = []
            in_call = False
            paren_count = 0
            
            for i in range(context_start, context_end):
                line = lines[i]
                if not in_call and function_name in line and '(' in line:
                    in_call = True
                
                if in_call:
                    call_lines.append(line)
                    paren_count += line.count('(') - line.count(')')
                    if paren_count <= 0 and ')' in line:
                        break
            
            if call_lines:
                # Clean up: remove leading/trailing whitespace, keep meaningful context
                usage = '\n'.join(call_lines).strip()
                # Remove excessive indentation
                if usage:
                    # Find minimum indentation
                    non_empty_lines = [l for l in usage.split('\n') if l.strip()]
                    if non_empty_lines:
                        min_indent = min(len(l) - len(l.lstrip()) for l in non_empty_lines)
                        usage = '\n'.join(l[min_indent:] if len(l) > min_indent else l for l in usage.split('\n'))
                    
                    examples.append(usage)
                    
                    if len(examples) >= max_examples:
                        break
        
        if len(examples) >= max_examples:
            break
    
    return examples[:max_examples]


def extract_package(source: str) -> Optional[str]:
    """Extract package name from source code.
    
    Args:
        source: Full source code
        
    Returns:
        Package name (e.g., "lib.tekton") or None
    """
    match = re.search(r'^package\s+(\S+)', source, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def scan_usage_sites_in_rules(repo_root: Path, function_name: str, import_prefix: str, max_examples: int = 2) -> List[str]:
    """Scan policy/** for usage sites of a function, prioritizing test files.
    
    Tests are scanned first because they show correct usage patterns.
    Then regular rule files are scanned.
    
    Args:
        repo_root: Repository root directory
        function_name: Name of function to find
        import_prefix: Import prefix (e.g., "data.lib.tekton")
        max_examples: Maximum number of examples to return
        
    Returns:
        List of usage example strings (prioritized from tests, then rules)
    """
    examples = []
    test_examples = []
    rule_examples = []
    
    # Determine the short name (e.g., "tekton" from "data.lib.tekton")
    parts = import_prefix.split('.')
    if len(parts) >= 3:
        short_prefix = parts[-1]  # e.g., "tekton" from "data.lib.tekton"
    else:
        short_prefix = None
    
    # Scan directories: policy/release/** and policy/lib/**
    scan_dirs = [
        repo_root / "policy" / "release",
        repo_root / "policy" / "lib",
    ]
    
    for scan_dir in scan_dirs:
        if not scan_dir.exists():
            continue
        
        # First pass: collect test examples (prioritized)
        # Limit to first 50 test files to avoid scanning everything
        test_files = list(scan_dir.rglob("*_test.rego"))[:50]
        for rego_file in test_files:
            if len(test_examples) >= max_examples:
                break
            try:
                source = rego_file.read_text(encoding='utf-8')
                
                # Look for usage with import prefix (e.g., tekton.tasks(...))
                if short_prefix:
                    full_name = f"{short_prefix}.{function_name}"
                    usages = extract_usage_sites(source, full_name, max_examples=1)
                    if usages:
                        test_examples.extend(usages)
                        if len(test_examples) >= max_examples:
                            break
                
                # Also look for direct usage (e.g., tasks(...))
                if len(test_examples) < max_examples:
                    usages = extract_usage_sites(source, function_name, max_examples=1)
                    if usages:
                        test_examples.extend(usages)
            except Exception:
                continue
        
        # Second pass: collect rule examples (if we need more)
        # Limit to first 30 rule files to avoid scanning everything
        if len(test_examples) < max_examples:
            rule_files = [f for f in scan_dir.rglob("*.rego") if not f.name.endswith("_test.rego")][:30]
            for rego_file in rule_files:
                if len(test_examples) + len(rule_examples) >= max_examples:
                    break
                try:
                    source = rego_file.read_text(encoding='utf-8')
                    
                    # Look for usage with import prefix (e.g., tekton.tasks(...))
                    if short_prefix:
                        full_name = f"{short_prefix}.{function_name}"
                        usages = extract_usage_sites(source, full_name, max_examples=1)
                        if usages:
                            rule_examples.extend(usages)
                            if len(test_examples) + len(rule_examples) >= max_examples:
                                break
                    
                    # Also look for direct usage (e.g., tasks(...))
                    if len(test_examples) + len(rule_examples) < max_examples:
                        usages = extract_usage_sites(source, function_name, max_examples=1)
                        if usages:
                            rule_examples.extend(usages)
                except Exception:
                    continue
    
    # Combine: tests first, then rules
    examples = test_examples + rule_examples
    return examples[:max_examples]
