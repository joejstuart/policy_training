#!/usr/bin/env python3
"""
Extract components from log files and run download-attestation for each component.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

def extract_json_from_log(log_file):
    """Extract JSON object from log file that contains a 'components' array."""
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # First, try to find JSON that might be in a shell variable assignment
    # Pattern: MAPPING='{...}' (can span multiple lines)
    shell_var_pattern = r"MAPPING='(\{.*?\})'"
    match = re.search(shell_var_pattern, content, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Find the JSON block starting with { and containing "components"
    match = re.search(r'\{[^{}]*"components"\s*:\s*\[', content)
    if match:
        start = match.start()
        # Now find the matching closing brace
        brace_count = 0
        json_str = ''
        for char in content[start:]:
            json_str += char
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try to fix common issues
                        json_str = re.sub(r',\s*}', '}', json_str)
                        json_str = re.sub(r',\s*]', ']', json_str)
                        try:
                            return json.loads(json_str)
                        except:
                            pass
                    break
    
    return None

def extract_components_from_log(log_file):
    """Extract containerImage from components in a log file."""
    data = extract_json_from_log(log_file)
    if not data or 'components' not in data:
        print(f"Warning: Could not find 'components' array in {log_file}", file=sys.stderr)
        return []
    
    components = []
    for comp in data['components']:
        # Extract containerImage field
        if 'containerImage' in comp:
            components.append(comp['containerImage'])
        else:
            print(f"Warning: Component missing 'containerImage': {comp}", file=sys.stderr)
    
    return components

def main():
    log_files = [
        '/Users/jstuart/Downloads/managed-7xrt2-apply-mapping.log',
        '/Users/jstuart/Downloads/managed-b7nhx-verify-conforma.log',
        '/Users/jstuart/Downloads/managed-dsq5h-verify-conforma.log',
        '/Users/jstuart/Downloads/managed-jv6xz-verify-conforma.log',
        '/Users/jstuart/Downloads/managed-mdftn-verify-conforma.log',
        '/Users/jstuart/Downloads/managed-r4l6m-verify-enterprise-contract.log',
        '/Users/jstuart/Downloads/managed-rsmxv-verify-conforma.log',
        '/Users/jstuart/Downloads/managed-tvtbd-verify-conforma.log',
        '/Users/jstuart/Downloads/managed-vrqch-verify-conforma.log',
    ]
    
    all_components = []
    
    for log_file in log_files:
        log_path = Path(log_file)
        if not log_path.exists():
            print(f"Warning: Log file not found: {log_file}", file=sys.stderr)
            continue
        
        print(f"Processing {log_path.name}...", file=sys.stderr)
        components = extract_components_from_log(log_file)
        all_components.extend(components)
        print(f"  Found {len(components)} components", file=sys.stderr)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_components = []
    for comp in all_components:
        if comp not in seen:
            seen.add(comp)
            unique_components.append(comp)
    
    print(f"\nTotal unique components: {len(unique_components)}", file=sys.stderr)
    
    # Run download-attestation for each component
    # download-attestation is a shell function, so we need to source zshrc first
    home = os.path.expanduser('~')
    
    for component in unique_components:
        # Sanitize component for filename (replace /, @, : with safe characters)
        safe_name = component.replace('/', '_').replace('@', '_').replace(':', '_')
        output_file = f"{safe_name}.json"
        print(f"Running: download-attestation {component} > {output_file}", file=sys.stderr)
        try:
            # Source zshrc and run the function
            cmd = f'source {home}/.zshrc 2>/dev/null || true; download-attestation {component}'
            result = subprocess.run(
                ['zsh', '-c', cmd],
                capture_output=True,
                text=True,
                check=True
            )
            with open(output_file, 'w') as f:
                f.write(result.stdout)
            print(f"  ✓ Saved to {output_file}", file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            if e.stderr:
                print(f"    stderr: {e.stderr[:500]}", file=sys.stderr)
            if e.stdout:
                print(f"    stdout: {e.stdout[:500]}", file=sys.stderr)

if __name__ == '__main__':
    main()

