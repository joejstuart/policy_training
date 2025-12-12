#!/usr/bin/env python3
"""Build Knowledge Base from policy/lib and attestations.

This script extracts helpers and schemas into the KB format.

Usage:
    # Build everything
    python scripts/build_kb.py
    
    # Build only helpers
    python scripts/build_kb.py --helpers-only
    
    # Build only schemas
    python scripts/build_kb.py --schemas-only
    
    # Specify output directory
    python scripts/build_kb.py --output data/knowledge_base/

Per architecture spec:
- Extract helpers with AST parser (not regex) - uses `opa parse`
- Two-tier chunks: card (compact) and full (with body)
- Grounding: source file, line numbers
- KB versioning via manifest
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_base import KnowledgeBase, HelperFull
from schema_extractor import SchemaExtractor
from kb_manifest import KBManifest
from rego_ast_parser import RegoASTParser, extract_function_body


def find_repo_root() -> Path:
    """Find repository root."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "policy").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent


def extract_helpers(repo_root: Path) -> List[HelperFull]:
    """Extract all helpers from policy/lib using AST parser.
    
    Per architecture spec: Extract helpers with AST parser (not regex).
    
    Args:
        repo_root: Repository root path
        
    Returns:
        List of HelperFull objects
    """
    lib_dir = repo_root / "policy" / "lib"
    if not lib_dir.exists():
        print(f"Warning: {lib_dir} does not exist")
        return []
    
    # Initialize AST parser
    try:
        parser = RegoASTParser()
    except RuntimeError as e:
        print(f"Warning: {e}. Falling back to regex-based extraction.")
        return extract_helpers_regex_fallback(repo_root)
    
    helpers = []
    
    # Find all .rego files (excluding tests)
    rego_files = [f for f in lib_dir.rglob("*.rego") if "_test.rego" not in f.name]
    
    for rego_file in rego_files:
        try:
            file_helpers = extract_helpers_from_file_ast(parser, rego_file, repo_root)
            helpers.extend(file_helpers)
        except Exception as e:
            print(f"Warning: Could not process {rego_file}: {e}")
    
    print(f"Extracted {len(helpers)} helpers from {len(rego_files)} files (using AST)")
    return helpers


def extract_helpers_from_file_ast(parser: RegoASTParser, file_path: Path, repo_root: Path) -> List[HelperFull]:
    """Extract helpers from a single file using AST parser.
    
    Args:
        parser: RegoASTParser instance
        file_path: Path to file
        repo_root: Repository root
        
    Returns:
        List of HelperFull objects
    """
    # Parse file with AST
    module = parser.parse_file(file_path)
    if not module:
        return []
    
    # Read source for body extraction
    source = file_path.read_text(encoding='utf-8')
    
    # Relative path for source_file
    try:
        rel_path = file_path.relative_to(repo_root)
    except ValueError:
        rel_path = file_path
    
    # Determine module path from package
    package = module.package
    module_path = f"data.{package}" if not package.startswith("data.") else package
    
    helpers = []
    
    for rule in module.rules:
        # Skip private functions
        if rule.is_private:
            continue
        
        # Skip default rules (they're values, not functions)
        if rule.is_default:
            continue
        
        try:
            # Extract function body using line numbers from AST
            body = extract_function_body(source, rule.start_line, rule.end_line)
            
            # Generate signature
            signature = rule.signature
            
            # Use doc comment from AST
            description = rule.doc_comment or ""
            
            # Generate use_when based on function name and description
            use_when = generate_use_when(rule.name, description)
            
            # Generate imports required
            imports_required = [module_path]
            
            # Build helper ID
            # Use module path parts after "lib"
            # e.g., "lib.tekton" -> "tekton", "lib" -> ""
            parts = package.split(".")
            if "lib" in parts:
                lib_idx = parts.index("lib")
                remaining = parts[lib_idx + 1:]
                if remaining:
                    helper_id = f"lib.{'.'.join(remaining)}.{rule.name}"
                else:
                    helper_id = f"lib.{rule.name}"
            else:
                helper_id = f"lib.{rule.name}"
            
            helper = HelperFull(
                id=helper_id,
                name=rule.name,
                module_path=module_path,
                signature=signature,
                description=description,
                use_when=use_when,
                source_file=str(rel_path),
                source_lines=(rule.start_line, rule.end_line),
                body=body,
                usage_examples=[],  # TODO: Mine from rules
                imports_required=imports_required,
            )
            
            helpers.append(helper)
            
        except Exception as e:
            print(f"  Warning: Could not extract {rule.name}: {e}")
    
    return helpers


def extract_helpers_regex_fallback(repo_root: Path) -> List[HelperFull]:
    """Fallback regex-based extraction if OPA is not available.
    
    Args:
        repo_root: Repository root path
        
    Returns:
        List of HelperFull objects
    """
    import re
    
    lib_dir = repo_root / "policy" / "lib"
    if not lib_dir.exists():
        return []
    
    helpers = []
    rego_files = [f for f in lib_dir.rglob("*.rego") if "_test.rego" not in f.name]
    
    for rego_file in rego_files:
        try:
            source = rego_file.read_text(encoding='utf-8')
            lines = source.split('\n')
            
            # Extract package
            package_match = re.search(r'^package\s+(\S+)', source, re.MULTILINE)
            package = package_match.group(1) if package_match else "unknown"
            module_path = f"data.{package}" if not package.startswith("data.") else package
            
            try:
                rel_path = rego_file.relative_to(repo_root)
            except ValueError:
                rel_path = rego_file
            
            # Find function definitions with regex
            patterns = [
                r'^([a-zA-Z][a-zA-Z0-9_]*)\s*:=',
                r'^([a-zA-Z][a-zA-Z0-9_]*)\s+if\s*\{',
                r'^([a-zA-Z][a-zA-Z0-9_]*)\([^)]*\)\s*:=',
                r'^([a-zA-Z][a-zA-Z0-9_]*)\([^)]*\)\s+if\s*\{',
            ]
            
            found_funcs = {}
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                for pattern in patterns:
                    match = re.match(pattern, stripped)
                    if match:
                        func_name = match.group(1)
                        if not func_name.startswith('_') and func_name not in found_funcs:
                            found_funcs[func_name] = i
                        break
            
            for func_name, line_num in found_funcs.items():
                # Build helper ID
                parts = package.split(".")
                if "lib" in parts:
                    lib_idx = parts.index("lib")
                    remaining = parts[lib_idx + 1:]
                    helper_id = f"lib.{'.'.join(remaining)}.{func_name}" if remaining else f"lib.{func_name}"
                else:
                    helper_id = f"lib.{func_name}"
                
                helpers.append(HelperFull(
                    id=helper_id,
                    name=func_name,
                    module_path=module_path,
                    signature=func_name,
                    description="",
                    use_when=[],
                    source_file=str(rel_path),
                    source_lines=(line_num, line_num),
                    body="",
                    usage_examples=[],
                    imports_required=[module_path],
                ))
        except Exception as e:
            print(f"Warning: Regex fallback failed for {rego_file}: {e}")
    
    print(f"Extracted {len(helpers)} helpers (using regex fallback)")
    return helpers


def generate_use_when(func_name: str, description: str) -> List[str]:
    """Generate use_when hints from function name and description."""
    hints = []
    combined = f"{func_name} {description}".lower()
    
    if "task" in combined:
        hints.append("working with tasks")
    if "attestation" in combined or "provenance" in combined:
        hints.append("processing attestations")
    if "sbom" in combined or "bom" in combined:
        hints.append("SBOM processing")
    if "result" in combined:
        hints.append("formatting results")
    if "bundle" in combined:
        hints.append("bundle validation")
    if "ref" in combined:
        hints.append("reference checks")
    if "error" in combined or "deny" in combined or "warn" in combined:
        hints.append("policy violations")
    if "rule_data" in combined or "data" in func_name:
        hints.append("configurable policy data")
    
    return hints if hints else ["general helper"]


def extract_schemas(repo_root: Path) -> Dict:
    """Extract schemas from attestations.
    
    Args:
        repo_root: Repository root path
        
    Returns:
        Dictionary of schema_id -> SchemaField
    """
    att_dir = repo_root / "data" / "attestations"
    
    if not att_dir.exists():
        print(f"Warning: {att_dir} does not exist")
        return {}
    
    extractor = SchemaExtractor(att_dir)
    schemas = extractor.extract_all()
    
    print(f"Extracted {len(schemas)} schemas from attestations")
    return schemas


def build_kb(
    repo_root: Path,
    output_dir: Path,
    helpers_only: bool = False,
    schemas_only: bool = False
) -> KnowledgeBase:
    """Build complete knowledge base.
    
    Args:
        repo_root: Repository root path
        output_dir: Output directory for KB
        helpers_only: Only extract helpers
        schemas_only: Only extract schemas
        
    Returns:
        Built KnowledgeBase
    """
    kb = KnowledgeBase()
    
    # Extract helpers
    if not schemas_only:
        print("Extracting helpers from policy/lib...")
        helpers = extract_helpers(repo_root)
        for helper in helpers:
            kb.add_helper(helper)
    
    # Extract schemas
    if not helpers_only:
        print("Extracting schemas from attestations...")
        schemas = extract_schemas(repo_root)
        for schema in schemas.values():
            kb.add_schema(schema)
    
    # Create manifest
    kb.manifest = KBManifest.create(
        repo_root,
        helper_count=len(kb.helper_cards),
        schema_count=len(kb.schemas)
    )
    kb.manifest.attestation_types = list(set(s.attestation_type for s in kb.schemas.values()))
    
    # Save KB
    print(f"Saving KB to {output_dir}...")
    kb.save(output_dir)
    kb.manifest.save(output_dir / "manifest.yaml")
    
    print("\n" + kb.summary())
    print(f"\nKB saved to: {output_dir}")
    
    return kb


def main():
    parser = argparse.ArgumentParser(
        description="Build Knowledge Base from policy/lib and attestations"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: data/knowledge_base/)"
    )
    parser.add_argument(
        "--helpers-only",
        action="store_true",
        help="Only extract helpers, skip schemas"
    )
    parser.add_argument(
        "--schemas-only",
        action="store_true",
        help="Only extract schemas, skip helpers"
    )
    
    args = parser.parse_args()
    
    repo_root = find_repo_root()
    output_dir = args.output or (repo_root / "data" / "knowledge_base")
    
    print(f"Repository root: {repo_root}")
    print(f"Output directory: {output_dir}")
    print()
    
    build_kb(
        repo_root=repo_root,
        output_dir=output_dir,
        helpers_only=args.helpers_only,
        schemas_only=args.schemas_only
    )


if __name__ == "__main__":
    main()

