"""Builds context dynamically from helper signatures and usage examples."""

import re
from typing import Optional, List
from pathlib import Path

from library_indexer import LibraryIndexer
from library_mapper import LibraryMapper


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 characters for English).
    
    This is a simple heuristic. For more accuracy, you'd use the actual tokenizer,
    but that's expensive. This is good enough for budgeting.
    """
    # Rough estimate: 1 token per 4 characters
    return len(text) // 4


class SmartContextBuilder:
    """Builds context dynamically with strict token budget."""
    
    def __init__(self, indexer: LibraryIndexer, mapper: LibraryMapper, max_tokens: int = 500):
        self.indexer = indexer
        self.mapper = mapper
        self.max_tokens = max_tokens
    
    def build_context(self, instruction: str, package: str = None) -> str:
        """Build context dynamically with strict token budget.
        
        Args:
            instruction: User instruction text
            package: Optional package name (e.g., "tasks")
            
        Returns:
            Context string with package, imports, and helper signatures + usage examples
        """
        # Detect package from instruction if not provided
        if not package:
            package = self._detect_package(instruction)
        
        # Start building context
        parts = []
        tokens_used = 0
        
        # 1. Package declaration
        package_decl = f"package {package}\n" if package else "package tasks\n"
        parts.append(package_decl)
        tokens_used += estimate_tokens(package_decl)
        
        # 2. Core imports (always include)
        core_imports = [
            "import rego.v1",
            "import data.lib",
        ]
        imports_section = '\n'.join(core_imports) + '\n'
        parts.append(imports_section)
        tokens_used += estimate_tokens(imports_section)
        
        # 3. Find relevant helpers
        helpers = self.indexer.find_relevant_helpers(instruction, package=package, max_results=5)
        
        if not helpers:
            # No helpers found, return minimal context
            parts.append("\n# No relevant helpers found. Use only standard Rego features.")
            return '\n'.join(parts)
        
        # 4. Group helpers by import prefix and add imports
        import_prefixes = set()
        for helper in helpers:
            import_prefixes.add(helper.import_prefix)
        
        # Add imports for libraries we'll use
        additional_imports = []
        for import_prefix in sorted(import_prefixes):
            if import_prefix not in ["data.lib", "data.release.lib"]:
                # Convert to import statement
                # e.g., "data.lib.tekton" -> "import data.lib.tekton"
                import_stmt = f"import {import_prefix}"
                additional_imports.append(import_stmt)
        
        if additional_imports:
            imports_text = '\n'.join(additional_imports) + '\n'
            parts.append(imports_text)
            tokens_used += estimate_tokens(imports_text)
        
        # 5. Add helper context (signatures + usage examples)
        parts.append("\n# Relevant helpers (signatures & usage):\n")
        tokens_used += estimate_tokens(parts[-1])
        
        # Add helpers one by one until we hit token budget
        remaining_budget = self.max_tokens - tokens_used
        helpers_added = 0
        
        for helper in helpers:
            helper_context = self.indexer.get_helper_context(helper.name)
            if not helper_context:
                continue
            
            # Format: # data.lib.tekton.tasks(obj) -> set[task]
            import_short = helper.import_prefix.split('.')[-1] if '.' in helper.import_prefix else helper.import_prefix
            # Show import prefix and function name
            helper_header = f"\n# {helper.import_prefix}.{helper.name}"
            
            # Add return type hint if we can infer it from signature
            signature_line = helper.signature.split('\n')[-1] if '\n' in helper.signature else helper.signature
            if ':=' in signature_line:
                if '[ ' in signature_line or '[' in signature_line:
                    return_type = "-> array"
                elif '{ ' in signature_line or '{' in signature_line:
                    return_type = "-> set"
                else:
                    return_type = ""
                if return_type:
                    helper_header += f" {return_type}"
            
            # Estimate tokens for this helper
            helper_tokens = estimate_tokens(helper_header + '\n' + helper_context)
            
            if tokens_used + helper_tokens > self.max_tokens:
                # Would exceed budget, stop here
                break
            
            parts.append(helper_header)
            parts.append(helper_context)
            tokens_used += helper_tokens
            helpers_added += 1
        
        # 6. Add instruction about using only provided helpers
        instruction_text = (
            "\n\n# Use only the helpers and imports provided in this context.\n"
            "# Do not invent new modules or helper functions; if a needed helper is missing,\n"
            "# add a TODO comment instead of creating a new import."
        )
        parts.append(instruction_text)
        
        return '\n'.join(parts)
    
    def _detect_package(self, instruction: str) -> str:
        """Try to detect package name from instruction.
        
        Args:
            instruction: User instruction text
            
        Returns:
            Detected package name or "tasks" as default
        """
        instruction_lower = instruction.lower()
        
        # Common package names
        package_keywords = {
            'task': 'tasks',
            'tasks': 'tasks',
            'image': 'images',
            'attestation': 'attestations',
            'sbom': 'sbom',
            'cve': 'cve',
            'provenance': 'provenance',
            'label': 'labels',
            'git': 'git',
        }
        
        for keyword, package in package_keywords.items():
            if keyword in instruction_lower:
                return package
        
        return 'tasks'  # Default
