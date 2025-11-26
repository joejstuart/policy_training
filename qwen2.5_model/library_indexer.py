"""Indexes all library functions for search."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass

from library_mapper import LibraryMapper
from context_extractor import extract_signature, scan_usage_sites_in_rules, extract_package


@dataclass
class HelperInfo:
    """Information about a helper function."""
    name: str
    package: str
    file_path: Path
    import_prefix: str
    signature: str
    doc: str
    usage_examples: List[str]
    keywords: Set[str]  # Keywords extracted from name and doc


class LibraryIndexer:
    """Indexes all library functions for search."""
    
    def __init__(self, repo_root: Path, mapper: LibraryMapper):
        self.repo_root = Path(repo_root)
        self.mapper = mapper
        self.index: Dict[str, HelperInfo] = {}  # function_name -> HelperInfo
        self.keyword_index: Dict[str, Set[str]] = {}  # keyword -> set of function names
    
    def index_all_libraries(self, scan_usage: bool = True):
        """Scan all library files and build searchable index.
        
        Args:
            scan_usage: If True, scan for usage examples (slower but more complete).
                       If False, skip usage scanning for faster indexing.
        """
        # Get all library directories
        for import_prefix in self.mapper.import_to_dir.keys():
            lib_files = self.mapper.get_all_library_files(import_prefix)
            
            for lib_file in lib_files:
                self._index_file(lib_file, import_prefix, scan_usage=scan_usage)
    
    def _index_file(self, file_path: Path, import_prefix: str, scan_usage: bool = True):
        """Index all public functions in a file.
        
        Args:
            file_path: Path to the .rego file
            import_prefix: Import prefix for this file
            scan_usage: If True, scan for usage examples (slower)
        """
        try:
            source = file_path.read_text(encoding='utf-8')
            package = extract_package(source) or import_prefix
            
            # Find all public functions (not starting with _)
            # Pattern 1: function_name := ... or function_name if ... (no args)
            # Pattern 2: function_name(args) := ... or function_name(args) if ... (with args)
            function_patterns = [
                r'^([a-zA-Z][a-zA-Z0-9_]*)\s+(?::=|if\s+\{)',  # function_name := or function_name if {
                r'^([a-zA-Z][a-zA-Z0-9_]*)\([^)]*\)\s*(?::=|if\s+\{)',  # function_name(args) := or function_name(args) if {
            ]
            
            matches = []
            for pattern in function_patterns:
                matches.extend(re.finditer(pattern, source, re.MULTILINE))
            
            # Remove duplicates (same function name, same position)
            seen = set()
            unique_matches = []
            for match in matches:
                key = (match.start(), match.group(1))
                if key not in seen:
                    seen.add(key)
                    unique_matches.append(match)
            
            for match in unique_matches:
                func_name = match.group(1)
                
                # Skip private functions (starting with _)
                if func_name.startswith('_'):
                    continue
                
                # Extract signature
                signature = extract_signature(source, func_name)
                if not signature:
                    continue
                
                # Extract docstring (first comment block)
                doc_lines = []
                start_pos = match.start()
                lines = source[:start_pos].split('\n')
                for line in reversed(lines):
                    stripped = line.strip()
                    if stripped.startswith('#'):
                        comment_text = re.sub(r'^#+\s*', '', stripped)
                        if comment_text:
                            doc_lines.insert(0, comment_text)
                    elif stripped == '':
                        continue
                    else:
                        break
                
                doc = ' '.join(doc_lines[:2]) if doc_lines else ''  # Keep it short
                
                # Scan for usage examples (prioritize tests, which show correct patterns)
                # Only scan if requested (can be slow for many functions)
                usage_examples = []
                if scan_usage:
                    usage_examples = scan_usage_sites_in_rules(
                        self.repo_root,
                        func_name,
                        import_prefix,
                        max_examples=2
                    )
                
                # Extract keywords
                keywords = self._extract_keywords(func_name, doc)
                
                # Create HelperInfo
                helper_info = HelperInfo(
                    name=func_name,
                    package=package,
                    file_path=file_path,
                    import_prefix=import_prefix,
                    signature=signature,
                    doc=doc,
                    usage_examples=usage_examples,
                    keywords=keywords
                )
                
                # Index by function name
                # Handle collisions: prefer more specific import prefix
                if func_name in self.index:
                    existing = self.index[func_name]
                    # Keep the one with more specific import (longer prefix)
                    if len(import_prefix) > len(existing.import_prefix):
                        self.index[func_name] = helper_info
                else:
                    self.index[func_name] = helper_info
                
                # Index by keywords
                for keyword in keywords:
                    if keyword not in self.keyword_index:
                        self.keyword_index[keyword] = set()
                    self.keyword_index[keyword].add(func_name)
        except Exception as e:
            # Skip files that can't be read or indexed
            print(f"Warning: Could not index {file_path}: {e}")
    
    def _extract_keywords(self, func_name: str, doc: str) -> Set[str]:
        """Extract keywords from function name and docstring."""
        keywords = set()
        
        # Add function name (lowercase)
        keywords.add(func_name.lower())
        
        # Add words from function name (split on _)
        for part in func_name.split('_'):
            if len(part) > 2:  # Skip very short parts
                keywords.add(part.lower())
        
        # Add words from docstring
        if doc:
            # Extract words (alphanumeric sequences)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', doc.lower())
            keywords.update(words)
        
        return keywords
    
    def find_relevant_helpers(self, instruction: str, package: str = None, max_results: int = 5) -> List[HelperInfo]:
        """Find helpers relevant to instruction using keyword matching.
        
        Args:
            instruction: User instruction text
            package: Optional package name to filter by
            max_results: Maximum number of helpers to return
            
        Returns:
            List of HelperInfo sorted by relevance
        """
        # Extract keywords from instruction
        instruction_lower = instruction.lower()
        instruction_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', instruction_lower))
        
        # Identify domain from instruction (task, image, sbom, attestation, etc.)
        domain_keywords = {
            'task': ['task', 'pipeline', 'pipelinerun'],
            'image': ['image', 'registry', 'container'],
            'sbom': ['sbom', 'bom', 'cyclonedx', 'spdx'],
            'attestation': ['attestation', 'provenance', 'slsa'],
        }
        
        detected_domains = []
        for domain, keywords in domain_keywords.items():
            if any(kw in instruction_lower for kw in keywords):
                detected_domains.append(domain)
        
        # Score each helper
        scores: Dict[str, float] = {}
        
        for func_name, helper_info in self.index.items():
            # Filter out clearly irrelevant helpers
            # Skip rule_data_* helpers unless instruction mentions "rule data" or "data"
            if func_name.startswith('rule_data_') and 'data' not in instruction_lower and 'rule' not in instruction_lower:
                continue
            
            # Skip very generic helpers unless explicitly mentioned
            # Check for whole word match, not substring
            generic_helpers = {'le', 'lt', 'ge', 'gt', 'eq', 'ne'}  # comparison operators
            if func_name in generic_helpers:
                # Only include if explicitly mentioned as a whole word
                if not re.search(rf'\b{re.escape(func_name)}\b', instruction_lower):
                    continue
            
            # Skip helpers from wrong domains
            if detected_domains:
                import_lower = helper_info.import_prefix.lower()
                # If we detected "task" domain, prioritize tekton/lib helpers
                if 'task' in detected_domains:
                    if 'sbom' in import_lower and 'sbom' not in detected_domains:
                        continue  # Skip SBOM helpers when looking for tasks
                # If we detected "sbom" domain, skip non-SBOM helpers
                if 'sbom' in detected_domains and 'sbom' not in import_lower:
                    if 'task' not in detected_domains:  # Unless also looking for tasks
                        continue
            
            # Filter by package if specified (but be lenient)
            if package:
                package_lower = package.lower()
                helper_package_lower = helper_info.package.lower()
                helper_import_lower = helper_info.import_prefix.lower()
                # Skip only if package is specified and doesn't match at all
                if (package_lower not in helper_package_lower and 
                    package_lower not in helper_import_lower and
                    helper_package_lower not in package_lower):
                    # But allow if instruction mentions the package domain
                    if package_lower not in instruction_lower:
                        continue
            
            score = 0.0
            
            # Exact function name match (very high score)
            func_name_lower = func_name.lower()
            if func_name_lower in instruction_lower:
                score += 20.0
            
            # Prioritize helpers from relevant modules
            import_lower = helper_info.import_prefix.lower()
            if 'task' in detected_domains:
                if 'tekton' in import_lower:
                    score += 5.0  # tekton helpers are highly relevant for tasks
                if 'lib' in import_lower and 'pipelinerun' in func_name_lower:
                    score += 5.0  # pipelinerun helpers are relevant
            if 'attestation' in detected_domains or 'provenance' in instruction_lower:
                if 'lib' in import_lower and 'attestation' in func_name_lower:
                    score += 5.0
            
            # Substring match in function name (e.g., "task" matches "tasks")
            for word in instruction_words:
                if word in func_name_lower:
                    # Exact word match gets higher score
                    if func_name_lower.startswith(word) or func_name_lower.endswith(word):
                        score += 4.0
                    else:
                        score += 2.0
                elif func_name_lower in word:
                    score += 1.0
            
            # Keyword matches (but lower weight to avoid noise)
            for keyword in helper_info.keywords:
                if keyword in instruction_words:
                    score += 1.5
                elif keyword in instruction_lower:
                    score += 0.5
            
            # Package/import prefix match
            if package:
                package_lower = package.lower()
                if (package_lower in helper_info.package.lower() or 
                    package_lower in helper_info.import_prefix.lower()):
                    score += 3.0
            
            # Only include helpers with meaningful scores
            if score >= 1.0:
                scores[func_name] = score
        
        # If no matches found, return top helpers anyway (fallback)
        if not scores:
            # Return helpers that might be generally useful
            all_helpers = list(self.index.values())
            # Filter by package if specified
            if package:
                package_lower = package.lower()
                filtered = [
                    h for h in all_helpers
                    if (package_lower in h.package.lower() or 
                        package_lower in h.import_prefix.lower() or
                        h.package.lower() in package_lower)
                ]
                if filtered:
                    return filtered[:max_results]
            # Otherwise return first N helpers
            return all_helpers[:max_results]
        
        # Sort by score (descending)
        sorted_helpers = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top N
        results = []
        for func_name, _ in sorted_helpers[:max_results]:
            results.append(self.index[func_name])
        
        return results
    
    def get_helper_context(self, helper_name: str) -> Optional[str]:
        """Get context for a helper: signature + doc + usage example.
        
        Args:
            helper_name: Name of the helper function
            
        Returns:
            Formatted string with signature, doc, and usage example, or None
        """
        if helper_name not in self.index:
            return None
        
        helper = self.index[helper_name]
        
        parts = []
        
        # Signature (already includes doc if available)
        parts.append(helper.signature)
        
        # Usage example if available
        if helper.usage_examples:
            parts.append("# Used like:")
            for example in helper.usage_examples[:1]:  # Only first example
                # Indent the example
                example_lines = example.split('\n')
                indented = '\n'.join(f"#   {line}" for line in example_lines)
                parts.append(indented)
        
        return '\n'.join(parts)
