"""Mine helper usage patterns from production rules.

Finds real examples of how helpers are used in policy/release/ rules,
providing grounded context for LLM enrichment.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class UsageExample:
    """A single usage example of a helper."""
    file_path: str
    rule_name: str
    context: str  # The surrounding code
    line_number: int
    related_helpers: List[str]  # Other helpers used in same rule
    attestation_type: Optional[str] = None


@dataclass 
class HelperUsageInfo:
    """Aggregated usage info for a helper."""
    helper_name: str
    usage_count: int
    examples: List[UsageExample]
    co_occurring_helpers: Dict[str, int]  # helper -> count
    common_patterns: List[str]


class UsageMiner:
    """Mine helper usage patterns from production rules."""
    
    # Patterns to detect attestation type from imports/code
    ATTESTATION_PATTERNS = {
        "slsa_provenance": [
            r"lib\.pipelinerun_attestations",
            r"lib\.taskrun_attestations", 
            r"predicate\.buildConfig",
            r"slsa",
        ],
        "spdx_sbom": [
            r"sbom\.spdx_sboms",
            r"spdx",
            r"packages\[",
        ],
        "cyclonedx_sbom": [
            r"sbom\.cyclonedx_sboms",
            r"cyclonedx",
            r"components\[",
        ],
    }
    
    def __init__(self, policy_dir: Path):
        """Initialize miner.
        
        Args:
            policy_dir: Path to policy directory (contains release/, lib/)
        """
        self.policy_dir = Path(policy_dir)
        self.release_dir = self.policy_dir / "release"
        self.lib_dir = self.policy_dir / "lib"
        
        # Cache for parsed rules
        self._rule_cache: Dict[str, List[dict]] = {}
        self._helper_usages: Dict[str, List[UsageExample]] = defaultdict(list)
        self._scanned = False
    
    def scan_all_rules(self):
        """Scan all production rules and build usage index."""
        if self._scanned:
            return
        
        if not self.release_dir.exists():
            print(f"Warning: {self.release_dir} does not exist")
            return
        
        # Find all non-test rego files
        rego_files = [f for f in self.release_dir.rglob("*.rego") 
                      if "_test.rego" not in f.name]
        
        for rego_file in rego_files:
            try:
                self._scan_file(rego_file)
            except Exception as e:
                print(f"Warning: Could not scan {rego_file}: {e}")
        
        self._scanned = True
        print(f"Scanned {len(rego_files)} rule files, found {len(self._helper_usages)} unique helpers used")
    
    def _scan_file(self, file_path: Path):
        """Scan a single file for helper usages."""
        source = file_path.read_text(encoding='utf-8')
        lines = source.split('\n')
        
        # Find all rules in the file
        rules = self._extract_rules(source, lines)
        
        rel_path = str(file_path.relative_to(self.policy_dir))
        
        for rule in rules:
            # Find helper calls in this rule
            helpers_in_rule = self._find_helper_calls(rule['body'])
            
            # Detect attestation type
            att_type = self._detect_attestation_type(rule['body'])
            
            # Record usage for each helper
            for helper_name, line_offset in helpers_in_rule:
                example = UsageExample(
                    file_path=rel_path,
                    rule_name=rule['name'],
                    context=rule['body'],
                    line_number=rule['start_line'] + line_offset,
                    related_helpers=[h for h, _ in helpers_in_rule if h != helper_name],
                    attestation_type=att_type,
                )
                self._helper_usages[helper_name].append(example)
    
    def _extract_rules(self, source: str, lines: List[str]) -> List[dict]:
        """Extract deny/warn/allow rules from source."""
        rules = []
        
        # Pattern for rule definitions
        rule_patterns = [
            r'^(deny|warn|allow|violation)\s+contains\s+',
            r'^(deny|warn|allow|violation)\s*\[',
            r'^(deny|warn|allow|violation)\s+if\s*\{',
        ]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check if this is a rule start
            rule_name = None
            for pattern in rule_patterns:
                match = re.match(pattern, stripped)
                if match:
                    rule_name = match.group(1)
                    break
            
            if rule_name:
                # Extract the full rule body
                start_line = i + 1  # 1-indexed
                body_lines = [line]
                brace_count = line.count('{') - line.count('}')
                
                j = i + 1
                while j < len(lines) and brace_count > 0:
                    body_lines.append(lines[j])
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    j += 1
                
                # Include closing brace line if we stopped on it
                if j < len(lines) and '}' in lines[j]:
                    body_lines.append(lines[j])
                
                rules.append({
                    'name': rule_name,
                    'start_line': start_line,
                    'body': '\n'.join(body_lines),
                })
                
                i = j
            else:
                i += 1
        
        return rules
    
    def _find_helper_calls(self, code: str) -> List[Tuple[str, int]]:
        """Find helper function calls in code.
        
        Returns list of (helper_name, line_offset) tuples.
        """
        helpers = []
        lines = code.split('\n')
        
        # Patterns for helper calls
        patterns = [
            # lib.helper_name or lib.module.helper_name
            r'\blib\.([a-z_][a-z0-9_.]*)',
            # tekton.helper_name
            r'\btekton\.([a-z_][a-z0-9_]*)',
            # sbom.helper_name
            r'\bsbom\.([a-z_][a-z0-9_]*)',
            # image.helper_name
            r'\bimage\.([a-z_][a-z0-9_]*)',
        ]
        
        for i, line in enumerate(lines):
            for pattern in patterns:
                for match in re.finditer(pattern, line):
                    full_match = match.group(0)
                    # Normalize to consistent naming
                    if full_match.startswith('lib.'):
                        helper_name = full_match[4:]  # Remove lib. prefix
                    else:
                        helper_name = full_match
                    helpers.append((helper_name, i))
        
        return helpers
    
    def _detect_attestation_type(self, code: str) -> Optional[str]:
        """Detect attestation type from code patterns."""
        code_lower = code.lower()
        
        for att_type, patterns in self.ATTESTATION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code_lower):
                    return att_type
        
        return None
    
    def get_helper_usages(self, helper_name: str, max_examples: int = 3) -> HelperUsageInfo:
        """Get usage info for a specific helper.
        
        Args:
            helper_name: Helper name (e.g., "tekton.tasks" or "pipelinerun_attestations")
            max_examples: Maximum examples to return
            
        Returns:
            HelperUsageInfo with examples and co-occurrence data
        """
        self.scan_all_rules()
        
        # Try different name variations
        name_variations = [
            helper_name,
            helper_name.split('.')[-1],  # Just the function name
            f"tekton.{helper_name}" if not '.' in helper_name else helper_name,
        ]
        
        examples = []
        for name in name_variations:
            if name in self._helper_usages:
                examples = self._helper_usages[name]
                break
        
        # Calculate co-occurrence
        co_occurring: Dict[str, int] = defaultdict(int)
        for ex in examples:
            for related in ex.related_helpers:
                co_occurring[related] += 1
        
        # Find common patterns in the usage
        patterns = self._extract_common_patterns(examples)
        
        return HelperUsageInfo(
            helper_name=helper_name,
            usage_count=len(examples),
            examples=examples[:max_examples],
            co_occurring_helpers=dict(co_occurring),
            common_patterns=patterns,
        )
    
    def _extract_common_patterns(self, examples: List[UsageExample]) -> List[str]:
        """Extract common usage patterns from examples."""
        patterns = []
        
        # Look for common iteration patterns
        iteration_patterns = defaultdict(int)
        for ex in examples:
            # Find "some X in Y" patterns
            for match in re.finditer(r'some\s+\w+\s+in\s+[\w.()]+', ex.context):
                iteration_patterns[match.group(0)] += 1
        
        # Return most common
        sorted_patterns = sorted(iteration_patterns.items(), key=lambda x: -x[1])
        patterns = [p for p, _ in sorted_patterns[:3]]
        
        return patterns
    
    def get_co_occurring_helpers(self, helper_name: str, min_count: int = 2) -> List[Tuple[str, int]]:
        """Get helpers that commonly appear with this one.
        
        Args:
            helper_name: Helper name
            min_count: Minimum co-occurrence count
            
        Returns:
            List of (helper_name, count) sorted by count
        """
        info = self.get_helper_usages(helper_name)
        
        filtered = [(h, c) for h, c in info.co_occurring_helpers.items() if c >= min_count]
        return sorted(filtered, key=lambda x: -x[1])
    
    def find_schema_usages(self, field_name: str) -> List[UsageExample]:
        """Find rules that access a schema field.
        
        Args:
            field_name: Field name to search for (e.g., "bundle", "ref", "status")
            
        Returns:
            List of usage examples
        """
        self.scan_all_rules()
        
        examples = []
        
        # Search all cached usages for patterns mentioning this field
        for helper_name, usages in self._helper_usages.items():
            for usage in usages:
                if field_name.lower() in usage.context.lower():
                    examples.append(usage)
        
        # Deduplicate by file+rule
        seen = set()
        unique = []
        for ex in examples:
            key = (ex.file_path, ex.rule_name)
            if key not in seen:
                seen.add(key)
                unique.append(ex)
        
        return unique[:5]

