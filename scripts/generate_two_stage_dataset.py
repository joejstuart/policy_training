#!/usr/bin/env python3
"""
Generate training dataset for two-stage Rego policy model.

This script generates:
- Stage 1 examples: REQUIREMENTS → CONTEXT (schema, helpers, rule_data_keys)
- Stage 2 examples: REQUIREMENTS + CONTEXT → ANALYSIS + RULE + TESTS

It reuses proven components from the existing codebase:
- library_indexer.py: Indexes all helpers with signatures and usage examples
- library_mapper.py: Maps import prefixes to directories
- generate_dataset.py: Rule parsing and validation patterns

Uses Ollama LLM for generating high-quality ANALYSIS sections.
"""

import json
import re
import sys
import random
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from paths import POLICY_RELEASE_DIR, POLICY_LIB_DIR, REPO_ROOT
    from library_mapper import LibraryMapper
    from library_indexer import LibraryIndexer
except ImportError:
    # Fallback paths
    REPO_ROOT = Path(__file__).parent.parent
    POLICY_RELEASE_DIR = REPO_ROOT / "policy" / "release"
    POLICY_LIB_DIR = REPO_ROOT / "policy" / "lib"
    
    # Try importing from src directly
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from library_mapper import LibraryMapper
    from library_indexer import LibraryIndexer

# Output directory
OUTPUT_DIR = REPO_ROOT / "data" / "training" / "two_stage"

# LLM Configuration
OLLAMA_MODEL = "qwen3-coder:30b"
OLLAMA_URL = "http://localhost:11434/api/generate"
USE_LLM_ANALYSIS = True  # Set to False to use rule-based analysis

# System prompt for Stage 1 (constant - tells model what to output)
STAGE1_SYSTEM_PROMPT = "Analyze the requirements and identify the attestation schema, available helpers, and rule data keys needed to implement this Rego rule."

# Stage 2 instruction (fixed format - receives structured input from Stage 1)
STAGE2_INSTRUCTION = "Write a Rego rule that enforces the requirements below using the provided context."


def generate_natural_language_instruction(
    metadata: 'RuleMetadata', 
    rule_type: str,
    llm: 'OllamaClient' = None
) -> str:
    """Generate varied natural language instruction from rule metadata.
    
    Uses LLM if available for natural, varied phrasings.
    Falls back to template-based generation.
    """
    title = metadata.title or ""
    description = metadata.description or ""
    
    # Try LLM-based generation first
    if llm and llm.is_available() and description:
        # Randomly select a style for variety
        styles = [
            "a direct statement like 'Verify that...' or 'Ensure that...'",
            "a question like 'How do I check if...' or 'How can I verify...'",
            "a task request like 'Create a rule to...' or 'Write a policy that...'",
            "a need statement like 'I need to enforce...' or 'I want to validate...'",
            "an informal request like 'Make sure...' or 'Check that...'",
        ]
        style = random.choice(styles)
        
        prompt = f"""/no_think
Rephrase this policy requirement as a natural user request. Use {style}.

Title: {title}
Description: {description}
Rule type: {rule_type}

Rules:
- Output ONLY the rephrased request (one sentence)
- Keep it concise (under 100 characters if possible)
- Sound natural, like a developer asking for help
- Do not include quotes around the output

Output:"""
        
        result = llm.generate(prompt, max_tokens=80)
        if result:
            # Clean up the response
            result = result.strip().strip('"\'')
            # Remove any "Output:" prefix if the model included it
            if result.lower().startswith('output:'):
                result = result[7:].strip()
            # Validate it's reasonable
            if 15 < len(result) < 200 and not result.startswith('#'):
                return result
    
    # Fallback to template-based generation
    purpose = title if title else "policy rule"
    
    templates = [
        f"{description}" if description else f"Implement a {rule_type} rule for: {purpose}",
        f"Create a Rego {rule_type} rule to {description.lower().rstrip('.')}" if description else f"Create a {rule_type} rule for {purpose}",
        f"I need to {description.lower().rstrip('.')}" if description else f"I need a rule that handles {purpose}",
    ]
    
    valid_templates = [t for t in templates if len(t) > 20]
    
    if valid_templates:
        return random.choice(valid_templates)
    
    return description if description else f"Implement a {rule_type} rule for {purpose}"


class OllamaClient:
    """Simple Ollama API client for generating ANALYSIS sections."""
    
    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url
        self._available = None
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if self._available is not None:
            return self._available
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=2) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False
        return self._available
    
    def generate(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Generate text using Ollama."""
        if not self.is_available():
            return None
        
        try:
            data = json.dumps({
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,
                }
            }).encode('utf-8')
            
            req = urllib.request.Request(
                self.url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status == 200:
                    result = json.loads(resp.read().decode('utf-8'))
                    return result.get("response", "").strip()
        except Exception as e:
            print(f"    LLM error: {e}")
        return None


@dataclass
class RuleMetadata:
    """Extracted rule metadata."""
    title: str = ""
    description: str = ""
    short_name: str = ""
    failure_msg: str = ""
    solution: str = ""
    collections: List[str] = field(default_factory=list)
    effective_on: str = ""


@dataclass
class ExtractedRule:
    """A complete extracted rule with all associated code."""
    package: str
    imports: List[str]
    private_helpers: List[str]  # Helper functions starting with _
    rule_code: str  # The main deny/warn/allow rule
    rule_type: str  # deny, warn, allow
    metadata: RuleMetadata
    test_code: Optional[str] = None
    source_file: Path = None
    
    def get_complete_code(self) -> str:
        """Assemble complete, compilable Rego code."""
        parts = []
        parts.append(f"package {self.package}")
        parts.append("")
        
        for imp in self.imports:
            parts.append(f"import {imp}")
        parts.append("")
        
        # Add private helpers
        for helper in self.private_helpers:
            parts.append(helper)
            parts.append("")
        
        # Add main rule (includes METADATA)
        parts.append(self.rule_code)
        
        return "\n".join(parts)


@dataclass
class Stage1Example:
    """Stage 1 training example: natural language instruction → context.
    
    The instruction is what the user types (varied natural language).
    The input contains the system prompt.
    The output is the inferred context.
    """
    natural_instruction: str  # What user writes (varied)
    requirements: str  # Structured requirements for reference
    attestation_schema: str
    available_helpers: str
    rule_data_keys: str
    
    def format_instruction(self) -> str:
        """The natural language instruction (user-facing, varied)."""
        return self.natural_instruction
    
    def format_input(self) -> str:
        """System prompt telling the model what to do."""
        return STAGE1_SYSTEM_PROMPT
    
    def format_output(self) -> str:
        parts = []
        parts.append(f"ATTESTATION_SCHEMA:\n{self.attestation_schema}")
        parts.append(f"\nAVAILABLE_HELPERS:\n{self.available_helpers}")
        if self.rule_data_keys:
            parts.append(f"\nRULE_DATA_KEYS:\n{self.rule_data_keys}")
        return "\n".join(parts)


@dataclass
class Stage2Example:
    """Stage 2 training example: requirements + context → analysis + rule + tests."""
    requirements: str
    context: str  # Stage 1 output
    analysis: str
    rule_code: str  # Complete code with helpers
    test_code: Optional[str] = None
    
    def format_input(self) -> str:
        return f"REQUIREMENTS:\n{self.requirements}\n\n{self.context}"
    
    def format_output(self) -> str:
        parts = []
        parts.append(f"ANALYSIS:\n{self.analysis}")
        parts.append(f"\nRULE:\n```rego\n{self.rule_code}\n```")
        if self.test_code:
            parts.append(f"\nTESTS:\n```rego\n{self.test_code}\n```")
        return "\n".join(parts)


class TwoStageDataGenerator:
    """Generates Stage 1 and Stage 2 training examples from existing policies."""
    
    def __init__(self, repo_root: Path, use_llm: bool = USE_LLM_ANALYSIS):
        self.repo_root = Path(repo_root)
        self.use_llm = use_llm
        
        # Initialize LLM client for ANALYSIS generation
        self.llm = OllamaClient() if use_llm else None
        if use_llm:
            if self.llm.is_available():
                print(f"LLM enabled: {OLLAMA_MODEL}")
            else:
                print("Warning: Ollama not available, falling back to rule-based ANALYSIS")
                self.use_llm = False
        
        # Initialize library indexer (reuse existing component)
        print("Initializing library indexer...")
        self.mapper = LibraryMapper(self.repo_root)
        self.mapper.build_mappings()
        self.indexer = LibraryIndexer(self.repo_root, self.mapper)
        self.indexer.index_all_libraries(scan_usage=False)  # Faster without usage scanning
        print(f"  Indexed {len(self.indexer.index)} helper functions")
    
    def extract_rule(self, rego_file: Path) -> List[ExtractedRule]:
        """Extract all rules from a Rego file with their private helpers."""
        try:
            content = rego_file.read_text()
        except Exception as e:
            print(f"  Error reading {rego_file}: {e}")
            return []
        
        # Extract package
        package_match = re.search(r'^package\s+(\S+)', content, re.MULTILINE)
        if not package_match:
            return []
        package = package_match.group(1)
        
        # Extract imports
        imports = []
        for match in re.finditer(r'^import\s+(\S+(?:\s+as\s+\S+)?)', content, re.MULTILINE):
            imports.append(match.group(1))
        
        # Find all private helpers (functions starting with _)
        all_private_helpers = self._extract_all_private_helpers(content)
        
        # Find all rules with METADATA
        rules = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if re.search(r'#\s*METADATA', line):
                # Extract metadata block
                metadata = self._extract_metadata(lines, i)
                if not metadata.title:
                    continue
                
                # Find the rule that follows
                rule_info = self._find_rule_after_metadata(lines, i, content)
                if not rule_info:
                    continue
                
                rule_type, rule_code, rule_with_metadata = rule_info
                
                # Find which private helpers this rule uses
                used_helpers = self._find_used_helpers(rule_code, all_private_helpers)
                
                # Check for test file
                test_file = rego_file.parent / rego_file.name.replace(".rego", "_test.rego")
                test_code = None
                if test_file.exists():
                    test_code = self._extract_relevant_tests(test_file, rule_type, metadata.short_name)
                
                rules.append(ExtractedRule(
                    package=package,
                    imports=imports,
                    private_helpers=used_helpers,
                    rule_code=rule_with_metadata,
                    rule_type=rule_type,
                    metadata=metadata,
                    test_code=test_code,
                    source_file=rego_file,
                ))
        
        return rules
    
    def _extract_all_private_helpers(self, content: str) -> Dict[str, str]:
        """Extract all private helper functions (starting with _) from content."""
        helpers = {}
        lines = content.split('\n')
        
        # Pattern for helper function definitions
        # Matches: _name := ... or _name(...) := ... or _name if { ... or _name(...) if {
        helper_pattern = re.compile(r'^(_\w+)(?:\([^)]*\))?\s*(?::=|if\s+\{|contains)')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            match = helper_pattern.match(line.strip())
            if match:
                helper_name = match.group(1)
                # Extract the complete helper (handle multi-line)
                helper_code = self._extract_complete_block(lines, i)
                if helper_code:
                    helpers[helper_name] = helper_code
                    # Skip past this helper
                    i += helper_code.count('\n') + 1
                    continue
            i += 1
        
        return helpers
    
    def _extract_complete_block(self, lines: List[str], start_idx: int) -> str:
        """Extract a complete code block (handling braces/brackets/parens)."""
        result_lines = []
        brace_count = 0
        bracket_count = 0
        paren_count = 0
        started = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            result_lines.append(line)
            
            # Count braces, brackets, and parentheses
            for char in line:
                if char == '{':
                    brace_count += 1
                    started = True
                elif char == '}':
                    brace_count -= 1
                elif char == '[':
                    bracket_count += 1
                    started = True
                elif char == ']':
                    bracket_count -= 1
                elif char == '(':
                    paren_count += 1
                    started = True
                elif char == ')':
                    paren_count -= 1
            
            # Check for simple assignment (no braces/brackets/parens)
            if ':=' in line and brace_count == 0 and bracket_count == 0 and paren_count == 0:
                if not '{' in line and not '[' in line and not '(' in line:
                    # Simple one-line assignment
                    break
            
            # Block is complete when all delimiters balance
            if started and brace_count == 0 and bracket_count <= 0 and paren_count <= 0:
                break
        
        return '\n'.join(result_lines)
    
    def _extract_metadata(self, lines: List[str], start_idx: int) -> RuleMetadata:
        """Extract METADATA block starting at start_idx."""
        metadata = RuleMetadata()
        current_key = None
        current_value = []
        has_custom_section = False
        
        for i in range(start_idx + 1, min(start_idx + 50, len(lines))):
            line = lines[i]
            stripped = line.strip()
            
            # End of metadata block
            if not stripped.startswith('#'):
                break
            
            # Remove comment marker
            text = stripped[1:].strip() if stripped.startswith('#') else stripped
            
            # Skip empty comment lines
            if not text:
                continue
            
            # Track if we have a custom section (indicates rule-level metadata)
            if text.strip() == 'custom:':
                has_custom_section = True
            
            # Check for key: value pattern
            if ':' in text and not text.startswith('-'):
                # Save previous key
                if current_key:
                    self._set_metadata_field(metadata, current_key, ' '.join(current_value))
                
                parts = text.split(':', 1)
                current_key = parts[0].strip().lower()
                value = parts[1].strip()
                
                # Handle multi-line indicator
                if value.startswith('>-'):
                    value = value[2:].strip()
                
                current_value = [value] if value else []
            elif current_key:
                # Continuation of previous value
                current_value.append(text)
        
        # Save last key
        if current_key:
            self._set_metadata_field(metadata, current_key, ' '.join(current_value))
        
        # Only return metadata if it has a custom section (rule-level, not package-level)
        # Package-level METADATA blocks don't have custom: sections
        if not has_custom_section:
            return RuleMetadata()  # Return empty metadata to skip this block
        
        return metadata
    
    def _set_metadata_field(self, metadata: RuleMetadata, key: str, value: str):
        """Set a metadata field by key name."""
        key = key.lower().replace(' ', '_')
        if key == 'title':
            metadata.title = value
        elif key == 'description':
            metadata.description = value
        elif key == 'short_name':
            metadata.short_name = value
        elif key == 'failure_msg':
            metadata.failure_msg = value
        elif key == 'solution':
            metadata.solution = value
        elif key == 'effective_on':
            metadata.effective_on = value
    
    def _find_rule_after_metadata(self, lines: List[str], metadata_start: int, content: str) -> Optional[Tuple[str, str, str]]:
        """Find the deny/warn/allow rule after a METADATA block."""
        # Look for rule in next 50 lines
        for i in range(metadata_start + 1, min(metadata_start + 50, len(lines))):
            line = lines[i].strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Match rule patterns
            rule_match = re.match(r'^(deny|warn|allow|violation|error)(\s+contains\s+\w+)?(\s+if\s+)?\{?', line)
            if rule_match:
                rule_type = rule_match.group(1)
                
                # Find the complete METADATA + rule block
                # Go back to find METADATA start
                metadata_line = metadata_start
                while metadata_line > 0 and not lines[metadata_line].strip().startswith('# METADATA'):
                    metadata_line -= 1
                
                # Extract from METADATA to end of rule
                rule_block = self._extract_complete_block(lines, i)
                
                # Get METADATA comments
                metadata_comments = []
                for j in range(metadata_line, i):
                    if lines[j].strip():
                        metadata_comments.append(lines[j])
                
                # Combine METADATA + rule
                full_rule = '\n'.join(metadata_comments) + '\n' + rule_block
                
                return rule_type, rule_block, full_rule
        
        return None
    
    def _find_used_helpers(self, rule_code: str, all_helpers: Dict[str, str]) -> List[str]:
        """Find which private helpers are used by this rule (recursively)."""
        used_names = set()
        used_code = []
        
        def find_helpers_in_code(code: str):
            """Recursively find helper references."""
            # Find all _helper_name references in code
            helper_refs = set(re.findall(r'\b(_\w+)\s*\(', code))
            # Also check for _helper_name without parentheses (set/object references)
            helper_refs.update(re.findall(r'\b(_\w+)\b', code))
            
            for helper_name in helper_refs:
                if helper_name in all_helpers and helper_name not in used_names:
                    used_names.add(helper_name)
                    helper_code = all_helpers[helper_name]
                    # Recursively find helpers this helper depends on
                    find_helpers_in_code(helper_code)
        
        # Start with the main rule
        find_helpers_in_code(rule_code)
        
        # Build ordered list (dependencies first would be ideal, but simple order for now)
        for helper_name in sorted(used_names):
            if helper_name in all_helpers:
                used_code.append(all_helpers[helper_name])
        
        return used_code
    
    def _extract_relevant_tests(self, test_file: Path, rule_type: str, short_name: str) -> Optional[str]:
        """Extract test functions relevant to this rule."""
        try:
            content = test_file.read_text()
        except Exception:
            return None
        
        # Find tests that reference this rule type
        # Pattern: test_* functions that use the rule
        test_pattern = re.compile(rf'^test_\w+.*if\s*\{{', re.MULTILINE)
        
        tests = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if re.match(r'^test_\w+', line.strip()):
                test_code = self._extract_complete_block(lines, i)
                # Check if it references our rule type
                if f".{rule_type}" in test_code or rule_type in test_code:
                    tests.append(test_code)
                    if len(tests) >= 3:  # Limit to 3 tests per rule
                        break
        
        if not tests:
            return None
        
        # Also get test fixtures (variables used by tests)
        fixtures = self._extract_test_fixtures(content, tests)
        
        # Build complete test code
        result_parts = []
        
        # Package and imports
        pkg_match = re.search(r'^package\s+(\S+)', content, re.MULTILINE)
        if pkg_match:
            result_parts.append(f"package {pkg_match.group(1)}")
        result_parts.append("")
        result_parts.append("import rego.v1")
        result_parts.append("")
        
        # Add tests
        for test in tests:
            result_parts.append(test)
            result_parts.append("")
        
        # Add fixtures
        for fixture in fixtures:
            result_parts.append(fixture)
            result_parts.append("")
        
        return '\n'.join(result_parts)
    
    def _extract_test_fixtures(self, content: str, tests: List[str]) -> List[str]:
        """Extract fixture variables used by the tests.
        
        Only extracts module-level fixtures (not indented, starts with _).
        Variables like 'expected' inside test functions are NOT fixtures.
        """
        fixtures = []
        
        # Find variable references in tests that start with _ (fixture naming convention)
        all_refs = set()
        for test in tests:
            # Look for _fixture_name references (underscore prefix = fixture)
            refs = re.findall(r'\b(_[a-z][a-z0-9_]*)\b', test)
            all_refs.update(refs)
        
        # Find fixture definitions at module level only
        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Only match NON-INDENTED lines that start with underscore (fixture convention)
            # Pattern: _fixture_name := { or _fixture_name := [ or _fixture_name := "
            # Must NOT be indented (module-level definition)
            if line and not line[0].isspace():
                match = re.match(r'^(_[a-z][a-z0-9_]*)\s*:=', line)
                if match:
                    fixture_name = match.group(1)
                    if fixture_name in all_refs:
                        fixture_code = self._extract_complete_block(lines, i)
                        fixtures.append(fixture_code)
        
        return fixtures[:5]  # Limit fixtures
    
    def generate_stage1_example(self, rule: ExtractedRule) -> Stage1Example:
        """Generate a Stage 1 training example from an extracted rule."""
        
        # Generate natural language instruction (what user would type)
        # Uses LLM for varied, natural phrasings
        natural_instruction = generate_natural_language_instruction(
            rule.metadata, 
            rule.rule_type,
            llm=self.llm if self.use_llm else None
        )
        
        # Generate REQUIREMENTS (structured, for reference)
        requirements = self._generate_requirements(rule)
        
        # Generate ATTESTATION_SCHEMA (infer from rule code)
        schema = self._infer_attestation_schema(rule)
        
        # Generate AVAILABLE_HELPERS (from library indexer)
        helpers = self._generate_available_helpers(rule)
        
        # Generate RULE_DATA_KEYS
        rule_data = self._extract_rule_data_keys(rule)
        
        return Stage1Example(
            natural_instruction=natural_instruction,
            requirements=requirements,
            attestation_schema=schema,
            available_helpers=helpers,
            rule_data_keys=rule_data,
        )
    
    def generate_stage2_example(self, rule: ExtractedRule, stage1: Stage1Example) -> Stage2Example:
        """Generate a Stage 2 training example."""
        
        # Generate ANALYSIS
        analysis = self._generate_analysis(rule)
        
        # Get complete rule code
        complete_code = rule.get_complete_code()
        
        return Stage2Example(
            requirements=stage1.requirements,
            context=stage1.format_output(),
            analysis=analysis,
            rule_code=complete_code,
            test_code=rule.test_code,
        )
    
    def _clean_error_message(self, msg: str) -> Optional[str]:
        """Clean up error message by replacing format placeholders with readable text."""
        if not msg:
            return None
        
        # Strip outer quotes if present
        msg = msg.strip().strip("'\"")
        
        # Skip messages that are ONLY format strings
        if re.match(r'^[%sqvd\s\[\]]+$', msg):
            return None
        
        # Replace format placeholders with readable descriptions
        # %s, %q, %v -> <value>
        cleaned = re.sub(r'%[sqv]', '<value>', msg)
        # %d -> <number>
        cleaned = re.sub(r'%d', '<number>', cleaned)
        # Clean up multiple consecutive placeholders
        cleaned = re.sub(r'(<value>\s*,?\s*)+', '<value> ', cleaned)
        # Ensure space after <value> when followed by word
        cleaned = re.sub(r'<value>(\w)', r'<value> \1', cleaned)
        # Remove trailing format-only parts like ": %v" at end
        cleaned = re.sub(r':\s*<value>\s*$', '', cleaned)
        # Clean up double spaces
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # If cleaning removed all content, return None
        final = cleaned.strip()
        if not final or final == '<value>':
            return None
        
        return final
    
    def _generate_requirements(self, rule: ExtractedRule) -> str:
        """Generate REQUIREMENTS section from rule metadata."""
        lines = []
        lines.append(f"- Package: {rule.package}")
        lines.append(f"- Rule type: {rule.rule_type}")
        
        if rule.metadata.short_name:
            lines.append(f"- Short name: {rule.metadata.short_name}")
        
        if rule.metadata.title:
            lines.append(f"- Purpose: {rule.metadata.title}")
        
        if rule.metadata.description:
            lines.append(f"- {rule.metadata.description}")
        
        # Only include error message if it's informative (not just format strings)
        if rule.metadata.failure_msg:
            msg = self._clean_error_message(rule.metadata.failure_msg)
            if msg:
                lines.append(f"- Error message: {msg}")
        
        return '\n'.join(lines)
    
    # Static schema mappings for rules with complex/indirect attestation access
    _SCHEMA_OVERRIDES = {
        # CVE rules access task results via _grouped_vulns/_clair_report helper chain
        ("cve", "cve_blockers"): ".statement.predicate.buildConfig.tasks[].results[] (CVE scan results)",
        ("cve", "unpatched_cve_blockers"): ".statement.predicate.buildConfig.tasks[].results[] (CVE scan results)",
        ("cve", "cve_warnings"): ".statement.predicate.buildConfig.tasks[].results[] (CVE scan results)",
        ("cve", "unpatched_cve_warnings"): ".statement.predicate.buildConfig.tasks[].results[] (CVE scan results)",
        # Tasks data validation rules check rule data but apply to PipelineRun context
        ("tasks", "data_provided"): ".statement.predicate (PipelineRun attestation)",
        ("tasks", "required_tasks_list_provided"): "(None - validates rule data format only)",
        ("tasks", "pipeline_required_tasks_list_provided"): "(None - validates rule data format only)",
        # Trusted task data validation
        ("trusted_task", "data"): ".statement.predicate.buildConfig.tasks[] (task references)",
        ("trusted_task", "data_format"): "(None - validates rule data format only)",
        # rpm_ostree validation
        ("rpm_ostree_task", "builder_image_param"): ".statement.predicate.buildConfig.tasks[].params[]",
        ("rpm_ostree_task", "rule_data"): "(None - validates rule data format only)",
        # SLSA source reference
        ("slsa_source_correlated", "source_code_reference_provided"): "input.source (expected source code reference)",
        # Attestation format checks
        ("attestation_type", "deprecated_policy_attestation_format"): "input (attestation format check)",
        ("rhtap_multi_ci", "attestation_found"): ".statement.predicate (RHTAP Multi-CI SLSA v1.0)",
        # SBOM rule data validation (uses lib.sbom.rule_data_errors)
        ("sbom", "disallowed_packages_provided"): "(None - validates rule data format only)",
        # OLM related images - uses lib.results_named to get RELATED_IMAGES task result
        ("olm", "allowed_registries_related"): ".statement.predicate.buildConfig.tasks[].results[] (RELATED_IMAGES result)",
        # Base image registries - validates rule data format only
        ("base_image_registries", "allowed_registries_provided"): "(None - validates rule data format only)",
        # RPM repos - accesses SBOM for RPM purl repo_id checking
        ("rpm_repos", "ids_known"): "CycloneDX/SPDX SBOM structure",
    }
    
    # Schema enrichments - detailed paths for each generic schema description
    _SCHEMA_ENRICHMENTS = {
        # PipelineRun attestation
        ".statement.predicate (PipelineRun attestation)": [
            "predicate.buildType (e.g., 'https://tekton.dev/chains/v2/slsa')",
            "predicate.buildConfig.tasks[]",
            "predicate.materials[]",
            "predicate.builder.id",
        ],
        # Task-level paths
        ".statement.predicate.buildConfig.tasks[]": [
            "predicate.buildConfig.tasks[].name",
            "predicate.buildConfig.tasks[].ref",
            "predicate.buildConfig.tasks[].results[]",
            "predicate.buildConfig.tasks[].results[].name",
            "predicate.buildConfig.tasks[].results[].value",
        ],
        ".statement.predicate.buildConfig.tasks[].results[]": [
            "predicate.buildConfig.tasks[].results[].name",
            "predicate.buildConfig.tasks[].results[].value",
            "predicate.buildConfig.tasks[].results[].type",
        ],
        ".statement.predicate.buildConfig.tasks[].params[]": [
            "predicate.buildConfig.tasks[].params[].name",
            "predicate.buildConfig.tasks[].params[].value",
        ],
        ".statement.predicate.buildConfig.tasks[].ref": [
            "predicate.buildConfig.tasks[].ref.name",
            "predicate.buildConfig.tasks[].ref.kind",
            "predicate.buildConfig.tasks[].ref.bundle",
        ],
        # CVE scan results - from clair-scan task
        ".statement.predicate.buildConfig.tasks[].results[] (CVE scan results)": [
            "Task 'clair-scan' produces REPORTS result",
            "REPORTS value: {image_digest: report_digest} mapping",
            "Report blob (OCI): mediaType 'application/vnd.redhat.clair-report+json'",
            "vulnerabilities[].name (CVE ID, e.g., 'CVE-2024-1234')",
            "vulnerabilities[].severity (critical, high, medium, low, unknown)",
            "vulnerabilities[].fixed_in_version (if patched)",
        ],
        # OLM related images result
        ".statement.predicate.buildConfig.tasks[].results[] (RELATED_IMAGES result)": [
            "Task result RELATED_IMAGES_DIGEST contains digest of referring manifest",
            "Referring manifest layers contain related images JSON blob",
            "Related images blob: mediaType 'application/vnd.konflux.related-images+json'",
            "related_images[].image (full image reference)",
            "related_images[].name (component name)",
        ],
        # Materials
        ".statement.predicate.materials[]": [
            "predicate.materials[].uri (git URL)",
            "predicate.materials[].digest.sha1 (commit SHA)",
        ],
        # Builder
        ".statement.predicate.builder": [
            "predicate.builder.id",
        ],
        # SPDX SBOM
        "SPDX SBOM structure": [
            "spdxVersion",
            "name (image reference)",
            "documentNamespace",
            "packages[]",
            "packages[].name",
            "packages[].versionInfo",
            "packages[].externalRefs[].referenceLocator (purl)",
            "files[].fileName",
        ],
        # CycloneDX SBOM
        "CycloneDX SBOM structure": [
            "bomFormat",
            "specVersion",
            "components[]",
            "components[].name",
            "components[].version",
            "components[].purl",
            "components[].externalReferences[]",
        ],
        # Combined SBOM
        "CycloneDX/SPDX SBOM structure": [
            "(CycloneDX) components[].purl",
            "(CycloneDX) components[].externalReferences[]",
            "(SPDX) packages[].externalRefs[].referenceLocator",
            "(SPDX) files[].fileName",
        ],
        # Image paths
        "input.image.ref (image reference)": [
            "input.image.ref (full image reference with digest)",
        ],
        "input.image.config (image configuration)": [
            "input.image.config.config.Labels",
            "input.image.config.rootfs",
        ],
        "input.image.signatures[] (Sigstore/Fulcio certificates)": [
            "input.image.signatures[].certificate.Extensions[]",
            "input.image.signatures[].certificate.Subject",
        ],
    }
    
    def _enrich_schema(self, schema_line: str) -> str:
        """Enrich a schema description with detailed paths."""
        desc = schema_line.strip().lstrip('-').strip()
        
        if desc in self._SCHEMA_ENRICHMENTS:
            enriched = [f"- {desc}"]
            for path in self._SCHEMA_ENRICHMENTS[desc][:5]:
                enriched.append(f"  - {path}")
            return '\n'.join(enriched)
        
        return f"- {desc}"
    
    def _infer_attestation_schema(self, rule: ExtractedRule) -> str:
        """Infer attestation schema paths from rule code."""
        paths = []
        code = rule.get_complete_code()
        
        # Check static overrides first
        key = (rule.package, rule.metadata.short_name)
        if key in self._SCHEMA_OVERRIDES:
            return self._enrich_schema(self._SCHEMA_OVERRIDES[key])
        
        # Check if this is a rule data validation rule (no attestation access)
        # These rules only validate rule_data format, don't access attestations
        is_rule_data_only = (
            (re.search(r'j\.validate_schema', code) or re.search(r'lib\.sbom\.rule_data_errors', code)) and
            not re.search(r'lib\.(pipelinerun|slsa_provenance)_attestations', code) and
            not re.search(r'input\.image\.(ref|config|signatures)', code) and
            not re.search(r'sbom\.(cyclonedx|spdx)_sboms', code)
        )
        if is_rule_data_only:
            return "- (None - validates rule data format only)"
        
        # Common patterns to detect - ordered by specificity
        patterns = [
            # Attestation entry points
            (r'lib\.pipelinerun_attestations', '.statement.predicate (PipelineRun attestation)'),
            (r'lib\.slsa_provenance_attestations', '.statement.predicate (SLSA Provenance)'),
            
            # Task-level accessors (more specific patterns first)
            (r'tekton\.task_ref\(', '.statement.predicate.buildConfig.tasks[].ref'),
            (r'tekton\.task_result\(', '.statement.predicate.buildConfig.tasks[].results[]'),
            (r'tekton\.task_param\(', '.statement.predicate.buildConfig.tasks[].params[]'),
            (r'tekton\.task_params\(', '.statement.predicate.buildConfig.tasks[].params[]'),
            (r'tekton\.task_results\(', '.statement.predicate.buildConfig.tasks[].results[]'),
            (r'tekton\.task_annotations\(', '.statement.predicate.buildConfig.tasks[].metadata.annotations'),
            (r'tekton\.task_labels\(', '.statement.predicate.buildConfig.tasks[].metadata.labels'),
            
            # Task collection accessors
            (r'tekton\.tasks\(', '.statement.predicate.buildConfig.tasks[]'),
            (r'tekton\.build_tasks', '.statement.predicate.buildConfig.tasks[]'),
            (r'tekton\.pre_build_tasks', '.statement.predicate.buildConfig.tasks[]'),
            (r'tekton\.source_build_tasks', '.statement.predicate.buildConfig.tasks[]'),
            (r'tekton\.git_clone_tasks', '.statement.predicate.buildConfig.tasks[]'),
            (r'lib\.tasks_from_pipelinerun', '.statement.predicate.buildConfig.tasks[]'),
            
            # Result accessors (map to results array)
            (r'lib\.results_named\(', '.statement.predicate.buildConfig.tasks[].results[]'),
            (r'lib\.results_from_tests', '.statement.predicate.buildConfig.tasks[].results[]'),
            
            # SBOM accessors
            (r'sbom\.cyclonedx_sboms', 'CycloneDX SBOM structure'),
            (r'sbom\.spdx_sboms', 'SPDX SBOM structure'),
            (r'sbom\.all_sboms', 'CycloneDX/SPDX SBOM structure'),
            
            # Material/builder accessors
            (r'predicate\.materials', '.statement.predicate.materials[]'),
            (r'predicate\.builder', '.statement.predicate.builder'),
            
            # Image accessors
            (r'input\.image\.signatures', 'input.image.signatures[] (Sigstore/Fulcio certificates)'),
            (r'input\.image\.ref', 'input.image.ref (image reference)'),
            (r'input\.image\.config', 'input.image.config (image configuration)'),
            (r'image\.files\(', 'input.image.files (OCI image filesystem)'),
            (r'image\.config\(', 'input.image.config (image configuration)'),
            (r'image\.signatures\(', 'input.image.signatures[] (Sigstore/Fulcio certificates)'),
        ]
        
        seen_descs = set()
        for pattern, path_desc in patterns:
            if re.search(pattern, code) and path_desc not in seen_descs:
                # Enrich each path with detailed sub-paths
                paths.append(self._enrich_schema(path_desc))
                seen_descs.add(path_desc)
        
        # Fallback: if we detected nothing but there's rule data access only
        if not paths:
            if re.search(r'lib\.rule_data\(', code):
                return "- (None - accesses rule data configuration only)"
            paths.append("- (Unable to infer schema - review manually)")
        
        return '\n'.join(paths)
    
    def _generate_available_helpers(self, rule: ExtractedRule) -> str:
        """Generate AVAILABLE_HELPERS section from library functions used."""
        code = rule.get_complete_code()
        helpers_used = []
        
        # Find library function calls
        # Pattern: module.function_name( or module.function_name without parens (for values)
        lib_calls = re.findall(r'\b(lib|tekton|sbom|image|j)\.(\w+)', code)
        
        seen = set()
        for module, func in lib_calls:
            full_name = f"{module}.{func}"
            if full_name in seen:
                continue
            seen.add(full_name)
            
            # Try to get info from indexer
            # The indexer stores by function name without module prefix
            desc = None
            if func in self.indexer.index:
                helper_info = self.indexer.index[func]
                desc = helper_info.doc
            
            # Also try with common prefixes
            if not desc:
                for try_name in [func, f"{func}_", f"_{func}"]:
                    if try_name in self.indexer.index:
                        helper_info = self.indexer.index[try_name]
                        desc = helper_info.doc
                        break
            
            if desc:
                # Truncate long descriptions
                desc = desc[:150].rstrip() + ("..." if len(desc) > 150 else "")
                helpers_used.append(f"- name: {full_name}\n  description: {desc}")
            else:
                helpers_used.append(f"- name: {full_name}")
        
        # Also add rego.metadata.chain if used
        if 'rego.metadata.chain()' in code:
            helpers_used.append("- name: rego.metadata.chain()\n  description: Returns metadata chain for current rule")
        
        if not helpers_used:
            helpers_used.append("- (No library helpers detected)")
        
        return '\n'.join(helpers_used)
    
    def _extract_rule_data_keys(self, rule: ExtractedRule) -> str:
        """Extract rule_data keys from rule code."""
        code = rule.get_complete_code()
        
        # Find lib.rule_data("key") calls
        keys = re.findall(r'lib\.rule_data\(["\']([^"\']+)["\']\)', code)
        
        if not keys:
            return ""
        
        lines = []
        for key in set(keys):
            lines.append(f"- {key}")
        
        return '\n'.join(lines)
    
    def _generate_analysis(self, rule: ExtractedRule) -> str:
        """Generate ANALYSIS section explaining field-to-logic mapping.
        
        Uses LLM if available for high-quality explanations, falls back to rule-based.
        """
        code = rule.get_complete_code()
        
        # Try LLM-based generation first
        if self.use_llm and self.llm:
            llm_analysis = self._generate_analysis_llm(rule)
            if llm_analysis:
                return llm_analysis
        
        # Fallback to rule-based analysis
        return self._generate_analysis_rulebased(rule)
    
    def _generate_analysis_llm(self, rule: ExtractedRule) -> Optional[str]:
        """Generate ANALYSIS using LLM."""
        code = rule.get_complete_code()
        
        prompt = f"""/no_think
You are analyzing a Rego policy rule. Write a brief ANALYSIS section (3-5 bullet points) explaining:
1. What data source the rule accesses (attestations, SBOMs, image config, etc.)
2. What the rule iterates over or checks
3. The core validation logic (what condition triggers a violation)
4. How the result is generated

Be specific and concise. Use bullet points starting with "- ".

RULE CODE:
```rego
{code[:2000]}
```

METADATA:
- Title: {rule.metadata.title}
- Purpose: {rule.metadata.description[:200] if rule.metadata.description else 'N/A'}
- Rule type: {rule.rule_type}

Write ONLY the bullet points, no introduction or conclusion:"""

        result = self.llm.generate(prompt, max_tokens=300)
        if result:
            # Clean up the response - extract just the bullet points
            lines = []
            for line in result.split('\n'):
                line = line.strip()
                if line.startswith('- ') or line.startswith('* '):
                    # Normalize to "- " format
                    lines.append('- ' + line[2:])
                elif line.startswith('-') or line.startswith('*'):
                    lines.append('- ' + line[1:].strip())
            if lines:
                return '\n'.join(lines[:6])  # Limit to 6 bullet points
        return None
    
    def _generate_analysis_rulebased(self, rule: ExtractedRule) -> str:
        """Generate ANALYSIS using rule-based extraction (fallback)."""
        code = rule.get_complete_code()
        analysis_parts = []
        
        # Identify the main data source
        if 'lib.pipelinerun_attestations' in code:
            analysis_parts.append("- Data source: PipelineRun attestations")
        elif 'lib.slsa_provenance_attestations' in code:
            analysis_parts.append("- Data source: SLSA Provenance attestations")
        elif 'sbom.cyclonedx_sboms' in code:
            analysis_parts.append("- Data source: CycloneDX SBOMs")
        elif 'sbom.spdx_sboms' in code:
            analysis_parts.append("- Data source: SPDX SBOMs")
        elif 'sbom.all_sboms' in code:
            analysis_parts.append("- Data source: All SBOMs (CycloneDX + SPDX)")
        elif 'input.image' in code:
            analysis_parts.append("- Data source: Image metadata")
        elif 'lib.rule_data' in code and 'attestations' not in code.lower():
            analysis_parts.append("- Data source: Rule configuration data")
        
        # Identify iteration patterns with context
        if 'some' in code:
            iterations = re.findall(r'some\s+(\w+)\s+in\s+([^\n{]+)', code)
            for var, collection in iterations[:3]:
                col = collection.strip()
                if 'tasks' in col.lower():
                    analysis_parts.append("- Iterates over pipeline tasks")
                elif 'sbom' in col.lower() or 'packages' in col.lower() or 'components' in col.lower():
                    analysis_parts.append("- Iterates over SBOM components/packages")
                elif 'results' in col.lower():
                    analysis_parts.append("- Iterates over task results")
                elif 'attestation' in col.lower():
                    analysis_parts.append("- Iterates over attestations")
                else:
                    analysis_parts.append(f"- Iterates: {var} in {col}")
        
        # Identify the check logic
        if 'count(' in code and '== 0' in code:
            analysis_parts.append("- Check: Verifies collection is not empty")
        elif 'count(' in code and '> 0' in code:
            analysis_parts.append("- Check: Verifies collection has items")
        if 'not ' in code:
            analysis_parts.append("- Check: Uses negation for condition")
        if 'regex.match' in code or 'startswith' in code:
            analysis_parts.append("- Check: Pattern/prefix matching")
        if 'time.' in code or 'effective_on' in code:
            analysis_parts.append("- Check: Time-based validation")
        
        # Identify the output format
        if 'result_helper_with_term' in code:
            analysis_parts.append("- Output: Result with searchable term")
        elif 'result_helper_with_severity' in code:
            analysis_parts.append("- Output: Result with severity level")
        elif 'result_helper' in code:
            analysis_parts.append("- Output: Standard violation result")
        
        # Fallback if nothing was detected
        if not analysis_parts:
            analysis_parts.append("- Validates rule data configuration")
        
        return '\n'.join(analysis_parts)
    
    def process_all_rules(self) -> Tuple[List[Stage1Example], List[Stage2Example]]:
        """Process all rules in policy/release and generate training examples."""
        stage1_examples = []
        stage2_examples = []
        
        # Find all Rego files (excluding tests)
        rego_files = [
            f for f in POLICY_RELEASE_DIR.rglob("*.rego")
            if not f.name.endswith("_test.rego")
        ]
        
        print(f"\nProcessing {len(rego_files)} Rego files...")
        
        for rego_file in rego_files:
            rel_path = rego_file.relative_to(POLICY_RELEASE_DIR)
            rules = self.extract_rule(rego_file)
            
            if rules:
                print(f"  {rel_path}: {len(rules)} rules")
            
            for rule in rules:
                try:
                    # Generate Stage 1 example
                    stage1 = self.generate_stage1_example(rule)
                    stage1_examples.append(stage1)
                    
                    # Generate Stage 2 example
                    stage2 = self.generate_stage2_example(rule, stage1)
                    stage2_examples.append(stage2)
                except Exception as e:
                    print(f"    Error processing rule in {rel_path}: {e}")
        
        return stage1_examples, stage2_examples


def save_examples(examples: list, output_path: Path, stage: int):
    """Save examples to JSONL file.
    
    Stage 1: instruction=natural language (varied), input=system prompt
    Stage 2: instruction=fixed prompt, input=requirements+context
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            if stage == 1:
                # Stage 1: Natural language instruction (user-facing, varied)
                data = {
                    "instruction": example.format_instruction(),
                    "input": example.format_input(),
                    "output": example.format_output(),
                }
            else:
                # Stage 2: Fixed instruction, structured input
                data = {
                    "instruction": STAGE2_INSTRUCTION,
                    "input": example.format_input(),
                    "output": example.format_output(),
                }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(examples)} examples to {output_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Two-Stage Training Data Generator")
    print("=" * 60)
    
    # Initialize generator
    generator = TwoStageDataGenerator(REPO_ROOT)
    
    # Process all rules - returns paired lists (stage1[i] corresponds to stage2[i])
    stage1_examples, stage2_examples = generator.process_all_rules()
    
    print(f"\nGenerated {len(stage1_examples)} Stage 1 examples")
    print(f"Generated {len(stage2_examples)} Stage 2 examples")
    
    # IMPORTANT: Keep paired examples in the same split (per TWO_STAGE_INFERENCE.md)
    # Shuffle indices first, then split
    indices = list(range(len(stage1_examples)))
    random.seed(42)  # Reproducible shuffle
    random.shuffle(indices)
    
    split_idx = int(len(indices) * 0.9)
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]
    
    # Split using shuffled indices
    stage1_train = [stage1_examples[i] for i in train_indices]
    stage1_eval = [stage1_examples[i] for i in eval_indices]
    stage2_train = [stage2_examples[i] for i in train_indices]
    stage2_eval = [stage2_examples[i] for i in eval_indices]
    
    # Save Stage 1 examples
    save_examples(stage1_train, OUTPUT_DIR / "stage1_train.jsonl", 1)
    save_examples(stage1_eval, OUTPUT_DIR / "stage1_eval.jsonl", 1)
    
    # Save Stage 2 examples
    save_examples(stage2_train, OUTPUT_DIR / "stage2_train.jsonl", 2)
    save_examples(stage2_eval, OUTPUT_DIR / "stage2_eval.jsonl", 2)
    
    print("\nDone!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nNote: Paired Stage 1 and Stage 2 examples are kept in the same split.")


if __name__ == "__main__":
    main()

