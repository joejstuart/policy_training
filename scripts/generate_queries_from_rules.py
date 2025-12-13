#!/usr/bin/env python3
"""Generate diverse retrieval training queries from actual policy rules.

For each policy rule in the codebase:
1. Extract what schemas it accesses
2. Extract what helpers it uses  
3. Use LLM to generate 5-10 varied natural language queries
4. Create training pairs: (query, correct_schema, correct_helper)

This ensures training data covers ALL actual use cases in the codebase.
"""

import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class PolicyRule:
    """Extracted information from a policy rule."""
    file_path: str
    package: str
    rule_name: str  # deny, warn, etc.
    title: str
    description: str
    short_name: str
    
    # What schemas this rule accesses
    schema_paths: List[str] = field(default_factory=list)
    
    # What helpers this rule uses
    helpers_used: List[str] = field(default_factory=list)
    
    # The actual rule body
    rule_body: str = ""
    
    # Domain: slsa, sbom, image
    domain: str = "slsa"


class OllamaClient:
    """Simple Ollama client for query generation."""
    
    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        """Generate completion."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.7}
        }
        
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(data).encode('utf-8'),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result.get("message", {}).get("content", "")
        except Exception as e:
            print(f"  LLM error: {e}")
            return ""


class PolicyRuleExtractor:
    """Extract policy rules and their dependencies from the codebase."""
    
    # Patterns for schema access
    SCHEMA_PATTERNS = [
        # Direct attestation access patterns
        (r'att\.statement\.(\w+)', 'statement'),
        (r'att\.predicate\.(\w+(?:\.\w+)*)', 'predicate'),
        (r'\.predicate\.buildConfig\.tasks\b', 'tasks'),
        (r'\.predicate\.materials\b', 'materials'),
        (r'\.subject\b', 'subject'),
        (r'task\.ref\.params', 'ref.params'),
        (r'task\.ref\.resolver', 'ref.resolver'),
        (r'task\.ref\.bundle', 'ref.bundle'),
        (r'task\.results', 'results'),
        (r'task\.status', 'status'),
        (r'task\.name', 'tasks.name'),
        (r's\.packages', 'packages'),
        (r's\.components', 'components'),
        (r'sbom\.packages', 'packages'),
    ]
    
    # Patterns for helper usage
    HELPER_PATTERNS = [
        r'lib\.(\w+(?:\.\w+)*)\s*[\(\[]',
        r'tekton\.(\w+)\s*[\(\[]',
        r'sbom\.(\w+)\s*[\(\[]',
        r'image\.(\w+)\s*[\(\[]',
    ]
    
    def __init__(self, policy_dir: Path):
        self.policy_dir = Path(policy_dir)
        self.rules: List[PolicyRule] = []
    
    def extract_all(self) -> List[PolicyRule]:
        """Extract all policy rules from the codebase."""
        release_dir = self.policy_dir / "release"
        
        if not release_dir.exists():
            print(f"Warning: {release_dir} does not exist")
            return []
        
        # Find all non-test rego files
        rego_files = [f for f in release_dir.rglob("*.rego") 
                      if "_test.rego" not in f.name]
        
        print(f"Found {len(rego_files)} policy files")
        
        for rego_file in rego_files:
            try:
                self._extract_from_file(rego_file)
            except Exception as e:
                print(f"  Error extracting {rego_file}: {e}")
        
        print(f"Extracted {len(self.rules)} rules")
        return self.rules
    
    def _extract_from_file(self, file_path: Path):
        """Extract rules from a single file."""
        source = file_path.read_text(encoding='utf-8')
        lines = source.split('\n')
        
        # Extract package
        package = ""
        for line in lines:
            if line.strip().startswith("package "):
                package = line.strip().split()[1]
                break
        
        # Find all rules with METADATA
        rule_blocks = self._find_rule_blocks(source, lines)
        
        rel_path = str(file_path.relative_to(self.policy_dir))
        
        for block in rule_blocks:
            rule = PolicyRule(
                file_path=rel_path,
                package=package,
                rule_name=block['rule_name'],
                title=block.get('title', ''),
                description=block.get('description', ''),
                short_name=block.get('short_name', ''),
                rule_body=block['body'],
            )
            
            # Extract schema access patterns
            rule.schema_paths = self._extract_schemas(block['body'])
            
            # Extract helper usage
            rule.helpers_used = self._extract_helpers(block['body'])
            
            # Detect domain
            rule.domain = self._detect_domain(block['body'], package)
            
            if rule.title or rule.description:  # Only include documented rules
                self.rules.append(rule)
    
    def _find_rule_blocks(self, source: str, lines: List[str]) -> List[dict]:
        """Find rule blocks with their metadata."""
        blocks = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Look for METADATA comment block
            if line == "# METADATA":
                metadata = {}
                j = i + 1
                
                # Parse metadata
                while j < len(lines):
                    meta_line = lines[j]
                    if not meta_line.strip().startswith('#'):
                        break
                    
                    meta_line = meta_line.strip().lstrip('#').strip()
                    
                    if meta_line.startswith('title:'):
                        metadata['title'] = meta_line.split(':', 1)[1].strip()
                    elif meta_line.startswith('description:'):
                        # Multi-line description
                        desc_parts = [meta_line.split(':', 1)[1].strip()]
                        k = j + 1
                        while k < len(lines):
                            next_line = lines[k].strip()
                            if next_line.startswith('#') and not any(
                                next_line.lstrip('#').strip().startswith(kw) 
                                for kw in ['title:', 'custom:', 'short_name:', 'failure_msg:']
                            ):
                                desc_parts.append(next_line.lstrip('#').strip())
                                k += 1
                            else:
                                break
                        metadata['description'] = ' '.join(desc_parts).strip().strip('>-').strip()
                    elif 'short_name:' in meta_line:
                        metadata['short_name'] = meta_line.split('short_name:', 1)[1].strip()
                    
                    j += 1
                
                # Find the rule after metadata
                while j < len(lines):
                    rule_line = lines[j].strip()
                    if rule_line.startswith(('deny ', 'warn ', 'allow ', 'violation ')):
                        rule_name = rule_line.split()[0]
                        
                        # Extract rule body
                        body_lines = [lines[j]]
                        brace_count = lines[j].count('{') - lines[j].count('}')
                        k = j + 1
                        while k < len(lines) and brace_count > 0:
                            body_lines.append(lines[k])
                            brace_count += lines[k].count('{') - lines[k].count('}')
                            k += 1
                        
                        blocks.append({
                            **metadata,
                            'rule_name': rule_name,
                            'body': '\n'.join(body_lines),
                        })
                        i = k
                        break
                    elif rule_line and not rule_line.startswith('#'):
                        break
                    j += 1
            
            i += 1
        
        return blocks
    
    def _extract_schemas(self, body: str) -> List[str]:
        """Extract schema paths accessed in rule body."""
        schemas = set()
        
        for pattern, schema_hint in self.SCHEMA_PATTERNS:
            if re.search(pattern, body):
                schemas.add(schema_hint)
        
        return list(schemas)
    
    def _extract_helpers(self, body: str) -> List[str]:
        """Extract helper functions used in rule body."""
        helpers = set()
        
        for pattern in self.HELPER_PATTERNS:
            for match in re.finditer(pattern, body):
                helper_name = match.group(1)
                # Reconstruct full helper name
                prefix = pattern.split(r'\.')[0]  # lib, tekton, etc.
                helpers.add(f"{prefix}.{helper_name}")
        
        return list(helpers)
    
    def _detect_domain(self, body: str, package: str) -> str:
        """Detect domain from rule body and package."""
        body_lower = body.lower()
        
        if 'sbom' in package.lower() or 'sbom.' in body_lower or 'spdx' in body_lower:
            return 'sbom'
        if 'image' in package.lower() and 'task' not in package.lower():
            return 'image'
        return 'slsa'


class QueryGenerator:
    """Generate diverse queries for policy rules using LLM."""
    
    SYSTEM_PROMPT = """You are helping generate training data for a retrieval system.
Given a policy rule description, generate diverse natural language queries that a user might ask
when they want to write or understand this rule.

Generate 8 varied queries, ranging from:
- Direct: "check if X"
- Exploratory: "how does X work"  
- Requirement-focused: "I need to ensure X"
- Problem-focused: "block Y when Z"
- Specific: with concrete examples

Output ONLY the queries, one per line. No numbering, no explanations."""

    def __init__(self, llm: OllamaClient):
        self.llm = llm
    
    def generate_queries(self, rule: PolicyRule) -> List[str]:
        """Generate diverse queries for a policy rule."""
        
        # Build context for LLM
        prompt = f"""Generate 8 diverse natural language queries for this policy rule:

Rule Title: {rule.title}
Description: {rule.description}
Rule Type: {rule.rule_name}
Domain: {rule.domain}
Helpers Used: {', '.join(rule.helpers_used) if rule.helpers_used else 'none specific'}
Schema Accessed: {', '.join(rule.schema_paths) if rule.schema_paths else 'attestation data'}

Example query styles:
- "Check if [condition]"
- "How do I verify [requirement]"  
- "Write a rule that [action]"
- "I need to ensure [condition]"
- "Block [subject] when [condition]"
- "How does [feature] work"

Generate 8 varied queries:"""

        response = self.llm.generate(prompt, system=self.SYSTEM_PROMPT)
        
        # Parse queries from response
        queries = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # Remove numbering if present
            line = re.sub(r'^[\d]+[\.\)]\s*', '', line)
            line = re.sub(r'^[-*]\s*', '', line)
            line = line.strip('"\'')
            
            if line and len(line) > 10:  # Skip empty or too short
                queries.append(line)
        
        return queries[:10]  # Cap at 10


class RetrievalDataGenerator:
    """Generate retrieval training data from policy rules."""
    
    def __init__(
        self, 
        policy_dir: Path, 
        kb_dir: Path,
        llm: OllamaClient,
    ):
        self.policy_dir = Path(policy_dir)
        self.kb_dir = Path(kb_dir)
        self.llm = llm
        
        self.extractor = PolicyRuleExtractor(policy_dir)
        self.query_gen = QueryGenerator(llm)
        
        # Load KB for schema/helper mapping
        self.schemas = self._load_schemas()
        self.helpers = self._load_helpers()
    
    def _load_schemas(self) -> Dict[str, dict]:
        """Load schemas from KB."""
        schemas = {}
        schema_file = self.kb_dir / "schemas.jsonl"
        
        if schema_file.exists():
            for line in schema_file.read_text().strip().split('\n'):
                if line:
                    data = json.loads(line)
                    schemas[data.get('schema_id', '')] = data
        
        return schemas
    
    def _load_helpers(self) -> Dict[str, dict]:
        """Load helpers from KB."""
        helpers = {}
        helper_file = self.kb_dir / "helpers.jsonl"
        
        if helper_file.exists():
            for line in helper_file.read_text().strip().split('\n'):
                if line:
                    data = json.loads(line)
                    helpers[data.get('id', '')] = data
        
        return helpers
    
    def _find_matching_schema(self, schema_hint: str) -> Optional[str]:
        """Find KB schema matching a hint from rule body."""
        hint_lower = schema_hint.lower()
        
        # Mapping of rule body patterns to schema path patterns
        hint_to_path = {
            'tasks': 'tasks[',
            'tasks.name': 'tasks[*].name',
            'results': 'results[',
            'status': 'status',
            'ref.params': 'ref.params',
            'ref.resolver': 'ref.resolver',
            'ref.bundle': 'ref.bundle',
            'materials': 'materials[',
            'subject': 'subject[',
            'predicate': 'predicate',
            'packages': 'packages[',
            'components': 'components[',
        }
        
        # Get the path pattern to match
        path_pattern = hint_to_path.get(hint_lower, hint_lower)
        
        # Find matching schemas
        matches = []
        for schema_id, schema in self.schemas.items():
            path = schema.get('canonical_path', '').lower()
            
            if path_pattern in path:
                matches.append(schema_id)
        
        # Return first match (or most specific if multiple)
        if matches:
            # Prefer more specific paths (longer)
            matches.sort(key=lambda x: len(self.schemas[x].get('canonical_path', '')), reverse=True)
            return matches[0]
        
        return None
    
    def _find_matching_helper(self, helper_hint: str) -> Optional[str]:
        """Find KB helper matching a hint from rule body."""
        hint_lower = helper_hint.lower()
        
        for helper_id, helper in self.helpers.items():
            if hint_lower in helper_id.lower():
                return helper_id
        
        # Try with lib. prefix
        for helper_id, helper in self.helpers.items():
            if f"lib.{hint_lower}" in helper_id.lower():
                return helper_id
        
        return None
    
    def _get_schema_text(self, schema_id: str) -> str:
        """Get searchable text for a schema."""
        schema = self.schemas.get(schema_id, {})
        parts = [
            f"Path: {schema.get('canonical_path', '')}",
            f"Description: {schema.get('description', '')}",
        ]
        keywords = schema.get('keywords', [])
        if keywords:
            parts.append(f"Keywords: {', '.join(keywords)}")
        return '\n'.join(parts)
    
    def _get_helper_text(self, helper_id: str) -> str:
        """Get searchable text for a helper."""
        helper = self.helpers.get(helper_id, {})
        parts = [
            f"Helper: {helper_id}",
            f"Signature: {helper.get('signature', '')}",
            f"Description: {helper.get('description', '')}",
        ]
        return '\n'.join(parts)
    
    def _get_hard_negative_schema(self, positive_id: str, domain: str) -> Optional[Tuple[str, str]]:
        """Get a hard negative schema (same domain, different field)."""
        positive_schema = self.schemas.get(positive_id, {})
        positive_type = positive_schema.get('attestation_type', '')
        
        for schema_id, schema in self.schemas.items():
            if schema_id == positive_id:
                continue
            if schema.get('attestation_type', '') == positive_type:
                return (schema_id, self._get_schema_text(schema_id))
        
        # Any different schema
        for schema_id, schema in self.schemas.items():
            if schema_id != positive_id:
                return (schema_id, self._get_schema_text(schema_id))
        
        return None
    
    def _get_hard_negative_helper(self, positive_id: str, domain: str) -> Optional[Tuple[str, str]]:
        """Get a hard negative helper (same module, different function)."""
        parts = positive_id.split('.')
        module = parts[1] if len(parts) > 2 else parts[0]
        
        for helper_id, helper in self.helpers.items():
            if helper_id == positive_id:
                continue
            if module in helper_id:
                return (helper_id, self._get_helper_text(helper_id))
        
        # Any different helper
        for helper_id, helper in self.helpers.items():
            if helper_id != positive_id:
                return (helper_id, self._get_helper_text(helper_id))
        
        return None
    
    def generate(self, output_file: Path, max_rules: Optional[int] = None):
        """Generate retrieval training data."""
        
        # Extract rules
        rules = self.extractor.extract_all()
        
        if max_rules:
            rules = rules[:max_rules]
        
        examples = []
        
        for i, rule in enumerate(rules):
            print(f"\n[{i+1}/{len(rules)}] {rule.title or rule.short_name}")
            
            # Generate queries
            queries = self.query_gen.generate_queries(rule)
            print(f"  Generated {len(queries)} queries")
            
            if not queries:
                continue
            
            # Find matching schemas
            matched_schemas = []
            for hint in rule.schema_paths:
                schema_id = self._find_matching_schema(hint)
                if schema_id:
                    matched_schemas.append(schema_id)
            
            # Find matching helpers
            matched_helpers = []
            for hint in rule.helpers_used:
                helper_id = self._find_matching_helper(hint)
                if helper_id:
                    matched_helpers.append(helper_id)
            
            print(f"  Matched {len(matched_schemas)} schemas, {len(matched_helpers)} helpers")
            
            # Create training examples
            for query in queries:
                # Schema examples
                for schema_id in matched_schemas:
                    neg = self._get_hard_negative_schema(schema_id, rule.domain)
                    if neg:
                        examples.append({
                            "query": query,
                            "positive": self._get_schema_text(schema_id),
                            "negative": neg[1],
                            "_positive_id": schema_id,
                            "_negative_id": neg[0],
                            "_type": "schema",
                            "_source": "rule_generated",
                            "_domain": rule.domain,
                            "_rule": rule.short_name or rule.title,
                        })
                
                # Helper examples
                for helper_id in matched_helpers:
                    neg = self._get_hard_negative_helper(helper_id, rule.domain)
                    if neg:
                        examples.append({
                            "query": query,
                            "positive": self._get_helper_text(helper_id),
                            "negative": neg[1],
                            "_positive_id": helper_id,
                            "_negative_id": neg[0],
                            "_type": "helper",
                            "_source": "rule_generated",
                            "_domain": rule.domain,
                            "_rule": rule.short_name or rule.title,
                        })
            
            # Rate limit
            time.sleep(0.5)
        
        # Save examples
        print(f"\nSaving {len(examples)} examples to {output_file}")
        with open(output_file, 'w') as f:
            for ex in examples:
                f.write(json.dumps(ex) + '\n')
        
        # Stats
        schema_count = len([e for e in examples if e['_type'] == 'schema'])
        helper_count = len([e for e in examples if e['_type'] == 'helper'])
        print(f"  Schema examples: {schema_count}")
        print(f"  Helper examples: {helper_count}")
        
        return examples


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate queries from policy rules")
    parser.add_argument("--policy-dir", default="policy", help="Policy directory")
    parser.add_argument("--kb-dir", default="data/knowledge_base", help="KB directory")
    parser.add_argument("--output", default="data/training/retrieval/rule_generated.jsonl")
    parser.add_argument("--ollama-model", default="llama3.2", help="Ollama model for query generation")
    parser.add_argument("--max-rules", type=int, help="Max rules to process (for testing)")
    
    args = parser.parse_args()
    
    llm = OllamaClient(model=args.ollama_model)
    
    generator = RetrievalDataGenerator(
        policy_dir=Path(args.policy_dir),
        kb_dir=Path(args.kb_dir),
        llm=llm,
    )
    
    generator.generate(
        output_file=Path(args.output),
        max_rules=args.max_rules,
    )


if __name__ == "__main__":
    main()

