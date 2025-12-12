#!/usr/bin/env python3
"""Enrich Knowledge Base with LLM-generated metadata.

Uses real usage examples from production rules to provide grounded context
to the LLM for generating descriptions and use_when tags.

Usage:
    # Enrich using Ollama (default)
    python scripts/enrich_kb.py
    
    # Enrich using specific model
    python scripts/enrich_kb.py --model qwen3-coder:30b
    
    # Dry run (show context without calling LLM)
    python scripts/enrich_kb.py --dry-run
    
    # Enrich only items without descriptions
    python scripts/enrich_kb.py --skip-existing
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from knowledge_base import KnowledgeBase, HelperFull, HelperCard
from schema_extractor import SchemaField
from usage_miner import UsageMiner


class OllamaClient:
    """Simple Ollama API client."""
    
    def __init__(self, model: str = "qwen3-coder:30b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._available = None
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if self._available is not None:
            return self._available
        
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                self._available = resp.status == 200
        except:
            self._available = False
        
        return self._available
    
    def generate(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Generate completion from Ollama."""
        if not self.is_available():
            return None
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.3,
            }
        }
        
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=json.dumps(data).encode('utf-8'),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=180) as resp:  # 3 minutes for large models
                result = json.loads(resp.read().decode('utf-8'))
                return result.get("response", "")
        except Exception as e:
            print(f"  LLM error: {e}")
            return None


def find_repo_root() -> Path:
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "policy").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent


def build_helper_context(helper: HelperFull, miner: UsageMiner) -> str:
    """Build rich context for a helper function."""
    
    # Get usage info
    usage_info = miner.get_helper_usages(helper.name, max_examples=2)
    
    parts = []
    
    # Header
    parts.append(f"HELPER FUNCTION: {helper.id}")
    parts.append(f"Module: {helper.module_path}")
    parts.append("")
    
    # Source code
    parts.append(f"SOURCE CODE (from {helper.source_file}:{helper.source_lines[0]}-{helper.source_lines[1]}):")
    parts.append("```rego")
    parts.append(helper.body)
    parts.append("```")
    parts.append("")
    
    # Usage examples
    if usage_info.examples:
        parts.append("REAL USAGE IN PRODUCTION RULES:")
        for i, ex in enumerate(usage_info.examples, 1):
            parts.append(f"\nExample {i} (from {ex.file_path}):")
            parts.append(f"Rule type: {ex.rule_name}")
            if ex.attestation_type:
                parts.append(f"Attestation type: {ex.attestation_type}")
            parts.append("```rego")
            # Truncate if too long
            context = ex.context
            if len(context) > 500:
                context = context[:500] + "\n... (truncated)"
            parts.append(context)
            parts.append("```")
        parts.append("")
    
    # Co-occurring helpers
    if usage_info.co_occurring_helpers:
        top_related = sorted(usage_info.co_occurring_helpers.items(), key=lambda x: -x[1])[:5]
        related_list = [f"{h} ({c}x)" for h, c in top_related]
        parts.append(f"COMMONLY USED WITH: {', '.join(related_list)}")
        parts.append("")
    
    # Common patterns
    if usage_info.common_patterns:
        parts.append(f"COMMON ITERATION PATTERNS: {', '.join(usage_info.common_patterns)}")
        parts.append("")
    
    return "\n".join(parts)


def gather_example_values(schema: SchemaField, attestation_dir: Path) -> List[str]:
    """Gather diverse example values for a schema field from attestations."""
    if not attestation_dir or not attestation_dir.exists():
        return []
    
    examples = set()
    path = schema.canonical_path
    
    # Convert JSONPath to key sequence
    # $.predicate.buildConfig.tasks[*].ref.params[*].value
    # -> ['predicate', 'buildConfig', 'tasks', '*', 'ref', 'params', '*', 'value']
    keys = path.replace('$', '').replace('[*]', '.[*]').split('.')
    keys = [k for k in keys if k]
    
    for att_file in list(attestation_dir.glob("*.json"))[:10]:  # Sample 10 files
        try:
            with open(att_file) as f:
                data = json.load(f)
            
            values = extract_values_at_path(data, keys)
            for v in values:
                if isinstance(v, str) and len(v) < 200:
                    examples.add(v)
                    if len(examples) >= 10:
                        break
        except:
            pass
        
        if len(examples) >= 10:
            break
    
    return list(examples)


def extract_values_at_path(data: Any, keys: List[str]) -> List[Any]:
    """Extract all values at a given path (handling [*] wildcards)."""
    if not keys:
        return [data] if data is not None else []
    
    key = keys[0]
    remaining = keys[1:]
    
    if key == '[*]':
        # Iterate over array
        if isinstance(data, list):
            results = []
            for item in data:
                results.extend(extract_values_at_path(item, remaining))
            return results
        return []
    else:
        # Access dict key
        if isinstance(data, dict) and key in data:
            return extract_values_at_path(data[key], remaining)
        return []


def gather_param_examples(attestation_dir: Path) -> List[Tuple[str, str]]:
    """Gather param name/value pairs to show what params contain."""
    if not attestation_dir or not attestation_dir.exists():
        return []
    
    examples = []
    seen_names = set()
    
    for att_file in list(attestation_dir.glob("*.json"))[:5]:
        try:
            with open(att_file) as f:
                data = json.load(f)
            
            # Navigate to tasks
            tasks = data.get('predicate', {}).get('buildConfig', {}).get('tasks', [])
            for task in tasks:
                params = task.get('ref', {}).get('params', [])
                for param in params:
                    name = param.get('name', '')
                    value = param.get('value', '')
                    if name and name not in seen_names and isinstance(value, str):
                        seen_names.add(name)
                        examples.append((name, value))
                        if len(examples) >= 10:
                            return examples
        except:
            pass
    
    return examples


def build_schema_context(schema: SchemaField, miner: UsageMiner, attestation_dir: Path = None) -> str:
    """Build rich context for a schema field."""
    
    # Extract key field name from path
    path_parts = schema.canonical_path.split('.')
    field_name = path_parts[-1].replace('[*]', '') if path_parts else ""
    
    # Find usages
    usages = miner.find_schema_usages(field_name) if field_name else []
    
    parts = []
    
    # Header
    parts.append(f"SCHEMA FIELD: {schema.schema_id}")
    parts.append(f"Path: {schema.canonical_path}")
    parts.append(f"Type: {schema.field_type}")
    parts.append(f"Attestation: {schema.attestation_type}")
    parts.append("")
    
    # Gather multiple example values from attestations
    example_values = gather_example_values(schema, attestation_dir)
    if example_values:
        parts.append("EXAMPLE VALUES FROM ATTESTATIONS:")
        for ex in example_values[:5]:  # Show up to 5 examples
            parts.append(f"  - {ex}")
        parts.append("")
    elif schema.example_value is not None:
        example_str = json.dumps(schema.example_value) if not isinstance(schema.example_value, str) else schema.example_value
        if len(example_str) > 200:
            example_str = example_str[:197] + "..."
        parts.append(f"EXAMPLE VALUE: {example_str}")
        parts.append("")
    
    # For params, show name/value pairs
    if "params" in schema.canonical_path.lower():
        param_examples = gather_param_examples(attestation_dir)
        if param_examples:
            parts.append("PARAM NAME/VALUE PAIRS (what this field contains):")
            for name, value in param_examples[:5]:
                val_str = value[:60] + "..." if len(value) > 60 else value
                parts.append(f"  - {name}: {val_str}")
            parts.append("")
    
    # Usage in rules
    if usages:
        parts.append("HOW IT'S USED IN RULES:")
        for i, ex in enumerate(usages[:2], 1):
            parts.append(f"\nExample {i} (from {ex.file_path}):")
            parts.append(f"Rule type: {ex.rule_name}")
            context = ex.context
            if len(context) > 400:
                context = context[:400] + "\n... (truncated)"
            parts.append("```rego")
            parts.append(context)
            parts.append("```")
        parts.append("")
    
    return "\n".join(parts)


def build_helper_prompt(context: str) -> str:
    """Build the prompt for helper enrichment."""
    return f"""/no_think
You are analyzing a Rego helper function from a policy library.

{context}

Based ONLY on the source code and usage examples above, respond in this exact JSON format:

{{
  "description": "One sentence describing what this helper does and returns",
  "use_when": ["scenario 1", "scenario 2", "scenario 3"],
  "expects": "What input type/format this function expects",
  "returns": "What this function returns",
  "gotchas": "Any edge cases or important notes (or null if none)"
}}

RULES:
- Base your answer ONLY on the provided code and examples
- Keep description under 100 characters
- Include 2-4 specific use_when scenarios
- Be precise about expects/returns based on the code

JSON response:"""


def build_schema_prompt(context: str) -> str:
    """Build the prompt for schema enrichment."""
    return f"""/no_think
You are analyzing a schema field from attestation data used in Rego policies.

{context}

Based ONLY on the path, example values, and param pairs above, respond in this exact JSON format:

{{
  "description": "One sentence describing what this field contains - INCLUDE specific value types seen in examples (e.g., bundle references, digests, task names)",
  "use_when": ["specific policy check 1", "specific policy check 2", "specific policy check 3"],
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}

RULES:
- Base your answer ONLY on the provided information
- Look at EXAMPLE VALUES and PARAM PAIRS to understand what this field actually contains
- If you see values like "quay.io/...@sha256:..." mention "bundle reference" and "digest" and "pinned"
- If you see task names, mention "task name"
- The use_when should be SPECIFIC policy checks like "verify bundle is pinned", "check task reference contains digest"
- Keywords should include terms that someone searching for this field would use
- Keep description under 100 characters
- Include 2-3 specific use_when scenarios
- Describe common_checks based on the rule examples

JSON response:"""


def parse_llm_response(response: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response."""
    if not response:
        return None
    
    # Try to find JSON in response
    response = response.strip()
    
    # Handle markdown code blocks
    if "```json" in response:
        start = response.find("```json") + 7
        end = response.find("```", start)
        if end > start:
            response = response[start:end].strip()
    elif "```" in response:
        start = response.find("```") + 3
        end = response.find("```", start)
        if end > start:
            response = response[start:end].strip()
    
    # Find JSON object
    if "{" in response:
        start = response.find("{")
        # Find matching closing brace
        brace_count = 0
        end = start
        for i, c in enumerate(response[start:], start):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        
        json_str = response[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    return None


def enrich_helper(helper: HelperFull, miner: UsageMiner, llm: OllamaClient, dry_run: bool = False) -> HelperFull:
    """Enrich a single helper with LLM-generated metadata."""
    
    context = build_helper_context(helper, miner)
    
    if dry_run:
        print(f"\n{'='*60}")
        print(f"CONTEXT FOR: {helper.id}")
        print('='*60)
        print(context)
        return helper
    
    prompt = build_helper_prompt(context)
    response = llm.generate(prompt, max_tokens=400)
    
    if not response:
        return helper
    
    parsed = parse_llm_response(response)
    
    if parsed:
        # Update helper with enriched data
        if "description" in parsed:
            helper.description = parsed["description"]
        if "use_when" in parsed and isinstance(parsed["use_when"], list):
            helper.use_when = parsed["use_when"]
        
        print(f"  ✓ Enriched: {parsed.get('description', '')[:50]}...")
    else:
        print(f"  ✗ Could not parse response")
    
    return helper


def enrich_schema(schema: SchemaField, miner: UsageMiner, llm: OllamaClient, attestation_dir: Path = None, dry_run: bool = False) -> SchemaField:
    """Enrich a single schema with LLM-generated metadata."""
    
    context = build_schema_context(schema, miner, attestation_dir)
    
    if dry_run:
        print(f"\n{'='*60}")
        print(f"CONTEXT FOR: {schema.schema_id}")
        print('='*60)
        print(context)
        return schema
    
    prompt = build_schema_prompt(context)
    response = llm.generate(prompt, max_tokens=300)
    
    if not response:
        return schema
    
    parsed = parse_llm_response(response)
    
    if parsed:
        if "description" in parsed:
            schema.description = parsed["description"]
        
        # Combine use_when and keywords for better retrieval
        use_when = parsed.get("use_when", []) if isinstance(parsed.get("use_when"), list) else []
        keywords = parsed.get("keywords", []) if isinstance(parsed.get("keywords"), list) else []
        
        # Add keywords as use_when entries for retrieval
        combined = use_when + [f"keyword: {kw}" for kw in keywords]
        if combined:
            schema.use_when = combined
        
        print(f"  ✓ Enriched: {parsed.get('description', '')[:50]}...")
        if keywords:
            print(f"    Keywords: {', '.join(keywords[:5])}")
    else:
        print(f"  ✗ Could not parse response")
    
    return schema


def main():
    parser = argparse.ArgumentParser(description="Enrich KB with LLM-generated metadata")
    parser.add_argument("--kb-dir", type=Path, help="Knowledge base directory")
    parser.add_argument("--model", default="qwen3-coder:30b", help="Ollama model")
    parser.add_argument("--dry-run", action="store_true", help="Show context without calling LLM")
    parser.add_argument("--skip-existing", action="store_true", help="Skip items with descriptions")
    parser.add_argument("--helpers-only", action="store_true", help="Only enrich helpers")
    parser.add_argument("--schemas-only", action="store_true", help="Only enrich schemas")
    parser.add_argument("--limit", type=int, help="Limit number of items to enrich")
    
    args = parser.parse_args()
    
    repo_root = find_repo_root()
    kb_dir = args.kb_dir or (repo_root / "data" / "knowledge_base")
    attestation_dir = repo_root / "data" / "attestations"
    
    print(f"Loading KB from: {kb_dir}")
    kb = KnowledgeBase(kb_dir)
    print(f"  Loaded {len(kb.helper_fulls)} helpers, {len(kb.schemas)} schemas")
    
    print(f"\nInitializing usage miner...")
    miner = UsageMiner(repo_root / "policy")
    miner.scan_all_rules()
    
    if not args.dry_run:
        print(f"\nInitializing LLM ({args.model})...")
        llm = OllamaClient(model=args.model)
        if not llm.is_available():
            print("  ✗ Ollama not available. Make sure it's running.")
            print("  Try: ollama serve")
            sys.exit(1)
        print("  ✓ LLM ready")
    else:
        llm = None
        print("\n[DRY RUN MODE - showing context only]")
    
    # Enrich helpers
    if not args.schemas_only:
        print(f"\n{'='*60}")
        print("ENRICHING HELPERS")
        print('='*60)
        
        helpers_to_enrich = list(kb.helper_fulls.values())
        if args.skip_existing:
            helpers_to_enrich = [h for h in helpers_to_enrich if not h.description]
        if args.limit:
            helpers_to_enrich = helpers_to_enrich[:args.limit]
        
        print(f"Enriching {len(helpers_to_enrich)} helpers...")
        
        for i, helper in enumerate(helpers_to_enrich, 1):
            print(f"\n[{i}/{len(helpers_to_enrich)}] {helper.id}")
            enriched = enrich_helper(helper, miner, llm, args.dry_run)
            kb.helper_fulls[helper.id] = enriched
            kb.helper_cards[helper.id] = enriched.to_card()
    
    # Enrich schemas
    if not args.helpers_only:
        print(f"\n{'='*60}")
        print("ENRICHING SCHEMAS")
        print('='*60)
        
        schemas_to_enrich = list(kb.schemas.values())
        if args.skip_existing:
            schemas_to_enrich = [s for s in schemas_to_enrich if not s.description or s.description == s.canonical_path.split('.')[-1]]
        if args.limit:
            schemas_to_enrich = schemas_to_enrich[:args.limit]
        
        print(f"Enriching {len(schemas_to_enrich)} schemas...")
        
        for i, schema in enumerate(schemas_to_enrich, 1):
            print(f"\n[{i}/{len(schemas_to_enrich)}] {schema.schema_id}")
            enriched = enrich_schema(schema, miner, llm, attestation_dir, args.dry_run)
            kb.schemas[schema.schema_id] = enriched
    
    # Save enriched KB
    if not args.dry_run:
        print(f"\nSaving enriched KB to {kb_dir}...")
        kb.save(kb_dir)
        print("✓ Done!")
        print("\nRemember to rebuild indexes:")
        print("  uv run python scripts/build_index.py")


if __name__ == "__main__":
    main()

