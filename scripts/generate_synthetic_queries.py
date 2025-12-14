#!/usr/bin/env python3
"""Generate synthetic queries for retrieval training.

Based on SID-1 Technical Report insight:
"Our synthetic question pipeline allows training to retrieve over any corpus
without human question data."

This script generates diverse, high-quality queries for each helper and schema
in the knowledge base, reducing reliance on hand-curated examples.
"""

import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set
import argparse


@dataclass
class SyntheticQuery:
    """A generated synthetic query."""
    query: str
    target_ids: List[str]  # Documents this query should retrieve
    query_type: str  # direct, problem, multi_hop, typo, etc.
    difficulty: float  # 0.0 = easy, 1.0 = hard
    source_doc: str  # Which document generated this


@dataclass
class HelperInfo:
    """Information about a helper function."""
    id: str
    signature: str
    description: str
    source: str = ""
    related_helpers: List[str] = field(default_factory=list)


@dataclass
class SchemaInfo:
    """Information about a schema field."""
    id: str
    path: str
    description: str
    attestation_type: str
    keywords: List[str] = field(default_factory=list)


# Query generation templates (no LLM required)
DIRECT_TEMPLATES = [
    "How do I use {name}",
    "What is {name}",
    "Show me how to use {name}",
    "{name} function",
    "{name} helper",
    "use {name}",
    "call {name}",
]

PROBLEM_TEMPLATES = [
    # For helpers
    "I need to {action}",
    "How can I {action}",
    "Write a rule that {action}",
    "Check if {condition}",
    "Verify that {condition}",
    "Ensure {condition}",
    "Policy to {action}",
    "Create a rule to {action}",
    
    # For schemas  
    "Access {field_desc}",
    "Get the {field_desc}",
    "Read {field_desc}",
    "What field contains {field_desc}",
    "Where is {field_desc} stored",
]

MULTI_HOP_TEMPLATES = [
    "Check all {subject} have {condition}",
    "Verify each {subject} satisfies {condition}",
    "For every {subject}, ensure {condition}",
    "Loop over {subject} and check {condition}",
    "Iterate {subject} and verify {condition}",
]

TYPO_PATTERNS = [
    # Common typos
    ("bundle", "bundel"),
    ("bundle", "bunlde"),
    ("attestation", "attestaion"),
    ("attestation", "attestion"),
    ("pipeline", "pipleine"),
    ("pipeline", "pipline"),
    ("tekton", "tektn"),
    ("digest", "digets"),
    ("schema", "shema"),
    ("verify", "verifiy"),
    ("check", "chek"),
]


class SyntheticQueryGenerator:
    """Generate synthetic queries from knowledge base."""
    
    def __init__(self, kb_dir: Path, llm_client=None):
        """
        Initialize generator.
        
        Args:
            kb_dir: Path to knowledge base
            llm_client: Optional LLM client for advanced generation
        """
        self.kb_dir = Path(kb_dir)
        self.llm = llm_client
        
        self.helpers: Dict[str, HelperInfo] = {}
        self.schemas: Dict[str, SchemaInfo] = {}
        
        self.queries: List[SyntheticQuery] = []
    
    def load_knowledge_base(self):
        """Load helpers and schemas from KB."""
        # Load helpers
        helpers_file = self.kb_dir / "helpers.jsonl"
        if helpers_file.exists():
            for line in helpers_file.read_text().strip().split('\n'):
                if not line:
                    continue
                data = json.loads(line)
                self.helpers[data['id']] = HelperInfo(
                    id=data['id'],
                    signature=data.get('signature', ''),
                    description=data.get('description', ''),
                    source=data.get('source', ''),
                    related_helpers=data.get('related', []),
                )
        
        print(f"Loaded {len(self.helpers)} helpers")
        
        # Load schemas
        schemas_file = self.kb_dir / "schemas.jsonl"
        if schemas_file.exists():
            for line in schemas_file.read_text().strip().split('\n'):
                if not line:
                    continue
                data = json.loads(line)
                self.schemas[data['schema_id']] = SchemaInfo(
                    id=data['schema_id'],
                    path=data.get('canonical_path', ''),
                    description=data.get('description', ''),
                    attestation_type=data.get('attestation_type', ''),
                    keywords=data.get('keywords', []),
                )
        
        print(f"Loaded {len(self.schemas)} schemas")
    
    def generate_all(self):
        """Generate queries for all documents."""
        print("\n=== Generating Synthetic Queries ===")
        
        # Generate for helpers
        for helper in self.helpers.values():
            self._generate_for_helper(helper)
        
        # Generate for schemas
        for schema in self.schemas.values():
            self._generate_for_schema(schema)
        
        # Generate multi-hop queries
        self._generate_multi_hop_queries()
        
        print(f"\nGenerated {len(self.queries)} total queries")
        
        # Summary by type
        by_type = {}
        for q in self.queries:
            by_type[q.query_type] = by_type.get(q.query_type, 0) + 1
        print("By type:", by_type)
    
    def _generate_for_helper(self, helper: HelperInfo):
        """Generate queries for a helper function."""
        # Extract name parts
        name = helper.id
        short_name = name.split('.')[-1] if '.' in name else name
        
        # 1. Direct queries (easy)
        for template in DIRECT_TEMPLATES:
            query = template.format(name=name)
            self.queries.append(SyntheticQuery(
                query=query,
                target_ids=[helper.id],
                query_type="direct",
                difficulty=0.1,
                source_doc=helper.id,
            ))
            
            # Also with short name
            if short_name != name:
                query = template.format(name=short_name)
                self.queries.append(SyntheticQuery(
                    query=query,
                    target_ids=[helper.id],
                    query_type="direct_short",
                    difficulty=0.2,
                    source_doc=helper.id,
                ))
        
        # 2. Problem-based queries (medium)
        if helper.description:
            # Extract action from description
            action = self._description_to_action(helper.description)
            if action:
                for template in PROBLEM_TEMPLATES[:8]:  # Only action templates
                    if "{action}" in template:
                        query = template.format(action=action)
                        self.queries.append(SyntheticQuery(
                            query=query,
                            target_ids=[helper.id],
                            query_type="problem",
                            difficulty=0.4,
                            source_doc=helper.id,
                        ))
        
        # 3. Typo queries (medium-hard)
        for query in list(self.queries[-5:]):  # Last few queries
            typo_query = self._add_typos(query.query)
            if typo_query != query.query:
                self.queries.append(SyntheticQuery(
                    query=typo_query,
                    target_ids=query.target_ids,
                    query_type="typo",
                    difficulty=0.6,
                    source_doc=helper.id,
                ))
        
        # 4. LLM-generated queries (if available)
        if self.llm:
            llm_queries = self._generate_llm_queries_for_helper(helper)
            self.queries.extend(llm_queries)
    
    def _generate_for_schema(self, schema: SchemaInfo):
        """Generate queries for a schema field."""
        # Extract field name
        path_parts = schema.path.replace('$', '').replace('[*]', '').split('.')
        field_name = path_parts[-1] if path_parts else schema.id
        
        # 1. Direct path queries
        for template in DIRECT_TEMPLATES:
            query = template.format(name=schema.path)
            self.queries.append(SyntheticQuery(
                query=query,
                target_ids=[schema.id],
                query_type="direct",
                difficulty=0.1,
                source_doc=schema.id,
            ))
        
        # 2. Field description queries
        if schema.description:
            field_desc = self._simplify_description(schema.description)
            for template in PROBLEM_TEMPLATES[8:]:  # Field templates
                if "{field_desc}" in template:
                    query = template.format(field_desc=field_desc)
                    self.queries.append(SyntheticQuery(
                        query=query,
                        target_ids=[schema.id],
                        query_type="field_query",
                        difficulty=0.4,
                        source_doc=schema.id,
                    ))
        
        # 3. Keyword-based queries
        for keyword in schema.keywords[:3]:  # Top 3 keywords
            queries = [
                f"{keyword} in {schema.attestation_type}",
                f"get {keyword}",
                f"access {keyword}",
                f"{keyword} field",
            ]
            for q in queries:
                self.queries.append(SyntheticQuery(
                    query=q,
                    target_ids=[schema.id],
                    query_type="keyword",
                    difficulty=0.3,
                    source_doc=schema.id,
                ))
    
    def _generate_multi_hop_queries(self):
        """Generate queries that require multiple documents."""
        print("  Generating multi-hop queries...")
        
        # Common multi-hop patterns
        patterns = [
            {
                "template": "Check all tasks in pipeline have pinned bundles",
                "targets": ["lib.pipelinerun_attestations", "tekton.tasks", "tekton.task_ref"],
                "subject": "tasks",
                "condition": "pinned bundles",
            },
            {
                "template": "Verify all build tasks produce images with digests",
                "targets": ["tekton.build_tasks", "tekton.task_result"],
                "subject": "build tasks",
                "condition": "images with digests",
            },
            {
                "template": "Check all SBOM packages have valid licenses",
                "targets": ["sbom.spdx_sboms", "packages[*].licenseDeclared"],
                "subject": "SBOM packages",
                "condition": "valid licenses",
            },
            {
                "template": "Ensure all attestations have proper signatures",
                "targets": ["lib.pipelinerun_attestations", "subject[*].digest"],
                "subject": "attestations",
                "condition": "proper signatures",
            },
        ]
        
        for pattern in patterns:
            # Find matching target IDs
            target_ids = []
            for t in pattern["targets"]:
                # Match by prefix
                for helper_id in self.helpers.keys():
                    if t in helper_id:
                        target_ids.append(helper_id)
                        break
                for schema_id in self.schemas.keys():
                    if t in schema_id:
                        target_ids.append(schema_id)
                        break
            
            if not target_ids:
                continue
            
            # Generate variations
            self.queries.append(SyntheticQuery(
                query=pattern["template"],
                target_ids=target_ids,
                query_type="multi_hop",
                difficulty=0.8,
                source_doc="multi_hop",
            ))
            
            # Also generate from templates
            for template in MULTI_HOP_TEMPLATES:
                query = template.format(
                    subject=pattern["subject"],
                    condition=pattern["condition"]
                )
                self.queries.append(SyntheticQuery(
                    query=query,
                    target_ids=target_ids,
                    query_type="multi_hop",
                    difficulty=0.85,
                    source_doc="multi_hop",
                ))
    
    def _description_to_action(self, description: str) -> Optional[str]:
        """Convert description to action phrase."""
        # Simple heuristics
        desc_lower = description.lower()
        
        # Remove common prefixes
        for prefix in ["returns", "gets", "checks", "verifies", "helper to"]:
            if desc_lower.startswith(prefix):
                desc_lower = desc_lower[len(prefix):].strip()
        
        # Truncate at sentence boundary
        if '.' in desc_lower:
            desc_lower = desc_lower.split('.')[0]
        
        # Skip if too short or too long
        if len(desc_lower) < 10 or len(desc_lower) > 80:
            return None
        
        return desc_lower
    
    def _simplify_description(self, description: str) -> str:
        """Simplify description to key phrase."""
        # Take first clause
        for sep in ['.', ',', ';', '-']:
            if sep in description:
                description = description.split(sep)[0]
        
        return description.lower().strip()
    
    def _add_typos(self, query: str) -> str:
        """Add realistic typos to a query."""
        result = query.lower()
        
        # Apply one random typo pattern
        for original, typo in random.sample(TYPO_PATTERNS, min(2, len(TYPO_PATTERNS))):
            if original in result:
                result = result.replace(original, typo, 1)
                break
        
        return result
    
    def _generate_llm_queries_for_helper(self, helper: HelperInfo) -> List[SyntheticQuery]:
        """Use LLM to generate high-quality queries."""
        if not self.llm:
            return []
        
        prompt = f"""
Given this Rego helper function:

Name: {helper.id}
Signature: {helper.signature}
Description: {helper.description}

Generate 5 diverse natural language queries that a developer might ask when they need this helper.
Include:
1. A simple direct question
2. A problem-focused question (describing what they want to achieve)
3. A policy-focused question ("write a rule that...")
4. A casual/informal question
5. A question with slight variation/misspelling

Return as JSON array of strings.
"""
        try:
            response = self.llm.generate(prompt)
            queries_text = json.loads(response)
            
            result = []
            for i, q in enumerate(queries_text):
                result.append(SyntheticQuery(
                    query=q,
                    target_ids=[helper.id],
                    query_type="llm_generated",
                    difficulty=0.3 + i * 0.1,  # Increasing difficulty
                    source_doc=helper.id,
                ))
            return result
        except Exception as e:
            print(f"  LLM generation failed for {helper.id}: {e}")
            return []
    
    def save(self, output_dir: Path):
        """Save generated queries as training data."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Shuffle and split
        random.shuffle(self.queries)
        split_idx = int(len(self.queries) * 0.9)
        train_queries = self.queries[:split_idx]
        eval_queries = self.queries[split_idx:]
        
        def save_queries(queries: List[SyntheticQuery], filepath: Path):
            with open(filepath, 'w') as f:
                for q in queries:
                    # For each target, create an entry
                    # This format matches the retrieval training format
                    for target_id in q.target_ids:
                        record = {
                            "query": q.query,
                            "_positive_id": target_id,
                            "_type": q.query_type,
                            "_difficulty": q.difficulty,
                            "_source_doc": q.source_doc,
                        }
                        f.write(json.dumps(record) + '\n')
        
        train_file = output_dir / "synthetic_train.jsonl"
        eval_file = output_dir / "synthetic_eval.jsonl"
        
        save_queries(train_queries, train_file)
        save_queries(eval_queries, eval_file)
        
        print(f"\nSaved {len(train_queries)} train queries to {train_file}")
        print(f"Saved {len(eval_queries)} eval queries to {eval_file}")
        
        # Stats
        stats = {
            "total_queries": len(self.queries),
            "train_queries": len(train_queries),
            "eval_queries": len(eval_queries),
            "by_type": {},
            "by_difficulty": {
                "easy": len([q for q in self.queries if q.difficulty < 0.3]),
                "medium": len([q for q in self.queries if 0.3 <= q.difficulty < 0.7]),
                "hard": len([q for q in self.queries if q.difficulty >= 0.7]),
            },
        }
        
        for q in self.queries:
            stats["by_type"][q.query_type] = stats["by_type"].get(q.query_type, 0) + 1
        
        stats_file = output_dir / "synthetic_stats.json"
        stats_file.write_text(json.dumps(stats, indent=2))
        print(f"Stats saved to {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic queries for retrieval training"
    )
    parser.add_argument(
        "--kb-dir",
        default="data/knowledge_base",
        help="Knowledge base directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data/training/retrieval",
        help="Output directory for generated queries",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM for advanced query generation (requires OPENAI_API_KEY)",
    )
    
    args = parser.parse_args()
    
    # Initialize LLM client if requested
    llm_client = None
    if args.use_llm:
        try:
            from openai import OpenAI
            llm_client = OpenAI()
            print("Using OpenAI for LLM-generated queries")
        except ImportError:
            print("Warning: openai package not installed, skipping LLM generation")
    
    # Create generator
    generator = SyntheticQueryGenerator(
        kb_dir=Path(args.kb_dir),
        llm_client=llm_client,
    )
    
    # Load KB
    generator.load_knowledge_base()
    
    # Generate queries
    generator.generate_all()
    
    # Save
    generator.save(Path(args.output_dir))


if __name__ == "__main__":
    main()

