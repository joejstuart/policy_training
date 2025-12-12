#!/usr/bin/env python3
"""Test retrieval quality interactively.

Usage:
    # Interactive mode
    python scripts/test_retrieval.py
    
    # Single query
    python scripts/test_retrieval.py --query "check if task bundle is pinned"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hybrid_retriever import HybridRetriever
from knowledge_base import KnowledgeBase


def find_repo_root() -> Path:
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "policy").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent


def test_query(retriever: HybridRetriever, kb: KnowledgeBase, query: str):
    """Test a single query and display results."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print('='*60)
    
    result = retriever.retrieve(query, helper_k=5, schema_k=3)
    
    print(f"\n📦 Retrieved {len(result.helpers)} helpers:")
    for i, h in enumerate(result.helpers, 1):
        helper_id = h.get('id', 'unknown')
        sig = h.get('signature', '')
        desc = h.get('description', '')[:60]
        print(f"  {i}. {helper_id}")
        print(f"     Signature: {sig}")
        if desc:
            print(f"     Description: {desc}...")
    
    print(f"\n📋 Retrieved {len(result.schemas)} schemas:")
    for i, s in enumerate(result.schemas, 1):
        schema_id = s.get('schema_id', 'unknown')
        path = s.get('canonical_path', '')
        att_type = s.get('attestation_type', '')
        print(f"  {i}. {schema_id}")
        print(f"     Path: {path}")
        print(f"     Type: {att_type}")
    
    print(f"\n📝 Formatted for prompt:")
    print("-" * 40)
    print(result.format_for_prompt()[:1000])
    if len(result.format_for_prompt()) > 1000:
        print("... (truncated)")


def interactive_mode(retriever: HybridRetriever, kb: KnowledgeBase):
    """Interactive query testing."""
    print("\n🔍 Rego Assistant - Retrieval Test")
    print("Type a query to test retrieval. Type 'quit' to exit.\n")
    
    # Sample queries to try
    print("Example queries to try:")
    print("  - check if task bundle is pinned")
    print("  - detect GPL licenses in SBOM")
    print("  - verify all required tasks are present")
    print("  - validate image signatures")
    print()
    
    while True:
        try:
            query = input("Query> ").strip()
            if query.lower() in ('quit', 'exit', 'q'):
                break
            if not query:
                continue
            
            test_query(retriever, kb, query)
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Test retrieval quality")
    parser.add_argument("--query", "-q", help="Single query to test")
    parser.add_argument("--kb-dir", type=Path, help="Knowledge base directory")
    args = parser.parse_args()
    
    repo_root = find_repo_root()
    kb_dir = args.kb_dir or (repo_root / "data" / "knowledge_base")
    
    print(f"Loading KB from: {kb_dir}")
    
    # Load KB
    kb = KnowledgeBase(kb_dir)
    print(f"  Loaded {len(kb.helper_cards)} helpers, {len(kb.schemas)} schemas")
    
    # Load retriever
    print("Loading retriever...")
    try:
        retriever = HybridRetriever.from_kb_dir(kb_dir)
        print("  ✓ Retriever loaded")
    except Exception as e:
        print(f"  ✗ Error loading retriever: {e}")
        print("\nMake sure you've run: python scripts/build_index.py")
        sys.exit(1)
    
    if args.query:
        test_query(retriever, kb, args.query)
    else:
        interactive_mode(retriever, kb)


if __name__ == "__main__":
    main()

