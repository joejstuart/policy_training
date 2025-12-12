#!/usr/bin/env python3
"""Build vector and BM25 search indexes from knowledge base.

Usage:
    # Build all indexes
    python scripts/build_index.py
    
    # Build only vector indexes
    python scripts/build_index.py --vector-only
    
    # Build only BM25 indexes
    python scripts/build_index.py --bm25-only
    
    # Use different embedding model
    python scripts/build_index.py --embedding-model bge-small-en-v1.5

Per architecture spec:
- Vector search for semantic matching
- BM25 for exact symbol matching
- Hybrid retrieval combines both
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def find_repo_root() -> Path:
    """Find repository root."""
    current = Path(__file__).parent
    while current != current.parent:
        if (current / "policy").exists():
            return current
        current = current.parent
    return Path(__file__).parent.parent


def load_jsonl(path: Path):
    """Load JSONL file."""
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_vector_indexes(kb_dir: Path, embedding_model: str):
    """Build FAISS vector indexes.
    
    Args:
        kb_dir: Knowledge base directory
        embedding_model: Sentence transformer model name
    """
    from vector_index import VectorIndex, create_helper_chunks, create_schema_chunks
    
    index_dir = kb_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Build helpers index
    helpers_path = kb_dir / "helpers.jsonl"
    if helpers_path.exists():
        print(f"\nBuilding helpers vector index...")
        helpers = load_jsonl(helpers_path)
        chunks = create_helper_chunks(helpers)
        
        index = VectorIndex(embedding_model)
        index.build(chunks)
        index.save(index_dir / "helpers_vector")
        print(f"  Indexed {index.size} helpers")
    else:
        print(f"Warning: {helpers_path} not found")
    
    # Build schemas index
    schemas_path = kb_dir / "schemas.jsonl"
    if schemas_path.exists():
        print(f"\nBuilding schemas vector index...")
        schemas = load_jsonl(schemas_path)
        chunks = create_schema_chunks(schemas)
        
        index = VectorIndex(embedding_model)
        index.build(chunks)
        index.save(index_dir / "schemas_vector")
        print(f"  Indexed {index.size} schemas")
    else:
        print(f"Warning: {schemas_path} not found")


def build_bm25_indexes(kb_dir: Path):
    """Build BM25 keyword indexes.
    
    Args:
        kb_dir: Knowledge base directory
    """
    from bm25_index import BM25Index
    from vector_index import create_helper_chunks, create_schema_chunks
    
    index_dir = kb_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Build helpers index
    helpers_path = kb_dir / "helpers.jsonl"
    if helpers_path.exists():
        print(f"\nBuilding helpers BM25 index...")
        helpers = load_jsonl(helpers_path)
        chunks = create_helper_chunks(helpers)
        
        index = BM25Index()
        index.build(chunks)
        index.save(index_dir / "helpers_bm25.pkl")
        print(f"  Indexed {index.size} helpers")
    else:
        print(f"Warning: {helpers_path} not found")
    
    # Build schemas index
    schemas_path = kb_dir / "schemas.jsonl"
    if schemas_path.exists():
        print(f"\nBuilding schemas BM25 index...")
        schemas = load_jsonl(schemas_path)
        chunks = create_schema_chunks(schemas)
        
        index = BM25Index()
        index.build(chunks)
        index.save(index_dir / "schemas_bm25.pkl")
        print(f"  Indexed {index.size} schemas")
    else:
        print(f"Warning: {schemas_path} not found")


def main():
    parser = argparse.ArgumentParser(
        description="Build vector and BM25 search indexes"
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=None,
        help="Knowledge base directory (default: data/knowledge_base/)"
    )
    parser.add_argument(
        "--vector-only",
        action="store_true",
        help="Only build vector indexes"
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Only build BM25 indexes"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for vector index"
    )
    
    args = parser.parse_args()
    
    repo_root = find_repo_root()
    kb_dir = args.kb_dir or (repo_root / "data" / "knowledge_base")
    
    if not kb_dir.exists():
        print(f"Error: Knowledge base directory not found: {kb_dir}")
        print("Run 'python scripts/build_kb.py' first to create the knowledge base.")
        sys.exit(1)
    
    print(f"Knowledge base: {kb_dir}")
    print(f"Embedding model: {args.embedding_model}")
    
    # Build indexes
    if not args.bm25_only:
        try:
            build_vector_indexes(kb_dir, args.embedding_model)
        except ImportError as e:
            print(f"\nError building vector indexes: {e}")
            print("Install dependencies with: pip install sentence-transformers faiss-cpu")
            if args.vector_only:
                sys.exit(1)
    
    if not args.vector_only:
        try:
            build_bm25_indexes(kb_dir)
        except ImportError as e:
            print(f"\nError building BM25 indexes: {e}")
            print("Install dependencies with: pip install rank-bm25")
            if args.bm25_only:
                sys.exit(1)
    
    print(f"\nIndexes saved to: {kb_dir / 'index'}")


if __name__ == "__main__":
    main()

