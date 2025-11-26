#!/usr/bin/env python3
"""Test script for the dynamic context building system."""

from pathlib import Path
from library_mapper import LibraryMapper
from library_indexer import LibraryIndexer
from smart_context_builder import SmartContextBuilder


def find_repo_root() -> Path:
    """Find repository root."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / "policy").exists():
            return current
        current = current.parent
    return Path.cwd()


def main():
    """Test the context building system."""
    repo_root = find_repo_root()
    
    print("=" * 60)
    print("Testing Dynamic Context Building System")
    print("=" * 60)
    print()
    
    # 1. Test LibraryMapper
    print("1. Building library mappings...")
    mapper = LibraryMapper(repo_root)
    mapper.build_mappings()
    print(f"   ✓ Mapped {len(mapper.import_to_dir)} import prefixes")
    print(f"   Example: data.lib.tekton -> {mapper.get_library_dir('data.lib.tekton')}")
    print()
    
    # 2. Test LibraryIndexer
    print("2. Indexing library functions...")
    indexer = LibraryIndexer(repo_root, mapper)
    indexer.index_all_libraries()
    print(f"   ✓ Indexed {len(indexer.index)} functions")
    
    # Show some examples
    example_funcs = list(indexer.index.keys())[:5]
    print(f"   Examples: {', '.join(example_funcs)}")
    print()
    
    # 3. Test SmartContextBuilder
    print("3. Testing context building...")
    builder = SmartContextBuilder(indexer, mapper, max_tokens=500)
    
    test_instructions = [
        "Write a rule that checks if all tasks in a PipelineRun succeeded",
        "Check if an image is from a trusted registry",
        "Verify that SBOM attestations are present",
    ]
    
    for instruction in test_instructions:
        print(f"\n   Instruction: {instruction}")
        context = builder.build_context(instruction)
        print(f"   Context length: {len(context)} chars (~{len(context)//4} tokens)")
        print(f"   Context preview:")
        print("   " + "-" * 56)
        for line in context.split('\n')[:15]:
            print(f"   {line}")
        if len(context.split('\n')) > 15:
            print(f"   ... ({len(context.split('\n')) - 15} more lines)")
        print("   " + "-" * 56)
    
    print()
    print("=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

