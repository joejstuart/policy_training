#!/usr/bin/env python3
"""
Test suite to validate that all scripts work after repository reorganization.

Run with: python -m pytest tests/test_migration.py -v
Or: python tests/test_migration.py
"""

import os
import sys
import importlib.util
import subprocess
from pathlib import Path

# Get repository root
REPO_ROOT = Path(__file__).parent.parent.resolve()

# Add src directory to path for imports
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))


def test_repo_structure_exists():
    """Verify the new directory structure exists."""
    required_dirs = [
        REPO_ROOT / "src",
        REPO_ROOT / "scripts",
        REPO_ROOT / "data" / "attestations",
        REPO_ROOT / "data" / "training",
        REPO_ROOT / "docs",
        REPO_ROOT / "policy",
        REPO_ROOT / "policy" / "lib",
        REPO_ROOT / "policy" / "release",
    ]
    
    missing = []
    for d in required_dirs:
        if not d.exists():
            missing.append(str(d))
    
    assert not missing, f"Missing directories: {missing}"
    print("✓ All required directories exist")


def test_paths_module_exists():
    """Verify the central paths module exists and works."""
    paths_file = REPO_ROOT / "src" / "paths.py"
    assert paths_file.exists(), f"Missing: {paths_file}"
    
    # Try to import it
    spec = importlib.util.spec_from_file_location("paths", paths_file)
    paths = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(paths)
    
    # Check required attributes
    required_attrs = [
        "REPO_ROOT",
        "SRC_DIR",
        "DATA_DIR",
        "ATTESTATION_DIR",
        "TRAINING_DIR",
        "POLICY_DIR",
        "POLICY_RELEASE_DIR",
        "POLICY_LIB_DIR",
    ]
    
    for attr in required_attrs:
        assert hasattr(paths, attr), f"paths.py missing attribute: {attr}"
        path_val = getattr(paths, attr)
        assert path_val.exists(), f"Path does not exist: {attr} = {path_val}"
    
    print("✓ paths.py module works correctly")


def test_src_modules_importable():
    """Test that all src modules can be imported."""
    src_modules = [
        "paths",
        "logging_setup",
        "context_extractor",
        "library_mapper",
        "library_indexer",
        "smart_context_builder",
        "rego_validator",
    ]
    
    errors = []
    for module_name in src_modules:
        module_file = REPO_ROOT / "src" / f"{module_name}.py"
        if not module_file.exists():
            errors.append(f"Missing: {module_name}.py")
            continue
            
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_file)
            module = importlib.util.module_from_spec(spec)
            # Add src to sys.modules for cross-imports
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as e:
            errors.append(f"{module_name}: {e}")
    
    assert not errors, f"Import errors:\n" + "\n".join(errors)
    print("✓ All src modules importable")


def test_scripts_syntax_valid():
    """Test that all scripts have valid Python syntax."""
    script_dirs = [
        REPO_ROOT / "scripts",
        REPO_ROOT / "src",
    ]
    
    errors = []
    for script_dir in script_dirs:
        if not script_dir.exists():
            continue
        for py_file in script_dir.glob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    source = f.read()
                compile(source, py_file, 'exec')
            except SyntaxError as e:
                errors.append(f"{py_file.name}: {e}")
    
    assert not errors, f"Syntax errors:\n" + "\n".join(errors)
    print("✓ All scripts have valid syntax")


def test_attestation_files_moved():
    """Verify attestation JSON files are in the correct location."""
    attestation_dir = REPO_ROOT / "data" / "attestations"
    
    # Check that attestation files exist in new location
    json_files = list(attestation_dir.glob("*.json"))
    assert len(json_files) > 0, f"No JSON files found in {attestation_dir}"
    
    # Check that no attestation files remain in root
    root_attestation_files = list(REPO_ROOT.glob("quay.io_*.json"))
    assert len(root_attestation_files) == 0, \
        f"Attestation files still in root: {[f.name for f in root_attestation_files[:5]]}..."
    
    print(f"✓ {len(json_files)} attestation files in correct location")


def test_training_data_exists():
    """Verify training data files exist."""
    training_dir = REPO_ROOT / "data" / "training"
    
    expected_files = [
        training_dir / "attestation" / "train.jsonl",
        training_dir / "attestation" / "eval.jsonl",
        training_dir / "policy_rules" / "train.jsonl",
        training_dir / "policy_rules" / "eval.jsonl",
    ]
    
    missing = []
    for f in expected_files:
        if not f.exists():
            missing.append(str(f.relative_to(REPO_ROOT)))
    
    assert not missing, f"Missing training files: {missing}"
    print("✓ Training data files exist")


def test_policy_directory_unchanged():
    """Verify policy directory structure is intact."""
    policy_dir = REPO_ROOT / "policy"
    
    required = [
        policy_dir / "lib",
        policy_dir / "release",
        policy_dir / "release" / "lib",
    ]
    
    for d in required:
        assert d.exists(), f"Missing policy dir: {d}"
    
    # Check some rego files exist
    rego_files = list(policy_dir.rglob("*.rego"))
    assert len(rego_files) > 10, f"Expected many .rego files, found {len(rego_files)}"
    
    print(f"✓ Policy directory intact ({len(rego_files)} .rego files)")


def test_library_mapper_finds_libs():
    """Test that LibraryMapper can find policy libraries."""
    # Import from src
    sys.path.insert(0, str(REPO_ROOT / "src"))
    
    from paths import REPO_ROOT as repo_root
    from library_mapper import LibraryMapper
    
    mapper = LibraryMapper(repo_root)
    mapper.build_mappings()
    
    # Should find data.lib mappings
    assert "data.lib" in mapper.import_to_dir, "Missing data.lib mapping"
    assert "data.lib.tekton" in mapper.import_to_dir, "Missing data.lib.tekton mapping"
    
    print("✓ LibraryMapper finds policy libraries")


def test_generate_dataset_script():
    """Test that generate_dataset.py can be loaded and has correct paths."""
    script_file = REPO_ROOT / "scripts" / "generate_dataset.py"
    assert script_file.exists(), f"Missing: {script_file}"
    
    # Check it has correct imports
    with open(script_file, 'r') as f:
        content = f.read()
    
    assert "from paths import" in content, \
        "generate_dataset.py should import from paths module"
    
    # Check syntax is valid
    compile(content, script_file, 'exec')
    
    print("✓ generate_dataset.py uses paths module")


def test_train_policy_script():
    """Test that train_policy.py can be loaded."""
    script_file = REPO_ROOT / "src" / "train_policy.py"
    assert script_file.exists(), f"Missing: {script_file}"
    
    # Check syntax
    with open(script_file, 'r') as f:
        source = f.read()
    compile(source, script_file, 'exec')
    
    print("✓ train_policy.py has valid syntax")


def test_infer_policy_script():
    """Test that infer_policy.py can be loaded."""
    script_file = REPO_ROOT / "src" / "infer_policy.py"
    assert script_file.exists(), f"Missing: {script_file}"
    
    # Check syntax
    with open(script_file, 'r') as f:
        source = f.read()
    compile(source, script_file, 'exec')
    
    print("✓ infer_policy.py has valid syntax")


def test_docs_moved():
    """Verify documentation files are in docs directory."""
    docs_dir = REPO_ROOT / "docs"
    
    expected = [
        "training.md",
        "rego_style_guide.md",
    ]
    
    for doc in expected:
        doc_file = docs_dir / doc
        assert doc_file.exists(), f"Missing doc: {doc_file}"
    
    # Check root is clean
    root_docs = ["training.md", "rego_style_guide.md"]
    for doc in root_docs:
        assert not (REPO_ROOT / doc).exists(), f"Doc still in root: {doc}"
    
    print("✓ Documentation moved to docs/")


def test_temp_files_cleaned():
    """Verify temporary files are removed."""
    temp_files = [
        REPO_ROOT / "t.rego",
        REPO_ROOT / "demo.md",
    ]
    
    existing = [f for f in temp_files if f.exists()]
    # Also check for rego_validate_*.rego
    validate_files = list(REPO_ROOT.glob("rego_validate_*.rego"))
    existing.extend(validate_files)
    
    assert not existing, f"Temp files still exist: {[f.name for f in existing]}"
    print("✓ Temporary files cleaned up")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_repo_structure_exists,
        test_paths_module_exists,
        test_src_modules_importable,
        test_scripts_syntax_valid,
        test_attestation_files_moved,
        test_training_data_exists,
        test_policy_directory_unchanged,
        test_library_mapper_finds_libs,
        test_generate_dataset_script,
        test_train_policy_script,
        test_infer_policy_script,
        test_docs_moved,
        test_temp_files_cleaned,
    ]
    
    passed = 0
    failed = 0
    
    print("\n" + "=" * 60)
    print("MIGRATION VALIDATION TESTS")
    print("=" * 60 + "\n")
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: Unexpected error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

