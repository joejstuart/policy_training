"""
Central path definitions for the policy-training repository.

All scripts should import paths from here instead of computing them locally.
This ensures consistency after repository reorganization.
"""

from pathlib import Path

# Repository root - this file is in src/
REPO_ROOT = Path(__file__).parent.parent.resolve()

# Source code directory
SRC_DIR = REPO_ROOT / "src"

# Scripts directory (data generation, utilities)
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Data directories
DATA_DIR = REPO_ROOT / "data"
ATTESTATION_DIR = DATA_DIR / "attestations"
TRAINING_DIR = DATA_DIR / "training"

# Training data subdirectories
ATTESTATION_TRAINING_DIR = TRAINING_DIR / "attestation"
POLICY_RULES_TRAINING_DIR = TRAINING_DIR / "policy_rules"
COMPILER_ERRORS_TRAINING_DIR = TRAINING_DIR / "compiler_errors"

# Policy directories (Rego source files)
POLICY_DIR = REPO_ROOT / "policy"
POLICY_RELEASE_DIR = POLICY_DIR / "release"
POLICY_LIB_DIR = POLICY_DIR / "lib"
RELEASE_LIB_DIR = POLICY_RELEASE_DIR / "lib"

# Documentation
DOCS_DIR = REPO_ROOT / "docs"

# Logs
LOGS_DIR = REPO_ROOT / "logs"

# Default output paths for training
DEFAULT_MODEL_OUTPUT_DIR = REPO_ROOT / "models"


def get_training_paths(dataset_type: str = "attestation") -> tuple:
    """Get train and eval paths for a dataset type.
    
    Args:
        dataset_type: One of "attestation", "policy_rules", "compiler_errors"
        
    Returns:
        Tuple of (train_path, eval_path)
    """
    type_to_dir = {
        "attestation": ATTESTATION_TRAINING_DIR,
        "policy_rules": POLICY_RULES_TRAINING_DIR,
        "compiler_errors": COMPILER_ERRORS_TRAINING_DIR,
    }
    
    if dataset_type not in type_to_dir:
        raise ValueError(f"Unknown dataset type: {dataset_type}. "
                        f"Valid types: {list(type_to_dir.keys())}")
    
    training_dir = type_to_dir[dataset_type]
    return (
        training_dir / "train.jsonl",
        training_dir / "eval.jsonl",
    )


def ensure_dirs_exist():
    """Create all required directories if they don't exist."""
    dirs = [
        SRC_DIR,
        SCRIPTS_DIR,
        DATA_DIR,
        ATTESTATION_DIR,
        TRAINING_DIR,
        ATTESTATION_TRAINING_DIR,
        POLICY_RULES_TRAINING_DIR,
        COMPILER_ERRORS_TRAINING_DIR,
        DOCS_DIR,
        LOGS_DIR,
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


# Verify critical directories exist on import
def _verify_critical_dirs():
    """Verify that critical directories exist."""
    critical = [POLICY_DIR, POLICY_LIB_DIR, POLICY_RELEASE_DIR]
    for d in critical:
        if not d.exists():
            import warnings
            warnings.warn(f"Critical directory missing: {d}")


_verify_critical_dirs()

