"""Knowledge Base manifest and versioning.

Tracks KB version (git ref, build time) for reproducibility.
Per architecture spec: outputs must annotate which KB version was used.
"""

import datetime
import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class KBManifest:
    """Knowledge Base manifest with version information.
    
    Per architecture spec:
    - git_ref: tie KB to specific commit
    - built_at: when KB was built
    - policy_lib_hash: hash of policy/lib directory
    - Counts for helpers, schemas
    """
    
    git_ref: str
    built_at: str
    policy_lib_hash: str
    helper_count: int
    schema_count: int
    attestation_types: List[str] = field(default_factory=list)
    attestation_files_count: int = 0
    version: str = "1.0"
    
    @classmethod
    def create(cls, repo_root: Path, helper_count: int = 0, schema_count: int = 0) -> "KBManifest":
        """Create manifest from current repo state.
        
        Args:
            repo_root: Root of the repository
            helper_count: Number of helpers indexed
            schema_count: Number of schemas extracted
            
        Returns:
            KBManifest with current state
        """
        git_ref = cls._get_git_ref(repo_root)
        policy_lib_hash = cls._hash_directory(repo_root / "policy" / "lib")
        built_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Count attestation files
        att_dir = repo_root / "data" / "attestations"
        att_count = len(list(att_dir.glob("*.json"))) if att_dir.exists() else 0
        
        return cls(
            git_ref=git_ref,
            built_at=built_at,
            policy_lib_hash=policy_lib_hash,
            helper_count=helper_count,
            schema_count=schema_count,
            attestation_files_count=att_count,
        )
    
    @staticmethod
    def _get_git_ref(repo_root: Path) -> str:
        """Get current git commit ref.
        
        Returns:
            Git commit SHA or "unknown" if not in a git repo
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]  # Short SHA
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        
        return "unknown"
    
    @staticmethod
    def _hash_directory(directory: Path) -> str:
        """Create a hash of all .rego files in directory.
        
        Args:
            directory: Directory to hash
            
        Returns:
            SHA-256 hash of concatenated file contents
        """
        if not directory.exists():
            return "empty"
        
        hasher = hashlib.sha256()
        
        # Sort files for consistent ordering
        rego_files = sorted(directory.rglob("*.rego"))
        
        for file_path in rego_files:
            # Skip test files
            if "_test.rego" in file_path.name:
                continue
            
            try:
                content = file_path.read_bytes()
                # Include relative path in hash for structure sensitivity
                rel_path = file_path.relative_to(directory)
                hasher.update(str(rel_path).encode())
                hasher.update(content)
            except Exception:
                pass
        
        return hasher.hexdigest()[:16]
    
    def save(self, path: Path):
        """Save manifest to YAML file.
        
        Args:
            path: Path to save manifest
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": self.version,
            "git_ref": self.git_ref,
            "built_at": self.built_at,
            "policy_lib_hash": self.policy_lib_hash,
            "helper_count": self.helper_count,
            "schema_count": self.schema_count,
            "attestation_types": self.attestation_types,
            "attestation_files_count": self.attestation_files_count,
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    @classmethod
    def load(cls, path: Path) -> "KBManifest":
        """Load manifest from YAML file.
        
        Args:
            path: Path to manifest file
            
        Returns:
            KBManifest loaded from file
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        return cls(
            git_ref=data.get("git_ref", "unknown"),
            built_at=data.get("built_at", ""),
            policy_lib_hash=data.get("policy_lib_hash", ""),
            helper_count=data.get("helper_count", 0),
            schema_count=data.get("schema_count", 0),
            attestation_types=data.get("attestation_types", []),
            attestation_files_count=data.get("attestation_files_count", 0),
            version=data.get("version", "1.0"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "git_ref": self.git_ref,
            "built_at": self.built_at,
            "policy_lib_hash": self.policy_lib_hash,
            "helper_count": self.helper_count,
            "schema_count": self.schema_count,
            "attestation_types": self.attestation_types,
            "attestation_files_count": self.attestation_files_count,
        }
    
    def summary(self) -> str:
        """Human-readable summary of manifest."""
        return (
            f"KB Version: {self.version}\n"
            f"Git Ref: {self.git_ref}\n"
            f"Built At: {self.built_at}\n"
            f"Helpers: {self.helper_count}\n"
            f"Schemas: {self.schema_count}\n"
            f"Attestation Files: {self.attestation_files_count}"
        )

