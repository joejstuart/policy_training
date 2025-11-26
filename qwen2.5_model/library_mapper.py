"""Maps import prefixes to directory patterns and vice versa."""

from pathlib import Path
from typing import Dict, Optional, List


class LibraryMapper:
    """Maps import prefixes to directories and vice versa."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self.lib_dir = self.repo_root / "policy" / "lib"
        self.release_lib_dir = self.repo_root / "policy" / "release" / "lib"
        self.import_to_dir: Dict[str, Path] = {}
        self.dir_to_import: Dict[Path, str] = {}
    
    def build_mappings(self):
        """Scan library directories and build import prefix ↔ directory mappings."""
        # Map data.lib.* to policy/lib/**
        if self.lib_dir.exists():
            self._map_lib_directory(self.lib_dir, "data.lib")
        
        # Map data.release.lib.* to policy/release/lib/**
        if self.release_lib_dir.exists():
            self._map_lib_directory(self.release_lib_dir, "data.release.lib")
    
    def _map_lib_directory(self, lib_dir: Path, base_import: str):
        """Recursively map directories to import prefixes."""
        # Map the base directory
        self.import_to_dir[base_import] = lib_dir
        self.dir_to_import[lib_dir] = base_import
        
        # Map subdirectories
        for subdir in lib_dir.iterdir():
            if subdir.is_dir() and not subdir.name.startswith('.'):
                import_prefix = f"{base_import}.{subdir.name}"
                self.import_to_dir[import_prefix] = subdir
                self.dir_to_import[subdir] = import_prefix
                
                # Recursively map nested subdirectories
                self._map_lib_directory(subdir, import_prefix)
    
    def get_library_dir(self, import_path: str) -> Optional[Path]:
        """Get directory for an import prefix.
        
        Args:
            import_path: Import prefix like "data.lib.tekton"
            
        Returns:
            Path to directory, or None if not found
        """
        return self.import_to_dir.get(import_path)
    
    def get_import_prefix(self, file_path: Path) -> Optional[str]:
        """Get import prefix for a library file.
        
        Args:
            file_path: Path to a .rego file
            
        Returns:
            Import prefix like "data.lib.tekton", or None if not found
        """
        file_path = Path(file_path).resolve()
        
        # Check if file is in a mapped directory
        for dir_path, import_prefix in self.dir_to_import.items():
            try:
                file_path.relative_to(dir_path)
                return import_prefix
            except ValueError:
                continue
        
        return None
    
    def get_all_library_files(self, import_path: str) -> List[Path]:
        """Get all .rego files for an import prefix (excluding test files).
        
        Args:
            import_path: Import prefix like "data.lib.tekton"
            
        Returns:
            List of .rego file paths (excluding *_test.rego files)
        """
        lib_dir = self.get_library_dir(import_path)
        if not lib_dir or not lib_dir.exists():
            return []
        
        rego_files = []
        for file_path in lib_dir.rglob("*.rego"):
            # Skip test files
            if not file_path.name.endswith("_test.rego"):
                rego_files.append(file_path)
        
        return sorted(rego_files)
