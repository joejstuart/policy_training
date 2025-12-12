"""Unified Knowledge Base access for helpers and schemas.

Provides a single interface to load and query the KB.
Per architecture spec: KB is the source of truth for all helpers, schemas, and patterns.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from kb_manifest import KBManifest
    from schema_extractor import SchemaField, SchemaExtractor
except ImportError:
    from .kb_manifest import KBManifest
    from .schema_extractor import SchemaField, SchemaExtractor


@dataclass
class HelperCard:
    """Compact helper representation for retrieval (~100-150 tokens).
    
    Per architecture spec: two-tier chunks - card is the compact form.
    """
    
    id: str                     # Fully qualified: e.g., "lib.pipelinerun_attestations"
    name: str                   # Function name only
    module_path: str            # e.g., "data.lib"
    signature: str              # e.g., "task_ref(task)"
    description: str            # Short description
    use_when: List[str]         # When to use this helper
    source_file: str            # e.g., "policy/lib/tekton/tekton.rego"
    source_lines: Tuple[int, int]  # (start_line, end_line)
    
    def to_text(self) -> str:
        """Convert to searchable text for embedding."""
        parts = [
            f"Helper: {self.id}",
            f"Signature: {self.signature}",
            f"Description: {self.description}",
        ]
        if self.use_when:
            parts.append(f"Use when: {', '.join(self.use_when)}")
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "module_path": self.module_path,
            "signature": self.signature,
            "description": self.description,
            "use_when": self.use_when,
            "source_file": self.source_file,
            "source_lines": list(self.source_lines),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HelperCard":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            module_path=data["module_path"],
            signature=data["signature"],
            description=data.get("description", ""),
            use_when=data.get("use_when", []),
            source_file=data.get("source_file", ""),
            source_lines=tuple(data.get("source_lines", [0, 0])),
        )


@dataclass
class HelperFull:
    """Full helper representation with body (~400-600 tokens).
    
    Per architecture spec: two-tier chunks - full includes body and examples.
    """
    
    id: str
    name: str
    module_path: str
    signature: str
    description: str
    use_when: List[str]
    source_file: str
    source_lines: Tuple[int, int]
    body: str                   # Full function source code
    usage_examples: List[str]   # Real usage from rules/tests
    imports_required: List[str] # Imports needed to use this helper
    
    def to_card(self) -> HelperCard:
        """Convert to compact card form."""
        return HelperCard(
            id=self.id,
            name=self.name,
            module_path=self.module_path,
            signature=self.signature,
            description=self.description,
            use_when=self.use_when,
            source_file=self.source_file,
            source_lines=self.source_lines,
        )
    
    def to_text(self) -> str:
        """Convert to full text for embedding/context."""
        parts = [
            f"Helper: {self.id}",
            f"Signature: {self.signature}",
            f"Description: {self.description}",
            f"Source: {self.source_file}:{self.source_lines[0]}-{self.source_lines[1]}",
            "",
            "Body:",
            self.body,
        ]
        if self.usage_examples:
            parts.append("")
            parts.append("Usage examples:")
            for ex in self.usage_examples[:2]:
                parts.append(f"  {ex}")
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "module_path": self.module_path,
            "signature": self.signature,
            "description": self.description,
            "use_when": self.use_when,
            "source_file": self.source_file,
            "source_lines": list(self.source_lines),
            "body": self.body,
            "usage_examples": self.usage_examples,
            "imports_required": self.imports_required,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HelperFull":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            module_path=data["module_path"],
            signature=data["signature"],
            description=data.get("description", ""),
            use_when=data.get("use_when", []),
            source_file=data.get("source_file", ""),
            source_lines=tuple(data.get("source_lines", [0, 0])),
            body=data.get("body", ""),
            usage_examples=data.get("usage_examples", []),
            imports_required=data.get("imports_required", []),
        )


class KnowledgeBase:
    """Unified access to helpers and schemas.
    
    Per architecture spec:
    - Single source of truth
    - Versioned via manifest
    - Provides existence checks for validation
    """
    
    def __init__(self, kb_dir: Optional[Path] = None):
        """Initialize knowledge base.
        
        Args:
            kb_dir: Path to knowledge base directory. If None, creates empty KB.
        """
        self.kb_dir = Path(kb_dir) if kb_dir else None
        self.manifest: Optional[KBManifest] = None
        
        # Helper storage
        self.helper_cards: Dict[str, HelperCard] = {}  # id -> HelperCard
        self.helper_fulls: Dict[str, HelperFull] = {}  # id -> HelperFull
        
        # Schema storage
        self.schemas: Dict[str, SchemaField] = {}  # schema_id -> SchemaField
        
        # Indexes for lookup
        self._helper_by_name: Dict[str, str] = {}  # name -> id
        self._schema_by_path: Dict[str, str] = {}  # canonical_path -> schema_id
        
        if kb_dir and Path(kb_dir).exists():
            self.load()
    
    def load(self):
        """Load KB from directory."""
        if not self.kb_dir or not self.kb_dir.exists():
            return
        
        # Load manifest
        manifest_path = self.kb_dir / "manifest.yaml"
        if manifest_path.exists():
            self.manifest = KBManifest.load(manifest_path)
        
        # Load helper cards
        cards_path = self.kb_dir / "helpers.jsonl"
        if cards_path.exists():
            self._load_helper_cards(cards_path)
        
        # Load helper fulls
        fulls_path = self.kb_dir / "helpers_full.jsonl"
        if fulls_path.exists():
            self._load_helper_fulls(fulls_path)
        
        # Load schemas
        schemas_path = self.kb_dir / "schemas.jsonl"
        if schemas_path.exists():
            self.schemas = SchemaExtractor.load_schemas(schemas_path)
            self._build_schema_index()
    
    def _load_helper_cards(self, path: Path):
        """Load helper cards from JSONL."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    card = HelperCard.from_dict(data)
                    self.helper_cards[card.id] = card
                    self._helper_by_name[card.name] = card.id
    
    def _load_helper_fulls(self, path: Path):
        """Load full helpers from JSONL."""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    full = HelperFull.from_dict(data)
                    self.helper_fulls[full.id] = full
    
    def _build_schema_index(self):
        """Build index from canonical path to schema_id."""
        for schema_id, schema in self.schemas.items():
            self._schema_by_path[schema.canonical_path] = schema_id
    
    # -------------------------------------------------------------------------
    # Helper access methods
    # -------------------------------------------------------------------------
    
    def helper_exists(self, helper_id: str) -> bool:
        """Check if a helper exists in the KB.
        
        Per architecture spec: validation gate uses this to reject invalid plans.
        
        Args:
            helper_id: Helper ID (e.g., "lib.pipelinerun_attestations" or just "pipelinerun_attestations")
            
        Returns:
            True if helper exists
        """
        # Try direct ID match
        if helper_id in self.helper_cards:
            return True
        
        # Try name-only match
        name = helper_id.split(".")[-1] if "." in helper_id else helper_id
        return name in self._helper_by_name
    
    def get_helper_card(self, helper_id: str) -> Optional[HelperCard]:
        """Get helper card by ID.
        
        Args:
            helper_id: Helper ID or name
            
        Returns:
            HelperCard or None
        """
        if helper_id in self.helper_cards:
            return self.helper_cards[helper_id]
        
        # Try name lookup
        name = helper_id.split(".")[-1] if "." in helper_id else helper_id
        if name in self._helper_by_name:
            full_id = self._helper_by_name[name]
            return self.helper_cards.get(full_id)
        
        return None
    
    def get_helper_full(self, helper_id: str) -> Optional[HelperFull]:
        """Get full helper by ID.
        
        Args:
            helper_id: Helper ID or name
            
        Returns:
            HelperFull or None
        """
        if helper_id in self.helper_fulls:
            return self.helper_fulls[helper_id]
        
        # Try name lookup
        name = helper_id.split(".")[-1] if "." in helper_id else helper_id
        if name in self._helper_by_name:
            full_id = self._helper_by_name[name]
            return self.helper_fulls.get(full_id)
        
        return None
    
    def get_all_helper_cards(self) -> List[HelperCard]:
        """Get all helper cards."""
        return list(self.helper_cards.values())
    
    def get_helper_ids(self) -> Set[str]:
        """Get all helper IDs."""
        return set(self.helper_cards.keys())
    
    # -------------------------------------------------------------------------
    # Schema access methods
    # -------------------------------------------------------------------------
    
    def schema_exists(self, schema_ref: str, attestation_type: Optional[str] = None) -> bool:
        """Check if a schema exists in the KB.
        
        Per architecture spec: validation gate uses this.
        
        Args:
            schema_ref: Schema ID or canonical path
            attestation_type: Optional attestation type to filter by
            
        Returns:
            True if schema exists
        """
        # Try direct ID match
        if schema_ref in self.schemas:
            schema = self.schemas[schema_ref]
            if attestation_type and schema.attestation_type != attestation_type:
                return False
            return True
        
        # Try path match
        if schema_ref in self._schema_by_path:
            schema_id = self._schema_by_path[schema_ref]
            schema = self.schemas[schema_id]
            if attestation_type and schema.attestation_type != attestation_type:
                return False
            return True
        
        return False
    
    def get_schema(self, schema_ref: str) -> Optional[SchemaField]:
        """Get schema by ID or path.
        
        Args:
            schema_ref: Schema ID or canonical path
            
        Returns:
            SchemaField or None
        """
        if schema_ref in self.schemas:
            return self.schemas[schema_ref]
        
        if schema_ref in self._schema_by_path:
            schema_id = self._schema_by_path[schema_ref]
            return self.schemas.get(schema_id)
        
        return None
    
    def get_all_schemas(self) -> List[SchemaField]:
        """Get all schemas."""
        return list(self.schemas.values())
    
    def get_schemas_for_type(self, attestation_type: str) -> List[SchemaField]:
        """Get all schemas for an attestation type.
        
        Args:
            attestation_type: e.g., "slsa_provenance_v02"
            
        Returns:
            List of matching schemas
        """
        return [s for s in self.schemas.values() if s.attestation_type == attestation_type]
    
    # -------------------------------------------------------------------------
    # Save methods
    # -------------------------------------------------------------------------
    
    def save(self, kb_dir: Path):
        """Save KB to directory.
        
        Args:
            kb_dir: Directory to save to
        """
        kb_dir.mkdir(parents=True, exist_ok=True)
        
        # Save helper cards
        cards_path = kb_dir / "helpers.jsonl"
        with open(cards_path, 'w', encoding='utf-8') as f:
            for card in self.helper_cards.values():
                f.write(json.dumps(card.to_dict()) + "\n")
        
        # Save helper fulls
        fulls_path = kb_dir / "helpers_full.jsonl"
        with open(fulls_path, 'w', encoding='utf-8') as f:
            for full in self.helper_fulls.values():
                f.write(json.dumps(full.to_dict()) + "\n")
        
        # Save schemas
        schemas_path = kb_dir / "schemas.jsonl"
        with open(schemas_path, 'w', encoding='utf-8') as f:
            for schema in self.schemas.values():
                f.write(json.dumps(schema.to_dict()) + "\n")
        
        # Save manifest if exists
        if self.manifest:
            self.manifest.save(kb_dir / "manifest.yaml")
    
    def add_helper(self, helper: HelperFull):
        """Add a helper to the KB.
        
        Args:
            helper: HelperFull to add
        """
        self.helper_fulls[helper.id] = helper
        self.helper_cards[helper.id] = helper.to_card()
        self._helper_by_name[helper.name] = helper.id
    
    def add_schema(self, schema: SchemaField):
        """Add a schema to the KB.
        
        Args:
            schema: SchemaField to add
        """
        self.schemas[schema.schema_id] = schema
        self._schema_by_path[schema.canonical_path] = schema.schema_id
    
    # -------------------------------------------------------------------------
    # Utility methods
    # -------------------------------------------------------------------------
    
    def stats(self) -> Dict[str, Any]:
        """Get KB statistics."""
        return {
            "helper_count": len(self.helper_cards),
            "schema_count": len(self.schemas),
            "attestation_types": list(set(s.attestation_type for s in self.schemas.values())),
            "git_ref": self.manifest.git_ref if self.manifest else "unknown",
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        stats = self.stats()
        return (
            f"Knowledge Base Summary:\n"
            f"  Helpers: {stats['helper_count']}\n"
            f"  Schemas: {stats['schema_count']}\n"
            f"  Attestation Types: {', '.join(stats['attestation_types']) or 'none'}\n"
            f"  Git Ref: {stats['git_ref']}"
        )

