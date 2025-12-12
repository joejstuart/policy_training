"""Validate plans against Knowledge Base.

Per architecture spec:
- Validation gate between Planner and Codegen
- Reject plans if helper/schema doesn't exist
- Provide repair suggestions with alternatives
- This is tooling-based validation, not LLM self-check
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

try:
    from knowledge_base import KnowledgeBase
except ImportError:
    from .knowledge_base import KnowledgeBase


@dataclass
class ValidationError:
    """A single validation error."""
    error_type: str  # "helper_not_found", "schema_not_found", "import_error"
    message: str
    field: str  # Which plan field caused the error
    value: str  # The problematic value
    suggestions: List[str] = field(default_factory=list)  # Alternative suggestions


@dataclass
class ValidationResult:
    """Result of plan validation.
    
    Per architecture spec:
    - valid: True if plan passes all checks
    - errors: List of validation errors
    - warnings: Non-fatal issues
    """
    valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid": self.valid,
            "errors": [
                {
                    "error_type": e.error_type,
                    "message": e.message,
                    "field": e.field,
                    "value": e.value,
                    "suggestions": e.suggestions,
                }
                for e in self.errors
            ],
            "warnings": self.warnings,
        }
    
    def error_summary(self) -> str:
        """Human-readable error summary for repair prompts."""
        if not self.errors:
            return "No errors"
        
        lines = ["Validation errors:"]
        for e in self.errors:
            lines.append(f"  - {e.message}")
            if e.suggestions:
                lines.append(f"    Suggestions: {', '.join(e.suggestions[:3])}")
        
        return "\n".join(lines)


class PlanValidator:
    """Validates plans against the Knowledge Base.
    
    Per architecture spec:
    - Check helper existence
    - Check schema existence for attestation type
    - Check importability (future)
    - Check type compatibility (future, best-effort)
    """
    
    def __init__(self, kb: KnowledgeBase):
        """Initialize validator with KB.
        
        Args:
            kb: Knowledge Base to validate against
        """
        self.kb = kb
    
    def validate(self, plan: Dict[str, Any]) -> ValidationResult:
        """Validate a plan against the KB.
        
        Per architecture spec, reject plans if:
        - Helper/schema doesn't exist
        - Helper not importable for target package
        - Plan references unselected modules
        - Schema incompatible with selected attestation type(s)
        
        Args:
            plan: Plan dictionary with helpers, schemas, attestation_type, etc.
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        
        # Get plan fields
        attestation_type = plan.get("attestation_type", "")
        helpers = plan.get("helpers", [])
        schemas = plan.get("schemas", [])
        
        # Validate helpers
        for helper_entry in helpers:
            # Handle both formats: string ID or {"id": ..., "reason": ...}
            if isinstance(helper_entry, dict):
                helper_id = helper_entry.get("id", "")
            else:
                helper_id = str(helper_entry)
            
            if not helper_id:
                continue
            
            if not self.kb.helper_exists(helper_id):
                # Find suggestions
                suggestions = self._find_similar_helpers(helper_id)
                
                errors.append(ValidationError(
                    error_type="helper_not_found",
                    message=f"Helper '{helper_id}' not found in knowledge base",
                    field="helpers",
                    value=helper_id,
                    suggestions=suggestions,
                ))
        
        # Validate schemas
        for schema_entry in schemas:
            # Handle both formats: string ID or {"id": ..., ...}
            if isinstance(schema_entry, dict):
                schema_id = schema_entry.get("id", schema_entry.get("schema_id", ""))
            else:
                schema_id = str(schema_entry)
            
            if not schema_id:
                continue
            
            if not self.kb.schema_exists(schema_id, attestation_type if attestation_type else None):
                # Check if schema exists but for wrong type
                if self.kb.schema_exists(schema_id):
                    schema = self.kb.get_schema(schema_id)
                    errors.append(ValidationError(
                        error_type="schema_type_mismatch",
                        message=f"Schema '{schema_id}' exists but is for '{schema.attestation_type}', not '{attestation_type}'",
                        field="schemas",
                        value=schema_id,
                        suggestions=[f"Use attestation_type='{schema.attestation_type}' or choose different schema"],
                    ))
                else:
                    # Find suggestions
                    suggestions = self._find_similar_schemas(schema_id, attestation_type)
                    
                    errors.append(ValidationError(
                        error_type="schema_not_found",
                        message=f"Schema '{schema_id}' not found in knowledge base",
                        field="schemas",
                        value=schema_id,
                        suggestions=suggestions,
                    ))
        
        # Check new_helpers format (if present)
        new_helpers = plan.get("new_helpers", [])
        for new_helper in new_helpers:
            if isinstance(new_helper, dict):
                if "name" not in new_helper:
                    warnings.append(f"new_helper missing 'name' field: {new_helper}")
                if "implementation" not in new_helper and "signature" not in new_helper:
                    warnings.append(f"new_helper '{new_helper.get('name', '?')}' missing implementation/signature")
        
        # Validate attestation_type is known (warning only)
        if attestation_type:
            known_types = set(s.attestation_type for s in self.kb.schemas.values())
            if attestation_type not in known_types and known_types:
                warnings.append(
                    f"Attestation type '{attestation_type}' not found in KB. "
                    f"Known types: {', '.join(sorted(known_types))}"
                )
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
    
    def _find_similar_helpers(self, helper_id: str, max_suggestions: int = 3) -> List[str]:
        """Find similar helper names for suggestions.
        
        Args:
            helper_id: The helper ID that wasn't found
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of similar helper IDs
        """
        # Extract the function name part
        name = helper_id.split(".")[-1] if "." in helper_id else helper_id
        name_lower = name.lower()
        
        suggestions = []
        
        for card in self.kb.helper_cards.values():
            card_name = card.name.lower()
            
            # Exact substring match
            if name_lower in card_name or card_name in name_lower:
                suggestions.append(card.id)
                continue
            
            # Word overlap
            name_words = set(name_lower.replace("_", " ").split())
            card_words = set(card_name.replace("_", " ").split())
            if name_words & card_words:
                suggestions.append(card.id)
        
        # Sort by similarity (shorter edit distance = more similar)
        suggestions.sort(key=lambda x: abs(len(x) - len(helper_id)))
        
        return suggestions[:max_suggestions]
    
    def _find_similar_schemas(
        self, 
        schema_id: str, 
        attestation_type: Optional[str] = None,
        max_suggestions: int = 3
    ) -> List[str]:
        """Find similar schema IDs for suggestions.
        
        Args:
            schema_id: The schema ID that wasn't found
            attestation_type: Optional attestation type to filter by
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of similar schema IDs
        """
        id_lower = schema_id.lower()
        suggestions = []
        
        for sid, schema in self.kb.schemas.items():
            # Filter by attestation type if specified
            if attestation_type and schema.attestation_type != attestation_type:
                continue
            
            sid_lower = sid.lower()
            path_lower = schema.canonical_path.lower()
            
            # Check for keyword overlap
            keywords = id_lower.replace("_", " ").split()
            for keyword in keywords:
                if keyword in sid_lower or keyword in path_lower:
                    suggestions.append(sid)
                    break
        
        return suggestions[:max_suggestions]
    
    def suggest_alternatives(self, error: ValidationError) -> str:
        """Generate a human-readable suggestion for an error.
        
        Per architecture spec: repair loop provides explicit alternatives.
        
        Args:
            error: Validation error
            
        Returns:
            Suggestion text for repair prompt
        """
        if not error.suggestions:
            return f"No alternatives found for '{error.value}'"
        
        if error.error_type == "helper_not_found":
            return (
                f"Helper '{error.value}' not found. "
                f"Available alternatives: {', '.join(error.suggestions)}"
            )
        elif error.error_type == "schema_not_found":
            return (
                f"Schema '{error.value}' not found. "
                f"Available alternatives: {', '.join(error.suggestions)}"
            )
        elif error.error_type == "schema_type_mismatch":
            return error.message
        
        return f"Error with '{error.value}': {error.message}"
    
    def generate_repair_prompt(self, plan: Dict[str, Any], result: ValidationResult) -> str:
        """Generate a prompt for the planner to repair the plan.
        
        Per architecture spec: repair loop reruns planner with explicit alternatives.
        
        Args:
            plan: The invalid plan
            result: Validation result with errors
            
        Returns:
            Repair prompt text
        """
        lines = ["Your plan has validation errors. Please revise:\n"]
        
        for error in result.errors:
            lines.append(f"- {error.message}")
            suggestion = self.suggest_alternatives(error)
            lines.append(f"  → {suggestion}")
        
        if result.warnings:
            lines.append("\nWarnings (non-fatal):")
            for warning in result.warnings:
                lines.append(f"- {warning}")
        
        lines.append("\nPlease update your plan to use only helpers and schemas that exist in the knowledge base.")
        
        return "\n".join(lines)

