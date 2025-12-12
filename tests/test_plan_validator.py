"""Tests for plan_validator module.

Tests follow the architecture spec requirements:
- Validation gate rejects invalid plans
- Provides suggestions for repair
- Checks helper and schema existence
"""

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from plan_validator import PlanValidator, ValidationResult, ValidationError
from knowledge_base import KnowledgeBase, HelperFull
from schema_extractor import SchemaField


@pytest.fixture
def sample_kb():
    """Create a sample KB for testing."""
    kb = KnowledgeBase()
    
    # Add helpers
    kb.add_helper(HelperFull(
        id="lib.tekton.task_ref",
        name="task_ref",
        module_path="data.lib.tekton",
        signature="task_ref(task)",
        description="Returns task reference info",
        use_when=["bundle pinning"],
        source_file="policy/lib/tekton/tekton.rego",
        source_lines=(142, 158),
        body="task_ref(task) := {...}",
        usage_examples=[],
        imports_required=["data.lib.tekton"],
    ))
    
    kb.add_helper(HelperFull(
        id="lib.tekton.tasks",
        name="tasks",
        module_path="data.lib.tekton",
        signature="tasks(attestation)",
        description="Returns all tasks",
        use_when=["iterating tasks"],
        source_file="policy/lib/tekton/tekton.rego",
        source_lines=(50, 70),
        body="tasks(att) := {...}",
        usage_examples=[],
        imports_required=["data.lib.tekton"],
    ))
    
    kb.add_helper(HelperFull(
        id="lib.pipelinerun_attestations",
        name="pipelinerun_attestations",
        module_path="data.lib",
        signature="pipelinerun_attestations",
        description="Returns attestations",
        use_when=["iterating attestations"],
        source_file="policy/lib/lib.rego",
        source_lines=(100, 110),
        body="pipelinerun_attestations := {...}",
        usage_examples=[],
        imports_required=["data.lib"],
    ))
    
    kb.add_helper(HelperFull(
        id="lib.result_helper",
        name="result_helper",
        module_path="data.lib",
        signature="result_helper(chain, args)",
        description="Creates result object",
        use_when=["error messages"],
        source_file="policy/lib/result_helper.rego",
        source_lines=(10, 25),
        body="result_helper(chain, args) := {...}",
        usage_examples=[],
        imports_required=["data.lib"],
    ))
    
    # Add schemas
    kb.add_schema(SchemaField(
        schema_id="slsa_task_bundle",
        canonical_path="$.predicate.buildConfig.tasks[*].ref.bundle",
        attestation_type="slsa_provenance_v02",
        field_type="string",
        description="OCI bundle reference",
        use_when=["bundle pinning"],
        example_value="quay.io/org/task@sha256:abc",
        source_file="test.json",
    ))
    
    kb.add_schema(SchemaField(
        schema_id="slsa_task_name",
        canonical_path="$.predicate.buildConfig.tasks[*].name",
        attestation_type="slsa_provenance_v02",
        field_type="string",
        description="Task name",
        use_when=["identifying tasks"],
        example_value="build-task",
        source_file="test.json",
    ))
    
    kb.add_schema(SchemaField(
        schema_id="spdx_package_license",
        canonical_path="$.packages[*].licenseConcluded",
        attestation_type="spdx_sbom",
        field_type="string",
        description="Package license",
        use_when=["license checks"],
        example_value="Apache-2.0",
        source_file="test.json",
    ))
    
    return kb


@pytest.fixture
def validator(sample_kb):
    """Create a validator with sample KB."""
    return PlanValidator(sample_kb)


class TestPlanValidatorBasics:
    """Basic validation tests."""
    
    def test_valid_plan_passes(self, validator):
        """Test that a valid plan passes validation."""
        plan = {
            "package": "task_bundle_pinning",
            "attestation_type": "slsa_provenance_v02",
            "helpers": [
                {"id": "lib.tekton.task_ref", "reason": "Check pinning"},
                {"id": "lib.tekton.tasks", "reason": "Iterate tasks"},
            ],
            "schemas": ["slsa_task_bundle", "slsa_task_name"],
        }
        
        result = validator.validate(plan)
        
        assert result.valid is True
        assert len(result.errors) == 0
    
    def test_valid_plan_with_string_helpers(self, validator):
        """Test validation with helpers as strings (not dicts)."""
        plan = {
            "attestation_type": "slsa_provenance_v02",
            "helpers": ["lib.tekton.task_ref", "lib.tekton.tasks"],
            "schemas": ["slsa_task_bundle"],
        }
        
        result = validator.validate(plan)
        
        assert result.valid is True
    
    def test_helper_not_found_fails(self, validator):
        """Test that non-existent helper fails validation."""
        plan = {
            "helpers": [
                {"id": "lib.tekton.nonexistent_helper"},
            ],
            "schemas": [],
        }
        
        result = validator.validate(plan)
        
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "helper_not_found"
        assert "nonexistent_helper" in result.errors[0].message
    
    def test_schema_not_found_fails(self, validator):
        """Test that non-existent schema fails validation."""
        plan = {
            "helpers": [],
            "schemas": ["nonexistent_schema"],
        }
        
        result = validator.validate(plan)
        
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "schema_not_found"


class TestSchemaTypeValidation:
    """Tests for schema/attestation type matching."""
    
    def test_schema_wrong_attestation_type(self, validator):
        """Test that schema for wrong attestation type fails."""
        plan = {
            "attestation_type": "slsa_provenance_v02",
            "helpers": [],
            "schemas": ["spdx_package_license"],  # This is for spdx_sbom, not slsa
        }
        
        result = validator.validate(plan)
        
        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "schema_type_mismatch"
        assert "spdx_sbom" in result.errors[0].message
    
    def test_schema_correct_attestation_type(self, validator):
        """Test that schema for correct attestation type passes."""
        plan = {
            "attestation_type": "spdx_sbom",
            "helpers": [],
            "schemas": ["spdx_package_license"],
        }
        
        result = validator.validate(plan)
        
        assert result.valid is True


class TestSuggestions:
    """Tests for repair suggestions."""
    
    def test_helper_not_found_suggests_alternatives(self, validator):
        """Test that helper not found error includes suggestions."""
        plan = {
            "helpers": [{"id": "lib.tekton.task"}],  # Close to "tasks"
            "schemas": [],
        }
        
        result = validator.validate(plan)
        
        assert result.valid is False
        assert len(result.errors[0].suggestions) > 0
        # Should suggest lib.tekton.tasks
        assert any("tasks" in s for s in result.errors[0].suggestions)
    
    def test_schema_not_found_suggests_alternatives(self, validator):
        """Test that schema not found error includes suggestions."""
        plan = {
            "attestation_type": "slsa_provenance_v02",
            "helpers": [],
            "schemas": ["slsa_bundle"],  # Close to "slsa_task_bundle"
        }
        
        result = validator.validate(plan)
        
        assert result.valid is False
        suggestions = result.errors[0].suggestions
        # Should suggest slsa_task_bundle
        assert len(suggestions) > 0


class TestRepairPrompt:
    """Tests for repair prompt generation."""
    
    def test_generate_repair_prompt(self, validator):
        """Test repair prompt generation."""
        plan = {
            "helpers": [{"id": "lib.nonexistent"}],
            "schemas": ["bad_schema"],
        }
        
        result = validator.validate(plan)
        prompt = validator.generate_repair_prompt(plan, result)
        
        assert "validation errors" in prompt.lower()
        assert "lib.nonexistent" in prompt
        assert "bad_schema" in prompt
    
    def test_error_summary(self, validator):
        """Test error summary for repair."""
        plan = {
            "helpers": [{"id": "bad_helper"}],
            "schemas": [],
        }
        
        result = validator.validate(plan)
        summary = result.error_summary()
        
        assert "bad_helper" in summary
        assert "not found" in summary.lower()


class TestNewHelpers:
    """Tests for new_helpers validation."""
    
    def test_new_helpers_with_valid_format(self, validator):
        """Test that well-formed new_helpers don't cause errors."""
        plan = {
            "helpers": ["lib.tekton.tasks"],
            "schemas": [],
            "new_helpers": [
                {
                    "name": "_format_result",
                    "signature": "_format_result(task)",
                    "reason": "Format for output",
                    "implementation": "sprintf(...)",
                }
            ],
        }
        
        result = validator.validate(plan)
        
        assert result.valid is True
        assert len(result.warnings) == 0
    
    def test_new_helpers_missing_name_warns(self, validator):
        """Test that new_helpers without name generates warning."""
        plan = {
            "helpers": [],
            "schemas": [],
            "new_helpers": [
                {"implementation": "something()"}  # Missing name
            ],
        }
        
        result = validator.validate(plan)
        
        # Still valid (warnings don't invalidate)
        assert result.valid is True
        assert len(result.warnings) > 0
        assert "name" in result.warnings[0].lower()


class TestValidationResultSerialization:
    """Tests for ValidationResult serialization."""
    
    def test_to_dict(self, validator):
        """Test conversion to dictionary."""
        plan = {
            "helpers": [{"id": "bad_helper"}],
            "schemas": [],
        }
        
        result = validator.validate(plan)
        data = result.to_dict()
        
        assert "valid" in data
        assert data["valid"] is False
        assert "errors" in data
        assert len(data["errors"]) == 1
        assert "error_type" in data["errors"][0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

