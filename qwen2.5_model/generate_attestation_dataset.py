#!/usr/bin/env python3
"""
Generate training dataset for fine-tuning on Rego attestation parsing.

This script:
1. Scans all JSON attestation files in repo root
2. Generates instruction-response pairs for parsing attestations
3. Trims large attestations to only relevant parts
4. Generates Rego code that evaluates attestations
5. Outputs training data in JSONL format
"""

import json
import re
import subprocess
import sys
import tempfile
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import random
import copy

# Import logging setup
try:
    from logging_setup import setup_logging, log_exception
except ImportError:
    # Fallback if logging_setup not available
    import logging
    logging.basicConfig(level=logging.INFO)
    def setup_logging(name, **kwargs):
        return logging.getLogger(name)
    def log_exception(logger, exc, context=""):
        logger.error(f"{context}: {exc}" if context else f"Exception: {exc}", exc_info=True)

# Configuration
REPO_ROOT = Path(__file__).parent.parent
TRAIN_SPLIT = 0.9  # 90% train, 10% eval
MAX_CONTEXT_LINES = 200  # Target max lines per trimmed attestation


@dataclass
class AttestationExample:
    """A single training example for attestation parsing."""
    instruction: str
    context: str  # Trimmed attestation JSON as string
    output_code: str  # Rego code
    task_type: str  # "rego_attestation_parse"
    source_file: str


class AttestationAnalyzer:
    """Analyzes attestation JSON files to understand structure."""
    
    def __init__(self, json_file: Path):
        self.json_file = json_file
        self.data = None
        self.attestation_type = None  # "slsa_v0.2", "slsa_v1", "in-toto"
        
    def load(self) -> bool:
        """Load and analyze the JSON file."""
        try:
            with open(self.json_file, 'r') as f:
                content = f.read()
            
            # Try to parse as single JSON first
            try:
                raw_data = json.loads(content)
            except json.JSONDecodeError as e:
                # Might be multiple JSON objects - try to parse first one
                decoder = json.JSONDecoder()
                try:
                    raw_data, idx = decoder.raw_decode(content, 0)
                    # Skip whitespace after first object
                    while idx < len(content) and content[idx].isspace():
                        idx += 1
                except json.JSONDecodeError:
                    # If that fails, try reading just first 100KB (most files are single objects)
                    try:
                        raw_data = json.loads(content[:100000])
                    except Exception as e:
                        logging.warning(f"Could not parse JSON from {self.json_file.name}: {e}")
                        return False
            
            # Normalize to standard format: {"attestations": [{"statement": {...}}]}
            if isinstance(raw_data, list):
                # Array of attestations
                self.data = {"attestations": [{"statement": item} if "statement" not in item else item for item in raw_data]}
            elif isinstance(raw_data, dict):
                if "attestations" in raw_data:
                    # Already in correct format
                    self.data = raw_data
                elif "statement" in raw_data:
                    # Wrapper with statement
                    self.data = {"attestations": [raw_data]}
                else:
                    # Direct attestation object (most common case)
                    # Check if it looks like an attestation (has _type or predicateType)
                    if "_type" in raw_data or "predicateType" in raw_data:
                        self.data = {"attestations": [{"statement": raw_data}]}
                    else:
                        # Unknown format, try to wrap it
                        self.data = {"attestations": [{"statement": raw_data}]}
            
            # Determine type from first attestation
            if self.data.get("attestations"):
                att = self.data["attestations"][0]
                stmt = att.get("statement", att)  # statement might be the root
                
                if stmt.get("predicateType") == "https://slsa.dev/provenance/v1":
                    self.attestation_type = "slsa_v1"
                elif stmt.get("predicate", {}).get("buildConfig"):
                    self.attestation_type = "slsa_v0.2"
                elif stmt.get("predicate", {}).get("buildDefinition"):
                    self.attestation_type = "slsa_v1"
                else:
                    self.attestation_type = "in-toto"
            
            return True
        except Exception as e:
            logging.error(f"Error loading {self.json_file}: {e}", exc_info=True)
            return False
    
    def get_tasks(self) -> List[Dict]:
        """Extract all tasks from attestations."""
        tasks = []
        for att in self.data.get("attestations", []):
            stmt = att.get("statement", att)  # statement might be the root
            pred = stmt.get("predicate", {})
            
            # SLSA v0.2
            if pred.get("buildConfig", {}).get("tasks"):
                tasks.extend(pred["buildConfig"]["tasks"])
            
            # SLSA v1 - tasks are in resolvedDependencies (base64 encoded)
            # For now, we'll focus on v0.2 which is more common
            if pred.get("buildDefinition", {}).get("resolvedDependencies"):
                # Would need to decode base64, skip for now
                pass
        
        return tasks
    
    def get_subjects(self) -> List[Dict]:
        """Extract all subjects from attestations."""
        subjects = []
        for att in self.data.get("attestations", []):
            stmt = att.get("statement", att)  # statement might be the root
            if stmt.get("subject"):
                subjects.extend(stmt["subject"])
        return subjects
    
    def get_materials(self) -> List[Dict]:
        """Extract all materials from attestations."""
        materials = []
        for att in self.data.get("attestations", []):
            stmt = att.get("statement", att)  # statement might be the root
            pred = stmt.get("predicate", {})
            if pred.get("materials"):
                materials.extend(pred["materials"])
        return materials


class AttestationTrimmer:
    """Trims large attestations to only relevant parts."""
    
    @staticmethod
    def trim_for_task_query(data: Dict, task_name: Optional[str] = None) -> Dict:
        """Trim attestation for task-related queries."""
        trimmed = {}
        
        for att in data.get("attestations", []):
            stmt = att.get("statement", att)  # statement might be the root
            pred = stmt.get("predicate", {})
            build_config = pred.get("buildConfig", {})
            tasks = build_config.get("tasks", [])
            
            if not tasks:
                continue
            
            # Find relevant tasks
            relevant_tasks = []
            for task in tasks:
                if task_name is None or task.get("name") == task_name:
                    # Keep this task but trim it
                    trimmed_task = AttestationTrimmer._trim_task(task)
                    relevant_tasks.append(trimmed_task)
                    if task_name:  # If looking for specific task, stop after first match
                        break
            
            if relevant_tasks:
                trimmed_att = {
                    "statement": {
                        "predicate": {
                            "buildConfig": {
                                "tasks": relevant_tasks
                            }
                        }
                    }
                }
                
                if "attestations" not in trimmed:
                    trimmed["attestations"] = []
                trimmed["attestations"].append(trimmed_att)
        
        return trimmed if trimmed.get("attestations") else data
    
    @staticmethod
    def _trim_task(task: Dict) -> Dict:
        """Trim a single task, keeping only essential fields."""
        trimmed = {}
        
        # Always keep these fields
        for field in ["name", "status", "ref", "results", "startedOn", "finishedOn"]:
            if field in task:
                if field == "ref":
                    # Keep ref but simplify
                    ref = task["ref"]
                    trimmed_ref = {}
                    for ref_field in ["bundle", "name", "kind", "resolver", "params"]:
                        if ref_field in ref:
                            trimmed_ref[ref_field] = ref[ref_field]
                    if trimmed_ref:
                        trimmed["ref"] = trimmed_ref
                elif field == "results":
                    # Keep all results (they're usually small)
                    trimmed["results"] = task["results"]
                else:
                    trimmed[field] = task[field]
        
        # Keep invocation.parameters if present (usually small)
        if "invocation" in task and "parameters" in task["invocation"]:
            trimmed["invocation"] = {"parameters": task["invocation"]["parameters"]}
        
        return trimmed
    
    @staticmethod
    def trim_for_subject_query(data: Dict, subject_name: Optional[str] = None) -> Dict:
        """Trim attestation for subject-related queries."""
        trimmed = {}
        
        for att in data.get("attestations", []):
            stmt = att.get("statement", att)  # statement might be the root
            subjects = stmt.get("subject", [])
            
            if not subjects:
                continue
            
            relevant_subjects = []
            for subject in subjects:
                if subject_name is None or subject.get("name") == subject_name:
                    # Keep subject with name and digest
                    trimmed_subject = {
                        "name": subject.get("name"),
                        "digest": subject.get("digest", {})
                    }
                    relevant_subjects.append(trimmed_subject)
                    if subject_name:
                        break
            
            if relevant_subjects:
                trimmed_att = {
                    "statement": {
                        "subject": relevant_subjects
                    }
                }
                
                if "attestations" not in trimmed:
                    trimmed["attestations"] = []
                trimmed["attestations"].append(trimmed_att)
        
        return trimmed if trimmed.get("attestations") else data
    
    @staticmethod
    def trim_for_material_query(data: Dict, uri: Optional[str] = None) -> Dict:
        """Trim attestation for material-related queries."""
        trimmed = {}
        
        for att in data.get("attestations", []):
            stmt = att.get("statement", att)  # statement might be the root
            pred = stmt.get("predicate", {})
            materials = pred.get("materials", [])
            
            if not materials:
                continue
            
            relevant_materials = []
            for material in materials:
                if uri is None or material.get("uri") == uri:
                    # Keep material with uri and digest
                    trimmed_material = {
                        "uri": material.get("uri"),
                        "digest": material.get("digest", {})
                    }
                    relevant_materials.append(trimmed_material)
                    if uri:
                        break
            
            if relevant_materials:
                trimmed_att = {
                    "statement": {
                        "predicate": {
                            "materials": relevant_materials
                        }
                    }
                }
                
                if "attestations" not in trimmed:
                    trimmed["attestations"] = []
                trimmed["attestations"].append(trimmed_att)
        
        return trimmed if trimmed.get("attestations") else data


class RegoCodeGenerator:
    """Generates Rego code for attestation parsing."""
    
    # Track which format to use (bare expression vs full rule)
    USE_FULL_RULES = True  # Set to True to include package/import/deny structure
    USE_DENY_RULES = True  # Set to True to use "deny" rules (common in policy)
    
    @staticmethod
    def _wrap_in_rule(code: str, rule_name: str = "deny", use_full: bool = True, use_deny: bool = True) -> str:
        """Wrap Rego code in a proper rule structure."""
        if not use_full:
            return code
        
        # Check if it's already a full rule (has package)
        if code.strip().startswith("package"):
            return code
        
        # For deny rules, use "deny contains result" pattern (common in policy)
        if use_deny and rule_name != "deny":
            # Use the specific rule name but also show deny pattern
            # Randomly choose between specific rule name and deny pattern
            import random
            if random.random() < 0.3:  # 30% chance to use deny
                rule_name = "deny"
                result_code = 'result := {"msg": "Policy violation found"}'
            else:
                result_code = None
        else:
            result_code = None
        
        # Check if it's a variable assignment (set comprehension or list)
        if ":=" in code and ("{" in code or "[" in code):
            # Variable assignment - keep as is, just add package/import
            return f"""package attestation_check

import rego.v1

{code}"""
        elif "if {" in code:
            # Already has rule structure, just add package/import
            return f"""package attestation_check

import rego.v1

{code}"""
        else:
            # Simple expressions - wrap in a rule
            if result_code:
                return f"""package attestation_check

import rego.v1

{rule_name} contains result if {{
    {code}
    {result_code}
}}"""
            else:
                return f"""package attestation_check

import rego.v1

{rule_name} if {{
    {code}
}}"""
    
    @staticmethod
    def generate_task_name_check(task_name: str) -> str:
        """Generate Rego code to check for a task by name."""
        code = f"""some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
"""
        return RegoCodeGenerator._wrap_in_rule(code, "task_found", RegoCodeGenerator.USE_FULL_RULES, RegoCodeGenerator.USE_DENY_RULES)
    
    @staticmethod
    def generate_task_status_check(task_name: str, status: str, use_in: bool = False) -> str:
        """Generate Rego code to check task status. Optionally use 'in' for membership (style guide)."""
        if use_in:
            # Style guide: use 'in' for membership when checking against multiple values
            valid_statuses = [status, "Completed", "Running"]  # Include related statuses
            status_set = "{" + ", ".join(f'"{s}"' for s in valid_statuses) + "}"
            code = f"""some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    task.status in {status_set}"""
        else:
            # Standard equality check (also valid)
            code = f"""some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    task.status == "{status}"
"""
        return RegoCodeGenerator._wrap_in_rule(code, "task_status_check", RegoCodeGenerator.USE_FULL_RULES, RegoCodeGenerator.USE_DENY_RULES)
    
    @staticmethod
    def generate_check_task_status_value(task_name: str, status: str) -> str:
        """Generate Rego code to check if task has a specific status value (validation/deny rule)."""
        # Multiple instruction variations should map to this same code
        code = f"""deny contains result if {{
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    task.status == "{status}"
    result := {{"msg": "Policy violation found"}}
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "deny", RegoCodeGenerator.USE_FULL_RULES, use_deny=True)
    
    @staticmethod
    def generate_check_material_value(uri: str, commit: Optional[str] = None) -> str:
        """Generate Rego code to check if material exists with specific URI/commit (validation/deny rule)."""
        # Multiple instruction variations should map to this same code
        if commit:
            code = f"""deny contains result if {{
    some att in input.attestations
    some material in att.statement.predicate.materials
    material.uri == "{uri}"
    material.digest.sha1 == "{commit}"
    result := {{"msg": "Policy violation found"}}
}}"""
        else:
            code = f"""deny contains result if {{
    some att in input.attestations
    some material in att.statement.predicate.materials
    material.uri == "{uri}"
    result := {{"msg": "Policy violation found"}}
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "deny", RegoCodeGenerator.USE_FULL_RULES, use_deny=True)
    
    @staticmethod
    def generate_check_subject_digest_value(digest: str) -> str:
        """Generate Rego code to check if subject has a specific digest (validation/deny rule)."""
        # Multiple instruction variations should map to this same code
        code = f"""deny contains result if {{
    some att in input.attestations
    some subject in att.statement.subject
    subject.digest.sha256 == "{digest}"
    result := {{"msg": "Policy violation found"}}
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "deny", RegoCodeGenerator.USE_FULL_RULES, use_deny=True)
    
    @staticmethod
    def generate_list_task_names() -> str:
        """Generate Rego code to list all task names."""
        code = """task_names := {name |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    name := task.name
}"""
        return RegoCodeGenerator._wrap_in_rule(code, "task_names", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_get_task_results(task_name: str) -> str:
        """Generate Rego code to get results from a task.
        
        Note: Uses 'r' instead of 'result' to avoid variable shadowing conflicts
        when this pattern is used inside 'deny contains result if {...}' rules.
        """
        code = f"""task_results := [r |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    some r in task.results
]"""
        return RegoCodeGenerator._wrap_in_rule(code, "task_results", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_get_task_bundle(task_name: str) -> str:
        """Generate Rego code to get bundle reference for a task."""
        # Handle both ref.bundle (direct) and ref.params[].bundle (via params)
        code = f"""bundle_ref := ref if {{
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    ref := task.ref.bundle
}} else := ref if {{
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    some param in task.ref.params
    param.name == "bundle"
    ref := param.value
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "bundle_ref", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_check_task_bundle_value(task_name: str, bundle_value: str) -> str:
        """Generate Rego code to check if task has a specific bundle value (validation/deny rule)."""
        # This generates a deny rule that checks if bundle equals a specific value
        # Multiple instruction variations should map to this same code
        # Use separate rules for ref.bundle and ref.params (can't use else with deny contains)
        code = f"""deny contains result if {{
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    bundle := task.ref.bundle
    bundle == "{bundle_value}"
    result := {{"msg": "Policy violation found"}}
}}

deny contains result if {{
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    some param in task.ref.params
    param.name == "bundle"
    param.value == "{bundle_value}"
    result := {{"msg": "Policy violation found"}}
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "deny", RegoCodeGenerator.USE_FULL_RULES, use_deny=True)
    
    @staticmethod
    def generate_get_subject_digest(index: int = 0) -> str:
        """Generate Rego code to get subject digest. Uses unconditional assignment (style guide)."""
        # Style guide: prefer unconditional assignment in rule head
        code = f"""subject_digest := digest if {{
    some att in input.attestations
    att.statement.subject[{index}].digest.sha256 == digest
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "subject_digest", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_check_material(uri: str, commit: Optional[str] = None) -> str:
        """Generate Rego code to check for a material."""
        if commit:
            code = f"""some att in input.attestations
    some material in att.statement.predicate.materials
    material.uri == "{uri}"
    material.digest.sha1 == "{commit}"
"""
        else:
            code = f"""some att in input.attestations
    some material in att.statement.predicate.materials
    material.uri == "{uri}"
"""
        return RegoCodeGenerator._wrap_in_rule(code, "material_found", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_list_subject_names() -> str:
        """Generate Rego code to list all subject names."""
        code = """subject_names := {name |
    some att in input.attestations
    some subject in att.statement.subject
    name := subject.name
}"""
        return RegoCodeGenerator._wrap_in_rule(code, "subject_names", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_find_tasks_by_status(status: str) -> str:
        """Generate Rego code to find tasks by status."""
        # Use set instead of array (style guide: prefer sets over arrays where applicable)
        code = f"""tasks_with_status := {{task |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.status == "{status}"
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "tasks_with_status", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_all_tasks_succeeded() -> str:
        """Generate Rego code using 'every' for FOR ALL within a single attestation."""
        code = """all_tasks_succeeded if {
    some att in input.attestations
    every task in att.statement.predicate.buildConfig.tasks {
        task.status == "Succeeded"
    }
}"""
        return RegoCodeGenerator._wrap_in_rule(code, "all_tasks_succeeded", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_all_tasks_succeeded_universal() -> str:
        """Generate Rego code using nested 'every' for FOR ALL across all attestations (universal condition)."""
        code = """all_tasks_succeeded if {
    every att in input.attestations {
        every task in att.statement.predicate.buildConfig.tasks {
            task.status == "Succeeded"
        }
    }
}"""
        return RegoCodeGenerator._wrap_in_rule(code, "all_tasks_succeeded", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_no_failed_tasks() -> str:
        """Generate Rego code using 'every' with negation (style guide pattern)."""
        # Use 'every' with != instead of 'not some' (idiomatic Rego)
        code = """no_failed_tasks if {
    some att in input.attestations
    every task in att.statement.predicate.buildConfig.tasks {
        task.status != "Failed"
    }
}"""
        return RegoCodeGenerator._wrap_in_rule(code, "no_failed_tasks", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_deny_any_task_not_succeeded() -> str:
        """Generate Rego code using deny pattern for negative existence (any task not succeeded)."""
        code = """deny contains result if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.status != "Succeeded"
    result := {"msg": sprintf("task %q did not succeed", [task.name])}
}"""
        return RegoCodeGenerator._wrap_in_rule(code, "deny", RegoCodeGenerator.USE_FULL_RULES, use_deny=True)
    
    @staticmethod
    def generate_deny_any_task_failed() -> str:
        """Generate Rego code using deny pattern for negative existence (any task failed)."""
        code = """deny contains result if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.status == "Failed"
    result := {"msg": sprintf("task %q failed", [task.name])}
}"""
        return RegoCodeGenerator._wrap_in_rule(code, "deny", RegoCodeGenerator.USE_FULL_RULES, use_deny=True)
    
    @staticmethod
    def generate_deny_task_with_status(task_name: str, status: str) -> str:
        """Generate Rego code using deny pattern for specific task with specific status."""
        code = f"""deny contains result if {{
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    task.status == "{status}"
    result := {{"msg": sprintf("task %q has status %q", [task.name, task.status])}}
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "deny", RegoCodeGenerator.USE_FULL_RULES, use_deny=True)
    
    @staticmethod
    def generate_deny_task_result_value(task_name: str, result_name: str, expected_value: str) -> str:
        """Generate Rego deny rule that checks a specific result value.
        
        Note: Uses 'r' instead of 'result' to avoid variable shadowing with 'deny contains result'.
        """
        code = f"""deny contains result if {{
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    some r in task.results
    r.name == "{result_name}"
    r.value == "{expected_value}"
    result := {{"msg": sprintf("task %q result %q has value %q", [task.name, r.name, r.value])}}
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "deny", RegoCodeGenerator.USE_FULL_RULES, use_deny=True)
    
    @staticmethod
    def generate_deny_task_missing_result(task_name: str, result_name: str) -> str:
        """Generate Rego deny rule that checks if a task is missing a required result.
        
        Note: Uses 'r' instead of 'result' to avoid variable shadowing with 'deny contains result'.
        """
        code = f"""deny contains result if {{
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    not task_has_result(task, "{result_name}")
    result := {{"msg": sprintf("task %q is missing required result %q", [task.name, "{result_name}"])}}
}}

task_has_result(task, name) if {{
    some r in task.results
    r.name == name
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "deny", RegoCodeGenerator.USE_FULL_RULES, use_deny=True)
    
    @staticmethod
    def generate_valid_task_status(task_name: str, valid_statuses: List[str]) -> str:
        """Generate Rego code using 'in' for membership check (style guide pattern)."""
        status_set = "{" + ", ".join(f'"{s}"' for s in valid_statuses) + "}"
        code = f"""some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    task.status in {status_set}"""
        return RegoCodeGenerator._wrap_in_rule(code, "valid_task_status", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_task_not_found(task_name: str) -> str:
        """Generate Rego code using 'every' with negation to check if task doesn't exist."""
        # Use 'every' with != instead of 'not some' (idiomatic Rego)
        code = f"""task_not_found if {{
    some att in input.attestations
    every task in att.statement.predicate.buildConfig.tasks {{
        task.name != "{task_name}"
    }}
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "task_not_found", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_task_bundle_with_helper(task_name: str) -> str:
        """Generate Rego code using helper rule (style guide pattern)."""
        # Style guide: Use helper rules for readability, leading underscore for internal
        # This returns code that already has package/import, so _wrap_in_rule won't double-wrap
        code = f"""package attestation_check

import rego.v1

# Helper rule (style guide: leading underscore for internal use)
_task_by_name(name) := task if {{
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == name
}}

# Main rule using helper
bundle_ref := ref if {{
    task := _task_by_name("{task_name}")
    ref := task.ref.bundle
}} else := ref if {{
    task := _task_by_name("{task_name}")
    some param in task.ref.params
    param.name == "bundle"
    ref := param.value
}}"""
        return code  # Already has package/import, don't wrap
    
    @staticmethod
    def generate_get_task_timestamp(task_name: str, field: str = "startedOn") -> str:
        """Generate Rego code to get task timestamp (startedOn or finishedOn)."""
        code = f"""task_timestamp := timestamp if {{
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    timestamp := task.{field}
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "task_timestamp", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_find_subject_by_digest(digest: str) -> str:
        """Generate Rego code to find subject by digest."""
        code = f"""subject_found if {{
    some att in input.attestations
    some subject in att.statement.subject
    subject.digest.sha256 == "{digest}"
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "subject_found", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_list_material_uris() -> str:
        """Generate Rego code to list all material URIs."""
        code = """material_uris := {uri |
    some att in input.attestations
    some material in att.statement.predicate.materials
    uri := material.uri
}"""
        return RegoCodeGenerator._wrap_in_rule(code, "material_uris", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_return_tasks_by_name(task_name: str) -> str:
        """Generate Rego code to return all tasks with a specific name (navigation/query pattern)."""
        code = f"""tasks_named_{task_name.replace("-", "_")} := [task |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
]"""
        return RegoCodeGenerator._wrap_in_rule(code, f"tasks_named_{task_name.replace('-', '_')}", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_return_task_names_by_name(task_name: str) -> str:
        """Generate Rego code to return just the names of tasks with a specific name (set comprehension)."""
        code = f"""{task_name.replace("-", "_")}_task_names := {{task.name |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, f"{task_name.replace('-', '_')}_task_names", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_return_task_statuses_by_name(task_name: str) -> str:
        """Generate Rego code to return (name, status) tuples/objects for tasks with a specific name."""
        code = f"""{task_name.replace("-", "_")}_task_statuses := [{{"name": task.name, "status": task.status}} |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
]"""
        return RegoCodeGenerator._wrap_in_rule(code, f"{task_name.replace('-', '_')}_task_statuses", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_get_param_value(task_name: str, param_name: str) -> str:
        """Generate Rego code to get a specific parameter value by name (first-class parameter navigation)."""
        rule_name = f"{task_name.replace('-', '_')}_{param_name.replace('-', '_')}"
        code = f"""{rule_name} := value if {{
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    some param in task.ref.params
    param.name == "{param_name}"
    value := param.value
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, rule_name, RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_get_result_names(task_name: str) -> str:
        """Generate Rego code to return the names of all result keys for a task.
        
        Note: Uses 'r' instead of 'result' to avoid variable shadowing conflicts.
        """
        code = f"""result_names := {{r.name |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    some r in task.results
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, "result_names", RegoCodeGenerator.USE_FULL_RULES)
    
    @staticmethod
    def generate_get_result_by_name(task_name: str, result_name: str) -> str:
        """Generate Rego code to get a specific result value by name (e.g., exitCode).
        
        Note: Uses 'r' instead of 'result' to avoid variable shadowing conflicts.
        """
        rule_name = f"{task_name.replace('-', '_')}_{result_name.replace('-', '_')}"
        code = f"""{rule_name} := value if {{
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    some r in task.results
    r.name == "{result_name}"
    value := r.value
}}"""
        return RegoCodeGenerator._wrap_in_rule(code, rule_name, RegoCodeGenerator.USE_FULL_RULES)


class InstructionGenerator:
    """Generates instruction templates and variations."""
    
    TASK_TEMPLATES = [
        "In an attestation, check all tasks for a task named '{task_name}'",
        "Find a task named '{task_name}' in the attestation",
        "Check if there is a task called '{task_name}'",
        "Look for a task with name '{task_name}' in the attestation",
        "Does the attestation contain a task named '{task_name}'?",
        "Search for task '{task_name}' in the attestation",
        "Verify if task '{task_name}' exists in the attestation",
        "Check whether task '{task_name}' is present",
        # More natural language variations
        "Is there a task called '{task_name}'?",
        "Do we have a task named '{task_name}'?",
        "Can you find task '{task_name}'?",
        "Look up task '{task_name}'",
        "See if '{task_name}' task exists",
        "Check for '{task_name}' task",
        "Does '{task_name}' exist as a task?",
        "Is '{task_name}' present in the tasks?",
    ]
    
    TASK_STATUS_TEMPLATES = [
        "Get the status of task '{task_name}'",
        "Check the status of task '{task_name}'",
        "Find the status for task '{task_name}'",
        "What is the status of task '{task_name}'?",
        "Retrieve the status for task '{task_name}'",
        "Determine the status of task '{task_name}'",
        # More natural language variations
        "What's the status of '{task_name}'?",
        "Show me the status for task '{task_name}'",
        "Tell me the status of '{task_name}'",
        "How did task '{task_name}' do?",
        "What status does '{task_name}' have?",
        "Give me the status of '{task_name}'",
        "Print the status of task '{task_name}'",
        "Display the status for '{task_name}'",
    ]
    
    # Validation templates for status - multiple variations that should produce the same deny/check code
    TASK_STATUS_CHECK_TEMPLATES = [
        "Check if task '{task_name}' has status '{status}'",
        "Check if task '{task_name}' status is '{status}'",
        "Verify task '{task_name}' has status '{status}'",
        "Verify task '{task_name}' status equals '{status}'",
        "Deny if task '{task_name}' has status '{status}'",
        "Deny if task '{task_name}' status is '{status}'",
        "Check if any attestation has task '{task_name}' with status '{status}'",
        "Check that task '{task_name}' has status '{status}'",
        "Ensure task '{task_name}' status is '{status}'",
        "Check if task '{task_name}' status equals '{status}'",
    ]
    
    LIST_TASKS_TEMPLATES = [
        "List all task names in the attestation",
        "Get all task names from the attestation",
        "Find all task names",
        "What are all the task names?",
        "Retrieve all task names",
        "Show all task names in the attestation",
        # More natural language variations
        "Print all task names",
        "What tasks are in here?",
        "Show me all the task names",
        "Give me a list of all task names",
        "Display all task names",
        "What are the names of all tasks?",
        "List every task name",
        "Show what tasks we have",
    ]
    
    # Navigation/query patterns - return tasks directly
    RETURN_TASKS_BY_NAME_TEMPLATES = [
        "Return a list of all tasks named '{task_name}'",
        "Return all tasks named '{task_name}'",
        "List every task named '{task_name}' across all attestations",
        "Show me all tasks named '{task_name}'",
        "Get all tasks with name '{task_name}'",
        "Return all tasks called '{task_name}'",
        "Find all tasks named '{task_name}' and return them",
        # More natural language variations
        "Print all tasks named '{task_name}'",
        "List tasks with name '{task_name}'",
        "Return all {task_name} tasks",
        "Which tasks use the name '{task_name}'?",
        "Show all {task_name} tasks",
        "Give me all tasks named '{task_name}'",
        "What tasks are called '{task_name}'?",
        "Display all tasks with the name '{task_name}'",
        "Find every task that's named '{task_name}'",
        "I need all tasks named '{task_name}'",
    ]
    
    # Navigation/query patterns - return just names
    RETURN_TASK_NAMES_BY_NAME_TEMPLATES = [
        "Return just the names of all tasks named '{task_name}'",
        "List the names of all tasks named '{task_name}'",
        "Get the names of all tasks called '{task_name}'",
        "Return task names for all tasks named '{task_name}'",
        "Show me the names of all '{task_name}' tasks",
        # More natural language variations
        "What are the names of all '{task_name}' tasks?",
        "Print the names of tasks called '{task_name}'",
        "Give me just the names for '{task_name}' tasks",
        "List all task names that are '{task_name}'",
        "Show task names matching '{task_name}'",
    ]
    
    # Navigation/query patterns - return (name, status) objects
    RETURN_TASK_STATUSES_BY_NAME_TEMPLATES = [
        "Show me all tasks named '{task_name}' and their status",
        "Return all tasks named '{task_name}' with their name and status",
        "List every task named '{task_name}' with name and status",
        "Get all '{task_name}' tasks with their name and status",
        "Return name and status for all tasks named '{task_name}'",
        "Show all '{task_name}' tasks and their status",
        # More natural language variations
        "What are all '{task_name}' tasks and their statuses?",
        "Print all '{task_name}' tasks with name and status",
        "Give me '{task_name}' tasks and how they did",
        "List '{task_name}' tasks and whether they succeeded",
        "Show '{task_name}' tasks and their completion status",
    ]
    
    TASK_RESULTS_TEMPLATES = [
        "Get all results from task '{task_name}'",
        "Find all results for task '{task_name}'",
        "List the results from task '{task_name}'",
        "What are the results from task '{task_name}'?",
        "Retrieve results for task '{task_name}'",
        "Show all results from task '{task_name}'",
        "Return all results for task '{task_name}'",
        "Get all result objects from task '{task_name}'",
        "List every result from task '{task_name}'",
        # More natural language variations
        "Print all results for '{task_name}'",
        "What did '{task_name}' produce?",
        "Show me what '{task_name}' returned",
        "Give me all the results from '{task_name}'",
        "Display results for task '{task_name}'",
        "What outputs did '{task_name}' generate?",
        "List everything '{task_name}' produced",
    ]
    
    # Parameter navigation templates - first-class parameter access
    PARAM_VALUE_TEMPLATES = [
        "Find the value of the {param_name} parameter for task '{task_name}'",
        "Get the {param_name} parameter value for task '{task_name}'",
        "Return the value of the {param_name} parameter for task '{task_name}'",
        "What is the {param_name} parameter value for task '{task_name}'?",
        "Retrieve the {param_name} parameter for task '{task_name}'",
        "Get the {param_name} param value from task '{task_name}'",
        "Find the {param_name} param for task '{task_name}'",
        # More natural language variations
        "What's the {param_name} param for '{task_name}'?",
        "Show me the {param_name} parameter for task '{task_name}'",
        "Print the {param_name} param value from '{task_name}'",
        "Give me the {param_name} value for '{task_name}'",
        "What {param_name} did '{task_name}' use?",
        "Display the {param_name} parameter for '{task_name}'",
        "Tell me the {param_name} value for task '{task_name}'",
    ]
    
    # Result navigation templates - result names
    RESULT_NAMES_TEMPLATES = [
        "Return the names of all result keys for task '{task_name}'",
        "Get all result names from task '{task_name}'",
        "List the names of all results for task '{task_name}'",
        "What are the result names for task '{task_name}'?",
        "Show all result key names for task '{task_name}'",
        "Return all result names from task '{task_name}'",
        # More natural language variations
        "Print all result names for '{task_name}'",
        "What result keys does '{task_name}' have?",
        "Show me the result names from '{task_name}'",
        "List what results '{task_name}' produced",
        "Give me all result key names for '{task_name}'",
        "What are the names of '{task_name}' results?",
    ]
    
    # Result navigation templates - specific result by name
    RESULT_BY_NAME_TEMPLATES = [
        "Return the {result_name} result for task '{task_name}' if present",
        "Get the {result_name} result value for task '{task_name}'",
        "Find the {result_name} result for task '{task_name}'",
        "What is the {result_name} result value for task '{task_name}'?",
        "Retrieve the {result_name} result from task '{task_name}'",
        "Get the {result_name} result if present for task '{task_name}'",
        # More natural language variations
        "What's the {result_name} for '{task_name}'?",
        "Show me the {result_name} result from '{task_name}'",
        "Print the {result_name} value for task '{task_name}'",
        "Give me the {result_name} from '{task_name}'",
        "What did '{task_name}' return for {result_name}?",
        "Display the {result_name} result for '{task_name}'",
        "Tell me the {result_name} value from '{task_name}'",
    ]
    
    TASK_BUNDLE_TEMPLATES = [
        "Get the bundle reference for task '{task_name}'",
        "Find the bundle used by task '{task_name}'",
        "Get the bundle for task '{task_name}'",
        "What bundle is used by task '{task_name}'?",
        "Retrieve the bundle reference for task '{task_name}'",
        "Find the bundle image for task '{task_name}'",
        # More natural language variations
        "What bundle did '{task_name}' use?",
        "Show me the bundle for '{task_name}'",
        "Print the bundle reference from '{task_name}'",
        "Give me the bundle that '{task_name}' used",
        "What's the bundle image for '{task_name}'?",
        "Display the bundle for task '{task_name}'",
        "Tell me which bundle '{task_name}' used",
    ]
    
    # Validation templates - multiple variations that should produce the same deny/check code
    TASK_BUNDLE_CHECK_TEMPLATES = [
        "Check if task '{task_name}' has bundle reference '{bundle_value}'",
        "Check if task '{task_name}' uses bundle '{bundle_value}'",
        "Verify task '{task_name}' has bundle reference '{bundle_value}'",
        "Verify task '{task_name}' uses bundle '{bundle_value}'",
        "Deny if task '{task_name}' has bundle reference '{bundle_value}'",
        "Deny if task '{task_name}' uses bundle '{bundle_value}'",
        "Check if any attestation has task '{task_name}' with bundle reference '{bundle_value}'",
        "Check if any attestation has task '{task_name}' using bundle '{bundle_value}'",
        "Verify any attestation has task '{task_name}' with bundle '{bundle_value}'",
        "Deny if any attestation has task '{task_name}' with bundle reference '{bundle_value}'",
        "Check that task '{task_name}' has bundle reference '{bundle_value}'",
        "Ensure task '{task_name}' uses bundle '{bundle_value}'",
        "Check if task '{task_name}' bundle equals '{bundle_value}'",
        "Verify task '{task_name}' bundle is '{bundle_value}'",
    ]
    
    TASK_TIMESTAMP_TEMPLATES = [
        "When did task '{task_name}' start?",
        "Get the start time for task '{task_name}'",
        "Find when task '{task_name}' started",
        "What is the startedOn timestamp for task '{task_name}'?",
        "When did task '{task_name}' finish?",
        "Get the finish time for task '{task_name}'",
        "Find when task '{task_name}' finished",
        "What is the finishedOn timestamp for task '{task_name}'?",
        # More natural language variations
        "What time did '{task_name}' start?",
        "Show me when '{task_name}' started",
        "Print the start time for '{task_name}'",
        "When was '{task_name}' started?",
        "What's the start timestamp for '{task_name}'?",
        "Give me the start time of '{task_name}'",
        "What time did '{task_name}' finish?",
        "Show me when '{task_name}' finished",
        "Print the finish time for '{task_name}'",
        "When was '{task_name}' completed?",
    ]
    
    SUBJECT_DIGEST_TEMPLATES = [
        "Get the SHA256 digest of the first subject",
        "Find the digest of the first subject",
        "Get the first subject's SHA256 digest",
        "What is the SHA256 digest of the first subject?",
        "Retrieve the digest for the first subject",
    ]
    
    LIST_SUBJECTS_TEMPLATES = [
        "List all subject names in the attestation",
        "Get all subject names",
        "Find all subject names",
        "What are all the subject names?",
        "Show all subject names",
        "Retrieve all subject names from the attestation",
    ]
    
    SUBJECT_BY_DIGEST_TEMPLATES = [
        "Find the subject with SHA256 digest '{digest}'",
        "Get the subject that has digest '{digest}'",
        "Which subject has the digest '{digest}'?",
        "Locate the subject with digest '{digest}'",
    ]
    
    # Validation templates for subject digest - multiple variations that should produce the same deny/check code
    SUBJECT_DIGEST_CHECK_TEMPLATES = [
        "Check if subject has SHA256 digest '{digest}'",
        "Check if any subject has digest '{digest}'",
        "Verify subject has SHA256 digest '{digest}'",
        "Verify any attestation has subject with digest '{digest}'",
        "Deny if subject has SHA256 digest '{digest}'",
        "Check that subject digest equals '{digest}'",
        "Ensure subject has SHA256 digest '{digest}'",
    ]
    
    MATERIAL_TEMPLATES = [
        "Check if material exists for git repo '{uri}' and commit '{commit}'",
        "Find material with URI '{uri}' and commit '{commit}'",
        "Check for material matching URI '{uri}' and commit '{commit}'",
        "Does the attestation contain material with URI '{uri}' and commit '{commit}'?",
        "Verify material exists with URI '{uri}' and commit '{commit}'",
    ]
    
    MATERIAL_URI_ONLY_TEMPLATES = [
        "Check if material exists with URI '{uri}'",
        "Find material with URI '{uri}'",
        "Does the attestation contain material with URI '{uri}'?",
        "Check for material matching URI '{uri}'",
        "Verify material exists with URI '{uri}'",
    ]
    
    # Validation templates for materials - multiple variations that should produce the same deny/check code
    MATERIAL_CHECK_TEMPLATES = [
        "Check if material exists with URI '{uri}' and commit '{commit}'",
        "Verify material exists with URI '{uri}' and commit '{commit}'",
        "Deny if material exists with URI '{uri}' and commit '{commit}'",
        "Check that material has URI '{uri}' and commit '{commit}'",
        "Verify any attestation has material with URI '{uri}' and commit '{commit}'",
        "Check if any attestation contains material with URI '{uri}' and commit '{commit}'",
    ]
    
    MATERIAL_URI_CHECK_TEMPLATES = [
        "Check if material exists with URI '{uri}'",
        "Verify material exists with URI '{uri}'",
        "Deny if material exists with URI '{uri}'",
        "Check that material has URI '{uri}'",
        "Verify any attestation has material with URI '{uri}'",
        "Check if any attestation contains material with URI '{uri}'",
    ]
    
    LIST_MATERIALS_TEMPLATES = [
        "List all material URIs in the attestation",
        "Get all material URIs",
        "Find all material URIs",
        "What are all the material URIs?",
        "Show all material URIs",
    ]
    
    TASK_STATUS_FILTER_TEMPLATES = [
        "Find all tasks with status '{status}'",
        "Get all tasks that have status '{status}'",
        "List tasks with status '{status}'",
        "What tasks have status '{status}'?",
        "Show all tasks with status '{status}'",
        "Retrieve all tasks that succeeded",
        "Find all tasks that failed",
        "Get all succeeded tasks",
        "List all failed tasks",
        # More natural language variations
        "Print all tasks with status '{status}'",
        "Show me tasks that are '{status}'",
        "Which tasks are '{status}'?",
        "Give me all '{status}' tasks",
        "List every task that's '{status}'",
        "What tasks ended up '{status}'?",
        "Display all '{status}' tasks",
        "Show tasks that completed with '{status}'",
    ]
    
    # Style guide patterns: every (FOR ALL) - single attestation
    ALL_TASKS_SUCCEEDED_TEMPLATES = [
        "Check if all tasks succeeded",
        "Verify all tasks have status 'Succeeded'",
        "Ensure all tasks completed successfully",
        "Check that every task succeeded",
        "Verify every task has status 'Succeeded'",
        "Ensure all tasks in the attestation succeeded",
    ]
    
    # Style guide patterns: every (FOR ALL) - universal across all attestations
    ALL_TASKS_SUCCEEDED_UNIVERSAL_TEMPLATES = [
        "Ensure all tasks in the attestation have status Succeeded",
        "Check that every task across all attestations succeeded",
        "Verify all tasks in all attestations have status 'Succeeded'",
        "Ensure every task in every attestation succeeded",
        "Check if all tasks in all attestations are Succeeded",
    ]
    
    # Style guide patterns: not (negation)
    NO_FAILED_TASKS_TEMPLATES = [
        "Check if no tasks failed",
        "Verify no tasks have status 'Failed'",
        "Ensure no tasks failed",
        "Check that no tasks have status 'Failed'",
        "Verify there are no failed tasks",
    ]
    
    # Negative existence patterns: deny if any task not succeeded
    DENY_ANY_TASK_NOT_SUCCEEDED_TEMPLATES = [
        "Deny if any task is not Succeeded",
        "Deny if any task did not succeed",
        "Deny if any task has a status other than Succeeded",
        "Deny if any task is not successful",
        "Deny if any task failed to succeed",
    ]
    
    # Negative existence patterns: deny if any task failed
    DENY_ANY_TASK_FAILED_TEMPLATES = [
        "Deny if any task failed",
        "Deny if any task has status Failed",
        "Deny if any task is failed",
        "Deny if there is a failed task",
    ]
    
    TASK_NOT_FOUND_TEMPLATES = [
        "Check if task '{task_name}' does not exist",
        "Verify task '{task_name}' is not present",
        "Ensure task '{task_name}' does not exist in the attestation",
        "Check that task '{task_name}' is not found",
        "Verify no task named '{task_name}' exists",
    ]
    
    # Deny patterns for specific task with specific status
    DENY_TASK_WITH_STATUS_TEMPLATES = [
        "Deny if task '{task_name}' has status '{status}'",
        "Deny if task '{task_name}' status is '{status}'",
        "Deny if any attestation has task '{task_name}' with status '{status}'",
        "Deny if task '{task_name}' is '{status}'",
    ]
    
    # Style guide patterns: in (membership)
    VALID_TASK_STATUS_TEMPLATES = [
        "Check if task '{task_name}' has a valid status",
        "Verify task '{task_name}' status is valid",
        "Check if task '{task_name}' status is one of the allowed values",
        "Verify task '{task_name}' has an acceptable status",
    ]
    
    @staticmethod
    def generate_task_instructions(tasks: List[Dict]) -> List[Tuple[str, str, Dict]]:
        """Generate instructions for task-related queries."""
        examples = []
        
        # Get unique task names
        task_names = {task.get("name") for task in tasks if task.get("name")}
        
        for task_name in task_names:
            # Get task object once for reuse
            task = next((t for t in tasks if t.get("name") == task_name), None)
            
            # Task name check
            template = random.choice(InstructionGenerator.TASK_TEMPLATES)
            instruction = template.format(task_name=task_name)
            rego_code = RegoCodeGenerator.generate_task_name_check(task_name)
            examples.append((instruction, rego_code, {"task_name": task_name}))
            
            # Navigation/query patterns - return tasks directly (NEW)
            # Add these patterns with 30% probability to balance dataset
            if random.random() < 0.3:
                # Return tasks directly
                template = random.choice(InstructionGenerator.RETURN_TASKS_BY_NAME_TEMPLATES)
                instruction = template.format(task_name=task_name)
                rego_code = RegoCodeGenerator.generate_return_tasks_by_name(task_name)
                examples.append((instruction, rego_code, {"task_name": task_name, "query_type": "navigation"}))
            
            # Navigation/query patterns - return just names (NEW)
            if random.random() < 0.2:
                template = random.choice(InstructionGenerator.RETURN_TASK_NAMES_BY_NAME_TEMPLATES)
                instruction = template.format(task_name=task_name)
                rego_code = RegoCodeGenerator.generate_return_task_names_by_name(task_name)
                examples.append((instruction, rego_code, {"task_name": task_name, "query_type": "navigation"}))
            
            # Navigation/query patterns - return (name, status) objects (NEW)
            if task and task.get("status") and random.random() < 0.2:
                template = random.choice(InstructionGenerator.RETURN_TASK_STATUSES_BY_NAME_TEMPLATES)
                instruction = template.format(task_name=task_name)
                rego_code = RegoCodeGenerator.generate_return_task_statuses_by_name(task_name)
                examples.append((instruction, rego_code, {"task_name": task_name, "query_type": "navigation"}))
            
            # Task status check
            if task and task.get("status"):
                status = task["status"]
                
                # 50% retrieval queries, 50% validation queries (with value check)
                if random.random() < 0.5:
                    # Generate validation queries - multiple instruction variations for same code
                    num_variations = random.randint(3, 5)  # Generate 3-5 variations per status
                    for _ in range(num_variations):
                        template = random.choice(InstructionGenerator.TASK_STATUS_CHECK_TEMPLATES)
                        instruction = template.format(task_name=task_name, status=status)
                        rego_code = RegoCodeGenerator.generate_check_task_status_value(task_name, status)
                        examples.append((instruction, rego_code, {"task_name": task_name, "status": status, "query_type": "validation"}))
                    
                    # NEW: Add deny pattern for specific task with specific status (negative existence)
                    # 30% chance to also generate deny pattern with sprintf message
                    if random.random() < 0.3:
                        template = random.choice(InstructionGenerator.DENY_TASK_WITH_STATUS_TEMPLATES)
                        instruction = template.format(task_name=task_name, status=status)
                        rego_code = RegoCodeGenerator.generate_deny_task_with_status(task_name, status)
                        examples.append((instruction, rego_code, {"task_name": task_name, "status": status, "query_type": "negative_existence"}))
                else:
                    # Generate retrieval queries
                    # 80% use standard equality, 20% use 'in' for membership (style guide)
                    use_in = random.random() < 0.2
                    template = random.choice(InstructionGenerator.TASK_STATUS_TEMPLATES)
                    instruction = template.format(task_name=task_name)
                    rego_code = RegoCodeGenerator.generate_task_status_check(task_name, status, use_in=use_in)
                    examples.append((instruction, rego_code, {"task_name": task_name, "status": status, "query_type": "retrieval"}))
                    
                    # Add valid status check (style guide: use 'in' for membership)
                    if use_in:
                        template = random.choice(InstructionGenerator.VALID_TASK_STATUS_TEMPLATES)
                        instruction = template.format(task_name=task_name)
                        valid_statuses = [status, "Completed", "Running"]
                        rego_code = RegoCodeGenerator.generate_valid_task_status(task_name, valid_statuses)
                        examples.append((instruction, rego_code, {"task_name": task_name, "valid_statuses": valid_statuses}))
            
            # Task results - enhanced with more navigation patterns
            if task and task.get("results"):
                # Basic: return all results
                template = random.choice(InstructionGenerator.TASK_RESULTS_TEMPLATES)
                instruction = template.format(task_name=task_name)
                rego_code = RegoCodeGenerator.generate_get_task_results(task_name)
                examples.append((instruction, rego_code, {"task_name": task_name}))
                
                # NEW: Return result names
                if random.random() < 0.4:
                    template = random.choice(InstructionGenerator.RESULT_NAMES_TEMPLATES)
                    instruction = template.format(task_name=task_name)
                    rego_code = RegoCodeGenerator.generate_get_result_names(task_name)
                    examples.append((instruction, rego_code, {"task_name": task_name, "query_type": "navigation"}))
                
                # NEW: Return specific result by name (e.g., exitCode)
                results = task.get("results", [])
                if results:
                    # Pick a random result or specifically look for exitCode
                    result_names = [r.get("name") for r in results if r.get("name")]
                    if result_names:
                        # 30% chance to specifically ask for exitCode if it exists, otherwise random
                        if "exitCode" in result_names and random.random() < 0.3:
                            result_name = "exitCode"
                        else:
                            result_name = random.choice(result_names)
                        
                        if random.random() < 0.4:
                            template = random.choice(InstructionGenerator.RESULT_BY_NAME_TEMPLATES)
                            instruction = template.format(task_name=task_name, result_name=result_name)
                            rego_code = RegoCodeGenerator.generate_get_result_by_name(task_name, result_name)
                            examples.append((instruction, rego_code, {"task_name": task_name, "result_name": result_name, "query_type": "navigation"}))
            
            # Task bundle and parameters - check both ref.bundle and ref.params
            ref = task.get("ref", {}) if task else {}
            params = ref.get("params", []) if isinstance(ref.get("params"), list) else []
            has_bundle = ref.get("bundle") or any(p.get("name") == "bundle" for p in params)
            
            if task and has_bundle:
                bundle_value = ref.get("bundle")
                if not bundle_value:
                    bundle_param = next((p for p in params if p.get("name") == "bundle"), None)
                    bundle_value = bundle_param.get("value") if bundle_param else None
                
                # 50% retrieval queries, 50% validation queries (with value check)
                # Increased to 50% to generate more instruction variations
                if bundle_value and random.random() < 0.5:
                    # Generate validation queries - multiple instruction variations for same code
                    # This teaches the model that different phrasings = same output
                    num_variations = random.randint(3, 5)  # Generate 3-5 variations per bundle (increased from 2-4)
                    for _ in range(num_variations):
                        template = random.choice(InstructionGenerator.TASK_BUNDLE_CHECK_TEMPLATES)
                        instruction = template.format(task_name=task_name, bundle_value=bundle_value)
                        rego_code = RegoCodeGenerator.generate_check_task_bundle_value(task_name, bundle_value)
                        examples.append((instruction, rego_code, {"task_name": task_name, "bundle_value": bundle_value, "query_type": "validation"}))
                else:
                    # Generate retrieval queries
                    # 80% use standard pattern, 20% use helper rule (style guide)
                    use_helper = random.random() < 0.2
                    template = random.choice(InstructionGenerator.TASK_BUNDLE_TEMPLATES)
                    instruction = template.format(task_name=task_name)
                    if use_helper:
                        rego_code = RegoCodeGenerator.generate_task_bundle_with_helper(task_name)
                    else:
                        rego_code = RegoCodeGenerator.generate_get_task_bundle(task_name)
                    examples.append((instruction, rego_code, {"task_name": task_name, "query_type": "retrieval"}))
            
            # NEW: Parameter navigation - first-class parameter access
            # Generate examples for each parameter found (excluding bundle which is handled above)
            if params:
                for param in params:
                    param_name = param.get("name")
                    # Skip bundle parameter as it's already handled above
                    if param_name and param_name != "bundle":
                        # 40% chance to generate parameter navigation example
                        if random.random() < 0.4:
                            template = random.choice(InstructionGenerator.PARAM_VALUE_TEMPLATES)
                            instruction = template.format(task_name=task_name, param_name=param_name)
                            rego_code = RegoCodeGenerator.generate_get_param_value(task_name, param_name)
                            examples.append((instruction, rego_code, {"task_name": task_name, "param_name": param_name, "query_type": "navigation"}))
        
        # List all task names (once per attestation)
        if task_names:
            template = random.choice(InstructionGenerator.LIST_TASKS_TEMPLATES)
            instruction = template
            rego_code = RegoCodeGenerator.generate_list_task_names()
            examples.append((instruction, rego_code, {}))
        
        # Style guide: Add 'every' FOR ALL queries
        if tasks:
            # Check if all tasks succeeded (single attestation)
            all_succeeded = all(task.get("status") == "Succeeded" for task in tasks if task.get("status"))
            if all_succeeded or random.random() < 0.3:  # 30% chance to add this query
                template = random.choice(InstructionGenerator.ALL_TASKS_SUCCEEDED_TEMPLATES)
                instruction = template
                rego_code = RegoCodeGenerator.generate_all_tasks_succeeded()
                examples.append((instruction, rego_code, {}))
            
            # NEW: Universal condition across all attestations (nested every)
            if random.random() < 0.2:  # 20% chance to add universal pattern
                template = random.choice(InstructionGenerator.ALL_TASKS_SUCCEEDED_UNIVERSAL_TEMPLATES)
                instruction = template
                rego_code = RegoCodeGenerator.generate_all_tasks_succeeded_universal()
                examples.append((instruction, rego_code, {"query_type": "universal"}))
            
            # Style guide: Add 'not' negation queries
            has_failed = any(task.get("status") == "Failed" for task in tasks if task.get("status"))
            if not has_failed or random.random() < 0.3:  # 30% chance to add this query
                template = random.choice(InstructionGenerator.NO_FAILED_TASKS_TEMPLATES)
                instruction = template
                rego_code = RegoCodeGenerator.generate_no_failed_tasks()
                examples.append((instruction, rego_code, {}))
            
            # NEW: Negative existence patterns - deny if any task not succeeded
            if random.random() < 0.25:  # 25% chance to add deny pattern
                template = random.choice(InstructionGenerator.DENY_ANY_TASK_NOT_SUCCEEDED_TEMPLATES)
                instruction = template
                rego_code = RegoCodeGenerator.generate_deny_any_task_not_succeeded()
                examples.append((instruction, rego_code, {"query_type": "negative_existence"}))
            
            # NEW: Negative existence patterns - deny if any task failed
            if has_failed or random.random() < 0.25:  # 25% chance to add deny pattern
                template = random.choice(InstructionGenerator.DENY_ANY_TASK_FAILED_TEMPLATES)
                instruction = template
                rego_code = RegoCodeGenerator.generate_deny_any_task_failed()
                examples.append((instruction, rego_code, {"query_type": "negative_existence"}))
        
        # Style guide: Add task not found queries (negation)
        # Use a task name that doesn't exist in this attestation
        fake_task_names = ["non-existent-task", "missing-task", "unknown-task"]
        if random.random() < 0.2:  # 20% chance to add negation query
            fake_name = random.choice(fake_task_names)
            template = random.choice(InstructionGenerator.TASK_NOT_FOUND_TEMPLATES)
            instruction = template.format(task_name=fake_name)
            rego_code = RegoCodeGenerator.generate_task_not_found(fake_name)
            examples.append((instruction, rego_code, {"task_name": fake_name}))
        
        # Find tasks by status - generate multiple examples per status for better coverage
        statuses = {task.get("status") for task in tasks if task.get("status")}
        for status in statuses:
            # Generate 3-4 examples per status to increase coverage
            # Filter templates that work with the status
            status_lower = status.lower()
            if "succeed" in status_lower:
                # Use templates that mention "succeeded" or work generically
                available_templates = [t for t in InstructionGenerator.TASK_STATUS_FILTER_TEMPLATES if "{status}" in t or "succeed" in t.lower()]
            elif "fail" in status_lower:
                # Use templates that mention "failed" or work generically
                available_templates = [t for t in InstructionGenerator.TASK_STATUS_FILTER_TEMPLATES if "{status}" in t or "fail" in t.lower()]
            else:
                # Use templates with {status} placeholder
                available_templates = [t for t in InstructionGenerator.TASK_STATUS_FILTER_TEMPLATES if "{status}" in t]
            
            if not available_templates:
                available_templates = [t for t in InstructionGenerator.TASK_STATUS_FILTER_TEMPLATES if "{status}" in t]
            
            num_examples = min(4, len(available_templates))
            selected_templates = random.sample(available_templates, num_examples)
            for template in selected_templates:
                # Handle templates that don't need status parameter
                if "{status}" in template:
                    instruction = template.format(status=status)
                else:
                    # For templates like "Retrieve all tasks that succeeded"
                    instruction = template
                rego_code = RegoCodeGenerator.generate_find_tasks_by_status(status)
                examples.append((instruction, rego_code, {"status": status}))
        
        # Task timestamps (startedOn, finishedOn)
        for task_name in task_names:
            task = next((t for t in tasks if t.get("name") == task_name), None)
            if task:
                if task.get("startedOn"):
                    template = random.choice([t for t in InstructionGenerator.TASK_TIMESTAMP_TEMPLATES if "start" in t.lower()])
                    instruction = template.format(task_name=task_name)
                    rego_code = RegoCodeGenerator.generate_get_task_timestamp(task_name, "startedOn")
                    examples.append((instruction, rego_code, {"task_name": task_name, "field": "startedOn"}))
                
                if task.get("finishedOn"):
                    template = random.choice([t for t in InstructionGenerator.TASK_TIMESTAMP_TEMPLATES if "finish" in t.lower()])
                    instruction = template.format(task_name=task_name)
                    rego_code = RegoCodeGenerator.generate_get_task_timestamp(task_name, "finishedOn")
                    examples.append((instruction, rego_code, {"task_name": task_name, "field": "finishedOn"}))
        
        return examples
    
    @staticmethod
    def generate_subject_instructions(subjects: List[Dict]) -> List[Tuple[str, str, Dict]]:
        """Generate instructions for subject-related queries."""
        examples = []
        
        if subjects:
            # First subject digest
            template = random.choice(InstructionGenerator.SUBJECT_DIGEST_TEMPLATES)
            instruction = template
            rego_code = RegoCodeGenerator.generate_get_subject_digest(0)
            examples.append((instruction, rego_code, {}))
            
            # List subject names
            template = random.choice(InstructionGenerator.LIST_SUBJECTS_TEMPLATES)
            instruction = template
            rego_code = RegoCodeGenerator.generate_list_subject_names()
            examples.append((instruction, rego_code, {}))
            
            # Subject by digest lookup (for first few subjects with digests)
            for subject in subjects[:4]:  # Limit to first 4 (increased from 2 to allow more validation variations)
                digest = subject.get("digest", {}).get("sha256")
                if digest:
                    # 50% retrieval queries, 50% validation queries (with value check)
                    if random.random() < 0.5:
                        # Generate validation queries - multiple instruction variations for same code
                        num_variations = random.randint(3, 5)  # Generate 3-5 variations per digest
                        for _ in range(num_variations):
                            template = random.choice(InstructionGenerator.SUBJECT_DIGEST_CHECK_TEMPLATES)
                            instruction = template.format(digest=digest)
                            rego_code = RegoCodeGenerator.generate_check_subject_digest_value(digest)
                            examples.append((instruction, rego_code, {"digest": digest, "query_type": "validation"}))
                    else:
                        # Generate retrieval queries
                        template = random.choice(InstructionGenerator.SUBJECT_BY_DIGEST_TEMPLATES)
                        instruction = template.format(digest=digest)
                        rego_code = RegoCodeGenerator.generate_find_subject_by_digest(digest)
                        examples.append((instruction, rego_code, {"digest": digest, "query_type": "retrieval"}))
        
        return examples
    
    @staticmethod
    def generate_material_instructions(materials: List[Dict]) -> List[Tuple[str, str, Dict]]:
        """Generate instructions for material-related queries."""
        examples = []
        
        # List all materials (once per attestation)
        if materials:
            template = random.choice(InstructionGenerator.LIST_MATERIALS_TEMPLATES)
            instruction = template
            rego_code = RegoCodeGenerator.generate_list_material_uris()
            examples.append((instruction, rego_code, {}))
        
        for material in materials:
            uri = material.get("uri")
            digest = material.get("digest", {})
            commit = digest.get("sha1") or digest.get("sha256")
            
            if uri:
                # Material with URI and commit (if available)
                if commit:
                    # 50% retrieval queries, 50% validation queries (with value check)
                    if random.random() < 0.5:
                        # Generate validation queries - multiple instruction variations for same code
                        num_variations = random.randint(3, 5)  # Generate 3-5 variations per material
                        for _ in range(num_variations):
                            template = random.choice(InstructionGenerator.MATERIAL_CHECK_TEMPLATES)
                            instruction = template.format(uri=uri, commit=commit)
                            rego_code = RegoCodeGenerator.generate_check_material_value(uri, commit)
                            examples.append((instruction, rego_code, {"uri": uri, "commit": commit, "query_type": "validation"}))
                    else:
                        # Generate retrieval queries
                        template = random.choice(InstructionGenerator.MATERIAL_TEMPLATES)
                        instruction = template.format(uri=uri, commit=commit)
                        rego_code = RegoCodeGenerator.generate_check_material(uri, commit)
                        examples.append((instruction, rego_code, {"uri": uri, "commit": commit, "query_type": "retrieval"}))
                else:
                    # Material with URI only
                    # 50% retrieval queries, 50% validation queries (with value check)
                    if random.random() < 0.5:
                        # Generate validation queries - multiple instruction variations for same code
                        num_variations = random.randint(3, 5)  # Generate 3-5 variations per material
                        for _ in range(num_variations):
                            template = random.choice(InstructionGenerator.MATERIAL_URI_CHECK_TEMPLATES)
                            instruction = template.format(uri=uri)
                            rego_code = RegoCodeGenerator.generate_check_material_value(uri)
                            examples.append((instruction, rego_code, {"uri": uri, "query_type": "validation"}))
                    else:
                        # Generate retrieval queries
                        template = random.choice(InstructionGenerator.MATERIAL_URI_ONLY_TEMPLATES)
                        instruction = template.format(uri=uri)
                        rego_code = RegoCodeGenerator.generate_check_material(uri, None)
                        examples.append((instruction, rego_code, {"uri": uri, "query_type": "retrieval"}))
        
        return examples


@dataclass
class PolicyRule:
    """A policy rule extracted from a Rego file."""
    package: str
    package_title: str  # Title from package-level METADATA
    package_description: str  # Description from package-level METADATA
    title: str
    description: str
    short_name: str
    failure_msg: str
    solution: str
    rule_type: str  # "deny" or "warn"
    rule_code: str  # The actual rule code
    full_code: str  # Full code including package, imports, and helpers
    source_file: str
    collections: List[str] = None
    depends_on: List[str] = None
    effective_on: str = None
    imports_used: List[str] = None
    helpers_used: List[str] = None


class PolicyRuleParser:
    """Parses Rego policy files to extract rules and metadata."""
    
    def __init__(self, policy_dir: Path):
        self.policy_dir = policy_dir
        self.logger = logging.getLogger(__name__)
    
    def parse_file(self, rego_file: Path) -> List[PolicyRule]:
        """Parse a single Rego file and extract all rules with metadata."""
        rules = []
        
        try:
            content = rego_file.read_text()
        except Exception as e:
            self.logger.warning(f"Failed to read {rego_file}: {e}")
            return rules
        
        # Skip test files
        if rego_file.name.endswith("_test.rego"):
            return rules
        
        # Extract package name
        package_match = re.search(r'^package\s+(\S+)', content, re.MULTILINE)
        if not package_match:
            return rules
        package = package_match.group(1)
        
        # Extract package-level metadata (first METADATA block before any rule)
        pkg_metadata = self._extract_package_metadata(content)
        
        # Extract imports
        imports = re.findall(r'^import\s+.+$', content, re.MULTILINE)
        
        # Find all deny/warn rules with their preceding METADATA blocks
        rules_data = self._extract_rules_with_metadata(content)
        
        for rule_data in rules_data:
            metadata = rule_data['metadata']
            rule_code = rule_data['rule_code']
            
            if not metadata.get('title') or not metadata.get('description'):
                continue
            
            # Determine rule type
            rule_type = "deny" if rule_code.strip().startswith("deny") else "warn"
            
            # Extract helper functions used by this rule
            helpers = self._extract_helpers(content, rule_code)
            
            # Build full code with package, imports, and helpers
            full_code = self._build_full_code(package, imports, rule_code, helpers)
            
            rule = PolicyRule(
                package=package,
                package_title=pkg_metadata.get('title', ''),
                package_description=pkg_metadata.get('description', ''),
                title=metadata.get('title', ''),
                description=metadata.get('description', ''),
                short_name=metadata.get('short_name', ''),
                failure_msg=metadata.get('failure_msg', ''),
                solution=metadata.get('solution', ''),
                rule_type=rule_type,
                rule_code=rule_code,
                full_code=full_code,
                source_file=str(rego_file),
                collections=metadata.get('collections', []),
                depends_on=metadata.get('depends_on', []),
                effective_on=metadata.get('effective_on', ''),
                imports_used=imports,
                helpers_used=[h[:50] + "..." if len(h) > 50 else h for h in helpers]
            )
            rules.append(rule)
        
        return rules
    
    def _extract_package_metadata(self, content: str) -> Dict[str, Any]:
        """Extract the package-level METADATA block (first one in file)."""
        # Find first METADATA block that's at the start of file
        match = re.search(
            r'^#\s*\n#\s*METADATA\s*\n((?:#[^\n]*\n)+)',
            content,
            re.MULTILINE
        )
        if match:
            return self._parse_metadata(match.group(1))
        return {}
    
    def _extract_rules_with_metadata(self, content: str) -> List[Dict]:
        """Extract all deny/warn rules with their metadata blocks."""
        rules = []
        
        # Find all rule definitions (deny/warn contains result if {...})
        rule_pattern = re.compile(
            r'((?:deny|warn)\s+contains\s+result\s+if\s*\{)',
            re.MULTILINE
        )
        
        for match in rule_pattern.finditer(content):
            rule_start = match.start()
            
            # Find matching closing brace (handle nested braces)
            brace_count = 1
            pos = match.end()
            while pos < len(content) and brace_count > 0:
                if content[pos] == '{':
                    brace_count += 1
                elif content[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            rule_code = content[rule_start:pos]
            
            # Look backwards for METADATA block
            # Pattern: METADATA followed by comment lines, ending with #\n before the rule
            before_rule = content[:rule_start]
            
            # Try pattern with trailing #\n (common format)
            metadata_match = re.search(
                r'#\s*METADATA\s*\n((?:#[^\n]*\n)+)#\s*\n\s*$',
                before_rule
            )
            
            if not metadata_match:
                # Fallback: find the last METADATA block before this rule
                all_metadata = list(re.finditer(
                    r'#\s*METADATA\s*\n((?:#[^\n]*\n)+)',
                    before_rule
                ))
                if all_metadata:
                    metadata_match = all_metadata[-1]  # Use closest one
            
            if metadata_match:
                metadata = self._parse_metadata(metadata_match.group(1))
                rules.append({
                    'metadata': metadata,
                    'rule_code': rule_code
                })
        
        return rules
    
    def _parse_metadata(self, metadata_block: str) -> Dict[str, Any]:
        """Parse a METADATA comment block into a dictionary."""
        metadata = {}
        
        # Remove comment markers and join continuation lines
        lines = []
        for line in metadata_block.split('\n'):
            line = re.sub(r'^#\s?', '', line)
            lines.append(line)
        text = '\n'.join(lines)
        
        # Extract title
        title_match = re.search(r'title:\s*(.+?)(?:\n|$)', text)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        # Extract multi-line description (>- format)
        desc_match = re.search(r'description:\s*>-\s*\n((?:\s+.+\n?)+)', text)
        if desc_match:
            desc_lines = desc_match.group(1).split('\n')
            desc = ' '.join(line.strip() for line in desc_lines if line.strip())
            metadata['description'] = desc
        else:
            # Try single-line description
            desc_match = re.search(r'description:\s*(.+?)(?:\n(?!\s)|$)', text)
            if desc_match:
                metadata['description'] = desc_match.group(1).strip()
        
        # Extract custom fields
        custom_match = re.search(r'custom:\s*\n((?:\s+.+\n?)+)', text)
        if custom_match:
            custom_text = custom_match.group(1)
            
            # short_name
            short_name_match = re.search(r'short_name:\s*(\S+)', custom_text)
            if short_name_match:
                metadata['short_name'] = short_name_match.group(1)
            
            # failure_msg (can be multi-line with >-)
            failure_msg_match = re.search(r'failure_msg:\s*>-\s*\n((?:\s{4,}.+\n?)+)', custom_text)
            if failure_msg_match:
                msg_lines = failure_msg_match.group(1).split('\n')
                msg = ' '.join(line.strip() for line in msg_lines if line.strip())
                metadata['failure_msg'] = msg
            else:
                failure_msg_match = re.search(r'failure_msg:\s*[\'"]?(.+?)[\'"]?\s*(?:\n|$)', custom_text)
                if failure_msg_match:
                    metadata['failure_msg'] = failure_msg_match.group(1).strip().strip("'\"")
            
            # solution (multi-line)
            solution_match = re.search(r'solution:\s*>-\s*\n((?:\s{4,}.+\n?)+)', custom_text)
            if solution_match:
                sol_lines = solution_match.group(1).split('\n')
                sol = ' '.join(line.strip() for line in sol_lines if line.strip())
                metadata['solution'] = sol
            else:
                solution_match = re.search(r'solution:\s*(.+?)(?:\n|$)', custom_text)
                if solution_match:
                    metadata['solution'] = solution_match.group(1).strip()
            
            # collections
            collections_match = re.search(r'collections:\s*\n((?:\s+-\s+\S+\n?)+)', custom_text)
            if collections_match:
                collections = re.findall(r'-\s+(\S+)', collections_match.group(1))
                metadata['collections'] = collections
            
            # depends_on
            depends_match = re.search(r'depends_on:\s*\n((?:\s+-\s+\S+\n?)+)', custom_text)
            if depends_match:
                depends = re.findall(r'-\s+(\S+)', depends_match.group(1))
                metadata['depends_on'] = depends
            
            # effective_on
            effective_match = re.search(r'effective_on:\s*(\S+)', custom_text)
            if effective_match:
                metadata['effective_on'] = effective_match.group(1)
        
        return metadata
    
    def _extract_helpers(self, content: str, rule_code: str) -> List[str]:
        """Extract helper functions referenced by the rule, recursively.
        
        Handles:
        - Function helpers: _helper(x) := value if {...}
        - Set comprehension helpers: _helper contains x if {...}
        - Simple constant helpers: _helper := "value"
        - Recursive extraction of helpers used by other helpers
        """
        helpers = []
        extracted_names = set()  # Track which helpers we've already extracted
        
        def extract_helper_code(helper_name: str) -> Optional[str]:
            """Extract the code for a single helper definition."""
            # Pattern 1: Function or simple assignment with :=
            # Matches: _name(args) := ... or _name := ...
            pattern1 = rf'^({re.escape(helper_name)}(?:\([^)]*\))?\s*:=\s*)'
            
            # Pattern 2: Set/object with contains
            # Matches: _name contains ...
            pattern2 = rf'^({re.escape(helper_name)}\s+contains\s+)'
            
            # Pattern 3: Function with if {...} (no :=)
            # Matches: _name(args) if {
            pattern3 = rf'^({re.escape(helper_name)}\([^)]*\)\s+if\s*\{{)'
            
            for pattern in [pattern1, pattern2, pattern3]:
                for match in re.finditer(pattern, content, re.MULTILINE):
                    start = match.start()
                    brace_count = 0
                    in_braces = False
                    end = start
                    
                    # Scan from start to find the end of definition
                    i = start
                    while i < len(content):
                        char = content[i]
                        
                        if char == '{':
                            brace_count += 1
                            in_braces = True
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and in_braces:
                                # End of braced definition
                                end = i + 1
                                break
                        elif char == '\n':
                            if in_braces:
                                # Inside braces, continue
                                pass
                            elif brace_count == 0:
                                # Check if this is a simple one-line definition
                                # (no opening brace yet and we hit newline)
                                line_so_far = content[start:i]
                                if ':=' in line_so_far and '{' not in line_so_far:
                                    # Simple assignment like: _key := "value"
                                    end = i
                                    break
                                # Check next line - if it starts with whitespace, it's continuation
                                next_line_start = i + 1
                                if next_line_start < len(content):
                                    next_char = content[next_line_start] if next_line_start < len(content) else ''
                                    if next_char in ' \t':
                                        # Continuation - keep going
                                        pass
                                    elif next_char == '#':
                                        # Comment line - skip
                                        pass
                                    else:
                                        # New definition or blank line
                                        end = i
                                        break
                        i += 1
                    
                    if end > start:
                        return content[start:end].strip()
            
            return None
        
        def find_helpers_in_code(code: str) -> Set[str]:
            """Find all helper references (identifiers starting with _) in code."""
            return set(re.findall(r'\b(_[a-zA-Z_][a-zA-Z0-9_]*)\b', code))
        
        # Start with helpers referenced in the main rule
        to_process = find_helpers_in_code(rule_code)
        
        # Recursively extract helpers
        while to_process:
            helper_name = to_process.pop()
            
            # Skip if already extracted
            if helper_name in extracted_names:
                continue
            
            extracted_names.add(helper_name)
            
            # Extract this helper's code
            helper_code = extract_helper_code(helper_name)
            if helper_code:
                helpers.append(helper_code)
                
                # Find any helpers this helper references
                nested_helpers = find_helpers_in_code(helper_code) - extracted_names
                to_process.update(nested_helpers)
        
        return helpers
    
    def _build_full_code(self, package: str, imports: List[str], rule_code: str, helpers: List[str]) -> str:
        """Build complete Rego code with package, imports, rule, and helpers."""
        parts = [f"package {package}", "", "import rego.v1"]
        
        # Add relevant imports (deduplicate)
        seen_imports = {"import rego.v1"}
        for imp in imports:
            if imp not in seen_imports:
                parts.append(imp)
                seen_imports.add(imp)
        
        parts.append("")
        
        # Add helpers
        for helper in helpers:
            parts.append(helper)
            parts.append("")
        
        # Add the main rule
        parts.append(rule_code)
        
        return '\n'.join(parts)
    
    def scan_policy_directory(self) -> List[PolicyRule]:
        """Scan the policy directory for all rules."""
        all_rules = []
        
        # Scan policy/release for deny/warn rules
        release_dir = self.policy_dir / "release"
        if release_dir.exists():
            for rego_file in release_dir.rglob("*.rego"):
                rules = self.parse_file(rego_file)
                all_rules.extend(rules)
                if rules:
                    self.logger.info(f"  Extracted {len(rules)} rules from {rego_file.name}")
        
        return all_rules


class InstructionPolisher:
    """Uses Ollama with Qwen3 to polish instructions for better training quality."""
    
    def __init__(self, model: str = "qwen3:latest", enabled: bool = True):
        self.model = model
        self.enabled = enabled
        self.cache = {}  # Cache polished instructions
        self._ollama_available = None
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        if self._ollama_available is None:
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                self._ollama_available = result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                self._ollama_available = False
        return self._ollama_available
    
    def polish(self, instruction: str, rule_type: str = "deny", title: str = "") -> str:
        """Polish an instruction using Qwen3 via Ollama.
        
        Args:
            instruction: The raw instruction to polish
            rule_type: "deny" or "warn"
            title: Optional rule title for context
            
        Returns:
            Polished instruction, or original if polishing fails
        """
        if not self.enabled or not self.is_available():
            return instruction
        
        # Check cache
        cache_key = f"{instruction}:{rule_type}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Build the prompt - request no thinking, just the answer
        prompt = f"""/no_think
Rewrite this instruction to be clear, grammatical, and natural-sounding for training data.
Keep it concise (1-3 sentences). Output ONLY the rewritten instruction, nothing else.
The instruction is for writing a Rego {rule_type} rule.

Original: {instruction}

Rewritten:"""
        
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=90  # Increased timeout for models with thinking
            )
            
            if result.returncode == 0:
                polished = result.stdout.strip()
                
                # Remove terminal escape sequences (colors, cursor movement)
                polished = re.sub(r'\[\?[0-9]+[hl]', '', polished)
                polished = re.sub(r'\[[0-9]*[GK]', '', polished)
                polished = re.sub(r'\[[0-9;]*m', '', polished)
                
                # Handle Qwen3 thinking output - extract content after "...done thinking."
                if "...done thinking." in polished:
                    parts = polished.split("...done thinking.")
                    if len(parts) > 1:
                        polished = parts[-1].strip()
                
                # Remove any thinking tags if present
                polished = re.sub(r'<think>.*?</think>', '', polished, flags=re.DOTALL).strip()
                
                # Remove "Thinking..." prefix and other thinking indicators
                polished = re.sub(r'^Thinking\.+\s*', '', polished)
                polished = re.sub(r'^Okay,.*?\.', '', polished, flags=re.DOTALL)
                
                # Remove quotes if the model wrapped the output
                polished = polished.strip('"\'')
                
                # Remove any "Rewritten:" prefix if model included it
                if polished.lower().startswith("rewritten:"):
                    polished = polished[10:].strip()
                
                # Clean up whitespace
                polished = ' '.join(polished.split())
                
                # Basic validation - must be non-empty and not too long
                if polished and len(polished) < 500 and len(polished) > 10:
                    self.cache[cache_key] = polished
                    print(f"Polished instruction: {polished}")
                    return polished
        except (subprocess.TimeoutExpired, Exception) as e:
            pass  # Fall back to original
        
        return instruction


# Global polisher instance (can be disabled via flag)
_instruction_polisher = None

def get_instruction_polisher(enabled: bool = True) -> InstructionPolisher:
    """Get or create the instruction polisher."""
    global _instruction_polisher
    if _instruction_polisher is None:
        _instruction_polisher = InstructionPolisher(enabled=enabled)
    return _instruction_polisher


class PolicyExampleGenerator:
    """Generates training examples from policy rules with rich context."""
    
    @staticmethod
    def _polish_description(desc: str) -> str:
        """Clean up and polish a description to make it a clear, grammatical instruction."""
        # Clean up whitespace
        desc = ' '.join(desc.split())
        
        # Fix common grammatical issues
        # Remove trailing periods before question marks
        desc = re.sub(r'\.\?', '?', desc)
        
        # Add missing articles for common patterns
        desc = re.sub(r'\bensure that task\b', 'ensure that the task', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\bensure task\b', 'ensure the task', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\bverify that task\b', 'verify that the task', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\bverify task\b', 'verify the task', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\bthat image\b', 'that the image', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\bproducing the images\b', 'producing images', desc, flags=re.IGNORECASE)
        
        # Fix missing articles
        desc = re.sub(r'\bthat task producing\b', 'that the task producing', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\bthat task containing\b', 'that the task containing', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\bensure that a list\b', 'ensure that the list', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\bthat will a list\b', 'that the list', desc, flags=re.IGNORECASE)
        
        # Clean up "Ensure that at least one of the tasks" -> cleaner form
        desc = re.sub(r'Ensure that at least one', 'Ensure at least one', desc, flags=re.IGNORECASE)
        
        # Remove duplicate words
        desc = re.sub(r'\bthe the\b', 'the', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\ba a\b', 'a', desc, flags=re.IGNORECASE)
        
        # Fix common lowercase starts after colons
        desc = re.sub(r':\s+([a-z])', lambda m: ': ' + m.group(1).upper(), desc)
        
        # Remove jargon explanations in parentheses if too long
        if len(desc) > 150:
            desc = re.sub(r'\([^)]{50,}\)', '', desc)
        
        # Ensure first character is uppercase
        if desc and desc[0].islower():
            desc = desc[0].upper() + desc[1:]
        
        # Clean up extra spaces
        desc = ' '.join(desc.split())
        
        return desc.strip()
    
    @staticmethod
    def _create_instruction(rule: PolicyRule, style: str = "direct") -> str:
        """Create a clear, polished instruction based on rule metadata.
        
        Styles:
        - direct: Use the polished description as an imperative statement
        - imperative: "Write a Rego rule that will..."
        - question: "How can I write a Rego policy to verify that...?"
        - task: "Create a Rego deny/warn rule that checks..."
        - detailed: Full structured requirements
        - from_failure: Based on the error message
        - with_solution: Include the solution/fix
        """
        desc = PolicyExampleGenerator._polish_description(rule.description)
        title = rule.title
        
        if style == "direct":
            # For direct style, ensure it's a complete imperative sentence
            desc_lower = desc.lower()
            # If it starts with a verb, it's already a good instruction
            if desc_lower.startswith(("verify", "ensure", "check", "confirm", "produce", "validate")):
                return desc
            # Otherwise, make it imperative
            return f"Verify that {desc_lower}" if desc_lower else desc
        
        elif style == "imperative":
            # Convert description to clear imperative form
            desc_lower = desc.lower()
            if desc_lower.startswith(("verify that ", "ensure that ", "confirm that ", "check that ")):
                # Already has "that", just wrap it
                return f"Write a Rego {rule.rule_type} rule that will {desc_lower}"
            elif desc_lower.startswith(("verify ", "ensure ", "confirm ", "check ")):
                # Starts with verb but no "that"
                return f"Write a Rego {rule.rule_type} rule that will {desc_lower}"
            elif desc_lower.startswith("produce"):
                return f"Write a Rego rule that will {desc_lower}"
            else:
                # Generic case - make it imperative
                return f"Write a Rego {rule.rule_type} rule to {desc_lower}"
        
        elif style == "question":
            # Convert to a natural, grammatical question
            desc_lower = desc.lower()
            
            # Handle "produce" descriptions differently (they describe what happens)
            if desc_lower.startswith("produce"):
                return f"How do I write a Rego {rule.rule_type} rule that will {desc_lower}?"
            
            # Extract the core requirement by removing leading verbs
            core = desc_lower
            for verb in ["verify that ", "verify ", "ensure that ", "ensure ", 
                        "check if ", "check that ", "check ", "confirm that ", "confirm "]:
                if core.startswith(verb):
                    core = core[len(verb):]
                    break
            
            # Form a grammatical question
            if core:
                return f"How can I write a Rego policy to verify that {core}?"
            return f"How do I implement a Rego {rule.rule_type} rule for \"{title}\"?"
        
        elif style == "task":
            # Brief but complete task description
            desc_short = desc if len(desc) < 80 else desc[:77] + "..."
            if desc_short.lower().startswith(("verify", "ensure", "check", "confirm")):
                return f"Create a Rego {rule.rule_type} rule that will {desc_short.lower()}"
            return f"Create a Rego {rule.rule_type} rule: {title}. {desc_short}"
        
        elif style == "detailed":
            # Detailed instruction with full context
            lines = [f"Write a Rego {rule.rule_type} rule with the following requirements:"]
            lines.append("")
            lines.append(f"Purpose: {title}")
            lines.append(f"Description: {desc}")
            if rule.failure_msg and '%' not in rule.failure_msg:
                lines.append(f"Error message when violated: {rule.failure_msg}")
            return "\n".join(lines)
        
        elif style == "from_failure":
            # Use failure message as guidance
            if rule.failure_msg and '%' not in rule.failure_msg:
                clean_msg = rule.failure_msg.replace("'", "").replace('"', '')
                return f"Write a Rego {rule.rule_type} rule that reports: \"{clean_msg}\""
            # Fallback to using description
            return f"Create a Rego {rule.rule_type} rule that will {desc.lower()}"
        
        elif style == "with_solution":
            # Include solution context for richer instruction
            instruction = desc
            if rule.solution:
                clean_solution = PolicyExampleGenerator._polish_description(rule.solution)
                if len(clean_solution) < 100:
                    instruction += f" The fix for violations is: {clean_solution}"
            return instruction
        
        return desc
    
    @staticmethod
    def _create_context(rule: PolicyRule) -> str:
        """Create helpful context about the rule."""
        context_parts = []
        
        # Add package context if available
        if rule.package_description:
            context_parts.append(f"# Package: {rule.package}")
            context_parts.append(f"# {rule.package_description[:200]}...")
        
        # Add schema hints based on what the rule checks
        rule_lower = rule.rule_code.lower()
        if "pipelinerun_attestations" in rule_lower or "attestation" in rule_lower:
            context_parts.append("# Input: SLSA Provenance attestation")
            context_parts.append("# Path: input.attestations[] → statement → predicate")
        
        if "tekton.tasks" in rule_lower:
            context_parts.append("# Checks: Tekton Pipeline tasks")
        
        if "task_result" in rule_lower:
            context_parts.append("# Accesses: Task results (task.results[])")
        
        if "task_param" in rule_lower:
            context_parts.append("# Accesses: Task parameters (task.invocation.parameters)")
        
        if "builder.id" in rule_lower:
            context_parts.append("# Checks: Builder ID in predicate.builder.id")
        
        return '\n'.join(context_parts) if context_parts else ""
    
    @staticmethod
    def generate_examples(rules: List[PolicyRule], polish_instructions: bool = True) -> List[Tuple[str, str, Dict]]:
        """Generate training examples from policy rules with varied instructions.
        
        Args:
            rules: List of PolicyRule objects to generate examples from
            polish_instructions: If True, use Ollama/Qwen3 to polish instructions
        """
        examples = []
        polisher = get_instruction_polisher(enabled=polish_instructions)
        
        # Log polisher status
        if polish_instructions and polisher.is_available():
            logging.info("Using Qwen3 via Ollama for instruction polishing")
        elif polish_instructions:
            logging.info("Ollama not available, using template-based polishing only")
        
        # Instruction styles to use
        styles = ["direct", "imperative", "question", "task", "detailed", "from_failure", "with_solution"]
        
        for rule in rules:
            # Skip rules without good descriptions
            if not rule.description or len(rule.description) < 20:
                continue
            
            # Generate multiple instruction variations per rule
            # More important rules (in more collections) get more variations
            num_collections = len(rule.collections) if rule.collections else 0
            num_variations = min(4, 2 + num_collections)
            
            used_styles = set()
            
            for _ in range(num_variations):
                # Pick an unused style
                available_styles = [s for s in styles if s not in used_styles]
                if not available_styles:
                    available_styles = styles
                
                style = random.choice(available_styles)
                used_styles.add(style)
                
                # Generate instruction
                instruction = PolicyExampleGenerator._create_instruction(rule, style)
                
                # Polish instruction with LLM if available (skip "detailed" style as it's structured)
                if polish_instructions and style != "detailed":
                    instruction = polisher.polish(instruction, rule.rule_type, rule.title)
                
                # Generate context (optional, for some variations)
                context = ""
                if style in ["detailed", "with_solution"] and random.random() < 0.5:
                    context = PolicyExampleGenerator._create_context(rule)
                
                # Build the output code (always the full_code)
                output_code = rule.full_code
                
                # Metadata for tracking
                metadata = {
                    "source": "policy_rule",
                    "package": rule.package,
                    "short_name": rule.short_name,
                    "title": rule.title,
                    "rule_type": rule.rule_type,
                    "source_file": rule.source_file,
                    "collections": rule.collections or [],
                    "instruction_style": style,
                    "polished": polish_instructions and polisher.is_available(),
                }
                
                # The context field will include any schema hints
                full_context = context
                
                examples.append((instruction, output_code, metadata, full_context))
        
        return examples
    
    @staticmethod
    def generate_from_package(rules: List[PolicyRule]) -> List[Tuple[str, str, Dict]]:
        """Generate examples for entire packages (multiple rules together)."""
        # Group rules by package
        packages = {}
        for rule in rules:
            if rule.package not in packages:
                packages[rule.package] = []
            packages[rule.package].append(rule)
        
        examples = []
        for package, package_rules in packages.items():
            if len(package_rules) < 2:
                continue
            
            # Create a combined instruction
            if package_rules[0].package_description:
                instruction = f"Implement the {package} policy package: {package_rules[0].package_description}"
            else:
                instruction = f"Write all policy rules for the {package} package"
            
            # Combine all rules and their helpers
            all_imports = set()
            all_helpers = set()  # Use set to deduplicate
            all_rules = []
            
            for rule in package_rules:
                if rule.imports_used:
                    all_imports.update(rule.imports_used)
                all_rules.append(rule.rule_code)
                # Extract helpers from full_code
                if rule.full_code:
                    # Get helper section from full_code (between imports and first deny/warn)
                    import re
                    lines = rule.full_code.split('\n')
                    in_helper = False
                    helper_lines = []
                    for line in lines:
                        if line.startswith('_') and (':=' in line or ' contains ' in line):
                            in_helper = True
                        if in_helper:
                            if line.strip().startswith('deny ') or line.strip().startswith('warn '):
                                break
                            helper_lines.append(line)
                    if helper_lines:
                        helper_block = '\n'.join(helper_lines).strip()
                        if helper_block:
                            all_helpers.add(helper_block)
            
            # Build combined code
            parts = [f"package {package}", "", "import rego.v1"]
            for imp in sorted(all_imports):
                if imp != "import rego.v1":
                    parts.append(imp)
            parts.append("")
            
            # Add helpers (sorted for consistency)
            for helper in sorted(all_helpers):
                parts.append(helper)
                parts.append("")
            
            # Add rules
            for rule_code in all_rules:
                parts.append(rule_code)
                parts.append("")
            
            combined_code = '\n'.join(parts)
            
            metadata = {
                "source": "policy_package",
                "package": package,
                "num_rules": len(package_rules),
                "rule_types": list(set(r.rule_type for r in package_rules)),
            }
            
            examples.append((instruction, combined_code, metadata, ""))
        
        return examples


class ExampleBuilder:
    """Builds training examples from instructions and attestations."""
    
    @staticmethod
    def _generate_schema_header(instruction: str, metadata: Dict) -> str:
        """Generate query-specific schema header based on instruction type."""
        instruction_lower = instruction.lower()
        
        if "task" in instruction_lower:
            # Task queries - show buildConfig.tasks path
            if "bundle" in instruction_lower:
                # Bundle queries need special note about ref structure
                return "# Attestation Structure:\n# input.attestations[] → statement → predicate → buildConfig.tasks[]\n# Task fields: name, status, ref.bundle (direct) OR ref.params[] where param.name == 'bundle'\n"
            elif "param" in instruction_lower:
                # Parameter navigation queries
                return "# Attestation Structure:\n# input.attestations[] → statement → predicate → buildConfig.tasks[]\n# Task fields: name, ref.params[] (array of {name, value} objects)\n# Access parameter: some param in task.ref.params; param.name == 'X'; value := param.value\n"
            elif "result" in instruction_lower:
                # Result navigation queries
                # Note: Use 'r' instead of 'result' to avoid conflicts with 'deny contains result'
                if "name" in instruction_lower and "result" in instruction_lower:
                    # Result names query
                    return "# Attestation Structure:\n# input.attestations[] → statement → predicate → buildConfig.tasks[]\n# Task fields: name, results[] (array of {name, value} objects)\n# Access result names: some r in task.results; r.name\n"
                else:
                    # Specific result or all results query
                    return "# Attestation Structure:\n# input.attestations[] → statement → predicate → buildConfig.tasks[]\n# Task fields: name, results[] (array of {name, value} objects)\n# Access result: some r in task.results; r.name == 'X'; value := r.value\n"
            else:
                return "# Attestation Structure:\n# input.attestations[] → statement → predicate → buildConfig.tasks[]\n# Task fields: name, status, ref.bundle, ref.params[], startedOn, finishedOn, results[]\n"
        elif "material" in instruction_lower:
            # Material queries - show materials path
            return "# Attestation Structure:\n# input.attestations[] → statement → predicate → materials[]\n# Material fields: uri, digest.sha1, digest.sha256\n"
        elif "subject" in instruction_lower:
            # Subject queries - show subject path
            return "# Attestation Structure:\n# input.attestations[] → statement → subject[]\n# Subject fields: name, digest.sha256\n"
        else:
            # Default - generic structure
            return "# Attestation Structure:\n# input.attestations[] → statement → predicate\n"
    
    @staticmethod
    def build_example(
        instruction: str,
        rego_code: str,
        analyzer: AttestationAnalyzer,
        metadata: Dict
    ) -> Optional[AttestationExample]:
        """Build a single training example."""
        # Determine trimming strategy based on instruction
        trimmed_data = ExampleBuilder._trim_attestation(analyzer.data, instruction, metadata)
        
        # Generate query-specific schema header
        schema_header = ExampleBuilder._generate_schema_header(instruction, metadata)
        
        # Convert to JSON string
        json_str = json.dumps(trimmed_data, indent=2, ensure_ascii=False)
        
        # Combine schema header with JSON
        context = f"{schema_header}\n{json_str}"
        
        # Validate context size (schema header adds ~4-5 lines)
        lines = context.split('\n')
        
        if len(lines) > MAX_CONTEXT_LINES:
            # Try more aggressive trimming
            trimmed_data = ExampleBuilder._aggressive_trim(trimmed_data, metadata)
            json_str = json.dumps(trimmed_data, indent=2, ensure_ascii=False)
            # Regenerate context with schema header
            context = f"{schema_header}\n{json_str}"
            lines = context.split('\n')
            if len(lines) > MAX_CONTEXT_LINES * 1.5:  # Allow some flexibility
                schema_lines = len(schema_header.split('\n'))
                json_lines = len(json_str.split('\n'))
                logging.warning(f"Context still large ({len(lines)} lines, {schema_lines} schema + {json_lines} JSON) for {analyzer.json_file.name}")
        
        return AttestationExample(
            instruction=instruction,
            context=context,
            output_code=rego_code,
            task_type="rego_attestation_parse",
            source_file=analyzer.json_file.name
        )
    
    @staticmethod
    def _trim_attestation(data: Dict, instruction: str, metadata: Dict) -> Dict:
        """Trim attestation based on instruction type."""
        instruction_lower = instruction.lower()
        
        if "task" in instruction_lower:
            task_name = metadata.get("task_name")
            return AttestationTrimmer.trim_for_task_query(data, task_name)
        elif "subject" in instruction_lower:
            subject_name = metadata.get("subject_name")
            return AttestationTrimmer.trim_for_subject_query(data, subject_name)
        elif "material" in instruction_lower:
            uri = metadata.get("uri")
            return AttestationTrimmer.trim_for_material_query(data, uri)
        else:
            # Default: minimal trim, keep structure
            return data
    
    @staticmethod
    def _aggressive_trim(data: Dict, metadata: Dict) -> Dict:
        """More aggressive trimming if initial trim wasn't enough."""
        # Further reduce by removing more fields
        trimmed = copy.deepcopy(data)
        
        for att in trimmed.get("attestations", []):
            stmt = att.get("statement", {})
            pred = stmt.get("predicate", {})
            build_config = pred.get("buildConfig", {})
            
            for task in build_config.get("tasks", []):
                # Remove large fields
                task.pop("steps", None)
                task.pop("invocation", None)
                if "ref" in task and isinstance(task["ref"], dict):
                    task["ref"].pop("params", None)
        
        return trimmed


def check_opa_available() -> bool:
    """Check if opa binary is available."""
    try:
        result = subprocess.run(
            ["opa", "version"],
            capture_output=True,
            text=True,
            timeout=2
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _validate_attestation_parsing(rego_code: str, instruction: str) -> bool:
    """Validate that Rego code correctly parses attestations (primary requirement)."""
    # Must navigate attestations
    if "input.attestations" not in rego_code:
        return False
    
    # Must navigate statement
    if "statement" not in rego_code:
        return False
    
    # Check for correct field access based on instruction
    instruction_lower = instruction.lower()
    
    if "task" in instruction_lower:
        # Task queries must access tasks
        if "tasks" not in rego_code and "buildConfig" not in rego_code:
            return False
    
    if "subject" in instruction_lower:
        # Subject queries must access subject
        if "subject" not in rego_code:
            return False
    
    if "material" in instruction_lower:
        # Material queries must access materials
        if "materials" not in rego_code:
            return False
    
    # Must use 'some' for iteration (declarative style)
    if "some " not in rego_code:
        return False
    
    return True


def validate_rego_syntax(rego_code: str) -> bool:
    """Validate Rego code syntax using opa parse. Returns True if opa is not available."""
    # Check if opa is available first
    if not check_opa_available():
        return True  # Skip validation if opa not found
    
    try:
        # Wrap in a package if not already wrapped
        if not rego_code.strip().startswith("package"):
            wrapped_code = "package test\n\n" + rego_code
        else:
            wrapped_code = rego_code
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as f:
            f.write(wrapped_code)
            temp_file = f.name
        
        result = subprocess.run(
            ["opa", "parse", temp_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        Path(temp_file).unlink()
        
        if result.returncode != 0:
            # Debug: print first error
            if result.stderr:
                error_lines = result.stderr.split('\n')[:3]
                # Only print if it's not a common expected error
                if "undefined" not in result.stderr.lower():
                    pass  # Don't print, too verbose
        
        return result.returncode == 0
    except FileNotFoundError:
        # OPA not installed, skip validation
        return True
    except Exception as e:
        # Other errors, skip validation
        return True


def example_to_jsonl(example: AttestationExample) -> str:
    """Convert example to JSONL format."""
    data = {
        "instruction": example.instruction,
        "context": example.context,
        "output_code": example.output_code,
        "task_type": example.task_type,
    }
    return json.dumps(data, ensure_ascii=False)


def policy_example_to_jsonl(instruction: str, output_code: str, metadata: Dict, context: str = "") -> str:
    """Convert a policy rule example to JSONL format."""
    data = {
        "instruction": instruction,
        "context": context if context else "",
        "output_code": output_code,
        "task_type": "rego_policy_rule",
    }
    return json.dumps(data, ensure_ascii=False)


def main():
    """Main function to generate dataset."""
    parser = argparse.ArgumentParser(
        description="Generate training dataset for Rego attestation parsing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default repo root (one level up from script)
  python generate_attestation_dataset.py

  # Specify custom directory for JSON files
  python generate_attestation_dataset.py --json-dir /path/to/json/files

  # Specify output directory
  python generate_attestation_dataset.py --output-dir ./output
  
  # Include policy rules from policy directory
  python generate_attestation_dataset.py --policy-dir /path/to/policy
        """
    )
    parser.add_argument(
        "--json-dir",
        type=str,
        default=None,
        help="Directory containing JSON attestation files (default: repo root, one level up from script)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for JSONL files (default: same as script directory)",
    )
    parser.add_argument(
        "--policy-dir",
        type=str,
        default=None,
        help="Directory containing policy Rego files (default: repo root/policy)",
    )
    parser.add_argument(
        "--include-policy-rules",
        action="store_true",
        default=True,
        help="Include training examples from actual policy rules (default: True)",
    )
    parser.add_argument(
        "--no-policy-rules",
        action="store_true",
        help="Disable including training examples from policy rules",
    )
    parser.add_argument(
        "--polish-instructions",
        action="store_true",
        default=True,
        help="Use Qwen3 via Ollama to polish policy rule instructions (default: True)",
    )
    parser.add_argument(
        "--no-polish",
        action="store_true",
        help="Disable LLM-based instruction polishing",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging("generate_attestation_dataset")
    
    logger.info("=" * 70)
    logger.info("Generating attestation parsing training dataset")
    logger.info("=" * 70)
    
    # Determine JSON files directory
    if args.json_dir:
        json_dir = Path(args.json_dir).resolve()
    else:
        json_dir = REPO_ROOT
    
    logger.info(f"Looking for JSON files in: {json_dir}")
    
    # Check if opa is available
    if not check_opa_available():
        logger.warning("OPA binary not found - skipping Rego syntax validation")
    else:
        logger.info("OPA found - validating Rego syntax")
    
    # Find all JSON files
    json_files = []
    if json_dir.exists() and json_dir.is_dir():
        for json_file in json_dir.glob("*.json"):
            if json_file.name.startswith("att") or "sha256" in json_file.name:
                json_files.append(json_file)
    else:
        logger.error(f"Directory not found: {json_dir}")
        return
    
    logger.info(f"Found {len(json_files)} JSON attestation files")
    
    if not json_files:
        logger.error(f"No JSON files found in {json_dir}!")
        logger.error("  Looking for files starting with 'att' or containing 'sha256'")
        return
    
    # Generate examples
    all_examples = []
    
    for json_file in json_files:
        logger.info(f"Processing {json_file.name}...")
        
        analyzer = AttestationAnalyzer(json_file)
        if not analyzer.load():
            logger.warning(f"  Skipped {json_file.name} (failed to load)")
            continue
        
        # Generate instructions based on content
        tasks = analyzer.get_tasks()
        subjects = analyzer.get_subjects()
        materials = analyzer.get_materials()
        
        logger.info(f"  Found: {len(tasks)} tasks, {len(subjects)} subjects, {len(materials)} materials")
        
        # Generate task-related examples (limit to avoid too many per file)
        task_examples = InstructionGenerator.generate_task_instructions(tasks)
        # Limit to max 30 task examples per file (increased from 20 to allow more validation variations)
        if len(task_examples) > 30:
            task_examples = random.sample(task_examples, 30)
        
        for instruction, rego_code, metadata in task_examples:
            # Primary validation: Rego syntax (must pass)
            if not validate_rego_syntax(rego_code):
                logger.warning(f"Invalid Rego code for instruction: {instruction[:50]}...")
                continue
            
            # Secondary validation: Attestation parsing correctness (must pass)
            if not _validate_attestation_parsing(rego_code, instruction):
                logger.warning(f"Attestation parsing issue for instruction: {instruction[:50]}...")
                continue
            
            example = ExampleBuilder.build_example(instruction, rego_code, analyzer, metadata)
            if example:
                all_examples.append(example)
        
        # Generate subject-related examples
        subject_examples = InstructionGenerator.generate_subject_instructions(subjects)
        for instruction, rego_code, metadata in subject_examples:
            if not validate_rego_syntax(rego_code):
                continue
            if not _validate_attestation_parsing(rego_code, instruction):
                continue
            example = ExampleBuilder.build_example(instruction, rego_code, analyzer, metadata)
            if example:
                all_examples.append(example)
        
        # Generate material-related examples (limit to avoid too many)
        material_examples = InstructionGenerator.generate_material_instructions(materials[:8])  # Limit to first 8 (increased from 5 to allow more validation variations)
        for instruction, rego_code, metadata in material_examples:
            if not validate_rego_syntax(rego_code):
                continue
            if not _validate_attestation_parsing(rego_code, instruction):
                continue
            example = ExampleBuilder.build_example(instruction, rego_code, analyzer, metadata)
            if example:
                all_examples.append(example)
        
        logger.info(f"  Generated {len(all_examples)} total examples so far")
    
    logger.info(f"Total attestation examples generated: {len(all_examples)}")
    
    # Process policy rules if enabled
    policy_examples = []
    if args.include_policy_rules and not args.no_policy_rules:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Extracting training examples from policy rules")
        logger.info("=" * 70)
        
        # Determine policy directory
        if args.policy_dir:
            policy_dir = Path(args.policy_dir).resolve()
        else:
            policy_dir = REPO_ROOT / "policy"
        
        if policy_dir.exists():
            logger.info(f"Scanning policy directory: {policy_dir}")
            
            # Parse policy files
            parser_instance = PolicyRuleParser(policy_dir)
            rules = parser_instance.scan_policy_directory()
            
            logger.info(f"Found {len(rules)} policy rules")
            
            # Determine if we should polish instructions
            polish = args.polish_instructions and not args.no_polish
            
            # Generate examples from rules
            rule_examples = PolicyExampleGenerator.generate_examples(rules, polish_instructions=polish)
            logger.info(f"Generated {len(rule_examples)} examples from individual rules")
            
            # Also generate package-level examples
            package_examples = PolicyExampleGenerator.generate_from_package(rules)
            logger.info(f"Generated {len(package_examples)} examples from packages")
            
            # Validate and add to policy examples
            for example_data in rule_examples + package_examples:
                if len(example_data) == 4:
                    instruction, output_code, metadata, context = example_data
                else:
                    instruction, output_code, metadata = example_data
                    context = ""
                
                # Validate Rego syntax
                if validate_rego_syntax(output_code):
                    policy_examples.append({
                        "instruction": instruction,
                        "output_code": output_code,
                        "metadata": metadata,
                        "context": context
                    })
                else:
                    logger.warning(f"Invalid Rego in policy rule: {metadata.get('short_name', 'unknown')}")
            
            logger.info(f"Total valid policy examples: {len(policy_examples)}")
            
            # Log some sample instructions for verification
            if policy_examples:
                logger.info("")
                logger.info("Sample policy rule instructions:")
                for i, ex in enumerate(random.sample(policy_examples, min(5, len(policy_examples)))):
                    logger.info(f"  {i+1}. {ex['instruction'][:80]}...")
        else:
            logger.warning(f"Policy directory not found: {policy_dir}")
    
    logger.info("")
    logger.info(f"Total attestation examples: {len(all_examples)}")
    logger.info(f"Total policy rule examples: {len(policy_examples)}")
    
    # Shuffle and split attestation examples
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * TRAIN_SPLIT)
    train_examples = all_examples[:split_idx]
    eval_examples = all_examples[split_idx:]
    
    # Shuffle and split policy examples
    random.shuffle(policy_examples)
    policy_split_idx = int(len(policy_examples) * TRAIN_SPLIT)
    policy_train = policy_examples[:policy_split_idx]
    policy_eval = policy_examples[policy_split_idx:]
    
    logger.info(f"Attestation - Train: {len(train_examples)}, Eval: {len(eval_examples)}")
    logger.info(f"Policy Rules - Train: {len(policy_train)}, Eval: {len(policy_eval)}")
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(__file__).parent
    
    train_path = output_dir / "attestation_train.jsonl"
    eval_path = output_dir / "attestation_eval.jsonl"
    
    with open(train_path, 'w') as f:
        for example in train_examples:
            f.write(example_to_jsonl(example) + '\n')
    
    with open(eval_path, 'w') as f:
        for example in eval_examples:
            f.write(example_to_jsonl(example) + '\n')
    
    logger.info("Attestation dataset written successfully")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Eval: {eval_path}")
    
    # Write policy rule examples if any
    if policy_examples:
        policy_train_path = output_dir / "policy_rules_train.jsonl"
        policy_eval_path = output_dir / "policy_rules_eval.jsonl"
        
        with open(policy_train_path, 'w') as f:
            for ex in policy_train:
                f.write(policy_example_to_jsonl(
                    ex['instruction'], 
                    ex['output_code'], 
                    ex['metadata'],
                    ex.get('context', '')
                ) + '\n')
        
        with open(policy_eval_path, 'w') as f:
            for ex in policy_eval:
                f.write(policy_example_to_jsonl(
                    ex['instruction'], 
                    ex['output_code'], 
                    ex['metadata'],
                    ex.get('context', '')
                ) + '\n')
        
        logger.info("Policy rules dataset written successfully")
        logger.info(f"  Train: {policy_train_path}")
        logger.info(f"  Eval: {policy_eval_path}")
    
    # Generate summary
    summary = {
        "attestation_examples": {
            "total": len(all_examples),
            "train": len(train_examples),
            "eval": len(eval_examples),
            "task_type": "rego_attestation_parse",
            "source_files": len(json_files),
        },
        "policy_rule_examples": {
            "total": len(policy_examples),
            "train": len(policy_train),
            "eval": len(policy_eval),
            "task_type": "rego_policy_rule",
        },
        "combined_total": len(all_examples) + len(policy_examples),
    }
    
    summary_path = output_dir / "attestation_dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"  Summary: {summary_path}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

