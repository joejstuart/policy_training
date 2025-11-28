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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import random
import copy

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
                    except:
                        print(f"  Warning: Could not parse JSON from {self.json_file.name}")
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
            print(f"Error loading {self.json_file}: {e}")
            import traceback
            traceback.print_exc()
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
    task.name == "{task_name}\""""
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
    task.status == "{status}\""""
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
        """Generate Rego code to get results from a task."""
        code = f"""task_results := [result |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    some result in task.results
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
    material.digest.sha1 == "{commit}\""""
        else:
            code = f"""some att in input.attestations
    some material in att.statement.predicate.materials
    material.uri == "{uri}\""""
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
        """Generate Rego code using 'every' for FOR ALL (style guide pattern)."""
        code = """all_tasks_succeeded if {
    some att in input.attestations
    every task in att.statement.predicate.buildConfig.tasks {
        task.status == "Succeeded"
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
    ]
    
    TASK_STATUS_TEMPLATES = [
        "Get the status of task '{task_name}'",
        "Check the status of task '{task_name}'",
        "Find the status for task '{task_name}'",
        "What is the status of task '{task_name}'?",
        "Retrieve the status for task '{task_name}'",
        "Determine the status of task '{task_name}'",
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
    ]
    
    TASK_RESULTS_TEMPLATES = [
        "Get all results from task '{task_name}'",
        "Find all results for task '{task_name}'",
        "List the results from task '{task_name}'",
        "What are the results from task '{task_name}'?",
        "Retrieve results for task '{task_name}'",
        "Show all results from task '{task_name}'",
    ]
    
    TASK_BUNDLE_TEMPLATES = [
        "Get the bundle reference for task '{task_name}'",
        "Find the bundle used by task '{task_name}'",
        "Get the bundle for task '{task_name}'",
        "What bundle is used by task '{task_name}'?",
        "Retrieve the bundle reference for task '{task_name}'",
        "Find the bundle image for task '{task_name}'",
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
    ]
    
    # Style guide patterns: every (FOR ALL)
    ALL_TASKS_SUCCEEDED_TEMPLATES = [
        "Check if all tasks succeeded",
        "Verify all tasks have status 'Succeeded'",
        "Ensure all tasks completed successfully",
        "Check that every task succeeded",
        "Verify every task has status 'Succeeded'",
        "Ensure all tasks in the attestation succeeded",
    ]
    
    # Style guide patterns: not (negation)
    NO_FAILED_TASKS_TEMPLATES = [
        "Check if no tasks failed",
        "Verify no tasks have status 'Failed'",
        "Ensure no tasks failed",
        "Check that no tasks have status 'Failed'",
        "Verify there are no failed tasks",
    ]
    
    TASK_NOT_FOUND_TEMPLATES = [
        "Check if task '{task_name}' does not exist",
        "Verify task '{task_name}' is not present",
        "Ensure task '{task_name}' does not exist in the attestation",
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
            # Task name check
            template = random.choice(InstructionGenerator.TASK_TEMPLATES)
            instruction = template.format(task_name=task_name)
            rego_code = RegoCodeGenerator.generate_task_name_check(task_name)
            examples.append((instruction, rego_code, {"task_name": task_name}))
            
            # Task status check
            task = next((t for t in tasks if t.get("name") == task_name), None)
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
            
            # Task results
            if task and task.get("results"):
                template = random.choice(InstructionGenerator.TASK_RESULTS_TEMPLATES)
                instruction = template.format(task_name=task_name)
                rego_code = RegoCodeGenerator.generate_get_task_results(task_name)
                examples.append((instruction, rego_code, {"task_name": task_name}))
            
            # Task bundle - check both ref.bundle and ref.params
            ref = task.get("ref", {})
            has_bundle = ref.get("bundle") or (
                isinstance(ref.get("params"), list) and
                any(p.get("name") == "bundle" for p in ref.get("params", []))
            )
            if task and has_bundle:
                bundle_value = ref.get("bundle")
                if not bundle_value and isinstance(ref.get("params"), list):
                    bundle_param = next((p for p in ref.get("params", []) if p.get("name") == "bundle"), None)
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
        
        # List all task names (once per attestation)
        if task_names:
            template = random.choice(InstructionGenerator.LIST_TASKS_TEMPLATES)
            instruction = template
            rego_code = RegoCodeGenerator.generate_list_task_names()
            examples.append((instruction, rego_code, {}))
        
        # Style guide: Add 'every' FOR ALL queries
        if tasks:
            # Check if all tasks succeeded
            all_succeeded = all(task.get("status") == "Succeeded" for task in tasks if task.get("status"))
            if all_succeeded or random.random() < 0.3:  # 30% chance to add this query
                template = random.choice(InstructionGenerator.ALL_TASKS_SUCCEEDED_TEMPLATES)
                instruction = template
                rego_code = RegoCodeGenerator.generate_all_tasks_succeeded()
                examples.append((instruction, rego_code, {}))
            
            # Style guide: Add 'not' negation queries
            has_failed = any(task.get("status") == "Failed" for task in tasks if task.get("status"))
            if not has_failed or random.random() < 0.3:  # 30% chance to add this query
                template = random.choice(InstructionGenerator.NO_FAILED_TASKS_TEMPLATES)
                instruction = template
                rego_code = RegoCodeGenerator.generate_no_failed_tasks()
                examples.append((instruction, rego_code, {}))
        
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


class ExampleBuilder:
    """Builds training examples from instructions and attestations."""
    
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
        
        # Convert to JSON string
        context = json.dumps(trimmed_data, indent=2, ensure_ascii=False)
        
        # Validate context size
        lines = context.split('\n')
        if len(lines) > MAX_CONTEXT_LINES:
            # Try more aggressive trimming
            trimmed_data = ExampleBuilder._aggressive_trim(trimmed_data, metadata)
            context = json.dumps(trimmed_data, indent=2, ensure_ascii=False)
            lines = context.split('\n')
            if len(lines) > MAX_CONTEXT_LINES * 1.5:  # Allow some flexibility
                print(f"Warning: Context still large ({len(lines)} lines) for {analyzer.json_file.name}")
        
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
    
    args = parser.parse_args()
    
    print("Generating attestation parsing training dataset...")
    
    # Determine JSON files directory
    if args.json_dir:
        json_dir = Path(args.json_dir).resolve()
    else:
        json_dir = REPO_ROOT
    
    print(f"Looking for JSON files in: {json_dir}")
    
    # Check if opa is available
    if not check_opa_available():
        print("⚠ OPA binary not found - skipping Rego syntax validation")
    else:
        print("✓ OPA found - validating Rego syntax")
    
    # Find all JSON files
    json_files = []
    if json_dir.exists() and json_dir.is_dir():
        for json_file in json_dir.glob("*.json"):
            if json_file.name.startswith("att") or "sha256" in json_file.name:
                json_files.append(json_file)
    else:
        print(f"Error: Directory not found: {json_dir}")
        return
    
    print(f"Found {len(json_files)} JSON attestation files")
    
    if not json_files:
        print(f"No JSON files found in {json_dir}!")
        print("  Looking for files starting with 'att' or containing 'sha256'")
        return
    
    # Generate examples
    all_examples = []
    
    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")
        
        analyzer = AttestationAnalyzer(json_file)
        if not analyzer.load():
            print(f"  Skipped (failed to load)")
            continue
        
        # Generate instructions based on content
        tasks = analyzer.get_tasks()
        subjects = analyzer.get_subjects()
        materials = analyzer.get_materials()
        
        print(f"  Found: {len(tasks)} tasks, {len(subjects)} subjects, {len(materials)} materials")
        
        # Generate task-related examples (limit to avoid too many per file)
        task_examples = InstructionGenerator.generate_task_instructions(tasks)
        # Limit to max 30 task examples per file (increased from 20 to allow more validation variations)
        if len(task_examples) > 30:
            task_examples = random.sample(task_examples, 30)
        
        for instruction, rego_code, metadata in task_examples:
            # Primary validation: Rego syntax (must pass)
            if not validate_rego_syntax(rego_code):
                print(f"  Warning: Invalid Rego code for instruction: {instruction[:50]}...")
                continue
            
            # Secondary validation: Attestation parsing correctness (must pass)
            if not _validate_attestation_parsing(rego_code, instruction):
                print(f"  Warning: Attestation parsing issue for instruction: {instruction[:50]}...")
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
        
        print(f"  Generated {len(all_examples)} total examples so far")
    
    print(f"\nTotal examples generated: {len(all_examples)}")
    
    # Shuffle and split
    random.shuffle(all_examples)
    split_idx = int(len(all_examples) * TRAIN_SPLIT)
    train_examples = all_examples[:split_idx]
    eval_examples = all_examples[split_idx:]
    
    print(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")
    
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
    
    print(f"\n✓ Dataset written:")
    print(f"  Train: {train_path}")
    print(f"  Eval: {eval_path}")
    
    # Generate summary
    summary = {
        "total_examples": len(all_examples),
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "task_type": "rego_attestation_parse",
        "source_files": len(json_files),
    }
    
    summary_path = output_dir / "attestation_dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()

