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
    def generate_task_status_check(task_name: str, status: str) -> str:
        """Generate Rego code to check task status."""
        code = f"""some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "{task_name}"
    task.status == "{status}\""""
        return RegoCodeGenerator._wrap_in_rule(code, "task_status_check", RegoCodeGenerator.USE_FULL_RULES, RegoCodeGenerator.USE_DENY_RULES)
    
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
    def generate_get_subject_digest(index: int = 0) -> str:
        """Generate Rego code to get subject digest."""
        code = f"""some att in input.attestations
    att.statement.subject[{index}].digest.sha256 == digest"""
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
        code = f"""tasks_with_status := [task |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.status == "{status}"
]"""
        return RegoCodeGenerator._wrap_in_rule(code, "tasks_with_status", RegoCodeGenerator.USE_FULL_RULES)


class InstructionGenerator:
    """Generates instruction templates and variations."""
    
    TASK_TEMPLATES = [
        "In an attestation, check all tasks for a task named '{task_name}'",
        "Find a task named '{task_name}' in the attestation",
        "Check if there is a task called '{task_name}'",
        "Look for a task with name '{task_name}' in the attestation",
    ]
    
    TASK_STATUS_TEMPLATES = [
        "Get the status of task '{task_name}'",
        "Check the status of task '{task_name}'",
        "Find the status for task '{task_name}'",
    ]
    
    LIST_TASKS_TEMPLATES = [
        "List all task names in the attestation",
        "Get all task names from the attestation",
        "Find all task names",
    ]
    
    TASK_RESULTS_TEMPLATES = [
        "Get all results from task '{task_name}'",
        "Find all results for task '{task_name}'",
        "List the results from task '{task_name}'",
    ]
    
    TASK_BUNDLE_TEMPLATES = [
        "Get the bundle reference for task '{task_name}'",
        "Find the bundle used by task '{task_name}'",
        "Get the bundle for task '{task_name}'",
    ]
    
    SUBJECT_DIGEST_TEMPLATES = [
        "Get the SHA256 digest of the first subject",
        "Find the digest of the first subject",
        "Get the first subject's SHA256 digest",
    ]
    
    LIST_SUBJECTS_TEMPLATES = [
        "List all subject names in the attestation",
        "Get all subject names",
        "Find all subject names",
    ]
    
    MATERIAL_TEMPLATES = [
        "Check if material exists for git repo '{uri}' and commit '{commit}'",
        "Find material with URI '{uri}' and commit '{commit}'",
        "Check for material matching URI '{uri}' and commit '{commit}'",
    ]
    
    TASK_STATUS_FILTER_TEMPLATES = [
        "Find all tasks with status '{status}'",
        "Get all tasks that have status '{status}'",
        "List tasks with status '{status}'",
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
                template = random.choice(InstructionGenerator.TASK_STATUS_TEMPLATES)
                instruction = template.format(task_name=task_name)
                rego_code = RegoCodeGenerator.generate_task_status_check(task_name, task["status"])
                examples.append((instruction, rego_code, {"task_name": task_name, "status": task["status"]}))
            
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
                template = random.choice(InstructionGenerator.TASK_BUNDLE_TEMPLATES)
                instruction = template.format(task_name=task_name)
                rego_code = RegoCodeGenerator.generate_get_task_bundle(task_name)
                examples.append((instruction, rego_code, {"task_name": task_name}))
        
        # List all task names (once per attestation)
        if task_names:
            template = random.choice(InstructionGenerator.LIST_TASKS_TEMPLATES)
            instruction = template
            rego_code = RegoCodeGenerator.generate_list_task_names()
            examples.append((instruction, rego_code, {}))
        
        # Find tasks by status
        statuses = {task.get("status") for task in tasks if task.get("status")}
        for status in statuses:
            template = random.choice(InstructionGenerator.TASK_STATUS_FILTER_TEMPLATES)
            instruction = template.format(status=status)
            rego_code = RegoCodeGenerator.generate_find_tasks_by_status(status)
            examples.append((instruction, rego_code, {"status": status}))
        
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
        
        return examples
    
    @staticmethod
    def generate_material_instructions(materials: List[Dict]) -> List[Tuple[str, str, Dict]]:
        """Generate instructions for material-related queries."""
        examples = []
        
        for material in materials:
            uri = material.get("uri")
            digest = material.get("digest", {})
            commit = digest.get("sha1") or digest.get("sha256")
            
            if uri and commit:
                template = random.choice(InstructionGenerator.MATERIAL_TEMPLATES)
                instruction = template.format(uri=uri, commit=commit)
                rego_code = RegoCodeGenerator.generate_check_material(uri, commit)
                examples.append((instruction, rego_code, {"uri": uri, "commit": commit}))
        
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
    print("Generating attestation parsing training dataset...")
    print(f"Repo root: {REPO_ROOT}")
    
    # Check if opa is available
    if not check_opa_available():
        print("⚠ OPA binary not found - skipping Rego syntax validation")
    else:
        print("✓ OPA found - validating Rego syntax")
    
    # Find all JSON files in repo root
    json_files = []
    for json_file in REPO_ROOT.glob("*.json"):
        if json_file.name.startswith("att") or "sha256" in json_file.name:
            json_files.append(json_file)
    
    print(f"Found {len(json_files)} JSON attestation files")
    
    if not json_files:
        print("No JSON files found in repo root!")
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
        # Limit to max 15 task examples per file
        if len(task_examples) > 15:
            task_examples = random.sample(task_examples, 15)
        
        for instruction, rego_code, metadata in task_examples:
            # Validate Rego syntax
            if not validate_rego_syntax(rego_code):
                print(f"  Warning: Invalid Rego code for instruction: {instruction[:50]}...")
                continue
            
            example = ExampleBuilder.build_example(instruction, rego_code, analyzer, metadata)
            if example:
                all_examples.append(example)
        
        # Generate subject-related examples
        subject_examples = InstructionGenerator.generate_subject_instructions(subjects)
        for instruction, rego_code, metadata in subject_examples:
            if not validate_rego_syntax(rego_code):
                continue
            example = ExampleBuilder.build_example(instruction, rego_code, analyzer, metadata)
            if example:
                all_examples.append(example)
        
        # Generate material-related examples (limit to avoid too many)
        material_examples = InstructionGenerator.generate_material_instructions(materials[:3])  # Limit to first 3
        for instruction, rego_code, metadata in material_examples:
            if not validate_rego_syntax(rego_code):
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
    
    # Write output files
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

