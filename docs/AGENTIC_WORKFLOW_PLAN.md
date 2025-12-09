# Agentic Workflow Plan for `infer_policy.py`

## Overview

Transform the current inference script into an agentic workflow that follows a structured **Plan → Implement → Check → Repair** loop. This will make the inference process more transparent, reliable, and capable of handling complex tasks.

## Key Feature: Execution Validation with Real Attestation Data

**NEW**: The workflow now includes execution validation that tests generated Rego code against real attestation JSON files from your repository. This catches runtime errors (undefined references, type mismatches, navigation errors) that syntax checking alone cannot detect.

**How it works:**
1. Automatically discovers attestation JSON files in the repo root
2. Wraps them in the expected `input.attestations` format
3. Uses `opa eval` to execute the generated Rego code against the data
4. Reports any runtime errors or execution failures
5. Only runs for attestation-related instructions (auto-detected)

**Benefits:**
- Catches errors that pass syntax validation but fail at runtime
- Validates code works with real-world data structures
- Increases confidence that generated code will work in production
- Helps identify navigation path errors (e.g., wrong field names)

## Current State Analysis

### Existing Capabilities
- Basic validation loop in `generate_response_with_validation()`
- Syntax validation via `rego_validator.validate_rego_syntax()`
- Code extraction from model responses
- Simple error correction prompts
- Dynamic context building via `SmartContextBuilder`

### Limitations
- No explicit planning phase
- Limited validation (only syntax)
- No semantic/logical validation
- No style guide compliance checking
- Single-pass error correction without structured feedback
- No visibility into the agent's reasoning process

## Proposed Agentic Workflow

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic Inference Loop                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────┐
        │  1. PLANNING PHASE                   │
        │  - Analyze instruction               │
        │  - Identify requirements             │
        │  - Plan approach                     │
        │  - Select helpers/patterns           │
        └─────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │  2. IMPLEMENTATION PHASE             │
        │  - Generate Rego code                │
        │  - Follow plan                       │
        │  - Use context helpers               │
        └─────────────────────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │  3. CHECKING PHASE                   │
        │  ┌───────────────────────────────┐   │
        │  │ 3a. Syntax Check             │   │
        │  │    - opa parse               │   │
        │  └───────────────────────────────┘   │
        │  ┌───────────────────────────────┐   │
        │  │ 3b. Semantic Check            │   │
        │  │    - Instruction compliance    │   │
        │  │    - Structure validation      │   │
        │  └───────────────────────────────┘   │
        │  ┌───────────────────────────────┐   │
        │  │ 3c. Style Check               │   │
        │  │    - Regal lint               │   │
        │  │    - Style guide compliance   │   │
        │  └───────────────────────────────┘   │
        │  ┌───────────────────────────────┐   │
        │  │ 3d. Execution Check           │   │
        │  │    - Test against real data   │   │
        │  │    - Runtime validation       │   │
        │  │    - Verify no crashes        │   │
        │  └───────────────────────────────┘   │
        └─────────────────────────────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ All checks    │
              │ passed?       │
              └───────────────┘
                  │        │
            YES   │        │   NO
                  │        │
                  ▼        ▼
        ┌─────────────┐  ┌─────────────────────┐
        │  4. SUCCESS │  │  5. REPAIR PHASE    │
        │  - Return   │  │  - Analyze errors   │
        │  - Format   │  │  - Prioritize fixes │
        │  - Log      │  │  - Generate fix     │
        └─────────────┘  └─────────────────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │ Max iterations   │
                        │ reached?         │
                        └──────────────────┘
                            │        │
                      NO    │        │   YES
                            │        │
                            └────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Return with errors     │
                    │ or best attempt        │
                    └───────────────────────┘
```

## Implementation Plan

### Phase 1: Core Agentic Infrastructure

#### 1.1 Create Agent State Management
**File**: `qwen2.5_model/infer_policy.py` (new classes)

```python
@dataclass
class AgentState:
    """Tracks the agent's state through the workflow."""
    iteration: int = 0
    plan: Optional[str] = None
    implementation: Optional[str] = None
    syntax_valid: bool = False
    semantic_valid: bool = False
    execution_valid: bool = False
    style_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)  # Conversation history
```

#### 1.2 Create Planning Function
**Function**: `generate_plan()`

```python
def generate_plan(
    tokenizer, model, device, instruction: str, context: str = None
) -> str:
    """Generate a structured plan for implementing the instruction.
    
    Returns a plan that includes:
    - Understanding of requirements
    - Approach/strategy
    - Relevant helpers/patterns to use
    - Expected structure
    """
    planning_prompt = f"""Analyze this instruction and create a plan for implementing it.

Instruction: {instruction}

{context if context else ""}

Create a structured plan that includes:
1. What the instruction is asking for
2. What Rego patterns/constructs are needed
3. Which helpers from the context should be used
4. The expected rule structure (deny/warn/allow, conditions, etc.)
5. Any potential challenges or considerations

Provide a clear, structured plan."""
    
    messages = [
        {"role": "system", "content": "You are a Rego policy planning assistant. Create clear, structured plans."},
        {"role": "user", "content": planning_prompt}
    ]
    
    plan = generate_response(tokenizer, model, device, messages, max_tokens=512, temperature=0.3)
    return plan
```

#### 1.3 Create Multi-Layer Validation System
**Function**: `check_code_comprehensively()`

```python
def check_code_comprehensively(
    code: str,
    instruction: str,
    package: str = None,
    imports: List[str] = None,
    include_style: bool = True,
    include_execution: bool = True,
    attestation_files: List[Path] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Perform comprehensive validation across multiple layers.
    
    Returns:
        (is_valid, validation_results)
        validation_results contains:
        - syntax: {valid, error_msg, formatted_code}
        - semantic: {valid, issues: []}
        - style: {valid, violations: []}
        - execution: {valid, errors: [], tested_files: []}
    """
    results = {
        "syntax": {"valid": False, "error_msg": "", "formatted_code": code},
        "semantic": {"valid": False, "issues": []},
        "style": {"valid": True, "violations": []},
        "execution": {"valid": True, "errors": [], "tested_files": []}
    }
    
    # 1. Syntax check
    from rego_validator import validate_rego_syntax
    is_valid, formatted_code, error_msg = validate_rego_syntax(
        code, package=package or "", imports=imports or []
    )
    results["syntax"] = {
        "valid": is_valid,
        "error_msg": error_msg,
        "formatted_code": formatted_code
    }
    
    if not is_valid:
        return False, results
    
    # 2. Semantic check
    semantic_issues = check_semantic_compliance(code, instruction)
    results["semantic"] = {
        "valid": len(semantic_issues) == 0,
        "issues": semantic_issues
    }
    
    # 3. Style check (optional)
    if include_style:
        style_violations = check_style_compliance(code)
        results["style"] = {
            "valid": len(style_violations) == 0,
            "violations": style_violations
        }
    
    # 4. Execution check (optional, only for attestation-related code)
    if include_execution and attestation_files:
        execution_errors, tested_files = check_execution_against_attestations(
            code, attestation_files, package=package, imports=imports
        )
        results["execution"] = {
            "valid": len(execution_errors) == 0,
            "errors": execution_errors,
            "tested_files": tested_files
        }
    
    # Overall validity: syntax, semantic, and execution must pass; style is optional
    overall_valid = (
        results["syntax"]["valid"] and 
        results["semantic"]["valid"] and
        results["execution"]["valid"]
    )
    
    return overall_valid, results
```

#### 1.4 Create Semantic Validation
**Function**: `check_semantic_compliance()`

```python
def check_semantic_compliance(code: str, instruction: str) -> List[str]:
    """Check if code semantically matches the instruction.
    
    Validates:
    - Required rule names/function names are present
    - Expected patterns are used (e.g., 'every' for FOR ALL)
    - Required navigation paths exist
    - Return values match expectations
    """
    issues = []
    instruction_lower = instruction.lower()
    code_lower = code.lower()
    
    # Check for required rule/function names
    quoted_names = re.findall(r"['\"]([^'\"]+)['\"]", instruction)
    for name in quoted_names:
        # Check if this looks like a rule/function name requirement
        context = instruction[max(0, instruction.find(f"'{name}'")-50):instruction.find(f"'{name}'")]
        if any(kw in context.lower() for kw in ['named', 'called', 'rule', 'function']):
            if name not in code:
                issues.append(f"Required name '{name}' not found in code")
    
    # Check for FOR ALL patterns
    if any(phrase in instruction_lower for phrase in ['all', 'every', 'for all']):
        if 'every' not in code_lower and 'not some' not in code_lower:
            issues.append("FOR ALL pattern expected but 'every' or 'not some' not found")
    
    # Check for attestation parsing patterns
    if 'attestation' in instruction_lower:
        required_paths = []
        if 'task' in instruction_lower:
            required_paths.append('task')
        if 'subject' in instruction_lower:
            required_paths.append('subject')
        if 'material' in instruction_lower:
            required_paths.append('material')
        
        for path in required_paths:
            if path not in code_lower:
                issues.append(f"Expected to navigate to '{path}' but not found in code")
    
    return issues
```

#### 1.5 Create Style Validation
**Function**: `check_style_compliance()`

```python
def check_style_compliance(code: str) -> List[str]:
    """Check style guide compliance using Regal and custom checks."""
    violations = []
    
    # Try Regal lint
    try:
        regal_ok, regal_issues = validate_with_regal(code)
        if not regal_ok and regal_issues:
            violations.extend(regal_issues)
    except Exception:
        pass  # Regal not available
    
    # Custom style checks
    # Check for 'in' vs '==' for membership
    if re.search(r'==\s*["\']', code) and 'in {' in code:
        # Mixed usage, check if 'in' should be preferred
        pass  # Could add specific check
    
    # Check for 'every' usage
    if 'not some' in code and 'every' not in code:
        if re.search(r'not\s+some\s+.*\s+in\s+.*\s+\{.*\s+!=', code):
            violations.append("Consider using 'every' instead of 'not some' for FOR ALL queries")
    
    return violations
```

#### 1.6 Create Execution Validation Function
**Function**: `check_execution_against_attestations()`

```python
def check_execution_against_attestations(
    code: str,
    attestation_files: List[Path],
    package: str = None,
    imports: List[str] = None,
    max_files: int = 3
) -> Tuple[List[str], List[str]]:
    """Test Rego code against real attestation JSON files.
    
    Args:
        code: Rego code to test
        attestation_files: List of paths to attestation JSON files
        package: Package name for code
        imports: List of imports
        max_files: Maximum number of files to test (for performance)
        
    Returns:
        (list_of_errors, list_of_tested_file_names)
        
    Uses `opa eval` to execute the code against wrapped attestation data.
    Checks for runtime errors (undefined references, type errors, etc.).
    """
    errors = []
    tested_files = []
    
    # Limit number of files to test
    files_to_test = attestation_files[:max_files]
    
    # Determine OPA command (try 'ec opa' first, fall back to 'opa')
    opa_base = ["opa"]
    try:
        result = subprocess.run(
            ["ec", "opa", "--version"],
            capture_output=True,
            timeout=1,
            text=True
        )
        if result.returncode == 0:
            opa_base = ["ec", "opa"]
    except:
        pass
    
    # Build complete code with package/imports
    complete_code = code
    if package and f"package {package}" not in code:
        code_parts = [f"package {package}\n"]
        if imports:
            code_parts.append("import rego.v1\n")
            for imp in imports:
                if not imp.startswith("rego.v1") and f"import {imp}" not in code:
                    code_parts.append(f"import {imp}\n")
        code_parts.append("\n")
        code_parts.append(code)
        complete_code = "".join(code_parts)
    
    # Write Rego code to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as rego_file:
        rego_path = Path(rego_file.name)
        rego_file.write(complete_code)
        rego_file.flush()
    
    try:
        # Test against each attestation file
        for att_file in files_to_test:
            if not att_file.exists():
                continue
            
            tested_files.append(att_file.name)
            
            try:
                # Read and wrap attestation in expected format
                with open(att_file, 'r') as f:
                    att_data = json.load(f)
                
                # Wrap in input.attestations format
                # If file already has attestations array, use it; otherwise wrap single object
                if isinstance(att_data, list):
                    wrapped_input = {"attestations": att_data}
                elif "attestations" in att_data:
                    wrapped_input = att_data
                else:
                    # Single attestation object, wrap it
                    wrapped_input = {"attestations": [att_data]}
                
                # Write input to temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as input_file:
                    input_path = Path(input_file.name)
                    json.dump(wrapped_input, input_file)
                    input_file.flush()
                
                try:
                    # Evaluate code against input
                    # Query for 'deny' or 'warn' or any rule that might exist
                    # We'll try multiple queries to see if code executes
                    queries_to_try = [
                        "data.deny",
                        "data.warn", 
                        "data.allow",
                        "data"  # Catch-all to see if anything evaluates
                    ]
                    
                    execution_errors = []
                    any_query_worked = False
                    
                    for query in queries_to_try:
                        try:
                            result = subprocess.run(
                                opa_base + [
                                    "eval",
                                    "--data", str(rego_path),
                                    "--input", str(input_path),
                                    query,
                                    "--format", "json"
                                ],
                                capture_output=True,
                                timeout=5,
                                text=True
                            )
                            
                            if result.returncode == 0:
                                any_query_worked = True
                                break  # At least one query worked
                            else:
                                # Check if it's a "not found" error (acceptable)
                                error_output = result.stderr or result.stdout
                                if "undefined" not in error_output.lower():
                                    # Real error, not just "rule not found"
                                    execution_errors.append(
                                        f"Query '{query}' failed: {error_output[:200]}"
                                    )
                        except subprocess.TimeoutExpired:
                            execution_errors.append(f"Query '{query}' timed out")
                        except Exception as e:
                            execution_errors.append(f"Query '{query}' error: {e}")
                    
                    # If no queries worked and we have errors, report them
                    if not any_query_worked and execution_errors:
                        # Only report if it's a real execution error
                        # (undefined rules are okay - code might not define deny/warn)
                        real_errors = [e for e in execution_errors if "undefined" not in e.lower()]
                        if real_errors:
                            errors.append(
                                f"{att_file.name}: {real_errors[0]}"
                            )
                
                finally:
                    try:
                        input_path.unlink()
                    except:
                        pass
            
            except json.JSONDecodeError as e:
                errors.append(f"{att_file.name}: Invalid JSON - {e}")
            except Exception as e:
                errors.append(f"{att_file.name}: Error loading file - {e}")
    
    finally:
        try:
            rego_path.unlink()
        except:
            pass
    
    return errors, tested_files
```

#### 1.7 Create Repair Function
**Function**: `generate_repair()`

```python
def generate_repair(
    tokenizer, model, device,
    instruction: str,
    plan: str,
    current_code: str,
    validation_results: Dict[str, Any],
    iteration: int,
    max_iterations: int
) -> str:
    """Generate a repair for the code based on validation results.
    
    Prioritizes fixes:
    1. Syntax errors (must fix)
    2. Semantic issues (must fix)
    3. Style violations (should fix if possible)
    """
    # Build error summary
    error_summary = []
    
    if not validation_results["syntax"]["valid"]:
        error_summary.append(f"SYNTAX ERROR:\n{validation_results['syntax']['error_msg']}")
    
    if not validation_results["semantic"]["valid"]:
        semantic_issues = "\n".join(f"- {issue}" for issue in validation_results["semantic"]["issues"])
        error_summary.append(f"SEMANTIC ISSUES:\n{semantic_issues}")
    
    if not validation_results["style"]["valid"]:
        style_violations = "\n".join(f"- {violation}" for violation in validation_results["style"]["violations"])
        error_summary.append(f"STYLE VIOLATIONS:\n{style_violations}")
    
    repair_prompt = f"""The generated Rego code has validation errors. Please fix them.

Original instruction:
{instruction}

Original plan:
{plan}

Current code (iteration {iteration}/{max_iterations}):
```rego
{current_code}
```

Validation errors:
{chr(10).join(error_summary)}

Please provide the corrected Rego code that fixes all syntax and semantic errors.
Prioritize fixing syntax errors first, then semantic issues.
Style violations can be addressed if time permits."""
    
    messages = [
        {"role": "system", "content": QWEN_SYSTEM_PROMPT_ATTESTATION if 'attestation' in instruction.lower() else QWEN_SYSTEM_PROMPT},
        {"role": "user", "content": f"Instruction: {instruction}\n\nPlan: {plan}"},
        {"role": "assistant", "content": f"```rego\n{current_code}\n```"},
        {"role": "user", "content": repair_prompt}
    ]
    
    repair = generate_response(tokenizer, model, device, messages, max_tokens=1024, temperature=0.5)
    return repair
```

### Phase 2: Main Agentic Workflow Function

#### 2.1 Create Agentic Inference Function
**Function**: `agentic_inference()`

```python
def agentic_inference(
    tokenizer, model, device,
    instruction: str,
    context: str = None,
    package: str = None,
    imports: List[str] = None,
    max_iterations: int = 5,
    include_planning: bool = True,
    include_style_check: bool = True,
    include_execution_check: bool = True,
    attestation_files: List[Path] = None,
    verbose: bool = True
) -> Tuple[str, AgentState]:
    """Run agentic inference with Plan → Implement → Check → Repair loop.
    
    Returns:
        (final_code, agent_state)
    """
    state = AgentState()
    
    if verbose:
        print("=" * 60)
        print("AGENTIC INFERENCE WORKFLOW")
        print("=" * 60)
        print()
    
    # PHASE 1: PLANNING
    if include_planning:
        if verbose:
            print("📋 Phase 1: Planning...")
        try:
            state.plan = generate_plan(tokenizer, model, device, instruction, context)
            if verbose:
                print("✓ Plan generated")
                print("-" * 60)
                print(state.plan)
                print("-" * 60)
                print()
        except Exception as e:
            if verbose:
                print(f"⚠ Planning failed: {e}, continuing without plan")
            state.plan = None
    
    # PHASE 2-5: IMPLEMENT → CHECK → REPAIR LOOP
    while state.iteration < max_iterations:
        state.iteration += 1
        
        if verbose:
            print(f"🔄 Iteration {state.iteration}/{max_iterations}")
        
        # PHASE 2: IMPLEMENTATION
        if verbose:
            print("  📝 Phase 2: Implementation...")
        
        # Build messages for implementation
        messages = build_implementation_messages(instruction, context, state.plan)
        implementation_response = generate_response(
            tokenizer, model, device, messages, max_tokens=1024, temperature=0.7
        )
        
        # Extract code
        from rego_validator import extract_rego_code
        code = extract_rego_code(implementation_response)
        
        if not code:
            if verbose:
                print("  ⚠ No Rego code found in response")
            state.errors.append("No Rego code found in model response")
            if state.iteration == 1:
                # First iteration, return the response as-is
                return implementation_response, state
            continue
        
        state.implementation = code
        
        # PHASE 3: CHECKING
        if verbose:
            print("  🔍 Phase 3: Checking...")
        
        is_valid, validation_results = check_code_comprehensively(
            code, instruction, package, imports, 
            include_style=include_style_check,
            include_execution=include_execution_check,
            attestation_files=attestation_files
        )
        
        state.syntax_valid = validation_results["syntax"]["valid"]
        state.semantic_valid = validation_results["semantic"]["valid"]
        state.style_valid = validation_results["style"]["valid"]
        state.execution_valid = validation_results["execution"]["valid"]
        
        # Collect errors and warnings
        if not state.syntax_valid:
            state.errors.append(f"Syntax: {validation_results['syntax']['error_msg']}")
        if not state.semantic_valid:
            state.errors.extend(validation_results["semantic"]["issues"])
        if not state.execution_valid:
            state.errors.extend(validation_results["execution"]["errors"])
        if not state.style_valid:
            state.warnings.extend(validation_results["style"]["violations"])
        
        if verbose:
            print(f"    Syntax: {'✓' if state.syntax_valid else '✗'}")
            print(f"    Semantic: {'✓' if state.semantic_valid else '✗'}")
            print(f"    Execution: {'✓' if state.execution_valid else '✗'}")
            if validation_results["execution"].get("tested_files"):
                print(f"      (tested against {len(validation_results['execution']['tested_files'])} attestation files)")
            print(f"    Style: {'✓' if state.style_valid else '⚠'}")
        
        # PHASE 4: SUCCESS
        if is_valid:
            if verbose:
                print("  ✓ All checks passed!")
                print()
            
            # Use formatted code if available
            final_code = validation_results["syntax"]["formatted_code"]
            return final_code, state
        
        # PHASE 5: REPAIR
        if state.iteration < max_iterations:
            if verbose:
                print("  🔧 Phase 5: Repairing...")
            
            try:
                repair_response = generate_repair(
                    tokenizer, model, device,
                    instruction, state.plan or "",
                    code, validation_results,
                    state.iteration, max_iterations
                )
                
                # Extract repaired code
                repaired_code = extract_rego_code(repair_response)
                if repaired_code:
                    state.implementation = repaired_code
                    # Continue loop with repaired code
                    continue
                else:
                    if verbose:
                        print("  ⚠ No code found in repair response")
                    state.errors.append("Repair response contained no code")
            except Exception as e:
                if verbose:
                    print(f"  ⚠ Repair failed: {e}")
                state.errors.append(f"Repair failed: {e}")
        else:
            if verbose:
                print("  ⚠ Max iterations reached")
            print()
    
    # Max iterations reached, return best attempt
    if verbose:
        print("=" * 60)
        print("⚠ Max iterations reached. Returning best attempt.")
        print("=" * 60)
    
    final_code = state.implementation or ""
    return final_code, state
```

#### 2.2 Helper Function: Build Implementation Messages
**Function**: `build_implementation_messages()`

```python
def build_implementation_messages(
    instruction: str, context: str = None, plan: str = None
) -> List[Dict]:
    """Build messages for implementation phase."""
    # Detect task type
    is_attestation_task = any(kw in instruction.lower() for kw in [
        'attestation', 'task', 'subject', 'material'
    ])
    
    system_prompt = (
        QWEN_SYSTEM_PROMPT_ATTESTATION if is_attestation_task 
        else QWEN_SYSTEM_PROMPT
    )
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # Build user content
    parts = []
    if context:
        parts.append(context)
    if plan:
        parts.append(f"Plan:\n{plan}")
    parts.append(f"Instruction: {instruction}")
    
    user_content = "\n\n".join(parts)
    messages.append({"role": "user", "content": user_content})
    
    return messages
```

### Phase 3: Integration with Existing Code

#### 3.1 Add Attestation File Discovery
**Function**: `find_attestation_files()`

```python
def find_attestation_files(repo_root: Path, max_files: int = 5) -> List[Path]:
    """Find attestation JSON files in the repository root.
    
    Looks for files matching patterns like:
    - *.json files in root
    - Files with 'attestation' in name
    - Files from quay.io (likely attestations)
    
    Returns:
        List of Path objects to attestation files
    """
    attestation_files = []
    
    # Look for JSON files in root
    for json_file in repo_root.glob("*.json"):
        # Skip dataset/summary files
        if any(skip in json_file.name.lower() for skip in [
            "dataset", "summary", "eval", "train"
        ]):
            continue
        
        # Check if it looks like an attestation (has _type or subject or predicate)
        try:
            with open(json_file, 'r') as f:
                content = f.read(1000)  # Read first 1KB
                if any(marker in content for marker in [
                    '"subject"', '"predicate"', '"_type"', '"attestations"',
                    '"buildConfig"', '"tasks"'
                ]):
                    attestation_files.append(json_file)
        except:
            continue
        
        if len(attestation_files) >= max_files:
            break
    
    return attestation_files
```

#### 3.2 Update `single_inference()` Function
Replace the current implementation to use `agentic_inference()`:

```python
def single_inference(
    tokenizer, model, device, instruction, 
    context=None, builder=None, package=None, max_tokens=1024,
    validate=True, max_corrections=3, include_style_guide=False, 
    enhance_instruction=True, agentic=True, verbose=True
):
    """Run inference with optional agentic workflow."""
    
    if agentic:
        # Use new agentic workflow
        # Build context if builder available
        context_parts = []
        if include_style_guide:
            context_parts.append(STYLE_GUIDE_CONDENSED)
        
        if builder and not context:
            built_context = builder.build_context(instruction, package=package)
            context_parts.append(built_context)
        elif context:
            context_parts.append(context)
        
        combined_context = "\n\n".join(context_parts) if context_parts else None
        
        # Extract package/imports from context
        package_from_context = package
        imports_from_context = []
        if combined_context:
            package_match = re.search(r'package\s+(\S+)', combined_context)
            if package_match:
                package_from_context = package_match.group(1)
            import_matches = re.findall(r'import\s+([^\n]+)', combined_context)
            imports_from_context = [imp.strip() for imp in import_matches if imp.strip()]
        
        # Find attestation files if this is an attestation task
        attestation_files = None
        if include_execution_check and any(kw in instruction.lower() for kw in [
            'attestation', 'task', 'subject', 'material'
        ]):
            repo_root = find_repo_root()
            attestation_files = find_attestation_files(repo_root, max_files=3)
            if verbose and attestation_files:
                print(f"✓ Found {len(attestation_files)} attestation files for execution testing")
        
        final_code, state = agentic_inference(
            tokenizer, model, device,
            instruction,
            context=combined_context,
            package=package_from_context,
            imports=imports_from_context,
            max_iterations=max_corrections,
            include_planning=True,
            include_style_check=include_style_guide,
            include_execution_check=include_execution_check,
            attestation_files=attestation_files,
            verbose=verbose
        )
        
        # Format output
        print("\n" + "=" * 60)
        print("Final Result:")
        print("=" * 60)
        if "```" not in final_code:
            print(f"```rego\n{final_code}\n```")
        else:
            print(final_code)
        print("=" * 60)
        
        if verbose and state.errors:
            print("\nErrors encountered:")
            for error in state.errors:
                print(f"  - {error}")
        if verbose and state.warnings:
            print("\nWarnings:")
            for warning in state.warnings:
                print(f"  - {warning}")
    else:
        # Use existing workflow (backward compatibility)
        # ... existing code ...
```

#### 3.2 Update `interactive_chat()` Function
Add agentic mode option:

```python
def interactive_chat(
    tokenizer, model, device, builder=None, default_package=None, 
    validate=True, max_corrections=3, include_style_guide=False, 
    enhance_instruction=True, agentic=True, verbose=True
):
    """Interactive chat with optional agentic workflow."""
    
    # ... existing setup code ...
    
    while True:
        # ... get user input ...
        
        if agentic:
            # Use agentic workflow
            context_parts = []
            if include_style_guide:
                context_parts.append(STYLE_GUIDE_CONDENSED)
            
            if builder:
                built_context = builder.build_context(instruction, package=package)
                context_parts.append(built_context)
            
            combined_context = "\n\n".join(context_parts) if context_parts else None
            
            final_code, state = agentic_inference(
                tokenizer, model, device,
                instruction,
                context=combined_context,
                package=package or default_package,
                max_iterations=max_corrections,
                include_planning=True,
                include_style_check=include_style_guide,
                verbose=verbose
            )
            
            print(f"\nAssistant: {final_code}")
            
            if verbose and (state.errors or state.warnings):
                if state.errors:
                    print("\nErrors:")
                    for error in state.errors:
                        print(f"  - {error}")
                if state.warnings:
                    print("\nWarnings:")
                    for warning in state.warnings:
                        print(f"  - {warning}")
        else:
            # Use existing workflow
            # ... existing code ...
```

#### 3.3 Update Command-Line Arguments
Add new flags:

```python
parser.add_argument(
    "--agentic",
    action="store_true",
    default=True,  # Make it default
    help="Use agentic workflow (Plan → Implement → Check → Repair)",
)
parser.add_argument(
    "--no-agentic",
    action="store_false",
    dest="agentic",
    help="Disable agentic workflow, use simple validation loop",
)
parser.add_argument(
    "--no-planning",
    action="store_true",
    help="Skip planning phase in agentic workflow (faster but less structured)",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    default=True,
    help="Show detailed workflow progress",
)
parser.add_argument(
    "--quiet",
    action="store_false",
    dest="verbose",
    help="Minimal output",
)
```

### Phase 4: Additional Enhancements

#### 4.1 Add Regal Integration
Create helper to run Regal lint:

```python
def validate_with_regal(code: str) -> Tuple[bool, List[str]]:
    """Run Regal lint on code.
    
    Returns:
        (is_valid, list_of_violations)
    """
    violations = []
    
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.rego', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        result = subprocess.run(
            ["regal", "lint", "--format", "json", temp_file],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        Path(temp_file).unlink(missing_ok=True)
        
        if result.returncode != 0:
            try:
                lint_data = json.loads(result.stdout) if result.stdout else {}
                if "violations" in lint_data:
                    for violation in lint_data["violations"]:
                        violations.append(
                            f"{violation.get('title', 'Violation')}: "
                            f"{violation.get('description', '')}"
                        )
            except:
                violations.append("Regal lint found issues (could not parse)")
    except FileNotFoundError:
        # Regal not available
        return True, []
    except Exception:
        return True, []  # Fail gracefully
    
    return len(violations) == 0, violations
```

#### 4.2 Add State Logging
Optional logging of agent state for debugging:

```python
def log_agent_state(state: AgentState, output_file: Path = None):
    """Log agent state to file or stdout."""
    log_data = {
        "iteration": state.iteration,
        "plan": state.plan,
        "syntax_valid": state.syntax_valid,
        "semantic_valid": state.semantic_valid,
        "style_valid": state.style_valid,
        "errors": state.errors,
        "warnings": state.warnings
    }
    
    if output_file:
        import json
        with open(output_file, 'a') as f:
            json.dump(log_data, f, indent=2)
            f.write("\n")
    else:
        print(json.dumps(log_data, indent=2))
```

## Testing Strategy

### Unit Tests
1. Test planning phase with various instruction types
2. Test semantic validation with known patterns
3. Test style validation with Regal
4. Test repair generation with mock errors

### Integration Tests
1. End-to-end agentic workflow with simple instruction
2. Complex instruction requiring multiple repair iterations
3. Instruction that should pass on first try
4. Instruction that fails after max iterations

### Manual Testing
1. Test with real attestation parsing instructions
2. Test with policy rule generation instructions
3. Compare agentic vs non-agentic outputs
4. Verify verbose output is helpful

## Migration Path

1. **Phase 1**: Implement core functions alongside existing code
2. **Phase 2**: Add `--agentic` flag (default: False for backward compatibility)
3. **Phase 3**: Test thoroughly with various instructions
4. **Phase 4**: Make agentic mode default (`--no-agentic` to disable)
5. **Phase 5**: Deprecate old validation loop (keep for compatibility)

## Benefits

1. **Transparency**: Users can see the planning and reasoning process
2. **Reliability**: Multi-layer validation catches more issues
3. **Iterative Improvement**: Structured repair loop improves success rate
4. **Debugging**: Agent state provides insights into failures
5. **Extensibility**: Easy to add new validation layers or repair strategies
6. **Real-World Testing**: Execution validation against actual attestation data catches runtime errors that syntax checking misses
7. **Confidence**: Code that passes execution tests is more likely to work in production

## Potential Challenges

1. **Token Usage**: Planning phase adds tokens, but improves quality
2. **Latency**: Multiple validation passes increase time, but reduce failures
3. **Complexity**: More moving parts to maintain
4. **Regal Dependency**: Optional but recommended for style checking
5. **Execution Testing**: Requires OPA to be installed and attestation files to be available
6. **File Discovery**: Need to reliably find attestation files in the repo
7. **Performance**: Execution testing adds latency (but catches real errors)

## Future Enhancements

1. **Test Generation**: Auto-generate tests for generated code
2. **Performance Analysis**: Check for inefficient patterns
3. **Security Checks**: Validate against security best practices
4. **Context Learning**: Learn from repair patterns to improve planning
5. **Multi-Model**: Use different models for planning vs implementation

