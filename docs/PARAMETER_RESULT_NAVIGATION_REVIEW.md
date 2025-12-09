# Parameter and Result Navigation Patterns Review and Implementation

## Review of Suggestions

The suggestions to add explicit parameter and result navigation patterns are **excellent** and address important gaps in teaching the model first-class navigation of these structures.

## What Was Requested

1. **First-class parameter navigation** - Explicit patterns for accessing `task.ref.params[]`
   - Example: "Find the value of the bundle parameter for the build-images task"
   - Pattern: `some param in task.ref.params; param.name == 'X'; value := param.value`

2. **Enhanced result navigation** - More variations and phrasings
   - "Return all results for task X" (already exists, but needs more phrasings)
   - "Return the names of all result keys for task X" (NEW)
   - "Return the exitCode result for task X if present" (NEW)

## What Was Implemented

### 1. New Rego Code Generators

Added three new methods to `RegoCodeGenerator`:

#### `generate_get_param_value(task_name, param_name)`
- **Purpose**: First-class parameter navigation
- **Returns**: `{task_name}_{param_name} := value if { ... }`
- **Pattern**: Explicitly navigates `task.ref.params[]` array
- **Example**:
```rego
build_images_bundle := value if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "build-images"
    some param in task.ref.params
    param.name == "bundle"
    value := param.value
}
```

#### `generate_get_result_names(task_name)`
- **Purpose**: Get all result key names
- **Returns**: `result_names := {result.name | ...}`
- **Pattern**: Set comprehension to extract result names
- **Example**:
```rego
result_names := {result.name |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "clamav-scan"
    some result in task.results
}
```

#### `generate_get_result_by_name(task_name, result_name)`
- **Purpose**: Get specific result value by name (e.g., exitCode)
- **Returns**: `{task_name}_{result_name} := value if { ... }`
- **Pattern**: Conditional rule to extract specific result
- **Example**:
```rego
clamav_scan_exitCode := value if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "clamav-scan"
    some result in task.results
    result.name == "exitCode"
    value := result.value
}
```

### 2. New Instruction Templates

Added three new template groups:

#### `PARAM_VALUE_TEMPLATES` (7 variations)
- "Find the value of the {param_name} parameter for task '{task_name}'"
- "Get the {param_name} parameter value for task '{task_name}'"
- "Return the value of the {param_name} parameter for task '{task_name}'"
- etc.

#### `RESULT_NAMES_TEMPLATES` (6 variations)
- "Return the names of all result keys for task '{task_name}'"
- "Get all result names from task '{task_name}'"
- "List the names of all results for task '{task_name}'"
- etc.

#### `RESULT_BY_NAME_TEMPLATES` (6 variations)
- "Return the {result_name} result for task '{task_name}' if present"
- "Get the {result_name} result value for task '{task_name}'"
- "Find the {result_name} result for task '{task_name}'"
- etc.

### 3. Enhanced Existing Templates

Added 3 more variations to `TASK_RESULTS_TEMPLATES`:
- "Return all results for task '{task_name}'"
- "Get all result objects from task '{task_name}'"
- "List every result from task '{task_name}'"

### 4. Integration into Dataset Generation

#### Parameter Navigation
- **Location**: After bundle handling in `generate_task_instructions()`
- **Frequency**: 40% probability per parameter (excluding bundle which is handled separately)
- **Coverage**: Automatically generates examples for all parameters found in `task.ref.params[]`
- **Smart filtering**: Skips "bundle" parameter to avoid duplication with bundle-specific queries

#### Result Navigation
- **Result names**: 40% probability when task has results
- **Specific results**: 40% probability, with special handling for "exitCode" (30% chance to specifically request it if present)
- **Coverage**: Generates examples for all result names found in task results

### 5. Enhanced Schema Headers

Updated `_generate_schema_header()` to provide query-specific guidance:

- **Parameter queries**: Shows `ref.params[]` structure and access pattern
- **Result name queries**: Shows `results[]` structure and name extraction pattern
- **Specific result queries**: Shows `results[]` structure and value extraction pattern

## Example Outputs

### Parameter Navigation
**Instruction**: "Find the value of the bundle parameter for the build-images task"

**Output Rego**:
```rego
package attestation_check

import rego.v1

build_images_bundle := value if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "build-images"
    some param in task.ref.params
    param.name == "bundle"
    value := param.value
}
```

### Result Names
**Instruction**: "Return the names of all result keys for task 'clamav-scan'"

**Output Rego**:
```rego
package attestation_check

import rego.v1

result_names := {result.name |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "clamav-scan"
    some result in task.results
}
```

### Specific Result (exitCode)
**Instruction**: "Return the exitCode result for task 'clamav-scan' if present"

**Output Rego**:
```rego
package attestation_check

import rego.v1

clamav_scan_exitCode := value if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "clamav-scan"
    some result in task.results
    result.name == "exitCode"
    value := result.value
}
```

## Benefits

1. **First-Class Parameter Access**: Model learns to navigate `ref.params[]` explicitly, not just as a fallback for bundle
2. **Result Navigation**: Model learns multiple ways to access results (all, names, specific)
3. **exitCode Priority**: Special handling ensures exitCode is well-covered (common use case)
4. **Multiple Phrasings**: Different instruction variations reinforce the same patterns
5. **Balanced Coverage**: Probability-based generation ensures good coverage without over-representation

## Technical Details

- **Variable Name Sanitization**: Parameter and result names with hyphens/special chars are converted to underscores
- **Automatic Discovery**: Patterns generated for all parameters and results found in attestations
- **Smart Filtering**: Bundle parameter excluded from generic parameter navigation (handled separately)
- **exitCode Bias**: 30% chance to specifically request exitCode when present (common use case)

## Comparison with Existing Patterns

### Before
- Bundle access: Only via `if/else` pattern (ref.bundle OR ref.params)
- Results: Only "get all results" pattern
- Parameters: Not explicitly navigated as first-class structure

### After
- Bundle access: Still has `if/else` pattern, but also explicit parameter navigation
- Results: Three patterns (all results, result names, specific result)
- Parameters: First-class navigation for any parameter name

## Next Steps

1. **Regenerate Dataset**: Run `generate_attestation_dataset.py` to create new training data
2. **Verify Examples**: Check that parameter and result navigation patterns appear for relevant tasks
3. **Test Model**: Fine-tune and test to ensure model learns these navigation patterns correctly

## Summary

✅ First-class parameter navigation implemented
✅ Result names pattern added
✅ Specific result by name pattern added (with exitCode priority)
✅ Multiple instruction variations included
✅ Enhanced schema headers for better guidance
✅ Balanced integration into dataset generation
✅ Ready for dataset regeneration


