# Navigation/Query Patterns Review and Implementation

## Review of Suggestions

The suggestions to add explicit "navigation/query" examples are **excellent** and address an important gap in the training dataset. Here's what was requested and what was implemented:

## What Was Requested

1. **Add explicit "navigation/query" examples** - Patterns where rules return data directly (not just boolean checks)
2. **Include buildah use case** - Specific examples for tasks named "buildah" or similar
3. **Three variant patterns**:
   - Return tasks directly: `tasks_named_buildah := [task | ...]`
   - Return just names: `buildah_task_names := {task.name | ...}`
   - Return (name, status) objects: `buildah_task_statuses := [{"name": ..., "status": ...} | ...]`
4. **Different phrasings** - Multiple instruction variations for the same query pattern

## What Was Implemented

### 1. New Rego Code Generators

Added three new methods to `RegoCodeGenerator`:

- **`generate_return_tasks_by_name(task_name)`**
  - Returns: `tasks_named_X := [task | ...]`
  - Returns the full task objects matching the name

- **`generate_return_task_names_by_name(task_name)`**
  - Returns: `X_task_names := {task.name | ...}`
  - Returns just the names as a set

- **`generate_return_task_statuses_by_name(task_name)`**
  - Returns: `X_task_statuses := [{"name": task.name, "status": task.status} | ...]`
  - Returns structured objects with name and status

### 2. New Instruction Templates

Added three new template groups to `InstructionGenerator`:

- **`RETURN_TASKS_BY_NAME_TEMPLATES`** (7 variations)
  - "Return a list of all tasks named '{task_name}'"
  - "List every task named '{task_name}' across all attestations"
  - "Show me all tasks named '{task_name}'"
  - etc.

- **`RETURN_TASK_NAMES_BY_NAME_TEMPLATES`** (5 variations)
  - "Return just the names of all tasks named '{task_name}'"
  - "List the names of all tasks named '{task_name}'"
  - etc.

- **`RETURN_TASK_STATUSES_BY_NAME_TEMPLATES`** (6 variations)
  - "Show me all tasks named '{task_name}' and their status"
  - "Return all tasks named '{task_name}' with their name and status"
  - etc.

### 3. Integration into Dataset Generation

The new patterns are integrated into `generate_task_instructions()`:

- **Return tasks directly**: 30% probability per task name
- **Return just names**: 20% probability per task name
- **Return (name, status) objects**: 20% probability per task name (only if task has status)

This ensures balanced coverage without overwhelming the dataset.

### 4. Buildah Use Case Coverage

The implementation automatically covers buildah use cases:
- Tasks with names like "build-images", "build-container" (which use buildah bundles) will automatically get these patterns
- The variable name sanitization (`replace("-", "_")`) handles hyphenated task names correctly
- Examples will be generated for any task name found in attestations

## Example Output

### Pattern 1: Return Tasks Directly
**Instruction**: "Return a list of all tasks named 'buildah'"

**Output Rego**:
```rego
package attestation_check

import rego.v1

tasks_named_buildah := [task |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "buildah"
]
```

### Pattern 2: Return Just Names
**Instruction**: "Return just the names of all tasks named 'buildah'"

**Output Rego**:
```rego
package attestation_check

import rego.v1

buildah_task_names := {task.name |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "buildah"
}
```

### Pattern 3: Return (Name, Status) Objects
**Instruction**: "Show me all tasks named 'buildah' and their status"

**Output Rego**:
```rego
package attestation_check

import rego.v1

buildah_task_statuses := [{"name": task.name, "status": task.status} |
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "buildah"
]
```

## Benefits

1. **Teaches Navigation Patterns**: Model learns to return data structures, not just boolean checks
2. **Covers Real Use Cases**: Matches the buildah use case and similar query patterns
3. **Multiple Variations**: Different phrasings reinforce the same pattern
4. **Balanced Dataset**: Probability-based generation ensures good coverage without over-representation

## Technical Details

- **Variable Name Sanitization**: Task names with hyphens are converted to underscores for valid Rego identifiers
- **Automatic Generation**: Patterns are generated for all task names found in attestations
- **Context-Aware**: Only generates status-based patterns when tasks have status fields

## Next Steps

1. **Regenerate Dataset**: Run `generate_attestation_dataset.py` to create new training data with these patterns
2. **Verify Examples**: Check that buildah-related tasks get the new patterns
3. **Test Model**: Fine-tune and test to ensure model learns these navigation patterns correctly

## Summary

✅ All requested patterns implemented
✅ Buildah use case automatically covered
✅ Multiple instruction variations included
✅ Balanced integration into dataset generation
✅ Ready for dataset regeneration

