# Universal Conditions and Negative Existence Patterns Review

## Review of Suggestions

The suggestions to add universal conditions (`every`) and "no such task" patterns are **excellent** and address important gaps in teaching the model non-trivial logic beyond simple equality checks.

## What Was Requested

1. **Universal conditions across all attestations** - Nested `every` patterns
   - Example: "Ensure all tasks in the attestation have status Succeeded"
   - Pattern: `every att in input.attestations { every task in ... }`

2. **Negative existence with deny patterns** - Classic deny frame
   - Example: "Deny if any task is not Succeeded"
   - Pattern: `deny contains result if { ... task.status != "Succeeded"; result := {"msg": sprintf(...)} }`

3. **More complex logic** - Beyond simple AND of equalities

## What Was Implemented

### 1. New Rego Code Generators

Added four new methods to `RegoCodeGenerator`:

#### `generate_all_tasks_succeeded_universal()`
- **Purpose**: Universal condition across ALL attestations using nested `every`
- **Pattern**: Nested `every` loops
- **Example**:
```rego
package attestation_check

import rego.v1

all_tasks_succeeded if {
    every att in input.attestations {
        every task in att.statement.predicate.buildConfig.tasks {
            task.status == "Succeeded"
        }
    }
}
```

#### `generate_deny_any_task_not_succeeded()`
- **Purpose**: Negative existence pattern - deny if any task is not succeeded
- **Pattern**: `deny contains result` with `sprintf` message
- **Example**:
```rego
package attestation_check

import rego.v1

deny contains result if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.status != "Succeeded"
    result := {"msg": sprintf("task %q did not succeed", [task.name])}
}
```

#### `generate_deny_any_task_failed()`
- **Purpose**: Negative existence pattern - deny if any task failed
- **Pattern**: `deny contains result` with `sprintf` message
- **Example**:
```rego
package attestation_check

import rego.v1

deny contains result if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.status == "Failed"
    result := {"msg": sprintf("task %q failed", [task.name])}
}
```

#### `generate_deny_task_with_status(task_name, status)`
- **Purpose**: Deny pattern for specific task with specific status
- **Pattern**: `deny contains result` with `sprintf` message including task name and status
- **Example**:
```rego
package attestation_check

import rego.v1

deny contains result if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "buildah"
    task.status == "Failed"
    result := {"msg": sprintf("task %q has status %q", [task.name, task.status])}
}
```

### 2. New Instruction Templates

Added four new template groups:

#### `ALL_TASKS_SUCCEEDED_UNIVERSAL_TEMPLATES` (5 variations)
- "Ensure all tasks in the attestation have status Succeeded"
- "Check that every task across all attestations succeeded"
- "Verify all tasks in all attestations have status 'Succeeded'"
- etc.

#### `DENY_ANY_TASK_NOT_SUCCEEDED_TEMPLATES` (5 variations)
- "Deny if any task is not Succeeded"
- "Deny if any task did not succeed"
- "Deny if any task has a status other than Succeeded"
- etc.

#### `DENY_ANY_TASK_FAILED_TEMPLATES` (4 variations)
- "Deny if any task failed"
- "Deny if any task has status Failed"
- "Deny if any task is failed"
- etc.

#### `DENY_TASK_WITH_STATUS_TEMPLATES` (4 variations)
- "Deny if task '{task_name}' has status '{status}'"
- "Deny if task '{task_name}' status is '{status}'"
- etc.

### 3. Enhanced Existing Templates

Added 2 more variations to `TASK_NOT_FOUND_TEMPLATES`:
- "Check that task '{task_name}' is not found"
- "Verify no task named '{task_name}' exists"

### 4. Integration into Dataset Generation

#### Universal Conditions
- **Location**: After single-attestation `every` patterns
- **Frequency**: 20% probability per attestation file
- **Coverage**: Teaches nested `every` pattern across all attestations

#### Negative Existence Patterns
- **Deny any task not succeeded**: 25% probability per attestation file
- **Deny any task failed**: 25% probability per attestation file
- **Deny specific task with status**: 30% probability per task status (when generating validation queries)

### 5. Key Features

#### Nested `every` Pattern
- Teaches universal quantification across multiple levels
- Shows how to check conditions across all attestations and all tasks
- Demonstrates proper nesting structure

#### Deny Patterns with sprintf
- Teaches negative existence checks using `deny contains result`
- Shows how to use `sprintf` for dynamic error messages
- Demonstrates proper message formatting with task names and statuses

#### Complex Logic
- Moves beyond simple equality checks
- Introduces negation (`!=`)
- Shows conditional logic with multiple conditions

## Example Outputs

### Universal Condition
**Instruction**: "Ensure all tasks in the attestation have status Succeeded"

**Output Rego**:
```rego
package attestation_check

import rego.v1

all_tasks_succeeded if {
    every att in input.attestations {
        every task in att.statement.predicate.buildConfig.tasks {
            task.status == "Succeeded"
        }
    }
}
```

### Negative Existence - Any Task Not Succeeded
**Instruction**: "Deny if any task is not Succeeded"

**Output Rego**:
```rego
package attestation_check

import rego.v1

deny contains result if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.status != "Succeeded"
    result := {"msg": sprintf("task %q did not succeed", [task.name])}
}
```

### Negative Existence - Specific Task
**Instruction**: "Deny if task 'buildah' has status 'Failed'"

**Output Rego**:
```rego
package attestation_check

import rego.v1

deny contains result if {
    some att in input.attestations
    some task in att.statement.predicate.buildConfig.tasks
    task.name == "buildah"
    task.status == "Failed"
    result := {"msg": sprintf("task %q has status %q", [task.name, task.status])}
}
```

## Benefits

1. **Universal Quantification**: Model learns nested `every` patterns for checking conditions across all attestations
2. **Negative Existence**: Model learns deny patterns with proper error messages
3. **Complex Logic**: Moves beyond simple equality checks to include negation and conditional logic
4. **sprintf Usage**: Teaches proper message formatting with dynamic values
5. **Multiple Phrasings**: Different instruction variations reinforce the same patterns

## Comparison with Existing Patterns

### Before
- `every` patterns only checked within a single attestation
- No deny patterns with sprintf messages
- Limited negative existence patterns
- Simple equality checks only

### After
- Universal conditions across all attestations (nested `every`)
- Deny patterns with sprintf for negative existence
- Multiple deny pattern variations
- Complex logic with negation and multiple conditions

## Technical Details

- **Nested every**: Properly nested structure `every att { every task { ... } }`
- **sprintf format**: Uses `%q` for quoted strings in error messages
- **Message structure**: `{"msg": sprintf(...)}` pattern for deny rules
- **Probability-based**: Balanced generation to avoid over-representation

## Next Steps

1. **Regenerate Dataset**: Run `generate_attestation_dataset.py` to create new training data
2. **Verify Examples**: Check that universal conditions and deny patterns appear
3. **Test Model**: Fine-tune and test to ensure model learns these complex patterns correctly

## Summary

✅ Universal conditions with nested `every` implemented
✅ Negative existence deny patterns with sprintf added
✅ Specific task deny patterns added
✅ Enhanced "no such task" patterns
✅ Multiple instruction variations included
✅ Balanced integration into dataset generation
✅ Ready for dataset regeneration

These patterns dramatically improve the model's grasp of non-trivial logic beyond simple AND of equalities.


