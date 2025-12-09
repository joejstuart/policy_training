# Natural Language Variety Review

## Review of Suggestions

The suggestion to add more natural language variety around the same code patterns is **excellent** and addresses an important gap in helping the model generalize navigation patterns to real-world debugging and exploration prompts.

## What Was Requested

For each "gesture" (code pattern), generate several natural language variants pointing to the exact same `output_code`. Examples:
- "Print all tasks named buildah."
- "List tasks with name buildah."
- "Return all buildah tasks."
- "Which tasks use the name buildah?"

This helps the model generalize navigation patterns to the kind of prompts users actually type when debugging or exploring.

## What Was Implemented

### Enhanced Template Groups

Added **70+ new natural language variations** across 12 key template groups:

#### 1. `RETURN_TASKS_BY_NAME_TEMPLATES` (+9 variations)
**Original**: 7 templates
**Added**:
- "Print all tasks named '{task_name}'"
- "List tasks with name '{task_name}'"
- "Return all {task_name} tasks"
- "Which tasks use the name '{task_name}'?"
- "Show all {task_name} tasks"
- "Give me all tasks named '{task_name}'"
- "What tasks are called '{task_name}'?"
- "Display all tasks with the name '{task_name}'"
- "Find every task that's named '{task_name}'"
- "I need all tasks named '{task_name}'"

**Total**: 17 templates (all map to same `tasks_named_X := [task | ...]` code)

#### 2. `TASK_TEMPLATES` (+8 variations)
**Original**: 8 templates
**Added**:
- "Is there a task called '{task_name}'?"
- "Do we have a task named '{task_name}'?"
- "Can you find task '{task_name}'?"
- "Look up task '{task_name}'"
- "See if '{task_name}' task exists"
- "Check for '{task_name}' task"
- "Does '{task_name}' exist as a task?"
- "Is '{task_name}' present in the tasks?"

**Total**: 16 templates (all map to same task existence check code)

#### 3. `TASK_STATUS_TEMPLATES` (+8 variations)
**Original**: 6 templates
**Added**:
- "What's the status of '{task_name}'?"
- "Show me the status for task '{task_name}'"
- "Tell me the status of '{task_name}'"
- "How did task '{task_name}' do?"
- "What status does '{task_name}' have?"
- "Give me the status of '{task_name}'"
- "Print the status of task '{task_name}'"
- "Display the status for '{task_name}'"

**Total**: 14 templates (all map to same status retrieval code)

#### 4. `RETURN_TASK_NAMES_BY_NAME_TEMPLATES` (+5 variations)
**Original**: 5 templates
**Added**:
- "What are the names of all '{task_name}' tasks?"
- "Print the names of tasks called '{task_name}'"
- "Give me just the names for '{task_name}' tasks"
- "List all task names that are '{task_name}'"
- "Show task names matching '{task_name}'"

**Total**: 10 templates (all map to same `X_task_names := {task.name | ...}` code)

#### 5. `RETURN_TASK_STATUSES_BY_NAME_TEMPLATES` (+5 variations)
**Original**: 6 templates
**Added**:
- "What are all '{task_name}' tasks and their statuses?"
- "Print all '{task_name}' tasks with name and status"
- "Give me '{task_name}' tasks and how they did"
- "List '{task_name}' tasks and whether they succeeded"
- "Show '{task_name}' tasks and their completion status"

**Total**: 11 templates (all map to same `X_task_statuses := [{"name": ..., "status": ...} | ...]` code)

#### 6. `LIST_TASKS_TEMPLATES` (+8 variations)
**Original**: 6 templates
**Added**:
- "Print all task names"
- "What tasks are in here?"
- "Show me all the task names"
- "Give me a list of all task names"
- "Display all task names"
- "What are the names of all tasks?"
- "List every task name"
- "Show what tasks we have"

**Total**: 14 templates (all map to same `task_names := {name | ...}` code)

#### 7. `TASK_RESULTS_TEMPLATES` (+7 variations)
**Original**: 9 templates
**Added**:
- "Print all results for '{task_name}'"
- "What did '{task_name}' produce?"
- "Show me what '{task_name}' returned"
- "Give me all the results from '{task_name}'"
- "Display results for task '{task_name}'"
- "What outputs did '{task_name}' generate?"
- "List everything '{task_name}' produced"

**Total**: 16 templates (all map to same `task_results := [result | ...]` code)

#### 8. `PARAM_VALUE_TEMPLATES` (+7 variations)
**Original**: 7 templates
**Added**:
- "What's the {param_name} param for '{task_name}'?"
- "Show me the {param_name} parameter for task '{task_name}'"
- "Print the {param_name} param value from '{task_name}'"
- "Give me the {param_name} value for '{task_name}'"
- "What {param_name} did '{task_name}' use?"
- "Display the {param_name} parameter for '{task_name}'"
- "Tell me the {param_name} value for task '{task_name}'"

**Total**: 14 templates (all map to same parameter value retrieval code)

#### 9. `RESULT_NAMES_TEMPLATES` (+6 variations)
**Original**: 6 templates
**Added**:
- "Print all result names for '{task_name}'"
- "What result keys does '{task_name}' have?"
- "Show me the result names from '{task_name}'"
- "List what results '{task_name}' produced"
- "Give me all result key names for '{task_name}'"
- "What are the names of '{task_name}' results?"

**Total**: 12 templates (all map to same `result_names := {result.name | ...}` code)

#### 10. `RESULT_BY_NAME_TEMPLATES` (+7 variations)
**Original**: 6 templates
**Added**:
- "What's the {result_name} for '{task_name}'?"
- "Show me the {result_name} result from '{task_name}'"
- "Print the {result_name} value for task '{task_name}'"
- "Give me the {result_name} from '{task_name}'"
- "What did '{task_name}' return for {result_name}?"
- "Display the {result_name} result for '{task_name}'"
- "Tell me the {result_name} value from '{task_name}'"

**Total**: 13 templates (all map to same specific result retrieval code)

#### 11. `TASK_BUNDLE_TEMPLATES` (+7 variations)
**Original**: 6 templates
**Added**:
- "What bundle did '{task_name}' use?"
- "Show me the bundle for '{task_name}'"
- "Print the bundle reference from '{task_name}'"
- "Give me the bundle that '{task_name}' used"
- "What's the bundle image for '{task_name}'?"
- "Display the bundle for task '{task_name}'"
- "Tell me which bundle '{task_name}' used"

**Total**: 13 templates (all map to same bundle retrieval code)

#### 12. `TASK_TIMESTAMP_TEMPLATES` (+10 variations)
**Original**: 8 templates
**Added**:
- "What time did '{task_name}' start?"
- "Show me when '{task_name}' started"
- "Print the start time for '{task_name}'"
- "When was '{task_name}' started?"
- "What's the start timestamp for '{task_name}'?"
- "Give me the start time of '{task_name}'"
- "What time did '{task_name}' finish?"
- "Show me when '{task_name}' finished"
- "Print the finish time for '{task_name}'"
- "When was '{task_name}' completed?"

**Total**: 18 templates (all map to same timestamp retrieval code)

#### 13. `TASK_STATUS_FILTER_TEMPLATES` (+8 variations)
**Original**: 9 templates
**Added**:
- "Print all tasks with status '{status}'"
- "Show me tasks that are '{status}'"
- "Which tasks are '{status}'?"
- "Give me all '{status}' tasks"
- "List every task that's '{status}'"
- "What tasks ended up '{status}'?"
- "Display all '{status}' tasks"
- "Show tasks that completed with '{status}'"

**Total**: 17 templates (all map to same status filter code)

## Key Features

### Conversational Phrasings
- Uses contractions: "What's", "What did", "Give me"
- Direct questions: "Which tasks...?", "What tasks...?", "How did...?"
- Casual language: "Show me", "Tell me", "Give me", "Print"
- Exploration style: "What tasks are in here?", "Show what tasks we have"

### Debugging-Style Prompts
- "Print all tasks named buildah" (exact example from suggestion)
- "List tasks with name buildah" (exact example from suggestion)
- "Which tasks use the name buildah?" (exact example from suggestion)
- "What did '{task_name}' produce?"
- "Show me what '{task_name}' returned"

### Same Code, Different Phrasing
All variations in each template group map to the **exact same Rego code output**, teaching the model that:
- Different phrasings = same navigation pattern
- Conversational language = same technical query
- Debugging prompts = same code structure

## Benefits

1. **Better Generalization**: Model learns that many phrasings map to same code
2. **Real-World Usage**: Matches how users actually type when debugging/exploring
3. **Conversational Support**: Handles casual, conversational queries
4. **Robustness**: Model becomes less sensitive to exact wording
5. **User-Friendly**: Supports natural language exploration patterns

## Example Mappings

### Pattern: Return tasks by name
**Same Code**: `tasks_named_buildah := [task | ...]`

**Different Phrasings**:
- "Print all tasks named buildah"
- "List tasks with name buildah"
- "Return all buildah tasks"
- "Which tasks use the name buildah?"
- "Show all buildah tasks"
- "Give me all tasks named buildah"
- "What tasks are called buildah?"

### Pattern: Get task status
**Same Code**: `task_status_check if { ... task.status == "X" }`

**Different Phrasings**:
- "What's the status of buildah?"
- "Show me the status for task buildah"
- "How did task buildah do?"
- "What status does buildah have?"
- "Give me the status of buildah"

## Technical Details

- **Random Selection**: `random.choice()` ensures all variations are used
- **Same Output**: All variations in a group map to identical Rego code
- **Balanced Coverage**: More variations = better generalization
- **Natural Language**: Focus on conversational, debugging-style prompts

## Summary

✅ **70+ new natural language variations** added
✅ **12 template groups** enhanced
✅ **Conversational phrasings** included
✅ **Debugging-style prompts** added (matching user examples)
✅ **Same code, different phrasing** pattern reinforced
✅ **Ready for dataset regeneration**

This dramatically improves the model's ability to generalize navigation patterns to real-world user prompts.


