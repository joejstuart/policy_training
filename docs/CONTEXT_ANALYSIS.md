# Training Data Context Analysis

## Current Approach: Raw Attestation JSON

**Current context**: Trimmed attestation JSON (100% of examples)
- Average size: ~2,189 chars
- Range: 353 - 21,151 chars
- Shows actual data structure with real values

## What the Model Needs to Learn

From analyzing the output Rego code, the model must learn:

1. **Navigation patterns**:
   - `input.attestations` → array of attestation objects
   - `att.statement.predicate.buildConfig.tasks` → array of task objects
   - `att.statement.predicate.materials` → array of material objects
   - `att.statement.subject` → array of subject objects

2. **Field access patterns**:
   - `task.name`, `task.status`, `task.ref.bundle`
   - `material.uri`, `material.digest.sha1`
   - `subject.digest.sha256`

3. **Structure understanding**:
   - Arrays vs objects
   - Nested paths
   - Optional fields (e.g., `task.ref.bundle` vs `task.ref.params`)

## Pros of Current JSON Context

✅ **Shows actual structure** - Model sees real data layout
✅ **Shows data types** - Arrays, objects, strings, numbers
✅ **Shows field names** - Exact field names used in Rego code
✅ **Shows relationships** - How fields nest and relate
✅ **Realistic** - Matches what model will see at inference time

## Cons of Current JSON Context

❌ **Noisy** - Lots of irrelevant data even after trimming
❌ **Large** - Average 2,189 chars, some up to 21K chars
❌ **No schema guidance** - Model must infer structure from examples
❌ **No navigation hints** - Doesn't explicitly show common paths
❌ **Redundant** - Same structure repeated across many examples

## Alternative Context Approaches

### Option 1: Schema + Minimal JSON

**Structure**:
```
# Attestation Structure Schema
input.attestations[] → array of attestation objects
  └─ statement
      └─ predicate
          ├─ buildConfig.tasks[] → array of task objects
          │   ├─ name (string)
          │   ├─ status (string: "Succeeded", "Failed", etc.)
          │   ├─ ref.bundle (string, optional)
          │   └─ ref.params[] → array of param objects
          │       ├─ name (string)
          │       └─ value (string)
          ├─ materials[] → array of material objects
          │   ├─ uri (string)
          │   └─ digest.sha1 (string, optional)
          └─ subject[] → array of subject objects
              └─ digest.sha256 (string)

# Example Data (relevant to query)
{actual trimmed JSON}
```

**Pros**:
- ✅ Explicit structure guidance
- ✅ Shows all navigation paths
- ✅ Smaller context (schema + minimal JSON)
- ✅ Teaches structure systematically

**Cons**:
- ❌ More complex to generate
- ❌ Schema might not cover all edge cases
- ❌ Less "realistic" (model won't see schema at inference)

### Option 2: Navigation Guide + JSON

**Structure**:
```
# Common Navigation Patterns:
- Tasks: input.attestations → statement → predicate → buildConfig.tasks
- Materials: input.attestations → statement → predicate → materials
- Subjects: input.attestations → statement → subject

# Example Data:
{trimmed JSON}
```

**Pros**:
- ✅ Simple navigation hints
- ✅ Still shows real data
- ✅ Smaller than full schema

**Cons**:
- ❌ Less comprehensive than schema
- ❌ Still has JSON noise

### Option 3: Structured Description

**Structure**:
```
Attestation contains:
- attestations[] (array)
  - statement.predicate.buildConfig.tasks[] (array)
    - name: "task-name"
    - status: "Succeeded" | "Failed"
    - ref.bundle: "oci://..." (optional)
  - statement.predicate.materials[] (array)
    - uri: "oci://..."
    - digest.sha1: "..." (optional)
  - statement.subject[] (array)
    - digest.sha256: "..."
```

**Pros**:
- ✅ Very compact
- ✅ Clear structure
- ✅ Easy to parse

**Cons**:
- ❌ Not realistic (model won't see this at inference)
- ❌ Loses actual data values
- ❌ May not teach JSON navigation well

### Option 4: Hybrid - Schema + Relevant JSON Snippet

**Structure**:
```
# Attestation Structure:
input.attestations[] → statement → predicate → buildConfig.tasks[]
  Fields: name, status, ref.bundle, ref.params[]

# Relevant Data (for this query):
{
  "attestations": [{
    "statement": {
      "predicate": {
        "buildConfig": {
          "tasks": [
            {"name": "init", "status": "Succeeded", ...}
          ]
        }
      }
    }
  }]
}
```

**Pros**:
- ✅ Best of both worlds
- ✅ Schema teaches structure
- ✅ JSON shows real data
- ✅ Smaller than full JSON

**Cons**:
- ❌ More complex generation
- ❌ Need to maintain schema

## Recommendation

**Keep JSON context BUT add a query-specific schema header**:

The schema header should **change based on the query type**, just like the JSON trimming already does:

### Task Queries
```json
{
  "instruction": "Get the status of task 'init'",
  "context": "# Attestation Structure:\n# input.attestations[] → statement → predicate → buildConfig.tasks[]\n# Task fields: name, status, ref.bundle, ref.params[], startedOn, finishedOn, results[]\n\n{trimmed JSON with tasks}",
  "output_code": "..."
}
```

### Material Queries
```json
{
  "instruction": "Check for material with URI 'oci://...'",
  "context": "# Attestation Structure:\n# input.attestations[] → statement → predicate → materials[]\n# Material fields: uri, digest.sha1, digest.sha256\n\n{trimmed JSON with materials}",
  "output_code": "..."
}
```

### Subject Queries
```json
{
  "instruction": "Find subject with digest '...'",
  "context": "# Attestation Structure:\n# input.attestations[] → statement → subject[]\n# Subject fields: name, digest.sha256\n\n{trimmed JSON with subjects}",
  "output_code": "..."
}
```

### Bundle Queries
```json
{
  "instruction": "Get bundle reference for task 'buildah'",
  "context": "# Attestation Structure:\n# input.attestations[] → statement → predicate → buildConfig.tasks[]\n# Task.ref fields: bundle (direct) OR ref.params[] where param.name == 'bundle'\n\n{trimmed JSON with tasks}",
  "output_code": "..."
}
```

**Why query-specific schemas work better**:
1. ✅ **Focused** - Only shows relevant paths (less noise)
2. ✅ **Smaller** - ~50-100 chars vs 200+ for full schema
3. ✅ **Matches trimming** - Schema aligns with what JSON contains
4. ✅ **Teaches patterns** - Model learns which paths to use for which queries
5. ✅ **Efficient** - No irrelevant information

**Implementation**:
- Generate schema based on instruction type (task/material/subject/bundle)
- Prepend to trimmed JSON
- Keep schema concise (just the relevant navigation path + key fields)
- Match the trimming logic already in place

## Implementation Details

The schema generation should mirror the existing trimming logic:

```python
def generate_schema_header(instruction: str, metadata: Dict) -> str:
    """Generate query-specific schema header."""
    instruction_lower = instruction.lower()
    
    if "task" in instruction_lower:
        return "# Attestation Structure:\n# input.attestations[] → statement → predicate → buildConfig.tasks[]\n# Task fields: name, status, ref.bundle, ref.params[], startedOn, finishedOn, results[]\n"
    elif "material" in instruction_lower:
        return "# Attestation Structure:\n# input.attestations[] → statement → predicate → materials[]\n# Material fields: uri, digest.sha1, digest.sha256\n"
    elif "subject" in instruction_lower:
        return "# Attestation Structure:\n# input.attestations[] → statement → subject[]\n# Subject fields: name, digest.sha256\n"
    elif "bundle" in instruction_lower:
        return "# Attestation Structure:\n# input.attestations[] → statement → predicate → buildConfig.tasks[]\n# Task.ref: bundle (direct) OR ref.params[] where param.name == 'bundle'\n"
    else:
        return "# Attestation Structure:\n# input.attestations[] → statement → predicate\n"
```

Then in `ExampleBuilder.build_example()`:
```python
# Generate query-specific schema
schema_header = generate_schema_header(instruction, metadata)

# Build context with schema + JSON
context = f"{schema_header}\n{json.dumps(trimmed_data, indent=2, ensure_ascii=False)}"
```

## Testing the Hypothesis

To validate if query-specific schema helps:
1. Generate a small test set with schema + JSON
2. Compare training performance vs JSON-only
3. Check if model learns navigation patterns faster
4. Evaluate inference quality on different query types

## Alternative: Improve JSON Trimming Only

If you prefer to keep JSON-only:
- Better trimming (only show exact paths needed)
- Add comments in JSON showing navigation
- Use more aggressive trimming for common queries

But query-specific schema is probably more valuable as it explicitly teaches structure patterns for each query type.

