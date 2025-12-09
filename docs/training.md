# Fine-Tuning Output Structure Guidelines

A practical guide for structuring LLM fine-tuning outputs across multiple task types.

---

## 🔑 Core Principles

### 1. Always use a single output field in your dataset

Your fine-tuning examples should follow the simple schema:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}
```

You do **not** add top-level fields like `analysis`, `tests`, or `plan`.  
All structure goes inside the `output` string itself.

### 2. Structured output is task-dependent

Some tasks benefit from structured sections such as:

```
ANALYSIS:
...

TESTS:
...
```

Other tasks should output only:

- text
- code
- a summary
- a fix
- a rewritten rule

There is no universal format for all examples.

### 3. Consistency is required within each task type

Each category of task should have its own predictable format.

| Task Type | Recommended Output Structure | Notes |
|-----------|------------------------------|-------|
| Coverage → Tests | `ANALYSIS` + `TESTS` | Multi-step reasoning task |
| Fix the code | Optional `ANALYSIS` + `FIXED CODE` | If reasoning is needed |
| Summaries | Direct output | No structure needed |
| Rego rule generation | Code only | Simple generation |
| Explain code | Direct natural-language explanation | No analysis needed |

Using consistent formatting teaches the model:

> "When I see this type of instruction, I respond using this structure."

### 4. Section labels (e.g., `ANALYSIS:`) are arbitrary

You may use:

- `ANALYSIS:`
- `PLAN:`
- `REASONING:`
- `NOTES:`
- `REVIEW:`

The word choice does not matter.  
**Only consistency matters.**

---

## 🧭 When Should You Add an `ANALYSIS:` Section?

A simple rule-of-thumb:

> Add an `ANALYSIS:` section when the correct output requires hidden, multi-step reasoning that the model cannot reliably infer without externalizing its thought process.

If the task is direct or single-step, do **NOT** include an analysis section.

### ✅ Add an `ANALYSIS:` section when ALL of these are true:

**1. The task requires multiple logical steps**

Examples:
- coverage → missing branch → necessary input → test generation
- compiler error → diagnose cause → propose fix → output code
- spec → infer rules → generate implementation
- ambiguous input → infer constraints → generate Rego logic

**2. The reasoning is NOT directly stated in the input**

If the model must infer what's missing or why something is uncovered, use analysis.

**3. You benefit from inspectable reasoning**

Especially useful for:
- agent loops
- debugging
- verifying correctness
- teaching the model a decomposition pattern

**4. You want the model to perform similar reasoning in the future**

Structured reasoning helps the model learn:
- decomposition
- state tracking
- branch recognition
- high-accuracy code synthesis

### ❌ Do NOT add an analysis section when the task is simple or direct:

- Summarize this text
- Rewrite this sentence
- Convert JSON to YAML
- Generate a Rego rule from a complete specification
- Extract data
- Basic question answering

Adding analysis to simple tasks is unnecessary and may dilute model patterns.

---

## 🎯 The Two-Question Heuristic

A fast and reliable approach:

1. **Would a human need scratch work to do this task correctly?**
   - If yes → include `ANALYSIS:`
   - If no → skip it

2. **Would skipping reasoning cause the model to hallucinate or miss steps?**
   - If yes → include `ANALYSIS:`
   - If no → skip it

If both answers are "no", you do not need structured reasoning.

---

## 🧩 Examples

### Example: Coverage → Test Generation (needs analysis)

```json
{
  "instruction": "Create tests for full coverage.",
  "input": "coverage output...",
  "output": "ANALYSIS:\n- File ...\n- Missing case ...\n\nTESTS:\n```rego\n...\n```"
}
```

**Reason:** multi-step logical inference.

### Example: Summarization (no analysis)

```json
{
  "instruction": "Summarize the following Jira issue.",
  "input": "text...",
  "output": "This issue describes..."
}
```

### Example: Rego Rule Generation (only code)

```json
{
  "instruction": "Write a rule that validates metadata.",
  "input": "",
  "output": "package metadata\n\nvalidate_metadata := ..."
}
```

### Example: Code Fixing (optional analysis)

If the cause isn't obvious:

```
ANALYSIS:
The error occurs because...

FIXED_RULE:
```rego
...
```
```

If the input is trivial, output only the fixed rule.

---

## ✔️ Summary

- You always use a **single** `output` field.
- You may structure that output **differently per task type**.
- Use `ANALYSIS:` **only when the task requires reasoning steps**.
- Don't force reasoning where it isn't needed.
- Consistency within each task category is essential for reliable model behavior.
