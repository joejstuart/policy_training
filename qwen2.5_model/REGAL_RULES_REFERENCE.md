# Regal Rules Reference

This document contains all Regal linting rules organized by category. Use this as a reference when improving Rego code to ensure compliance with Regal's style guide and best practices.

**Source**: [Regal Rules Documentation](https://www.openpolicyagent.org/projects/regal/rules)

## Table of Contents

- [Bugs](#bugs) - Rules that catch potential bugs and errors
- [Idiomatic](#idiomatic) - Rules for idiomatic Rego patterns
- [Style](#style) - Style and formatting rules
- [Imports](#imports) - Import-related rules
- [Performance](#performance) - Performance optimization rules
- [Testing](#testing) - Testing-related rules
- [Custom](#custom) - Custom rules

---

## Bugs

Rules that catch potential bugs, errors, and problematic patterns.

### `bugs/zero-arity-function`
**Summary**: Avoid functions without args

Functions should have at least one argument. Zero-arity functions are not idiomatic in Rego.

### `bugs/var-shadows-builtin`
**Summary**: Variable shadows built-in

A variable name shadows a built-in function name, which can cause confusion and errors.

### `bugs/unused-output-variable`
**Summary**: Unused output variable

An output variable is declared but never used in the rule body.

### `bugs/unassigned-return-value`
**Summary**: Non-boolean return value unassigned

A function returns a non-boolean value but it's not assigned to a variable.

### `bugs/top-level-iteration`
**Summary**: Iteration in top-level assignment

Iteration should not be used in top-level assignments. Use comprehensions or helper rules instead.

### `bugs/time-now-ns-twice`
**Summary**: Repeated calls to time.now_ns

Multiple calls to `time.now_ns` in the same evaluation can cause inconsistent results. Cache the value.

### `bugs/sprintf-arguments-mismatch`
**Summary**: Mismatch in sprintf arguments count

The number of format specifiers in `sprintf` doesn't match the number of arguments provided.

### `bugs/rule-shadows-builtin`
**Summary**: Rule name shadows built-in

A rule name shadows a built-in function name, which can cause confusion.

### `bugs/rule-named-if`
**Summary**: Rule named if

A rule is named `if`, which is a reserved keyword in Rego.

### `bugs/rule-assigns-default`
**Summary**: Rule assigned its default value

A rule is assigned its default value, which is redundant.

### `bugs/redundant-loop-count`
**Summary**: Redundant count before loop

A `count()` operation is performed before a loop that could be optimized.

### `bugs/redundant-existence-check`
**Summary**: Redundant existence check

An existence check is redundant (e.g., checking if something exists when it's already required).

### `bugs/not-equals-in-loop`
**Summary**: Use of != in loop

Using `!=` in a loop can cause unexpected behavior. Use `every` or restructure the logic.

### `bugs/leaked-internal-reference`
**Summary**: Outside reference to internal rule or function

A rule or function marked as `internal` is referenced from outside its intended scope.

### `bugs/invalid-metadata-attribute`
**Summary**: Invalid attribute in metadata annotation

A metadata annotation contains an invalid attribute.

### `bugs/internal-entrypoint`
**Summary**: Entrypoint can't be marked internal

An entrypoint rule cannot be marked as `internal` since it needs to be accessible.

### `bugs/inconsistent-args`
**Summary**: Inconsistently named function arguments

Function arguments should be consistently named across all function definitions.

### `bugs/impossible-not`
**Summary**: Impossible not condition

A `not` condition can never be true, indicating a logic error.

**Type**: Aggregate - runs both on single files as well as when more than one file is provided for linting

### `bugs/import-shadows-rule`
**Summary**: Import shadows rule

An import name shadows an existing rule name.

### `bugs/if-object-literal`
**Summary**: Object literal following if

An object literal follows an `if` keyword, which is not valid syntax.

### `bugs/if-empty-object`
**Summary**: Empty object following if

An empty object follows an `if` keyword, which is not valid syntax.

### `bugs/duplicate-rule`
**Summary**: Duplicate rule

A rule is defined multiple times with the same name.

### `bugs/deprecated-builtin`
**Summary**: Deprecated built-in function

A deprecated built-in function is being used. Use the recommended alternative.

### `bugs/constant-condition`
**Summary**: Constant condition

A condition always evaluates to the same value (true or false), indicating dead code.

### `bugs/argument-always-wildcard`
**Summary**: Argument is always a wildcard

A function argument is always a wildcard, making it unnecessary.

### `bugs/annotation-without-metadata`
**Summary**: Annotation without metadata

An annotation is used without a corresponding metadata block.

---

## Idiomatic

Rules for writing idiomatic Rego code that follows best practices.

### `idiomatic/use-strings-count`
**Summary**: Use strings.count where possible

Prefer `strings.count()` over manual counting when checking string occurrences.

### `idiomatic/use-some-for-output-vars`
**Summary**: Use some to declare output variables

Use `some` to declare output variables in comprehensions for clarity.

### `idiomatic/use-object-keys`
**Summary**: Prefer to use object.keys

Use `object.keys()` instead of manual iteration when you need object keys.

### `idiomatic/use-in-operator`
**Summary**: Use in to check for membership

Use the `in` operator to check if a value is in a collection instead of manual iteration.

### `idiomatic/use-if`
**Summary**: Use the if keyword

Use the `if` keyword in rule bodies for clarity and consistency with modern Rego.

### `idiomatic/use-contains`
**Summary**: Use the contains keyword

Use the `contains` keyword for set membership checks instead of manual iteration.

### `idiomatic/single-item-in`
**Summary**: Avoid in for single item collection

Don't use `in` operator when checking against a single-item collection. Use direct comparison.

### `idiomatic/prefer-set-or-object-rule`
**Summary**: Prefer set or object rule over comprehension

When possible, use set or object rules instead of comprehensions for better performance.

### `idiomatic/prefer-equals-comparison`
**Summary**: Prefer == for equality comparison

Use `==` for equality comparisons instead of other patterns.

### `idiomatic/non-raw-regex-pattern`
**Summary**: Use raw strings for regex patterns

Use raw strings (backticks) for regex patterns to avoid escaping issues.

**Automatically fixable**: Yes

### `idiomatic/no-defined-entrypoint`
**Summary**: Missing entrypoint annotation

An entrypoint rule is missing the `@entrypoint` annotation.

**Type**: Aggregate - only runs when more than one file is provided for linting

### `idiomatic/in-wildcard-key`
**Summary**: Unnecessary wildcard key

A wildcard key is used unnecessarily in an object iteration.

### `idiomatic/equals-pattern-matching`
**Summary**: Prefer pattern matching in function arguments

Use pattern matching in function arguments when possible instead of equality checks.

### `idiomatic/directory-package-mismatch`
**Summary**: Directory structure should mirror package

The directory structure should mirror the package path for better organization.

### `idiomatic/custom-in-construct`
**Summary**: Custom function may be replaced by in keyword

A custom function that checks membership can be replaced with the `in` keyword.

---

## Style

Style and formatting rules for consistent code appearance.

### `style/yoda-condition`
**Summary**: Yoda condition, it is

Avoid Yoda conditions (e.g., `"value" == variable`). Use normal order: `variable == "value"`.

### `style/use-assignment-operator`
**Summary**: Prefer := over = for assignment

Use `:=` for assignment instead of `=` for clarity and consistency.

**Automatically fixable**: Yes

### `style/unnecessary-some`
**Summary**: Unnecessary use of some

The `some` keyword is used unnecessarily when it doesn't add value.

### `style/unconditional-assignment`
**Summary**: Unconditional assignment in rule body

An assignment in a rule body is unconditional and could be moved to the rule head.

### `style/trailing-default-rule`
**Summary**: Default rule should be declared first

Default rules should be declared before their non-default counterparts.

### `style/todo-comment`
**Summary**: Avoid TODO Comments

Remove TODO comments before committing code.

### `style/rule-name-repeats-package`
**Summary**: Avoid repeating package path in rule names

Rule names shouldn't repeat parts of the package path (e.g., `package auth` with rule `auth.allow`).

### `style/rule-length`
**Summary**: Max rule length exceeded

A rule exceeds the maximum recommended length. Consider breaking it into smaller rules.

### `style/prefer-some-in-iteration`
**Summary**: Prefer some .. in for iteration

Use `some ... in` for iteration instead of other patterns for better readability.

### `style/prefer-snake-case`
**Summary**: Prefer snake_case for names

Use snake_case for variable names, rule names, and function names instead of camelCase.

### `style/pointless-reassignment`
**Summary**: Pointless reassignment of variable

A variable is reassigned to the same value, which is unnecessary.

### `style/opa-fmt`
**Summary**: File should be formatted with opa fmt

The file should be formatted using `opa fmt` for consistent formatting.

**Automatically fixable**: Yes

### `style/no-whitespace-comment`
**Summary**: Comment should start with whitespace

Comments should start with a space after `#` for consistency.

**Automatically fixable**: Yes

### `style/mixed-iteration`
**Summary**: Mixed iteration style

Different iteration styles are used in the same code. Use consistent iteration patterns.

### `style/messy-rule`
**Summary**: Messy incremental rule

An incremental rule (using `+=` or similar) is structured in a confusing way.

### `style/line-length`
**Summary**: Line too long

A line exceeds the maximum recommended length. Consider breaking it into multiple lines.

### `style/function-arg-return`
**Summary**: Return value assigned in function argument

A function's return value is assigned in the function argument list, which is confusing.

### `style/file-length`
**Summary**: Max file length exceeded

The file exceeds the maximum recommended length. Consider splitting into multiple files.

### `style/external-reference`
**Summary**: External reference in function

A function references external data that should be passed as an argument.

### `style/double-negative`
**Summary**: Avoid double negatives

Avoid double negatives (e.g., `not not allowed`) which are confusing.

### `style/detached-metadata`
**Summary**: Detached metadata annotation

A metadata annotation is not directly above the rule or function it describes.

### `style/default-over-not`
**Summary**: Prefer default assignment over negated condition

Use default assignment instead of negated conditions when possible.

### `style/default-over-else`
**Summary**: Prefer default assignment over fallback else

Use default assignment instead of else clauses when possible.

### `style/comprehension-term-assignment`
**Summary**: Assignment can be moved to comprehension term

An assignment can be moved into the comprehension term for better readability.

### `style/chained-rule-body`
**Summary**: Avoid chaining rule bodies

Avoid chaining rule bodies together. Use separate rules or helper functions.

### `style/avoid-get-and-list-prefix`
**Summary**: Avoid get_ and list_ prefix for rules and functions

Don't prefix rule or function names with `get_` or `list_`. The name should be descriptive without these prefixes.

---

## Imports

Rules related to import statements and package imports.

### `imports/unresolved-import`
**Summary**: Unresolved import

An import cannot be resolved. Check that the package exists and the path is correct.

**Type**: Aggregate - only runs when more than one file is provided for linting

### `imports/redundant-data-import`
**Summary**: Redundant import of data

A data import is redundant and can be removed.

### `imports/redundant-alias`
**Summary**: Redundant alias

An import alias is redundant (e.g., `import data.foo as foo`).

### `imports/prefer-package-imports`
**Summary**: Prefer importing packages over rules

Import entire packages instead of individual rules when possible.

**Type**: Aggregate - only runs when more than one file is provided for linting

### `imports/pointless-import`
**Summary**: Importing own package is pointless

A package imports itself, which is unnecessary.

### `imports/import-shadows-import`
**Summary**: Import shadows import

An import name shadows another import name.

### `imports/import-shadows-builtin`
**Summary**: Import shadows built-in namespace

An import name shadows a built-in namespace (e.g., `import data`).

### `imports/import-after-rule`
**Summary**: Import declared after rule

Imports should be declared at the top of the file, before any rules.

### `imports/implicit-future-keywords`
**Summary**: Implicit future keywords

The code uses future keywords without importing `rego.v1`.

### `imports/ignored-import`
**Summary**: Reference ignores import

A reference ignores an import that was intended to be used.

### `imports/confusing-alias`
**Summary**: Confusing alias of existing import

An import alias is confusing or conflicts with existing names.

### `imports/circular-import`
**Summary**: Avoid circular imports

A circular import dependency exists between packages.

### `imports/avoid-importing-input`
**Summary**: Avoid importing input

Don't import `input` as a package. Access it directly.

---

## Performance

Rules for optimizing Rego code performance.

### `performance/with-outside-test-context`
**Summary**: with used outside of test context

The `with` keyword is used outside of test context, which may indicate a mistake.

### `performance/walk-no-path`
**Summary**: Call to walk can be optimized

A call to `walk()` can be optimized by providing a path parameter.

### `performance/non-loop-expression`
**Summary**: Non loop expression in loop

A non-loop expression is used inside a loop, which can cause performance issues.

### `performance/defer-assignment`
**Summary**: Assignment can be deferred

An assignment can be deferred to improve performance by avoiding unnecessary computation.

---

## Testing

Rules related to test files and test code.

### `testing/todo-test`
**Summary**: TODO test encountered

A test contains a TODO comment, which should be addressed.

### `testing/test-outside-test-package`
**Summary**: Test outside of test package

A test is defined outside of a test package (should be in a package ending with `.test`).

### `testing/print-or-trace-call`
**Summary**: Call to print or trace function

A test contains calls to `print()` or `trace()`, which should be removed in production code.

### `testing/metasyntactic-variable`
**Summary**: Metasyntactic variable name

A test uses a metasyntactic variable name (e.g., `foo`, `bar`) instead of a descriptive name.

### `testing/identically-named-tests`
**Summary**: Multiple tests with same name

Multiple tests have the same name, which can cause confusion.

### `testing/file-missing-test-suffix`
**Summary**: Files containing tests should have a _test.rego suffix

Test files should have the `_test.rego` suffix for clarity.

### `testing/dubious-print-sprintf`
**Summary**: Dubious use of print and sprintf

A test uses `print()` or `sprintf()` in a way that may indicate a problem.

---

## Custom

Custom rules that can be configured or extended.

### `custom/prefer-value-in-head`
**Summary**: Prefer value in rule head

Prefer assigning values directly in the rule head when possible.

### `custom/one-liner-rule`
**Summary**: Rule body could be made a one-liner

A rule body could be simplified to a single line.

### `custom/narrow-argument`
**Summary**: Function argument can be narrowed

A function argument type can be narrowed for better type safety.

### `custom/naming-convention`
**Summary**: Naming convention violation

A naming convention violation is detected (configurable).

### `custom/missing-metadata`
**Summary**: Package or rule missing metadata

A package or rule is missing metadata annotations.

### `custom/forbidden-function-call`
**Summary**: Forbidden function call

A forbidden function is being called (configurable list).

---

## Key Takeaways for LLM Code Improvement

When improving Rego code based on Regal violations, prioritize:

1. **Bugs**: Fix all bug-related violations first as they indicate actual problems
2. **Idiomatic**: Use modern Rego patterns (`if`, `in`, `contains`, `some ... in`)
3. **Style**: Follow snake_case naming, proper formatting, and consistent patterns
4. **Performance**: Optimize loops and defer assignments when possible
5. **Imports**: Ensure imports are correct, not redundant, and properly ordered

**Most Common Issues to Fix**:
- Use `if` keyword in rule bodies
- Use `in` operator for membership checks
- Use `some ... in` for iteration
- Use snake_case for all names
- Format code with `opa fmt`
- Avoid unnecessary `some` keywords
- Prefer `:=` over `=` for assignment

**Reference**: For detailed information on each rule, visit: https://www.openpolicyagent.org/projects/regal/rules

