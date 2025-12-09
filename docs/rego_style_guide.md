# Rego Style Guide

The purpose of this style guide is to provide a collection of recommendations and best practices for authoring Rego. From the maintainers of Open Policy Agent (OPA), and some of the most experienced members of the community, we hope to share lessons learnt from authoring and reviewing hundreds of thousands of lines of Rego over the years.

With new features, language constructs, and other improvements continuously finding their way into OPA, we aim to keep this style guide a reflection of what we consider current best practices. Make sure to check back every once in a while, and see the changelog for updates since your last visit.

## Regal

Inspired by this style guide, Regal is a new linter for Rego that allows you to enforce many of the recommendations in this guide, as well as identifying issues, bugs and potential problems in your Rego policies. If you enjoy this style guide, make sure to check it out!

If you want to automatically enforce the rules in this style guide, consider using Regal, the Rego linter. Regal codifies many style guidelines and also identifies problematic patterns, bugs, or risks in Rego policies.

## General Advice

### Optimize for Readability, Not Performance

Rego is a declarative language, which in the best case means you express **what** you want rather than **how** it should be retrieved. When authoring policy, do not try to be "smart" about assumed performance characteristics or optimizations. That's what OPA should worry about!

Optimize for **readability** and **obviousness**. Optimize for performance _only_ if you've identified performance issues in your policy, and even if you do — making your policy more compact or "clever" almost never helps at addressing the problem at hand.

- Rego is a declarative language. Write policy that expresses what you want, not how to optimize internal execution.
- Don't attempt cleverness that obscures intent. Most micro-optimizations do not improve performance.
- If you need performance improvements, measure first and only optimize after profiling indicates a real bottleneck.

### Use `opa fmt`

The `opa fmt` tool ensures consistent formatting across teams and projects. While certainly not perfect (yet!), unified formatting is a big win, and saves a lot of time in code reviews arguing over details around style.

A good idea could be to run `opa fmt --write` on save, which can be configured in most editors. If you want to enforce `opa fmt` formatting as part of your build pipeline, use `opa fmt --fail`.

Format your policies using:

```bash
opa fmt
```

To enforce formatting consistency:

- Set up your editor to run `opa fmt --write` on save.
- Configure your CI pipeline to run `opa fmt --fail` to catch formatting drift.

**Notes:**

- `opa fmt` uses tabs for indentation.
- Tabs can render inconsistently across systems (especially web UIs like GitHub).
- Consider including an `.editorconfig` to define tab width.
- Formatting rules handled by `opa fmt` are not duplicated in this guide.

**Tip**: `opa fmt` uses tabs for indentation. By default, GitHub uses 8 spaces to display tabs, which is arguably a bit much. You can change this preference for your account in github.com/settings/appearance, or provide an `.editorconfig` file in your policy repository:

```ini
[*.rego]
end_of_line = lf
insert_final_newline = true
charset = utf-8
indent_style = tab
indent_size = 4
```

### Use Strict Mode

Strict mode provides extra checks for common mistakes like redundant imports, or unused variables. Include an `opa check --strict path/to/polices` step as part of your build pipeline.

Enable strict mode in your CI/build pipeline:

```bash
opa check --strict .
```

Strict mode catches mistakes such as:

- Unused imports
- Unused variables
- Deprecated syntax
- Other static analysis warnings

### Use Metadata Annotations

Favor metadata annotations over regular comments.

Metadata annotations allow external tools and editors to parse their contents, potentially leveraging them for something useful, like in-line explanations, generated docs, etc.

Annotations are also a good way to de-duplicate information such as documentation links, contact emails and error codes where explanations are returned as part of the result.

Use structured METADATA annotations instead of free-form comments for:

- Rule descriptions
- Error codes
- References
- Links
- Documentation

This allows tools to inspect metadata programmatically.

**Avoid**

```rego
# Example package with documentation
package example

import future.keywords.contains
import future.keywords.if

# E123: Deny non admin users.
# Only admin users are allowed to access these resources, see https://docs.example.com/policy/rule/E123
deny contains {
	"code": 401,
	"message": "Unauthorized due to policy rule (E123, https://docs.example.com/policy/rule/E123)",
} if {
	input.admin == false
}
```

**Prefer**

```rego
# METADATA
# title: Example
# description: Example package with documentation
package example

import future.keywords.contains
import future.keywords.if

# METADATA
# title: Deny non admin users
# description: Only admin users are allowed to access these resources
# related_resources:
# - https://docs.example.com/policy/rule/E123
# custom:
#   code: 401
#   error_id: E123
deny contains {
	"code": metadata.custom.code,
	"message": sprintf("Unauthorized due to policy rule (%s, %s)", [
		metadata.custom.error_id,
		concat(",", [ref | ref := metadata.related_resources[_].ref]),
	]),
} if {
	input.admin == false

	metadata := rego.metadata.rule()
}
```

**Notes / Exceptions**

Use regular comments inside of rule bodies, or for packages and rules you consider "internal". It's still fine to use comments for:

- Explanations inside rule bodies
- Notes that don't belong in formal metadata

### Get to Know the Built-in Functions

With more than 150 built-in functions tailor-made for policy evaluation, there's a good chance that some of them can help you accomplish your goal.

Rego has over 150 built-ins. Many style concerns vanish when you use built-ins effectively.

Areas:

- **Collections**: `sum`, `count`, `intersection`, `union`
- **Strings**: `split`, `sprintf`, `indexof`, `replace`
- **JSON**: `json.unmarshal`, `json.is_valid`
- **Regex**: `re_match`
- **Time**: `time.parse_rfc3339`, `time.now_ns`

### Consider Using JSON Schemas for Type Checking

As you author Rego policy, providing JSON schemas for your `input` (and possibly `data`) enables strict type checking, letting you avoid simple — but common — mistakes, like typos, or referencing nested attributes in the wrong location. This extra level of verification improves both the developer experience as well as the quality of your policies.

- Helps catch typos or wrong attribute paths.
- Improves editor integration.
- Ensures type stability.

## Style

### Prefer `snake_case` for Rule Names and Variables

The built-in functions use `snake_case` for naming — follow that convention for your own rules, functions, and variables.

All built-in functions use `snake_case`. Best practice:

- Use `snake_case` for all rule names.
- Use `snake_case` for all variables.

**Avoid**

```rego
userIsAdmin if "admin" in input.user.roles
```

**Prefer**

```rego
user_is_admin if "admin" in input.user.roles
```

**Notes / Exceptions**

In many cases, you might not control the format of the `input` data — if the domain of a policy (e.g. Envoy) mandates a different style, making an exception might seem reasonable. Adapting policy format after `input` is however prone to inconsistencies, as you'll likely end up mixing different styles in the same policy (due to imports of common code, etc).

Avoid:

- `camelCase`
- `PascalCase`
- `mixedCase`

### Optionally, Use Leading Underscore for Rules Intended for Internal Use

While OPA doesn't have "private" rules or functions, a pretty common convention that we've seen in the community is to use a leading underscore for rules and functions that are intended to be internal to the package that they are in:

```rego
developers contains user if {
    some user in input.users
    _is_developer(user)
}

_is_developer(user) if {
    # some conditions
}

_is_developer(user) if {
    # some other conditions
}
```

While an `is_developer` function may seem like a good candidate for reuse, it could easily be the case that this should be considered to what **this** package considers a developer, and not necessarily a universal truth. Using a leading underscore to denote this is a good way to communicate this intent, but there are also other ways to do this, like agreed upon naming conventions, or using custom metadata annotation attributes.

One benefit of sticking to the leading underscore convention is that tools like Regal, and the language server for Rego that it provides, may use this information to provide better suggestions, like not adding references to these rules and functions from other packages.

Rego has no private visibility. But you may use a leading underscore for rules intended for internal use:

```rego
_is_internal_helper(...)
```

This is optional but helps readability.

### Keep Line Length `<=` 120 Characters

Long lines are tedious to read. Keep line length at 120 characters or below.

Long lines reduce readability. Break expressions across multiple lines.

### Use Helper Rules and Functions to Decompose Logic

If a rule's body grows large or complex, create helper rules.

**Avoid**

```rego
allow if {
    "developer" in input.user.roles
    input.request.method in {"GET", "HEAD"}
    startswith(input.request.path, "/docs")
}

allow if {
    "developer" in input.user.roles
    input.request.method in {"GET", "HEAD"}
    startswith(input.request.path, "/api")
}
```

**Prefer**

```rego
allow if {
    is_developer
    read_request
    startswith(input.request.path, "/docs")
}

allow if {
    is_developer
    read_request
    startswith(input.request.path, "/api")
}

read_request if input.request.method in {"GET", "HEAD"}
is_developer if "developer" in input.user.roles
```

This improves modularity and readability.

### Use Negation to Handle Undefined

When checking for the absence of something, use negation carefully. Negation interacts with undefined values.

### Consider Partial Helper Rules Over Comprehensions in Rule Bodies

For complex logic, consider using partial helper rules instead of complex comprehensions directly in rule bodies.

### Avoid Prefixing Rules and Functions with `get_` or `list_`

These prefixes are redundant. The rule or function name should be descriptive enough.

### Prefer Unconditional Assignment in Rule Head Over Rule Body

If a rule always produces a value, define that in the rule head.

**Avoid**

```rego
full_name := name {
    name := concat(", ", [input.first_name, input.last_name])
}
```

**Prefer**

```rego
full_name := concat(", ", [input.first_name, input.last_name])
```

Same with functions:

**Avoid**

```rego
divide_by_ten(x) := y {
    y := x / 10
}
```

**Prefer**

```rego
divide_by_ten(x) := x / 10
```

## Variables and Data Types

### Use `in` to Check for Membership

Clearer and safer than using equality with iteration.

**Avoid**

```rego
allow {
    "admin" == input.user.roles[_]
}
```

**Prefer**

```rego
allow if "admin" in input.user.roles
```

### Prefer `some ... in` for Iteration

Use the modern pattern:

```rego
some host in data.network.hosts
```

Avoid old-style iteration:

```rego
host := data.network.hosts[_]
```

### Use `every` to Express FOR ALL

Use `every` when you mean "all items must satisfy X".

**Avoid**

```rego
allow if not any_old_registry

any_old_registry if {
    some container in input.request.object.spec.containers
    startswith(container.image, "old.docker.registry/")
}
```

**Prefer**

```rego
allow if {
    every container in input.request.object.spec.containers {
        not startswith(container.image, "old.docker.registry/")
    }
}
```

### Don't Use Unification Operator for Assignment or Comparison

Use `:=` for assignment and `==` for comparison. Don't use `=` for either.

### Don't Use Undeclared Variables

Always declare variables explicitly using `some` or `:=`.

### Prefer Sets Over Arrays (Where Applicable)

Sets indicate uniqueness and unordered semantics.

- Set membership is O(1).
- They support natural set operations.

**Avoid**

Using arrays when order doesn't matter.

**Prefer**

Use sets for roles, permissions, unique identifiers, etc.

## Functions

### Prefer Using Arguments Over `input`, `data` or Rule References

Avoid referencing `input` or `data` deep inside helper functions.

**Better:**

```rego
is_allowed(user) := ...
```

**Worse:**

```rego
is_allowed := input.user ...
```

Argument-based functions are more reusable and clearer.

### Avoid Using the Last Argument for the Return Value

Older Rego policies sometimes contain an unusual way to declare where the return value of a function call should be stored — the last argument of the function. True to its Datalog roots, return values may be stored either using assignment (i.e. `:=`) or by appending a variable name to the argument list of a function.

**Avoid**

```rego
first_a := i if {
    indexof("answer", "a", i)
}
```

**Prefer**

```rego
first_a := i if {
    i := indexof("answer", "a")
}
```

While the first form is valid, it is almost guaranteed to confuse developers coming from the most common programming languages. Again, optimize for readability!

## Regex

### Use Raw Strings for Regex Patterns

Raw strings are interpreted literally, allowing you to avoid having to escape special characters like `\` in your regex patterns.

**Avoid**

```rego
all_digits if {
    regex.match("[\\d]+", "12345")
}
```

**Prefer**

```rego
all_digits if {
    regex.match(`[\d]+`, "12345")
}
```

## Packages

### Package Name Should Match File Location

When naming packages, the package name should reflect the file location. This makes the package implementation easier to find when looking up from elsewhere in a project as well.

When choosing to follow this recommendation, there are two options:

- **Matching the directory and filename**
  - Pros: Reduced nesting for simple policies.
  - Cons: Large packages can become unwieldy in long files.
- **Matching the directory only**
  - Pros: Large packages can be broken into many files.
  - Cons: Exception needed to co-locate test files (i.e. `package foo_test` should still be in `foo/`).

Either is acceptable, just remember to use the same convention throughout your project.

#### Matching the Directory and Filename

**Avoid**

```rego
# foo/bar.rego
package bar.foo

# ...
```

**Prefer**

```rego
# foo/bar.rego
package foo.bar

# ...
```

#### Matching the Directory Only

**Avoid**

```rego
# foo/bar.rego
package baz

# ...
```

**Prefer**

```rego
# foo/bar.rego
package foo

# ...
```

## Imports

### Prefer Importing Packages Over Rules and Functions

Importing packages rather than specific rules and functions allows you to reference them by the package name, making it obvious where the rule or function was declared. Additionally, well-named packages help provide context to assertions.

**Avoid**

```rego
import data.user.is_admin

allow if is_admin
```

**Prefer**

```rego
import data.user

allow if user.is_admin
```

### Avoid Importing `input`

While importing attributes from the global `input` variable might eliminate some levels of nesting, it makes the origin of the attribute(s) less apparent. Clearly differentiating `input` and `data` from values, functions, and rules defined inside of the same package helps in making things _obvious_, and few things beat obviousness!

**Avoid**

```rego
import input.request.context.user

# ... many lines of code later

fin_dept if {
    # where does "user" come from?
    contains(user.department, "finance")
}
```

**Prefer**

```rego
fin_dept if {
    contains(input.request.context.user.department, "finance")
}
```

**Prefer**

```rego
fin_dept if {
    # Alternatively, assign an intermediate variable close to where it's referenced
    user := input.request.context.user
    contains(user.department, "finance")
}
```

**Notes / Exceptions**

In some contexts, the source of data is obvious even when imported and/or renamed. A common practice is to rename `input` in Terraform policies for example, either via `import` or a new top-level variable.

```rego
import input as tfplan

violations contains message if {
    # still obvious where "tfplan" comes from, perhaps even more so — this is generally acceptable
    some change in tfplan.resource_changes
    # ...
}
```

## Additional Recommendations

### Use JSON Schemas for Input and Data Structure Validation

- Helps catch typos or wrong attribute paths.
- Improves editor integration.
- Ensures type stability.

### Learn Built-in Functions

Rego has over 150 built-ins. Many style concerns vanish when you use built-ins effectively.

### Use Negation Carefully

Negation interacts with undefined values.

**Example:**

```rego
not some_condition
```

is true if `some_condition` is false or undefined.

For "deny" style policies, prefer making missing fields explicit, for example:

```rego
has_field if input.x != null
```

or use:

```rego
input.x == true
```

instead of relying on negation.

## Older Advice

### Use Explicit Imports for Future Keywords

**With the introduction of the `import rego.v1` construct in OPA v0.59.0, this is no longer needed**

In order to evolve the Rego language without breaking existing policies, many new features require importing "future" keywords, like `contains`, `every`, `if` and `in`. While it might seem convenient to use the "catch-all" form of `import future.keywords` to import all of the future keywords, this construct risks breaking your policies when new keywords are introduced, and their names happen to collide with names you've used for variables or rules.

**Avoid**

```rego
import future.keywords

severe_violations contains violation if {
    some violation in input.violations
    violation.severity > 5
}
```

**Prefer**

```rego
import future.keywords.contains
import future.keywords.if
import future.keywords.in

severe_violations contains violation if {
    some violation in input.violations
    violation.severity > 5
}
```

**Tip**: Importing the `every` keyword implicitly imports `in` as well, as it is required by the `every` construct. Leaving out the import of `in` when `every` is imported is considered okay.

## Summary

- Prefer clarity over cleverness.
- Use `snake_case` everywhere.
- Use METADATA annotations.
- Use helper rules, sets, and `some ... in` / `every ... in` patterns.
- Use `opa fmt` and Regal for consistency.
- Structure policies to be modular, readable, and easy to maintain.

## Contributing

This document is meant to reflect the style preferences and best practices as compiled by the OPA community. As such, we welcome contributions from any of its members. Since most of the topics in a guide like this are likely subject to discussion, please open an issue, and allow some time for people to comment, before opening a PR.

If you'd like to add or remove items for your own company, team or project, forking this repo is highly encouraged!
