---
description: "Contractor likec4@1 validation reference: exact no-argument tool contract, fixed CLI invocation, bounded result shape, and failure semantics."
---

# CLI Reference (validation behavior of `validate_likec4`)

The agent does **not** invoke `likec4` directly — there is no shell tool. All
validation goes through the `validate_likec4` tool, which wraps the LikeC4 CLI
internally. This file documents the contract so you can interpret tool output
correctly and avoid asking for CLI capabilities that are not exposed.

## What `validate_likec4` runs

After `load_likec4` or `write_likec4` establishes the current exact document,
call:

```text
validate_likec4()
```

The tool accepts no path or other arguments. It validates the persisted current
artifact, not uncommitted filesystem content. The runtime requires an installed
`likec4` executable; it does not fall back to package runners or inspect a
project `package.json`.

Internally, the runtime creates an isolated directory, writes the current
document as `main.c4`, and executes a fixed bounded invocation:

```text
likec4 validate --json --no-layout --file <temporary-main.c4> <temporary-project-dir>
```

## Validate flags (and why they matter for output)

- `--json` — emits structured output which the tool parses and sanitizes.
- `--no-layout` — skips layout drift checks; only syntax + semantic errors
  are reported. This is intentional: layout is not the agent's concern.
- `--file <temporary-main.c4>` — selects the one runtime-owned document.
- `<temporary-project-dir>` — contains only that self-contained document.

## Output shape

`validate_likec4` returns one stable envelope. A successful validation resembles:

```jsonc
{
  "artifact": {"namespace": "likec4", "name": "architecture", "revision": "..."},
  "valid": true,
  "validator": "likec4",
  "validatorAvailable": true,
  "validatorExecutionError": null,
  "issues": [],
  "issuesTruncated": false,
  "stats": {}
}
```

Diagnostic objects are bounded and recursively sanitized. Any reported file
name is rewritten to `main.c4`; never treat it as a host path. At most 100
issues are returned. The optional `stats` object retains only bounded scalar
fields from CLI output.

Validation is clean only when `valid` is true, the validator is available,
`validatorExecutionError` is null, and `issues` is empty. An empty issue list
does not mean success when the executable is unavailable, the process times
out/fails, output is invalid or oversized, or `issuesTruncated` is true.

## Anti-substitutions (what the tool will NOT do)

- It will not run `check`, `lint`, `verify`, or `build`. Validation uses
  `likec4 validate` only.
- It will not export PNG / JSON / DrawIO, run `serve`, run `codegen`, run an
  MCP server, list icons, format files, or sync to LeanIX. None of those
  capabilities are exposed to this agent — do not plan around them.

If a task requires one of those capabilities, report that the assigned Toolset
does not expose it rather than pretending the command ran.
