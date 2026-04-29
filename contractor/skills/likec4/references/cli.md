# CLI Reference (validation behavior of `validate_likec4`)

The agent does **not** invoke `likec4` directly — there is no shell tool. All
validation goes through the `validate_likec4` tool, which wraps the LikeC4 CLI
internally. This file documents the contract so you can interpret tool output
correctly and avoid asking for CLI capabilities that are not exposed.

## What `validate_likec4` runs

Internally, the tool executes:

```text
likec4 validate --json --no-layout --file <path> <project-dir>
```

- The runtime resolves a launcher in this order: `likec4` (if installed) →
  `bunx` → `pnpx` → `npx`. Fallback runners are auto-confirmed (no interactive
  prompt). You do not pick a launcher — it is selected for you.
- The required minimum LikeC4 version is `>= 1.53.0`. If the project pins
  another version in `package.json`, the runtime will respect the pin.

## Validate flags (and why they matter for output)

- `--json` — emits a structured JSON payload on stdout; logs go to stderr.
  The tool parses the payload and returns it to you.
- `--no-layout` — skips layout drift checks; only syntax + semantic errors
  are reported. This is intentional: layout is not the agent's concern.
- `--file <path>` — scopes results to the file you passed in. The payload's
  `filteredFiles` / `filteredErrors` reflect this scope.
- `<project-dir>` — implied by the file location; the tool determines it
  internally from a temp project directory.

## Output shape

`validate_likec4` returns one of:

```jsonc
// Success path — empty list means clean
{ "result": [ /* issue objects */ ] }
```

```jsonc
// Failure path — execution problem, parse error, or unexpected shape
{ "error": "...", "details": "...", "raw_output": ... }
```

Each issue object carries `message`, `file`, `line`, and a `range`. When the
underlying CLI emits the dict form, the report also includes a `stats` block:

| Field            | Meaning                                                |
| ---------------- | ------------------------------------------------------ |
| `totalFiles`     | Total `.c4` / `.likec4` source files in the project    |
| `totalErrors`    | All errors across the full project                     |
| `filteredFiles`  | Files actually included by the `--file` filter         |
| `filteredErrors` | Errors only in the filtered subset                     |

If `filteredErrors == 0` but `totalErrors > 0`, the file you edited is clean —
the cascade is from another file you do not own. Self-check: `filteredFiles`
should equal the number of files you focused on.

## Anti-substitutions (what the tool will NOT do)

- It will not run `check`, `lint`, `verify`, or `build` — none of those exist
  in `likec4`. Validation is `likec4 validate` only.
- It will not export PNG / JSON / DrawIO, run `serve`, run `codegen`, run an
  MCP server, list icons, format files, or sync to LeanIX. None of those
  capabilities are exposed to this agent — do not plan around them.

If a task asks for one of those capabilities, surface that the agent cannot
execute it and stop, rather than emitting a CLI command the agent cannot run.
