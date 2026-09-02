# 13 — Structured taint annotations

Status: **Working agreement**

Depends on: [01](01-agent-template.md), [02](02-runtime-and-a2a.md),
[10](10-runtime-filesystems-and-edit-tools.md),
[12](12-code-analysis-tools.md)

## Purpose and boundary

`taint-annotations@1` is the specialized source-mutation Toolset for the
machine-readable comments used while tracing data and control flow:

```text
@trace target=<target> [args=<name:state,...>] [calls=<symbol,...>]
@validate arg=<name> kind=<kind>
@sink kind=<kind> arg=<name-or-unknown>
```

It ports the useful `contractor-old` annotation operations while retaining the
v2 ownership model:

- Workflow supplies exact ZIP and optional overlay-state artifacts through
  `context.workspace`;
- Scheduler pins those refs in `AllocationSpec` and performs ordinary exact
  capability placement;
- Runtime alone hydrates disposable local or memory storage;
- AgentTemplate grants the exact model-visible operations;
- Worker receives tool instances over a narrow `WorkspaceWriter`, not an
  AgentTemplate, Artifact catalog, fsspec object, provider root or host path.

The Toolset mutates only the effective allocation workspace. It does not mutate
Trailmark or any persistent graph. A later `code-analysis@1` call observes the
new workspace digest and rebuilds its derived state under [12]. No Server-side
annotation service, database table, Artifact endpoint or infrastructure
channel is introduced.

## Exact Toolset and capability

The exact ref is `taint-annotations@1`; it exports exactly:

```text
annotate_trace
annotate_validate
annotate_sink
```

All three operations are available together after the Runtime's exact pinned
Tree-sitter dependencies and all nineteen v1 language parsers pass their
bounded offline startup probe. They are portable across `local` and `memory`
workspace storage and work in both `direct` and `overlay` mode. A Runtime with
no workspace provider or a failed parser probe advertises no
`taint-annotations@1` capability. There is no partial language capability and
no grep fallback.

The factory declares `requires_workspace=true`, `workspace_access=write` and
no infrastructure channels. Selecting any operation therefore requires
`context.workspace`, but does not by itself require overlay mode. In direct
mode the mutation lives only in the disposable allocation copy. A Workflow
that needs durable annotation output selects overlay mode and ordinary
workspace `state`/`diff` result slots; the pre-terminal auto-export contract in
[10] remains the only persistence path.

`taint-annotations@1` is independent of `code-analysis@1`. A template may
select either or both. The annotation resolver uses Tree-sitter internally but
does not accept or emit the graph Toolset's allocation-private `symbolId`.

## Model-visible operations

All paths are normalized workspace-relative POSIX paths. `symbol` is the exact
unqualified structural name of a function-like declaration. Matching is
case-sensitive. Callers may omit `definition_line` by passing its default `0`
when `(path, symbol)` identifies exactly one declaration. If more than one real
declaration matches, the operation fails with `taint_annotation_target_ambiguous`;
the caller then supplies a positive 1-based `definition_line` observed in the
current source or structural-search result. Wrapper rows belonging to the same
decorated/exported declaration are deduplicated and accepted as selectors for
that one declaration.

### `annotate_trace`

```text
annotate_trace(
  path,
  symbol,
  target="unknown",
  args="",
  calls="",
  definition_line=0
)
```

`args` is a comma-separated ordered list of `name:state` entries. State is
exactly one of `tainted`, `validated`, `clean` or `derived`. `calls` is a
comma-separated ordered list of direct target-relevant callees. Empty
`args`/`calls` fields are omitted from the inserted line. `target` defaults to
the literal `unknown`; it is never inferred by Runtime.

One declaration may have multiple trace lines for different targets. Their
order is insertion order. A second line for the same target with different
arguments or callees is a conflict rather than a silent rewrite.

### `annotate_validate`

```text
annotate_validate(path, symbol, arg, kind, definition_line=0)
```

This inserts one `@validate arg=<arg> kind=<kind>` line. Runtime validates only
the bounded machine syntax. The trace Skill supplies the semantic vocabulary
and evidence discipline; Runtime does not guess whether visible code really
validates the argument.

### `annotate_sink`

```text
annotate_sink(path, symbol, kind, arg="unknown", definition_line=0)
```

This inserts one `@sink kind=<kind> arg=<arg>` line. `arg` defaults to
`unknown`. As with validation, the selected Skill/Worker owns semantic
classification and Runtime owns only safe structural mutation.

Every successful response has the closed shape:

```json
{
  "path": "src/handlers.py",
  "symbol": "get_order",
  "kind": "trace",
  "annotationLine": 7,
  "definitionLine": 9,
  "changed": true
}
```

It contains neither source excerpts nor annotation text. On exact replay,
`changed` is false and the line numbers describe the already-current file.

## Structural resolution and placement

Version 1 recognizes only executable function-like declarations in the same
nineteen languages as the shallow code-analysis surface. Classes, interfaces,
types, imports, call sites and non-callable variables never qualify. Named
arrow/closure assignments supported by the pinned grammar qualify under their
assigned variable name.

The resolver parses only the named current UTF-8 managed-text file. It finds
all exact structural matches, folds parser wrapper nodes belonging to one real
declaration, and applies `definition_line` only after that deduplication. It
never selects the first ambiguous result.

The inserted comment uses the target declaration's indentation and existing
file newline style. Marker selection is fixed:

| Languages | Marker |
|---|---|
| Python, Ruby, Bash, Elixir | `#` |
| Haskell, Lua | `--` |
| JavaScript, TypeScript, TSX, Go, Rust, Java, Kotlin, C, C++, C#, PHP, Scala, Swift | `//` |

Insertion precedes the declaration's complete syntax-owned prefix. This means
it lands above Python/TypeScript decorators, Rust/PHP attributes, Java/C#
annotations, JavaScript exports and C++ templates rather than splitting those
constructs. Existing immediately adjacent canonical taint annotations remain
one contiguous block, and new lines append to that block so multi-target order
is stable. Unrelated whitespace, comments and source bytes do not change.

## Input limits and canonical form

Existing workspace path and tree limits apply first. Additionally:

| Boundary | Limit |
|---|---:|
| one parsed/updated file | 4 MiB UTF-8 |
| `symbol` | 256 Unicode scalar values |
| one emitted token (`target`, argument, call or kind) | 128 ASCII characters |
| `args` entries | 32 |
| `calls` entries | 32 |
| canonical annotation line | 4 KiB UTF-8 |
| `definition_line` | `0` or `1..2147483647` |

Emitted tokens use only ASCII letters, digits and the reviewed punctuation
`_ . $ : / { } [ ] * ? + - < >`. Argument names additionally exclude `:`
because it separates state. Empty entries, duplicates, whitespace, control
characters, `=`, comma, comment introducers and newline injection are rejected.
Runtime canonicalizes comma-separated lists without spaces and preserves their
declared order.

## Atomicity, replay and lifecycle

One allocation owns one annotation session and calls through it serialize. A
mutation follows this sequence:

1. read one current file snapshot through `WorkspaceWriter`;
2. parse and resolve it off the asyncio event loop;
3. enter `WorkspaceWriter.update_text`, compare the locked current bytes with
   the parsed bytes, and apply the already-computed single-line insertion only
   if they still match.

Any Edit tool or other writer that wins between steps 1 and 3 causes the
retryable `taint_annotation_workspace_changed` error. The annotation call never
overwrites that change. There is no lock shared between Toolsets and no
non-atomic read-then-write fallback.

An exact canonical annotation already in the target's adjacent annotation
block is successful idempotent replay (`changed=false`). This covers a lost
tool response after a committed write. For `@trace`, the same target with a
different canonical body returns `taint_annotation_conflict`; different target
lines remain valid. Exact duplicate `@validate` or `@sink` lines are no-ops;
distinct validation and sink lines may coexist.

Cancellation while Tree-sitter is running waits for the bounded worker thread
before releasing session ownership, then reports cancellation. Workspace
commit itself is short and non-awaiting under the workspace lock: cancellation
may happen before it or after a complete commit, never in the middle. A replay
after an uncertain response reconciles through the exact no-op rule.

`close()` is idempotent, prevents new calls and waits for any active parse or
mutation. Normal finalize, abort, lease loss and release use the existing
Toolset-before-workspace teardown order. No parser tree, source snapshot,
physical path or annotation body survives allocation cleanup.

## Stable failures and telemetry

The model-visible error vocabulary is closed:

- `workspace_required`;
- `taint_annotation_input_invalid`;
- `taint_annotation_language_unsupported`;
- `taint_annotation_target_not_found`;
- `taint_annotation_target_ambiguous`;
- `taint_annotation_conflict`;
- `taint_annotation_workspace_changed` (retryable);
- `taint_annotation_capacity_exceeded`;
- `taint_annotation_cancelled` (retryable);
- `taint_annotation_closing`;
- `taint_annotation_unavailable` (retryable).

Errors never include source, annotation text, input values, logical or physical
paths, parser diagnostics or arbitrary exceptions. Raw ADK arguments are
validated before framework binding so unknown fields and wrong types also map
to `taint_annotation_input_invalid`.

Worker metrics may contain operation, kind, outcome, duration and `changed`.
They must not retain path, symbol, target, argument/call/kind values, source,
annotation text, parser diagnostics, workspace digest, physical path or
credentials. Tool results sent intentionally to the model may contain the
closed success projection above; automatic logs, final reports and OTLP do not.

## AgentTemplate and trace Skill

A trace-oriented template explicitly selects its authority:

```yaml
spec:
  toolsets:
    - ref: filesystem@1
      tools: [ls, glob, read_file, grep]
    - ref: code-analysis@1
      tools: [search_def, list_symbols]
    - ref: taint-annotations@1
      tools: [annotate_trace, annotate_validate, annotate_sink]
    - ref: workspace-changes@1
      tools: [changed_paths, diff, rollback_changes]
  skills:
    - namespace: skills
      name: trace
```

The Skill adds guidance, not tools. The checked-in trace template intentionally
does not select generic `edit-files@1`; annotation comments therefore go
through the structured surface. This is an AgentTemplate policy, not a hidden
Runtime ban: another explicitly versioned template may select both Toolsets.

The Worker uses `changed_paths` and `diff` to verify intended edits. It never
calls a materialize operation. When its Workflow selects overlay export, the
resulting cumulative state can be applied by a later Stage and the text diff can
be analyzed as an ordinary artifact, as defined in [10].

## Port boundary and acceptance

The old implementation and tests are semantic references, not authoritative
architecture. Version 1 intentionally changes these behaviors:

- raw fsspec/root access becomes one narrowed allocation Writer;
- `file/function` become normalized `path/symbol`, with a line selector for
  real ambiguity;
- first-match resolution and grep fallback are rejected;
- decorators/attributes/exports/templates remain syntactically intact;
- different trace targets may coexist and exact replay is a no-op;
- source/comment text and arbitrary filesystem/parser errors are not returned
  or retained;
- PHP follows the checked-in trace Skill and uses `//`; Haskell and Lua use
  their actual `--` line comment marker.

The initial increment is accepted when:

1. Go and Python agree on the exact ref/tools and local/memory Runtime Agents
   advertise them only after a complete parser/workspace probe.
2. All three operations produce byte-minimal valid edits across the nineteen
   language fixtures, including decorator/wrapper and CRLF cases.
3. Undefined and ambiguous symbols, unsafe inputs, capacity limits, exact
   replay, conflicts and cross-Toolset write races have deterministic outcomes.
4. Direct and overlay providers behave equally; overlay state/diff auto-export
   remains the only durable output mechanism.
5. Cancellation/close/release cannot leave work running or state retained, and
   metrics/logs/reports contain no content/path/value canaries.
6. A checked-in trace AgentTemplate/Workflow resolves the bundled Skill,
   selects no generic source writer and completes through PostgreSQL, mTLS,
   Scheduler, A2A and a real Runtime process with a reusable clean slot.
