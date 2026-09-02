# 12 — Workspace code-analysis tools

Status: **Working agreement**

Depends on: [01](01-agent-template.md), [02](02-runtime-and-a2a.md),
[10](10-runtime-filesystems-and-edit-tools.md)

## Purpose and ownership

This document owns the first read-only structural code-analysis Toolset for a
hydrated allocation workspace. It ports the useful Tree-sitter and Trailmark
behavior from `contractor-old` while preserving the v2 Runtime boundary:

- Workflow supplies exact source/state artifacts through `context.workspace`;
- Scheduler pins those artifacts in `AllocationSpec` and places only against
  the Runtime Agent's frozen positive capability snapshot;
- Runtime alone chooses `local` or `memory` workspace storage at process start;
- AgentTemplate grants an exact set of model-visible operations;
- the Worker receives selected tool instances and a narrow `WorkspaceReader`,
  never an AgentTemplate, Artifact catalog, fsspec object or host path.

The Toolset is domain-neutral. OpenAPI, LikeC4, security review and general
repository understanding may all select it, but Scheduler contains no
code-analysis-specific workflow behavior.

## One Toolset with two positive capability levels

The exact ref is `code-analysis@1`. Its complete exported operation set is:

```text
search_def
list_symbols
graph_summary
find_symbol
find_callers
find_callees
paths_between
entrypoint_paths_to
attack_surface
complexity_hotspots
functions_that_raise
```

The first two operations are the **shallow** Tree-sitter surface. They operate
directly on the current immutable `WorkspaceSnapshot` and are portable across
both workspace providers.

The remaining nine operations are the **graph** surface backed by
`trailmark==0.5.0`. They are advertised only by a Runtime configured with
`workspace.storage: local` whose complete offline Trailmark probe succeeds.
Trailmark runs in an allocation-local child process so releasing one allocation
reclaims its graph RSS even though the Runtime Agent process is long-lived.

Typical frozen registrations therefore contain:

```json
{
  "workspaceCapabilities": {"storage": "memory", "modes": ["direct", "overlay"]},
  "supportedToolsets": [
    {"ref": "code-analysis@1", "tools": ["list_symbols", "search_def"]}
  ]
}
```

or:

```json
{
  "workspaceCapabilities": {"storage": "local", "modes": ["direct", "overlay"]},
  "supportedToolsets": [
    {
      "ref": "code-analysis@1",
      "tools": [
        "attack_surface",
        "complexity_hotspots",
        "entrypoint_paths_to",
        "find_callees",
        "find_callers",
        "find_symbol",
        "functions_that_raise",
        "graph_summary",
        "list_symbols",
        "paths_between",
        "search_def"
      ]
    }
  ]
}
```

The examples omit unrelated capability fields and workspace limits. A local
Runtime whose Trailmark probe fails may still advertise only the two shallow
operations. A Runtime without a usable workspace provider advertises none of
`code-analysis@1`.

This is the existing operation-level Toolset capability model, not a new
`shallow`/`deep` capability type. Server's code-backed descriptor knows all
eleven legal operation names. AgentTemplate selects exact names, and placement
requires that exact selected set to be contained in one candidate Runtime's
registered set. There is no silent graph-to-shallow fallback. A Stage requiring
`find_callers` waits for compatible local capacity; a template selecting only
`search_def` and `list_symbols` can run on either provider.

## AgentTemplate and workspace binding

A shallow template can select:

```yaml
spec:
  toolsets:
    - ref: code-analysis@1
      tools:
        - search_def
        - list_symbols
```

A graph-oriented template lists every graph operation it intends to expose:

```yaml
spec:
  toolsets:
    - ref: code-analysis@1
      tools:
        - graph_summary
        - find_symbol
        - find_callers
        - find_callees
        - paths_between
        - entrypoint_paths_to
        - attack_surface
        - complexity_hotspots
        - functions_that_raise
```

Selecting any `code-analysis@1` operation requires the corresponding Workflow
Stage to declare `context.workspace`. Configuration validation rejects a
code-analysis Worker binding without that logical workspace. The same
operations work in `direct` and `overlay` mode and require read access only.
They see the effective tree, including imported overlay state and edits already
committed by earlier tools in the allocation.

`code-analysis@1` is intentionally separate from `source-analysis@1`:

- `source-analysis@1` explicitly opens and searches one ZIP artifact and can be
  used without a hydrated project workspace;
- `code-analysis@1` never accepts an ArtifactRef and always analyzes the current
  effective workspace snapshot;
- adding structural tools does not change the behavior or version of existing
  archive-oriented templates.

The internal Trailmark child is an implementation detail of a trusted local
Toolset. It is not the `runtime-subprocess-launcher` infrastructure channel used
for external validators, is not configured by Runtime labels and is not a new
model-visible process-execution authority. The Server descriptor records no
infrastructure channel for these tools.

## Exact snapshot semantics

Every call begins by obtaining one `WorkspaceSnapshot` through the narrow
`WorkspaceReader`. The snapshot contains the complete effective managed-text
projection, binary path metadata and its canonical digest. The tool call
answers only from that immutable value even if an Edit tool commits a later
workspace version concurrently.

One allocation owns one code-analysis session. The session remembers only its
current snapshot digest and bounded derived state:

- Tree-sitter parses one file at a time and discards every AST after extracting
  compact symbol metadata;
- its bounded cache contains compact per-file symbol rows, never source bytes
  or AST objects, and is keyed by snapshot digest plus relative path;
- Trailmark owns at most one child and one graph for the current digest;
- a call observing a different digest first invalidates all shallow metadata,
  stops the old graph child and builds or queries only the new snapshot;
- no model-visible `refresh` operation exists.

Code-analysis calls serialize within their session. They do not retain the
workspace lock while parsing: the immutable snapshot is their consistency
boundary. A cursor is bound to its operation, normalized arguments and snapshot
digest. Reusing it after an edit fails with `code_analysis_workspace_changed`
rather than combining two trees. The caller restarts the query without a
cursor.

## Bounded coverage and common response fields

Version 1 uses fixed implementation ceilings so a Toolset capability has the
same semantic strength on every Runtime that advertises it:

| Boundary | Ceiling |
|---|---:|
| candidate managed-text source files per snapshot | 20,000 |
| aggregate source bytes parsed/materialized | 128 MiB |
| one analyzed source file | 4 MiB |
| compact cache entries retained for one digest | 20,000 files |
| compact symbols retained for one digest | 100,000 |
| ordinary result page | 200 items |
| one model-visible result after JSON encoding | 256 KiB |
| symbol/query string | 256 Unicode scalar values |
| one shallow scan CPU budget | 10 seconds, checked between bounded files |
| one Trailmark graph query | 10 seconds hard wall time |
| first Trailmark graph build for one digest | 120 seconds |
| Trailmark child address space | 1 GiB |
| path results returned by one traversal | 50 |
| call-path depth | 20 nodes |
| parent/child protocol request | 16 KiB |
| parent/child protocol response | 2 MiB |

Workspace path and total hydration limits still apply first. Source candidates
recognized by the selected engine are ordered by normalized relative path,
then admitted until the file/byte ceilings are reached. A file above the
per-file ceiling is skipped. This makes partial coverage deterministic instead
of depending on filesystem walk order. `unsupportedSourceFiles` counts files
recognized by the other v1 engine's source-extension registry but unsupported
by the selected one; unknown non-source files are ignored.

Every successful analysis response includes bounded coverage metadata:

```json
{
  "coverage": {
    "analyzedFiles": 143,
    "analyzedBytes": 921337,
    "binaryFiles": 4,
    "unsupportedSourceFiles": 2,
    "oversizedFiles": 0,
    "parseErrors": 1,
    "incomplete": true,
    "reasons": ["parse_errors"]
  }
}
```

`incomplete` is true when a file/byte/symbol/deadline ceiling or parser error
can make the answer omit otherwise supported source. Its sorted unique reasons
are drawn from `file_limit`, `byte_limit`, `symbol_limit`, `deadline`, and
`parse_errors`. Binary and unsupported-language counts describe intentional
coverage boundaries and do not alone make the supported-language result
incomplete. Unknown non-source files are ignored rather than counted.

Collection responses use `items`, `nextCursor`, `truncated` and
`observedTotal`. `observedTotal` counts the complete bounded result set, not an
unexamined suffix when `coverage.incomplete` is true. Cursors are opaque,
allocation-local, integrity-protected and at most 2 KiB. Invalid, cross-tool or
cross-allocation cursors fail closed. A page stops before the item or encoded
JSON response ceiling and returns a cursor for the first unreturned ordinary
item. No operation silently cuts off results.

Path traversals are different: they stop traversal after finding `limit + 1`
paths, return at most `limit`, set `truncated` when another path was observed,
or when another path would exceed the encoded response ceiling, and do not
claim a total. They intentionally have no cursor because continuing an
exponential traversal would retain unbounded frontier state; callers narrow the
symbols or depth instead.

## Shallow Tree-sitter surface

The portable implementation pins `tree-sitter==0.25.2` and
`tree-sitter-language-pack==1.14.3`. It supports the retained v1 language names
from `contractor-old`:

```text
python javascript typescript tsx go rust java kotlin c cpp c_sharp ruby php
scala swift lua elixir haskell bash
```

The extension mapping is code-backed and versioned with the Toolset. Startup
probe loads every v1 parser from the installed wheel without network access;
if the fixed set cannot be honored, both shallow operations are omitted rather
than advertising an unexpressed language subset. Runtime startup never
downloads grammars.

### `search_def`

```text
search_def(symbol, path="", language="", cursor="", limit=50)
```

- `symbol` is mandatory and matched against extracted definition names using
  exact and case-folded bare-name comparison;
- `path` is the normalized relative workspace root/subtree;
- `language` is empty or one exact v1 language name;
- `limit` is 1..200.

For efficiency, the implementation first performs a bounded case-folded text
prefilter for the bare symbol, then parses only candidate files and validates
actual definition nodes. Results are sorted by path, start line, column, name
and node type. Each row contains `name`, `path`, `line`, `endLine`, `column`,
`nodeType`, `language` and an optional definition preview capped at 12 lines
and 4 KiB. A caller can use `filesystem@1/read_file` for more context.

Unlike the old implementation, absence of a structural definition returns an
empty result. It does not silently turn into a grep result; a template that
wants textual occurrences explicitly grants `filesystem@1/grep`.

### `list_symbols`

```text
list_symbols(path="", language="", node_type="", cursor="", limit=100)
```

This returns structural definition rows with the same location fields but no
source preview. `language` and `path` have the same meaning as `search_def`.
`node_type` is empty or an exact parser node type; it is deliberately
language-specific rather than pretending v1 has a lossless common symbol-kind
taxonomy. Rows are sorted by path, start line, column, name and node type.

Both tools perform CPU parsing outside the asyncio event-loop thread. The
10-second budget is cooperative between files because a Tree-sitter C parse
cannot be safely preempted inside a Python thread; the 4 MiB per-file ceiling
bounds one uninterrupted parser call. Budget exhaustion after that call returns
explicit `deadline` coverage and retains no partial AST. Their cache may
accelerate later calls but cannot change results or cursor ordering.

## Trailmark graph surface

### Build boundary

Graph support pins exactly `trailmark==0.5.0`; a version change requires a new
review, lock update and either proven compatibility or a new Toolset version.
All Python packages and grammar data are installed in the Runtime image or
environment before startup. Capability probing is finite and offline.

On the first graph call for a digest, the Runtime:

1. takes the exact current `WorkspaceSnapshot`;
2. creates a uniquely named mirror below the allocation's private scratch
   directory;
3. writes only admitted managed UTF-8 files at normalized relative paths;
4. starts one child with direct argv, no shell, a sanitized environment and no
   Allocation RuntimeSettings, LLM/API credentials or Artifact grant;
5. has the child build a public Trailmark `CodeGraph` from that mirror in
   `auto` language mode, detect entrypoints and construct query indexes;
6. serves typed bounded requests over a framed, size-bounded local protocol.

The graph never reads `ProjectWorkspaceStorage.root`, the local provider's live
physical underlay or a model-supplied path. This matters for both modes: the
direct storage tree is an implementation detail, and an overlay's lower tree
is intentionally stale after edits. Snapshot materialization makes the graph
match the same effective tree as shallow and filesystem tools and prevents
host paths from escaping in results.

Only `local` workspace Runtime Agents advertise graph operations even though a
memory workspace could technically be copied to allocation scratch. This is a
deliberate first-version resource policy: graph RSS belongs in a killable child
on a Runtime provisioned for disk-backed project work, while memory Runtime
Agents retain the low-overhead shallow surface.

The child uses Trailmark's public parse-only API and retained `CodeGraph` data.
The adapter does not reach through `QueryEngine._store._graph`, install a
process-global parser monkeypatch or expose raw Trailmark objects. The mirror
already excludes binary/non-UTF-8 data, so the old global UTF-8 workaround is
unnecessary.

Trailmark 0.5.0 graph language coverage is the exact supported set returned by
its pinned public API: Python, JavaScript, TypeScript/TSX, PHP, Ruby, C, C++,
C#, Java, Go, Rust, Solidity, Cairo, Circom, Haskell, Erlang, MASM, Swift,
Objective-C, Kotlin, Dart, Move, Tact, FunC, Sway, Rego, Protobuf, Thrift,
GraphQL and SQL. A shallow-supported language outside this set is counted as
unsupported for graph coverage.

### Stable symbol identity

Raw Trailmark IDs and its ambiguous first-name resolver are not model-facing.
For each graph, Contractor maps every Trailmark node to a deterministic-within-
allocation opaque `symbolId`, integrity-protected by an allocation-local random
key and bound to the workspace digest plus complete upstream node ID. The token
contains a bounded digest tag so the session can distinguish stale from unknown
IDs without retaining an old graph. Collisions fail the build, cross-allocation
IDs fail validation, and the key is erased on close. All graph results expose
only this ID, symbol name, kind and normalized relative source location; they
never expose the mirror, workspace-provider or Runtime host path.

`find_symbol` is the only name-to-ID operation. Every relationship/path tool
requires an exact `symbolId`; it never chooses the first equal bare name. An ID
from an older digest fails with `code_analysis_stale_symbol`.

### Operations

| Tool | Contract |
|---|---|
| `graph_summary()` | Bounded node/call-edge/entrypoint/language counts and coverage; no full dependency map. |
| `find_symbol(query, cursor="", limit=50)` | Exact/case-folded name and qualified-name candidates with opaque `symbolId` and relative locations. |
| `find_callers(symbol_id, cursor="", limit=100)` | Direct call-edge predecessors with confidence and slim source locations. |
| `find_callees(symbol_id, cursor="", limit=100)` | Direct call-edge successors, including bounded unresolved proxy nodes. |
| `paths_between(source_id, target_id, max_depth=20, limit=20)` | Deterministic simple call paths, capped before unbounded Trailmark enumeration. |
| `entrypoint_paths_to(symbol_id, max_depth=20, limit=20)` | Deterministic simple paths from detected entrypoints to one exact node. |
| `attack_surface(cursor="", limit=100)` | Slim detected entrypoints with bounded known route/trust metadata. |
| `complexity_hotspots(threshold=10, cursor="", limit=100)` | Nodes with detected cyclomatic complexity at or above 1..10,000, highest first. |
| `functions_that_raise(exception, cursor="", limit=100)` | Nodes whose parsed exception type exactly matches the bounded name. |

`limit` for ordinary collections is 1..200. Path `limit` is 1..50 and
`max_depth` is 1..20. Rows and adjacency lists use deterministic sorting before
pagination/traversal. A path is a list of slim symbol projections, not a list
of raw Trailmark IDs.

Trailmark's public `paths_between` and `entrypoint_paths_to` implementations
materialize all simple paths before the caller can slice them. Contractor must
not invoke those unbounded operations. The child performs its own deterministic
depth-limited traversal over the retained CodeGraph call edges and stops after
`limit + 1` results.

## Child failure and lifecycle

The child is lazy: selecting a graph tool does not build a repository graph at
allocation prepare. Build/query/mirror replacement serialize per allocation
and run outside the Runtime asyncio event-loop thread.

The parent enforces build/query deadlines, protocol bounds and the fixed child
memory ceiling. A Runtime whose OS cannot install and verify that memory/process-
group boundary omits graph operations during probe. A timeout, malformed frame,
unexpected exit or memory-limit termination kills and reaps the complete child.
A definitely-unprocessed read-only operation may be replayed once after one
clean child start/rebuild; an operation that
already exceeded its deadline is not automatically repeated. The next model
call may cause a fresh build. There is no mutation or external side effect to
reconcile.

On graceful Worker terminalization, Runtime closes the analysis session and
reaps the child while retaining the ordinary project workspace for existing
release semantics. The later release idempotently confirms that teardown and
removes any residual bounded state; ordinary success leaves no mirror or cache.
Abort, lost lease, failed prepare and process shutdown cancel analysis, kill
and reap the child, and remove the mirror before a slot can become reusable.
Tool close is idempotent and composes with the allocation-wide bounded cleanup
task in [10](10-runtime-filesystems-and-edit-tools.md). Failure to confirm child
termination or mirror cleanup leaves the Runtime fenced; it cannot return an
apparently clean slot.

No graph, index or mirror is persisted as an Artifact or reused by another
allocation. Router Workers receive independent workspace sessions, mirrors,
children, allocation keys and symbol IDs even when their initial source
artifacts are equal.

## Built-in Workflow variants

The repository publishes additive workspace-backed discovery choices; it does
not mutate the original identities:

| Workflow versions | Discovery AgentTemplate | Placement effect |
|---|---|---|
| `openapi-from-workspace@1`, `likec4-from-workspace@1` | `workspace_source_analyst@1` | Original filesystem-only behavior |
| `openapi-from-workspace@2`, `likec4-from-workspace@2` | `workspace_source_analyst@2` | Selects only `search_def` and `list_symbols`; local or memory is eligible |
| `openapi-from-workspace@3`, `likec4-from-workspace@3` | `workspace_source_graph_analyst@1` | Selects all eleven operations; waits for complete local graph capacity |

Only `dependency_discovery` and `project_discovery` change template selection.
The four-Stage graph, Planner instructions, input/output declarations, explicit
Markdown report handoff, overlay state/diff handoff, retry/escalation rules,
execution configuration and builder/validator template versions are byte-for-
byte-equivalent to `@1` after accounting for Workflow/template identity. There
is no analyzer-aware Scheduler branch and no fallback from `@3` to `@2`.

Both new discovery templates retain only bounded filesystem reads and explicit
text-artifact read/write operations in addition to their exact analysis
allowlist. They have no Edit or workspace-change tool authority. Their Worker
instructions explain model-visible operation semantics, opaque graph identity,
coverage and truncation without describing configuration objects, placement,
process identity, scratch paths or implementation protocols.

## Stable errors and telemetry

The initial model-visible error vocabulary is:

- `workspace_required` — no project workspace was supplied;
- `code_analysis_input_invalid` — invalid query, language, node type, limit,
  depth or path;
- `code_analysis_cursor_invalid` — malformed or wrong-operation cursor;
- `code_analysis_workspace_changed` — cursor or pending cache belongs to an
  older effective tree; retry without the cursor;
- `code_analysis_symbol_not_found` — the exact graph `symbolId` is absent;
- `code_analysis_stale_symbol` — the ID belongs to an older graph digest;
- `code_analysis_build_timeout`;
- `code_analysis_query_timeout`;
- `code_analysis_capacity_exceeded` — fixed source, protocol or child-memory
  boundary prevented the requested graph result;
- `code_analysis_engine_failed` — bounded Trailmark/child failure not covered
  by a more specific code;
- `code_analysis_cancelled` — the invocation was cancelled while an off-loop
  operation was being joined;
- `code_analysis_closing` — allocation termination has begun.

Invalid input, stale/not-found symbol and deterministic capacity failures are
non-retryable for the same request. Workspace change is retryable by restarting
the tool query. Unexpected child failure is retryable only after the one
internal read-only recovery attempt has failed. Errors contain a stable code,
retryability and bounded structural counts; they never contain source text,
query strings, relative/host paths, raw child stderr or arbitrary exceptions.

Worker metrics may contain Toolset/operation, shallow or graph engine, outcome,
duration, analyzed file/byte/symbol counts, skip/error counts, cache
hit/invalidation, build/rebuild count and classified child termination. Metrics,
logs, final reports and OTLP must not contain symbol/query text, source paths,
source/preview content, raw IDs, cursors, scratch paths or credentials. Child
stdout is protocol-only; stderr is bounded and reduced to a safe failure class
instead of being copied to logs.

## Port boundary

The initial port retains the two read-only Tree-sitter tools and nine read-only
Trailmark tools listed above. Existing unit fixtures may be adapted as semantic
references, but old architecture is not authoritative.

The following old behavior is explicitly excluded:

- `annotate_trace`, `annotate_validate`, `annotate_sink` and all other mutable
  graph annotations; a future reviewed `code-annotations@1` may own them;
- preanalysis/findings, SARIF, weAudit and binary graph augmentation;
- raw AST, CodeGraph or graph serialization exposed to the model;
- model-visible code/script execution;
- automatic host-root discovery or direct reads through a concrete fsspec
  backend;
- process-global Trailmark patches, private `_store._graph` access and
  first-match ambiguous symbol resolution;
- persisted/shared graph artifacts, cross-allocation caches and hidden
  cross-Run reuse;
- a model-visible refresh/materialize operation or silent shallow fallback.

## Initial acceptance

1. Server and Runtime agree on exactly eleven exported operation names, while
   Runtime registration may report only a positive subset.
2. Memory and local processes with shallow dependencies advertise and execute
   `search_def`/`list_symbols` over equal snapshots with equal normalized rows.
3. Only a successfully probed local process advertises graph operations; a
   graph-requiring Stage waits rather than landing on memory or silently using
   shallow analysis.
4. Direct and overlay edits are visible on the next call because every engine
   builds from the effective snapshot rather than the provider underlay.
5. Shallow parsing leaves no retained AST/source cache and never blocks the
   asyncio heartbeat during a deliberately slow parse.
6. Trailmark runs in one allocation-local child, receives no RuntimeSettings or
   credentials, exposes no host path and is reaped on digest change, terminal
   completion, abort, lease loss and release.
7. Duplicate symbol names produce multiple `find_symbol` candidates; every
   graph relationship query accepts only one returned opaque ID.
8. Adversarial high-branching graphs prove path traversal stops at `limit + 1`
   and depth, response and deadline bounds without first materializing all
   simple paths.
9. File/byte/symbol/parser limits produce deterministic explicit incomplete
   coverage; pagination never claims an exact total for unexamined data.
10. Binary, unsupported, oversized, malformed and polyglot fixtures preserve
    confinement and report bounded coverage without a global parser patch.
11. Child hang, crash, OOM, malformed protocol, teardown timeout and concurrent
    edit tests fail closed, keep heartbeats responsive and prevent dirty slot
    reuse.
12. A real two-process heterogeneous test runs a shallow template on memory,
    holds a graph Stage until local capacity appears, then completes it and
    proves both slots can be reused.

## Invariants

1. AgentTemplate grants tools; installed analyzers and labels never add them.
2. Code analysis reads one exact effective workspace snapshot per call.
3. Shallow capability is portable; graph capability is local-only in version 1.
4. Capability placement is exact and has no semantic fallback.
5. Trailmark is pinned, offline, allocation-local and killable.
6. No model or Server contract contains a Runtime host path or raw filesystem
   handle.
7. No graph operation resolves an ambiguous name by choosing the first match.
8. Every scan, result, traversal, child and cache has a finite explicit bound.
9. Workspace mutation invalidates derived state automatically by digest.
10. Release cannot make a slot reusable while an analysis child, mirror or
    source-bearing cache remains live.
