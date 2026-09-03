# 16 — WorkflowRun metadata labels

This document owns immutable, queryable metadata labels attached to one
`WorkflowRun`. These labels identify and group executions. They are distinct
from the Runtime labels in [07](07-runtime-labels-and-infrastructure-config.md),
which select infrastructure configuration.

The first motivating use is evaluation. One eval invocation may create many
ordinary WorkflowRuns for fixtures, cases, repeated samples and A/B legs. A
shared label set must make those Runs discoverable without introducing a
second execution path into Scheduler.

## Boundary and terminology

There are two deliberately separate label domains:

| Field | Shape | Meaning |
|---|---|---|
| `runtimeLabels` | sorted unique array of names | Selects and pins RuntimeConfig bindings such as `debug` or `caido` under [07] |
| `labels` | key/value string map | Immutable metadata describing one WorkflowRun |

Runtime labels may change infrastructure resolution. WorkflowRun metadata
labels never change Workflow semantics, Scheduler eligibility, queue priority,
Runtime capabilities, authorization, budgets, tools, Skills, model selection
or artifact access.

The pre-feature public `labels` array is renamed to `runtimeLabels`. The first
metadata-label version is a strict contract cutover: Server does not accept the
old array and the new map under one overloaded field. Server, UI and bundled
clients are upgraded together. Durable RuntimeConfig snapshot tables and refs
need no semantic migration merely because their public field is renamed.

## Public Run contract

`POST /v1/runs` accepts both domains independently:

```json
{
  "workflow": "openapi-from-source@1",
  "runtimeLabels": ["debug"],
  "labels": {
    "purpose": "eval",
    "eval.name": "openapi-regression",
    "eval.id": "eval_01k4example",
    "eval.leg": "a",
    "eval.fixture": "vulnyapi",
    "eval.case": "sqli-01",
    "eval.sample": "2"
  },
  "parameters": {},
  "artifacts": {}
}
```

Omitting `labels` is equivalent to an empty map. Explicit `null`, an array or
non-string values are invalid. Successful Run-create, Run detail and Run list
projections expose the complete normalized map. Lifecycle events continue to
carry `run_id`; they do not repeat the full map in every event.

The first slice permits at most 32 entries. A key is 1–63 ASCII bytes and uses:

```text
[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*
```

A value is non-empty UTF-8 of at most 256 bytes after JSON decoding; U+0000 is
excluded because it has no PostgreSQL `text` representation. Values are
opaque: they may contain paths, spaces or punctuation, and Server performs no
case folding or semantic parsing. `contractor.` is reserved for future
Server-owned keys; caller-supplied keys in that namespace are rejected.
Duplicate JSON object keys are rejected by the strict request decoder rather
than resolved by last-value-wins behavior.

Labels are safe metadata, not a secret container. A caller must not place
credentials, tokens, prompts, source excerpts or other confidential payloads
in them. The Server cannot reliably infer secrecy from arbitrary text and does
not promise value redaction after accepting a label.

## Identity, immutability and idempotency

The normalized metadata-label map is immutable for the complete WorkflowRun
lifetime. Version 1 has no add, replace or delete endpoint. A future mutable
operator-tag feature requires its own compare-and-set contract and must not
silently weaken eval provenance.

Run creation stores all metadata labels in the same transaction as the Run,
its immutable Workflow snapshot, input forks, RuntimeConfig pins and Skill
snapshot. A committed Run therefore never exists with a partially stored label
set.

The canonical Run-create request digest includes the complete label map sorted
by key, independently of the sorted `runtimeLabels` array. Exact idempotency
replay returns the original Run and original labels. Reusing an idempotency key
with any changed label key or value conflicts, even when Workflow inputs and
executionConfig are otherwise equal.

Labels belong to the Run owner and follow existing owner isolation. They grant
no access and are not trusted claims about the caller. Operations may display
them, but neither Scheduler nor Control Plane authorizes behavior from their
contents.

## Durable storage and queries

Metadata labels are durable first-class indexed data, not a substring encoded
into `run_id`, a log field or an artifact name. Storage must enforce one value
per `(run_id, key)`, the entry/key/value bounds and atomic insertion with the
Run. Terminal retention follows the WorkflowRun retention policy.

The owner Run-list endpoint accepts a repeated exact selector:

```text
GET /v1/runs?label=purpose=eval&label=eval.id=eval_01k4example
GET /v1/runs?label=eval.name=openapi-regression&label=eval.leg=b
```

The first `=` separates key from value; the remaining decoded characters
belong to the value. Repeated selectors are logical AND. Version 1 supports
only exact key/value equality: no existence, inequality, set, regex, prefix or
OR syntax. Invalid selectors fail the request; an unknown or unmatched pair
returns an empty page.

Filtering composes with the existing owner, state and keyset-pagination
constraints. Pagination remains stable newest-first and must neither leak
foreign-owner labels nor produce duplicate Runs when a Run has several
matching labels. The implementation must use an index-backed query rather than
scan and deserialize every historical Run.

## Evaluation convention

The eval subsystem owns a convention over generic labels; Server does not add
eval-specific branches or validation.

| Label | Meaning |
|---|---|
| `purpose=eval` | Classifies an ordinary WorkflowRun as an eval execution |
| `eval.name` | Stable human-readable suite or experiment name |
| `eval.id` | Opaque identity of one invocation of that suite, shared by all its Runs |
| `eval.leg` | Comparison leg such as `a`/`b` or `baseline`/`candidate` |
| `eval.fixture` | Stable fixture identity |
| `eval.case` | Stable case identity within the fixture |
| `eval.sample` | One-based decimal sample number within pass@N |

One eval sample is one fresh WorkflowRun and therefore receives natural Run
artifact, memory, workspace and Worker-session isolation. `eval.sample` is not
a Stage attempt number: Scheduler retry/escalation remains visible through the
ordinary StageExecution history inside that Run.

`eval.id` is not a WorkflowRun ID. It correlates a set of Run IDs created by
one eval invocation. The eval runner is responsible for using identical exact
input ArtifactRefs where an A/B comparison requires shared inputs and for
recording the immutable Workflow/config refs that define each leg. Label values
are indexes for discovery, not a replacement for those snapshots or the
versioned eval result envelope.

## Planner, Runtime and observability propagation

Server makes the immutable map available to Planner and Worker telemetry so an
operator can correlate model traces with an eval invocation and leg:

1. Planner spans obtain the labels from the authoritative WorkflowRun row.
2. Scheduler copies the same bounded map into each `AllocationSpec` as
   `runMetadataLabels`.
3. Runtime validates the wire bounds and exposes labels only to its telemetry
   adapter as per-execution span attributes. They are not process-wide resource
   attributes because one Runtime process serves different Runs over time.
4. Runtime never injects metadata labels into prompts, ADK State, tool
   arguments, WorkspaceFS, Skills or model-visible results.

Telemetry uses a namespaced attribute projection such as
`contractor.run.label.eval.id`. Exporter-specific adapters may additionally
map the conventional eval labels to native trace tags when that mapping loses
no values. The exact original map remains authoritative in Server.

Metadata labels are intentionally not added as metric dimensions by default.
Keys such as `eval.id`, `eval.case` and `run_id` are high-cardinality and would
create unsafe time-series growth. Existing Run/Stage correlation IDs and trace
attributes remain sufficient for detailed drill-down.

Allocation replay identity includes the complete metadata-label map. A stale
or altered replay is rejected before Worker construction. The labels do not
participate in Runtime capability matching or RuntimeConfig merge provenance,
and a Runtime process does not interpret `purpose` or `eval.*`.

## UI surface

The Run-create UI presents Runtime labels separately as infrastructure
configuration. Metadata labels use a bounded key/value editor; the UI does not
offer them as RuntimeConfig checkboxes.

Run list and detail show metadata labels independently from pinned Runtime
configuration provenance. The first filter surface supports exact conjunctions
and may provide shortcuts for the conventional `purpose=eval`, `eval.name`,
`eval.id` and `eval.leg` keys without changing the generic API.

The UI must explain that labels are immutable and telemetry-visible. It must
not infer permissions, execution success or A/B comparability from a label.

## Failure behavior

| Failure | Result |
|---|---|
| Invalid key/value shape or bounds | `400`; no Run, input fork or label row is created |
| Reserved caller key | `400`; no mutation |
| Changed labels under an existing idempotency key | Existing idempotency conflict; original Run is unchanged |
| Invalid list selector | `400`; no partial interpretation |
| Valid selector with no match | Empty owner-scoped page |
| Allocation label-map mismatch or invalid bounds | Private protocol invariant failure before Worker construction |
| Telemetry exporter failure | Bounded best-effort telemetry failure; semantic Run outcome is unchanged |

## Invariants

1. `runtimeLabels` configure infrastructure; `labels` describe one Run.
2. Metadata labels are immutable, owner-scoped, bounded and atomically durable
   with Run creation.
3. Metadata labels are neither authorization claims nor Scheduler inputs.
4. An eval sample is an ordinary WorkflowRun; eval adds no hidden execution
   path to Scheduler, Control Plane or Runtime.
5. Planner and Runtime may attach labels to traces but never expose them to the
   model or use them as unbounded metric dimensions.
6. Filtering is exact, conjunctive, owner-isolated and compatible with stable
   keyset pagination.
7. Labels aid discovery but never replace immutable Workflow, config, input or
   result provenance.
