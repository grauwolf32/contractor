# 14 — Worker results, deterministic observations and live state

Status: **Working agreement**

Depends on: [00](00-workflow-and-planner.md),
[01](01-agent-template.md), [02](02-runtime-and-a2a.md),
[04](04-execution-lifecycle-and-metrics.md) and
[10](10-runtime-filesystems-and-edit-tools.md)

## Goal

This document owns four closely related boundaries:

- the terminal semantic text produced by the main Worker model and its strict
  structured projection;
- the trusted `WorkerResult` assembled by Runtime Agent for Planner;
- deterministic invocation observations accumulated in allocation-local State;
- the read-only path by which explicit Server-side Planner tools inspect that
  live State without exposing arbitrary keys to the model.

It supersedes the plain-text Worker completion projection previously described
by V13. It does **not** move `StageResult` ownership to Worker: a model-backed
Planner still decides Stage completion through `finish`, and Workflow Scheduler
still validates and durably accepts that candidate.

## Four result layers

The word “result” refers to four different facts. They are separate contracts:

1. terminal Worker text is authored by the main tool-using model. It says what
   one exact subtask produced, and nothing about Contractor lifecycle;
2. `WorkerModelResult` is emitted by a separate tool-free ADK result finalizer
   which must copy that text and the Runtime-supplied subtask ID exactly;
3. `WorkerResult` is assembled by Runtime. It combines the validated semantic
   result with Runtime-owned observations, exact trusted artifact refs and a
   Runtime-owned summarization flag.
4. `StageResult` is authored by Planner and accepted by Scheduler. It decides
   whether the complete Stage succeeded or failed.

The model-facing schema is intentionally small:

```python
class WorkerModelResult(BaseModel):
    subtask_id: str
    result: str
```

Both fields are mandatory and non-empty; unknown fields are rejected.
`subtask_id` is 1..128 bytes and matches
`[A-Za-z0-9][A-Za-z0-9._:-]{0,127}`. `result` is bounded to 64 KiB of UTF-8.
It is the factual result intended for the calling Planner, not a transport
envelope, retry decision, Stage outcome or telemetry container.

The main Worker does not serialize this object and does not receive it as an
output schema. `WorkerModelResult` is internal to the Python Runtime/ADK
adapter. It never crosses A2A and has no Server-domain counterpart. Only the
Runtime-authored `WorkerCompletion` below is a shared Go/Python wire contract.

Every private Worker request contains a mandatory `subtask_id`. Streamline and
Router copy the ID of the exact claimed `PlannerSubtask`; the baseline
Passthrough Planner uses the fixed ID `0`. The ID is opaque to Worker and is
distinct from A2A Task ID, invocation ID, allocation ID and StageExecution ID.
Runtime includes it in semantic task input. After ordinary Worker completion,
Runtime supplies the exact ID and bounded terminal text to the result finalizer
described below.

The finalizer must echo that exact ID. Runtime compares the parsed value to the
request and rejects a mismatch as `worker_result_subtask_mismatch`; the value
placed in the trusted result is always copied from the request, never trusted
from model output. Runtime also requires byte-for-byte equality between
finalized `result` and the main Worker's terminal text. A rewrite is
`worker_result_finalizer_mismatch`. These checks detect stale, malformed or
semantic-changing serialization without letting either model redirect or
rewrite a result.

Runtime then constructs:

```python
class WorkerResult(BaseModel):
    subtask_id: str
    result: str
    observations: WorkerObservations
    artifacts: dict[str, ArtifactRef]  # mandatory non-null; may be empty
    summarized: bool


class WorkerFailure(BaseModel):
    code: str
    message: str
    retryable: bool


class WorkerCompletion(BaseModel):
    api_version: Literal["contractor/v1alpha1"]
    result: WorkerResult | None = None
    failure: WorkerFailure | None = None
    invocation_id: str
    state_revision: int
```

`WorkerCompletion` is a versioned strict private A2A union: exactly one of
`result` and `failure` is present. `invocation_id` and `state_revision` are
control metadata for Server correlation and never enter a model-facing Planner
tool result.

The A2A request DataPart uses
`application/vnd.contractor.stage-content+json`; the terminal response DataPart
uses `application/vnd.contractor.worker-completion+json`. The Agent Card and
each request advertise the latter as the accepted output mode. A missing or
different response media type fails closed before payload decoding.

`WorkerResult` has no outcome or error field. Receiving it means that the
Worker model produced one valid semantic completion. Runtime/tool/provider,
budget and contract failures use `WorkerFailure`; a Worker-facing `fail` tool
or model-authored failure status is deliberately not defined in this increment.
`WorkerFailure.code` matches `[a-z][a-z0-9_]{0,63}`; its already-sanitized
non-empty `message` is at most 4,096 UTF-8 bytes. `invocation_id` is an opaque
Runtime-generated 1..128-byte identifier and `state_revision` is positive.

For Streamline and Router, `execute_current_subtask` returns the bounded
model-facing portion of `WorkerResult`. A `WorkerFailure` becomes its existing
closed tool error and marks that dispatch failed, leaving Planner to reason or
finish within its remaining limits. Passthrough deterministically maps a
`WorkerResult` to a succeeded `StageResult` candidate and a `WorkerFailure` to
a failed candidate with the same safe error. Neither mapping lets Worker
select retry or escalation.

## Structured output and model boundary

The selected `adk@1` implementation never combines tools and an output schema
in the main Worker Agent. This rule is unconditional: Runtime does not branch
on a model or adapter capability flag. A generic LLM Gateway may route the same
alias to backends with different grammar behavior, and an adapter's advertised
capability is not proof that the selected downstream model accepts both
features in one request.

Every ordinary completion therefore has two phases:

```text
main ADK Worker: AgentTemplate behavior + task + selected tools, no output_schema
  -> bounded terminal semantic text
tool-free ADK result finalizer: exact subtask ID + exact terminal text
  -> WorkerModelResult(subtask_id, result)
Runtime: exact-copy validation + observations + trusted artifacts
  -> WorkerResult(summarized=false)
```

The finalizer is an ephemeral ADK `LlmAgent`, `Runner` and in-memory Session.
It uses the same allocation model client, pinned ModelPolicy and LLM Gateway as
the main Worker, but receives no tools, Agent Skills, workspace handles,
Artifact client, task objective/instructions, parameters, input refs, prior
conversation or arbitrary State. Its deterministic input document contains
only `subtaskId` and `resultText`, is at most 256 KiB encoded, and is destroyed
with the invocation. Its instruction permits exact serialization only. It may
not summarize, correct, reinterpret or reformat either value.

The finalizer makes exactly one provider call. It has no repair or retry loop.
That call and its provider-reported usage count against the same normal Worker
`maxModelCalls` and `maxTotalTokens` as main turns, and appear in the same
Worker metrics/live State. Model spans mark `model.phase=result_finalizer`.
Workflow authors must include this mandatory call in the normal Worker budget;
Runtime does not grant hidden finalization capacity. Missing, blank, over-64
KiB or known-secret-bearing terminal text is rejected before the finalizer.

The optional terminal summarizer in [15](15-worker-summarization.md) is the one
exception to this ordinary two-phase path: it already is an independent
tool-free structured terminal Agent. Its valid `WorkerModelResult` is checked
and projected directly with `summarized=true`, without a redundant result
finalizer call or normal-budget charge.

Runtime does not ask either model to serialize `WorkerResult`,
`WorkerCompletion` or `StageResult`.

The model sees only:

- resolved AgentTemplate behavioral instructions;
- Planner-supplied objective and task instructions;
- the exact opaque `subtask_id`;
- immutable string parameters and named exact input ArtifactRefs;
- its explicitly selected tools and skills;
- no Contractor result schema or lifecycle envelope.

The result finalizer sees only:

- the exact opaque `subtask_id` supplied by Runtime;
- the exact bounded terminal text emitted by the main Worker;
- ADK's schema guidance for the two-field `WorkerModelResult`.

It does not see result bindings, artifact grants, Runtime observations,
`summarized`, state revision, physical placement, RuntimeSettings, lifecycle,
retryability or Scheduler policy. An unsupported or invalid provider
structured response fails closed with a bounded stable Worker error. Runtime
accepts main final text only after the mandatory finalizer has returned the
strict schema and exact-copy validation has succeeded. It never makes a second
finalizer/repair call.

The existing trusted artifact projection remains authoritative. Runtime maps
only exact refs observed through completed allocation-bound tool calls in the
same invocation to immutable Server-declared result bindings. It ignores any
artifact-like text or unknown field produced by the model. Overlay auto-export
adds its Runtime-owned exact refs before `WorkerCompletion` is published, as
defined by [10](10-runtime-filesystems-and-edit-tools.md).

Initial stable result failures include:

- `worker_result_missing`;
- `worker_result_invalid`;
- `worker_result_subtask_mismatch`;
- `worker_result_finalizer_mismatch`;
- `worker_result_finalizer_failed`;
- `worker_result_too_large`;
- `unsafe_worker_result`;
- the existing `worker_budget_exhausted` and domain/runtime failures.

Malformed/missing model output, a subtask mismatch, an internal finalizer
failure and a finalizer exact-copy mismatch are retryable Worker failures
because a fresh StageExecution may succeed. Provider transport failures retain
the shared `worker_gateway_unavailable` code; an internal finalizer boundary
failure uses `worker_result_finalizer_failed`. Deterministic size, secret-retention and
invalid binding violations are non-retryable. Workflow policy, not Runtime,
decides whether that flag causes another attempt.

## One instrumentation plugin, separate reducers

An ADK Worker uses one Contractor `WorkerInstrumentationPlugin`, registered on
the ADK `App` supplied to `Runner`. It observes run, model and tool callback
boundaries and feeds independent reducers:

```text
WorkerInstrumentationPlugin
  -> MetricsReducer                allocation-wide report counters/details
  -> InvocationMetricsReducer      bounded facts for the current invocation
  -> WorkspaceObservationReducer   model-visible filesystem interaction facts
  -> future typed reducers         only through reviewed safe extractors
```

This is one plugin for lifecycle correlation, not one untyped data bucket.
Metrics and observations have different retention and projection rules.
Refactoring existing per-tool metrics into the plugin must preserve the current
`ExecutionReport` exactly and must not count a call twice.

The result finalizer has a separate ephemeral ADK Runner but reports its one
model call through typed hooks on this same invocation plugin. It therefore
updates the same normal budget, metrics and Contractor-owned State reducers; it
does not create a second uncorrelated instrumentation State. Its prompt and
output are never retained in that State.

The plugin never generically copies tool arguments or results into State. Each
selected tool may register a narrow `ObservationExtractor` which converts a
completed call into a typed, bounded, content-free fact. Unknown tools have
metrics counts but contribute no domain observation. An extractor failure is a
safe instrumentation error: it sets observation completeness false and cannot
change or fabricate the tool result.

Only a completed callback changes semantic observations. A failed/cancelled
call increments safe metrics but does not claim that a file was read or
changed. Parallel callback order is normalized by the plugin's monotonic call
ordinal and committed under one state lock.

## Workspace observations

Workspace observation coverage measures what the Worker actually exposed to
the model during one A2A invocation. It is not code-analysis scan coverage and
does not claim that every file internally parsed by Tree-sitter or Trailmark
was read by the model.

At invocation start, Runtime creates one `WorkspaceObservationState` over the
effective managed-text checkpoint:

```python
class WorkspaceInteraction(BaseModel):
    path: str
    first_ordinal: int
    last_ordinal: int
    discovery_calls: int
    read_calls: int
    match_calls: int
    mutation_calls: int


class WorkspaceObservationState(BaseModel):
    workspace_digest: str
    scope_paths: list[str]             # lexical, bounded checkpoint universe
    scope_complete: bool
    interactions: list[WorkspaceInteraction]
    detail_complete: bool
```

The initial semantics are:

- a successful `read_file` records its normalized relative path as read;
- successful `grep` records only paths whose returned matches were visible;
- successful `glob` and `ls` discover returned normalized managed-text file
  paths but do not mark their contents read; directory entries are not file
  coverage;
- successful Edit/annotation mutations record affected managed-text file paths
  as modified;
- `workspace-changes` records as modified only file paths present in its
  returned bounded page;
- code-analysis coverage remains its own typed response/observation section and
  does not mark every internally scanned file as read;
- retries increment operation counts while unique coverage counts each path
  once.

Interactions are ordered by first completed observation ordinal with relative
path as a deterministic tie-breaker. Model-visible projections never contain
host paths, file contents, grep patterns, edit text, diffs, parser payloads or
Artifact revisions.

The checkpoint scope admits at most 10,000 paths and 2 MiB of encoded path
detail. Interaction detail independently admits at most 10,000 paths. A
Workspace larger than either bound remains usable: the corresponding
`*_complete` flag becomes false, counts are reported as observed lower bounds,
and no projection may claim exact unread coverage. The complete encoded
`AgentStateSnapshot` containing Contractor State is capped at 4 MiB. State
admission reserves the fixed envelope overhead rather than allowing a valid
State value to become an oversized HTTP response. The per-section bounds are
maxima, not reservations: optional path/detail admission stops earlier and
marks the corresponding completeness/truncation field when the shared ceiling
would otherwise be crossed. Mandatory schema fields and saturating aggregate
counters have reserved capacity, including while current and last-completed
invocations coexist.

The immediate result uses one fixed initial `lean@1` projection:

```python
class ToolObservationCount(BaseModel):
    calls: int
    failures: int


class WorkspaceObservationSummary(BaseModel):
    scoped_files: int
    scope_complete: bool
    discovered_files: int
    read_files: int
    matched_files: int
    modified_files: int
    detail_complete: bool
    unread_files: int | None = None
    files_read: list[str]              # at most 25, first-observation order
    files_read_truncated: bool


class WorkerObservations(BaseModel):
    profile: Literal["lean@1"]
    tools: dict[str, ToolObservationCount]
    workspace: WorkspaceObservationSummary | None = None
    truncated: bool
```

`discovered_files`, `read_files`, `matched_files` and `modified_files` are
unique managed-text path counts under the operation semantics above.
`unread_files` is present only when both scope and interaction detail are
complete and is the cardinality of `scope_paths - observed_read_paths`; it is
not arithmetic subtraction of the two exposed counts because a Worker may read
a file created after the checkpoint. `files_read_truncated` is true exactly
when `files_read` omits part of the `read_files` set. Tool names are sorted;
zero-call tools are omitted. This compact projection is always Runtime-authored.
Its precise future composition remains an evaluation subject; a later named
profile may change what is surfaced, but cannot reinterpret results already
stamped `lean@1`.

## Contractor-owned Worker State

The allocation's ADK Session contains one reserved `contractor` subtree. It is
the only State surface exported by Runtime:

```python
class ContractorWorkerState(BaseModel):
    schema_version: Literal[2]
    state_revision: int
    metrics: AllocationMetricsState
    current_invocation: InvocationState | None
    last_completed_invocation: InvocationState | None
```

`current_invocation` is created immediately before `Runner.run_async` and
contains Runtime-generated invocation ID, authoritative subtask ID, phase,
invocation metrics and typed observation sections. On every success, failure
or cancellation it is atomically closed and becomes
`last_completed_invocation`. Beginning the next invocation does not mutate the
previous completed snapshot; completing it replaces that one-slot history.
Allocation-wide metrics continue across sequential invocations.

State schema version 2 adds the nullable most-recent provider prompt-token
count to invocation metrics and a required closed summarizer section. That
section is `disabled` when the pinned AgentTemplate has no summarizer, otherwise
it moves monotonically through `not_requested -> requested -> succeeded|failed`.
It retains only the request-causing state revision, one-call numeric usage and
a stable failure code; it never retains the summary prompt or result. The
endpoint ETag format remains its independently versioned `v1` cache contract.

Every committed mutation increments `state_revision`. Snapshot reads take an
immutable deep copy under the same lock; JSON encoding happens after releasing
the lock so a large read cannot block tool callbacks. Counters saturate and
detail admission stops at the declared bounds rather than evicting facts and
changing earlier ordering.

“Full state” always means this complete Contractor-owned, versioned and bounded
subtree. It explicitly excludes:

- arbitrary/framework-private ADK Session keys and conversation events;
- prompts, model output history, hidden reasoning and complete tool payloads;
- activated-skill framework State;
- Runtime Agent lifecycle/registry internals;
- RuntimeSettings, endpoints, headers, tokens and credentials.

Secrets are held in typed allocation context and injected into already
constructed clients/adapters. They are forbidden in ADK State by invariant,
not merely redacted on export. Runtime may initialize safe closed non-secret
state before Runner starts; there is no remote generic state mutation API.
Future ADK instruction templating may consume an explicitly specified safe
initial-state schema, but arbitrary placeholders and secret-backed values are
outside this increment.

State is volatile. It exists only in the allocated Runtime process and is
destroyed with Worker/release. Durable metrics continue to use the bounded
final report; durable domain state continues to use ArtifactStore and
MemoryTools.

## Private read endpoint

The Python Runtime Agent exposes:

```http
GET /private/v1/allocations/{allocation_id}/agent-state
```

Its exact successful JSON body is:

```python
class AgentStateSnapshot(BaseModel):
    api_version: Literal["contractor/v1alpha1"]
    state: ContractorWorkerState
```

Unknown fields are rejected. The response deliberately does not repeat
`allocation_id`: request routing and the live allocation lookup establish that
identity, while `state.state_revision` establishes snapshot identity.

The thin ASGI handler lives beside allocation lifecycle routes and delegates
to `AllocationService`, which verifies that the exact allocation owns the live
Worker. The Worker implementation returns a framework-neutral
`AgentStateSnapshot`; the HTTP layer never reaches directly into ADK Session.

The endpoint:

- accepts only a Control Plane mTLS peer under [02];
- has no request body and no model/user-provided state key;
- returns the whole exportable Contractor State and the strong ETag
  `"contractor-agent-state-v1-<stateRevision>"` derived only from its positive
  revision;
- honors a matching `If-None-Match` with `304` and no body for bounded
  Server-side caching;
- sends `Cache-Control: private, no-cache` so revision revalidation remains
  usable without accepting an unvalidated cached body, and a body no larger
  than 4 MiB including the response envelope;
- is available for the active allocation while its Worker State exists;
- returns closed safe `allocation_not_found` or `agent_state_unavailable`
  errors for stale, preparing, draining-with-destroyed-State or released slots;
- never persists, mutates, resumes or reconstructs State.

There is deliberately no `PUT`, `PATCH` or generic command counterpart.
Configuration and credentials arrive once through the pinned allocation
prepare contract. Per-invocation semantic parameters arrive through
`StageContentRequest`. Cancellation/finalization already have explicit typed
lifecycle commands, and shared mutable Planner/Worker information belongs in
MemoryTools/artifacts.

Go Control Plane adds a bounded Runtime client operation behind a narrow
framework-neutral `WorkerStateReader`; Planner packages do not construct HTTP
requests or receive physical Runtime endpoint data.

## Planner projection tools

Raw State is never a Planner model tool. Server-side tools load the complete
snapshot, validate its allocation/invocation/subtask correlation, run one
fixed projection and return only that projection. They accept no state-key,
JSONPath, expression, field list or arbitrary aggregation program.

The initial Streamline tools are:

| Tool | Result |
|---|---|
| `get_workspace_coverage()` | Counts/completeness for the sole Worker's last completed invocation. |
| `list_read_files(cursor="", limit=100)` | Bounded first-observation-ordered read paths. |
| `list_unread_files(cursor="", limit=100)` | Lexically ordered checkpoint scope minus read paths; unavailable when coverage is incomplete. |
| `get_worker_tool_usage()` | Sorted per-tool calls/failures and bounded invocation model/token counters. |

Router exposes the same names with mandatory `worker_name`, constrained in each
schema to logical bindings for which that projection is valid. Streamline has
no `worker_name` parameter. Passthrough has no model-facing Planner tools.
Workspace tools are present only when the immutable Stage/Agent binding has a
workspace and at least one selected workspace-observing tool. The Planner
toolset includes `get_worker_tool_usage` for every modeled Worker binding; this
does not add that tool to the Worker toolset. A Worker never invokes State
projections and remains unaware of Planner, AgentTemplate and Runtime endpoint
details.

The tools always select the newest `WorkerCompletion` already returned for
that logical Worker; the model cannot select allocation, invocation, state
revision or a stale subtask. Before any matching completion they return
`worker_state_unavailable`. The hidden selector contains allocation ID,
invocation ID, subtask ID and state revision. A fetched snapshot must contain a
matching `last_completed_invocation`, otherwise the tool fails closed instead
of showing another invocation.

Collection cursors are opaque and bind projection name, logical Worker and
state revision. A changed revision invalidates the cursor with
`worker_state_changed`. `limit` is 1..100. The Server may cache a validated
snapshot by allocation ID and ETag; the cache is StageExecution-local and is
erased during Planner cleanup.

State-read failure does not rewrite an already returned `WorkerResult`, change
subtask status or fail the Stage automatically. It is a bounded correctable
Planner tool error. The compact observations already returned with
`execute_current_subtask` remain usable.

Only the live Planner conversation sees detailed projection results. Durable
Planner facts may record tool name, logical Worker, counts, outcome and stable
error code, but never paths, raw State, model/tool content, allocation ID or
state revision. No public Operations/UI endpoint is added by this contract.

## Lifecycle and races

- One allocation permits one active Worker invocation; state transitions and
  snapshot reads serialize without holding a lock across network/model/tool I/O.
- A result is published only after the matching invocation State has been
  atomically closed, so `WorkerCompletion.state_revision` names a snapshot that
  already contains its observations.
- A later invocation may replace `last_completed_invocation`; hidden
  correlation prevents an old Planner result from reading the new invocation.
- Finalize/abort stop new reads once Worker State is destroyed. Final reports
  are built from the same reducer snapshot before destruction and remain the
  only durable metrics path.
- Release erases State, ETag/cache material, observations, prompts and clients
  before the slot may become idle.
- Observation/state export failure can reduce telemetry completeness, but it
  cannot invent a semantic Worker success or artifact ref.

## Deliberately deferred

- a Worker-facing semantic `fail` tool/status;
- generic Planner `get_state`, arbitrary state keys or model-defined filters;
- any endpoint that mutates ADK State during execution;
- generic or secret-backed ADK instruction templating; a later closed schema
  may seed approved non-secret allocation/task values without adding a remote
  State-write endpoint;
- public/UI access, historical snapshots or persistence of detailed paths;
- dynamic configuration/credential replacement inside an allocation;
- user-defined observation reducers or YAML A/B profiles;
- exact byte/range coverage and claims that static-analysis scanning equals
  model-visible file reading;
- ADK event compaction and terminal summarization, which are specified
  separately in [15](15-worker-summarization.md).

## Invariants

1. Worker model authors only exact subtask ID and semantic result text.
2. Runtime authors observations, summarized state, artifact refs and all
   technical failures; Planner authors Stage outcome.
3. The outbound subtask ID is copied from the trusted request and must match the
   model's echo.
4. Observations describe completed model-visible operations, never inferred
   filesystem access by an implementation detail.
5. Metrics and observations share callback correlation but have independent
   typed reducers and retention.
6. The private endpoint exports only bounded Contractor-owned State and cannot
   mutate it.
7. Credentials and secret-bearing configuration never enter ADK State.
8. Planner models can call only explicit typed projections and never select a
   State key, physical Worker, allocation, invocation or revision.
9. Live State is advisory context; artifacts remain durable data and final
   reports remain durable telemetry.
10. A stale snapshot, truncated coverage or unavailable Runtime is explicit
    and can never be represented as complete coverage.
