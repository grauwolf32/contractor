# Worker

Requirement prefix: `WRK` · [Specification index](../README.md)

## Purpose and scope

This document owns four closely related boundaries of a Worker invocation:

- the terminal semantic text produced by the main Worker model and its strict
  structured projection;
- the trusted `WorkerResult` assembled by Runtime Agent for Planner;
- deterministic invocation observations accumulated in allocation-local State;
- the read-only path by which explicit Server-side Planner tools inspect that
  live State without exposing arbitrary keys to the model.

Around those boundaries it also owns the Worker context built from an
AgentTemplate, the Worker's LLM Gateway client and its failure classification,
Worker budget enforcement, the one Runtime-owned completion boundary and its
ordinary result finalizer, ADK session modes and allocation-local State, the
schema and lifecycle of the private agent-state endpoint, and the optional
terminal summarizer owned by Runtime Agent.

It leaves to other documents:

- the Runtime Agent process and its one exclusive slot, the Worker runtime's
  deployment dependencies and credential boundary, A2A transport, allocation
  and lease checks, the Agent Card, A2A Task handling and the HTTP handler
  beside the allocation lifecycle routes: [Runtime Agent](runtime-agent.md);
- the private control protocol, including the allocation-correlated
  agent-state transport: [Control Plane](control-plane.md);
- AgentTemplate, Worker runtime selection, ModelPolicy field requirements and
  the `summarizer` configuration block:
  [AgentTemplates](../configuration/agent-templates.md);
- the Stage Worker session policy field and its immutability:
  [Workflows](../configuration/workflows.md);
- Runtime labels and LLM Gateway routes:
  [Runtime configuration](../configuration/runtime-config.md);
- Planner task rendering, the Passthrough pipeline, `finish` and the Planner
  projection tools over live State: [Planners](planners.md);
- StageExecution, retry and escalation: [Scheduler](scheduler.md);
- Gateway `failureSignatures` and the shared Gateway failure-classification
  fixture: [Failure model](../platform/failure-model.md);
- the `audit-check-results@1` completion strategy:
  [Audit Worker completion](../audits/worker-completion.md);
- overlay workspace auto-export: [Workspace](../tools/workspace.md);
- Agent Skill packages and activated-skill State: [Skills](../tools/skills.md);
- the ArtifactStore, result-binding carriage in private A2A input and the ADK
  artifact service adapter: [Artifact store](../artifacts/artifact-store.md).

## Concepts

Terms are defined once in the [glossary](../glossary.md).

| Term | Meaning here |
| --- | --- |
| Worker | The temporary, allocation-scoped role and runtime instance created in process from one AgentTemplate for one allocation; not another service, daemon, subprocess, container or operating-system process ([Runtime Agent](runtime-agent.md), [AgentTemplates](../configuration/agent-templates.md)). |
| Main Worker model | The tool-using ADK Worker Agent that runs AgentTemplate behavior over one task. |
| Terminal Worker text | The bounded semantic text the main Worker model ends with: what one exact subtask produced, and nothing about Contractor lifecycle. |
| Subtask ID | The opaque, mandatory `subtask_id` of a private Worker request, supplied by Planner. |
| Result finalizer | The ephemeral, tool-free ADK agent that copies the subtask ID and the terminal Worker text exactly into a `WorkerModelResult`. |
| `WorkerModelResult` | The internal two-field model-produced result `(subtask_id, result)`. |
| `WorkerResult` | The trusted result Runtime assembles from the validated semantic result, Runtime-owned observations, exact trusted artifact refs and the Runtime-owned `summarized` flag. |
| `WorkerFailure` | A Runtime-authored technical failure `(code, message, retryable)`. |
| `WorkerCompletion` | The versioned private A2A union carrying exactly one `WorkerResult` or `WorkerFailure`. |
| `StageResult` | The executor-authored, Scheduler-accepted decision whether a whole Stage succeeded ([Planners](planners.md)). |
| Completion boundary | The one Runtime-owned point at which an invocation's semantic completion resolves to `continue`, `complete` or `fail`. |
| Session mode | The Stage's `isolated` or `shared` Worker session policy. |
| Contractor-owned Worker State | The reserved, versioned, bounded `contractor` subtree of ADK State; "full state" always means this subtree. |
| `WorkerStateStore` | The canonical allocation-local store of the Contractor-owned Worker State. |
| `WorkerInstrumentationPlugin` | The one ADK plugin that observes run, model and tool callbacks and feeds typed reducers. |
| `ObservationExtractor` | A narrow per-tool converter from one completed tool call to a typed, bounded, content-free fact. |
| Workspace observation coverage | What the Worker actually exposed to the model during one A2A invocation. |
| `lean@1` | The fixed, Runtime-authored compact observation projection carried by `WorkerResult`. |
| Terminal summarization | Stopping the normal Worker loop, making one separate tool-free model call and returning a final `WorkerResult` with `summarized: true`. |
| Context compaction | Replacing the oldest complete history groups of an isolated invocation with one Runtime-committed summary checkpoint so that the same Worker can continue using tools (`compact@1`, WRK-R184 to WRK-R195). |
| Context boundary | The prompt-token boundary at which an enabled summarizer or a `compact@1` checkpoint is requested. |
| Cumulative budget | The optional soft boundary on summed provider-reported tokens of the normal loop in one invocation. |

## Model

### Result layers

The word "result" refers to four different facts. They are separate
contracts:

1. **Terminal Worker text** is authored by the main tool-using model. It says
   what one exact subtask produced, and nothing about Contractor lifecycle.
2. **`WorkerModelResult`** is emitted by a separate tool-free ADK result
   finalizer, which must copy that text and the Runtime-supplied subtask ID
   exactly.
3. **`WorkerResult`** is assembled by Runtime. It combines the validated
   semantic result with Runtime-owned observations, exact trusted artifact refs
   and a Runtime-owned summarization flag.
4. **`StageResult`** is authored by Planner and accepted by Scheduler. It
   decides whether the complete Stage succeeded or failed.

The model-produced schema is intentionally small. The result finalizer and the
terminal summarizer both produce it:

```python
class WorkerModelResult(BaseModel):
    subtask_id: str
    result: str
```

The ordinary completion path has two phases:

```text
main ADK Worker: AgentTemplate behavior + task + selected tools, no output_schema
  -> bounded terminal semantic text
tool-free ADK result finalizer: exact subtask ID + exact terminal text
  -> WorkerModelResult(subtask_id, result)
Runtime: exact-copy validation + observations + trusted artifacts
  -> WorkerResult(summarized=false)
```

The flow below combines that path with the safe-boundary check of the optional
terminal summarizer and the failure exits. Every hard-budget exit sits before
the next side effect (WRK-R23, WRK-R25):

```mermaid
flowchart TD
    resp["Normal model response completes"] --> tools["Every tool call selected by that response completes or fails"]
    tools --> text{"Ordinary terminal semantic text?"}
    text -- yes --> fin["Tool-free result finalizer: exact subtask ID + exact terminal text"]
    fin --> val["Runtime: exact-copy validation, observations, trusted artifacts"]
    val --> res1["WorkerResult, summarized=false"]
    text -- no --> soft{"Summarizer enabled and context boundary or cumulative budget reached?"}
    soft -- no --> next["Next normal model call"]
    next --> resp
    soft -- yes --> sum["Terminal summarizer: one tool-free call"]
    sum --> val2["Runtime: same checks and projection"]
    val2 --> res2["WorkerResult, summarized=true"]
    resp -.->|"token ceiling crossed"| fail["WorkerFailure"]
    tools -.->|"tool-call ceiling reached"| fail
    next -.->|"model-call or token ceiling reached"| fail
    fin -. failure .-> fail
    val -. failure .-> fail
    sum -. failure .-> fail
    val2 -. failure .-> fail
```

### Observations

One instrumentation plugin feeds independent reducers:

```text
WorkerInstrumentationPlugin
  -> MetricsReducer                allocation-wide report counters/details
  -> InvocationMetricsReducer      bounded facts for the current invocation
  -> WorkspaceObservationReducer   model-visible filesystem interaction facts
  -> future typed reducers         only through reviewed safe extractors
```

Workspace observation State, created per invocation:

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

The `lean@1` observation projection carried by `WorkerResult`:

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

### Contractor-owned Worker State

```python
class ContractorWorkerState(BaseModel):
    schema_version: Literal[2]
    state_revision: int
    metrics: AllocationMetricsState
    current_invocation: InvocationState | None
    last_completed_invocation: InvocationState | None
```

The closed summarizer section of the State moves through these values:

```mermaid
stateDiagram-v2
    [*] --> disabled: pinned AgentTemplate has no summarizer
    [*] --> not_requested: pinned AgentTemplate has a summarizer
    not_requested --> requested
    requested --> succeeded
    requested --> failed
```

## Requirements

### Worker context

**WRK-R1.** AgentTemplate is the sole authored, reusable
Worker-behavior contract. The Runtime Agent allocation layer consumes the
resolved template and builds a minimal Worker context containing the
description, instruction text, card version, effective model policy, optional
resolved summarizer policy, instantiated tools and prepared skills.

**WRK-R2.** `WorkerBuildContext` does not contain an
`AgentTemplate`; neither the ADK implementation nor its model sees a template
ref, digest, manifest or allocation projection.

**WRK-R3.** The Stage objective and instructions remain separate
per-task semantic input ([Workflows](../configuration/workflows.md),
[Planners](planners.md)). The AgentTemplate's explicit Toolset selections own
the domain tools, and its explicit `skills` refs own the bounded native ADK
SkillToolset ([Skills](../tools/skills.md)).

**WRK-R4.** A Runtime label may configure a proxy, a telemetry
exporter or an already selected Toolset adapter, but cannot add a tool,
Toolset, skill, instruction or Worker behavior
([Runtime configuration](../configuration/runtime-config.md)). Runtime Agent
receives typed settings and never interprets label names.

**WRK-R5.** The main Worker model sees only the resolved
AgentTemplate behavioral instructions, the Planner-supplied objective and task
instructions, the exact opaque `subtask_id`, immutable string parameters and
named exact input ArtifactRefs, and its explicitly selected tools and skills.
It sees no Contractor result schema or lifecycle envelope.

**WRK-R166.** Every ADK agent that Runtime Agent constructs (the main
Worker, the result finalizer and the terminal summarizer) receives its
instruction through an instruction provider that returns the exact text, so
ADK never applies its session-State or artifact templating to it. A `{name}`
sequence therefore neither fails the invocation before the first model call
nor takes a value from ADK State, such as the `contractor` subtree
(WRK-R105). Authored instruction text keeps its braces
([AgentTemplates](../configuration/agent-templates.md), TPL-R114 and
TPL-R51).

How the semantic task is rendered for the Worker model, and which transport,
allocation and lifecycle facts the rendered task never contains, is owned by
[Planners](planners.md).

**WRK-R6.** The private `StageContentRequest.deadline` is required and
carries the Planner's effective Worker-call deadline: the Stage deadline for passthrough,
the smaller of the Stage and Planner wall-clock deadlines for streamline and
router, and the smaller of the Stage and scan-job deadlines for scan workers.
Model-free `tool@1` Workers enforce it locally on new receipt creation and tool
execution, independently of Server cancellation; their nonretryable timeout,
completed replay and separate settlement budgets are specified by
[SCN-R15, SCN-R38 and SCN-R51](../tools/scanners.md).

**WRK-R7.** The deadline is Runtime control data and is not
placed in the Worker model prompt.

**WRK-R157.** Each outbound call retains its own request timeout.

**WRK-R158.** Audit completion uses that absolute deadline for the whole
invocation, including reminders, model recovery and publication; it does not
turn `requestTimeoutSeconds` into a whole-invocation cap.

**WRK-R167.** An ordinary ADK invocation applies the same absolute deadline to
its whole logical invocation: model and tool execution, model recovery waits,
the result finalizer and the terminal summarizer. Runtime converts it to a
monotonic deadline once, at invocation start. A request whose deadline has
already passed starts no model or tool call and adds nothing to the session's
conversation. When the deadline passes, Runtime cancels the work in flight and joins
it, starts no further model, tool, finalizer or summarizer call, and returns
the retryable `worker_timeout` failure; it does not depend on the Server's
best-effort A2A cancellation. Externally requested cancellation is never
reported as `worker_timeout`, a sandbox failure keeps its own code, and the
timeout of a nested operation, such as the Gateway attempt series (WRK-R13,
WRK-R157), keeps that operation's failure. A summarizer call cut off by the
deadline, and a summarization request the deadline overtakes before that call
starts, are recorded with the `cancelled` cause (WRK-R107, WRK-R155).

**WRK-R168.** The deadline bounds the semantic work of WRK-R158 and WRK-R167,
not what follows its outcome. Pre-terminal workspace auto-export
([WSP-R143](../tools/workspace.md)) keeps the request timeouts of its writes,
and Worker State finalization and session release run after an expired
deadline under their own bounded, cancellation-shielded cleanup budget
(FAIL-WRK-4). A release that exceeds that budget fences the Worker exactly as
it does without a deadline.

**WRK-R170.** Immediately before admitting a main model, tool,
result-finalizer or terminal-summarizer call, Runtime checks the active
logical invocation, authorized SDK turn where applicable, absolute deadline,
cancellation, allocation admission and sandbox state. A rejected admission
starts no new semantic attempt, budget charge, metric span or provider/tool
dispatch. Continuation grants no new budget or deadline. Already dispatched
remote work may have committed and keeps its existing receipt reconciliation;
terminal cleanup follows WRK-R168.

**WRK-R171.** Contractor tool admission preserves the model's original
values for integer and integer-list arguments, including optional forms,
before the SDK can cast them. The owning callable or its raw-argument guard
therefore sees a float, boolean, string, null or invalid list element as
supplied, and applies that tool's strict validation before effects. This
boundary preserves the declared schema and injected context and does not
change the owning tool's missing-field, unknown-field or domain-error policy.

How the Audit strategy converts and applies the deadline is owned by
[Audit Worker completion](../audits/worker-completion.md).

### LLM Gateway client

**WRK-R8.** The Python Runtime implements `openai-compatible@1`
with a Contractor-owned ADK `BaseLlm` adapter and an allocation-owned HTTPX
Chat Completions client. The Runtime dependency set excludes the OpenAI and
LiteLLM Python SDKs; LiteLLM, when selected, remains the one shared external
Gateway.

**WRK-R9.** The adapter sends the exact resolved model alias,
translates ADK text and function-call history, tools and strict response
schemas to Chat Completions, and projects model text and reasoning, function
calls, finish reason and provider-reported usage back to ADK. Successful JSON
responses are projected directly into ADK types without constructing
general-purpose SDK models.

An unanswered function call does not prove nonexecution: a tool may have made
an external change before cancellation or loss of its response. History repair
without admission or outcome evidence pairs it with a bounded
`tool_call_outcome_unknown` response. Only execution evidence may establish
that a call was never dispatched; an existing completed response is retained
exactly. A repaired message never authorizes replay of an uncertain effect.

**WRK-R10.** Non-text task data remains an Artifact. An unsupported
inline content part fails closed instead of disappearing from the model
request.

**WRK-R11.** The Gateway client uses `max_retries=3`, meaning one
initial HTTP attempt and at most three transport retries (LIM-WRK-1).
Its retry classification includes connection loss, timeouts and retryable HTTP
statuses such as 408, 409, 429 and 5xx (FAIL-WRK-1).

**WRK-R12.** Transport retries honor `retry-after-ms`, numeric or
HTTP-date `Retry-After`, and explicit `x-should-retry` overrides. A finite
server delay over 120 seconds stops transport retries
(LIM-WRK-2); an invalid or nonpositive delay uses bounded exponential
backoff with jitter.

**WRK-R13.** One outer Runtime deadline caps the entire attempt
series, retry waits included, at the resolved `requestTimeoutSeconds` plus 60
seconds (LIM-WRK-3). An enclosing Stage cancellation or deadline can
stop it earlier.

**WRK-R14.** Every attempt retains the same allocation-owned Gateway
route and never falls back around a configured proxy.

**WRK-R15.** The attempt series is one logical ModelPolicy call,
because Runtime receives at most one usable response. If a response was lost
after the Gateway accepted it, repeated Gateway work and billing are possible,
and only the final received response's usage can be added to the Worker token
counters; Gateway-side quotas therefore remain the hard authority for all
physical attempts (FAIL-WRK-1).

**WRK-R16.** A received response that the adapter then rejects
(invalid tool-call JSON, an unsupported part, an empty choice) is still a
failed model call, but its reported usage is added to the Worker, invocation,
budget and summarizer token counters like an accepted response. The boundary
error keeps only those numeric counters, never response content.

**WRK-R17.** The client logs neither request payloads nor provider
responses, and releases provider response buffers before retry waits.

**WRK-R18.** After the transport attempt series ends, Runtime
classifies the final failure from bounded error metadata. Connection failures,
timeouts and HTTP 408, 409, 429 or 5xx are retryable. Other HTTP errors,
including 400, 401, 403, 404, 413 and 422, are non-retryable. Explicit
`insufficient_quota`, `budget_exceeded` and `context_length_exceeded` codes are
non-retryable even when the status would otherwise allow retry. Provider
bodies and headers are not retained in errors (FAIL-WRK-1).

Gateway-specific `failureSignatures`, and the shared fixture that keeps the
Server-side Planner and Runtime Worker classifiers identical, are owned by the
[failure model](../platform/failure-model.md).

**WRK-R19.** The ordinary Worker and the result finalizer preserve
that retryable flag under `worker_gateway_unavailable`, including for permanent
request, authorization and context or quota errors; terminal summarization
preserves it under `worker_summarization_failed`. Wrapping an error never
grants another semantic attempt.

**WRK-R159.** Cancellation of a Gateway call propagates through the
existing lifecycle.

**WRK-R20.** Workflow policy, not Runtime, decides whether a
retryable Worker failure causes another attempt by creating a fresh
StageExecution ([Scheduler](scheduler.md)). The retryable flag is separate from
the client's bounded transport retries.

**WRK-R21.** An adapter failure while constructing a request from
ADK content is a local, deterministic `worker_gateway_invalid_request` failure
with `retryable: false`, distinct from a Gateway transport or HTTP response
failure (FAIL-WRK-2).

### Worker budgets

**WRK-R22.** Each prepared Python ADK Worker independently
enforces the cumulative model-call, tool-call and provider-reported token
ceilings embedded in its exact ModelPolicy. Each main SDK run receives an explicit
finite call ceiling from that policy; SDK defaults or environment overrides cannot
shorten or enlarge the logical budget. Auxiliary one-shot sessions also receive an
explicit finite SDK ceiling, allowing their owned wrapper to classify a refused
repair attempt while still dispatching at most one provider call. The ModelPolicy fields that an
`adk@1` Worker, a Worker with terminal summarization and a Worker summarizer
policy require are owned by [AgentTemplates](../configuration/agent-templates.md).

**WRK-R23.** Worker Runtime checks call and tool capacity before
starting the next operation, and adds the provider-reported
`total_token_count` after each completed model response. Crossing the token
ceiling stops the loop before a response tool call is executed; reaching the
ceiling permits a valid structured semantic result but no later operation.

**WRK-R24.** Missing token usage is recorded explicitly and
never disables the independent model-call and tool-call ceilings.

**WRK-R25.** Exhaustion stops before the next side effect.
Runtime then constructs one safe `WorkerFailure` with code
`worker_budget_exhausted`, the exhausted dimension in bounded metrics, and
`retryable: true`; the model does not serialize that failure. It is neither a
Planner budget termination nor a direct StageExecution write: Passthrough
supplies a normal failed candidate to Scheduler, whose Workflow transition
owns whole-Stage retry.

### Result layers and the wire contract

**WRK-R26.** Worker model, Runtime, Planner and Scheduler own separate
result layers ([Result layers](#result-layers)). The main tool-using Worker
model returns bounded terminal semantic text; Runtime completes it through the
mandatory one-shot result finalizer, attaches deterministic observations and
trusted exact result artifacts, and returns a `WorkerResult` or a separate
technical `WorkerFailure`. Planner alone turns one or more such completions
into a `StageResult` candidate.

**WRK-R27.** The Worker does not own `StageResult`: a
model-backed Planner decides Stage completion through `finish`, and Workflow
Scheduler validates and durably accepts that candidate.

**WRK-R28.** Allocation drain and finalization perform no model
or tool calls.

**WRK-R29.** Every private Worker request contains a mandatory
`subtask_id`. Streamline and Router copy the ID of the exact claimed
`PlannerSubtask`; the baseline Passthrough Planner uses the fixed ID `0`. The
ID is opaque to the Worker and distinct from the A2A Task ID, invocation ID,
allocation ID and StageExecution ID. Runtime includes it in the semantic task
input and, after ordinary Worker completion, supplies the exact ID and the
bounded terminal text to the result finalizer.

**WRK-R30.** Both `WorkerModelResult` fields are mandatory and
non-empty, and unknown fields are rejected. `subtask_id` is 1..128 bytes and
matches `[A-Za-z0-9][A-Za-z0-9._:-]{0,127}` (LIM-WRK-4). `result` is
bounded to 64 KiB of UTF-8 (LIM-WRK-5).

**WRK-R31.** `result` is the factual result intended for the
calling Planner, not a transport envelope, retry decision, Stage outcome or
telemetry container.

**WRK-R32.** The main Worker does not serialize
`WorkerModelResult` and does not receive it as an output schema.
`WorkerModelResult` is internal to the Python Runtime and its ADK adapter: it
never crosses A2A and has no Server-domain counterpart. Only the
Runtime-authored `WorkerCompletion` is a shared Go/Python wire contract.

**WRK-R33.** Runtime does not ask either model, the main Worker
model or the result finalizer, to serialize `WorkerResult`,
`WorkerCompletion` or `StageResult`.

**WRK-R34.** `WorkerCompletion` is a versioned, strict
private A2A union in which exactly one of `result` and `failure` is present.
`invocation_id` and `state_revision` are control metadata for Server
correlation and never enter a model-facing Planner tool result.

**WRK-R35.** `WorkerResult` has no outcome or error field:
receiving it means that the Worker model produced one valid semantic
completion. Runtime, tool, provider, budget and contract failures use
`WorkerFailure`.

**WRK-R36.** `WorkerFailure.code` matches
`[a-z][a-z0-9_]{0,63}` (LIM-WRK-6), and its already-sanitized,
non-empty `message` is at most 4,096 UTF-8 bytes (LIM-WRK-7).
`invocation_id` is an opaque Runtime-generated 1..128-byte identifier
(LIM-WRK-8), and `state_revision` is positive.

**WRK-R37.** The 256 KiB encoded limits for `StageContentRequest`,
`WorkerCompletion`, the Planner-facing `WorkerResult` and the resulting
`StageResult` (LIM-WRK-9) count compact UTF-8 JSON bytes. Optional HTML (`<`,
`>`, `&`) and JavaScript line-separator escaping must not consume extra budget
in one language; required JSON escaping, object and envelope overhead and
exact integer digits still count. A `StageResult` also has a non-null
`artifacts` map of at most 128 entries and a `summary` of at most 64 KiB. In
the Server each message's own validator is the single definition of its
bounds: every Planner, the Workflow Scheduler and RunStore apply the same
`StageResult` check, so a candidate one of them accepts is never rejected by
another for its size. The 64 KiB result-text bound (LIM-WRK-5),
model-output bounds and outer transport limits are separate limits.

**WRK-R165.** The Worker State byte bounds count compact UTF-8 JSON bytes as
WRK-R37 does: the `AgentStateSnapshot` body (LIM-WRK-17), the 4,096-byte
arguments of each retained tool call and the 2 MiB workspace scope and
interaction path lists. The Runtime that produces the State and the Server that
reads it measure with the same rule, pinned by
`api/testdata/v1alpha1/json-byte-size-cases.json`, so State the Runtime exports
at a bound is never rejected by the Server for its size.

**WRK-R38.** The A2A request DataPart uses
`application/vnd.contractor.stage-content+json` and the terminal response
DataPart uses `application/vnd.contractor.worker-completion+json`. The Agent
Card ([Runtime Agent](runtime-agent.md)) and each request advertise the latter
as the accepted output mode. A missing or different response media type fails
closed before payload decoding.

**WRK-R164.** The Planner decodes the terminal `WorkerCompletion` from the
bytes of the A2A response that carried it with the strict private decoder
([CTL-R149](control-plane.md)), never from the A2A SDK's generic value: a
duplicate key anywhere in it is rejected and an integer keeps its exact
digits. An A2A DataPart is a `google.protobuf.Value`, whose numbers are
doubles, and the Runtime renders every integer that way (`7.0`); a number
written as an integral double therefore counts as that integer, while a
fractional number or one outside the member's range is rejected.

The contiguous exact-integer boundary is a fixed format property, not Runtime
or Server configuration. Go and Python each define it once; shared boundary
cases hold the definitions and decimal parsing to the same value.

Because a double carries integers exactly only up to 2^53-1, every
`WorkerCompletion` integer, `state_revision` (1 to 2^53-1) and each lean@1
observation count (tool calls and failures, scoped, discovered, read,
matched, modified and unread files; 0 to 2^53-1), is bounded to that range
(LIM-WRK-27) by the private schema, the Runtime model and the Server decoder.
The Runtime rejects a larger value when it builds the completion, before it
encodes the DataPart, so the A2A executor answers with its bounded
`worker_execution_failed` failure instead (or fails the Task without a
message when that failure cannot be built either); the Planner rejects a
larger value it decodes with `invalid_worker_result`. Every integer above
2^53-1 rounds to a double of at least 2^53, so the Planner detects a rounded
value instead of reading it as another count. The Worker State and the
allocation lifecycle responses are JSON bodies, not A2A DataParts, and keep
their uint64 range.

Decision: bound the integers rather than change the transport. The
alternative, carrying integers exactly through the DataPart, needs a
different part encoding or replacing the SDK transport, which is out of scope
for this boundary ([proposal 0038](../../proposals/0038-server-runtime-transport.md)
weighs a transport change); no producer comes near the bound (workspace counts are at most 10,000
paths, LIM-WRK-15, and tool calls and State revisions grow by one per event).
The shared cases `exact_integer_max` and `exact_integer_over` in
`api/testdata/v1alpha1/worker-result-size-cases.json` and the
`worker-completion-*-above-exact.json` invalid fixtures hold the schema, Go,
Python and the A2A decode path to one verdict. Reopen when a concrete
`WorkerCompletion` member needs values above 2^53-1 or the A2A transport gains
an exact integer encoding.

**WRK-R39.** For Streamline and Router,
`execute_current_subtask` returns the bounded model-facing portion of
`WorkerResult`. A `WorkerFailure` becomes that tool's existing closed tool
error and marks the dispatch failed, leaving Planner to reason or finish
within its remaining limits.

**WRK-R40.** Passthrough deterministically maps a
`WorkerResult` to a succeeded `StageResult` candidate and a `WorkerFailure` to
a failed candidate with the same safe error; Scheduler and Planner retain
their independent Stage result-contract validation. Neither mapping lets the
Worker select retry or escalation.

The complete Passthrough path from `StageSpec` to `StageResult` is owned by
[Planners](planners.md).

### Model boundary and result finalizer

**WRK-R41.** The main Worker model authors only the semantic result
text. Runtime computes observations and matches each declared
`resultArtifacts` binding against exact refs actually observed through
allocation-bound tools. Neither the main Worker model nor the result finalizer
can select the Stage outcome, artifact slot, revision, `summarized`,
retryability or lifecycle transition. Runtime, tool, provider and budget
failures are a separate `WorkerFailure`.

**WRK-R42.** The selected `adk@1` implementation never
combines tools and an output schema in the main Worker Agent. The rule is
unconditional: Runtime does not branch on a model or adapter capability flag.

Rationale: a generic LLM Gateway may route the same alias to backends with
different grammar behavior, and an adapter's advertised capability is not
proof that the selected downstream model accepts both features in one request.

**WRK-R43.** Every ordinary completion therefore has two phases
([Result layers](#result-layers)): the main ADK Worker, with AgentTemplate
behavior, the task and its selected tools but no `output_schema`, produces
bounded terminal semantic text; a tool-free ADK result finalizer turns the
exact subtask ID and exact terminal text into
`WorkerModelResult(subtask_id, result)`; Runtime performs exact-copy
validation, adds observations and trusted artifacts and returns
`WorkerResult(summarized=false)`. This is the default ordinary completion
strategy. The optional terminal summarizer (WRK-R149) and the
`audit-check-results@1` strategy (WRK-R64) are the exceptions.

**WRK-R44.** The result finalizer is an ephemeral ADK `LlmAgent`,
`Runner` and in-memory Session. It uses the same allocation model client,
pinned ModelPolicy and LLM Gateway as the main Worker, but receives no tools,
Agent Skills, workspace handles, Artifact client, task objective or
instructions, parameters, input refs, prior conversation or arbitrary State.

**WRK-R45.** The result finalizer sees only the exact opaque
`subtask_id` supplied by Runtime, the exact bounded terminal text emitted by
the main Worker, and ADK's schema guidance for the two-field
`WorkerModelResult`. It does not see result bindings, artifact grants, Runtime
observations, `summarized`, state revision, physical placement,
RuntimeSettings, lifecycle, retryability or Scheduler policy.

**WRK-R46.** The finalizer's deterministic input document contains
only `subtaskId` and `resultText`, is at most 256 KiB encoded
(LIM-WRK-10), and is destroyed with the invocation
(LIFE-WRK-1). Its instruction permits exact serialization only:
it may not summarize, correct, reinterpret or reformat either value.

**WRK-R47.** Missing, blank, over-64 KiB (LIM-WRK-5) or
known-secret-bearing terminal text is rejected before the finalizer.

**WRK-R48.** The finalizer makes exactly one provider call
(LIM-WRK-11). It has no repair or retry loop, and Runtime never
makes a second finalizer or repair call (FAIL-WRK-3).

**WRK-R49.** The finalizer call and its provider-reported usage
count against the same normal Worker `maxModelCalls` and `maxTotalTokens` as
main turns, and appear in the same Worker metrics and live State. Workflow
authors must include this mandatory call in the normal Worker budget; Runtime
does not grant hidden finalization capacity.

**WRK-R50.** The finalizer must echo the exact subtask ID. Runtime
compares the parsed value to the request and rejects a mismatch as
`worker_result_subtask_mismatch`. The value placed in the trusted result is
always copied from the request, never trusted from model output.

**WRK-R51.** Runtime requires byte-for-byte equality between the
finalized `result` and the main Worker's terminal text; a rewrite is
`worker_result_finalizer_mismatch`.

Rationale: these checks detect stale, malformed or semantic-changing
serialization without letting either model redirect or rewrite a result.

**WRK-R52.** An unsupported or invalid provider structured response
fails closed with a bounded stable Worker error (FAIL-WRK-3). Runtime accepts the main
final text only after the mandatory finalizer has returned the strict schema
and exact-copy validation has succeeded.

**WRK-R53.** Ordinary result-finalizer and terminal-summarizer
outputs pass through the bounded model-JSON decoder before the common Runtime
result assembly. How the Audit strategy's trusted completion fields enter that
assembly is owned by [Audit Worker completion](../audits/worker-completion.md).

### Trusted artifact projection

**WRK-R54.** The trusted artifact projection is authoritative.
Runtime, not the Worker model, maps only exact refs observed through completed
allocation-bound tool calls in the same invocation to the immutable
Server-declared result bindings
([ART-R101](../artifacts/artifact-store.md)). It ignores any artifact-like
text or unknown field produced by the model.

**WRK-R55.** The model may receive refs as ordinary selected-tool
arguments and results, but never serializes the A2A envelope or chooses a
result slot or revision.

**WRK-R56.** Two Runtime-owned sources extend the observed-ref
projection, and neither accepts model-authored refs: overlay auto-export adds
its exact refs after the ordinary observed-ref projection and before
`WorkerCompletion` is published ([Workspace](../tools/workspace.md)), and the
`audit-check-results@1` strategy supplies only Runtime-verified publication
receipts ([Audit Worker completion](../audits/worker-completion.md)).

ArtifactStore bytes, the ADK `BaseArtifactService` adapter and its use of
`save_artifact` are owned by [Artifact store](../artifacts/artifact-store.md)
(ART-R100, ART-R102 to ART-R105).

### Result failures

**WRK-R57.** Malformed or missing model output, a subtask
mismatch, an internal finalizer failure and a finalizer exact-copy mismatch
are retryable Worker failures, because a fresh StageExecution may succeed. An
internal finalizer boundary failure uses `worker_result_finalizer_failed`
(FAIL-WRK-3).

**WRK-R58.** A `MAX_TOKENS` finish from the Worker or its result
finalizer uses retryable `worker_output_limit_exceeded`, not
`worker_result_missing`. Runtime accounts the consumed tokens but does not
accept the truncated text (even valid-looking JSON), call the finalizer on it,
or start an unbudgeted continuation. Workflow policy owns any retry; operators
can adjust the per-response output limit or the model reasoning
configuration.

**WRK-R59.** Deterministic size, secret-retention and invalid
binding violations are non-retryable.

**WRK-R172.** Empty, whitespace-only or thought-only `STOP` output cannot
authorize a successful result. The ordinary completion path returns its
missing or invalid result failure without another main model call or a
finalizer call on missing or blank text. An empty result-finalizer response
fails its existing result validation without repair. The Audit completion
boundary still decides from its trusted submitted results: it may finish or
issue only its existing bounded incomplete-result reminder. Known usage is
accounted even when the SDK reports `MODEL_RETURNED_NO_CONTENT`; output-limit
and provider safety failures keep their own classification and precedence.

**WRK-R60.** The secret-retention scan (`unsafe_worker_result`)
covers every allocation RuntimeSettings credential: the LLM Gateway token,
telemetry header values, HTTP proxy credentials, the Caido bearer token and
the credentials Runtime injects for the HTTP origin target. Endpoints and CA
bundles are not credentials, and results may name them (a same-host deployment
audits services next to its own); the Agent Card check additionally rejects
those.

**WRK-R61.** A credential value of at least 16 UTF-8 bytes
(LIM-WRK-12) is rejected anywhere in the result text; a shorter value,
such as a proxy username, only when it is the whole text, so that an ordinary
word cannot fail a result. The LLM Gateway token is rejected at any length.

The stable result failure codes are listed under
[Limits, failures and errors](#limits-failures-and-errors).

### Completion boundary

**WRK-R62.** Every Worker has one Runtime-owned semantic completion
boundary, reached while the invocation and its session can still continue and
before `WorkerCompletion` or terminal invocation State is published. It is not
allocation drain or finalize. The boundary returns exactly one of:

- `continue`: a bounded Runtime-authored reminder; the same invocation resumes;
- `complete`: a trusted `WorkerResult` ready for normal lifecycle processing;
- `fail`: a bounded `WorkerFailure`, with no fabricated successful result.

Internal SDK turns do not create another logical Worker invocation. Every
Runtime-authored reminder reaches the next actual model request exactly once;
the logical invocation, completion owner, State, observations, budgets and
original deadline remain the same. SDK-local identifiers never replace the
logical invocation identity seen by business tools or trusted results. Model
and tool admission is bound to the currently authorized turn; a stale callback
cannot start unaccounted work.

**WRK-R173.** Every additional Runner turn uses a fresh SDK invocation ID
in the same selected session, so its Runtime-authored reminder is appended
and reaches the next model request exactly once. SDK callbacks retain that
turn identity for admission and history correlation; business tool context,
metrics, observations, completion ownership and `WorkerCompletion` retain the
original logical Worker invocation ID. A turn starts only after the preceding
Runner, tool callbacks and interrupted-history settlement have joined.
At most 1,001 SDK turn identities are retained per logical invocation
(LIM-WRK-24), including the final turn that may be refused by the 1,000-call
ModelPolicy ceiling.

**WRK-R63.** The shared boundary lives in `core/worker_completion.py`;
the Audit implementation lives with `toolsets/audit_results`. Worker
orchestration binds a prepared completion implementation to the identical
trusted contract and all its required tools under one owner; missing, mixed or
forged owners fail preparation. The common runner does not infer Audit
behavior from tool names or import an Audit-specific completion policy. This
organization does not enable additional publicly selectable completion
strategies.

**WRK-R64.** When no completion contract is selected, the ordinary
strategy applies: the terminal text passes through the one-shot tool-free LLM
serializer to the trusted projection, the optional terminal summarizer applies
unchanged, and there is no mandatory Audit tool check. The opt-in
`audit-check-results@1` strategy at the same boundary, including its
completeness gate, continuation within one logical invocation, publication and
accounting, is owned by [Audit Worker completion](../audits/worker-completion.md).

### Session modes

**WRK-R65.** `isolated` gives every admitted sequential Worker
invocation a fresh ADK conversation while retaining only the bounded eligible
allocation-local State (WRK-R73). `shared` deliberately keeps one
conversation for sequential invocations sent to the same logical Worker
allocation. In a Router Stage, `shared` still never joins conversations between
logical Workers: each binding owns a distinct allocation and session service.

The Stage's choice of mode and its immutability are owned by
[Workflows](../configuration/workflows.md).

**WRK-R66.** An ADK-based Worker uses one allocation-local
in-memory SessionService inside the Runtime Agent process. AllocationSpec
carries the Stage's explicit mode to the generic `adk@1` runtime. The mode is
not a WorkerRuntime factory ref and does not create a second
behavior-specific implementation.

**WRK-R67.** A non-ADK Worker runtime provides equivalent
execution and report behavior without exposing an ADK contract.

**WRK-R68.** Runtime Agent receives no database credentials, and
its A2A Task-to-session mapping remains an implementation detail within the
allocation.

**WRK-R69.** Allocation identity, A2A Task and context identity,
Contractor Worker invocation identity and ADK session identity are four
separate domains. Runtime creates opaque ADK session IDs according to the
pinned Stage mode; none is derived from, equal by contract to, or exposed
through the allocation ID, A2A payload, `WorkerCompletion`, Agent Card, public
API or Planner tools. A rejected request therefore cannot infer whether a
prior session exists.

**WRK-R70.** Sessions are lazy invocation resources: Runtime creates
none while preparing the allocation. A request must first pass A2A payload,
tenant and lease validation, the accepting-versus-draining check and the
single-invocation lock. Invalid, stale, draining and concurrent-busy requests
are rejected without creating a session.

Once a request is admitted:

- **WRK-R71.** `shared` creates at most one opaque session per
  allocation (LIM-WRK-13) and reuses it for every sequential
  invocation in that allocation. Its ADK events and eligible State survive an
  ordinary success, a modeled Worker failure or a budget failure until
  allocation shutdown.
- **WRK-R72.** `isolated` creates a new opaque session for every
  invocation and supplies its ID explicitly to Runner and State
  synchronization. It copies no prior event, deletes the completed session
  inside the shielded invocation finalizer, and releases the invocation lock
  only after that deletion succeeds.

**WRK-R73.** In isolated mode Runtime carries a deep copy of eligible
State from one completed invocation to the next invocation of the same
allocation. The combined JSON encoding of that non-Contractor State is at most
4 MiB (LIM-WRK-14). Ordinary unprefixed State plus ADK `app:` and
`user:` State is eligible; `temp:` State is discarded, and no event is
converted into State. App and user maps remain owned by the same
allocation-local service and are never process-global. Values that cannot be
copied or encoded safely, or that exceed the bound, fail closed
(FAIL-WRK-4).

**WRK-R74.** The `contractor` subtree is never carried from
the ADK session snapshot. Every new isolated session receives it only from the
latest canonical `WorkerStateStore` revision, so a model, tool or stale
session cannot replace the trusted metrics and observation reducer. Allocation
secrets never enter either carried State or conversation events.

Activated-skill State, which is separate from the `contractor` subtree and may
follow the carry rule above, is owned by [Skills](../tools/skills.md).

**WRK-R75.** State already committed by ADK is snapshotted after
success, modeled failure, budget failure or cancellation. Snapshot and
isolated-session deletion execute as part of the existing
cancellation-shielded invocation finalizer.

**WRK-R76.** Failure to create, snapshot, bound or delete a session
fences the Worker from subsequent work (FAIL-WRK-4).

**WRK-R77.** Finalize, abort and release retry cleanup
idempotently. The Runtime slot cannot be reused until all owned sessions,
carried values, app and user maps and session identifiers are cleared. If
bounded cleanup cannot prove this before the control deadline, the allocation
remains fenced and the Runtime process is not reused (LIFE-WRK-2).

### Interrupted tool history

**WRK-R174.** Before another SDK turn or shared-session invocation can use
the conversation, Runtime settles each unresolved persisted tool call after
joining the Runner and its tools. It retains each known completed result
exactly once, using an immutable JSON snapshot rather than rerunning the tool.
Only a call whose tool callback was never admitted receives
`tool_call_not_executed`. An admitted call interrupted by cancellation receives
`tool_call_cancelled`; an admitted call without a known result or proven
cancellation receives `tool_call_outcome_unknown`. Neither dispatched outcome
claims that a remote effect was rolled back or absent. Existing operation
receipts remain authoritative before any retry of an uncertain effect.

**WRK-R175.** The invocation-local history ledger admits at most 10,000
observed tool-call identities cumulatively across SDK turns (LIM-WRK-22),
even after earlier results have settled. Unresolved call metadata and immutable
result snapshots together admit at most 32 MiB of UTF-8 data (LIM-WRK-23).
Raw tool arguments already stored in the SDK session are not copied into the
ledger. Resolved entries are released immediately. Snapshot encoding is
bounded before allocation, runs off the event loop and joins before cleanup;
it holds no instrumentation accounting lock. These ledger bounds do not bound
or compact the SDK session's retained conversation.

**WRK-R176.** History settlement uses the public SessionService and reads
at most twice the unresolved call count plus three recent events
(LIM-WRK-25). It preserves selecting call IDs, source-event identity, SDK turn,
session and branch/isolation scope. Existing saved responses win over missing
callback acknowledgements. A synthetic response group has a deterministic
identity and is never blindly appended again after an uncertain write:
Runtime accepts a lost acknowledgement only after a bounded read proves that
the identical event committed. Missing acknowledged source history, mismatched
scope, malformed or unencodable data, or an exceeded ledger bound fail closed
instead of dropping required context or inventing nonexecution.

**WRK-R177.** History settlement has its own five-second,
cancellation-shielded, joined cleanup budget (LIM-WRK-26). Failure, timeout or
unprovable settlement fences the allocation before any later semantic work;
it does not override an earlier cancellation or failure merely to report a
successful cleanup. If this boundary supplies the Worker failure, it uses
`worker_session_lifecycle_failed`: `invalid_tool_history` and
`tool_history_limit_exceeded` are non-retryable, while
`tool_history_unavailable` is retryable only through a fresh StageExecution.
Every case still fences this allocation. Release clears the transient ledger,
pending seal events and SDK turn identities after owned work has joined
(FAIL-WRK-7, LIFE-WRK-5).

### Secrets

**WRK-R78.** RuntimeSettings secrets supplied by Control Plane
are held outside ADK Session, State, events and model-visible instruction and
context. They configure clients such as the LLM Gateway adapter and are never
a telemetry source.

**WRK-R79.** Secrets are held in typed allocation context and
injected into already constructed clients and adapters. They are forbidden in
ADK State by invariant, not merely redacted on export. Runtime may initialize
safe, closed, non-secret State before Runner starts; there is no remote generic
State mutation API.

### Instrumentation

**WRK-R80.** An ADK Worker reserves one bounded `contractor` subtree in
its in-memory State and uses one Contractor `WorkerInstrumentationPlugin`,
registered on the ADK `App` supplied to `Runner`. The plugin observes run,
model and tool callback boundaries and feeds the independent allocation
metrics and invocation-observation reducers in that subtree
([Observations](#observations)).

**WRK-R81.** The plugin exists for lifecycle correlation; it is
not one untyped data bucket. Metrics and observations have different retention
and projection rules. Collecting per-tool metrics through the plugin must
not change the content of the `ExecutionReport` and must not count a call
twice.

**WRK-R82.** During finalization the Worker converts the metrics
reducer to the framework-neutral `ExecutionReport`. Raw ADK State is never a
Contractor wire or persistence schema.

**WRK-R83.** The result finalizer has a separate ephemeral ADK
Runner but reports its one model call through typed hooks on the same
invocation plugin. It therefore updates the same normal budget, metrics and
Contractor-owned State reducers and does not create a second uncorrelated
instrumentation State. Its prompt and output are never retained in that State.

The before-call hook owns its pending span until the hook returns successfully.
If cancellation interrupts its State publication after registration, it closes
that span synchronously under the plugin lock and propagates cancellation;
no model call starts and no extra publication or cleanup task is required.
Subsequent error callbacks and invocation teardown cannot close that span
again. Teardown keeps the authoritative terminal invocation phase separate
from auxiliary model phase names (WRK-R106, WRK-R125); the invocation's
existing bounded cleanup owns terminal State publication (WRK-R168).

**WRK-R84.** Model spans of the result-finalizer call mark
`model.phase=result_finalizer`.

**WRK-R85.** The plugin never generically copies tool arguments or
results into State. Each selected tool may register a narrow
`ObservationExtractor` that converts a completed call into a typed, bounded,
content-free fact. Unknown tools have metrics counts but contribute no domain
observation. An extractor failure is a safe instrumentation error: it sets
observation completeness false and cannot change or fabricate the tool result
(FAIL-WRK-5).

**WRK-R179.** Before retaining a tool name in metrics, spans or Worker
State, the plugin checks the allocation's selected tool-name set; an unknown
placeholder is recorded only as `unknown_tool`. Every admitted callback
attempt has one outcome and releases its correlation context once, including
raw-argument rejection, domain failure and parallel sibling cancellation.
`asyncio.CancelledError` reaching the SDK's tool-error callback closes the
attempt with `tool_call_cancelled` under OBS-R24 and is re-raised, so that
callback cannot recover cancelled execution or create a second attempt.
Runtime observes authorized tool cancellation independently of whether the SDK
propagates it or ends its iterator normally. It admits no later semantic call
and preserves cancelled invocation State before joined history settlement;
prior deadline, sandbox, history-validation and refused-budget failures retain
their existing authority. A valid new logical invocation resets its own
cancellation evidence, and foreign callbacks cannot alter it.

**WRK-R86.** Only a completed callback changes semantic
observations. A failed or cancelled call increments safe metrics but does not
claim that a file was read or changed. Parallel callback order is normalized by
the plugin's monotonic call ordinal and committed under one state lock.

### Workspace observations

**WRK-R87.** Workspace observation coverage measures what the Worker
actually exposed to the model during one A2A invocation. It is not
code-analysis scan coverage and does not claim that every file internally
parsed by Tree-sitter or Trailmark was read by the model.

**WRK-R88.** At invocation start, Runtime creates one
`WorkspaceObservationState` over the effective managed-text checkpoint
([Observations](#observations)).

The operation semantics are:

- **WRK-R89.** A successful `read_file` records its normalized relative
  path as read.
- **WRK-R90.** A successful `grep` records only paths whose returned
  matches were visible.
- **WRK-R91.** A successful `glob` or `ls` discovers the returned
  normalized managed-text file paths but does not mark their contents read;
  directory entries are not file coverage.
- **WRK-R92.** Successful Edit and annotation mutations record the
  affected managed-text file paths as modified.
- **WRK-R93.** `workspace-changes` records as modified only the file
  paths present in its returned bounded page.
- **WRK-R94.** Code-analysis coverage remains its own typed response
  and observation section and does not mark every internally scanned file as
  read.
- **WRK-R95.** Retries increment operation counts, while unique
  coverage counts each path once.

**WRK-R96.** Interactions are ordered by first completed observation
ordinal, with the relative path as a deterministic tie-breaker.

**WRK-R97.** Model-visible projections never contain host paths,
file contents, grep patterns, edit text, diffs, parser payloads or Artifact
revisions.

**WRK-R98.** The checkpoint scope admits at most 10,000 paths and
2 MiB of encoded path detail (LIM-WRK-15). Interaction detail
independently admits at most 10,000 paths (LIM-WRK-16).

**WRK-R99.** A Workspace larger than either bound remains usable:
the corresponding `*_complete` flag becomes false, counts are reported as
observed lower bounds, and no projection may claim exact unread coverage.

**WRK-R100.** The complete encoded `AgentStateSnapshot` containing
Contractor State is capped at 4 MiB (LIM-WRK-17). State admission
reserves the fixed envelope overhead rather than allowing a valid State value
to become an oversized HTTP response.

**WRK-R101.** The per-section bounds are maxima, not reservations:
optional path and detail admission stops earlier and marks the corresponding
completeness or truncation field when the shared ceiling would otherwise be
crossed. Mandatory schema fields and saturating aggregate counters have
reserved capacity, including while current and last-completed invocations
coexist.

**WRK-R102.** The immediate result uses one fixed `lean@1` projection
([Observations](#observations)). `discovered_files`, `read_files`,
`matched_files` and `modified_files` are unique managed-text path counts under
the operation semantics above. `files_read` holds at most 25 paths in
first-observation order (LIM-WRK-18), and `files_read_truncated`
is true exactly when `files_read` omits part of the `read_files` set. Tool
names are sorted and zero-call tools are omitted.

**WRK-R103.** `unread_files` is present only when both scope and
interaction detail are complete, and is the cardinality of
`scope_paths - observed_read_paths`. It is not arithmetic subtraction of the
two exposed counts, because a Worker may read a file created after the
checkpoint.

**WRK-R104.** The compact projection is always Runtime-authored. A
later named profile may change what is surfaced, but cannot reinterpret
results already stamped `lean@1`.

The precise future composition of the projection remains an evaluation
subject.

### Worker State lifecycle and contents

**WRK-R105.** Every live allocation-local ADK Session contains one
reserved `contractor` subtree. In `shared` mode this is the allocation's reused
session; in `isolated` mode Runtime rehydrates each fresh session from the
canonical allocation-local `WorkerStateStore`. That canonical subtree
(`ContractorWorkerState`, [Contractor-owned Worker State](#contractor-owned-worker-state))
is the only State surface Runtime exports.

**WRK-R106.** `current_invocation` is created immediately before
`Runner.run_async` and contains the Runtime-generated invocation ID, the
authoritative subtask ID, phase, invocation metrics and typed observation
sections. On every success, failure or cancellation it is atomically closed and
becomes `last_completed_invocation`. Beginning the next invocation does not
mutate the previous completed snapshot; completing it replaces that one-slot
history. Allocation-wide metrics continue across sequential invocations.

**WRK-R107.** State schema version 2 includes bounded invocation metrics
and a required closed summarizer section. Prompt-token trigger tracking stays
internal to the invocation budget; workspace integrity digests stay in the
workspace owner. Neither is an exported State field. The section is `disabled` when the pinned AgentTemplate has
no summarizer; otherwise it moves monotonically through
`not_requested -> requested -> succeeded|failed`. No completed invocation
keeps a `requested` section: when an invocation ends after the request and
before the summarizer records its outcome, for example through its deadline,
cancellation or a sandbox failure, completing it closes the section as
`failed` with the `cancelled` cause and no summarizer usage, and the summarizer
aggregate (WRK-R155) counts that cancelled attempt.

**WRK-R108.** The summarizer section retains only the normal-loop
state revision that caused the request, bounded one-call numeric summarizer
model and token usage, and one stable failure code. It never retains the
summary prompt or result.

**WRK-R109.** Every committed mutation increments `state_revision`.

**WRK-R160.** Snapshot reads take an immutable deep copy under the same
lock as State mutations; JSON encoding happens after the lock is released, so
a large read cannot block tool callbacks.

**WRK-R169.** A State revision does not serialize retained metric detail
again. The allocation metrics reducer serializes each tool-call record and
error once, when it admits it, and drops that projection when overflow removes
the record (LIM-OBS-4, LIM-OBS-5); a revision without a metric mutation
serializes no record. The reducer and the canonical State share these
projections read-only: a mutation copies only the invocation sections it may
change, admission (WRK-R101) replaces rather than edits a value it trims, and
every snapshot that leaves the store remains an independent deep copy
(WRK-R160). Copying the retained references stays linear in the retained
detail.

**WRK-R161.** Counters saturate at their declared bounds. Semantic
observation detail preserves its first-observation admission order and stops
admitting entries at its bounds (WRK-R97 to WRK-R102). Allocation metric
tool-call and error detail follows the rolling retention policy of OBS-R41:
overflow removes the oldest detail, preserves aggregate counters and sets
`truncated`. Its cached State projections follow WRK-R169.

**WRK-R110.** "Full state" always means the complete
Contractor-owned, versioned and bounded subtree. It excludes arbitrary or
framework-private ADK Session keys and conversation events; prompts, model
output history, hidden reasoning and complete tool payloads; activated-skill
framework State; Runtime Agent lifecycle and registry internals; and
RuntimeSettings, endpoints, headers, tokens and credentials.

**WRK-R111.** State is volatile. It exists only in the allocated
Runtime process and is destroyed with the Worker or on release. Durable metrics
use the bounded final report; durable domain state uses ArtifactStore and
MemoryTools (LIFE-WRK-3).

### Private agent-state endpoint

The read-only allocation-correlated route is transported by the private control
protocol ([Control Plane](control-plane.md)); it is neither an A2A operation
nor a public or generic state API. Its schema and lifecycle are defined here.

**WRK-R112.** The Python Runtime Agent exposes
`GET /private/v1/allocations/{allocation_id}/agent-state`. Its exact successful
JSON body is `AgentStateSnapshot` ([Interfaces](#interfaces)), and unknown fields
are rejected.

**WRK-R113.** The response deliberately does not repeat
`allocation_id`: request routing and the live allocation lookup establish that
identity, while `state.state_revision` establishes snapshot identity.

**WRK-R114.** The Worker implementation returns a framework-neutral
`AgentStateSnapshot`; the HTTP layer never reaches directly into the ADK
Session. The handler itself is owned by [Runtime Agent](runtime-agent.md).

The endpoint:

- **WRK-R115.** accepts only a Control Plane mTLS peer
  ([Control Plane](control-plane.md));
- **WRK-R116.** has no request body and no model- or user-provided
  state key;
- **WRK-R117.** returns the whole exportable Contractor State and the
  strong ETag `"contractor-agent-state-v1-<stateRevision>"`, derived only from
  its positive revision; the ETag format is its own independently versioned
  `v1` cache contract;
- **WRK-R118.** honors a matching `If-None-Match` with `304` and no body,
  for bounded Server-side caching;
- **WRK-R119.** sends `Cache-Control: private, no-cache`, so revision
  revalidation remains usable without accepting an unvalidated cached body,
  and a body no larger than 4 MiB including the response envelope
  (LIM-WRK-17);
- **WRK-R120.** is available for the active allocation while its Worker
  State exists;
- **WRK-R121.** returns the closed, safe `allocation_not_found` or
  `agent_state_unavailable` errors for stale, preparing,
  draining-with-destroyed-State or released slots. A missing, preparing or
  draining Worker returns retryable `409 agent_state_unavailable`; a snapshot
  that fails its own validation or exceeds LIM-WRK-17 returns non-retryable
  `503 agent_state_unavailable`. Other temporary snapshot infrastructure
  failures return retryable `503 agent_state_unavailable`. Invalid snapshots
  are never served partially;
- **WRK-R122.** never persists, mutates, resumes or reconstructs
  State.

### Planner access to live State

The Server-side Planner projection tools that read the snapshot are owned by
[Planners](planners.md).

**WRK-R123.** Planner models can call only explicit typed
projections, and never select a State key, physical Worker, allocation,
invocation or revision.

### Lifecycle and races

- **WRK-R124.** One allocation permits one active Worker invocation
  (LIM-WRK-19). State transitions and snapshot reads serialize without
  holding a lock across network, model or tool I/O.
- **WRK-R125.** A result is published only after the matching
  invocation State has been atomically closed, so
  `WorkerCompletion.state_revision` names a snapshot that already contains its
  observations.
- **WRK-R126.** A later invocation may replace
  `last_completed_invocation`; hidden correlation prevents an old Planner result
  from reading the new invocation.
- **WRK-R127.** Finalize and abort stop new reads once Worker State is
  destroyed. Final reports are built from the same reducer snapshot before
  destruction and remain the only durable metrics path.
- **WRK-R128.** Release erases State, ETag and cache material,
  observations, prompts and clients before the slot may become idle
  (LIFE-WRK-3).
- **WRK-R129.** Observation or State export failure can reduce
  telemetry completeness, but cannot invent a semantic Worker success or an
  artifact ref (FAIL-WRK-5).
- **WRK-R130.** A stale snapshot, truncated coverage or an
  unavailable Runtime is explicit and is never represented as complete
  coverage.

### Terminal summarization

**WRK-R131.** A long-running Worker should be able to stop before its
hard context or token budget and still return one useful typed semantic
result. For this, Runtime Agent owns an optional, deterministic terminal
summarizer, enabled by the AgentTemplate's `summarizer` block with fixed
`terminal@1` semantics ([AgentTemplates](../configuration/agent-templates.md)).

**WRK-R132.** Absence of `summarizer` preserves hard-budget behavior
exactly.

**WRK-R133.** The summarizer uses the same pinned Worker
LLMGatewayConfig and credential as the normal Worker but its own exact
ModelPolicy and model alias, whose required fields are owned by
[AgentTemplates](../configuration/agent-templates.md). Runtime enforces that
policy's one-call bound and, when `maxTotalTokens` is supplied, checks it
against the reported usage after that call.

#### Trigger

**WRK-R134.** An enabled normal Worker ModelPolicy must pin
`contextWindowTokens`.

**WRK-R162.** Runtime derives one prompt boundary from the effective Worker
policy and the resolved summarizer configuration as below; the second term
reserves the configured maximum normal response.

```text
context_boundary = min(
  floor(contextWindowTokens * contextWindowRatio),
  contextWindowTokens - maxOutputTokens,
)
```

**WRK-R163.** Configuration is invalid when
`maxOutputTokens >= contextWindowTokens`.

**WRK-R135.** Runtime requests summarization when the most recent
completed normal response reports `prompt_token_count >= context_boundary`.

**WRK-R136.** The optional `cumulativeBudget` independently
compares the sum of provider-reported `total_token_count` across all completed
normal-loop model calls in this A2A invocation. It is a soft cumulative
spending boundary that can stop a long sequence of individually small calls
before its expensive tail. It does not describe model context capacity and
does not replace the normal Worker policy's hard cumulative `maxTotalTokens`.

**WRK-R137.** Missing provider usage never invents a token estimate
and therefore cannot trigger the affected rule. A zero-valued total produced by
an adapter for an omitted provider `usage` object is also unavailable.

**WRK-R138.** Non-negative counters are internally
inconsistent when `prompt + completion > total` or cached input exceeds prompt
input. Runtime then drops that response's token counters closed, increments
`tokenUsageUnavailable`, and lets the independent model-call and tool-call
limits continue to bound execution.

**WRK-R139.** Independent model, tool and hard-token budgets remain
mandatory alongside the soft triggers.

Rationale: the context rule observes the last completed valid provider prompt,
not an exact tokenization of the prospective prompt after new tool results. Its
ratio and output reserve are a deterministic pre-emptive boundary, not a
guarantee that every provider will accept the next request.

These trigger rules apply to the ordinary completion strategy; the
`audit-check-results@1` contract rejects summarizer configuration
([Audit Worker completion](../audits/worker-completion.md)).

**WRK-R140.** The trigger is checked at a safe boundary:

1. a model response completes;
2. every tool call selected by that response either completes or fails;
3. if the main Worker response contains ordinary terminal semantic text,
   Runtime completes it through the mandatory result finalizer and returns it
   with `summarized: false`;
4. otherwise, before starting another normal model call, reaching either the
   derived context boundary or the optional cumulative budget requests terminal
   summarization.

**WRK-R141.** No in-flight tool is cancelled merely because a
completed response crossed a soft threshold, and no new normal tool or model
call starts after the request: a soft trigger starts no additional normal
Worker side effect.

**WRK-R142.** Hard exhaustion or a provider context failure that
occurs before this boundary uses the existing Worker failure path and does not
retroactively invoke the summarizer.

**WRK-R143.** The instrumentation plugin records only a private
`summarization_requested` control signal and must not make a nested LLM call
inside an ADK callback. The Runtime invocation coordinator unwinds the normal
Runner and invokes the independent summarizer afterward.

#### Input

**WRK-R144.** The summarizer has no Contractor or domain tools,
Agent Skills, Artifact client, MemoryTools or workspace handles. Its input is
one Runtime-built bounded document containing:

- the authoritative subtask ID, objective and task instructions;
- the immutable string parameters and named inputs already visible to the
  Worker;
- a chronological projection of model-visible normal-loop events, without
  thought parts unless WRK-R181 enables provider reasoning;
- the deterministic latest `WorkerObservations`.

**WRK-R145.** Trusted result bindings remain private Runtime data
and are not supplied as a separate summarizer input. An ArtifactRef may
reappear only if it was already part of a bounded model-visible event
projection.

**WRK-R146.** Runtime never includes system or runtime internals,
hidden thought other than provider reasoning admitted by WRK-R181, credentials, headers, provider exceptions, host paths or
arbitrary ADK State. Before projection it replaces the allocation's host paths,
the LLM Gateway token and every RuntimeSettings credential or private endpoint
of at least 16 UTF-8 bytes (LIM-WRK-12) with `[REDACTED]`; the
credentials are those scanned by WRK-R60, plus the private
endpoints.

**WRK-R147.** The event projection is capped at 512 KiB
(LIM-WRK-20). Immutable task data is retained first; the newest
complete event groups that fit are retained next, preserving their original
order, and a deterministic `transcriptTruncated` marker reports any omission.
Tool payloads remain subject to their existing model-visible bounds. Runtime
records the projection only for a Worker whose pinned AgentTemplate has a
summarizer; without one it records no transcript (WRK-R132).

**WRK-R148.** The 512 KiB projection is the only local
document-size admission; it does not establish that the complete request fits
the summarizer model's context. There is no additional byte-to-token estimate,
framing reserve or context-based history trimming. Runtime sends the bounded
projection with the pinned instructions and output schema, and the Gateway
decides context acceptance. Context failures keep their non-retryable
classification (WRK-R18).

**WRK-R181.** An AgentTemplate summarizer with `includeReasoning: true`
(TPL-R47) adds provider reasoning to the summarizer input. Authoring omission
normalizes to false; the resolved value is an explicit required boolean (TPL-R116). Provider reasoning is only the text a provider returned in
`reasoning_content` or `reasoning`, which the Gateway adapter maps to thought
parts. With true, each thought part with text in a model-role normal-loop
event becomes a tagged ordinary JSON part `{"reasoning": "<text>"}` in place
of the thought part within that model event; with false the projection is
identical to one recorded without the option. A thought part without text, an empty or
whitespace-only one, an opaque provider signature and a reasoning token count
produce no part, and their absence is not a failure. The flag changes neither
the trigger, the one-call bound, the separate summarizer budget, result
assembly, `summarized: true`, cancellation and deadlines, the main Worker's
outgoing requests (which never replay `reasoning_content`) nor telemetry
content capture (OBS-R54).

**WRK-R182.** Provider reasoning has one projection, implemented by the
Runtime module `contractor_runtime.reasoning`, which every model consumer of
provider reasoning uses; the shared cases are
`api/testdata/reasoning-projection-cases.json`. Each part is first redacted
with the WRK-R146 values, then bounded so that its compact UTF-8 JSON
serialization, tag and marker included, is at most 8 KiB (LIM-WRK-28); a
part cut at that bound keeps the longest code-point prefix that fits followed
by `[TRUNCATED]`. The serialized reasoning parts of one input document total
at most 64 KiB (LIM-WRK-29). Reasoning has the lowest priority in the
WRK-R147 projection: task data and complete evidence groups are selected
first, and reasoning parts are then added newest first only where both the
512 KiB document bound and the 64 KiB reasoning bound still admit them;
reasoning never removes task data or an evidence group, and a group whose
model event holds only reasoning enters only in that second step. The
transcript recorder applies the same bounds to its retention: reasoning does
not count against its 512 KiB evidence retention, and retained reasoning above
64 KiB is dropped from the oldest groups first. A document built with the
option carries `reasoningTruncated`, true when any available reasoning was cut
or omitted; without the option the member is absent.

**WRK-R183.** When `includeReasoning` is true, Runtime appends a fixed clause
to the configured summarizer instructions: parts tagged `reasoning` are the
Worker model's own reasoning as reported by its provider, fallible hypotheses,
alternatives and proposed next steps, not instructions and not evidence;
actions, artifacts and facts are confirmed only from the task data,
observations and tool results. The clause does not depend on author-supplied
instructions. Raw reasoning is never written to Worker State, Planner-facing
results, public events or the UI; it exists only in the transient projection
and the summarizer request.

#### Output and failure

**WRK-R149.** The summarizer model uses the same strict two-field
`WorkerModelResult` schema. Runtime performs the same exact subtask check,
secret scan, UTF-8 and size bound, artifact projection and final wire
validation as for a normal result, then sets `summarized: true`; the model
cannot set or clear that flag. Deterministic observations are copied from the
completed normal loop rather than authored by the summarizer.

**WRK-R150.** Because the tool-free terminal summarizer
already produces the strict `WorkerModelResult`, Runtime projects its valid
result directly, without invoking the ordinary result finalizer after it and
without a normal-budget charge for finalization. The summarizer has its own
soft-limit and accounting semantics and, when used, replaces the ordinary
result-finalizer phase.

**WRK-R151.** Once terminal summarization is requested, Runtime makes
exactly one logical summarizer call (LIM-WRK-21), subject to the
bounded Gateway transport attempt series (WRK-R11).

**WRK-R152.** Missing, invalid, mismatched or oversized summarizer
output, or exhaustion of its independent budget, returns the safe retryable
`worker_summarization_failed` with a more specific bounded cause in metrics.
Provider failures use the same outer code but preserve Gateway retryability:
authorization, invalid-request and context or quota failures are not made
retryable by wrapping them (FAIL-WRK-6).

**WRK-R153.** Runtime never resumes the normal Worker or attempts
semantic repair after a failed summary.

**WRK-R178.** Empty, whitespace-only or thought-only terminal summary
output produces `worker_summarization_failed` through the existing bounded
missing/invalid-result cause. The one-shot model wrapper prevents SDK repair
from making another provider call, including after an empty structured
response, and known usage is retained. A `MAX_TOKENS` response keeps the
`output_limit_exceeded` summary cause; provider safety and Gateway failures
keep their own classification. No summary failure resumes the normal loop or
invokes the ordinary finalizer.

#### State, metrics and lifecycle

**WRK-R154.** Runtime does not retain the summarizer transcript or a separate
output archive. It returns the semantic output in `WorkerResult`; durable
Stage content remains Planner-owned. Passthrough records its trusted
incomplete completion and Scheduler publishes its ordinary summary artifact
before continuation or ordinary failure ([SCH-R171](scheduler.md)).
Worker State keeps only bounded state/usage, not summary text (LIFE-WRK-4).

**WRK-R155.** `ExecutionMetrics` carries a separate optional
summarizer aggregate, so that normal Worker usage and terminal summarization
cost can be evaluated independently. The aggregate contains attempt, success
and failure counts, model calls, provider-reported input, output and total
tokens, a missing-usage count and bounded stable failure-code counts; each
count saturates at the execution report integer bound
([OBS-R111](../platform/observability.md)). It contains no transcript or
result text and no endpoint or model-provider body. The immutable
AllocationSpec supplies exact summarizer ModelPolicy and Gateway attribution.

**WRK-R156.** Summarization is semantic work and therefore occurs only
inside the active A2A invocation while allocation writes are allowed. It never
runs during finalize, abort, lease-loss drain, report collection or release.
Cancellation winning before its result stops the call and produces no
`WorkerResult`; normal bounded abort semantics apply
(FAIL-WRK-6).

### Context compaction

A Workflow binding may select `compact@1`
([WFL-R91 to WFL-R95](../configuration/workflows.md)). The Worker then
continues the same isolated subtask past its context boundary: Runtime
summarizes the oldest complete history groups into one checkpoint and keeps
the recent ones. Proposal 0046 records the design and the ADK 2.13 evidence
it builds on. Shared sessions and Audit completion Workers are outside this
version.

#### Selection and boundary

**WRK-R184.** AllocationSpec carries exactly one effective
`contextManagement` branch: `none` (the template has no summarizer),
`terminal@1` (the template summarizer of WRK-R140 onward) or `compact@1`
with every member of the binding policy, including the pinned compactor
ModelPolicy and instruction text. Server derives it from the binding and the
template; the template itself is not rewritten. Go and Runtime validation and
the allocation schema reject a branch that disagrees with the template
(`none` with a summarizer, `terminal@1` without one), `compact@1` for a
`tool@1` template, a `shared` session, an Audit completion contract or an
`audit-results@2` template, a `summarizerModelPolicy` outside `terminal@1`,
and a compacted Worker policy without `contextWindowTokens` or
`maxOutputTokens`. Runtime never falls back from `compact@1` to terminal
summarization.

**WRK-R185.** The compaction boundary is
`min(floor(contextWindowTokens * contextWindowRatio), contextWindowTokens -
maxOutputTokens)` of the effective main Worker policy. Before each main model
request, Runtime compares the provider-reported prompt tokens of the latest
main model response of the invocation with the boundary; when it is reached,
the request is not sent or charged and a checkpoint starts instead. Missing
usage never triggers a checkpoint and is never estimated from bytes. After a
committed checkpoint the measurement is cleared, so the next decision waits
for the first main response in the new context.

**WRK-R186.** The compactor is one tool-free, one-shot model call per
checkpoint through its own model client, built from the pinned compactor
ModelPolicy with no tools, Skills, workspace or summarizer. It charges the
owning invocation's hard budgets and metrics like the result finalizer: one
model call and its reported tokens, under the separate model phase
`context_compactor`. Its usage never replaces the main prompt measurement of
WRK-R185. It also stays within the compactor policy's own `maxModelCalls: 1`
and `maxTotalTokens`. Exhausting the owning invocation's budget returns
`worker_budget_exhausted`; exceeding the compactor's own limits returns
`worker_compaction_failed`.

#### History and selection

**WRK-R187.** Runtime indexes the persisted events of the invocation into
ordered complete groups: one plain model event, or one function-call event
with exactly the function responses that answer its call identifiers. Events
authored by the user or Runtime are not history. A duplicate, partial or
error event, a branch or isolation scope, native ADK compaction, a new call
before the previous group is answered, a response to an unknown call or plain
content inside an open group makes the history unsafe; a checkpoint is then
refused with `worker_compaction_failed` and its cause. Each group is
projected once, off the event loop, with the terminal summarizer projection
(WRK-R146, WRK-R147): credentials and private endpoints redacted, part bounds
applied, and a Skill tool response (`list_skills`, `load_skill`,
`load_skill_resource`) replaced by a fixed notice so history cannot bypass the
Skill disclosure budget. A checkpoint selects the longest prefix of
uncompacted groups that leaves the newest `keepRecentGroups` untouched and
fits the compactor input bound. Bounds and their outcomes
(`worker_context_limit_exceeded` unless stated):

- at most `maxCompactions` checkpoints per invocation (LIM-WFL-9) and 32 per
  allocation (LIM-WRK-30);
- the protected recent groups at most 128 KiB (LIM-WRK-31);
- the compactor input document at most 8 MiB (LIM-WRK-32), failing only when
  not even one eligible group fits;
- the retained history of an invocation at most 4,096 groups (LIM-WRK-35)
  and 32 MiB (LIM-WRK-34);
- no eligible group at all is `worker_compaction_failed` with cause
  `ineffective`.

**WRK-R188.** With `includeReasoning: true`, each projected group keeps the
tagged provider-reasoning parts of WRK-R181 and WRK-R182, and the compactor
instruction receives the fixed clause of WRK-R183. Evidence is selected first:
reasoning bytes do not count against LIM-WRK-31, the input bound reserves
LIM-WRK-29 for reasoning, and reasoning is admitted newest group first within
that budget. The checkpoint header then carries `reasoningTruncated`, true
when any reasoning was omitted or cut. With `includeReasoning: false`,
reasoning is neither retained nor sent. The main model never sees reasoning
from compacted or retained groups.

#### Output and commit

**WRK-R189.** The compactor's only output is a strict object with one
non-blank `summary` string. Empty or thought-only output, invalid JSON or
members, a summary above 16 KiB of UTF-8 (LIM-WRK-33), a summary not smaller
than the evidence bytes it replaces, a response that reached its output
limit, and a summary exposing a credential or private endpoint value each
return `worker_compaction_failed` with a bounded cause. The new summary
replaces the previous one; the compactor receives the previous summary in its
input.

**WRK-R190.** A valid summary is committed by appending one marker event to
the ADK session: a Runtime-authored event whose metadata records only the
generation and the first and last compacted group, never summary text. Before
appending, Runtime rechecks that the history has not changed since the
selection. When the append fails, Runtime re-reads the session once: an
exactly equal stored marker counts as committed; no stored marker returns
`worker_compaction_failed` (`marker_not_committed`) without further effect; a
conflicting marker or an unreadable session fences the allocation's session
lifecycle and returns `worker_compaction_failed` (`marker_unverified`). The
checkpoint becomes visible to the Worker only after the commit. Runtime never
appends a second marker for the same checkpoint.

#### Continuation

**WRK-R191.** A compacted Worker has one extra read-only tool,
`read_context_history(group, position)`. It returns compacted groups of the
current invocation only, marked as historical data: whole groups as projected
JSON while they fit one page of 32 KiB and 8 groups (LIM-WRK-36), otherwise
one bounded fragment of one group with its position, a completion flag and
the next position. A group outside the compacted range, a position that is not
a returned continuation, or a call from another invocation is a tool input
error. The tool returns the same redacted projection the compactor saw, without
reasoning unless it was selected, and with the Skill notice in place of Skill
content.

**WRK-R192.** After a commit, the same logical invocation continues in a new
SDK turn. Each main model request is projected by exact event identity: one
leading user content holding the subtask prompt, the checkpoint summary
labelled as model-written historical data with unverified references, and a
continuation hint, followed by the retained uncompacted events with thought
parts removed. Markers and compacted events are omitted; an unknown event in
the session returns `worker_compaction_failed` (`foreign_event`). The
continuation keeps the invocation's deadline, budgets, tool history,
observations and artifact authority. A valid result after compaction is an
ordinary `WorkerResult` with `summarized: false`, finalized as usual
(WRK-R46).

**WRK-R193.** When the first main response after a commit still reports a
prompt at or above the boundary, Runtime stops with
`worker_compaction_failed` (`ineffective`) instead of starting another
checkpoint.

#### Failures and lifecycle

**WRK-R194.** Compaction failures use two codes. `worker_compaction_failed`
covers unsafe history, compactor provider and output failures, stale
selections and marker commit outcomes; it is retryable except when it wraps a
permanent Gateway failure, whose retryability it preserves (FAIL-WRK-8).
`worker_context_limit_exceeded` covers the bounds of WRK-R187 and is not
retryable, because a new attempt would reach the same bound. Messages carry
only the bounded cause, never content.

**WRK-R195.** Checkpoint state (archived group projections, summary, markers
known to the invocation and selection cursors) lives only in the allocation's
memory for the active invocation and is cleared when it ends; only the
allocation-wide checkpoint count survives until release (LIFE-WRK-6).
Besides the session's own events, which are not copied, one invocation holds
at most the retained projection (LIM-WRK-34), one compactor input document
(LIM-WRK-32; it is assembled off the event loop from per-group strings, so
at most twice its bound while it is joined), one summary (LIM-WRK-33) and one
history page (LIM-WRK-36). A history page addresses its first group
by ordinal and serializes at most one page, so a read never rescans the
archive. Successful and failed checkpoints are reported through the span
below and the `context_compactor` model phase; richer counters are not part
of this version. The
session itself, including markers, follows LIFE-WRK-2. A checkpoint is
semantic work: it runs only inside the active invocation, checks the deadline
and cancellation before the compactor call and before the commit, and never
runs during finalize, abort, report collection or release. Cancellation
before the commit leaves no marker and no visible checkpoint
(FAIL-WRK-8). Each checkpoint emits one `contractor.worker.context_compaction`
span with the generation, outcome and bounded cause.

## Interfaces

### A2A Worker completion

The private A2A request carries Stage content
(`application/vnd.contractor.stage-content+json`); the terminal response
carries a `WorkerCompletion`
(`application/vnd.contractor.worker-completion+json`), WRK-R38:

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

### Agent-state endpoint

```http
GET /private/v1/allocations/{allocation_id}/agent-state
```

Exact successful JSON body:

```python
class AgentStateSnapshot(BaseModel):
    api_version: Literal["contractor/v1alpha1"]
    state: ContractorWorkerState
```

### LLM Gateway

Worker model calls use the `openai-compatible@1` Chat Completions protocol
through the allocation's pinned Gateway route (WRK-R8).

## Limits, failures and errors

Limits ([registry](../platform/limits.md)):

- LIM-WRK-1: Gateway client `max_retries=3`, one initial attempt and
  at most three transport retries (WRK-R11).
- LIM-WRK-2: a finite server retry delay over 120 seconds stops
  transport retries (WRK-R12).
- LIM-WRK-3: the attempt series is capped at `requestTimeoutSeconds`
  plus 60 seconds (WRK-R13).
- LIM-WRK-4: `subtask_id` 1..128 bytes (WRK-R30).
- LIM-WRK-5: result text 64 KiB of UTF-8 (WRK-R30, WRK-R37, WRK-R47).
- LIM-WRK-6: `WorkerFailure.code` 1 to 64 characters
  (WRK-R36).
- LIM-WRK-7: `WorkerFailure.message` 4,096 UTF-8 bytes
  (WRK-R36).
- LIM-WRK-8: `invocation_id` 1..128 bytes
  (WRK-R36).
- LIM-WRK-9: `StageContentRequest`, `WorkerCompletion`, Planner-facing
  `WorkerResult` and `StageResult` 256 KiB each (WRK-R37).
- LIM-WRK-10: result-finalizer input document 256 KiB
  (WRK-R46).
- LIM-WRK-11: one result-finalizer provider call per ordinary
  completion (WRK-R48).
- LIM-WRK-12: 16 UTF-8 bytes, the length from which a credential is
  rejected anywhere in a result, and from which a RuntimeSettings credential
  or private endpoint is replaced with `[REDACTED]` in the summarizer input
  (WRK-R61, WRK-R146).
- LIM-WRK-13: at most one ADK session per `shared` allocation
  (WRK-R71).
- LIM-WRK-14: isolated-mode carried State 4 MiB (WRK-R73).
- LIM-WRK-15: checkpoint scope 10,000 paths and 2 MiB
  (WRK-R98).
- LIM-WRK-16: interaction detail 10,000 paths (WRK-R98).
- LIM-WRK-17: `AgentStateSnapshot` and agent-state response body
  4 MiB (WRK-R100, WRK-R119).
- LIM-WRK-18: `files_read` 25 paths (WRK-R102).
- LIM-WRK-19: one active Worker invocation per allocation
  (WRK-R124).
- LIM-WRK-20: summarizer event projection 512 KiB
  (WRK-R147).
- LIM-WRK-21: one logical summarizer call per invocation
  (WRK-R151).
- LIM-WRK-22: 10,000 observed tool-call identities cumulatively per logical
  invocation (WRK-R175).
- LIM-WRK-23: 32 MiB of unresolved call metadata and result snapshots
  (WRK-R175).
- LIM-WRK-24: 1,001 SDK turn identities per logical invocation
  (WRK-R173).
- LIM-WRK-25: at most twice the unresolved call count plus three recent
  SessionService events per history reconciliation read (WRK-R176).
- LIM-WRK-26: five-second joined history-settlement cleanup budget
  (WRK-R177).
- LIM-WRK-27: every `WorkerCompletion` integer at most 2^53-1
  (WRK-R164).
- LIM-WRK-28: one serialized tagged provider-reasoning part 8 KiB
  (WRK-R182).
- LIM-WRK-29: serialized provider reasoning per consumer input document
  64 KiB (WRK-R182).
- LIM-WRK-30: 32 committed compactions per allocation (WRK-R187).
- LIM-WRK-31: protected recent groups 128 KiB (WRK-R187).
- LIM-WRK-32: compactor input document 8 MiB (WRK-R187).
- LIM-WRK-33: compactor summary 16 KiB (WRK-R189).
- LIM-WRK-34: retained history 32 MiB per invocation (WRK-R187).
- LIM-WRK-35: retained history 4,096 groups per invocation (WRK-R187).
- LIM-WRK-36: `read_context_history` page 32 KiB and 8 groups (WRK-R191).
- Used here, owned by Workflows: LIM-WFL-9 to LIM-WFL-12 (`compact@1`
  members, WFL-R92).

Errors ([registry](../platform/errors.md)):

- `worker_gateway_unavailable`, `worker_gateway_invalid_request`
  (WRK-R19, WRK-R21);
- `worker_budget_exhausted` (WRK-R25);
- `worker_timeout` (WRK-R158, WRK-R167);
- result failures: `worker_result_missing`, `worker_output_limit_exceeded`,
  `worker_result_invalid`, `worker_result_subtask_mismatch`,
  `worker_result_finalizer_mismatch`, `worker_result_finalizer_failed`,
  `worker_result_too_large` and `unsafe_worker_result`, alongside
  `worker_budget_exhausted` and domain and runtime failures
  (WRK-R57 to WRK-R61);
- `worker_summarization_failed` (WRK-R152);
- `worker_compaction_failed` and `worker_context_limit_exceeded`
  (WRK-R194);
- `worker_session_lifecycle_failed` and its bounded history causes
  (WRK-R177);
- `tool_call_not_executed`, `tool_call_cancelled` and
  `tool_call_outcome_unknown` in interrupted history (WRK-R174);
- `allocation_not_found`, `agent_state_unavailable` (WRK-R121).

`worker_busy` and `worker_execution_failed` are owned by
[Runtime Agent](runtime-agent.md).

Failure outcomes ([failure model](../platform/failure-model.md)):

- FAIL-WRK-1: an LLM Gateway model call and its transport attempt
  series (WRK-R11 to WRK-R20).
- FAIL-WRK-2: request construction from ADK content
  (WRK-R21).
- FAIL-WRK-3: the result-finalizer call
  (WRK-R48, WRK-R52, WRK-R57).
- FAIL-WRK-4: ADK session creation, snapshot, bounding, carry and
  deletion (WRK-R73, WRK-R76).
- FAIL-WRK-5: observation extraction and State export
  (WRK-R85, WRK-R129).
- FAIL-WRK-6: the terminal summarizer call
  (WRK-R152, WRK-R156).
- FAIL-WRK-7: interrupted tool-history reconciliation and settlement
  (WRK-R174 to WRK-R177).
- FAIL-WRK-8: one context checkpoint: compactor call and marker commit
  (WRK-R186, WRK-R189, WRK-R190, WRK-R195).

Lifecycle ([registry](../platform/lifecycle.md)):

- LIFE-WRK-1: the result finalizer's ephemeral Runner, Session and
  input document (WRK-R46).
- LIFE-WRK-2: allocation-local ADK sessions and carried State
  (WRK-R72, WRK-R77).
- LIFE-WRK-3: the Contractor-owned Worker State
  (WRK-R111, WRK-R128).
- LIFE-WRK-4: summarizer input and output (WRK-R154).
- LIFE-WRK-5: the transient interrupted-history ledger and SDK turn
  identities (WRK-R175, WRK-R177).
- LIFE-WRK-6: compaction checkpoint state of one invocation (WRK-R195).

## Security

The security-relevant rules of this document are:

- the Worker context and what each model sees: WRK-R2,
  WRK-R4, WRK-R5, WRK-R44,
  WRK-R45, WRK-R144, WRK-R145, WRK-R181, WRK-R183;
- models cannot select outcome, slot, revision, retryability or lifecycle:
  WRK-R41, WRK-R50, WRK-R54,
  WRK-R55, WRK-R149;
- secrets: WRK-R17, WRK-R60,
  WRK-R61, WRK-R78,
  WRK-R79, WRK-R74, WRK-R146, WRK-R182;
- identity separation: WRK-R69;
- semantic admission, raw arguments and interrupted history: WRK-R170,
  WRK-R171, WRK-R173 to WRK-R177;
- the read-only endpoint and Planner access: WRK-R110,
  WRK-R115, WRK-R116, WRK-R122,
  WRK-R123, WRK-R97.

mTLS peer identity itself is owned by [Security](../platform/security.md).

## Observability

- Instrumentation plugin, reducers and `ExecutionReport`: WRK-R80,
  WRK-R81, WRK-R82, WRK-R85,
  WRK-R86, WRK-R179.
- Result-finalizer accounting and the `model.phase=result_finalizer` span:
  WRK-R49, WRK-R83, WRK-R84.
- Token usage accounting: WRK-R16, WRK-R24,
  WRK-R138.
- Summarizer State and the `ExecutionMetrics` summarizer aggregate:
  WRK-R108, WRK-R155.
- Context compaction: the `contractor.worker.context_compaction` span and the
  `model.phase=context_compactor` span (WRK-R186, WRK-R195).

Metrics and OTLP conventions shared by every component are owned by
[Observability](../platform/observability.md).

**WRK-R180.** Allocation composition may install a bounded infrastructure
checkpoint at invocation completion, after tool producers settle and before
returning the Worker result. Automatic collection failures produce technical logs and leave HTTP results,
Worker results, Runtime reports and Run state unchanged.
Resource I/O has its own allowance outside the five-second State finalization
budget. The sitemap checkpoint runs while allocation writes are still authorized;
[SMT-R21](../artifacts/project-sitemap.md) owns its flush and loss semantics.

## Invariants

These properties follow from the cited requirements, which are authoritative.

1. The Worker model authors only the exact subtask ID echo and the semantic
   result text (WRK-R41, WRK-R50).
2. Runtime authors observations, the `summarized` state, artifact refs and all
   technical failures; Planner authors the Stage outcome (WRK-R26,
   WRK-R27, WRK-R149).
3. The outbound subtask ID is copied from the trusted request and must match
   the model's echo (WRK-R50).
4. Observations describe completed model-visible operations, never filesystem
   access inferred from an implementation detail (WRK-R86,
   WRK-R87).
5. Metrics and observations share callback correlation but have independent
   typed reducers and retention (WRK-R81).
6. The private endpoint exports only bounded Contractor-owned State and cannot
   mutate it (WRK-R122).
7. Credentials and secret-bearing configuration never enter ADK State
   (WRK-R79).
8. Planner models call only explicit typed projections and never select a
   State key, physical Worker, allocation, invocation or revision
   (WRK-R123).
9. Live State is advisory context; artifacts remain durable data and final
   reports remain durable telemetry (WRK-R111).
10. A stale snapshot, truncated coverage or unavailable Runtime is explicit and
    never represented as complete coverage (WRK-R130).
11. Absence of `summarizer` preserves hard-budget behavior exactly
    (WRK-R132).
12. A soft trigger starts no additional normal Worker side effect
    (WRK-R141).
13. Runtime, never either model, owns `summarized` (WRK-R149).
14. The summarizer has one call, no tools and an independent pinned ModelPolicy
    (WRK-R133, WRK-R144, WRK-R151).
15. Summarizer failure cannot resume the normal loop or invoke a repair loop
    (WRK-R153).
16. Normal and summarizer usage remain separately attributable and secret-free
    (WRK-R155).
17. Finalize, abort and release never invoke the summarizer
    (WRK-R156).
18. A completion reminder uses a fresh SDK turn while business tools and
    results retain the logical invocation identity (WRK-R173); tested by
    `test_real_runner_continuation_preserves_one_invocation_and_tool_order`
    in `runtime/tests/test_audit_completion_continuation.py`.
19. Interrupted history preserves known sibling results and never changes an
    admitted uncertain call into proven nonexecution (WRK-R174, WRK-R176);
    tested by
    `test_real_runner_next_request_keeps_completed_sibling_and_truthful_interrupted_call`
    in `runtime/tests/test_worker_history.py`.
20. SDK preprocessing cannot replace raw integer arguments before their
    owning validation (WRK-R171); tested by
    `test_function_tool_keeps_raw_numbers_and_native_structured_conversion`
    in `runtime/tests/test_worker_function_tools.py`.

## Non-goals

- A Worker-facing semantic `fail` tool or model-authored failure status is
  deliberately not defined.
- There is deliberately no `PUT`, `PATCH` or generic command counterpart to the
  agent-state endpoint, and no endpoint mutates ADK State during execution.
  Configuration and credentials arrive once through the pinned allocation
  prepare contract; per-invocation semantic parameters arrive through
  `StageContentRequest`; cancellation and finalization already have explicit
  typed lifecycle commands; and shared mutable Planner and Worker information
  belongs in MemoryTools and artifacts.
- A generic Planner `get_state`, arbitrary state keys or model-defined filters.
- Generic or secret-backed ADK instruction templating; instructions reach the
  model literally (WRK-R166). Future ADK instruction
  templating may consume an explicitly specified safe initial-state schema: a
  later closed schema may seed approved non-secret allocation and task values
  without adding a remote State-write endpoint. Arbitrary placeholders and
  secret-backed values are out of scope.
- Public or UI access, historical snapshots or persistence of detailed paths.
- Dynamic configuration or credential replacement inside an allocation.
- User-defined observation reducers or YAML A/B profiles.
- Exact byte or range coverage, and claims that static-analysis scanning equals
  model-visible file reading.
- ADK native compaction: Contractor neither enables nor wraps ADK
  `EventsCompactionConfig`, token-triggered or post-invocation compaction, or
  SDK resumability. `compact@1` (WRK-R184) is Runtime-owned, selected by a
  Workflow binding and separate from the terminal path; compaction of shared
  sessions, of Audit completion Workers, cumulative-spend triggers and durable
  checkpoint recovery after a process restart are not supported.
- Rolling, hierarchical, extractive and artifact-backed summarization
  strategies; introducing one requires an explicit discriminator rather than
  changing `terminal@1` semantics.
- Summarizer retries, tool access or model-authored observations.
- A separate summarizer Gateway or credential, or an executionConfig override
  of the summarizer.
- Estimated-token triggers when provider usage is absent.
- Reactive recovery after a provider has already rejected the context length.
- UI mutation of summarizer policy for an existing Run or allocation.
