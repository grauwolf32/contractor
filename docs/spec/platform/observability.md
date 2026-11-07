# Observability

Requirement prefix: `OBS` · [Specification index](../README.md)

## Purpose and scope

This document owns the observability rules that every domain relies on:

- request correlation through `X-Request-ID` and the content of server-side 5xx
  diagnostics;
- execution reports: the `ExecutionReport`, `RuntimeReport`,
  `AllocationFinalReport` and `StageMetrics` shapes, the trusted Server
  envelope, usage attribution, the meaning and capture rules of reports,
  redaction of tool arguments and errors with the shared secret-substitution
  policy, report bounds and the retention of durable telemetry;
- external trace export through `otlp-http@1`: span and attribute content,
  content capture and sensitive-output tools, the Worker exporter's queue,
  batching, retry, acknowledgement and flush behaviour, its dropped-span
  counters, and Planner telemetry built in the Server;
- propagation of WorkflowRun metadata labels into telemetry.

It leaves to other documents:

- StageMetrics as a StageExecution field, separate from the stored terminal
  facts: [Scheduler](../execution/scheduler.md)
  ([SCH-R51](../execution/scheduler.md));
- metadata labels themselves, their bounds, the UI treatment of label values
  and the public `AttemptDiagnostics` projection of report errors in Run status:
  [Runs](../execution/runs.md) ([RUN-R77](../execution/runs.md),
  [RUN-R91](../execution/runs.md), [RUN-R92](../execution/runs.md));
- selection of the telemetry adapter (`worker.telemetry`, `planner.telemetry`),
  the export settings with their defaults and allowed values, credential
  records and label-resolution provenance:
  [Runtime configuration](../configuration/runtime-config.md)
  ([RCF-R97](../configuration/runtime-config.md),
  [RCF-R98](../configuration/runtime-config.md),
  [RCF-R99](../configuration/runtime-config.md));
- the Worker instrumentation plugin, its reducers and the summarizer aggregate:
  [Worker](../execution/worker.md) ([WRK-R80](../execution/worker.md),
  [WRK-R155](../execution/worker.md));
- the content of durable Planner facts: [Planners](../execution/planners.md);
- closing and releasing allocation adapters on the Runtime side:
  [Runtime Agent](../execution/runtime-agent.md)
  ([RTA-R58](../execution/runtime-agent.md));
- what the metrics of each toolset may contain: the documents listed under
  [Telemetry rules owned by other documents](#telemetry-rules-owned-by-other-documents);
- performance collection, the optional `RuntimeReport.resources` extension,
  Server and database performance history with its startup switch, HTTP
  listener metrics and profiling: [Operations](../operations/operations.md)
  ([OPS-R69](../operations/operations.md),
  [OPS-R70](../operations/operations.md),
  [OPS-R99](../operations/operations.md),
  [OPS-R100](../operations/operations.md));
- trust boundaries and the handling of RuntimeSettings secrets:
  [Security](security.md);
- the REST error envelope: [Public API](../api/public-api.md).

## Concepts

Terms are collected in the [glossary](../glossary.md); this document owns the
following ones.

- **Request ID.** The bounded `X-Request-ID` value that correlates one HTTP
  request across hops and diagnostics.
- **Execution report.** The framework-neutral `ExecutionReport` that a Planner
  or an in-process Worker runtime produces: aggregate metrics, tool-call records
  and errors.
- **Runtime report.** The `RuntimeReport` of allocation lifecycle facts that the
  Runtime Agent around a Worker observes, including Runtime adapter metrics.
- **Final allocation report.** The `AllocationFinalReport` that carries one
  allocation's Worker execution report and Runtime report.
- **Participant.** The Planner or one logical Worker of a StageExecution; each
  participant reports separately.
- **Trusted Server envelope.** The identities, attribution and correlation
  metadata that the Server binds to a report from its own records, never from
  the report payload.
- **Incomplete placeholder.** The report, with unknown fields, that Scheduler
  records for an expected participant that started but produced no report.
- **Related telemetry.** A report delivered after its StageExecution is
  terminal, retained without changing the frozen terminal facts.
- **Runtime adapter metrics.** Per-adapter counters in a Runtime report, keyed
  by the exact RuntimeAdapter ref.
- **Content capture.** The opt-in `captureContent=true` mode in which prompts,
  responses and tool arguments and results are exported to the telemetry sink.
- **Trusted sink.** A telemetry backend that content capture trusts with
  unredacted content.
- **Sensitive-output tool.** A tool that declares `contractor_sensitive_output`
  because its results carry command output, captured HTTP traffic or HTTP
  session material.
- **Metadata-only instrumentation.** Instrumentation that exports no content,
  whatever the content-capture setting.
- **Local discard.** A span that the Worker exporter itself drops, counted in
  `droppedSpans`.
- **PlannerTelemetryAdapter registry.** The Server-owned, versioned registry of
  the telemetry adapters a Planner may use.
- **Closed instrumentation vocabulary.** The fixed set of Planner span kinds and
  metadata attributes; content never enters it.

## Model

### Report shapes

A Planner and the in-process Worker runtime produce the same
`ExecutionReport`. The Runtime Agent that surrounds a Worker produces a
separate `RuntimeReport`. One `AllocationFinalReport` carries a Worker's
execution report and its Runtime report, and the StageExecution's
`StageMetrics` holds the Planner report, one execution report per logical
Worker and one Runtime report per logical Worker.

```python
class ExecutionError(BaseModel):
    code: str
    message: str
    retryable: bool | None = None


class ToolMetrics(BaseModel):
    calls: int | None = None
    succeeded: int | None = None
    failed: int | None = None


class WorkerBudgetMetrics(BaseModel):
    max_model_calls: int
    max_tool_calls: int | None = None
    max_total_tokens: int
    observed_model_calls: int
    observed_tool_calls: int
    observed_total_tokens: int
    token_usage_unavailable: int
    exhausted: Literal["model_calls", "tool_calls", "total_tokens"] | None = None


class ExecutionMetrics(BaseModel):
    duration_ms: int | None = None
    model_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    tools: dict[str, ToolMetrics] = Field(default_factory=dict)
    worker_budget: WorkerBudgetMetrics | None = None


class ToolCallRecord(BaseModel):
    call_id: str
    tool: str
    arguments: dict[str, Any] | None = None
    arguments_truncated: bool = False
    outcome: Literal["succeeded", "failed"]
    duration_ms: int | None = None
    result_size_bytes: int | None = None
    error: ExecutionError | None = None


class ExecutionReport(BaseModel):
    report_id: str
    complete: bool
    metrics: ExecutionMetrics
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    errors: list[ExecutionError] = Field(default_factory=list)
    truncated: bool = False


class RuntimeAdapterMetrics(BaseModel):
    operations: int
    failed_operations: int
    flush_attempted: bool | None = None
    flush_succeeded: bool | None = None
    last_error_code: str | None = None


class RuntimeReport(BaseModel):
    complete: bool
    duration_ms: int | None = None
    stop_reason: str | None = None
    adapters: dict[str, RuntimeAdapterMetrics] = Field(default_factory=dict)


class AllocationFinalReport(BaseModel):
    report_id: str
    allocation_id: AllocationID
    started_at: datetime
    finished_at: datetime
    worker: ExecutionReport
    runtime: RuntimeReport


class StageMetrics(BaseModel):
    planner: ExecutionReport | None = None
    workers: dict[AgentName, ExecutionReport] = Field(default_factory=dict)
    runtime: dict[AgentName, RuntimeReport] = Field(default_factory=dict)
```

`ExecutionMetrics` also carries the optional summarizer aggregate specified in
[WRK-R155](../execution/worker.md). The Worker Runtime report of an
`otlp-http@1` adapter adds the optional `droppedSpans` counters
(OBS-R81). Every integer of `ExecutionMetrics` is bounded (OBS-R111).

### Report flow

The trusted Server envelope, not the reporting participant, supplies every
identity and attribution field.

```mermaid
sequenceDiagram
    participant W as Worker runtime
    participant RA as Runtime Agent
    participant P as Planner
    participant S as Server
    participant SM as StageMetrics
    W->>RA: ExecutionReport (metrics, tool_calls, errors)
    RA->>S: AllocationFinalReport (report_id, worker ExecutionReport, runtime RuntimeReport)
    P->>S: Planner ExecutionReport
    S->>S: Bind Run, StageExecution, allocation ID and logical Agent name
    S->>S: Derive ModelPolicy, Gateway and credential refs from the Run snapshot and pinned allocation provenance
    S->>S: When label-driven configuration is active, derive label names, RuntimeConfig refs/digests and RuntimeAdapter refs
    S->>S: Optionally copy the Run's metadata labels
    S->>SM: Record planner, workers and runtime entries (repeated report_id is idempotent)
    Note over S,SM: A started participant without a report gets an incomplete placeholder
    Note over S,SM: A report after the StageExecution is terminal is related telemetry only
```

### Worker trace export

An allocation-scoped Worker `otlp-http@1` exporter queues finished spans and
sends them in bounded batches through one asynchronous sender. Every local
discard is counted under one `droppedSpans` reason.

```mermaid
flowchart TD
    A["Span finishes"] --> Q{"Fits both queue limits?"}
    Q -- no --> QO["Drop span: queueOverflow; wake the sender if another whole span cannot fit before batchSizeBytes is reached"]
    Q -- yes --> E["Append to the queue"]
    E --> T{"Queue reaches batchSizeBytes or maxPendingSpans?"}
    T -- no --> WT["Wait for more spans or the final flush"]
    T -- yes --> B["Sender encodes a batch of whole spans, at most batchSizeBytes"]
    B -- "encoding fails or a whole span cannot fit" --> EF["Discard: encodingFailed"]
    B --> H["HTTP attempt"]
    H --> R{"Outcome"}
    R -- "acknowledged" --> OK["Remove the delivered prefix"]
    R -- "partial success" --> PS["Discard rejectedSpans only: collectorRejected"] --> OK
    R -- "permanent failure or invalid acknowledgement" --> NR["Discard batch: nonRetryable"]
    R -- "temporary failure" --> RT{"Attempt left and delay fits the batch budget?"}
    RT -- yes --> BO["Wait Retry-After or backoff"] --> H
    RT -- "no attempt left" --> RE["Discard batch: retryExhausted"]
    RT -- "delay does not fit or deadline expired" --> DE["Discard batch: deadlineExceeded"]
    H -- "sender cancelled" --> CA["Discard batch: cancelled"]
    E -- "close" --> SD["Discard active batch and queued tail: shutdown"]
```

### Metadata labels in telemetry

```mermaid
flowchart LR
    WR[("WorkflowRun row: immutable metadata labels")] --> PS["Planner spans"]
    WR --> SC["Scheduler"] --> AS["AllocationSpec.runMetadataLabels"]
    AS --> RV["Runtime validates the wire bounds"]
    RV --> TA["Telemetry adapter: per-execution span attributes"]
    RV -. never .-> MV["Prompts, ADK State, tool arguments, WorkspaceFS, Skills, model-visible results"]
```

## Requirements

### Request correlation

**OBS-R1.** Every public and private HTTP response carries one bounded
`X-Request-ID` (LIM-OBS-1).

**OBS-R2.** Public ingress always generates its own request ID.

**OBS-R3.** Private Control Plane, lifecycle, Artifact and A2A hops
propagate one syntactically valid incoming value and replace a missing,
duplicated or malformed value.

**OBS-R4.** The Server's private Control Plane and Artifact APIs
propagate an incoming value only from a peer that presented a verified Runtime
Agent certificate. A request that fails mTLS gets a generated value, so an
unauthenticated caller cannot choose its logged ID.

**OBS-R5.** REST error objects repeat the request ID as `requestId`
([Public API](../api/public-api.md)). A2A keeps its protocol error envelope and
carries correlation in the HTTP header.

### Server-side diagnostics

**OBS-R6.** Server-side 5xx diagnostics record the request ID, boundary,
method, status and safe error type. They do not record raw URL paths, request
bodies, artifact bytes, provider exception messages, RuntimeSettings or tokens.

Rationale: an operator gets a stable lookup key without a user-controlled path
or a nested tool argument becoming a logging channel.

### Execution reports and the trusted envelope

**OBS-R7.** The Planner and the in-process Worker runtime use the same
framework-neutral report shape, `ExecutionReport`. Allocation lifecycle facts
observed by the surrounding Runtime Agent remain a separate `RuntimeReport`,
because their source and completeness differ.

**OBS-R8.** The trusted Server envelope binds each report to the
Run, the StageExecution, the globally unique allocation ID and the logical Agent
name. A Worker-supplied payload cannot select or override those identities.

**OBS-R9.** `report_id` makes repeated delivery of a report
idempotent (FAIL-OBS-1). A repeated report matches the stored one when the two
documents are equal as jsonb, compared by PostgreSQL itself, so a number the
column stored in another form (`5e-05` as `0.00005`, `-0.0` as `0.0`) still
matches; another document under a stored identity is a conflict.

**OBS-R10.** The envelope derives ModelPolicy attribution from the
Run's immutable ResolvedExecutionConfig. The Planner's LLMGatewayConfig and
LLMCredential refs also come from that Run snapshot.

**OBS-R11.** For a logical Worker, the final LLMGatewayConfig
and LLMCredential refs come from the Server-pinned allocation configuration
provenance, because the default, Run and Agent infrastructure layers may
override the Workflow route
([Runtime configuration](../configuration/runtime-config.md)).

**OBS-R12.** None of the ModelPolicy, LLMGatewayConfig or
LLMCredential refs is accepted from a Runtime report. They let Operations
aggregate calls and tokens by policy, route and credential without persisting
or exposing the secret value.

**OBS-R13.** When label-driven infrastructure configuration is
active, the envelope also derives the sorted Run and Agent label names, the
exact RuntimeConfig refs and digests and the RuntimeAdapter refs from
Server-owned resolution provenance. Runtime reports cannot claim or replace
them. Endpoint credentials and resolved secret-bearing RuntimeSettings remain
absent. The pinning and merge contract is owned by
[Runtime configuration](../configuration/runtime-config.md).

**OBS-R14.** The envelope may additionally copy the WorkflowRun's
immutable metadata labels ([Runs](../execution/runs.md)) as trusted correlation
metadata. These labels describe the Run, are not accepted from Runtime reports
and are not metric dimensions (OBS-R108).

### Usage attribution

**OBS-R15.** Attribution records the ModelPolicy, LLMGatewayConfig and
credential refs together with the WorkflowRun, the StageExecution and the
Planner or Worker role.

**OBS-R16.** Provider-reported tokens and calls are operational
measurements, not billing-grade proof.

**OBS-R17.** Gateway-reported spend and rate state may be displayed as
operational information, but a Gateway quota rejection follows the ordinary
bounded LLM error path (FAIL-OBS-2).

**OBS-R18.** ModelPolicy call and token ceilings remain
independently enforced, because they bound one Planner or Worker execution
rather than account consumption.

### Report content

**OBS-R19.** `complete` describes report completeness, not
execution success.

**OBS-R20.** A known zero is `0`; an unobserved or unsupported value is
`null` or absent.

**OBS-R21.** `metrics` contains only aggregate values. Tool
arguments and errors are diagnostic records, not counters.

**OBS-R22.** `worker_budget` is present for an ADK Worker invocation.
It records the exact configured ceilings, the operations actually started, the
sum of available provider token usage, the count of responses without usage and
at most one exhausted dimension (LIM-OBS-2). `max_tool_calls` is absent only
for a compatible tool-free policy. `worker_budget` contains no prompt, response
or tool payload.

**OBS-R23.** Every started tool call is recorded exactly once (FAIL-OBS-3).

**OBS-R24.** A tool call cancelled while it runs, including an Agent
Skill or `audit-results@2` call and a call still pending when the Worker closes,
fails with the retryable `tool_call_cancelled` unless its toolset records a
domain error (FAIL-OBS-3).

**OBS-R25.** Full tool results are not retained by default. Durable or
large results belong in the ArtifactStore
([Artifact store](../artifacts/artifact-store.md)); metrics may retain their
size and outcome without their body (LIFE-OBS-1).

**OBS-R26.** External observability exporters are optional
adapters over Contractor-owned reports; no LangChain- or Langfuse-style service
is required.

### Redaction of tool arguments and errors

**OBS-R27.** Tool arguments are recursively redacted before they leave
the Worker and again at the Server persistence boundary. They are bounded
(LIM-OBS-6) and carry an explicit truncation marker. The value of a member
with a sensitive name becomes the placeholder `{"redacted": true, "size": n}`,
where `size` is a non-negative integer or null; every name the Server treats
as sensitive the Worker treats as sensitive too. Redacting a redacted record
again changes nothing: a value that already is that placeholder, a
non-negative integer under a size name the Worker derives from content it
omits (`content_bytes`, `description_bytes`, `result_content_bytes`,
`result_description_bytes`) and the summary that replaces arguments above
LIM-OBS-6 (`{"summary": "[TRUNCATED]", "originalSizeBytes": n}`) pass
unchanged.

**OBS-R28.** MemoryTools applies a stricter projection: note
content, description and tags are always dropped rather than merely truncated.
Only the operation, the logical Worker (`worker_name`) when applicable, the note
name, byte sizes, outcome and duration may remain. The memory-specific
diagnostic and content contracts are in [Memory](../tools/memory.md).

**OBS-R29.** Agent Skill tools apply a similar stricter projection:
only the validated logical skill name, the bounded normalized reference or asset
path, outcome, duration, result size and stable error code may remain.
Instructions, resource and package bytes, extracted paths and generated context
are always dropped ([SKL-R114](../tools/skills.md)).

**OBS-R30.** RuntimeSettings tokens and other known deployment
secrets are always removed, even when a tool argument or error accidentally
contains them.

**OBS-R31.** One secret-matching policy applies to metrics, allocation
checks and Worker results: a secret of at least 16 UTF-8 bytes (LIM-OBS-3)
matches wherever it occurs, while a shorter one, such as a proxy username or a
short header value, matches only a complete string, so it never garbles
ordinary text. Long secrets are replaced longest first, so a short secret
that prefixes a longer one cannot leave the longer one's remainder. The policy
holds at the Server's telemetry persistence boundary and for Planner span
attributes as well as in the Runtime; the Server and the Runtime each define it
once and both read the cases in
`api/testdata/v1alpha1/secret-matching-cases.json`. In metrics a match is
replaced by `[REDACTED]`. How a Worker
result that contains a secret is handled is specified in the Worker document
([WRK-R60](../execution/worker.md), [WRK-R61](../execution/worker.md)). How an
allocation check applies it is specified in the Control Plane document
([CTL-R110](../execution/control-plane.md)).

**OBS-R32.** Tool names and error codes are Runtime vocabulary
and are never redacted. An error code that is not an identifier, or that
contains a long secret, is recorded as `tool_call_failed` (FAIL-OBS-3). A
tool call that raises an exception is recorded once as failed, and the Worker
answers the model with a failed tool result instead of failing the
invocation. Its code is the exception's own identifier code, or
`tool_call_failed` when the exception has none; it is retryable only when the
exception declares itself retryable, so an unclassified exception is
non-retryable; and its message names only the tool and the exception class,
as in `list_symbols failed (TypeError)`, never the exception text. Only a tool
input validation error replaces that message with its bounded Runtime-authored
diagnostic.

### Runtime adapter metrics

**OBS-R33.** Runtime adapter metrics are keyed by the exact
RuntimeAdapter ref selected by trusted Server provenance.

**OBS-R34.** Adapter counters are non-negative and saturating.
`flush_*` fields are absent for an adapter without a flush operation.

**OBS-R35.** Adapter error codes are a bounded allowlist and
never contain an endpoint, provider body or exception message.

**OBS-R36.** The Server accepts adapter metrics only for refs present
in the pinned allocation provenance. Unknown keys and malformed counters are
discarded as incomplete telemetry with a bounded safe diagnostic; they cannot
relabel attribution or block semantic terminalization and allocation release
(FAIL-OBS-1).

### Report collection

**OBS-R37.** For every expected participant that started but produced
no report, Scheduler records an incomplete placeholder with unknown fields;
a Worker allocation's placeholder carries the retryable error
`allocation_report_unavailable`.
The one exception is a model-free Planner, which does no model work: its
placeholder states zero model calls and tokens, as its own report would
([EVM-R81](../evals/managed-evals.md)). Absence from a `StageMetrics` map means
the participant was not started or was not applicable (FAIL-OBS-1).

**OBS-R38.** Report collection is best-effort and cannot raise into the
Planner or change a semantically valid StageResult or a durable
StageTermination (FAIL-OBS-1).

**OBS-R39.** A report delivered after the StageExecution is terminal
may be retained as related telemetry but never mutates the frozen StageResult or
StageTermination (FAIL-OBS-1).

### Report bounds and retention

**OBS-R40.** The bounded report policy is fixed: at most 1,000
tool-call records (LIM-OBS-4) and 100 errors (LIM-OBS-5) per participant, at
most 4,096 UTF-8/JSON bytes per argument summary or error message (LIM-OBS-6)
and at most 1 MiB per final allocation report (LIM-OBS-7). The report shape
does not depend on these values, so they can become deployment policy without
a change to the shape.

**OBS-R41.** Overflow removes the oldest detail, preserves aggregate
counters and sets `truncated`.

**OBS-R42.** JSON sizes count compact UTF-8 bytes without HTML
escaping; U+2028 and U+2029 count as three-byte UTF-8 characters. Error-message
sizes count UTF-8 text bytes, independently of their JSON escaping. These rules
apply both to final reports and live Worker State, including per-tool errors.
The shared `execution-error-cases.json` fixture pins both diagnostic bounds. An
empty tool `arguments` object counts as absent, as both sides encode it, and a
number in tool arguments counts as the literal its producer wrote: the Server
keeps that literal instead of re-formatting it, so it measures and bounds the
bytes the Runtime measured, and an integer beyond 2^53 keeps its value. Such a
number is zero or the decimal exponent of its leading digit lies between -7 and
19, and any exponent it writes lies in that range too (LIM-OBS-16); the Server
rejects a received report with any other number. The range keeps probabilities
down to one in ten million, float32 epsilon and every 64-bit integer, so ports,
counts, sizes and timestamps in seconds, milliseconds or nanoseconds stay as
written; it is no wider because PostgreSQL stores a number as its full decimal
expansion (OBS-R110). Both sides judge a number as written, the Runtime by
Python's `repr` of a float, which switches to exponent form below 1e-4 and at
1e16. The Runtime writes any other number, NaN and infinities as
`"[NUMBER_OUT_OF_RANGE]"` and marks the arguments truncated; the Server writes
the same marker for such a number in a summary it builds itself, such as a
Planner tool call. The shared `telemetry-report-trim-cases.json` fixture pins
the range, the marker and the numbers on each side of both edges, and the
private allocation fixtures hold a report with numbers just inside both edges
and reports each with one just outside.

**OBS-R110.** A report over its byte bound loses detail in one order: the
oldest tool-call record first, the oldest error only after no tool-call record
remains (OBS-R41). The result is the first report of that removal sequence
whose compact JSON (OBS-R42) fits; once a record is removed `truncated` is
`true`, and a report that does not fit loses at least one record even when the
shorter `"truncated":true` alone would make it fit. When no detail is left and
the report still does not fit, the report is rejected as an oversized
aggregate. Records are redacted and bounded (OBS-R27, OBS-R30) before anything
is measured, so unredacted content is never sized. Since the JSON of a list is
its elements joined by commas, each side measures every candidate record once
and the complete report at most three times per bound (the candidate, the
report without detail and the retained result), however many records it
removes; the retained result must measure exactly the computed size, or the
report is rejected. The Worker producer and the Server persistence boundary
implement the rule separately. Given the same records they keep the same
suffix, and they measure a record the Runtime built at the same size
(OBS-R42). The Server applies the rule again after its own redaction, first at
the Worker report bound and then at the 1 MiB allocation bound (LIM-OBS-7); it
bounds a Planner report at 1 MiB. That redaction leaves a record the Runtime
redacted unchanged (OBS-R27), except where it replaces a deployment secret it
holds (OBS-R30) or a text containing `://` that its URL check treats as
credential-bearing. Only such a change makes the Server remove more detail
than the Runtime removed. The shared `telemetry-report-trim-cases.json` fixture
pins the retained suffix at every removal boundary, the equal sizes and the
unchanged records, including records the Runtime's producer built.

The byte bounds measure the compact JSON a report is received in, and only
that: the private response bounds the report a Runtime sent at 1 MiB before it
decodes it (LIM-OBS-7). PostgreSQL stores each number as its exact decimal
expansion and renders a stored document with a space after each separator, so
a stored report reads back as longer text and its decoded numbers keep that
expansion. A stored report is decoded as strictly as a received one and passes
every rule of a received one except the byte bounds. The replay identity
lookups decode a stored report this strictly, and the StageMetrics rebuild
also checks every report rule before it stores the document. Every reader of
a stored StageMetrics document applies the same decoding and rules: Run detail
omits a document that fails them ([RUN-R90](../execution/runs.md)), and Eval
usage collection, which reads each report's completeness and metrics, fails
as a data-integrity error ([EVM-R81](../evals/managed-evals.md)). Run detail
and the rebuild decode a stored Stage summary the same way, with every member
present and no counter negative (OBS-R112). The Eval token threshold reads
only the summary's `totalTokens` inside one bounded SQL sum
([EVM-R248](../evals/managed-evals.md)) and relies on the rebuild being the
summary's only writer: it derives the summary from validated reports and
decodes the stored copy strictly. LIM-OBS-16 bounds the expansion: a kept
number has at most 20 integer digits, and at most 7 more fraction digits than
its literal, so no element expands more than `1e19,` does
into `10000000000000000000, `, 4.4 times, and a stored report reads back as at
most 4.4 times its 1 MiB bound. Measured on PostgreSQL 17 with 1 MiB reports of
250 tool calls, arrays of `1e19` read back as 4.27 times the bound, the
Runtime's `1e+19` as 3.57 times, `1e-07` as 1.81 times, one-character strings
as 1.24 times, and a mix of ports, counts, timestamps, fractions and paths as
1.09 times. The StageMetrics rebuild decodes the Planner's report and the
report of every logical Agent of the Stage at once, encodes them into the
StageMetrics document and decodes that again; Run detail decodes that
document, and the replay identity lookup (OBS-R9) decodes one stored report.
Their memory grows by at most that factor over the received bounds.

**OBS-R111.** Every integer of an execution report's `metrics`, with its
`tools`, `worker_budget` and summarizer members, lies between 0 and
2^53-1 (LIM-OBS-17), the largest integer every JSON consumer, TypeScript
included, reads exactly. The allocation schema, the Server's report validation
and the Runtime model apply the same bound, and the shared
`execution-metric-integer-cases.json` fixture pins it on both sides of the
bound for each member. A producer saturates at the bound: the Runtime keeps
its allocation counters at the Worker State bound and projects each one into
the report as at most 2^53-1, so a Gateway that reports an absurd token count
yields a valid report. A Planner cannot reach the bound: its usage arrives as
32-bit counts per response and it stops at its ModelPolicy ceilings
(LIM-TPL-3, LIM-TPL-6). A received report with an integer above the bound
violates the contract and takes the outcome of a participant that produced no
report: the Scheduler records the incomplete `allocation_report_unavailable`
placeholder for that allocation, and a Planner report is not recorded
(FAIL-OBS-1). The bound also keeps every sum the Server builds from reports
exact for any realistic Stage: a Stage of 1,000 Workers and its Planner, every
counter at the bound, sums to 9,016,206,453,995,731,991, below 2^63-1.

**OBS-R112.** A Stage summary (`StageMetrics.summary`, the public
`MetricsSummary`) adds its reports' counters with saturation: a sum beyond
2^63-1 is 2^63-1, never a wrapped negative value. The Operations totals of
one allocation report are that allocation's Stage summary and follow the same
rule. The Eval token lower bound that adds Stage summaries saturates the same
way ([EVM-R248](../evals/managed-evals.md)); the Eval usage read model instead
marks an overflowing counter unavailable
([EVM-R240](../evals/managed-evals.md)).

**OBS-R43.** Durable telemetry expires after 30 days (LIM-OBS-8). Cleanup
is batch-bounded and may delete only telemetry that belongs to a terminal
StageExecution (LIFE-OBS-2). Expired StageMetrics of a Run that an existing
Eval member executed, directly or as a child Run of the Audit it executed,
are renewed for another retention period instead of deleted
([EVM-R239](../evals/managed-evals.md)). A renewal changes only the expiry,
so it does not mark the member for recollection, and it takes the row out of
the expired range, so later passes do not scan it again.

**OBS-R44.** Each telemetry table has a bounded deletion budget; StageMetrics
renewals count against the StageMetrics budget. Full batches are drained with
a short pause before the next retention interval, so a metrics backlog cannot
starve reports.

### Trace export with `otlp-http@1`

The `otlp-http@1` adapter itself and its selection are specified in
[RCF-R98](../configuration/runtime-config.md).

**OBS-R45.** `otlp-http@1` exports OTLP/HTTP protobuf traces to the
configured complete traces endpoint. It does not export OpenTelemetry metrics or
logs and does not replace Contractor's durable `ExecutionReport`.

**OBS-R46.** Telemetry instrumentation emits bounded
framework-neutral spans and events for model calls, selected tool calls, the A2A
task lifecycle and Worker errors.

**OBS-R47.** Bounded resource attributes contain the Contractor
service and version, the Run, StageExecution and allocation or Planner session
correlation IDs, the logical Worker name, the exact adapter and configuration
refs and safe label names.

**OBS-R48.** Span attributes may contain the operation kind, model alias,
tool name, outcome, duration and token and count aggregates.

**OBS-R49.** One span has at most 64 attributes (LIM-OBS-9). Metadata
strings are at most 256 UTF-8 bytes (LIM-OBS-10).

**OBS-R50.** Transport credentials and provider error bodies are
never added automatically.

**OBS-R51.** Export uses a bounded queue. Each Planner and Worker exporter
queue holds at most 2,048 span records (LIM-OBS-11) and 64 MiB of encoded
pending data (LIM-OBS-12).

**OBS-R109.** A record that would exceed either bound is dropped and counted;
overflow increments adapter failures without blocking model or tool execution
(FAIL-OBS-4).

The Worker's queue limits are configured within these bounds in
`worker.telemetry.export` ([RCF-R99](../configuration/runtime-config.md)).

**OBS-R52.** There is no durable exporter spool and no infinite retry. A
process crash may lose unflushed external telemetry, while Contractor's own
durable lifecycle remains authoritative (LIFE-OBS-3).

### Content capture

**OBS-R53.** Content capture is off by default
(`captureContent=false`): prompts, model responses, artifact bytes and content,
tool arguments and results and RuntimeSettings secrets are absent from exported
telemetry.

**OBS-R54.** Setting `captureContent=true`, independently on Worker
telemetry, Planner telemetry or both, explicitly trusts the configured sink with
unredacted content. No secret-content filter is applied to captured content.
With capture enabled, content may itself contain secrets and
is not filtered, so secrets included in messages or tool results can be
exported.

**OBS-R55.** Content capture does not include automatic export of
client configuration or authentication headers.

**OBS-R56.** Tools whose results carry command output, captured
HTTP traffic or HTTP session material declare sensitive output
(`contractor_sensitive_output`). A Worker that selects any such tool uses
metadata-only instrumentation for the whole allocation, even when
`captureContent=true`.

Rationale: later model requests, result finalizers and summaries can repeat the
earlier output.

**OBS-R57.** `exec_command` declares sensitive output on the tool. The
`http-tools@1` and `caido@1` toolset factories declare it for every tool they
export (`sensitive_output_tools`); the Runtime derives the set from the
registered toolsets and passes it to the Worker.

**OBS-R58.** Captured model input includes conversation messages,
system instructions and tool declarations; captured output includes the model
response. Worker tool spans include arguments and results. Planner invocation
and passthrough dispatch spans include their semantic input and output. Worker
result-finalizer and terminal-summarizer model calls also capture input and
output; a summarizer input that includes provider reasoning
([WRK-R181](../execution/worker.md)) is captured only under these rules, and
`includeReasoning` enables no capture.

**OBS-R59.** Model spans use OpenTelemetry GenAI semantic conventions
v1.41.1: `gen_ai.operation.name=chat`, `gen_ai.provider.name=contractor.gateway`
(the client-visible proxy, without guessing the upstream provider), and
`gen_ai.request.model` containing the requested model alias. Captured content
uses JSON-serialized `gen_ai.input.messages`, `gen_ai.output.messages`,
`gen_ai.system_instructions` and `gen_ai.tool.definitions`, preserving roles,
text, reasoning and function calls/results. A missing output finish reason is
explicitly `unknown`. Per-call reported counts use integer
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` and
`gen_ai.usage.cache_read.input_tokens`; unavailable counts are absent and no
duplicate total attribute is emitted. Non-model content uses
`contractor.input` and `contractor.output`. Neither exporter emits collector
vendor attributes or requires a collector SDK. Go and Python validate
[`telemetry-genai-cases.json`](../../../api/testdata/telemetry-genai-cases.json).

The content shapes and their JSON-string encoding follow the
[pinned GenAI conventions](https://github.com/open-telemetry/semantic-conventions/blob/v1.41.1/docs/gen-ai/gen-ai-spans.md).

**OBS-R60.** Each captured content field is bounded to 256 KiB of
UTF-8 (LIM-OBS-13), separately from the metadata-string bound. Oversized content
uses a valid schema-shaped array with a bounded text preview marked
`truncated: true` for GenAI fields; generic Contractor content uses a valid
JSON object with `truncated: true` and a preview.

**OBS-R61.** The 64 MiB pending-queue bound (LIM-OBS-12) still applies
to captured content, so content-heavy spans may be dropped. The bound is per
exporter, not a process-wide memory limit: simultaneous allocations multiply the
retained capacity, and serialization and transport may need additional memory.

**OBS-R62.** Content capture does not change durable reports,
ordinary logs or metrics.

**OBS-R63.** Existing Runs keep their pinned configuration; enabling
capture affects only new Runs resolved against the new RuntimeConfig version.

### Worker export scheduling

**OBS-R64.** A Worker exporter starts one asynchronous sender when the
queue reaches `batchSizeBytes` of encoded pending data or `maxPendingSpans`.

**OBS-R65.** Requests contain whole spans and are bounded to
`batchSizeBytes`, including the protobuf envelope.

**OBS-R66.** An overflow also wakes the sender to drain accepted spans
when another whole span cannot fit before the byte threshold is reached
(FAIL-OBS-4).

**OBS-R67.** The sender transmits ready batches sequentially while Worker
callbacks continue appending. The in-flight batch remains charged to both queue
limits. Only its prefix is removed after delivery, which preserves spans added
during the request.

**OBS-R68.** Each batch gets at most `maxAttempts` HTTP attempts; retries
preserve the payload and the span IDs.

**OBS-R69.** One absolute `flushTimeoutSeconds` budget covers a batch's
encoding, HTTP attempts and waits. Each HTTP attempt is also bounded by
`min(requestTimeoutSeconds, flushTimeoutSeconds)`.

**OBS-R70.** Final draining has its own `flushTimeoutSeconds` cap,
shortened by the enclosing allocation deadline. None of these deadlines is
extended for retries.

**OBS-R71.** Adapter operation and error counters account for each
HTTP attempt.

**OBS-R72.** Subsequent batches can proceed after a failed batch is
discarded. Cancellation is not retried (FAIL-OBS-5).

**OBS-R73.** There is no periodic timer: a sub-threshold tail waits for
more spans or the final flush.

### Worker delivery and acknowledgements

**OBS-R74.** Worker delivery retries temporary transport failures
and HTTP 429, 502, 503 and 504. Other HTTP errors, including 500, are terminal
(FAIL-OBS-5). A temporary transport failure is any member of the Runtime's one
HTTP transport failure family (httpx's errors and `OSError`, such as
`ssl.SSLError` for a corrupt TLS record or a connection reset), raised while
sending the batch or reading the acknowledgement, except an unsupported
protocol and a local protocol error, which are terminal, and a body that fails
content decoding, which is a malformed acknowledgement (OBS-R77). Any other
exception is an implementation failure and terminal. The exporter applies its
own retry budget, backoff and `Retry-After` rules (OBS-R75, OBS-R76) to this
family.

**OBS-R75.** A valid `Retry-After` header is honoured as seconds or as an
HTTP date. Without it, exponential backoff starts at
`initialBackoffMilliseconds` and doubles up to `maxBackoffMilliseconds`; equal
jitter selects a delay between half that ceiling and the ceiling.

**OBS-R76.** If the delay cannot fit the remaining batch budget, the
batch expires without another attempt (FAIL-OBS-5).

**OBS-R77.** Exporters accept bounded standard OTLP protobuf or JSON
acknowledgements. JSON decoding ignores unknown message fields, including
additional receipt metadata, as required by OTLP/JSON and consistent with binary
protobuf. Known fields such as `partialSuccess.rejectedSpans` retain their
validation; malformed and oversized acknowledgements are rejected. The Worker
applies the acknowledgement byte ceiling (LIM-OBS-22) separately to received
and decoded bytes, using the shared HTTP-body reader. Compressed responses
are decoded incrementally, with each decoding layer bounded before output is
accumulated and at most a one-byte overflow probe. Wire overflow, decoded
overflow and malformed compression are invalid acknowledgements. The exporter
closes the response under the existing delivery deadline.
Worker delivery treats invalid acknowledgements as terminal
(FAIL-OBS-5); Planner delivery reports its existing export failure.

**OBS-R78.** OTLP partial success is never retried: only the
`rejectedSpans` are discarded. A warning with zero rejected spans counts as
successful delivery (FAIL-OBS-5).

These rules follow the
[OTLP retry and partial-success rules](https://opentelemetry.io/docs/specs/otlp/).

**OBS-R79.** Provider error messages are not retained in
diagnostics.

**OBS-R80.** The Worker export scheduling, retry and
acknowledgement rules above do not change Planner export scheduling or
acknowledgement handling, which follow [Planner telemetry](#planner-telemetry).

### Dropped-span counters

**OBS-R81.** The Worker Runtime report adds optional
`adapters["otlp-http@1"].droppedSpans` counters, separate from HTTP operation
failures. They count spans, not requests or attempts, under these local discard
reasons:

| Counter | Local discard reason |
| --- | --- |
| `queueOverflow` | A new span cannot fit either queue limit. |
| `encodingFailed` | Encoding fails or a whole span cannot fit a bounded request. |
| `nonRetryable` | Permanent HTTP/transport failure or invalid acknowledgement. |
| `retryExhausted` | A temporary failure uses the last allowed attempt. |
| `collectorRejected` | The collector explicitly rejects this many spans. |
| `deadlineExceeded` | The batch deadline expires or cannot accommodate Retry-After/backoff. |
| `cancelled` | The sender is cancelled, including by an enclosing flush deadline. |
| `shutdown` | Close discards an active batch or queued tail, or a span finishes after close. |

**OBS-R82.** Each local discard is counted once, including during close;
successful retries add no discarded spans.

**OBS-R83.** The counters saturate at the uint64 maximum
(LIM-OBS-14) and remain in the final Runtime report after adapter cleanup
(LIFE-OBS-3).

**OBS-R84.** The counters record local discard decisions only. A lost
acknowledgement can make remote acceptance unknown, and retries can produce
duplicates; the counters do not measure confirmed collector loss or spans lost
in a process crash (FAIL-OBS-5).

**OBS-R85.** The counters include no endpoint, response body,
credential or captured content.

### Export failure, flush and teardown

**OBS-R86.** An OTLP exporter delivery error, timeout or unavailable
backend increments bounded adapter error metrics but does not change the
StageResult, the StageTermination or release eligibility (FAIL-OBS-6).

**OBS-R87.** Finalization and abort perform one best-effort flush,
bounded by both the configured timeout and the remaining finalization or abort
deadline (FAIL-OBS-6). With telemetry enabled, entering allocation teardown
reserves `min(flushTimeoutSeconds, remainingStopSeconds / 2)` of the original
stop budget before HTTP, Worker, Toolset and execution cleanup can consume it.
Those producers receive the earlier deadline. Lease loss, reconciliation and
process shutdown use the same rule. No deadline or retry budget is restarted.
An exhausted deadline or failure to acquire lifecycle ownership does not
authorize extending mandatory process termination.

If producer cleanup fails, the allocation is fenced and the same bounded flush
is attempted before controlled process exit. It exports completed spans while
producer handles remain owned; it does not close adapters underneath producers
whose stop is unconfirmed. Repeated caller cancellation cannot skip this attempt.
Export failure never replaces the original cleanup failure or allows slot reuse.
A safe diagnostic records the cleanup phase and bounded exception class name,
never the exception message, credentials, endpoint or response body.

```mermaid
flowchart TD
    S["Stop: one absolute deadline"] --> R["Reserve final flush time"]
    R --> P["Stop HTTP, Worker, Toolset and execution producers"]
    P -- confirmed --> F["Existing bounded final flush"]
    F --> C["Close adapters, then report and release"]
    P -- "failure, timeout or cancellation" --> X["Fence allocation and log safe cleanup phase"]
    X --> E["Existing bounded flush with original deadline; retain producer handles"]
    E --> Q["Exit 70; slot cannot be reused"]
```

**OBS-R88.** For a Worker the flush joins the active sender and drains
the remaining spans in the same bounded batches; the deadline bounds the entire
drain, not each batch independently.

**OBS-R89.** Closing cancels and joins any sender before erasing its queue and
transport credentials (LIFE-OBS-3).

**OBS-R90.** The Runtime then returns the Contractor
`ExecutionReport` regardless of export success when resource cleanup is confirmed.
An unconfirmed cleanup instead forces process exit after the bounded export
attempt and cannot manufacture a successful final report.

**OBS-R91.** Release erases all credential values, closes clients and
removes temporary proxy trust material before the slot can become idle
(LIFE-OBS-3).

### Planner telemetry

**OBS-R92.** Planner telemetry uses the same pinned Run
configuration as Worker telemetry but is constructed in the Server, not in a
Runtime Agent. Its exporter follows the same no-content default, bounded flush
and best-effort delivery rules.

**OBS-R93.** Planner telemetry never receives Agent labels or Worker
proxy settings.

**OBS-R94.** The Server owns a versioned PlannerTelemetryAdapter
registry, and publication rejects a `planner.telemetry.adapter` unknown to that
registry. This Server capability is deployment code, not a Runtime Agent
placement fact.

**OBS-R95.** One selected Planner `otlp-http@1` adapter is
invocation-local. It does not install a global OpenTelemetry provider, read
ambient proxy variables, follow redirects or retry.

**OBS-R96.** The Passthrough, Streamline and Router Planners use
a closed instrumentation vocabulary for invocation or session, model, Worker
dispatch, subtask transition and finish spans. Content never enters the closed
metadata vocabulary; explicit capture uses the separate bounded input and output
fields of OBS-R59 and OBS-R60.

**OBS-R97.** Planner session IDs are attached only after durable
session creation.

**OBS-R98.** Every span of one invocation shares safe Run,
Stage and Planner correlation and the exact pinned RuntimeConfig resource
correlation; Agent-label provenance is absent.

**OBS-R99.** After the Planner returns, the Server makes at most
one OTLP request (LIM-OBS-15). Its deadline is the minimum of the configured
flush timeout, the remaining Stage deadline and the finalization bound
(FAIL-OBS-7).

**OBS-R100.** Adapter creation, credential decryption, encoding,
delivery, queue overflow and flush failures are supplementary. They are logged
with a closed safe code and, when a Planner report identity exists, recorded as
the `telemetry.export` tool outcome in the durable `ExecutionReport`. They never
change the candidate, the retry or escalation decision, the StageTermination or
allocation release (FAIL-OBS-7).

**OBS-R101.** When no Planner telemetry is selected, no exporter, client
or request is created.

### Metadata labels

**OBS-R102.** The Server makes the immutable metadata-label map of a
WorkflowRun ([Runs](../execution/runs.md)) available to Planner and Worker
telemetry, so that an operator can correlate model traces with an eval
invocation and leg. Planner and Worker trace exporters may attach the bounded
map without exposing it to either model.

**OBS-R103.** Planner spans obtain the labels from the authoritative
WorkflowRun row.

**OBS-R104.** Scheduler copies the same bounded map into each
`AllocationSpec` as `runMetadataLabels`.

**OBS-R105.** The Runtime validates the wire bounds and exposes labels
only to its telemetry adapter, as per-execution span attributes. They are not
process-wide resource attributes, because one Runtime process serves different
Runs over time.

**OBS-R106.** The Runtime never injects metadata labels into
prompts, ADK State, tool arguments, WorkspaceFS, Skills or model-visible
results.

**OBS-R107.** Telemetry uses a namespaced attribute projection such
as `contractor.run.label.eval.id`. An exporter-specific adapter may additionally
map the conventional eval labels to native trace tags when that mapping loses no
values. The exact original map remains authoritative in the Server.

**OBS-R108.** Metadata labels are not metric dimensions.

Rationale: keys such as `eval.id`, `eval.case` and `run_id` are high-cardinality
and would create unsafe time-series growth. The existing Run and Stage
correlation IDs and trace attributes are sufficient for detailed drill-down.

## Interfaces

- `X-Request-ID` on every public and private HTTP response (OBS-R1), and
  `requestId` in REST error objects of
  [`contractor-public-v1.yaml`](../../../api/openapi/contractor-public-v1.yaml)
  (OBS-R5); the error envelope itself is specified in
  [Public API](../api/public-api.md).
- Execution report shapes: [Report shapes](#report-shapes).
- OTLP/HTTP protobuf trace export to the configured complete traces endpoint
  (OBS-R45); accepted acknowledgements (OBS-R77); the content attributes
  `gen_ai.*`, `contractor.input` and `contractor.output` (OBS-R59); the label attribute
  namespace `contractor.run.label.*` (OBS-R107).
- Configuration consumed here and specified in
  [Runtime configuration](../configuration/runtime-config.md):
  `worker.telemetry` and `planner.telemetry` with `captureContent` and
  `flushTimeoutSeconds`; `worker.telemetry.export` (`batchSizeBytes`,
  `maxAttempts`, `maxPendingSpans`, `maxPendingBytes`;
  [RCF-R99](../configuration/runtime-config.md)) and its `retry` object
  (`initialBackoffMilliseconds`, `maxBackoffMilliseconds`;
  [RCF-R103](../configuration/runtime-config.md));
  `requestTimeoutSeconds`.

## Limits, failures and errors

Limits ([registry](limits.md)):

- LIM-OBS-1 — one `X-Request-ID` per HTTP response (OBS-R1).
- LIM-OBS-2 — at most one exhausted dimension in `worker_budget`
  (OBS-R22).
- LIM-OBS-3 — 16 UTF-8 bytes, the secret length from which a secret matches
  anywhere in the text (OBS-R31).
- LIM-OBS-4 — at most 1,000 tool-call records per participant
  (OBS-R40).
- LIM-OBS-5 — at most 100 errors per participant (OBS-R40).
- LIM-OBS-6 — at most 4,096 UTF-8/JSON bytes per argument summary or error
  message (OBS-R40, OBS-R27).
- LIM-OBS-7 — at most 1 MiB per final allocation report (OBS-R40).
- LIM-OBS-8 — durable telemetry retention of 30 days (OBS-R43).
- LIM-OBS-9 — at most 64 attributes per span (OBS-R49).
- LIM-OBS-10 — metadata strings of at most 256 UTF-8 bytes (OBS-R49).
- LIM-OBS-11 — at most 2,048 span records per exporter queue (OBS-R51).
- LIM-OBS-12 — at most 64 MiB of encoded pending data per exporter queue
  (OBS-R51, OBS-R61).
- LIM-OBS-13 — at most 256 KiB of UTF-8 per captured content field
  (OBS-R60).
- LIM-OBS-14 — dropped-span counters saturate at the uint64 maximum
  (OBS-R83).
- LIM-OBS-15 — at most one OTLP request after a Planner returns
  (OBS-R99).
- LIM-OBS-16 — each tool argument number zero or with a leading-digit and
  written decimal exponent between -7 and 19 (OBS-R42, OBS-R110).
- LIM-OBS-17 — each integer of an execution report's metrics between 0 and
  2^53-1 (OBS-R111).
- LIM-OBS-22 — received and incrementally decoded Worker acknowledgement bytes,
  including each decoding layer's overflow probe (OBS-R77).
- The Worker export settings and their ranges belong to
  [Runtime configuration](../configuration/runtime-config.md); the
  `AttemptDiagnostics` bounds to [Runs](../execution/runs.md)
  ([LIM-RUN-8](limits.md), [LIM-RUN-9](limits.md)).

Failure outcomes ([failure model](failure-model.md)):

- FAIL-OBS-1 — delivery and collection of execution reports
  (OBS-R9, OBS-R36, OBS-R37,
  OBS-R38, OBS-R39, OBS-R111).
- FAIL-OBS-2 — an LLM call rejected by a Gateway quota (OBS-R17).
- FAIL-OBS-3 — a started tool call that is cancelled or fails (OBS-R23,
  OBS-R24, OBS-R32).
- FAIL-OBS-4 — appending a span to a full exporter queue (OBS-R109,
  OBS-R66).
- FAIL-OBS-5 — Worker OTLP batch delivery (OBS-R72,
  OBS-R74, OBS-R76, OBS-R77, OBS-R78,
  OBS-R84).
- FAIL-OBS-6 — the final telemetry flush at finalization or abort
  (OBS-R86, OBS-R87).
- FAIL-OBS-7 — Planner telemetry export by the Server
  (OBS-R99, OBS-R100).

Entity lifecycle ([lifecycle](lifecycle.md)):

- LIFE-OBS-1 — full tool results in execution telemetry (OBS-R25).
- LIFE-OBS-2 — durable execution telemetry (OBS-R43).
- LIFE-OBS-3 — the Worker exporter's queue, sender and transport credentials
  (OBS-R52, OBS-R89, OBS-R91, OBS-R83).

Error codes ([registry](errors.md)): `tool_call_cancelled` (OBS-R24)
and `tool_call_failed` (OBS-R32).

## Security

- The Server's private Control Plane and Artifact APIs propagate an incoming
  request ID only from a peer with a verified Runtime Agent certificate
  (OBS-R4); server-side diagnostics exclude paths, bodies, provider messages
  and secrets (OBS-R6).
- Report identities and attribution come from the trusted Server envelope, never
  from a Runtime report (OBS-R8, OBS-R12,
  OBS-R13).
- Tool arguments are redacted before leaving the Worker and again at the
  Server persistence boundary, known secrets are removed from tool arguments
  and errors, and one secret-matching policy covers metrics, allocation checks
  and Worker results (OBS-R27 to OBS-R32).
- Exported telemetry is content-free unless content capture explicitly trusts
  the sink, and sensitive-output tools force metadata-only instrumentation
  (OBS-R53, OBS-R54, OBS-R56).
- Metadata labels never reach a model (OBS-R106).
- Trust boundaries, mTLS identity and the handling of RuntimeSettings secrets
  are in [Security](security.md).

## Observability

This whole document is the shared observability model.

#### Owner-visible retained Runtime diagnostics

**OBS-R113.** The authenticated Run owner can read one exact Stage attempt's
Runtime diagnostics through `GET /v1/runs/{runId}/attempts/{stageExecutionId}/runtime-diagnostics`.
The reader authorizes the owner and Stage-to-Run association in the same SQL
statement before projecting data. Retained `stage_metrics` is authoritative,
including Eval-held metrics after source report rows expire. It reads Runtime
participant projections only, never full Worker reports or tool arguments.

**OBS-R114.** A page defaults to five and permits at most ten logical Workers,
128 KiB compact UTF-8 JSON, eleven participant selections including lookahead,
and 2 MiB of returned Runtime-only JSON before Go decoding (LIM-OBS-18).
Participants use deterministic bytewise name order. A signed cursor binds the
owner, Run, Stage, last name and retained metric generation. Generation hashes
the row update time and MVCC tuple generation, so updating the same timestamp
still invalidates a cursor. A changed generation returns 409 conflict and
requires a fresh first page; reads create no snapshot rows or retention holds.
The SQL extracts the containing Runtime map and enumerates its keys once per
page. Subsequent materialized projections carry only selected participant
values and bounded documents, rather than a copy of that map per participant.

**OBS-R115.** Public adapter counters are canonical unsigned decimal strings
bounded by uint64; optional Runtime milliseconds use nonnegative int64 decimal
strings (LIM-OBS-19). Browser clients preserve them without passing through a
JavaScript number. The projection preserves absent flush/duration facts, exact
adapter refs, closed error codes and optional local dropped-span counters.
Stop reasons are URL-redacted and limited to 4096 UTF-8 bytes with explicit
truncation. A missing live report is pending; missing terminal telemetry is
unavailable (it may have expired). A malformed participant remains unavailable
without hiding healthy siblings. Reports marked incomplete remain partial.
These supplementary facts never change execution results, retries or Eval
usage. The UI reads on disclosure open, observed Stage state change and
explicit refresh, backs off retryable failures and does not poll settled data.

#### Owner-visible retained tool-call diagnostics

**OBS-R116.** The authenticated Run owner can read retained call detail through
`GET /v1/runs/{runId}/attempts/{stageExecutionId}/tool-calls`. The same SQL
statement authorizes the owner and exact Stage-to-Run association before
selecting only `planner`, or `worker` with a required logical Agent name in the
shared name grammar. A Planner selector forbids a logical Agent name. Retained
StageMetrics is authoritative, including Eval-held metrics after source report
cleanup; no new report, retention hold or call ledger is created. Selected
reports are decoded strictly under stored-report rules (OBS-R110); malformed
selected data is unavailable and does not hide a healthy other participant.

**OBS-R117.** One request decodes at most one selected participant projection,
limited in PostgreSQL to 8 MiB of JSON text before it leaves the database. A
larger projection is unavailable with `report_oversized`. Pages default to 25
records and permit at most 100 and 128 KiB of compact UTF-8 response JSON,
including string escaping (LIM-OBS-20). Retained source order is preserved. A
byte-limited page stops before its first nonfitting row; every bounded row fits
an empty page. Opaque signed cursors bind the owner, Run, Stage, participant,
logical name, retained metric generation and record offset. Each scope member
is framed unambiguously, including resource IDs containing colons. A changed
generation requires a fresh first page with 409 conflict. Reads do no per-tool
query or all-participant report scan and stop when the request is cancelled.
The SQL selects a Worker directly from an object-valued Worker map and
serializes the selected report once for both its byte guard and returned text.

**OBS-R118.** Each public row contains an opaque 64-character hexadecimal
`callReference` derived from the selected report generation and record ordinal,
never a source CallID or report identity. It contains the Runtime vocabulary
tool name, succeeded/failed outcome, nullable nonnegative int64 decimal-string
`durationMs` and `resultSizeBytes`, and the sanitized argument/error summary.
The public boundary reuses the common persistence sanitizer and removes
URL-shaped text recursively, including argument keys, while preserving
`json.Number` values. `argumentsJson` is canonical JSON **text**, at most 4096
UTF-8 bytes under OBS-R42, rendered without parsing through browser numbers.
Oversized argument text becomes an explicit valid JSON truncation summary.
Names outside the public 128-byte vocabulary bound become an unavailable-name
marker with `toolTruncated`; error messages are bounded to 4096 UTF-8 bytes
with `messageTruncated` (LIM-OBS-21). Error codes follow OBS-R32's existing name
grammar and fallback. Arguments preserve `argumentsTruncated` and additionally
mark public redaction or truncation. No raw results, model/provider content,
physical identities, URLs, credentials or invented call timestamps are added.

**OBS-R119.** This surface is retained call detail, not a full event history.
Report completeness, report-level detail truncation, argument truncation, an
empty retained call list, a pending live report, unavailable terminal telemetry
and read failure are distinct. Availability follows OBS-R115 and failures are
FAIL-OBS-9. The UI selects one participant, expands arguments on demand and
loads later pages explicitly. It refreshes on disclosure open, that Stage's
observed state change or explicit action, uses bounded retry backoff and
preserves loaded facts visibly on failed refresh. Other Run updates, projection
remounts and settled objects do not cause perpetual polling. Diagnostic reads
never change execution results, aggregate counters or Eval usage.

## Telemetry rules owned by other documents

| Topic | Owner |
| --- | --- |
| StageMetrics is a separate StageExecution field populated while finalizing or aborting and never rewrites the StageResult or StageTermination | [SCH-R51](../execution/scheduler.md) |
| `AttemptDiagnostics` in owner Run status: content, ordering and bounds | [RUN-R91](../execution/runs.md), [RUN-R92](../execution/runs.md) |
| Metadata labels as telemetry-visible safe metadata, the UI warning against secrets, and labels never used as authorization, scheduling or Run success signals | [RUN-R77](../execution/runs.md) |
| `otlp-http@1` as the initial backend-neutral telemetry adapter | [RCF-R98](../configuration/runtime-config.md) |
| `WorkerInstrumentationPlugin` and its reducers | [WRK-R80](../execution/worker.md) |
| The `ExecutionMetrics` summarizer aggregate | [WRK-R155](../execution/worker.md) |
| Durable Planner facts from projection tools | [Planners](../execution/planners.md) |
| Memory adapter diagnostics and memory content under content capture | [Memory](../tools/memory.md) |
| The retained Skill projection and disclosed skill text under content capture | [SKL-R113](../tools/skills.md) to [SKL-R115](../tools/skills.md) |
| Workspace metric content | [WSP-R176](../tools/workspace.md), [WSP-R177](../tools/workspace.md) |
| HTTP and Caido metric content and the durable metrics planes | [HTTP and Caido tools](../tools/http-caido.md) |
| Code-analysis and taint-annotation metric content | [CAN-R141](../tools/code-analysis.md) to [CAN-R145](../tools/code-analysis.md) |
| Podman sandbox metrics and metadata-only instrumentation of execution-enabled Workers | [Code execution](../tools/code-execution.md) |
| Performance collection, `RuntimeReport.resources`, Server and database performance history, HTTP listener dimensions; performance telemetry never affects placement, outcomes, leases, fencing or release | [OPS-R69](../operations/operations.md), [OPS-R70](../operations/operations.md), [OPS-R99](../operations/operations.md), [OPS-R100](../operations/operations.md) |
| Closing allocation adapters after the flush and keeping them closed at release | [RTA-R58](../execution/runtime-agent.md) |
| Eval purge progress records: pass count, rows, bytes and age of each experiment and Project purge | [EVM-R243](../evals/managed-evals.md) |

## Invariants

These properties follow from the cited requirements, which are authoritative.

1. An unauthenticated caller cannot choose the request ID that the Server's
   private Control Plane or Artifact API logs (OBS-R4); test
   `TestPrivateArtifactTrustsIncomingRequestIDOnlyAfterMTLS` in
   [`internal/httpapi/privateartifacts/handler_test.go`](../../../internal/httpapi/privateartifacts/handler_test.go).
2. Telemetry never decides a semantic outcome: report collection, adapter
   metrics and trace export never change a valid StageResult, a durable
   StageTermination, a Planner candidate, a retry or escalation decision or
   allocation release (OBS-R36, OBS-R38, OBS-R39,
   OBS-R86, OBS-R100).
3. Every started tool call appears exactly once in its report, and a call
   pending when the Worker closes is recorded as `tool_call_cancelled`
   (OBS-R23, OBS-R24); test
   `test_plugin_close_records_pending_tool_calls_as_cancelled` in
   [`runtime/tests/test_instrumentation.py`](../../../runtime/tests/test_instrumentation.py).
4. Exported spans contain no prompts, responses, artifact content or tool
   arguments and results unless content capture is enabled for that telemetry
   and, for a Worker, the Worker selected no sensitive-output tool (OBS-R53,
   OBS-R54, OBS-R56); tests
   `test_otlp_protobuf_is_bounded_content_free_and_header_scoped` in
   [`runtime/tests/test_otlp_adapter.py`](../../../runtime/tests/test_otlp_adapter.py)
   and `test_adk_worker_suppresses_captured_content_for_sensitive_output_tools`
   in [`runtime/tests/test_adk_runtime.py`](../../../runtime/tests/test_adk_runtime.py).
5. A permanent export failure is never retried, and no retry extends a batch
   deadline (OBS-R74, OBS-R70, OBS-R76); tests
   `test_permanent_http_failures_never_retry` and
   `test_retry_after_cannot_extend_deadline` in
   [`runtime/tests/test_otlp_retries.py`](../../../runtime/tests/test_otlp_retries.py).
6. Each local span discard is counted once, and the counters saturate
   (OBS-R82, OBS-R83); tests
   `test_overflow_and_shutdown_are_counted_once` and
   `test_drop_counters_saturate_and_reports_are_detached` in
   [`runtime/tests/test_otlp_retries.py`](../../../runtime/tests/test_otlp_retries.py).
7. Repeated report delivery is idempotent, persisted reports are redacted,
   every reader decodes a stored report whose rendering exceeds its byte
   bound, and durable telemetry expires in bounded batches (OBS-R9, OBS-R27,
   OBS-R43, OBS-R44, OBS-R110); tests
   `TestMetricsPostgresIdempotencyRedactionAggregationAndRetention`,
   `TestMetricsPostgresReplayComparesNumbersAsStored` and
   `TestMetricsPostgresReadsReportsTrimmedToTheByteBound` in
   [`internal/telemetry/postgres_integration_test.go`](../../../internal/telemetry/postgres_integration_test.go)
   and `TestRetentionDrainsFullBatchesBeforeLongWait` in
   [`internal/telemetry/retention_test.go`](../../../internal/telemetry/retention_test.go).
8. Metadata labels reach telemetry only as bounded attributes and never reach a
   model (OBS-R102, OBS-R106).
9. Given the same records, the Worker producer and the Server keep the same
   newest detail of an oversized report and measure the complete report at
   most three times per bound; they measure a record the Runtime built at the
   same size, and the Server's redaction leaves it unchanged unless it holds a
   secret the record contains or treats a `://` text as credential-bearing;
   both reject an argument number outside LIM-OBS-16 on receipt and write the
   same marker for one in a summary they build
   (OBS-R27, OBS-R41, OBS-R42, OBS-R110); tests
   `TestSharedReportTrimKeepsTheSharedSuffixAtEveryBoundary`,
   `TestSharedArgumentNumbersAgreeWithTheRuntime`,
   `TestPolicyMarksNumbersTheServerBuiltOutsideTheirRange` and
   `TestReportTrimMeasuresTheCompleteReportAtMostThreeTimes` in
   [`internal/telemetry/trim_test.go`](../../../internal/telemetry/trim_test.go),
   `TestToolArgumentsKeepTheRuntimeNumberLiterals` and
   `TestToolArgumentsRejectNumbersOutsideTheirRange` in
   [`internal/contracts/reporting/telemetry_test.go`](../../../internal/contracts/reporting/telemetry_test.go),
   and `test_shared_report_trim_keeps_the_shared_suffix_at_every_boundary`,
   `test_shared_argument_redaction_leaves_redacted_arguments_unchanged`,
   `test_shared_argument_numbers_agree_with_the_server` and
   `test_shared_report_trim_measures_the_complete_report_at_most_three_times`
   in [`runtime/tests/test_metrics.py`](../../../runtime/tests/test_metrics.py).

## Non-goals

- Incremental report delivery per A2A Task. It could reduce data loss from a
  hard Worker crash without changing the terminal report schema.
- Secret filtering of captured content and a separate retention policy for it;
  neither is a prerequisite for the trusted-sink mode (OBS-R54).
- Export of OpenTelemetry metrics or logs, a durable exporter spool and infinite
  retry (OBS-R45, OBS-R52).
- A required external observability service (OBS-R26).
- Metadata labels as metric dimensions (OBS-R108).
- Route-specific HTTP metric breakdowns, owned by
  [Operations](../operations/operations.md).
