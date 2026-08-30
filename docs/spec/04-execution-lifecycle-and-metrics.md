# 04 — Stage execution lifecycle, sessions and metrics

Status: **Working agreement**

Depends on: [00](00-workflow-and-planner.md) through
[03](03-artifact-plane.md)

## Goal

This document owns the durable boundary around one Stage attempt: identity,
Planner and Worker sessions, finalization, recovery and execution reports.
Planner decides when its strategy has finished; Workflow Scheduler remains the
only component that commits durable `StageExecution` state.

## Stage and StageExecution identity

`Stage` is a reusable named node in a Workflow definition. `StageExecution` is
one attempt to execute that node in one WorkflowRun. There is no additional
`StageAttempt` entity.

Each StageExecution records at least:

- its `stage_execution_id`, `run_id` and stable Stage name;
- an attempt number for that stable Stage name within the Run and optional
  `previous_execution_id`;
- a reference to the WorkflowRun's exact validated snapshot and the selected
  Stage specification within it;
- exact AgentTemplate and WorkerRuntime refs plus Runtime Agent process
  identities for participants that were prepared;
- globally unique allocation IDs and, when Planner started, its invocation and
  session IDs;
- lifecycle timestamps, one terminal outcome, separate StageMetrics and any
  accepted exact artifact versions.

One StageExecution owns at most one Planner instance, one Planner invocation
and one durable Contractor Planner Session. An ADK-backed strategy also owns
one live in-memory ADK conversation bound to that Session; deterministic
`passthrough@1` does not. They do not exist until preparation succeeds and
Workflow Scheduler starts Planner. A Workflow retry creates a new
StageExecution and, if preparation succeeds, a new Planner and Session; it
never reopens a terminal execution.

Cross-attempt state is explicit. A retry may receive Workflow context, accepted
prior results and Run-scoped artifact refs selected by Workflow policy. The new
StageExecution resolves and pins its own StageContext snapshot, so it may see
bindings advanced by the previous attempt but never inherits that attempt's
pinned context implicitly. It also does not inherit the previous Planner's
framework-private state or subtask plan.

## Terminal outcome

`StageResult` and `StageTermination` represent different terminal facts:

- `StageResult` is the normalized semantic result of a Planner invocation. It
  produces terminal execution state `succeeded` or `failed` and may reference
  exact result artifacts;
- `StageTermination` is a durable Scheduler decision that execution must end
  without an accepted semantic result. It is recorded when execution enters
  `aborting`, produces terminal execution state `cancelled` or `interrupted`,
  and never publishes declared outputs.

```python
class StageTermination(BaseModel):
    outcome: Literal["cancelled", "interrupted"]
    code: str
    message: str
    retryable: bool
    phase: Literal["preparing", "running"]
    occurred_at: datetime
```

`code` is stable and machine-readable; `message` is diagnostic. `retryable` is
an input to Workflow policy, not an instruction to reopen or automatically
retry the same StageExecution.

The cardinalities depend on lifecycle state:

- before Planner starts, Planner invocation, Planner Session, StageResult and
  StageTermination may all be absent;
- `running` and normal `finalizing` require exactly one Planner invocation and
  one Planner Session;
- `aborting` requires exactly one StageTermination and no accepted StageResult.
  Planner invocation and Session may be absent when abort began during
  preparation;
- `succeeded` and `failed` require exactly one accepted StageResult and no
  StageTermination;
- `cancelled` and `interrupted` require exactly one StageTermination and no
  accepted StageResult. Planner invocation and Session are absent when
  termination happened during preparation and present when Planner had
  already started.

Consequently every terminal StageExecution contains exactly one of an accepted
`StageResult` or a `StageTermination`, never both. A rejected or superseded
Planner candidate may be retained for audit but is not an accepted
StageResult.

## Completion ownership

A Planner implementation owns the semantic condition that ends its invocation:

- `PassthroughPlanner` waits for its required remote Worker invocation to
  produce an immediate A2A Message, a terminal Task, or an interrupted Task
  state that the baseline maps to a stable failed candidate;
- a Streamline-style Planner completes successfully only through its explicit
  `finish` operation and produces a semantic failed candidate only through
  explicit `escalate`;
- exhausting a hard model-call/token/Worker-call/deadline budget without a
  valid terminal tool stops the strategy without a candidate; Scheduler records
  the stable budget error as a retryable, running-phase interrupted
  StageTermination through bounded `aborting`, without awaiting outstanding A2A
  Tasks;
- an expected Worker failure is mapped by the Planner strategy into a failed
  candidate with a stable error; this includes Runtime-enforced
  `worker_budget_exhausted`, which is retryable and carries complete bounded
  Worker budget metrics.

Planner does not update StageExecution storage. Its `finish` operation ends the
Planner invocation and produces a candidate result. Workflow Scheduler validates
and normalizes that candidate, may replace an invalid claimed success with a failed
`result_contract_violation`, and owns the durable transition to a terminal
state. An exception handled by the Planner strategy may become a failed
candidate with a stable Planner error. An exception escaping the Planner
invocation, or loss of the active Server/Planner before any candidate is
durable, produces an `interrupted` StageTermination.

This separation keeps Planner strategies independent of RunStore and gives
cancellation, recovery and result acceptance one durable writer.

## Lifecycle

The minimum StageExecution lifecycle is:

```text
preparing -> running -> finalizing -> succeeded
                                 \-> failed

preparing / running -> aborting -> cancelled
                              \-> interrupted
```

- `preparing`: the StageContext artifact snapshot is pinned, exact templates
  are loaded and required allocations are prepared;
- `running`: one Planner invocation may use the fixed Worker set;
- `finalizing`: Planner has stopped and its candidate result plus exact artifact
  versions are durable; allocations are draining and reports are collected;
- `aborting`: Scheduler has durably rejected semantic completion, stored the
  StageTermination and fenced further work; cancellation and drain are bounded
  by the recorded abort deadline;
- `succeeded` and `failed` contain an accepted StageResult; `cancelled` and
  `interrupted` contain a StageTermination;
- terminal states are immutable. Workflow policy may create another
  StageExecution for retry or escalation.

Preparation failure before Planner start still produces a durable execution
outcome. Temporary lack of capacity is not such a failure: it keeps the
StageExecution in `preparing` without reserving any slot. A missing required
Stage context artifact is detected before allocation and produces
`context_artifact_missing`, a non-retryable interrupted StageTermination. Once
all required slots have been atomically reserved, an initialization failure
drains/releases the whole batch and makes the StageExecution `interrupted` with
a retryable StageTermination whose code identifies the infrastructure error.
Workflow policy then selects the next action. No Planner or Planner Session is
created for either execution.

### Entering finalizing

Workflow Scheduler atomically records the following before asking any Worker to
stop:

- the candidate StageResult without assuming telemetry completeness;
- the exact committed versions/revisions referenced by that result;
- Planner session identity and available Planner report;
- a unique `finalization_id`, deadline and the allocation set to drain;
- a Server-side fence that makes Artifact API writes from those allocations
  fail from this transition onward.

Every candidate ArtifactRef must already contain the revision selected by
Planner. Scheduler verifies that `(RunScope, namespace, name, revision)` exists
and resolves to a retained immutable version, then pins that version in the same
transition. It never substitutes the current binding, even when that binding
has advanced since Planner selected the result.

After this transition neither Planner nor Worker may change the semantic result
or its referenced artifact versions. The private Artifact API rejects new
writes even if a stale Worker Task remains active. Declared `outputs/<slot>`
bindings are created only when Scheduler accepts the terminal successful result
and the WorkflowRun is still `running` in that same transaction. If Run
`cancelling` won first, StageResult remains valid for audit but cannot advance
the Workflow or create Run output bindings.

StageMetrics is a separate StageExecution field populated while finalizing or
aborting. It is not embedded into or used to rewrite the stored StageResult or
StageTermination.

### Entering aborting

`aborting` is the bounded path for external cancellation, loss of a required
participant, an unrecoverable Planner invocation error, or another
Scheduler-owned interruption. Workflow Scheduler atomically records:

- the StageTermination, including its origin phase and stable reason;
- a unique `abort_id`, abort deadline and every allocation that must stop;
- the write fence for those allocations and the fact that no Planner candidate
  may now be accepted.

The transition to `finalizing` and the transition to `aborting` compete through
one durable compare-and-set from `running`; only one may win. A Planner
candidate arriving after `aborting` starts may be retained for diagnostics but
cannot become the StageResult. Once `finalizing` is durable, the Stage result
has won this StageExecution race and a later cancellation cannot replace it
with a StageTermination.

Scheduler then cancels the in-process Planner invocation and requests A2A Task
cancellation best-effort. Neither a successful `CancelTask` response nor an
observed terminal A2A Task is required for progress. Control Plane drains the
allocations through the authoritative private control protocol. At the abort
deadline, any allocation that has not confirmed shutdown is marked lost and
fenced, its reports are recorded as incomplete, and Scheduler commits the
terminal state named by StageTermination.

An execution aborted during `preparing` follows the same contract. If it has no
Planner and no reserved allocations, `aborting` can complete immediately after
the durable transition.

## Sessions and State

### Planner

Every Planner runs in Server with one database-backed Contractor session. Its
identity is created for one StageExecution and recorded with that execution.
`passthrough@1` writes bounded request/completion events directly.
`streamline@1` additionally supplies Google ADK with a SessionService adapter:
live conversation contents remain in memory, while every non-partial ADK event
is reduced before PostgreSQL append to author, allowed function names, action
flags and aggregate token counts. Prompt/model text, tool arguments/results,
provider bodies and unknown provider-controlled function names are not durable.

The database record supports inspection, statistics, audit and completed-result
recovery. A completed Planner session returns its recorded candidate/failure
without invoking ADK, Gateway or Worker again. It is not authority for resuming
a partially completed model invocation; a still-running session after process
loss yields a stable retryable invocation-in-progress error and Workflow policy
decides whether a fresh StageExecution should retry.

### Worker

An ADK-based Worker uses an in-memory SessionService inside the Runtime Agent
process; that SessionService and its State are created and destroyed with the
allocation. A non-ADK Worker runtime provides equivalent execution/report
behavior without exposing an ADK contract. Runtime Agent receives no database
credentials, and its A2A Task/session mapping remains an implementation detail
within the allocation.

An ADK Worker reserves a bounded `metrics` section in its in-memory State. Its
callbacks, guardrails or agent code accumulate counters, redacted/truncated
tool-call arguments and error details there across all A2A Tasks handled by the
allocation. A dedicated collector/plugin may manage that section. During
finalization the Worker converts it to the framework-neutral `ExecutionReport`;
ADK State itself is only an accumulator implementation, never the Contractor
wire or persistence schema.

RuntimeSettings secrets supplied by Control Plane are held outside ADK Session,
State, events and model-visible instruction/context. They configure clients
such as the LLM Gateway adapter and are never a telemetry source.

## Execution reports

Planner and the in-process Worker runtime use the same framework-neutral report
shape. Allocation lifecycle facts observed by the surrounding Runtime Agent
remain a separate report because their source and completeness differ.

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


class RuntimeReport(BaseModel):
    complete: bool
    duration_ms: int | None = None
    stop_reason: str | None = None


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

The trusted Server envelope binds each report to Run, StageExecution, the
globally unique allocation ID and logical Agent name. A Worker-supplied payload
cannot select or override those identities. `report_id` makes repeated delivery
idempotent.

The same trusted envelope derives model attribution from the Run's immutable
ResolvedExecutionConfig: effective ModelPolicyRef, LLMGatewayConfigRef and the
non-secret LLMCredentialRef for the Planner or logical Worker. These
refs are never accepted from a Runtime report. They let Operations aggregate
calls/tokens by policy, route and credential without persisting or exposing the
secret value.

### Meaning and capture rules

- `complete` describes report completeness, not execution success.
- A known zero is `0`; an unobserved or unsupported value is `null`/absent.
- `metrics` contains only aggregate values. Tool arguments and errors are
  diagnostic records rather than counters.
- `worker_budget` is present for an ADK Worker invocation. It records the exact
  configured ceilings, operations actually started, the sum of available
  provider token usage, missing-usage response count, and at most one exhausted
  dimension; `max_tool_calls` is absent only for a compatible tool-free policy.
  It contains no prompt, response, or tool payload.
- Tool arguments are recursively redacted before leaving Worker and again at
  the Server persistence boundary. They are bounded and carry an explicit
  truncation marker.
- RuntimeSettings tokens and other known deployment secrets are always removed,
  even if a tool argument or error accidentally contains them.
- Full tool results are not retained by default. Durable or large results belong
  in ArtifactStore; metrics may retain their size/outcome without their body.
- For every expected participant that started but produced no report, Scheduler
  records an incomplete placeholder with unknown fields. Absence from a map
  means the participant was not started or was not applicable.
- Report collection is best-effort and cannot raise into Planner or change a
  semantically valid StageResult or a durable StageTermination.
- Reports delivered after StageExecution is terminal may be retained as related
  telemetry but never mutate the frozen StageResult or StageTermination.
- External observability exporters are optional adapters over Contractor-owned
  reports; no LangChain/Langfuse-style service is required.

The first-slice bounded policy is fixed: at most 1,000 tool-call records and 100
errors per participant, at most 4,096 UTF-8/JSON bytes per argument summary or
error message, and at most 1 MiB per final allocation report. Overflow removes
the oldest detail, preserves aggregate counters and sets `truncated`. Durable
telemetry expires after 30 days; cleanup is batch-bounded and may delete only
telemetry belonging to a terminal StageExecution. These values may become
deployment policy later without changing the report shape.

The public Run status exposes only aggregate counts and report completeness.
Tool arguments, report/session/allocation identities and error messages remain
on the authenticated Server side and are never part of that projection.

## Worker drain and finalization

Finalization is part of the private control/lifecycle protocol, not an A2A Task:

```text
Workflow Scheduler
  -> Control Plane: finalize(finalization_id, allocations, deadline)
  -> Runtime Agent: mark allocation draining and reject new A2A Tasks
  -> same process: flush one allocation-wide ExecutionReport and destroy Worker instance
  -> Runtime Agent: return ExecutionReport plus RuntimeReport
  -> Workflow Scheduler: accept terminal StageResult
  -> Control Plane: release route, sandbox, lease and slot
```

Worker finalization performs no model/tool calls and cannot mutate artifacts. It
only serializes already accumulated data and destroys the in-process runtime
instance. An active A2A Task is asked to cancel locally but finalization does not
wait for the A2A protocol to report a terminal state. The command is idempotent
for `finalization_id`. Runtime Agent waits only until the supplied deadline; if
it cannot guarantee that the Worker stopped, it exits its own process rather
than reusing the slot. Control Plane then records an incomplete report from the
facts it can observe, and Scheduler accepts the already durable candidate.

Runtime Agent keeps the resulting `AllocationFinalReport` with the allocation
until `release`. If the response is lost, repeating the same `finalization_id`
returns that cached report even though the Worker has already exited; a
different finalization ID for the same draining allocation is rejected.

### Abort drain

Abort uses a separate idempotent private control command:

```text
Workflow Scheduler
  -> RunStore: record aborting, StageTermination, abort_id and deadline
  -> Control Plane: abort(abort_id, allocations, deadline)
  -> Runtime Agent: reject new A2A Tasks and request local Task cancellation
  -> same process: flush bounded reports and destroy Worker instance
  -> Runtime Agent: return ExecutionReport plus RuntimeReport
  -> Workflow Scheduler: commit the cancelled/interrupted terminal state
  -> Control Plane: release route, sandbox, lease and slot
```

The command performs no semantic work and does not depend on A2A `CancelTask`
delivery. It is idempotent for `abort_id`; retries return the cached report when
available. Scheduler waits no longer than the durable abort deadline. At that
deadline an unconfirmed allocation is marked lost, its reports are incomplete,
and the StageExecution becomes terminal deterministically.

A lost allocation is not offered for placement. If its Runtime Agent later
reconnects, reconciliation reissues the same abort before the slot may become
idle. If it remains disconnected, the agent's control-lease watchdog
ultimately destroys the Worker or exits the process. Late reports may enrich
related telemetry but never change the StageTermination.

The allocation remains reserved through `draining`/stopped state until
StageExecution becomes terminal. Only then does two-phase release begin. The
private Runtime request idempotently removes the SandboxProfile workspace,
tools and secrets but leaves the old allocation identity `fenced`; after its
successful response, Control Plane removes its route, grant and lease. A
subsequent heartbeat `release` action confirms that authoritative edge and lets
Runtime Agent clear its retained identity/report and become `idle`. A lost
private response therefore leaves both sides retry-safe, while cleanup failure
keeps that Runtime Agent fenced without rewriting the terminal Stage outcome.
Control Plane does not offer the slot until a confirmed heartbeat reports the
matching idle state.

Incremental per-A2A-Task report delivery is not required for the first slice.
It may later reduce data loss from a hard Worker crash without changing the
terminal report schema.

## Recovery

WorkflowRun recovery uses durable Scheduler state, not live ADK sessions:

- the validated Workflow snapshot, exact input forks, completed
  StageExecutions, accepted results and exact artifact versions survive Server
  restart;
- the in-memory Runtime Agent Registry does not survive Server restart and is
  not a recovery authority; still-running agent processes register again;
- every allocation reported after re-registration is reconciled by its globally
  unique ID: `finalizing` resumes idempotent finalization, `aborting` resumes
  the idempotent abort, and terminal or unknown allocations are drained and
  released;
- a terminal StageExecution is never rerun;
- `finalizing` with a durable candidate and pinned versions reissues the same
  idempotent finalization, then accepts the candidate even if reports remain
  incomplete;
- `preparing` or `running` without a candidate enters `aborting`; Scheduler
  records a retryable interrupted StageTermination, remaining allocations stop
  or become lost at the abort deadline, and Workflow policy then chooses retry,
  escalation or Run failure;
- a Runtime Agent process restart never reattaches its old Worker; an affected
  in-process Worker no longer exists, and an affected `preparing` or `running`
  StageExecution follows the same `aborting -> interrupted` path;
- expiry of either side's 60-second confirmed Runtime Agent control lease
  follows that same path; the agent independently drains and terminates its
  Worker after 60 seconds without a new acknowledged heartbeat, then remains
  fenced with the allocation ID until release is acknowledged;
- retry always creates a fresh StageExecution and, if preparation succeeds, a
  fresh Planner Session.

## Invariants

1. One StageExecution is one attempt and owns at most one Planner invocation
   and one Planner Session; both are required only after Planner starts.
2. Planner owns completion semantics; Workflow Scheduler is the only durable
   StageExecution writer.
3. Every terminal StageExecution contains exactly one accepted StageResult or
   one StageTermination. A StageResult belongs to Planner completion; a
   StageTermination belongs to Scheduler-controlled interruption or
   cancellation.
4. `finalizing` and `aborting` are mutually exclusive compare-and-set outcomes;
   a late Planner candidate cannot replace a durable StageTermination.
5. Candidate result and exact referenced artifact versions are durable before
   normal Worker drain begins; StageTermination and its abort deadline are
   durable before abort drain begins.
6. Finalization and abort cannot perform semantic work or artifact mutation.
7. Neither A2A cancellation delivery nor terminal Task observation can extend
   finalization or abort past its durable deadline.
8. Missing telemetry never invalidates an otherwise valid StageResult or
   StageTermination.
9. Planner Session persistence does not imply Planner resume.
10. Runtime Agent has no direct PostgreSQL or external telemetry credentials.
11. Allocations are released only after StageExecution is terminal.
