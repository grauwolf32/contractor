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
- an attempt number within that Run and optional `previous_execution_id`;
- a reference to the WorkflowRun's exact validated snapshot and the selected
  Stage specification within it;
- exact AgentTemplate and WorkerRuntime refs plus Runtime Agent process identity;
- globally unique allocation IDs and the Planner session ID;
- lifecycle timestamps, the one normalized semantic result, separate
  StageMetrics and accepted exact artifact versions.

One StageExecution owns exactly one Planner instance, one Planner invocation,
one Planner ADK Session and at most one terminal StageResult. A Workflow retry
creates a new StageExecution with a new Planner and Session; it never reopens a
terminal execution.

Cross-attempt state is explicit. A retry may receive Workflow context, accepted
prior results and Run-scoped artifact refs selected by Workflow policy. It does
not inherit the previous Planner's ADK State or private subtask plan.

## Completion ownership

A Planner implementation owns the semantic condition that ends its invocation:

- `PassthroughPlanner` waits for its required remote Worker invocation to
  complete, accepting either an immediate A2A Message or a terminal A2A Task,
  and maps that outcome to a candidate StageResult;
- a Streamline-style Planner completes successfully only through its explicit
  `finish` operation;
- exhausting a hard token/step/deadline budget without a valid `finish`
  produces a failed candidate with a stable budget-exhaustion error;
- an expected Worker failure is mapped by the Planner strategy into a failed or
  cancelled candidate as appropriate.

Planner does not update StageExecution storage. Its `finish` operation ends the
Planner invocation and produces a candidate result. Workflow Scheduler validates
and normalizes that candidate, may replace an invalid claimed success with a failed
`result_contract_violation`, and owns the durable transition to a terminal
state. A caught Planner exception becomes a failed candidate with a stable
Planner error; loss of the active Server/Planner before any candidate is durable
becomes `interrupted` during recovery.

This separation keeps Planner strategies independent of RunStore and gives
cancellation, recovery and result acceptance one durable writer.

## Lifecycle

The minimum StageExecution lifecycle is:

```text
preparing -> running -> finalizing -> succeeded
                                 \-> failed
                                 \-> cancelled

preparing / running -------------> interrupted
```

- `preparing`: templates are resolved and required allocations are prepared;
- `running`: one Planner invocation may use the fixed Worker set;
- `finalizing`: Planner has stopped and its candidate result plus exact artifact
  versions are durable; allocations are draining and reports are collected;
- terminal states are immutable. Workflow policy may create another
  StageExecution for retry or escalation.

Preparation failure before Planner start still produces a durable execution
outcome. Temporary lack of capacity is not such a failure: it keeps the
StageExecution in `preparing` without reserving any slot. Once all required
slots have been atomically reserved, an initialization failure drains/releases the whole
batch and makes the StageExecution `interrupted` with a retryable infrastructure
error before Workflow policy selects the next action.

### Entering finalizing

Workflow Scheduler atomically records the following before asking any Worker to
stop:

- the candidate StageResult without assuming telemetry completeness;
- the exact committed versions/revisions referenced by that result;
- Planner session identity and available Planner report;
- a unique `finalization_id`, deadline and the allocation set to drain.

After this transition neither Planner nor Worker may change the semantic result
or its referenced artifact versions. Declared `outputs/<slot>` bindings are
created only when Scheduler accepts the terminal successful result.

StageMetrics is a separate StageExecution field populated while finalizing. It
is not embedded into or used to rewrite the stored StageResult.

## Sessions and State

### Planner

Planner runs in Server with a database-backed ADK SessionService. Its Session is
created for one StageExecution and its ID is recorded with that execution. The
database record supports inspection, statistics and audit; it is not a recovery
authority and Scheduler never resumes an interrupted Planner invocation from
its event history or private State.

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


class ExecutionMetrics(BaseModel):
    duration_ms: int | None = None
    model_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    tools: dict[str, ToolMetrics] = Field(default_factory=dict)


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


class RuntimeReport(BaseModel):
    complete: bool
    duration_ms: int | None = None
    stop_reason: str | None = None


class StageMetrics(BaseModel):
    planner: ExecutionReport | None = None
    workers: dict[AgentName, ExecutionReport] = Field(default_factory=dict)
    runtime: dict[AgentName, RuntimeReport] = Field(default_factory=dict)
```

The trusted Server envelope binds each report to Run, StageExecution, the
globally unique allocation ID and logical Agent name. A Worker-supplied payload
cannot select or override those identities. `report_id` makes repeated delivery
idempotent.

### Meaning and capture rules

- `complete` describes report completeness, not execution success.
- A known zero is `0`; an unobserved or unsupported value is `null`/absent.
- `metrics` contains only aggregate values. Tool arguments and errors are
  diagnostic records rather than counters.
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
  semantically valid StageResult into failure.
- Reports delivered after StageExecution is terminal may be retained as related
  telemetry but never mutate the frozen StageResult.
- External observability exporters are optional adapters over Contractor-owned
  reports; no LangChain/Langfuse-style service is required.

Exact byte/count limits and retention periods are deployment policy left to the
first implementation, but an unbounded argument, error or record list is not a
conforming implementation.

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
instance. The command is idempotent for `finalization_id`. Runtime Agent waits
only until the supplied deadline; if it cannot guarantee that the Worker stopped,
it exits its own process rather than reusing the slot. Control Plane then records
an incomplete report from the facts it can observe.

Runtime Agent keeps the resulting `AllocationFinalReport` with the allocation
until `release`. If the response is lost, repeating the same `finalization_id`
returns that cached report even though the Worker has already exited; a
different finalization ID for the same draining allocation is rejected.

The allocation remains reserved through `draining`/stopped state until
StageExecution becomes terminal. Only then does `release` remove routing,
grants, sandbox and lease and make the Runtime Agent slot available again.

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
  unique ID: `finalizing` resumes idempotent finalization, while terminal,
  interrupted or unknown allocations are drained and released;
- a terminal StageExecution is never rerun;
- `finalizing` with a durable candidate and pinned versions reissues the same
  idempotent finalization, then accepts the candidate even if reports remain
  incomplete;
- `preparing` or `running` without a candidate becomes `interrupted`; remaining
  allocations stop or expire and Workflow policy chooses retry, escalation or
  Run failure;
- a Runtime Agent process restart never reattaches its old Worker; an affected
  in-process Worker no longer exists, and an affected `preparing` or `running`
  StageExecution follows the same `interrupted` path;
- expiry of the 60-second Runtime Agent control lease follows that same path;
  the agent independently drains and terminates its Worker after 60 seconds
  without an acknowledged heartbeat;
- retry always creates a fresh StageExecution and Planner Session.

## Invariants

1. One StageExecution is one attempt, one Planner invocation and one Planner
   Session.
2. Planner owns completion semantics; Workflow Scheduler is the only durable
   StageExecution writer.
3. Candidate result and exact referenced artifact versions are durable before
   Worker drain begins.
4. Finalization cannot perform semantic work or artifact mutation.
5. Missing telemetry never invalidates an otherwise valid StageResult.
6. Planner Session persistence does not imply Planner resume.
7. Runtime Agent has no direct PostgreSQL or external telemetry credentials.
8. Allocations are released only after StageExecution is terminal.
