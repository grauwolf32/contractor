# 00 — Workflow Scheduler and Planner

Status: **Working agreement**

Depends on: none

## Goal

Workflow defines the product-specific sequence or graph. Workflow Scheduler
executes that graph and owns Stage lifecycle. Planner owns decisions inside one
prepared Stage. Worker runtimes and their heavy dependencies remain outside
Server, Scheduler and Planner packages.

The execution model has no Server-side Task DAG in addition to the Workflow.
Planner may keep a private plan, but the durable outer unit is a
`StageExecution`: one attempt with one Planner invocation and one terminal
`StageResult`. Retry creates another StageExecution rather than reopening the
previous one. Lifecycle, sessions and recovery are owned by
[04](04-execution-lifecycle-and-metrics.md).

## Workflow, Scheduler and Stage

One Workflow invocation creates one `WorkflowRun` (`Run` below). A Workflow
definition declares:

- named input artifact slots, including required/optional status and accepted
  media types;
- named output artifact slots and their acceptance contract;
- Stages, dependencies and product transition/escalation rules.

Workflow definitions are YAML files. At Server startup `WorkflowCatalog` loads,
parses and validates them into internal DTOs; the baseline has no hot reload.
Creating a Run stores the complete validated Workflow snapshot used by that Run,
so later file edits cannot change its graph or contracts. A formal Workflow
version/ref/digest scheme is not required initially: the durable snapshot is the
execution authority.

The reusable definition owns these slot contracts; concrete input and output
bindings belong to one Run. A Stage declares:

- one exact Planner factory/version;
- one or more uniquely named Worker bindings;
- its Stage context and references to available Run artifacts;
- named result-artifact slots and their kinds/cardinalities required for
  success.

Conceptually:

```yaml
planner: passthrough@1
agents:
  - name: oas_builder
    template: oas_builder@2
  - name: reviewer
    template: oas_reviewer@1
    namespace: shared_oas
inputs: ...
expected_result: ...
```

In Workflow authoring, `template` is a versioned selector. Before execution it
is resolved to the exact digest-bearing `AgentTemplateRef` owned by
[01](01-agent-template.md). `name` is the stable logical name visible to Planner,
not a physical Runtime Agent or process identity. `namespace` defaults to
`name` and affects artifact tools only.

At Run creation, the caller maps Workflow input slots to artifacts in its
authenticated UserScope. Before selecting the first Stage, Workflow Scheduler
pins their exact versions and forks independent working bindings into the Run's
reserved `inputs` Namespace. Run output bindings live in reserved `outputs` and
are created from accepted Stage results. The one-store, scoped fork and
publication contract is owned by [03](03-artifact-plane.md).

Workflow Scheduler owns this sequence for every StageExecution:

1. resolve and verify every AgentTemplate declared by the selected Stage;
2. ask Control Plane to prepare all required Worker allocations;
3. construct one Planner through the selected `PlannerFactory`;
4. invoke the Planner once with the fixed prepared Worker set;
5. validate and durably record the candidate `StageResult` plus exact referenced
   artifact versions, entering `finalizing`;
6. ask Control Plane to drain the Workers and collect execution/runtime reports;
7. atomically persist the terminal result and apply any declared Workflow output
   mapping;
8. release every Stage allocation.

Planner is not invoked when any required allocation fails to become ready.
Already prepared allocations are released before Workflow Scheduler applies the
Workflow's declared Stage failure/retry policy.

After a Planner terminates, Workflow Scheduler interprets `StageResult` under
that policy and selects the next Stage, a retry or an explicit escalation. The
Planner cannot advance or rewrite the Workflow graph itself. Retry creates a new
StageExecution with an incremented attempt number, a new Planner and a new ADK
Session.

A Run can succeed only when all of its required output slots have frozen
`outputs/<slot name>` bindings. Run success does not implicitly publish or
overwrite a user-scoped artifact.

## Planner

Planner is an ADK agent, normally an `LlmAgent`, constructed for one prepared
Stage. The replaceable Contractor abstraction is `PlannerFactory`; the name
avoids collision with ADK's unrelated `google.adk.planners.BasePlanner`.

```python
class PlannerFactory(Protocol):
    def create(
        self,
        stage: StageSpec,
        workers: Mapping[str, WorkerHandle],
        context: StageContext,
    ) -> BaseAgent: ...
```

Exactly one root Planner ADK agent owns the Stage invocation. A factory converts
the name-to-handle map into `RemoteA2aAgent` subagents:

```python
sub_agents = [
    RemoteA2aAgent(
        name=name,
        agent_card=handle.agent_card,
        a2a_client_factory=a2a_client_factory(handle),
    )
    for name, handle in workers.items()
]
```

`RemoteA2aAgent` is the initial ADK adapter, not a Contractor domain contract.
It is experimental in the current pinned ADK dependency and requires ADK's
optional A2A package, so construction stays behind PlannerFactory and is covered
by a compatibility test. Allocation routing comes from the selected Agent
Card's A2A interface; request metadata is not used as a substitute for the A2A
1.0 `tenant` field.

The ADK tree stays inside Server memory. Invoking a remote subagent crosses the
process boundary through A2A. Planner may decompose, iterate and route among the
fixed names, but it cannot:

- add or replace an Agent binding;
- choose a physical Runtime Agent;
- reserve capacity or initialize a Runtime Agent's Worker instance;
- change an AgentTemplate or Namespace;
- access a Worker sandbox directly.

Cross-Stage state passes through explicit Workflow context, Stage results or
artifacts, never through an implicitly shared Planner instance.

### Completion semantics

Planner strategy determines when its invocation is semantically complete, while
Workflow Scheduler owns the durable StageExecution transition. Planner returns
a candidate StageResult and does not write RunStore directly.

`PassthroughPlanner` completes when its required remote Worker invocation
completes. An immediate A2A `Message` is already complete; a task-based response
waits until its A2A `Task` becomes terminal. A Streamline-style Planner completes
successfully only through an explicit `finish`; reaching a hard
token/step/deadline limit without a valid finish produces a failed candidate.
Scheduler independently validates the Stage result contract before accepting a
claimed success. See [04](04-execution-lifecycle-and-metrics.md).

A Planner may return a candidate only after every A2A invocation it started is
terminal. In particular, `finish` cannot strand a running Worker Task; the
Planner must first await it or request cancellation and observe a terminal
state. This makes the following Worker finalization a telemetry-and-shutdown
step rather than additional semantic execution.

### Passthrough baseline

`PassthroughPlanner` is the first concrete factory and the baseline integration
path. It requires one prepared Worker, sends the Stage input through its
`RemoteA2aAgent`, waits for the remote invocation to complete and maps the
response and artifact refs into `StageResult`.

```text
StageSpec + StageContext + WorkerHandle
  -> A2A SendMessage
  -> immediate Message or Worker A2A Task
  -> completed response / artifact refs
  -> StageResult
```

It knows neither which Runtime Agent hosts the Worker nor how that process
configured the selected runtime.
More capable Planners preserve the same outer Stage contract.

## StageResult

One Planner invocation produces one framework-neutral semantic result:

```python
class StageError(BaseModel):
    code: str
    message: str
    retryable: bool | None = None


class StageResult(BaseModel):
    outcome: StageOutcome
    artifacts: dict[str, ArtifactRef] = Field(default_factory=dict)
    content: StageContent | None = None
    error: StageError | None = None
```

`StageContent` is a bounded Contractor presentation DTO; an ADK adapter may map
to and from `google.genai.types.Content`, but the core result does not expose an
ADK type. Its exact text/structured-part shape remains an implementation
decision. `StageMetrics` and report completeness are stored separately on
StageExecution and are defined in [04](04-execution-lifecycle-and-metrics.md).

Planner produces a candidate. Scheduler validates and stores one normalized
`StageResult` while entering `finalizing`; final report collection may update
StageMetrics but never rewrites this semantic result.

`StageOutcome` is `succeeded`, `failed` or `cancelled`.

- success has no `error` and satisfies the Stage's required artifact contract;
- failure has an `error`;
- cancellation may include a cancellation error;
- failed/cancelled results may retain useful partial content or artifacts;
- presentation `content` is bounded and excludes thoughts/function protocol;
- large or durable output is an `ArtifactRef`, never inline content;
- result refs resolve in the current RunScope, never directly in UserScope;
- every `artifacts` key is a Stage-local result name declared by the Stage;
  Workflow output mappings select results by that name rather than by list
  position or by parsing an ArtifactRef;
- metrics are not part of StageResult and cannot change its outcome.

Exact AgentTemplate refs, WorkerRuntime refs, Runtime Agent process identities
and allocation IDs are recorded with the StageExecution for provenance; they
are not duplicated inside presentation content.

## Stage execution scope

One StageExecution has one root Planner instance, one Planner invocation, one
database-backed Planner Session, a fixed set of allocations and one terminal
StageResult. An allocation lives through Planner invocation and finalization and
may serve multiple sequential A2A Tasks while running. A Worker handles at most
one Task at a time; different prepared Workers run on different allocated
Runtime Agent processes and may execute concurrently.

Workflow Scheduler persists the candidate before stopping Workers, accepts the
terminal result or a defined failure/cancellation/interruption outcome, then
releases allocations. Planner-private plans and A2A Task IDs do not become
additional Workflow stages.
