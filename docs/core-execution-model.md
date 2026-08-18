# Contractor v2 core execution model

Status: **Working agreement**

This document records the architecture agreed during design discussion. It is
the current source of truth for the core Workflow/Planner/Worker separation.
The detailed implementation specifications and LikeC4 model still describe an
earlier candidate in several places and must be reconciled after the remaining
open decisions are resolved.

## Primary goal

Contractor v2 reduces the coupling of v1 by separating Workflow and planning
logic from Worker implementations and their often-heavy dependencies.

The replaceable boundaries are:

- a Workflow defines stages, their permitted executors and their configuration;
- a Planner is an ADK agent constructed for one prepared stage;
- a replaceable Planner factory selects the Planner strategy and constructs its
  ADK agent tree;
- an ephemeral Worker Agent performs the work without importing or knowing the
  Workflow or Planner implementation;
- Planner and Worker Agent communicate through A2A;
- Control Plane and Runtime Agents prepare execution capacity before Planner is
  invoked.

## Terms

### Workflow

A Workflow defines the product-specific sequence or graph of stages. A stage
declares the known Worker types that may execute it, such as `oas_builder` or
`swe_agent`, and the capacity/configuration needed for that execution.

Each stage selects exactly one Planner implementation/version for one stage
execution. The initial conceptual contract is:

```text
StageSpec
  planner
  agents: unique logical Worker Agent names
  inputs
  expected result
```

For example:

```text
agents: [oas_builder, oas_analyzer, swe_agent]
```

The list is a set of unique logical executors, not a `worker_type + count`
request. A stage may name one or several agents, but the same logical name does
not occur twice. Control Plane prepares exactly one Worker allocation for each
declared name before Workflow invokes Planner.

Workflow owns stage progression. It asks Control Plane to prepare the required
Worker capacity, invokes the selected Planner with the resulting Worker handles,
accepts the stage result and releases the allocations.

Workflow also owns Planner selection and construction. It may select different
Planner factories or versions for different stages of the same run; Planner is
not a globally assigned service and is not selected by Control Plane.

### Planner

Planner is an ADK agent, normally an `LlmAgent`, created for a prepared stage.
Multiple Planner implementations or versions may use different planning,
decomposition and routing strategies.

A Planner factory is the replaceable Contractor-level abstraction. Workflow
selects a factory from StageSpec and asks it to construct the Planner agent
tree. This avoids naming the Contractor abstraction `BasePlanner`: ADK already
uses `google.adk.planners.BasePlanner` for the separate `LlmAgent.planner`
thinking/planning configuration.

A Planner instance is created by Workflow for a selected stage. Exactly one
root Planner ADK agent owns that stage invocation. Its internal agent tree may
decompose, route or delegate, but those details do not introduce additional
Workflow-level Planner identities. Cross-stage state passes through explicit
Workflow context, stage results or artifacts rather than an implicitly shared
Planner instance.

After Control Plane has prepared every allocation, Workflow converts the
name-to-handle mapping into ADK remote agent proxies. Each ready Worker is
represented by one `RemoteA2aAgent`, constructed from the Worker's external
Agent Card and using allocation-scoped request metadata/authentication. These
proxies are passed to the root Planner through ADK's `sub_agents` composition:

```python
sub_agents = [
    RemoteA2aAgent(
        name=agent_name,
        agent_card=worker_handle.agent_card,
        a2a_request_meta_provider=allocation_metadata(worker_handle),
    )
    for agent_name, worker_handle in worker_handles.items()
]

planner = LlmAgent(
    name=stage.planner_instance_name,
    instruction=stage.planner_instruction,
    sub_agents=sub_agents,
)
```

The `sub_agents` list is a local ADK agent tree; it is not serialized over A2A.
Delegation to a `RemoteA2aAgent` crosses the process boundary through A2A to the
corresponding Worker Agent. A2A is not used for reservation or Worker launch.

#### Example: Passthrough Planner

`PassthroughPlanner` illustrates a minimal Planner ADK agent. It performs no
decomposition and creates no private subplan. It requires exactly one prepared
Worker subagent, sends the stage input to it through the `RemoteA2aAgent`,
observes the A2A Task to a terminal state and maps the terminal
response/artifacts to `StageResult`.

Conceptually:

```text
StageSpec + StageContext + WorkerHandle
  -> A2A SendMessage
  -> Worker A2A Task
  -> terminal Task/artifacts
  -> StageResult
```

The passthrough strategy must not know how the Worker was allocated, where it
runs, how its sandbox is laid out or which framework implements it.
`PassthroughPlanner` is the first planned concrete implementation and the
baseline end-to-end path for the Workflow/Planner/A2A/Worker separation, while
remaining only one of the available Planner strategies.

Other Planner implementations may decompose the stage, iterate, maintain
stage-local planning state and route work across multiple prepared Worker
subagents while preserving the same outer `StageResult` contract.

Planner receives an already prepared stage and a fixed `sub_agents` list for all
declared Worker Agents. Its factory receives the name-to-handle mapping and
constructs those remote proxies. Planner may route work between the named
subagents, but it does not add undeclared names, select physical Runtime Agents,
prepare sandboxes, launch processes or manage fleet capacity.

### Stage result

Planner completes its invocation with one framework-visible `StageResult`:

```python
from google.genai import types


class StageResult(BaseModel):
    outcome: StageOutcome
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    content: types.Content | None = None
    metrics: StageMetrics | None = None
    error: ErrorEnvelope | None = None
```

`StageOutcome` contains the terminal values `succeeded`, `failed` and
`cancelled`. A successful result has no `error`; a failed result must have one.
A cancelled result may carry an `ErrorEnvelope` describing the cancellation
reason. `content` and `artifacts` may also be present on failed or cancelled
results when they provide useful partial output or diagnostics.

`content` is the terminal Planner response and uses ADK's
`google.genai.types.Content`. It is normally text but may contain multiple
presentation-oriented parts. It replaces a separate unstructured `summary`
field. Internal reasoning and protocol parts such as `thought`,
`function_call` and `function_response` are not published as terminal stage
content.

Large or durable outputs are represented by immutable `ArtifactRef` values,
not embedded into `content`. Inline content is bounded by a configurable size
limit. Duplicate artifact references in one result are invalid; list order may
be preserved for presentation but must not be used as the artifact's semantic
identity. StageSpec decides which artifact kinds and cardinalities are required
for successful acceptance.

`StageMetrics` is a Contractor-owned normalized aggregate across the Planner
and its Worker invocations. It is not ADK's provider-specific usage metadata,
although ADK event usage is one input to aggregation. Metrics are optional;
unavailable measurements are represented as unknown/absent rather than zero.
The exact metric fields and aggregation/deduplication rules remain a separate
contract decision.

### Control Plane

Control Plane owns Runtime Agent registration, availability and allocation. It
translates a stage's capacity requirement into one or more exclusive Runtime
Agent reservations and returns ready Worker handles to Workflow.

Control Plane manages allocation lifecycle, not Planner algorithms or Worker
implementation semantics.

### Runtime Agent

A Runtime Agent is a long-running, lightweight, single-slot launcher and A2A
gateway. It can host at most one active Worker Agent allocation at a time.

For one allocation it:

1. reserves its slot;
2. receives and validates a `LaunchSpec`;
3. pulls/materializes the required state;
4. prepares an isolated sandbox;
5. starts the requested Worker Agent as a separate process or container;
6. waits for its private A2A endpoint to become ready;
7. exposes that Worker through the stable Runtime Agent gateway;
8. destroys the Worker, cleans its sandbox and frees the slot after release.

Runtime Agent must not import concrete Worker packages or their heavy
dependencies.

### Worker Agent

A Worker Agent is an ephemeral, isolated A2A Server for one allocation. It owns
its A2A Task state and implements one declared Worker type/version. It may use
ADK or another implementation without changing Workflow or Planner code.

The Worker Agent is private to its Runtime Agent. It may listen on loopback or a
Unix socket rather than exposing a directly routable deployment endpoint.

### Worker handle

A ready Worker handle contains only information needed by Workflow and Planner
to use the Worker through A2A:

- Worker type and version;
- allocation identity and generation;
- external Agent Card;
- allocation-scoped A2A credentials;
- lease/deadline information.

It contains no host path, sandbox handle, process/container handle or concrete
Worker object.

## Execution flow

```text
Workflow stage
  -> Control Plane: reserve required capacity
  -> Runtime Agent(s): reserve slot + accept LaunchSpec
  -> Runtime Agent(s): prepare state, sandbox and Worker Agent
  -> Worker Agent(s): start private A2A server
  -> Runtime Agent gateway: publish external Agent Card
  -> Control Plane: return ready Worker handle(s)
  -> Workflow: build RemoteA2aAgent subagents from ready handles
  -> Workflow: construct and run the selected Planner ADK agent
  -> Planner <-> A2A <-> Worker Agent(s)
  -> Workflow: accept stage result
  -> Control Plane: release allocations
  -> Runtime Agent(s): destroy Worker, clean sandbox, free slot
```

The infrastructure path precedes Planner execution:

```text
Workflow -> Control Plane -> Runtime Agent -> Worker preparation
```

The work path begins only after all required Worker handles are ready:

```text
Workflow -> Planner <-> A2A Gateway <-> Worker Agent
```

## Stage execution scope

One stage execution has:

- one root Planner instance;
- one ADK invocation of that Planner;
- one fixed set of Worker allocations prepared before the invocation starts;
- one terminal `StageResult` returned to Workflow.

The allocations live for the complete Planner invocation. Within that
invocation, Planner may send multiple sequential A2A Tasks to the same Worker
Agent and may invoke different Worker Agents concurrently. A single Worker
Agent processes at most one A2A Task at a time; it does not run concurrent Tasks
inside its allocation.

Consequently, one allocation is not the identity of one A2A Task. It is the
exclusive execution scope in which one ephemeral Worker Agent may serve a
sequence of A2A Tasks for the owning stage. Task identifiers remain Worker-owned
A2A identifiers beneath the allocation.

Planner cannot replace or extend its `sub_agents` during the invocation. After
Planner returns, Workflow first accepts or persists the terminal `StageResult`
and then releases all allocations belonging to the stage.

## Capacity and routing

Each Runtime Agent has one exclusive slot. A StageSpec declaring three named
Worker Agents therefore requires three available Runtime Agents and three
allocations before its Planner starts.

Workflow/stage declares the finite unique set of Worker Agent names. Planner may
act as a router among the corresponding prepared handles, but it cannot
introduce an undeclared agent, duplicate a declared name, expand the pool during
execution or address a physical Runtime Agent directly.

## A2A gateway model

The Runtime Agent exposes one stable, authenticated A2A gateway address. The
ephemeral Worker Agent exposes A2A only on a private local address. For each
ready allocation, Runtime Agent publishes an external Agent Card that describes
the actual Worker but points its interface at the stable gateway.

Under A2A 1.0, the interface `tenant` value can carry the opaque allocation
routing identity. It is routing data, not an authentication credential.
Allocation-scoped authentication is checked separately.

Conceptually:

```text
external Agent Card
  worker_type = oas_builder
  interface.url = https://runtime-agent-7.internal/a2a
  interface.tenant = allocation-123

gateway route
  allocation-123 -> private Worker endpoint on Runtime Agent 7
```

Gateway responsibilities are deliberately narrow:

- terminate transport security;
- authenticate the caller and validate the current allocation generation;
- route every A2A operation and stream to the private Worker endpoint;
- reject expired, released or stale allocations;
- close the route during release.

Gateway must not plan work, interpret Worker objectives, translate A2A into a
Worker-specific Python API, assemble Worker results or import Worker packages.
The Worker Agent remains the A2A Server that owns task semantics.

## Control plane versus A2A

Worker provisioning is not represented as an A2A skill.

The private control protocol owns:

- Runtime Agent registration and heartbeat;
- slot reservation;
- Worker launch/readiness;
- allocation lease and release;
- process/sandbox cleanup.

A2A owns interaction with an already prepared Worker Agent:

- send/stream message;
- task status and continuation;
- task cancellation;
- progress and artifacts.

## Initial interface shape

The exact DTOs are not yet accepted, but responsibilities imply interfaces of
this shape. `BaseAgent` below is ADK's agent type:

```python
class PlannerFactory(Protocol):
    def create(
        self,
        stage: StageSpec,
        workers: Mapping[str, WorkerHandle],
        context: StageContext,
    ) -> BaseAgent: ...


class ControlPlane(Protocol):
    async def prepare_stage(
        self,
        requirement: StageCapacityRequirement,
        launch_spec: LaunchSpec,
    ) -> list[WorkerHandle]: ...

    async def release_stage(
        self,
        handles: list[WorkerHandle],
        reason: ReleaseReason,
    ) -> None: ...
```

Workflow calls `ControlPlane`; the Planner factory converts the resulting
handles to `RemoteA2aAgent` subagents. The constructed Planner does not call
capacity-management APIs directly.

## Agreed invariants

1. Workflow, Planner, Runtime Agent and Worker Agent are different entities.
2. Planner is an ADK agent; Planner construction strategies are replaceable
   behind a Contractor `PlannerFactory` abstraction.
3. Each stage selects exactly one Planner implementation/version, and Workflow
   creates that Planner instance; Control Plane does not select it.
4. A stage declares a unique set of one or more logical Worker Agent names;
   Control Plane prepares exactly one allocation and handle per name before
   Planner starts.
5. `PassthroughPlanner` is the first planned Planner implementation and
   baseline integration path; the architecture does not special-case it.
6. Physical Runtime Agent selection belongs to Control Plane.
7. Planner receives a fixed name-to-handle map rather than physical Agent
   identities and cannot expand it during the stage.
8. Planner and Worker Agent communicate through A2A.
9. Every ready Worker handle is represented in the Planner tree by one ADK
   `RemoteA2aAgent`; the list itself remains local and each invocation uses A2A.
10. Runtime Agent is a launcher/gateway, not a semantic Worker adapter.
11. Runtime Agent hosts at most one active Worker Agent allocation.
12. N simultaneously declared Worker Agents require N Runtime Agents.
13. One stage execution creates one Planner instance and one ADK invocation;
    its fixed allocations may carry multiple sequential A2A Tasks.
14. A Worker Agent handles at most one A2A Task at a time, while different
    Worker Agents allocated to the same stage may run concurrently.
15. Concrete Worker code and heavy dependencies remain outside Server,
    Workflow, Planner and Runtime Agent packages.
16. Workflow releases allocations only after it has accepted/persisted the
    stage result or selected a failure/cancellation outcome.
17. A Stage returns one terminal `StageResult`; large or durable outputs use
    immutable `ArtifactRef` values, while bounded presentation content uses ADK
    `types.Content`.
18. Successful `StageResult` has no error, failed `StageResult` has an error,
    and missing metric values are never silently treated as zero.

## Open decisions

- Which ADK subagent interaction mode is required for each Planner strategy:
  transfer, task, or single-turn invocation?
- Which A2A 1.0 transport binding is required initially?
- Where are stage inputs and results stored, and which component materializes
  them for Worker?
- How does a Worker publish output bytes and obtain the immutable `ArtifactRef`
  returned through A2A?
- Which normalized fields, source identities and deduplication rules comprise
  `StageMetrics`?
- Which allocation/task state must survive Server, Runtime Agent or Planner
  restart in the first release?
