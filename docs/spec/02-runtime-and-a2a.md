# 02 — Runtime and A2A

Status: **Working agreement**

Depends on: [00](00-workflow-and-planner.md), [01](01-agent-template.md)

## Control Plane

Control Plane owns Runtime Agent registration, availability and allocation. It
checks the WorkerRuntime selected by each AgentTemplate, chooses free Runtime
Agent capacity and returns ready `WorkerHandle` values to Workflow Scheduler.

It manages allocation lifecycle, not Planner algorithms, Worker objectives or
A2A Task semantics.

## Control-plane trust and mTLS

Server and Runtime Agents form one deployment-owned trust domain rooted in a
single CA. The private control protocol uses mutual TLS. All Runtime Agents are
equally trusted and have the same control-protocol capabilities; the certificate
does not encode per-agent roles or an authorization policy.

Control Plane accepts any otherwise-valid Runtime Agent certificate that chains
to this CA. A Runtime Agent accepts a Control Plane peer only when both of the
following hold:

- its certificate chains to the same CA;
- one URI SAN starts with the exact canonical prefix
  `urn:contractor:control-plane:` and has a non-empty instance identifier after
  the prefix, for example
  `urn:contractor:control-plane:01951d4e-ec24-7df2-a9c1-18d55d123456`.

The URI SAN, rather than CN or OU, distinguishes Control Plane certificates from
the otherwise-equivalent Runtime Agent certificates. Prefix matching permits
certificate rotation and multiple future Control Plane instances without an
agent-side allowlist. CA issuance must ensure that Runtime Agent certificates do
not contain the reserved Control Plane URI SAN prefix.

For HTTPS/TCP, normal endpoint-name verification against a DNS/IP SAN still
applies; the reserved URI SAN is an additional Control Plane role check, not a
replacement for standard TLS endpoint validation. The same peer check protects
Server calls to both the Runtime Agent control endpoint and its A2A endpoint.

This is a node trust boundary, not user authentication: user clients do not
receive certificates from this CA, and an in-process Worker has no identity or
certificate separate from its Runtime Agent. Separately from TLS, Control Plane
keeps the normal allocation invariant that lifecycle messages and final reports
are associated with the Runtime Agent to which the allocation was issued. That
association protects execution-state consistency; it does not give different
Runtime Agents different privileges.

## Registration and process identity

Runtime Agent registration represents one running process, not a durable
machine or deployment identity. On process start the agent generates a fresh
random `instance_id`. It retains that value in memory across registration
retries and Control Plane reconnects, but does not reuse it after its own
restart.

The minimal registration describes:

```text
RuntimeAgentRegistration
  instance_id
  control endpoint
  A2A endpoint
  private protocol version
```

Registration with the same `instance_id` is idempotent. The `instance_id` is a
correlation and routing identity, not an authentication credential; mTLS
authenticates the agent. Control Plane records the last heartbeat and the
single-slot state for the live instance. The initial homogeneous fleet does not
advertise per-agent capability allowlists.

A restarted Runtime Agent registers with a new `instance_id`. It cannot adopt a
Worker or allocation belonging to its previous process instance. The old
registration becomes unavailable after its heartbeat lease expires, and any
active StageExecution using that allocation becomes `interrupted`; Workflow
policy may then retry with a new StageExecution, escalate or fail the Run.

This first model deliberately has no stable `AgentId`, incarnation nonce,
registration generation, Server epoch, rotated fleet token or cross-process
Worker continuation protocol.

## Heartbeat and control lease

The first-slice heartbeat interval is 10 seconds and the Runtime Agent control
lease is 60 seconds. Control Plane returns both values on registration so that
the Server remains their authority and the agent does not depend on matching
local configuration.

Each heartbeat reports the `instance_id`, observed single-slot state
(`idle`, `allocated` or `draining`) and the current `allocation_id` when one
exists. Control Plane's allocation record remains authoritative; a heartbeat is
an observation and cannot by itself complete, replace or release an allocation.

Control Plane renews the registration lease when it accepts a heartbeat. After
60 seconds without an accepted heartbeat it excludes the Runtime Agent from
placement and reports any active allocation as lost to Workflow Scheduler. The
Scheduler then moves an affected `preparing` or `running` StageExecution to
`interrupted` and applies Workflow retry, escalation or failure policy.

Runtime Agent maintains its own monotonic watchdog. It resets that watchdog only
after receiving a successful heartbeat response, not merely after sending a
request. If 60 seconds pass without an acknowledged heartbeat, the agent:

1. rejects new A2A Tasks and enters `draining`;
2. asks its active in-process Worker runtime to finalize the accumulated report;
3. stops and destroys that Worker instance within the configured shutdown grace;
4. if in-process termination cannot be guaranteed, exits the Runtime Agent
   process instead of reusing the slot;
5. otherwise deactivates the allocation's A2A identity and access context before
   becoming idle again.

The agent continues registration/heartbeat retries while disconnected. A late
report may be accepted as incomplete telemetry, but loss of the lease cannot be
reversed into successful completion of the interrupted StageExecution.

## Runtime Agent Registry persistence

The first implementation keeps the live Runtime Agent Registry in Control Plane
memory. Registrations, last-heartbeat times, control leases and current slot
availability are not PostgreSQL state, and heartbeat processing does not write
one database row every 10 seconds.

PostgreSQL retains only the durable execution facts that refer to allocations,
including their StageExecution association and terminal outcome. After a Server
restart the in-memory Registry starts empty. A Runtime Agent process that
remained alive registers again with its existing process-scoped `instance_id`;
an agent process that also restarted uses a new one. Recovery still interrupts
every affected `preparing` or `running` StageExecution because its Planner is
not resumable, then drains any Worker reported by the reconnected agent.

Registration or heartbeat reporting an active `allocation_id` triggers
reconciliation with durable StageExecution state before that slot can receive
new work. A matching `finalizing` execution may reissue its recorded
finalization; an interrupted, terminal or unknown allocation is drained and
released. Reconnection never adopts a Worker for continued Planner execution.

This design permits one active Control Plane instance in the first slice.
Shared fleet coordination, durable liveness state and multiple active Control
Plane replicas are deferred together; they must not be approximated by sharing
an unfenced `runtime_agents` table.

## Runtime Agent

A Runtime Agent is one long-running process with one exclusive slot. Runtime
Agent and Worker behavior are implemented by the same code and execute in that
same process. `Worker` names the temporary allocation-scoped role/runtime
instance, not another service, daemon, subprocess or container.

For one allocation the Runtime Agent:

1. reserves its only slot and validates `AllocationSpec`;
2. creates one in-process Worker runtime from the complete AgentTemplate;
3. prepares allocation-local State, tools and any tool sandbox/work directory;
4. configures its own A2A Server and Agent Card for that allocation;
5. binds its Artifact client and model access to the allocation context;
6. reports ready and handles the Worker's A2A Tasks itself;
7. on finalization, rejects new Tasks, serializes the accumulated execution
   report and destroys the Worker runtime instance;
8. after terminal Stage acceptance, clears allocation State, RuntimeSettings
   and access tokens, then frees the slot.

The Worker runtime may use ADK or another future in-process adapter without
changing Workflow Scheduler or Planner contracts. Its dependencies live in the
Runtime Agent deployment, not in Contractor Server. It receives no PostgreSQL
or S3 credential and cannot outlive the Runtime Agent process.

Finalization enters through the Runtime Agent's existing private control
endpoint. It performs no new semantic work: the same process stops its current
Worker runtime, converts accumulated metrics to `ExecutionReport` and returns
the report. Finalization is not exposed as an A2A skill.

### RuntimeSettings from Control Plane

Control Plane includes a resolved `RuntimeSettings` snapshot in every
AllocationSpec. It may contain the LLM Gateway URL and token, the Server Artifact
API endpoint, timeouts, limits and other deployment-owned adapter settings.
Runtime Agent does not resolve these values from AgentTemplate or local Worker
configuration.

`LLM Gateway` is a backend-neutral role in this contract. LiteLLM is the initial
backend, not a required Contractor component; a compatible backend can replace
it without changing Workflow, AgentTemplate or allocation semantics.

Secret fields are accepted only over the private mTLS control channel, retained
in memory for the active allocation and redacted from logs, Agent Cards,
WorkerHandle, metrics and durable StageExecution state. They are erased during
release. A token that must expire during a long allocation requires an explicit
private refresh operation; silently replacing the active settings snapshot is
not allowed.

## WorkerHandle

A ready handle contains only what Workflow Scheduler and Planner need to address
the prepared Worker:

- exact AgentTemplate ref;
- exact WorkerRuntime ref;
- globally unique allocation_id;
- external Agent Card;
- lease/deadline information.

It contains no host path, tool-sandbox handle or in-process Worker object.
It also contains no LLM Gateway token or other RuntimeSettings secret.

## Preparation and execution flow

```text
Workflow Scheduler selects a ready Stage and resolves its AgentTemplates
  -> Control Plane checks runtime refs and selects free Runtime Agents
  -> Runtime Agents reserve their slots and accept AllocationSpecs
  -> each Runtime Agent creates one in-process Worker instance
  -> each Runtime Agent publishes its allocation-scoped Agent Card
  -> Control Plane returns ready WorkerHandles
  -> Workflow Scheduler constructs Planner RemoteA2aAgent subagents
  -> Planner <-> A2A <-> allocated Runtime Agents acting as Workers
  -> Planner returns candidate StageResult
  -> Workflow Scheduler durably enters finalizing
  -> Control Plane asks Runtime Agents to finalize their Worker instances
  -> Workflow Scheduler accepts terminal StageResult
  -> Control Plane releases allocation contexts, leases and slots
```

Infrastructure preparation completes before Planner starts. If any required
Worker fails preparation, Workflow Scheduler receives no partial Planner
subagent set.

### Atomic Stage reservation

Control Plane reserves capacity for the complete Stage Worker set as one
in-memory operation. Under the single Control Plane instance's registry lock it
either moves enough `idle` slots to `reserved` or changes no slot at all. A
Stage requiring three Workers therefore never holds two slots while waiting for
the third.

Temporary lack of sufficient capacity is not a Stage failure. StageExecution
remains `preparing`, no slot is reserved, and Workflow Scheduler retries with
bounded backoff until capacity appears, the execution deadline expires or
cancellation is requested.

After reserving the full set, Control Plane creates one globally unique
`allocation_id` and one complete AllocationSpec per logical Stage Agent binding,
then may initialize those Runtime Agents concurrently. Planner starts only after
every Runtime Agent reports ready and Control Plane can return the complete
`WorkerHandle` map.

If any initialization fails, times out or returns an incompatible Agent Card,
Control Plane drains/releases every allocation from that batch, including
Runtime Agents that became ready. Workflow Scheduler receives no partial map and
terminates that StageExecution as `interrupted` with a retryable infrastructure
error; Workflow policy decides whether to create a new attempt.

`prepare_stage` is idempotent for `stage_execution_id`. A retry after a lost
response observes the same in-memory reservation/preparation operation and
cannot create a second Worker set. Loss of the Control Plane process instead
uses the already-defined interruption and cleanup recovery path.

```python
class ControlPlane(Protocol):
    async def prepare_stage(
        self,
        bindings: Collection[ResolvedStageAgentBinding],
        context: StageAllocationContext,
    ) -> Mapping[AgentName, WorkerHandle]: ...

    async def finalize_stage(
        self,
        handles: Collection[WorkerHandle],
        finalization_id: FinalizationId,
        deadline: datetime,
    ) -> Mapping[AgentName, AllocationFinalReport]: ...

    async def release_stage(
        self,
        handles: Collection[WorkerHandle],
        reason: ReleaseReason,
    ) -> None: ...
```

Exact transport DTO encoding remains open, but preparation, finalization and
release are always a private control protocol rather than A2A skills. The
StageExecution lifecycle and report contracts are defined by
[04](04-execution-lifecycle-and-metrics.md).

## Capacity and execution scope

One Runtime Agent has one active slot. A Stage declaring three named Workers
needs three available Runtime Agent processes before Planner starts. Those
processes may all run on the same VM.

One allocation lasts for the complete Planner invocation and its finalization.
Planner may send multiple sequential A2A Tasks to its Worker while the
allocation is active. A Worker executes at most one Task at a time; separate
Stage allocations may execute concurrently. A draining allocation rejects new
A2A work and remains reserved until its StageExecution is terminal.

Every allocation receives a globally unique `allocation_id` that is never
reused. Retry, escalation and replacement create new StageExecutions and new
allocation IDs. This ID is the sole lifecycle, routing and report-correlation
key; there is no separate allocation generation. Repeated control commands for
an active `allocation_id` are idempotent, while commands and A2A requests for a
released or unknown ID are rejected.

Planner cannot expand or replace the prepared set during the invocation.
Changing the set requires a new StageExecution decision by Workflow Scheduler
under the Workflow policy.

## Allocation-scoped A2A endpoint

Runtime Agent itself exposes one stable authenticated A2A address. While its
slot is allocated, the same process configures its A2A Server and external Agent
Card from the active AgentTemplate. An
opaque allocation routing key may be carried in the
[`AgentInterface.tenant`](https://a2a-protocol.org/v1.0.0/specification) field
defined by A2A 1.0; the protocol leaves its routing semantics opaque. It is not
an authentication credential.

```text
external Agent Card
  skills = actual Worker skills
  supportedInterfaces[0].url = https://runtime-agent-7.internal/a2a
  supportedInterfaces[0].protocolBinding = HTTP+JSON
  supportedInterfaces[0].protocolVersion = 1.0
  supportedInterfaces[0].tenant = allocation-123

Runtime Agent active slot
  allocation-123 -> in-process Worker runtime created from AgentTemplate
```

The external card declares the endpoint's mutual-TLS security requirement. The
Planner client is configured with the Server's Control Plane certificate; the
advertised `tenant` remains only the opaque route selector and is never treated
as a credential.

The A2A endpoint responsibilities are limited to:

- mTLS authentication of the Server as a Control Plane peer;
- validation that `tenant` matches the active allocation and lease;
- delivery of every A2A operation and stream to the current in-process Worker
  runtime;
- rejection of stale, expired or released allocations;
- deactivation during finalization/release.

The transport layer does not plan work or assemble results. The in-process
Worker runtime owns Task semantics while the surrounding Runtime Agent owns
transport, allocation and lease checks.

## Protocol ownership

The private control protocol owns registration, heartbeat, reservation,
AllocationSpec and RuntimeSettings delivery, readiness, lease, idempotent Worker
finalization, execution/runtime report transport, release and cleanup.

A2A owns interaction with a Runtime Agent's already prepared Worker instance:
messages/streams, Task state and continuation, cancellation, progress and
artifact refs.

The artifact plane owns data access, authenticated scope and Namespace grants,
and never sends normal durable artifact bytes through A2A. See
[03](03-artifact-plane.md).

## Single-VM baseline

The baseline deployment may use:

```text
one VM
  Contractor Server process
  PostgreSQL
  Runtime Agent process 1 (one slot, Worker runtime executes in-process)
  Runtime Agent process 2 (one slot, only if concurrency is needed)
```

Control, A2A and artifact transports may use loopback on the single VM.
Kubernetes, distributed scheduling, a child Worker process and a remotely
managed AgentTemplate service are not prerequisites.

## Invariants

1. Physical placement belongs to Control Plane, not Workflow Scheduler or
   Planner.
2. One declared Stage Agent name maps to one allocation and handle.
3. One Runtime Agent process has one slot and at most one in-process Worker
   instance; it never hosts multiple Workers concurrently. It may create the
   next Worker only after the prior allocation is released and fully cleared.
4. Runtime Agent owns transport/allocation checks; its current Worker runtime
   owns every A2A Task under the allocation.
5. Allocation identity is not A2A Task identity; one allocation may serve a
   sequence of Tasks.
6. Planner completion first moves its StageExecution to durable `finalizing`;
   Runtime Agent then finalizes its Worker instance and returns bounded reports.
7. Worker finalization cannot perform model/tool calls or artifact mutations.
8. Workflow Scheduler releases allocations only after recording terminal Stage
   outcome.
9. Stale, draining or released allocation identities cannot address the current
   or a replacement Worker instance.
10. A Runtime Agent recognizes a Control Plane peer by the reserved URI SAN
    prefix in a certificate signed by the deployment CA; Control Plane applies
    no per-agent authorization policy beyond allocation ownership consistency.
11. Runtime Agent process restart creates a new registration identity and never
    resumes or adopts the previous process instance's allocation.
12. Sixty seconds without an acknowledged heartbeat makes both sides stop
    treating the allocation as live; Runtime Agent must drain and ultimately
    terminate its Worker before reusing the slot.
13. Live registrations and slot availability are in-memory Control Plane state;
    PostgreSQL does not serve as a heartbeat registry in the first slice.
14. An `allocation_id` is globally unique and never reused; no allocation
    generation exists in the first model.
15. Capacity for one Stage is reserved all-or-nothing; Planner never receives a
    partial WorkerHandle map.
