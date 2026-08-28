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
authenticates the agent. Control Plane records the last-seen request, issued and
confirmed heartbeat sequences, plus the single-slot state for the live
instance. The initial homogeneous fleet does not advertise per-agent capability
allowlists.

A restarted Runtime Agent registers with a new `instance_id`. It cannot adopt a
Worker or allocation belonging to its previous process instance. The old
registration becomes unavailable after its confirmed control lease expires,
and any active StageExecution using that allocation enters `aborting`; Workflow
Scheduler records an interrupted StageTermination, completes the bounded abort
path and then lets Workflow policy retry with a new StageExecution, escalate or
fail the Run.

This first model deliberately has no stable `AgentId`, incarnation nonce,
registration generation, Server epoch, rotated fleet token or cross-process
Worker continuation protocol.

## Heartbeat and confirmed control lease

The first-slice heartbeat interval is 10 seconds and the confirmed Runtime
Agent control lease is 60 seconds. Control Plane returns both values on
registration so that the Server remains their authority and the agent does not
depend on matching local configuration. Each side measures expiry with its own
monotonic clock; no shared wall-clock timestamp is a lease authority.

A Runtime Agent starts a strictly increasing heartbeat `sequence` at process
start and does not reset it across reconnects. A request contains:

```text
RuntimeAgentHeartbeat
  instance_id
  sequence
  last_ack_sequence?
  observed_state       idle | allocated | draining | fenced
  allocation_id?
  worker_stopped?
```

Control Plane responds with:

```text
RuntimeAgentHeartbeatAck
  ack_sequence         equal to the accepted request sequence
  heartbeat_interval
  lease_duration
  authoritative_state
  allocation_id?
  action               continue | drain | release | reregister
```

`last_ack_sequence` echoes the newest response the Runtime Agent actually
received. Control Plane maintains two different liveness facts:

- registration `last_seen` advances on every valid authenticated heartbeat
  request and is useful for observation and routing diagnostics;
- the confirmed control lease advances only when `last_ack_sequence` advances
  and names an ack that this current in-memory registration entry previously
  issued. Repeating the same ack never extends the lease again.

Runtime Agent likewise renews its local confirmed lease only after receiving a
new ack for a heartbeat sequence that it actually sent. A duplicate, delayed or
unknown ack does not extend the watchdog. Consequently requests reaching
Control Plane while all responses are lost cannot keep the confirmed lease
alive indefinitely: the echoed ack stops advancing and both sides expire.

An active allocation is live only while the confirmed control lease is valid
and the observed allocation matches Control Plane's authoritative record. An
idle process is eligible for placement only after the same round trip is
confirmed and both sides agree that no allocation exists. A heartbeat is only
an observation: `idle`, `fenced` or a different `allocation_id` can trigger
reconciliation but can never release or replace the authoritative allocation.

When the confirmed lease expires on Control Plane, it excludes the Runtime
Agent from placement. Any active allocation is irreversibly reported lost to
Workflow Scheduler, which enters `aborting` with an interrupted
StageTermination. A late ack cannot revive that allocation; only reconciliation
and a new allocation may return the slot to service.

Runtime Agent maintains the symmetric monotonic watchdog. If 60 seconds pass
without a new valid ack, the agent:

1. rejects new A2A Tasks and enters `draining`;
2. asks its active in-process Worker runtime to finalize the accumulated report;
3. stops and destroys that Worker instance within the configured shutdown grace;
4. if in-process termination cannot be guaranteed, exits the Runtime Agent
   process instead of reusing the slot;
5. otherwise deactivates the allocation's A2A identity, secrets and access
   context, retains its allocation ID and cached final report, and enters
   `fenced` rather than `idle`.

The fenced agent continues registration and heartbeat retries. It cannot accept
another allocation until Control Plane explicitly acknowledges release of the
old allocation. If no allocation existed when the local lease expired, the
agent still remains fenced until a confirmed response establishes that the
authoritative slot is unallocated. A lost release response is safe: the agent
continues reporting the old allocation as fenced and Control Plane repeats the
idempotent release action.

### Observed/authoritative reconciliation

- matching live allocation IDs and a confirmed lease allow `continue`;
- observed `draining` is valid only when Control Plane has issued finalization
  or abort for that allocation;
- observed `fenced`, `idle` or another allocation while Control Plane still
  owns an active allocation makes that allocation lost and starts/resumes the
  bounded abort path;
- an allocation reported for a `finalizing` or `aborting` StageExecution causes
  the corresponding idempotent control command to be reissued;
- an allocation reported for a terminal or unknown StageExecution receives an
  idempotent drain/release action;
- no mismatched process is offered for placement until it reports the released
  state and completes a fresh confirmed round trip.

A late report may be accepted as incomplete telemetry, but lease loss cannot be
reversed into successful completion of the interrupted StageExecution.

## Runtime Agent Registry persistence

The first implementation keeps the live Runtime Agent Registry in Control Plane
memory. Registrations, last-seen times, issued/confirmed heartbeat sequences,
control leases and current slot availability are not PostgreSQL state, and
heartbeat processing does not write one database row every 10 seconds.

PostgreSQL retains only the durable execution facts that refer to allocations,
including their StageExecution association and terminal outcome. After a Server
restart the in-memory Registry starts empty. A Runtime Agent process that
remained alive registers again with its existing process-scoped `instance_id`;
an agent process that also restarted uses a new one. Recovery still interrupts
every affected `preparing` or `running` StageExecution through durable
`aborting` because its Planner is not resumable, then drains any Worker reported
by the reconnected agent.

After Registry loss, an echoed ack from the former Control Plane process is not
recognized because the new in-memory entry did not issue it. After
re-registration, Control Plane acknowledges the next heartbeat; only echoing
that new response establishes the fresh confirmed control lease. The
process-monotonic heartbeat sequence avoids collision with a delayed response
without introducing a durable Server epoch or registration generation.

Registration or heartbeat reporting an active `allocation_id` triggers
reconciliation with durable StageExecution state before that slot can receive
new work. A matching `finalizing` execution reissues its recorded finalization;
a matching `aborting` execution reissues its recorded abort; a terminal or
unknown allocation is drained and released. Reconnection never adopts a Worker
for continued Planner execution.

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

1. reserves its only slot and validates `AllocationSpec`, including the exact
   registered SandboxProfile ref;
2. asks that profile to prepare the allocation-local workspace;
3. prepares allocation-local State and selected tools, then creates one
   in-process Worker runtime from the complete AgentTemplate;
4. configures its own A2A Server and Agent Card for that allocation;
5. binds its Artifact client and model access to the allocation context;
6. reports ready and handles the Worker's A2A Tasks itself;
7. on finalization or abort, rejects new Tasks, requests cancellation of any
   active Task, serializes the accumulated execution report and destroys the
   Worker runtime instance;
8. after the terminal Stage outcome is committed and Control Plane's
   idempotent release is acknowledged, clears allocation State, tools,
   RuntimeSettings, access tokens, retained allocation identity and cached
   report, cleans the profile workspace, then becomes `idle` and frees the
   slot.

The Worker runtime may use ADK or another future in-process adapter without
changing Workflow Scheduler or Planner contracts. Its dependencies live in the
Runtime Agent deployment, not in Contractor Server. It receives no PostgreSQL
or S3 credential and cannot outlive the Runtime Agent process.

Finalization and abort enter through the Runtime Agent's existing private
control endpoint. They perform no new semantic work: the same process stops its
current Worker runtime, converts accumulated metrics to `ExecutionReport` and
returns the report. Neither operation is exposed as an A2A skill or waits for
A2A cancellation to reach a terminal Task state.

The first-slice `local-workdir@1` SandboxProfile is prepared before any Toolset
factory or Worker can run. Failure to create its fresh allocation directory is
an allocation-preparation failure. During release, Runtime Agent reports the
allocation cleared only after removing that directory. Cleanup failure keeps
the old allocation identity `fenced` and the slot unavailable; an idempotent
release retry repeats cleanup. Before registering an idle slot after process
startup, Runtime Agent also removes recognized orphan allocation directories
under its dedicated configured work root. The profile does not add a process,
container, filesystem-permission or network security boundary.

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
moves that StageExecution through `aborting` with a retryable interrupted
StageTermination whose code identifies the infrastructure error. When the
batch is cleared or the abort deadline expires, the execution becomes terminal
and Workflow policy decides whether to create a new attempt.

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

    async def abort_stage(
        self,
        handles: Collection[WorkerHandle],
        abort_id: AbortId,
        deadline: datetime,
    ) -> Mapping[AgentName, AllocationFinalReport]: ...

    async def release_stage(
        self,
        handles: Collection[WorkerHandle],
        reason: ReleaseReason,
    ) -> Mapping[AgentName, AllocationReleaseAck]: ...
```

Exact transport DTO encoding remains open, but preparation, finalization, abort
and acknowledged release are always a private control protocol rather than A2A
skills. The StageExecution lifecycle and report contracts are defined by
[04](04-execution-lifecycle-and-metrics.md).

## Capacity and execution scope

One Runtime Agent has one active slot. A Stage declaring three named Workers
needs three available Runtime Agent processes before Planner starts. Those
processes may all run on the same VM.

One allocation lasts for the complete Planner invocation and its finalization.
Planner may send multiple sequential A2A Tasks to its Worker while the
allocation is active. A Worker executes at most one Task at a time; separate
Stage allocations may execute concurrently. A draining allocation rejects new
A2A work and remains reserved until its StageExecution is terminal. A fenced
slot rejects all work and remains unavailable until authoritative
reconciliation and release acknowledgement complete.

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
- deactivation during finalization, abort and release.

The transport layer does not plan work or assemble results. The in-process
Worker runtime owns Task semantics while the surrounding Runtime Agent owns
transport, allocation and lease checks.

## Protocol ownership

The private control protocol owns registration, heartbeat, reservation,
AllocationSpec and RuntimeSettings delivery, readiness, lease, idempotent Worker
finalization and abort, execution/runtime report transport, release and cleanup.

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
   Scheduler-owned cancellation/interruption instead moves it to durable
   `aborting`. The two transitions are mutually exclusive.
7. Worker finalization and abort cannot perform model/tool calls or artifact
   mutations, wait indefinitely for A2A Task cancellation, or exceed their
   durable control deadline.
8. Workflow Scheduler releases allocations only after recording terminal Stage
   outcome.
9. Stale, draining or released allocation identities cannot address the current
   or a replacement Worker instance.
10. A Runtime Agent recognizes a Control Plane peer by the reserved URI SAN
    prefix in a certificate signed by the deployment CA; Control Plane applies
    no per-agent authorization policy beyond allocation ownership consistency.
11. Runtime Agent process restart creates a new registration identity and never
    resumes or adopts the previous process instance's allocation.
12. The confirmed control lease advances only through a new monotonic
    request/ack/echo round trip; receiving heartbeat requests alone does not
    keep an allocation live or a slot eligible for placement.
13. Local lease expiry drains and destroys Worker, then leaves Runtime Agent
    fenced with the old allocation identity until explicit release is
    acknowledged; it never silently becomes idle.
14. Observed and authoritative allocation mismatch triggers reconciliation and
    cannot itself release, adopt or replace an allocation.
15. Live registrations, heartbeat sequence state and slot availability are
    in-memory Control Plane state; PostgreSQL does not serve as a heartbeat
    registry in the first slice.
16. An `allocation_id` is globally unique and never reused; no allocation
    generation exists in the first model.
17. Capacity for one Stage is reserved all-or-nothing; Planner never receives a
    partial WorkerHandle map.
