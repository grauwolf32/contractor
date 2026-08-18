# 01 — Architecture boundaries

Status: **Draft**  
Depends on: [00-product-scope](00-product-scope.md)

The canonical component map is [`../architecture.c4`](../architecture.c4).

## Ownership

| Area | ADK-owned | Contractor-owned |
|---|---|---|
| Agent execution | Optional ADK binding: `Runner`, `LlmAgent`, tool declaration/dispatch hooks | portable Worker strategy, capability providers and the enforcing `ToolInvoker`/policies |
| Workflow | Optional decomposing-Planner binding: graph nodes, routing, concurrency and local replay | Planner strategy selection, run semantics, authoritative RunState and distributed dispatch |
| Sessions | `DatabaseSessionService` and its tables | engine lifecycle and session identity policy |
| A2A | Optional ADK client/server bridge | Contractor A2A profile, adapter ports/translation, registry, routing, leases, fencing and effective cancellation |
| Artifacts | `BaseArtifactService` contract and context API | PostgreSQL implementation, CAS, ACL and retention |
| Telemetry | ADK session detail and optional OTel spans | authoritative-domain views plus bounded PostgreSQL rollups/samples and query adapters |
| Sandbox | optional executor/environment primitives | workspace ownership, Podman policy and cleanup |

The ADK-owned column describes the default ADK-backed adapters. It is not a
requirement that every Worker Agent use ADK. A custom Worker strategy remains
inside the Agent process and implements the same Contractor ports and A2A wire
contract.

A2A is the stable process/interoperability seam, not the owner of planning
semantics. `PlannerStrategy` is independently replaceable inside Server and
`WorkerStrategy` is independently replaceable inside Agent. Consequently the
existing Server decomposer, a deterministic static/passthrough stub, an ADK
Worker that plans and works internally, and a non-ADK Worker all compose through
one dispatch/result profile without importing one another.

## Required ports

The domain/application layers depend on ports with no ADK, SQLAlchemy, FastAPI
or A2A types in their signatures:

- `RunStateStore` — load and compare-and-swap `PlannerRunState`;
- `PlannerStrategy` — propose typed Planner commands from one exact RunState;
  decomposing, static-manifest and deterministic passthrough Planners implement
  this same Server-local port and produce `TaskSpec`, not `WorkerJob`, values;
- `RunRepository` — idempotency, status projection, budget ledger and dispatch
  outbox;
- `ServerAttemptRepository` — authoritative Attempt leases, fencing, outbound
  A2A delivery identity, operation-start gates and ingested terminal results;
- `AgentTaskRepository` — Agent-owned durable inbound A2A task mapping and
  cancellation/effect state plus append-only recovery reports, using guarded
  generation-authenticated mutations and no write path to Server run-control
  rows;
- `EffectEvidenceReader` — Server read-only access to Agent-scoped durable tool
  intent/outcome/reconciliation evidence when retry safety must be decided;
- `WorkerGateway` — execute/stream an exact committed
  `WorkerDispatchEnvelope`, recover a task by committed context and cancel the
  stored A2A task using an already committed Agent assignment;
- `AttemptControlClient` — install an idempotent unknown-task cancellation
  tombstone at the already assigned Agent;
- `WorkerRegistry` — register, heartbeat and resolve capabilities;
- `AgentRegistryClient` — private fleet-control transport used by Agent;
- `WorkerStrategy` — Agent-local start, inspection/reconciliation, cancellation
  by handle or durable reference, recovery of an ambiguous start by its
  pre-persisted key, and termination acknowledgement for one portable
  `WorkerJob`; an implementation may directly work or plan and work internally;
- `ToolInvoker` — Attempt-scoped, policy-enforcing access to versioned
  Contractor tools for every Worker framework adapter;
- `ArtifactStore` — typed artifact read/write/list operations;
- `ModelInvocationRepository` — scoped requested/resolved model, status and
  usage evidence; only Server accounting accepts and settles it;
- `AuditRepository` — append/query authorized, non-lossy security facts with
  retention independent of diagnostic telemetry;
- `EventPublisher` — non-blocking domain event publication;
- `SandboxManager` — acquire and release an attempt-scoped workspace lease.

## Requirements

- **ARC-001** — Domain modules MUST NOT import Google ADK, SQLAlchemy, A2A SDK,
  web-framework or Podman client modules.
- **ARC-002** — ADK integration MUST be isolated in adapters and composition
  roots. ADK objects MUST NOT be persisted as domain DTOs.
- **ARC-003** — Every Server Planner strategy MUST reach Workers only through
  committed dispatch and `WorkerGateway`; direct invocation of
  `AgentTool.run_async()` or an in-process Worker builder is forbidden.
- **ARC-004** — Server MUST NOT receive an Agent host path or filesystem
  object. It uses `ArtifactRef` and `WorkspaceRef` only.
- **ARC-005** — Agent owns sandbox creation, cancellation and destruction.
- **ARC-006** — Execution correctness MUST depend on RunState and result
  commits, not on successful telemetry publication.
- **ARC-007** — Each process MUST construct its dependencies in one explicit
  composition root. Configuration reads and client construction at import time
  are forbidden.
- **ARC-008** — Cross-boundary errors MUST use the versioned error envelope
  from spec 02 rather than framework exceptions.
- **ARC-009** — Agent task persistence and Server Attempt authority MUST use
  separate repository ports, logical tables and database grants even though
  both live in one physical PostgreSQL database.
- **ARC-010** — `AttemptIoWorker`, not `WorkerGateway`, MUST resolve an Agent for
  execute and commit the exact `WorkerDispatchEnvelope` plus
  endpoint/message/context before calling the gateway. The gateway MUST NOT
  consult `WorkerRegistry` for execute, recovery or cancel.
- **ARC-011** — Attempt-control is a Contractor-owned private HTTPS adapter
  co-hosted with Agent, not a custom A2A method or another deployable service.
- **ARC-012** — Selecting decomposing, static or passthrough planning MUST be a
  `PlannerStrategy` composition/configuration decision. It MUST NOT require a
  different A2A method, Worker DTO or run lifecycle implementation.
- **ARC-013** — `AgentRuntime` MUST invoke Worker logic through
  `WorkerStrategy`; ADK Runner is one adapter and MUST NOT be referenced by
  domain, A2A profile or fleet-routing contracts.
- **ARC-014** — Agent-internal planning, working and local subtask state MUST
  remain behind the Worker boundary. Only normalized progress and terminal
  `WorkerResult` data may cross it unless a later version defines an explicit
  delegation contract.
- **ARC-015** — Server result acceptance/accounting MUST read the authoritative
  model-invocation ledger and pin its evidence version; diagnostics/session
  events cannot substitute for that path.
- **ARC-016** — Required run-state audit appends and Attempt operation-gate
  changes MUST participate in the `RunStateStore` caller-owned Unit of Work.
  Control Plane MUST NOT perform a separate best-effort submission/cancellation
  audit write after the state mutation.

## Acceptance

1. A dependency test fails when a domain package imports an adapter package.
2. An in-process fake `WorkerGateway` can replace A2A without changing Planner.
3. A fake `RunStateStore` can execute Planner state-transition tests without
   PostgreSQL or ADK.
4. Architecture tests assert that only Agent packages import sandbox adapters.
5. Architecture and database-grant tests prove that Agent runtime packages
   cannot mutate Server Attempt leases, fencing or dispatch outbox records.
6. A gateway contract test requires a committed assignment and has no registry
   dependency; cancel/recovery remain routable after heartbeat expiry.
7. A transport test proves standard A2A routes and the private Attempt-control
   route have separate DTOs while sharing only authenticated domain identity.
8. Replacing the decomposing Planner with the passthrough Planner changes only
   composition/configuration and yields one root Task whose first Attempt uses
   the normal committed `WorkerJob` path.
9. The same gateway contract suite passes against an ADK-backed Agent and a
   minimal non-ADK Agent.
10. A Worker strategy can alternate planning and working internally without
    adding Agent-private subtasks to authoritative Server RunState.
11. A strategy start accepted remotely before its reference is persisted is
    recovered/cancelled by the pre-persisted `start_operation_id` without
    duplicate work.
12. A fault between submission/cancellation state mutation and its required
    audit append commits both or neither; result settlement reads one exact
    model-invocation evidence snapshot.
