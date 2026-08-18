# 03 — Google ADK integration boundary

Status: **Draft**  
Depends on: [01](01-architecture-boundaries.md), [02](02-domain-contracts.md)

Reference baseline: [workflows](https://adk.dev/workflows/),
[sessions](https://adk.dev/sessions/session/),
[A2A](https://adk.dev/a2a/), [artifacts](https://adk.dev/artifacts/) and
[observability](https://adk.dev/observability/).

This specification governs the default ADK-backed Planner and Worker adapters.
The Contractor domain DTOs and A2A Worker profile are framework-neutral: a
passthrough Server Planner does not require ADK planning primitives, and a
conforming Worker Agent may implement its local strategy without ADK.

## Reused ADK capabilities

The initial baseline is Google ADK `2.7.1` with the `a2a` and `db` extras.
The lock file selects one exact tested version even when the project metadata
permits compatible `2.x` upgrades.

The default decomposing Planner and ADK Worker strategy reuse public ADK APIs
for:

- `Runner`, `LlmAgent` and callbacks/plugins; ADK Worker tool
  declaration/dispatch hooks adapt into Contractor `ToolInvoker`, while
  decomposing-Planner tool-shaped functions are pure typed-command builders;
- graph `Workflow`, typed/function nodes, retries and local concurrency;
- `DatabaseSessionService(db_engine=...)`;
- `BaseArtifactService` and context-facing artifact operations;
- the reference A2A exposure and consumption adapters;
- built-in OpenTelemetry instrumentation;
- code executors or Environment APIs only where their isolation contract fits.

## Explicitly not delegated to ADK

- authoritative `PlannerRunState` and its CAS protocol;
- Server run lifecycle and public API;
- Worker registry, health, capability routing, leases and fencing;
- PostgreSQL ArtifactService and telemetry repository;
- effective cancellation of Worker execution and sandbox processes;
- pre-task-ID cancellation tombstones and the private Attempt-control endpoint;
- Contractor tool implementations and sandbox security policy.

## Session policy

- Every ADK-backed decomposing-Planner invocation has a distinct session ID
  linked to `run_id`, RunState version and stable `planner_invocation_id`.
- Every ADK-backed Worker Attempt has a distinct session ID.
- A session has one active writer at a time.
- ADK session events are invocation history, not independent transitions of the
  durable Contractor run/task state machine.
- Persisted ADK events are reused in place for authorized diagnostic timelines
  and bounded derived metrics through the public session-service/callback APIs;
  an ADK event is not copied into a second Contractor row merely to make it
  observable.
- A failed/pre-CAS Planner session is never reused as context for the retry;
  accepted facts are carried by RunState or referenced artifacts only.
- Contractor code uses `DatabaseSessionService` instead of querying or mutating
  ADK-owned tables directly.

The passthrough Planner creates no ADK Workflow, Planner `LlmAgent` or Planner
session. A non-ADK Worker creates no ADK Worker session. Their authoritative
Contractor run/Attempt, artifact and A2A mapping records are unchanged.

Persistent ADK detail is an adapter policy, not a correctness prerequisite. If
enabled, one session history is retained and reused in place for diagnosis and
bounded metric derivation. If its privacy or measured write cost is not
acceptable, the profile may use nonpersistent/bounded session detail plus
callback aggregates and external sampled tracing; no RunState, accounting,
effect or audit fact is lost. Persisted prompts, responses and tool history are
classified as sensitive application data, receive least-privilege access,
encryption/backup treatment and short explicit retention, and never contain
runtime credentials or injected tool secrets.

## Workflow execution policy

For the decomposing strategy, ADK Workflow is an in-process executor for one
Planner invocation, not the durable run scheduler. Each invocation is
constructed from a verified RunState version and returns typed Planner
commands. ADK Workflow replay/resume is not used to infer which distributed
tasks are complete. The passthrough strategy reaches the same command/state
machine boundary without constructing an ADK Workflow.

The durable crash boundary is the RunState CAS:

- before CAS, a crash discards the proposed turn and a fresh turn may run from
  the same RunState version;
- the Planner CAS commits only the next RunState TaskSpecs/edges, Planner
  accounting and status projection in one PostgreSQL transaction;
- a later deterministic scheduler CAS creates an Attempt, WorkerJob, exact input
  grants and dispatch outbox in one transaction;
- after the Planner CAS, recovery skips the accepted planning decision and runs
  the scheduler; after the scheduler CAS, recovery delivers its committed
  outbox without rerunning either decision.

ADK session events written by a failed/pre-CAS turn may remain as diagnostic
history. They do not authorize dispatch or a domain state transition.

## ADK schema preparation

A deployment migration job using a DDL-capable role prepares and validates the
ADK schema before an ADK-backed Server or Agent adapter becomes ready. Runtime
roles have no general DDL permission. Because `DatabaseSessionService` calls
`prepare_tables()` lazily, a capability test must prove that it can inspect an
already prepared schema and operate under the runtime role without attempting
privileged DDL.

## Requirements

- **ADK-001** — ADK imports MUST be confined to adapter and application
  composition packages.
- **ADK-002** — Only documented public APIs may be used. Any unavoidable
  internal API requires a wrapper, a pinned-version compatibility test and an
  explicit issue to remove it.
- **ADK-003** — When configured, `DatabaseSessionService` MUST receive the
  process-wide shared `AsyncEngine`; it MUST NOT create a second pool from a
  URL.
- **ADK-004** — ADK Workflow replay/session state MUST NOT become a second
  authoritative run state. Recovery starts from committed Contractor state.
- **ADK-005** — The reference ADK A2A adapter MUST translate at the boundary
  between A2A models and v2 DTOs. Those models MUST NOT leak into Planner.
- **ADK-006** — ADK upgrades MUST run capability tests before dependency lock
  changes are accepted.
- **ADK-007** — The ADK Worker strategy MUST provide cancellation around ADK
  execution until a tested ADK API proves that active Runner and tool work is
  interrupted.
- **ADK-008** — Durable decomposing-Planner recovery MUST start a fresh ADK
  Workflow invocation from committed RunState when a plan mutation is needed;
  ADK session replay MUST NOT authorize a dispatch.
- **ADK-009** — A deployment enabling persistent ADK adapters MUST prepare ADK
  tables before those adapters become ready, and production runtime roles MUST
  NOT require database DDL grants.
- **ADK-010** — The reference ADK A2A adapter MUST pass A2A 1.0 capability
  tests for A2A-server-generated (Agent-generated) task IDs, accepted Contractor
  context IDs, task listing by context, get/cancel, required Contractor Worker-
  profile extension negotiation and its exact structured dispatch/progress/result/
  error Parts, plus an injected durable task store or Contractor wrapper.
- **ADK-011** — The private Contractor Attempt-control HTTPS adapter MUST be
  co-hosted beside, but remain separate from, the standard A2A routes exposed
  by the selected protocol adapter. Its tombstone handler MUST complete before
  any Worker strategy invocation can start.
- **ADK-012** — Selecting the passthrough Planner MUST NOT construct an ADK
  Workflow, Planner `LlmAgent`, Planner session or Planner budget reservation.
- **ADK-013** — A Worker Agent that does not select the ADK Worker strategy MUST
  NOT be required to import ADK, create ADK sessions or access ADK-owned tables
  to conform with Contractor `WorkerJob`, `WorkerResult` and A2A contracts.
- **ADK-014** — The runtime adapter, not Worker-authored input, MUST derive ADK
  app/user/session identity from authenticated tenant, Agent, Run and Attempt
  identity. Worker code MUST NOT receive the database engine or an unrestricted
  session service. Persisted ADK detail is diagnostic, bounded by retention and
  exposed only through the authorized/redacted observability adapter.
- **ADK-015** — An ADK-backed adapter MUST reuse persisted session events through
  documented public APIs or callback-maintained aggregates for optional
  diagnostics/metrics. It MUST NOT create a row-for-row Contractor telemetry
  mirror, scan ADK internal tables directly or infer authoritative lifecycle,
  audit or accounting facts solely from those events.
- **ADK-016** — Any tool-shaped function exposed to the decomposing Planner LLM
  MUST be a bounded, side-effect-free constructor for typed plan proposals. It
  MUST NOT receive `ToolInvoker`, Worker/fleet ports, filesystem/network access
  or persistence authority; only validation plus RunState CAS may apply its
  proposed command.
- **ADK-017** — Persistent ADK session detail MUST be explicitly enabled per
  adapter profile with sensitivity classification, authorization, byte/event
  caps and retention. Disabling or pruning it MUST leave authoritative
  lifecycle, accounting, effect and audit records intact; credentials and tool
  secrets MUST be removed before any session event is persisted.

## ADK-profile and isolation capability tests

1. `DatabaseSessionService` accepts and reuses an injected `AsyncEngine`.
2. A graph workflow runs a route, fan-out/join, timeout and retry.
3. A crash before RunState CAS reruns a fresh turn from the same version; a
   crash after CAS dispatches only the committed outbox.
4. A2A execute and streaming events round-trip through the boundary adapter.
5. A cancellation request interrupts Runner work and kills sandbox subprocesses.
6. ADK artifact context calls work with the Contractor PostgreSQL adapter.
7. Every enabled ADK-backed Server/Agent adapter starts and uses sessions with
   pre-created ADK tables under database roles that cannot create or alter them.
8. A crash before the first A2A response recovers the A2A-server-generated
   (Agent-generated) task ID
   by the committed context ID and does not create a second execution.
9. A delayed execute racing an unknown-task cancel is serialized against the
   durable Agent tombstone and never starts after cancellation acknowledgement.
10. A passthrough run reaches Worker dispatch with no ADK Planner invocation or
    session, while the same RunState/outbox assertions still pass.
11. A Worker-supplied session/user identifier cannot read or overwrite another
    Agent or Attempt's ADK history, and pruning that history does not change
    authoritative Run/Attempt status or accounting.
12. A high-volume ADK event fixture grows the ADK session store once; Contractor
    telemetry grows only by configured bounded aggregates/samples, and public
    diagnostic reads are paginated/capped rather than full-table scans.
13. A retained Planner command-tool fixture can propose subtasks but cannot
    access a Worker, external target or repository; malformed commands fail
    validation and leave RunState/outbox unchanged.
14. Running the same Worker profile with bounded persistent versus
    nonpersistent ADK detail changes only optional diagnostics/database load.
    Canary credentials never enter session history, and pruning it leaves final
    status, accepted usage, effects and audit unchanged.
