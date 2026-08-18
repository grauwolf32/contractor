# Contractor v2 architecture

> **Architecture revision in progress.**
> [`core-execution-model.md`](core-execution-model.md) records the current
> Workflow/Planner/Control Plane/Runtime Agent/Worker agreement. The LikeC4
> model and detailed specifications still contain parts of an earlier candidate
> and are not authoritative where they conflict with that document.

[`architecture.c4`](architecture.c4) is the canonical architecture map. It
contains seven focused views:

- `index` — processes and external infrastructure;
- `orchestration` — decomposing/static/passthrough strategy selection through
  CAS/outbox and A2A to a Worker;
- `workerAttempt` — Agent-side deduplication, framework-neutral Worker strategy,
  optional ADK/native bindings, tools and sandbox;
- `serverPersistence` — Server adapters sharing one engine and transaction
  boundary;
- `agentPersistence` — Agent sessions, staged artifacts, task mappings and
  scoped model/effect evidence, audit and telemetry sharing one engine;
- `fleetControl` — private registration, heartbeat, drain and routing path;
- `observability` — authoritative lifecycle/accounting sources, append-only
  audit, bounded PostgreSQL projections and optional external OpenTelemetry.

## Ownership legend

| Color | Owner | Meaning |
|---|---|---|
| Blue | Contractor | Code and contracts implemented in this repository |
| Green | Google ADK | Public ADK primitives reused through adapters |
| Gray | External | PostgreSQL, LLM Proxy/providers, OpenTelemetry and API clients |

The colors describe ownership, not process placement. Green components are the
optional default ADK profile; a conforming non-ADK Worker binds the blue
Contractor ports and does not instantiate those components.

## Fixed architecture decisions

1. Server, Agents, PostgreSQL and LLM Proxy are separate failure domains.
2. There is one Server process and N Agent processes in v2. High availability
   for Server is a non-goal for the first release.
3. There is one physical PostgreSQL database. ADK sessions, Contractor
   artifacts, run/Attempt state, model/effect evidence, audit and bounded
   telemetry use separate logical table groups.
4. Every process owns one shared SQLAlchemy `AsyncEngine`; individual
   operations use short-lived `AsyncSession` instances.
5. `PlannerRunState` is a versioned artifact and the authoritative state of
   the plan. It pins the exact immutable workflow profile, RunSpec and
   decomposing, static or passthrough Planner strategy.
   ADK session events are invocation history, not a competing source of truth.
6. Contractor owns the framework-neutral Planner/Worker, tool-policy and A2A
   boundaries. The default ADK profile supplies optional Runner, LlmAgent,
   workflow, session and adapter implementations; a Worker may use another
   implementation without changing wire or domain contracts.
7. Server and Agents call models only through the shared LLM Proxy.
8. The decomposing Planner may execute a fresh ADK Workflow invocation from a
   committed RunState version. Static planning loads a versioned task manifest;
   deterministic passthrough is its one-root specialization. All three use the
   same state machine/outbox.
9. Worker dispatch uses a transactional PostgreSQL outbox committed with
   RunState CAS; duplicate delivery keeps the same Attempt identity.
10. Agent outputs are staged first and become run-visible only when Server
    promotes their refs in a fencing-token-checked CAS transaction. Direct
    Agent artifact access is constrained by PostgreSQL RLS.
11. Agent registration/heartbeat uses a private mTLS HTTPS protocol, separate
    from A2A task execution. Fleet leases are durable and Server-epoch fenced.
12. Server Attempt/fencing rows and Agent A2A task mappings use separate tables
    and grants inside the same physical database.
13. Standard A2A owns task execution/query/cancel once `task_id` is known.
    Unknown-task cancellation installs a durable tombstone through a small mTLS
    Attempt-control API co-hosted in Agent; this is not another service.
14. A Worker strategy may work directly or alternate planning and working
    internally. Those private steps share one Attempt, budget, deadline,
    cancellation tree and result and never become Server Tasks implicitly.
15. Telemetry is derived: authoritative lifecycle/usage comes from domain
    stores, ADK detail remains in ADK sessions, and PostgreSQL telemetry holds
    bounded rollups/indexes/samples rather than duplicating every event. Raw
    sampled traces may go to an optional external OpenTelemetry backend.
16. Source projects enter as canonical, checksummed snapshot artifacts. Server
    and A2A contracts never carry a caller's local filesystem path.
17. Security audit is append-only and retained independently from lossy
    telemetry; model invocation evidence and budget settlement remain
    authoritative even when ADK/OTel diagnostic detail is pruned.

## Local commands

```shell
likec4 validate docs
likec4 start docs
```

Any change that moves a responsibility across the ADK/Contractor boundary must
update both the model and the relevant document in [`spec/`](spec/README.md).
