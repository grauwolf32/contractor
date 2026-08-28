# 05 — First slice and open decisions

Status: **Working agreement**

Depends on: [00](00-workflow-and-planner.md) through
[04](04-execution-lifecycle-and-metrics.md)

## First implementation slice

The first slice proves boundaries, not the final distributed runtime:

```text
one user-scoped uploaded input
  -> one Workflow Run with that exact version forked into RunScope inputs
  -> Workflow Scheduler selects one Stage
  -> one exact AgentTemplate
  -> one local single-slot Runtime Agent
  -> one in-process Worker runtime instance
  -> one PassthroughPlanner invocation
  -> one A2A invocation producing an immediate Message or terminal Task
  -> one committed working artifact
  -> one persisted candidate StageResult and finalizing StageExecution
  -> one graceful Worker finalization report
  -> one terminal StageExecution and one Scheduler-bound Run output
  -> allocation cleanup
```

Server, PostgreSQL and Runtime Agent may run on one VM. Runtime Agent and its
current Worker runtime are one process; tool execution may still use a separate
sandbox where a tool requires it.

This slice must demonstrate:

- template resolution happens before allocation and the exact ref is retained;
- every Runtime Agent runs the same code and can instantiate the
  AgentTemplate's `runtime: adk@1` in-process;
- Control Plane supplies the allocation's LLM Gateway URL/token and other
  RuntimeSettings over mTLS; the Agent keeps secrets only in memory and clears
  them on release;
- Workflow Scheduler records `preparing -> running -> finalizing -> terminal`;
- Planner knows only the logical Worker name and WorkerHandle;
- Worker runtime code/heavy dependencies are absent from Server and live in the
  Runtime Agent deployment;
- A2A reaches the allocated Runtime Agent's own A2A Server; there is no proxy to
  a child Worker process;
- Planner completion produces a candidate result while Workflow Scheduler owns
  its durable acceptance;
- Planner uses one database-backed ADK Session associated with the
  StageExecution, while Worker ADK state remains in the allocated Runtime Agent
  process;
- Runtime Agent finalizes and destroys its in-process Worker instance through
  its private control endpoint and returns bounded reports before terminal
  acceptance;
- private Control Plane/Runtime Agent traffic uses the deployment CA for mTLS;
  Runtime Agents additionally require the Control Plane URI SAN prefix
  `urn:contractor:control-plane:` while Control Plane treats all valid Runtime
  Agent certificates as peers with equal capabilities;
- each Runtime Agent process registers one in-memory `instance_id`; restarting
  it creates a new identity and interrupts rather than adopts its old
  allocation;
- Runtime Agent sends a heartbeat every 10 seconds, both sides expire its
  control lease after 60 seconds, and lease loss makes the agent drain and
  ultimately kill its Worker before reusing the slot;
- live Runtime Agent registrations, heartbeat leases and slot availability are
  held in Control Plane memory rather than written to PostgreSQL;
- Control Plane reserves the complete Stage Worker set atomically and returns
  either every ready WorkerHandle or no handles; preparation is idempotent for
  `stage_execution_id`;
- one ArtifactStore binds public calls to UserScope and Worker calls to
  RunScope;
- input fork records an exact source version and lineage without requiring a
  blob copy, and Worker mutation leaves the user source unchanged;
- artifact bytes use the Server's private allocation-bound Artifact API while
  A2A carries only the ref;
- accepted output is frozen under `outputs/<slot>` and is not implicitly
  published back to UserScope;
- release clears the allocation's A2A identity, State, tools and access context.

## Deliberately deferred

The following are open contract decisions, not permission to inherit behavior
from the older `docs/spec` candidate:

- exact DTOs and size limits for AgentTemplate instruction/model/tool/sandbox
  policies;
- exact Workflow YAML schema and catalog directory layout, including named
  Stage-result and Workflow-output mappings;
- exact framework-neutral StageContent DTO;
- canonical serialization and digest algorithm for AgentTemplate;
- AgentTemplate YAML file schema and catalog directory layout;
- exact in-process WorkerRuntime factory contract;
- ADK subagent interaction mode for each Planner strategy;
- initial A2A 1.0 transport binding; direct Planner-to-Runtime-Agent
  authentication already uses the deployment mTLS identity defined in
  [02](02-runtime-and-a2a.md);
- Planner artifact authority and whether it receives domain toolsets;
- exact numeric limits and retention for tool-call detail and execution
  telemetry;
- optional incremental Worker metric delivery for retaining detail across a
  hard process crash;
- initial blob backend and upload/streaming/retention limits;
- public User Artifact API DTOs and explicit Run-output publication endpoint;
- retry/cancellation semantics after ambiguous allocation initialization or A2A
  delivery;
- concrete graceful-drain timeout before forced Worker termination after lease
  loss;
- shared Runtime Agent Registry and coordination for multiple active Control
  Plane replicas;
- public API representation for exact historical Stage-output versions;
- exact RuntimeSettings DTO and LLM Gateway client protocol;
- RuntimeSettings token issuance, expiry/refresh and concrete redaction rules;
- concrete CA bootstrap, certificate delivery, lifetime, rotation and
  revocation procedures;
- multi-tenant authorization and quota policy.

Each decision should be added to its owning spec only when a concrete first-
slice implementation needs it.

## Explicit non-goals for the first slice

- Kubernetes CRDs/controllers or autoscaling;
- multi-host placement optimization;
- dynamic Planner expansion of the Worker pool;
- Planner-selected AgentTemplates;
- concurrent Tasks inside one Worker allocation;
- resuming a Planner invocation from its persisted ADK Session or private plan;
- artifact change subscriptions or semantic merge service;
- multiple production Planner strategies before passthrough works end to end.

## Exit question

The slice is successful when the `adk@1` runtime executes different
AgentTemplates without changes to Workflow Scheduler, PassthroughPlanner, A2A
or artifact contracts.
