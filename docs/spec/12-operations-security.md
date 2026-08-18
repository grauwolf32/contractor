# 12 — Operations and security

Status: **Draft**  
Depends on: [04](04-database-runtime.md) through [11](11-llm-proxy.md)

## Configuration and startup

Configuration is parsed into immutable Server or Agent settings in the process
composition root. Module import performs no `.env` read, model-client creation,
PostgreSQL connection or sandbox startup.

Startup order:

1. parse and validate configuration;
2. construct engine and verify database compatibility;
3. construct repositories, the selected Planner/Worker strategies and only the
   ADK services required by configured ADK-backed adapters, using the shared
   engine;
4. verify A2A/fleet-control mTLS and only the Proxy/framework dependencies
   required by the configured Planner and Worker strategies;
5. for Agent, register the protected runtime incarnation, rotate generation
   credentials and obtain exact continuation/recovery grants;
6. reconcile owned or cleanup-only work under those grants;
7. become ready.

## Failure behavior

| Failure | Required behavior |
|---|---|
| PostgreSQL unavailable | reject new durable operations; bounded retries; no unbounded queues |
| LLM Proxy unavailable | classify retryably and apply persisted budget/retry policy |
| Agent lost | seal/revoke old operation-start gate, expire lease, fence stale result, reconcile evidence, then retry only if eligible |
| Server restart | recover from RunState and run projection |
| Telemetry projection/OTel sink unavailable | degrade visibility only; preserve bounded memory |
| External tool effect unknown | reconcile by the tool protocol or block automatic retry |
| Live-target scope absent/mismatch | deny before external I/O; append redacted audit fact |
| Required audit write unavailable | fail privileged mutation closed; keep denials denied and alert |
| Sandbox cleanup failure | quarantine Agent/workspace and surface an operator error |

## Requirements

- **OPS-001** — Configuration MUST have explicit types, safe defaults and a
  startup validation error for missing required values.
- **OPS-002** — Secrets MUST come from runtime secret sources and be redacted
  from exceptions, ADK sessions, telemetry, A2A payloads and artifacts.
- **OPS-003** — Server shutdown MUST stop submissions, stop new dispatches,
  persist current state, request/observe Attempt drain, flush telemetry and
  dispose the engine within a deadline.
- **OPS-004** — Agent shutdown MUST stop new tasks, drain or cancel each active
  Worker strategy, obtain or report its termination acknowledgement, then
  release/quarantine every recorded owned resource before unregistering and
  disposing process dependencies.
- **OPS-005** — Database backup and restore MUST cover all logical table groups
  including the append-only audit log at one consistent point in time.
- **OPS-006** — Artifact and A2A access MUST follow least privilege. Agents use
  a database role distinct from Server and cannot mutate run-control rows.
- **OPS-007** — Every externally supplied artifact, filename, URL and tool
  argument MUST be validated at its trust boundary.
- **OPS-008** — Dependency lock files, container images and migrations MUST be
  reproducible and versioned together with the release.
- **OPS-009** — Append-only audit records MUST cover run
  submission/cancellation, Agent registration, capability changes, artifact
  access denial, policy denial, dispatch-provenance mismatch, fence rejection
  and unconfirmed external effects. They have explicit authorization, integrity
  metadata and retention independent of diagnostic telemetry.
- **OPS-010** — A runbook MUST document database, Proxy, Agent-loss, stuck-run,
  telemetry-backlog and sandbox-leak recovery.
- **OPS-011** — A separate migration role MUST own DDL. Server/Agent runtime
  roles MUST have no general schema-creation privilege; Agent MUST have no
  `BYPASSRLS` and artifact RLS MUST remain forced for its access path.
- **OPS-012** — Artifact transaction scope (`agent_id`, `attempt_id`) MUST be set
  from authenticated runtime identity, never from an untrusted tool argument or
  arbitrary A2A payload field.
- **OPS-013** — Fleet-control certificates, lease tokens and Server epochs MUST
  be rotatable. Revoked or stale credentials MUST stop new routing without
  granting Control Plane or database privileges. The Agent database-persistence
  capability is distinct from the fleet lease token, rotates per registration
  generation and authorizes only guarded Agent persistence operations.
- **OPS-014** — Server-to-Agent A2A/Attempt-control credentials MUST be distinct
  from operator and database credentials, rotatable, and authorized only for
  the assigned Agent task operations.
- **OPS-015** — A custom/non-ADK Worker strategy MUST run under the same Agent
  identity, database grants, artifact grants, sandbox policy, secret policy and
  outbound-network policy as an ADK-backed strategy for the same `WorkerJob`.
- **OPS-016** — An authenticated cancellation or effective WorkerJob deadline
  MUST close the Agent's scoped execution context, stop or reconcile all of the
  boundary Attempt's internal planning/working and owned resources, and reject
  later local tool, artifact or model operations. Missing termination proof is
  `termination_unconfirmed`, not `canceled`.
- **OPS-017** — Planner and Worker strategy factories, versions and capabilities
  MUST be startup-configured and allow-listed. An untrusted `WorkerJob` MUST NOT
  select a Python module, class, executable or unregistered implementation.
- **OPS-018** — Readiness dependency checks MUST be conditional: passthrough
  with a non-model, non-ADK Worker MUST NOT require ADK or LLM Proxy services;
  enabling a dependent strategy MUST make its missing dependency fail readiness.
- **OPS-019** — Agent Worker strategies MUST receive scoped runtime services,
  including only the central `ToolInvoker`, scoped artifact/model clients,
  sandbox access, cancellation/deadline and usage/progress sinks. They MUST NOT
  receive `WorkerRegistry`, `WorkerGateway`, fleet-control credentials, raw tool
  implementations or a direct path to create Server Tasks/Attempts.
- **OPS-020** — Server fence loss MUST reject stale result acceptance, usage
  settlement and artifact promotion and SHOULD trigger best-effort cancellation.
  A partitioned Agent is not assumed to observe the new fence instantly;
  Agent-scoped grants MUST expire no later than the effective WorkerJob deadline
  and MAY support earlier explicit revocation. Before overlap/retry, Server MUST
  seal or revoke the Attempt operation-start gate so no later model/tool intent
  can reach external I/O even if cancellation has not arrived.
- **OPS-021** — Fencing MUST NOT be treated as rollback or deduplication for
  external tool effects. Effecting tools require a versioned effect class,
  stable logical invocation key and the idempotency/reconciliation protocol from
  spec 08. `external_effect_unconfirmed` MUST block automatic Task retry unless
  that tool protocol subsequently proves retry safe. A failed/lost/cancelled
  Attempt whose TaskSpec permits non-idempotent effects MUST also block
  automatic retry when durable evidence cannot prove whether the effect was
  dispatched; a new fence alone is not sufficient evidence.
- **OPS-022** — A security-sensitive state mutation whose required audit record
  cannot commit atomically or through its durable audit protocol MUST fail
  closed. Failure to record an access/policy denial MUST NOT turn it into an
  allow and MUST make audit health visibly degraded for operators.
- **OPS-023** — Network, exploit and effecting tools MUST default deny. An exact
  immutable `TaskSpec` policy MUST authorize target/environment, operation,
  egress, credential and time scope; `ToolInvoker` and sandbox/network policy
  MUST enforce it independently before external I/O. Planner/model/private Agent
  state cannot widen it, and every decision MUST be audited.
- **OPS-024** — In-process WorkerStrategy adapters MUST be trusted,
  startup-allow-listed deployment code. Model-generated/untrusted code MUST run
  only through sandboxed tools; an untrusted third-party Worker implementation
  requires a separate Agent/process identity and isolation boundary rather than
  relying on scoped Python objects for containment.
- **OPS-025** — A fenced registration's ordinary persistence capability MUST
  fail even through an old live connection pool. Same-nonce continuation and
  replacement-nonce cleanup require exact record-scoped grants. Continuation may
  recover the named already-started durable execution, perform only original-
  policy subordinate operations, stage/submit its result and append monotonic
  evidence while the original fence, open gate and deadline remain valid; it
  MUST NOT start a second Worker execution. After gate closure, continuation is
  limited to exact-reference query/cancel and monotonic finalization of a
  pre-existing record through the evidence window; it cannot make a new provider
  request, publish output or create a WorkerResult. Cleanup MUST NOT adopt/resume
  the execution, initiate a business/effecting operation, stage/publish output
  or submit success. Neither mode may accept/settle usage.
- **OPS-026** — Reuse of a runtime-incarnation nonce MUST be protected by an
  external single-owner mechanism. If exclusion of the prior process is
  unconfirmed, rollout/recovery MUST register a new nonce and accept only cleanup
  authority for prior mappings; an expected-generation registration CAS is
  necessary but insufficient to choose between two live same-nonce processes.

## Acceptance

1. Backup/restore into an empty environment passes the end-to-end smoke run.
2. SIGTERM during planning and during a Worker Attempt follows the documented
   drain sequence without corrupting RunState.
3. Server and Agent database roles fail forbidden SQL operations.
4. Configuration/import tests prove that importing packages performs no I/O.
5. Fault injection for every table row above produces bounded, observable
   behavior and a documented recovery path.
6. A migration job always prepares Contractor schemas and prepares ADK schemas
   only for enabled persistent ADK adapters; configured Server/Agents then pass
   applicable session/artifact operations under no-DDL roles.
7. RLS tests prove that forged A2A/tool scope fields cannot widen the Agent's
   artifact access.
8. A revoked Agent certificate and an old-epoch lease both fail registration or
   heartbeat and cannot place the Agent back into the routing pool.
9. Revoking the Server Agent credential prevents execute/list/get/cancel and
   tombstone installation without granting access to Control Plane,
   fleet-control or PostgreSQL.
10. Replacing an ADK Worker strategy with a custom strategy does not widen its
    database, artifact, secret, sandbox or network permissions.
11. Cancelling a boundary Attempt whose Worker strategy internally alternates
    planning and working terminates that execution tree and rejects every later
    staged result with the stale fence.
12. A passthrough Server and deterministic non-ADK/non-model Worker start and
    finish a run without ADK session or Proxy configuration; enabling an
    ADK/model-backed strategy makes missing ADK schema or Proxy fail readiness.
13. A job containing an unregistered provider/module name is rejected before
    Worker invocation, and the Worker cannot obtain fleet-routing services.
14. During a Server-Agent partition, lease expiry and retry with a new
    Attempt/WorkerJob/fence reject the stale result and promotion; the original
    Agent stops local access at its effective deadline without relying on an
    instantaneous fence notification.
15. Fault injection or Agent loss after an external non-idempotent request may
    have been sent produces a durable or conservative
    `external_effect_unconfirmed` classification and no automatic Task retry.
    An idempotent or reconcilable fixture retries only through its exact stable
    cross-Attempt key/provider-reference protocol.
16. Deleting telemetry projections and external OTel data leaves append-only
    audit queries unchanged; unauthorized principals cannot enumerate audit
    targets or records.
17. SIGTERM while a custom strategy ignores cancellation never reports A2A
    `canceled`: it records `termination_unconfirmed`, quarantines its owned
    resources and removes the affected capability from routing.
18. Exploit/live-target fixtures default deny and send no outside-scope bytes.
    Exact authorized replay/disposable targets work only for the pinned
    operation, egress and time scope; target substitution, redirect, model
    instruction and post-deadline requests fail before I/O and are audited.
19. Model-generated code cannot execute in the Agent process and reaches host
    resources only through sandboxed allowed tools; an untrusted third-party
    Worker fixture is deployed under a separate Agent/process identity.
20. An old live Agent pool cannot impersonate a replacement by copying its
    registration fields. Only proof of the current persistence capability passes
    ordinary writes; a new-nonce recovery grant can report cleanup for its exact
    old record but cannot resume it or produce a newly accepted success. An exact
    same-nonce continuation grant resumes business execution only while the
    original open gate/fence/deadline remain valid; after closure it can only
    query/cancel exact references or finalize pre-existing evidence through the
    bounded window.
21. A concurrent late model/tool start races Server gate closure. The start is
    either durably visible before the final evidence decision or rejected before
    external I/O; there is no snapshot-then-unrecorded-effect window.
22. Two same-nonce registration contenders with one expected predecessor produce
    one new generation. Without external exclusive-incarnation proof, neither is
    allowed same-nonce continuation and recovery uses a new nonce plus cleanup-
    only grants.
