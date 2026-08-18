# 04 — Database runtime

Status: **Draft**  
Depends on: [01](01-architecture-boundaries.md), [03](03-adk-integration.md)

## Topology

v2 uses one physical PostgreSQL database. The following are logical ownership
areas and do not require separate database servers:

- ADK sessions and events for ADK-backed Planner/Worker adapters;
- Contractor run catalog, status projection, dispatch outbox and budget ledger;
- scoped model-invocation provenance/usage evidence, with Server-owned
  acceptance and settlement;
- Server-owned Attempt catalog, leases and fencing;
- Agent-owned inbound A2A task mappings, cancellation state and durable tool
  effect/reconciliation records;
- Agent registrations and heartbeat leases;
- artifact metadata, versions and content;
- append-only security/audit facts with independent retention;
- bounded telemetry rollups, indexes and retained samples (not a mandatory
  row-for-event copy of ADK or OpenTelemetry data).

One database is the initial deployment simplification, not a domain coupling:
all ownership areas are reached through ports and migrations. If the declared
capacity/SLO envelope cannot be met after bounded retention, aggregation and
admission control, ADK diagnostic sessions or optional telemetry are the first
candidates for a separate service/store adapter with its own process-level
connection ownership. RunState, Attempt, budget, effect and audit authority must
not be reconstructed from that optional store after a split.

## Engine and transaction lifecycle

Each Server or Agent process constructs exactly one SQLAlchemy `AsyncEngine`.
When configured, ADK `DatabaseSessionService` receives that engine;
`PgArtifactService`, repositories and the telemetry sink always receive it. A
non-ADK Worker does not construct an otherwise unused ADK session service or
second pool.

An engine is long-lived; an `AsyncSession` and transaction are short-lived.
No transaction may remain open while waiting for an LLM, A2A response,
sandbox command, file upload or telemetry flush timer.

`RunStateStore` is the one exception to repository-local transaction creation:
it opens one Unit-of-Work `AsyncSession` and passes that session explicitly to
the artifact, run-projection, Attempt/fencing, budget and outbox persistence
functions plus required audit appends. Those functions join the caller-owned
transaction and must not open nested independent sessions. Their unrelated
standalone reads/uploads may open their own short session, but never while
participating in that Unit of Work.

## Agent write authentication and operation sealing

The deployment-provisioned Agent database login authenticates the stable
`AgentId`; it does not distinguish two processes that reused that identity.
Fleet registration therefore also issues a random, per-registration-generation
`persistence_capability`, distinct from the heartbeat lease token. Only its
verifier is stored in protected Server registry state. The trusted persistence
adapter proves possession on every generation-sensitive transaction, and a
hardened database mutation function/RLS helper derives registration ID, nonce,
generation and Server epoch from that proof. Copying those values into row data
or a caller-settable transaction variable is not authentication. Runtime code
has no direct table-DML alternative that bypasses the guard.

Normal mapping, artifact, model-invocation, effect-intent and scoped-audit writes
require the current capability and exact assigned Attempt. Same-nonce recovery
after a Server epoch change uses an explicit continuation grant naming existing
mappings. Its guarded mapping mutations atomically advance a Server-held grant
version/snapshot cursor, and it may resume only the named already-started durable
execution while the original fence, open gate and deadline remain valid. A
replacement nonce receives, at most, record-scoped recovery grants that permit
cleanup and append-only reconciliation reports; they do not transfer the old
execution, permit a new business/effecting operation or make a stale result
acceptable.

Each Server Attempt also owns a monotonic `operation_start_gate`. Before loss,
cancellation, registration replacement or retry can overlap old work,
`RunStateStore` seals or revokes that gate through the Server Attempt repository.
Agent mutation functions serialize creation of every model/tool invocation
intent against the same gate before provider I/O. Thus an intent either commits
before the seal/revocation and appears in the final evidence snapshot, or is
rejected. Making the gate non-open does not discard provider evidence: during a
bounded evidence window, the proper current capability or exact recovery grant
may only finalize/reconcile a pre-existing record through monotonic transitions.
It cannot create an invocation, change immutable request/provider identity, or
accept/settle usage.

## Requirements

- **DBR-001** — A process MUST create at most one normal application
  `AsyncEngine` for the Contractor database.
- **DBR-002** — Repositories MUST acquire a fresh `AsyncSession` per unit of
  work and close it on every success, failure and cancellation path.
- **DBR-003** — Pool sizes MUST be configured from a global connection budget:
  `server_pool + agent_count × agent_pool + operational_reserve <= max_connections`.
- **DBR-004** — Pool acquisition and SQL statements MUST have bounded timeouts.
  Exhaustion maps to a retryable infrastructure error.
- **DBR-005** — Custom schema changes MUST use versioned migrations. Startup
  MUST fail readiness when the database is older or incompatibly newer than
  the application.
- **DBR-006** — ADK-owned tables MUST be managed through the supported ADK
  migration path and MUST NOT be included in custom ORM migrations.
- **DBR-007** — Transaction retries MUST be limited to serialization/deadlock
  failures and execute only idempotent transaction functions.
- **DBR-008** — Engine disposal MUST happen after API drain, A2A drain and the
  final bounded telemetry flush.
- **DBR-009** — Database credentials MUST be provided at runtime and MUST NOT
  be stored in artifacts, telemetry payloads or committed configuration.
- **DBR-010** — RunState CAS, projection update, outbox append and any required
  same-owner audit fact plus required Attempt operation-gate closure MUST share
  one `AsyncSession` and one PostgreSQL transaction, with rollback of all
  participants on any failure.
- **DBR-011** — A dedicated migration role MUST prepare Contractor schemas and,
  for deployments enabling ADK-backed adapters, ADK schemas; Server and Agent
  runtime roles MUST operate without general DDL privileges.
- **DBR-012** — Server Attempt/fencing rows, Agent task mappings and Agent
  registrations MUST use distinct logical tables and explicit grants. The Agent
  role MUST have no insert, update or delete privilege on Server run-control,
  Attempt, lease, fencing, budget or outbox rows.
- **DBR-013** — Agent-owned Contractor rows MUST use forced RLS scoped by the
  authenticated per-Agent database login, protected principal mapping and, for
  generation-sensitive operations, proof of the protected per-generation
  persistence capability; a pooled connection MUST fail closed when
  identity/Attempt/capability context is absent and MUST NOT retain prior
  transaction scope.
- **DBR-014** — Agent execute mapping and pre-execution cancellation tombstone
  MUST contend on one database-enforced unique identity and one transaction;
  application-only check-then-insert is insufficient.
- **DBR-015** — Server Attempt and Agent task-mapping schemas MUST enforce
  uniqueness of `(authenticated_server_principal, a2a_context_id)` and reject a
  different Attempt/message identity for an existing context.
- **DBR-016** — Telemetry projection/rollup writes MUST have bounded concurrency
  and a pool-acquisition budget subordinate to correctness traffic. They MUST
  defer or drop optional detail before consuming the connection reserve needed
  by RunState, Attempt, cancellation or artifact operations.
- **DBR-017** — Every Server or Agent engine MUST authenticate with a
  deployment-provisioned database login for exactly that runtime identity. The
  Agent login, mTLS principal and stable `AgentId` mapping is provisioned outside
  registration; an Agent-supplied ID cannot create or change that mapping.
- **DBR-018** — Forced RLS for Agent-owned Contractor tables MUST derive scope
  from the authenticated database login through a protected principal mapping,
  not from a caller-settable session variable. Runtime roles MUST lack
  `BYPASSRLS`, role-switching and arbitrary principal-mapping mutation. Pool
  checkout MUST verify the expected login and reset all transaction-local
  settings. A generation-sensitive mutation MUST additionally prove the
  persistence capability to a hardened database guard; caller-supplied
  registration fields are never proof.
- **DBR-019** — Persisted ADK session history MUST have explicit terminal-run
  retention, partition/index, sensitivity/encryption and maximum-content
  policies. It may feed derived diagnostics through `DatabaseSessionService`,
  but MUST NOT be copied row-for-row into Contractor telemetry or be required
  after authoritative domain/accounting records are finalized. An adapter
  profile MAY disable persistent detail without disabling correctness stores.
- **DBR-020** — Worker strategy and sandbox code MUST NOT receive database
  credentials, `AsyncEngine`, unrestricted repositories or raw ADK-table
  access. Only trusted Agent runtime adapters use the Agent engine and enforce
  the authenticated Attempt/session namespace.
- **DBR-021** — A release MUST declare and load-test one database capacity
  envelope covering concurrent Runs/Attempts, RunState versions/bytes and CAS
  latency, project/artifact bytes, ADK event rows/bytes and model-invocation
  records per Attempt, audit retention, telemetry projection lag, peak pool use
  and operational query latency.
  Exceeding the supported envelope MUST produce admission backpressure or a
  documented topology change, not an unbounded queue inside the shared
  database.
- **DBR-022** — Agent roles MAY create model-invocation evidence only for an
  authenticated assigned Attempt whose operation-start gate is open, and MUST
  NOT accept or settle usage. A bounded late-evidence operation MAY only
  monotonically finalize/reconcile an exact pre-existing invocation. It may use
  the reconciliation protocol's exact-reference status/read/cancel I/O but MUST
  NOT initiate another billed/model request. Server owns acceptance/settlement,
  cross-checks it against `WorkerResult` and records the exact evidence version/
  hash; stable invocation identity and provider-reference uniqueness MUST be
  database-enforced.
- **DBR-023** — Runtime roles MAY append audit facts within their authenticated
  scope but MUST NOT update or delete them. Retention/deletion uses a separate
  maintenance role and bounded partition/chunk operations. Agent audit inserts
  and tool-effect records are RLS-scoped to an exact assigned Attempt; Server can
  query them. New effect intents additionally require an open operation-start
  gate. Audit rows remain immutable; a separate pre-existing effect record may
  change after gate closure only through its documented monotonic
  outcome/reconciliation transitions and never by dispatching new I/O.
- **DBR-024** — Agent mapping, model-invocation, effect and scoped audit writes
  MUST retain immutable assignment provenance, while writer authorization MUST
  be derived from proof of a distinct random per-generation persistence
  capability whose verifier is in Server-owned registry state. Re-registering
  an AgentId with a new nonce/generation MUST make its old ordinary capability
  fail even when the stable per-Agent login remains alive; presenting copied
  current registration fields MUST also fail. Same-nonce continuation or
  replacement-nonce cleanup requires an exact scoped grant. Continuation may
  resume business execution only for the named already-started durable execution
  under its unchanged envelope/fence/open-gate/deadline; after gate closure it is
  limited to the exact-reference control and pre-existing-evidence operations in
  `DBR-022`/`DBR-023`. Cleanup cannot resume business execution or stage/submit
  output.
- **DBR-025** — Server retry reconciliation MAY read Agent-owned effect
  evidence through a dedicated read-only repository/grant, but MUST NOT mutate
  Agent task mappings or manufacture a confirmed effect outcome. The decision
  records the exact evidence version/hash it observed; missing or changing
  evidence remains conservative.
- **DBR-026** — Server loss/cancel/fence/retry transitions MUST seal or revoke
  the exact Attempt's operation-start gate before relying on a final model/tool
  evidence snapshot. The seal/revocation and Agent intent creation MUST
  serialize in the database; snapshot hashing alone is not a concurrency
  barrier. Existing intent outcome/reconciliation updates remain permitted only
  by `DBR-022` and `DBR-023`.
- **DBR-027** — Agent runtime roles MUST use guarded mutation APIs (or an
  equivalently proven RLS mechanism) for generation- and gate-sensitive writes
  and MUST lack direct DML that bypasses those checks. Guard functions MUST use
  fixed object resolution, least privilege and tests proving that forged row or
  transaction-context values cannot select another capability/generation.
- **DBR-028** — The one-database topology remains supported only within a
  versioned measured capacity envelope. A topology split MUST preserve the
  existing repository/session ports and per-process engine ownership, and keep
  authoritative Run/Attempt,
  accounting, effect and audit data independent of optional ADK diagnostics and
  telemetry projections.
- **DBR-029** — A registration, mapping, cancellation, effect or other
  security-sensitive mutation and its required same-owner audit fact MUST share
  a caller-owned transaction. If a future topology prevents that, the owner MUST
  atomically append a durable audit-outbox record and fail closed until its
  delivery contract is satisfied; an after-the-fact best-effort audit call is
  forbidden.
- **DBR-030** — Ingesting a terminal WorkerResult/protocol error and normalized
  usage MUST atomically seal/revoke the exact Attempt operation-start gate and
  record its resulting version. RunState acceptance MUST require that version to
  remain non-open before artifact promotion or usage settlement.
- **DBR-031** — A mutation-capable recovery grant MUST have one Server-owned
  current mapping-version/snapshot cursor. A hardened guard MUST atomically CAS
  and advance mapping plus cursor; query is read-only, replay/mismatch fails, and
  no mapping may have two unexpired mutating continuation grants. Replacement-
  nonce cleanup reports never change mapping ownership.
- **DBR-032** — Registration generation/credential rotation MUST compare the
  exact expected predecessor generation and update durable registry state in one
  transaction. A stale contender MUST NOT rotate or receive current credentials.
  Same-nonce continuation also requires external exclusive-incarnation proof.
- **DBR-033** — Resource creation that can outlive its call MUST first persist a
  stable operation ID, immutable spec hash and creating state. Provider creation/
  lookup/cleanup MUST be idempotent by that ID or atomically reserved; ambiguous
  attachment after a crash remains quarantined rather than silently absent.
- **DBR-034** — Agent terminal mapping state and its immutable canonical outcome
  record MUST commit atomically before terminal A2A emission. The record stores
  the exact result-or-error DTO, normalized usage, task state, profile media type
  and Part-data digest; runtime roles may replay it but cannot rewrite or combine
  it with a second terminal outcome.

## Schema ownership

Custom tables have explicit owners in code. Cross-area foreign keys can
reference stable Contractor IDs, but do not point into ADK internal tables.
ADK session IDs are stored as opaque values.

## Acceptance

1. Repeated API calls and Worker attempts do not increase the number of
   engines or pools.
2. A load test with the configured Agent count stays below the declared
   PostgreSQL connection budget.
3. Cancelling a coroutine during a transaction returns the connection to the
   pool and leaves no idle-in-transaction session.
4. An application restart after migrations retains all committed sessions,
   artifacts, run projections, accounting/effect/audit evidence and retained
   telemetry.
5. A deliberately exhausted pool produces bounded retryable failures rather
   than an unbounded hang.
6. Fault injection after each step of RunState/artifact/projection/outbox write
   leaves either the complete new version or no visible change.
7. Every ADK-backed Server/Agent adapter uses pre-created ADK tables under a
   no-DDL runtime role; a non-ADK Agent starts and completes work without access
   to those tables.
8. Agent A can persist its scoped task mappings but cannot read, update or delete
   Agent B mappings; both fail every attempted mutation of Server Attempt, lease
   and outbox rows, including after pooled-connection reuse.
9. Concurrent execute and tombstone transactions produce one serialized mapping
   whose cancellation state prevents any post-ack Worker-strategy start.
10. Reusing one context for two Attempts fails at both Server and Agent database
    boundaries; paginated recovery never sees valid multi-Attempt context data.
11. Sustained telemetry aggregation and ADK event writes cannot consume the
    configured connection reserve or prevent a RunState/cancellation transaction
    from acquiring a connection within its declared bound.
12. An Agent that declares another AgentId, runs `SET LOCAL` with another scope
    or reuses a pooled connection still cannot read/write the other Agent's
    mappings, staged artifacts or audit rows; the same login/certificate
    mismatch also fails fleet registration.
13. Worker code and sandboxed subprocesses receive no database descriptor or
    credential. ADK-session retention removes old diagnostic history while
    terminal RunState, Attempt usage and audit queries remain unchanged.
14. A representative capacity test stays within the declared database bytes,
    write rate, pool reserve and query-latency bounds; overload rejects or
    delays new work before correctness transactions starve.
15. Agent A cannot alter Agent B's model-invocation evidence or any Server
    settlement field; duplicate stable invocation IDs reconcile to one record
    and a conflicting provider reference fails closed.
16. Server/Agent runtime roles cannot rewrite or delete audit history. Agent A
    cannot append an audit/effect record for Agent B or an unassigned Attempt,
    while the maintenance role expires one bounded retention partition without
    blocking a RunState transaction.
17. Keep an old Agent process and database pool alive while a new instance
    registers the same AgentId. Its old persistence capability and forged
    current registration fields cannot create/mutate mappings or start model or
    effect operations. A new-nonce scoped recovery grant can append only cleanup
    and monotonic observations for named existing records. A same-nonce
    continuation grant can resume its exact already-started execution only while
    the original fence/gate/deadline remain valid; the new generation can create
    unrelated work only for its newly assigned Attempts.
18. Server can gate retry from one exact effect-evidence snapshot but every
    attempted update/delete of the Agent mapping/effect rows through that reader
    fails; an absent or concurrently changing outcome never becomes retry-safe.
19. Race gate closure and retry against an Agent effect-intent insert. Exactly
    one serialization wins: a committed intent appears in the final snapshot,
    or the insert fails before provider I/O. No later intent can invalidate the
    retry decision.
20. A model/tool response after Attempt termination can monotonically seal its
    exact previously created record during the evidence window. Attempts to
    create a new invocation, change immutable identity, overwrite a terminal
    observation or mark usage accepted/settled all fail.
21. A stale process that knows the current registration fields but not the
    current persistence capability fails guarded SQL mutation APIs; direct DML,
    unsafe search-path substitution and caller-settable scope bypasses fail too.
22. A capacity fixture first proves the one-database deployment within its
    envelope. The documented split drill moves optional ADK diagnostic/session
    or telemetry load without changing Run recovery, accepted accounting,
    effect evidence or audit queries.
23. Fault injection between each registration/mapping/cancellation/effect
    mutation and its required audit append commits both or neither (or commits
    the documented audit outbox atomically); no successful mutation lacks a
    durable audit obligation.
24. A terminal-result insert racing an invocation-intent insert commits either
    the intent first and includes it in evidence, or the result/usage plus gate
    seal first and rejects the invocation before Proxy/tool I/O. Acceptance pins
    the terminal-ingest gate version.
25. A continuation grant advances its mapping and Server-held cursor together
    through multiple mutations; stale/replayed cursors and a second mutating
    grant fail, while a cleanup report leaves mapping ownership unchanged.
26. Two registration rotations with the same expected predecessor race and only
    one receives the next generation/credentials. Same-nonce continuation fails
    without external proof that the old incarnation cannot contend.
27. A crash in each resource-create/attach window resolves one resource from its
    precommitted operation ID or quarantines the ambiguous resource; no orphan is
    treated as absent and recreated blindly.
28. Crash after Agent terminal mapping/outcome commit but before A2A delivery
    replays the identical terminal DTO/state/media/digest/usage after restart;
    crash before the atomic commit exposes neither terminal state nor outcome.
