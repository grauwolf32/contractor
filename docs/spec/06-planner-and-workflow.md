# 06 — Planner and workflow execution

Status: **Draft**  
Depends on: [02](02-domain-contracts.md), [03](03-adk-integration.md), [05](05-artifacts-and-run-state.md)

## Responsibility split

- `RunOrchestrator` owns run commands, scheduler ticks, cancellation and
  recovery.
- `PlannerStrategy` is a Server-local port selected by the exact
  `PlannerStrategyRef` pinned with the immutable RunSpec when the run is created.
- The decomposing strategy uses the current subtask state model and may use ADK
  `Workflow` plus `LlmAgent` to propose `TaskSpec` and dependency changes.
- The static strategy deterministically instantiates a pinned, versioned
  one-or-more-Task template manifest against the accepted RunSpec. Passthrough
  is its constrained specialization: it proposes exactly one root TaskSpec and
  zero edges without semantic decomposition or a Planner model call.
- `PlannerStateMachine` validates and applies TaskSpec/edge proposals to a pure
  domain `PlannerRunState`.
- The deterministic scheduler owns readiness, retry and terminal-state
  derivation. Only after a TaskSpec is committed does it create an Attempt and
  materialize that Attempt's immutable `WorkerJob`.
- `RunStateStore` commits RunState, projection and outbox atomically.
- `AttemptIoWorker` claims execute/cancel outbox commands, resumes observation
  of mapped A2A tasks and is the only caller of `WorkerGateway`; that gateway is
  the only route to Worker execution or remote cancellation.

Every scheduler/recovery turn loads and verifies the exact pinned `RunSpec` and
`PlannerStrategyRef` from one exact RunState version. The strategy is invoked
only when committed state requires initial plan formation or an authorized plan
revision that adds a new, superseding Task definition. A
decomposing-strategy invocation may start a fresh ADK Workflow; a
static or passthrough strategy deterministically instantiates its pinned
manifest and is not reinvoked after that plan is committed. Ordinary readiness,
retry and terminal-state derivation remain deterministic scheduler operations.
ADK event replay is not used for durable completion or dispatch decisions. A
Planner LLM call, when the selected strategy makes one, is allowed before CAS
and may repeat after a crash. A2A, tool and sandbox execution cannot start from
an uncommitted TaskSpec proposal.

Before any Planner LLM call, Orchestrator durably reserves the maximum allowed
invocation budget under a stable `planner_invocation_id` and exact RunState
version. Actual Proxy usage settles the reservation. An unresolved invocation
is charged conservatively until reconciliation, so a pre-CAS crash cannot evade
the run budget. Static and passthrough ticks make no Planner LLM reservation.

The decomposing binding may adapt or refactor the existing Contractor Planner
and its subtask-state logic in the Server process. That does not preserve its
old runtime authority: proposed subtasks are translated to typed TaskSpecs and
edges, durable state is committed only through `PlannerStateMachine`/RunState
CAS, and every Worker call is scheduled through the outbox and A2A
`WorkerGateway`. Legacy in-process Worker builders or `TaskRunner` dispatch are
not alternate paths.

## Planner strategies and execution granularity

The decomposing, static and passthrough implementations are alternative bindings
of the same `PlannerStrategy`; they are not different distributed protocols.

The decomposing strategy proposes immutable TaskId/TaskSpec definitions and
edges for the Server-owned subtask DAG. The static strategy validates its pinned
manifest's template-local node/edge keys, binds the accepted RunSpec and applies
the pinned ID-derivation version to produce exact run-specific TaskIds,
TaskSpecs and edges. Passthrough applies the same mapping to the RunSpec-derived
high-level objective and must yield exactly one root TaskSpec and zero edges. No
Planner strategy creates an Attempt, WorkerJob, budget reservation, fence or
dispatch envelope.

After a TaskSpec-containing plan version commits, the deterministic scheduler
computes readiness from that committed version. A dispatch CAS creates a new
Attempt, reserves its concrete budget, derives its effective deadline and
materializes one immutable `WorkerJob` from the exact TaskSpec before appending
the execute outbox record. Transport redelivery reuses that job byte-for-byte;
a policy retry creates a new Attempt and WorkerJob for the same immutable
TaskId/TaskSpec. Revised work is not a retry: it is a newly identified Task with
an explicit `supersedes_task_id` relation.

The receiving Worker Agent may directly execute the objective or alternate
between planning and working using Agent-private state. Those phases do not
feed the Server Planner state machine. The Server accepts only normalized
best-effort `WorkerProgress`, artifacts, usage and the terminal `WorkerResult`
for the boundary Attempt. Progress never authorizes a state transition.

A TaskSpec may use normal capability routing or an authorized exact
Agent/capability/implementation constraint narrowed from the RunSpec. The state
machine validates that constraint; the dispatcher still verifies a current
healthy registration and compatible `job_contract`, then commits a
`WorkerDispatchEnvelope` containing the exact WorkerJob and selected capability/
implementation identity before sending it through `WorkerGateway`. No Planner
strategy performs network I/O or selects an Agent from mutable registry state.
If an Attempt already has a committed dispatch envelope, recovery reuses it and
never resolves another Agent; loss or unavailability follows Attempt policy,
and any reassignment creates a new Attempt.

## Planner commands

The Planner may propose only typed plan commands such as:

- add a new TaskId permanently bound to an immutable `TaskSpec` whose policies
  narrow the accepted RunSpec;
- add a dependency edge;
- mark planning complete;
- add a new TaskId/TaskSpec with `supersedes_task_id` after an authorized
  revision decision; rebinding or reopening the prior TaskId is forbidden.

The deterministic state machine validates every command before CAS commit.
Readiness, Attempt creation, retry and terminal Run derivation are scheduler
transitions and are not Planner commands.

The decomposing Planner may preserve a tool-shaped LLM interface for proposing
subtasks, but those handlers are pure typed-command constructors. They have no
WorkerGateway, ToolInvoker, external-I/O or repository authority; nothing they
construct becomes durable or executable before deterministic validation and a
winning RunState CAS.

## Requirements

- **PLN-001** — A plan MUST be a directed acyclic graph. Cycles, unknown
  dependencies, duplicate task IDs, rebinding a TaskId to another TaskSpec hash,
  unknown supersedes targets and cycles in the supersedes relation are rejected
  before execution.
- **PLN-002** — A task becomes `ready` only when every required predecessor has
  an explicit committed resolution that satisfies the dependency edge and every
  required output binding is available. A reused predecessor satisfies an edge
  only through its validated exact output refs; a skipped/not-applicable
  predecessor satisfies only an edge whose declared policy permits it. Each
  dependency-bound TaskSpec input references exactly one committed edge ID; that
  edge is the sole source of predecessor/output and successor/input identity.
- **PLN-003** — Planner output MUST be parsed into typed commands; arbitrary
  mutation of session state or RunState by model-generated code is forbidden.
- **PLN-004** — Planner MUST NOT construct or invoke a Worker `LlmAgent`
  directly and MUST NOT propose an Attempt-specific `WorkerJob`. It proposes
  immutable TaskSpecs only. After the TaskSpec is committed, only the
  deterministic scheduler may create an Attempt and materialize the WorkerJob
  in a winning CAS with its budget reservation and dispatch outbox record.
- **PLN-005** — Every scheduler decision MUST be reproducible from one loaded
  RunState version and committed repository data.
- **PLN-006** — Retry policy MUST specify maximum Attempts, retryable error
  classes, deadline and backoff. A retry creates a new immutable Attempt ID and
  WorkerJob for the same exact TaskId/TaskSpec. A revised TaskSpec is new work
  under a new TaskId, not a retry of the prior Task.
- **PLN-007** — Completed tasks MUST NOT be executed again after Server restart.
- **PLN-008** — Cancellation prevents new dispatches, requests cancellation of
  active Attempts and reaches a defined terminal state after a bounded drain.
- **PLN-009** — LLM token/cost/time budgets MUST be checked before each Planner
  or Worker dispatch and persisted in RunState.
- **PLN-010** — Only the CAS winner may emit dispatch outbox work for the new
  state version.
- **PLN-011** — Outbox workers MUST claim records with a renewable/expiring
  claim, retry with the same Attempt/idempotency identity, and acknowledge only
  after durable Server-side assignment/task mapping or a terminal response is
  recorded in `ServerAttemptRepository`.
- **PLN-012** — A crash after CAS but before A2A send MUST leave a claimable
  outbox record; a crash after send but before acknowledgement MUST recover by
  stored endpoint/context or redeliver the same eligible execute identity and
  rely on Agent deduplication.
- **PLN-013** — A pre-CAS ADK Workflow crash in the decomposing strategy MUST be
  retried as a fresh turn from the same RunState version and MUST NOT infer
  completion from ADK events.
- **PLN-014** — Projection and RunState version mismatch MUST block dispatch and
  trigger deterministic projection repair from the verified artifact.
- **PLN-015** — Outbox ordering MUST be deterministic, but an execute record's
  producing RunState version is provenance rather than a latest-version lock.
  Before send, dispatcher MUST verify the exact current Attempt, WorkerJob hash,
  fence, deadline and cancellation/supersession eligibility. Unrelated later
  CASes MUST NOT suppress it; an invalidating CAS MUST explicitly fence it.
  Committed `cancel` records remain deliverable until acknowledged, the remote
  task is terminal/proven absent, or their own deadline expires.
- **PLN-016** — Every pre-CAS Planner LLM call MUST have a durable budget
  reservation and stable invocation identity; crash recovery MUST settle or
  conservatively charge unresolved reservations before another call.
- **PLN-017** — Run cancellation MUST append durable cancel commands in the same
  CAS transaction that records cancellation, fences pending execute records and
  seals/revokes active Attempt operation-start gates; no direct best-effort call
  from Control Plane or Orchestrator may bypass the outbox.
- **PLN-018** — Every `PlannerStrategy`, including static and passthrough, MUST
  return typed immutable TaskId/TaskSpec and edge proposals to
  `PlannerStateMachine`; only that deterministic state machine may validate and
  apply plan transitions, and only `RunStateStore` may own their commit.
- **PLN-019** — Dispatcher MUST resolve the routing constraint embedded in the
  exact TaskSpec/WorkerJob and persist a `WorkerDispatchEnvelope`, endpoint and
  A2A message/context identity and atomically open the previously closed
  operation-start gate for that exact registration/deadline before network send.
  It MUST call
  `WorkerGateway` with that committed envelope and reject result provenance that
  does not match it. A committed envelope MUST be reused for every delivery of
  that Attempt and MUST NOT be retargeted; a different registration requires a
  new Attempt. Dispatcher MUST persist terminal `WorkerResult`/error/usage
  and seal/revoke the Attempt operation-start gate in the same transaction before
  acknowledgement. The next scheduler CAS validates the fence and exact terminal-
  ingest non-open gate version before staged-ref promotion and Worker-budget
  settlement. A terminal response may be accepted directly from `leased` without
  fabricating a running event.
- **PLN-020** — Cancel delivery MUST use the committed assigned endpoint and
  stored/recovered A2A task identity. It MUST NOT resolve a new healthy Agent
  through `WorkerRegistry`.
- **PLN-021** — Worker dispatch MUST durably reserve its maximum model budget;
  accepting terminal usage MUST settle that reservation atomically with the
  corresponding RunState transition.
- **PLN-022** — Acknowledging execute after durable A2A mapping MUST NOT abandon
  result ingestion. `AttemptIoWorker` MUST resume stream/query for every
  nonterminal Server Attempt after its own or Server restart without sending a
  new execute.
- **PLN-023** — For ambiguous execute delivery, an empty task lookup MUST NOT
  acknowledge cancel until the assigned Agent durably acknowledges the
  Attempt-control cancellation tombstone. Tombstone and execute mapping MUST
  serialize so a delayed original request cannot start later.
- **PLN-024** — Attempt I/O MUST use a context unique to the authenticated
  caller/Attempt, exhaust paginated task listing and match exact Attempt/message
  metadata. Multiple matches or incomplete listing MUST fail closed.
- **PLN-025** — The exact `RunSpec` and `PlannerStrategyRef`, including verified
  implementation/configuration identity and hashes, MUST be persisted in the
  initial RunState and reused for every scheduler/recovery turn of that run.
- **PLN-026** — Static planning MUST deterministically instantiate and commit
  exact run-specific TaskIds/TaskSpecs/edges from template-local node/edge keys,
  the pinned manifest/ID-derivation version and accepted RunSpec. Repeating the
  instantiation MUST be byte-identical. Passthrough MUST be exactly the
  one-root-TaskSpec, zero-edge specialization and MUST make no Planner LLM call.
  A retry creates a new Attempt and WorkerJob for that root TaskId/TaskSpec, not
  a new plan or Agent-private child Tasks.
- **PLN-027** — Decomposing, static and passthrough planning MUST produce
  TaskSpec/edge proposals only. Their committed Tasks MUST traverse the same
  deterministic scheduling, CAS, budget, outbox, assignment-envelope,
  `WorkerGateway` and result-acceptance path.
- **PLN-028** — No Planner strategy may infer authoritative Task state from an
  Agent's private planning/working phases. Only a valid boundary `WorkerResult`
  or defined Server evidence may transition its Attempt; only deterministic
  scheduler rules may derive Task and Run state from that evidence.
- **PLN-029** — Every capability or exact Agent/capability/implementation routing
  constraint MUST come from the accepted TaskSpec, be no wider than RunSpec
  policy and be revalidated against current health, identity, capabilities and
  `job_contract` before assignment. It MUST NOT bypass the outbox,
  `WorkerDispatchEnvelope` commit or fencing protocol.
- **PLN-030** — Readiness, retry eligibility and terminal Run outcome for every
  strategy MUST be derived by deterministic scheduler rules and the accepted
  RunSpec completion/failure policy. An executed Task success is accepted only
  with its current fence, structurally valid declared outputs, staged-ref
  promotion and usage settlement; a reused resolution instead requires the
  exact validated evidence in `PLN-032`. Static/passthrough completion MUST NOT
  require reinvoking the strategy after its plan is committed.
- **PLN-031** — Every Attempt's Worker budget, deadline, cancellation and sandbox
  policy MUST cover all of its Agent-internal planning and working. Internal
  activity MUST report aggregate usage and MUST NOT widen those constraints.
- **PLN-032** — Conditional, reused and not-applicable work MUST be represented
  by explicit committed Task resolution and exact evidence in RunState. Merely
  finding an artifact with a familiar logical name MUST NOT be interpreted as a
  cache hit or successful predecessor.
- **PLN-033** — Every dependency edge/join MUST declare deterministic
  satisfaction rules for executed, reused, skipped/not-applicable, failed and
  cancelled predecessors. The scheduler MUST evaluate those rules only from the
  exact committed RunState and artifact refs. A predecessor's terminal
  state/resolution is independent of those outgoing rules; the rules determine
  only successor readiness or branch outcome.
- **PLN-034** — An adapter around retained decomposing-Planner/subtask logic
  MUST implement the same `PlannerStrategy` typed-command contract and MUST NOT
  retain direct Worker invocation, legacy durable scheduling or mutable
  deployment-default authority. Its Worker work reaches only the normal
  scheduler/outbox/A2A path.
- **PLN-035** — Before retrying a TaskSpec that permits external effects, the
  scheduler MUST first CAS-seal/revoke the old Attempt's Server-owned
  operation-start gate, then read the sealed durable effect evidence and apply
  the tool protocol from spec 08. Effect-intent insertion MUST serialize with
  the exact open gate version. Outcomes for intents committed before closure MAY
  finalize monotonically, so the scheduler MUST reconcile/reload them before
  deciding; a snapshot-only pre-seal check is insufficient. Missing,
  conflicting or unconfirmed evidence blocks automatic retry. Server has
  read-only evidence access and MUST NOT manufacture a safe outcome in
  Agent-owned records.
- **PLN-036** — A permanent activation/binding/policy error found before Attempt
  creation MUST terminally fail the Task with exact evidence. A created Attempt
  whose deadline expires or whose assignment/dispatch validation fails
  permanently MUST become failed, revoke its still-closed operation-start gate
  and drive the accepted Task retry/failure policy without fabricating a
  WorkerResult.
- **PLN-037** — Accepting `supersedes_task_id` MUST atomically cancel an eligible
  nonterminal prior Task with reason `superseded`, fence its pending/active
  execute and Attempt and seal/revoke its operation-start gate. An
  already-terminal prior Task remains unchanged; no new Task state is introduced
  and its resolution cannot be transferred to the replacement.
- **PLN-038** — Tool-shaped subtask operations exposed to a decomposing Planner
  LLM MUST be pure typed-command constructors. They MUST NOT receive
  `WorkerGateway`, `ToolInvoker`, external-I/O clients or repository mutation
  authority; validation plus a winning RunState CAS precedes every durable or
  executable effect.
- **PLN-039** — Result acceptance MUST consume terminal evidence whose Server
  ingestion atomically stored result/error/usage and sealed/revoked the Attempt
  operation-start gate. It MUST pin that exact non-open gate version; an open,
  absent or changed version blocks promotion and settlement.

## Dispatch outbox

After its source TaskSpec is committed and the Task becomes ready, a scheduler
CAS creates the Attempt-specific WorkerJob and appends an immutable `execute`
`DispatchOutboxRecord`. Cancellation CASes append immutable `cancel` records.
Workers claim due rows
using PostgreSQL row locking (`FOR UPDATE SKIP LOCKED` or an equivalently tested
mechanism), set a bounded claim lease and deliver the unchanged command.
The producing `run_state_version` is retained for audit/order reconstruction but
does not require equality with the current RunState version. Before claim and
again before network send, the worker verifies the exact current Attempt,
WorkerJob hash, fence, deadline and cancellation/supersession facts. An unrelated
CAS leaves the record eligible; only an explicit invalidating transition fences
it.
For `execute`, dispatcher reads the immutable routing constraint from the
TaskSpec embedded in the exact WorkerJob, resolves a healthy compatible Agent,
then atomically commits the `WorkerDispatchEnvelope`, endpoint, stable A2A
message/context IDs and operation-start gate opening for that exact
registration/deadline before sending that envelope. The envelope selects the
exact capability and Worker implementation without adding a planning/working
mode or mutating the already committed WorkerJob/outbox payload.
For `cancel`, dispatcher never consults `WorkerRegistry`; it targets that
committed endpoint and stored or context-recovered A2A-server-generated
`a2a_task_id`.
Delivery retries never create a new Attempt ID.

On successful A2A submission, dispatcher adds the A2A-server-generated
`a2a_task_id` to the Server Attempt record before acknowledgement. If the first
response is lost, it lists tasks by the committed context ID on the same
endpoint; it does not guess an `a2a_task_id`. A terminal response/error and
normalized usage are stored there with the resulting operation-start-gate seal/
revocation version in one transaction before acknowledgement. This Server
outbound record is distinct from the Agent's inbound task-to-execution mapping.

An execute record may be acknowledged once its outbound task mapping is durable.
`AttemptIoWorker` still owns observation of that nonterminal mapping. It resumes
the stream when supported or polls `GetTask`; on restart it scans all nonterminal
Server Attempt records before accepting new dispatch work. A terminal result is
ingested exactly once by Attempt/result identity.

A cancellation CAS fences pending execute records and appends cancel commands
for active or ambiguously delivered Attempts. A cancel without a known
`a2a_task_id` first durably installs the tombstone through the private
Attempt-control HTTPS endpoint on the assigned Agent, then exhausts the context
lookup: it cancels the one exact recovered task or records that no remote task
can start, but never sends execute. An empty/incomplete lookup without a
tombstone acknowledgement remains ambiguous and retryable; multiple exact
matches fail closed and quarantine the Agent. Only the Attempt I/O worker may
turn committed cancel commands into `WorkerGateway.cancel()` calls. Later
RunState versions do not fence an unacknowledged cancel command before its own
deadline.

## Recovery algorithm

1. Load the run projection and current RunState ref.
2. Verify/upcast the artifact, exact RunSpec, PlannerStrategyRef and TaskSpecs,
   then check the projection-version invariant; repair the projection before
   continuing when it is stale.
3. Reconcile active Attempt leases, ambiguous A2A mappings and committed results,
   then CAS deterministic result/loss/cancellation transitions and seal/revoke
   the affected Server-owned operation-start gates atomically.
4. For an effect-capable retry candidate, only after that gate closure commits,
   read the sealed effect evidence. Permit preexisting committed intents to
   append monotonic outcomes, reconcile/reload until the bounded policy decides,
   and CAS retry eligibility or a blocked/failure outcome.
5. Invoke the exact pinned Planner strategy only if the resulting committed
   state requires initial plan formation or an authorized revision that adds a
   new TaskId/TaskSpec with `supersedes_task_id`. Static/passthrough initialization
   is skipped when its manifest/root TaskSpec is already committed; only the
   decomposing ADK binding may start an ADK Workflow.
6. Validate immutable TaskId/TaskSpec bindings, supersedes relations, edge-ID
   input references and DAG rules, then CAS accepted proposals without creating
   a WorkerJob or execute outbox record and reload the winning committed version.
7. Compute explicit conditional/reuse resolutions, dependency/join satisfaction,
   exact resolved inputs, readiness, retry eligibility, predispatch failures and
   terminal outcome deterministically from that committed version.
8. CAS scheduler transitions. For every newly dispatched/retried Task, create a
   new Attempt with a closed operation-start gate, reservation, fence and
   Attempt-specific WorkerJob from its exact TaskSpec and append the execute
   outbox record in that transaction.
9. `AttemptIoWorker` resumes nonterminal mappings, then independently claims and
   delivers committed outbox records using their committed dispatch identity.

## Acceptance

1. Decomposing DAG fixtures cover conditional, fan-out/join and dynamic
   extension; static manifests cover one and many TaskSpecs with byte-identical
   Task/edge IDs across reinstantiation; passthrough accepts exactly one root
   TaskSpec and zero edges.
2. A cycle, missing dependency and missing/duplicate/mismatched edge-ID input
   binding fail before any Worker dispatch.
3. Server is killed after result commit but before the next scheduler tick;
   restart does not repeat the completed task.
4. Two concurrent scheduler ticks cannot dispatch two active Attempts for one
   TaskId.
5. Cancellation during planning, dispatch and Worker execution produces the
   documented monotonic transitions.
6. Crash after CAS/before send eventually delivers the committed Attempt once
   logically; crash after send/before ack redelivers the same Attempt and Agent
   starts no duplicate execution.
7. A decomposing-strategy crash after ADK session-event append but before CAS
   causes a fresh planning turn and no dispatch from the uncommitted TaskSpecs.
8. An unrelated newer RunState version does not suppress a current execute row;
   explicit cancellation/supersession or a stale Attempt/job/fence does, while an
   already committed cancel row remains deliverable under its own deadline.
9. Projection-version corruption blocks dispatch, repairs from RunState and
   resumes without changing the plan.
10. Crash during a Planner LLM call preserves its reservation; retry cannot
    exceed the run budget or silently discard already reported Proxy usage.
11. Crashes before and after cancel-command delivery retain one committed cancel
    identity, durably install the Agent tombstone and eventually invoke
    `WorkerGateway.cancel()` without reopening a fenced execute command.
12. Server crashes after Agent accepts execute but before returning
    `a2a_task_id`;
    recovery finds it by endpoint/context, persists the terminal result before
    ack and promotes staged refs exactly once in the next fencing-checked CAS.
13. Cancellation of an Agent whose heartbeat expired still targets the committed
    endpoint/`a2a_task_id` identity and never routes the cancel to another healthy
    Agent.
14. Server restarts after execute acknowledgement but before terminal response;
    Attempt I/O resumes the stored task without a new execute and ingests its
    terminal result exactly once.
15. Decomposing, static and passthrough proposals contain TaskSpecs/edges only.
    Their plan CAS creates no Attempt-specific WorkerJob or execute outbox row;
    a later scheduler CAS creates both from the winning committed TaskSpec.
16. Passthrough creates one root Task. Its first Attempt and a policy retry use
    different immutable WorkerJobs derived from the same exact TaskSpec, while
    transport redelivery reuses byte-identical job data for one Attempt.
17. The passthrough root job succeeds when its Agent alternates planning and
    working internally; no mode flag or private subtask becomes part of a Server
    Task, Attempt or RunState transition.
18. An exact-target TaskSpec dispatches only through a committed
    `WorkerDispatchEnvelope` selecting the configured healthy, compatible
    capability/implementation whose transaction also opens the closed
    operation-start gate for that exact registration/deadline; unavailable or
    incompatible targets follow the declared policy without silently routing
    elsewhere.
19. Restart after a static or passthrough plan commit skips strategy invocation,
    recovers its exact pinned RunSpec, PlannerStrategyRef and TaskId/TaskSpec
    bindings and does not duplicate a Task or WorkerJob.
20. Passthrough completion requires no second strategy invocation; all internal
    model/tool usage settles against the root Worker reservation and exceeding
    its budget or deadline stops the Attempt.
21. Static conditional fixtures commit explicit executed, reused and
    skipped/not-applicable resolutions. Join outcomes are deterministic, reused
    outputs name exact validated artifact refs, and mere artifact-name existence
    never causes a cache hit or satisfies an edge.
22. A retained decomposing-Planner adapter preserves a representative subtask
    graph but has no direct Worker hook: a gateway spy observes work only after
    typed-command validation, RunState CAS and execute-outbox commit.
23. A lost effecting Attempt is retried only when the read-only evidence
    read after the old Attempt's gate-seal CAS proves idempotent/reconciled
    safety. An intent cannot insert after closure; a pre-closure intent may
    finalize monotonically and is reloaded before the decision. Missing,
    changing or unconfirmed evidence blocks retry, and Server cannot update it
    through the reader.
24. Rebinding a TaskId to a changed TaskSpec is rejected; an authorized revision
    creates a new TaskId with an exact supersedes relation, atomically cancels an
    eligible nonterminal prior Task with reason `superseded`, fences its
    execute/Attempt/gate and cannot accept a stale result as the new Task. An
    already-terminal prior Task remains unchanged.
25. A permanent input/policy failure before Attempt creation terminals the Task;
    permanent assignment failure or deadline expiry terminals a created Attempt,
    revokes its closed operation-start gate and deterministically retries or fails
    its Task without a WorkerResult.
26. A predecessor's terminal state/evidence remains unchanged when an outgoing
    edge rejects it; only the successor branch outcome changes under the declared
    edge/join and Run completion policies.
27. A terminal result arriving immediately after assignment transitions the
    Attempt from `leased` and Task from `dispatched` directly to success/failure;
    no synthetic running event is required.
28. A decomposing Planner's tool-shaped subtask fixture can construct typed
    commands, while gateway/tool/external-I/O/repository spies observe no call
    before validation and the winning plan CAS.
29. Race terminal-result ingestion against a new Agent invocation intent. The
    intent either commits first and appears in evidence, or result/usage plus
    gate seal commit first and it performs no provider I/O. The scheduler pins
    that exact terminal-ingest gate version before promotion/settlement.
