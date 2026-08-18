# 07 — Agent runtime and A2A

Status: **Draft**  
Depends on: [02](02-domain-contracts.md), [05](05-artifacts-and-run-state.md), [06](06-planner-and-workflow.md)

Wire baseline: [A2A Protocol 1.0](https://a2a-protocol.org/latest/specification/).
The compatibility suite requires A2A-server-generated (Agent-generated) task IDs, accepted
client-supplied context IDs and task listing by context for recovery.

## Agent lifecycle

An Agent process hosts an A2A server adapter and a Contractor `AgentRuntime`.
The reference adapter uses ADK, but any A2A 1.0-compatible implementation may
bind the same Contractor DTOs. On startup the process registers a stable
`AgentId`, a protected runtime-incarnation `instance_nonce`, endpoint, protocol
versions and capabilities. A crash restart of the same deployment incarnation
reuses that nonce, registers a new generation and receives continuation grants
only for its exact prior mappings; the rotated persistence capability fences a
zombie old process. Planned replacement, loss of trusted local incarnation state
or concurrent rollout uses a new nonce and cannot adopt prior execution. The
live process renews a heartbeat lease until it begins graceful drain.

`WorkerRegistry` chooses only a healthy Agent whose registered capability
descriptor satisfies the immutable `TaskSpec` carried by `WorkerJob`. After
selection, dispatcher commits the exact `WorkerDispatchEnvelope` from spec 02;
initial routing may be round-robin, but selection remains behind the registry
port.

## Local Worker strategy

`AgentRuntime` owns the protocol-facing invariants: authenticated identity,
deduplication, durable task mapping, dispatch-envelope verification, artifact
grants, fencing data, cancellation, sandbox ownership and terminal result
staging. It verifies that the envelope's selected capability and Worker
implementation ID/version/digest match the current registration and loaded
implementation before it reads input content, acquires resources or invokes an
Agent-local `WorkerStrategy` with the portable `WorkerJob`.

A strategy may directly perform the objective or use its own logic and state to
alternate between planning and working. Those internal phases are not selected
by a `WorkerJob` mode flag and are not Server Tasks. ADK `Runner`/`LlmAgent` is
one strategy adapter; another framework or custom implementation may bind the
same port. All strategies return the same `WorkerResult` and remain inside the
enclosing Attempt's capability, artifact, budget, deadline, cancellation and
sandbox boundaries.

An in-process `WorkerStrategy` adapter is trusted, startup-allow-listed
deployment code. Scoped context prevents ambient application services from
becoming part of its supported API; it is not a security sandbox against
malicious Python in the same process. Model-generated or otherwise untrusted
code runs only through `ToolInvoker` and the sandbox. A third-party Worker that
is not trusted at the Agent-process boundary requires a separate Agent/process
identity and isolation boundary while keeping the same A2A contracts.

The framework-neutral port has an explicit lifecycle. Names are illustrative;
the semantics are mandatory:

```text
start(worker_job, scoped_context, start_operation_id) -> WorkerExecutionHandle
recover_start(start_operation_id, worker_job_sha256) -> StartRecoveryReport
cancel(handle_or_execution_ref, reason, deadline) -> CancellationAck
inspect_or_reconcile(execution_ref, worker_job_sha256) -> ReconcileReport
await_terminated(handle_or_ref, deadline) -> TerminationAck
```

`scoped_context` is process-local and contains only Attempt-scoped artifact
access, the central `ToolInvoker`, an optional model client, sandbox access,
progress publication, usage accumulation, cancellation and the effective
deadline. It is not serialized into `WorkerJob`. `WorkerExecutionHandle`
exposes the normalized progress/terminal-result stream and declares one provider
recovery class:

- `durable` — the strategy returns an opaque execution reference that remains
  inspectable and cancellable after Agent restart;
- `non_resumable` — the strategy itself is process-local, and restart may only
  clean up its persistently recorded owned resources and report interruption.

The first production Worker profile SHOULD declare `non_resumable`. A durable
profile is enabled only when its provider passes start-recovery, reference-based
inspection/cancellation and external single-owner continuation tests; exposing a
nominal durable mode without those capabilities is forbidden. The runtime
instead reports a conservative lost Attempt and applies the accepted retry
policy.

The selected capability descriptor declares the recovery class before
`start()`. Runtime first persists the mapping, exact implementation provenance,
recovery class and deterministic `start_operation_id`; it then invokes `start()`
with a context still closed to model, tool, artifact publication and unrecorded
owned resources. Before creating a subprocess, sandbox, container or external
execution whose identity is returned after creation, Runtime persists a
`resource_operation_id`, immutable creation-spec hash and `creating` intent.
Creation must be idempotent/recoverable by that identity (or use an atomic
provider reservation); the returned reference is attached by mapping CAS before
use. A provider that supports neither cannot claim recoverable resource
ownership, and an ambiguous crash quarantines the capability. Runtime opens the
full scoped context only after the Worker handle/reference is durably attached.

A durable provider must support `inspect_or_reconcile`, cancellation by opaque
execution reference and `recover_start`. The last operation looks up the one
provider execution created by `start_operation_id` after a crash or lost start
response; `absent`, one exact recovered reference and `ambiguous` are distinct outcomes.
If obtaining a provider reference itself starts remote work, provider-side
idempotency plus this lookup is mandatory: merely passing an idempotency key
without a recovery operation is insufficient. A non-resumable provider must not
start externally durable work before its reference can be recorded and must not
claim that in-memory work survived restart. `TerminationAck` positively states
that the strategy and every recorded owned resource are stopped or reconciled
to a terminal state. Absence of that proof is `termination_unconfirmed`, never
successful cancellation.

## Fleet-control protocol

Registration is deliberately separate from Worker A2A traffic. Agent
`AgentRegistryClient` calls the Server `AgentRegistryApi` using a private,
versioned JSON API over HTTPS with mutual TLS:

- `POST /internal/v1/agent-registrations` — register or idempotently recover one
  `(agent_id, instance_nonce)`;
- `POST /internal/v1/agent-registrations/{registration_id}/heartbeat` — renew a
  lease with the next monotonic sequence;
- `POST /internal/v1/agent-registrations/{registration_id}/drain` — immediately
  stop new routing and declare a drain deadline;
- `DELETE /internal/v1/agent-registrations/{registration_id}` — idempotently
  unregister after drain.

The mTLS service identity is mapped to an allowed `AgentId` and capability
allow-list. Registration returns an opaque fleet lease token and a distinct
opaque, per-generation database-persistence capability, plus monotonic lease
generation, Server epoch, heartbeat interval and expiry computed from Server
time. Only the trusted Agent persistence adapter receives the persistence
capability; Worker strategy code never does. The database stores its verifier
and derives writer generation from proof of that capability rather than trusting
registration fields copied into a row or transaction setting. Heartbeat payload
time is diagnostic only; duplicate or lower sequences do not extend a lease. A
new `instance_nonce` for the same `AgentId` fences the prior registration and
ordinary persistence capability.

The Agent permits one registration call in flight per instance. Network retries
reuse one `registration_request_id`; Server returns the same generation and
credential set for that request. A later registration operation uses a new
request ID and supplies the exact expected predecessor generation. Server
atomically compares it with durable current state before incrementing
`lease_generation` and rotating both credentials; one contender wins and a
stale contender receives no current credential or replacement generation. The
Agent accepts only the highest generation for the current Server epoch, so
reordered responses cannot roll it back to fenced credentials.

Reusing an `instance_nonce` for crash continuation also requires exclusive
ownership outside this protocol, such as an orchestrator/local lease proving the
prior process cannot register concurrently. Overlapping rollout, an unconfirmed
zombie or loss of that ownership proof uses a new nonce and cleanup-only
recovery. Registration CAS prevents repeated rotation by a stale contender but
does not manufacture priority between two legitimately live processes sharing a
nonce.

Registrations and leases are persisted in the shared PostgreSQL database. A
Server process restart starts a new `server_epoch`; old-epoch leases become
ineligible for routing. Each live Agent re-registers idempotently with the same
instance nonce and a new registration request ID, receives a current-epoch
lease, then resumes heartbeat. This preserves one logical registration while
fencing stale tokens and old processes. Under the configured recovery policy,
that same runtime incarnation may continue only exact mappings named by its new
generation's continuation grants; their dispatch envelopes are not rewritten,
and the already-started durable execution may resume, stage and submit its result
only while the original fence, open operation-start gate, deadline and policy
remain valid. It cannot start a second Worker execution. Every new assignment
uses the current epoch/generation. A replacement
incarnation with a different nonce cannot adopt or restart them. It may read its
stable Agent identity's prior mappings solely to inspect/terminate recorded
resources and append a registration-scoped recovery report linked to the exact
prior mapping/hash. That report transfers no execution ownership and Server
still applies the original Attempt fence and result-acceptance rules.

## Attempt protocol

1. Server creates an Attempt, its immutable `WorkerJob`, Worker budget
   reservation and fencing token in a committed transaction.
2. For execute only, dispatcher chooses an exact registered capability and
   commits the `WorkerDispatchEnvelope`, endpoint, stable A2A `message_id` and
   accepted client `context_id` before network I/O.
3. `WorkerGateway` sends that versioned dispatch envelope over A2A.
4. Agent deduplicates by `attempt_id`, idempotency key, A2A message ID, exact
   WorkerJob hash and assignment hash.
5. Agent validates the exact `TaskSpec`/`job_contract`, verifies the selected
   capability and implementation provenance against its loaded strategy,
   verifies the resolved-input/grant-set hash, and persists the mapping. Only
   then may it load the exact granted inputs,
   acquire resources required by the sandbox policy and call
   `WorkerStrategy.start()`.
6. Agent publishes immutable output as Attempt-scoped `staged` artifacts. Before
   emitting terminal `WorkerResult`/error it closes the scoped context to new
   model/tool/artifact operations, records the local execution terminal and
   returns the exact refs plus aggregate internal usage.
7. Dispatcher persists the terminal result/error and normalized usage and
   atomically seals/revokes the Server Attempt operation-start gate in one
   transaction before acknowledging outbox delivery.
8. A scheduler turn accepts the result only if the fencing token is current,
   the exact terminal-ingest gate version is non-open and evidence/accounting is
   reconciled; it then promotes exact staged refs and settles Worker usage in the
   RunState CAS.

Streaming events are progress information; only committed result artifacts and
RunState transitions determine correctness.

## Contractor A2A Worker profile

Contractor uses standard A2A operations plus the required, versioned profile
extension identifier `urn:contractor:a2a:worker-profile:v1`. The Agent Card
advertises the extension as required together with streaming/list/get/cancel
support and the media types below; the client opts in using the standard A2A
extension-negotiation mechanism. A different major profile URI is a different
wire contract and is never selected by silent fallback.

Every Contractor structured A2A Part uses the canonicalization profile from
spec 02. Its `metadata` object contains exactly one authoritative digest entry:

`"urn:contractor:a2a:part-digest:jcs-sha256:v1": "<64 lowercase hex>"`

The value is SHA-256 over the RFC 8785 canonical UTF-8 bytes of the Part's
`data` value only. The enclosing Part, its media type, metadata and A2A
Message/Artifact are not included. The `data` value is a JSON object, never a
string containing serialized JSON. Unknown metadata is diagnostic; it cannot
replace this key or contribute authoritative fields. A missing, malformed or
mismatched digest fails before Worker start or result acceptance.

An execute `Message` contains exactly one structured `data` Part with media type
`application/vnd.contractor.worker-dispatch+json;version=1`, containing the
canonical `WorkerDispatchEnvelope`. Progress status messages contain at most one
`application/vnd.contractor.worker-progress+json;version=1` data Part containing
`WorkerProgress`. The terminal result Artifact contains exactly one
`application/vnd.contractor.worker-result+json;version=1` data Part containing
`WorkerResult`. Each such Part carries the required digest metadata above.

A task-associated profile, adapter or runtime failure for which no valid
`WorkerResult` exists is carried in `TaskStatus.message`. That Agent-role
Message contains exactly one structured `data` Part with media type
`application/vnd.contractor.error+json;version=1`, containing one
`ErrorEnvelope` and the same required digest metadata. Its task and context IDs
must match the enclosing A2A Task. The status is `rejected` when work was refused
before Worker start and `failed` for a failure after task acceptance.

A Worker execution outcome, including an ordinary Worker failure or
cancellation, continues to use `WorkerResult` and its result Artifact; the
standalone ErrorEnvelope carrier MUST NOT coexist as a second authoritative
terminal outcome. Before an A2A Task exists, authentication, negotiation and
request-validation failures use the selected A2A binding's standard error
response, which the boundary adapter translates to a local `ErrorEnvelope`.
The profile deliberately defines no second Contractor envelope representation
inside JSON-RPC `error.data`, HTTP error details or gRPC status details.

The A2A `message_id`/`context_id` are the committed transport identities;
duplicate copies inside untrusted metadata have no authority. Text instructions,
raw/file parts and multiple candidate envelopes are rejected rather than
concatenated or interpreted as prompts. The adapter maps standard A2A task
states separately and rejects an outcome/state contradiction. Optional
human-readable status text is diagnostic only and can never supply a job,
result, artifact ref, usage value or error classification.

## Durable task mapping and cancellation

The Contractor A2A executor wraps the configured A2A server adapter. Before
starting `WorkerStrategy` it stores `a2a_task_id`, Attempt/fencing identity,
Agent ID, exact WorkerJob and assignment hashes, selected capability and
implementation provenance, provider recovery class, optional opaque execution
reference, owned-resource set, durable tool-effect/reconciliation record refs,
current state, optional sandbox lease, cancel flag, and the immutable canonical
terminal-outcome record when terminal in `AgentTaskRepository`. The terminal
record includes the exact WorkerResult-or-ErrorEnvelope, normalized usage, A2A
terminal state, profile media type and Part-data digest; it commits with the
terminal mapping state after scoped-context closure and before emission. This
repository owns only
Agent-side inbound mappings; Server Attempt leases and fencing live in separate
Server-only tables. In memory the executor additionally maps the task to the
active execution handle and cancellation signal.

A2A task IDs are generated by the Agent, not the Server. Contractor therefore
uses a stable client-generated A2A message ID and requires its Agent to accept a
stable client context ID unique to the authenticated Server principal and
Attempt. Before the first send, Server stores those IDs and the selected
endpoint. Agent rejects reuse of that context for another Attempt and copies
Attempt/message identity into A2A task metadata.

If delivery is ambiguous before a task ID is received, dispatcher lists tasks
by that context on the same endpoint and follows every `next_page_token` until
the result is complete. Incomplete pagination before the operation deadline
remains ambiguous and retryable. Exactly one task with matching authenticated
Attempt/message metadata is stored and queried/cancelled; zero allows execute
redelivery only while the explicit Attempt/fence/cancellation eligibility in
the current RunState permits it. A merely newer RunState version does not make
the same Attempt ineligible. More than one exact
match is an invariant violation: Server accepts none, fences the Attempt,
quarantines the Agent for routing and best-effort cancels every matched task.

Cancellation when `task_id` is still unknown uses a Contractor-owned control
endpoint co-hosted in the Agent process, not a custom A2A method:

```text
PUT /internal/v1/attempt-cancellations/{attempt_id}
Content-Type: application/json
mutual TLS: authenticated Contractor Server principal
```

The exact request/ack DTOs are defined in spec 02. The endpoint is versioned and
idempotent by `cancel_command_id`; an exact replay returns the same or a newer
tombstone acknowledgement, identity mismatch returns `409`, incompatible
protocol returns `426`, and other errors use `application/problem+json`. Its URL
and supported versions are registered with the Agent assignment.

The handler atomically persists a cancellation tombstone in
`AgentTaskRepository` and acknowledges it without invoking a Worker strategy,
model or tool.

The same unique-key transaction serializes tombstone and execute mapping: if
execute won, it sets `cancel_requested`, returns the task identity and signals
the local cancellation path; Server follows with idempotent standard A2A
`CancelTask` to converge protocol state. If tombstone won, a late execute is
returned as canceled and cannot start. Only after tombstone acknowledgement and
a complete empty task listing may Server record `no_remote_task`. Tombstones
remain through the Attempt deadline plus the maximum delivery/retention window.

On an authenticated cancellation or local WorkerJob deadline, the wrapper
durably records intent, closes the scoped execution context to new model, tool
and artifact operations, calls `WorkerStrategy.cancel()`, and cancels every
recorded sandbox/process resource. It emits terminal A2A `canceled` only after
`await_terminated()` proves that the strategy and all owned resources stopped
or reached a reconciled terminal state. If that proof is absent by the
cancellation deadline, the mapping becomes `termination_unconfirmed`, affected
resources are quarantined, and the Agent stops advertising the affected
capability until reconciliation or operator recovery. Server persists the
failure; only its state machine may mark the Attempt `lost` and revoke the
fencing token. A late success remains staged and cannot be promoted.

Server-side lease expiry or replacement of a fencing token is not an
instantaneous signal to a partitioned Agent. Server invalidates the stale
Attempt for result acceptance and artifact promotion and best-effort sends
cancellation. Agent-local work is stopped by a received cancellation or the
effective WorkerJob deadline. Artifact, tool and model grants are bounded by
that deadline and may be explicitly revoked sooner where their adapter supports
revocation; no Agent is assumed to observe Server fence loss synchronously.

Before Server authorizes another Attempt for a Task that can cause an external
effect, its RunState transaction seals or revokes the old Attempt's Server-owned
`operation_start_gate`. Agent persistence guards serialize every new effect
intent against that gate, so an intent either commits before the seal/revocation
and is included in the final evidence snapshot, or is rejected before provider
I/O. Making the gate non-open does not erase evidence: an already-recorded
invocation may still append only its monotonic terminal outcome/reconciliation
under the narrow late-evidence path. Snapshot hashing without this write gate is
not sufficient retry safety.

After Agent restart, nonterminal durable mappings are reconciled before the
Agent becomes ready. A `durable` execution is inspected through its exact
stored provider reference and implementation provenance; a mapping left in
`starting` without a reference uses `recover_start` and fails closed on an
ambiguous lookup. A `non_resumable` execution is never restarted from mutable
defaults: its orphaned recorded resources are terminated/quarantined and it is
reported as interrupted, or as `termination_unconfirmed` if cleanup cannot be
proved. A replacement nonce appends the recovery report described above rather
than rewriting/adopting the old mapping. By itself that report may drive only
cleanup, `lost` or cancellation. Terminal success/failure requires a
WorkerResult already durable before replacement and independently valid under
the original fence and terminal-ingest gate version. After Server restart,
a stored A2A task ID or the endpoint/context recovery identity permits
query/cancel of already submitted work without issuing a new Attempt.

## Requirements

- **A2A-001** — Every A2A adapter MUST pin and test its protocol binding against
  the Contractor profile. The reference ADK adapter additionally installs the
  compatible ADK `a2a` extra. Core conformance MUST remain runnable against an
  Agent adapter that does not import ADK.
- **A2A-002** — Agent Card/protocol discovery MUST be cached with a bounded TTL
  and invalidated on incompatible responses.
- **A2A-003** — Duplicate delivery MUST attach to or return the existing
  Attempt; it MUST NOT start a second Worker execution or sandbox.
- **A2A-004** — Heartbeat expiry makes an Agent ineligible for new work. Active
  Attempts become `lost` only after the configured lease/grace policy.
- **A2A-005** — A stale Agent result MUST be rejected by fencing-token check
  even if its A2A task reports success.
- **A2A-006** — Cancellation MUST durably record intent, close the Attempt-scoped
  execution context, cancel the active Worker strategy and stop every recorded
  owned resource after grace. If termination cannot be positively acknowledged,
  Agent MUST report `termination_unconfirmed` and Server MUST revoke the fence
  before any retry.
- **A2A-007** — A2A task-to-Attempt mapping MUST be stored in
  `AgentTaskRepository` before execution, including exact job/assignment hashes,
  implementation provenance, recovery class, optional execution reference and
  owned resources. Its terminal state MUST include the immutable terminal-
  outcome replay record; the default in-memory task store alone is insufficient
  for recovery.
- **A2A-008** — Registration and heartbeats MUST be authenticated and MUST NOT
  allow an Agent to advertise capabilities outside its configured allow-list.
- **A2A-009** — Agent shutdown MUST stop accepting new tasks, drain or cancel
  active Attempts, await or report their termination state, flush bounded
  telemetry and dispose its engine within one deadline.
- **A2A-010** — Transport DTOs and errors MUST be translated at the adapter;
  Planner and domain code see only v2 contracts.
- **A2A-011** — Terminal A2A `canceled` MUST NOT be emitted without a positive
  termination acknowledgement for the mapped Worker execution and all recorded
  owned resources.
- **A2A-012** — Agent restart MUST reconcile every nonterminal task mapping
  according to its persisted recovery class, `start_operation_id` and exact execution
  reference, and terminate/quarantine orphaned resources before readiness. A
  replacement nonce MUST append a scoped recovery report rather than adopt or
  rewrite the old mapping.
- **A2A-013** — Server restart MUST reuse a stored `a2a_task_id` or recover it by
  the committed endpoint/context identity for query or cancellation; it MUST NOT
  submit a second Attempt for that mapping.
- **A2A-014** — Server-side fence loss MUST reject a late result and leave its
  artifacts staged even when a partitioned Agent has not yet received
  cancellation. It MUST NOT become a run-visible artifact.
- **A2A-015** — Fleet control MUST use the private versioned HTTPS endpoints
  above with mTLS; it MUST NOT be encoded as an undocumented A2A extension.
- **A2A-016** — Registration MUST be idempotent for
  `(agent_id, instance_nonce, registration_request_id)`. An exact request retry
  MUST return the same lease generation and credential set; a later request MUST
  supply the exact expected predecessor generation and atomically compare it
  before incrementing generation and rotating both the fleet lease token and
  distinct persistence capability. Only one contender may advance; stale
  predecessors receive no current credential. A new nonce for the same Agent ID
  MUST fence the former registration and ordinary use of both credentials.
- **A2A-017** — Heartbeat renewal MUST require the current registration, lease
  token, lease generation, Server epoch and a strictly increasing sequence.
  Lease expiry MUST use Server receipt time, never Agent clock.
- **A2A-018** — Registry state MUST be durable. After Server restart, routing
  MUST remain disabled for old-epoch leases until an Agent re-registers and
  heartbeats under the current epoch.
- **A2A-019** — Drain MUST remove the Agent from new routing immediately while
  applying the declared bounded policy to existing Attempts; unregister MUST be
  idempotent.
- **A2A-020** — The authenticated mTLS identity MUST bind the allowed Agent ID
  and capability set. Spoofed IDs, capability escalation and mismatched
  capability hashes MUST be rejected and audited.
- **A2A-021** — The Agent database role MUST be able to mutate only its scoped
  task-mapping/cancellation rows and MUST have no write grant on Server Attempt,
  lease, fencing, run, budget or outbox rows.
- **A2A-022** — New-task A2A IDs MUST remain Agent-generated. Ambiguous delivery
  MUST recover by the committed endpoint/context identity; execute redelivery
  MUST retain message identity and MUST NOT start a task after cancellation.
- **A2A-023** — Agent-side task states and errors are reports only. Every
  authoritative `failed`, `cancelled` or `lost` Attempt transition and fencing
  change MUST be applied by Server after durable evidence is recorded: a
  terminal report, lease expiry, protocol failure or cancellation deadline.
- **A2A-024** — Dispatcher MUST commit the exact `WorkerDispatchEnvelope` before
  the first send and MUST commit A2A task mapping before acknowledging execute.
  Terminal result/error, usage and the exact operation-start-gate seal/revocation
  version MUST commit in one Server transaction before acknowledging terminal
  delivery. Recovery MUST reuse that envelope and endpoint for the Attempt; it
  MUST NOT retarget the same Attempt to another registration.
- **A2A-025** — Execute, list/get and cancel MUST authenticate the Contractor
  Server principal according to the Agent Card. Agent MUST derive caller scope
  from that transport identity, not message metadata, and MUST not expose one
  caller's task mappings to another.
- **A2A-026** — An unknown-task cancel MUST persist an Agent-side tombstone by
  authenticated Attempt/message/context identity before an empty lookup is
  accepted. Tombstone and execute mapping MUST be serialized by one unique key
  and transaction so no later execute can start.
- **A2A-027** — The Contractor Attempt-control protocol MUST be versioned,
  implemented by the exact private HTTPS binding above, advertised during Agent
  registration and handled before any Worker strategy invocation. It MUST NOT
  be exposed as a non-standard A2A method or leak into Planner/domain
  state-machine APIs.
- **A2A-028** — A2A context ID MUST be unique per authenticated Server
  principal/Attempt. Agent MUST reject reuse for different Attempt/message
  identity and include that identity in task metadata used during recovery.
- **A2A-029** — Context recovery MUST consume all task-list pages and select by
  exact authenticated Attempt/message metadata. Zero, one and multiple matches
  MUST follow the fail-closed rules above; incomplete pagination MUST never be
  interpreted as an empty result.
- **A2A-030** — Attempt-control HTTPS MUST require the assigned Server mTLS
  principal, validate path/body identity and fence stale cancellation tokens;
  its credentials MUST authorize no public, fleet-control or database action.
- **A2A-031** — `AgentRuntime` MUST invoke local work through a framework-neutral
  `WorkerStrategy` lifecycle port with start/recover-start, cancellation by
  live handle or durable reference, inspect/reconcile and
  termination-acknowledgement semantics. Strategy replacement MUST NOT alter A2A
  routes, `WorkerJob`, `WorkerResult`, task mapping, cancellation or fencing
  semantics.
- **A2A-032** — An Agent MUST advertise every supported `worker_kind` and exact
  `job_contract` version. It MUST reject an unsupported `TaskSpec` or an envelope
  whose selected capability/implementation does not exactly match the loaded
  registration before reading inputs, acquiring a sandbox or invoking a model,
  and MUST NOT infer compatibility from a framework label.
- **A2A-033** — A Worker strategy MAY alternate planning and working internally
  without declaring those phases to Server. Internal states and steps MUST NOT
  create or mutate Server RunState, Task, Attempt, budget or outbox records.
- **A2A-034** — All local planning, model, tool and sandbox activity MUST remain
  subordinate to the boundary Attempt. Its deadline and cancellation signal
  apply to the entire strategy, and `WorkerResult.usage` MUST aggregate all such
  activity.
- **A2A-035** — A non-ADK Worker implementation MUST pass the same Contractor
  A2A profile, DTO, deduplication, artifact, cancellation, usage and fencing
  conformance suite as the ADK-backed implementation.
- **A2A-036** — Agent-private progress or intermediate artifacts MUST NOT be
  interpreted as independently retryable or terminal Server Tasks. Only the
  boundary `WorkerResult` can report the Attempt outcome.
- **A2A-037** — The `WorkerDispatchEnvelope` MUST snapshot the selected
  capability and Worker implementation ID/version/digest. Agent MUST verify and
  persist that assignment before execution and return matching
  `execution_provenance`; Server MUST reject a mismatched result before usage
  settlement or artifact promotion.
- **A2A-038** — A strategy MUST declare exactly one recovery class. A durable
  strategy MUST return an opaque execution reference that supports
  inspect/reconcile and cancellation after restart, and MUST recover an
  accepted-but-unrecorded start by the pre-persisted deterministic
  `start_operation_id`.
  A non-resumable strategy MUST be reported interrupted after restart and MUST
  NOT silently restart from current defaults.
- **A2A-039** — Every strategy MUST persist a `resource_operation_id`, immutable
  creation-spec hash and `creating` intent before resource creation. Creation
  MUST be idempotent/recoverable by that identity or use an atomic reservation,
  and the resolved descriptor MUST attach before use. It MUST provide positive
  termination acknowledgement; unknown resource or execution state fails closed
  as `termination_unconfirmed` and quarantines the capability.
- **A2A-040** — Cancellation and the effective WorkerJob deadline are
  Agent-visible stop signals. Server fence loss MUST always invalidate result
  acceptance and promotion, but the protocol MUST NOT assume that a partitioned
  Agent observes the new fence synchronously.
- **A2A-041** — Every ordinary Agent task/effect/model-evidence mutation MUST
  authenticate its writer with the distinct per-generation persistence
  capability and validate the derived registration/nonce/generation/epoch plus
  exact assignment. Row fields, stable AgentId/database login or caller-settable
  transaction context are insufficient. Same-nonce epoch continuation and
  replacement-nonce cleanup MAY affect only mappings named by an explicit
  record-scoped grant. Continuation may resume the already-started durable
  execution and its original-policy subordinate operations, stage output and
  submit its result only while the original fence/gate/deadline remain valid; it
  cannot create a second Worker execution or change dispatch provenance. After
  gate closure it may only perform the bounded exact-reference query/cancel and
  pre-existing-evidence finalization in `A2A-044`, never new provider work,
  output publication or WorkerResult creation.
  Replacement cleanup cannot resume business execution, stage/publish output or
  create/change/submit success; it may replay only a terminal record committed
  before replacement as constrained by `A2A-045`.
- **A2A-042** — Agent MUST verify the WorkerJob's exact resolved bindings and
  transitive input-grant-set hash before reading or materializing any input. A
  missing/extra snapshot blob, mutable alias or grant not usable by the exact
  assigned mapping MUST fail before strategy/model/tool start.
- **A2A-043** — Server MUST seal or revoke a Server-owned Attempt
  operation-start gate before declaring a lost/cancelled/fenced Attempt safe to
  overlap or retry. Agent model/tool intent creation MUST serialize with and
  fail against the non-open gate before provider I/O; already-created records
  MAY only progress monotonically to outcome/reconciliation evidence.
- **A2A-044** — Post-terminal evidence authorization MUST distinguish creating a
  new operation from finalizing an exact pre-existing invocation. During the
  bounded evidence window, a current registration or explicit recovery grant
  MAY append monotonic observations and perform only the exact-reference status/
  read/cancel I/O declared by the reconciliation protocol. It cannot initiate a
  new business/effecting/model operation, change immutable request identity,
  publish artifacts or accept/settle usage.
- **A2A-045** — A replacement process MUST NOT turn inspection into successful
  execution under an old envelope. It may append a hash-linked recovery report,
  terminate recorded resources and expose a result already durable before the
  crash; the report alone can drive only cleanup/loss/cancellation. Server may
  accept success/failure only from that pre-replacement result when the original
  fence and terminal-ingest gate evidence independently validate it; otherwise
  it marks the old Attempt lost and uses a new Attempt, WorkerJob, assignment and
  fence for retry.
- **A2A-046** — Agent Card discovery and every execute MUST negotiate the exact
  required Contractor Worker-profile extension. Execute, progress and terminal
  result MUST use their single structured-data Part/media-type mappings and
  canonical hash above. Missing/duplicate/wrong-kind parts, unsupported major
  profile versions and hash mismatches MUST fail before Worker start or result
  acceptance.
- **A2A-047** — A2A metadata and human-readable text are diagnostic and MUST NOT
  override authenticated transport identity or any field in the canonical
  dispatch/result DTO. Business artifacts cross the boundary as ArtifactRefs;
  inline output bytes and framework-native event/session objects are forbidden.
- **A2A-048** — Each mutation-capable continuation grant MUST have one
  Server-held current mapping-version/snapshot cursor. A guarded mapping mutation
  MUST atomically CAS and advance both mapping and cursor; query does not advance
  it. Cursor replay/mismatch, partial advance or a second unexpired mutating grant
  for the same mapping MUST fail closed. Cleanup reports never rewrite mapping
  ownership.
- **A2A-049** — Before emitting terminal result/error, AgentRuntime MUST close
  the Attempt-scoped context to new model/tool/artifact operations and durably
  mark its local execution terminal. Server terminal ingestion MUST serialize
  result/error and normalized usage with gate seal/revocation; later acceptance
  MUST pin that exact non-open gate version.
- **A2A-050** — Same-nonce continuation MUST require an externally enforced
  single-owner incarnation in addition to registration CAS. When concurrent old
  process exclusion cannot be proven, replacement MUST use a new nonce and may
  receive cleanup-only grants; registration CAS alone MUST NOT be described as
  choosing a legitimate owner between two live same-nonce processes.
- **A2A-051** — Dispatch, progress, result and task-associated error data Parts
  MUST use their exact profile media type and the metadata key
  `urn:contractor:a2a:part-digest:jcs-sha256:v1`. The digest MUST cover only the
  Part's structured `data` value under `contractor-jcs-sha256-v1`. A standalone
  ErrorEnvelope is permitted only when no valid WorkerResult exists; pre-task
  failures remain standard A2A binding errors translated at the adapter.
- **A2A-052** — AgentRuntime MUST atomically persist one immutable terminal-
  outcome record with the mapping's terminal state after local context closure
  and before A2A emission. `GetTask`, redelivery and authorized replacement
  inspection MUST replay its exact canonical DTO, terminal state, media type,
  digest and usage. Provider/session observations MUST NOT synthesize or alter a
  terminal outcome.

## Acceptance

1. Two Agents register different capabilities and matching jobs route correctly.
2. Replaying the same A2A request concurrently starts one Attempt.
3. Killing an Agent expires its lease and safely retries eligible work on a
   second Agent with a new fencing token.
4. A late result from the killed Agent cannot replace the retry result.
5. A cancel request terminates the active tool subprocess and produces one
   terminal Attempt result.
6. Cancellation that exceeds grace revokes the fence, quarantines the sandbox
   and rejects a later success; database audit proves Server, not Agent, performs
   the authoritative transition.
7. When a durable profile is enabled, crash-restarting the same protected Agent
   incarnation rotates its persistence capability, receives exact continuation
   grants, reconciles the orphaned sandbox/provider by stored
   `start_operation_id`/reference and returns a stable terminal/query result; the
   fenced zombie generation cannot write. A `non_resumable` profile instead
   reports interruption and cleans up/quarantines its recorded resources.
8. Restarting Server after A2A submit but before response recovers the
   Agent-generated task ID by committed endpoint/context and never starts a
   duplicate Attempt.
9. Cancellation before Worker-strategy start, during a model stream and during
   a tool subprocess follows the same monotonic state mapping.
10. An exact registration-request retry returns the same generation and
    credential set;
    reordered responses from later generations cannot roll the Agent back, and
    registering a new nonce fences the old process.
11. Replayed, out-of-order and forged heartbeats neither extend a lease nor make
    an Agent routable.
12. After Server restart, an old-epoch heartbeat is rejected, re-registration
    recovers one logical Agent, and no duplicate healthy instance appears.
13. Drain removes the Agent from new routing immediately while existing Attempts
    follow the configured completion/cancel deadline.
14. Agent A can persist its scoped task mappings but cannot read, change or
    delete Agent B mappings, and neither Agent can mutate any Server Attempt,
    lease, fencing or outbox row.
15. Server crashes after Agent accepts execute but before returning task ID;
    cancellation races a delayed original execute. The tombstone/mapping unique
    transaction either stops that exact task or proves no remote task can start;
    no new execute is sent.
16. An unauthenticated or differently authenticated caller cannot create, list,
    query or cancel Contractor Server tasks even with a copied context/task ID.
17. A paginated lookup with duplicate-context corruption reads every page,
    detects multiple exact matches, accepts no result, fences the Attempt and
    quarantines the Agent rather than selecting an arbitrary task.
18. The same Attempt-specific `WorkerJob` fixture for one immutable root
    `TaskSpec` passes the shared protocol, lifecycle and normalized-result-schema
    assertions through an ADK-backed strategy and a minimal non-ADK strategy;
    semantic output bytes need not be identical.
19. A strategy alternates through multiple private planning and working steps;
    Server observes one Attempt, one aggregate usage record and one terminal
    `WorkerResult`.
20. Cancelling or reaching the deadline of that boundary Attempt stops all
    private strategy steps and owned resources before a terminal `canceled`
    response is emitted.
21. An Agent advertising a familiar framework but not the exact `job_contract`
    is rejected for routing, while an Agent using any implementation that
    advertises the semantic contract and required policy capabilities is
    eligible.
22. Replacing an Agent binary between registration and result cannot pass a
    different strategy/build digest as the originally assigned execution.
23. An Agent advertising two implementations for one job contract receives an
    envelope selecting one exact descriptor. Changing its local default before
    start is rejected before any input read, sandbox, model or tool activity.
24. Restart reconciliation inspects a durable provider by its stored opaque
    execution reference. A crash after provider start acceptance but before
    reference persistence recovers exactly one execution by the pre-persisted
    `start_operation_id`; `absent` and `ambiguous` do not silently start
    another. A non-resumable provider is reported interrupted and never
    restarted from a deployment default.
25. A strategy that ignores cancellation or leaves any recorded owned resource
    running yields `termination_unconfirmed`; the Agent emits no terminal
    `canceled` state and stops advertising the affected capability.
26. Partitioning Agent from Server, expiring its lease and retrying with a new
    Attempt/WorkerJob/fence rejects every stale result and promotion. Agent-local
    access stops by its original deadline without assuming instant fence
    notification.
27. Keep an old Agent process/database pool alive while a new nonce registers
    the same AgentId. Copying current registration fields cannot authenticate
    the old persistence capability; new mappings/operations and ordinary
    mutations fail. Only a record-scoped recovery grant can append a monotonic
    cleanup/evidence report, and only the new registration accepts newly
    assigned envelopes.
28. A snapshot envelope whose manifest is valid but whose grant closure omits
    or adds one blob is rejected before workspace, model, tool or Worker start.
29. Race Server retry against an old Agent creating an effect intent. Row-level
    serialization makes the intent either commit before gate closure and appear
    in the final evidence snapshot, or fail before external I/O; no unsafe
    snapshot-then-late-intent interleaving exists.
30. A Proxy/tool response arriving after Attempt loss can monotonically finalize
    its exact pre-existing evidence during the bounded window, but cannot create
    another invocation, overwrite a terminal observation, publish an artifact
    or make stale usage accepted.
31. A replacement nonce reads an old durable mapping, cancels/inspects its exact
    provider reference and appends a hash-linked recovery report, but cannot
    resume business execution or return a newly produced success under the old
    dispatch envelope.
32. Golden A2A fixtures negotiate the required profile and round-trip one
    dispatch, reordered/gapped progress and one result through ADK-backed and
    minimal non-ADK adapters. Text-only, two-envelope, wrong-media-type,
    unsupported-extension and changed-hash inputs all fail before execution.
33. A result with A2A `completed` plus `WorkerResult.outcome=failed`, or with
    inline business output replacing an ArtifactRef, is rejected before usage
    settlement and promotion.
34. Golden A2A fixtures cover dispatch, progress, result and standalone
    ErrorEnvelope Parts. Reordering source object members preserves the digest;
    changing one data value, hashing the whole Part, using the wrong metadata
    key/media type, omitting the digest or supplying both WorkerResult and a
    standalone ErrorEnvelope is rejected at the boundary. Every ADK-backed and
    non-ADK adapter runs these same fixtures.
35. Crash after terminal-record commit but before the A2A response leaves one
    byte-identical result/error replayable by `GetTask`; crash before commit
    leaves no terminal outcome. A replacement can expose the former under the
    original fence/gate evidence but cannot synthesize the latter from provider
    or ADK session state.
