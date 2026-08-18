# 02 — Domain contracts

Status: **Draft**  
Depends on: [00](00-product-scope.md), [01](01-architecture-boundaries.md)

## Identity and envelope rules

`RunId`, `TaskId`, `AttemptId`, `AgentId`, `ArtifactId`, `EventId`,
`PrincipalId` and `TenantId` are opaque strings. Producers generate them once;
consumers treat them as opaque values. The first release may use one fixed
tenant, but tenant/principal identity is still present at authorization
boundaries so it does not have to be inferred from artifact names later.

An accepted `TaskId` is bound exactly once to one verified `task_spec_sha256`.
Neither retry nor plan revision may change that binding. If revised work is
required, the Planner proposes a new `TaskId`/`TaskSpec` pair with an explicit
`supersedes_task_id` relation to the prior Task; the prior Task and all of its
Attempts remain immutable history. In the accepting CAS, an eligible nonterminal
prior Task becomes `cancelled` with reason `superseded`, and its pending/active
execute records, Attempt fence and operation-start gate are invalidated
atomically. An already-terminal prior Task remains in its existing terminal
state; supersession never rewrites its resolution.

All wire DTOs contain `protocol_version`. Timestamps use UTC RFC 3339 and
deadlines are absolute, not relative durations.

## Canonical JSON and SHA-256

The canonicalization profile for this specification is
`contractor-jcs-sha256-v1`:

1. Parse and validate the JSON value.
2. Serialize it with the JSON Canonicalization Scheme (JCS), RFC 8785.
3. Encode the JCS result as UTF-8 with no BOM, trailing newline or other bytes.
4. Compute SHA-256 over those bytes.

Implementations MUST use an RFC 8785-conforming implementation; ordinary
lexicographic `sort_keys` serialization is not an equivalent algorithm. JCS
member ordering, escaping and IEEE-754 number formatting apply exactly as
specified by [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785). Strings are not
Unicode-normalized by this profile. Duplicate object member names and strings
containing unpaired Unicode surrogates are rejected before DTO construction.

Every JSON number MUST be finite and exactly accepted by the RFC 8785 binary64
input model. A field whose value is semantically an integer MUST be in the
inclusive JSON safe-integer range `[-9007199254740991, 9007199254740991]`.
Larger counters, identifiers and quantities requiring exact decimal semantics
MUST be strings with a field-specific canonical format; they MUST NOT be JSON
numbers. `NaN`, positive/negative infinity and implementation-specific numeric
extensions are forbidden. Negative zero follows RFC 8785 and canonicalizes as
`0`.

Unless a DTO definition names a narrower hash projection, a DTO's own
`*_sha256` field is calculated over the complete DTO with only that
self-referential field omitted. Nested digest fields and unknown additive
members remain in the hashed value. An implementation may ignore an unknown
member for behavior under `CON-002`, but it MUST include that member when
verifying or forwarding the original value.

Every field named `sha256` or ending in `_sha256` is encoded as exactly 64
lowercase ASCII hexadecimal characters matching `^[0-9a-f]{64}$`, with no
`sha256:` prefix. Base64, base64url and uppercase hexadecimal are rejected.
Changing the canonicalization or digest algorithm requires a new Contractor
protocol/profile major version.

A minimal canonicalization probe is:

- source JSON: `{"s":"€","b":2,"a":1.0}`
- canonical UTF-8 text: `{"a":1,"b":2,"s":"€"}`
- SHA-256:
  `94c03ee60a01d89f55bf751941d09753491adea86e7d69a1774ac4a42c6ef2ea`

## Core DTOs

### `ArtifactRef`

Required fields: `artifact_id`, `version`, `scope`, `kind`, `media_type`,
`sha256`, `size_bytes`. The reference is immutable and identifies one exact
version.

Publication of a compound artifact MUST atomically record its ordered exact
child `ArtifactRef` dependencies. Those dependency records are immutable and
participate in authorization, grant-closure calculation and retention; a
compound artifact never resolves children through a mutable name or alias.

### `ProjectSnapshotManifest`

Source input crosses the public boundary as artifacts, never as a Server host
path. `ProjectSnapshotManifest` required fields are `protocol_version`,
`snapshot_manifest_version`, `path_canonicalization_version`, an ordered entry
list and `snapshot_sha256`. Each entry contains one canonical relative POSIX
path, an exact blob `ArtifactRef` and a Boolean executable flag. Snapshot v1
accepts regular files only; directories are implicit and empty directories,
devices, symbolic links and hard links are not represented.

Path-canonicalization v1 strictly decodes UTF-8, normalizes to Unicode NFC using
the pinned Unicode 15.1 tables and accepts `/` as the only separator. It rejects
an empty path or segment, leading/trailing `/`, `.`, `..`, backslash, NUL and
any code point in Unicode general category `Cc`. Entries are sorted by the
unsigned UTF-8 bytes of the canonical path. Duplicate paths and collisions under
the key `NFC(default_casefold(canonical_path))`, using the same Unicode 15.1
tables, are rejected. Materialization writes non-executable/executable files
with normalized modes `0644`/`0755`; no other source mode bits survive.

`snapshot_sha256` is SHA-256 over the deterministic JSON serialization from
`CON-001` of a domain-separated object containing the manifest and path-rule
versions plus the ordered entries (`path`, canonical complete blob
`ArtifactRef`, and executable flag), excluding `snapshot_sha256` itself. The
published manifest bytes, including that field, have their separate
`ArtifactRef.sha256`. The manifest is published only as an immutable
`ArtifactRef` of reserved kind `project-snapshot` and media type
`application/vnd.contractor.project-snapshot+json`; generic blob publication
cannot mint that kind.

The upload API may stream blobs in multiple transport chunks, but a blob or
snapshot becomes referencable only after declared size and digest verification
and an atomic publication commit of the manifest plus exact dependency records.
A later Git/object-store importer is another producer of the same manifest
contract; it does not add filesystem paths to `RunSpec`.

### `PlannerStrategyRef`

Required fields: `protocol_version`, `strategy_id`, `strategy_version`,
`strategy_implementation_sha256`, `strategy_config` and
`strategy_config_sha256`. Initial `strategy_id` values are `decomposing`,
`static` and `passthrough`. `static` deterministically instantiates a pinned
one-or-more-task template manifest; `passthrough` is its constrained
one-root-Task specialization/stub.

Strategy configuration contains only inputs that change how the Server forms a
plan, such as a static manifest ref or decomposing-Planner prompt/model policy.
Worker routing, retry, output and budget policy belong to `RunSpec`/`TaskSpec`,
not this reference. If `strategy_config` is an `ArtifactRef`, the artifact
contains canonical versioned configuration JSON and
`strategy_config_sha256` verifies those content bytes. Mutable deployment
defaults and reassignable version labels are not sufficient recovery identity.

### `RunSpec`, `TaskSpec` and routing

`RunSpec` is the immutable accepted run input. Required fields:
`protocol_version`, `run_spec_version`, objective contract/value, expected run
outputs, exact input artifacts, default Worker routing/capability policy, retry
policy, deterministic run completion/failure policy, budget/deadline allocation
and exact tool/model/sandbox policy refs, plus `run_spec_sha256`. Large objective
content is referenced as an `ArtifactRef`.

`TaskSpec` is an immutable unit of planned work. Required fields:
`protocol_version`, `task_spec_version`, `worker_kind`, `job_contract`, typed
objective, expected output bindings, input bindings,
`required_capabilities`, `routing_constraint`, retry policy, Worker-budget
policy, tool/model/sandbox policy refs, a versioned `activation_condition`
(`always` by default), a versioned `reuse_policy` (`never` by default) and
`task_spec_sha256`.

An accepted Task definition contains `task_id`, the exact `task_spec` and hash,
and optional `supersedes_task_id`. RunState permanently binds that TaskId to the
hash. A superseding Task uses a new TaskId; it does not mutate, reopen or reuse
the identity of the prior Task.

Each named TaskSpec input binding is either an exact `ArtifactRef` or a
versioned `DependencyInputRef` containing only an `edge_id`. The referenced
committed edge is the single source of predecessor Task, predecessor output and
successor input identity: its `successor_task_id` must equal the Task being
validated and its `successor_input_binding` must equal the binding name. The
TaskSpec does not repeat a predecessor ID or output name. Before dispatch, every
dependency input must resolve from committed RunState to one exact ArtifactRef.
Conditions and reuse policies are declarative, allow-listed contracts evaluated
by the deterministic scheduler; they are not Planner- or Worker-supplied
executable code.

`routing_constraint` is a versioned `DispatchRoutingConstraint`: capability
matching by default, or an authorized exact Agent/capability/implementation
constraint. It selects a semantic implementation, never an internal
planning/working phase. A Planner may narrow `RunSpec` policy for a Task but may
not widen it.

### Dependency and Task-resolution contracts

A `StaticPlanManifest` is immutable reusable configuration. Required fields are
`protocol_version`, `static_plan_manifest_version`, `id_derivation_version`,
uniquely keyed Task templates, uniquely keyed dependency templates,
objective/input/output mappings, activation/reuse policies, join rules and the
manifest content digest pinned by `PlannerStrategyRef`. A Task template has a
stable `template_node_key`; a dependency template has a stable
`template_edge_key` and refers only to predecessor/successor template node keys
plus their named output/input bindings. Template-local keys are not runtime
TaskIds or edge IDs.

Static instantiation first validates and canonically orders the complete
manifest, then binds its mappings against the exact accepted RunSpec. The pinned
`id_derivation_version` deterministically derives each runtime TaskId/edge ID
from a domain separator, `run_id`, manifest digest and template-local key;
collisions fail closed. Only after all IDs are fixed does the strategy construct
exact run-specific TaskSpecs and `DependencyEdge` values. Repeating
instantiation for the same inputs produces byte-identical Task definitions and
edges. Passthrough uses the same algorithm with one reserved `root` node key and
no dependency templates.

A versioned `DependencyEdge` contains `edge_id`, exact `predecessor_task_id` and
`successor_task_id`, named predecessor output and successor input bindings, and
deterministic satisfaction behavior for predecessor resolutions (`executed`,
`reused`, `skipped`), failures and cancellation. It also contains an explicit
join group/rule where more than one edge feeds a Task. Each dependency-bound
TaskSpec input references exactly one edge, and every edge's successor binding
must reference it; duplicates, unreferenced edges and mismatches fail closed.

`TaskResolution` is immutable evidence that a Task was `executed`, `reused` or
`skipped`. Required common fields are exact `task_id`, its bound TaskSpec hash,
resolution type, resolved inputs, declared outputs and evidence hash. Executed
evidence pins the accepted Attempt/result and output refs. Reused evidence pins
the versioned reuse fingerprint, source provenance and validated exact output
refs. Skipped evidence pins the activation-condition version, exact facts it
evaluated and the deterministic result. Artifact name/existence alone is never
evidence.

### `WorkerJob`

Required fields: `protocol_version`, `run_id`, `task_id`, `attempt_id`,
exact `task_spec` plus `task_spec_sha256`, exact resolved input bindings,
`input_grant_set_sha256`, `workspace`, concrete reserved `budget`, effective
`deadline`, `idempotency_key`, `fencing_token` and `trace_context`.

The deterministic scheduler materializes a new immutable `WorkerJob` for each
Attempt from the accepted `TaskSpec` by adding Attempt-specific identity,
reservation, deadline, workspace, fence and trace data. Transport redelivery of
one Attempt reuses its exact `WorkerJob`; a policy retry creates a new Attempt
and `WorkerJob` for the same Task/TaskSpec.

The same envelope therefore represents a narrow decomposed/static Task or the
passthrough root objective. It contains no `planning`/`working` execution-mode
flag and no ADK prompt, session, Agent or tool object. The receiving Agent
chooses its internal execution strategy.

### `WorkerDispatchEnvelope`

After resolving an Agent, dispatcher commits a `WorkerDispatchEnvelope` in the
Server Attempt record before network I/O. It contains `protocol_version`, the
exact `WorkerJob`, assigned Agent/registration identity, selected
`capability_id`, Worker implementation ID/version/digest and an assignment
generation/hash. The authenticated A2A adapter sends this envelope. It is not
caller-authored and does not modify the already committed WorkerJob or outbox
payload. Once committed it is immutable for that Attempt: delivery recovery
reuses it, while reassignment to another registration requires a new Attempt,
fence, WorkerJob and envelope.

### `WorkerResult`

Required fields: `protocol_version`, `attempt_id`, `outcome`,
`fencing_token`, `execution_provenance`, `output_artifacts`, `summary`, `usage`,
and optional `error`. `execution_provenance` identifies the selected Worker
strategy/capability implementation ID, version and immutable build/configuration
digest; framework name/version is optional diagnostic metadata. `outcome` is
exactly `succeeded`, `failed` or `cancelled`; `lost` is inferred by Server when
a lease expires and is never asserted by an Agent result.

`usage` is a versioned normalized aggregate for the whole Attempt. It includes
model invocation/token/cost totals, tool-call totals by versioned tool contract,
elapsed time and configured sandbox/resource measurements when available.
Unavailable measurements are explicit `unknown`, never silently reported as
zero; provider-native detail may be referenced for diagnostics but is not the
accounting contract.

### `WorkerProgress`

`WorkerProgress` is a framework-neutral, best-effort streaming DTO. Required
fields are `protocol_version`, `run_id`, `task_id`, `attempt_id`, `event_id`,
opaque `source_id`, monotonic per-source `source_sequence`, source timestamp,
closed versioned `kind`, bounded normalized summary and `trace_context`. It may
carry exact staged artifact refs and normalized cumulative usage, both
explicitly provisional.
Kinds describe transport-safe facts such as heartbeat, message, staged-artifact
or usage update; they never expose an Agent planning/working mode or private Task
state.

Consumers deduplicate by event ID, retain the highest sequence per authenticated
source and explicitly mark gaps or reordering; they do not fabricate missing
events. Progress cannot transition Run/Task/Attempt state, promote an artifact,
settle usage or prove completion. Only committed Server records,
`WorkerResult`/failure evidence and RunState CAS are authoritative.

### `WorkspaceRef` and `SandboxLease`

`WorkspaceRef` contains an opaque workspace ID plus exact input artifact refs;
it never contains a Server or Agent host path. `SandboxLease` contains
`attempt_id`, workspace ID, fencing token, owner Agent, creation/expiry times
and the sandbox policy version.

### `DispatchOutboxRecord`

Required fields: `outbox_id`, `run_id`, `run_state_version`, `attempt_id`,
`command_type`, serialized payload hash/reference, state, `claim_owner`,
`claim_expires_at`, delivery count and `next_delivery_at`. `command_type` is
`execute` or `cancel`. An execute payload is an exact `WorkerJob`; a cancel
payload identifies the Attempt, current fencing token, assigned endpoint,
deterministic A2A context/message identity, optional learned `a2a_task_id`, reason
and deadline. States are `pending`, `claimed`, `delivered` and `acknowledged`; a
claim may expire back to `pending` without changing Attempt identity.

`run_state_version` records the CAS that created the command; it is provenance,
not a latest-version lock. A later unrelated RunState CAS does not invalidate an
execute. At claim and again before send, execute eligibility requires that the
exact Attempt remains current and nonterminal, its WorkerJob hash and fencing
token still match committed state, its deadline remains valid and neither run
cancellation nor an explicit scheduler fence has invalidated it. Cancellation,
Task supersession and retry decisions explicitly fence affected execute records.

### Server and Agent attempt records

`ServerAttemptRecord` is Server-owned and contains Attempt state, lease and
fencing token; the Agent assignment; deterministic A2A `message_id` and
client-supplied `context_id` unique to this authenticated caller/Attempt;
optional A2A-server-generated `a2a_task_id`; terminal
`WorkerResult` or protocol failure; the immutable `WorkerDispatchEnvelope`;
normalized usage; the exact `terminal_ingest_gate_version` when terminal
evidence is stored; Server-owned `operation_start_gate_state` (`closed`, `open`,
`sealed` or `revoked`), monotonic `operation_start_gate_version`, optional exact
authorized registration identity and gate `not_after`; and settlement state. An
Attempt is created with a `closed` pre-assignment gate. The dispatcher CAS that
commits its immutable assignment envelope/message/context atomically opens the
gate for that exact registration and no later than the effective job deadline,
before network send. This is safe only because intent insertion also requires
the exact durable AgentTaskMapping, assignment and persistence/recovery
capability. Only a Server transition may open or irreversibly seal/revoke the
gate. Model/tool/effect intent insertion must serialize against the exact open
gate version. Sealing/revoking prevents a new operation from starting but does
not prevent a previously committed intent from recording its monotonic
outcome/reconciliation evidence. A terminal result is committed before its
outbox delivery is acknowledged. Ingesting a terminal result/error and sealing
or revoking its gate are one Server transaction; later RunState acceptance must
observe that exact non-open gate version before promotion or settlement.

`AgentTaskMapping` is Agent-owned and contains authenticated caller/Agent ID,
assigned registration ID, instance nonce, lease generation and Server epoch,
stable `mapping_id`, monotonic `mapping_version`, `attempt_id`, idempotency key,
stable `start_operation_id`, A2A `message_id`, `context_id` and optional
`a2a_task_id`, local execution state, exact WorkerJob hash, selected
capability/implementation provenance, provider recovery classification and
optional opaque execution reference, owned-resource set, durable
tool-effect/reconciliation record refs, optional sandbox lease, cancel/tombstone
state, optional exact `AgentTerminalOutcomeRecord` and timestamps.
`mapping_identity_sha256` covers the immutable caller, assignment, Attempt,
message/context, WorkerJob and start-operation identity;
`mapping_snapshot_sha256` covers the deterministic complete record at its stated
mapping version. Every mutation CASes that version, preserves the identity hash
and derives a new snapshot hash. `start_operation_id` is bound before the first
Worker-strategy start and is unchanged by duplicate delivery or recovery. The
owned-resource set records a stable `resource_operation_id`, resource kind,
immutable creation-spec hash, lifecycle state and optional resolved provider/
process reference for every resource whose creation can outlive the creating
call. The operation identity and `creating` intent are durable before creation;
attachment advances that same record rather than inserting an after-the-fact
descriptor.

`AgentTerminalOutcomeRecord` is the immutable Agent-side A2A replay record.
Required fields are `protocol_version`, outcome-record version, exact mapping/
Attempt identity, terminal A2A task state, exactly one canonical `WorkerResult`
or standalone `ErrorEnvelope`, its exact profile media type and Part-data digest,
normalized usage observed at terminal close, `recorded_at` and
`terminal_record_sha256`. It is committed atomically with the mapping's terminal
state only after the scoped execution context is closed and local termination
requirements are satisfied, and before terminal emission. `GetTask`, transport
redelivery and authorized replacement inspection replay that record rather than
reconstructing an outcome from mutable framework state. A standalone error and a
WorkerResult cannot coexist in one mapping.

The
mapping may exist as a cancellation tombstone before any `a2a_task_id`. A
unique authenticated Attempt/message/context key serializes tombstone and
execute mapping. It never owns or mutates the Server Attempt state or
authoritative fencing token.

`AgentMappingRecoveryGrant` is a Server-issued, record-scoped credential. Its
required fields are `protocol_version`, stable `recovery_grant_id`, mode
(`continue_same_incarnation` or `cleanup_replacement`), current and prior exact
registration IDs, instance nonces, lease generations and Server epochs, exact
prior `mapping_id`, `mapping_version`, `mapping_identity_sha256` and
`mapping_snapshot_sha256` at issuance, a closed allow-list of actions, expiry,
`evidence_until` no later than expiry, and an opaque `grant_capability`. Server
stores only the capability verifier plus a mutable grant cursor initialized to
the issuance `mapping_version`/`mapping_snapshot_sha256`. Each permitted mapping
mutation must atomically compare the mapping and cursor to that exact pair,
preserve immutable identity/lineage, and advance both to the resulting version/
snapshot. A query does not advance the cursor. Mismatch, replay or partial
advance fails closed and requires reconciliation or grant reissue. At most one
unexpired mutation-capable continuation grant may exist for a mapping. A
registration response may return a bounded grant list or a one-time
authenticated opaque bundle reference; grant secrets are never ArtifactRefs or
diagnostic data.

`continue_same_incarnation` requires the same protected runtime-incarnation
nonce and may allow recovery/observation of the one already-started durable
Worker execution. Resuming business execution, subordinate operations permitted
by the original immutable policy, staging and terminal-result creation/submission
require the original Attempt fence to remain current, its operation-start gate
to remain open for the original assignment and its deadline to remain valid.
Query, exact-reference status/read/cancel and monotonic finalization of an
already-created invocation remain allowed only through `evidence_until` under
their narrow reconciliation rules even after the gate becomes non-open; they
cannot start a new provider request, publish output or create a WorkerResult.
The grant never authorizes a second Worker execution, changed job or widened
policy.
`cleanup_replacement` requires a new nonce and may allow only inspect,
cancel/terminate recorded resources and append monotonic recovery or
preexisting-evidence outcomes through `evidence_until`. It may replay an
immutable `AgentTerminalOutcomeRecord` already committed before replacement,
but never permits provider/model/tool business-operation start, new or changed
WorkerResult creation/submission, mapping adoption, output staging or an
operation-start gate open. Actions absent from the mode's closed allow-list are
denied even if named by the serialized grant.

`AgentMappingRecoveryReport` is append-only evidence produced under an exact
`cleanup_replacement` grant when a current registration observes or cleans up a
mapping owned by a prior instance nonce. Required fields are `protocol_version`,
stable `recovery_report_id`, exact `recovery_grant_id`, reporting
Agent/registration/nonce/generation/Server epoch, exact prior mapping and owner
identity, monotonic observation sequence, observed execution/resource state,
cleanup action/outcome, immutable evidence refs, observation time and report
hash. A report never changes mapping ownership, adopts the prior execution,
opens an operation-start gate, acknowledges termination by itself or authorizes a
WorkerResult. By itself it may drive only cleanup, `lost` or cancellation
decisions. A terminal success/failure still requires a WorkerResult already
durable before replacement and independently valid under the original Attempt
fence, gate-seal version, deadline and evidence rules.

### Attempt-control DTOs

`AttemptCancellationTombstoneRequest` contains `protocol_version`, stable
`cancel_command_id`, `attempt_id`, Worker idempotency key and fencing-token
snapshot, exact A2A message/context IDs, reason, cancellation deadline and
tombstone retention deadline.

`AttemptCancellationTombstoneAck` contains `protocol_version`,
`cancel_command_id`, `attempt_id`, monotonic tombstone version, state
(`installed`, `task_known` or `already_terminal`), optional
A2A-server-generated `a2a_task_id` and tombstone expiry. Errors use
`application/problem+json` plus
`ErrorEnvelope` fields.

### Agent fleet-control DTOs

`AgentRegistrationRequest` contains `protocol_version`, stable `agent_id`, a
protected runtime-incarnation `instance_nonce`, stable `registration_request_id`
for one retry set, nullable exact `expected_predecessor_generation`, Agent Card URL, supported A2A
protocol versions, versioned capability descriptors and their aggregate hash,
Attempt-control URL/protocol versions, capacity and `started_at`. A crash restart
of the same trusted local incarnation reuses its protected nonce. Planned
replacement, concurrent rollout or loss of that protected state generates a new
nonce and cannot claim same-incarnation continuation.

Each capability descriptor has a stable `capability_id` within one registration
and declares supported `worker_kind` values, exact `job_contract` versions and
immutable Worker implementation ID/version/digest in addition to
resource/policy capabilities. Framework identity is diagnostic and MUST NOT
determine semantic compatibility. Ordinary routing uses semantic contracts and
policy capabilities; an authorized routing constraint may pin a specific
descriptor or implementation digest.

`AgentRegistrationLease` contains `registration_id`, opaque `lease_token`,
monotonic `lease_generation`, a distinct opaque per-generation
`persistence_capability`, `lease_expires_at`,
`heartbeat_interval_seconds`, `server_epoch` and either a bounded exact
`recovery_grants` list or one-time authenticated `recovery_grant_bundle_ref`.

`persistence_capability` authorizes only persistence operations for that exact
Agent registration generation. It is not the fleet-control lease token, cannot
renew/drain/unregister a lease and is rotated whenever the lease generation
changes. Server stores only its verifier; plaintext values are returned once to
the authenticated Agent runtime (or replayed only for the exact idempotent
registration request) and are excluded from logs, later DTO echoes and diagnostic
storage.

`AgentHeartbeat` contains `protocol_version`, `registration_id`, `lease_token`,
`lease_generation`, `server_epoch`, a monotonically increasing `sequence`,
runtime state, available capacity, bounded active-Attempt summary and capability
hash. Server records its own receipt time; Agent timestamps never extend a
lease.

`AgentDrainRequest` contains `protocol_version`, `registration_id`,
`lease_token`, `lease_generation`, `server_epoch`, reason and absolute drain
deadline. Unregister uses the same lease identity and is idempotent.

### `ErrorEnvelope`

`ErrorEnvelope` is a versioned domain DTO. Required fields are
`protocol_version`, stable machine-readable `code`, bounded human-readable
`message`, Boolean `retryable`, versioned `origin`, object-valued `details` and
`correlation_id`. `details` is subject to `contractor-jcs-sha256-v1`, is bounded
by the receiving contract and contains no traceback, credential, prompt, source
contents or other secret by default. `message` is diagnostic and MUST NOT be
parsed to determine retry, origin or state-transition behavior.

## State machines

| Entity | From | Allowed next states |
|---|---|---|
| Run | `accepted` | `planning`, `cancelling`, `failed` |
| Run | `planning` | `running`, `cancelling`, `failed` |
| Run | `running` | `succeeded`, `cancelling`, `failed` |
| Run | `cancelling` | `cancelled`, `failed` |
| Task | `pending` | `ready`, `reused`, `skipped`, `failed`, `cancelled` |
| Task | `ready` | `dispatched`, `reused`, `skipped`, `failed`, `cancelled` |
| Task | `dispatched` | `running`, `succeeded`, `ready`, `failed`, `cancelled` |
| Task | `running` | `succeeded`, `ready`, `failed`, `cancelled` |
| Attempt | `created` | `leased`, `failed`, `cancelled` |
| Attempt | `leased` | `running`, `succeeded`, `failed`, `lost`, `cancelled` |
| Attempt | `running` | `succeeded`, `failed`, `cancelled`, `lost` |
| Operation-start gate | `closed` | `open`, `revoked` |
| Operation-start gate | `open` | `sealed`, `revoked` |

Terminal states have no outgoing transition. `succeeded`, `reused` and
`skipped` require valid immutable `TaskResolution` evidence for that Task and
are terminal independently of downstream topology. Edge/join policy separately
determines successor readiness or branch failure; it never changes a
predecessor's terminal state or resolution. `pending`/`ready` may become
`failed` for a deterministic permanent condition, binding or policy error found
before Attempt creation. A `created` Attempt may become `failed` when its job
deadline expires or assignment/dispatch validation fails permanently before a
lease; no WorkerResult is fabricated for that transition. Moving a dispatched
or running Task back to `ready` requires the current Attempt first to become
`failed` or `lost`, and retry policy to authorize a newly identified Attempt.
An assigned Agent may return a terminal result before Server observes a running
event, so `leased` and its `dispatched` Task may transition directly to a valid
terminal result. A gate opens only from its initial pre-assignment `closed`
state; `sealed` and `revoked` are terminal, so a loss/cancel/retry/supersession
cannot reopen the old Attempt. In the other specifications, “gate closure” is
the generic operation that makes a gate non-open: it revokes an initial
`closed` gate or seals/revokes an `open` gate; it never means returning to the
pre-assignment `closed` state. Internal `cancelled` maps to A2A task state
`canceled` at the protocol adapter.

## Requirements

- **CON-001** — Every hashed or fixture-compared DTO MUST use
  `contractor-jcs-sha256-v1` exactly. Duplicate keys, invalid Unicode, forbidden
  numbers and non-canonical digest encodings MUST be rejected before
  persistence or execution.
- **CON-002** — Unknown additive fields MUST be ignored within the same major
  protocol version. Removing or changing a field requires a major version.
- **CON-003** — Invalid state transitions MUST be rejected before persistence.
- **CON-004** — Every command and emitted event MUST carry `run_id`, and when
  applicable `task_id` and `attempt_id`.
- **CON-005** — `idempotency_key` identifies the logical operation;
  `fencing_token` identifies the currently authorized Attempt lease. They MUST
  NOT be treated as the same value.
- **CON-006** — Large or binary values MUST cross service boundaries as
  `ArtifactRef`, not inline payloads.
- **CON-007** — Domain contracts MUST have no dependency on ADK events,
  `google.genai.types.Part`, SQLAlchemy models or A2A SDK models.
- **CON-008** — Cancellation is monotonic: once cancellation is requested, no
  new Attempts may be leased for the run.
- **CON-009** — `WorkspaceRef`, `SandboxLease`, `DispatchOutboxRecord` and
  principal/tenant identity MUST obey the same deterministic/versioned wire
  rules as `WorkerJob` and `WorkerResult`.
- **CON-010** — Boundary adapters MUST map internal `cancelled` to A2A
  `canceled`, and MUST reject unknown outcome/state strings.
- **CON-011** — Agent registration, lease, heartbeat and drain DTOs MUST follow
  the same deterministic and versioned wire rules as execution DTOs.
- **CON-012** — Lease tokens, persistence capabilities and recovery-grant
  capabilities are credentials: they MUST be compared through Server-held
  verifiers without leaking their values and MUST be excluded from logs,
  telemetry and errors. The credential classes are purpose-distinct and MUST
  NOT authorize each other's operations.
- **CON-013** — An outbox record MUST use a closed, versioned command type and
  an immutable payload hash; changing execute into cancel or changing its
  payload after commit is forbidden. Its producing `run_state_version` is audit
  provenance; delivery eligibility MUST be checked against the exact current
  Attempt/job/fence/cancellation facts and MUST NOT fail merely because an
  unrelated CAS committed a later RunState version.
- **CON-014** — `ServerAttemptRecord` and `AgentTaskMapping` MUST remain separate
  persistence contracts. Agent-reported status is input to a Server transition,
  never authority to mutate Server state or revoke a fence.
- **CON-015** — A2A `message_id` and accepted client `context_id` MUST be stable
  and unique per authenticated caller/Attempt; reuse for another Attempt MUST be
  rejected. `a2a_task_id` is generated by the A2A server hosted by the Agent and
  MUST NOT be guessed or supplied when creating a task; it is distinct from the
  Contractor `TaskId` carried as `task_id` in a WorkerJob.
- **CON-016** — `AgentTaskMapping` MUST represent a pre-execution cancellation
  tombstone without inventing an `a2a_task_id`, and its unique identity MUST
  conflict atomically with any later execute mapping for that Attempt.
- **CON-017** — Attempt-control request/ack DTOs MUST be deterministic,
  versioned and idempotent by `cancel_command_id`; the acknowledged tombstone
  version MUST never decrease.
- **CON-018** — `RunSpec` and every `TaskSpec` MUST validate their objectives
  and expected outputs against exact declared contracts. `worker_kind` alone is
  not a complete description of work, and a Task policy MUST NOT widen its
  accepted Run policy.
- **CON-019** — A `WorkerJob` MUST contain one exact, hash-verified `TaskSpec`
  and only scalar/DTO fields plus exact artifact references. Framework state,
  Python callables and implicit Server filesystem context are forbidden.
- **CON-020** — Agent capability matching MUST include a compatible
  `job_contract` version. The Agent's use of ADK, another framework or custom
  logic MUST NOT change the wire DTO.
- **CON-021** — Agent-internal planning/working phases and local subtasks are
  not Server `Task` or `Attempt` records. A `WorkerResult` terminates the one
  boundary Attempt identified by its `attempt_id`, regardless of the internal
  strategy that produced it.
- **CON-022** — `PlannerStrategyRef` MUST be committed before planning, obey the
  deterministic/versioned DTO rules and remain immutable for the run. Exact
  implementation/configuration digests are required; recovery MUST stop safely
  instead of using a deployment default or reassigned version label.
- **CON-023** — `RunSpec` is authoritative for submitted objective, outputs and
  run-wide Worker execution constraints; `PlannerStrategyRef` is authoritative
  only for plan formation. Conflicting or widened Planner/Task configuration
  MUST be rejected rather than resolved by undocumented precedence.
- **CON-024** — Planner strategies MUST produce immutable `TaskSpec` values,
  never Attempt-specific `WorkerJob` values. The scheduler MUST create one new
  WorkerJob per Attempt; redelivery MUST reuse the exact job for that Attempt. A
  retry retains the TaskId/TaskSpec binding; revised work uses a new TaskId with
  an explicit `supersedes_task_id`.
- **CON-025** — Dispatcher MUST commit a `WorkerDispatchEnvelope` with selected
  `capability_id` and Worker implementation digest before send. Agent MUST
  verify and persist it before reading inputs or starting work, and
  `WorkerResult.execution_provenance` MUST match it before result acceptance.
- **CON-026** — `WorkerResult.usage` MUST aggregate direct work and every
  Agent-internal planning/working step under one normalized schema. Server MUST
  preserve unknown measurements and MUST NOT infer zero usage from missing ADK
  or telemetry detail.
- **CON-027** — Static planning MUST deterministically instantiate a pinned,
  versioned one-or-more-Task template manifest against the exact RunSpec. Only
  template-local node/edge keys occur in the reusable manifest; the pinned ID
  derivation maps them to byte-identical run-specific Task definitions and
  edges. Passthrough MUST be exactly the one-root-Task, zero-edge specialization;
  neither changes the Worker wire profile.
- **CON-028** — A source project input MUST be an exact, authorized
  `project-snapshot` ArtifactRef. Snapshot materialization MUST validate the
  pinned canonicalization/hash algorithm, manifest and every referenced blob
  before writing only beneath the Attempt workspace; Server/Agent host paths are
  not wire inputs. Only the validated snapshot-publication operation may mint
  the reserved `project-snapshot` kind.
- **CON-029** — Before materializing a WorkerJob, the scheduler MUST resolve
  every TaskSpec dependency input through its exact committed edge to an exact
  ArtifactRef and include
  those bindings in the job. For a compound snapshot/manifest it MUST also
  compute and hash the exact transitive blob-grant closure. Missing, ambiguous
  or policy-incompatible bindings MUST fail before Attempt creation or
  tool/model access and take a defined predispatch Task failure transition.
- **CON-030** — Conditional, reuse and join decisions MUST use versioned
  declarative policies plus immutable `TaskResolution` evidence. A cache/reuse
  fingerprint MUST cover the exact TaskSpec, resolved inputs, effective policy
  refs and permitted producer provenance; a familiar artifact name or mutable
  latest-version lookup is insufficient.
- **CON-031** — Every Agent-owned execution/effect record MUST pin the exact
  assigned registration identity/generation/Server epoch. A newer registration
  with a different instance nonce fences writes/reconciliation from the prior
  process just as it fences routing; stable AgentId or database login alone is
  insufficient. A crash restart with the same protected runtime-incarnation
  nonce MAY resume business execution only for exact mappings named by
  `continue_same_incarnation` grants while their original Attempt fence/open
  gate/deadline remain valid; after gate closure only the narrow control/evidence
  actions in `CON-037` remain. Its rotated persistence capability fences the
  zombie generation. Reusing a nonce requires an external single-owner
  guarantee. A new nonce receives cleanup-only grants.
  The append-only cross-nonce recovery report in `CON-037` is evidence, not a
  mutation or adoption exception.
- **CON-032** — Run terminal outcome MUST follow the accepted deterministic
  completion/failure policy over committed Task states/resolutions and required
  output bindings. Tolerated branch failure, partial joins and optional skipped
  work MUST be explicit; the scheduler MUST NOT infer success from “some output
  exists” or fail every fan-out merely because one optional branch failed.
- **CON-033** — Every accepted TaskId MUST remain permanently bound to its first
  TaskSpec hash. A revision MUST create a new TaskId and explicit
  `supersedes_task_id`. In the accepting CAS, an eligible nonterminal prior Task
  MUST become `cancelled` with reason `superseded` and its execute/fence/gate be
  invalidated; an already-terminal prior Task MUST keep its terminal state and
  resolution. All prior definitions, Attempts and evidence remain immutable.
- **CON-034** — `DependencyEdge` MUST be the single source of predecessor/output
  and successor/input identity. A TaskSpec dependency input references only its
  edge ID; missing, duplicate, unreferenced or mismatched bindings MUST be
  rejected before plan commit or dispatch.
- **CON-035** — Ordinary Agent persistence writes MUST prove the exact
  per-generation `persistence_capability` in addition to database/transport
  identity. A recovery operation instead MUST prove an exact unexpired
  record-scoped recovery grant whose mode/action/mapping all match. Rotation MUST
  fence the prior generation without treating the fleet lease token as a
  persistence credential.
- **CON-036** — `AgentTaskMapping.start_operation_id` MUST be stable for the
  mapping, and model/tool/effect intent insertion MUST remain denied until the
  corresponding Server-owned Attempt operation-start gate is explicitly open
  for its exact registration and deadline. Attempt creation sets `closed`; the
  assignment-envelope CAS opens it before send; loss, cancel, retry and
  supersession irreversibly seal/revoke it. Intent insertion MUST serialize with
  the exact gate version. Only Server may transition the gate; sealing it still
  permits a preexisting committed intent to finalize its outcome monotonically.
- **CON-037** — Recovery grants MUST be exact, expiring and action-closed.
  Same-incarnation continuation may recover/observe the already-started durable
  execution, perform original-policy subordinate work, stage output and submit a
  terminal result only for its named mapping while the original fence, open gate
  and deadline remain valid; it MUST NOT start a second Worker execution. After
  gate closure, it may only query/cancel exact references and monotonically
  finalize already-created evidence through the bounded evidence window, with no
  new provider request, output publication or WorkerResult creation. A new-nonce
  cleanup grant MUST NOT adopt that mapping, start a business
  operation, stage output, open its gate or create/change/submit success. It MAY
  replay an exact terminal record committed before replacement and append
  monotonic `AgentMappingRecoveryReport` evidence; duplicate/conflicting
  observations fail closed and only an explicit Server transition may consume
  them. Every mapping mutation uses the grant cursor's monotonic version/snapshot
  CAS and preserves its immutable identity/lineage.
- **CON-038** — `WorkerProgress` MUST remain bounded, best-effort and
  non-authoritative. Event IDs/sequences expose duplicate, gap and reordering
  behavior explicitly; provisional refs/usage MUST NOT promote, settle or
  transition state, and no event kind may encode Agent-private planning/working
  mode or subtask state.
- **CON-039** — A mutation-capable recovery grant MUST own a Server-held current
  mapping-version/snapshot cursor initialized at issuance. The guarded mapping
  mutation and cursor advance MUST commit atomically; stale/replayed cursors and
  more than one unexpired mutating continuation grant for a mapping MUST fail.
  Query-only actions do not advance the cursor, and cleanup grants never rewrite
  mapping ownership.
- **CON-040** — Server terminal-result/error ingestion MUST atomically store the
  evidence and usage and seal/revoke the Attempt operation-start gate. RunState
  acceptance MUST pin that exact non-open gate version, so no model/tool intent
  can commit after the terminal aggregate and before promotion/settlement.
- **CON-041** — A resource whose creation can precede discovery of its provider/
  process reference MUST have a durable `resource_operation_id`, immutable spec
  hash and `creating` intent before creation. Its provider/supervisor MUST support
  idempotent create-or-lookup/cleanup by that identity, or an atomic reservation;
  otherwise restart MUST classify the resource state as unconfirmed and
  quarantine the capability.
- **CON-042** — A non-idempotent registration request MUST atomically compare
  `expected_predecessor_generation` with current durable registry state before
  rotating credentials. Exactly one contender may advance a generation; a stale
  predecessor fails without revealing or rotating current credentials. Same-
  nonce crash continuation additionally requires an externally enforced
  single-owner incarnation, otherwise the process registers with a new nonce and
  receives cleanup-only grants.
- **CON-043** — A version-controlled canonical-JSON golden corpus MUST contain
  original JSON, expected canonical UTF-8 bytes, expected lowercase SHA-256 and
  expected acceptance/rejection for each case. Every Server and Agent adapter,
  independent of language or framework, MUST consume the same corpus.
- **CON-044** — Before any terminal A2A emission, AgentRuntime MUST atomically
  persist the mapping's immutable `AgentTerminalOutcomeRecord` and terminal
  state. Query, redelivery and replacement inspection MUST replay that exact
  canonical DTO/media type/digest; they MUST NOT reconstruct, replace or combine
  terminal outcomes from session events, provider state or mutable defaults.

## Acceptance

1. Golden JSON fixtures round-trip for all public DTOs.
2. Property tests cover legal and illegal state transitions.
3. Duplicate `WorkerJob` delivery with the same `attempt_id` and idempotency
   key yields one logical Attempt.
4. A stale fencing token cannot publish a successful result.
5. Golden fixtures cover register, heartbeat, drain, execute-outbox and
   cancel-outbox payloads.
6. Unknown fleet-control fields are tolerated only according to the declared
   protocol-version compatibility rules.
7. Golden records distinguish Server outbound delivery identity from Agent
   inbound execution mapping and contain no shared mutable database row.
8. Property tests cover both serialization orders of execute mapping versus
   cancellation tombstone and never produce two tasks or a post-cancel start.
9. Golden fixtures cover Attempt-control request, acknowledgement, idempotent
   replay, stale fence and incompatible-version errors.
10. Golden `TaskSpec`/`WorkerJob` fixtures cover both a typed narrow Task and a
    root objective using the same envelope and A2A mapping.
11. Capability matching accepts ADK and non-ADK Agents implementing the same
    `job_contract` and rejects an Agent supporting only an incompatible version.
12. No serialized DTO reveals an Agent-internal planning/working phase or
    private subtask object.
13. Separate `PlannerStrategyRef` and `RunSpec` fixtures survive restart with
    exact Planner code/config and Worker execution policy; changing current
    process defaults does not change or retarget the recovered run.
14. A Planner proposal contains TaskSpecs only; first execution and a policy
    retry create different Attempt-specific WorkerJobs, while transport
    redelivery reuses byte-identical job data.
15. A committed dispatch-envelope fixture selects one of two compatible local
    implementations by `capability_id`; the Agent rejects a changed/unregistered
    implementation before reading inputs or starting work.
16. Result fixtures accept matching Worker execution provenance and reject a
    changed or unregistered strategy/build digest before artifact promotion.
17. A multi-step Worker strategy produces one aggregate usage record;
    deliberately unavailable counters remain `unknown` rather than zero.
18. Static-plan fixtures accept one and many Tasks, reproduce byte-identical
    Task/edge IDs from template-local keys and reject an ID collision; the
    passthrough fixture accepts exactly one root Task and zero dependency edges.
19. Project-snapshot fixtures reject invalid UTF-8/NFC/path segments, case-fold
    collisions, links, wrong ordering, digest mismatch and a blob outside the
    caller's artifact scope; a valid snapshot materializes byte-for-byte with
    normalized modes beneath a temporary Attempt workspace.
20. Dependency fixtures cover executed, reused, skipped, failed and cancelled
    predecessors plus all/threshold joins; missing policy and missing,
    duplicated or mismatched edge/input references fail closed before dispatch.
21. Reuse fixtures accept only an exact fingerprint/provenance/output match and
    commit immutable TaskResolution evidence; same-name or changed-input
    artifacts do not satisfy the Task.
22. After a new instance nonce/lease generation registers for one AgentId, the
    old process cannot create or mutate task mappings, effect records or model
    evidence even though its stable Agent database login is unchanged.
23. Completion-policy fixtures cover all-required success, an explicitly
    tolerated failed fan-out branch and a missing required output; only the
    declared first two reach the specified terminal result.
24. Attempting to bind an existing TaskId to a different TaskSpec hash fails. An
    authorized revision creates a new TaskId with an exact supersedes relation;
    retry and redelivery retain the original TaskId/TaskSpec binding.
25. Permanent binding/policy failure before Attempt creation terminals the Task;
    permanent assignment failure or job-deadline expiry terminals the created
    Attempt and drives the declared Task retry/failure transition without a
    fabricated WorkerResult.
26. An unrelated later RunState CAS leaves a current execute record deliverable;
    explicit cancellation, supersession, stale job hash or stale fence prevents
    delivery before network I/O.
27. A predecessor remains terminal when a successor rejects its resolution;
    edge/join policy deterministically fails, skips or blocks only the successor
    branch according to the accepted completion policy.
28. Server/Agent attempt fixtures use `a2a_task_id` for the A2A-server-generated
    identifier and cannot confuse it with Contractor `task_id`.
29. Fleet fixtures prove the lease token cannot authorize persistence, the
    per-generation persistence capability cannot renew a lease, and neither can
    substitute for an exact recovery-grant capability. Rotation rejects the
    prior capability without storing plaintext.
30. Attempt creation persists a closed operation-start gate; the exact
    assignment-envelope CAS opens it for one registration/deadline before send.
    Duplicate delivery retains one `start_operation_id`; intent insertion also
    requires the exact durable mapping. Loss/cancel/retry/supersession seals the
    gate, after which a pre-closure intent may append its outcome but no new
    operation can start or reopen the Attempt.
31. A same-protected-nonce crash restart receives only exact
    `continue_same_incarnation` grants and can resume the one already-started
    durable execution, perform only original-policy subordinate work, stage
    output and submit its terminal result while the original fence, open gate and
    deadline remain valid. Its rotated persistence capability fences a zombie
    process, and no second Worker execution can start. A new nonce receives only
    exact `cleanup_replacement` grants: it can append idempotent monotonic cleanup
    reports but cannot adopt a mapping, open its gate, submit success or rewrite
    conflicting observations; a stale mapping version/snapshot or changed
    identity fails its grant CAS. Once the gate is non-open, either grant is
    limited to its allow-listed exact-reference query/cancel/reconciliation
    actions and cannot initiate business/provider work.
32. An Agent that returns success/failure immediately after assignment can move
    the Attempt directly from `leased` and its Task directly from `dispatched`
    to the corresponding terminal state without a synthetic running event.
33. Golden progress fixtures cover duplicate, gap and reordered sequences,
    bounded summaries, provisional staged refs/usage and ADK/non-ADK producers;
    dropping the entire stream leaves terminal state, promotion and settlement
    unchanged, and no fixture exposes private planning/working state.
34. A same-incarnation continuation performs resume → staged output → terminal
    mapping updates by atomically advancing one recovery-grant cursor each time;
    replay, a stale cursor and a second mutating grant all fail. A replacement
    nonce can append cleanup evidence but cannot mutate mapping ownership or
    submit the result.
35. Terminal-result ingestion races a new model/tool intent. Either the intent
    commits first and appears in the sealed evidence snapshot, or terminal
    evidence and gate seal commit first and the intent performs no provider I/O;
    acceptance pins that exact gate version.
36. A resource-creation crash after provider acceptance but before reference
    attachment recovers exactly one resource by `resource_operation_id`, or
    quarantines an unconfirmed provider; it never creates an untracked duplicate.
37. Two non-idempotent registration requests with the same expected predecessor
    race: one generation/credential rotation commits and the other fails without
    receiving current credentials. Same-nonce recovery is rejected when the
    deployment cannot prove exclusive incarnation ownership.
38. The shared canonical corpus covers RFC 8785 object ordering, escaping and
    number vectors; non-ASCII member names; equivalent source member orders;
    nested unknown fields; and self-digest omission. It rejects duplicate keys,
    unpaired surrogates, non-finite numbers, unsafe integer-valued numbers,
    uppercase/base64 digests and an incorrectly included self-digest.
39. A crash after Agent terminal-record commit but before/during A2A emission
    makes `GetTask` replay byte-identical canonical data, media type, digest,
    state and usage. A crash before that commit exposes no terminal outcome, and
    a replacement cannot manufacture one from provider/session observations.
