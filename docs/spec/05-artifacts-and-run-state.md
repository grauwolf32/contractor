# 05 — Artifacts and PlannerRunState

Status: **Draft**  
Depends on: [02](02-domain-contracts.md), [04](04-database-runtime.md)

## PostgreSQL ArtifactService

`PgArtifactService` implements ADK `BaseArtifactService` and the richer
Contractor `ArtifactStore` port over the same tables. The first release stores
both metadata and content in PostgreSQL; introducing object storage is a later
adapter decision.

Every save creates an immutable numbered version. Metadata includes artifact
kind, scope, media type, content length, SHA-256, schema version, creator,
creation time and correlation IDs.

Content is uploaded as bounded chunks into PostgreSQL while SHA-256 and size
are computed incrementally. An incomplete upload is invisible and removable.
The ADK `types.Part` API is used only below a configured materialization limit;
larger Contractor tools stream content and exchange `ArtifactRef` instead of
loading the full value into ADK session memory.

## Artifact scopes

- `run` — shared by Planner and attempts in one run;
- `attempt` — private staging area for one Attempt;
- `user` — explicitly published reusable data;
- `system` — controlled templates or static assets.

Agents receive grants for exact input refs and their Attempt staging scope.
They cannot list or read unrelated run/user artifacts.

PostgreSQL row-level security enforces this boundary for the Agent database
role. RLS derives `agent_id` from the deployment-provisioned database login and
protected principal mapping, never from a caller-settable variable. The trusted
runtime establishes `attempt_id` context only after proving the exact current
per-generation persistence capability, or an unexpired record/action-scoped
recovery grant, and validating its durable mapping and exact artifact grants.
Policies require the mapping to match the committed dispatch
assignment/registration before allowing those input refs or writes to that
Attempt's staging scope. A grant by Attempt ID, copied current registration
fields or the stable Agent login alone does not let another or fenced generation
claim it. Cleanup grants cannot read arbitrary inputs or stage a successful
result. Server uses a distinct privileged application role. Application checks
remain defense in depth and do not replace RLS. A transaction with missing,
stale or mismatched capability/grant/Attempt context fails closed, and pool reset
prevents the previous transaction's context from leaking. Bytes already cached
inside an old local workspace remain constrained by the Attempt deadline,
operation-start gate, sandbox termination and result fence; database revocation
does not pretend to erase them synchronously.

A source project is an immutable `ProjectSnapshotManifest` artifact plus its
exact immutable blob refs. The snapshot becomes visible atomically only after
the manifest, every referenced blob, canonicalization/hash rules, immutable
artifact-dependency records and grants have been validated. Server passes only
the resulting exact snapshot ref; the Agent revalidates the manifest hash and
dependency closure and materializes those refs beneath its Attempt workspace
without receiving a Server host path.

## Staging and promotion

Agent output is first committed as `staged` Attempt-scoped content. Only Server
and the owning Attempt can read it; run/user consumers cannot. `WorkerResult`
returns exact staged refs and the current fencing token.

When Server accepts the result, RunState CAS atomically inserts promotion/grant
records and references the immutable content from the new RunState. No artifact
bytes move between processes in that transaction. A stale, failed or cancelled
Attempt's staged output is never promoted and is removed by retention policy.

## Authoritative PlannerRunState

`PlannerRunState` is a `run`-scoped versioned artifact containing:

- `schema_version`, `run_id` and logical plan generation;
- the exact immutable `RunSpec` and `PlannerStrategyRef` selected when the run
  is created, including their verified hashes;
- immutable Task definitions that permanently bind each `TaskId` to one exact
  `TaskSpec` hash, optional explicit `supersedes_task_id` relations and
  dependency edges accepted from the selected Planner strategy;
- task states, explicit executed/reused/skipped resolution evidence,
  deterministic predispatch failure evidence and immutable Attempt references;
- current budgets, accepted Planner invocation references, cancellation flag
  and terminal outcome;
- the decisions needed to reconstruct the next scheduler step.

Chat history, full tool output and telemetry remain referenced artifacts or
sessions instead of being copied wholesale into RunState.

With the static Planner, the authoritative graph is the deterministic
run-specific instantiation of the pinned one-or-more-Task template manifest.
Passthrough is the static Planner's constrained specialization: its graph
contains exactly one logical root Task and zero dependency edges.
Retries add immutable Attempts for that existing Task and do not add another
root Task or change its `TaskSpec`. Planning/working phases and subtasks created
privately by the receiving Agent are not copied into `PlannerRunState`; the
Server observes that Agent through the root Attempt's normalized progress and
`WorkerResult` only.

For every strategy, an accepted TaskId/TaskSpec binding is permanent. An
authorized plan revision adds a new TaskId and exact `supersedes_task_id`
relation; it never replaces a TaskSpec under an existing TaskId. The superseded
definition, Attempts and resolutions remain immutable. In the revision CAS, an
eligible nonterminal prior Task becomes `cancelled` with reason `superseded` and
its execute records, active Attempt fence and operation-start gate are
invalidated; an already-terminal prior Task retains its terminal state and
resolution.

The run catalog is a query/status projection with a pointer to the current
RunState version. If projection data disagrees with the artifact, recovery uses
the verified RunState and rebuilds the projection; the catalog is not a second
plan owner.

## Atomic commit protocol

The Contractor extension provides:

```text
compare_and_swap(run_id, expected_version, next_state, side_effects)
  -> new_version | VersionConflict
```

The transaction inserts the immutable RunState artifact version, promotes any
accepted staged refs, verifies/accepts the Server-owned Attempt result and
fencing token plus its exact terminal-ingest non-open gate version, seals/revokes
gates for scheduler-owned loss/cancel/retry transitions as applicable, settles
its Worker budget reservation, updates the run's current-state pointer/status
projection and commits execute/cancel outbox records plus any required same-owner
audit facts. A Planner-strategy CAS may
commit `TaskSpec`/edge changes and Planner accounting, but it does not create an
Attempt-specific WorkerJob or execute outbox record. After those TaskSpecs are
committed, a deterministic scheduler CAS may create a new Attempt, reserve its
budget, materialize its exact Attempt-specific `WorkerJob` from the accepted
TaskSpec and append the execute outbox record. That CAS also inserts grants for
exactly the resolved input refs and creates the Attempt with a closed
operation-start gate. `RunStateStore` owns the only `AsyncSession` and transaction
for each such CAS; Artifact, Run and Attempt repositories plus required Audit
appends receive that caller-owned session and cannot commit independently.

Separately, before network send, the dispatcher transaction atomically commits
the immutable assignment envelope/endpoint/message/context and opens the gate
only for that exact registration and effective deadline; failure rolls back both
assignment and gate opening. Input grants become usable only by an Agent whose
verified task mapping and persistence/recovery capability match that committed
assignment.

Before a scheduler can accept terminal Worker evidence, the Server dispatcher
ingests the terminal result/error and normalized usage and seals/revokes that
Attempt's operation-start gate in one Server-owned transaction. The resulting
`terminal_ingest_gate_version` is part of the immutable evidence read by the
acceptance CAS. This prevents an Agent invocation intent from committing between
the terminal aggregate and gate seal; acceptance never performs a later,
separate first closure for a successful result.

A CAS that records Attempt loss/cancellation/terminal state or begins retry
authorization seals or revokes that Attempt's operation-start gate atomically.
Only after the closure commits may the scheduler read the resulting sealed
effect-evidence view. Agent intent insertion serializes against the gate in the
same database; outcomes for intents committed before closure may still advance
monotonically and are reconciled before a retry decision. Neither
`EffectEvidenceReader` nor the Agent mutates the Server-owned gate.
The invariant after commit is
`projection.run_state_version == current RunState artifact version`. A normal
ADK `save_artifact()` call is insufficient because it has no `expected_version`
precondition or multi-record Unit of Work.

## Requirements

- **ART-001** — Artifact bytes MUST be immutable after a version is committed.
- **ART-002** — Reads by `ArtifactRef` MUST verify the declared size and SHA-256.
- **ART-003** — RunState changes MUST use compare-and-swap against the exact
  previously loaded version.
- **ART-004** — Exactly one of two concurrent updates with the same expected
  version may commit.
- **ART-005** — Repeating an artifact publication with the same idempotency key
  and content MUST return the existing reference. Different content with the
  same key MUST fail.
- **ART-006** — A successful Attempt result MUST become visible atomically with
  the RunState transition that accepts its fencing token.
- **ART-007** — RunState readers MUST support explicit schema upcasting. Writers
  emit only the current schema version.
- **ART-008** — Delete and retention operations MUST preserve any version still
  referenced by a non-deleted RunState, retained session or retained compound
  artifact, plus the full transitive closure of immutable artifact-dependency
  records. Collection MUST fail closed on a missing/cyclic/corrupt dependency
  graph and MUST NOT delete a project blob while a retained snapshot can reach
  it.
- **ART-009** — Maximum inline artifact size MUST be configurable and enforced
  before buffering the whole request in memory.
- **ART-010** — The mapping between ADK `(app_name, user_id, session_id,
  filename)` namespaces and Contractor scopes MUST be explicit, reversible and
  covered by compatibility fixtures.
- **ART-011** — Run catalog status rows MUST be treated as rebuildable
  projections; they MUST NOT override a verified RunState plan decision.
- **ART-012** — Agent output MUST remain `staged` until a fencing-token-checked
  RunState CAS promotes its exact immutable refs.
- **ART-013** — Agent artifact access MUST be protected by PostgreSQL RLS using
  verified transaction scope and a no-`BYPASSRLS` runtime role.
- **ART-014** — Large content MUST use incremental chunked upload/download with
  bounded memory. Incomplete uploads MUST never produce a readable ArtifactRef.
- **ART-015** — Every successful CAS MUST preserve
  `projection.run_state_version == committed artifact version`; mismatch MUST
  fail readiness for the run until deterministic projection repair completes.
- **ART-016** — RLS Agent identity MUST derive from the authenticated database
  login, while every ordinary generation-sensitive operation additionally proves
  the exact current per-generation persistence capability. Attempt context MUST
  be validated against the durable assignment/mapping/artifact grant, remain
  transaction-local and clear automatically before a pooled connection is
  reused. Recovery requires an exact unexpired mapping/action-scoped grant;
  copied registration fields, Attempt ID or `SET LOCAL agent_id` are not
  authentication.
- **ART-017** — Transaction-bound Artifact, Run and Attempt repository methods
  and required Audit appends MUST use the `RunStateStore` caller-owned session
  and MUST NOT begin, commit or roll back an independent transaction.
- **ART-018** — RunState MUST reference only a terminal result already ingested
  in the Server Attempt record together with its exact gate seal/revocation
  version. Its fencing/gate-version acceptance, staged-ref promotion and Worker-
  budget settlement MUST commit or roll back together.
- **ART-019** — The exact `RunSpec` and `PlannerStrategyRef`, including their
  recoverable content, implementation/configuration identity and verified
  hashes, MUST be pinned in the first RunState version and remain unchanged for
  that run. Recovery MUST fail safely rather than substitute deployment defaults
  or a reassigned strategy version.
- **ART-020** — Every authoritative task definition MUST be an exact immutable
  TaskId/`TaskSpec`-hash binding accepted under the pinned RunSpec. A Planner
  proposal MUST NOT contain an Attempt-specific `WorkerJob` or rebind an existing
  TaskId; revised work creates a new TaskId with explicit `supersedes_task_id`.
  Only after that Task definition is committed may a deterministic scheduler CAS
  materialize a new WorkerJob for a new Attempt, insert its exact resolved-input
  grants and append its execute outbox record. A policy retry MUST reuse the
  exact TaskId/TaskSpec while creating a new Attempt ID, reservation, fence,
  grants and WorkerJob; transport redelivery MUST reuse the exact committed
  WorkerJob/grants for the existing Attempt.
- **ART-021** — RunState MUST record an explicit Task resolution and exact
  evidence when static/conditional work is executed, reused or
  skipped/not-applicable. Reuse MUST identify the validated immutable output
  refs and cache/provenance decision; artifact-name existence alone MUST NOT
  change Task state. Dependency and join evaluation MUST consume this committed
  resolution rather than infer it from the artifact store.
- **ART-022** — A project snapshot manifest and every referenced blob MUST be
  immutable exact inputs. Snapshot publication MUST atomically validate the
  versioned canonical path/hash algorithm, manifest, blobs and authorization and
  record exact artifact dependencies. Agent materialization MUST revalidate the
  exact refs and write only beneath the Attempt workspace. A Server or Agent host
  path MUST NOT substitute for a snapshot ref.
- **ART-023** — Attempt input grants MUST be the immutable, hash-identified
  closure of resolved TaskSpec bindings, including every blob referenced by a
  compound project snapshot. Grant creation MUST NOT follow mutable aliases or
  grant unrelated artifacts; Agent MUST verify the grant-set hash before
  materialization.
- **ART-024** — Execute-outbox authorization MUST be represented by the exact
  current Attempt, WorkerJob hash, fence and cancellation/supersession facts.
  `run_state_version` records command provenance only; an unrelated later CAS
  MUST NOT revoke delivery, while a CAS that invalidates work MUST explicitly
  fence its execute record.
- **ART-025** — Attempt creation MUST persist a closed operation-start gate. The
  assignment-envelope transaction MUST atomically open it for the exact selected
  registration/deadline before send. Loss, cancellation, supersession and retry
  authorization MUST seal/revoke the prior gate in the CAS before effect evidence
  is read. New intent insertion MUST serialize with that gate; already committed
  intents MAY append only monotonic outcome/reconciliation evidence. The
  read-only evidence adapter MUST NOT open, close or otherwise mutate the gate.
- **ART-026** — Terminal WorkerResult/error ingestion MUST store the evidence,
  normalized usage and resulting non-open operation-start-gate version in one
  Server transaction. Result acceptance MUST reject an absent, open or changed
  terminal-ingest gate version before promotion or settlement.

## Acceptance

1. Concurrent CAS writers produce one new version and one conflict.
2. A stale Agent cannot commit output after its Attempt has been replaced.
3. Restarting Server reconstructs scheduling solely from the current RunState
   artifact plus immutable referenced records.
4. Authorization tests prove that an Agent cannot enumerate or load another
   Attempt's artifacts.
5. Corrupted content fails hash validation and is never passed to a tool.
6. A cancelled/stale Attempt may leave staged bytes, but they are invisible to
   run readers, cannot be promoted and are later collected.
7. Faults before and after staged-ref promotion prove that promotion, RunState,
   projection and outbox commit or roll back together.
8. Upload/download of a fixture larger than the ADK materialization limit stays
   within the configured process-memory bound.
9. Tests run with RLS enabled and prove exact input grants, staging writes and
   cross-Attempt denial even when the adapter omits its application check.
10. Deliberately corrupting the projection version triggers rebuild from the
    verified RunState and never dispatches from the stale projection.
11. Reusing one physical pooled connection for two Attempts cannot carry the
    first Attempt's RLS identity or grants into the second transaction.
12. A repository spy/fault test proves there is exactly one transaction owner
    for RunState version, promotions, Attempt/fence changes, projection,
    required audit facts and execute/cancel outbox records.
13. A crash after terminal-result ingestion but before RunState CAS leaves the
    result durable but unaccepted; recovery promotes and settles it exactly once.
14. Recovery verifies and reuses the exact pinned RunSpec, PlannerStrategyRef
    and immutable TaskId/TaskSpec bindings; changing deployment defaults does
    not change, retarget or re-plan the run.
15. A passthrough run recovers from its one root TaskSpec without requiring any
    Agent-private subtask or reasoning state and never creates a second root
    Task.
16. A Planner proposal commits no Attempt-specific WorkerJob. The first
    scheduler dispatch and a policy retry create different WorkerJobs from the
    same exact TaskId/TaskSpec, while redelivery reuses byte-identical job data
    for the existing Attempt. Rebinding that TaskId fails; an authorized revision
    creates a new TaskId and supersedes relation, atomically cancels/fences an
    eligible nonterminal prior Task and leaves terminal history unchanged.
17. Static conditional fixtures commit explicit executed, reused and
    skipped/not-applicable resolutions. Reused outputs name exact validated refs,
    and an unrelated artifact with the same logical name never satisfies a
    dependency or join.
18. A project snapshot is invisible until its complete manifest/blob dependency
    set and canonical `snapshot_sha256` preimage pass validation; the Agent
    materializes the exact refs and normalized modes inside its workspace, and
    malformed paths, changed blobs and host-path inputs fail before tool use.
19. Fault injection around Attempt creation proves the WorkerJob, resolved-input
    grants and execute outbox commit atomically. Before a verified assigned
    Agent mapping no Agent can use the grants; the exact assigned generation can
    only after proving its persistence capability, while another/fenced
    generation cannot claim them with copied registration fields or Attempt ID.
20. Replacing, omitting or adding one blob in a snapshot grant closure changes
    its hash and fails before materialization; the exact closure succeeds without
    granting another same-scope artifact.
21. Retention keeps every blob reachable from a retained project snapshot even
    when no RunState names the blob directly; after the final manifest/RunState
    reference expires, the exact dependency closure becomes collectible.
22. A corrupt, cyclic or incomplete compound-artifact dependency graph blocks
    collection and materialization instead of deleting a still-reachable child.
23. Attempt creation commits a closed operation-start gate; assignment commit
    opens it for the exact registration/deadline or rolls back before send. A
    loss/cancel/retry race then proves the CAS seals the old gate before evidence
    is read: no later intent can start, a pre-seal intent can finalize
    monotonically, and retry waits for reconciliation without
    `EffectEvidenceReader` mutating the gate.
24. A fenced old generation cannot read new artifact bytes or stage output even
    when it copies current registration fields. Any bytes cached in its old local
    workspace remain unable to start operations after gate/deadline closure or
    pass the result fence.
25. Race terminal WorkerResult/error ingestion against a new model/tool intent.
    Either the intent commits first and is present in the evidence snapshot, or
    result/usage plus gate seal commit first and the intent reaches no provider.
    The acceptance CAS pins the exact terminal-ingest gate version.
