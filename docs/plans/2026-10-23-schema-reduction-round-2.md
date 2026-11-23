# Schema reduction, round two — 2026-10-23

Design assessment, revised after source/specification review on `main`
`14e56f91840ca0b87002329b0e98d7b433c6e44b`. The migrated schema has **72
application tables**, excluding `contractor_schema_migrations`; the earlier
70-table inventory omitted `scan_sweeps` and `scan_sweep_members`. This document
creates no implementation task and changes no task status. The proposals below
are not ready migrations or authorization to implement them.

The [first round](2026-10-04-schema-reduction.md) reduced 91 tables to 70.
That result does not establish that another table with a matching key is
redundant. The [SQL review](../research/2026-10-23-sql-review.md) is historical
input; current source, owning specifications and the corrected V407/V409 task
files determine the contracts and retained work.

## Summary

Physical consolidation is an **optional chore/refactoring** when it preserves
behavior. A confirmed defect requires a specific violated contract and a
regression case. The corrected V409 batch retains defects in transaction
retries (V409-001), Run/Stage lock ordering (V409-003), review-write preconditions
(V409-005) and bounded Eval purge (V409-009). Those tasks own their fixes;
reducing the number of tables is not a substitute for completing them.

This review found errors in the original proposal and conditional risks of
its implementation, not evidence that the current satellites are all defects.
In particular, Gateway failure receipts must survive recovery; Eval selections
are mutable; resume receipts carry transaction authority; and counter scope,
settings CAS, report retention and command recovery cannot be inferred from
similar column names.

| Area | Current decision | What could justify reconsideration |
| --- | --- | --- |
| Eval member satellites | Keep for now; optional partial fold; projections stay separate | Explicit mutation/constraint/lock design and demonstrated benefit |
| Stage transition and resume receipts | Optional two-table fold, not ready to implement | Equivalent receipt/thaw authority and measured tradeoff |
| Audit/Eval controller claims | Optional shared storage, not a kind/id-only substitution | Parent FKs, cascades, epochs and all callers preserved |
| Settings singletons | Keep separate | Benefit that justifies additional dispatch lock coupling without changing public contracts |
| Execution reports | Optional shared storage | Per-kind identity, complete-only uniqueness and cleanup budgets preserved |
| Eval collections and Gateway failure receipts | Keep both | Independent scope/retention design with a named benefit |
| Eval commands and progress observations | Keep both | Recovery/status or history-compaction design, respectively |

There is **no committed target table count or blanket performance guarantee**.
A smaller count is a consequence of an accepted design, not an acceptance
criterion. Keeping a table with a recorded reason is a valid outcome. Reopening
a proposal below is a design decision; it does not reactivate a deferred task.

## 1. Evidence and method

For each candidate inspect the fully migrated schema, not only the baseline:
PK/UNIQUE/FK dependencies, partial indexes, CHECK and deferred constraints,
trigger events and functions, production readers/writers, lock order and
parent deletion. A zero-or-one extension is established by a unique child
reference to an actual parent key. Equal key names or a shared Go helper do
not establish equivalent authority, lifecycle or storage cost.

Trace each field as immutable identity, write-once identity, mutable state,
derived data or retained evidence. Inventory failure outcomes, ownership,
replay, purge and read visibility before proposing the replacement. Measure
per-operation work on a pinned revision with documented data distributions;
counts of matching source files and unqualified table counts are not a cost
model. Each future implementation task collects its own pre-change evidence;
V409-013 is a final verification task, not the provider of this preflight.

Review evidence on this revision included disposable PostgreSQL 17.11 checks:

- `TestRecoveryRoutesOutliveRunsAndReceiptsDoNot` passed. An external probe
  performed acquire, failure and successful recovery through the real service:
  replaying the old failure left the route open while its receipt remained;
  deleting that receipt made the identical replay block the route again.
- `TestPostgresBulkExperimentWritesBumpCollectionsOncePerStatement` and
  `TestSharedPostgresEvalUsageTicksKeepListCursors` passed, covering owner-wide
  and Project collection revisions and usage-only cursor preservation.
- `TestPostgresResumeEscalatedStage` and
  `TestPostgresResumeFailedStageIsAtomicAndIdempotent` passed. An external
  probe confirmed that reusing the source's configuration `escalation_ordinal`
  is rejected with `23514`; merely adding nullable receipt fields does not
  extend the existing Stage guards or decoded Stage equality tests.

All five existing tests and both external probes ran without skips. These
checks establish present guarantees and controlled counterexamples, not
correctness of a new migration. The Eval lock cycles below are conditional
inferences for a direct fold, not reproduced deadlocks on current `main`.
No production latency or before/after schema workload was measured.

## 2. Tables by domain at the reviewed revision

The inventory is descriptive; its domain grouping is not a redundancy score.


| Domain | Tables |
| --- | ---: |
| Runs, Stages, Scheduler (`workflow_runs`, `stage_executions`, `stage_allocations`, `workflow_run_events`, `workflow_run_output_publications`, `planner_sessions`, `planner_execution_reports`, `allocation_execution_reports`, `stage_metrics`, `stage_transition_decisions`, `run_stage_resumptions`, `owner_queue_controls`, `scheduler_settings`, `model_selection_settings`) | 14 |
| Gateway recovery (`gateway_recovery_routes`, `gateway_run_routes`, `gateway_allocation_routes`, `gateway_recovery_waits`, `gateway_recovery_failures`) | 5 |
| Artifacts (`artifact_scopes`, `artifact_bindings`, `artifact_binding_revisions`, `artifact_versions`, `artifact_blobs`, `artifact_lineage`) | 6 |
| Projects and receipts (`projects`, `mutation_receipts`) | 2 |
| Configuration and credentials (`runtime_agent_principals`, `runtime_config_versions`, `runtime_label_bindings`, `runtime_credentials`, `llm_credentials`, `credential_operations`, `git_ssh_keys`) | 7 |
| Telemetry (`performance_minutes`) | 1 |
| Audits (`audits`, `audit_rounds`, `audit_items`, `audit_executions`, `audit_execution_items`, `audit_artifact_links`, `audit_events`, `audit_report_pages`, `audit_controller_claims`, `audit_findings`, `audit_finding_assessments`, `audit_finding_contributions`, `audit_review_requests`, `audit_review_decisions`) | 14 |
| Findings (`finding_proposal_receipts`, `finding_proposal_audit_holds`) | 2 |
| Evals (`eval_experiments`, `eval_frozen_plans`, `eval_plan_resources`, `eval_dataset_revisions`, `eval_collections`, `eval_members`, `eval_submissions`, `eval_suboperations`, `eval_selections`, `eval_records`, `eval_evidence_refs`, `eval_member_projections`, `eval_execution_tombstones`, `eval_projection_queue`, `eval_view_pairs`, `eval_view_charts`, `eval_commands`, `eval_progress_observations`, `eval_controller_claims`) | 19 |
| Scan sweeps (`scan_sweeps`, `scan_sweep_members`) | 2 |
| **Total** | **72** |

## 3. Candidate decisions and required guarantees

Each proposal below is optional refactoring. Its specific acceptance is
required in addition to section 7 if implementation is later selected.

### 3.1 The Eval member family (5 tables; optional fold)

`eval_members`, `eval_submissions`, `eval_selections`,
`eval_execution_tombstones` and `eval_member_projections` each have primary
key `(experiment_id, member_id)`. Their zero-or-one cardinality makes some
of them candidates for a behavior-preserving refactor. It does not establish
that their mutation rules, locks or storage costs are interchangeable. No
migration is ready, and this proposal commits to no table reduction.

- **Candidate scope.** Consider submission, selection and tombstone columns
  on `eval_members` only after specifying their individual invariants.
  Keep `eval_member_projections` and `eval_projection_queue` separate. The
  projection row contains both `document` and `observed_document`, each
  bounded at 1 MiB, and is repeatedly dirtied and rewritten. Replacing those
  documents or moving their locks needs a separate design preserving complete
  view generations, not just a list of scalar filter columns
  ([EVM-R164–168](../spec/evals/managed-evals.md#read-model-and-summaries)).
- **Mutation rules.** Distinguish frozen member identity, nullable identity
  fields that become immutable once assigned, evolving submission state,
  token high-water marks and owner-selected records. Selection is mutable:
  [SelectRecords](../../internal/evalstore/selections.go) replaces exact
  result/assessment SHA-256 references, records the actor and increments the
  selection revision under owner CAS. Preserve both record FKs, assessment
  association checks, dirty marking and view-generation changes. A second
  valid **Save and use** must still work; a blanket set-once trigger would
  violate [EVM-R159–162](../spec/evals/managed-evals.md#assessments-and-selection).
- **Admission and recovery.** Preserve the frozen-plan/member relationship,
  submission key, actor and execution-kind identity, once-bound execution
  and workspace identities, terminal-state protection and observed usage.
  `eval_suboperations` currently references an admitted submission, not just
  any frozen member. A replacement must enforce that distinction, operation
  key/request identity, replay and claim/cancel fences. Retain durable intent,
  ordinary service call and acknowledgement as separate phases
  ([EVM-R108–111](../spec/evals/managed-evals.md#coordinator-claims-and-native-controls)).
- **Tombstone lifetime.** Tombstones survive ordinary execution deletion but
  already cascade with the Eval member. Their lifetime alone does not prevent
  a fold. Preserve execution provenance, terminal outcome, `never_started`,
  deletion time and recovery without recreating a deleted execution. Do not
  confuse them with mutation receipts, which survive experiment purge until
  Project purge ([EVM-R223–226](../spec/evals/managed-evals.md#retention-and-deletion)).
- **Locks.** The current settlement path locks the experiment, then updates
  its submission; ordinary execution deletion writes a tombstone, then
  updates the experiment. Moving both writes onto a member without changing
  that order can introduce experiment→member versus member→experiment waits.
  The ordinary Run deletion path does not first take the Eval experiment
  lock; Eval's Project `FOR SHARE` is not an exclusive serialization gate.
  This is a conditional risk of a future fold, not a current deadlock finding.
  If projections were folded too, a submission/selection UPDATE could hold
  the member before its dirty trigger asks for the queue, while source
  Run/Audit dirtying takes queue→member. Keeping the queue alone would not
  prevent that inversion. Keep the order documented by
  [migration 000018](../../internal/persistence/migrations/000018_eval_projection_lock_order.sql)
  and EVM-R166, or prove an equivalent order across all affected writers.
- **Reads.** Inventory and admission reads can lose particular satellite
  lookups. `selectedMemberPageSQL` instead reads published `eval_view_pairs`
  plus frozen member identity; this fold does not replace that snapshot with
  one live member row. Inventory still needs ordinary Run/Audit facts, and
  reconciliation still needs ownership and bounded eligibility reads. List
  savings for each actual statement; there is no established eight-query
  constant or proof that every page becomes cheaper.
- **Indexes and writes.** The final schema must retain equivalent execution
  and workspace uniqueness, active-submission access and reconciliation
  ordering. Today these are two partial unique indexes plus outstanding and
  reconciliation indexes. `ObserveTokens` updates `updated_at` to rotate
  bounded polling, and that column is indexed. Such updates do not become
  HOT merely because the old member indexes omit the new columns. Include
  required predicates, row width, WAL/index work and lock waiting in the
  comparison; do not promise equal writes or a universal HOT ratio.

**Acceptance before a fold is selected.** The Eval store/service owner must
supply the final constraint/index inventory, writer lock matrix and migration
mapping. Given a fully populated experiment, migration and transaction-abort
checks must preserve frozen membership, admitted-submission identity, exact
selections, suboperations and tombstone recovery. Given two valid selections
and a stale CAS, the first two succeed and the stale mutation remains rejected.
Given synchronized deletion/settlement and source-dirtying/reselection races,
no new inverted lock order may remain. The existing collection concurrency,
claim-loss, replay and partial-purge cases must retain their outcomes on real
PostgreSQL. Given representative workloads before and after, measured reads,
write work and lock waiting must justify the change without dropping required
indexes or replacing published views with live data. Use admitted native
workloads up to 1000 members and label any 10000-member store-seeded fixture
separately; neither is a whole-database cardinality bound.

**Verdict and reopening.** Keep this as an optional chore proposal owned by
the Eval store/service area. Reopen when the constraint and lock design above
is available and measured benefits justify it. Coordinate with V407-028/029
and V409-009/014 if their changes land first: V409-009 now owns bounded purge;
V409-014 owns safe Eval page/query optimizations. Neither promises a replacement
for the member projection documents. Deferring this proposal does not change
any task's status or require a migration merely to reach a table-count target.

### 3.2 Stage execution satellites: optional fold of two receipts

`stage_transition_decisions` and `run_stage_resumptions` each record one
immutable receipt per source execution. They have production writers and
readers; their separate tables are not a demonstrated defect. Folding them
onto `stage_executions` is an optional chore that could remove two tables if
the replacement preserves their authority, replay and lifecycle guarantees.
Keep `stage_metrics` and `planner_sessions` separate: metrics have their own
derived-data expiry, while Planner state has a separate update pattern and a
2 MiB bound. Trigger scoping alone does not justify merging either.

**Representation.** Use distinct nullable receipt groups on the source Stage:

- Transition: `transition_action`, `transition_target_stage_name`,
  `transition_target_execution_id`, `transition_decided_at`,
  `transition_escalation_ordinal`, `transition_escalation_exhausted`.
- Manual continuation: `resume_target_execution_id`, `resume_requested_by`,
  `resume_requested_at`, `resume_transaction_id`; retain the target's
  `resume_source_execution_id` and immutable execution lineage.

The existing `stage_executions.escalation_ordinal` belongs to the current
attempt's pinned configuration. It must remain distinct from the outgoing
decision's ordinal: a base attempt has NULL while its first escalation
decision has 1 (`stage_finalization.go:350–357,399–431`; migration 000016).
Do not relax the existing configuration-shape or immutability checks.

**Receipt and referential guarantees.** Before replacing the tables, define
the following constraints and transaction behavior in the migration design:

- Initialize each group once with a compare-and-set on its presence marker.
  A present transition freezes its entire tuple, including NULL target and
  ordinal members; a later manual continuation freezes a separate tuple.
  Preserve the original automatic `fail` decision after manual continuation.
  Reuse the action/target/escalation CHECKs, with explicit absent/present
  group shapes; SQL NULL must not bypass an incomplete-receipt check.
- Preserve one decision per source and unique non-null transition targets,
  plus the same-Run/target-Stage FK. A unique target index alone does not
  establish a receipt, especially for targetless `succeed`/`fail` decisions.
- Preserve one manual target per source, unique resume targets and both
  directions of source/target agreement. The target must copy the source's
  Run, Stage, attempt progression, exact snapshots and effective configuration.
  Replace the existing deferred `stage_execution_resume_receipt` FK and
  receipt-validation trigger with an explicit equivalent commit-time check;
  a bare self-FK to an existing source is insufficient. A forged manual
  target without its matching receipt must still fail at commit.
- Move the artifact-binding thaw predicate as part of the same change.
  It must still require the receipt's transaction ID to equal
  `pg_current_xact_id()`, the ordinary Run to be pending and the matching
  target to be preparing, while preserving the current output revision
  (`000001_baseline.sql:714–725`). An old receipt must not authorize a later
  transaction to thaw outputs.
- Keep transition completion, next-attempt creation and receipt atomic
  (SCH-R77/78); preserve Project-before-Run authority for resume and
  Run-before-Stage order where both are locked. Read an existing resume
  receipt before rechecking current eligibility: response-loss replay must
  return its original target even after that target fails (RUN-R47–51).
  Preserve existing conflict/rollback handling and Run/Project purge; new
  self-references must neither retain orphan receipts nor block parent purge.

**Cost and verification.** Transition listing currently serves Run detail
(`runservice/detail.go:132–135`); resume uses a separate receipt lookup
(`resume_store.go:77`). A fold may remove a read, but it replaces satellite
insertion with a source-Stage update. Completion and decision recording are
currently separate statements, and resume does not update its source Stage.
Measure the chosen implementation's statements, row versions, applicable
index work and trigger calls for each transition and resume. Do not promise
cheaper reads, equal writes, HOT eligibility or one added index entry without
before/after evidence. Coordinate with V407-025, but add protection for the
new receipt groups: that task only scopes existing Stage checks.

Acceptance for an implementation task must include real PostgreSQL tests for:

- Concurrent receipt initialization, duplicate/conflicting targets and all
  transition actions, including exhaustion, summary retry and manual resume.
- Direct SQL changes to every new field: replacement, clearing and filling a
  previously NULL member after its group exists; incomplete groups, target
  grafts, foreign-Run targets and missing reciprocal receipts must be rejected.
  Existing `workflow_runs` trigger checks and decoded Stage equality do not
  cover new Stage columns; test those columns and their guards directly.
- Concurrent and late resume replay, unchanged source terminal facts and
  automatic decision, same-transaction thaw, rejected old-receipt thaw,
  cancellation/rollback and fully populated Run/Project deletion.
- Migration from populated source tables, preserving receipt identities,
  timestamps, transaction IDs, target links and replay results; malformed
  input fails atomically. Verify the resulting constraints and purge on the
  migrated schema, without compatibility readers or fallback storage.

The RunStore/Scheduler owners may select this optional fold only when the
contract and measured tradeoff above are established. Retaining either table
with recorded evidence is a valid outcome.

### 3.3 Controller claims: optional consolidation, with typed ownership

The Audit and Eval claim tables already share the lease shape and
`persistencepostgres.ControllerClaims` implementation. Migration `000014`
added Eval's `claimed_at` and the same free/held shape check. Both tables have
an epoch, a holder and lease timestamps, a primary key, an expiry index and a
foreign key to their own parent with `ON DELETE CASCADE`.

- **Current behavior.** Audit reconciliation leases a bounded batch and releases
  each claim after its pass (`auditcontroller/controller.go:49-82`). Its explicit
  renewal is for deletion's separate cleanup budget (`:95-103`), not a timer
  renewing every active Audit each third of a lease. Renewals change indexed
  `expires_at`; the Scheduler's inline Run claim is not evidence of identical
  update cost here.
- **Candidate.** One table keyed by `(subject_kind, subject_id)` could remove
  one table, but that key alone cannot preserve the two parent FKs. Before
  implementation, specify typed nullable/generated references with strict
  kind/subject checks, or another design that preserves both parent existence
  and cascading deletion. Account for the resulting indexes: the primary key
  and expiry access path alone already require more than one index.
- **Callers and authority.** Every lease, live/held proof, release, renewal and
  direct claim join must include the correct kind. Preserve epoch advancement,
  stale-holder rejection and the narrower rule allowing an expired holder only
  to release or purge its own work. The standalone Sweep caller is removed by the user decision of 2026-10-31
  (V425-033); preserving its inline claim path is no longer a requirement.
- **Acceptance for any fold.** Real-store migration and concurrency tests must
  cover equal textual IDs in different kinds, rejection of missing/wrong-kind
  parents, both cascades, free and held claims, takeover, release and deletion.
  Migration must not reset epochs or revive an expired/released lease. Retain
  remaining controller claim tests and audit every direct join. Compare representative
  lease/release and deletion-renewal work before claiming a performance result.
- **Decision and owner.** Keep the current tables until the Persistence owner,
  with Audit and Eval owners, has a complete FK-preserving design and
  evidence of a maintenance or workload benefit. This is an optional chore,
  not an approved implementation task. Record any changed storage decision in
  [PER-R25](../spec/platform/persistence.md). Reopen an inline Audit
  alternative only after relevant trigger scoping and measurements of actual
  claim writes and parent-row contention; trigger scoping alone does not prove
  that the fold is beneficial.

### 3.4 Settings singletons: keep separate

`scheduler_settings` owns `max_concurrent_runs`, `max_concurrent_tool_runs`
and its revision/timestamp. `model_selection_settings` owns the default model,
role overrides and a different revision/timestamp. Their separate API resources
have independent conditional updates and no-op behavior; the separation is
specified by [OPS-R47–50 and OPS-R225](../spec/operations/operations.md).

- **Initialization differs.** A migration seeds Scheduler settings. Model
  selection is optional: without a deployment catalog its row can be absent;
  startup initializes it once from the catalog and must never overwrite a
  saved operator choice. A combined row would need an explicit, strictly
  validated uninitialized model group and atomic seed-if-uninitialized logic.
  Inserting a default model or treating an existing Scheduler row as an already
  initialized model selection would change supported behavior.
- **Concurrency differs.** Independent revisions and `updated_at` values are
  required by the existing resources, not by the arrangement of UI cards.
  A model update must not invalidate the Scheduler ETag or publish a Scheduler
  change. Even with separate counters, a combined row adds lock coupling:
  each Audit execution-intent reservation locks Scheduler settings `FOR UPDATE`
  before its claim, Audit and Round (`auditstore/internal/pgstore/execution.go:32-47`;
  [AUX-R6](../spec/audits/execution.md)). Model edits would then share
  that dispatch lock. Rare settings writes do not establish unchanged cost.
- **Decision and owner.** Keep both tables and the existing typed storage
  contract. Operations/settingsstore, with Audit dispatch and Model Selection
  owners, may reopen consolidation only for a demonstrated benefit and a
  reviewed lock design. Any future candidate must preserve both revision and
  timestamp pairs, field-specific checks/notifications, optional catalog
  startup, seed-once behavior and saved choices. Acceptance requires real-store
  tests for independent concurrent edits, stale/no-op requests, absent catalog,
  first initialization, restart and dispatch ordering. A shared revision or
  changed API semantics would require a separate explicit product decision;
  neither is part of this schema-reduction proposal.

### 3.5 Execution reports: optional shared storage, separate report semantics

Planner and allocation reports share some envelope and retention fields, but
both can retain multiple physical rows. An incomplete placeholder and a later
complete report can coexist. The effective reader prefers completeness; it
must not mistake the number of effective participants for storage cardinality
(`telemetry/repository.go:96-138,209-256`, `repository_sql.go:5-23`).

- **Candidate and constraints.** A shared table must keep report-kind-scoped
  identity/replay and immutable payloads, not introduce unconditional uniqueness
  on allocation or session. Preserve the complete-only keys: allocation reports
  use `(stage_execution_id, allocation_id, logical_agent_name)` when both
  Worker and Runtime reports are complete; Planner reports use
  `(stage_execution_id, session_id)` when complete. Retain Planner start/finish
  timestamps and ordering checks. Kind checks must require the proper envelope
  and owner columns, including non-nullness, while preserving the allocation's
  composite FK and the Planner session/Stage FKs with their cascades.
- **Historical receipt flag.** `stage_allocations.execution_report_received`
  means that a report **was ever stored**, not that one exists now. Migration
  `000030` makes it sticky so history distinguishes `report_expired` from
  `report_missing` after retention ([OPS-R183](../spec/operations/operations.md)).
  Preserve existing true values during migration; only allocation-kind inserts
  set the flag, and cleanup never resets it. Operations already joins the best
  unexpired report for resources (`telemetry/allocation_resources_sql.go:38-53`);
  the flag records information that the join cannot recover after deletion.
- **Retention and cost.** Preserve terminal-Stage-only deletion and an independent
  bounded budget for each report kind. The current cleanup gives allocation
  and Planner reports separate batches (`telemetry/repository.go:319-358`);
  one global oldest-first limit could starve one kind behind the other's
  backlog. One SQL round trip can still contain two kind-specific candidate
  scans, so fewer statements do not prove cheaper work. Keep StageMetrics and
  its Eval retention/renewal budget separate. V409-007 investigates index
  candidates; it does not promise a mandatory index change or a particular
  scan plan. Measure any proposed index/cleanup benefit on representative
  disposable fixtures.
- **Acceptance for any fold.** Real-store tests must preserve redaction, exact
  replay/conflict handling, placeholders followed by late complete reports,
  complete-first selection, frozen semantic outcomes, owner cascades and
  `report_expired` after cleanup. A mixed backlog must make bounded progress
  for both report kinds without deleting nonterminal telemetry. Verify the
  actual reader and cleanup plans; do not assert unchanged cost without data.
- **Decision and owner.** Telemetry, with Operations and execution owners, may
  pursue this as an optional storage chore only after the constraints and
  per-kind budgets above are explicit. Reopen on demonstrated maintenance or
  workload benefit, preserving [OBS-R8–9, R37–44](../spec/platform/observability.md).
  No table-count target or approved implementation task follows from this
  candidate alone.

### 3.6 Eval collection revisions and Gateway failure receipts: keep both

**Eval collections.** `eval_collections (owner_id, project_id, revision)`
tracks mutable dataset and experiment lists in two namespaces: a Project
and the owner's entire collection, represented by `project_id=''`. Triggers
advance both scopes for relevant changes, once per statement where specified;
usage-only and semantic no-op changes retain usable cursors. See
[collection readers](../../internal/evalstore/public_read_sql.go),
[collection tests](../../internal/evalstore/collections_postgres_test.go) and
[EVM-R212](../spec/evals/managed-evals.md).

A `project_counters (project_id, counter_kind, value)` table or columns on
`projects` alone cannot represent the owner-wide scope. Omitting owner identity
would also mix different owners' empty-Project keys. V409-001 preserves the
existing Project correlation revision and early lock; it neither creates a
counter table nor chooses a new home for Eval revisions. Replacing one table
with another would not by itself reduce the table count.

Keep `eval_collections`. The Eval list/pagination owner may reopen a shared
counter design only with a demonstrated caller/benefit and explicit namespace,
locking and deletion rules. Acceptance must preserve owner isolation, both
cursor scopes, dataset and experiment invalidation, once-per-statement updates,
usage/no-op behavior and concurrent mutations across several Projects. Do not
trade those guarantees for a generic table name or a different lock protocol.

**Gateway failures.** `(route_key, request_id)` is an idempotency receipt,
with `run_id` owning its retention. The result of `INSERT ... ON CONFLICT DO
NOTHING` is a production consumer of its identity, even without a separate
SELECT reader. [The update path](../../internal/gatewayrecovery/requests.go)
keeps receipts after a successful probe, so an old duplicated failure cannot
re-block a recovered route. Stage end also retains them; hard Run deletion
cascades its receipts while the shared route can survive
([SCH-R169](../spec/execution/scheduler.md)). **Do not delete receipts
on route recovery or replace their retention with a route-prefix delete.**

`mutation_receipts` is not a drop-in destination. Its `key_scope` is a text
array with family-specific checks; it requires operation/digest/response data,
has a 128-byte idempotency-key limit and has no Run FK today. Gateway request
IDs admit up to 256 bytes. A proposed key encoding, owner/FK/cascade design,
immutable receipt shape and interaction with the existing Run→route lock order
would need to be specified and measured. Reusing `KeyedMutation` also requires
accounting for its advisory-lock/replay protocol; it cannot be justified as
adding a scope with no work or index consequences.

Keep `gateway_recovery_failures`. The Gateway recovery owner, with Persistence,
may reopen a storage-only consolidation when those guarantees and a maintenance
or workload benefit are demonstrated. Acceptance must cover repeated failures
before and after successful recovery, shared routes across Runs, maximum legal
request IDs, Stage end, Run purge, concurrent delivery and lost responses.
The result must retain the same receipt identity and lifetime without a new
retry/lease protocol. The unsafe recovery-time deletion is rejected, not a
candidate awaiting a performance test.

### 3.7 Journals (keep both; optional redesign)

`eval_commands` and `eval_progress_observations` have production readers and
different guarantees. Keep both tables. Their absence of a public history
pagination endpoint does not make them unused, interchangeable with receipts
or bounded by a known number of writes per experiment.

- **Commands are mutable recovery state.**
  [CompleteCommand](../../internal/evalstore/commands.go) updates state,
  diagnostic and completion time. `CommandRecord` reads an exact command ID;
  `PendingCommands` reads bounded pending work, and the claim query includes
  experiments with accepted/running commands. After a crash between a
  lifecycle transition and its acknowledgement, the controller completes
  those commands even if the experiment is already ready or terminal. The
  public GET and UI already poll them; these needs are not an undecided UI
  feature ([EVM-R100–107](../spec/evals/managed-evals.md#commands-and-idempotency)).
- **A command status is not its mutation receipt.** Command POST stores an
  immutable acceptance response; GET observes subsequent completion. Under
  [PER-R24](../spec/platform/persistence.md#keyed-mutations-and-controller-claims),
  a claimed `mutation_receipts` response cannot change. Command rows leave
  with their experiment, while Eval mutation receipts remain replayable
  until Project purge. A future shared table would need to preserve separate
  replay bytes and mutable status, command-ID lookup, pending-work claiming
  and indexes, and both retention rules. Simply adding state to the current
  immutable receipt does not provide that design.
- **Progress is a chart read model.** View publication persists real server
  observations. The chart reader selects the last observation in each of at
  most 200 display buckets, with suite-specific counts, gaps and a cutoff at
  the selected publication. It is a separate chart query, not history already
  returned with every experiment read. LIM-EVM-13 limits the response, not
  the number of stored observations. See
  [EVM-R200–201](../spec/evals/managed-evals.md#charts) and
  [progressSQL](../../internal/evalstore/progress_sql.go).
- **An array cap is a retention decision.** Keeping only the latest N values
  can discard the start of an experiment; a fixed coarse sample need not
  retain the last real observation for later recomputed display buckets.
  Any bounded representation needs an explicit compaction/retention policy
  preserving the required range, suites, gaps and snapshot semantics. It also
  adds repeated JSON rewriting on the authority row. A new LIM entry alone
  proves neither semantic equivalence nor acceptable write cost.

**Acceptance before reopening.** Given a crash after a command's lifecycle
transition but before acknowledgement, restart must discover and finish the
same command; replay must return the original acceptance bytes, while GET
shows the correct completion. Experiment and Project purge must retain their
separate command/receipt outcomes. Given more observations than the proposed
storage cap, long pauses, missing suites and snapshot cutoffs, the progress
chart must retain the specified real-observation bucket and gap semantics;
no silent truncation or fabricated points may pass. Real-store migration,
recovery and chart tests must cover these cases, and a representative workload
must demonstrate bounded storage and acceptable write/lock cost.

**Owner and decision.** The Eval control/read-model owner keeps these proposals
outside the implementation sequence. Reopen commands only with an explicit
status-versus-replay and retention design; reopen progress only with an
agreed storage/compaction contract and measured workload need. Reuse existing
recovery mechanisms. No current task is put into `deferred` by this decision,
and neither optional redesign contributes to a promised table reduction.

### 3.8 Other tables retained in this round

| Table | Reason to retain it |
| --- | --- |
| `eval_projection_queue` | Source Run/Audit dirtying avoids taking the experiment row and orders queue before projection rows (000018). Some private projection writers also hold the experiment; the queue does not make folding other rows automatically safe. |
| `eval_frozen_plans` | Parent of frozen members, with immutability and deferred complete-matrix authority. V407-029 narrows metadata reads; it does not remove the stored plan or its required full readers. |
| `stage_metrics`, `planner_sessions` | Independent derived-data retention or frequent bounded document updates; see 3.2. Trigger scoping is not a sufficient reopening condition. |
| `artifact_scopes` | Run/Project binding scope identity and revision-count admission, with its own parent/child contract. |
| `artifact_versions`, `artifact_blobs` | Versions reference deduplicated content: this is N:1, not a redundant 1:1 pair. |
| `gateway_run_routes`, `gateway_allocation_routes`, `gateway_recovery_waits` | Distinct Run×route, allocation×model and participant×model identities and recovery locks. |
| `audit_events`, `audit_report_pages`, `workflow_run_events` | Paged event/report contracts with distinct sequence, ordinal and retention rules. |
| `audit_finding_contributions` | Implemented finding↔receipt provenance: admission writes `first` contributions; assessment, collection, provenance and publication readers use them. The table is not unimplemented. |
| `owner_queue_controls`, `git_ssh_keys` | Separate per-owner Scheduler and credential authority; sharing an owner ID does not justify merging them. |
| `mutation_receipts`, `credential_operations`, `eval_suboperations` | Existing distinct replay and outbound-operation mechanisms. Sections 3.6/3.7 do not automatically belong in these tables. |
| `scan_sweeps`, `scan_sweep_members` | Sweep authority and member execution lifecycle introduced by 000031; claims already reuse the shared helper on the sweep row. |

For `audit_finding_contributions`, production evidence includes
[admission](../../internal/persistence/migrations/000003_audit_event_allocator.sql),
[collection reads](../../internal/auditstore/internal/pgstore/finding_collection.go)
and [publication authority](../../internal/findingintake/collection_publication_sql.go).
An incomplete feature for another contribution kind would not make these
existing writers/readers dispensable. For a future proposal affecting any
retained table, its owning subsystem must identify a concrete new benefit and
an equivalent authority/lifecycle design before reopening it; lack of a new
proposal creates no implementation work.

## 4. Performance and migration evidence required for a candidate

A shared row can save a lookup while adding contention, row versions or index
work. Compare actual statements and update mixes before and after a concrete
design. In particular:

- Preserve the union of required uniqueness, FK-probe, pagination, eligibility
  and reconciliation access paths. Count only indexes that actually disappear;
  constraint-backed indexes are not redundant because they have few scans.
- HOT eligibility depends on the **final** indexes and their predicates as
  well as page space, not the old parent definition. Submission polling changes
  indexed `updated_at`; Stage transition targets would become indexed. Use
  [PostgreSQL's HOT conditions](https://www.postgresql.org/docs/17/storage-hot.html)
  when interpreting statistics. There is no universal 90% acceptance threshold.
- Account for inline row density, document size, TOAST access/rewrite, WAL,
  buffer work and trigger invocations per named operation. TOAST can avoid
  rewriting unchanged out-of-line values, but does not remove row locks,
  index maintenance or the cost of rewriting a changed document.
- Test the combined lock graph with synchronized concurrent writers, including
  deletion and recovery. Measure representative waits separately from proving
  ordering; timing thresholds alone cannot establish correctness.
- State admitted producer limits and whole-database populations separately.
  Native Eval plans admit at most 1000 members within a 1 MiB document; a
  store-seeded 10000-member test is a defensive fixture, not that producer's
  normal maximum. Neither bounds the number of experiments or dependent rows.

Each candidate records PostgreSQL/schema revision, fixture generation, rows
and bytes, operation mix, parameters, statistics context and plans. Use a
disposable database for mutating `EXPLAIN ANALYZE`: it executes the statement.
No demo connection is required for this work. Optional live statistics are
bounded read-only observations and cannot replace reproducible fixtures or
justify dropping a constraint. Prefer existing test helpers to a mandatory
new generic harness.

The migrator applies **all pending files in one transaction** and checks their
immutable checksums. Its default statement timeout is 120 seconds, configurable
by the operator; it is not proof that a fold fits, nor a separate transaction
budget for each file. Inventory total backfill rows/bytes, locks and dependent
trigger/cascade work, and test the full upgrade route. If it does not fit the
chosen budgets, design bounded migration work separately before implementation.

Use new incremental migrations under the existing ledger. A new baseline is
outside this round and outside V409-012. Any later baseline proposal needs an
explicit upgrade/ledger design and verified backup/data preservation for a demo
database; pre-production does not authorize discarding business data. Runtime
compatibility readers, legacy fallbacks and edits to applied migration bytes
are not substitutes for a correct migration.

## 5. Review one-to-one schema extensions by their guarantees

A one-to-one table can represent phase authority, immutable evidence or an
independent lifecycle as well as a convenient data split. A future schema-review
gate may require a recorded decision for new extensions; equal keys alone
are not a reason to reject a table. Its implementation is a separate optional
chore, not a prerequisite for this documentation correction.
Identify candidates through an actual FK to a unique parent key together
with uniqueness of the child reference; matching column names/types alone
do not establish the relationship. Indirect or polymorphic relations need
explicit review rather than an inferred catalog result.

For each candidate, record the production caller, cardinality, authority and
failure/lifecycle contract, then compare a separate row with parent columns.
Allow documented authority, lifecycle, retention, immutability, lock-order,
contention and size/update-pattern reasons. Each retained exception names its
owner, evidence and condition for reconsideration. The gate checks that this
decision exists; it does not prove the proposed replacement correct. Receipt
constraints, new nullable-column guards, replay, migration and deletion still
need their own executable tests before a table can be removed.

## 6. Dependencies and a conditional implementation sequence

The following are coordination points with the **corrected** task scopes,
not a blanket requirement to finish every task before doing design work.
All are pending on the reviewed revision. A future implementation task must
name its actual technical dependencies; only completed tasks satisfy them.
Rebase and recheck overlapping work before execution rather than assuming that
a task ID supplies a design or a performance improvement.

| Existing work | Actual scope and relationship to this proposal |
| --- | --- |
| [V407-025](../../tasks/done/v407-025-persistence-indexes-and-triggers.yml) | Stage trigger scoping plus measured query/index work. Sequence a Stage fold after it if relying on its scoped definitions; add separate guards for new receipt tuples. |
| [V407-028](../../tasks/done/v407-028-eval-member-bulk-freeze.yml), [V407-029](../../tasks/done/v407-029-eval-reads-plan-metadata.yml) | Bounded member-freeze batching and metadata-only plan reads; preserve their limits and remaining full-document readers in any Eval design. |
| [V409-001](../../tasks/done/v409-001-finding-correlation-project-lock.yml) | Finite retries around existing finding import/retention transactions, retaining the Project fence/revision. It supplies no new counter table or counter-home decision. |
| [V409-003](../../tasks/done/v409-003-runstore-lock-order-and-cas.yml) | Correct Run-before-Stage locking in StartPlanner. Keep this defect separate; coordinate any Stage lock design with its corrected order and regression coverage. |
| [V409-006](../../tasks/done/v409-006-trigger-column-scoping.yml) | Scoped trigger work preserving admission and projection side effects. StageExecution triggers belong to V407-025; this task does not establish that parent-inline claims are cheap. |
| [V409-007](../../tasks/done/v409-007-dead-and-redundant-indexes.yml) | Per-candidate index preflight and justified decisions, including telemetry cleanup readers. Reuse the evidence and preserve required indexes; no predetermined DROP or cleanup-plan outcome. |
| [V409-009](../../tasks/done/v409-009-eval-store-bounds.yml) | Bounded, restartable Eval purge. Any Eval/claim consolidation must preserve its dependent-row/byte budgets and phases if it lands first, or have those changes explicitly rebased onto the selected schema. No projection redesign is supplied. |
| [V409-011](../../tasks/done/v409-011-execution-store-n1-and-duplicates.yml) | Safe execution-side query/round-trip improvements (binding, principal and release-marker batching). Recheck the actual statements after it lands. |
| [V409-014](../../tasks/done/v409-014-eval-comparison-and-recovery-queries.yml) | Safe Eval comparison-page and composed recovery-query improvements; complete-view redesign remains excluded. Recheck the actual statements after it lands. |
| [V409-012](../../tasks/done/v409-012-migration-and-sql-hygiene.yml) | Upgrade tests for 000008/000014 and migration-budget/ledger documentation. No baseline regeneration or historical-migration rewrite. |
| [V409-013](../../tasks/done/v409-013-verify-sql-review-on-database.yml) | Final SQL-review gate after V409-001…012 and V409-014. It is neither stage-zero preflight nor a harness prerequisite; no implementation dependency may point back to this exit task. |

Proposed sequence, only if a candidate is later selected:

1. **Candidate preflight.** Its owning subsystem establishes contracts, named
   callers, workload bounds and existing guarantees on the intended base.
   Record keep/refactor/reject, reason, evidence and reopening condition. Do
   not create code work solely to meet a table-count goal.
2. **Complete the design.** For a retained optional fold, specify schema,
   field mutation rules, read/write paths, lock order, lifecycle and migration.
   Reconcile overlapping tasks using the table above. Update the owning spec
   in the eventual implementation change where storage guarantees change;
   this non-normative plan cannot override it.
3. **Create a scoped implementation task.** Use `kind: chore` for equivalent
   storage. A separately discovered contract violation needs its own defect
   evidence and regression, not a relabelled consolidation. Record technical
   dependencies, failure outcomes and the acceptance in sections 3 and 7.
4. **Implement and verify only that selected change.** Use a new migration
   where needed and real-store upgrade/behavior tests. Do not automatically
   proceed from claims to settings, receipts, journals or a new baseline.
5. **Record the outcome.** Name implementation revision, evidence and actual
   table/index changes, or the supported decision to retain the current
   design. Update the related review disposition without claiming that pending
   or deferred tasks are complete. There is no fixed final table count.

## 7. Acceptance for any future implementation

These requirements supplement each candidate's specific cases. They are not
claims that a migration has been implemented or validated by this document.

- **A1 — Contract equivalence.** Given the original and proposed migrated
  schemas, enumerate every removed constraint, index and trigger and its
  replacement or evidenced redundancy. Direct invalid-write tests preserve
  identity, ownership, null-sensitive tuple shape, immutability and deferred
  commit-time rules. No guarantee may move silently into an untested caller.
- **A2 — Failure and recovery.** Given duplicate delivery, definite transaction
  abort, ambiguous commit, cancellation, timeout and process restart, preserve
  the subsystem's existing outcomes and bounded retries. Reconcile ambiguous
  outcomes through existing receipts; do not blindly repeat service effects.
  Lost-response replay must preserve original identities and response bytes
  where the owning contract requires them.
- **A3 — Concurrency.** Given synchronized overlapping production writers,
  prove the chosen order and authority/claim fences for admission, observation,
  selection, source updates, resume and deletion as applicable. Preserve
  classified conflict handling; retries must not conceal an introduced lock
  inversion or use a whole-tick transaction across external effects.
- **A4 — Lifecycle and read visibility.** Given populated parents and mixed
  retained/expired children, deletion and partial purge preserve per-pass row
  and byte budgets, owner isolation, progress, cascades and retention evidence.
  Lists/cursors, selected view generations, command status and report history
  retain their defined visibility during concurrent updates and cleanup.
- **A5 — Upgrade.** Given populated predecessor schemas, apply the new
  migration through the real runner and verify field/receipt/epoch/revision
  mappings, constraints and post-upgrade operations. Failures roll back schema,
  data and ledger together. Test the chosen budgets against total work; do not
  reset business data, rewrite historical migrations or add legacy readers.
- **A6 — Evidence for the tradeoff.** Before implementation, state operation,
  cardinality and byte bounds plus acceptable resource/work budgets. Compare
  named queries, write/index/trigger work and concurrency on representative
  fixtures, with stress fixtures labelled separately. Demonstrate the claimed
  maintenance or workload benefit and record remaining costs; a lower table
  count, absent index-scan name or global HOT percentage is insufficient.
- **A7 — Verification and delivery.** Tests must distinguish a broken mapping
  or lost guarantee, not merely count columns/tables. Run `make lint` and the
  full suites of touched packages, with `-tags=integration -count=1` and a
  nonempty `CONTRACTOR_TEST_DATABASE_URL` for real-store coverage. Update and
  run spec/API/fixture gates where their boundaries change. Unreachable-DB,
  skipped and zero-match runs are not passes. Record exact commands/outcomes,
  implementation commit and whether the branch is merged under task policy.

Any future schema-review guard in section 5 checks the recorded decision;
passing that guard or lint alone cannot satisfy A1–A6. A proposal that remains
unselected keeps its stated owner/reopening condition without becoming an open
implementation obligation or an implicitly completed task.
