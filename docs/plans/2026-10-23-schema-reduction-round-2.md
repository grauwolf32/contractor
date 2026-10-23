# Schema reduction, round two — 2026-10-23

Analysis only; no task exists yet. Revision analysed: `main` `fa235aeb`
(70 tables, excluding `contractor_schema_migrations`). Source data: the
[SQL and schema review](../research/2026-10-23-sql-review.md) and a per-table
count of production readers, writers, triggers and keys (method below). The
[first round](2026-10-04-schema-reduction.md) (V365, 91 → 70) folded strict
1:1 extensions and dropped tables without readers; this document asks what
is left, why, and what merging it would do to performance.

## Summary

About 50–55 of the 70 tables carry a distinct entity, a distinct lifecycle or
a distinct cardinality. The remaining 15–20 are of four kinds:

1. **1:1 satellites** of a parent row, keyed by the parent's key, that exist
   because a lifecycle phase was modelled as a table instead of as nullable
   set-once columns: five tables keyed by `(experiment_id, member_id)`, four
   keyed by `stage_execution_id`, one keyed by `experiment_id`.
2. **Pairs of the same shape** served by one Go type: the two controller
   claim tables, the two singleton settings tables, the two execution report
   tables.
3. **A counter or a receipt in its own table**: `eval_collections`,
   `gateway_recovery_failures`.
4. **Journals that are never paged**: `eval_commands`,
   `eval_progress_observations`.

Merging the first three kinds removes 13–16 tables. Performance does not get
worse as long as two rules hold, both stated in the detail below: a column
that is written on a hot path never moves onto a row whose triggers or
indexes fire on every update, and a large document never moves onto a row
that is locked or scanned often. Two merges that look like 1:1 folds violate
those rules and are kept out: `eval_projection_queue` (lock order) and
`stage_metrics` (until `stage_executions` triggers are column-scoped).

Recommendation: do not start before V407-025, V409-006 and V409-009 land,
because they change the same tables and triggers, then run the stages in
section 6. Target: 54 tables.

## 1. Method

For every table: primary key, unique constraints and foreign keys from the
migrations; the number of production Go files that write it (`INSERT`,
`UPDATE`, `DELETE`), read it (`FROM`, `JOIN`) and the triggers defined on it;
the columns that production statements update. A table is a 1:1 satellite
when its primary key (or a unique constraint) equals the parent's key and
nothing else distinguishes the rows. A pair is "same shape" when one Go type
is parametrised by the table name. "Hot path" means written per heartbeat,
per tool call, per claim renewal or per controller tick.

The counts are static; no `pg_stat_user_tables` was available. V409-013
collects the live numbers.

## 2. Tables by domain

| Domain | Tables | Of which candidates |
| --- | ---: | ---: |
| Runs, Stages, Scheduler (`workflow_runs`, `stage_executions`, `stage_allocations`, `workflow_run_events`, `workflow_run_output_publications`, `planner_sessions`, `planner_execution_reports`, `allocation_execution_reports`, `stage_metrics`, `stage_transition_decisions`, `run_stage_resumptions`, `owner_queue_controls`, `scheduler_settings`, `model_selection_settings`) | 14 | 5 |
| Gateway recovery (`gateway_recovery_routes`, `gateway_run_routes`, `gateway_allocation_routes`, `gateway_recovery_waits`, `gateway_recovery_failures`) | 5 | 1 |
| Artifacts (`artifact_scopes`, `artifact_bindings`, `artifact_binding_revisions`, `artifact_versions`, `artifact_blobs`, `artifact_lineage`) | 6 | 0 |
| Projects and receipts (`projects`, `mutation_receipts`) | 2 | 0 |
| Configuration and credentials (`runtime_agent_principals`, `runtime_config_versions`, `runtime_label_bindings`, `runtime_credentials`, `llm_credentials`, `credential_operations`, `git_ssh_keys`) | 7 | 0 |
| Telemetry (`performance_minutes`) | 1 | 0 |
| Audits (`audits`, `audit_rounds`, `audit_items`, `audit_executions`, `audit_execution_items`, `audit_artifact_links`, `audit_events`, `audit_report_pages`, `audit_controller_claims`, `audit_findings`, `audit_finding_assessments`, `audit_finding_contributions`, `audit_review_requests`, `audit_review_decisions`) | 14 | 1 (+1 product decision) |
| Findings (`finding_proposal_receipts`, `finding_proposal_audit_holds`) | 2 | 0 |
| Evals (`eval_experiments`, `eval_frozen_plans`, `eval_plan_resources`, `eval_dataset_revisions`, `eval_collections`, `eval_members`, `eval_submissions`, `eval_suboperations`, `eval_selections`, `eval_records`, `eval_evidence_refs`, `eval_member_projections`, `eval_execution_tombstones`, `eval_projection_queue`, `eval_view_pairs`, `eval_view_charts`, `eval_commands`, `eval_progress_observations`, `eval_controller_claims`) | 19 | 8 |
| **Total** | **70** | **15–16** |

Evals hold 19 of 70 tables for a secondary capability, against 14 for Run
execution. That is where most of the fold is.

## 3. Candidates, with reasons and performance consequences

Each entry states what the table holds, why it exists separately today, what
the merge looks like, and what changes for the database: row width, index
entries per write, trigger firings per write, lock scope, joins per read.
"HOT" means a heap-only tuple update, which PostgreSQL performs when no
indexed column changes and the page has room; a HOT update writes no index
entries. Large `bytea`/`jsonb` values are stored out of line (TOAST) and are
not rewritten when other columns of the row change, so row "width" below
means the inline part.

### 3.1 The Eval member family (5 → 1 or 2 tables)

`eval_members` (13 columns, the frozen plan member), `eval_submissions` (11,
the one execution started for it), `eval_selections` (6, which record was
selected), `eval_execution_tombstones` (7, written only by a trigger when the
execution is deleted), `eval_member_projections` (8, a comparison document up
to 1 MB) all have primary key `(experiment_id, member_id)`.

- **Why separate today.** Each phase arrived as its own task and modelled
  "exists or not" as a row. `eval_submissions` carries the FK target for
  `eval_suboperations`; the tombstone trigger wanted a row that survives the
  execution's deletion.
- **Merge.** Nullable, set-once columns on `eval_members`: `submission_state`,
  `submission_actor_id`, `execution_id`, `execution_project_id`,
  `terminal_state`, `observed_tokens`, `submitted_at`, `selected_result_kind`,
  `selected_assessment_kind`, `selection_revision`, `tombstone_*`. A
  column-scoped trigger keeps the set-once fields immutable (the V365 pattern).
  `eval_suboperations` references `eval_members`; its `CHECK` that a
  submission exists becomes `WHERE submission_state IS NOT NULL` in the
  trigger. `eval_member_projections` stays separate only if V409-009 keeps a
  document there; if V409-009 projects the fields into columns, it folds too.
- **Reads.** Every member page (`selectedMemberPageSQL`, inventory pages,
  `nextMembersSQL`, reconcile) joins two to four of these tables today; after
  the merge they read one row. The coordinator tick's per-member
  `recovery()`/`locked()` sequence (8 round trips, V409-009) drops the
  submission lookup.
- **Writes.** `UPDATE eval_submissions SET state, terminal_state,
  observed_tokens` (per member settlement, per token observation) becomes an
  update of `eval_members`. `eval_members` has four unique constraints plus
  `(experiment_id, ordinal)`; none includes the moved columns, so these
  updates stay HOT as long as the page has room. The inline row grows by about
  100 bytes (texts and a bigint); `recipe bytea` is already out of line.
  `eval_members` is inserted once per plan freeze (V407-028 makes that one
  statement), so write amplification is unchanged.
- **Locks.** Settling a member locks the member row instead of the
  submission row. Nothing else locks member rows during execution (the
  experiment row and the projection queue row are the contention points),
  so no new contention.
- **Verdict.** Faster reads, equal writes, 3–4 tables fewer. Prerequisite:
  V409-009 (it rewrites the same statements).

### 3.2 Stage execution satellites (4 → 1 or 2 tables)

`stage_transition_decisions` (PK `source_execution_id`, 8 columns: action,
target stage, target execution, escalation ordinal), `run_stage_resumptions`
(PK `source_execution_id`, 6 columns: target execution, requested by, at,
transaction id), `stage_metrics` (PK `stage_execution_id`, metrics and
summary `jsonb` with their own expiry), `planner_sessions` (PK `session_id`
but `UNIQUE (stage_execution_id)`, state `jsonb` up to 2 MiB).

- **Why separate today.** Transition decisions and resumptions are
  natural-key receipts (AGENTS 3.2 lists both): the unique key proves the
  decision was recorded once. `stage_metrics` is derived and expires
  (V365-016 deferred it for that reason). `planner_sessions` isolates a large,
  frequently rewritten blob.
- **Merge.** `transition_action`, `transition_target_stage_name`,
  `transition_target_execution_id`, `transition_decided_at`,
  `escalation_ordinal`, `escalation_exhausted`, `resume_target_execution_id`,
  `resume_requested_by`, `resume_requested_at`, `resume_transaction_id` as
  nullable set-once columns on `stage_executions`. The receipt property is a
  partial unique index `(run_id, transition_target_execution_id) WHERE NOT
  NULL`, which is what `stage_transition_decisions.target_execution_id UNIQUE`
  is today. `stage_executions` already carries
  `resume_source_execution_id` pointing back at the satellite, so the pair of
  FKs collapses into one column.
- **Reads.** The Scheduler's transition and resume paths read one row
  instead of a join; `listTerminalStageExecutionsWithAllocationsSQL` and the
  recovery queries do not touch these columns.
- **Writes.** Transition and resume are written once per Stage end; the
  update fires `stage_executions` triggers. Today those triggers fire on every
  `UPDATE` (V407-025); after V407-025 scopes them with `UPDATE OF`, a
  transition write fires only `capture_lifecycle_event` if `state` changes in
  the same statement, which it does today anyway. No index includes the new
  columns except the partial receipt index, so the write adds one index entry.
- **Not merged.** `stage_metrics` is updated repeatedly (summary refresh,
  expiry bumps: five `UPDATE stage_metrics SET expires_at` sites). Folding it
  into `stage_executions` would run the four `stage_executions` triggers on
  every metrics refresh and widen the row the Scheduler locks on every
  transition. Keep it until V407-025 lands and the metrics write rate is
  measured; then reconsider. `planner_sessions` stays: a 2 MiB `jsonb`
  rewritten per Planner turn belongs away from the row that holds the Stage
  lock, and TOAST would not help because the whole value changes.
- **Verdict.** Two tables fewer, reads cheaper, writes equal. Prerequisite:
  V407-025.

### 3.3 Controller claims (2 → 1 table, or 0)

`audit_controller_claims` (`audit_id, epoch, holder_id, claimed_at,
expires_at`) and `eval_controller_claims` (`experiment_id, epoch, holder_id,
expires_at`) are served by one type,
`persistencepostgres.NewControllerClaims(table, subject)`. The Scheduler's
claim for Runs is inline on `workflow_runs` (`scheduler_claim_id`,
`scheduler_claim_expires_at`), renewed with a heap-only update (SCH-R166).

- **Option A, one table** `controller_claims (subject_kind, subject_id,
  epoch, holder_id, claimed_at, expires_at)` with PK `(subject_kind,
  subject_id)`. Same statements, one more predicate. Renewals touch a
  five-column row with one index; cost unchanged. Claim lookups by expiry
  (`audit_controller_claim_expiry_idx`, `eval_claims_expiry_idx`) become one
  partial index per kind or one index led by `subject_kind`.
- **Option B, inline on the parent** like `workflow_runs`. Rejected for now:
  `audits` has 36 columns, 5 triggers and `audits_protect_preparation_baseline`
  runs an `EXISTS` on every update (V409-006); a claim renewal every third of
  the lease on every active Audit would pay that until V409-006 scopes the
  triggers, and the renewal would compete with the controller's own `audits
  FOR UPDATE` in the same transaction family. Reopen when V409-006 is done and
  the Audit row update rate is measured.
- **Verdict.** Option A: one table fewer, no performance change. Make
  `ControllerClaims` take a kind instead of a table name.

### 3.4 Settings singletons (2 → 1 table)

`scheduler_settings` (`max_concurrent_runs`, revision) and
`model_selection_settings` (`default_model`, `role_models`, revision) are
single rows with a `singleton` CHECK. One `operations_settings` row with both
groups and one revision, or two revision columns if the UI edits them
separately (it does, on different cards). Reads are one row either way;
writes are rare. One table fewer, no performance change.

### 3.5 Execution reports (2 → 1 table)

`planner_execution_reports` (several per Stage, by session and invocation)
and `allocation_execution_reports` (one per allocation and logical agent)
share `report_id`, `stage_execution_id`, schema version, `report jsonb`,
`received_at`, `expires_at`; they differ in the owner columns. One
`execution_reports (report_id, stage_execution_id, kind, session_id,
invocation_id, allocation_id, logical_agent_name, …)` with a CHECK per kind.
Migration 000030 added `stage_allocations.execution_report_received`, a flag
that mirrors the existence of a report row; it stays, because the Operations
read that needed it (V404-002) must not join reports.

- **Performance.** Cleanup (`telemetry/repository.go:340-358`) runs two
  statements over two indexes today and one after; the index mismatch
  V409-007 fixes applies once instead of twice. Reads by allocation or by
  session use the same composite indexes with `kind` added.
- **Verdict.** One table fewer, cleanup cheaper, reads equal.

### 3.6 Counters and receipts in their own tables (2 → 0)

- `eval_collections (owner_id, project_id, revision)` is the Project's
  dataset revision counter. It is the same pattern V409-001 proposes for the
  finding correlation fence, for the same reason: a counter that must not
  lock the parent row. The right end state is one small table
  `project_counters (project_id, counter_kind, value)` holding both, or both
  on `projects` if V409-001 shows the parent-row lock is acceptable. Either
  way one table fewer; the choice is V409-001's.
- `gateway_recovery_failures (route_key, request_id, run_id)` is a
  natural-key receipt (no production reader; the `INSERT … ON CONFLICT DO
  NOTHING` result is the signal). V370-007 folded three such tables into
  `mutation_receipts`; this one belongs there with `key_scope =
  'gateway-recovery-failure'`. The receipt table is already written on every
  keyed mutation, so one more scope adds nothing measurable; the per-route
  `DELETE` on route recovery becomes a scoped delete on `mutation_receipts`
  by `(key_scope, idempotency_key LIKE route_key || ':%')`, which needs the
  existing PK prefix only.

### 3.7 Journals (2 → 0 or 1)

`eval_commands` (command journal, 9 columns) and
`eval_progress_observations` (counts per observation) are append-only and
read only as "the latest" or "all for one experiment", bounded by the
experiment's lifetime. `eval_commands` could live in `mutation_receipts`
(it is the Eval mutation receipt with a state); `eval_progress_observations`
could be a bounded `jsonb` array on `eval_experiments`, which the UI reads
with the experiment anyway. The second needs a bound (LIM entry) before it
moves onto the parent row, or it becomes the unbounded-column problem the
first round avoided. Defer both until the Eval UI's needs are settled; count
them as optional.

### 3.8 Not candidates, and why

| Table | Looks like | Why it stays |
| --- | --- | --- |
| `eval_projection_queue` | 1:1 to `eval_experiments` | Migration 000018 moved projection locking onto this row so projection writers never hold the experiment row (lock order: Run/Audit → queue → members). Folding it back reintroduces the contention 000018 removed. |
| `eval_frozen_plans` | 1:1 to `eval_experiments` | The plan document (`bytea`) is the FK target that proves members exist only for a frozen plan. It could be a column plus a trigger, but V407-029 is about to stop reading it on hot paths; revisit after. |
| `artifact_scopes` | polymorphic parent | Gives bindings one key `(scope_kind, scope_id)` for Run and Project scopes and carries `revision_count` for the per-Run limit (000008). Replacing it with two FKs changes every artifact key. |
| `artifact_versions` vs `artifact_blobs` | 1:1 | N:1 by design: versions dedupe onto content-addressed blobs. |
| `gateway_run_routes`, `gateway_allocation_routes`, `gateway_recovery_waits` | small | Different keys (Run × route, allocation × model, participant × model); they are the recovery lock graph, not satellites. |
| `audit_events`, `audit_report_pages`, `workflow_run_events` | journals | Paged by sequence or ordinal with their own retention; they are the API's event and report pages. |
| `audit_finding_contributions` | unimplemented | Specifications 19 and 28 define it; removal is a product decision, unchanged from round one. |
| `owner_queue_controls`, `git_ssh_keys` | per-owner rows | There is no owners table; these are the only per-owner state. Merging them into one `owners` row is possible but mixes a Scheduler control with a credential. |
| `mutation_receipts`, `credential_operations`, `eval_suboperations` | receipts and journals | Already the consolidated mechanisms (V370-007); they absorb 3.6 and 3.7 rather than being candidates. |

## 4. Does performance get worse?

Overall no, with the two exclusions above. Per effect:

- **Reads.** Every fold replaces a join or a second lookup with columns of
  the row already read. The Eval member family is read per page and per
  tick; the Stage satellites per transition. Nothing becomes wider on a path
  that scans many rows, because the folded columns are small and the large
  documents stay separate.
- **Writes.** A fold adds columns to a row that is already written once per
  entity; the extra columns are set once, so no new update traffic. The
  updates that do repeat (`eval_submissions.state` and `observed_tokens`) move
  onto `eval_members`, whose indexes do not include those columns, so they
  stay heap-only. The folds that would move a repeated write onto a
  trigger-heavy row (`stage_metrics`, claims onto `audits`) are excluded until
  the triggers are scoped and the rates are measured.
- **Indexes.** Each dropped table removes its primary key index and its
  foreign key probe index (V407-025 and V409-007 already target index count);
  the Eval family alone drops five primary keys and two FK indexes, adding one
  partial unique index. Fewer index entries per insert, fewer pages to keep
  hot.
- **Triggers.** Set-once immutability moves from "the row cannot change" to
  column-scoped `UPDATE OF … WHEN` triggers, the shape 000012 established.
  Per write that is one trigger instead of one trigger plus an FK check.
- **Locks.** Lock scope stays with the entity: a member settlement locks the
  member row, a transition locks the Stage row, exactly as now minus the
  satellite row. The one place a fold would widen a lock scope is the
  projection queue, which is excluded.
- **Migrations.** Each fold moves rows in a migration under the 120 s
  migration budget; the largest table involved is `eval_members` (10 000 per
  experiment). The project is pre-production, so a new baseline after the
  round is the cheaper path (V409-012 decides).

What could get worse, and how it is detected: a fold onto `eval_members`
makes the row wider than a page fraction so HOT updates start failing and
settlements write index entries. Detection: `pg_stat_user_tables.n_tup_hot_upd`
versus `n_tup_upd` on `eval_members` before and after, collected by the
V409-013 harness. Reopen condition: HOT ratio drops below 90 % at 10 000
members.

## 5. Why the tables exist (the pattern to stop)

Round one's folds and this list have one cause: a new phase of an existing
entity was added as a table because a table is the easiest way to express
"this exists now" with a primary key and a foreign key, and because
whole-row immutability was easier than a column-scoped trigger. The project
already has the alternative (set-once columns with `UPDATE OF` triggers,
natural-key receipts in `mutation_receipts`, counters off the parent row).
The guard that would stop a third round: a schema test that fails when a new
table's primary key equals another table's primary key or unique key and the
new table has no cardinality of its own, unless an allowance names the lock
or size reason (as 000018 and `planner_sessions` have).

## 6. Proposed stages

Order follows the dependencies on open tasks. Counts exclude
`contractor_schema_migrations`.

| Stage | Change | Depends on | Tables |
| --- | --- | --- | ---: |
| 0 | V409-013 collects `pg_stat_user_tables` and HOT ratios for every table named here | — | 70 |
| 1 | One `controller_claims` table (3.3 option A); `ControllerClaims` takes a kind | — | 69 |
| 2 | One `operations_settings` row (3.4) | — | 68 |
| 3 | One `execution_reports` table (3.5) | V409-007 | 67 |
| 4 | `gateway_recovery_failures` into `mutation_receipts`; `eval_collections` into the counter home V409-001 chooses (3.6) | V409-001 | 65 |
| 5 | Transition decisions and resumptions onto `stage_executions` (3.2) | V407-025 | 63 |
| 6 | Eval member family onto `eval_members`; projections per V409-009 (3.1) | V409-009, V407-028 | 59–60 |
| 7 | Optional: `eval_commands` and `eval_progress_observations` (3.7) | Eval UI decision | 57–58 |
| 8 | Reconsider `stage_metrics`, claims inline, `eval_frozen_plans` with measured rates | V407-025, V409-006, V407-029 | 54–57 |
| 9 | New baseline migration; the satellite-table schema guard (section 5) | V409-012 | — |

Each stage is one task with one migration and a migration test, as in round
one. The round is complete when the guard in section 5 is in `make lint` and
the Status section of the SQL review names each table's outcome.
