# Database schema review — 2026-10-03

First pass against commit `8304dd7d` (71 migrations). Updated on 2026-10-04
against `origin/main` at `831f21f1` (79 migrations). This review covers all 91
tables created by the forward-only migrations in
`internal/persistence/migrations/`; no migration has ever dropped a table. It is
non-normative analysis: proposals here are not implemented unless the
[changes section](#changes-since-the-first-pass) says so.

## Method

- All migrations were applied to a throwaway PostgreSQL 16 container and the
  catalog was introspected: 91 tables, 121 foreign keys, 852 columns.
- Production code (non-test Go files under `internal/` and `cmd/`) was searched
  for SQL references to each table: `FROM`, `JOIN`, `INTO`, `UPDATE` and
  `CopyFrom` identifiers.
- In the first pass, seven parallel reviews read the writers, readers, triggers
  and owning specifications of each subsystem.
- The defects, the undeletable-row guards and every removal candidate were
  rechecked directly, on both baselines. Other per-table notes come from the
  subsystem reviews; items not rechecked are listed separately as unverified
  risks.

## Changes since the first pass

Eight migrations (000072–000079) landed between the two baselines. None adds or
drops a table; they add two foreign keys, one column and replace one trigger.

| Item | Status on `831f21f1` | Change |
| --- | --- | --- |
| DB-01 — Audit purge blocked by review decisions | **Resolved** | 000077 (`1d28f597`, V278-001) cascades decisions and finding assessments directly from `audits`. The reproduction below now deletes every case |
| DB-02 — eval view generations accumulate | **Resolved** | 000078 (`37cafa82`, V303-001) makes superseded generations deletable; publishing skips unchanged content and deletes every generation except the current one (`internal/evalstore/projections.go`) |
| `audit_events` verdict | **Remove → Questionable** | `f42f4e4d` made finding intake read `finding.proposal_rejected` events to skip rejected receipts; 000074 indexes that lookup |
| `gateway_run_routes` keeps earlier stages' routes | **Resolved** | Admission now deletes the Run's routes the current stage does not use, and the scheduler deletes them when the Run ends |
| Eval If-Match revision consumed by background writes | **Resolved** | 000073 (`776ffb2a`): coordinator observations advance `updated_at` without consuming the revision |
| `eval_collections` contention | **Partly resolved** | 000079 (`ed3f53c9`, V304-001) bumps collections once per insert or delete statement; the per-row update trigger from 000066 still bumps the owner-wide row on every state change |
| Unused eval state `interrupted` | **Resolved** | 000075 removes it from the CHECK constraint |
| Eval claim candidate scan | **Resolved** | 000076 adds partial indexes for actionable experiments and dirty projections |
| `credential_operations` lookup without an index | **Withdrawn** | The first pass was wrong: `HasPreparedDelete` filters on `phase = 'prepared'`, which the partial index `credential_operations_prepared_idx` covers |
| Queue partial index mismatch | **Confirmed** | Rechecked by reading; now DB-04 |

## Summary

| Verdict | Tables | Meaning |
| --- | ---: | --- |
| Keep | 64 | Has production writers and readers and an invariant no other table holds |
| Merge | 15 | Strict 1:1 or 0..1 extension of a parent; columns on the parent would carry the same data |
| Remove | 7 | No production reader, or an exact duplicate of another table |
| Questionable | 5 | Works, but the mechanism looks replaceable |

The table count follows from two things. The database hosts eight subsystems,
and each one re-implements the same reliability patterns in its own tables:
idempotency receipts, controller leases, tombstones, append-only logs and
materialized read models. Immutability triggers that reject every `UPDATE` also
push mutable state into 1:1 side tables.

| Subsystem | Tables | Keep | Merge | Remove | Questionable |
| --- | ---: | ---: | ---: | ---: | ---: |
| Run execution | 14 | 10 | 2 | 2 | 0 |
| Artifacts | 10 | 6 | 1 | 1 | 2 |
| Projects and queue | 3 | 3 | 0 | 0 | 0 |
| Secrets | 8 | 4 | 4 | 0 | 0 |
| Runtimes and LLM gateway | 11 | 9 | 1 | 1 | 0 |
| Audits | 13 | 8 | 4 | 0 | 1 |
| Findings | 8 | 6 | 1 | 1 | 0 |
| Managed evals | 24 | 18 | 2 | 2 | 2 |
| **Total** | **91** | **64** | **15** | **7** | **5** |

Removing the seven tables and merging the fifteen brings the schema to 69 tables
without losing behaviour. Resolving the five questionable ones could take it
to 64.

## Defects

### DB-01 — P2, resolved: an Audit with any human review decision could not be purged

On the first baseline, `audit_review_decisions.(request_id, audit_id)`
referenced `audit_review_requests` with `ON DELETE RESTRICT`, and decisions had
no other path to `audits`. `PurgeClaimed` in `internal/auditstore/lifecycle.go`
issues `DELETE FROM audits`; the cascade reached `audit_review_requests`, and the
`RESTRICT` check failed. Decisions are written by `DecideFinding` and
`DecideActionReview`, so every Audit in which a human triaged a finding or
approved an action was affected, and so was deletion of its Project.

Migration 000077 adds cascading foreign keys from `audit_review_decisions` and
`audit_finding_assessments` to `audits`. Its comment also confirms the
assessment case that the first pass could not reproduce.

Reproduction used a scratch database. To insert minimal rows, CHECK constraints
and user triggers were removed from `projects`, `audits`, `audit_rounds`,
`audit_review_requests`, `audit_review_decisions` and `audit_report_candidates`.
Every foreign key was kept.

| Audit contents | 71 migrations | 79 migrations |
| --- | --- | --- |
| A review request without a decision | `DELETE 1` | `DELETE 1` |
| A pending report candidate without a decision | `DELETE 1` | `DELETE 1` |
| A decided review request | `ERROR: … violates foreign key constraint "audit_review_decisions_request_fkey"` | `DELETE 1` |

### DB-02 — P3, resolved: managed-eval view generations accumulated without limit

Every publish inserted a new `eval_view_generations` row and copied all members,
pairs and charts, while readers resolved only the latest generation. Since
000078, publishing skips content that has not changed and deletes every
generation except the current one. The view tables now hold one generation per
experiment.

### DB-03 — P3, open: rows that can never be deleted

- `finding_proposal_receipts`: the `BEFORE UPDATE OR DELETE` trigger raises
  unconditionally (migration 000040). `run_id`, `project_id`, `owner_id` and
  `audit_id` have no foreign keys. Receipts, including evidence of up to 1 MiB,
  outlive their Run, Project and owner. Once the source Run is deleted without
  an Audit hold, no API can reach them.
- `runtime_credentials`: the same unconditional trigger (migration 000016).
  Deletion only adds a `runtime_credential_tombstones` row, so the ciphertext of
  a deleted secret stays in the database permanently.

### DB-04 — P3, open: the Run queue partial index no longer matches the queue query

`workflow_runs_owner_nonterminal_queue_idx` (000029) is restricted to
`state IN ('initializing', 'running', 'cancelling')`. Migration 000064 added the
`pending` and `waiting` states, and the queue query in
`internal/runstore/queue_store.go` now filters on all five. PostgreSQL can use a
partial index only when the query's condition implies the index predicate, so
this index cannot serve that query. Either widen the predicate or drop the
index.

## Unverified risks

These come from the subsystem reviews and were not rechecked beyond the code
still being present.

- `stage_metrics` has a 30-day TTL, while managed evals still read it live
  (`internal/evalstore/observed_usage.go`, `metric_snapshots.go`). A
  recomputation after expiry could report zero observed tokens.
- The Audit execution-intent SQL still takes `FOR UPDATE` on the singleton
  `scheduler_settings` row, which serializes intent creation across all Audits.
- `allocation_execution_reports` expires after 30 days while `stage_allocations`
  is kept, so older allocations show a missing report.

## Cross-cutting patterns

- **Idempotency** is implemented nine ways: inline keys on `workflow_runs` and
  `projects`, `credential_operations` (a two-phase saga),
  `runtime_credential_creations`, `runtime_config_publications`,
  `runtime_management_operations`, `audit_idempotency`, review-request key
  columns, `eval_mutation_receipts` and `eval_suboperations` (outbound intents).
  They differ in key storage (raw key or digest), scope and retention. The plain
  receipts could collapse into one table; the saga and outbound intents are
  different mechanisms and should stay separate.
- **Controller leases** come in three styles: `audit_controller_claims` and
  `eval_controller_claims` (identical shape and nearly identical SQL), and inline
  claim columns on `workflow_runs` and `projects`. A shared Go helper gives most
  of the benefit; a single polymorphic table would lose cascading foreign keys.
- **Whole-row immutability triggers** create 1:1 side tables for the few mutable
  fields: `finding_proposal_retention`, `runtime_credential_tombstones` and the
  pattern behind several other merge candidates. Triggers scoped to the immutable
  columns would let those fields live on the parent row.
- **Cross-subsystem triggers**: about ten eval-invalidation triggers sit on
  `workflow_runs`, `stage_executions`, `stage_metrics`, `audits`,
  `audit_executions`, `audit_artifact_links`, `artifact_bindings` and
  `artifact_binding_revisions`. Every write to those tables pays for them, even
  when no experiment exists.
- **Event logs as state**: `audit_events` now also stores the rejection state of
  finding proposals. A log whose general history nobody reads has become the
  source for one decision.
- **Retention** is uneven. Runs, artifacts, Audits and evals have no TTL and go
  away only through explicit deletion. Telemetry expires after 30 days, and
  performance minutes after 7. Receipts, journals and user-scope artifact
  history grow without limit.

## Per-table reference

Verdicts: **Keep**, **Merge** (into the named parent), **Remove**, **Questionable**.

### Run execution (14)

| Table | Purpose | Writers → readers | Verdict and notes |
| --- | --- | --- | --- |
| `workflow_runs` | Aggregate root of a Run: snapshot, parameters, state machine, scheduler lease, event sequence | `CreateRun`, Audit-managed runs, scheduler transitions → run APIs, queue, scheduler, Audits, evals, credentials, gateway recovery | **Keep.** Later migrations added 13 column groups. Kept until explicit deletion. See DB-04 |
| `stage_executions` | One stage attempt: immutable spec and context, lineage, per-attempt state | Scheduler, resume → run detail, scheduler recovery, evals, findings, Audit scan history | **Keep.** `accepted_stage_result` is always a byte copy of `candidate_stage_result` |
| `stage_allocations` | Runtime-agent grant per stage and logical agent; release tracking | `RecordStageAllocation` → reports, delete and resume gates, allocation history, credential-in-use checks | **Keep** |
| `planner_sessions` | Durable Planner state (up to 2 MiB) so a crash can resume | `StartPlanner`, `AppendPlannerEvent` → session recovery, run-detail plan, Audit scan history | **Keep.** Separating a large, frequently updated JSONB from `stage_executions` is justified |
| `planner_events` | Intended as the Planner event journal | Written with every run event → `ListPlannerEvents` has no production caller | **Remove.** Write-only; `workflow_run_events` already holds the facts and is the replay authority |
| `stage_transition_decisions` | Records once what followed an attempt: next, retry, escalate, succeed, fail | Scheduler, in the same transaction as the next step → run detail | **Keep.** Prevents double progression. 1:0..1 with the source stage, so it could become columns |
| `run_stage_resumptions` | Receipt for a manual resume of a failed Run | `ResumeFailedRun` → its own replay; the artifact-binding trigger uses it to allow thawing outputs | **Keep** while that trigger needs it. `requested_by` and `requested_at` are never read |
| `workflow_run_events` | Gap-free per-run event log for the WebSocket stream | Triggers, Planner → stream replay, `pg_notify` listener | **Keep** |
| `workflow_run_metadata_labels` | User labels on a Run (at most 32) for filtering | Inserted with the Run → list filter, queue, run detail | **Merge** into `workflow_runs` as `jsonb` with a GIN index; labels are immutable |
| `stage_execution_reports` | Original per-allocation report | None since `a1db02a5` (2026-08-29) | **Remove.** Dead; referenced only by tests |
| `allocation_execution_reports` | Final worker and Runtime telemetry per allocation | Scheduler at stage end → metric rebuild, allocation history | **Keep.** 30-day TTL |
| `planner_execution_reports` | Planner report per stage | Scheduler → only the `stage_metrics` rebuild | **Merge** into `stage_metrics`; it is a staging input, never exposed |
| `stage_metrics` | Materialized per-stage metrics | Rebuilt from both report tables → run detail, evals | **Keep.** 30-day TTL; see unverified risks |
| `performance_minutes` | Per-minute server health: process, database pool, HTTP, GPU | Diagnostics worker → performance history API | **Keep.** 7-day TTL; cleanup runs only while metrics are enabled |

### Artifacts (10)

| Table | Purpose | Writers → readers | Verdict and notes |
| --- | --- | --- | --- |
| `artifact_scopes` | Ownership registry: user, project, run | Inserted before each write; read only as an existence check and lock anchor | **Questionable.** Holds no data; removing it requires redesigning the fork/purge lock protocol |
| `artifact_blobs` | Content-addressed, deduplicated payloads in the database or on the filesystem | Writes → every read path, garbage collection | **Keep** |
| `artifact_versions` | One write: blob plus media type, shared across forks and publications | Writes → metadata joins, output checks, garbage collection | **Keep.** Keeps git provenance from attaching to unrelated writes of the same bytes |
| `artifact_bindings` | Mutable name → current revision pointer with a freeze flag | All write, fork, publish and import paths → all artifact APIs | **Keep.** The store's only mutable row (CAS, freeze) |
| `artifact_binding_revisions` | Immutable history of each binding | One row per operation → versioned reads, history, evals | **Keep.** History is never compacted |
| `artifact_lineage` | Provenance edges between exact revisions | Same statement as fork, bind, publish, import → lineage API and UI, two correctness checks | **Keep** |
| `artifact_pins` | Intended to protect used revisions from collection | `PinExact` and fork/bind SQL → only its own duplicate check; purge deletes pins before revisions | **Remove** together with the `PinExact` plumbing |
| `artifact_blob_settings` | Single row: the blob backend this installation committed to | Startup claim → a trigger on every blob insert | **Questionable.** Configuration plus a startup check gives the same protection and removes a trigger from the hottest insert. Do not merge with `scheduler_settings` |
| `artifact_git_sources` | Git provenance: URL, ref, commit | Git import → a LEFT JOIN in every metadata query | **Merge** into `artifact_versions` (0..1:1) |
| `workflow_run_output_publications` | Receipt of automatic output publication to the Project | Scheduler at run end → run detail, resume guard | **Keep.** A `published` row duplicates the lineage edge |

### Projects and queue (3)

| Table | Purpose | Writers → readers | Verdict and notes |
| --- | --- | --- | --- |
| `projects` | Owner-scoped container with an in-row deletion state machine | Project API, deletion controller → nearly every subsystem | **Keep.** Two unique constraints on the same columns in different order; one is redundant |
| `owner_queue_controls` | Per-owner queue pause | Queue control API; locked on every stage admission | **Keep** |
| `scheduler_settings` | Runtime-mutable limit on concurrent Runs | Settings API → scheduler, Audit execution-intent SQL | **Keep** |

### Secrets (8)

| Table | Purpose | Writers → readers | Verdict and notes |
| --- | --- | --- | --- |
| `llm_credential_identities` | Reserves an LLM credential ID forever; the LiteLLM key alias derives from it | Credential create → no Go reader; foreign keys and a trigger lock only | **Merge** into `credential_operations` via a partial unique index on create operations |
| `llm_credentials` | Encrypted LLM gateway token | After LiteLLM mints the key → credential resolution, API | **Keep.** The UPDATE ban prevents re-encrypting rows during master-key rotation |
| `llm_credential_tombstones` | Who deleted a key and when | Credential delete → only delete replay | **Merge:** the completed delete row in `credential_operations` already holds the same data |
| `credential_operations` | Two-phase saga around remote LiteLLM calls | Create and delete; prepared rows replay at startup → every credential lookup | **Keep.** Stores the raw idempotency key instead of a digest |
| `runtime_credentials` | Encrypted Runtime secrets: OTLP, proxy, Caido, HTTP targets | Create API → decryption at allocation | **Keep.** Soft delete keeps the ciphertext (DB-03) |
| `runtime_credential_tombstones` | "Deleted" flag | Delete → an anti-join in every active read | **Merge** into `runtime_credentials` as `deleted_at` and `deleted_by` |
| `runtime_credential_creations` | Create receipt, with an HMAC instead of a request hash | Written with the secret → replay | **Merge** into `runtime_credentials` (strictly 1:1) |
| `git_ssh_keys` | Owner SSH key for `ssh://` imports | Settings API → git import | **Keep** |

### Runtimes and LLM gateway (11)

| Table | Purpose | Writers → readers | Verdict and notes |
| --- | --- | --- | --- |
| `runtime_config_versions` | Immutable, content-addressed RuntimeConfig documents | Publish API → Run pinning, placement, credential checks | **Keep.** No garbage collection |
| `runtime_config_publications` | Publish idempotency receipt | Written with the version → replay | **Merge** into `runtime_config_versions` (1:1; actor and time are copies) |
| `runtime_label_bindings` | Label → exact config ref, with CAS | Label API → Run pinning, placement | **Keep** |
| `runtime_agent_principals` | Registered agents (by mTLS key) and their labels | Registration, label API → placement | **Keep.** Each certificate rotation adds a row; stale rows are not cleaned |
| `runtime_management_operations` | Generic receipt that stores the result | Label and principal mutations | **Keep.** Natural home for the other plain receipts |
| `configuration_publications` | Audit trail of model-policy and LLM-gateway publications | `configaudit` → no reader | **Remove,** or make it the durable idempotency store for `config.Manager`, which keeps idempotency only in memory |
| `gateway_recovery_routes` | Circuit breaker per model route | Admission, failure reports, retry → run status | **Keep.** Never collected |
| `gateway_run_routes` | Routes the Run's current stage uses | Admission replaces them; removed when the Run ends → status, retry | **Keep** |
| `gateway_allocation_routes` | Trusted (allocation, model) → route binding | Scheduler → Runtime report handling | **Keep.** `run_id` is derivable |
| `gateway_recovery_waits` | Participants waiting for a blocked route | Failure, acquire → admission gate | **Keep.** Cleans itself up |
| `gateway_recovery_failures` | Deduplicates repeated failure reports | Insert-only; the dedup is in `ON CONFLICT` and the row count | **Keep.** Retained until the Run is deleted, though needed only for minutes |

### Audits (13)

| Table | Purpose | Writers → readers | Verdict and notes |
| --- | --- | --- | --- |
| `audits` | Aggregate root: profile, limits, budgets, revision | All Audit mutations → Audit APIs, controller, evals | **Keep.** `next_event_sequence` exists only for `audit_events` |
| `audit_rounds` | Round worklist and state | Controller → APIs, report | **Keep.** State `proposed` is allowed but never written |
| `audit_items` | Check task: dispatch worklist and final disposition | Rounds, dispatch, collection, reviews | **Keep** |
| `audit_executions` | Intent for exactly one Run; survives Run deletion | Controller with budget reservation → controller, run-deletion gates, evals | **Keep** |
| `audit_execution_items` | Attempt history per item within a batch Run | Intent, collection | **Keep.** The result reference is stored in four places |
| `audit_collection_receipts` | A Run's output was collected exactly once | Collection → replay, report, run-deletion gates | **Merge** into `audit_executions` (1:1) |
| `audit_artifact_links` | Logical key → retained artifact | Start, collection, report → report API, controller, evals | **Keep** |
| `audit_coverage_rows` | Per-item coverage for the report and UI | Created with items; overwritten by collection and review | **Merge** into `audit_items` (1:1) |
| `audit_events` | Event log, originally for a deferred WebSocket feed | 28 Go statements and 2 migration-defined writes → finding intake reads `finding.proposal_rejected` events; `ListEvents` has no production caller | **Questionable.** Only one event kind is read. Store rejection state on the receipt or hold, then the log and `next_event_sequence` can go |
| `audit_idempotency` | Replay fence for create, start, transition, delete | Each mutation → replay, eval tombstone trigger | **Keep.** `response_snapshot` and `resource_id` are never read |
| `audit_controller_claims` | Epoch-fenced controller lease | Claim, renew, release; checked in every controller write | **Keep** |
| `audit_proposal_items` | Consume-once link from a proposed check to its item | Next-round acceptance | **Merge** into `audit_items` (1:1) |
| `audit_report_candidates` | Report frozen while human acceptance is pending | `ProposeReport` → report API, controller | **Merge** into `audit_artifact_links` plus the review request |

### Findings (8)

| Table | Purpose | Writers → readers | Verdict and notes |
| --- | --- | --- | --- |
| `finding_proposal_receipts` | Unforgeable receipt of a Runtime finding proposal | Finding intake → run and Audit inboxes, imports | **Keep.** Undeletable (DB-03) |
| `finding_proposal_retention` | Mutable retention state of a receipt | 1:1 with the receipt; updated by a trigger and by run deletion with duplicated logic | **Merge** into the receipt; it exists only to bypass the immutability trigger |
| `finding_proposal_audit_holds` | An Audit's retained copy of a receipt | Import into an Audit; a trigger creates the finding | **Keep** |
| `audit_findings` | Per-Audit finding with triage state | Created by the hold trigger; reviews, collection | **Keep** |
| `audit_finding_contributions` | Intended to merge several proposals into one finding | Only the hold trigger writes it, always `relation='first'` | **Remove** (or build the merge feature): today an exact copy of `audit_findings` |
| `audit_finding_assessments` | History of machine assessments | Collection, direct verification | **Keep.** Cascades from `audits` since 000077 |
| `audit_review_requests` | Requests for human authority: triage, action approval, report acceptance | Reviews, rounds, report → APIs, dispatch gate | **Keep.** Kinds `plan-approval` and `provide-evidence` are unused |
| `audit_review_decisions` | Immutable human decision | Decision API → findings, dispatch gate, resume | **Keep.** Cascades from `audits` since 000077 (DB-01) |

### Managed evals (24)

| Table | Purpose | Writers → readers | Verdict and notes |
| --- | --- | --- | --- |
| `eval_dataset_revisions` | Immutable dataset revisions pinned by experiments | Dataset import → plan preparation, lists | **Keep.** Old revisions cannot be deleted |
| `eval_experiments` | Experiment root and authority | Nearly all eval code | **Keep** |
| `eval_frozen_plans` | Plan frozen at start | Written once → every pin check | **Keep** |
| `eval_plan_resources` | Full resource closure behind the plan | Written with the plan → only `bindings/*` and `private/*` are read | **Questionable:** four of six path families are write-only |
| `eval_members` | Matrix cell: suite × case × sample × variant | Written with the plan → admission, execution, projections | **Keep** |
| `eval_submissions` | At most one execution per cell | Admission, binding, settlement | **Keep** |
| `eval_suboperations` | Write-ahead intents for effects in other services | Coordinator; keys become downstream Idempotency-Keys | **Keep** |
| `eval_execution_tombstones` | Remembers a Run or Audit deleted before the cell observed it | Trigger on Run and Audit deletion → reconcile, settlement | **Keep:** without it the cell hangs or the Run is created twice |
| `eval_records` | Content-addressed results and assessments | API, coordinator | **Keep** |
| `eval_selections` | Which record counts for a cell | Selection API, automatic selection | **Keep** |
| `eval_selection_history` | History of selections | Written on each selection → no reader | **Remove** |
| `eval_commands` | Status of an operator command | Command API, coordinator | **Keep** |
| `eval_controller_claims` | Coordinator lease | Coordinator | **Keep.** `RenewClaim` has no production caller |
| `eval_mutation_receipts` | Inbound idempotency; survives experiment purge | Every mutation | **Keep.** Up to about 10,000 rows per experiment until Project purge |
| `eval_project_dependencies` | Workspace Projects created by an experiment | Same transaction as `eval_submissions.execution_project_id` | **Merge** into that column (exact copy) |
| `eval_progress_observations` | Progress-chart time series | One row per publish | **Keep:** with superseded generations now deleted, this is the only history |
| `eval_collections` | Revision counter that fences list cursors | Statement triggers on insert and delete, row trigger on update | **Questionable:** both lists already page on immutable keys |
| `eval_projection_queue` | Per-experiment dirty counter and current snapshot | About ten invalidation triggers | **Keep** |
| `eval_member_projections` | Per-member view cache and work queue | Coordinator, up to 100 dirty members per tick | **Keep;** it also drives result collection. `observed_document` (up to 1 MiB) is never read |
| `eval_view_generations` | Header of the current published snapshot | Publish, which now deletes superseded rows | **Merge** into `eval_projection_queue`: since 000078 it holds one row per experiment |
| `eval_view_members` | Per-generation copy of each cell's view | Copied on publish | **Remove:** `eval_view_pairs` already embeds both cells of each pair |
| `eval_view_pairs` | A/B pairs with filter and sort columns | Copied on publish → pair and member pages | **Keep** |
| `eval_view_charts` | Precomputed chart aggregates | Copied on publish → chart pages | **Keep;** derivable from pairs |
| `eval_evidence_refs` | Reverse index artifact → cells, for invalidation | Written with results → a trigger on artifact deletion | **Keep** while views are materialized |

## Proposed cleanup

1. Remove the seven tables and their writer code: `stage_execution_reports`,
   `planner_events`, `artifact_pins`, `configuration_publications`,
   `audit_finding_contributions`, `eval_selection_history` and
   `eval_view_members`.
2. Fold the fifteen 1:1 extensions into their parents.
3. Decide on the five questionable tables: `artifact_scopes`,
   `artifact_blob_settings`, `audit_events` (move rejection state first),
   `eval_collections` and `eval_plan_resources`.
4. Fix DB-03 (deletion paths for receipts and Runtime secret ciphertext) and
   DB-04 (queue index predicate).
5. The project does not require backward compatibility, so once the schema
   settles, the migrations can be squashed into one baseline.
