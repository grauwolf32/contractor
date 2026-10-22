# Database schema review — 2026-10-03

[Research index](README.md)

- **Date:** 2026-10-03 at `8304dd7d` (71 migrations); updated 2026-10-04 at
  `831f21f1` (79 migrations); status rechecked 2026-10-23 at `af9fd522`
  (baseline plus migrations 000002–000025).
- **Scope:** all 91 tables of that schema: purpose, writers and readers,
  retention, and a verdict per table.
- **Kind:** non-normative analysis. The executed cleanup is the
  [schema reduction plan](../plans/2026-10-04-schema-reduction.md) (V365).
- **Current picture:** the [database ERD](../spec/database-erd.html)
  (`make docs-erd`) shows every current table, the tables each one absorbed and
  the remaining candidates. It replaces the per-table reference of the original
  report, which is in git history.

## Status on 2026-10-23

The schema has 70 tables, down from 91. V365 removed 6 and folded 14, which
gave the 71 tables of `000001_baseline.sql` (V367-001). Later changes:
`audit_report_pages` added (000004, V368-001); `mutation_receipts` replaced
`audit_idempotency`, `eval_mutation_receipts` and
`runtime_management_operations` (000013, V370-007); `artifact_blob_settings`
dropped (000015); `model_selection_settings` added (000020).

| Verdict in the review | Tables | Outcome |
| --- | ---: | --- |
| Keep | 64 | Kept, except the three receipt tables folded into `mutation_receipts` |
| Merge | 15 | 14 folded (V365-011–015, V365-017–022); `planner_execution_reports` deferred (V365-016): a Stage can have several immutable reports, while `stage_metrics` is a derived row that expires |
| Remove | 7 | 6 dropped (V365-005–010); `audit_finding_contributions` kept: the proposal and findings specifications describe contributing proposals, so removing it is a product decision |
| Questionable | 5 | `artifact_blob_settings` dropped; `artifact_scopes`, `audit_events`, `eval_collections` and `eval_plan_resources` remain |

### Still open

| Item | State |
| --- | --- |
| `planner_execution_reports` → `stage_metrics` | Deferred, V365-016 |
| `audit_finding_contributions` | Only `relation = 'first'` is written; keep it for multi-proposal findings or remove it (product decision) |
| `artifact_scopes` | Lock anchor for fork and purge, and the Run's revision counter (LIM-ART-6). Removing it means redesigning those locks |
| `audit_events` | Only `finding.proposal_rejected` is read. Storing rejection on the receipt or hold would let the log go. V370-003 gave it one allocator (`contractor_append_audit_events`) |
| `eval_collections` | Both lists already page on immutable keys; the owner-wide counter is still bumped on every experiment state change |
| `eval_plan_resources` | Four of six path families are write-only; only `bindings/*` and `private/*` are read |
| Risk: `stage_metrics` TTL | Not rechecked. Eval metric snapshots and observed usage still read the 30-day `stage_metrics` live |
| Risk: `allocation_execution_reports` TTL | Not rechecked. Reports expire after 30 days while `stage_allocations` stays |

Apart from V365-016, no task tracks these items.

Accepted on 2026-10-23 by the user: a deleted Runtime credential keeps its
ciphertext (DB-03). Deletion sets `deleted_at` and `deleted_by`; the get and
list reads filter on `deleted_at IS NULL`, and the row stays encrypted
under the master key named by its `key_id` (SEC-R19). Reopen it if master-key
rotation or a requirement to erase deleted secrets is planned: deleted rows
would then need re-encryption or removal too.

## Method

- All migrations were applied to a throwaway PostgreSQL 16 container and the
  catalog was introspected: 91 tables, 121 foreign keys, 852 columns.
- Production Go code under `internal/` and `cmd/` was searched for SQL
  references to each table (`FROM`, `JOIN`, `INTO`, `UPDATE`, `CopyFrom`).
- Seven parallel reviews read the writers, readers, triggers and owning
  specifications of each subsystem.
- Defects, the undeletable-row guards and every removal candidate were
  rechecked directly on both baselines.

## Defects

| ID | Severity | Finding | State |
| --- | --- | --- | --- |
| DB-01 | P2 | An Audit with any human review decision could not be purged: `audit_review_decisions` referenced `audit_review_requests` with `ON DELETE RESTRICT`, so `PurgeClaimed` and Project deletion failed. Reproduced on a scratch database (`DELETE` failed on `audit_review_decisions_request_fkey`) | Resolved by V278-001 (cascades from `audits`) |
| DB-02 | P3 | Every eval view publish inserted a new `eval_view_generations` row and copied all members, pairs and charts; readers used only the latest | Resolved by V303-001; V365-022 then folded the header into `eval_projection_queue` |
| DB-03 | P3 | Undeletable rows: `finding_proposal_receipts` outlived their Run, Project and owner; deleted Runtime credentials kept their ciphertext | Receipts: purged with their Project since V368-009 (000010); a receipt of a Run outside any Project stays by design (DEL-R29). Credential ciphertext: accepted |
| DB-04 | P3 | The Run queue partial index covered three active states while the queue query filters on five, so PostgreSQL could not use it | Resolved by V367-003 (000002) |

Findings withdrawn or resolved during the 2026-10-04 update:
`gateway_run_routes` now drops routes of earlier stages; coordinator
observations no longer consume the Eval If-Match revision; the unused eval
state `interrupted` was removed; Eval claim scans got partial indexes; the
`credential_operations` lookup was already covered by a partial index. The
Audit execution-intent SQL no longer locks `scheduler_settings`.

## Cross-cutting patterns

The review explained the table count by eight subsystems each re-implementing
the same reliability patterns in their own tables.

| Pattern | Then | Now |
| --- | --- | --- |
| Idempotency receipts | Nine implementations with different key storage, scope and retention | V370-007 folded the plain receipts into `mutation_receipts` through `KeyedMutation` (PER-R24). The other mechanisms are listed in AGENTS.md §3.2 |
| Controller leases | Two identical claim tables plus inline claims on `workflow_runs` and `projects` | One claim shape and `persistencepostgres.ControllerClaims` (PER-R25, 000014); inline Run and Project claims stay |
| Whole-row immutability triggers | Forced 1:1 side tables for a few mutable fields | V365 replaced them with column-scoped triggers on the parent |
| Cross-subsystem triggers | About ten eval-invalidation triggers on Run, Stage, metric, Audit and artifact tables | `workflow_runs` row triggers fire only when their columns change (V370-028, 000012); the eval triggers remain |
| Event log as state | `audit_events` is the source of proposal rejection | Unchanged (see `audit_events` above) |
| Retention | Runs, artifacts, Audits and evals have no TTL; telemetry 30 days, performance minutes 7; receipts, journals and user-scope artifact history grow without limit | Unchanged, except that finding receipts now go with their Project |
