# SQL and schema review — 2026-10-23

**Revision reviewed:** `main` `725b863e`. File:line references are as of that revision.

**Scope:** every statement in `internal/**/*_sql.go` and inline SQL in `internal/**/*.go`, the migrations (`000001_baseline.sql` plus 000002–000031), triggers, functions, indexes, constraints, the pgx pool configuration and query budgets.

**Method:** three static reviewers (schema and migrations; Audit, Eval and Findings SQL; execution and artifact SQL), the top findings re-verified by hand. No PostgreSQL was available, so no EXPLAIN or `pg_stat_user_indexes`; dead-index claims rest on grepping every non-test Go file for the index's distinguishing predicate. Findings of the 2026-10-23 Server review already owned by V407 tasks are not repeated.

**Outcome:** the SQL is careful where it matters (lock order, claims, receipts, keyset pages, batched purge, connection budgets). The defects are concentrated in triggers and indexes that accumulated without readers, a few hot statements without a supporting index, one lock-mode conflict on the Project row, and reads bounded by rows but not bytes.

## Status (2026-10-24): verified on a database, review closed by V409-013

**Verified revision:** branch `claude/rv-x1`, base `efd5d6ab` plus the `--no-ff` merge `5be7e24c` of `claude/review-v407-v409`, which carries every V409 task and the pending V414 follow-ups. PostgreSQL 17.11, disposable database `contractor_test_x1`. Section H has the new evidence. No demo or production database was used, so their statistics are **not collected**.

Every original claim in A–D and every open question in F has a disposition. When a claim splits into parts, each part has its own:

- **Fixed**: a confirmed defect, fixed by a completed task (its `implementation_commit`).
- **Optimized**: a change that keeps behaviour and has measured evidence.
- **Refuted** or **unsupported**: the claim does not hold on the named revision, with the evidence.
- **Retained**: an accepted risk, listed in the risk table below with its owner, enforcement or detection, and reopen condition.
- **Follow-up (pending)**: a confirmed cost owned by a pending V414 task. It is **not a fix**. Exit task V414-027 tracks them.

Implementation commits: V409-001 `3b654ad7`, V409-002 `1d1edc23`, V409-003 `793c9d53`, V409-004 `02e84782` (migration 000036), V409-005 `fb838e7e`, V409-006 `ed08c26d` (000037), V409-007 `20ef8ffc` (000039; this is the record's renumber commit, and the implementation is `9c6f6e64`), V409-008 `e9f08301`, V409-009 `f58ec279`, V409-010 `89c8038f`, V409-011 `c87e01c1`, V409-012 `052acac4`, V409-014 `ddeb2f1d` (000038; `7a2cacc5` tightened its page budgets to exact reads after the close, test only). Each task record holds its fixtures, plan evidence and can-fail checks. This section only cites them.

### A. Top findings

| # | Disposition | Evidence |
| --- | --- | --- |
| A1 | Fixed in part (V409-001) | Confirmed: `ImportIntoAudit` returned 40001 with no local retry while another transaction held the Project `FOR SHARE`. Intake import, `retainAuditCollectionPrefix` and `DecideFinding` now use bounded `InTxWithRetry` (FAIL-REV-7, FAIL-AUX-3). Refuted: the model-facing failure. Worker `Service.Submit` writes a proposal receipt, never takes the fence, and succeeded under the same lock. The Project fence stays. Follow-up (pending): V414-012 classifies a closed connection or lost COMMIT as unavailable. |
| A2 | Optimized (V409-002) | The inbox now has a native branch and a held branch with an `OFFSET 0` fence. It reads 175–180 receipts at 20,110 and 23,110 receipts; the original read every receipt (seq scan) at the early cursor. No `owner_id` index was needed: the Audit partial index and the receipt and hold primary keys serve it. The held branch is O(holds) (FND-R186): follow-up (pending) V414-019. The quota count is C11. |
| A3 | Fixed (V409-003) | `StartPlanner` now locks the Run, then the Stage (RUN-R29). On the original statement the regression's NOWAIT Stage probe got 55P03. |
| A4 | Optimized (V409-004) | Per-Round `LATERAL` keyset. Fixture: 32 Rounds × 3,125 items. Unfiltered pages read 235/236/236/25/1 items; the original read 100,002–100,004 for every page in generic plans. The subject filter is retained (risk table). |
| A5 | Optimized (V409-004) | The workspace counts findings in one pass: 4,002 findings read, against 8,004 before. Exact counts are kept; an exact count reads each counted row once through the Audit's index. Review counts are kept without a new index (risk table). |
| A6 | Fixed (V409-006, 000037) | The identity and performance-policy guards are `UPDATE OF` their columns. On UPDATE, the completion check only rejects a change of `completion_contract` and no longer joins Stage and Run. Pinned by `TestRowTriggerInventory` and `RequireColumnScopedGuards`. |
| A7 | Fixed (V409-006) | The guard fires only for an UPDATE of the phase or a baseline column, and compares them before probing `audit_executions`. |
| A8 | Fixed in part (V409-007, 000039) | `workflow_run_events_time_idx` is removed. Refuted for `UNIQUE(event_id)`: it guards an invariant distinct from the `(run_id, sequence_number)` primary key, so having no reader does not make it dead. |
| A9 | Fixed (V409-007) | Both removed. No statement applies an operator to `runtime_labels` or reads through the `publication_mode` index. |
| A10 | Fixed (V409-009) | Bounded purge passes (LIM-EVM-20, EVM-R243), accounted against `pg_stat_get_xact_tuples_*` on every pass. Section G has the details. Follow-up (pending): V414-022 (purge guard lexer and rewrite rules). |

### B. Schema and migrations

| Claim | Disposition | Evidence |
| --- | --- | --- |
| 10 dead-index candidates | 5 removed, 5 kept (V409-007) | Removed: `workflow_run_events_time_idx`, `workflow_runs_runtime_labels_idx`, `workflow_runs_publication_mode_idx`, `stage_executions_recovery_idx`, `eval_experiments_owner_page_idx`. Kept and refuted as dead: `UNIQUE(event_id)` (A8), and the `eval_experiments` state, dataset, control and project page indexes. The list being ordered by `created_at` does not make filter indexes dead. Without them, measured custom reads rose: paused cursor page 30 → 1,589, dataset 75 → 1,724, external control 31 → 9,001. The project index is also the FK probe index. |
| 5 redundant indexes | 2 removed, 3 kept (V409-007) | Removed: `audits_project_owner_identity` and `projects_project_owner_key`, with equivalent uniqueness and FKs kept. Kept: `eval_records_native_recent` (1 record read against 201 plus a Sort); the `audit_findings` list and `"C"` pair, because there are two order contracts and the `finding_id` CHECK admits IDs whose order differs between the collations; and `audit_executions_audit_identity`, which the V409-004 inventory keyset probes (449–486 reads without it). |
| `audit_findings`: 18 indexes, row triggers | Fact confirmed; "exists only for the FK probe" refuted; write cost retained | The catalog on this revision has 18 indexes, 2 row triggers (`initialize_correlation` BEFORE INSERT, `protect_correlation` BEFORE UPDATE OF its columns) and 3 statement triggers (section H). Every index has a reader, an FK target or probe role, or a uniqueness role. Dropping `correlation_scope_idx` failed `TestPostgresAuditReadsAtFindingScale`: a seq scan read 12,001 findings against a budget of 4,009. The per-admission write cost was not measured (risk table). |
| Triggers without `UPDATE OF`/`WHEN` | Fixed, with two retained by reason (V409-006) | Scoped: `stage_allocations` ×3, the `audits` guard, the `artifact_bindings` identity guard and Eval projection (split per event), `audit_executions` ×3, and the Eval retention trigger, which now fires only for Runs with a Project, request key and digest. Retained (risk table): Project binding admission (DEL-R40). Refuted: making it INSERT-only would allow a revision write after `BeginDeletion`. The Eval Audit child projection is also retained, because every execution write changes what Eval reads. |
| State CHECK sets without a Go↔schema test | Fixed (V409-008) | Parity tests now cover the 10 tables and their state-dependent partial-index predicates. No drift was found. |
| Migrations | Function redefinition refuted; tests added and runner documented (V409-012) | `CREATE OR REPLACE` in older files is a series of schema versions, and the database has one installed definition. Upgrade tests now cover the 000008 and 000014 backfills. The single transaction, the immutable ledger, budgets and the unavailability of `CREATE INDEX CONCURRENTLY` are in PER-R29–R33, LIM-PER-9..13 and FAIL-PER-5, enforced by `TestEmbeddedMigrationsStayInsideTheMigrationTransaction`. Follow-up (pending): V414-021, a migration file's client deadline. |
| Keyset vs index | Follow-up for allocation history; the other two resolved (V409-007) | Allocation history is confirmed by measurement: 9,306 Stage reads at a cursor 1,500 rows deep, and 17,944 in the generic plan with the Run filter. Follow-up (pending): V414-025. Telemetry cleanup is kept: an aligned index saves at most one tie group per batch. The Project deletion claim index was replaced to match its order. |

### C. Audit, Eval and Findings SQL

| Claim | Disposition | Evidence |
| --- | --- | --- |
| `recordReviewDecisionSQL` | Fixed (V409-005) | The decision insert is gated on a pending request, with `ErrPrecondition` otherwise. On the original statement the regression failed in all four refusal cases. |
| PublishView reads every member document | Retained by decision (V409-009, V409-014) | It still reads every member's projection document (`evalstore/projections.go:198-199`). Bounded by the 1 MiB CHECK per projection document and the member limits. A redesign must keep EVM-R164–R168 (risk table). |
| Comparison pages run the CTE twice | Fixed (V409-014, 000038) | One statement per page returns the exact count and the page (EVM-R245). Exact read budgets (`7a2cacc5`). Follow-up (pending): V414-017, the jsonb NUL escape on the remaining casts. |
| Two clocks for one review rule | Refuted | REV-R43 requires the PostgreSQL clock under the request lock, not one SQL function (V409-005 validation). The read path uses `statement_timestamp()` for both the page basis and the unexpired filter (`finding_reads_sql.go:60,66,140`). |
| `listItems` dead branch; `ListExecutions` | Fixed (V409-004) | The zero-limit branch moved to `export_test.go`; `ListExecutions` moved to `auditstoretest`. |
| `state <> 'collected'` | Fixed (V409-004) | One `uncollectedExecutionStatesSQL`, tied to the Go state set and to the reconcile index: 20,004 executions read before, 0 after. |
| Review-request index | Fixed (V409-004, 000036) | `audit_review_requests_item_subject_idx`: the renewal search read 58,290 requests before and 24 after. The index also serves the `waiting_review` probe of `claimAuditSQL`. |
| `nextMembersSQL` anti-join | Retained by decision (V409-009) | Native admission only: at most 1,000 members and 10 admissions per tick (risk table). |
| `recovery()` per member, "100 per tick" | Fixed (V409-014); "100" refuted (V409-009) | Composed recovery reads authority once per transaction (EVM-R246). Reconciliation selects at most 10 intents plus 10 accepted submissions, not 100. |
| Keyset on `'1' \|\| execution_id` | Fixed (V409-004) | An execution-ID range probe reads 39–52 executions per page; the original read 152, 2,552 and 5,000. |
| `count(*)` per SubmitProposal | Retained (V409-002) | The count runs under the Run's quota advisory lock. LIM-FND-46 and FND-R185 bound the matched receipts at 4,096 (risk table). |
| `audit_artifact_links` collisions | Unsupported | Logical keys carry their source identity: `result/<itemID>`, `evidence/<executionID>/…`, `prepare:<executionID>:…`, `finding/<receiptID>/…`, `standard/…`, `report/machine`. V409-005 found no reachable cross-source collision, so a collision test would need input that no writer produces. |
| `GREATEST(...)` repeated | Declined (V409-012) | Refactoring, not a defect. Reconsider only with named production callers and tests that keep the distinct observe and advance semantics. |
| Inline 10+-line SQL | Refuted (V409-012) | The cited expressions are separate statements shorter than ten lines. The SQL placement test passes. |

### D. Execution and artifact SQL

| Claim | Disposition | Evidence |
| --- | --- | --- |
| ListRunEvents bounded by rows, not bytes | Fixed (V409-010) | Byte-bounded pages (LIM-API-24, 8 MiB; LIM-API-25). Follow-up (pending): V414-008, the CHECK 23514 → `ErrInvalid` classification at every store writer. |
| GetPlannerSessions reads whole state | Retained by decision (V409-010) | The strict decode validates the whole state (RUN-R89, one query per collection, LIM-RUN-7). Bounded by `planner_session_state_size` (2 MiB) and one session per StageExecution (LIM-SCH-2) (risk table). |
| Placement `GetRun` under the Stage lock | Not present at `725b863e`; now present and retained | V407-014 found that the claim referred to uncommitted V405-002 code. Since V405-002 (`efeba911`), `controlplane/placement.go:480` reads the Run by primary key under the Stage `FOR UPDATE`, once per placement commit. It copies the Workflow snapshot (≤ 2 MiB, LIM-CAT-4) without decoding it; the other JSON columns are decoded and validated (risk table). |
| PurgeRun in one transaction | Retained by decision (V409-010) | Metadata GC stays transactional (DEL-R13/R22/R59). Bounded by LIM-ART-6 (50,000 revisions); `TestPostgresRunRevisionLimitAndBoundedPurge` passes at that bound (risk table). |
| Credential usage `OR` across two tables | Retained, rewrite rejected (V409-011) | The target index covers only active Runs, but terminal unreleased allocations must remain references, and a target-only probe still chose a seq scan (risk table). |
| FindEvalReceipt `LIMIT 1` without `ORDER BY` | Unchanged by decision (V409-014) | This is a retained-receipt lookup of the Project. Experiment IDs carry 128-bit randomness, and no production writer creates a second receipt (risk table). |
| Lenient `metrics_batch` read | Refuted | RUN-R90 requires per-row omission. `TestPostgresStageMetricsBatchOmitsOnlyUndecodableRows` (V409-011) pins it. |
| N+1 loops | Fixed (V409-011) | Batched binding and principal locks, label reads, release markers and the Retry owner read. Example: a 16-label Run pin went from 17+17 to 1+1 statements. |
| Two clocks in allocation release | Refuted | Completion writes both timestamps from one `statement_timestamp()` (`runstore/allocation_store_sql.go:43-44`), so the CHECK holds. |
| Planner append requires an unexpired claim | Refuted as a defect | SCN-R204 specifies that behaviour, and the expiry and takeover test covers it (V409-003). |
| Resume `UPDATE` without CAS | Refuted | The Run is locked `FOR UPDATE` (`resume_store.go:69`), and `resumableStageSQL` rechecks `run.state = 'failed'` in the same transaction before the update (V409-003). |
| Duplicate reads | Fixed in part (V409-011) | The gateway-recovery second read is removed. Rejected by decision: merging the artifact ref-page statements and consolidating the credential JSON paths. The Resume Project-first lookup is kept for lock order (`TestPostgresResumeLocksTheProjectBeforeTheRun`). |

### F. Open questions

| Question | Disposition | Evidence |
| --- | --- | --- |
| Plans of optional filters (`$n = '' OR ...`) under pgx prepared statements | Confirmed class; partly fixed, rest pending | V409-004 removed the shape from the Audit finding filter: the owner-wide generic plan read 12,001 findings before. V409-007 measured allocation history (V414-025) and the Eval summary (V414-026), both pending. Every other statement with the shape is open question (1) of V414-027. |
| Partial-index choice for `claimAuditSQL` and `nextLiveRunForDeletionSQL` | Measured by V409-013: both read beyond their bound; follow-ups pending | Section H. `claimAuditSQL` reads every claim row (50,011 at 50,009 Audits, both plans), so the work grows with the retained Audits: follow-up (pending) V414-028. `nextLiveRunForDeletionSQL` reads the whole execution table or every execution of the Audit, plus every Run, on each call, and deletion calls it once per child Run: follow-up (pending) V414-029. The Project deletion claim was aligned by V409-007. |
| `pg_stat_user_indexes` | Not used; demo statistics not collected | Index decisions rest on catalog dependencies, named readers and EXPLAIN evidence on disposable fixtures (V409-007). V409-013 had no demo access. |
| Real frequency of the A1 40001 | Not measured; retained | V409-001 reproduced it deterministically and bounded the retry. Production frequency is not collected (risk table). |
| Real sizes of `eval_view_pairs`, `eval_member_projections`, `audit_report_pages` | "4,096 × 64 MB" refuted; documents bounded; distributions not collected | `report_build.go:145-149` refuses a page that would take the build past `artifacts.MaxPayloadSize` (64 MiB, LIM-REV-8). The running sum is rebuilt from recorded pages on replay (`apply`, `report_build.go:82-85`), so a whole build stays at or under 64 MiB. Projection documents have a 1 MiB CHECK. View pair documents are Server-assembled, under about 2 MiB (section G). No test drives the 64 MiB rejection: open question (7) of V414-027. |
| `workflow_run_events` per Run is unbounded | Confirmed; detection gap pending | Neither schema nor spec bounds the count. Reads are bounded per page (LIM-API-24/25), but Run deletion removes the whole history in one transaction. V409-010 declined a cap until a representative workload exists. No signal detects a large history: open question (6) of V414-027. |

### Accepted risks

| Risk | Owner | Enforcement or detection | Reopen when |
| --- | --- | --- | --- |
| Subject-filtered Audit item pages read every item of the Audit (100,000 custom, 39,625 generic) | audits (auditstore reads) | `MaxItemsTotal` ≤ 100,000 (`validateLimits`); the V409-004 budget test (100,000 + 6,432) fails above it | the subject filter serves a polling or hot path, or a measured workload repays an index's write cost |
| Exact workspace and review counts read every counted row | audits | each counted row is read once through the Audit's index; `TestPostgresAuditReadsAtFindingScale` budgets | workspace polling latency is measured as material, or counts are needed beyond the Audit limits |
| 18 `audit_findings` indexes on every admission | findings | each index has a measured reader or an FK, probe or uniqueness role (V409-007); PER-R21 probe test | admission write latency or WAL volume is measured as material |
| Broad triggers kept: Project binding admission, Eval Audit child projection | projects-artifacts (DEL-R40), evals | `TestRowTriggerInventory` and `RequireColumnScopedGuards` pin every trigger of the five tables | a write path measures trigger cost, or admission moves to another enforcement point |
| PublishView reads every member projection | evals | 1 MiB CHECK per document, member limits (LIM-EVF-8) | a measured admitted-size workload, with a design that keeps EVM-R164–R168 (V409-009) |
| `nextMembersSQL` rescans members | evals | native admission only, ≤ 1,000 members, ≤ 10 admissions per tick | representative plans show the scan is material and cursor recovery is specified (V409-009) |
| Quota `count(*)` per SubmitProposal | findings | advisory lock; LIM-FND-46 and FND-R185 cap matched receipts at 4,096 per Run | the per-Run limit rises or submit latency is measured as material |
| GetPlannerSessions whole state | execution-core | 2 MiB CHECK per session; one session per StageExecution; one query (RUN-R89) | an equivalent strict-read design and measured admitted-size bytes (V409-010) |
| Placement `GetRun` under the Stage lock | execution-core (control plane) | one primary-key row per placement commit; snapshot ≤ 2 MiB (LIM-CAT-4), copied, not decoded | lock waits on Stage rows or placement-commit latency are measured, or LIM-CAT-4 rises |
| PurgeRun transactional metadata GC | projects-artifacts | LIM-ART-6 (50,000 revisions); `TestPostgresRunRevisionLimitAndBoundedPurge` | a deferred metadata-GC design with durable recovery is proposed (V409-010) |
| Credential usage query scans allocations | runtime-config | result capped by `LIMIT`; usage checks run on credential deletion and inspection only | a representative full-allocation fixture shows a semantics-preserving gain (V409-011) |
| FindEvalReceipt without `ORDER BY` | evals | 128-bit random experiment IDs; no production writer of a second receipt | such a writer becomes reachable or corruption detection is required (V409-014) |
| Project-fence 40001 on finding writes | findings | bounded retry (`InTxWithRetry`); detected by the controller's "Audit Controller iteration failed" log and the documented 503 `storage_transaction_conflict` | retries are seen exhausted, or V414-012 changes the classification |

Sections A–G below describe the reviewed revision `725b863e`; section H records the V409-013 verification.

---

## A. Топ по важности

| # | Где | Что | Класс |
|---|-----|-----|-------|
| 1 | `migrations/000024:147-166`, `findingintake/postgres.go:547` | Каждая запись в `audit_findings` берёт `projects FOR NO KEY UPDATE NOWAIT`; это конфликтует с `FOR SHARE`, который берут триггеры `contractor_require_active_project` при создании Run, AuditExecution и project-артефактов. Параллельная запись finding и создание Run в одном Project → синтетический 40001 без retry (`InTx`, не `InTxWithRetry`), ошибка доходит до модели как failed tool call | locking, high |
| 2 | `findingintake/postgres_sql.go:39-52` | `listAuditInboxSQL`: `owner_id = $1 AND (audit_id = $2 ... OR EXISTS(...))`. Индекса по `owner_id` нет, `OR EXISTS` не превращается в semi-join → seq scan всей `finding_proposal_receipts` (все владельцы, все Run'ы, до purge Project) на каждую страницу публичного inbox | index, high |
| 3 | `runstore/stage_store_sql.go:92` | `startPlannerSQL` лочит Stage, потом Run (UPDATE stage_executions → INSERT planner_sessions → UPDATE workflow_runs) на pool-store без внешней транзакции; все остальные точки коммита держат порядок Run → Stage (RUN-R29). Второй Run-first locker на `preparing` Stage даст 40P01 | locking, medium |
| 4 | `auditstore/.../read_sql.go:46-57` | `listItemsPageSQL` сортирует по `round.ordinal, item.ordinal`, индекса с `round.ordinal` нет → сортировка всех items Audit (до 100k) на каждую страницу | index, medium |
| 5 | `auditstore/.../workspace_sql.go:13-32`, `finding_reads_sql.go:47-56` | Workspace GET (UI polling) делает три `count(*)` по всем findings Audit (до 100k), unreviewed-count без покрывающего индекса | count on hot path, medium |
| 6 | `baseline:2084-2088`, `000030:16-28` | `stage_allocations` BEFORE UPDATE триггеры без `UPDATE OF`; `_require_completion` делает JOIN двух таблиц на каждый UPDATE, включая release и per-report `execution_report_received` | trigger, medium |
| 7 | `baseline:3386`, функция `:850-884` | `audits_protect_preparation_baseline` делает EXISTS по `audit_executions` на каждый UPDATE `audits`, а строка обновляется на каждый event append | trigger, medium |
| 8 | `baseline:2231,2236` | На самой горячей append-таблице `workflow_run_events` два индекса без читателей: `UNIQUE(event_id)` и `(run_id, occurred_at, sequence_number)`; единственный reader идёт по PK | dead index, medium |
| 9 | `baseline:2329,2333` | `workflow_runs_runtime_labels_idx` (GIN) и `workflow_runs_publication_mode_idx` не используются ни одним запросом; GIN на массив дорог при каждой записи | dead index, medium |
| 10 | `evalstore/deletion.go:71-125` | Purge Eval: один `DELETE eval_experiments` каскадом через members (10k) → records (без границы) → projections (по 1 MB) в одной транзакции под `projects FOR SHARE`; Audit purge батчится по 1000 строк | unbounded, medium |

## B. Схема и миграции

- **Индексы:** 146 обычных + ~113 PK/UNIQUE на 73 таблицы. Кандидаты в мёртвые (10): два на `workflow_run_events`, два на `workflow_runs`, `stage_executions_recovery_idx (state, updated_at)` (все чтения несут `run_id`), пять `eval_experiments_*_page_idx` (единственный list упорядочен по `created_at DESC`). Избыточные (5): `eval_records_native_recent` (частичная копия `eval_records_recent`), `audit_findings_correlation_audit_scan_idx` vs `audit_findings_list_idx` (отличаются только `COLLATE "C"` tie-breaker, два reader'а выбрали разные collation), `audit_executions_audit_identity`, `audits_project_owner_identity`, одна из пары `projects_owner_identity`/`projects_project_owner_key`.
- **`audit_findings` несёт 18 btree-индексов + 3 row-триггера + statement-триггер**: каждая admission пишет 18 entries. `audit_findings_correlation_scope_idx` существует только ради FK-probe правила.
- **Триггеры без `UPDATE OF`/`WHEN`** (класс, который 000012 починил для `workflow_runs`): `stage_allocations` ×3, `audits_protect_preparation_baseline`, `artifact_bindings` ×2 (projection фильтрует `scope_kind` внутри plpgsql; admission берёт `projects FOR SHARE` на каждый bump revision), `audit_executions` ×3 (включая deferred constraint trigger, ставящий commit-time probe на каждый UPDATE), `contractor_eval_retain_deleted_execution` (4-table join на каждый DELETE Run).
- **State CHECK-наборы без теста Go↔схема**: `stage_executions`, `audits`, `audit_items`, `audit_executions`, `projects`, `eval_experiments`, `eval_submissions`, `audit_review_requests`, `audit_findings`, `credential_operations`; тест есть только для `workflow_runs`. Предикаты partial-индексов (`audits_reconcile_idx`, `eval_experiments_claim_active_idx` и др.) тоже копии Go-наборов.
- **Миграции:** `contractor_protect_finding_proposal_receipt_retention` определена пять раз; ещё восемь функций переопределены по разу; backfill в 000008 и 000014 без `*_migration_test.go`. Раннер применяет все pending файлы в одной транзакции, поэтому `CREATE INDEX CONCURRENTLY` невозможен; это ограничение стоит задокументировать.
- **Keyset vs индекс**: `telemetry/allocation_resources_sql.go:13-29` сортирует по `(terminal_at DESC, allocation_id DESC)`, индекс заканчивается `stage_execution_id DESC` и не ведёт по owner; `telemetry/repository.go:340-358` cleanup сортирует по `(expires_at, report_id)` при индексе `(expires_at, stage_execution_id)`; `projectstore/deletion_claim_sql.go:10` сортирует по `(deletion_requested_at, project_id)`, индекс ведёт `deletion_claim_expires_at`.

## C. SQL домена Audit / Eval / Findings

- `review_write_sql.go:18-30` **`recordReviewDecisionSQL`**: INSERT решения в data-modifying CTE выполняется всегда, внешний UPDATE с `state='pending'` может затронуть 0 строк, command tag игнорируется → решение записано, запрос не переведён. Сегодня защищено сервисом под локом, но SQL обещает гарантию, которую не даёт.
- `evalstore/projections.go:198-205` **PublishView** читает полные `document` всех members (10k × 1 MB) при каждой републикации.
- `evalstore/comparison_pages_sql.go` + `comparison_pages.go:76-115`: `MATERIALIZED` CTE со всеми документами пар выполняется дважды на страницу (count + page), `selectedMemberPageSQL` парсит `convert_from(document)::jsonb` дважды на пару.
- `finding_reads_sql.go:127` **два часа для одного правила**: `statement_timestamp()` в `reviewUnexpiredSQL` против `clock_timestamp()` в view, claim и expire; REV-R43 называет один DB-clock.
- `read.go:141-167` listItems: мёртвая ветка `$2 = 0 → LIMIT 100001`; `read.go:199-221` ListExecutions `LIMIT 1000001` без продакшен-вызова (уже в testonlyexports baseline).
- `report_review_sql.go:94-97`, `report_sql.go:39-42`, `lifecycle_sql.go:165-168`: `state <> 'collected'` не использует partial `audit_executions_reconcile_idx`; переписать как `state IN ('intent','submitted','collecting')`.
- `item_review_renewal_sql.go:13-43`, `claimAuditSQL:50-77`: join по `(subject_kind, subject_id)` индексирован только для `state='pending'`; нужен `(audit_id, subject_kind, subject_id, kind)`.
- `evalstore/reconcile_sql.go:8-20` `nextMembersSQL`: anti-join по всем 10k members каждый тик, когда большинство уже submitted; нужен курсор `next_ordinal`.
- `evalstore` `recovery()`/`project()`/`locked()`: ~8 round-trip и повторный lock experiment на каждого member в цикле до 100 за тик.
- `inventory_pages_sql.go:39-52`: keyset по выражению `'1' || execution_id > $3` не использует индекс.
- `findingintake/service.go:173`: `count(*)` receipts на каждый SubmitProposal → O(n²) за Run у лимита 4096.
- `audit_artifact_links` INSERT без `ON CONFLICT` в четырёх местах: коллизия logical_key → `ErrConflict` → execution вечно в `collecting`. Нужен тест.
- `GREATEST(clock_timestamp(), updated_at + '1 microsecond')` повторяется ~40 раз; в evalstore есть константы `observe`/`advance`.
- Несколько 10+-строчных SQL инлайном (`read.go:145-153,230-236`, `execution.go:156-163`, `evalstore/projections.go`, `submissions.go:91-105`) при 3–8-строчных в `*_sql.go`.

## D. SQL ядра исполнения и артефактов

- `runstore/run_event_store.go:444-449` **ListRunEvents** ограничен строками (1000), не байтами: `data` до 2 MiB → страница до 2 GiB. `readExactBatchSQL` уже показывает шаблон с `sum(octet_length) OVER`.
- `runstore/planner_batch.go:10-30` **GetPlannerSessions** читает `state` (2 MiB) всех сессий Run (до 1024), callers используют только `state->'plan'` и scan-часть.
- `controlplane/placement.go:423` полный `GetRun` (со snapshot Workflow, декод и валидация) под `FOR UPDATE` Stage ради `required_placement_labels` (уже в V407-014).
- `artifacts/postgres_purge.go:77-102` PurgeRun: до 50k ревизий в одной транзакции, все version ID в памяти, затем `FOR UPDATE` всех versions и blobs чанками в той же транзакции.
- `credentials/runtime_repository_sql.go:64`: `OR` через две таблицы; ни GIN, ни expression-индекс не применимы.
- `persistence/postgres/receipts.go:168-173` FindEvalReceipt: `LIMIT 1` без `ORDER BY` при возможных двух receipts в разных Project.
- `telemetry/metrics_batch.go:34-36`: битая JSON-строка StageMetrics молча пропускается (`continue`) вместо ошибки.
- N+1: `scheduler/allocation_release.go:98-135` (32 executions × ListStageAllocations × per-allocation mark при существующем `ListStageAllocationsBatch`); `runtimeconfig/repository.go:348-370` и `agentprincipal/repository.go:131-153` лочат по одной строке в цикле вместо `= ANY($1) ORDER BY ... FOR UPDATE`.
- `runstore/allocation_store.go:312-336`: `clock_timestamp()` для attempted и `statement_timestamp()` для completed при CHECK `completed >= attempted`.
- `runstore/planner_store_sql.go:10`: append в журнал планировщика требует `scheduler_claim_expires_at > clock_timestamp()`; между потерей renewal и сдачей lane (SCH-R120/R166) поздний renewal даёт `ErrConflict`. Либо зафиксировать в спеке, либо проверять только `scheduler_claim_id`.
- `runstore/resume_store.go:97-99`: `UPDATE workflow_runs SET state='pending'` без CAS `state='failed'`.
- Дубли: `artifacts/ref_page_sql.go:11` vs `:26`; `credentials/repository.go:176-183` vs `runtime_repository_sql.go:17` (оба кастят `canonical_document::jsonb` из text на каждую версию); `gatewayrecovery/waiting.go:119-152` и `resume_store.go:46-69` читают одну строку дважды.

## E. Что сделано хорошо

- Порядок локов Run → Stage соблюдён во всех точках коммита Scheduler, Memory, gateway recovery (кроме п. A3); `FOR UPDATE OF <alias>` везде, где есть join; claim-statements через `ControllerClaims.LeaseSQL` с epoch; `SKIP LOCKED` по индексированному порядку; lease и expiry по часам БД.
- `KeyedMutation` для Audit и Eval, `ON CONFLICT DO NOTHING` + replay для Run, Project, RuntimeConfig, credentials, allocations; все `ON CONFLICT` цели совпадают с unique-индексами.
- Keyset-пагинация с tie-breaker и совпадающим индексом в подавляющем большинстве списков; `IS NOT DISTINCT FROM` для nullable, типизированные optional-фильтры; `= NULL` и `NOT IN (nullable)` не найдены.
- Audit purge батчится по строкам и байтам (`pg_column_size`), pointer-колонки очищаются до целей; `readExactBatchSQL` удерживает payload сверх бюджета; CopyFrom для eval pairs; statement-level триггеры с transition tables (000008, 000024, eval_members).
- Раннер миграций: checksum-ledger, schema-scoped advisory lock с polling под `lock_timeout`, `NewAdvisoryKey` с length-prefix без коллизий; все timestamps `timestamptz`, 44 enum-CHECK, 119 `octet_length`-границ.

## F. Не проверено без базы

- Планы для `($n = '' OR col = $n)` под prepared statements pgx; выбор partial-индексов для claim-probe (`claimAuditSQL`, `nextLiveRunForDeletionSQL`).
- `pg_stat_user_indexes` для подтверждения мёртвых индексов.
- Реальная частота 40001 из п. A1 (нужен параллельный intake при создании Run).
- Реальные размеры `eval_view_pairs.document`, `eval_member_projections.document`, `audit_report_pages.content` (`string_agg` до 4096 × 64 MB).
- Число `workflow_run_events` на Run не ограничено ни схемой, ни спекой.

## G. A10 verification: bounded Eval purge (V409-009, 2026-10-24)

**Landing revision:** branch `claude/rv-c2` on base `9cc34c0b`: first implementation `56ca02d3ef1ca528c391e1d06c74c3f7ab7a0865`, merge of `claude/review-v407-v409` in `17de731b`, review fixes `f58ec2799a2d67042d820757a5eeb6571b098528`. PostgreSQL 17, disposable database `contractor_test_c2`, one migrated schema per test.

**Budget:** LIM-EVM-20, EVM-R243. One pass is one transaction of at most `CleanupBatchRows` = 1,000 row mutations and 16 MiB of stored row bytes. A row mutation is every row the pass deletes or rewrites, including the rows its cascades and triggers delete or rewrite. Stored bytes are the sum of `pg_column_size` over every column of each such row, read before the statement deletes it, so a compressed or TOAST value counts at its stored size. This is row-byte accounting, not a claim about heap memory or WAL volume. The first unit of a fresh pass always goes. The Eval coordinator commits one pass per claimed tick. The Project controller commits an Eval pass alone and goes on to finding receipts and Artifacts only in a transaction whose Eval pass found nothing left.

**Phase table**, in pass order. The order follows the FK graph of `000001_baseline.sql` and the triggers listed. Each statement is `DeleteWithinBudget` (`internal/persistence/postgres/cleanup_sql.go`): at most limit+1 candidates by `ctid`, a running row and byte sum, and a DELETE of the prefix that fits.

| # | Table | Row cost | Delete-time triggers and cascades | Why nothing uncounted happens |
|---|---|---|---|---|
| 0 | `eval_projection_queue` (UPDATE: view pointer and summary set to NULL) | 1 | none | taken `FOR UPDATE` first; the dirty triggers take this row before they write a projection |
| 1 | `eval_view_pairs` | 1 | `contractor_eval_view_immutable` (check only) | leaf; FK to the queue |
| 2 | `eval_view_charts` | 1 | same | leaf |
| 3 | `eval_progress_observations` | 1 | none | leaf; FK to the experiment |
| 4 | `eval_selections` | 1 | none on DELETE | NO ACTION FK to records, so it goes before records |
| 5 | `eval_evidence_refs` | 1 | none | lookup source of the artifact dirty triggers, so it goes before the projections |
| 6 | `eval_suboperations` | 1 | `contractor_eval_immutable` (check only) | lookup source of the tombstone trigger; child of submissions |
| 7 | `eval_submissions` | 1 | none on DELETE | its suboperations are gone; lookup source of the Run and Audit dirty triggers |
| 8 | `eval_records` | 1 | `contractor_eval_immutable` (check only) | its selections are gone |
| 9 | `eval_execution_tombstones` | 1 | same | no lookup source is left that could insert another |
| 10 | `eval_member_projections` | 1 | none | same |
| 11 | `eval_members` | 1 | same as 8 | every child (5–10) is gone, so the CASCADE FKs remove nothing |
| 12 | `eval_plan_resources` | 1 | same as 8 | child of the frozen plan |
| 13 | `eval_commands` | 1 | none | readable until here |
| 14 | `eval_experiments` (final unit) | 3 + plan + queue + claim (at most 6) | CASCADE to `eval_frozen_plans`, `eval_projection_queue` and `eval_controller_claims` (one row each at most; every other child is already empty); the statement trigger `eval_experiments_collection_delete` upserts the two `eval_collections` rows | measured by `finalPurgeUnitSQL` before the DELETE |
| P1 | `eval_dataset_revisions` (Project pass, after every experiment) | 3 | the row trigger `eval_datasets_collection` upserts two collection rows; `eval_datasets_immutable` (check only) | the RESTRICT FK from experiments is satisfied; the trigger rows are charged |
| P2 | the Project's `eval_collections` row | 1 | none | — |
| P3 | Eval `mutation_receipts` | 1 | `contractor_protect_mutation_receipt` (allows it only while `contractor.eval_purge` is on) | last, so receipts replay until nothing else of the Project's Eval data is left |

**Exclusion during a pass:** the pass holds the Project row (`FOR SHARE`, under the Project controller also its deletion-claim lock), the experiment row and the queue row (`FOR UPDATE`) for the whole transaction. Every dirty trigger takes the queue row first. The tombstone trigger on Run and Audit DELETE must update the experiment row. PublishView, Command, SelectRecords, PutRecord, Admit and PutSuboperation lock the experiment and refuse a deletion-requested one. So no writer adds a dependent row the pass has not counted, and no cascade removes one.

**Accounting check:** every committed pass in `internal/evalstore/purge_postgres_test.go` compares `Rows` with the sum of `pg_stat_get_xact_tuples_inserted + _updated + _deleted` over all tables of the test schema, taken inside the pass transaction (minus the counters from before the pass). They are equal on every pass. Bytes charged agree with an independent `information_schema`-driven measurement of what was removed, within the rewritten queue and collection rows. Measured standalone purge of an admitted, fully populated experiment: 8 members; 3,171 dependent rows (3,115 evidence references from 3 results of 1,024 references each, 24 records, a human assessment and selection, a resolved run-create suboperation, a seeded tombstone, a published view, receipts); 18,332,784 stored bytes. Passes: 1,000 rows / 197,250 bytes; 1,000 / 188,989; 1,000 / 188,989; 149 / 15,971,999 (byte-bound); 26 / 1,785,643 (done).

**Fixed authority overhead:** per pass, lock reads of the Project, experiment, claim (`FOR UPDATE`, no write) and queue rows; the drain checks (`purgeBlockedSQL`, or `ProjectBlocked` and `purgeProjectSQL`); `set_config`; and one candidate read per phase, each capped at limit+1 rows. A Project pass adds one `nextPurgedExperimentSQL` lookup per experiment it finishes. None of these writes. The final unit adds at most 6 row mutations; its bytes are the experiment row (draft at most 1 MiB), the frozen plan (document and setup, each at most 1 MiB), the already invalidated queue row, the claim and the two collection rows, about 3.1 MiB at most.

**Largest legal row:** documents that the database bounds take at most 1 MiB each and at most two per row (about 2 MiB). Rows the Server assembles are larger:

- The queue `suites` document has at most 5,000 suites, because 10,000 members (LIM-EVF-8) in A/B pairs form at most 5,000 suites. Each suite ID has 128 characters, both variant IDs have 128 characters, every counter is 10,000 and every ratio has its longest value. That document is 9,295,001 bytes (bytea, stored uncompressed under `STORAGE EXTERNAL`).
- A progress observation's `counts` jsonb (`{"a","b","suites"}`) of the same shape is 11,260,056 bytes stored uncompressed, about 10.74 MiB, or 211,807 bytes compressed.
- View pair documents stay under about 2 MiB.

16 MiB holds the largest of these with about 5.5 MB to spare. `TestSharedPostgresEvalPurgeFitsLargestLegalRowInFreshPass` stores both worst-case rows under `STORAGE EXTERNAL`. The first pass spends the queue row's 9.3 MB and stops before the observation, because the remaining budget is too small for it. The next fresh pass removes the observation. A directly seeded row larger than 16 MiB, which no validated write stores, goes alone as the first row of a fresh pass, so there is neither a size rejection nor a zero-progress pass.

**Original implementation (A5):** the aggregate regressions were copied onto a temporary worktree at `9cc34c0b` and run against PostgreSQL. `TestProjectDeletionPurgesEvalDataInBoundedTransactions` failed with "step 3 spent 5705 Eval row mutations, more than 1000". `TestSharedPostgresEvalPurgeYieldsBetweenTicksAndRefusesPartialReads` failed with "one tick left 0 evidence references, want one bounded pass". The evalstore tests use the new pass API, so they cannot compile against the base revision.

**Review changes (2026-10-24):** the first review found that the schema guard compared only the columns of `eval_*` tables. A new cascading child (`eval_member_notes`, or `review_eval_links` outside the `eval_` names) or a new writing DELETE trigger passed it, and a pass then counted 63 rows while PostgreSQL counted 4,863. `TestSharedPostgresEvalPurgeCoversCascadesAndTriggers` now builds the graph from `pg_constraint` and `pg_trigger`, starting from every table the purge deletes from (`experimentPurgePhases`, the experiment row, `projectPurgeTables` and the Eval receipts). Every referencing table of any delete action must be emptied by an earlier step or be a cascading child of `eval_experiments` counted in the final unit (`finalPurgeCascades`, from which `finalPurgeUnitSQL` is built). Every trigger that a purge statement fires and whose PL/pgSQL source, or the source of a schema function it calls, writes a table or runs dynamic SQL must be declared in `purgeDeleteTriggers`, from which the row costs and charged bytes are derived. The guard follows triggers on the rows a declared trigger writes. Each counted table's columns must equal `purgeRowColumns`. Thirteen fixtures, each a DDL change in a rolled-back transaction, show the guard failing: cascading children inside and outside the `eval_` names, the child of an uncounted cascade, an uncounted cascade of the experiment row, SET NULL and RESTRICT children, writing DELETE triggers (direct, through a helper function, and through dynamic SQL), triggers on the queue update and on the collection rows, a declared trigger that no longer writes, and an unmeasured column. Reordering the phases or dropping a final cascade also fails it. The rule "row and byte limits per pass, and the first unit always goes" now has one definition, `persistencepostgres.CleanupBudget` with `DeleteWithinBudget` (optionally ordered); the Eval purge (LIM-EVM-20) and the Project purge of finding proposal receipts (LIM-DEL-3, in `receipt_id` order) both use it. The Audit purge stays separate because LIM-REV-10 bounds it by rows alone. The accepted purge rate (one experiment pass per coordinator tick) is recorded in EVM-R243 with its detection signal: the Server logs each experiment and Project purge's first pass, every 600th pass and its last, with the pass count, rows, bytes and age since the deletion request, and warns once a purge is older than one hour.

**Commands (all with `CONTRACTOR_TEST_DATABASE_URL` set; none of the purge tests skipped):** after the review fixes, `go test -tags=integration -count=1 ./internal/evalstore/... ./internal/evalservice/... ./internal/evalcoordinator/... ./internal/projectlifecycle/... ./internal/projectstore/... ./internal/persistence/... ./internal/httpapi/public/... ./internal/findingintake/... ./internal/app/...` passed in every package except `internal/persistence`. That package fails only `TestSQLAccessFollowsTableOwnership` (`internal/cli/scan_sweep_remote.go:196`) and `TestKeyedMutationsUseTheSharedHelpers` (`internal/app/single_active_budget_test.go`), both of which fail identically on the integration branch head `c36e3caa`. `make verify-public-api`, `make lint` and `make test-wire-cross-language` passed.

## H. Database verification of the review (V409-013, 2026-10-24)

**Revision and environment.** `claude/rv-x1`, base `efd5d6ab` plus merge `5be7e24c` (`claude/review-v407-v409`). Go 1.26.9 and PostgreSQL 17.11 on the local cluster. Every test and probe used the disposable database `contractor_test_x1`, its per-test schemas and one scratch catalog schema (`v409013_catalog`, all migrations through 000039 applied, dropped afterwards). No demo or production database was opened. Optional demo statistics: **not collected**.

**Dependencies.** V409-013 has 17 transitive dependencies. The 13 direct V409 dependencies (V409-001…012, V409-014) and the transitive V407-016, V407-025, V407-028 and V407-029 are `completed` in `tasks/done/`. Each has a verification record, and each `implementation_commit` is an ancestor of the verified revision. No task depends on V409-013, directly or transitively. Two records need a note: V409-007's `implementation_commit` `20ef8ffc` is its renumber commit (the implementation is `9c6f6e64`), and V409-014 has the test-only commit `7a2cacc5` after its close.

**`audit_findings` catalog** (scratch schema, `pg_index` and `pg_trigger`). 18 indexes:

- uniqueness and FK targets: the primary key; `finding_id_audit_id_key` (target of five FKs); `audit_id_first_receipt_id_key`;
- public pages and reports: `list_idx`, `state_idx`, `global_created_idx` (owner-wide pages);
- correlation and comparison reads: `correlation_audit_scan_idx`, `correlation_audit_group_idx`, `correlation_audit_group_order_idx`, `correlation_group_idx`, `correlation_project_group_order_idx`, `correlation_project_scan_idx`;
- the counts and the FK probe: `correlation_scope_idx`;
- FK probe indexes: `current_assessment_idx`, `current_decision_idx`, `duplicate_target_idx`, `first_receipt_idx`;
- purge: `purge_pointers_idx`.

Triggers: 2 row triggers (`audit_findings_initialize_correlation`, `audit_findings_protect_correlation`) and 3 statement triggers (`audit_findings_correlation_inserted`, `_updated` and `_deleted`).

**Claim and deletion probes.** The review asked how `claimAuditSQL` (`auditstore/internal/pgstore/claims_sql.go`) and `nextLiveRunForDeletionSQL` (`lifecycle_sql.go`) choose partial indexes. No completed task had measured either. A throwaway harness in `internal/auditstore/internal/pgstore`, deleted afterwards and never committed, measured both with `explaintest.ExplainPrepared` in `force_custom_plan` and `force_generic_plan`. Each statement ran as EXPLAIN (ANALYZE, BUFFERS) EXECUTE in a rolled-back transaction with `SET LOCAL statement_timeout = '30s'` and `lock_timeout = '5s'`, inside the test's own schema. Reads are `Analysis.Reads`, the rows that table scans returned plus the index entries returned, and "blocks" are shared buffers. Each table was ANALYZEd before measuring.

- Fixture: `newFindingScaleFixture` (one collected Audit with 20,001 executions and 4,001 findings, 8 background Audits with 5,000 executions each, and 2,000 empty Audits). The empty Audits were reshaped into 1,600 `completed`, 101 `draft`, 100 `active` with free claims, 100 `active` with live foreign claims and 99 `paused` without executions, each with its claim row. That gives 306 entries of `audits_reconcile_idx`. Then 48,000 more `completed` Audits with claim rows were added: every Audit keeps its claim row from `CreateDraft` until purge.
- For the deletion probe, the Runs of background Audits 1–3 (15,000) and of the scale Audit's last 50 executions were cloned. `run_deleted_at` was stamped on every execution whose Run is absent, the scale Audit was set to `deleting`, and later background Audit 0 (all 5,000 Runs gone). The bulk steps ran with user triggers disabled; this fixture is not a model of a store write path.

| Statement | Case | Plan | Rows | Reads | Blocks |
| --- | --- | --- | --- | --- | --- |
| `claimAuditSQL` | 2,009 Audits, limit 8 | custom / generic | 8 | claims 2,012 / 4,020; audits 920 | 364 / 380 |
| `claimAuditSQL` | 2,009 Audits, limit 64 | custom / generic | 64 | claims 4,020; audits 920 | 942 / 998 |
| `claimAuditSQL` | 50,009 Audits, limit 8 | custom / generic | 8 | claims 50,011; audits 920 | 3,992 / 3,995 |
| `claimAuditSQL` | 50,009 Audits, limit 64 | custom / generic | 64 | claims 50,011; audits 920 | 4,746 / 4,782 |
| `nextLiveRunForDeletionSQL` | 20,001 executions, 51 live Runs | custom | 1 | executions 60,001; Runs 15,053 | 22,370 |
| `nextLiveRunForDeletionSQL` | same | generic | 1 | executions 39,954; Runs 15,053 | 9,404 |
| `nextLiveRunForDeletionSQL` | drained, 5,000 executions | custom / generic | 0 | executions 10,000; Runs 15,053 | 3,391 / 3,388 |

The claim plan seq-scans `audit_controller_claims` and hashes it against the candidates. `audits_reconcile_idx` is BitmapAnd-ed with whole scans of `audits_project_state_idx`, so blocks grow with the retained Audits even though the reconcilable set and the audit rows read stay the same. The Eval claim already avoids this pattern with per-candidate lookups (`evalstore/claims_sql.go`). The deletion probe has no `run_deleted_at` predicate, so it joins every execution of the Audit (in the custom plan, the whole table) with every Run. `deleteReleasedRunBatch` calls it once per child Run. Index entries above the live row count are versions that the fixture's bulk update superseded and VACUUM had not removed. A first `run_deleted_at IS NULL` rewrite was not clean evidence for the same reason (20,003 entries with the live tail). The follow-ups record it as a direction only: V414-028 and V414-029, both pending.

**Static re-checks on the verified revision.**

- Report bound: the check at `auditimport/report_build.go:145-149` and the replayed running sum at `:82-85`. Each page also has its own 64 MiB CHECK (`audit_report_pages_content_check`, migration 000004).
- Allocation release clock: `runstore/allocation_store_sql.go:43-44`.
- Resume lock and state recheck: `runstore/resume_store.go:69,83` and `resume_store_sql.go:18`.
- Review read clocks: `finding_reads_sql.go:60,66,140`.
- PublishView document read: `evalstore/projections.go:198-199`.
- Placement `GetRun`: `controlplane/placement.go:480`.
- Artifact-link logical keys, by source identity, as listed in Status C.

**Commands** (`CONTRACTOR_TEST_DATABASE_URL=postgres://ruslan@localhost:5432/contractor_test_x1?sslmode=disable`):

- The task's test command, `go test -tags=integration -count=1` over the 25 package trees listed in V409-013 (agentprincipal … telemetry), on the tree of `5be7e24c` (only documents and task files change after it): exit 0. 30 packages passed, and 11 have no test files. 3,899 tests passed, 0 failed and 2 skipped, neither of them a database test: `TestPinnedLiteLLMContract` (pinned LiteLLM contract environment not configured) and `TestPlannerOTLPCollectorProjection` (`CONTRACTOR_TEST_OTLP_ENDPOINT` unset). All 117 tests in the 50 integration-tagged files of these packages ran and passed. No `-run` filter was used, so no selection could match zero tests.
- `make verify-public-api`: passed.
- `make lint`: passed (task records and spec documents consistent, layer and import contracts kept).
