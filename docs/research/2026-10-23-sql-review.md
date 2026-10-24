# SQL and schema review — 2026-10-23

**Revision reviewed:** `main` `725b863e`. File:line references are as of that revision.

**Scope:** every statement in `internal/**/*_sql.go` and inline SQL in `internal/**/*.go`, the migrations (`000001_baseline.sql` plus 000002–000031), triggers, functions, indexes, constraints, the pgx pool configuration and query budgets.

**Method:** three static reviewers (schema and migrations; Audit, Eval and Findings SQL; execution and artifact SQL), the top findings re-verified by hand. No PostgreSQL was available, so no EXPLAIN or `pg_stat_user_indexes`; dead-index claims rest on grepping every non-test Go file for the index's distinguishing predicate. Findings of the 2026-10-23 Server review already owned by V407 tasks are not repeated.

**Outcome:** the SQL is careful where it matters (lock order, claims, receipts, keyset pages, batched purge, connection budgets). The defects are concentrated in triggers and indexes that accumulated without readers, a few hot statements without a supporting index, one lock-mode conflict on the Project row, and reads bounded by rows but not bytes.

## Status (2026-10-23)

All findings open. Tasks V409-001 … V409-014 (milestone `sql-review-2026-10`); V409-013 verifies the static claims on a real database and closes the review.

| Section | Findings | Tasks |
| --- | --- | --- |
| A top | A1 | V409-001 |
| A top | A2, C count per submission | V409-002 |
| A top, D | A3, D resume CAS, D planner-append fence, D two clocks | V409-003 |
| A top, C | A4, A5, C `state <> 'collected'`, C review-request index, C keyset on expression, C listItems dead branch, C ListExecutions | V409-004 |
| C | C recordReviewDecisionSQL, C audit_artifact_links, C two review clocks | V409-005 |
| A top, B | A6, A7, B triggers | V409-006 |
| A top, B | A8, A9, B dead and redundant indexes, B keyset/index mismatches, B audit_findings collation | V409-007 |
| B | B state CHECK sets | V409-008 |
| A top, C | A10, C PublishView, C nextMembers | V409-009 |
| D | D ListRunEvents bytes, D GetPlannerSessions, D PurgeRun transaction | V409-010 |
| D | D N+1 and duplicates, D metrics_batch lenient read, D credential OR | V409-011 |
| B, C | B migrations hygiene, C advanceUpdatedAt, C SQL placement | V409-012 |
| C, D | C comparison pages, C recovery() per member, D FindEvalReceipt | V409-014 |
| F | open questions | V409-013 |

The findings below describe the reviewed revision.

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

**Landing revision:** branch `claude/rv-c2`, implementation commit `56ca02d3ef1ca528c391e1d06c74c3f7ab7a0865` on base `9cc34c0b`. PostgreSQL 17, disposable database `contractor_test_c2`, one migrated schema per test.

**Budget:** LIM-EVM-19, EVM-R242. One pass is one transaction of at most `CleanupBatchRows` = 1,000 row mutations and 16 MiB of stored row bytes. A row mutation is every row the pass deletes or rewrites, including the rows its cascades and triggers delete or rewrite. Stored bytes are the sum of `pg_column_size` over every column of each such row, read before the statement deletes it, so a compressed or TOAST value counts at its stored size. This is row-byte accounting, not a claim about heap memory or WAL volume. The first unit of a fresh pass always goes. The Eval coordinator commits one pass per claimed tick. The Project controller commits an Eval pass alone and goes on to finding receipts and Artifacts only in a transaction whose Eval pass found nothing left.

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

**Commands (all with `CONTRACTOR_TEST_DATABASE_URL` set; none of the purge tests skipped):** `go test -tags=integration -count=1 ./internal/evalstore/... ./internal/evalservice/... ./internal/evalcoordinator/... ./internal/projectlifecycle/... ./internal/projectstore/... ./internal/persistence/... ./internal/httpapi/public/...` passed in every package except `internal/persistence`. That package fails only `TestSQLAccessFollowsTableOwnership` at `internal/cli/scan_sweep_remote.go:196`, which already fails on the base revision. `make verify-public-api`, `make lint` and `make test-wire-cross-language` passed.
