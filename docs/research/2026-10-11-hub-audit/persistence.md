# Аудит узла: persistence и владение SQL (main @ 02bbdcbf)

Работа шла только на чтение. В репозитории ничего не менялось, к базе данных я не подключался. Вспомогательные AST-инструменты и промежуточные данные лежат в `/tmp/claude-1000/-home-ruslan-src-contractor/0fecd9f4-f328-4d49-ab01-b67e6eace15e/scratchpad/hub-audit/persistence/`:
- `sqlmap/` строит карту «таблица → пакет, R/W»; результат в `access_prod.tsv`;
- `nested/` ищет использование pool внутри tx-scope;
- `uses/` считает, какой API persistence использует каждый импортёр;
- `sqlloc/` считает SQL-литералы;
- `erd.json` — данные, извлечённые из ERD.

Отметки: **V** = VERIFIED (путь прочитан или подтверждён скриптом), **S** = SUSPECTED (нужен тест или прогон).

---

## A. Карта узла

| Пакет | non-test LOC | Экспорт | Fan-in (prod) | Churn с 2026-08-12 |
|---|---|---|---|---|
| internal/persistence | 2 (`doc.go`) + `sql_placement_test.go` | 0 | 0 | 118 коммитов на всё поддерево |
| internal/persistence/postgres | 879 в 11 файлах | 44 (14 const, 3 var, 5 типов, 22 функции) + 2 метода | 24 | 76; non-test Go всего 13 |
| internal/persistence/migrations | 10 строк Go + 6959 строк SQL в 102 файлах | 1 (`Files`) | 1 | 100 (каждый файл добавлен один раз, правился только 000062) |

**Fan-in.** 24 production-пакета: app, artifacts, auditservice, auditstore, controlplane, credentials, evalservice, evalstore, findingintake, gatewayrecovery, gitimport, httpapi/public, memory, performance, planner, projectlifecycle, projectstore, rundeletion, runservice, runstore, runtimeconfig, scheduler, settingsstore, telemetry. Ещё 6 пакетов импортируют узел только из тестов: agentskills, auditstandards, planner/scan, planner/session, planner/streamline, tests/integration/streamline. С тегами `integration,e2e` число production-импортёров то же — 24.

**Fan-out.** Только `persistence/migrations`, pgx v5 и stdlib.

**Использование API.** 
- DBTX импортируют 13 пакетов, InTx — 10, SQLState — 8, InTxWithRetry — 8. 
- 31 из 44 экспортов используют не больше двух импортёров. 
- Типичный импортёр берёт 1–5 идентификаторов.

**Схема как узел** (на 000102):
- 71 таблица, плюс ledger и view `audit_live_item_reviews`;
- 97 FK;
- 88 триггеров на версии 000096 (84 из них row-level) и ещё 2 добавлены в 000098;
- 121 вхождение `ERRCODE='23514'` в RAISE.

**SQL-текст в Go:**
- 638 statement-литералов, 4874 строки, в 20 пакетах (19 со ссылками на таблицы плюс app с advisory-lease);
- 71 файл `*_sql.go`, 4709 строк, в 14 пакетах;
- больше всего SQL в evalstore (125 statements), auditstore (102), runstore (71), artifacts (69).

**Коммиты с миграциями (100).** У 54 из них затронуто не больше двух других prod-пакетов, у 29 — пять и больше.

**Top-10 файлов хаба по churn:** postgres_integration_test.go 49, migrate_test.go 17, migrate.go 6, tx.go 5, budgets.go 5, pool.go 4, budgets_postgres_test.go 4, audit_finding_identity_migration_test.go 4, eval_view_generation_migration_test.go 3, sqlstate.go 2.

**Co-change вне хаба** (по 118 коммитам хаба): contractor-public-v1.yaml 36, httpapi/public/types.go 34, app/app.go 33, ui/.../generated/public.ts 30, runstore/store.go 27, runstore/types.go 26, openapi_contract_test.go 25, router.go 24, handler_test.go 24, fakes_test.go 23. По каталогам: httpapi/public 50, runstore 39, docs/spec 38, app 36, scheduler 28, artifacts 26.

**Команды:**
- `go list -f '{{.ImportPath}} {{join .Imports " "}}' ./...` (также с `-tags integration,e2e`) и awk по импорту;
- `git log --no-merges --since=2026-08-12 --format=%H -- <path> | wc -l`;
- `git log ... --name-only | sort | uniq -c`;
- цикл `git show --name-only` по коммитам хаба;
- AST-инструменты из scratch (см. начало);
- `rg`.

---

## B. Обязанности

Go-часть — небольшое ядро, оно связное:
1. Pool и бюджеты: `pool.go`, `budgets.go`, `budget_tracer.go`.
2. Транзакции: `tx.go` (InTx, BeginTx), `retry.go` (InTxWithRetry), `commit.go` (AfterCommit).
3. Классификация ошибок: `sqlstate.go`, `transient.go`, `errors.go`.
4. Мигратор и ledger: `migrate.go`.
5. `Timestamp`.

God-файлов и широких интерфейсов нет: DBTX — 3 метода, самый большой файл — `migrate.go` на 279 строк.

Низкая связность у второй половины узла — схемы как общего ресурса:
- **Все контексты в одной папке миграций.** God-объекты схемы: `workflow_runs` (34 колонки, 9 row-триггеров, 3 пакета-писателя, 12 пакетов-читателей), `audits` (35 колонок), `000034_audit_state.sql` на 546 строк.
- **Тесты чужих контекстов живут в `persistence/postgres`.** 17 upgrade-тестов fold-миграций Audit, Eval и Credentials занимают 1727 строк. Рядом лежат `testdata/audit-profile-current.json` и центральный инвентарь таблиц `postgres_integration_test.go:63-86`.
- **Репо-уровневые проверки:** `sql_placement_test.go` и ERD-генератор `scripts/erd/build_erd.py:107-121`, который грепает Go-код, чтобы понять, какой пакет пользуется таблицей.

Это должно жить у владельцев: upgrade-тесты — в пакетах-владельцах таблиц, инвентарь таблиц — генерироваться из одной карты владения (см. F1).

---

## C. Связи, которые не видит archtest

**C1. Карта «таблица → пакет» (V, через `sqlmap`).** Неопределённость: CopyFrom и динамические имена таблиц учтены отдельно — `evalstore/comparison_projection.go:30,42`, `evalstore/deletion.go:120`, `evalstore/projections.go:303`.

Предлагаемые владельцы, 71 таблица:

| Владелец | Таблицы |
|---|---|
| runstore | 9 (workflow_runs, stage_*, events, planner_sessions, resumptions, owner_queue_controls, output_publications) |
| artifacts | 7 |
| telemetry | 3 |
| performance | 1 |
| projectstore | 1 |
| settingsstore | 1 |
| credentials | 4 |
| runtimeconfig | 4 |
| gatewayrecovery | 5 |
| auditstore | 14 (audit_* вместе с findings) |
| findingintake | 2 (finding_proposal_*) |
| evalstore | 20 |

**У 11 таблиц писатели из нескольких пакетов.** Записи не владельцем:

| Пишет | Таблица (владелец) | Где |
|---|---|---|
| scheduler | workflow_runs, stage_executions (runstore) | `scheduler/postgres.go:264`, `:372`, `:384`; `postgres_sql.go:11`; `admission.go:35` |
| scheduler | gateway_recovery_waits, gateway_run_routes (gatewayrecovery) | `scheduler/postgres.go:258,261` |
| gatewayrecovery | workflow_runs (runstore) | `admission.go:65`; `waiting.go:18,30` |
| findingintake | audit_findings, audit_finding_assessments, audit_artifact_links (auditstore) | `direct_verification.go:385,407`; `direct_verification_sql.go:11` |
| runstore | artifact_bindings (artifacts) | `resume_store.go:136-137` |
| artifacts | workflow_run_output_publications (runstore) | `postgres_purge.go:119,174` |
| rundeletion | finding_proposal_receipts (findingintake) | `deleter_sql.go:11` |
| projectlifecycle | projects (projectstore) | `controller.go:376,396,414,471`; `controller_sql.go:20` |

**Чтения чужих таблиц** (число мест): auditservice 69 (все — таблицы auditstore), evalstore 51 (15 таблиц пяти контекстов), findingintake 29, auditstore 14, projectlifecycle 12, credentials 10, telemetry 9, scheduler 7.

**SQL-зависимости вверх по слоям (V)** — то, что запретил бы `layers_test.go`, если бы видел SQL:
- platform/gatewayrecovery **пишет** execution-таблицу workflow_runs (`admission.go:65`);
- platform/telemetry читает workflow_runs, stage_executions, stage_allocations (`allocation_resources_sql.go:16-18`);
- platform/artifacts читает workflow_runs (`postgres_audit_import_sql.go:16`) и удаляет output_publications;
- execution/credentials читает product-таблицы audits и projects (`runtime_repository.go:214,258`);
- execution/runstore читает projects (`queue_store_sql.go:18`).

**C2. Что должно уйти за store владельца:**
- переходы состояний Run и Stage, которые делают scheduler и gatewayrecovery: state machine из трёх писателей;
- пересчёт оценки finding в findingintake;
- оттаивание bindings в runstore;
- удаление publications в artifacts;
- 69 мест прямого SQL в auditservice по таблицам auditstore.

**Прецедент:** `rundeletion/boundary_test.go:55-84`, но он проверяет только runstore и только audit/finding-таблицы.

**C3. Межконтекстные триггеры (V).** Миграции Eval вешают проекционные триггеры на таблицы других контекстов:
- `eval_runs_projection` на workflow_runs (000059:195);
- `eval_stages_projection` на stage_executions (:257);
- `eval_outputs_projection` на artifact_bindings (:290);
- `eval_artifact_projection` на artifact_binding_revisions (:310);
- `eval_audits_projection` / `eval_audit_children_projection` (:198,218);
- tombstone-триггеры на workflow_runs и audits (000057:59-62).

Изменение колонок state или finished_at в runs или audits меняет поведение eval, и ни импорт, ни тест этого не покажут.

**C4. JSON-формы внутри SQL (V, по счёту скрипта).** SQL разбирает 114 разных JSON-ключей. Примеры:
- `outcome` из StageResult: `scheduler/postgres.go:375,387`, `postgres_sql.go:11` и 7 мест в миграциях;
- `runtimeCredentialIds`: SQL в `credentials/runtime_repository.go:237,260`, а Go-теги в `runtimeconfig/run_snapshot.go:35` и `auditbaseline/baseline.go:65`.

Если переименовать поле wire-контракта, SQL молча начнёт видеть NULL.

**C5. Строковые константы протокола «БД → Go» (V):**
- SQLSTATE `"55000"` («Project is deleting») стоит литералом в 25 местах Go: artifacts 8, auditstore 13, runstore 3, evalstore 1. В `sqlstate.go` такой константы нет. Поднимается эта ошибка в 000033:67,144; 000034:460; 000039:189,224.
- `23514` перегружен: им помечаются и CHECK-ограничения, и бизнес-отказы триггеров.

**C6. DBTX скрывает разницу «autocommit или транзакция» (V).** Три способа обойти это:
- четыре одинаковых switch «pool → InTxWithRetry, Tx → inline»: `rundeletion/deleter.go:56-66`, `runstore/resume_store.go:143-150`, `auditstore/item_review_renewal.go:30-40`, `auditstore/lifecycle.go:122-130`;
- пять runtime-проверок `s.db.(pgx.Tx)`: `runstore/delete_store.go:30,100`, `auditstore/run_deletion.go:19`, `auditstore/resume_renewal.go:38`, `artifacts/git_source.go:27`;
- при этом уже есть типизированные конструкторы `evalstore.NewTxStore` и `artifacts.NewPostgresPurger(tx)`.

**C7. Механизмы идемпотентности, receipts и leases (V).** Их как минимум 16, больше, чем девять из AGENTS.md.

1. Ключ прямо в строке сущности: workflow_runs и projects (`request_idempotency_key`, `request_digest`), runtime_credentials и runtime_config_versions (`idempotency_key_digest`), audit_executions (`submission_key`), eval_members (`submission_key`), audit_review_requests/decisions (`idempotency_key`).
2. Отдельные receipt-таблицы с ключом (scope, operation, key): `audit_idempotency`, `eval_mutation_receipts`, `runtime_management_operations`, журнал `credential_operations`.
3. Естественный ключ: finding_proposal_receipts, run_stage_resumptions (`resume_store.go:34`), stage_transition_decisions, output_publications, gateway_recovery_failures, collection_receipt_id.
4. Receipt в виде артефакта: `findingintake/collection_publication.go:85-90`.

Как они защищаются от гонок:
- advisory lock первым: 12 мест, ключи строятся пятью разными способами — строковый префикс, deterministicID, JSON-массив (`evalstore/store.go:145`), сырой digest (`runtimeconfig/management.go:334`), два int (`runtimeconfig/credentials.go:165`);
- ON CONFLICT DO NOTHING с последующим чтением;
- unique-violation и replay;
- RR-снимок с повтором при 40001.

Как хранится ключ: сырым (audit, eval, runs) или digest-ом (runtime_*). Колонка digest запроса называется по-разному: `request_digest` и `request_sha256` (eval).

Leases и claims:
- `workflow_runs.scheduler_claim_*`;
- `projects.deletion_claim_*`;
- `audit_controller_claims` и `eval_controller_claims` — по сути одинаковые (`auditstore/claims_sql.go:85-97` против `evalstore/claims_sql.go:51-56`);
- `gateway_recovery_routes.probe_*`;
- session lease (`app/single_active.go:105`).

**C8. Экспорты без production-вызывающего (V):**
- `WithMigrationBudget` (`budgets.go:115-121`, содержит panic) — только в тесте;
- `ApplyMigrations` — 33 тестовых файла, production зовёт `ApplyMigrationsWithBudgets`;
- `MaxTransactionAttempts` — только тест;
- `DefaultBudgets`, `ErrInvalidDatabaseConfiguration`, `SQLStateStringDataRightTruncation`, `SQLStateInvalidTextRepresentation`, `SQLStateTooManyConnections` — используются только внутри пакета.

---

## D. Классы дефектов из AGENTS.md §3

**3.1. Одно правило в нескольких местах:**
- **Отображение SQLSTATE расходится (V).**
  - 23514: `ErrInvalid` в `ConstraintError` (`sqlstate.go:27-29`), `ErrConflict` в `telemetry/repository.go:560`, `ErrPrecondition` в `auditstore/collection.go:42` и `execution.go:39`.
  - 23503: `ErrInvalid` (`sqlstate.go:28`), `ErrNotFound` (`telemetry/repository.go:558`, `runstore/stage_store.go:58`, `auditstore/store.go:121`), `eval_not_ready` (`evalstore/store.go:102`).
  - Всего 54 вызова `SQLState()` в 28 файлах.
- **Список таблиц в четырёх местах (V):** миграции, `postgres_integration_test.go:63-86`, `scripts/erd/meta.json`, ERD.
- **Пересчёт оценки finding дублируется (V):** `auditstore/collection_sql.go:200-214` и `findingintake/direct_verification.go:398-410`.
- **Длина media type 255 (V):** CHECK в 000101, Go, OpenAPI, eval-schema.
- **JSON из БД читается нестрого (S, расхождение с политикой).** В stores около 100 вызовов `json.Unmarshal` (auditstore 23, evalstore 19, runstore 18, findingintake 11), тогда как AGENTS §2 требует «validate strictly … on read». Strictjson при этом используют credentials, runtimeconfig и scheduler.

**3.2. Исходы сбоев:**
- **Публикация выходов (G1, V):** временная ошибка становится постоянным `failed`.
- **Retry-замыкания безопасны (V).** Все 9 `InTxWithRetry` сбрасывают захваченное состояние. Единственный внешний эффект — filesystem-кандидат блоба в `collection_publication.go:79-90`; осиротевший блоб допускается спецификацией `blob-storage.md:85-100`.
- **Вложенного захвата pool внутри транзакции не найдено (V).** Три AST-проверки: pool или `.pool` в tx-scope; `recv.field.M(ctx)` без tx-аргумента; локальные переменные, полученные из pool, внутри tx-замыкания. Совпадения только ложные: `runtimeconfig/management.go:158` — это срез данных. Ограничение: вызовы через интерфейсы этим анализом не видны.
- **Скрытая проблема (S).** Восстановление после SQL-ошибки на том же DBTX (`auditstore/store.go:124-125`, `execution.go:43-57`, `collection.go:45-66`, `initial_round.go:113-121`) работает только на autocommit-pool. На tx-store оно получит 25P02. Сейчас tx-вызывающих не найдено: `auditcontroller` и `auditimport` работают через pool.

**3.3. Ограничение работы:** FK без индекса при purge (I1), `count(*)` на каждую страницу (I2), триггеры на каждое добавление события Run (I3), N+1 в публикации коллекции (I4).

**3.4. Жизненный цикл.** Receipts предложений не удаляются никогда (G3).

**3.5. Тесты, которые не могут упасть:**
- `TestLifecycleForeignKeyProbeIndexes` проверяет только FK на 3 родителей (`lifecycle_fk_indexes_test.go:28-34`), поэтому FK purge в Audit и Eval им не покрыты (V).
- Свежесть ERD ничем не проверяется (V).
- 103 тестовых файла в 24 пакетах пишут в таблицы сырым SQL: workflow_runs из 9 пакетов, audits из 8 (V). Каждая NOT NULL-колонка ломает их все.
- Upgrade-тесты сравнивают `to_jsonb(row)-ARRAY[...]`, и новая колонка ломает старые тесты (V, см. E1).

**3.7.** См. C8.

**Spec ↔ code:**
- `platform/persistence.md:111-121` («PostgreSQL state and its owners») называет 6 состояний на 71 таблицу. Правил для InTx, InTxWithRetry, AfterCommit там нет (V).
- Бюджеты БД описаны только в `docs/operations/timeout-configuration.md`.

---

## E. Цена изменений (3 коммита)

**E1. c13dfbb9 feat(audit): persist preparation attempts** (2026-10-09). 21 файл, слои product (auditstore, runservice), domain (auditdomain), platform (тест artifacts), kernel (миграция 000098, тесты, testdata).

Правки, которые потребовала структура узла, а не сама фича:
- два чужих upgrade-теста, где пришлось исключить новую колонку `phase` из `to_jsonb(a)-ARRAY['continuation_count','phase']` (`audit_continuation_removal_test.go`, `audit_finding_identity_migration_test.go`);
- 284 строки testdata Audit-профиля в пакете persistence.

ERD при этом не перегенерирован (G2).

**E2. 8d385042 fix(eval): share media type contracts** (2026-10-09). 10 файлов: миграция 000101, тест artifacts, OpenAPI, `api/evals/v1` schema, сгенерированный UI-клиент, httpapi-тесты, evaldomain, sync-скрипт. Правило длины уже существовало в четырёх представлениях, CHECK в БД стал пятым копированием. Единого источника, из которого генерировались бы остальные, нет.

**E3. b97fa2b6 feat(auditstore): fold collection receipts (V365-017)** (2026-10-04). 16 файлов. Внутреннее изменение схемы Audit потребовало правок:
- в SQL runstore (execution-слой): `delete_store_sql.go`, `store_sql.go`;
- в тестах evalservice, projectlifecycle, auditservice, e2e;
- в центральном `wantTables`.

Позже 02bbdcbf перенёс этот SQL в rundeletion и добавил boundary-тест. Это подтверждает: без карты владения такие правки неизбежны.

---

## F. Рекомендации (по отношению ценности к трудоёмкости)

**F1. Карта владения таблицами и fitness-тест** — структура, M, риск низкий.
- *Проблема:* записи не владельцем и upward-зависимости из C1 (`scheduler/postgres.go:264`, `gatewayrecovery/admission.go:65`, …), список таблиц в четырёх копиях, `persistence.md:111` неполон.
- *Изменение:*
  - один файл владения (таблица → пакет и режим R/W), который начинается с групп из `scripts/erd/meta.json`;
  - тест по образцу `rundeletion/boundary_test.go`, но для всех пакетов и с allowlist «чужих» доступов, в котором у каждой записи указана причина;
  - генерация из этого файла `wantTables`, `meta.json` и таблицы владельцев в spec.
- *Эффект:* число записей не владельцем видно и уменьшается по храповику (сейчас ≈16 мест); копий списка таблиц станет одна.

**F2. Индексы для FK-проверок purge и обобщённый тест** — дефект по производительности (S), S–M.
- *Проблема:* I1.
- *Изменение:*
  - индексы `audit_executions(audit_id, round_id)`, `audit_findings(current_assessment_id)`, `audit_findings(current_decision_id)`, `eval_commands(experiment_id)`, `gateway_run_routes(route_key)`, `gateway_allocation_routes(route_key)`;
  - убрать фильтр трёх родителей в `lifecycle_fk_indexes_test.go:28-34` и сверять все FK с явным allowlist.
- *Как закрепить:* тот же тест и проверка `pg_stat_user_tables.seq_scan` в тесте purge на реалистичном объёме.

**F3. Классификация сбоев публикации выходов** — дефект, S.
- *Проблема:* G1.
- *Изменение:* в savepoint при `IsTransactionConflict` или `IsTransientFailure` прерывать всю транзакцию для replay. Дальше одно из двух: реализовать повтор `failed`, либо исправить PRJ-R58 и FAIL-PRJ-4.
- *Как закрепить:* integration-тест с инъекцией 40P01 и 57014.

**F4. Единый контракт ошибок «БД → Go»** — структура с профилактикой дефектов, S–M.
- *Изменение:* константа для 55000; бизнес-отказы триггеров вынести в собственные SQLSTATE (класс P0xxx) или различать по имени constraint; одна функция `Classify`; убрать 25 литералов и около 45 ручных switch.
- *Как закрепить:* тест, запрещающий 5-символьные SQLSTATE-литералы вне `sqlstate.go`.

**F5. Один механизм receipts и claims** — структура, M–L.
- *Изменение:* объединить `audit_idempotency`, `eval_mutation_receipts` и `runtime_management_operations` в одну таблицу с одним helper «lock → replay → claim». Сделать один построитель ключей advisory-lock и общий helper claim для audit и eval.
- *Срок:* до `tasks/v64-002-audit-priority-store.yml:24-26`, иначе там появятся новые intents и claims.

**F6. Проверка свежести ERD** — S, дефект документации.
- *Изменение:* без БД сверять `schemaVersion` в `database-erd.html` с последней миграцией, затем перегенерировать.
- *Отдельно, на решение пользователя:* свернуть 102 миграции в одну baseline — см. H6.

---

## G. Баги

**G1. Временная ошибка БД при публикации выхода записывается навсегда как `failed`, хотя по spec должна повторяться (V, medium).**
- *Путь:* `scheduler/postgres.go:685-728` (savepoint): любой `publishErr`, кроме ErrArtifactConflict, становится `OutputPublicationFailed` с кодом `publication_failed` (`:730-739`). То есть 40P01, 40001, 55P03 и 57014 внутри savepoint откатываются, а Run коммитится как `succeeded` с постоянным failed-receipt.
- `runstore/output_publication_sql.go` с `ON CONFLICT DO NOTHING` сохраняет первую запись; кода, который читал бы `failed` и повторял публикацию, нет.
- *Spec:* `artifacts/projects.md:139,443-447` и `failure-model.md:426` требуют «records and retries a bounded failure».
- *Тест:* `scheduler/postgres_integration_test.go:395-430` проверяет только P0001 как постоянный отказ.
- *introduced_by:* v23-005 (0c88a0f9).

**G2. ERD отстал на 6 миграций (V, low-medium, расхождение spec и code).**
- `docs/spec/database-erd.html` собран на `schemaVersion 000096`; в нём нет `audits.phase`, `audit_executions.preparation_snapshot`/`preparation_outputs` и триггеров из 000098.
- `docs/decisions/spec-history.md:1619` фиксирует 000096. Проверки нет.

**G3. Receipts предложений переживают удаление Project навсегда (V, low-medium, пробел в lifecycle).**
- 000093:27-46: триггер отклоняет любой DELETE. В Go нет ни одного DELETE по `finding_proposal_receipts`, FK на projects тоже нет.
- После purge (`projectlifecycle/controller.go:432-480`) остаются строки с owner_id, project_id, client_key, ссылками на proposal и evidence.
- Spec: LIFE-DEL-6 — «retention: not specified». В `projects.md` о receipts ничего не сказано.

**G4. Scheduler переводит waiting → running, не очищая причину (V, low).**
- `scheduler/postgres.go:264-265` меняет только state. `gatewayrecovery/waiting.go:30-31` в аналогичном месте ставит `gateway_recovered` и очищает message.
- Если дальше запускается следующая Stage, Run остаётся `running` с сообщением «Waiting for the model gateway». Публично StateReason не отдаётся, поэтому эффект внутренний. Тестов нет.

**G5. Несогласованная реакция на исчерпание retry (V, low):**
- `auditservice/start.go:65-66` делает replay только при 40001, а 40P01 пропускает, хотя рядом есть `IsTransactionConflict`;
- `findingintake/collection_publication.go:96` превращает исчерпанные serialization-retry в 409 `conflict` вместо 503 `storage_transaction_conflict` (`httpapi/public/errors.go:169-170`).

**G6. Восстановление после ошибки на том же DBTX (S, latent).** Описано в D 3.2. При первом же вызове на tx-store ответ станет internal error вместо replay.

**G7. Детерминированный отказ классифицируется как retryable (S).** Telemetry превращает CHECK-violation в ErrConflict (`telemetry/repository.go:556-563`), дальше он становится `runstore.ErrConflict` (`report_store.go:104`). Это класс V280. Последствия для ретраев отчётов не прослежены.

---

## H. Упрощение

| # | Что удалить или объединить | Экономия | Риск |
|---|---|---|---|
| H1 | Четыре dispatch-switch и пять `.(pgx.Tx)` → helper `postgres.Within(ctx, db, opts, fn)` и tx-only конструкторы | ≈50 строк, одно правило вместо девяти | низкий |
| H2 | Claims Audit и Eval (`auditstore/claims*.go`, `evalstore/claims*.go`) → общий SQL/helper | ≈100 строк | средний (fencing) |
| H3 | 25 литералов «55000» и ≈45 ручных switch → `Classify` (F4) | ≈80 строк | низкий |
| H4 | Удалить `WithMigrationBudget` (с panic), сделать неэкспортируемыми 6 идентификаторов, `ApplyMigrations` превратить в тестовый helper | ≈15 строк | низкий |
| H5 | Пересчёт оценки в findingintake → метод auditstore (рядом с `ProjectFindingDecision`, `review_write.go:46`) | ≈15 строк | низкий |
| H6 | Свернуть миграции в одну baseline (pre-production) | −6959 строк SQL, −1727 строк upgrade-тестов в 17 файлах, −284 строки testdata; уходит класс правок из E1 | высокий: ledger демо-стенда получит ErrMigrationDrift. Решение пользователя, противоречит правилу §6 «fold-миграция требует migration_test» |
| H7 | `memory.withActiveStage` (`memory/postgres.go:113-185`) — частичная копия InTx без `ApplyTransactionBudget`. Свою классификацию commit оставить, переиспользовать budget | ≈10 строк | низкий |

---

## I. Оптимизация

**I1. FK-проверки при purge Audit (S, высокий эффект).**
- Purge выполняет один `DELETE FROM audits` (`auditstore/lifecycle.go:153`). Каскад обходит FK, у которых нет непартиального индекса с ведущей колонкой:
  - `audit_executions(audit_id)` и `(round_id, audit_id)`: есть только партиальные индексы 000034:230-232, 000039:123-125, 000098:40-44, поэтому RI-запрос `WHERE audit_id=$1` идёт seq scan-ом — R+1 раз на purge (по одному на audit и на каждый round);
  - `audit_findings.current_assessment_id` и `current_decision_id` (NO ACTION): проверка на каждую удалённую оценку и decision, итого O((A+D) × все findings в БД).
- Это пропущенный сосед фикса 6e509ad7 (000099), который проиндексировал две другие ссылки.
- Ещё кандидаты: `eval_commands(experiment_id)` (только партиальный индекс), `gateway_*_routes(route_key)`, `eval_experiments(project_id, dataset_id, dataset_revision)`, `audit_finding_contributions(receipt_id)`, `audit_findings(first_receipt_id)`.
- *Проверить:* EXPLAIN или seq_scan на БД с ~10⁵ исполнений.

**I2. `ListEventsPage` считает `count(*)` всех событий на каждой странице (S).** `auditservice/events.go:67`. События append-only, нумеруются подряд с 1 (`createDraftSQL` стартует с `next_event_sequence=2`), так что Total, вероятно, равен ThroughSequence. Пролистать все N событий стоит O(N²/limit). Плотность нумерации при групповом выделении номеров (`collection_sql.go:93`) нужно подтвердить.

**I3. Каждое добавление события Run обновляет `workflow_runs` (S).**
- Места: `planner_store_sql.go:26`, `stage_store_sql.go:53`, 000012:42-45.
- Каждое такое UPDATE запускает 3 BEFORE-триггера и ставит в очередь до commit два DEFERRED constraint-триггера (000054, 000036), которые сразу возвращаются при неизменном `audit_completion`.
- *Фикс:* `UPDATE OF …` или `WHEN (OLD.audit_completion IS DISTINCT FROM NEW.audit_completion)`. Событий на Run — тысячи. Эффект измеряется через `pg_stat_user_functions`.

**I4. Публикация коллекции — N+1 внутри одной RR-транзакции под advisory lock (S).**
- На каждую запись (до 256) идёт 5–8 запросов: `collection_publication.go:181-260,309-324`; `postgres.go:49-64`; запрос на каждый hold `:222-224`.
- Итого ≈1.5–2 тыс. statements, плюс чтение блобов, пока держится lock.
- *Фикс:* пакетные запросы `ANY($1)` для authorize, receipts и holds.

---

## Топ-3

1. **G1/F3:** успешный Run навсегда фиксирует публикацию как `failed` при временной ошибке БД (deadlock, timeout), а обещанного spec повтора нет. Фикс S, нужен integration-тест с 40P01.
2. **F1:** ≈16 мест записи в чужие таблицы и 10 SQL-зависимостей вверх по слоям (gatewayrecovery и scheduler меняют state machine Run, findingintake пишет audit_findings) не видит archtest. Нужна карта владения с fitness-тестом по образцу `rundeletion/boundary_test.go`.
3. **I1/F2:** purge Audit проверяет FK по `audit_executions` и `audit_findings.current_*` без индексов (сосед 6e509ad7), а FK-тест покрывает только трёх родителей. Нужно обобщить тест и добавить индексы.
