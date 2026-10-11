# Аудит узла «Аудит» (product layer), main @ 02bbdcbf, 2026-10-11

Анализ только для чтения. В репозитории ничего не менялось, к БД не подключался. Запустил один тест без БД: `env -u CONTRACTOR_TEST_DATABASE_URL go test ./internal/persistence -run TestSQLPlacement`, он PASS (см. G8). Скрипты для подсчётов лежат в `…/scratchpad/hub-audit/audits/` (expcount, structw, sqlconcat, expnames, sqltables.py).

## A. Карта узла

**Команды.** Ниже LOC — это `find <pkg> -maxdepth 1 -name '*.go' ! -name '*_test.go' | xargs cat | wc -l`. Экспорты считал через go/ast (`expcount/main.go`). Fan-in и fan-out — через `go list -f '{{.ImportPath}}|{{join .Imports " "}}|…' ./...`. Churn: `git log --no-merges --since=2026-08-12 -- <pkg>`. Co-change: для каждого коммита узла `git show --name-only`, затем `uniq -c`.

| Пакет | LOC | файлов | экспорт (func/method/type/value) | fan-in prod | fan-out | коммиты (все/нетест) | создан |
|---|---|---|---|---|---|---|---|
| auditstore | 8573 | 49 | 264 (5/81/83/95) | 11: app, auditbaseline, auditcontroller, auditimport, auditservice, evalservice, findingintake, httpapi/public, projectlifecycle, rundeletion, runservice (+тесты artifacts, scheduler) | 7 | 69/66 | 09-05 |
| auditservice | 5380 | 30 | 163 | 4: app, auditcontroller, evalservice, httpapi/public | 16 | 88/70 | 09-05 |
| findingintake | 2776 | 12 | 57 | 5: app, auditimport, auditservice, httpapi/privateartifacts, httpapi/public | 8 | 46/37 | 09-06 |
| auditimport | 2814 | 11 | 18 | 2: app, auditcontroller | 14 | 55/39 | 09-06 |
| auditcontroller | 2324 | 10 | 33 | 1: app | 12 | 54/32 | 09-05 |
| planner/scan | 1018 | 6 | 13 | 1: app | 9 | 12/9 | 09-20 |
| auditscan | 308 | 2 | 6 | 2: auditimport, planner/scan | 6 | 2/2 | 09-21 |
| auditbaseline | 90 | 2 | 9 | 2: auditimport, auditservice | 7 | 3/3 | 09-19 |
| **Итого узел** | **23 283** | | | | | **166 уникальных** | |
| auditdomain (domain) | 4726 | 22 | 204 | 12 | 8 | 35 | |
| auditpriority (domain) | 615 | 5 | 59 | **0** | 3 | 3 | |
| rundeletion (product) | 222 | 2 | 12 | 3: app, auditcontroller, httpapi/public | 4 | 1 | 10-11 |

Контекст существует с 2026-09-05. За 36 дней 166 из 2245 коммитов репозитория задели узел, это примерно 4,6 в день. Из них 91 (55%) задели два пакета узла и больше, 62 (37%) заодно трогали OpenAPI, UI или httpapi.

**Топ-10 файлов по churn** (нетест): auditimport/importer.go 28 (987 строк); auditstore/validation.go 26 (1003); auditstore/types.go 26 (874); auditservice/start.go 26 (735); auditstore/read.go 19 (1059); auditcontroller/controller.go 19 (723); findingintake/postgres.go 18 (794); auditservice/finding_review.go 18 (897); auditservice/service.go 16; auditimport/report.go 16 (608).

**Co-change вне узла:** docs/spec/19-audits.md 54; api/openapi/contractor-public-v1.yaml 28; migrations/* 28 коммитов; ui/src/api/generated/public.ts 25; httpapi/public/types.go 20; audit_handlers.go 14; app/app.go 14; ui audits.test.tsx 14; openapi_contract_test.go 13; runservice/audit.go 11; controlplane/registry.go 11; auditdomain codec.go и inventory.go по 11.

## B. Обязанности

**auditstore** — один тип `PostgresStore` с 70 экспортными методами. Внутри семь швов, LOC по группам файлов:
- owner-жизненный цикл: create, transition, delete, project-deletion, lifecycle — 1319;
- старт и раунды: preparation, round, initial_round — 1159;
- claim-bound исполнение для контроллера: claims, execution*, run_deletion — 1067;
- сбор и находки: collection*, finding_events, findings — 754;
- ревью и отчёты: review_*, item_review_*, report* — 1248;
- чтения: read.go — 1124;
- общие validation.go, types.go, errors.go — 1902.

Методы делятся так: 21 метод принимает ControllerClaim, остальные 49 — owner-команды, журнал ревью, чтения и хуки для чужих контекстов (rundeletion 4, findingintake 2, credentials 1, projectlifecycle 1). Самые большие файлы — read.go, validation.go и types.go (83 типа). У `Audit` 27 полей, у `Execution` 21.

**auditservice** совмещает пять ролей:
1. каталог профилей и совместимость (model, compatibility, standards);
2. старт и инвентарь, включая построение следующего раунда (start, initial_inventory, next_round*). Это вызывает контроллер через `RoundBuilder`, поэтому auditcontroller импортирует auditservice;
3. owner-жизненный цикл (lifecycle, resume);
4. read-модели со своим SQL: finding_read, owner_lists, workspace, events, coverage;
5. журнал ревью: finding_review, action_review.

У `Service` 36 методов, у `public.AuditManagement` 30. auditstore создаётся ad hoc 32 раза (`auditstore.NewPostgresStore(tx)`), то есть зависимость не внедряется.

**Что не на своём месте:**
- next-round building лучше перенести в auditcontroller или auditdomain;
- read-SQL из auditservice — в query-часть стора;
- durable-типы сканов (`planner.ScanAttempt`, `ScanJobRecord`, статусы) живут в planner (execution layer), хотя это доменные понятия audit-скана. auditscan использует 12 идентификаторов planner, planner/scan — более 20.

## C. Связи, которые не видит archtest

**C1. Кто пишет и читает audit-таблицы** (`sqltables.py`, по строкам SQL; триггеры добавлены вручную).

| Таблица | Writers | Readers вне auditstore |
|---|---|---|
| audits | auditstore; триггер `contractor_admit_audit_finding` (062) при вставке hold от findingintake | auditservice 16, findingintake 13, evalstore 10, projectlifecycle 3, **credentials 1** |
| audit_events | auditstore (30 INSERT) + тот же триггер | auditservice, findingintake |
| audit_findings | auditstore (collection_sql.go:208, review_write.go:50), **findingintake** (direct_verification.go:407), триггер 062 | auditservice 13 |
| audit_finding_assessments | auditstore, **findingintake** (direct_verification_sql.go:11) | auditservice, findingintake |
| audit_artifact_links | auditstore, **findingintake** (direct_verification.go:385) | evalstore |
| audit_finding_contributions | только триггер (062:28) | auditservice 6, findingintake 2 |
| finding_proposal_receipts | findingintake, **rundeletion** (deleter_sql.go:11), триггер reconcile (093) | auditstore |
| audit_idempotency | auditstore | **evalstore** (submissions_sql.go:31-43) |

Больше одного писателя у шести таблиц. SQL пишется не только в auditstore: в auditservice 36 call sites (только чтения и advisory-локи), в findingintake 34, в rundeletion 1. VERIFIED.

**C2. Межконтекстный SQL:**
- credentials (execution layer) читает таблицу `audits` и JSON-ключ baseline `runtimeCredentialIds` напрямую (runtime_repository.go:256-262). Та же проверка для LLM-ключей идёт через порт `ListHeldAuditIDsByLLMCredential` (auditstore/store.go:23-52; credentials/lifecycle.go:113, 613). Это layer violation через SQL, archtest его не видит.
- evalstore связывает эвал с Audit через внутренности журнала идемпотентности `i.operation='audit.create'` (submissions_sql.go:41).
- auditstore читает `scheduler_settings` (read.go:631, execution_intent_sql.go:29) и `eval_submissions`/`eval_experiments` (store.go:208).
- Триггеры eval `eval_audit_children_projection` и `eval_audit_outputs_projection` (059) срабатывают на каждую строку audit_executions и audit_artifact_links; `audits_eval_tombstone` пишет в eval-таблицы при purge.

Всё VERIFIED.

**C3. Связь через форму JSON.** Ключи `llmCredentialIds` и `runtimeCredentialIds` из `auditbaseline.BaselineSnapshot` (baseline.go:64-65) читаются сырым SQL в store.go:33 и credentials:259. Переименование поля компилятор не поймает. Read-проекции декодируются лениво: комментарий в auditbaseline/projections.go прямо говорит, что они сохраняют «existing unknown-field semantics».

**C4. Контрабанда интерфейса.** `auditimport.RunReader` объявлен как узкий интерфейс из одного `GetRun` (types.go:31-33). В рантайме scan_recovery.go:31 приводит его к `plannersession.ScanHistoryStore`; если тип другой, получаем ошибку «history is unavailable». Настоящая зависимость шире объявленной. VERIFIED.

**C5. Широкие поверхности.** Сколько методов `PostgresStore` использует каждый импортёр: auditservice 29, auditcontroller 23 (интерфейс `Store` из 20 методов, types.go:35-56), auditimport 11, rundeletion 4, runservice 3, findingintake 2, projectlifecycle 1. httpapi методов стора не вызывает, но тянет 24 типа строк auditstore (`Audit`, `Item`, `AuditState`…) через `AuditManagement`. Поэтому новая колонка в стор-типе доходит до edge-слоя.

**C6. Экспорты только для тестов:**
- `auditstore.ListEvents` (read.go:447; 13 тестовых файлов, ноль прод-вызовов) и `ListExecutions` (read.go:249);
- `auditservice.ActiveCheckReviewKind` и `ApplicabilityReviewKind` (finding_review_types.go:16-17);
- `findingintake.RetentionSourceHeld`, `RetentionAuditHeld`, `RetentionDiscarded` (types.go:86-88) — сама логика retention живёт в SQL;
- `findingintake.StableSubmissionID` (types.go:219).

`TransitionTrusted` и `AppendReviewEvents` вызываются только внутри пакета.

## D. Классы дефектов (AGENTS §3)

**3.1. Одно правило в нескольких местах** (всё VERIFIED):
- **Матрица переходов и действий Audit** — четыре копии: auditstore/validation.go:222-241 (`transitionAllowed`) и :285-294 (`ownerTransitionAllowed`), auditservice/lifecycle.go:85-101 (`lifecycleTarget`), SQL CASE в deletion.go:96-99, UI controls.tsx:44-57 (STOPPABLE, DELETABLE). Поля `allowedActions` в API нет.
- **Выделение событий**: «revision += k, next_event_sequence += k, k событий подряд». Это 30 вручную написанных `INSERT INTO audit_events` в 16 файлах auditstore плюс plpgsql 062:33-47. Плотность последовательности ничем не проверяется. Дефект V279-001 (9c0c162c) появился именно отсюда.
- **Сброс находки в `proposed` при новой оценке** — три копии: collection_sql.go:208-216, findingintake/direct_verification.go:407-413, триггер 062.
- **Правило retention_state** — две копии: триггер 093 `contractor_reconcile_finding_retention` и rundeletion/deleter_sql.go:11-21.
- **Предикат видимости проекта** (скрыть eval-воркспейсы) — три копии: projectstore/store_sql.go:18-22, auditstore/store.go:204-210, auditservice/owner_lists.go:11-17. Две последние добавлены одним коммитом cc167827.
- **Грамматика Idempotency-Key**: auditstore/validation.go:20, auditservice/finding_review.go:885, projectstore/store.go:21, credentials/repository.go:30, httpapi run_handlers.go:266, requestid.go:22, ui/src/mutations/idempotency.ts:1, OpenAPI :6407 (`*` + maxLength), SQL CHECK в миграциях. Плюс более слабая проверка в action_review.go:218 (`validReviewIdentity`), см. G5.
- **Корзины покрытия**: workspace_sql.go:14-17 и auditimport/report.go:499-501 расходятся, см. G1.
- **Строки операций** `audit.start` и `audit.transition` записаны литералами больше чем в 15 местах, хотя константы есть (types.go:306-309). Ещё одна копия в evalstore.
- **Множество терминальных состояний Run**: auditcontroller/controller.go:675 против runstore/types.go:46.
- **Ограниченный strict-JSON обход токенов** — четыре реализации: planner/scan/report.go:284-345, auditdomain/canonical.go:63-160, auditpriority/verdict.go, scanplan/document.go. Отдельно существует strictjson.

**3.2. Исходы отказов и идемпотентность.** В одном этом контексте девять долговечных механизмов replay плюс три пространства advisory-локов. По AGENTS их было «девять на всю схему», это заметно меньше реального числа.

| # | Механизм | Где |
|---|---|---|
| 1 | audit_idempotency (owner, operation, key, digest) | create/start/transition/resume/delete |
| 2 | audit_review_requests (audit, key) + reuse по subject | CreateFindingReview |
| 3 | audit_review_decisions (audit, key) | DecideFinding / DecideActionReview |
| 4 | audit_executions.submission_key + request_digest | CreateExecutionIntent, runservice.CreateAudit |
| 5 | collection_receipt_id / collection_request_digest | Collect |
| 6 | finding_proposal_receipts, детерминированный receipt_id | приём от Worker |
| 7 | finding_proposal_audit_holds PK | ImportIntoAudit / RetainAuditCollectionBatch |
| 8 | детерминированный assessment_id + сравнение digest | direct_verification.go:323-353 |
| 9 | digest коммита отчёта (только в summary события) + replay по состоянию | report_sql.go:75 |
| — | advisory: `audit-finding-review:`, `audit-action-review:`, `audit-resume:owner:key` | finding_review.go:48/148, action_review.go:28, resume.go:52 |

Нерешённые исходы:
- **Replay DELETE после purge.** Строки idempotency удаляются каскадом (`ON DELETE CASCADE`, 000034), поэтому повтор потерянного ответа получает 404. Спека говорит только «Begin/replay durable delete» (audits.md:1298). VERIFIED, решение не записано.
- **Finalize по таймауту** повторяется с нуля, см. G6.

**3.3. Ограничения работы:**
- Финализация отчёта читает до 100 001 элементов (read.go:189), до 500 страниц покрытия (report.go:221) и до 100 001 находки (findings_sql.go:26). Для каждой находки — отдельный `ReadProjectExact` (report.go:406). Всё это в одном `Finalize` под общим `OperationTimeout`, по умолчанию 10 с (auditcontroller/types.go:23).
- Ни лимит числа находок на Audit, ни лимит отчёта не записаны в docs/spec/platform/limits.md. Это расхождение спеки и кода.
- Тестов на реалистичном объёме почти нет: единственный цикл на 200 строк — в provenance_read_postgres_test.go:226.

**3.4. Жизненный цикл.** Purge хорошо покрыт: auditstore/purge_postgres_test.go:37 проверяет каждый FK, есть projectlifecycle …WithDecidedReviews, evalservice. Пробелы:
- Не решено, что делать с pending finding-triage запросами в терминальном Audit. Время их не закрывает, а счётчик врёт (G2).
- Не решён replay после purge (см. 3.2).
- Решение UI скрывать Delete в активных состояниях противоречит AUD-R222 (G4).

**3.5. Тесты, которые не могут упасть:**
- Gate размещения SQL проверяет только одиночный литерал (sql_placement_test.go:78-99), конкатенация его обходит (G8).
- `fakeControllerStore.TransitionClaimed` (controller_test.go:854-867) не применяет таблицу переходов, которую проверяет настоящий стор (validation.go:296-304). Это класс V280.
- 13 тестовых файлов проверяют события через тестовый `auditstore.ListEvents`, а прод читает через `auditservice.ListEventsPage` с другим SQL и allowlist для summary.
- Тест корзин workspace покрывает 6 статусов покрытия из 10 (service_postgres_test.go:180-183), `excluded` среди них нет.

**3.7. Нет вызывающего:**
- `auditpriority` — 615 LOC без единого прод-импортёра. V64-000 завершён, вызывающие появятся в V64-001…011, все pending. Это нарушение «build only what has a caller», риск принят неявно, без владельца (§3.9).
- Экспорты только для тестов перечислены в C6.

## E. Цена изменений

**1. cc167827 feat: owner-wide APIs.** 49 файлов: auditstore, auditservice, httpapi/public, миграция 097, OpenAPI, сгенерированный TS-клиент, около 30 файлов UI. Слои product, edge, kernel, api. Правки, которые навязала структура, а не фича:
- предикат видимости вписан дважды (store.go и owner_lists.go), потому что owner-список Audit живёт в auditstore, а списки находок и ревью — в SQL auditservice;
- рядом с `ListParams.State` появилось параллельное `States`;
- любое новое чтение требует метод в `AuditManagement` (types.go), фейк, хендлер, OpenAPI, генерацию, обёртку UI и query-key.

**2. c13dfbb9 feat(audit): preparation attempts.** 21 файл: 12 в auditstore, runservice, auditdomain, artifacts-тесты, миграция 098, persistence/postgres. Навязанные правки:
- новая колонка `phase` потребовала правок в двух копиях списка колонок (store.go:70-97), в `scanAudit` и в round_sql;
- сломались два посторонних миграционных теста со снимками `to_jsonb(audits)`;
- в runservice/audit.go:227-246 пришлось добавить второй компаратор «submission == intent». Он дублирует `auditcontroller.matchingRoleIntent` (controller.go:519-532).

**3. 9c0c162c fix V279-001.** 10 файлов, auditstore и auditservice. Причина дефекта: выход из waiting_review в одном из трёх SQL-путей не закрывал report review. Пришлось вынести общие CTE-фрагменты и переписать арифметику revision и событий в deletion.go, project_deletion.go, transition.go и report_review.go, а также сигнатуру хелпера в пяти местах. Корень — инлайн-учёт событий в каждом statement (см. F3).

## F. Рекомендации (по ценности и усилию)

**F1. Один владелец audit-таблиц.**
- Проблема: матрица C1 — шесть таблиц с несколькими писателями, credentials:256 и evalstore submissions_sql.go:31.
- Изменение: весь SQL по `audit_*` и `finding_proposal_*` перенести в auditstore (или его подпакеты). credentials и evalstore ходят через порты, как это уже сделано для LLM (store.go:23).
- Эффект: у каждой таблицы ровно один писатель-пакет; число пакетов с SQL по audit-таблицам уменьшится с 8 до 1–2.
- Трудоёмкость M–L, риск средний.
- Как закрепить: тест «table ownership» по образцу sql_placement_test, где для каждого пакета перечислены разрешённые таблицы.
- Тип: структура.

**F2. Единый аллокатор событий.**
- Проблема: 30 `INSERT INTO audit_events` + 062.
- Изменение: одна SQL-функция `contractor_append_audit_events(audit_id, jsonb)` или Go-билдер CTE; плюс CHECK или тест на плотность sequence.
- Эффект: 31 копия → 1, класс V279 закрыт.
- Трудоёмкость M, риск средний.
- Как закрепить: grep-тест, что `INSERT INTO audit_events` встречается не больше одного раза.
- Тип: структура.

**F3. Доменные таблицы в auditdomain.**
- Проблема: см. 3.1 и G1/G4.
- Изменение: матрица действий, корзины покрытия и множества состояний — одна таблица в auditdomain. Списки `IN (...)` в SQL генерировать из неё; добавить `allowedActions` в read-модель Audit; фейк стора должен использовать ту же таблицу.
- Эффект: копий меньше в 4–5 раз.
- Трудоёмкость M, риск низкий.
- Как закрепить: cross-language fixture `api/testdata/audit-actions.json` и тест UI на нём.
- Тип: дефект (G1, G4) + структура.

**F4. Разрезать auditstore по швам** после F2: claim-bound контроллерные команды (около 1070 LOC), owner lifecycle (1320), журнал ревью и отчётов (1250), находки и сбор (750), read/query (1120 + read-SQL из auditservice). Каждому шву — свой тип стора.
- Эффект: импортёр видит 3–23 метода вместо 70; churn validation.go и types.go распределится.
- Трудоёмкость L, риск средний.
- Как закрепить: archtest-правило «auditservice не импортирует claim-пакет».
- Тип: структура.

**F5. Ограничить финализацию отчёта.**
- Проблема: см. 3.3 и G6.
- Изменение: при приёме сохранять проекцию находки (title/severity…), читать батчами, дать Finalize свой бюджет, сделать его возобновляемым.
- Эффект: Finalize не зависит от O(N) чтений артефактов.
- Трудоёмкость M, риск средний.
- Как закрепить: PostgreSQL-тест на 10 тыс. элементов и 2 тыс. находок с инжектированным бюджетом.
- Тип: дефект (SUSPECTED) / масштаб.

**F6. Закрыть лазейку SQL-gate.**
- Проблема: G8.
- Изменение: разворачивать `+`-цепочки (готовый код — `sqlconcat/main.go`).
- Эффект: 13 нарушений станут видны, 8 из них в узле.
- Трудоёмкость S, риск низкий.
- Тип: структура.

## G. Баги

**G1. Средний, VERIFIED. Workspace считает `excluded` выполненной проверкой без пробела.**
- Путь: отклонение элемента ставит `coverage_status='excluded'` и пробел `human-approval-rejected` (item_review_decision_sql.go:10-18). workspace_sql.go:14 засчитывает это в completedChecks, а :16 не считает пробелом. В UI «done» и 0 follow-ups (inbox/present.ts:170-172).
- При этом отчёт для тех же данных пишет `completed-with-gaps` (report.go:499-501). Спека требует «visible coverage gap» (review-and-reports.md:379, AUD-R178).
- Тест: этот случай не покрыт (service_postgres_test.go:180-183).

**G2. Низкий–средний, VERIFIED. Просроченные pending-запросы ревью считаются в «Decisions waiting» бессрочно.**
- workspace_sql.go:20 считает `state='pending'` без фильтра по `expires_at`.
- Finding-triage истекает только лениво: при новом ревью этой же находки или при попытке решения (finding_review.go:85, review_write.go:72-79). Предикат claim такие запросы не рассматривает (claims_sql.go:28-73), а терминальные Audit вообще не берутся в работу.
- UI показывает счётчик как есть (overview.tsx:195), хотя для списков вычисляет истечение на клиенте (decisions/model.ts:173-177).
- Это тот же класс, что V279-001, которое исправили только для report review. Задачи нет.

**G3. Низкий, VERIFIED по трассе.** Replay старта возвращает все элементы всех раундов: start.go:717 → `ListItems` с лимитом 100 001 → audit_handlers.go:300-303. Ответ при этом содержит текущий Round, а OpenAPI ограничивает items `maxItems: 10000` (:11956). Нарушение контракта наступает, когда сумма элементов больше 10 000; `maxItemsTotal` допускает до 100 000.

**G4. Расхождение VERIFIED, дефект SUSPECTED.** UI скрывает Delete для active, paused, waiting_review, finalizing и cancelling (controls.tsx:52-57). Сервер принимает удаление из любого состояния, кроме deleting (deletion.go:96-99). AUD-R222 говорит, что удаление остаётся доступным во время решения по отчёту.

**G5. Низкий, VERIFIED.** `validateActionDecision` принимает любой непустой UTF-8 ключ до 128 байт (action_review.go:218). SQL CHECK (000041:173-175) отвергает такой ключ ошибкой 23514, а `reviewWriteError` (review_write.go:102-107) отображает в ErrConflict только 23505. Для не-HTTP вызывающего это 500 вместо 400. Публичный путь защищён `requireIdempotencyKey` (audit_review_handlers.go:375).

**G6. Средний, SUSPECTED. Большой Audit может навсегда застрять в finalizing.** Работа Finalize описана в 3.3. Ошибка таймаута не классифицирована как постоянная (settlement.go:66-71), поэтому весь отчёт пересобирается каждый тик. Проверить можно только нагрузочным тестом.

**G7. Низкий, SUSPECTED.** `GetReconcileSnapshot` делает около 10 отдельных autocommit-чтений через pool (read.go:610-700; composition_audit.go:76). Это безопасно только при неявном инварианте: любая параллельная owner-мутация поднимает `audits.revision` до CAS контроллера.

**G8. Низкий, VERIFIED.** SQL-gate обходится конкатенацией. В transition.go:33 инлайн-statement на 37 строк, в deletion.go:29 и store.go:195 — по 19, есть ещё 5 таких мест в узле. Тест проходит (запускал).

## H. Упрощение

| Что | Действие | Экономия | Риск |
|---|---|---|---|
| Двойные позиционные списки колонок: store.go:70-97, read.go:18-22 + execution.go:62-70 (сейчас совпадают, проверял скриптом) | один `columns(prefix, list)` | ~40 строк, минус точка дрейфа (ср. c13dfbb9) | низкий |
| Тестовые `ListEvents` и `ListExecutions` (read.go:249-276, 445-490) | удалить, тесты перевести на прод-ридер | ~70 | низкий |
| Предикат видимости ×3 | SQL-функция или view | ~15 | низкий |
| 4 ограниченных JSON-обходчика | `strictjson.DecodeBounded(depth, nodes)` | 100–150 | средний: различается семантика чисел |
| Мёртвые константы: Retention* (types.go:84-89), ActiveCheck/ApplicabilityReviewKind; `ListParams.State` рядом с `States`; обёртка StableSubmissionID; литералы операций вместо констант types.go:306-309 | удалить, слить, заменить | ~25 | низкий |
| auditpriority (615 + 943 тест) | отложить до V64-006 — решение пользователя | 1558 | низкий |
| Advisory-локи ×3 перед `FOR UPDATE OF audit` | брать блокировку строки Audit до проверки replay и убрать advisory | ~15, минус три пространства локов | средний, нужен порядок audit→review→item (SUSPECTED) |

## I. Оптимизация

1. **Финализация отчёта** (report.go:209/221/397/406). При 10 тыс. элементов и 2 тыс. находок это 50 запросов покрытия, 2 тыс. чтений артефактов и около 200 тыс. строк в памяти за одну попытку, под 10 с, с повтором с нуля. Исправление — F5.
2. **Тик контроллера.** До 8 claim в секунду, на каждый около 12 запросов снапшота, даже когда Audit простаивает. Дальше `observeOneTerminal` делает до 32 `GetRun` по одному (controller.go:416), а dispatch — до 64 `NextItemAttempt` (dispatch_selection.go:38). Исправление: подтянуть состояние Run join-ом в `listExecutions`, получать попытки одним `GROUP BY item_id = ANY`, читать снапшот одной RR read-only транзакцией. Сейчас только polling (`PollInterval` 1 с), поэтому задержка растёт как O(active / ClaimBatch); нужен wake-хинт от scheduler.
3. **Workspace** (workspace_sql.go:10-20). Пять проходов по audit_items раунда (до 10 тыс.) и два по findings на каждый poll; UI опрашивает раз в 5 с (audits.ts:844-846). Исправление: два агрегата с `FILTER`. `ListEventsPage` делает `count(*)` по всем событиям на каждой странице (events.go:67), хотя при плотной последовательности (F2) Total равен ThroughSequence.
4. **Owner-wide findings** (owner_lists.go:68-76): гидрация по каждому Audit, около 4 запросов × до 201 Audit на страницу. Исправление: одна батч-гидрация по receipt_id с owner-проверкой.
5. **Start** (start.go:47-49, 109, 255). Одна REPEATABLE READ транзакция пишет до 4096 пакетов задач (до 64 МиБ, LIM-AUD-19/20) и держит Project `FOR SHARE` и credential guard; при serialization failure всё повторяется. Исправление: content-addressed артефакты писать до транзакции, в транзакции оставить только строки.

## Топ-3

1. **F2 и F1.** Единый аллокатор событий и один писатель на таблицу: 31 копия учёта revision и событий и шесть таблиц с несколькими писателями — источник класса V279, и SQL-связи, которых не видит archtest.
2. **G1 и G2.** Workspace показывает неправду о прогрессе и ожидающих решениях: excluded считается выполненным, просроченные pending-запросы считаются вечно. Исправлять одной доменной таблицей корзин и состояний (F3).
3. **F5 и G6.** Финализация отчёта без ограничений (O(находок) чтений артефактов под 10 с, повтор с нуля): нужен тест на реальном объёме и возобновляемый Finalize.

Ключевые файлы: /tmp/claude-1000/-home-ruslan-src-contractor/0fecd9f4-f328-4d49-ab01-b67e6eace15e/scratchpad/hub-audit-src/internal/auditservice/workspace_sql.go, …/internal/auditstore/{store.go,transition.go,deletion.go,validation.go,read.go,claims_sql.go}, …/internal/auditimport/report.go, …/internal/findingintake/direct_verification.go, …/internal/credentials/runtime_repository.go, …/internal/evalstore/submissions_sql.go, …/internal/persistence/sql_placement_test.go, …/ui/src/routes/projects/audits/controls.tsx.