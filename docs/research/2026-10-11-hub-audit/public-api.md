# Аудит узла «Публичный API» (main @ 02bbdcbf, 2026-10-11)

Анализ шёл только на чтение. Рабочее дерево `hub-audit-src` не изменено: `git status --porcelain` пуст. Пробные тесты запускались на копии Go-дерева в scratch (`…/scratchpad/hub-audit/public-api/probe`), все с `env -u CONTRACTOR_TEST_DATABASE_URL`.

## A. Карта узла

| Пакет / артефакт | non-test LOC (файлов) | Exported | Fan-in (prod) | Fan-out (internal) | Коммитов с 2026-08-12 |
|---|---|---|---|---|---|
| `internal/httpapi/public` | 10 380 (61) | 38 | 1: `internal/app` | 39 | 185 |
| `…/public/events` | 1 220 (5) | 18 | `app`, `public` (+`tests/e2e`) | 3 | 4 |
| `…/privateartifacts` | 686 (6) | 5 | `app` (+`auditimport` в тестах, e2e) | 11 | 29 |
| `…/artifacttransfer` | 172 (2) | 12 | `public`, `privateartifacts` | 1 | 3 |
| `…/httpx` | 94 (1) | 7 | `public`, `privateartifacts` | 0 | 1 |
| `internal/publicclient` (ручной код) | 603 (3) | 24 | `internal/cli` | 2 | 10 (40 вместе с generated) |
| `api/openapi/contractor-public-v1.yaml` | 14 840 строк: 112 path, 138 операций, 461 схема | — | генераторы TS и Go | — | 106 |
| `ui/src/api` (ручной код) | 9 487 (29), 438 строк `export` | — | 240 non-test UI-файлов; `query-keys.ts` ← 103 | — | 110 (154 вместе с generated) |

Узел затрагивают 275 из 2245 non-merge коммитов (12,2%). Медиана такого коммита — 18 файлов, среднее — 24,4. Fan-in у `public` равен 1, значит это не хаб по входящим связям. Это хаб по fan-out (39 пакетов) и по частоте правок контракта.

**Топ-10 файлов по churn:** OpenAPI 106; `ui/src/api/generated/public.ts` 92; `public/types.go` 73; `handler_test.go` 62; `openapi_contract_test.go` 49; `run_handlers.go` 47; `router.go` 43; `fakes_test.go` 41; `query-keys.ts` 36; `publicclient/generated/public.gen.go` 33.

**После разделения типов (aa95d592, 2026-10-04):** узел затронули 36 из 338 коммитов. Общие файлы `types*.go`, `router.go`, `errors.go` правились только в 6 из них. Разделение типов и маршрутов в основном сработало.

**Частые со-изменения вне узла:** `internal/app/app.go` 37; `runstore/store.go` 31; `ui/src/styles.css` 27; `runstore/types.go` 26; `scheduler_test.go` 26; `Makefile` 24. По каталогам: `ui/src/routes` 104, `docs/spec` 91, `internal/app` 54, `internal/persistence` 53, `tasks` 52, `internal/config` 50, `scheduler` 49, `runstore` 49, `auditservice` 41.

**Команды:**
- `go list -f '{{.ImportPath}} {{join .Imports " "}}' ./...`, а также с `-tags 'e2e integration'` и `TestImports` — fan-in/fan-out;
- `git log --no-merges --since=2026-08-12 --name-only` плюс скрипт на Python — churn и со-изменения;
- AST-счётчик экспортов на `go/parser` (scratch);
- `go tool cover -func` по тестам `verify-public-api`;
- инструментированная копия `serveAndValidatePublicContract`, которая пишет `operationId` каждой провалидированной операции.

## B. Обязанности

1. **Транспорт и middleware.** `router.go`, `authorization.go`, `auth_handlers.go`, `error_diagnostics.go`, `request.go`, `httpx/request.go`.
2. **Ручные типы контракта.** `types.go`, `types_run_project.go`, `types_configuration.go`, `types_operations.go`, `audit_types.go`, `eval_types.go`: 109 структур `*Request`/`*Response`, 537 json-тегов. Ещё 124 поля в них имеют тип из доменных пакетов (15 пакетов, например `auditservice.Finding`, `planner.PlannerPlanProjection`, `telemetry.Summary`). Часть ответов сериализует доменный тип напрямую: `run_handlers.go:411` (`runstore.ResumeRunResult`), `audit_review_handlers.go:135` (`auditservice.Finding`), `credential_handlers.go:63` (`credentials.Record`).
3. **Отображение ошибок.** `errors.go:27-172` (`handleError`) импортирует sentinel-ошибки из 15 пакетов. Рядом есть локальные мапперы: `gitImportError` (`git_import_handlers.go:84`), `handleArchiveError` (`artifact_archive_handlers.go:167`), `evalError` (`eval_helpers.go:21`). Отдельные хендлеры переопределяют результат до вызова `handleError` (`runtime_configuration_handlers.go:90,200`).
4. **Пагинация.** `pagination.go` (HMAC-курсоры). Соглашение «limit+1» живёт в 31 месте вызова.
5. **Дайджесты идемпотентности** — доменное правило, которое живёт в edge: `run_handlers.go:281-323`, `project_handlers.go:248`, `audit_models.go:139-161`, `audit_review_handlers.go:386`.
6. **Оркестрация.** `getRun` собирает ответ из 14 чтений (`run_handlers.go:438-535`, `run_detail_batch.go:22-60`). `cancelRun` выбирает notifier через type assertion (`run_handlers.go:363-369`).
7. **Live-события** (`events/*`) — отдельная подсистема со своим жизненным циклом и поллингом.
8. **Слоты и дедлайны передачи артефактов, превью архивов** (`artifacttransfer`, `artifact_archive_handlers.go`).

**God-типы.** Один `handler` с 210 методами, `Dependencies` с 41 полем (`types.go:216`), `AuditManagement` с 30 методами (`types.go:177`), `EvalManagement` с 21 методом (`eval_types.go:10`). Самый большой файл — `run_handlers.go`, 819 строк.

**Что должно жить в другом месте.**
- Дайджесты идемпотентности и чтение деталей Run — в `runservice`, как у Audit в `auditservice`.
- Классификацию ошибок лучше делать в домене через интерфейс «вид ошибки». Сейчас сосуществуют три механизма: sentinel-switch, `auditdomain.ErrorCode` и `evaldomain.Error`.

## C. Связи, которые не видит archtest

1. **Хендлеры ходят в хранилища напрямую, и это непоследовательно.** `Runs`, `RunQueue` и `RunLifecycle` — это `runstore.PostgresStore` (`app/composition_http.go:78`). `Projects` — `projectstore` (`:84`), `Metrics` и `AllocationResources` — репозиторий telemetry (`:90`). Audit и Eval при этом идут через сервисы. Мутации Run (`RequestRunCancellation`, `ResumeFailedRun`) хендлер вызывает прямо на хранилище, а мутации Audit — на сервисе. Сам `auditservice` местами лишь пробрасывает вызов: `service.go:159-163` (`ListItems` → `auditstore.NewPostgresStore(s.pool).ListItemsPage`).
2. **Один sentinel несёт разный смысл в разных контекстах.** Централизованно `runtimeconfig.ErrNotFound` даёт 400 `runtime_label_unknown`, а в GET версий и привязок его локально переопределяют на 404. В production-коде узла SQL нет. Слой хранения протекает только через ошибки: `persistencepostgres.IsTransactionConflict` и SQLSTATE в `error_diagnostics.go:44-58`.
3. **JSON-формы контракта задаются в ~15 доменных пакетах.** Переименование json-тега, например в `runstore`, меняет публичный контракт. Поймать это можно, только если операция проходит валидацию схемы (см. D-3.5).
4. **Дубли.** Таблица маршрутов существует в трёх копиях: пути OpenAPI; 128 регистраций хендлеров плюс 78 строк `methodNotAllowed` в `routes_*.go`; 138 строк `wantImplemented` (`openapi_contract_test.go:106-247`). Тест сравнивает OpenAPI со списком, но не с роутером. Грамматика ResourceId скопирована 20 раз (8 в Go, 12 в TS), сейчас все копии совпадают.
5. **Широкие интерфейсы.** Каждый из 30 методов `AuditManagement` вызывается ровно в одном месте. Интерфейс просто зеркалит список audit-маршрутов, поэтому каждая новая audit-операция правит и интерфейс, и `fakeAuditManagement`. Поле `Audits` используют 9 файлов, включая `eval_handlers.go`; `Projects` — 6 файлов, включая `run_handlers.go` и `audit_handlers.go`.
6. **Экспорты без внешних production-вызовов.**
   - `events.StreamRun`, `events.StreamOperations`, `events.Stream`;
   - `artifacttransfer.MinimumThroughput`, `artifacttransfer.StorageBudget`, `artifacttransfer.TransferGrace` (последний нужен только тесту `public`);
   - `httpx.ErrUnquotedETag`;
   - `privateartifacts.RuntimeInstanceHeader` (только тесты и e2e).

## D. Классы дефектов из AGENTS.md §3

**3.1. Одно правило в нескольких местах.**

| Правило | Где определено |
|---|---|
| Максимальный размер страницы | OpenAPI `Limit` max 200 (`yaml:6460-6464`); `maxPageLimit=200` (`pagination.go:15`); CLI `validatedLimit` 200 (`cli/project_commands.go:239`); UI 100 (`ui/src/api/projects.ts:338`, `audits.ts:407`); хранилища — не меньше 12 разных проверок: `projectstore` 200, `auditstore` 200 и 200+1, `runstore` 201, `artifacts` 201 (×3), `credentials` 201/200, `runtimeconfig` 200/201, `findingintake` 200, `telemetry` 101, `evaldomain` 100. Расхождения порождают дефект G1. |
| Класс повторяемости | Флаг `retryable` для каждого кода на сервере против `transientStatus` в Go-клиенте (`publicclient/retry.go:124-127,141-146`) → G5 |
| Коды ошибок | 67 литералов в handler-коде; поле `code` в OpenAPI — свободный `ConfigId` (`yaml:12513`). 50 из 67 кодов не упомянуты ни в одном документе `docs/spec`, хотя `errors.md` объявлен реестром кодов: среди них `invalid_request`, `conflict`, `precondition_failed`, `storage_transaction_conflict`. UI сравнивает литералы (`"not_found"` встречается 15 раз). VERIFIED. |

**3.2. Исходы мутаций.**
- Конфликт ключа идемпотентности помечен как повторяемый — см. G3.
- Приватный API отдаёт конфликт транзакции как 500 `internal_error` с `retryable:true` (`privateartifacts/errors.go:61-62`), публичный — как 503 `storage_transaction_conflict`. Это две разные политики для одного исхода.
- Retry-After на повторяемых 503 не ставится, но спека его и не требует.

**3.3. Неограниченная работа.** `getRun` делает 14 последовательных запросов (I2). Поллинг событий (I1). Owner-список находок гидратирует по аудитам в цикле (I3). Цикл `runArtifactsByNamespace` (`run_handlers.go:774-797`) не ограничен по числу итераций. SUSPECTED, что фактически его ограничивает число слотов Workflow.

**3.4. Жизненный цикл.** Новых пробелов в узле не нашёл: удаление и отмена делегированы сервисам, ошибки `*_deleting` дают 409 `project_deleting`.

**3.5. Тесты, которые не могут упасть.** Всё VERIFIED.
- Фейки игнорируют `Limit`: `runtime_configuration_fakes_test.go:187-205`, `audit_handlers_test.go:598-608` и `:710-728`. Поэтому G1 не ловится.
- Без БД валидацию схемы проходят 75 из 138 операций. DB-тесты добавляют все Eval-операции и ещё 8 audit-операций. 33 не-Eval операции не валидируются нигде, в том числе create/start/pause/resume/cancel/delete/get Audit, `listAuditItems`, `listAuditCoverage`, finding-proposals, `resumeRun`, `deleteProject`, queue control.
- Тест конфликта идемпотентности проверяет только 409 (`handler_test.go:1316-1322`), но не `retryable`.
- Свежесть Go-клиента ничем не проверяется (G2).

**3.7. Лишнее.**
- 13 «опциональных» зависимостей с ~26 проверками на nil, хотя production всегда передаёт все поля (`composition_http.go:72-102`).
- Мёртвая ветка `project_handlers.go:195`: `RuntimeCredentials` уже обязателен в `router.go:33`.
- Экспорты из C6.

## E. Цена изменений

1. **cc167827 `feat: replace cross-project UI fan-out with owner-wide APIs` (2026-10-07).** 49 файлов. Затронуты слои kernel (миграция 000097), product (`auditstore`, `auditservice`), edge (`httpapi/public`), контракт (OpenAPI +94) и UI (api, routes, e2e, generated +166). Правки, которые навязала структура узла:
   - +2 метода в `AuditManagement` (`types.go`) и +2 в `fakeAuditManagement`;
   - `routes_audit.go`: 3 маршрута и 3 строки `methodNotAllowed`;
   - +3 строки в `wantImplemented`;
   - перегенерация `public.ts`;
   - +19 строк в `query-keys.ts`;
   - регенерация Go-клиента была обязательной, но её пропустили (отсюда G2).

   Итого около 8 точек и ~40 строк учёта на каждую новую операцию.
2. **99e91963 `feat: expose durable check event history` (2026-10-07).** 28 файлов. Тот же шаблон: `types.go` +1 метод, фейк +6 строк, маршруты +2, `wantImplemented` +1, `query-keys` +2, `public.ts` +99. Go-клиент снова не перегенерирован.
3. **592271f5 `fix(api): allow the cursor sentinel on full audit pages` (2026-10-07).** 2 файла. Исправлен только `auditstore.validateList` (`validation.go:142-144`), то есть один GET `/v1/audits`. Однотипные места в том же файле (`:178`) и в `read.go:453,501` остались — это G1. Цену создаёт структура: соглашение limit+1 живёт в edge (31 вызов), а максимум объявляет каждое хранилище само. Каждое место приходится находить вручную; так раньше исправили 3 списка в V59-001 (`tasks/done/v59-001-audit-report-and-pagination.yml:31`).

Попутно 8d385042 показывает цепочку из четырёх звеньев для Eval: `managed.schema.json` → `sync-eval-openapi.py` → OpenAPI → TS. Go-клиент в этой цепочке тоже пропущен.

## F. Рекомендации

1. **Единый контракт страницы.** Тип: дефект. Трудоёмкость S, риск низкий.
   - Проблема: G1 — 8 операций отвечают 400 на документированный `limit=200`.
   - Изменение: в нижнем слое объявить `PublicPageMax=200` и `PageFetchMax=201`; все хранилища за публичными списками проверяют `Limit <= PageFetchMax`. Альтернатива — хендлер передаёт тип `PageRequest`, а не голый `int`.
   - Эффект: 0 из 12 мест с расхождением.
   - Как закрепить: DB-тест, который проходит все операции с параметром `#/components/parameters/Limit` и вызывает их с `limit=200` на реальном хранилище. Можно расширить `TestPublicAuditPaginationBoundary`.
2. **Проверка свежести Go-клиента.** Тип: дефект. Трудоёмкость S, риск низкий.
   - Изменение: цель `generate-public-client-check` (перегенерировать во временный файл и сравнить) в `make lint`. `oapi-codegen` закрепить через `tool`-директиву в `go.mod`, чтобы генерация работала офлайн. Сразу перегенерировать клиент.
   - Эффект: дрейф 0 вместо +1805 строк.
3. **Одна таблица маршрутов и покрытие операций.** Тип: структура. Трудоёмкость M, риск низкий.
   - Изменение: Go-таблица `{method, path, operationId, handler}`, из которой регистрируются маршруты и автоматически ответ 405. Тест в обе стороны: таблица совпадает с операциями OpenAPI. Удалить `wantImplemented` и `x-contractor-implementation`: все 138 операций помечены `implemented`, `planned` — ни одной. В `serveAndValidatePublicContract` считать операции и падать, если какая-то не провалидирована ни разу.
   - Эффект: точек правки на новую операцию становится 5 вместо 8; минус ~300 строк; 33 непроверяемые операции станут видны.
4. **Классификация ошибок в домене и реестр кодов.** Тип: дефект и структура. Трудоёмкость M, риск средний (меняется семантика `retryable` для клиентов).
   - Изменение: отделить sentinel «конфликт идемпотентности» от «оптимистического конфликта» в `runstore`, `projectstore` и `auditstore`; новый код `idempotency_conflict` с `retryable:false`, как требует FAIL-RUN-1. Ввести интерфейс `Kind()` (код, статус, повторяемость) вместо switch по 15 пакетам. Таблицу кодов сделать источником для `errors.md`. Go-клиент должен уважать `retryable:false` на 5xx.
   - Как закрепить: тест, что каждый код из таблицы есть в реестре с тем же флагом.
5. **Строгий JSON.** Тип: дефект. Трудоёмкость S, риск низкий (pre-prod).
   - Изменение: `decodeJSON` и `decodeClosedJSON` переводятся на `strictjson` плюс `RejectDuplicateKeys` и проверку точного регистра ключей.
   - Как закрепить: тест на алиас регистра и дубликат ключа для одного REST-тела и одного кадра WebSocket.
6. **Контекстные хендлеры.** Тип: структура. Трудоёмкость M–L, риск средний.
   - Изменение: заменить один `handler` на структуры по контекстам (`runHandlers{Runs, RunQueue, …}`, `auditHandlers{Audits, Projects}`), каждая со своим узким `Dependencies`. Все зависимости сделать обязательными. Чтение деталей Run и дайджесты перенести в `runservice`.
   - Эффект: новая зависимость audit затрагивает только audit-файлы; минус ~26 nil-веток; закрывает G4 и I2.
   - Как закрепить: archtest-подобный тест, что файлы контекста не обращаются к чужим полям.

## G. Баги

**G1. `limit=200` → 400 на 8 операциях.** Серьёзность medium. VERIFIED.

Хендлер принимает 200 (`pagination.go:55`) и передаёт `limit+1 = 201` в хранилище, которое разрешает не больше 200:

| Операция | Хендлер | Проверка в хранилище |
|---|---|---|
| GET `/v1/projects` | `project_handlers.go:89` | `projectstore/store.go:328` |
| GET `/v1/audits/{id}/items` | `audit_handlers.go:431` | `auditstore/validation.go:178` |
| GET `/v1/audits/{id}/coverage` | `audit_handlers.go:527` | `auditstore/read.go:501` |
| GET `/v1/runs/{id}/finding-proposals` и `/v1/audits/{id}/finding-proposals` | `finding_handlers.go:92` | `findingintake/postgres.go:364` |
| GET `/v1/operations/runtime-configs` | `runtime_configuration_handlers.go:55` | `runtimeconfig/repository.go:152` |
| GET `/v1/operations/runtime-labels` | `runtime_configuration_handlers.go:165` | `repository.go:271` |
| GET `/v1/operations/runtime-credentials` | `runtime_configuration_handlers.go:348` | `credentials/runtime_repository.go:106` |

- Ошибка отображается в 400 (`errors.go:152-165` и `:74`).
- Пробный тест подтвердил отказ всех 8 вызовов хранилищ на значении 201.
- CLI достаёт до бага: `contractor project list --limit 200`. UI не достаёт, у него потолок 50 или 100.
- Тестов нет.
- Источники (`introduced_by`): 10f017c6 (Projects), c7632c22 (Audits), 610cab5f (intake). В каждом случае хендлер и хранилище писались в одном коммите.
- Открытой задачи нет. Частично исправлялось в V59-001 и 592271f5.

**G2. Go-клиент устарел, проверки нет.** Серьёзность medium. VERIFIED.
- Перегенерация закреплённым `oapi-codegen` v2.8.0 из закоммиченного YAML даёт +1805 и −18 строк. Отсутствуют методы `ListAuditEvents`, `ListOwnerAudits`, `ListOwnerFindings`, `ListOwnerReviews`; у типов Eval-селекторов есть расхождения.
- Последняя регенерация — be4b7338 (2026-10-04).
- `make/dev.mk:27-28` только генерирует; аналог `ui-generate-check` (`make/ui.mk:16-17`) для Go отсутствует.
- Задачи нет.

**G3. Конфликт ключа идемпотентности отдаётся как повторяемый.** Серьёзность low–medium. VERIFIED.
- Хранилища возвращают `ErrConflict` при повторе ключа с другим дайджестом: `runstore/store.go:284-285` и `:325-326`, `projectstore/store.go:61-68`.
- `errors.go:145-149` превращает это в 409 `conflict` с `retryable:true` и текстом «retry with the current revision».
- Спека говорит обратное: FAIL-RUN-1 (`failure-model.md:461`) называет этот исход permanent.
- Go-клиент повторяет такой запрос ещё дважды (`retry.go:104-121`).
- Тест не проверяет флаг. Источник — 70010625. Задачи нет.

**G4. JSON принимает алиасы регистра и дублирующиеся ключи.** Серьёзность low. VERIFIED пробой.
- `decodeJSON` (`request.go:37-40`) использует `encoding/json`. `{"Reason":"x"}` проходит; `{"reason":"first","reason":"second"}` даёт `"second"`; в `createProjectRequest` пара `name`+`NAME` даёт `"y"`.
- То же для кадров WebSocket (`events/protocol.go:159-163`): `Type` и `SubscriptionID` принимаются, дубликат `subscriptionId` берёт последнее значение.
- Это нарушает AGENTS §6, точные имена свойств OpenAPI при `additionalProperties:false` и API-R43 («rejected rather than guessed»).
- Задачи нет.

**G5. Курсор в деталях Run читается после состояния.** Серьёзность medium при срабатывании, вероятность низкая. Порядок чтений VERIFIED, последствие SUSPECTED.
- Run читается в `run_handlers.go:447`, `GetRunEventCursor` — в `:484`, после пяти других чтений, и не в одном снимке.
- UI подписывается после курсора (`ui/src/routes/runs/live.ts:231`), а детали опрашивает только во время recovery (`detail.tsx:1067-1088`).
- Если терминальный переход закоммитится в этом окне, UI покажет «Running» до перезагрузки страницы. Это противоречит API-R48 («corresponding event cursor»).
- Подтвердить можно тестом с перемежением на фейках.

**G6. Go-клиент повторяет 5xx вопреки `retryable:false`.** Серьёзность low. VERIFIED.
- `transientStatus` (`retry.go:124-127`) считает повторяемыми любые 502, 503 и 504.
- Сервер помечает `retryable:false` коды `artifact_content_corrupt`/`missing` (`errors.go:139-142`) и `git_import_capacity`/`unavailable`/`timeout` (`git_import_handlers.go:102-104`).
- `IsTransient` (`retry.go:141-146`) продолжает опрос при постоянной порче содержимого.

## H. Упрощение

1. **Удалить `wantImplemented` и `x-contractor-implementation`.** Это `openapi_contract_test.go:106-247` и 138 строк YAML. Вместо них — одна таблица маршрутов с автоматическим 405 (78 строк в `routes_*.go`). Экономия ~300 строк, риск низкий.
2. **Сделать все зависимости обязательными.** Удалить ~26 nil-веток, fallback для `ConfigurationPublisher` в `router.go:24-28` (production передаёт его явно, `composition_http.go:76`), ветку `Wake()` в `run_handlers.go:366-368` (scheduler реализует `Cancel`, `scheduler.go:144`) и `project_handlers.go:195`. Экономия ~80 строк, риск низкий: правятся только фикстуры.
3. **`errors.go`.** Пять одинаковых веток `ErrPrecondition` (`:36-39`, `:50-55`) свести в одну. Вместе с F4 switch по 15 пакетам уходит. Экономия ~40–100 строк.
4. **Лишние проверки capability.** `requireOperationsCapability` в `performance_handlers.go` (3 вызова) и `scheduler_settings_handlers.go` (2) повторяет middleware (`authorization.go:10-19`): все эти пути лежат под `/v1/operations/`. Экономия ~10 строк.
5. **Ручная пагинация.** `performance_handlers.go:88-106` повторяет `paginate` (`pagination.go:73-90`). Экономия ~10 строк.
6. **Legacy-ветка дайджеста.** `run_handlers.go:297-310`: «Preserve the pre-Runtime-label digest … before migration 000018». Это нарушает правило «no backward compat»; удалить условное включение. Экономия ~6 строк.

## I. Оптимизация

1. **Поллинг событий.**
   - Путь: каждая run-подписка раз в секунду вызывает `ListRunEvents` (`subscriptions.go:79-91`, `CatchUpInterval` 1 с) даже при живом NOTIFY (API-R55).
   - Масштаб: до 8 сокетов × 32 подписки (`hub.go:24`, `protocol.go:24`), то есть до 256 запросов в секунду на сессию в простое.
   - Исправление: один поллер на runID на уровне hub (`runWatchers` уже сгруппирован по runID, `hub.go:328-360`) и редкий страховочный catch-up, например раз в 15 с, пока LISTEN здоров.
2. **`getRun` — 14 последовательных round-trip'ов.**
   - Путь: `run_handlers.go:447-509` и `run_detail_batch.go:41-60`, на каждое lifecycle-событие (`live.ts:255-256` инвалидирует `runs.all`).
   - Исправление: одна read-only RR-транзакция в `runservice`. Ожидаемо ~15–70 мс вместо этого → 1–3 запроса, и заодно закрывается G5.
3. **`/v1/findings` (owner).**
   - Путь: `auditservice/owner_lists.go:70-78` гидратирует по каждому аудиту страницы отдельно: receipts + exact batch, decisions, assessments (`finding_read.go:125-140`).
   - Масштаб: страница в 201 строку с A аудитами → примерно 3–4·A запросов. Inbox вызывает список с `limit=200` (`ui/src/api/audits.ts:143`).
   - Исправление: батч по парам `(audit, receipt)` сразу на всю страницу. Форма цикла VERIFIED, масштаб SUSPECTED.
4. **Coverage без закрепления ревизии.**
   - Путь: UI читает `getAudit` до и после каждого батча и может повторить (`coverage-data.ts:60-79`). Сервер умеет `auditRevision` для findings и reviews (`audit_review_handlers.go:48,179`), но не для coverage (`audit_handlers.go:492`).
   - Исправление: аддитивный параметр `auditRevision` для coverage.

SQL в узле нет, поэтому индексы не оценивались.

**`query-keys.ts`.** Это реестр, в который в основном дописывают: 36 коммитов, +627/−41 строк, 103 non-test импортёра. Нагрузку несёт иерархия префиксов, на которой построена инвалидация, а тестов, которые фиксируют вложенность префиксов, нет. По сути это churn с узким радиусом последствий, а не структурный хаб.

## Топ-3

1. **G1 / F1.** 8 публичных операций отвечают 400 на документированный `limit=200` (VERIFIED). Исправление S: единая константа «limit+1» и DB-тест на максимум для всех пагинируемых операций.
2. **G2 / F2.** Go-клиент отстаёт от OpenAPI с 2026-10-04 (+1805 строк, 4 операции), и ни один гейт этого не ловит. Исправление S: generate-check в `make lint`.
3. **F3.** Проверка контракта выборочная: 33 не-Eval операции не валидируются ни одним тестом, таблица маршрутов существует в трёх копиях. Нужна одна таблица маршрутов и гейт покрытия операций. Именно этот разрыв дал G1 и G4.
