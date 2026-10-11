# Аудит узла «Артефакты» (platform). Worktree main @ 02bbdcbf, 2026-10-11

Анализ только для чтения: в репозитории ничего не менялось, к БД не подключался. Вспомогательные скрипты (подсчёт экспортов через go/ast и копия проверки размещения SQL) лежат в `…/scratchpad/hub-audit/artifacts/`.

**Команды.**
- LOC: `ls $d/*.go | grep -v _test.go | xargs cat | wc -l`.
- Экспорты: скрипт на go/ast по не-тестовым файлам.
- Fan-in и fan-out: `go list -f '{{.ImportPath}} {{join .Imports " "}}' ./...`, потом фильтр по пакету.
- Churn: `git log --no-merges --since=2026-08-12 --format=%H -- <pkg>`.
- Топ файлов: `--name-only | sort | uniq -c`.
- Соизменения: для каждого коммита узла `git show --name-only`, потом исключение путей узла и `tasks/`.
- Использование API: `rg -o 'artifacts\.[A-Z]\w*'` по каждому импортёру.

## A. Карта узла

| Пакет / модуль | LOC (без тестов) | Экспортов | Fan-in (без тестов) | Fan-out (внутр.) | Коммитов с 08-12 (без тестов) |
|---|---|---|---|---|---|
| internal/artifacts | 4033 | 133 (28 типов, 14 функций, 61 метод, 12 const, 18 var) | **23** | artifactpolicy, clone, contracts, persistence/postgres, randomid и pgx | 51 (42) |
| internal/artifactpolicy (foundation) | 48 | 12 | 8 | — | 6 |
| internal/artifactpreview | 228 | 18 | 1 (httpapi/public) | zipdirectory | 2 |
| internal/httpapi/artifacttransfer | 172 | 12 | 2 (privateartifacts, public) | artifacts | 3 |
| internal/httpapi/privateartifacts (edge) | 686 | 5 | 1 (app) | 11 пакетов, включая controlplane, findingintake, gatewayrecovery | 29 (17) |
| Python: `artifacts.py`, `contracts/artifacts.py`, `run_artifacts`, `text_artifacts`, `common/artifact_*` | 1428 | — | `contractor_runtime.artifacts` импортируют 27 модулей | — | artifacts.py 22, text_artifacts 10, run_artifacts 8 |

Итого по узлу: Go — 5167 LOC и 74 коммита; вместе с Python — 95 коммитов.

**Сверка с прошлым обзором.** Было «~4k строк, ~22 импортёра, ~42 коммита». Сейчас:
- 4033 LOC только в internal/artifacts;
- 23 импортёра без тестов, ещё 2 только из тестов (planner/streamline, tests/eval/project_workflows);
- 51 коммит, из них 42 по не-тестовым файлам.

**Импортёры artifacts:** agentskills, app, auditcontroller, auditimport, auditscan, auditservice, auditstandards, auditstore, cli, evalservice, findingintake, gitimport, httpapi/artifacttransfer, httpapi/privateartifacts, httpapi/public, memory, planner, planner/scan, projectlifecycle, rundeletion, runservice, runstore, scheduler.

**Что они используют:**
- 18 пакетов ходят в store через Service и ScopedStore.
- 2 пакета (projectlifecycle, rundeletion) используют только `NewPostgresPurger`.
- 2 пакета берут только типы и константы: cli (`MaxPayloadSize`) и auditscan (`ErrArtifactNotFound`, `Payload`).
- 1 пакет (artifacttransfer) использует `AcquireTransfer`.

**Ответ на вопрос о маленьком kernel-пакете с типами.** Он сократит fan-in только на 2–3 импортёра, около 10%. Fan-in определяется операциями над store. Реальная польза kernel-типов в другом: свернуть дубли дескриптора exact-артефакта (раздел C2) и вынести лимиты (раздел D/3.1).

**Топ-10 не-тестовых файлов по числу коммитов:**

| Файл | Коммитов |
|---|---|
| runtime/…/artifacts.py | 22 |
| artifacts/types.go | 16 |
| artifacts/service.go | 14 |
| artifacts/postgres_write.go | 12 |
| privateartifacts/artifact_handlers.go | 11 |
| text_artifacts/tools.py | 10 |
| artifacts/postgres_fork.go | 10 |
| run_artifacts/tools.py | 8 |
| privateartifacts/handler.go | 8 |
| artifacts/postgres_purge.go | 8 |

**Частые соизменения вне узла:**

| Файл | Совместных коммитов |
|---|---|
| api/openapi/contractor-public-v1.yaml | 24 |
| httpapi/public/types.go | 20 |
| ui/src/api/generated/public.ts | 19 |
| httpapi/public/fakes_test.go | 18 |
| public/handler_test.go | 16 |
| public/run_handlers.go | 15 |
| public/router.go | 15 |
| Makefile | 14 |
| openapi_contract_test.go | 14 |
| app/app.go | 14 |
| runstore/store.go | 13 |

## B. Обязанности

1. **Физический store и blob-бэкенды.** Файлы: blob*.go (PostgreSQL inline и filesystem), dedup, офлайн-очистка, transfer-слоты через context. Связность хорошая.
2. **Репозиторий bindings, revisions и lineage.** Файлы: postgres_write/read/query/fork/read_batch/prefix_*.go.
3. **Purge для Run, Project и Audit.** Файл: postgres_purge.go.
4. **Политики конкретных потребителей внутри platform.** `Service` (service.go, 625 строк) — god-тип. Кроме генерических User/Project/Run у него 14 доверенных операций. Каждая знает namespace и политику одного product-контекста. Из 14 операций 12 вызываются ровно из одного пакета:

   | Операция | Единственный потребитель |
   |---|---|
   | WriteRunRepeatRequest | runservice |
   | ReadRunRepeatRequest | public |
   | WriteFindingProposal | findingintake |
   | WriteAuditStandardPackage | auditstandards |
   | ForkSkill | agentskills |
   | ForkInput, ForkProjectInput | runservice |
   | BindOutputExact, PublishRunOutput, FreezeRunOutputs | scheduler |
   | ImportAuditArtifact | auditimport |
   | ImportFindingArtifact | findingintake |

   Каждая новая цель добавляет в узел метод, опциональный интерфейс, SQL и тест. Отсюда и churn.
5. **Python `artifacts.py` (577 строк).** Это не только клиент артефактов. Там же:
   - отправка finding proposal (artifacts.py:127-201);
   - фабрика GatewayRecoveryClient (:116-120);
   - журнал наблюдений exact-ref (:203-226).

   7 из 22 коммитов в нём пришли из findings, gateway recovery и memory.
6. **`privateartifacts` — это весь приватный API аллокации.** handler.go:63-67 обслуживает также `gateway-recovery` и `finding-proposals`. Имя пакета вводит в заблуждение. Fan-out 11, включая findingintake и gatewayrecovery.

**Что вынести:** доверенные операции — к потребителям (поверх общего create-only/fork примитива); finding и gateway-клиенты — в отдельный модуль приватного API.

## C. Связи, которые не видит archtest

**C1. SQL по чужим таблицам (VERIFIED).**

| Где | Что делает | С какой таблицей |
|---|---|---|
| artifacts/postgres_fork_sql.go:25-29 | читает | `workflow_runs` |
| artifacts/postgres_fork.go:164 | читает | `workflow_runs` |
| artifacts/postgres_project_output_sql.go:25 | читает | `workflow_runs` |
| artifacts/postgres_audit_import_sql.go:16 | читает | `workflow_runs` |
| artifacts/postgres_purge.go:119, :174 | **удаляет** строки | `workflow_run_output_publications` (таблица runstore, migration 000028) |
| **runstore/resume_store.go:137-138** | `UPDATE artifact_bindings SET frozen=false` — размораживает outputs | таблица артефактов |
| scheduler/postgres_sql.go:27-38 | копия запроса Metadata | таблицы артефактов |
| evalstore/evidence_sql.go:13-21, evidence.go:56-58 | копия запроса Metadata | таблицы артефактов |

Про строку runstore: это зеркальная операция к `FreezeOutputs` (postgres_fork.go:152). Конечный автомат флага `frozen` разнесён по двум слоям и триггеру из 000052.

Ещё 30 тестовых файлов вне узла пишут SQL прямо по `artifact_*`. Поэтому, например, e415ae38 правил persistence/postgres и public тесты.

**C2. Дубли JSON-формы «exact-артефакт» (ref, digest, mediaType, size) (VERIFIED).**
- auditstore/types.go:368 — mediaType и size с `omitempty`;
- findingintake/types.go:48 — поля обязательны;
- auditstandards/types.go:135 — ключ `artifact` вместо `ref`;
- auditdomain/types.go:233 — без size;
- `artifacts.Metadata` — Digest помечен `json:"-"`;
- OpenAPI `AuditExactArtifact` (11449) и `FindingExactArtifact` (11965).

Проверка тоже разошлась. auditcontroller/artifacts.go:28-55 сверяет SameExact, digest, mediaType и size. auditimport/artifacts.go:43-45 пропускает проверку mediaType и size, если `expected.MediaType==""`.

**C3. Обходы Service и адаптеры у потребителей.**
- 33 вызова `artifacts.NewPostgresRepository(` вне пакета; 27 из них в форме `NewService(NewPostgresRepository(tx))`, в 17 файлах.
- Методы конкретного репозитория вызываются напрямую:
  - `ReadExactBatch` — auditservice/coverage.go:53 и findingintake/audit_receipt_batch.go:198;
  - `RecordGitSource` — gitimport/importer.go:148;
  - `Read` — evalservice/native_check_execution.go:144.
- Пять адаптеров-обёрток над `*artifacts.Service` с пересекающимися операциями resolve, read-verified и create-immutable:
  - scheduler/artifacts.go;
  - planner/scan/artifacts.go;
  - auditcontroller/artifacts.go;
  - auditimport/artifacts.go;
  - auditstore/immutable_artifact.go.

**C4. Широкие интерфейсы при узком использовании.**
- `Repository` (7 методов) и `QueryRepository` (4) — единственный продакшен-реализатор `PostgresRepository`.
- Рядом ещё 7 опциональных интерфейсов с проверкой типа во время выполнения и ответом `ErrQueryUnsupported`: service.go:182, 478-488; prefix_list.go:9; prefix_read_batch.go:11.
- Зачем они нужны: ради 5 Go-фейков (artifacts/service_test.go:222, agentskills/catalog_test.go:408, auditstandards/catalog_test.go:405, public/fakes_test.go:368, privateartifacts/handler_test.go:844).

**C5. Экспорты только для тестов (VERIFIED по rg).**
- `Scope.ID`, `Scope.Kind` — нужны лишь фейкам репозитория.
- `ErrQueryUnsupported` — ни одной внешней продакшен-ссылки.
- `LineageInputFork`, `LineageOutputBind`, `LineageProjectOutputPublish`, `MaxBindingListLimit` — используются только внутри пакета.

## D. Классы дефектов AGENTS §3

**3.1 — одно правило в нескольких местах.**

| Правило | Копии | Что держит копии в синхроне |
|---|---|---|
| 64 MiB (LIM-ART-4) | types.go:11; CHECK в migrations/000048:14; Python artifacts.py:28; UI artifacts.ts:5 **и** audit-composition.ts:12; 12 литералов в OpenAPI (3459, 3469, 5244, 5254, 6576, 6580, 10922, 11457, 11919, 11973, 12643, 12688) | ничего. VERIFIED: ни один тест не сравнивает две копии, `x-max-request-body-bytes` никто не читает |
| Лимит prefix-list 256 (LIM-ART-3) | prefix_list.go:7; artifacts.py:32 | ничего |
| Грамматика revision | store types.go:189 (не пусто, без NUL, без ограничения длины); contracts/common.go:138 (не пусто); public request.go:25 (`^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$`, но только для `?revision`, не для If-Match в :130); OpenAPI `Revision` 12494; UI artifacts.ts:12; Python contracts/artifacts.py:37 и artifacts.py:339. Кодирование ETag: Go `strconv.Quote/Unquote` (httpx/request.go:75-94) против Python `json.dumps/loads` (artifacts.py:341, 541) | только связка UI↔OpenAPI (openapi-conformance.test.ts:109). Общей fixture нет |
| Зарезервированные namespace и префиксы | Go artifactpolicy/visibility.go:28-48 (5 namespace); contracts/worker_completion.go:237-244; service.go:141 (литералы); postgres_write.go:35; литерал `"audit-"` в service.go:548, 590, postgres_audit_import.go:32, postgres_audit_write.go:25, postgres_purge.go:29, auditstandards/validate.go:270; `"tool-invocation."` в config/scan_plan.go:153 и execution_config.go:542. Python: artifact_visibility.py:9-11 (**3** namespace), contracts/worker.py:96-98, worker/tool_runtime.py:43 и 206, memory/codec.py:17 | ничего. Префикс `http.body.` есть только в Python |
| Имена и media type | общие `artifact-name-cases.json` и `media-type-cases.json` (Go, Python, UI) | работает. Не покрыты: `EvalArtifactName` (6741) и inline `sourceMediaType` (11798) — SUSPECTED, нашёл по grep |
| Таблица ошибок | private errors.go:30-63 и public errors.go:135-170 | ничего. В private нет `storage_transaction_conflict` (public errors.go:169) |

**3.2 — исходы сбоев.** FAIL-ART-1 определяет неоднозначный CAS-исход: сверка через чтение. Для `retryable`, `cancelled` и `timeout` там стоит «not specified» (failure-model.md:279). Код при этом помечает conflict **и frozen** как `retryable:true` — см. G4.

**3.3 — ограничение работы.**
- Генерический List без LIMIT (G1). Спека ART-R47 (artifact-store.md:452) сама разрешает несписанный нефильтрованный список.
- Исключение префикса в ListMetadata делается не по индексу (I2).
- Purge ограничен только временем (I3).
- Чтения материализуют весь payload (I4).

**3.4 — жизненный цикл.**
- В PostgreSQL каскад полный: PurgeRun, PurgeProject, PurgeAuditNamespace, триггер `contractor.lifecycle_purge`.
- Физическое удаление при filesystem-бэкенде проверено только для Run (blob_lifecycle_postgres_test.go:18) и в одном тесте auditcontroller. Purge Project и Audit на filesystem не тестируется: в projectlifecycle и auditstore тестах нет `BlobFilesystem` (VERIFIED по grep).
- Квоты на число bindings или revisions в Run нет.

**3.5 — тесты, которые не могут упасть.**
- Матрица filesystem (`CONTRACTOR_TEST_ARTIFACT_BACKEND=filesystem`) есть только в opt-in цели `test-artifact-blob-backends` (make/platform.mk:129-133). В release.mk:29-35 её нет, хотя Go-часть не требует podman. Её `-run` берёт 8 из 13 тестов `TestPostgresIntegration*`.
- Фейк public пишет, не учитывая `frozen` (fakes_test.go:390-424), хотя сам сохраняет этот флаг в :549.
- 18 Python-тестов пишут собственный `write_artifact`; общего фейка с правилами Server нет.
- Проверка размещения SQL (persistence/sql_placement_test.go) смотрит только литерал, который начинается с ключевого слова. Склеенные запросы она пропускает. VERIFIED копией проверки: postgres_query.go:22 (15 строк), :62 (18), :112 (16); всего по репозиторию 12 пропусков.

**3.7 — экспорты без продакшен-вызова.**
- `ErrQueryUnsupported` и `Scope.ID/Kind` (C5).
- `Service.ImportFindingArtifact` (service.go:565-573) — чистый алиас.
- `DELETE FROM workflow_run_output_publications` в PurgeProject (postgres_purge.go:173-177) — мёртвый код, см. H3.

## E. Цена изменений

1. **40ac02fd fix(memory): bounded enumeration (V48-002).**
   - Масштаб: 14 файлов, +494 строки.
   - Слои: platform (artifacts), edge (privateartifacts), execution (memory), Runtime (artifacts.py, memory), спека, README приватного API.
   - Навязано структурой:
     - лимит 256 вручную записан дважды (prefix_list.go:7 и artifacts.py:32), fixture нет;
     - пришлось добавить опциональный интерфейс, проверку типа и `ErrQueryUnsupported` (prefix_list.go:9-27);
     - разбор двух режимов в обработчике (artifact_handlers.go:42-57).
   - Класс дефекта не закрыт: в задаче сказано «Preserve existing unfiltered Artifact clients». Модельный `list_artifacts` остался без ограничения (G1).
2. **e415ae38 feat(artifacts): fold Git provenance (V365-014).**
   - Масштаб: 8 файлов. Слои: platform, kernel (migration и тест), edge (тест).
   - Навязано структурой:
     - одинаковый `LEFT JOIN artifact_git_sources` удаляли в трёх копиях SQL метаданных (Metadata, ListMetadata, ListVersions в postgres_query.go);
     - два внешних теста зашивали имя таблицы (persistence/postgres/postgres_integration_test.go, public/git_import_handlers_test.go).
3. **0792d0a4 fix(artifacts): storage budgets vs socket deadlines (V287-001).**
   - Масштаб: 14 файлов. Слои: platform (artifacttransfer) и два edge-пакета.
   - Около половины файлов — одна и та же правка, сделанная дважды:
     - `readArtifactBody` «deliberately differs» в private request.go:43-54 и public request.go:102-110;
     - оба `artifact_handlers.go`;
     - два deadline-теста;
     - фейки.

(Дополнительно: c13dfbb9 feat(audit) положил тест подготовки Audit на 579 строк в `internal/artifacts/audit_preparation_postgres_test.go`. Тесты platform-пакета импортируют auditstore, runstore, projectstore, rundeletion и runtimeconfig.)

## F. Рекомендации (по соотношению ценность/трудоёмкость)

1. **Ограничить генерический список.** Дефект, трудоёмкость S, риск низкий.
   - Проблема: G1; postgres_read.go:83-88 и run_artifacts/tools.py:92.
   - Изменение:
     - `limit` и курсор для нефильтрованного приватного List;
     - фильтр скрытых префиксов (`memory.`, `http.body.`, `skills`) на стороне Server;
     - инструмент возвращает страницу и `nextCursor`;
     - лимит в LIM-ART;
     - исправить docstring.
   - Эффект: ответ не больше `limit` строк и меньше 1 MiB при любом размере Run.
   - Закрепить: тест на 15 000 скрытых и 7 000 обычных bindings через настоящий обработчик и Python-клиент; SQL-lint «SELECT FROM artifact_bindings без LIMIT вне purge».
2. **Фильтр filesystem — в release-gate.** Тесты, S, риск низкий.
   - Изменение:
     - перенести `CONTRACTOR_TEST_ARTIFACT_BACKEND=filesystem go test ./internal/artifacts` в release-verify-integration без `-run` (podman не нужен);
     - добавить purge Project и Audit на filesystem;
     - починить проверку SQL на склеенные запросы.
   - Эффект: 13 из 13 интеграционных тестов на обоих бэкендах; 12 запросов уходят в `*_sql.go`.
3. **Один путь неизменяемой Audit-записи.** Дефект и структура, S/M, риск средний.
   - Проблема: G3 и C2.
   - Изменение:
     - auditcontroller и auditservice вызывают `Service.WriteAuditArtifact` (frozen, create-only) и один helper сравнения при повторе;
     - удалить `auditstore.WriteImmutableArtifact` и две копии ReadProjectExact;
     - ввести `contracts.ExactArtifact` с одним `Verify`.
   - Эффект: 4 структуры превращаются в 1, примерно −100 LOC.
   - Закрепить: проверка после раунда Audit — ни одного `frozen=false` в namespace `audit-*`.
4. **Таблица политики bindings и лимитов как общая fixture.** Структура, M, риск низкий.
   - Изменение: `api/testdata/v1alpha1/artifact-binding-policy.json` и `artifact-revision-cases.json` с пунктами namespace/name, modelHidden, memoryReserved, workerResultReserved, runWritable; плюс `limits.json` с 64 MiB и 256, которые сверяются с OpenAPI `maximum`.
   - Потребители: Go (artifactpolicy, contracts), Python (artifact_visibility, worker), UI, `make test-wire-cross-language`.
   - Эффект: около 15 копий из D/3.1 под тестом.
5. **Вернуть артефактам владение их SQL.** Структура, M, риск средний.
   - Изменение:
     - `Service.ThawRunOutputs` рядом с FreezeOutputs вместо runstore/resume_store.go:137;
     - метаданные для scheduler и evalstore — через `ScopedStore.Metadata` или пакетное чтение;
     - удаление строк publications — в runstore (вызывает rundeletion);
     - убрать мёртвый DELETE из PurgeProject.
   - Закрепить: тест владения SQL в духе rundeletion/boundary_test.go — `artifact_*` разрешены только в internal/artifacts, `workflow_*` там запрещены.
6. **Упростить Service и Python-клиент.** Структура, M, риск низкий.
   - Изменение:
     - конкретный `*PostgresRepository` вместо 9 интерфейсов и `ErrQueryUnsupported`;
     - одна валидация вместо Service+Repository;
     - один классификатор SQLSTATE;
     - вынести finding и gateway из `artifacts.py`.
   - Эффект: примерно −150..200 LOC и меньше churn в `artifacts.py`.

## G. Баги

**G1. Модельный `list_artifacts` целиком отказывает на большом Run; Server делает неограниченную работу.** Средний. Механизм VERIFIED, достижимость SUSPECTED.
- Путь:
  1. run_artifacts/tools.py:92;
  2. artifacts.py:246-252 — лимит ответа `MAX_ARTIFACT_JSON_BYTES = 1 MiB`, :29;
  3. privateartifacts/artifact_handlers.go:56;
  4. postgres_read.go:83-88 — без LIMIT.
- Скрытые bindings тоже идут в ответ: фильтр в tools.py:93-97 работает уже после получения.
- HTTP toolset на каждый ответ с телом создаёт `http.body.<16hex>.<id>` (http/tools.py:435-443). По расчёту примерно 14 700 таких ответов, то есть больше 1 MiB, ломают любой `list_artifacts`. Воспроизведение V48-002: 7 000 имён длиной 126 байт дают 1 120 050 байт.
- Тестов на этом размере нет. Тот же класс, что V48-002; исправлен был только Memory.

**G2. Docstring `list_artifacts` врёт.** Низкий. VERIFIED.
- tools.py:84 обещает «namespace, name and current revision».
- Список по контракту versionless: Python contracts/artifacts.py:56, Go contracts/artifacts.go:175. Модель не получает revision для CAS (V178/V184).

**G3. Audit-артефакты пишутся незамороженными вопреки спеке.** Низкий–средний. Путь VERIFIED.
- Манифесты, задачи и worklist Audit пишутся в `audit-*` как незамороженные изменяемые bindings:
  1. auditcontroller/builder.go:105, 366, 505;
  2. auditcontroller/artifacts.go:97;
  3. auditstore/immutable_artifact.go:25 `store.Write`;
  4. service.go:118-150 — `audit-*` для ProjectScope не запрещён;
  5. writeSQL `created_binding` без `frozen`, а `frozen DEFAULT false` (migrations/000002:53).
- Так же auditservice/start.go:670-692.
- Спека: artifact-store.md:60 — «audit-* … created frozen»; REV-R146 (review-and-reports.md:804).
- Тест проверяет `Frozen` только у результатов (auditcontroller/controller_postgres_test.go:1176).

**G4. «Frozen» отдаётся как повторяемая ошибка.** Низкий. Код VERIFIED, последствия SUSPECTED.
- private errors.go:45 и public errors.go:145 отвечают `retryable:true`.
- Frozen постоянен (§3.2: детерминированный отказ не повторяется). В errors.md:297 retryable стоит как «not specified».
- В private нет 503 для 40001/40P01: падает в `default` → 500 (errors.go:61).

**G5. `IsAuditStandardCatalogNamespace` проверяет префикс, а спека резервирует точное имя.** Низкий. VERIFIED.
- Код: visibility.go:22-23. Спека: ART-R25 (artifact-store.md:310) резервирует ровно `audit-standards`.
- Из-за этого UserScope-namespace вида `audit-standards-notes` запрещён на запись и скрыт: service.go:138, public artifact_handlers.go:442 и 553, gitimport/importer.go:77.
- При этом доверенная запись требует точного совпадения (service.go:165).

**G6. Публичный If-Match не проверяет грамматику revision.** Низкий. VERIFIED.
- public request.go:130 разбирает If-Match через `ParseStrongETag` без `publicRevisionPattern` (:25, :157).
- Мусорный revision даёт 409 вместо 400.

**G7. Worker может создавать произвольные `inputs/<name>`.** Низкий. SUSPECTED, неоднозначность спеки.
- service.go:141-145 не резервирует `inputs`, грант это разрешает (handler.go:143).
- ART-R16 (artifact-store.md:252) говорит: имена в `inputs` — это имена слотов. ART-R37 говорит про *изменение* копий.
- Такие bindings попадают в проекцию детали Run (public run_handlers.go:479).

**G8. Скрытая зависимость от blob runtime в context.** Латентный. Fallback VERIFIED, достижимость SUSPECTED.
- Без runtime в context:
  - чтение filesystem-объекта превращается в `ErrArtifactIntegrity` (blob.go:109-111 через activeBlobStore, blob_context.go:76);
  - unlink после commit молча пропускается без лога (blob_context.go:114).
- Сейчас все продакшен-context наследуют app ctx (app.go:174, system.go:58 и 64). Новый `context.Background()` (уже есть в auditcontroller/controller.go:72 и scheduler/allocation_release.go:39) сломает это молча.

## H. Упрощение

1. **Три способа неизменяемой Audit-записи свести к одному.** auditstore/immutable_artifact.go, auditcontroller/artifacts.go:66-123, auditimport/artifacts.go:32-172 → Service.WriteAuditArtifact и один helper. Экономия примерно 80–100 строк. Риск средний: путь восстановления после сбоя.
2. **9 интерфейсов и `ErrQueryUnsupported` убрать** (service.go:182, 212-216, 248-251, 478-488, 518-521, 554-557, 599-602; prefix_*.go). Реализация одна. Около 60 строк. Тесты с фейками перевести на PostgreSQL (§3.5).
3. **Убрать мёртвый `DELETE workflow_run_output_publications` в PurgeProject** (postgres_purge.go:173-177). FK `(run_id, project_id)→workflow_runs` (000028), а projectlifecycle/controller.go:446-458 требует, чтобы Run в проекте не осталось. Минус 5 строк. VERIFIED.
4. **Свести одинаковую обработку SQLSTATE (5 копий):** postgres_write.go:80-112, postgres_audit_write.go:62-88, postgres_project_output.go:47-70, postgres_audit_import.go:48-70. Для `"55000"` нет именованной константы. Около 40 строк.
5. **Убрать двойную валидацию Service→Repository.** postgres_write.go:21-40, postgres_read.go:13-18, query.go повторяют service.go. Плюс `validateScope` дословно повторяет `newScope` (types.go:159 против :34). Около 40 строк.
6. **Удалить алиас `ImportFindingArtifact`** (service.go:565-573). 9 строк.
7. **Python.** Убрать дубли:
   - `_validate_media_type` (artifacts.py:520 и contracts/artifacts.py:46);
   - `_validate_ref` и `_validate_component` (artifacts.py:508-517) повторяют валидаторы pydantic ArtifactRef.

   Около 20 строк.

## I. Оптимизация

1. **Нефильтрованный List** (postgres_read.go:70-105). Сейчас берёт весь scope, Python-клиент отбрасывает результат больше 1 MiB. Нужны LIMIT, курсор и фильтр на стороне Server (см. F1).
2. **ListMetadata для списка Project.** Фильтр `left(namespace, length($5)) <> $5` (postgres_query.go:77) не использует PK `(scope_kind, scope_id, namespace, name)`. Каждый запрос первой страницы последовательно проходит и отбрасывает все bindings в `audit-*`, их примерно Audit × (манифесты, задачи, результаты, доказательства). Исправление: исключение диапазоном `namespace < 'audit-' OR namespace >= 'audit.'` — индексируемо; это вывод, с реальной БД не проверял. SUSPECTED по величине; тест есть только на корректность (postgres_integration_test.go:337).
3. **Purge одной транзакцией без ограничения по строкам.**
   - Все version_id собираются в Go (postgres_purge.go:228-250), затем FOR UPDATE по каждой версии и blob (:261-268).
   - Ограничение только по времени: 2 минуты (persistence/postgres/budgets.go:136-137).
   - Квоты на размер Run нет. Достаточно большой Run никогда не удалится: попытка за попыткой упирается в бюджет. SUSPECTED, нужен тест на размер.
   - Исправление: квота bindings и revisions на Run (проверка при записи под блокировкой scope) и purge-тест на 50k revisions.
4. **Полная материализация payload.** Python `read_artifact` скачивает до 64 MiB ради страницы 256 KiB и только потом отвергает бинарный артефакт (run_artifacts/tools.py:163-174). Versionless-страница каждый раз перекачивает заново (artifact_read_cache.py:22-29). Исправление: Range или HEAD-метаданные в приватном API; проверять media type до загрузки.
5. **Dedup на filesystem перечитывает существующий blob до 64 MiB внутри транзакции вызывающего** (blob_dedup.go:50) при совпадении digest с другим ключом. Это осознанная плата за восстановление пропавшего содержимого (ab63123a). Приоритет низкий; верификацию можно вынести до транзакции.
6. **Повторы на каждый запрос.**
   - GET детали Run проходит все страницы `inputs` и `outputs` при каждом опросе (public run_handlers.go:469-483, 774-797). Всё ограничено, кроме случая G7.
   - Вне узла: `auditstandards.Catalog.List` читает и валидирует все пакеты на каждый запрос (catalog.go:236-258), это до 64×8 MiB.

Недостающих индексов для просмотренных запросов не нашёл: lineage покрыт 000066, ссылочные проверки — query_indexes_postgres_test.go.

**Открытых задач в `tasks/` по этим находкам нет.** grep по `list_artifacts`, `frozen=false`, `WriteImmutableArtifact`, `CONTRACTOR_TEST_ARTIFACT_BACKEND` и `IsReservedMemoryBinding` выдаёт только задачи из `done/`. Будущие V62-008 и V64-002 добавят новых потребителей Service, это усилит раздел B4.

## Топ-3
1. **G1/F1:** модельный `list_artifacts` и приватный List без ограничения — класс V48-002 не закрыт; на большом Run, в том числе из-за скрытых `http.body.*`, инструмент отказывает целиком.
2. **D/3.1 и 3.5 (F4, F2):** зарезервированные namespace и префиксы, 64 MiB, 256 и грамматика revision живут в 7–15 копиях на Go, Python, UI и OpenAPI без fixture; filesystem-матрица в release-gate не запускается.
3. **C1 и G3 (F5, F3):** состоянием артефактов владеют несколько слоёв. runstore размораживает outputs сырым SQL, scheduler и evalstore читают таблицы артефактов, artifacts удаляет таблицу runstore. Audit-артефакты пишутся двумя механизмами, и манифесты остаются незамороженными вопреки artifact-store.md:60.
