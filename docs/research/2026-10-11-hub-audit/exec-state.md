# Аудит узла «Состояние исполнения» (main @ 02bbdcbf, 2026-10-11)

Анализ только для чтения, рабочее дерево `/tmp/claude-1000/-home-ruslan-src-contractor/0fecd9f4-f328-4d49-ab01-b67e6eace15e/scratchpad/hub-audit-src`. В репозитории ничего не менялось, к БД не подключался. Единственный запущенный тест: `env -u CONTRACTOR_TEST_DATABASE_URL go test -run 'TestWorkflowRunTransitionValidation|TestTypedConflictAndNoWorkResults' ./internal/runstore/`, он прошёл (PASS). Анализаторы (на go/parser и x/tools go/packages) лежат в `scratchpad/hub-audit/exec-state/`.

## Топ-3

1. **Устаревшие partial-индексы `workflow_runs`.** Это прямое следствие того, что множество активных состояний Run записано во многих местах. Миграция 000064 добавила `pending`/`waiting`, а предикаты пяти индексов остались старыми. Поэтому список очереди и проверки ссылок на credential не могут ими пользоваться (G1). Исправление: S, одна миграция плюс общий источник множеств состояний.
2. **У переходов Run нет одного хозяина.** Таблица в `store.go:661` шире спеки (RUN-R26). Переходы `running↔waiting`, `failed→pending` и `waiting→running` записываются сырым SQL в обход неё, в том числе из слоя platform. Фейки в тестах не проверяют переходы вообще (G2, G4, F2).
3. **Удалить поверхность, нужную только тестам, и слить три пути создания Run.** Эта поверхность вызывает почти половину правок `store.go`: 23 из 44 коммитов трогали интерфейс `Repository`, которым пользуются только e2e-тесты. В 7 коммитах пришлось параллельно править `CreateRun` и `CreateRunIdempotent` (F3, H1–H3).

## A. Карта узла

| Пакет | LOC (не тесты) | Файлов | Экспортов | Fan-in (не тесты) | Fan-out (internal) | Коммитов |
|---|---|---|---|---|---|---|
| runstore | 4967 | 31 | 195 (70 методов, 55 типов) | 15 | 12 | 76 |
| runtimeconfig | 4325 | 11 | 170 | 10 | 10 | 41 |
| credentials | 3726 | 16 | 200 | 9 | 11 | 44 |
| credentials/litellm | 697 | 3 | 19 | 1 (app) | 6 | 7 |
| memory | 1262 | 3 | 71 | 3 | 5 | 15 |
| runrepeat | 122 | 1 | 6 | 2 | 4 | 4 |
| rundeletion | 222 | 2 | 12 | 3 | 4 | 1 |
| **Итого** | **15 321** | 67 | 673 | 17 внешних пакетов | | 137 |

Fan-in:
- **runstore:** app, auditcontroller, auditimport, auditservice, controlplane, evalservice, findingintake, httpapi/public, httpapi/public/events, memory, planner/session, projectlifecycle, rundeletion, runservice, scheduler. С учётом тестов 23 пакета.
- **runtimeconfig:** app, auditbaseline, auditservice, controlplane, credentials, evalservice, httpapi/public, runservice, runstore, scheduler.
- **credentials:** app, auditservice, controlplane, litellm, evalservice, gitimport, httpapi/public, runservice, scheduler.
- **memory:** app, planner/router, planner/streamline.
- **rundeletion:** app, auditcontroller, httpapi/public.

История начинается 2026-08-29, поэтому «с 2026-08-12» означает всю историю, 44 дня.

Топ-10 файлов по правкам:

| Файл | Правок |
|---|---|
| runstore/store.go | 44 |
| runstore/types.go | 36 |
| runstore/postgres_integration_test.go | 33 |
| credentials/lifecycle.go | 16 |
| credentials/runtime_service_postgres_test.go | 15 |
| credentials/runtime_repository.go | 15 |
| runtimeconfig/resolver.go | 13 |
| runtimeconfig/normalize.go | 13 |
| runstore/allocation_store.go | 13 |
| runstore/scan.go | 11 |

Чаще всего вместе с узлом меняются (число общих коммитов из 137): httpapi/public/types.go 41, handler_test.go 36, contractor-public-v1.yaml 36, ui generated public.ts 31, httpapi/public/fakes_test.go 31, scheduler_test.go 30, run_handlers.go 30, app/app.go 29, scheduler.go 25, persistence/postgres integration test 24. Для одного `store.go` первые партнёры: types.go 27, интеграционный тест 25, httpapi/public/types.go 23, run_handlers.go 21, fakes_test.go 21.

Команды:
- `find … ! -name '*_test.go' | xargs cat | wc -l` — строки.
- `go list -f '{{.ImportPath}} {{join .Imports " "}}' ./...` (и с `-test`) — fan-in/fan-out.
- `git log --no-merges --since=2026-08-12 --format= --name-only -- <hub> | sort | uniq -c` и `git show --name-only` по каждому коммиту узла — правки и соизменения.
- `git log -L '/^type Repository interface/,/^}/:internal/runstore/store.go'` — правки по блокам.
- Анализаторы на go/parser и go/packages — экспорты, использование методов каждым импортёром, экспорты без продакшн-вызовов.

## B. Обязанности

В **runstore** 12 разных обязанностей при одном `PostgresStore` (56 экспортных методов):

| № | Обязанность | Где живёт |
|---|---|---|
| 1 | Агрегат Run: три пути создания, чтение, список, переход, отмена | store.go, store_sql.go, audit_run*.go, scan.go |
| 2 | Политика очереди Scheduler: порядок claim, отсрочка 30 с, продление и снятие lease | store.go:500-594, store_sql.go:155-181 |
| 3 | Пауза очереди владельца и проекция очереди | queue_*store*.go |
| 4 | StageExecution и решения переходов | stage_store*, transition_store.go |
| 5 | Allocations | allocation_store.go |
| 6 | Planner sessions и события Run, включая полную схему planner-событий | run_event_store.go (535 строк, около половины — валидация payload планировщика) |
| 7 | Сквозной прокси к telemetry | report_store.go (111 строк) |
| 8 | Продолжение упавшего Run | resume_store* |
| 9 | Публикация outputs | output_publication* |
| 10 | Инициализация Skills | run_skills.go |
| 11 | Проверки ссылок на credentials | store.go:22, 77-167 |
| 12 | Шаги удаления | delete_store* |

`types.go` — 637 строк и 46 типов. `WorkflowRun` содержит 26 полей (types.go:91-118): вход создания, lease Scheduler, привязку к Audit, skills, отмену. `StageExecution` содержит 31 поле (types.go:371).

Почему `store.go` правится так часто (VERIFIED по `git log -L`):
- 23 из 44 коммитов меняли интерфейс `Repository` (store.go:27-73, 45 методов). Комментарий на store.go:24-25 сам говорит, что он «used by e2e test helpers».
- По 9 коммитов трогали `CreateRun` и `CreateRunIdempotent`, 7 из них — обе функции сразу.
- Сюда же попадают политика claim (a2435bd4, 25f469e1) и валидатор runtime-credential (0b481c7d).

**runtimeconfig:**
- документы RuntimeConfig: normalize, merge, resolver — около 1700 строк;
- версии и привязки, management — около 880;
- закрепление снимка для Run (run_snapshot.go);
- advisory-блокировки ссылок на credential (credentials.go);
- **реестр principal Runtime Agent** (principal.go, 848 строк). Им пользуются только controlplane и app. Это отдельная сущность, которая попала сюда из-за общих блокировок привязок, и ей место в своём пакете.

**credentials:** жизненный цикл LLM-credential (lifecycle.go — 830 строк, это god file), runtime credentials (около 1160 строк), шифрование и мастер-ключ, git-ключи, dev-провайдер `StaticProvider`.

**memory:** инструмент памяти планировщика и PG-store, который сырым SQL читает таблицы runstore (postgres.go:192-240).

**rundeletion:** композиция удаления. Отражает решение V366-006 и соответствует ему.

## C. Связи, которые не видит archtest

1. **SQL по таблицам runstore из других пакетов.** 34 файла в 12 пакетах читают или пишут `workflow_runs`, `stage_executions`, `stage_allocations` и т.д.: evalstore 7, findingintake 5, artifacts 5, scheduler 3, gatewayrecovery 3, telemetry 2, projectlifecycle 2, credentials 2, auditstore 2, runtimeconfig, memory, controlplane. Записи вне runstore (VERIFIED):
   - `gatewayrecovery/waiting.go:18,30` — сам переводит Run `running→waiting` и `waiting→running`, хотя пакет в слое **platform**, ниже execution;
   - `gatewayrecovery/admission.go:65` — reason;
   - `scheduler/postgres.go:264` — `waiting→running`; `:372,:384`, `admission.go:35` — stage_executions;
   - `artifacts/postgres_purge.go:119,174` — удаляет output publications.
2. **runstore пишет SQL по чужим таблицам:**
   - `runtime_credentials` — store.go:155;
   - `projects` — resume_store.go:56, resume_store_sql.go:23;
   - `artifact_bindings` — resume_store.go:139 (оттаивание);
   - JSON-путь по provenance из `runtimesettings` — store_sql.go:30.
3. **JSON-пути, привязанные к Go json-тегам.**
   - Где читаются: store_sql.go:17-22 (`llmCredentialIds`, `$.**.credentialId`), credentials/runtime_repository.go:236-238 (`runtimeCredentialIds`, `{credential,credentialId}`), auditstore/store.go:33.
   - Чьи это теги: run_snapshot.go:34-35, resolver.go:166.
   - Если переименовать тег, проверка перед удалением credential молча вернёт пустой список и пропустит удаление. Защищено круговыми тестами на реальном PG (runstore/postgres_integration_test.go:1553-1630), но только когда БД доступна.
4. **Ширина интерфейса против реального использования.** Продакшн-потребители уже объявляют узкие роли:

   | Потребитель | Используемые методы |
   |---|---|
   | scheduler | Store: 18 (types.go:43) |
   | httpapi/public | RunReader 8 + RunQueue 3 + RunLifecycle 2 |
   | runservice | 2 / 4 / 5 — 8 различных |
   | planner/session | 5 + 3 |
   | auditcontroller | 3 |
   | events | 3 |
   | auditimport, projectlifecycle, credentials | по 1 |

   Ещё 7 пакетов создают `runstore.NewPostgresStore(tx)` прямо внутри своих транзакций; только в scheduler таких мест 13. Значит, сужение интерфейсов fan-in **не снизит**. Его создают типы и константы: scheduler использует 79 идентификаторов runstore, httpapi/public — 53, planner/session — 27. Ещё один источник — конкретный store, создаваемый в транзакции. Цена фейков реальна: `fakes_test.go` (1282 строки, у `fakeRunStore` 24 метода) соизменялся с узлом 31 раз.
5. **Экспорты только для тестов** (VERIFIED, 0 ссылок в продакшне):
   - `runstore.Repository`;
   - `PostgresStore.CreateRun` — 60 вызовов в 32 тестовых файлах;
   - `GetStageTransitionDecision`, `ListStageExecutionReports`;
   - `MaxRunMetadataLabelKey/Value` (metadata_labels.go:11-12) — вообще не используются;
   - `runtimeconfig.Repository.InsertVersion` (repository.go:29);
   - `runtimeconfig.BuiltInRunSnapshot` (run_snapshot.go:180) — 59 тестовых вызовов.

## D. Классы дефектов AGENTS §3

**3.1 Одно правило в нескольких местах.**

*Множества состояний Run* (VERIFIED):

| Где | Копии |
|---|---|
| Go | `types.go:18-25,41-46`; `store.go:585,663-675` |
| SQL в runstore | `store_sql.go:14,90-91,127-131,147,160,167`; `queue_store_sql.go:22` |
| SQL в других пакетах | `credentials/runtime_repository.go:235`; `projectlifecycle/controller_sql.go:38`, `controller.go:291`; `gatewayrecovery/waiting.go:20,55`, `admission.go:59`; `memory/postgres.go:198,237`; `auditstore/execution_sql.go:101`; `evalstore/submissions.go:340` |
| Миграции | CHECK в 000001:28-33 и 000064:2-3; **предикаты partial-индексов** (G1) |
| OpenAPI | 12720-12725 |
| UI | `runs.ts:35-49`, `queue.ts:7-13`, `workflows.ts:200-209` |
| CLI | `run_commands.go:458-460` |

Общего фикстура нет, ни один тест не сравнивает Go-множества с OpenAPI.

*Умолчания экспорта телеметрии* заданы в Go (`telemetry_export.go:16-21,50-52`) и в Python (`settings.py:240-265`). Когда блока нет, Server его не передаёт (`scheduler/execution_preparation.go:173-180`), и умолчания применяет Python (`otlp_http.py:185,408-412`) — см. G3. В `api/testdata/v1alpha1` нет ни одного граничного числового кейса RuntimeSettings: 32 заголовка, 4096 / 16 KiB, flush 1–10, Caido 120, ≤3 targets.

*Другие копии:*
- Лимиты меток 32 и 63: run_snapshot.go:16, principal.go:807, runtimesettings/validation.go:71,76, base.py:256, settings.py:556-573.
- Грамматика ResourceId: 8 regexp в Go (в узле — store.go:22; ещё evalstore, projectstore, auditstore, httpapi …), 4 в UI, плюс OpenAPI:10933. При этом сам runstore при записи проверяет только непустоту (types.go:605-610).
- Грамматика planner-событий: схема `contractor-events-v1.schema.json:128-129`, `run_event_store.go:370-399`, `planner/session/service.go:712-726` (мягче), `streamline/tools.go:635`.

**3.2 Исходы отказов.** Ключевые мутации продуманы:
- `CreateRunIdempotent` / `LookupRunIdempotency` — ключ и digest;
- `ResumeFailedRun` — квитанция `run_stage_resumptions`;
- `RequestRunCancellation` — идемпотентное повторное чтение;
- memory при сбое commit возвращает `ErrMutationOutcomeUnknown` (postgres.go:174-180);
- у credentials есть durable operation record и восстановление при старте (lifecycle.go:190-232, соответствует FAIL-RCF-1/2).

Пробел: `running↔waiting` в gatewayrecovery идёт без CAS по claim или участнику. Запоздавший `failed`/`acquire` от уже завершённого участника может снова поставить `waiting` (SUSPECTED, нужен тест с гонкой).

**3.3 Ограничение работы:** устаревшие индексы (G1), сортировка при claim, повторные claim приостановленных Run, полное чтение стадий на каждом тике (раздел I).

**3.4 Жизненный цикл. rundeletion держится (VERIFIED):**
- единственный `DELETE FROM workflow_runs` — `delete_store.go:105`;
- FK `projects→workflow_runs` без ON DELETE (000027:8);
- удаление Project идёт через `rundeletion` (projectlifecycle/controller.go:364), удаление Audit — через settlement.go:83.

Дыра в закреплении: `boundary_test.go` ищет только Go-имена методов. Сырой `DELETE FROM workflow_runs` в другом пакете тест пропустит. У `runtime_management_operations` и `runtime_config_publications` нет ни retention, ни описания в спеке (низкий приоритет).

**3.5 Тесты:**
- фейки `TransitionRun` принимают любой переход (G4);
- 43+ тестовых вызова используют запрещённый спекой `initializing→running`;
- юнит-тест `store_test.go:18` закрепляет это ребро как валидное.

**3.7** Список экспортов только для тестов — в C.5.

**Секреты (VERIFIED для Go-части узла).**
- В пакетах узла нет логирования.
- Расшифровка выполняется точно в момент выдачи allocation в `SecretString`.
- Сохраняемая конфигурация allocation содержит только provenance (`types.go:558-562`).
- Отчёты редактируются списком секретов (`RecordStageExecutionReportParams.Secrets`).
- LiteLLM-менеджер возвращает только стабильные ошибки (manager.go:59-60).

Риск: `contracts.SecretString.MarshalJSON` раскрывает открытый текст (`contracts/common.go:195`). Любой будущий `json.Marshal` над RuntimeSettings в БД или лог выдаст секрет, и fitness-теста на это нет.

## E. Цена изменений

1. **fae981a0 `fix(runs): queue pending work and recover model gateway outages`.** 105 файлов, 3454+. Пакеты: scheduler 13, httpapi 13, runtime 12, ui 9, gatewayrecovery 8, runstore 6, cli 2, persistence 2, OpenAPI, generated client. Слои: kernel (миграции), platform (gatewayrecovery), execution, edge, composition, UI, Python. Обязательными из-за структуры были правки множества состояний в каждой копии: CHECK, types/store/queue/resume в runstore, OpenAPI и сгенерированный клиент, `cli/run_commands.go` и `queue_commands.go`, `ui/runs.ts` и `queue.ts`. **Пять предикатов partial-индексов пропущены и остаются неверными до сих пор (G1).**
2. **8156e063 + ff3b31ed** (сворачивание меток метаданных Run, V365-015). 10 + 5 файлов. Перенос одной колонки потребовал правок в трёх SQL-запросах создания (store_sql.go, audit_run_sql.go, queue_store_sql.go). Первая версия добавила второй запрос на каждое чтение Run в семи методах: CreateRun, CreateRunIdempotent, CreateAuditRun, GetRun, TransitionRun, RequestRunCancellation, ClaimRunnableRun, LockRunSkillInitialization. Понадобился отдельный fix-коммит. Причина в структуре: нет одной точки создания и сборки `WorkflowRun`.
3. **a2435bd4 `fix(scheduler): age deferred Run claims`** (и 25f469e1). Исправление в Scheduler поменяло только `runstore/store.go`, интеграционный тест runstore и спеку. Политика справедливости очереди живёт в SQL runstore (store_sql.go:155-181), поэтому `store.go` меняется и из-за Scheduler.

## F. Рекомендации (по соотношению ценности и трудоёмкости)

**F1. Множества состояний из одного источника и пересоздание индексов.**
- Проблема: G1 и таблица 3.1.
- Изменение:
  - миграция, пересоздающая `workflow_runs_owner_nonterminal_queue_idx`, `_runtime_llm_credentials_idx`, `_runtime_credentials_idx` и `_project_http_target_credential_idx` с пятью активными состояниями, плюс удаление `workflow_runs_scheduler_idx`;
  - экспорт `ActiveRunStates` / `TerminalRunStates` (лучше в kernel-пакете `contracts/runstate`, чтобы им могли пользоваться gatewayrecovery и CLI) и передача их в SQL параметром `state = ANY($n)`;
  - тест: Go-множества == OpenAPI `WorkflowRunState` / `NonTerminalWorkflowRunState` == CHECK.
- Эффект: страница очереди и проверки credential используют индекс; уходят 15+ литералов.
- Трудоёмкость S/M. Риск низкий.
- Как закрепить: тест-сравнение плюс EXPLAIN-тест под тегом integration (по образцу V300-001).
- Тип: дефект и структура.

**F2. Один хозяин переходов Run.**
- Проблема: G2, C.1, store.go:661-675.
- Изменение:
  - полная таблица по спеке (runs.md:120-140) в `runstate`, включая `failed→pending`, `running↔waiting` и отмену;
  - убрать `initializing→running` и `waiting→succeeded`;
  - методы runstore `EnterWaiting` / `LeaveWaiting` с CAS; gatewayrecovery получает их через порт (или переезжает в слой execution);
  - экспортировать `ValidRunTransition` для фейков;
  - тестовые хелперы проводить через `pending`.
- Трудоёмкость M. Риск средний: 43+ тестовых вызова.
- Как закрепить: fitness-тест «`UPDATE workflow_runs … state` только в runstore» по образцу `TestRunstoreNamesNoAuditOrFindingTables`.
- Тип: дефект (расхождение со спекой) и структура.

**F3. Удалить поверхность только для тестов и слить пути создания.**
- Удаляется: `Repository`, `CreateRun` (общий тестовый хелпер), `GetStageTransitionDecision`, `ListStageExecutionReports`, `report_store.go` (scheduler обращается к `telemetry.Repository` напрямую — это вниз по слоям), `MaxRunMetadataLabelKey/Value`, `InsertVersion`; `BuiltInRunSnapshot` уходит в тестовый пакет.
- Эффект: около 350 строк меньше, `store.go` теряет примерно половину причин для изменений.
- Трудоёмкость S/M. Риск низкий.
- Как закрепить: линтер «экспортный идентификатор без продакшн-ссылки» (тот же анализатор go/packages).
- Тип: структура.

**F4. Server материализует экспорт телеметрии.**
- Проблема: G3.
- Изменение: всегда передавать `export` и `retry` с умолчаниями из Go (`execution_preparation.go:173`); в Python сделать поле обязательным и убрать `defaults()`; добавить `api/testdata/v1alpha1/runtime-settings-bounds-cases.json`.
- Трудоёмкость S. Риск низкий: digest хранимых документов не меняется (RCF-R100).
- Как закрепить: кросс-языковой фикстур.
- Тип: дефект.

**F5. Один валидатор runtime-credential.**
- Проблема: `store.go:148-167` дублирует `credentials/runtime_repository.go:298`, но без проверки ID, целостности записи и допустимых kinds, и отдаёт сырой `pgx.ErrNoRows`. Из-за этого потребитель вынужден разбирать `pgx.ErrNoRows` (`runtimeconfig/credentials.go:143`).
- Изменение: передавать в `NewRunCreationPostgresStore` `credentials.NewRuntimeCredentialRepository(tx)`, как уже делает `auditservice/start.go:139`.
- Трудоёмкость S. Риск низкий.
- Тип: структура.

**F6. Разрезать сущности.**
- Изменение:
  - claim/lease перенести в `claim_store.go`;
  - валидацию planner-событий перенести к планировщику или в contracts, вместе с грамматиками из `session/service.go:712-726`;
  - principal Runtime Agent вынести из runtimeconfig;
  - общий хелпер replay для management (`management.go:323-405` ≈ `principal.go:677-742`).
- Трудоёмкость M. Риск средний.
- Как закрепить: правило в archtest про место для новых пакетов.
- Тип: структура.

## G. Баги

**G1 (medium, VERIFIED по предикатам; влияние на производительность SUSPECTED до EXPLAIN). Устаревшие partial-индексы.**

| Индекс (миграция) | Предикат | Запрос | Его предикат |
|---|---|---|---|
| owner_nonterminal_queue (000029:4-8) | `state IN ('initializing','running','cancelling')` | `queue_store_sql.go:22` | 5 состояний |
| runtime_llm_credentials (000018:154-156) | то же | `store_sql.go:14` | 5 состояний |
| runtime_credentials (000018:158-160) | то же | `credentials/runtime_repository.go:235` | 5 состояний |
| project_http_target_credential (000030:78-81) | то же | `runtime_repository.go:235` | 5 состояний |

Partial-индекс применим, только если предикат запроса влечёт предикат индекса. Предикат на пять состояний не влечёт предикат на три. Поэтому очередь, которую опрашивает UI, сканирует всю историю Run владельца по `owner_created_idx`, а проверки credential делают seq scan. Плюс `workflow_runs_scheduler_idx` (000004:55-57) вытеснен `admission_idx` и только стоит записей. Внесено коммитом fae981a0, тестов нет.

**G2 (low, VERIFIED). Таблица переходов противоречит спеке.** `store.go:663` разрешает `initializing→running`, `store.go:665` — `waiting→succeeded`. Спека: runs.md:110-140, RUN-R26 (runs.md:352), RUN-R28 (:364). В продакшне эти рёбра не вызываются, но `TransitionRun` их примет, а `store_test.go:18` закрепляет первое как валидное. Запущенный тест это подтверждает.

**G3 (low сейчас, VERIFIED). Расхождение RCF-R102 с кодом.** Спека: «The Server pins the export values» (runtime-config.md:756). Фактически при отсутствии блока решает Python (`execution_preparation.go:173-180` → `otlp_http.py:408-412`). Сегодня значения совпадают, но расходятся молча. Тестов на равенство нет.

**G4 (low, VERIFIED, пробел в тестах). Фейки принимают любой переход.** `httpapi/public/fakes_test.go:1174-1191` и `scheduler/scheduler_test.go:2102-2117` пропускают, например, `succeeded→running`. Набор терминальных стадий тоже переписан заново (`scheduler_test.go:2132-2133`). Регрессия в runservice или scheduler с недопустимым переходом пройдёт юнит-тесты.

**G5 (low, SUSPECTED, латентный).** `planner/session/validatePlannerFact` (`service.go:712-726`) принимает `"01"` и `"dispatch-x"`, а `AppendPlannerEvent` отклонит их как `ErrInvalid` (`run_event_store.go:370-399`). Получился бы постоянный отказ планировщика вместо ответа модели. Сейчас недостижимо: ID генерирует сервер (`plan.go:131,178`).

**G6 (low, SUSPECTED, латентный).** runstore принимает при записи любой непустой run_id (`types.go:605`, CHECK только `btrim`, 000001:2). При этом `ListNonTerminalRunIDsByCredential` при первом же ID, не прошедшем `credentialRunIDPattern`, падает целиком (`store.go:99-101`) и навсегда блокирует удаление credential (`lifecycle.go:368-372`). Сейчас все пути создания проверяют ResourceId выше по стеку.

**G7 (low, VERIFIED, только внутреннее).** `clearStageGatewayWaits` (`scheduler/postgres.go:264-266`) возвращает `running`, но оставляет reason «Waiting for the model gateway». Если следом создаётся следующая стадия, Run остаётся `running` со старым reason. Публичный API reason не отдаёт.

## H. Упрощение

| Что убрать или слить | Где | Экономия строк | Риск |
|---|---|---|---|
| Три кодировщика создания Run → один кодировщик + один SQL с опциональными колонками; удалить тестовый `CreateRun` | `store.go:169-296`, `audit_run.go:15-69`, `store_sql.go:41-74` | ~110 | низкий |
| Интерфейс `Repository` | `store.go:24-73` (e2e объявляют свои 2 метода) | ~50 | низкий |
| Сквозной прокси к telemetry + 4 метода из `scheduler.Store` | `report_store.go` | ~110 | низкий |
| Дублирующий валидатор runtime-credential (F5) + ветка `pgx.ErrNoRows` | `store.go:148-167`, `runtimeconfig/credentials.go:143` | ~25 | низкий |
| Дублирующий replay в management | `management.go:323-405` ≈ `principal.go:677-742` | ~60 | низкий |
| Ручная строгость JSON → `strictjson.RejectDuplicateKeys/RequireEOF`; `runrepeat.Decode` → `strictjson.Decode` (заодно отказ на дублях ключей, AGENTS §6) | `memory/codec.go:314-399`; `snapshot.go:55-70` | ~50 | низкий |
| Мёртвые алиасы, `InsertVersion`, мёртвый `workflow_runs_scheduler_idx` | см. C.5 | ~30 + индекс | низкий |

## I. Оптимизация

1. **Очередь и проверки credential без индекса (G1).** Кардинальность — вся история Run владельца. Аудиты и eval порождают тысячи дочерних Run, а вкладка Queue опрашивается. Исправление — F1.
2. **Claim (`store_sql.go:155-181`).**
   - `ORDER BY CASE …` не может взять порядок из `admission_idx`. Поэтому на каждый claim сортируются все незахваченные активные Run (SUSPECTED, нужен EXPLAIN на 10k).
   - Pending-Run приостановленного владельца по-прежнему захватываются: claim не проверяет паузу; `ErrQueuePaused` превращается в `ErrDeferred` (`supervisor.go:170`), затем отсрочка на 30 с. SCH-R122/123 ограничивают это одним claim на Run за окно, но **не суммарно**: P приостановленных Run дают P/30 claim в секунду, каждый с `GetRun`, `ListStageExecutions` и декодированием workflow.
   - Исправление: в SQL пропускать pending-Run без admitted-стадии у приостановленных владельцев; вынести tier в генерируемую колонку с индексом.
3. **Тик Scheduler.** Каждый тик читает все StageExecution целиком, со snapshot, context и результатами (`run_execution.go:39,105`; `stage_store.go:113-138`), чтобы найти одну активную. Их до 1024 на Run (`resume_store_sql.go:32`). Ещё `GetRun` с полным `workflow_snapshot` ради одного `State` (`scheduler/postgres.go:607`). Исправление: `GetActiveStageExecution` и узкое `GetRunState`.
4. **Детальная страница Run без пагинации.** Все стадии, решения и allocations батчем на каждый опрос (`run_handlers.go:459`, `run_detail_batch.go:41`). N+1 нет, но объём растёт с числом попыток. Низкий приоритет: ограничить или пагинировать попытки.

Открытых задач в `tasks/` по этим находкам нет. Проверял runstore, runtimeconfig, credentials, memory, gatewayrecovery, переходы и индексы. Близкие задачи V365-016 и V61-* лежат в deferred и эти вопросы не затрагивают.
