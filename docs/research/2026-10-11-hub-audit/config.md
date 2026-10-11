# Аудит узла config + configload (слой catalog), main @ 02bbdcbf

Репозиторий не менялся (`git status` пуст). Тесты из БД не запускались. Находки подтверждал двумя способами:
- **виртуальные тесты через `go test -overlay`**: файлы лежат только в scratch, репозиторий они не трогают;
- **AST-утилита** в scratch (`hub-audit/config/tool`).

## A. Карта узла

| Пакет | Non-test LOC | Exported: top-level (типы/функции/константы/переменные) + методы + поля | Production fan-in | Fan-out (internal) | Churn с 2026-08-12 (все / non-test) |
|---|---|---|---|---|---|
| internal/config | 7 936 (28 файлов; ещё 38 тестовых файлов, 6 703 строки) | 170 (82/24/60/4) + 37 методов + 224 поля | **22**: app, auditbaseline, auditcontroller, auditimport, auditservice, auditstore, configload, controlplane, credentials, evaldomain, evalservice, evalstore, findingintake, httpapi/public, planner, planner/scan, planner/session, planner/streamline, runrepeat, runservice, runtimeconfig, scheduler | 8: artifactpolicy, clone, contentdigest, contracts, contracts/llmgateway, contracts/scan, randomid, strictjson | 136 / 92 |
| internal/configload | 51 | 4 функции | 1 (app); 19 тестовых | 3 | 2 / 1 |
| internal/agentskills | 1 291 | 55 + 12 | 6: cmd/contractor-skill, app, auditservice, configload, evalservice, runservice | 4 (в т.ч. artifacts) | 29 / 11 |
| internal/auditstandards | 1 287 | 63 + 14 | 8: app, auditbaseline, auditdomain, auditimport, auditservice, configload, evalservice, httpapi/public | 5 | 11 / 10 |
| internal/configtest | 65 | 2 | 0; 5 тестовых | 0 | 3 / 2 |

Прежние оценки подтвердились: около 8k строк и 22 пакета, зависящих от config. Число коммитов теперь 92 (non-test) против прежних 88.

Сам config чист: pgx среди зависимостей 0. А configload через agentskills/auditstandards → artifacts тянет 11 пакетов pgx, так что офлайн-валидатор каталога зависит от драйвера БД.

**Top-10 non-test файлов по churn:**

| Файл | Коммитов |
|---|---|
| descriptors.go | 34 |
| workflow.go | 26 |
| snapshot.go | 24 |
| types.go | 23 |
| manager.go | 20 |
| loader.go | 16 |
| audit_profile.go | 15 |
| digest.go | 14 |
| execution_config.go | 12 |
| template.go / persisted.go | по 9 |

**Co-change вне узла** (107 non-test коммитов узла):
- `api/openapi/contractor-public-v1.yaml` 25;
- `runtime/src/contractor_runtime/factories.py` 24;
- `ui/src/api/generated/public.ts` 19;
- `internal/app/app.go` 19;
- `api/descriptor-parity/toolset-infrastructure-channels.yaml` 19;
- `internal/scheduler/scheduler_test.go` 18;
- `configs/README.md` 18;
- `internal/httpapi/public/types.go` 17;
- `internal/controlplane/registry.go` 15.

По каталогам: httpapi/public 159, runtime/tests 147, runtime/src 107, docs/spec 92, scheduler 85, contracts 82.

Из 34 коммитов `descriptors.go`:
- 26 одновременно правили Python-тулсеты;
- 19 правили parity-фикстуру;
- 17 правили `config_test.go`.

Из 24 коммитов `snapshot.go` 17 шли вместе с `types.go`: clone-функции следуют за типами.

**Команды:**
- LOC: `find internal/<p> -name '*.go' ! -name '*_test.go' | xargs wc -l`.
- Экспорты: AST-утилита `hubtool exports` (go/parser, экспортируемые объявления верхнего уровня).
- Fan-in: `go list -f '{{.ImportPath}} {{join .Imports " "}}' ./...`; для тестов `go list -test -f '… {{.TestImports}} {{.XTestImports}}'`.
- Fan-out: `go list -f '{{join .Imports "\n"}}' ./internal/<p>`.
- Churn: `git log --no-merges --since=2026-08-12 --format=%H -- <path> ':!<path>/*_test.go' ':!<path>/testdata'`.
- Top files: `git log … --name-only --format= -- <hub> | grep -v _test | sort | uniq -c`.
- Co-change: `git show --name-only` по каждому коммиту узла минус пути узла.
- Использование идентификаторов по импортёрам: `hubtool uses` (selector-выражения по алиасу импорта).

## B. Обязанности

| Обязанность | Файлы | LOC | Кто потребляет |
|---|---|---|---|
| Реестр дескрипторов кода (planner/runtime/toolset/sandbox) | descriptors.go, tool_workflow_inputs.go, sandbox_compatibility.go | 507 | loader, controlplane, auditservice, findingintake |
| Resolved-модель — она же **персистентный формат** Run/Stage/Audit snapshot | types.go (resolved-часть), persisted.go, snapshot.go (clone) | 998 | 19 из 22 импортёров |
| Загрузка YAML (discovery, strict decode, envelope) | loader.go, source-структуры в types.go | ~750 | только Load |
| Разрешение Workflow/AgentTemplate и проверка графа | workflow.go (**1 134**, god file), template.go, validation.go, workflow_bindings.go | 1 685 | loader, scheduler, runservice |
| Слои executionConfig и проверка credential | execution_config*.go, llm_gateway.go | 1 361 | runservice, evalservice, auditservice |
| Контракты scan/auditTask | scan_plan.go, audit_scan.go, audit_task.go | 443 | planner/scan, scheduler |
| Модель и правила AuditProfile (режимы, interaction, лимиты раундов) | audit_profile.go (**968**), audit_*.go | 1 352 | 6 audit-пакетов, findingintake |
| Managed publication (атомарная запись, идемпотентность, API-проекции) | manager.go, resources.go | 931 | app, httpapi |
| Digest (поля собираются вручную) | digest.go | 180 | — |

**Низкая связность.** Manager, Snapshot, Publication* и Configuration* нужны только трём пакетам: app, configload, httpapi/public. Остальные 19 импортируют config ради data-модели и декодеров snapshot (≈98 различных идентификаторов; httpapi использует 44, auditservice 24). При этом 47 из 92 non-test коммитов не трогали модельные файлы (types/persisted/snapshot/audit_profile). После разделения такие правки не задевали бы 19 модельных потребителей.

**Чему место в другом пакете:**
- Персистентная модель — отдельный пакет: формат БД сейчас задаётся json-тегами config.
- Модель AuditProfile — это доменная политика Audit, а не каталог.
- Реестр дескрипторов — чистые данные.
- DB-backed `agentskills.Catalog` и `auditstandards.Catalog` (seed через artifacts.Service) лежат в слое «static catalogs».

Ручные digest-карты хрупкие. Прецедент — `ab799945` («Include preparation attempt budget in AuditProfile digest», audit_profile.go:913): поле `MaxRunAttempts` забыли включить в digest.

## C. Связи, которые не видит archtest

1. **SQL читает JSON-форму config.ResolvedAuditProfile. VERIFIED.**
   - Читаются теги `workflows`, `kind`, `maxRunAttempts`, `workerCompletion.{stage,agent,kind}`, `interaction.findingConfirmation` (audit_profile.go:133-203).
   - Места: `auditstore/execution.go:156`, `auditstore/execution_intent_sql.go:34,84`, `auditstore/preparation_sql.go:19,58,84`, `findingintake/service.go:265-269`, `findingintake/postgres_sql.go:28`, триггеры `migrations/000054_audit_completion.sql:31-40` и `000098_audit_preparation_storage.sql:65-69,109`.
   - Enum-литералы скопированы: `'prepare'`, `'human-required'`, `findingPolicy == "disabled"` (findingintake/service.go:274), `role_attempt BETWEEN 1 AND 10` (000098:20) — это MaxAuditRunAttempts=10 из audit_profile.go:26.
   - Переименование тега ловят только PostgreSQL-тесты.
   - Попутно: findingintake читает таблицу `audits`, которой владеет auditstore.
2. **Json-теги — это и формат БД, и вход digest для eval-пинов.** `ResolvedWorkflow` хранится в `workflow_runs.workflow_snapshot` и строго декодируется в 9 местах. Тот же JSON хэшируется в `contracts.CheckPinnedSelection` (pinned_selection.go:16-28). Любое новое не-omitempty поле меняет все eval-пины.
3. **Широких интерфейсов нет.** Потребители объявляют узкие интерфейсы: `runservice.WorkflowResolver` (1 метод), `evalservice.Catalog` (2), `controlplane.PlacementGatewayLookup` (1), `httpapi.ConfigurationCatalog` (7). Проблема ширины — в data-модели, а не в интерфейсах.
4. **Экспорты без production-вызова (VERIFIED).**
   - `Snapshot.AgentTemplate` и `Snapshot.Instructions` (snapshot.go:152-193) в комментариях прямо помечены как нужные только тестам.
   - `LoadUnion`, `IsModelFreePlanner`, `ValidateAuditTaskWorkflow` извне не вызываются.
   - `ValidateAuditPreparationProfile` и `ValidateAuditTaskProfile` извне вызываются только из тестов.
   - В agentskills/auditstandards около 70 экспортированных имён (в основном константы `Maximum*` и `Code*`) извне не используются.
5. **Параметр `Descriptors` декоративный (VERIFIED).** Production всегда передаёт `MVPDescriptors()` (app/config_command.go:38,71). При этом 5 путей читают глобальный реестр мимо параметра: sandbox_compatibility.go:11, tool_workflow_inputs.go:14, workflow.go:514 (variadic default), descriptors.go:289, а также auditservice/compatibility.go:69 и findingintake/service.go:283. Инжект, который делают тесты (tests/eval/agent_instructions/overlay_test.go:173), действует только на часть проверок.

## D. Классы дефектов AGENTS.md §3

### 3.1 Одно правило в нескольких местах

| Правило | Копии | Чем закреплено |
|---|---|---|
| Грамматика id/version конфигурации | contracts/common.go:73-80 (источник); OpenAPI 12466-12480; ui/src/api/workflows.ts:14-15; Python contracts/base.py:28-29 | Фикстура `config-identity-cases.json`. **Не покрыты:** ui/src/routes/studio/validation.ts:41,53 — пропускает заглавные буквы, `.`/`+` в имени и не проверяет длину; ui/server/static-server.mjs:36,42,50 — не допускает `+` в версии (G4). Спека LIM-CAT-1/2 утверждает, что UI применяет общую грамматику |
| Toolset: имена tools, active/finding-классификация, required inputs | descriptors.go:77-204; пины в config_test.go:78-442; Python `exported_tools` в 18 фабриках и их тесты; reader.py:38 литерал `inputs/findings` | Parity-фикстура проверяет **только каналы** (config_test.go:466-496, test_capabilities.py:44-67). Имена сейчас совпадают (сверено AST-извлечением), но расхождение проявится только при placement (controlplane/registry.go:1100-1114). `ActiveCheckTools` есть только в Go — это классификация безопасности для `activeChecks: prohibited` |
| Toolsets, которым нужен workspace | switch в workflow.go:532-535 и Python `requires_workspace` (6 фабрик) | Ничем. Сейчас эквивалентны: code-execution покрыт косвенно через podman@1 |
| Planner: кардинальность и model-free | config workflow.go:913-930 и scan_plan.go:15-18; scheduler/workflow.go:43-56; planner/types.go:16-19; UI validation.ts:110-118 | Ничем: `PlannerFactories` — просто набор имён |
| Лимиты исполнения Audit и enum-ы | audit_profile.go:15-28; SQL 000034:36-42; OpenAPI 11357-11365 и 11598-11604; UI validation.ts:591-630 | Ничем |
| Лимиты skill-пакета (15 констант), грамматика имени skill | agentskills/model.go:6-20 ↔ runtime agent_skills/package.py:26-40; имя: agentskills/path.go:58 ≡ contracts/artifacts.go:117 (побайтно одинаковые функции), package.py:42, contracts/base.py:31 | `testdata/agent-skills/cases.json` (31 кейс) фиксирует поведение, но лимиты не пинит |
| Digest ModelPolicy/AgentTemplate | digest.go ↔ runtime digests.py | **Хорошо:** покрыто `digest-cases.json` |

### 3.2 Исходы отказов
- Классификация ошибок поиска credential теряется (G1).
- Правка operator root во время работы Server и частичная публикация блокируют все публикации до рестарта (G5).
- Ключи идемпотентности хранятся только в памяти (manager.go:115,596-598): после рестарта переиспользованный ключ с другим телом принимается. Низкий риск.

### 3.3 Ограничение работы
- На каждую страницу каталога сериализуется весь каталог (I1).
- Чтение манифестов без ограничения размера (loader.go:291,313). При этом лимит 2 MiB есть только на чтении snapshot (G3).

### 3.4 Жизненный цикл: правка name@version на месте не обнаруживается. VERIFIED
CAT-R44 и §2 AGENTS.md требуют неизменности. Фактически защищены только:
- managed-публикация (`Linkat` → EEXIST, manager.go:580-584);
- Skills (seed_drift классифицируется, не фатально);
- Audit standards (ErrDrift);
- eval-пины.

Для Workflow, AgentTemplate, ModelPolicy, LLMGateway, ExecutionConfig, AuditProfile и инструкций в operator root после рестарта новое содержимое молча получает прежний name@version.

Репозиторий так делает регулярно. С 2026-08-12 **129 изменений** shipped-манифестов меняли не только комментарии и при этом не меняли `metadata.version`. Например, `worker@2` правили 5 раз (fb496d13, 456c9d68, 05488d20, 35ff8f60, 56dac45b): temperature 0.1→1.0→0.6, maxTotalTokens 2.5M→24M→25M. Ещё 9 коммитов правили инструкции, а это меняет digest потребляющих шаблонов. Открытой задачи на это нет.

### 3.5 Тесты, которые не могут упасть
- **UI-тесты Studio читают живой `configs/`** — нарушение ground rule: ui/src/routes/studio/document.test.ts:17,72-95 и advanced-forms.test.tsx:46. Вдобавок они доказывают лишь то, что мягкий TS-валидатор принимает реальные файлы.
- Пины списков tools в config_test.go — change-detector'ы: падают только при правке `descriptors.go`, расхождение с Python не ловят.
- `contractor server config validate --root ./configs` запускается только в `make test-config` (make/core.mk:37-39), а эта цель не входит ни в один gate: release.mk, verify, test, CI. Живой каталог всё же загружается нетегированным `tests/e2e/audit_programs_test.go:21`. Но дополнительная проверка CLI `validateBundledCatalogStandards` (app/catalog_standard_validation.go:13) и объединение с managed root в gate не попадают.

### 3.7 Нет production-вызова
См. C.4 и C.5.

## E. Цена изменений

1. **a57162ff feat(contracts): failure signatures на LLMGatewayConfig** (24 файла, +826/−23).
   - Слои: foundation (api ×7), kernel (contracts), catalog, execution (scheduler), edge (publicclient), плюс Python, UI, configs.
   - Вызвано узлом: одно поле потребовало 5 ручных отображений внутри config — source-struct types.go:294, resolve llm_gateway.go:52-94, digest digest.go:45, тело публикации manager.go:64/452, resource body resources.go:191. Шестое, `cloneLLMGatewayConfig`, забыли — это G2.
2. **1dcadfc5 fix(contracts): одна грамматика id/version (V323-001)** (35 файлов, +827/−188).
   - Затронуто 12 Go-пакетов в kernel/catalog/execution/edge, 8 файлов UI, тесты Python, OpenAPI, 2 схемы, спека.
   - Вызвано узлом: грамматика была переписана в config/validation.go, manager.go, audit_profile.go, auditstandards, runtimeconfig и других.
   - Класс не закрыт: static-server.mjs пропустили, а `d90560d0` (2026-10-07) через три дня добавил пятую, более мягкую грамматику в Studio.
3. **d3f80ff7 feat(scan): ffuf (V55-005)** (16 файлов).
   - Одно имя `scan_ffuf` пришлось вписать в 3 Go-списка (Tools, InfrastructureChannels, ActiveCheckTools), Go-тест-пин, parity yaml и Python-фабрику с тестами.
   - Вызвано узлом: параллельные списки внутри дескриптора и пины, повторяющие код.

## F. Рекомендации (по отношению ценность/трудозатраты)

1. **Сохранять класс ошибок поиска credential.**
   - Проблема: execution_config.go:742 выбрасывает причину; runservice/public.go:53-54 оборачивает всё в ErrInvalid.
   - Изменение: оборачивать санированный классифицированный sentinel (без текста — тест «secret-value must not escape» остаётся); runservice маппит только ошибки валидации; удалить обходной tracker в evalservice/preflight_errors.go:60-103.
   - Эффект: POST /v1/runs отдаёт 503 RunCreationUnavailable, как обещает OpenAPI, вместо 400.
   - S, риск низкий. Закрепить handler-тестом в духе credential_lookup_error_handlers_test.go для Workflow с credential. **Дефект.**
2. **Lock-файл идентичностей каталога** (`configs/catalog.lock`: kind name@version → digest).
   - Проверяет его `config validate` (production-читатель); `test-config` добавить в release-verify-lint.
   - Эффект: 129 правок на месте за 60 дней стали бы ошибками CI.
   - S–M, риск низкий: появляются принудительные bump версий. **Дефект** — правило §2 сейчас нарушается.
3. **UI: грамматики только по фикстурам и без чтения живого configs.**
   - static-server.mjs и studio/validation.ts тестировать против `config-identity-cases.json` и `media-type-cases.json`.
   - Тест «реальный каталог грузится» перенести в tests/ui-stack.
   - S, риск низкий. **Дефект** (G4, G6, нарушение ground rule).
4. **Расширить parity-фикстуру до полного дескриптора** (tools, active/finding, required inputs, requires_workspace).
   - Python сверяет свои фабрики с ней; Go-пины удалить (~220 строк в config_test.go:78-442).
   - S–M. Закрепить: Go и Python читают одну фикстуру. **Структура**, с аспектом безопасности ActiveCheckTools.
5. **Один реестр.**
   - Убрать параметр `Descriptors` и 5 параллельных путей доступа (C.5): один нормализованный `sync.OnceValue`.
   - Факты о planner'ах (min/max agents, modelFree) перенести в дескриптор, и пусть scheduler/workflow.go:43-56 читает оттуда.
   - Закрепить AST-тестом в духе boundary_test, запрещающим `MVPDescriptors()` вне config.
   - M, риск средний: eval overlay-тест опирается на инжект. **Структура.**
6. **Отделить персистентную модель от загрузчика:** `config/resolved` (Resolved*, Decode*, ValidateWorkflowGraph, clone) и `config/auditprofile`.
   - Fan-in загрузчика упадёт с 22 до 3; ~51% коммитов перестанут задевать модельных потребителей.
   - JSON-пути, которые читает SQL, сделать константами плюс unit-тест на маршалинг.
   - L, риск средний. **Структура.**

## G. Баги

- **G1 (medium, VERIFIED). Транзитная ошибка БД или recovery-in-progress при поиске credential превращается в постоянную ошибку.**
  - Путь: `validateStageExecutionConfigCredentials` (execution_config.go:740-742) теряет причину.
  - Overlay-тест: `errors.Is(err, transient)=false`, сообщение «Stage "copy" base agent builder credential is unavailable».
  - Последствия по путям:
    - **POST /v1/runs:** runservice/public.go:54 → ErrInvalid → errors.go:164-168 → **400 invalid_request, retryable=false**. Нарушен контракт OpenAPI 503 RunCreationUnavailable.
    - **Audit start** (auditservice/start.go:122-126): 500, и ретрай транзакции на conflict не срабатывает.
    - **Eval:** обходит через tracker (это побочный канал от фикса 0b481c7d).
  - Тестом не покрыто.
  - Соседний случай вне узла (SUSPECTED): findingintake/service.go:274 `err != nil || … → ErrToolNotSelected`.
- **G2 (low, VERIFIED). `Snapshot.LLMGateway` и clone в resolved Workflow делят `*FailureSignatures` со snapshot** (snapshot.go:195-204, execution_config.go:388).
  - Overlay-тест: мутация возвращённой «копии» поменяла snapshot. Нарушен контракт «каждый accessor возвращает deep copy» (snapshot.go:13-14).
  - Мутирующих вызывающих сейчас нет.
  - Появилось в a57162ff; это тот же класс, что «Align drifted Go clone…» (a8fd4104).
- **G3 (low–medium, VERIFIED). Лимит 2 MiB на Workflow snapshot существует только при чтении** (scheduler/workflow.go:18,28 → failUnsupportedRun, run_execution.go:35-37).
  - Каталог (CAT-R42 «no maximum», loader.go:291-297) и runstore (types.go:612-618) такой snapshot принимают.
  - Overlay-тест: инструкция 2.28 MB → snapshot 2 342 838 байт; загрузка успешна, а Run упадёт после 202. В спеке лимита нет.
- **G4 (low, VERIFIED). Static-server отдаёт 404 на deep link/refresh версий с `+`.**
  - Маршруты `/catalog/{workflows,agents,audit-presets}/n/v`, `/operations/configurations/...`, `/projects/.../workflows/n/v/run`; версии вроде `2026.1+eu` есть в valid-кейсах фикстуры.
  - static-server.mjs:36,42,50, проверка node: false; extname ≠ "" → 404 на строке 379.
  - static-server.test.mjs:228 проверяет `+` только для маршрута RuntimeConfig. Строка 49 корректна: это грамматика RuntimeConfig (≤63/≤128).
- **G5 (low–medium, VERIFIED). Правка operator root на месте или частичная публикация блокирует все публикации до рестарта.**
  - Частичная публикация — сбой после `Linkat`: Unlinkat/Fsync manager.go:586-592 или seam AfterDurablePublish.
  - Ошибка «configuration roots changed outside the publication manager» (manager.go:334,346) не классифицирована → 500 **retryable=true** (errors.go:172-173), а повтор никогда не пройдёт.
  - Overlay-тест: подтверждено в обоих сценариях. Исход в FAIL-CAT-2 не определён, тестов нет.
- **G6 (low, VERIFIED). Studio показывает «валидно» там, где Server отклонит:** имена `A.b+1`, media type `^\S+/\S+$` (validation.ts:41,53,82).

## H. Упрощение

| Что | Действие | Экономия | Риск |
|---|---|---|---|
| Пины списков tools (config_test.go:78-442) | Заменить расширенной parity-фикстурой | ~220 строк | низкий |
| Чтение файлов без follow symlinks (agentskills/read_nofollow_*.go, auditstandards/read_nofollow*.go) | Один хелпер в internal/securefile; заодно ограничить чтение манифестов в loader | ~90 строк | низкий |
| `validSkillName` (agentskills/path.go:58) ≡ `contracts.validAgentSkillName` | Использовать contracts | ~20 строк | нет |
| Variadic `registered ...Descriptors` (workflow.go:508-516) и 5 путей доступа к реестру | См. F5 | ~40 строк | средний |
| Повторная проверка графа: при старте Audit граф каждой роли проверяется 3 раза (persisted.go:87, execution_config.go:675, 686); `ValidateWorkflowGraph` не вызывается в 4 из 6 мест декодирования (app/run_skill_initializer.go:45, findingintake/direct_verification.go:201, runservice/audit_completion.go:34, planner/session/scan_history.go:38) | Встроить в `DecodeResolvedWorkflowSnapshot`, а `Validate…Credentials` и `…CredentialIDs` сделать чистыми | ~15 строк, единое строгое чтение | низкий |
| Двойное обнаружение bundle при старте (configload.go:22-31 отбрасывает план; composition_catalogs.go:39,60 строит заново) | Возвращать план из configload | ~30 ms из 69 ms загрузки | низкий |
| 11 ручных clone-функций (snapshot.go:195-384, audit_profile.go:931-961) | Оставить, но добавить reflection-тест «нет общих указателей» на все accessor'ы, либо generic deep copy | до ~220 строк | средний |

## I. Оптимизация

Узел в основном не на горячих путях.
- **I1. Каталог на каждую страницу.** GET /v1/workflows делает clone, `json.Marshal` и sha256 **всего** каталога ради fingerprint неизменяемого snapshot (workflow_handlers.go:30-31, catalog_discovery.go:66-73).
  - Замер: 21 workflow → 339 KB и 1.6 ms на запрос. UI-пикер, проходящий все страницы, платит O(страниц × каталог).
  - То же для `AuditProfiles()` (200 KB, 0.6 ms).
  - Исправление: fingerprint считать один раз в `newSnapshot`, итерировать без clone.
- **I2. Scheduler декодирует и заново проверяет весь snapshot на каждом шаге Run** (scheduler/workflow.go:26-40): 0.71 ms на самом большом shipped workflow (41 KB), до ~35 ms при 2 MiB. Ограничено maxConcurrentRuns ≤ 32 (LIM-OPS-7), так что низкий приоритет: кэш по (run_id, digest).
- **I3. Каждая публикация — полная перезагрузка вместе с упаковкой bundle под lock:** 69 ms. Публикации редки, действий не требуется.

## Топ-3
1. G1: при сбое поиска credential POST /v1/runs отвечает 400 invalid_request, хотя должен 503 (так обещает OpenAPI); Audit start отвечает 500. Фикс S (F1).
2. Неизменность name@version нигде не проверяется: 129 правок shipped-манифестов на месте за 60 дней, `worker@2` правили 5 раз. Нужен lock-файл digest плюс `test-config` в gate (F2).
3. Реестр toolset'ов и грамматики скопированы вручную в Go, Python и UI: parity-фикстура проверяет только каналы, UI Studio и static-server расходятся с общей фикстурой (G4, G6). Нужны полная фикстура и один реестр (F3–F5).
