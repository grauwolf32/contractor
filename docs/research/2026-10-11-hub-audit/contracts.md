# Аудит узла: `internal/contracts` ×7 и приватный протокол v1alpha1

Worktree `main@02bbdcbf`, только чтение. Расхождения подтверждал scratch-программами в `…/scratchpad/hub-audit/contracts/` (копия пакетов в отдельном модуле, `pyprobe.py` на `.venv` Runtime с `-I -B`). `git status` worktree после прогонов чистый. К БД не подключался. В `tasks/*.yml` ни одна открытая задача эти находки не покрывает (grep по duplicate key, ResultJSONSize, stage_finalization, workspace target, orphan-схемам, cloneSnapshot).

## A. Карта узла

| Пакет / артефакт | LOC без тестов | Экспорт: верхний уровень / поля | Прямой fan-in | Транзитивный (из 99) | Fan-out | Коммитов с 08-12 |
|---|---|---|---|---|---|---|
| internal/contracts (ядро) | 2069 | 190 / 144 | 43 (37 вне хаба) | 58 | contentdigest, strictjson | 81* |
| contracts/reporting | 1496 | 79 / 167 | 9 | 30 | contracts, strictjson | 1 |
| contracts/runtimesettings | 588 | 33 / 59 | 10 | 30 | cabundle, contracts, llmgateway | 2 |
| contracts/control | 572 | 35 / 66 | 1 (controlplane) | 15 | contentdigest, contracts, reporting, runlabels, runtimesettings | 4 |
| contracts/scan | 400 | 31 / 35 | 4 | 42 | contentdigest, contracts, strictjson | 1 |
| contracts/llmgateway | 240 | 16 / 16 | 13 | 44 | contracts | 1 |
| contracts/runlabels | 88 | 9 / 0 | 4 | 30 | contracts | 1 |
| contracts/contractstest | 66 | 4 | 0 в проде, 5 тестовых | — | — | — |
| api/v1alpha1 (17 схем + README) | 3289 | — | — | — | — | 43 |
| api/testdata/v1alpha1 | 37 valid, 48 invalid, 14 case-файлов + index.json | — | — | — | — | 41 |
| runtime/…/contracts (Python) | 2893: 98 классов, 412 полей | — | 72 файла (30 в toolsets) | — | — | 18 (+33 у прежнего contracts.py) |

\* История ядра по пути включает время до разделения 2026-10-09. Подпакеты созданы в тот день, их churn пока ни о чём не говорит. После коммита документации к разделению (972a6753) хаб затронул ровно один коммит (eb678d41).

Прямые импортёры ядра (37): agentskills, app, artifacts, auditbaseline, auditcontroller, auditdomain, auditimport, auditpriority, auditscan, auditservice, auditstandards, auditstore, cli, config, controlplane, credentials, credentials/litellm, evalservice, evalstore, findingintake, httpapi/privateartifacts, httpapi/public, mtls, planner, planner/a2a, planner/scan, planner/session, planner/stateview, planner/streamline, projectstore, runrepeat, runservice, runstore, runtimeconfig, scanplan, scheduler, telemetry.

Сверка гипотез прошлого обзора:
- Ядро импортируют 37 пакетов: подтверждено. Транзитивно от него зависят 58 из 99.
- reporting: 9 прямых импортёров, транзитивно 30.
- JSON-тегов сейчас 491, а не 508. В это число входят wire-структуры внутри `UnmarshalJSON`.
- Файлов схем 17, а не 18.

Самые часто менявшиеся файлы (`git log --follow`, без тестов): `private.go` 29, `agent_template.go` 23, `allocation.schema.json` 21, `api/v1alpha1/README.md` 19, `settings.py` 13, `reporting/telemetry.go` 13, `common.schema.json` 11, `artifacts.go` 9, `stage_content.go` 7, `llmgateway/llm_gateway.go` 7.

Хаб затронули 91 коммит. Медиана — 25 файлов на коммит, p90 — 60, максимум — 282. Чаще всего вместе с хабом меняются:
- `runtime/tests/test_contracts.py` — 35;
- прежний `runtime/src/contractor_runtime/contracts.py` — 33;
- `controlplane/runtime_client_test.go` и `registry.go` — по 25, `runtime_client.go` — 24;
- `scheduler_test.go` — 22, публичный OpenAPI — 22, `scheduler.go` — 20;
- `ui/src/api/generated/public.ts` — 18, `config/types.go` — 17.

Команды:
- LOC: `ls <pkg>/*.go | grep -v _test | xargs cat | wc -l`.
- Экспорт: свой счётчик на go/ast (`hub-audit/contracts/expcount`).
- Fan-in и транзитивные зависимости: `go list -f '{{.ImportPath}} {{join .Imports " "}}' ./...` и `… {{join .Deps " "}}` с разбором через awk.
- Churn: `git log --no-merges --since=2026-08-12 --format=%H -- <path>`, по файлам — с `--follow`.
- Co-change: `git show --name-only` по каждому коммиту хаба, считал файлы вне хаба.
- Какие символы ядра использует каждый импортёр: сканер селекторов на go/ast (`hub-audit/contracts/symuse`, результат в `coreuse.tsv`).

## B. Обязанности

Сгруппировал символы ядра по файлам-кластерам и посчитал, сколько импортёров берут каждый кластер (подтверждено сканером):

| Кластер | Файлы | Wire? | Импортёров |
|---|---|---|---|
| ArtifactRef, имена, media types | artifacts.go, media_type.go | общий доменный тип | 29 и 4 |
| Грамматики, SecretString, `Invalidf`, APIVersion | common.go | словарь ядра | 28 |
| Разрешённые AgentTemplate и ModelPolicy, бюджеты | agent_template.go, tool_execution.go, worker_session.go | значение конфига, которое едет по проводу | 15 |
| Skills: ResolvedSkill (wire) и RunSkillSnapshot (провенанс Run) | skills.go | смешанный | 11 |
| Ключи адаптеров и credential kinds, классы ошибок | private.go | по сути runtimesettings | 10 |
| Planner↔Worker: StageContent, WorkerCompletion, лимиты, handle | stage_content.go, worker_completion.go, result_limits.go, worker_handle.go | A2A wire | 6–10 |
| Кодек | private_codec.go | wire, но 5 доменных пакетов берут его как обычный JCS | 9 |
| Завершение Audit | audit_completion.go | wire плюс правило конкретного Toolset | 6 |
| Чистый домен | pinned_selection.go, audit_task.go, `FindingCollectionMediaType` | нет | 3, 4, 2 |

Выводы:
- **19 из 37 импортёров ядра вообще не трогают приватный протокол** (подтверждено). Им нужны только ArtifactRef, имена, media types, RunSkillSnapshot, CheckPinnedSelection, JCS или грамматика ID. Это agentskills, artifacts, auditbaseline, auditcontroller, auditdomain, auditimport, auditpriority, auditscan, auditservice, auditstandards, cli, evalservice, evalstore, findingintake, mtls, projectstore, runrepeat, runservice, app. Например, `auditpriority` импортирует ядро ради одного `MarshalPrivateCanonical` (`auditpriority/candidates.go:29`). Этот символ — обёртка над `strictjson.Canonical` с пометкой «private-wire encoder» (`private_codec.go:49-55`).
- В kernel-слое лежит доменная логика (подтверждено):
  - `CheckPinnedSelection` (`pinned_selection.go:12-28`) — предусловие run/audit/eval-сервисов;
  - `AuditTaskOpenAPIScanV1` (`audit_task.go:7`);
  - `FindingCollectionMediaType` (`media_type.go:17`);
  - `ValidateForPlanner` (`agent_template.go:94-114`) — политика Planner, которая никогда не уходит в Runtime;
  - правило `audit-results@2` с именами инструментов `read_audit_task` и `submit_check_result` (`audit_completion.go:47-68, 89-99`). Это знание Toolset-дескриптора каталога (`config/descriptors.go:86`).
- **`StageContentResult` — не приватный протокол** (подтверждено). Спека определяет его как результат `Planner.Run` (`docs/spec/execution/planners.md:175`), то есть это внутренний тип Server. Тем не менее он есть в `stage-content.schema.json` (вторая ветка `oneOf`), в Python (`worker.py:377-393`) и в 4 golden-фикстурах. В production-коде Runtime на Python-модель ссылок нет: она используется только в `contracts/__init__.py`.
- Крупные файлы: `allocation.schema.json` — 1417 строк, `reporting/agent_state.go` — 690 (из них около 250 — десять ручных `UnmarshalJSON`, строки 437–690), `settings.py` — 688, `worker.py` — 544.

## C. Связи, которые не видит archtest

- **SQL в хабе нет** (подтверждено). Зато wire-форма JSON стала форматом хранения: runstore пишет `contracts.StageContentResult` в БД и читает обратно обычным `json.Unmarshal` (`runstore/stage_scan.go:68-85`). Он же затыкает дыру контракта: `normalizeStageResult` подменяет `Artifacts == nil` на `{}` (`runstore/stage_store.go:365-370`), потому что `StageContentResult.Validate` nil не отвергает (`stage_content.go:83-114`).
- Продублированные строковые константы (подтверждено):
  - A2A media types: Go в двух местах (`planner/a2a/client.go:28-29`, `controlplane/runtime_client.go:668-669`), Python в двух (`a2a_server.py:44-45`, `allocation/redaction.py:46-59`). В contracts такой константы нет.
  - Литерал `"contractor/v1alpha1"` повторён в `config/validation.go:56`, `runtimeconfig/model.go:16` и `app/server_config_file.go:16`. Там это версия документов конфигурации — другое понятие под той же строкой.
- Хелперы клонирования типов хаба разбросаны примерно по 18 функциям вне хаба. `runtimeconfig/resolver.go:767-774` дословно повторяет `ResolvedModelPolicy.Clone` (`agent_template.go:77-84`). В `planner/passthrough.go:378-420` пары `clone*`/`Clone*` с однострочной косвенностью, плюс клонирование через JSON round-trip.
- Широких интерфейсов нет: единственный интерфейс — `Validatable` с одним методом. Широк сам пакет. config берёт 47 символов, scheduler — 35, controlplane — 29, planner — 22, а 21 пакет — 3 символа или меньше.
- Экспорты только для тестов: `contractstest` (4 функции, только `_test`-импортёры). Около 37 экспортов используются только внутри своего файла — в основном лимиты scan/llmgateway и типы полей. Экспортов совсем без production-ссылок не нашёл.

## D. Классы дефектов AGENTS §3

### 3.1. Одно правило в нескольких местах

| Правило | Копии | Чем связаны копии |
|---|---|---|
| Бюджеты 1000 / 10 000 / 100 000 000 | Go `agent_template.go:117-121`; схемы `common.schema.json:78-83,113`, `allocation.schema.json:71`, `agent-state.schema.json:38-40`; OpenAPI `13925-13998`; Python `settings.py:346-351,587`, `reports.py:50-52`, `worker.py:166-168`; UI `forms.tsx:304-338`, `model.ts:37-42` | общей фикстуры нет; Go и Python проверяют границы каждый у себя (`control/allocation_test.go:192-196`, `test_contracts.py:435-439`); OpenAPI и UI ничем не закреплены |
| Относительный POSIX-путь | Go `contracts/workspace.go:172-199`, `config/workflow.go:479-505`, `reporting/agent_state.go:409-435`; Python `base.py:228-250` и `115-141` | фикстуры нет; копии уже разошлись (G4) |
| Непустой текст | Go `strings.TrimSpace` в десятках `Validate`; Python `_require_text` / `.strip()` (`base.py:57-60`); схема только `minLength:1` | фикстуры нет; копии разошлись (G5) |
| Правило `audit-results@2` | Go `audit_completion.go:47-99`; Python `contracts/allocation.py:59-75`, `allocation/validation.py:33-36`, `capabilities.py:64-70,236-242`, `toolsets/audit_results/completion.py:47`; дескриптор `config/descriptors.go:86` | `audit-completion-cases.json` |
| Размер StageResult (256 KiB) | `planner/result.go:24-26` через `ResultJSONSize`, `scheduler/stage_finalization.go:84-86` через `json.Marshal` | копии разошлись (G2) |
| Строгость декодирования | `DecodePrivateStrict`, `DecodeStrict`, `strictjson.Decode`, ad hoc `RejectDuplicateKeys` в 4 `UnmarshalJSON`; Python `decode_private` | три уровня строгости (G7) |

Тройное описание протокола. Что проверяет каждая сторона (подтверждено пробами `probe` и `pyprobe.py`):

| Ограничение | Схема | Go | Python |
|---|---|---|---|
| required для скаляров и map | да | **нет**, если `Validate` не отвергает нулевое значение (около 80 полей bool/int вне agent_state) | да |
| enum | да, кроме `lastErrorCode`: в схеме свободная строка (`runtime-report.schema.json:42-48`) | да | да |
| Лимит длины | в символах (`maxLength`) | в байтах | в байтах |
| Ограничение на всю WorkerCompletion (256 KiB) | нет | да | да |
| Неизвестные поля | `additionalProperties:false` везде, кроме `arguments` | да | да (`extra=forbid`) |
| Дубликаты ключей | — | только `DecodePrivateStrict` и 4 типа отчётов | да, весь входящий трафик |
| Строка из одних пробелов | принимает | отвергает | отвергает |
| Зарезервированные namespace артефактов, кросс-полевые инварианты | нет | да | да |

Пример расхождения: результат из 40 000 символов «ж» (80 000 байт) схема принимает, Go и Python отвергают. Схема как оракул слабее обоих реализаций, поэтому дрейф байтовых лимитов она поймать не может.

Покрытие фикстурами:
- Ни одной фикстуры: `audit-worker-completion`, `workflow-transition`. У `execution-config-manifest` и `llm-gateway-config-manifest` проверяются только паттерны имени и версии.
- Нет негативных фикстур: `agent-registration-response`, `runtime-report`.
- `llm-gateway-config` проверяется только в Go (`GO_ONLY_TYPES`, `test_contracts.py:86-91`).
- Ни для одной схемы нет негатива «пропущено required-поле, у которого нулевое значение валидно».

Можно ли генерировать модели. Генерация модели Python из схем (datamodel-code-generator) убрала бы около 410 строк объявлений полей, генерация Go — около 490 тегов. Но останутся около 1260 строк Python-валидаторов и 108 Go-функций `validate`: сортировка, digest, пути, байты, взаимоисключающие варианты. Генераторы плохо понимают `if/then`, `oneOf`, SecretString и различие «поле отсутствует / поле null». Обратное направление (схемы из Go через рефлексию) теряет паттерны и `if/then`, а Python остаётся ручным.

Рекомендую средний путь. Схемы — источник машинных таблиц ограничений (required, enum, лимиты в байтах через расширение `x-maxBytes`) и мутационного набора для всех трёх сторон, а типы остаются ручными. Условие пересмотра решения MVP-002 («code generation is intentionally deferred», `tasks/done/mvp-002-wire-contracts.yml:124`) теперь выполнено: в этом аудите 5 подтверждённых дрейфов. Решение стоит записать в `docs/decisions`.

Используется ли strictjson последовательно: нет (G7). Кроме того, `strictjson.Decode` дубликаты не отвергает (`strictjson.go:22-35`). Утверждение AGENTS §6 о том, что strictjson отвергает дубликаты ключей, для этой функции неверно. В `internal/httpapi/public` нет ни одного вызова `RejectDuplicateKeys` — это вне хаба, публичный API.

### 3.2. Исходы отказов

Классы ошибок кодека (version, duplicate_key, schema, invariant) определены. A2A `invalid_worker_result` — постоянная ошибка (`planner/a2a/client.go:491-495`). Не решено и нигде не записано: путь A2A теряет точность чисел и обнаружение дубликатов (G3).

### 3.3. Ограничение работы

Тяжёлый путь — чтение Worker State: на один изменившийся revision около 300 мс CPU (I1).

### 3.4. Жизненный цикл

У хаба нет сущностей с удалением. Ближайшее — политика «исторические отчёты»: AWC-R114/R115 (`docs/spec/audits/worker-completion.md:694-696`) и снисходительный читатель (G6).

### 3.5. Тесты, которые не могут упасть

- **`make test-wire-cross-language` запускает обе стороны только для 3 из 15 общих фикстур** (`index.json`, `worker-result-size-cases`, `performance-cases`), для `audit-completion` — только Go (подтверждено). Причины: `-run 'Golden|SharedPython|ResolvedSkills'`, `-k 'golden or digest or resolved_skills'`, и только два pytest-файла (`make/core.mk:33-35`). В пакетах llmgateway, runlabels, runtimesettings, scan и contractstest фильтр выбирает 0 тестов, и `go test` молча проходит.
- Кейс `uint64_max` в `worker-result-size-cases.json:103` проверяется через `DecodePrivateStrict` (`worker_result_size_test.go:57`), а production-путь A2A декодирует иначе (G3).
- Python-оракул `jsonschema` использует `re.search`, где `$` срабатывает перед завершающим `\n`. Обход есть только для config identity (`test_contracts.py:699-705`).

### 3.7. Вещи без потребителя

- `audit-worker-completion.schema.json`. Описывает форму kind/stage/agent, а реальный `WorkerCompletionContract` — это kind/task/executionManifest/resultArtifact (`audit_completion.go:13-18`). Схема устарела с V39-001, читателей нет.
- `workflow-transition.schema.json`. Спека называет её частью валидации (`configuration/workflows.md:648-650`, `platform/limits.md:713`), но её не исполняет ни код, ни тест — спека и код расходятся.
- Python `StageContentResult` и `StageOutcome`.
- Наследие совместимости, которое README узаконивает:
  - `StageContentRequest.deadline` «optional for older callers» (README:14), хотя все три продюсера её выставляют (`passthrough.go:312`, `scan/run.go:240`, `streamline/tools.go:401`);
  - пустой `llmGatewayToken` принимается (README:40-41), хотя Server пустой токен не отправляет (`scheduler/execution_preparation.go:383-386`).

## E. Цена изменений

1. **a98289ab** `fix(contracts)`: длина media type до 255 (V324-001). 9 файлов: kernel (contracts), domain (auditdomain), Python kernel и toolset, приватная схема, OpenAPI, фикстура, спека. Само правило — одна строка (`media_type.go`). Остальные правки вынуждены дублированием: удаление собственных 127-символьных пределов в `auditdomain/package.go:262` и `security_findings/collection.py:265`, а также синхронизация с `MEDIA_TYPE_PATTERN`, схемой и OpenAPI.
2. **a57162ff** `feat(contracts)`: failure signatures на LLMGatewayConfig. 24 файла, +826 строк. Слои: kernel, catalog, execution (scheduler), edge (OpenAPI, сгенерированные клиенты), Python. Логика фичи — около 64 строк (`config/llm_gateway.go` +57, scheduler +7). Одна и та же грамматика записана 4 раза, примерно 364 строки: Go `llm_gateway.go` +119, Python `settings.py` +105, три схемы +108, OpenAPI +32. Форма digest — дважды (Go `config/digest.go`, Python `digests.py` +21). Python-digest позже удалили как GO_ONLY, то есть мёртвый код сначала отгрузили, а потом выкосили.
3. **ba329bdd** `feat(config)`: инструкции для summarizer. 23 файла, +466 строк. Вынужденные правки: валидация трижды (Go `allocation.go` +19, `common.schema.json` +6, Python `contracts.py` +5), digest дважды (Go +5, Python +13), ручной клон Summarizer в `controlplane/registry.go` +4 (позже заменён на `ResolvedAgentTemplate.Clone`), проза README +7, фикстура +99. Это около половины не-тестовых строк.

## F. Рекомендации (по соотношению ценности к трудоёмкости)

1. **Одно определение границ StageResult.**
   - Проблема: G2; `planner/result.go:24-26` и `scheduler/stage_finalization.go:84-86` считают размер по-разному; `stage_content.go:83-114` не проверяет nil и лимиты.
   - Изменение: перенести nil-артефакты, лимит summary, число артефактов и `ResultJSONSize` в `StageContentResult.Validate`; удалить обе внешние проверки и `normalizeStageResult`.
   - Эффект: одно определение вместо трёх.
   - Трудоёмкость S, риск низкий.
   - Закрепить: тест summary из 65 536 символов «<» — `Validate` обязан совпадать на всех путях; скрипт в `make lint`, который запрещает сравнивать `json.Marshal` с `contracts.Max*Bytes`.
   - Тип: дефект.
2. **Один строгий декодер с проверкой наличия полей.**
   - Проблема: G1, G7; `common.go:89-103`, `runtime_client.go:310,361`, `agent_state.go:437-690`.
   - Изменение: `DecodeStrict` получает семантику `DecodePrivateStrict` (дубликаты); добавить проверку наличия required-полей по таблице, выведенной из `required` схем.
   - Эффект: закрыты около 80 молча обнуляемых полей; минус около 200 строк шаблонного кода.
   - Трудоёмкость M, риск средний (затронуты все декодеры Runtime→Server).
   - Закрепить: мутационный набор из п. 3.
   - Тип: дефект.
3. **Мутационный набор из схем и честный gate.**
   - Проблема: 3.5 выше; ни одного негатива «пропущено required-поле».
   - Изменение: для каждого типа из index генерировать мутации — удаление required, выход за enum, длина на 1 больше лимита в байтах и в символах, ключи U+001C и `ж:`, дубликат ключа, лишнее поле — и прогонять на схеме, Go и Python. `test-wire-cross-language` должен запускать все тесты, которые читают `api/testdata` (соглашение имён `TestShared*` / `test_shared_*`), и падать, если тестов 0.
   - Эффект: любое из найденных расхождений G1–G5 становится красным тестом.
   - Трудоёмкость M, риск низкий.
   - Тип: структура.
4. **Общие фикстуры для правил на двух языках.**
   - Проблема: G4, G5, бюджеты, media types.
   - Изменение: `workspace-path-cases.json` (один параметризованный предикат на язык), `blank-text-cases.json` (Python использует набор пробелов Go), `limits.json` (бюджеты; читают Go, Python, тест OpenAPI и тест UI); константы A2A media types — в contracts.
   - Трудоёмкость S–M.
   - Тип: дефект (3.1).
5. **Убрать сирот и снисходительные пути.**
   - Изменение: удалить `audit-worker-completion.schema.json`; либо исполнять `workflow-transition.schema.json` в тестах config, либо удалить её и поправить спеку; сделать `deadline` обязательным; убрать приём пустого токена; убрать Python `StageContentResult` и `StageContentResult` из приватного каталога. AWC-R115 (отбрасывать неизвестные kind/phase) противоречит AGENTS §2 и схеме (`allocation.schema.json:254-264`) — это решение за пользователем.
   - Трудоёмкость S, риск низкий.
   - Тип: структура.
6. **Вывести домен из kernel.**
   - Изменение: pinned selection → общий доменный пакет; `AuditTask*` и `FindingCollectionMediaType` → auditdomain; `RunSkillSnapshot` → agentskills; правило `audit-results@2` → Toolset-дескриптор; `StageContentResult` → planner; вызывающим `MarshalPrivateCanonical` перейти на `strictjson.Canonical`; RuntimeAdapter и CredentialKind → runtimesettings.
   - Эффект: число импортёров ядра без wire-символов с 19 до 0.
   - Трудоёмкость M–L, риск средний (конфликты слияния).
   - Закрепить: allowlist импортёров contracts в archtest.
   - Тип: структура.

## G. Баги

- **G1. Средняя.** Подтверждено: проба Go — ACCEPTED, Python и схема — rejected.
  - Go молча обнуляет отсутствующие required-поля WorkerCompletion и StageContentResult. `failure` без `retryable` → `false`, то есть повторяемая ошибка Worker становится постоянной. Так же проходят `result` без `summarized`, `observations` без `truncated`, StageContentResult без `artifacts`.
  - Код: `worker_completion.go:55-79,187-212`, `stage_content.go:83-114`, плюс компенсация в `runstore/stage_store.go:365-370`.
  - Тестов нет.
- **G2. От низкой до средней.** Подтверждено (`sizeprobe`).
  - Один и тот же StageResult с summary из 65 536 символов «<» проходит проверку Planner (`ResultJSONSize` = 65 622 байта) и не проходит проверку Scheduler (`json.Marshal` = 393 302 байта > 262 144). Scheduler прерывает Stage с `result_contract_violation, Retryable:false` (`stage_finalization.go:23-33`) уже после того, как Worker израсходовал бюджет.
  - Это пропущенный соседний путь V60-026: в её scope прямо сказано «mapped StageResult boundaries» (`tasks/done/v60-026…yml:30`). `introduced_by` — V60-026.
  - Тестов нет.
- **G3. Низкая.** Подтверждено.
  - A2A SDK кладёт DataPart в `*any` с числами как float64 (`a2a-go@v2.5.0/a2a/core.go:686,738`), а клиент заново кодирует его в JSON (`planner/a2a/client.go:485-491`).
  - Следствия: `stateRevision` 2^53+1 молча превращается в 2^53; 2^64−1 отвергается, хотя общая фикстура считает его валидным; дубликаты ключей не видны в принципе.
  - Нарушает и комментарий `result_limits.go:80` («does not decode through float64»), и scope V60-026. Тест `planner/a2a/result_size_test.go` проверяет только HTML-экранирование.
- **G4. Низкая.** Подтверждено: Go — ACCEPTED, Python — rejected.
  - Цель workspace `ж:docs` Server принимает и при публикации Workflow (`config/workflow.go:479-505`), и в allocation (`workspace.go:195`). Runtime отвергает её: `isalpha()` для не-ASCII (`base.py:248`), из-за чего подготовка allocation падает. Внутри самого Python второй предикат проверяет `isascii()` (`base.py:135-137`).
- **G5. Низкая.** Подтверждено.
  - U+001C…U+001F: в Go это не пробел, в Python — пробел. `StageContentRequest` с `objective="\x1c"` проходит Go-валидацию и отвергается Runtime (`base.py:57-60`). То же касается всех проверок «непустое» в обе стороны.
- **G6. Дефект политики.** Подтверждено.
  - Диагностику с неизвестными kind/phase Go и Python отбрасывают (`worker_completion_diagnostics.go:45-63`, `base.py:95-112`), а схема её отвергает (`const` и `enum`).
  - Последствие: опечатка Runtime в phase (например, «Sealed») делает диагностику невидимой вместо ошибки.
- **G7. Низкая, плюс дефект документации.** Подтверждено.
  - Ответы Runtime→Server принимают дубликаты ключей: AgentStateSnapshot (`runtime_client.go:310`), Prepare/Finalize/Abort (`runtime_client.go:361`, частично спасают `UnmarshalJSON` в `telemetry.go:107,183`), WorkerCompletion (G3).
  - Python отвергает дубликаты на всём входе (`codec.py:35-37`).
  - Утверждение AGENTS §6 для `strictjson.Decode` неверно.

## H. Упрощение

| Что | Удалить или слить | Экономия | Риск |
|---|---|---|---|
| 10 `UnmarshalJSON` в agent_state | общий декодер с проверкой наличия (F2) | ≈150–200 строк | низкий при F3 |
| `DecodeStrict`, `DecodeStrictObject`, вложенные `RejectDuplicateKeys` | один декодер | ≈40 строк | средний |
| 3 Go- и 2 Python-копии предиката пути | по одной параметризованной функции на язык | ≈50 Go + 25 Python | низкий |
| `MarshalPrivateCanonical` | прямой `strictjson.Canonical`; auditpriority теряет импорт ядра | 7 строк, 1–5 рёбер импорта | низкий |
| Клоны вне хаба | удалить дубль `resolver.go:767-774`, слить пары в `passthrough.go:378-420`, убрать `cloneSnapshot` | ≈30 строк | низкий |
| Python `StageContentResult`, ветка схемы, 4 фикстуры | убрать из приватного каталога | ≈60 строк | низкий |
| Orphan-схемы | удалить, либо исполнять в тестах | 24 + 103 строки | низкий |
| test-contracts / verify-wire-contracts / test-wire-cross-language | одна цель (F3) | ≈10 строк | низкий |
| Необязательный deadline, пустой токен, отброс по AWC-R115 | обязательные поля и строгий отказ | ≈30 строк, минус ветвления | низкий; AWC-R115 — за пользователем |

## I. Оптимизация

- **I1. Чтение Worker State.**
  - Путь: инструмент Planner → `stateview.load` → `ReadWorkerState` → `DecodeStrict` + `Validate` → `cloneSnapshot` (`stateview/service.go:383-391, 560-566`).
  - Кардинальность: по контракту до 10 000 scope paths, 10 000 interactions, 4 MiB.
  - Замер scratch-программой, один прогон на этой машине. Снимок 1,98 МБ с 10k путей: `DecodeStrict` + `Validate` — 141 мс против 18 мс обычного разбора в map (в 7,7 раза медленнее); `cloneSnapshot` (marshal + повторный strict decode) добавляет ещё 156 мс.
  - Итого около 300 мс CPU на каждый изменившийся revision в каждом вызове инструмента Planner; ETag-кэш спасает только неизменённые revision.
  - Причины: каждый вложенный `UnmarshalJSON` разбирает своё поддерево 2–3 раза, а `Validate` заново кодирует весь снимок.
  - Исправление: убрать клон (минус около 50%) и проверять наличие полей за один проход (F2). Ожидаю меньше 60 мс.
- **I2.** Вложенные `RejectDuplicateKeys` (RuntimeReport, AllocationFinalReport, ExecutionReport) сканируют одни и те же байты на каждом уровне. Ответ ограничен 1 МиБ, поэтому эффект мал, и он исчезает вместе с F2. Отдельно не рекомендую.
- **I3.** Повторное кодирование в `Validate` для подсчёта размера (до 256 KiB) — микрооптимизация, не рекомендую. Правка ядра пересобирает 58 из 99 пакетов — это цена цикла разработки, а не runtime.

## Топ-3

1. G2 и F1: Scheduler окончательно отвергает StageResult, который принял Planner, из-за HTML-экранирования (`stage_finalization.go:84-86`) — пропущенный соседний путь V60-026, исправление размера S.
2. G1 и F2: Go молча обнуляет отсутствующие required-поля протокола (`retryable`, `summarized`, `artifacts`), а схема и Python их отвергают — один строгий декодер с проверкой наличия, M.
3. F3: gate `test-wire-cross-language` гоняет обе стороны лишь для 3 из 15 общих фикстур, и ни одна фикстура не мутирует required, байтовые лимиты или пробелы — мутационный набор из схем поймал бы G1–G5.
