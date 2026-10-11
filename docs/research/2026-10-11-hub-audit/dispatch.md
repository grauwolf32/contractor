# Аудит узла «Планирование и запуск» (scheduler · planner/* · controlplane · gatewayrecovery)

Аудит проведён на дереве 02bbdcbf (2026-10-11), только чтение. БД не использовалась. Два вывода я подтвердил unit-тестами без БД: тестовые файлы подключались через `go test -overlay`, сами файлы лежат в `…/scratchpad/hub-audit/dispatch/overlay/`, в репозиторий ничего не записано. Метки: **VERIFIED** — путь по коду прослежен или тест прогнан; **SUSPECTED** — для подтверждения нужен тест на PostgreSQL или прогон.

## A. Карта узла

Команды: `go list -f '{{.ImportPath}}|{{join .Imports " "}}|…' ./...`, `wc -l` по не-тестовым `*.go` (генерированных файлов в узле нет), парсер на go/ast в dispatch/expcount (считает экспортированные идентификаторы и самые широкие типы), `git log --no-merges --since=2026-08-12 [--name-only] -- <pkg>`. Партнёры по совместным изменениям посчитаны так: файлы, которые менялись в тех же 198 коммитах, что и узел.

| Пакет | LOC не-тест / тест | Экспорт | Fan-in, импортёры в prod | Fan-out (internal) | Коммиты с 08-12 (вкл. тесты / только не-тест) |
|---|---|---|---|---|---|
| scheduler | 5007 / 9277 | 54 | 1: app (тест: memory) | 19, в т.ч. runstore (65 символов), controlplane, planner, persistence/postgres, settingsstore | 112 / 84 |
| controlplane | 5145 / 5543 | 151 | 6: app, findingintake, httpapi/privateartifacts, httpapi/public, public/events, scheduler | 19, в т.ч. **planner**, runstore, runtimeconfig, persistence/postgres | 93 / 79 |
| gatewayrecovery | 857 / 154 | 29 | 6: app, privateartifacts, public, planner, streamline, scheduler | 3 | 11 / 11 |
| planner (корень) | 1810 / 1175 | 125 | 12: app, auditimport, auditscan, controlplane, public, a2a, router, scan, session, stateview, streamline, scheduler | 10 | 42 / 35 |
| planner/a2a | 551 / 1862 | 6 | app | 4 | 12 / 8 |
| planner/router | 67 / 751 | 10 | app | 3 | 13 / 6 |
| planner/session | 1657 / 1594 | 27 | app, auditimport, streamline | 8 | 26 / 20 |
| planner/stateview | 570 / 302 | 22 | streamline | 3 | 2 / 2 |
| planner/streamline | 3202 / 3381 | 31 | app, router | 11 | 40 / 32 |
| **Итого** | **18 866 / 24 039** | **455** | 8 внешних prod-импортёров | | **198 из 2245 коммитов (8,8%)** |

Топ-10 не-тестовых файлов по числу коммитов: controlplane/registry.go 45, scheduler/scheduler.go 41, controlplane/runtime_client.go 35, scheduler/types.go 24, streamline/planner.go 22, controlplane/allocation.go 21, scheduler/postgres.go 19, planner/passthrough.go 19, streamline/tools.go 17, controlplane/placement.go 17.

Пересмотр гипотез прошлого обзора:
- scheduler.go **больше не god-файл**. 2026-09-19 в рамках V45-005 его разрезали с 2332 до 186 строк (сейчас 253). После этого он менялся 6 раз. С 09-20 изменения распределены по stage_finalization (10), types, stage_preparation, postgres, execution_preparation (по 9).
- Главный горячий файл сейчас — registry.go: 1504 строки, 15 коммитов с 09-20.
- Счёт символов contracts подтверждается: controlplane использует 29, scheduler 35.
- Тест V62 claim-lease заново не мерил: ему нужна БД.

Партнёры по совместным изменениям вне узла (из 198 коммитов): docs/spec — 66, runtime/ (Python) — 59, internal/httpapi — 58, internal/contracts — 52, api/ — 51, internal/runstore — 48, internal/config — 43, ui/ — 30. Файлы: httpapi/public/types.go 31, runstore/types.go 28, runstore/store.go 27, OpenAPI 27, ui generated public.ts 25, runtime contracts.py 23. Коммит узла затрагивает в медиане 15 файлов, p90 — 47.

## B. Обязанности

**scheduler.** У пакета 9 причин меняться:
1. Лейны и claim: supervisor.go, scheduler.go.
2. Жизненный цикл Stage: stage_preparation, stage_finalization, stage_termination, run_execution.
3. Политика Workflow, то есть retry и escalation: workflow.go, buildProgression.
4. Проверка размещения и release: allocation_reservations, allocation_release.
5. Материализация RuntimeSettings и credentials: execution_preparation.go, 440 строк.
6. Телеметрия и отчёты, плюс ретенция метрик: reports, planner_telemetry, supervisor.go:278-305.
7. Допуск через gateway recovery: gateway_recovery.go.
8. Требования Audit completion: audit_completion.go.
9. SQL по чужим таблицам: postgres.go.

Широкие типы:
- `Options` — 20 полей; `Store` — 18 методов; `Scheduler` — 16 полей; `AtomicPersistence` — 9 методов.
- `Store` и `AtomicPersistence` в production реализованы по одному разу. Нужны они ради фейков.
- Ретенция телеметрии (supervisor.go:278-305) проходит транзитом scheduler → runstore.CleanupExpiredTelemetry → telemetry.Repository (runstore/report_store.go:~90). Ей место в собственном maintenance-цикле telemetry.

**controlplane.**
- registry.go совмещает: регистрацию и heartbeat, confirmed lease, резервирование рёбер, write fence, детекцию потерь, удаление и метки principal, журнал изменений Operations.
- `agentEntry` — 23 поля, `InMemoryRegistry` — 19.
- placement.go (800 строк) резолвит RuntimeConfig по SQL-каталогам runtimeconfig и закрепляет `stage_allocations` (таблица runstore) в своей транзакции. По сути это планировщик размещения со стороны scheduler.
- operations.go — read-модель для public API вместе с собственной валидацией.

**planner.**
- Корень: интерфейсы, passthrough-planner, модель плана.
- session: персистентная ADK-сессия поверх runstore.
- stateview: проекция состояния Worker.
- streamline: Go-planner на ADK-Go, его tools и собственный OpenAI-адаптер (gateway.go, 347 строк; аналог Python llm/openai.py).
- a2a: вызов Worker.

**gatewayrecovery** классифицирован как platform, но меняет `workflow_runs.state`, то есть таблицу слоя execution (см. C).

## C. Связи, которые не видит archtest

1. **SQL по чужим таблицам** (VERIFIED):
   - scheduler/postgres.go:124, :407 — FOR UPDATE по `workflow_runs`; :257-266 — DELETE из `gateway_recovery_waits`/`gateway_run_routes` (это таблицы gatewayrecovery) и UPDATE `workflow_runs.state`; :366-395 и postgres_sql.go:10-21 — массовые переходы `stage_executions`; postgres_sql.go:27-38 — чтение `artifact_bindings/revisions/versions` (таблицы artifacts); admission.go:35 — UPDATE `stage_executions`.
   - controlplane/placement.go:356-366 — FOR UPDATE по `stage_executions`, проверка `state='preparing'`.
   - gatewayrecovery (слой platform) пишет `workflow_runs.state`: waiting.go:17-21, :29-33, admission.go:64-67, читает его в store.go:62, :73. По SQL это зависимость вверх по слоям.
2. **Обращения во внутренности чужого пакета.** controlplane импортирует planner только ради алиасов `WorkerStateReadResult/Error` и проверки интерфейса (runtime_client.go:36-39). Эти типы стоит перенести в contracts/reporting, тогда ребро controlplane→planner исчезнет.
3. **Широкий интерфейс.** `controlplane.Registry` — 15 методов (registry.go:30-46). Единственный потребитель, `NewHTTPHandler`, вызывает 3: RegistrationResponse, RegisterAuthenticated, HeartbeatAuthenticated (controlplane/http.go:39, 67). Остальные потребители берут узкие интерфейсы: placement — конкретный `*InMemoryRegistry`, 12 методов; runtime_batch — `AllocationRegistry`, 6; principal_operations — 2; privateartifacts — GetGrant и WithWriteGrant. VERIFIED.
4. **Экспорт только для тестов** (VERIFIED, 0 вызовов в prod): `InMemoryRegistry.Register`, `Heartbeat`, `ReserveAll`, `GetAgent` (registry.go:333-347, 904-915). При этом ReserveAll и GetAgent входят в интерфейс Registry. Через `ReserveAll` 10 тестовых файлов идут в обход durable-pinning. `Scheduler.RunOnce` спецификация разрешает «для тестов и recovery tools» (SCH-R125), но recovery tools не существует.
5. **Дублированные строковые константы** — см. D-3.1.

## D. Классы дефектов (AGENTS §3)

**3.1 Одно правило в нескольких местах.**
- Переходы Run running↔waiting (VERIFIED). Белый список в Go — runstore/store.go:661-672. Но фактически переходы пишут сырым SQL: gatewayrecovery/waiting.go:17-21, :29-33 и scheduler/postgres.go:264-265. Причины расходятся: gatewayrecovery ставит `gateway_recovered`, scheduler оставляет старое `Waiting for the model gateway`. В public API state reason не попадает, поэтому эффект только внутренний.
- Множество активных состояний Run литералом в SQL (VERIFIED): runstore/store_sql.go:14, 90, 147, 160; store.go:585; queue_store_sql.go:22; projectlifecycle/controller_sql.go:38, controller.go:291; credentials/runtime_repository.go:235; gatewayrecovery/store.go:65, admission.go:59, waiting.go:55; scheduler/postgres.go:131. В Go — runstore/types.go:44. Общего определения нет.
- Терминальные переходы Stage с кодами `termination_committed`/`result_accepted` (VERIFIED): runstore/stage_store.go:286-294 и stage_store_sql.go:107-119, а их копии с другим условием WHERE — в scheduler/postgres.go:371-395 и postgres_sql.go:10-21.
- Захват Run FOR UPDATE с проверкой состояния — три копии (VERIFIED): scheduler/postgres.go:122-138, :398-417 и gatewayrecovery/store.go:60-69.
- Toolset capability (VERIFIED): wire-схема agent-registration.schema.json:141-148 (без maxItems, любая строка); Go contracts/control/agent.go:105-125 (лимита на число tools нет, имя — любое непустое); Python registration.py:31-42 (то же); controlplane/operations.go:24, 30, 268-275 (≤256 и регулярка `^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`); OpenAPI 14250-14255 и ConfigId 10972-10976. Коммит V323-001 (1dcadfc5) заявлял, что «копии в controlplane уходят», но `operationsConfigIDPattern` (operations.go:30) остался. Последствие — баг G3.
- Классификация ошибок Gateway (Go failure.go:36-118 и Python errors.py:31-82) **синхронизирована**: общий фикстур api/testdata/v1alpha1/gateway-failure-classification-cases.json (39 кейсов), его читают failure_test.go и test_gateway_error_classification.py. **Протокол участника recovery без общего фикстура.** Это набор action (requests.go:32-41), коды `ValidFailure` (policy.go:60-66), правило abandoned и повторная доставка при отмене (Go http.go:35-86 против Python client.py:180-245). В части отмены реализации уже разошлись (G4). Гипотезу «recovery реализован дважды» уточняю: authority один (Go и PostgreSQL), а продублированы участник протокола и OpenAI-адаптер.
- Значения таймаутов по умолчанию в трёх местах: scheduler.go:17-28 (operation 15s, planner 45s, claim 2m), runtime_batch.go:67-68 (cleanup 10s), app/operational_settings.go:116-122 (30s/30s), плюс таблица в docs/operations/timeout-configuration.md:19-22. ClaimDuration, PollInterval и LeaseScanInterval существуют только как умолчания пакета: в ServerConfig их нет, в документации тоже.

**3.2 Исход операции не решён.**
- Лайфциклические коммиты (CommitResult/Termination, EnterFinalizing/Aborting, FailRunWithActiveStages) защищены только CAS по состоянию Run/Stage. В условие коммита claim не входит. Единственная запись, огороженная claim, — журнал scan (runstore/planner_store_sql.go:29-30).
- Безопасность держится на advisory lease одного активного Server (app/single_active.go:21-31, 141-178: ping с допуском 10s меньше keepalive 30s). Это принятый риск без записанного владельца и без теста «два владельца не коммитят» (§3.9). VERIFIED.
- Таймауты:
  - `scheduler.operationTimeout` обслуживает около 25 вызовов. В том числе Runtime PrepareAll (stage_preparation.go:222) и Runtime ReleaseAll (allocation_release.go:48-50). Из-за последнего `runtimeLifecycle.cleanupTimeout` (runtime_batch.go:216-217) молча обрезается до operationTimeout, хотя документ отдаёт этому параметру «one release batch» (timeout-configuration.md:22) и требует раздельных бюджетов (OPS-R15).
  - `plannerTimeout` одновременно задаёт стену Stage (stage_preparation.go:245-249), лимит запроса модели Planner и аренду probe (composition_execution.go:125).
  - VERIFIED; вред при текущих умолчаниях SUSPECTED.
- Probe lease при отмене или фенсинге — G1, G2, G4.

**3.3 Ограничение работы.**
- Терминальный release ограничен (stage_store_sql.go:22 LIMIT), cleanup телеметрии ≤10k, snapshot Operations ≤10k.
- Цена renewal не ограничена разумно — см. I-1.
- N+1 при чтении истории сканов: scan_history.go:71 вызывает GetPlannerSession на каждый Stage, хотя есть батч GetPlannerSessions (plans_batch.go:27). Кардинальность мала.

**3.4 Жизненный цикл.**
- `gateway_recovery_routes` никогда не удаляются (DELETE по ним нет; FK из 4 таблиц без cascade).
- `gateway_recovery_failures` — одна квитанция на каждый failed-запрос, живёт до удаления Run.
- Probe не освобождается при завершении Stage (G1).
- Waits и routes не чистятся в путях FailRunWithActiveStages, CommitTerminationAndFinishRun, AcceptResultDuringCancellation (postgres.go:269-401). Вреда нет: Admit и Status фильтруют по активным Run.

**3.5 Тесты, которые не могут упасть / фейки слабее реального хранилища** (VERIFIED):
- `memorySchedulerStore.TransitionRun` принимает любой переход (scheduler_test.go:2102-2117); реальный проверяет (runstore/store.go:661-672).
- `RecordStageAllocation` в фейке не валидирует ничего (2159-2173); реальный — allocation_store.go:18-83.
- `memoryAtomicPersistence.EnterAbortingWithTermination` пускает только running/cancelling (2335); реальный — pending/running/waiting/cancelling (postgres.go:131).
- Тест V290 отключает production-поведение фейкового размещения (paused_admission_release_test.go:24-30, хук `record` задокументирован как «mirrors production placement»). Поэтому тест не может поймать G2.
- Python-тесты recovery работают на фейковом transport. Межпроцессного теста «огороженная (fenced) аллокация освобождает probe» нет.

**3.7 Экспорт без production-вызова** — см. C-4.

## E. Цена изменений

| Коммит | Файлы / пакеты / слои | Что потребовала структура, а не сама фича |
|---|---|---|
| **8112e65a** fix(runs): stop recovery from cutting… | 23 файла: gatewayrecovery (6), app (2), docs (4), тесты privateartifacts, streamline, scheduler; runtime llm/client.py, recovery.py и 2 Python-теста. Слои platform, execution, edge, composition + Python | Правило «abandoned → released, повторно не отправлять» пришлось писать дважды: Go http.go:127-182 и Python client.py:106-245. Общего фикстура протокола нет, два набора тестов правились отдельно. Здесь же появилась аренда probe «max из двух таймаутов», от неё растёт G1 |
| **84f0b406** fix: keep Gateway Run routes scoped to current Stage | 3 файла: gatewayrecovery/admission.go (platform) и scheduler/postgres.go (execution) + тест | Таблицей `gateway_run_routes` владеет gatewayrecovery, но scheduler удаляет из неё сырым SQL (postgres.go:261). Фикс пришлось делать в двух пакетах на одной таблице |
| **1dcadfc5** fix(contracts): one config id/version grammar (V323-001) | 35 файлов: contracts, config, runtimeconfig, controlplane, auditstandards, runstore, mtls, httpapi, сгенерированный клиент, OpenAPI, JSON-схемы, Python, UI | Валидация имён capability живёт в read-модели controlplane, а не в contracts/control. Копия operations.go:30 уцелела, отсюда G3 |

Для контраста — 22cb4f47 (V290-001): 2 файла, изменение локальное, но в production неверное. Фейк не воспроизводит durable-pinning (G2).

## F. Рекомендации (по отношению ценность/трудоёмкость)

1. **Освобождать probe при завершении Stage и при фенсинге** (дефект; S–M; риск низкий).
   - Проблема: G1.
   - Изменение: в той же транзакции, где `clearStageGatewayWaits` (postgres.go:257-266), а также в FailRunWithActiveStages и CommitTerminationAndFinishRun обнулять `probe_*`, если `probe_run_id = run`. Пускать `released`/`finished` для fenced-гранта (privateartifacts/gateway_recovery.go:21-25). Аренду probe сделать по участнику (worker 180s против planner 30m), вместо `max` (composition_execution.go:125, requests.go:148, admission.go:82).
   - Эффект: блокировка маршрута после abort падает с ≈30 мин до ≤InitialDelay.
   - Закрепить: тест на PostgreSQL «Worker держит probe → Stage abort → другой Run получает probe в пределах 1s».
2. **Одна граница для toolset capability** (дефект; S; риск низкий).
   - Проблема: G3.
   - Изменение: перенести лимиты 256 и грамматику ConfigId в contracts (`ValidCapabilityToolName`). Применять её в registration.go, в схеме agent-registration и в Python registration.py. operations.go:30 удалить.
   - Закрепить: фикстур `api/testdata/v1alpha1/runtime-capability-cases.json` для Go, Python, OpenAPI плюс свойство «каждая принятая регистрация даёт валидный snapshot».
3. **Исправить путь V290 и убрать двойную запись аллокаций** (дефект + §3.5; M; риск средний).
   - Проблема: G2, H-1.
   - Изменение: при `ErrQueuePaused` для неадмитированного Stage удалять durable-строки в той же транзакции, что и Release, либо не освобождать, а помечать «retained». Удалить `scheduler.recordReservations` (allocation_reservations.go:120-153 и вызов в stage_preparation.go:188-195). Фейку memoryAllocator сделать pinning обязательным (сейчас необязательный хук).
   - Закрепить: integration-тест V290 с реальным PlacementAllocator.
4. **Один владелец для записей в workflow_runs, stage_executions и gateway_*** (структура; M; риск средний).
   - Проблема: C-1, D-3.1.
   - Изменение: FailRunWithActiveStages, переходы waiting/running и lockRun вынести в runstore API; scheduler вызывает gatewayrecovery API для waits/routes/probe.
   - Эффект: −3 копии SQL-переходов, −3 копии захвата блокировки.
   - Закрепить: расширить archtest картой «таблица → пакет-владелец». Тест сканирует строки `UPDATE|INSERT INTO|DELETE FROM <table>` в `*.go` и падает на писателе, который не владеет таблицей.
5. **Сократить renewal claim** (оптимизация; S; риск низкий). См. I-1. Закрепить: тест с подставным clock — не больше 3 renew за минуту на лейн.
6. **Сузить Control Plane API и убрать мёртвые умолчания** (структура; S).
   - Изменение: Registry сократить до 3 методов; Register, Heartbeat, ReserveAll, GetAgent перенести в export_test.go. Удалить умолчания пакетов для параметров, которые конфигурирует app (scheduler.go:17-28, runtime_batch.go:67-68). ClaimDuration и PollInterval внести в ServerConfig и в таблицу таймаутов.
   - Закрепить: скрипт в `make lint`, который находит экспортированные методы с вызовами только из `_test`.

## G. Баги

**G1. Probe маршрута держится до ≈30 мин после abort или cancel. Ручной Retry не помогает.** Серьёзность: высокая (локальные модели часто падают по OOM).
- Путь: стадия прерывается (cancel, deadline, потеря аллокации). beginTermination сначала ставит write fence (stage_termination.go:57-59), потом AbortAll.
- Python-участник при отмене повторно шлёт `released` (client.py:224-227 shield → `_release_probe` :274-291). Server отвечает 409, потому что грант fenced (privateartifacts/gateway_recovery.go:21-25), а `_release_probe` глотает ошибку.
- При коммите Stage `clearStageGatewayWaits` чистит только waits и routes (postgres.go:257-266). `probe_*` вне gatewayrecovery нигде не обнуляется — проверено grep.
- Аренда probe равна `max(workerRequestTimeout, plannerTimeout)` + 1s, то есть ≈30m01s (composition_execution.go:125, requests.go:146-148).
- Другие Run получают отказ из-за `activeProbe` (requests.go:135-145, admission.go:62). За это время истекает automaticWindow (5m), а `Retry` сохраняет probe (store.go:79-98).
- Нарушен SCH-R108: «so the leased probe is not held until it expires». Проверка: каждое звено VERIFIED, сквозной сценарий SUSPECTED (нужен тест на PostgreSQL). Тестом не покрыто: TestGatewayRecoveryAutomaticWindowManualRetryAndCancellation не проверяет освобождение probe для других Run. Аренда «max из двух» появилась в 8112e65a.

**G2. Путь V290 (пауза владельца между probe и AdmitStage) в production не переразмещает Stage.** Серьёзность: средняя (гонка узкая, но последствие детерминировано).
- Путь: placement durable записывает `stage_allocations` в транзакции закрепления (placement.go:368-392). AdmitStage падает с ErrQueuePaused. `releaseUnprepared` освобождает только гранты в реестре (stage_preparation.go:177-186). Строки `stage_allocations` нигде не удаляются (`DELETE FROM stage_allocations` вне тестов нет).
- После resume: `liveOrNewReservations` видит записи без грантов и возвращает `errControlPlaneStateLost` (allocation_reservations.go:27-40). Дальше beginAbort `control_plane_state_lost` (stage_preparation.go:156-160): Stage прерывается, тратится попытка retry, а без политики retry падает весь Run.
- Тест V290 этого не видит: он выключает pinning (paused_admission_release_test.go:24-30).
- **VERIFIED overlay-тестом** `TestAuditPausedAdmissionWithDurablePlacement`. С хуком, который ведёт себя как production, первый RunOnce даёт deferred при 1 durable-строке и 0 грантах. Повторный RunOnce даёт «optimistic state conflict», reserves остаётся 1, переразмещения нет. В фейке вместо abort получаем conflict, потому что фейк уже реального (scheduler_test.go:2335).
- В тот же путь попадает gateway Admit: он резервирует probe для Run (admission.go:77-85), а повторный Admit этого же Run отклоняется его собственной резервацией (admission.go:62 — исключения для своего Run нет) на requestTimeout+MaxDelay ≈ 30m30s. Эта часть SUSPECTED.
- Внёс 22cb4f47.

**G3. Одна регистрация Runtime Agent ломает весь Operations API.** Серьёзность: средняя.
- Регистрация принимает toolset с больше чем 256 tools или с именем вида `server:tool` (agent.go:116-125, схема :141-148, Python :38-41).
- `SnapshotOperations().Validate()` отклоняет такой snapshot целиком (operations.go:268-275). Обработчик возвращает ошибку на всех трёх списках Operations (operations_handlers.go:21, 42, 83, 115-119).
- **VERIFIED overlay-тестом**: «257 tools: registration accepted; snapshot.Validate() = invalid … tool collection bound», «colon tool name: … invalid Runtime Agent tool capability name».
- Тестом не покрыто. Регрессия класса V323-001.

**G4. Go-участник Planner при отмене не освобождает probe и классифицирует отмену как сбой gateway.** Серьёзность: низкая–средняя.
- `context.Canceled` превращается в `gateway_unavailable`, retryable (http.go:159-164). `record` идёт с уже мёртвым контекстом запроса (http.go:67), поэтому повторной доставки нет.
- Python делает shield с отдельным таймаутом 5s (client.py:32, 226).
- Спецификация говорит только о «Runtime call» (SCH-R108): для Planner правило не записано, это пробел спецификации. VERIFIED по коду.

**G5. Бюджет release обрезан чужим таймаутом.** Серьёзность: низкая. `cleanupTimeout` эффективно равен `min(cleanupTimeout, operationTimeout)` (allocation_release.go:48 + runtime_batch.go:217). Документ это противоречит (timeout-configuration.md:22). VERIFIED.

## H. Упрощение

| Что убрать или слить | Экономия | Риск |
|---|---|---|
| `scheduler.recordReservations` дублирует закрепление в placement (placement.go:376 при записи с 2026-09-01, bd5e3b6a; scheduler пишет с 08-29). Удалить и заставить фейк делать pinning | ~45 строк, 2N SQL на каждое размещение | средний (нужны правки фейков) |
| FailRunWithActiveStages с сырым SQL (postgres.go:366-395, postgres_sql.go) — заменить run-scoped методами runstore | ~40 строк, единый источник кодов причин | низкий |
| Три копии «lock Run FOR UPDATE + проверка состояния» (postgres.go:122-138, :398-417, gatewayrecovery/store.go:60-69) свести к `runstore.LockRunState(tx, id, states...)` | ~35 строк | низкий |
| Ретенция телеметрии через Scheduler (supervisor.go:278-305, 2 поля Options, Store.CleanupExpiredTelemetry, транзит через runstore, 2 фейка) — перенести в цикл, принадлежащий telemetry | ~70 строк | низкий |
| Registry на 15 методов и 4 метода только для тестов (registry.go:30-46, 333-347, 904-915) | ~45 строк prod | низкий |
| Умолчания пакетов, которые всегда перезаписывает app (scheduler.go:17-28, 80-118; runtime_batch.go:67-68) | ~35 строк | низкий |
| Ребро controlplane→planner ради алиасов (runtime_client.go:36-39) — типы перенести в contracts/reporting | −1 зависимость | низкий |
| N+1 в scan_history.go:71 — использовать GetPlannerSessions | ~10 строк | низкий |

## I. Оптимизация

1. **Renewal claim каждую секунду с чтением полной строки.**
   - Интервал = min(ClaimDuration/3, PollInterval) = 1s (supervisor.go:335-340). Это в 40 раз чаще, чем нужно для аренды в 2 минуты.
   - Каждый тик — `UPDATE workflow_runs SET scheduler_claim_expires_at, updated_at` (runstore/store.go:580-586) плюс `GetRun`, который читает все JSONB-снимки: workflow, runtime_config, skill и audit_completion (supervisor.go:391, runstore/scan.go:13-21).
   - Оба изменяемых столбца входят в `workflow_runs_admission_idx` (000064_run_admission.sql:15). Значит, ни одно обновление не HOT, и каждое пишет во все индексы workflow_runs.
   - При 32 лейнах это 64 оператора/с, около 2,7 млн записей в сутки, и `updatedAt` публичного Run меняется раз в секунду.
   - Исправление: renew раз в ClaimDuration/3; брать `RETURNING state` вместо GetRun; `updated_at` при renew не трогать. На порядок claim это не влияет: кандидаты не захвачены.
   - Задержка обнаружения отмены не вырастет: все четыре места, которые пишут отмену, вызывают `Cancel` в процессе (public/run_handlers.go:367, projectlifecycle/controller.go:324, evalservice/workflow_adapter.go:124, auditcontroller/controller.go:624), а executeRun перечитывает состояние после регистрации (run_execution.go:14).
   - Ожидаемо: с 64 до ≈1 оператора/с.
   - Попутно: `executeRun` повторно читает Run, хотя только что получил его из RETURNING claim (run_execution.go:14 против supervisor.go:141).
2. **Hot row в gatewayrecovery.** Каждый вызов модели — две транзакции, каждая берёт FOR UPDATE и на `workflow_runs`, и на общую для владельца+модели строку `gateway_recovery_routes` (requests.go:80-91, store.go:38-47). Так происходит даже на здоровом маршруте, хотя acquire при `!blocked` ничего не меняет. Вызовы модели одного маршрута сериализуются. Fast path: читать без блокировки и не писать, если маршрут не заблокирован и wait нет. Эффект SUSPECTED, на малых масштабах невелик.
3. **A2A-опрос с фиксированными 100 ms** на всё время вызова Worker (a2a/client.go:32, 265; operational_settings.go:143): 10 GetTask/с на каждый активный вызов, вызов длится минутами. Нужен адаптивный backoff до 1–2s, это −90% mTLS-запросов. Payload мал (HistoryLength=1, client.go:296).
4. **Двойная запись аллокаций** — см. H-1: 2N лишних операторов на каждое размещение плюс лишний режим отказа `allocation_record_failed`.

## Топ-3
1. G1: после abort probe маршрута держится до ≈30 мин, ручной Retry не помогает. Освобождать probe при коммите Stage и для fenced-гранта, аренду задавать по участнику (F1).
2. G2: исправление V290 в production прерывает Stage вместо переразмещения, тест не может это поймать. Чинить путь, удалить `recordReservations`, сделать pinning в фейке обязательным (F3).
3. Один владелец SQL для `workflow_runs`/`stage_executions`/`gateway_*` и fitness-тест владения таблицами. Это закрывает класс G1/G3 и три параллельные копии переходов состояний (F4, плюс F2 как быстрый дефект-фикс за S).

Для повторного прогона: сценарии лежат в `/tmp/claude-1000/-home-ruslan-src-contractor/0fecd9f4-f328-4d49-ab01-b67e6eace15e/scratchpad/hub-audit/dispatch/overlay/` (opsbound_test.go, v290_test.go), запуск через `go test -overlay` с overlay.json и overlay2.json там же. Открытых задач по этим находкам нет. Ближайшая по коду scheduler — V64-003 (pending), но перечисленного она не покрывает.
