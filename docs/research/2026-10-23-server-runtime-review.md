# Server and Runtime review — 2026-10-23

## Проверенная база и статус

Первоначальный обзор был сделан на `39d89438` с незакоммиченными изменениями
V405-002. Повторная проверка выполнена на неизменяемом `main`
`dc7012e8b1dc11efb1be4505f3c31a7d743a04f4`. Последующий `725b863e` добавил
только материалы V408 и не изменил проверенный код Server/Runtime.
Описание ниже заменяет исходные утверждения и оценки этого обзора. Номера
B1–G15 сохранены для связи с задачами; номера строк в задачах относятся к
проверенной базе и требуют сверки перед реализацией.

Область: Go Server (`internal/`, `cmd/`) и Python Runtime (`runtime/src/`).
UI не проверялся; изменения публичных контрактов в будущем должны учитывать
его потребителей. Метод: четыре независимых разбора по областям, проверка
спецификаций, существующих тестов и production-путей, затем совместная сверка
и локальные диагностические пробы. Это проверка утверждений, не полный
security scan и не доказательство отсутствия других дефектов.

**Статус на 2026-10-24: V407-039 in_progress.** Все сохранённые задачи
V407-001…V407-038, кроме V407-036, завершены и лежат в `tasks/done/` с
`implementation_commit` (журнал решений ниже). V407-036 (G14) переведена в
archived 2026-10-24 с причиной и условием повторного открытия и удалена из
`depends_on` V407-039. Дефекты и вопросы, найденные при реализации и ревью,
оформлены как V414-001…V414-027 (milestone
`review-v407-v409-followups-2026-10`, exit V414-027). Они открыты и здесь не
считаются исправлениями. Исход 75 утверждений B1–G15: 22 подтверждённых
дефекта исправлены, 34 оптимизации/chore выполнены, 2 изменения дизайна
оставлены предложением (proposal 0044), 8 уже покрыты, 4 не воспроизведены,
5 отклонены. Вопросы Q1–Q7: четыре проверенных ответа, два ответа с принятым
остаточным риском (Q5, Q6) и один принятый риск без проверки (Q7, нет
Linux-прогона). Release gate запущен на `b82c7309`, код которого совпадает с
`e72c1180` и с landing-веткой. На момент записи он не завершён, а `test-go` и
`test-runtime-hardening` уже упали (раздел J). Критерий A4 V407-039 не
выполнен, поэтому задача остаётся in_progress.

- `defect` (15): 001, 007–010, 019, 021–024, 026, 030–032, 037 — completed.
- `chore` (24): 002–006, 011–018, 020, 025, 027–029, 033–036, 038–039.
  V407-036 archived, 039 — этот exit gate, остальные completed.
- Изменение credential lifecycle в 018 — возможное расширение поведения,
  а не доказанный дефект. Задача подготовила proposal 0044; реализация
  потребует отдельного принятого решения и отдельной задачи.

Подтверждённая лишняя работа не означает измеренный production impact.
Критерии оптимизаций проверяют число операций, ограниченность памяти,
сохранение поведения и отмену; они не обещают неподтверждённые миллисекунды
или недостижимые размеры входов. Проверки PostgreSQL требуют непустой
`CONTRACTOR_TEST_DATABASE_URL` отдельной тестовой базы: пропуск не является
успешной проверкой.

## Журнал решений V407-039 (2026-10-24)

Журнал закрывает каждое исходное утверждение B1–G15 и каждый вопрос Q1–Q7.
Основное доказательство — записи завершённых задач в `tasks/done/`
(`verification`, `decisions`, `declined`, проверки «падает до исправления»);
неясные места перепроверены по коду ревизии `e72c1180`. Владелец — область
(`epic`) из `tasks/epics.yml`, которой принадлежит задача или компонент.
Ссылка на V414 означает открытый follow-up (на момент записи pending или
in_progress), а не исправление.

Категории:

- **Дефект исправлен** — дефект подтверждён и исправлен задачей; указан полный
  `implementation_commit`, регрессия падала на исходном коде.
- **Оптимизация/chore** — подтверждённая лишняя работа или сопровождение;
  поведение и контракты сохранены, изменение выполнено задачей.
- **Изменение дизайна** — запрошенное изменение является продуктовым решением;
  подготовлено только сравнение вариантов, действующий контракт сохранён до
  отдельной принятой feature-задачи.
- **Уже покрыто** — существующий контракт или код уже обеспечивает требуемое;
  изменение поведения не нужно (возможна правка описания).
- **Не воспроизведено** — заявленный сбой не воспроизведён пробой, тестом или
  опровергнут анализом пути.
- **Отклонено** — наблюдение верно, но изменение сознательно не делается;
  указана причина и условие пересмотра.

### Решения B: HTTP API и контракты

| Finding | Категория | Решение и доказательства | Задача, implementation_commit | Владелец |
| --- | --- | --- | --- | --- |
| B1 | Дефект исправлен | На базе авторизованный HEAD на шести GET Audit review отвечал 200 (400 с неверным query), а `/v1/events/ws` входил в handler. Router владеет HEAD для всех public GET (API-R111): 405 после auth и до handler; route-table тесты падают при удалении правила. Тот же класс найден и исправлен в private Artifact API (ART-R120); Control Plane и `/healthz`/`/readyz` исключены с обоснованием. | V407-022, `83bbc7046851ade0217fb1b54507a9595fdfe445` | api-contracts |
| B2 | Дефект исправлен | `a!b`, `a/b`, `-a` и повторный заголовок доходили до Eval service. Одно правило `contracts.ValidIdempotencyKey` и общая фикстура для Go, OpenAPI, CHECK в PostgreSQL, CLI и UI; Eval отвечает документированным 422 `eval_invalid` до чтения и записи receipt (API-R69/R70). Найден и исправлен sibling `cancelScanSweep`. Отклонено: grammar CHECK для eval-receipts (пересмотр — запись eval receipt вне evalstore). | V407-021, `cb975a8fc72e90173ba771156354968bf3acae01` | api-contracts |
| B3 | Дефект исправлен | Login 8192 bytes и тела Git key/import объявлены в OpenAPI; `TestPublicRequestBodyBoundsMatchHandlers` сверяет каждую ограниченную операцию с handler и падал на базе. Новые LIM-AUD-28, LIM-GIT-21 и др. | V407-021, `cb975a8fc72e90173ba771156354968bf3acae01` | api-contracts |
| B4 | Оптимизация/chore | Real-store tracer: owner-страница из 200 preparing Audits — 4 statements (BEGIN, list, preparation, COMMIT) против 1201 на базе; ответ байт в байт равен прежнему. Подпункт о чтении inventory payload не воспроизведён (phase guard). По ходу работы исправлены необъявленные 409/500 при гонке со snapshot. Открытый follow-up: V414-018, V414-020. | V407-004, `3cd9826c7d5a3676f6cff53a712d6ea555b76736` | audits |
| B5 | Дефект исправлен | Оборванное тело RuntimeConfig publish давало retryable 500; теперь 400 `invalid_request` без записи, oversized остаётся 413 (FAIL-API-1). Тест падал на базе. | V407-023, `978164edced37e5b18b8ae1a16a9e977fac2754d` | api-contracts |
| B6 | Дефект исправлен | Неверные query, имя, media type и precondition Project Artifact отвечают 400 без чтения Project; порядок записан в API-R112; ownership и `project_deleting` сохранены. Тест падал на базе. | V407-023, `978164edced37e5b18b8ae1a16a9e977fac2754d` | api-contracts |
| B7 | Уже покрыто | 415 для Scheduler settings и model selection задан OPS-R60; V407-021 исправила только описание `unsupported_media_type` в реестре. Preservation-тесты проходят на обеих ревизиях. | V407-021 (описание) | api-contracts |
| B8 | Отклонено | Разные JSON decoders не дают доказанной разницы строгости: строковые массивы не имеют members, labels проверяют дубли. Общий decoder вне scope; общие helpers введены только там, где контракт совпадает (`requireJSONMediaType`, `readJSONBody`). Пересмотр — воспроизведённое расхождение строгости. | V407-023 (scope.out) | api-contracts |
| B9 | Отклонено | Лишние nil checks не меняют поведение mutation; это cleanup без дефекта. Пересмотр не требуется: чистка допустима в любой chore, затрагивающей файл. | V407-023 (scope.out) | api-contracts |
| B10 | Не воспроизведено | Вложенные Git `WithTimeout` берут ранний deadline, а не сумму; socket deadlines принадлежат handler. Таймауты сохранены; тело Git import следует FAIL-API-1 и под своим deadline (GIT-R72). | V407-023 (scope.out) | api-contracts |
| B11 | Дефект исправлен | Для проверенных free-text полей объявлен `x-max-utf8-bytes` рядом с `maxLength`; многобайтные фикстуры на границе и за ней; LIM-RUN-10, LIM-PRJ-5, LIM-REV-11 и др. Открытый follow-up: V414-009 (byte bounds машинных значений). | V407-021, `cb975a8fc72e90173ba771156354968bf3acae01` | api-contracts |
| B12 | Уже покрыто | Лимит 100 allocation history задан OPS-R168/LIM-OPS-28 и сохранён; preservation-тест проходит. | V407-021 (scope.out) | api-contracts |
| B13 | Дефект исправлен | Ключ и If-Match проверяются до чтения тела у всех keyed/precondition mutations, media type до заголовков (API-R112); `TestSharedMutationHeadersAreRejectedBeforeTheBodyAndStores` и precedence table падали на базе. | V407-023, `978164edced37e5b18b8ae1a16a9e977fac2754d` | api-contracts |

### Решения C: Persistence

| Finding | Категория | Решение и доказательства | Задача, implementation_commit | Владелец |
| --- | --- | --- | --- | --- |
| C1 | Оптимизация/chore | На native-потолке было 1000 INSERT и 1000 запусков statement-trigger (queue revision 1→1001). Теперь `unnest`-пакеты не более 1000 строк и 4 MiB (LIM-EVM-18, EVM-R241); production-тесты на 1000 members и на внешнем документе 1 MiB. Sibling `putResources` пакетирован. 10 000 public members недостижимы (1 MiB) — подпункт отклонён. | V407-028, `3b5ae75b667568e4113afdcec401f94affe89d8f` | evals |
| C2 | Оптимизация/chore | EXPLAIN на фикстуре 6000–9999 Runs: drain check 161 block против 267, purge recheck 4 против 33. Ни один исходный план не был seq scan — подпункт «seq scan на каждом poll» не воспроизведён. Block counts одного локального запуска, не latency. Открытый follow-up: V414-024 (перемерить через `explaintest` Analysis.Reads). | V407-025, `c9562ebeb4628a77dc0b697eccb8a5a2993f1d24` | persistence |
| C3 | Уже покрыто | Placement grammar V405-002 появилась на main (`contractor_valid_placement_labels`); `TestSharedPostgresPlacementLabels` держит её общей фикстурой и проходит. Дефекта main нет. | V407-024 (проверка) | persistence |
| C4 | Оптимизация/chore | `CollectionReceiptIDs`: 510 statements для 255 findings → 1; граница 256 receipts и порядок ошибок сохранены (FND-R159). | V407-002, `72b9c7666ea1bc76d35517b0b1828ffce9c205a4` | audits |
| C5 | Оптимизация/chore | Partial GIN index по плану на 3000 Audits (60 held): 104 block против seq scan 361; отсутствующий credential 3 против 361; generic plan тоже использует индекс. | V407-025, `c9562ebeb4628a77dc0b697eccb8a5a2993f1d24` | persistence |
| C6 | Не воспроизведено | Two-writer artifact race не воспроизведена на PostgreSQL; CTE и mapping `ClassReferenceViolation` не менялись. Пересмотр — production-path воспроизведение двух writers (в т.ч. смешанные filesystem/PostgreSQL кандидаты) с точным constraint. | V407-026 (declined) | projects-artifacts |
| C7 | Дефект исправлен | `insertRun` отдавал сырой 23503 для отсутствующего или чужого Project; теперь `ErrNotFound` без следов Run, RunScope и ключа; service-путь при purge между precheck и транзакцией тоже 404. Тесты падали на базе. | V407-026, `6ce41d31784e8b6d8ff95df77a84008983d6c28e` | persistence |
| C8 | Уже покрыто | Строгая валидация Run events на чтении требуется правилом репозитория (AGENTS §2); сохранена. | V407-026 (declined) | persistence |
| C9 | Оптимизация/chore | Stage triggers сужены до значимых колонок (UPDATE OF), `explaintest.RequireColumnScopedGuards` проверяет все NEW/OLD ссылки. Sibling-триггеры других таблиц закрыла V409-006 (completed). | V407-025, `c9562ebeb4628a77dc0b697eccb8a5a2993f1d24` | persistence |

### Решения D: Scheduler, Control Plane, credentials и telemetry

| Finding | Категория | Решение и доказательства | Задача, implementation_commit | Владелец |
| --- | --- | --- | --- | --- |
| D1 | Изменение дизайна | Barrier через remote delete — действующий RCF-R52. Proposal 0044 (решение D1) сравнивает варианты и рекомендует database-level fence (D1-C), если решение будет пересмотрено; поведение не менялось; список holders в RCF-R52 исправлен. Решение принимает владелец проекта (proposal 0044 §9–10). | V407-018, `a9bfa9ee5fbaaeb71ed66451c8f34fc3a10f0f49` | runtime-config |
| D2 | Оптимизация/chore | Control client: 10 вызовов — 1 handshake вместо 10; A2A SendMessage и 20 GetTask — 1 соединение вместо 21; identity проверяется на каждом новом соединении (CTL-R163–R166). Утечка idle connections не воспроизведена. | V407-013, `cfecd2db7a8a724dc19528ddbce4f80ba556ea1c` | execution-core |
| D3 | Оптимизация/chore | Кэш одной попытки `placementAttemptCache` (CTL-R162) убирает повторные чтения в пересекающихся проходах; cross-poll cache отклонён (operationsRevision меняется на heartbeat). Подпункты GetRun/gofmt относились к незакоммиченному V405 и на main отсутствуют. | V407-014, `a3a7984e675498e56de42e51640067d7ff0ad47f` | execution-core |
| D4 | Оптимизация/chore | Trim измеряет отчёт не более трёх раз (на базе 925 полных Go encodes и 926 Python измерений); общий fixture и differential-тесты против прежнего цикла. Прохождение 4 MiB через private wire не воспроизведено (1 MiB). По ходу исправлено нечитаемое сохранённое значение на границе (из V32-001) и введён LIM-OBS-16. Открытый follow-up: V414-023. | V407-020, `f41fdbdb18860290634a798c3b60240c37a0c9ee` | operations |
| D5 | Оптимизация/chore | Credentials разрешаются один раз при подготовке и повторно только при restart/recovery (SCH-R178, LIM-SCH-19); один rebuild metrics; redaction побайтно совпадает на обоих путях. Открытый follow-up: V414-011 (fmt-глаголы SecretString). | V407-015, `a525cf29ada6f007f61ae0d29db1aa7d8dd21969` | execution-core |
| D6 | Дефект исправлен | Липкий `deleteDirty` сверяется с durable operation (RCF-R53); найденный sibling — delete другого credential снимал fence — исправлен; сбой сверки отвечает 503 `credential_recovery_required`. Открытый follow-up: V414-010 (intent прежнего lease owner). | V407-019, `a5ee851da7ef7f1a6014acfbb7f16716839122d6` | runtime-config |
| D7 | Оптимизация/chore | Один claim probe на pool за idle-раунд вместо одного на свободную lane (до 64); pools и fairness сохранены (SCH-R181). | V407-016, `51d627a9dc9b6c5a2d092e10238f1168d18011c7` | execution-core |
| D8 | Оптимизация/chore | Нормализация resource sentinel — одно определение `RuntimeReport.ApplyResourcePolicy` (CTL-R136). Часть о classifiers отклонена в V407-019: роли различны, `WrapError` сохраняет `errors.Is`. | V407-020, `f41fdbdb18860290634a798c3b60240c37a0c9ee` | operations |
| D9 | Оптимизация/chore | Terminal release recovery идёт не более чем четырьмя операциями (LIM-SCH-20) с ротацией по `release_attempted_at`; медленный peer больше не блокирует остальных. «N×timeout на проход» не воспроизведено. | V407-016, `51d627a9dc9b6c5a2d092e10238f1168d18011c7` | execution-core |
| D10 | Оптимизация/chore | История Operations — фиксированное кольцо `[255]`, `AllocsPerRun` = 0 при append на ёмкости; cursor overflow и snapshot semantics сохранены. | V407-020, `f41fdbdb18860290634a798c3b60240c37a0c9ee` | operations |
| D11 | Не воспроизведено | Бесконечное зависание опровергнуто: `TestControlPlaneLeaseStartupStallEndsAtItsBudgetOwner` останавливает handshake, keepalive, первое acquire и holder lookup без deadline, и каждый шаг завершается своим унаследованным бюджетом (PER-R27, PER-R28). Новый SLA не вводился. Открытый follow-up: V414-012, V414-014. | V407-017, `6970c5171ec096bb4babca542d9d7fd76b305498` | execution-core |
| D12 | Оптимизация/chore | Cancellation без active Stage читает только попытки с незавершённым release и одним пакетом allocations (SCH-R179); тест на 22 попытках. | V407-015, `a525cf29ada6f007f61ae0d29db1aa7d8dd21969` | execution-core |
| D13 | Изменение дизайна | Recovery-before-traffic — действующий RCF-R72. Proposal 0044 (решение D4) описывает restricted startup как вариант; поведение не менялось. Пересмотр — принятая feature-задача. | V407-018, `a9bfa9ee5fbaaeb71ed66451c8f34fc3a10f0f49` | runtime-config |

### Решения E: Audit, Eval и Planner

| Finding | Категория | Решение и доказательства | Задача, implementation_commit | Владелец |
| --- | --- | --- | --- | --- |
| E1 | Дефект исправлен | Ошибка standard loader после сохранения proposals остаётся transient; на базе вторая попытка коммитила invalid-result receipt (тест на PostgreSQL). | V407-001, `cc3c821262bbe7e74803817ef1e83acc431a02c3` | audits |
| E2 | Дефект исправлен | Общий `artifacts.IsExactReferenceRejection` для Passthrough, Streamline/Router и Scheduler; missing/invalid — детерминированный класс; выбор ошибки отсортирован. Открытый follow-up: V414-004, V414-006. | V407-008, `ab8ce42549ed394f64c38afb9f9a3120dda18d5d` | planner-worker |
| E3 | Дефект исправлен | Повреждённые pinned scan plan/report → `auditscan.ErrInvalid` → `ErrPermanent` → collection-contract-invalid receipt. Недекодируемая scan history остаётся retryable как принятое исключение AUX-R96 с enforcement, detection и условием пересмотра. | V407-001, `cc3c821262bbe7e74803817ef1e83acc431a02c3` | audits |
| E4 | Оптимизация/chore | Inventory из 1000 operations: 1000 разборов и digest источника → 1 и 1 (SCN-R256); golden digests без изменений. Benchmark 16,2 → 0,4–0,5 s/op — наблюдение, не gate. | V407-006, `7ca9628ca267b9520def6b13a48c702e1a63d330` | scan-tools |
| E5 | Оптимизация/chore | 64 items × 15 evidence: 960 payload reads → 0, 959 artifact-write транзакций → 2 (по 512 retention, LIM-AUX-7, AUX-R98); фактические границы 4096/1024, а не 16 384. | V407-002, `72b9c7666ea1bc76d35517b0b1828ffce9c205a4` | audits |
| E6 | Оптимизация/chore | `ResolveAuditProposals`: 512 statements на 128 keys → 5; 4096 proposals — 32 вызова, 160 statements. | V407-002, `72b9c7666ea1bc76d35517b0b1828ffce9c205a4` | audits |
| E7 | Оптимизация/chore | Опрашиваемые представления читают `FrozenPlanMetadata`/`FrozenPlanView`; readiness материализован при freeze (EVM-R242, LIM-EVM-19); `Validate("Experiment")` и строгое чтение сохранены. Открытый follow-up: V414-017 (NUL escape). | V407-029, `1b74b73835c11d93737405cbf4f096e728979c13` | evals |
| E8 | Оптимизация/chore | До 32 drain mutations за шаг (LIM-AUX-14, AUX-R97): 64 children — 74 snapshots вместо 259. Обещание «≤3 прохода» отклонено. Принятый риск fairness записан в AUX-R97 с enforcement (cleanup budget), detection (`reportLongDeletionStep`) и условием пересмотра. | V407-003, `d6b5c6bcf46c2967e2fee78cd4f430bfe745fce9` | audits |
| E9 | Дефект исправлен | Отклонённый candidate теперь ставит `invokeFailure`; `a2a.invoke` считается неуспешным (PLN-R119). | V407-008, `ab8ce42549ed394f64c38afb9f9a3120dda18d5d` | planner-worker |
| E10 | Уже покрыто | Повтор preparing при инфраструктурной ошибке задан EVM-R90 (`recovery=wait`), Duplicate доступен для draft. Новый Cancel или предел retries — изменение продукта, вне scope. | V407-030 (scope.out) | evals |
| E11 | Отклонено | Обе границы равны 200, текущего дефекта нет; V407-001 исключила invariant из scope. Перепроверка: `auditimport/importer.go:326` листает страницы `pagelimit.MaxRows`, а `RetainAuditCollectionBatch` отвергает больше `MaxAuditReceiptBatchSize` (`findingintake/postgres.go:336`); связь держится только комментарием `audit_receipt_batch.go:14`, тестов на страницу >200 через importer нет. Принятый риск: владелец audits (findingintake/auditimport); enforcement — fail-closed: при расхождении `RetainAuditCollectionBatch` отвергает страницу больше 200 как `ErrInvalid`, сбор падает детерминированно и без потери данных; автоматического теста на расхождение нет; пересмотр — любое изменение LIM-API-18 или `MaxAuditReceiptBatchSize`. Рекомендован отдельный chore на invariant-тест (решает интегратор). | V407-001 (scope.out) | audits |
| E12 | Дефект исправлен | `evaldomain.CounterSum` и сортированный обход дают одинаковый результат при overflow; normalizer стал `contractor-output@2` (EVF-R53/R58). Открытый follow-up: V414-002. | V407-030, `a3baa1a4d58fe7ffe4b766fe9a86cc02592ce3b2` | evals |
| E13 | Отклонено | У `internal/auditpriority` нет production caller; выбор сортирует по Priority, затем candidate ID. Неверного ранжирования работающих Audits нет. Интеграция V64 отложена `execution_policy`; пересмотр — решение пользователя активировать V64. | V407-005 (проверка) | audits |
| E14 | Оптимизация/chore | `internal/claimpass` — общий lifecycle claims для Audit controller и Eval coordinator; политики сохранены (PER-R26); sweep controller теперь сообщает ошибки release. «time.Now как ошибка» не воспроизведено. Открытый follow-up: V414-007, V414-008. | V407-027, `62d5e1c5d51c24f8a5d3bff86050e3d28b64d724` | persistence |
| E15 | Оптимизация/chore | Удалены повторная инициализация source, недостижимые проверки и второй fence recheck; intake делит одно чтение Stage (FND-R167, FND-R179). Marshal fail-open не воспроизведён: значения получены успешным JSON decode. | V407-005, `623cec9ebccf0e0d69c65d208a874de2592963eb` | audits |

### Решения F: Runtime core

| Finding | Категория | Решение и доказательства | Задача, implementation_commit | Владелец |
| --- | --- | --- | --- | --- |
| F1 | Оптимизация/chore | 1000 tool calls и 200 errors: `model_dump` 1 000 000 и 150 400 → 1000 и 200; публикация при 1000 records 10,9 → 4,6 ms (профиль, не gate). | V407-011, `d2b98323b49baa5f9bd9a3638204a7db505ab5ee` | planner-worker |
| F2 | Уже покрыто | Subtree contractor требуется WRK-R105; тест зеркала для shared и isolated sessions; deepcopy снимков сохранён (WRK-R160). | V407-011 (scope.out) | planner-worker |
| F3 | Дефект исправлен | Worker и finalizer передают instruction literal provider; `{path}` больше не вызывает KeyError; AST-guard требует provider в каждом `LlmAgent` (TPL-R114, WRK-R166). | V407-009, `4299674b09ada58841e2767e5a6b04647f456265` | planner-worker |
| F4 | Оптимизация/chore | TranscriptRecorder и summarizer secrets создаются только при наличии summarizer. | V407-011, `d2b98323b49baa5f9bd9a3638204a7db505ab5ee` | planner-worker |
| F5 | Дефект исправлен | Effective deadline ограничивает каждую ADK invocation (WRK-R167/R168); просроченный запрос не делает model/tool calls; 13 из 15 новых тестов падали на базе; injected clock. | V407-010, `dad9694711b375a5d2154e22d43538d7d2645c62` | planner-worker |
| F6 | Оптимизация/chore | Runtime private HTTPS переиспользует проверенные соединения с lifetime/idle bounds (CTL-R163–R166, LIM-CTL-13–18); неидемпотентные вызовы не повторяются. | V407-013, `cfecd2db7a8a724dc19528ddbce4f80ba556ea1c` | execution-core |
| F7 | Оптимизация/chore | Exporter сначала читает metadata: разный size/media — запись без download; равный размер читается точно, если нет подтверждённого digest той же ревизии (WSP-R178). | V407-034, `5e2950318e98e7273d4984da358acc851c9cda8d` | workspaces |
| F8 | Оптимизация/chore | Эквивалентные helpers сведены к одному определению; различные политики JSON и UTF-8 оставлены раздельно и закреплены тестами. | V407-012, `17e5c5e9bf58c3425cae689631042155af678551` | planner-worker |
| F9 | Оптимизация/chore | Filesystem-работа prepare/probe вынесена в joined threads с rollback при cancel (WSP-R168). Прочие `asyncio.to_thread` пути отклонены; пересмотр — задача cancellation ownership. | V407-012, `17e5c5e9bf58c3425cae689631042155af678551` | planner-worker |
| F10 | Не воспроизведено | Heartbeat deadlock не доказан: heartbeat читает независимое состояние, lease expiry ставит fence; allocation locking, fencing и порядок drain не менялись. Пересмотр — воспроизведённая остановка heartbeat во время prepare. | V407-012 (scope.out) | planner-worker |

### Решения G: Runtime toolsets

| Finding | Категория | Решение и доказательства | Задача, implementation_commit | Владелец |
| --- | --- | --- | --- | --- |
| G1 | Дефект исправлен | openapi@1 принимает границы Server (2 MiB, depth 64, 100 000 nodes с keys; SCN-R253); `api/testdata/openapi-document-bounds.json` — 26 документов с одинаковым вердиктом в Go и Python; база Runtime принимала все 14 документов за границей. Открытый follow-up: V414-001, V414-003. | V407-024, `6d0467c001c01f64bb0494f427f9d3bc3fdd47b0` | api-contracts |
| G2 | Дефект исправлен | Decode, validation и serialize выполняются в потоках под session lock; одна проверка документа на upsert; close ждёт preparation (RTA-R115). | V407-031, `9d2c3a99b618e86e63fad53862eb0a33b804129c` | workflows |
| G3 | Дефект исправлен | Caido decode, redaction и preview вне loop, cancel и close присоединяются к потоку; canonical base64 сохранён. Сценарий 32–64 MiB недостижим (adapter 16 MiB). | V407-032, `c45a3739d40f21358fbf6590f32a4454a6a58831` | http-tools |
| G4 | Оптимизация/chore | 64 items: decode task set 70/7 → 1; encode prospective package вне loop под collector lock (AWC-R125). | V407-033, `b3875b5f13669846ca0c3de78dc1ace61a996d79` | findings |
| G5 | Оптимизация/chore | Glob компилируется один раз (20 643 нормализации → 5) и сопоставляется вне loop; stat-only listing отклонён (WSP-R91/R69); 88 ms не перемерялись. Открытый follow-up: V414-005. | V407-034, `5e2950318e98e7273d4984da358acc851c9cda8d` | workspaces |
| G6 | Оптимизация/chore | 30 findings одного ZIP — 1 payload read вместо 30 (FND-R184); AST не кэшируется (FND-R169). | V407-033, `b3875b5f13669846ca0c3de78dc1ace61a996d79` | findings |
| G7 | Оптимизация/chore | Десять страниц 32 MiB: 0 полных decode вместо 10, line index и окно (ART-R117, LIM-ART-7). LikeC4 `line_offset` — отдельная capability, отклонена. | V407-035, `8e5d096069f3fd53ca462757d1d07a152a730264` | projects-artifacts |
| G8 | Дефект исправлен | HTTP/Caido: TypeError доходит до tool-error boundary как `tool_call_failed` (V407-032, `c45a3739d40f21358fbf6590f32a4454a6a58831`); code analysis: исключение parser path больше не становится parse_error coverage (V407-037). Filesystem уже non-retryable (V407-034). Открытый follow-up: V414-015, V414-016. | V407-037, `9ab42a9260faace63c16abdef99c52b70fad8a9c` | code-analysis |
| G9 | Уже покрыто | Close документа отменяет validator и присоединяется (тесты V407-031); HTTP serialization по HTP-R75/R129 сохранена (V407-032); serialization code analysis с границами описана в CAN-R31 (V407-037). Остаточная гипотеза — Q5. | V407-031/032/037 (проверка) | workflows |
| G10 | Дефект исправлен | `.hh` разбирается как C++ в shallow tools; C/C++ наборы выводятся из одной таблицы (CAN-R49); регрессии на настоящем reader. | V407-037, `9ab42a9260faace63c16abdef99c52b70fad8a9c` | code-analysis |
| G11 | Оптимизация/chore | Размер страницы считается линейно, страницы байт в байт как прежде; unsigned cursor сохранён (FND-R137–R141). | V407-033, `b3875b5f13669846ca0c3de78dc1ace61a996d79` | findings |
| G12 | Дефект исправлен | Scanner parsing, prepare и cleanup вне loop с joined cancellation (SCN-R254); проверка create-only output по metadata (SCN-R255); 43 новых теста падали на базе. | V407-007, `dd5d65645e0c453496e8c93b2ab9d758b6ee8309` | scan-tools |
| G13 | Уже покрыто | 16 KiB Worker summary задан LIM-AWC-2/AWC-R60; gate `TestBundledCWEClassificationMatchesRuntime` существует и проходит; LikeC4 использует общий line window. Дублирование ID regex — cleanup без drift, отклонено. | V407-024 (проверка) | api-contracts |
| G14 | Отклонено | Параллельные reads конкурируют за 4 fail-fast transfer slots Server (LIM-BLB-1, BLB-R24–R26) и превращают успешные операции в `memory_unavailable`. Задача archived 2026-10-24 и удалена из depends_on V407-039. Пересмотр — Server обслуживает малые reads без fail-fast конкуренции или появляется private Worker batch read. | V407-036 (archived) | memory |
| G15 | Оптимизация/chore | `FilesystemToolError` перенесён в `toolsets.common`, protected chain удалена из import-linter. `read_openapi_document` сохранён (удаление отклонено); существование katana output проверяется по metadata (V407-007). | V407-038, `df1271b407db1de37ea6c1e9b9b451c3c7dbd078` | quality |

### Итог по категориям

| Категория | B | C | D | E | F | G | Всего |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Дефект исправлен | 7 | 1 | 1 | 5 | 2 | 6 | 22 |
| Оптимизация/chore | 1 | 5 | 9 | 7 | 6 | 6 | 34 |
| Изменение дизайна | 0 | 0 | 2 | 0 | 0 | 0 | 2 |
| Уже покрыто | 2 | 2 | 0 | 1 | 1 | 2 | 8 |
| Не воспроизведено | 1 | 1 | 1 | 0 | 1 | 0 | 4 |
| Отклонено | 2 | 0 | 0 | 2 | 0 | 1 | 5 |
| Всего | 13 | 9 | 13 | 15 | 10 | 15 | 75 |

Строка записана по основному исходу; частичные подпункты (например,
неподтверждённые части B4, C2, D4) названы в её доказательствах.

### Решения Q: вопросы закрывающей проверки

| ID | Решение | Ответ и доказательства | Владелец |
| --- | --- | --- | --- |
| Q1 | Проверенный ответ | V407-004 считает statements через pgx tracer на PostgreSQL: owner-страница из 200 preparing Audits и из одного — 4, Project list — 5, страница без preparing — 3; на базе 1201 для 200 и 61 для смешанной страницы из 12 items всех фаз. | audits |
| Q2 | Проверенный ответ | V407-010: детерминированные expiry и cancellation через injected loop clock в каждой фазе (model, tool, finalizer, summarizer), joined in-flight work; 13 из 15 тестов падали на базе. | planner-worker |
| Q3 | Проверенный ответ | V407-011 сохраняет WRK-R105: каждая terminal revision попадает в session как contractor stateDelta для shared и isolated sessions; публикация сериализует каждую запись один раз. Найденное удержание tools в module-level кэше ADK — открытая V414-013. | planner-worker |
| Q4 | Проверенный ответ | Цепочка producer → consumer: openapi@1 и scan preparation используют одно правило (2 MiB, depth 64, 100 000 nodes с member names; общая фикстура, V407-024); inventory openapi-operations@1 его охватывает: 8 MiB (LIM-AUD-14), depth 64 с той же конвенцией (LIM-AUD-17), 250 000 value nodes (`internal/auditdomain/limits.go`). Два разрыва вне byte/depth/node — cap строки 64 KiB и YAML 1.1/1.2 scalar/key/merge interop — оформлены как открытые V414-001 и V414-003. | api-contracts, audits |
| Q5 | Проверенный ответ + принятый риск | OpenAPI/LikeC4: синхронизированные тесты V407-031 (close ждёт preparation, отменяет validation до Vacuum). HTTP/Caido: serialization намеренна (HTP-R75/R129), cancel и close присоединяются к потоку (V407-032). Memory: MEM-R49, см. G14. Code analysis — принятый риск: владелец code-analysis; enforcement — CAN-R31 (LIM-CAN-11 и один parse в пределах LIM-CAN-5, LIM-CAN-12/13 с kill child по CAN-R83, joined cancellation CAN-R86, fence slot WSP-R169/R170); detection — длительности tool calls и число graph builds в Worker metrics (CAN-R141); пересмотр — воспроизводимое ожидание сверх этих границ или нагрузка, требующая параллельных вызовов в одной allocation. | workflows, http-tools, code-analysis |
| Q6 | Проверенный ответ + принятый риск | Смещение часов client/DB не влияет на захват. Expiry, takeover и fencing сравниваются с часами PostgreSQL в том же statement (`runstore/store_sql.go:188,213,239`, `runstore/store.go:562`, `persistence/postgres/claims.go:68-69,83,101,134`). Локальная оценка Scheduler использует только `time.Now` с monotonic reading и длительности, отсчитанные до запроса (`scheduler/supervisor.go:199,456,476,505`), поэтому постоянный offset сокращается, а оценка раньше срока в БД. После истечения SQL lease другая lane того же Server не заменяет живого владельца (SCH-R120). Детерминированные тесты перезапущены на этой ревизии (раздел J). Дефект не показан, задача не создаётся. Остаточный риск — скачок wall clock хоста БД вперёд больше `claimDuration/3` сокращает выданные leases; другой Server может взять Run только при одновременном failover. Владелец execution-core; enforcement — исключение живых lanes (SCH-R120), single-active session lease, claim ID в записях Run; detection — следующая renewal старой lane получает conflict и завершает её с `ErrClaimLost` (FAIL-SCH-16) после предупреждений «WorkflowRun claim renewal failed»; пересмотр — более одного активного Scheduler или takeover при живой прежней lane. | execution-core |
| Q7 | Принятый риск, не проверено | Linux-прогон на этом хосте невозможен: нет локального Linux-образа с Python 3.13, образы не скачивались. На macOS `test-runtime-hardening` release gate дал 107 падений (раздел J). 105 из них поимённо совпадают со списком V401-007: Podman 95 (`AF_UNIX SOCK_SEQPACKET`, `pidfd_open`, shim `/usr/bin/python3`), семь непереносимых тестов (`/proc/self/fd`, `setsid(1)`, `RLIM_INFINITY`), wall-clock parser gap 100 ms и два macOS-дефекта (APFS normalization против WSP-R88, `PATH_MAX` 1024 в Trailmark mirror против WSP-R13). Ещё два — зависящие от порядка `ResourceWarning` незакрытых AF_UNIX-сокетов вне списка; оба теста проходят отдельно и в своих модулях. Ни одно падение не засчитано как pass. Владелец quality — V401-007 (pending); enforcement — стадия `release-verify-unit` на ubuntu-24.04 в CI (PR, теги, ручной запуск), где у Runtime-набора нет allowlist; detection — падение этой стадии; пересмотр — результат V401-007, Linux-прогон release gate или падение Runtime-набора на Linux. | quality |

## B. HTTP API и контракты

| Finding | Проверенный результат и решение | Задача |
| --- | --- | --- |
| B1 | HEAD действительно доходит до public GET handlers, включая Audit review. Ответ зависит от входа: возможен 400, а не только 200. Нужен ранний 405 после существующей auth-проверки; нельзя приписывать всем private endpoints тот же контракт. | V407-022 defect |
| B2 | Eval допускает `a!b` и `a/b`, которые общая грамматика Idempotency-Key отвергает. Объединить правило и проверить всех потребителей общей фикстурой. | V407-021 defect |
| B3 | Login читает 8192 bytes при объявленных 2048; часть Git body limits не объявлена. Согласовать фактический и публичный контракт, включая представление максимальных строк. | V407-021 defect |
| B4 | Дополнительная гидратация есть для preparing/inventory Audit. GeneratedInventory защищён фазой rounds: чтение его payload из списка не подтвердилось. Нужна пакетная гидратация с сохранением всех полей ответа, не урезание list DTO. | V407-004 chore |
| B5 | Обрыв RuntimeConfig publish body (`unexpected EOF`) превращается в 500. Классифицировать как ошибку запроса, сохранив отдельный oversized-body outcome. | V407-023 defect |
| B6 | Project lookup выполняется до проверки query, имени артефакта, media type и preconditions. Уточнить порядок валидации без изменения auth, ownership и согласованных статусов. | V407-023 defect |
| B7 | HTTP 415 для неверного media type предусмотрен OPS-R60. Унификация к 400 исключена; устаревшее описание реестра можно исправить. | V407-021, сохранение контракта |
| B8 | Разные JSON helpers сами по себе не доказывают ошибку. Обязательное объединение удалено из defect scope. | V407-023, исключено |
| B9 | Избыточные nil checks — локальный cleanup, не основание для mutation defect. Обязательная зачистка исключена. | V407-023, исключено |
| B10 | Вложенные Git deadlines не складываются; handler дополнительно владеет socket deadline. Удаление таймаутов как «трёх одинаковых ожиданий» исключено. | V407-023, исключено |
| B11 | UTF-8 byte bounds отличаются от OpenAPI maxLength в code points. Описать и проверить обе границы, включая многобайтный ввод. | V407-021 defect |
| B12 | Лимит 100 у Operations предусмотрен OPS-R168. Общая замена на 200 исключена; совпадение некоторых чисел не делает все page contracts одинаковыми. | V407-021, сохранение контракта |
| B13 | Некоторые mutation handlers буферизуют тело до проверки обязательных заголовков. Исправить конкретные пути и precedence; общий универсальный preamble не является обязательным дизайном. | V407-023 defect |

## C. Persistence

| Finding | Проверенный результат и решение | Задача |
| --- | --- | --- |
| C1 | Построчный persistPlan и statement-trigger на каждый INSERT подтверждены. 10 000 public members не помещаются в документ 1 MiB; native максимум 1000. Production preparation test на 1000 уже есть: расширить его счётчиками и rollback cases. | V407-028 chore |
| C2 | Project-only lookup имеет возможность улучшить индексный доступ. Seq scan на каждом poll не измерен: сначала EXPLAIN на реалистичном наборе, затем обоснованный predicate/index. | V407-025 chore |
| C3 | Placement SQL grammar из V405-002 отсутствует на проверенном main. Не считать дефектом main, не удалять SQL validation; проверить committed реализацию при завершении V405. | V407-024, вне текущего scope |
| C4 | CollectionReceiptIDs делает отдельные запросы на entries; результат ограничен 256. Пакетировать с сохранением ownership и лимитов. | V407-002 chore |
| C5 | Held-Audit credential lookup допускает оптимизацию. Нужность GIN/другого индекса подтвердить планом и размером данных, а не требованием «всегда index scan». | V407-025 chore |
| C6 | Предложенная two-writer artifact race не воспроизведена на PostgreSQL. Не менять ClassReferenceViolation mapping без отдельного доказательства и точного constraint. | V407-026, не подтверждено / вне scope |
| C7 | Run store возвращает неклассифицированную ссылку на отсутствующий Project. Публичный путь уже проверяет Project и возвращает 404; это store-boundary defect, не доказанный public 500. | V407-026 defect |
| C8 | Строгая валидация Run events на чтении соответствует правилу репозитория. Не удалять её на основании immutability. | V407-026, сохранение контракта |
| C9 | Stage triggers назначены на любой UPDATE. Сузить только несущественные обновления; старый тест отсутствия lifecycle event уже проходит, поэтому проверять реальные вызовы/определение триггера и существенные изменения. | V407-025 chore |

## D. Scheduler, Control Plane, credentials и telemetry

| Finding | Проверенный результат и решение | Задача |
| --- | --- | --- |
| D1 | Эксклюзивный barrier удерживается через remote delete, как требует RCF-R52. Меньшая область блокировки требует доказательства безопасности всех reference writers. Подготовить сравнение вариантов, без реализации нового поведения. | V407-018 chore / proposal |
| D2 | Control Plane создаёт Transport на вызов; A2A — на Invoke, но keep-alive отключён и при polling. Повторные TLS connections подтверждены, утечка idle connections — нет. | V407-013 chore |
| D3 | Перекрывающиеся placement resolutions есть, но не всегда ровно 2×N. GetRun/gofmt из незакоммиченного V405 отсутствуют на базе. Повторное использование ограничить попыткой; Operations revision меняется и на heartbeat и сам по себе не обосновывает cross-poll cache. | V407-014 chore |
| D4 | Go trimming повторяет полную сериализацию; Runtime private wire ограничен 1 MiB, поэтому 4 MiB report через него не проходит. Локальные Server producers/redaction рассматриваются отдельно. Python build_report включён в scope. | V407-020 chore |
| D5 | Повторная материализация credentials и сборка metrics есть на normal path; после restart восстановление необходимо. Оптимизировать с сохранением redaction и recovery. | V407-015 chore |
| D6 | deleteDirty может остаться после неоднозначного DB INSERT; Recover его не снимает. Обычная ошибка manager.Delete не воспроизводит этот путь. Перед снятием fence надо доказать завершение неоднозначной транзакции. | V407-019 defect |
| D7 | Idle claim probes растут с числом lanes. Объединение должно сохранить независимые model/tool pools, fairness и доступную конкуренцию. | V407-016 chore |
| D8 | Дубли нормализации resource sentinel можно объединить при сохранении semantics. Credential classifiers имеют разные роли; доказанной ошибки классификации ErrKeyUnavailable/ErrCrypto здесь нет. | V407-020 chore; classifier часть исключена из 019 |
| D9 | Terminal release последователен и допускает head-of-line blocking на медленном вызове. Быстрый 409 не тратит 30 секунд. Проверять управление concurrency и retry/backoff, не N×timeout без условий. | V407-016 chore |
| D10 | Operations history копируется при heartbeat; ring ограничен 255. Сократить копирование без изменения cursor overflow/snapshot semantics. | V407-020 chore |
| D11 | Startup использует copied pool config: connect timeout 5 s и query tracer 20 s уже ограничивают ожидание. Пробы опровергли бесконечное зависание. Зафиксировать владельцев бюджетов и покрыть startup тестами; не объявлять новый SLA. | V407-017 chore |
| D12 | История Stage читается на cancelling path без active Stage, не на каждом tick. Оптимизировать этот условный путь с сохранением завершения и cleanup. | V407-015 chore |
| D13 | Gateway outage препятствует startup recovery только при существующей prepared operation. Recovery-before-traffic задан RCF-R72. Restricted startup — отдельный проектный вариант, не автоматически разрешённое изменение. | V407-018 chore / proposal |

## E. Audit, Eval и Planner

| Finding | Проверенный результат и решение | Задача |
| --- | --- | --- |
| E1 | Ошибка standard loader при сверке уже сохранённых proposals превращается в permanent invalid. Сохранить transient class; permanent validation должна использовать существующие receipt outcomes. | V407-001 defect |
| E2 | Missing/invalid result artifact ошибочно получает retryable. Классифицировать по причине и стабилизировать выбор ошибки; bounded attempt policy не означает бесконечный retry. | V407-008 defect |
| E3 | Часть повреждённых pinned scan data возвращается как обычная retryable ошибка. Missing plan/report уже дают coverage gap; часть decode paths защищена раньше. ErrPermanent приводит к collection-contract-invalid receipt, не обязательно Audit failed. | V407-001 defect |
| E4 | При подготовке inventory один OpenAPI повторно разбирается и хешируется на operation. Переиспользовать подготовленный source в batch; единичный Run caller уже обрабатывает одну operation. | V407-006 chore |
| E5 | Дескрипторы читаются вместе с payload и retention выполняется отдельно. Реальные границы: 4096 evidence links на результат, 1024 retained links в receipt; не 16 384. | V407-002 chore |
| E6 | ResolveAuditProposals имеет N+1, но batch ограничен 128, а proposals на Run — 4096. Счётчики и тесты должны следовать этим границам. | V407-002 chore |
| E7 | Metadata callers читают полный FrozenPlan, GET повторяет nativeReadiness. Validate("Experiment") проверяет собранный ответ, не portable plan. Сохранить строгие read/output проверки. | V407-029 chore |
| E8 | Один progress на snapshot подтверждён, но RunOnce сразу повторяет шаг. Fixed-budget batching полезен; завершение cancel/delete зависит от асинхронных детей и barriers, поэтому обещание ≤3 passes исключено. | V407-003 chore |
| E9 | Passthrough может отклонить candidate, но оставить invokeFailure пустым. ExecutionReport ошибочно считает tool invocation успешным; остальной report не обязательно успешен или пуст. | V407-008 defect |
| E10 | Preparing retry при инфраструктурной ошибке предусмотрен EVM-R90; Duplicate при наличии draft разрешён. Новый Cancel или предел числа retries — изменение продукта, исключённое из overflow defect. | V407-030, сохранение контракта |
| E11 | Оба batch limits сейчас равны 200. Это риск будущего drift, не текущий неправильный размер. Связать/test invariant, не заявлять воспроизведённый retry loop. | V407-001, защитный invariant |
| E12 | Overflow в usage aggregation зависит от map order; расхождение воспроизведено. P3: практический объём недостижим, но обработка границы должна быть детерминированной. | V407-030 defect |
| E13 | auditpriority не интегрирован в production; Confidence не участвует в ranking. Это не неправильное ранжирование работающих Audits. Сохранить отложенное решение V64. | V407-005, documented non-goal — перепроверено 2026-10-24: у `internal/auditpriority` нет production caller (только его тесты и baseline `tools/testonlyexports` для V64-005/006); выбор сортирует по Priority, затем по candidate ID, Confidence только валидируется. Интеграция V64 остаётся отложенной по `execution_policy`; ranking и API не менялись |
| E14 | Общую claim lifecycle plumbing можно выделить, но политика wake/fairness/budgets различается. Прямой time.Now сам по себе не доказывает runtime bug. | V407-027 chore |
| E15 | Есть повторные checks/reads. Marshal fail-open не достижим через нормализацию конкретного Workflow type; transaction-local provenance read имеет самостоятельную роль и не удаляется. | V407-005 chore — выполнено 2026-10-24: удалены повторная инициализация source, недостижимые проверки владельца evidence и суммы рабочих строк, второй fence check FindingGroupMembers; intake делит одно ленивое чтение Stage до транзакции, provenance read в транзакции сохранён. Сравнение preparation Workflow не менялось: у ResolvedWorkflow нет custom marshaler, оба значения получены успешным JSON decode |

## F. Runtime core

| Finding | Проверенный результат и решение | Задача |
| --- | --- | --- |
| F1 | Повторные полные metric dumps дают квадратичное накопление работы: 100/200 records в пробе дали 10 000/40 000 сериализаций. Уменьшить повторную работу без изменения публикации состояния. | V407-011 chore |
| F2 | Session deepcopy существует, но immutable строки не означают повторное копирование всех payload bytes. Subtree contractor требуется WRK-R105; отсутствие найденного reader не разрешает удалить его. | V407-011, сохранение контракта |
| F3 | Обычный Worker передаёт authored string в ADK: `{path}` вызывает ошибку до model call. Нужен literal instruction provider. Finalizer использует фиксированный безопасный текст; нового запрета фигурных скобок не нужно. | V407-009 defect |
| F4 | TranscriptRecorder делает ненужную работу при отключённом summarizer. Ограничить её, сохранив enabled path и cleanup. | V407-011 chore |
| F5 | Ordinary Worker не применяет effective call deadline локально; probe с истёкшим deadline сделал model calls. Audit completion path уже защищён, другие budgets существуют. | V407-010 defect |
| F6 | Runtime private HTTPS закрывает соединения. Нужен bounded reuse с проверкой identity, lifecycle и cancellation, без новых retries неидемпотентных вызовов. | V407-013 chore |
| F7 | Exporter скачивает прежний payload для сравнения. ArtifactReadResult не содержит digest, revision opaque. Использовать только доказанные size/media fast paths и bounded exact-revision comparison cache; неизвестные равные размеры требуют чтения. | V407-034 chore |
| F8 | Повторные helpers не все эквивалентны: JSON fallback и UTF-8 suffix policy различаются. Объединять только эквивалентные части, сохраняя различия явно. | V407-012 chore |
| F9 | Небольшие синхронные filesystem операции есть. Переносить с joined cancellation и сохранением ownership, не обещая измеренный выигрыш. | V407-012 chore |
| F10 | Prepare удерживает allocation lock, но heartbeat работает независимо, а lease expiry сначала ставит fence. Общий heartbeat deadlock не доказан; сохранить эти гарантии при рефакторинге. | V407-012 chore |

## G. Runtime toolsets

| Finding | Проверенный результат и решение | Задача |
| --- | --- | --- |
| G1 | OpenAPI parser принимает 4 MiB/depth 128, Server — 2 MiB/64; различается подсчёт keys в node bound. Scanner bounds уже есть в LIM-SCN-39/41/42. Согласовать документные bounds и общие boundary fixtures, сохранив разные semantic validators. | V407-024 defect |
| G2 | Синхронные YAML parse/serialize/validation блокируют loop. В пробе две проверки всего документа и три PathItem, не три whole-document passes. Устранить повторную работу и offload с сохранением lock ownership. | V407-031 defect |
| G3 | Caido decode/redaction/preview выполняются inline. Adapter ограничивает весь JSON 16 MiB, raw base64 меньше 12 MiB: 32–64 MiB fake обходит настоящий вход. Re-encode проверяет canonical pad bits; эту проверку сохранить. | V407-032 defect |
| G4 | Admission повторяет decode immutable ZIP и encode растущего набора до 64 items. Кэшировать immutable input, вынести CPU с loop, сохранив atomic collector lock и реальную стоимость output encoding. | V407-033 chore |
| G5 | Snapshot читает content для binary classification и content-based cursor; stat-only listing нарушает WSP-R91/R69. Disk I/O уже threaded. Оптимизировать glob matching и доказанные export comparisons; 88 ms не повторно измерены. | V407-034 chore |
| G6 | 30 findings из одного ZIP дали 30 metadata/payload reads и anchor derivations. Переиспользовать bounded immutable bytes; AST не хранить между файлами (FND-R169). Timeout anchors не означает потерю finding. | V407-033 chore |
| G7 | Payload cache работает, но 10 страниц дали 10 полных decodes при одном download. Кэшировать decoded representation с общей memory accounting. Новый LikeC4 line_offset — отдельная capability, не исправление честно описанного long-line ограничения. | V407-035 chore |
| G8 | HTTP/Caido TypeError превращается в retryable tool error. Code analysis TypeError становится parseErrors/incomplete; filesystem error уже retryable=false. Исправлять подтверждённые error boundaries, не объявлять все случаи одинаковыми. | V407-032 и 037 defect; filesystem часть исключена из 034 |
| G9 | Document close уже отменяет validator и joins process cleanup до lock: полного timeout ждать не требуется. HTTP request serialization задана HTP-R75/R129. Для code analysis не доказано превышение shutdown budget. Сохранить гарантии; общий unlock переписыванием не предписывать. | V407-031/032/037, уточнение scope |
| G10 | `.hh` объявлен C++ для graph, но отсутствует в shallow parser map; list_symbols пропускает header. Исправить с regression на настоящем reader. | V407-037 defect |
| G11 | Repeated page canonicalization квадратична; заменить точным incremental sizing. Unsigned cursor намеренно предусмотрен FND-R137–141; common HMAC codec имеет другой контракт и не подставляется. | V407-033 chore |
| G12 | Scanner parsing и часть prepare/cleanup I/O выполняются на loop. Offload синхронную работу с joined cancellation; async publication остаётся await. | V407-007 defect |
| G13 | Worker summary 16 KiB уже описан LIM-AWC-2/AWC-R60; generic 64 KiB не доказывает drift. CWE equality gate уже существует. ID regex duplication — cleanup; LikeC4 уже пользуется common line window. | V407-024, неподтверждённые подпункты исключены |
| G14 | До 128 payload reads Memory выполняются последовательно. Внутри serialized operation допустим bounded concurrency; MEM-R49 и последовательность writes сохраняются. Готового private Worker batch endpoint нет. | V407-036 — отклонено 2026-10-24: параллельные reads конкурируют за 4 fail-fast transfer slots Server (LIM-BLB-1, BLB-R24–R26) и превращают успешные операции в memory_unavailable; задача archived, reopen условия в её записи |
| G15 | read_openapi_document доступен custom templates и упоминается production guidance; отсутствие shipped selection не доказывает dead API. Оставить tool. Общий exception перенести для filesystem, edit_files и workspace_changes; katana existence read проверить только на реально существующем артефакте. | V407-038 chore; katana — V407-007 |

## H. Обвязка и ограничения проверки

`regexp2` имеет test-only caller; оснований удалять его нет. Локальные
брошенные worktrees не являются дефектом репозитория. Семейные gates могут
пропускать повторный Go run внутри release-verify согласно существующему
контракту; сам по себе этот механизм не является новым V407 defect.

При повторной проверке прошли Go packages `internal/evaldomain`,
`internal/scanplan`, выбранные Audit/Planner suites и целевые overlay probes.
Полные public HTTP и часть Planner suites были ограничены sandbox-запретом
localhost bind и не считаются прошедшими.

Runtime: core группа — 213 passed и три заблокированные сетевые fixtures
(отдельный повтор с явным deselect: 213 passed); Worker State/metrics —
18 passed. Toolset группы — 280 passed / 1 failed и 250 passed / 3 failed.
Они пересекаются, поэтому числа нельзя складывать. Повторяющееся падение —
существующий wall-clock parser-gap threshold 100 ms (наблюдались 126/204 ms).
Ещё два падения — обращения теста к Linux `/proc/self/fd` на macOS. Они не
скрыты и не объявлены исправлениями текущего пакета.

PostgreSQL URL не был задан: integration tests, EXPLAIN и artifact two-writer
race не выполнялись. Live models и release-verify не запускались. Число
SQL-запросов, production latency, последствия clock skew и platform-specific
сценарии ниже остаются предметом будущей проверки. Этот абзац описывает исходную
проверку; команды и результаты закрывающих запусков — в разделе J.

## I. Вопросы, оставленные закрывающей проверке

| ID | Текущий ответ | Что требуется для закрытия |
| --- | --- | --- |
| Q1 | Preparing/inventory list hydration делает дополнительную работу; точное число SQL queries не измерялось. | V407-004: real-store counters на реалистичной странице, включая остальные фазы. |
| Q2 | Локальный deadline ordinary Worker действительно отсутствует; удалённая отмена его не заменяет. | V407-010: deterministic expiry/cancellation tests и joined cleanup. |
| Q3 | Production reader ADK subtree не найден, но WRK-R105 требует зеркало. | V407-011: сохранить контракт и проверить bounded publication; удаление не входит в scope. |
| Q4 | Bounds отдельного Audit inventory generator не доказаны этой проверкой целиком. | V407-024/039: trace producer-to-consumer, JSON/YAML byte/depth/node cases; новый подтверждённый gap оформить отдельно, если не покрыт 024. |
| Q5 | Document close уже отменяет validator; HTTP/Memory serialization намеренна. Shutdown bound code analysis не воспроизведён. | V407-031/032/037/039: сохранить cancellation ownership; для оставшейся гипотезы дать synchronized test либо явный disposition с условием пересмотра. |
| Q6 | Client/DB clock skew в renewClaim остался гипотезой. | V407-039: deterministic skew/takeover test, затем defect task при подтверждении; не считать гипотезу исправленной другими chores. |
| Q7 | Две projectfs ошибки объясняются Linux-only `/proc` на macOS; первоначальный третий случай не подтверждён как product defect. | V407-039: Linux run и раздельная классификация тестовой переносимости, flaky threshold и runtime failures. |

Ответы и решения по Q1–Q7 — в таблице Q журнала решений V407-039.

V407-039 закрывает только проверенные и реализованные решения. Если отдельная
задача целиком отклонена, её архивирование с причиной и удаление соответствующей
dependency выполняются согласованно: archived/deferred не считаются completed.
Принятый риск получает владельца, механизм контроля и условие пересмотра.
Редактура этого отчёта и задач не заменяет implementation commits или release gate.

## J. Закрывающая проверка V407-039

**Ревизии.** Журнал решений сверялся с кодом `e72c1180` (база ветки
`claude/rv-x2`). Release gate запущен на
`b82c730984594e491de12edc5cc161901b0b7199`, который отличается от `e72c1180`
только записью задачи и `tasks/BACKLOG.md`. Затем в ветку влит `main`: merge
`d9c9db9e` (`main` `59906c2f`: переименование follow-ups в V414, ревалидация
backlog, V414-028…030) и merge `fccd443c` (`main` `ea9d7b3a`: старт delivery
lanes V410–V414). Ветка `claude/review-v407-v409` уже входит в `main`.
`git diff --name-only e72c1180 HEAD` вне `docs/` и `tasks/` пуст, поэтому
результат gate относится к коду landing-ревизии.

**Окружение.** macOS 27.0.1 arm64, Go 1.26.9, Python 3.13.16 (uv),
Node 24.20.0, PostgreSQL 17.11 (Homebrew). Кластер общий с параллельными
агентами, у каждого своя база. Отдельная тестовая база:
`CONTRACTOR_TEST_DATABASE_URL=postgres://localhost/contractor_test_x2?sslmode=disable`,
`database_tests_configured: true`.

**Команда.**

```sh
CONTRACTOR_TEST_DATABASE_URL='postgres://localhost/contractor_test_x2?sslmode=disable' make -k release-verify
```

Старт 2026-10-24 10:46:52 MSK (07:46:52Z). **На момент этой записи gate ещё
выполняется (стадии process-a, process-b, process-c). Итог не определён, и
успешным он уже не будет: две стадии упали.**

| Стадия | Результат |
| --- | --- |
| `release-verify-lint` (`lint`, `build`, `test-config`) | прошла |
| `release-verify-unit`: `test-go` | **упала**: 2704 passed, 8 skipped, 2 failed. Все пропуски вызваны окружением, тестов БД среди них нет: LiteLLM contract env, NVIDIA GPU, chown только от root, не задан OTLP endpoint, файловая система NFC/NFD, live Gateway ×3. Упали `TestPostgresControlPlaneLeaseHolderIgnoresOtherDatabases` (`internal/app`, `drop lease database: timeout: context deadline exceeded`) и `TestStandbyServerAnswersPublicAPIWithRetryableEnvelope` (`tests/integration/lease`, `drop standby database: timeout`). Обе — 10-секундный cleanup `DROP DATABASE … WITH (FORCE)` под нагрузкой параллельных агентов на общем кластере. Те же пакеты прошли в стадии race. |
| `release-verify-unit`: `test-runtime-hardening` | **упала** (`pytest -W error`, allowlist нет): 4726 tests, 4580 passed, 39 skipped, 107 failed. 105 падений поимённо совпадают со списком `tasks/v401-007-runtime-suite-on-macos.yml`. Podman, 95: capabilities 32, supervisor 18, allocation 15, command_coordination 13, network 9, recovery 5, workflow 2, engine 1. (d): `local_direct_faults[socket]`, `content_scan_stops_after_first_binary_chunk` ×2, `detached_descendant_holding_pipes_cannot_stall_stop` ×2, `address_space_budget_accounts_for_darwin_mappings_and_hard_limit` ×2. (e): `real_four_mib_parser_keeps_event_loop_gaps_below_100_ms`. (f): `limits_deadline_names_and_binary_classification` и `graph_collection_stops_at_encoded_result_ceiling_with_resumable_cursor`. Два падения вне списка: `test_projectfs_concurrency::test_export_generation_rejects_rollback_import_and_checkpoint` и `test_projectfs_local_direct::test_failed_temp_write_preserves_workspace_and_guard[28-write]`. Это `PytestUnraisableExceptionWarning` для незакрытых сокетов `AF_UNIX SOCK_STREAM`, созданных раньше в той же сессии; GC приписал их этим тестам, порядок тестов случайный. Оба теста проходят отдельно и вместе со своими модулями. Источник утечки не установлен. |
| `release-verify-unit`: `test-release-config-fixtures` | прошла |
| `release-verify-unit`: `test-release-shared-fixtures` | под `make -k` не запускалась: она зависит от упавших `test-go` и `test-runtime-hardening`. Её команда, выполненная отдельно на отчётах этого прогона (`python3 scripts/shared_fixture_gate.py --go-report reports/go-unit.jsonl --runtime-log reports/shared-fixture-reads.log`): 261 fixtures, 293 Go и 198 Python shared tests, rc 0. |
| `release-verify-ui` (`ui-verify`) | прошла |
| `release-verify-families` | прошла |
| `release-verify-browser-a` | прошла: mocked browser gate — 91 тест, ui-stack — 9 top-level tests |
| `release-verify-browser-b` | прошла: `TestManagedEvalsExternalStack` |
| `release-verify-race` | прошла: 45 пакетов |
| `release-verify-race-discovered` | прошла: 54 пакета |
| `release-verify-integration` | прошла: Go integration — 19 пакетов и 190 top-level tests, `verify-audit` ok; `internal/findingintake` без race ok; `test-artifact-filesystem` (`internal/artifacts`, `internal/findingintake`) ok |
| `release-verify-process-a`, `-b`, `-c` | выполняются. `TestAgentSkillsMVPProcesses` входит в process-c. |

**Отдельные перезапуски** (та же база, во время стадий process):

- `go test -count=1 -run '^TestStandbyServerAnswersPublicAPIWithRetryableEnvelope$' ./tests/integration/lease/` — PASS.
- `go test -count=1 -run '^TestPostgresControlPlaneLeaseHolderIgnoresOtherDatabases$' ./internal/app/` — FAIL с другим сообщением: `lease held for another database blocked this one = (false, <nil>)`. Control Plane lease — advisory lock на уровне базы, а тест изолирован схемой. Server стадии process в той же базе `contractor_test_x2` в это время держал ключ (видно в `pg_locks`). Повтор отложен до окончания gate.
- Q6: `go test -count=1 -run '^(TestPostgresLivePlannerSurvivesExpiredClaimBeforeRenewal|TestPostgresClaimLossDuringFinalizationPreventsReportAndProgression|TestClaimRenewalRetriesTransientErrorsUntilLeaseMargin|TestClaimRenewalRenewsEveryThirdOfTheLease)$' ./internal/scheduler/` — 4 PASS.
- Runtime: `uv run pytest -W error -p no:randomly` для двух падений вне списка V401-007 — PASS (5 cases). Их модули целиком дают только два известных падения (d) `content_scan_stops_after_first_binary_chunk`.

`TestRealGitReadOnlyContainer` (`internal/gitimport`) в release-verify не
входит: `scripts/release_integration_tests.py` исключает его («requires Podman
and the pinned read-only container image»), он запускается opt-in через
`make test-git-artifacts`. Тест пропускается только при отсутствии бинарника
`podman`. На этом хосте `podman` установлен, но локального образа нет
(`--pull=never docker.io/library/alpine:latest`), поэтому opt-in запуск упал
бы. Это предусловие opt-in gate, а не дефект. Расхождение между «pinned» в
исключении и плавающим тегом `alpine:latest` в тесте отмечено для интегратора.
Образы не скачивались, хост не менялся.

**Проверки ветки** (после merge `fccd443c`, с этой записью): `python3 scripts/check_spec.py` — 47 specification documents are consistent; `make lint` — rc 0.

**Состояние критериев V407-039.**

- A1 выполнен: у каждого из B1–G15 есть решение, доказательства и владелец (журнал решений).
- A2 выполнен: Q1–Q6 имеют проверенный ответ (у Q5 и Q6 — с принятым риском), Q7 закрыт явным решением о риске. Ни неизмеренный результат, ни незапущенный платформенный тест не засчитан как pass.
- A3 выполнен: archived V407-036, `depends_on`, решение G14 и `tasks/BACKLOG.md` согласованы.
- A4 не выполнен: gate не завершён, а `test-runtime-hardening` на macOS падает на платформенном наборе V401-007, на что не влияет никакой повтор на этом хосте. Зелёный `make release-verify` нужен на Linux: ubuntu-24.04 в CI через `workflow_dispatch` или тег `v*`. Другой путь — после V401-007.
