# Ревью архитектуры — 2026-09-15

## Итог

Проверены **все 42 views** из [artitecture.likec4](../spec/artitecture.likec4).
В запросе указан `architecture.likec4`; фактическое имя файла в репозитории —
`artitecture.likec4`. База ревью: `0e51f65a`, чистый `main` перед изменениями.

Найдены и исправлены **4 проблемы P1 и 3 проблемы P2**. P1 означает высокий
приоритет из-за потери корректного результата, нарушения исключительного
владения или остановки жизненного цикла. P2 — ошибка ограниченного сценария,
которую также следует устранить. Это классификация данного ревью.

Проверка включает связи модели, сценарии реализации, ошибочные переходы,
повторные операции, отмену, отказ хранения, конкуренцию и осмысленность
рефакторинга. Формулировка «новых дефектов нет» относится к описанным сценариям.
Она не доказывает отсутствие всех возможных ошибок и не заменяет release gates.

## Исправления по приоритету

| Приоритет | Ошибка и последствие | Что изменено |
| --- | --- | --- |
| P1 | Временная ошибка чтения evidence необратимо создавала `invalid-result` receipt и разрешала удаление исходного Run | [Importer](../../internal/auditimport/check_results.go) различает некорректную ссылку и отказ хранения; повторяет сбор того же результата без новой Worker attempt. |
| P1 | После отмены прекращалось продление claim; другой lane мог получить Run, пока старый выполнял cleanup | [Scheduler](../../internal/scheduler/scheduler.go) разделяет execution и ownership, продлевает claim при cleanup и отдельно обрабатывает потерю claim и terminal commit. |
| P1 | После настоящей потери claim прежний lane продолжал запись отчётов и мог завершить Run, стирая claim нового владельца | Общий [cleanup context](../../internal/scheduler/claim_context.go) сохраняет bounded cleanup при обычной отмене, но прекращает записи при потере владения. |
| P1 | `SKIP LOCKED` пропускал последний Audit, после чего Project навсегда оставался в `draining` | [Project controller](../../internal/projectlifecycle/controller.go) проверяет выставление всех deletion intents и восстанавливает уже застрявший `draining`. |
| P2 | При неудаче второго Toolset ресурсы первого терялись для rollback | [AllocationService](../../runtime/src/contractor_runtime/allocation/service.py) передаёт cleanup ownership сразу после создания каждого инструмента, до следующего await и валидации. |
| P2 | Ошибка единственного/последнего RuntimeAdapter принималась за успешную подготовку | [AdapterHost](../../runtime/src/contractor_runtime/adapters/host.py) возвращает host только после успешного завершения всех проверок; ошибка вызывает cleanup. |
| P2 | Отписка до subscribe ACK сбрасывала соседние WebSocket-потоки | [RunEventsManager](../../ui/src/events/run-events.ts) сохраняет отменённую подписку до подтверждения и корректно обрабатывает поздние ответы. |

Для исправленных сценариев добавлены регрессионные тесты; исходные ошибки
воспроизведены до изменения реализации. Отдельные подробности и команды
содержатся в отчётах ниже.

## Матрица покрытия

| № | View | Результат | Подробности: сценарии, код, проверки |
| --- | --- | --- | --- |
| 1 | `index` | Проверен; новых подтверждённых дефектов нет | [Оркестрация / хранилище](architecture-orchestration.md) |
| 2 | `platformOverview` | Проверен; новых подтверждённых дефектов нет | [Оркестрация / хранилище](architecture-orchestration.md) |
| 3 | `serverOverview` | Проверен; новых подтверждённых дефектов нет | [Оркестрация / хранилище](architecture-orchestration.md) |
| 4 | `runtimeOverview` | Исправлено: частичный prepare | [Runtime](architecture-runtime.md) |
| 5 | `productSurface` | Проверен; новых подтверждённых дефектов нет | [API / UI / Operations](architecture-surface.md) |
| 6 | `productEvents` | Исправлено: ранняя отписка | [API / UI / Operations](architecture-surface.md) |
| 7 | `workspaceLifecycle` | Исправлено: locked Audit при deletion | [Audits / lifecycle](architecture-audits.md) |
| 8 | `stageExecution` | Исправлено: claim при cleanup | [Оркестрация / хранилище](architecture-orchestration.md) |
| 9 | `plannerAndA2A` | Проверен; новых подтверждённых дефектов нет | [Оркестрация / хранилище](architecture-orchestration.md) |
| 10 | `plannerArtifacts` | Проверен; новых подтверждённых дефектов нет | [Оркестрация / хранилище](architecture-orchestration.md) |
| 11 | `runtimeLifecycle` | Исправлено: Toolsets и adapters | [Runtime](architecture-runtime.md) |
| 12 | `workerSessions` | Проверен; новых подтверждённых дефектов нет | [Runtime](architecture-runtime.md) |
| 13 | `runtimeFilesystems` | Проверен; новых подтверждённых дефектов нет | [Runtime](architecture-runtime.md) |
| 14 | `runtimeCodeAnalysis` | Проверен; новых подтверждённых дефектов нет | [Runtime](architecture-runtime.md) |
| 15 | `runtimeAdapters` | Исправлено: последний невалидный adapter | [Runtime](architecture-runtime.md) |
| 16 | `artifacts` | Проверен; новых подтверждённых дефектов нет | [Оркестрация / хранилище](architecture-orchestration.md) |
| 17 | `artifactCleanup` | Исправлен upstream Project deletion | [Audits / lifecycle](architecture-audits.md) |
| 18 | `workerArtifacts` | Проверен; новых подтверждённых дефектов нет | [Runtime](architecture-runtime.md) |
| 19 | `controlPlane` | Проверен; новых подтверждённых дефектов нет | [Оркестрация / хранилище](architecture-orchestration.md) |
| 20 | `executionTelemetry` | Исправлено: запись после потери claim | [Оркестрация / хранилище](architecture-orchestration.md) |
| 21 | `agentSkills` | Проверен; новых подтверждённых дефектов нет | [Runtime](architecture-runtime.md) |
| 22 | `runtimeConfiguration` | Проверен; новых подтверждённых дефектов нет | [Оркестрация / хранилище](architecture-orchestration.md) |
| 23 | `bootstrapSecrets` | Проверен; новых подтверждённых дефектов нет | [Оркестрация / хранилище](architecture-orchestration.md) |
| 24 | `audits` | Исправлен сбор evidence | [Audits / lifecycle](architecture-audits.md) |
| 25 | `auditBaseline` | Проверен; новых подтверждённых дефектов нет | [Audits / lifecycle](architecture-audits.md) |
| 26 | `auditCollection` | Исправлено: storage error / invalid result | [Audits / lifecycle](architecture-audits.md) |
| 27 | `auditReviewAndPrograms` | Проверен; новых подтверждённых дефектов нет | [Audits / lifecycle](architecture-audits.md) |
| 28 | `schedulerConcurrency` | Исправлено: claim renewal / потеря claim | [Оркестрация / хранилище](architecture-orchestration.md) |
| 29 | `catalog` | Проверен; новых подтверждённых дефектов нет | [API / UI / Operations](architecture-surface.md) |
| 30 | `artifactBackends` | Проверен; новых подтверждённых дефектов нет | [Оркестрация / хранилище](architecture-orchestration.md) |
| 31 | `gitArtifacts` | Проверен; новых подтверждённых дефектов нет | [API / UI / Operations](architecture-surface.md) |
| 32 | `gitSettings` | Проверен; новых подтверждённых дефектов нет | [API / UI / Operations](architecture-surface.md) |
| 33 | `podmanSandbox` | Код и обычные tests; real Podman gates пропущены | [Runtime](architecture-runtime.md) |
| 34 | `serverPerformance` | Проверен; новых подтверждённых дефектов нет | [API / UI / Operations](architecture-surface.md) |
| 35 | `allocationPerformance` | Проверен; новых подтверждённых дефектов нет | [API / UI / Operations](architecture-surface.md) |
| 36 | `processBootstrap` | Проверен; новых подтверждённых дефектов нет | [API / UI / Operations](architecture-surface.md) |
| 37 | `auditWorkerCompletion` | Runtime/Go tests; V39-007 release gate не запускался | [Runtime](architecture-runtime.md) |
| 38 | `findingsCollections` | Проверен; новых подтверждённых дефектов нет | [Audits / lifecycle](architecture-audits.md) |
| 39 | `portableEvaluations` | Исходники + 225 offline tests; live не проверен | [API / UI / Operations](architecture-surface.md) |
| 40 | `operationalBudgets` | Исправлено: ownership при cleanup | [Оркестрация / хранилище](architecture-orchestration.md) |
| 41 | `workerTelemetry` | Исправлена валидация adapter | [Runtime](architecture-runtime.md) |
| 42 | `durableState` | Проверен; новых подтверждённых дефектов нет | [Audits / lifecycle](architecture-audits.md) |

## Рефакторинг и уточнение схемы

Выполнены локальные изменения, связанные с найденными ошибками: отдельное
владение Run, ранняя передача ресурсов rollback, явное успешное завершение
подготовки адаптеров и общий путь Audit deletion intent для двух фаз.
Публичные API и схема БД не менялись.

В LikeC4 уточнена связь Project deletion с Audit lifecycle: Project controller
пишет SQL intent, а Audit controller выполняет drain/purge. В spec 13 уточнено,
что раздельные session locks Toolsets сочетаются с общим workspace guard.
Spec 20 описывает владение claim во время очистки.

Кандидаты на последующий рефакторинг: выделение supervisor/terminal recovery
из большого `scheduler.go` и общий интерфейс проверки exact artifacts для
Planner/Scheduler. Механическое слияние разных lifecycle/authority boundaries
и замена проверки bytes только чтением metadata не рекомендуются.
Обоснование приведено по каждому разделу.

## Проверки и границы результата

- Go: целевые пакеты оркестрации, artifacts, configuration, memory, API,
  telemetry, auth, performance и Git; PostgreSQL 17 в отдельном временном
  контейнере. Scheduler и audit/finding/project suites, а также
  orchestration/storage suites прошли с `-race`.
- Runtime: полный финальный pytest — **1788 passed, 30 skipped**; Ruff чист.
- UI: 70 целевых tests, дополнительно финальные 21 event tests, 7 Node server
  tests; TypeScript, ESLint и Prettier для изменений прошли.
- Portable evaluator: read-only проверка sibling `playground-v2` на
  `bd2a557`, **225 offline tests passed**. Внешние файлы не менялись.
- LikeC4: `likec4 validate docs/spec`; проверено покрытие всех 42 view IDs.
- `go vet ./...` и `git diff --check` прошли.

30 Runtime skips — 3 live Gateway теста и 27 opt-in Podman gates.
Полный browser E2E, production remote hosts, live evaluator campaign и
V39-007 release gate не запускались. UI проверялся на установленном Node
26.7, а проект декларирует 24.20.x. Утверждения схемы об исторических
release gates не выдаются за текущий прогон.

Постоянная порча artifact storage удерживает Audit collection и удаление до
восстановления bytes. Это согласованная политика сохранения evidence, а не
обещание бесконечно успешного retry. Отдельного восстановления из backup,
отключения питания и многосерверного deployment этот обзор не проверяет.
