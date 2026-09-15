# Ревью архитектуры: оркестрация и хранилище

Дата: 2026-09-15. Исходная версия: `0e51f65a`.
Схема: [artitecture.likec4](../spec/artitecture.likec4).
Это проверка сценариев и выбранных граничных случаев, а не доказательство
отсутствия всех ошибок в перечисленных подсистемах.

## Подтверждённые проблемы

### P1 — потеря исключительного владения Run во время отмены

Views: `schedulerConcurrency`, `stageExecution`, `operationalBudgets`.

В [Scheduler](../../internal/scheduler/scheduler.go) продление SQL claim зависело
от контекста выполнения Planner. Отмена через API или обнаруженная при polling
отмена завершала продление, хотя тот же lane продолжал очистку под родительским
контекстом. Если очистка занимала больше оставшегося срока claim, другой lane
мог получить тот же cancelling Run. Это нарушало исключительность обработки
и могло запускать параллельные abort/release и изменения жизненного цикла.

Исправлено: контекст владения claim отделён от контекста выполняемой работы;
продление продолжается во время очистки. Потеря claim отменяет контекст владения.
Продление, проигравшее атомарному завершению Run и снятию claim, больше не
интерпретируется как захват другим lane.

[PostgreSQL regression](../../internal/scheduler/cancellation_claim_postgres_test.go)
задерживает отмену дольше трёх исходных сроков claim и пытается захватить Run
из конкурирующего lane. До исправления захват успешно происходил для обоих
способов отмены; после исправления возвращается `ErrNoWork`, а исходный lane
завершает Run и снимает claim.

### P1 — запись состояния прежним владельцем после потери claim

Views: `schedulerConcurrency`, `stageExecution`, `executionTelemetry`, `operationalBudgets`.

Независимая перепроверка выявила связанную ошибку: после настоящей потери
claim операции сохранения отчёта и завершения Stage использовали
`context.Background`/`WithoutCancel`. Старый lane мог завершить Run и стереть
claim нового владельца. Добавлен общий [cleanup context](../../internal/scheduler/claim_context.go):
он сохраняет конечную очистку при обычной отмене, но отслеживает потерю
исходного claim, даже если execution уже получил другую причину отмены.
Этот контекст передаётся в запись отчётов, метрик и переходов состояния.
Отмена действует после обнаружения `ErrClaimLost`; она не отменяет SQL commit,
уже завершившийся до обнаружения потери владения. Проверка claim ID внутри
каждой транзакции не вводилась. Освобождение terminal resources остаётся
отдельной идемпотентной обязанностью maintenance/recovery.

[Регрессия потери claim](../../internal/scheduler/claim_loss_postgres_test.go)
останавливает запись отчёта, атомарно передаёт просроченный claim конкурирующему
lane и проверяет, что прежний владелец не записал report, не завершил Stage/Run
и не удалил replacement claim. До исправления все эти запреты нарушались.

## Проверка каждого view

| View | Проверенный сценарий и граничные случаи | Результат / код |
| --- | --- | --- |
| `index` | SPA отдельно от Go, отдельный приватный TLS listener, прямые model calls, PostgreSQL и optional filesystem | Соответствует composition; HTTPS публичного API предполагает deployment TLS termination. [app.go](../../internal/app/app.go), [composition_http.go](../../internal/app/composition_http.go), [composition_execution.go](../../internal/app/composition_execution.go). |
| `platformOverview` | Общий Run service, Audit child Run, доступ Runtime только к выделенному RunScope, необязательные метрики | Границы соответствуют composition. [composition_audit.go](../../internal/app/composition_audit.go), [composition_http.go](../../internal/app/composition_http.go). Детали Audit проверены в [отдельном разделе](architecture-audits.md). |
| `serverOverview` | Отдельные контроллеры Scheduler/Audit/Project, общие artifact/config services, независимый profiling | Подтверждена сборка зависимостей; диаграмма показывает логические зависимости, не исчерпывающий граф вызовов функций. [composition_execution.go](../../internal/app/composition_execution.go), [composition_control.go](../../internal/app/composition_control.go). |
| `stageExecution` | Атомарное создание Run, повтор idempotency key, pin input/Skill/config, pause между Stage, partial prepare и отмена | Исправлены два P1 владения Run, описанные выше. Fork и Run создаются в одной транзакции; pause блокирует обычный admission, не отмену. [runservice/public.go](../../internal/runservice/public.go), [scheduler/postgres.go](../../internal/scheduler/postgres.go), [runtime_client.go](../../internal/controlplane/runtime_client.go). |
| `plannerAndA2A` | Корреляция task/context/subtask, replay завершённого Planner, malformed/oversized response, deadline, redirects, URI SAN | Новых подтверждённых дефектов не найдено. A2A проверяет идентичность ответов и allocation tenant; recovered completion не даёт права повторного вызова. [a2a/client.go](../../internal/planner/a2a/client.go), [passthrough.go](../../internal/planner/passthrough.go), [session/service.go](../../internal/planner/session/service.go). |
| `plannerArtifacts` | Exact revision, чужой Run, запись Memory одновременно с завершением Stage, потеря ответа на CAS | Новых дефектов не найдено. Memory удерживает Run/Stage locks в порядке жизненного цикла; ссылки и revisions скрыты инструментом. [artifact_inspector.go](../../internal/planner/artifact_inspector.go), [memory/postgres.go](../../internal/memory/postgres.go), [streamline/memory_tools.go](../../internal/planner/streamline/memory_tools.go). |
| `artifacts` | Create/CAS/frozen output, isolated scopes, rollback forks, nested transfer admission, отсутствующие bytes | Новых дефектов не найдено в проверенных путях. Registry является authority; наличие metadata не заменяет проверку bytes. [postgres_write.go](../../internal/artifacts/postgres_write.go), [postgres_fork.go](../../internal/artifacts/postgres_fork.go), [blob_context.go](../../internal/artifacts/blob_context.go). |
| `controlPlane` | Повтор регистрации, чужой principal, смена instance на том же endpoint, expired lease, несовместимые capabilities, изменение label revision при placement | Новых дефектов не найдено. Assignment ищет полное распределение совместимых Workers; размещение повторно проверяет revisions и закрепляет выбранные настройки. [registry.go](../../internal/controlplane/registry.go), [placement.go](../../internal/controlplane/placement.go), [runtime_client.go](../../internal/controlplane/runtime_client.go). |
| `executionTelemetry` | Incomplete report, отсутствующий Runtime при finalize, поздний report, bound identity, отказ необязательного exporter | Исправлен P1: прежний lane прекращает запись отчётов после потери claim. Scheduler сохраняет placeholder report и обрабатывает semantic result независимо от доставки OTLP. [scheduler.go](../../internal/scheduler/scheduler.go), [telemetry/repository.go](../../internal/telemetry/repository.go), [planner_otlp.go](../../internal/telemetry/planner_otlp.go). Runtime exporter проверен [отдельно](architecture-runtime.md). |
| `runtimeConfiguration` | Дубликаты и порядок labels, same-layer conflict, immutable refs, поздняя смена binding/Agent labels, credential kind/gateway mismatch | Новых дефектов не найдено. Run snapshot фиксирует binding revision; placement перепроверяет выбранный Agent и credentials. [run_snapshot.go](../../internal/runtimeconfig/run_snapshot.go), [resolver.go](../../internal/runtimeconfig/resolver.go), [placement.go](../../internal/controlplane/placement.go). |
| `bootstrapSecrets` | Неверный master key, symlink/права файла, смена AAD identity, удаление credential с Run/Audit holds, незавершённая remote operation | Новых дефектов не найдено в проверенных путях. Secret-файлы читаются при bootstrap; ciphertext связан с purpose/identity, удаление отделено от использования lifecycle barrier. [master_key.go](../../internal/credentials/master_key.go), [crypto.go](../../internal/credentials/crypto.go), [lifecycle.go](../../internal/credentials/lifecycle.go). |
| `schedulerConcurrency` | Лимит lanes, resize/drain, lost wake, failed settings refresh, paused owners, claim renewal при cleanup | Исправлены два P1 выше; существующие supervisor и PostgreSQL tests также прошли. [scheduler.go](../../internal/scheduler/scheduler.go), [supervisor_test.go](../../internal/scheduler/supervisor_test.go), [supervisor_postgres_test.go](../../internal/scheduler/supervisor_postgres_test.go). Один активный Server остаётся обязательной предпосылкой. |
| `artifactBackends` | Installation backend pin, corruption против missing bytes, dedup repair race, lost commit acknowledgement, orphan cleanup, symlinks и payload limits | Новых дефектов не найдено. Файловая публикация создаёт новую генерацию; reuse разрешён только после чтения точного key. Cleanup выполняется offline и не удаляет referenced metadata из-за утраты файла. [blob_dedup.go](../../internal/artifacts/blob_dedup.go), [blob_filesystem.go](../../internal/artifacts/blob_filesystem.go), [blob_cleanup.go](../../internal/artifacts/blob_cleanup.go). Power-loss durability прямо исключена spec 23. |
| `operationalBudgets` | Отдельные transport/operation/cleanup budgets, повтор finalize/abort после restart, сохранение absolute deadline, более ранний родительский deadline | Исправлена независимость claim ownership от отмены работы. Сохранённые IDs/deadlines не обновляются при recovery; проверки настроек исключают невалидные значения. [configured_deadlines_test.go](../../internal/scheduler/configured_deadlines_test.go), [runtime_deadline_test.go](../../internal/controlplane/runtime_deadline_test.go), [operational_settings.go](../../internal/app/operational_settings.go). |

## Рефакторинг

- Выполнено разделение контекстов выполнения и владения Run: это устраняет
  смешение двух разных сроков жизни без изменения публичных контрактов.
- `scheduler.go` велик. Выделение supervisor/claim renewal и terminal recovery
  в отдельные файлы того же пакета упростит следующие проверки; перенос сам по
  себе не исправляет поведение, поэтому массовая перестановка сейчас не нужна.
- Planner Inspector и Scheduler ArtifactResolver читают payload ради проверки
  metadata. Возможен общий внутренний интерфейс проверки exact artifact;
  простая замена на SQL metadata-only read убрала бы проверку missing/corrupt
  bytes. Такой рефакторинг требует сохранения integrity-контракта и замеров.
- Уточнены две формулировки спецификации: renewal при cleanup и общий guard
  local-direct workspace при раздельных session locks у Toolsets.

## Проверки

На выделенной временной PostgreSQL 17 выполнены:

```sh
go test -race -count=1 ./internal/scheduler
go test -race -p 2 -count=1 ./internal/controlplane/... ./internal/planner/... \
  ./internal/artifacts/... ./internal/runtimeconfig/... ./internal/memory/... \
  ./internal/runservice/... ./internal/settingsstore/... \
  ./internal/httpapi/privateartifacts/...
```

Обе команды прошли с установленным `CONTRACTOR_TEST_DATABASE_URL`.
Финальный Scheduler-прогон после доработки cleanup context: 16.356 s;
целевые claim/cancellation tests дополнительно прошли с `-race -count=3`.
Artifact suite включает явные filesystem fixtures; это не полный production
container gate для обоих backend. `likec4 validate docs/spec` прошёл.
Проверки живых моделей, аварийного отключения питания и нескольких активных
Server не выполнялись.
