# V60-005 — execution recovery и конкурентные переходы

Дата: 2026-09-20. Исходный код: `8edeabf21194f5bba8dd258d53a072af713c630e`.
Ревью выполнено в отдельном worktree `v60-deep-review`; основной checkout и
внешние сервисы не изменялись. Задача проверяет Scheduler, RunStore и границу
Control Plane/Runtime; она не является заявлением о корректности всего проекта.

## Контракт и история

Прочитаны сценарии Scheduler, отмены и восстановления в
[spec 00](../spec/00-workflow-and-planner.md), жизненный цикл и durable deadlines
в [spec 04](../spec/04-execution-lifecycle-and-metrics.md), owner Queue Pause в
[spec 18](../spec/18-run-and-workspace-lifecycle-controls.md) и concurrency lanes
в [spec 20](../spec/20-scheduler-concurrency-control.md).

История определяет границы проверки:

- `78eef87e` ввёл Queue Pause, атомарное ожидание следующего Stage и освобождение
  завершивших работу allocations во время этого ожидания. Тест
  `TestSchedulerOwnerQueuePauseDrainsCurrentStageWithoutAdmittingNext` с самого
  введения функции проверяет `finalizing` вместе с `ReleaseCompletedAt`.
- `7ca8daa6` отделил владение claim от отмены Planner и запретил дальнейшие
  записи после обнаружения `ErrClaimLost`. [Предыдущее архитектурное ревью](../reviews/architecture-orchestration.md)
  явно не обещает SQL fencing до обнаружения потери claim.
- `9ef3eec6` выделил lifecycle/Planner части Scheduler без изменения этого
  поведения; `74d3ede3` сохранил escalation history при ручном продолжении.
  Эти изменения учтены, прежние исправления не объявляются новыми находками.

## Проверенные сценарии

| Сценарий | Граница и действующая защита | Исполняемая проверка |
| --- | --- | --- |
| Cancel конкурирует с успешным результатом | PostgreSQL блокирует Run перед Stage; принятие результата, точные outputs и Run state изменяются одной транзакцией. При победе отмены outputs не создаются. | `TestPostgresCancelAndSuccessRaceSerializesOnRunRow`, `TestPostgresAcceptanceRollsBackStageAndOutputWhenRunCASLoses` |
| Поздний Planner candidate после отмены | `running -> finalizing` и `running -> aborting` имеют CAS; сохранённая termination не заменяется результатом. | `TestSchedulerCancelInterruptsPlannerAndIgnoresLateCandidate`, `TestSchedulerCancelAcceptsAlreadyFinalizingResultForAuditOnly`, `TestPostgresIntegrationStageLifecycleAndSessions` |
| Cleanup дольше первоначального claim | Отдельный ownership context продолжает renewal после immediate и polled cancellation; конкурирующий SQL claimant не получает Run. | `TestPostgresCancellationCleanupRetainsRunClaim` — оба подслучая |
| Обнаруженная потеря claim во время finalization | Реальный SQL claim передаётся другому lane, прежний context отменяется; старый report, Stage/Run commit и снятие replacement claim запрещены. | `TestPostgresClaimLossDuringFinalizationPreventsReportAndProgression` |
| Потеря volatile state и повтор finalization | Новый Scheduler использует сохранённый candidate либо прерывает потерянный running attempt; не вызывает прежний Planner повторно. | `TestSchedulerRecoversDurableFinalizingCandidateWithoutPlannerInvocation`, `TestSchedulerAbortsRunningStageWhenVolatileControlPlaneStateWasLost` |
| Restart между terminal transaction и release | Durable markers сохраняют необходимость release; повторяются только оставшиеся allocations, отказ одного не блокирует другой Run. | `TestSchedulerRestartRetriesReleaseAfterTerminalTransaction`, `TestTerminalReleaseRecoveryRetriesOnlyRemainingLiveAllocations`, `TestTerminalReleaseFailureDoesNotBlockClaimPath` |
| Абсолютные terminal deadlines | Recovery использует прежние finalization/abort IDs и deadlines, даже при других process settings; неполные reports не превращаются в новую семантическую работу. | `TestConfiguredTerminalBudgetsAreSavedAndRecoveryDoesNotRenew`, `TestSchedulerMetricsCancelDeadlineTerminatesWithIncompleteReportAndFencedAllocation` |
| Pause конкурирует с admission | SQL lock упорядочивает оба направления гонки; чужой owner не затрагивается. Terminal-only commit и cancellation остаются доступны. | `TestPostgresOwnerQueueControlSerializesWithStageAdmission`, `TestPostgresQueuePauseAllowsTerminalResultCommit`, три `TestSchedulerOwnerQueuePause...` |
| Pause перед retry | Termination acceptance и новая попытка не расходятся: при pause транзакция не меняет Stage/decision, resume создаёт ровно следующую попытку. | `TestPostgresRetryProgressionAtomicallyCreatesFreshAttempt` |
| Concurrency, saturation и resize | Реальные SQL claims ограничены lanes; уменьшение лимита дожидается drain, deferred Run уступает очередь, отсутствующая settings row закрывает admission. | `TestPostgresSchedulerSupervisorBoundsRealRunClaims`, `TestPostgresSchedulerSupervisorResizesAndDrainsDurableLanes`, `TestPostgresClaimRunnableRunRotatesAfterDeferredRelease`, `TestSchedulerSupervisorFailsClosedAndShutdownReleasesClaim` |
| Асимметричная partition и Runtime restart | Подтверждённые heartbeat echoes и monotonic lease исключают продление старым ack и reuse до release. | `TestAsymmetricPartitionsFenceBothAuthorities`, `TestRuntimeRestartIsWithheldUntilOldAuthorityIsReleased`, пять Python `test_lease_watchdog.py` |

Основные прочитанные реализации:
[transaction boundaries](../../internal/scheduler/postgres.go),
[claim/supervisor](../../internal/scheduler/supervisor.go),
[cleanup context](../../internal/scheduler/claim_context.go),
[result finalization](../../internal/scheduler/stage_finalization.go),
[termination](../../internal/scheduler/stage_termination.go),
[release recovery](../../internal/scheduler/allocation_release.go),
[Stage CAS](../../internal/runstore/stage_store.go) и
[Queue admission](../../internal/runstore/queue_control_store.go).

## Результаты анализа

**Accepted: ограничение claim detection.** Наличие нового claim в PostgreSQL
само по себе не отменяет уже завершённую транзакцию прежнего владельца.
Проверки доказывают запрет записей после обнаружения потери владения. Один
активный Server/Scheduler остаётся поддерживаемой топологией; distributed
semaphore и проверка claim ID в каждой SQL mutation не добавлялись.

**Accepted, документация уточнена: release при Queue Pause.** Старый invariant
spec 04 безусловно требовал terminal Stage до release. Между тем атомарный
next/retry/escalation admission может ожидать resume, сохраняя уже записанный
`finalizing` candidate или `aborting` termination. Runtime уже завершает drain,
и его allocations освобождаются с обычным подтверждением teardown. Это
закреплено реализацией и regression с `78eef87e`, а не новая runtime-семантика.
Spec 04 и 18 теперь явно описывают узкое исключение, неизменные IDs/deadlines,
атомарный commit и отсутствие повторного Planner invocation. Общие требования
к fencing, teardown и release acknowledgement не ослаблены.

**Rejected: cancellation прекращает claim renewal.** PostgreSQL regression
удерживает cleanup дольше трёх первоначальных claim durations для обоих путей
отмены; конкурирующий claim отклоняется. Исправление `7ca8daa6` сохранено.

**Rejected: finalizing result обязан победить отмену всего Run.** Result может
сохраниться для аудита Stage, но Run-row ordering запрещает output publication
и новую попытку после победившего `cancelling`. Это разные durable решения.

В рассмотренных сценариях новый воспроизводимый runtime-дефект не подтверждён.
Продуктовый код не изменён; уточнение спецификации устраняет расхождение текста
с давно установленным поведением.

## Выполненные проверки

Результаты команд и SHA-256 локальных логов фиксируются в
[evidence V60-005](../../tasks/evidence/v60-005.json). Go counts включают
подтесты и проверки matrix, поэтому не являются числом уникальных сценариев.

Среда: Go 1.25.6, Python 3.13.14 из locked uv environment, Node 24.20.0,
pnpm 11.24.0, отдельный PostgreSQL 17 с уникальными test schemas.
`GOFLAGS=-p=2 -v`, `GOMAXPROCS=2`, `PYTEST_ADDOPTS=-ra`; connection credentials
не входят в report/evidence. Существующие gate-команды выполнены без исключения
их составных пакетов и без подмены обязательных случаев.

| Команда | Результат | Время |
| --- | --- | --- |
| `make test-faults` | PASS; 1536 Go tests/subtests; 2407 Python passed, 34 skips | 400.849 s |
| `make test-lease-integration` | PASS; 11 Go tests/subtests, 5 Python passed | 11.938 s |
| `make test-lifecycle-controls-hardening` | PASS; 153 Go tests/subtests, 45 UI tests / 6 files | 48.850 s |
| `make test-scheduler-concurrency-hardening` | PASS; 152 Go tests/subtests | 23.130 s |

В обязательных Go/PostgreSQL и offline lifecycle случаях пропусков нет.
34 Python skips: 1 отдельный audit-completion E2E (обязательный отдельный gate
V60-008), 3 live LLM, 27 opt-in real Podman и 3 случая отсутствующего optional
sqlmap. Точные причины сохранены в evidence; эти сценарии не объявляются
пройденными этим запуском.

**Confirmed P2, исправлено V60-029:** первый `make test-faults` завершился
с exit 2 через 78.255 s: validator не нашёл
`TestPrepareAllCleansEveryReservationAfterPartialFailure` в старом файле.
`5fad87a3` перенёс regression в `internal/controlplane/runtime_batch_test.go`,
но не обновил matrix. Исправлен только путь; проверены прежние assertions о
fence/abort/release всех reservations. Focused matrix + named regression PASS
(7.255 s); полный повторный gate приведён в таблице. Исходный провал и оба
последующих запуска сохранены в [evidence V60-029](../../tasks/evidence/v60-029.json).

## Границы доказательства

- PostgreSQL проверки используют настоящий отдельный сервер БД; Scheduler и
  Worker test doubles контролируют окна гонок и сетевые исходы.
- Go lease integration — детерминированный in-memory Runtime harness с fake
  clock. Python watchdog проверяет реальный AllocationService с test factories;
  эти результаты не выдаются за испытание реального сетевого разделения.
- Restart-сценарии восстанавливают новый объект Scheduler из durable state;
  новый процесс Server, power loss и несколько активных Scheduler здесь не
  запускались. Они не требуются четырьмя gate-командами V60-005.
- UI проверки в lifecycle gate — component tests, не browser gate. Отдельные
  process/browser journeys принадлежат V60-008/009.
- Live-model и opt-in Podman/scanner проверки полного Runtime suite отделены
  от обязательных offline lifecycle случаев в evidence. Их пропуск не
  объявляется подтверждением соответствующей возможности.
