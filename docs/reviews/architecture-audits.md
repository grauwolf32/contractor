# Ревью архитектуры: Audits, findings и lifecycle

Дата: 2026-09-15. Источник: `docs/spec/artitecture.likec4` (так файл называется в репозитории).

Проверены все восемь перечисленных ниже views, их связи в `model`, реализация и относящиеся к ним тесты. Вывод «соответствует» означает отсутствие дополнительных подтверждённых ошибок в проверенных сценариях, а не доказательство отсутствия любых ошибок.

## Исправленные проблемы по приоритету

### P1 — Project deletion мог навсегда зависнуть из-за занятого Audit

`requestOneAuditDeletion` пропускал занятые строки через `SKIP LOCKED`. Если последний Audit был заблокирован другой транзакцией, а активных Runs уже не было, контроллер переходил из `cancelling` в `draining`. В этой фазе он ждал удаления Audit, которому ещё никто не выставил `deletion_requested_at`.

В [controller.go](../../internal/projectlifecycle/controller.go) добавлен барьер: пока хотя бы один Audit не получил запрос удаления, переход в `draining` откладывается. Выставление Audit deletion intent также выполняется в `draining`, чтобы после перезапуска восстановить проекты, уже застрявшие на предыдущей версии. Фазы остаются монотонными; receipt, release и artifact purge barriers сохранены.

Регрессия `TestProjectDeletionWaitsForLockedAuditBeforeDraining` удерживает настоящую блокировку PostgreSQL, проверяет ожидание без преждевременного перехода, перезапуск контроллера и завершение удаления. Второй подслучай начинает с уже застрявшего `draining`. До исправления первый сценарий воспроизвёл ошибку: `RunOnce` вернул `(true, nil)` вместо ожидания.

### P1 — Сбой чтения evidence превращался в окончательный результат Audit

В `prepareCheckResults` любая ошибка `ReadRunExact` становилась `evidence-reference-invalid`. Временная ошибка БД/хранилища или отмена контекста поэтому могла создать окончательный `invalid-result` receipt, израсходовать попытку элемента и открыть барьер удаления исходного Run. Успешный результат работника терялся из-за ошибки инфраструктуры.

В [check_results.go](../../internal/auditimport/check_results.go) ошибки отсутствующей/некорректной ссылки отделены от ошибок чтения. Ошибка хранилища возвращается контроллеру; execution остаётся в `collecting` и повторно обрабатывается без новой попытки работника. Проверка целостности также не выдаётся за ошибку содержимого, предложенного работником: данные остаются доступными для восстановления хранилища.

Новые [регрессионные тесты](../../internal/auditimport/evidence_read_test.go) покрывают временную ошибку, deadline, cancellation, integrity failure, отсутствие receipt/retained output до восстановления и успешный повторный сбор того же terminal execution. Отдельный тест сохраняет ожидаемое поведение действительно отсутствующей ссылки.

## Покрытие views

### `audits` — соответствует после исправления классификации ошибок сборщика

- Проверены stable execution intents, повторная отправка после разрыва между intent и Run creation, независимые item attempts внутри batch, ограничения количества Runs, epoch claims и строгая trusted Audit association. Контроллер не создаёт отдельный Scheduler для Audit.
- Источник terminal outcome — PostgreSQL Run state и точная пара generation/sequence; повторные уведомления не принимаются за новые результаты. Следующий запуск/round проходит через receipt и settlement barriers.
- Реализация: [auditcontroller/controller.go](../../internal/auditcontroller/controller.go), [auditcontroller/builder.go](../../internal/auditcontroller/builder.go), [auditstore/execution.go](../../internal/auditstore/execution.go), [auditstore/claims.go](../../internal/auditstore/claims.go), [runservice/audit.go](../../internal/runservice/audit.go).
- Проверки: `TestPostgresControllersConvergeWithoutDuplicateExecutionAttempts`, `TestPostgresControllerBatchBuildsOnePinnedRunForTwoItems`, `TestControllerReplaysIntentAfterSubmissionFailure`, `TestControllerLeavesMembersOmittedByByteBoundedPreparationReady`.

### `auditBaseline` — соответствует

- Start использует caller-owned transaction с `REPEATABLE READ`; `LockActiveAuditProject` блокирует смену Project target и deletion во время фиксации baseline. Evaluation Project не допускается.
- Runtime config, Workflow closure, Skills, стандартные пакеты и исходные ревизии фиксируются точно. Изменение binding после Start не подменяет baseline. Runtime labels на стороне Agent остаются отдельным allocation-time слоем, как указано в view.
- Неподдерживаемая комбинация capability/input откатывает baseline, artifacts и holds. Replay проверяется до обращения к изменившимся зависимостям; транзакционный credential lookup не требует дополнительного свободного соединения в пуле.
- Реализация: [auditservice/start.go](../../internal/auditservice/start.go), [auditservice/compatibility.go](../../internal/auditservice/compatibility.go), [projectstore/store.go](../../internal/projectstore/store.go), [auditstandards/catalog.go](../../internal/auditstandards/catalog.go).
- Проверки: `TestAuditDraftStartReplayAndAtomicUnsupportedRollback`, `TestAuditStartUsesOwningTransactionWithSaturatedPool`, `TestCatalogPostgresConcurrentSeedAndExactRetention`.

### `auditCollection` — исправлен P1 с ошибками чтения evidence

- Полное соответствие batch manifest проверяется до per-item acceptance; нельзя принять только корректную часть неполного/чужого набора результатов. Evidence и proposal должны принадлежать точному элементу.
- Missing/invalid/failed/cancelled outputs получают явные dispositions и coverage gaps. Run success не означает положительную оценку безопасности. Новая регрессия отделяет временные ошибки чтения от этих dispositions.
- Receipt и settlement записываются одной SQL-операцией под live claim. Delete Run повторно проверяет terminal state, завершение allocation release, receipt и tombstone provenance под блокировкой Run.
- Future-dispatch credential hold освобождается после закрытия dispatch и разрешения intents. Evidence остаётся до удаления Audit; при удалении исходного Run используются удержанные копии.
- Реализация: [auditimport/check_results.go](../../internal/auditimport/check_results.go), [auditstore/collection.go](../../internal/auditstore/collection.go), [auditstore/lifecycle.go](../../internal/auditstore/lifecycle.go), [runstore/delete_store.go](../../internal/runstore/delete_store.go).
- Проверки: весь пакет `auditimport`; `TestPostgresAuditBatchFailureRequeuesEveryMemberAndAllowsRegrouping`, `TestPostgresControllerDeletesActiveAuditWhileOwnerQueuePaused`, `TestPostgresFindingReceiptReplayRetentionAndRunDeletion`.

### `auditReviewAndPrograms` — соответствует

- Worker proposal сохраняет происхождение allocation/invocation; наличие proposal/receipt не подтверждает finding автоматически. Human decision связан с actor, request revision и digest точного subject.
- Проверены истечение review TTL, устаревший subject, повторная команда, frozen report candidate, review после удаления исходного Run, повторное использование proposal в новых rounds. Следующий round фиксирует consume-once relation в транзакции; worker-defined method не обходит active-check approval.
- Top 10 и ASVS pilot берут identifiers/denominator/evidence contract из сохранённых пакетов. Пять требований ASVS pilot не выдаются за полный ASVS Level 1.
- Реализация: [findingintake/service.go](../../internal/findingintake/service.go), [auditservice/finding_review.go](../../internal/auditservice/finding_review.go), [auditservice/action_review.go](../../internal/auditservice/action_review.go), [auditservice/next_round.go](../../internal/auditservice/next_round.go), [auditstore/round.go](../../internal/auditstore/round.go).
- Проверки: `TestAuditFindingReviewHistoryAndDeletedRunProvenance`, `TestAuditReportAcceptanceUsesFrozenCandidate`, `TestPostgresAcceptNextRoundIsAtomicReplaySafeAndConsumeOnce`; unit tests `auditdomain`/`auditstandards`.

### `findingsCollections` — соответствует

- Publisher захватывает snapshot в `REPEATABLE READ`, авторизует каждый источник и receipt, раскрывает все contributions выбранного finding и проверяет точную revision. ZIP и replay receipt публикуются атомарно.
- Проверены foreign source/receipt, missing bytes, пустой выбор, неизвестный receipt, ошибка между записью ZIP и receipt, повторный запрос после удаления Run, изменение review после publication. Replay возвращает исходный snapshot до чтения изменяемых источников.
- После удаления Run разрешены только удержанные Audit copies с проверенным владельцем. Содержимое collection заморожено; последующие review не переписывают уже опубликованный ZIP. Сам ZIP остаётся обычным exact input следующего Run.
- Реализация: [findingintake/collection_publication.go](../../internal/findingintake/collection_publication.go), [auditservice/finding_collection.go](../../internal/auditservice/finding_collection.go), [auditdomain/finding_collection.go](../../internal/auditdomain/finding_collection.go). Worker-side `list_findings` проверяется отдельно в runtime review.
- Проверки: `TestFindingCollectionPublicationRunRetentionAndReplay`, `TestFindingCollectionPublicationRejectsPartialForeignAndInterruptedWrites`, `TestFindingCollectionAuditContributionsAndPinnedReview`.

### `workspaceLifecycle` — исправлен P1 с заблокированным Audit

- Project deletion durable и монотонен: cancellation → drain → Run purge → Project artifact purge. Новый барьер не путает пропущенный занятый Audit с отсутствующим Audit; уже застрявшая deletion восстанавливается без отката фаз.
- Queue pause сериализуется через owner control row только с normal Stage admission. Pause не препятствует cancellation/collection/release. Перед Project purge повторно проверяется отсутствие и Runs, и Audits.
- Реализация: [projectlifecycle/controller.go](../../internal/projectlifecycle/controller.go), [runstore/queue_control_store.go](../../internal/runstore/queue_control_store.go), [projectstore/store.go](../../internal/projectstore/store.go), [runstore/delete_store.go](../../internal/runstore/delete_store.go).
- Проверки: обе PostgreSQL-регрессии в `projectlifecycle`, существующий `TestProjectDeletionCancelsDrainsPurgesAndRetainsSharedResources`, `TestPostgresControllerDeletesActiveAuditWhileOwnerQueuePaused`.
- Уточнена диаграмма: связь `projectDeletionController -> auditService` логическая. Фактически `requestOneAuditDeletion` пишет SQL intent, а `auditController`/`auditStore` выполняют drain и purge. Подпись модели и двух views теперь явно обозначает SQL intent и ожидание drain/purge. Это уточнение структуры вызовов, а не отдельная ошибка жизненного цикла.

### `artifactCleanup` — соответствует после исправления Project deletion

- Run purge выполняется в транзакции под authority lock; Project purge стартует только после полного drain. Reference checks отделены от получения locks, чтобы после ожидания видеть committed forks/pins при `READ COMMITTED`.
- Проверены сохранение shared User artifacts/Skills/credentials и owner queue state после Project deletion, порядок version/blob locks и запрет удаления referenced metadata из-за отсутствующих bytes. Удаление filesystem bytes после commit может оставить orphan; rollback не должен удалить ещё используемые данные.
- Реализация: [runstore/delete_store.go](../../internal/runstore/delete_store.go), [projectlifecycle/controller.go](../../internal/projectlifecycle/controller.go), [artifacts/postgres_purge.go](../../internal/artifacts/postgres_purge.go).
- Проверки: `projectlifecycle` с реальным PostgreSQL; полный `internal/artifacts` с PostgreSQL и `-race` проверен в параллельном orchestration/backend review.

### `durableState` — соответствует

- Audits/claims/receipts/holds, Projects/deletion phases и owner queue controls находятся в PostgreSQL. Важные изменения lifecycle и revision проходят одним statement/transaction; periodic reconciliation восстанавливается по durable rows, а не по сохранности wake notification.
- Проверены ссылки view на typed singleton Scheduler settings, immutable RuntimeConfig versions/CAS bindings, зашифрованные credentials, Planner session state и execution report repositories. Artifact registry остаётся отдельным адаптером; контекст процесса не заменяет persistent authority.
- Реализация: [auditstore/claims.go](../../internal/auditstore/claims.go), [settingsstore/store.go](../../internal/settingsstore/store.go), [runtimeconfig/repository.go](../../internal/runtimeconfig/repository.go), [credentials/crypto.go](../../internal/credentials/crypto.go), [planner/session/service.go](../../internal/planner/session/service.go), [telemetry/repository.go](../../internal/telemetry/repository.go).
- Restart/claim takeover и lifecycle race checks выполнены пакетами `auditcontroller`, `auditstore`, `projectlifecycle`; остальные repositories дополнительно покрыты параллельным orchestration review. Потеря БД/восстановление из backup не эмулировались.

## Рефакторинг

Внесён небольшой структурный рефакторинг: общая отправка Audit deletion intent вынесена перед обработкой фаз `cancelling`/`draining`. Это сохраняет один путь формирования SQL intent и обеспечивает recovery.

Не рекомендуется механически объединять Audit claims с Run claims, human review с worker intake или Project/Run purge: у них разные authority и lifetime. Крупный перенос SQL из `projectlifecycle` в `auditService` потребует сохранения порядка блокировок и общей транзакции; для исправленных сценариев он не нужен. Отдельное улучшение читаемости — отмечать в диаграмме logical coordination и прямые вызовы разными подписями.

## Проверка

Выполнен полный запуск с временным PostgreSQL 17 и изолированными схемами тестов:

```sh
go test -count=1 -tags=integration \
  ./internal/auditcontroller ./internal/auditservice ./internal/auditstore \
  ./internal/auditimport ./internal/auditstandards ./internal/auditdomain \
  ./internal/findingintake ./internal/projectlifecycle ./internal/projectstore
```

Все девять пакетов прошли. После окончательных исправлений и дополнения Project recovery regression все девять пакетов также прошли тем же запуском с `-race`; пропуска PostgreSQL-проверок не было. `git diff --check` прошёл.

Ограничения: реальные external workers/LLM, аварийное выключение ОС и восстановление PostgreSQL из backup не запускались; вместо них использованы controller restart, настоящий PostgreSQL contention и fault injection в точках чтения/публикации. Схема API и миграции не менялись.
