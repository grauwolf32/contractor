# Ревью архитектуры: UI, публичные API и Operations

Дата: 2026-09-15. Исходная схема: `docs/spec/artitecture.likec4`.
Проверены девять представлений: `productSurface`, `productEvents`, `catalog`,
`gitArtifacts`, `gitSettings`, `processBootstrap`, `serverPerformance`,
`allocationPerformance`, `portableEvaluations`. Проверка включает чтение
соответствующих views и связей модели, трассировку реализации и существующие
негативные/интеграционные тесты. Выводы ниже относятся к указанным сценариям,
а не означают отсутствие любых возможных дефектов.

## Приоритеты и доработки

### P2 — отписка во время подключения сбрасывала соседние потоки событий

**Сценарий:** один WebSocket обслуживает несколько подписок. UI отправил
`subscribe`, затем компонент размонтировался до получения `subscribed`.
`RunEventsManager.#unsubscribe` удалял локальную запись, не отправляя
`unsubscribe`. Поздний ответ сервера считался нарушением протокола и вызывал
`#resyncAll`, закрывая соединение и перезагружая состояние оставшихся
Run/Operations подписок. На стороне сервера подписка создаётся асинхронно,
поэтому эта последовательность допустима.

**Исправление:** [run-events.ts](../../ui/src/events/run-events.ts) сохраняет
отменённую подписку в фазе `closing`, отправляет `unsubscribe` также для уже
отправленного, но ещё не подтверждённого `subscribe`, и ждёт ответа.
Поздние `subscribed`, события, `resync_required` и ошибки отменённой подписки
не вызывают callbacks размонтированного компонента. Два `not_found` от
одновременно завершившихся subscribe/unsubscribe обрабатываются без сброса
остальных потоков. Проверки формы, идентичности потока и cursor для ACK
сохранены.

**Регрессия:** [run-events.test.ts](../../ui/src/events/run-events.test.ts),
`cancels an in-flight subscription without resyncing another stream`:
пять вариантов позднего ответа; события оставшейся подписки продолжают
доставляться, соединение остаётся открытым. Два исходных варианта теста
падали до изменения реализации.

Иных подтверждённых дефектов в проверенных сценариях этих девяти views
не обнаружено. Спекулятивные изменения API и схем хранения не вносились.

## Покрытие по представлениям

### productSurface — соответствует проверенным границам

Трасса: `ui/server/static-server.mjs` → `ui/src/api/client.ts` →
`internal/httpapi/public/router.go`, `auth_handlers.go` →
`internal/auth/session.go`, `origin.go`.

- Node отдаёт SPA и закрытую публичную runtime-конфигурацию; API-пути и
  неизвестные маршруты не превращаются в SPA fallback. Проверены запреты
  мутаций, обхода путей и отдельные политики кеширования.
- Cookie-мутации требуют ровно один разрешённый Origin и CSRF до вызова
  доменной операции. Bearer-клиент использует отдельную ветку аутентификации.
  CORS проверяет известный маршрут, метод и заголовки, а не отражает любые
  клиентские значения.
- Проверены лимит login body, ограничение параллельных попыток входа,
  idle/absolute expiry, отзыв сессии, перезапуск и редактирование секретов
  в диагностике. Браузерный клиент ограничивает API origin и `/v1/`,
  проверяет версию ответа и не допускает ручные Cookie/Authorization.
- Реальная PostgreSQL-проверка покрыла создание Run и замороженный output.

**Рефакторинг:** границы Node/Go и auth/domain полезны и не требуют слияния.
Не вводился общий middleware для разнотипных precondition-контрактов:
это рисковало бы изменить семантику существующих API.

### productEvents — исправлен P2

Трасса: `internal/httpapi/public/event_handlers.go`, `events/hub.go`,
`events/connection.go`, `events/subscriptions.go`, `events/protocol.go`,
`events/queue.go` → `ui/src/events/socket.ts`, `run-events.ts`.

- WebSocket требует browser cookie, точный Origin и согласованный subprotocol;
  bearer не превращается в browser session. Owner Run и operations capability
  проверяются сервером до отправки данных.
- Проверены replay от точного cursor, смена generation, пропуск sequence,
  неизвестные/лишние поля и усечённая Planner projection. UI применяет
  последовательность через BigInt и запрашивает авторитетный snapshot при gap.
- Ограничения: 16 KiB client frame, 64 KiB server frame, 32 подписки,
  восемь сокетов на сессию, очередь до 256 frames / 1 MiB. Переполнение
  закрывает медленного потребителя. Отзыв и expiry закрывают сессию.
- Дополнительно разобрана серверная последовательность отмены: cancel pump,
  ожидание его завершения, затем `unsubscribed`. На ней основан новый тест.

**Рефакторинг:** исправлено общее управление жизненным циклом подписок;
отдельный второй механизм отмены не добавлен. Run и Operations сохраняют
различные типизированные projections.

### catalog — соответствует проверенным сценариям

Трасса: `internal/httpapi/public/agent_instructions.go`,
`catalog_discovery.go`, `internal/config/agent_instructions.go`,
`ui/src/routes/catalog/`, `ui/src/api/agents.ts`.

- Точный selector разбирается до чтения каталога. Template и base
  instructions берутся из одного загруженного snapshot; endpoint не читает
  произвольные пути и не раскрывает prompt конкретного invocation.
- Проверены аутентификация, отсутствующая версия, traversal selector,
  неизменность загруженных инструкций после изменения файла, поисковые
  ограничения и cursor, привязанный к фильтру и fingerprint каталога.
- UI показывает точный текст, Source/Copy и legacy redirects; Skills
  сохраняют обычные owner artifact permissions.

**Рефакторинг:** snapshot accessor уже устраняет риск раздельного чтения
template/instructions. Дополнительное кеширование файла в endpoint не нужно.

### gitArtifacts — соответствует проверенным сценариям

Трасса: `internal/httpapi/public/git_import_handlers.go` →
`internal/gitimport/{importer,client,pack,archive}.go` → ArtifactStore.

- Импорт требует create-only/CAS, owner scope и разрешённый namespace.
  Project lifecycle и revision проверяются перед сетью и повторно при
  публикации; origin и версия записываются в одной транзакции.
- Сеть выполняется без удержания DB-транзакции. Захватываются один import
  slot и transfer admission; общий deadline — 120 секунд. HTTPS redirects
  отключены, подключение идёт к проверенному allowlisted адресу; SSH
  проверяет known_hosts и захватывает одну генерацию ключа.
- Ограничены advertisement, pack и распакованные объекты; итоговый ZIP
  ограничен 64 MiB. Исполнения hooks или checkout рабочего дерева нет.
- Выполнены реальные локальные HTTPS/SSH fixtures с native git: branch/tag,
  фиксированный commit при изменении ветки, ключ владельца, отказ при
  неподтверждённом SSH host и cumulative decoded-pack budget. PostgreSQL
  API-тест проверил публикацию, ограничения admission и конфликт revision.

**Рефакторинг:** разделение transport/snapshot/registry оправдано; перенос
удалённого fetch внутрь транзакции нарушил бы проверяемую границу.

### gitSettings — соответствует проверенным сценариям

Трасса: `ui/src/routes/operations/settings/index.tsx`,
`ui/src/routes/settings/git-key.tsx` →
`internal/httpapi/public/git_key_handlers.go` →
`internal/credentials/git_keys.go`.

- Пользователь без operations capability видит Repository access,
  а запросы server-wide scheduler settings не выполняются.
- GET возвращает только metadata; PUT принимает ограниченный
  незашифрованный private key; шифротекст аутентифицирует owner, purpose
  и generation. Замена не меняет ранее захваченный signer импорта.
- UI отменяет прежний metadata-запрос до мутации, не складывает private key
  в mutation/query cache, очищает поле при успехе и abort при размонтировании.
  Неуспешная запись не повторяется автоматически.
- PostgreSQL-проверка покрыла owner isolation, replace/delete, подмену
  envelope и проверку ключей при старте. Существующие snapshots не удаляются.

**Рефакторинг:** owner API должен оставаться отдельным от operations-only
guard, несмотря на общее расположение в Settings UI.

### processBootstrap — соответствует проверенным сценариям

Трасса: `internal/app/config.go`, `server_config_file.go`,
`operational_settings.go`, `performance.go`, `profiling.go` →
`internal/profiling/{server,handler}.go`.

- Прочитаны defaults → YAML → environment → CLI; explicit false отличается
  от отсутствующего значения. YAML — один strict document, unique keys,
  known fields, обычный файл до 64 KiB; относительные пути разрешаются
  от каталога конфигурации. Полей с DB URL и secret bytes в YAML нет.
- pprof включается независимо от metrics, слушает numeric loopback,
  связывается до readiness и имеет конечный shutdown. Не используется
  DefaultServeMux; отсутствует browser profiling API.
- Handler ограничивает CPU до 60 секунд, trace до 10 секунд, delta capture
  до 60 секунд, symbol body до 64 KiB и параллельные профили. Проверены
  неизвестные/повторные query-параметры и недопустимые сочетания debug/gc.

**Рефакторинг:** process settings и durable Scheduler settings имеют разные
владение и жизненный цикл; объединять их в изменяемый общий объект не нужно.

### serverPerformance — соответствует проверенным сценариям

Трасса: `internal/app/performance.go`, `internal/performance/` →
`internal/httpapi/public/performance_handlers.go` →
`ui/src/routes/operations/performance/`.

- Выключенные metrics не создают recorder/collector/diagnostic pool.
  Current snapshot читает память; durable history использует обычный
  request pool. Фоновый diagnostic pool отдельный, с одной connection.
- Проверены bounded query/lock timeouts, повторное использование pool после
  ошибки, частично недоступная статистика, независимый сбой database size,
  generation/minute idempotency, TTL и предел 1000 history points / 7 дней.
- HTTP имеет фиксированные dimensions; monitoring/health/pprof исключены
  из наблюдаемого трафика. UI отключает фоновый interval polling и разрывает
  линии графика на gaps, неизвестных значениях и смене generation.
- Эти database-проверки выполнены на отдельном PostgreSQL 17 в временных
  схемах, а не только на mocks.

**Рефакторинг:** оставлены самостоятельные sampler, read service и writer;
это сохраняет изоляцию мониторинга от рабочего пула и горячего HTTP пути.

### allocationPerformance — соответствует проверенным сценариям

Трасса: `internal/telemetry/policy.go`, `allocation_resources.go`,
`internal/contracts/performance.go`,
`internal/httpapi/public/performance_handlers.go`, `ui/src/api/performance.ts`.

- История строится по durable terminal Stage/Allocation identity, с owner
  filter и LEFT JOIN отчёта. Отсутствующий/просроченный отчёт оставляет
  allocation видимым. Используется pinned policy, а не текущий metrics switch.
- Проверены requested/disabled/unsupported/legacy, pending до release,
  report_missing после release, partial/unavailable и invalid resource block,
  который не должен отбрасывать корректный execution report.
- API ограничивает страницу 100 записями, привязывает cursor к Run filter
  и верхней границе; terminal history не зависит от live Runtime registry.
  UI не выдаёт отсутствующую CPU/RSS метрику за ноль.
- PostgreSQL-тест `TestAllocationResourceHistoryUsesTerminalIdentityAndPinnedPolicy`
  выполнен. Проверка фактического Runtime sampler принадлежит соседнему
  обзору runtime/telemetry; здесь проверен downstream приём и чтение.

**Рефакторинг:** объединять эту историю с live allocation registry нельзя:
это потеряет missing/late reports и уже освобождённые allocations.

### portableEvaluations — подтверждено офлайн, live campaign не проверялась

Спецификация: `docs/spec/26-portable-evaluation-format.md`. Дополнительно
доступен sibling `/home/ruslan/src/playground-v2`, commit `bd2a557`.
Прочитаны `evals/src/playground_evals/experiments.py`, `journal.py`,
`comparison.py`, `publishers.py`, adapter/contracts и профильные тесты.
Внешние файлы не изменялись.

- Freeze фиксирует полную матрицу и pins; перед submit выполняется read-only
  recheck. Journal сохраняет intent до сетевой операции, receipt после неё,
  использует exclusive lock, immutable files, atomic index и fsync.
- Recovery различает prepared/uncertain/unknown/accepted; deadline не
  начинается заново после рестарта. Labels не становятся доказательством
  членства. При отмене каждого активного member сохраняется отдельный intent.
- Comparison сверяет receipts и полный denominator, сохраняет failed members,
  неизвестную стоимость и mismatched scopes. Publisher использует allowlisted
  projection, исключает opaque execution locators, проверяет точные bytes
  и выполняет create-only/CAS через обычный scoped Artifact API.
- Выполнены 225 офлайн-тестов formats, adapters, recovery, results,
  assessments, comparison и publication, включая потерянный ответ, stale CAS,
  скрытый ground truth, удалённые evidence и конфликт receipts.

**Рефакторинг:** внешняя граница evaluator/Contractor соответствует модели;
добавление evaluator или второго Scheduler внутрь Server не требуется.
Этот результат не доказывает успешную live target campaign, качество
LLM-инструкций или готовность отдельного V38 Evals UX.

## Выполненные проверки

- `go test ./internal/auth ./internal/profiling ./internal/performance ./internal/gitimport ./internal/httpapi/public/...` — успешно.
- С `CONTRACTOR_TEST_DATABASE_URL`, указывающим на отдельный PostgreSQL 17:
  `go test ./internal/performance ./internal/credentials ./internal/telemetry ./internal/httpapi/public ./internal/app ./internal/config` — успешно.
- `go test -tags=integration ./internal/gitimport -run 'TestRealHTTPSPinnedCommitAndTags|TestDecodedPackCumulativeBudget|TestRealSSHOwnerKeyAndStrictHostTrust' -count=1` — успешно.
- На той же тестовой БД: `go test -tags=integration ./internal/httpapi/public -run 'TestPostgresPublicRunInitializationAndFrozenOutput|TestPostgresRunDetailFixedBatchQueries' -count=1` — успешно.
- UI: 70 tests в 10 файлах client/agents/performance/events/catalog/Git/settings/application — успешно; после дополнительного варианта `not_found` оба event suites повторены: 21 test успешно.
- `pnpm test:server` — 7 tests; `pnpm typecheck`, ESLint и Prettier для изменённых event-файлов — успешно.
- В sibling `evals`: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_formats.py tests/test_adapters.py tests/test_experiment_recovery.py tests/test_results.py tests/test_assessments.py tests/test_comparison.py tests/test_publication.py` — 225 passed.

Окружение UI: Node 26.7.0 / pnpm 11.24.0; проект указывает Node 24.20.x.
Проверки прошли, но прогон на точной Node release, полный browser E2E,
удалённые production Git hosts и live evaluator campaign в этот обзор
не входят. Продуктовые процессы и данные не изменялись.
