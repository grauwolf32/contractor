# Независимое ревью кода — 2026-09-20

Найдены **11 воспроизводимых замечаний: одно P1, девять P2 и одно P3**.
Основание каждого — чтение текущей реализации и отдельная проверка.
Предыдущие отчёты не служат доказательством ни наличия, ни отсутствия ошибок.
На этапе обнаружения исходный код не исправлялся; существующие изменения документации и задач сохранены.

Начальный commit: `ea1bab91a8c3a12f972fbb2188597334e4652ce3`.
Во время работы main изменился до `5856beb5c13429f4be54856015ad7e86bd8b573d`:
добавились ScanPlan и связанные контракты. Файлы с замечаниями между этими
commit не изменились. Новый модуль дополнительно прочитан выборочно и проверен
отдельным запуском тестов. Описание и номера строк ниже относятся к коду до исправлений.

## Correction status

All CR-01–CR-11 findings are corrected by tasks V60-013–V60-023 with permanent
regression tests. See the [fix results](2026-09-20-independent-review-fix-results.md)
for implementation commits and verification. The original reproductions below
remain historical evidence. Broader readability work remains in V60-025.

## Подтверждённые дефекты

### CR-01 · P1 · Runtime подтверждает release при продолжающейся файловой операции

Место: [source_analysis/tools.py:213](../../runtime/src/contractor_runtime/toolsets/source_analysis/tools.py#L213),
обработка отмены на строке 218, запись файлов на строке 654.

`open_source_archive` запускает распаковку через `asyncio.to_thread`.
Отмена ожидающей coroutine не останавливает поток. Обработчик удаляет staging,
отпускает session lock, и дальнейшая очистка allocation считает работу завершённой.
Продолжающийся extractor вызывает `mkdir(parents=True)` и воссоздаёт каталог.

Воспроизведение использует настоящие `AllocationService`, `AdkWorkerRuntime`,
`SourceAnalysisToolsetFactory` и `LocalWorkdirFactory`. Модель и Artifact storage
заменены fixtures; чтение ZIP приостановлено в определённой точке. После
`abort → release → confirm_release` получено:

```text
slot=idle; workspaceExists=False; extractorFinished=False
forcedExit=[]; reportComplete=True
```

После продолжения настоящего extractor каталог снова существует и содержит
`.source-staging-*/src/private.py` с прежним исходным текстом. Проверен обычный
`local-workdir@1`, где supervisor subprocess не создаётся. Проверка проходит
через runtime lifecycle напрямую, без HTTP/mTLS.

Дополнительное проявление: [likec4/tools.py:295](../../runtime/src/contractor_runtime/toolsets/likec4/tools.py#L295)
тем же способом запускает синхронный validator. После отмены validation,
`tools.close()` и удаления workspace реальный тестовый CLI-child остаётся жив,
а поток ещё не завершён. Тест сам завершает и собирает процесс. Этот вариант
проверен на уровне tool/session, отдельно от полного AllocationService.

Исправление: явно учитывать незавершённые файловые операции и дочерние процессы.
Отмена должна запрещать новые операции; cleanup обязан дождаться старых либо
завершить их до подтверждения release. Для файловых операций можно использовать
подход существующего `WorkspaceOperationGuard`; для CLI нужен управляемый
subprocess с terminate/kill/wait и ограничением вывода.

### CR-02 · P2 · CloneWorkerHandle разделяет map с оригиналом и создаёт гонку

Место: [passthrough.go:505](../../internal/planner/passthrough.go#L505).

После `result := handle` поле `AgentCard` указывает на ту же map.
`json.Unmarshal(encoded, &result.AgentCard)` переиспользует ненулевую map,
поэтому само клонирование пишет в исходный объект. Helper вызывается на
границах A2A invocation и чтения Worker State.

Проверено: изменение `name` и вложенного A2A URL клона изменяет оригинал.
Два параллельных вызова `CloneWorkerHandle(original)` без внешней записи
дают `WARNING: DATA RACE` между Marshal и Unmarshal.
State reader вызывает helper после отпускания своего mutex.

Исправление: декодировать в новую map, затем присвоить результат. Временная
overlay-правка `result.AgentCard = nil` устраняет оба воспроизведения, включая
`-race`. Ошибку копирования также нельзя молча превращать в возврат общей map.

### CR-03 · P2 · Placement и Rebind ждут второе соединение, удерживая первое

Места:

- [placement.go:440](../../internal/controlplane/placement.go#L440), также строка 452;
- [binding_service.go:82](../../internal/runtimeconfig/binding_service.go#L82),
  затем `validateTargetWith → validateSpecRuntimeCredentials`;
- [runtime_service.go:169](../../internal/credentials/runtime_service.go#L169).

Обе операции открывают транзакцию, но вложенные credential readers используют
сервисы, созданные на общем `pgxpool.Pool`. В production composition это тот же
пул. Placement затрагивает managed LLM credentials и Runtime credentials;
Rebind — конфигурацию с credential для proxy/telemetry/Caido.

Две независимые проверки на настоящем PostgreSQL:

| Операция | MaxConns=2 | MaxConns=1 |
| --- | --- | --- |
| Placement с настоящим EncryptedProvider | allocation создан, около 19 мс | через 1 с deadline, `lookup encrypted credential metadata` |
| Rebind с настоящим RuntimeCredentialService | успешно, около 20 мс | через 1,2 с `context deadline exceeded` |

Прямые credential reads проходят. После выхода соединения освобождены: это
вложенное ожидание, не утечка. Такой же ресурсный конфликт возможен при
насыщении большего пула; конкурентный вариант отдельно не измерялся.

Исправление: передавать credential readers, привязанные к текущей транзакции.
В Run creation и Eval preflight такой подход уже реализован. Сохранить
credential lifecycle barrier; увеличивать пул как замену исправлению нельзя.

### CR-04 · P2 · Объявленное доступным продолжение Run после escalation не работает

Место: [resume_store.go:124](../../internal/runstore/resume_store.go#L124).

`ResumableStage` разрешает продолжение неуспешной escalated-попытки.
`ResumeFailedRun` создаёт новую попытку с прежними `ExecutionConfigVariant`
и `EscalationOrdinal`. [Unique index](../../internal/persistence/migrations/000008_scheduler_escalation.sql#L15)
запрещает эту комбинацию для того же Run/Stage.

На PostgreSQL контрольный `base` успешно продолжился. Варианты
`failed_escalation` и `interrupted_escalation` сначала объявлялись доступными
для Resume, затем оба получили `runstore optimistic state conflict`.

Исправление: различить identity автоматической escalation и ручного продолжения
с унаследованной effective configuration. Согласовать индекс и подсчёт попыток:
[stage_finalization.go:333](../../internal/scheduler/stage_finalization.go#L333)
тоже отвергает повтор ordinal. Простое удаление unique index недостаточно.

### CR-05 · P2 · HTTP tool меняет исходный query даже без новых параметров

Место: [http/tools.py:1109](../../runtime/src/contractor_runtime/toolsets/http/tools.py#L1109),
повторное кодирование на строке 1116.

`parse_qsl` и `urlencode` безусловно декодируют и заново собирают исходную query.
Настоящий callable `http_request` с `httpx.MockTransport` и без аргумента `query`
отправляет:

```text
?q=%FF      → ?q=%EF%BF%BD
?q=%20&flag → ?q=+&flag=
?q=a%2fb    → ?q=a%2Fb
```

Первый пример меняет сами байты значения. Это повреждает security-test payloads
и URL, подпись которых зависит от исходного request target.

Исправление: сохранять существующую raw query. Проверять её лимиты отдельно;
при добавлении параметров кодировать только новые пары.

### CR-06 · P2 · Запоздавший session response стирает CSRF после нового входа

Место: [client.ts:293](../../ui/src/api/client.ts#L293), особенно строка 298.

Сценарий одного пользователя: повторный `/auth/session` ещё ожидает ответа;
другой запрос получает 401, `SessionProvider` отменяет queries; пользователь
входит заново; старый session request возвращает 401 и вызывает `csrf.clear()`.
React Query отбрасывает отменённый результат, но не побочный эффект внутри API.

Проверка с настоящими `PublicAPI` и `SessionProvider` подтверждает: UI содержит
новую session, а `mutationHeaders()` падает с
`An authenticated CSRF token is required`. Это не гипотеза о нескольких owners.

Исправление: проверять поколение авторизации перед `csrf.replace/clear`
в `getSession`. Аналогичная защита для обычных 401 уже присутствует в transport.
Передача `AbortSignal` полезна дополнительно, но не заменяет защиту состояния.

### CR-07 · P2 · Coverage остаётся устаревшим после завершения Audit

Место: [coverage-data.ts:20](../../ui/src/routes/projects/audits/coverage-data.ts#L20),
отключение polling на строке 47.

Родитель получает Audit примерно раз в секунду, Coverage обновляется раз в пять
секунд. При переходе в terminal state в том же round polling отключается,
query key остаётся прежним, финального refetch нет.

Проверено: после active revision 1 передан completed revision 2 того же round;
через 5200 мс `listAuditCoverage` всё ещё вызван ровно один раз и отображаются
прежние данные. Последние результаты не появляются до нового refetch.

Исправление: гарантировать последнее обновление projections при terminal
transition либо привязать загрузку к принятой revision/state. Сохранить
проверки согласованности между страницами. У items/report заметен похожий
паттерн, но отдельные воспроизведения для них не выполнялись.

### CR-08 · P2 · HTTP target editor не показывает credentials после первой страницы

Место: [http-target-editor.tsx:133](../../ui/src/routes/projects/http-target-editor.tsx#L133).

`listRuntimeCredentials(api)` запрашивает 50 записей. Редактор использует только
`items`, игнорирует `hasMore/nextCursor`, не предлагает перехода на следующую
страницу или ввода ID. При большом каталоге нужный credential недоступен.

Component/API probe: первая страница содержит 50 записей и `nextCursor`,
нужный credential находится на второй. Выполнен один GET с `limit=50`;
нужной option и управления пагинацией нет.

Исправление: отдельный paginated picker либо сбор всех страниц для этого
ограниченного selector. Использовать отдельный query key для агрегированного
списка и сохранять видимость уже выбранного credential.

### CR-09 · P2 · Static server отвергает допустимые ссылки, созданные UI

Место: [static-server.mjs:211](../../ui/server/static-server.mjs#L211), route patterns на строках 40–41.

RuntimeConfig разрешает версию `1.0.0+local`. UI строит URL через
`encodeURIComponent`, получая `%2B`. Сервер проверяет decoded path на опасные
компоненты, но для route matching возвращает encoded pathname.

Настоящий обработчик `createStaticServer` возвращает:

```text
/runs/configuration/default/1.0.0        → 200
/runs/configuration/default/1.0.0+local  → 200
/runs/configuration/default/1.0.0%2Blocal → 404
```

Переход внутри SPA может работать, а прямое открытие и reload — нет.
Аналогичный отказ воспроизведён для `%3A` в Project ID.

Исправление: согласовать безопасное декодирование route segments с URL builder.
Сохранить отказы для encoded separators и traversal. Общая таблица допустимых
идентификаторов должна проверяться и browser routing, и static server.

### CR-10 · P2 · CLI --force расширяет права приватного скачанного файла

Место: [artifact_commands.go:595](../../internal/cli/artifact_commands.go#L595).

Обычное создание файла учитывает umask. При `--force` временный файл получает
`Chmod(0644)`, затем заменяет существующий через rename. Воспроизведение
настоящей функции при `umask 0077`: обычное скачивание создаёт `0600`,
повторное скачивание с `--force` меняет права на `0644`.

Затронуты Artifact download и Run output. Если родительский каталог доступен
другим пользователям ОС, ранее закрытые данные становятся читаемыми для них.

Исправление: при замене сохранять разрешения существующего файла; для нового
файла не расширять безопасные права `CreateTemp`. Сохранить атомарный rename.

### CR-11 · P3 · CLI теряет смысл разделителя --

Место: [root.go:263](../../internal/cli/root.go#L263), возврат аргументов на строке 288.

`interspersedFlags` удаляет `--`, собирает positional arguments отдельно,
но не отделяет их от options перед последующим `flag.Parse`.
Проверено: `source push -- -source` отклоняется с
`flag provided but not defined: -source` вместо принятия имени каталога.

Исправление: вставить `--` между собранными options и positionals; проверить
также обычные flags после resource name и literal `-` для stdin.

## Упрощение и читаемость

Эти рекомендации отделены от подтверждённых ошибок; сами размеры файлов
не считаются дефектом.

1. **Сделать владение фоновыми операциями единообразным.** Source/validators
   используют голые `to_thread`, тогда как projectfs/code-analysis уже имеют
   механизмы ожидания операций. Общий небольшой operation/subprocess layer
   устранит расхождение cancellation semantics, доказанное CR-01.
2. **Сделать транзакционность credential readers явной в типах/API.** Сейчас
   `resolveCandidate(ctx, db, …)` выглядит привязанным к транзакции, но обращается
   к полям allocator с собственным pool. Переиспользовать подход transaction
   lookup из Run creation, чтобы исключить класс CR-03.
3. **Собрать проверенные deep-copy helpers для контрактных структур.** Сейчас
   они распределены по planner/controlplane/scheduler. Начать с WorkerHandle;
   проверить независимость вложенных значений и обработку ошибок копирования.
   Не заменять deep copy на `maps.Clone` для вложенной JSON-структуры.
4. **Разделить Audit detail по существующим секциям.** Файл
   `ui/src/routes/projects/audits/detail.tsx` содержит около 2100 строк и
   смешивает checks, reviews, findings, report и polling. Вынести projections
   в hooks с общей политикой terminal refresh, а секции — в отдельные компоненты.
5. **Объединить повторяющуюся работу с paginated selectors.** Credentials,
   Project inventory и Audit collections должны иметь одинаково явную обработку
   `hasMore`. Это предотвращает пропуски вроде CR-08. UI route builders и Node
   route patterns также полезно связать общими fixtures.
6. **Разделить HTTP transport и batch orchestration.** `runtime_client.go`
   объединяет wire validation, endpoint handling, fan-out и cleanup; выделение
   `RuntimeBatchController` уменьшит число обязанностей файла. В Python
   `control_client.py` и `artifacts.py` дублируют HTTP framing; общий parser с
   настраиваемыми лимитами уменьшит расхождение проверок.
7. **Упростить RuntimeConfig merge без reflection-присваивания.** В
   `internal/runtimeconfig/merge.go` `any` и `reflect.Value.Set` скрывают
   соответствие destination/value. Типизированный generic helper оставит
   существующую политику конфликтов, но перенесёт ошибки типов на компиляцию.

## Проверки и границы

| Проверка | Результат |
| --- | --- |
| `go test -json ./...` | 2826 pass, 220 skip; ошибок пакетов нет. Счётчики включают subtests. |
| Go + настоящий PostgreSQL: auditservice, auditstore, evalservice, evalstore, runtimeconfig, credentials, persistence/postgres | 326 pass, без пропусков и ошибок |
| `runtime/.venv/bin/python -m pytest -W error tests` | 2352 passed, 34 skipped; 61,30 с |
| Полный UI Vitest | 62 файла, 435 tests passed |
| TypeScript приложения и E2E; Node static-server tests | Typecheck прошёл; серверные тесты прошли после запуска с разрешением локальных сокетов |
| ScanPlan и обновлённые contracts после изменения HEAD | оба пакета pass |
| Адресные probes | Все описанные дефекты воспроизведены; основные проверки повторены независимо |

Go regression probes с assertions ожидаемого корректного поведения закономерно
завершаются FAIL на неисправленном коде. Python/UI/CLI reproduction probes
проверяют наличие дефекта и завершаются успешно. Их PASS не означает исправление.

Первый запуск Go в sandbox не смог писать build cache и открывать loopback
listeners. Повтор с разрешённым доступом прошёл. PostgreSQL был запущен в
отдельном временном контейнере; тесты использовали изолированные схемы.
По завершении контейнер остановлен и удалён. Реальные модели и внешние
security targets не вызывались.

Прочитаны execution/recovery, placement/planner, Run storage/resume, blobs и
транзакции, части Audit/Evals, auth/credentials/configuration, Runtime lifecycle
и несколько toolsets, API/UI/CLI, CI и deployment. Это широкий проход по
подсистемам, а не заявление о построчной проверке каждого файла.
Полный real-browser stack, Podman release gates, live-model evals и сценарии
восстановления production backup не запускались. Отдельного измерения
производительности под production-нагрузкой нет.

Probes, raw logs и хеши файлов сохранены локально в
[.local/code-review-2026-09-20](../../.local/code-review-2026-09-20/), сводка —
[evidence.json](../../.local/code-review-2026-09-20/evidence.json).
Каталог `.local` не входит в Git; сам отчёт сохранён отдельно от существующих
dirty plans/tasks. Overlay-файлы не изменяют production-исходники.

Рекомендуемый порядок: CR-01; затем CR-02/03/04/10; затем CR-05–09 и CR-11.
Рефакторинг выполнять отдельно от исправлений, сохраняя проверки поведения.
