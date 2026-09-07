# Ревью жёстко заданных эксплуатационных параметров

Проверена рабочая копия Contractor 2026-09-07, HEAD
`5157896bb3160faa92c51f8f0ce0fb065c8f2ee6`, включая имеющиеся незакоммиченные
изменения. Основной охват: композиция Go Server, Scheduler/Audit Controller,
PostgreSQL, Planner/A2A, Python Runtime/workspace и CLI. Это проверка путей
настройки и их применения, а не полный функциональный или нагрузочный аудит.

Да, такие параметры есть. Выделены семь основных мест: одно подтверждённое
игнорирование пользовательской настройки и шесть пробелов в эксплуатационной
конфигурации. Наличие `Options` у Go-конструктора само по себе не означает,
что оператор может поменять параметр: важна передача из ServerConfig/CLI/env.

## 1. P2 — `context check` игнорирует `--timeout` и `CONTRACTOR_TIMEOUT`

- `internal/cli/root.go:153`: глобальный флаг `--timeout` разбирается.
- `internal/cli/root.go:76`: ветка `context` не получает `globalOptions`.
- `internal/cli/context_commands.go:137`: вместо разобранного значения
  передаётся `Timeout: 30 * time.Second`.

Два способа проверить один Server имеют разное поведение. Это ошибка применения
уже существующей настройки, а не предложение добавить ещё один параметр.

Изолированное воспроизведение: локальный HTTP Server задерживает ответ на 120 ms,
CLI использует временный контекст и тестовый токен:

```text
contractor --timeout 20ms check
  elapsed=21ms; Client.Timeout exceeded while awaiting headers
contractor --timeout 20ms context check probe
  elapsed=122ms; error=<nil>
```

Исправление: передать разрешённый timeout в `runContext`/общий построитель клиента.

## 2. P2 — завершение и прерывание allocation всегда получают 10 секунд

- `internal/scheduler/scheduler.go:36`: `defaultFinalizationTimeout = 10s`.
- `internal/scheduler/scheduler.go:37`: `defaultAbortTimeout = 10s`.
- `internal/app/composition_execution.go:129`: задаются `OperationTimeout`
  и `PlannerTimeout`, но не `FinalizationTimeout`/`AbortTimeout`.
- `internal/scheduler/scheduler.go:967` и `:1343`: из этих значений строятся
  абсолютные сохраняемые deadlines.
- `internal/scheduler/scheduler.go:1522`: FinalizeAll ограничен этим deadline.

Увеличение `runtimeRequestTimeout` или `workerRequestTimeout` не увеличивает
время остановки Worker, инструментов и адаптеров. В Runtime эти операции делят
один deadline (`runtime/src/contractor_runtime/allocation.py:1050`). Для медленного
cleanup это означает неполные terminal reports или неподтверждённую остановку;
Runtime может fence/завершить процесс. Успешный Stage candidate при неполном
отчёте сам по себе не становится failed — Scheduler сохраняет его по контракту.

Спецификация требует конечных deadlines и достоверной остановки, но не фиксирует
именно 10 секунд как доменный инвариант. Нужны отдельные настройки ServerConfig
для finalization/abort с сохранением существующих правил ownership и fencing.

## 3. P2 — весь проход Audit Controller ограничен 10 секундами

- `internal/auditcontroller/types.go:20`: poll interval 1s.
- `internal/auditcontroller/types.go:21`: claim lease 30s.
- `internal/auditcontroller/types.go:22`: operation timeout 10s.
- `internal/auditcontroller/types.go:23`: claim batch 8.
- `internal/app/composition_audit.go:75`: Options содержит только зависимости
  и Logger; все четыре параметра остаются значениями из кода.
- `internal/auditcontroller/controller.go:79`: один 10-секундный context
  охватывает reconcile целиком, включая сбор/подготовку результатов.

Это отдельные параметры Controller, а не `AuditProfile.execution.batchSize`,
`maxRounds` или `deadlineSeconds`. Увеличение срока Audit и timeout Runtime их
не меняет. Корректная тяжёлая операция, которой нужно больше 10 секунд, может
постоянно обрываться при повторной обработке. Восемь claims обрабатываются
параллельно; это не лимит восемь child Runs.

Спецификация 19 определяет fencing/claims и ограниченность обработки, но не
задаёт эти четыре числа. Нужна типизированная конфигурация Controller;
валидацию связи claim lease с operation timeout следует сохранить.

## 4. P2 — PostgreSQL budgets доступны только вызывающему Go-коду

- `internal/persistence/postgres/budgets.go:24`: acquire 2s, query 20s,
  statement 15s, lock 2s, idle transaction 30s.
- `internal/persistence/postgres/pool.go:60`: connect timeout 5s по умолчанию.
- `internal/persistence/postgres/pool.go:65`: ConnectTimeout переписывается.
- `internal/persistence/postgres/pool.go:70`: три server-side timeout из DSN
  переписываются нормализованными `PoolOptions.Budgets`.
- `internal/app/app.go:122`: production вызывает OpenPool только с Logger;
  ServerConfig не имеет полей этих budgets.

Проверка реального `PoolConfig` без соединения с БД:

```text
DSN requested: connect=42s statement=42000ms lock=6000ms idle=90000ms pool=9
Effective:     connect=5s  statement=15000ms lock=2000ms idle=30000ms pool=9
```

Числа и приоритет Options над DSN уже описаны в
[отчёте V29](2026-09-06-postgres-v29-budgets.md). Поэтому это не новая ошибка
парсинга DSN: остаётся пробел между внутренним конфигурационным API и настройкой
запущенного приложения. Эксплуатационные budgets не являются форматом сущности
или ограничением wire-контракта.

Нужно передать budgets из ServerConfig либо явно поддержать выбранные параметры
DSN. Сохранить конечные значения, отношение `lock < statement < query`, отдельную
политику maintenance и приоритет более короткого deadline вызывающей операции.

## 5. P2 — операции local direct workspace скрыто ограничены 30 секундами

- `runtime/src/contractor_runtime/projectfs/local_direct.py:30`:
  `_OPERATION_SECONDS = 30.0`.
- `:48`: внутренний deadline создаётся заново для обычных операций.
- `:198`: тот же фиксированный бюджет применяется при close.
- `runtime/src/contractor_runtime/settings.py:22`: WorkspaceLimits содержит
  file/byte limits, но не время операции.

Snapshot/read/mutations используют `_run` без переданного deadline. Поэтому
увеличение `requestTimeoutSeconds` или размера разрешённого workspace не даёт
больше времени на сканирование и операции с медленной файловой системой.
Истечение срока может переводить guard в fenced-состояние
(`projectfs/operation_guard.py:75`), а не только обрезать один ответ.

Проверка вызова публичного `snapshot()` с подменой только принимающего deadline
внутреннего guard показала бюджет ровно `30.0 seconds`. Реальная операция на
медленном диске и ожидание 30 секунд в этом ревью не моделировались.

Спецификация 10 требует ограниченности I/O и сохранения ownership при отмене,
но не фиксирует 30 секунд. Нужен параметр WorkspaceSettings для operation timeout
и явное согласование с lifecycle cleanup deadline.

## 6. P3 — A2A polling жёстко задан как 100 ms

- `internal/planner/a2a/client.go:30`: `defaultPollInterval = 100ms`.
- `internal/app/composition_execution.go:59`: production передаёт пустой
  `plannera2a.Options{}`.
- `internal/planner/a2a/client.go:248`: периодический GetTask использует
  этот интервал на всём протяжении Worker invocation.
- `internal/planner/a2a/client.go:148`: production transport одновременно
  использует `DisableKeepAlives: true`.

При быстром локальном ответе это почти десять запросов в секунду на ожидающий
Worker и повторные соединения mTLS. Долгий ответ модели не уменьшает частоту.
Зависимость нагрузки от длительности модели и числа активных Runs нельзя
настроить. В исследованных контрактах частота 100 ms не закреплена.

Нужен параметр polling/backoff для Server. Изменение переиспользования соединений
требует отдельной проверки привязки сертификата к allocation и не предлагается
как механическое отключение существующей защиты.

## 7. P3 — timeout управления ключами LiteLLM не настраивается оператором

- `internal/credentials/litellm/manager.go:27`: connect timeout 3s.
- `:28`: request timeout 15s.
- `internal/app/composition_credentials.go:100`: `litellmcredentials.Options{}`.

Это отдельный management HTTP client. Настройки inference timeout в LiteLLM,
Worker и Planner на него не влияют. При медленном `/key/generate`, обновлении
или отзыве ключа операция может не получить подтверждения за 15 секунд.
Спецификация 06 требует конечных timeout, но не закрепляет эти значения.

Нужны process settings для credential-manager timeouts. Семантика reconciliation
неоднозначных внешних мутаций при этом должна сохраняться.

## Что не считаю нарушением указанного критерия

- Memory: 32 KiB, 128 заметок, ограничения имени/тегов, внутренний prefix
  `memory.` — спецификация 08. Sentinel 129 выводится из квоты 128 и служит
  обнаружению превышения, а не скрытой дополнительной квотой.
- LLM Gateway: `max_retries=3` и grace 60s — прямо записаны в спецификации 02.
- ModelPolicy: call/token/output/temperature передаются из разрешённой политики;
  значения `DefaultLimits` не подменяют выбранную ModelPolicy в production.
- Heartbeat/lease 10s/60s — контракт 02; auth sessions 8h/24h/8 — контракт 06.
- Code analysis: 20 000 файлов, 128 MiB, build/query 120s/10s — контракт 12.
- OpenAPI provenance suffixes и Vacuum timeout 30s закреплены в
  `tasks/v2-003-openapi-toolset.yml:60`; это авторские правила выбранного Toolset,
  а не случайный литерал. Их изменение потребует решения о поведении Toolset.
- Summarizer: общий предел проекции 512 KiB закреплён в спецификации 15;
  V47-004/005 уже завершили настройку инструкций и отдельное решение об admission.
- Telemetry export configuration меняется в параллельной V39-006. Новые значения
  DefaultTelemetryExportSettings не считаются жёсткими лимитами автоматически:
  в текущей рабочей копии есть их путь через typed runtime configuration.
- Серверные адреса/порты, обычные request/planner timeout и размеры workspace,
  имеющие действующий override через ServerConfig/CLI/env, — допустимые defaults.

Найдены и менее значимые кандидаты на последующую унификацию: ReadHeaderTimeout
5s у HTTP Server, общие бюджеты startup capability probes и ряд внутренних
cleanup/backoff интервалов. Они не включены в семь основных замечаний: для их
вынесения стоит отдельно определить нужную оператору область настройки, а не
экспортировать каждый внутренний литерал как публичный параметр.

Проверка: статический анализ цепочек конфигурации, сверка со спецификациями и
задачами, локальный HTTP probe CLI, вызов PostgreSQL PoolConfig без БД и перехват
workspace deadline. Production-код и задачи этим ревью не изменены.


## Исправлено в V49-001

Все семь основных замечаний закрыты задачей
[V49-001](../../tasks/v49-001-operational-timeout-configuration.yml).
Настройки, значения по умолчанию, приоритет источников и миграция описаны в
[руководстве по таймаутам](../operations/timeout-configuration.md).

CLI использует существующий общий timeout. ServerConfig отдельно задаёт бюджеты
Scheduler, Runtime cleanup, Project lifecycle, Audit Controller, PostgreSQL,
A2A polling и управления ключами Gateway. Local direct workspace получает
process setting и оставшийся бюджет cleanup. HTTP timeout больше не меняет
бюджеты операций. Сохранённые terminal deadlines и владение ресурсами проверены
при восстановлении, отмене, истечении срока и повторном release.

Проверка реализации: девять Go-пакетов с `-race` и временной PostgreSQL 17,
229 Python-тестов с warnings-as-errors, Ruff и валидация конфигурации. Те же
проверки Go/Python и конфигурации прошли в чистой рабочей копии коммитов,
независимо от параллельных изменений. Хеши реализации записаны в задаче.
