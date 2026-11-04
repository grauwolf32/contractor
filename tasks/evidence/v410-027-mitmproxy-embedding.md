# Оценка встраивания mitmproxy в Runtime

Дата: 2026-11-03. Исходный Contractor: `03a31a26d`.
Проверено: mitmproxy **12.3.1**, Python **3.13.16**, macOS arm64.

**Вывод:** импортировать и запустить полезную часть mitmproxy внутри нашего
процесса можно. Для проверенных HTTP/TLS/WebSocket возможностей достаточно
трёх addons движка и небольшого обработчика событий. `DumpMaster`, UI,
архивирование flows и отдельный proxy-процесс не понадобились.

При внедрении следует переиспользовать существующий allocation/HTTP lifecycle
Contractor. Новый менеджер lifecycle, отдельная lease или аналог Podman guardian
для in-process proxy не нужны. Специфичным для mitmproxy останется закрытие
его listener, соединений, задач и ссылок на приватное состояние.

Это завершённая оценка возможности встраивания, не production-интеграция и не
результат отдельного сравнения sitemap sources из V410-026. Решение о замене
production transport остаётся открытым.

## Воспроизводимость

Исполняемый [spike](../../scripts/experiments/mitmproxy_embedding.py) содержит
PEP 723 dependencies и проверяет реальные локальные HTTP/TLS targets:

```sh
uv run --script scripts/experiments/mitmproxy_embedding.py \
  --output tasks/evidence/v410-027-mitmproxy-embedding.json
```

[JSON evidence](v410-027-mitmproxy-embedding.json) содержит версии, hash script,
результат dependency resolution и три последовательных прогона в одном Python
процессе. Protocol/lifecycle результаты трёх прогонов совпали. В каждом target
принял 14 HTTP requests. Никакие внешние цели не сканировались; package resolver
мог обращаться к package indexes.

Production `runtime/pyproject.toml` и `runtime/uv.lock` не изменены: harness
проверяет их hashes до и после работы. Он создаёт отдельный CA и target
certificates, затем удаляет их после закрытия ресурсов. Секреты, тела запросов
и сетевые архивы в evidence не сохраняются.

## Минимальная сборка

Фактически использована следующая композиция:

```python
engine = master.Master(options, with_termlog=False)
engine.addons.add(
    proxyserver.Proxyserver(),
    next_layer.NextLayer(),
    tlsconfig.TlsConfig(),
    probe_hooks,
)
task = asyncio.create_task(engine.run())
```

Стандартный `addons.default_addons()` содержит 31 addon. В spike подключены
только `Proxyserver`, `NextLayer`, `TlsConfig` и наши test-only hooks. В частности,
нет `Save`, `SaveHar`, replay, onboarding и script loader. Сборка с запуском и
cleanup занимает 57 строк класса; это размер экспериментального каркаса,
а не оценка готового production adapter. Остальной harness создаёт targets,
сертификаты, клиентов, выполняет probes и сохраняет evidence.

Уменьшение числа активных addons не превращает dependency в отдельную маленькую
transport library. Пакет по-прежнему требует Rust extension и UI-related
dependencies. Disposable universal Runtime lock получил 36 новых package records,
включая платформенные варианты; существующие версии остались прежними. Число 36
не является числом пакетов, которые устанавливаются на каждой платформе.

Проверен также настоящий совместный import allocation preparation, текущих
HTTPTransport/bridge и mitmproxy Master через `uv run --project runtime --locked
--with mitmproxy==12.3.1`. Сохранились Google ADK 2.13.0 и cryptography 50.0.1.
Это проверка разрешимости и import compatibility; она не заменяет полную
совместную нагрузку ADK и proxy.

## Проверенные возможности

| Проверка | Наблюдаемый результат |
| --- | --- |
| HTTP/1 GET и POST | Оба прошли; target получил правильный размер POST body; два запроса использовали одно origin connection |
| HTTPS interception | Клиент проверил сертификат, выпущенный временным CA mitmproxy; upstream verification включена |
| Недоверенный upstream certificate | Получен локальный proxy `502`, HTTP request к target не выполнен |
| HTTP/2 | Client и origin действительно использовали HTTP/2; пять запросов использовали одно origin connection |
| Streaming SSE | Первый chunk получен до разрешения fixture закончить response; streamed body не сохранён в flow |
| WebSocket | Выполнен реальный handshake и echo сообщения; hooks получили сообщения обоих направлений |
| Upstream mTLS | Без client certificate соединение отвергнуто (`502`); с certificate запрос выполнен (`200`) |
| Request admission hook | Запрещённый path вернул локальный `403`; для первого HTTPS request к этому origin не было upstream connect |
| Metadata hooks | Доступны method/path/status; flow recording addon не подключён |

Для admission использованы `connection_strategy="lazy"` и
`upstream_cert=False`. Они позволяют принять решение по HTTP request до
подключения к origin. Это не реализация всех engagement route/DNS/peer правил:
fixture policy проверяет только конкретный запрещённый path.

При lazy strategy с обычным HTTPX negotiation выбрал HTTP/1.1, хотя HTTP/2 был
разрешён. В этом режиме mitmproxy уважает порядок client ALPN offers. Test hook
в `tls_clienthello` явно выбрал `h2`, когда клиент его предлагал; после этого
прошёл HTTP/2 на обоих плечах. Политика выбора ALPN — конкретная часть будущего
adapter. Простое включение `http2=True` не гарантирует HTTP/2 для каждого клиента.

Не проверялись HTTP/3, gRPC semantics, raw/unsafe Nuclei modes, browser traffic,
реальные ffuf/Katana, throughput или экономия памяти под нагрузкой. Значение
`elapsed_seconds` в JSON — длительность harness, не benchmark преимущество.

## Lifecycle и повторное использование существующего кода

Вопрос пользователя на 2026-11-03 — переиспользовать уже имеющийся механизм
Contractor, похожий на используемый sandbox, вместо нового proxy lifecycle.
Проверка исходников подтвердила такой путь:

| Существующий механизм | Применение к mitmproxy |
| --- | --- |
| `AllocationService._close_http_or_exit` | Уже закрывает HTTP owner в общем deadline и fences allocation/останавливает Runtime при неподтверждённом cleanup |
| `HTTPTransport.close` | Уже владеет завершением HTTP invocations, закрытием streams и очисткой credentials/session buffers |
| `core.async_tasks.join_owned_task` | Готовое ожидание дочернего cleanup при повторной отмене вызывающей задачи; spike использует этот настоящий helper |
| `HTTPProxyBridge.__aexit__` | Уже показывает локальный шаблон: закрыть listener, отменить handlers, закрыть writers, дождаться задач и стереть приватные ссылки |
| `AllocationAdapterHost` | Существующий owner lifecycle для инфраструктурных adapters, если будущая граница конкретного engine потребует такой формы; второй owner одновременно с HTTPTransport не нужен |

Наиболее прямой вариант — включить engine в ресурс, который сейчас закрывается
через `context.http_transport.close()`. Общее состояние allocation, deadline,
lease/fencing и completion не переносятся в новый класс. Внутри HTTP owner
необходим engine-specific `close`: запретить новые запросы, остановить listener,
закрыть и join его соединения/задачи, затем удалить ключи и приватные ссылки.

Sandbox-specific `PreparedExecution` имеет `executor`, `stop` и `remove`
контейнера. `PodmanLifecycle` владеет surviving owner process, RPC и guardian.
Эти типы не стоит натягивать на in-process proxy. `OwnedOperations` также
привязан к sandbox error classes и сериализации CLI/RPC операций; прямой import
перенёс бы лишнюю предметную зависимость. Общий join уже находится в `core`.
Выделять ещё одну абстракцию без конкретной общей операции не требуется.

Фактические стадии остановки на открытом SSE stream:

| Действие | Listener | Активный target stream |
| --- | --- | --- |
| `Master.shutdown()` и ожидание `run()` | Всё ещё принимает соединения | Остаётся открыт |
| Дополнительно `options.update(server=False)` и ожидание server update | Закрыт | Остаётся открыт; один connection handler остаётся активен |
| Явное закрытие и join engine connection tasks | Закрыт | Закрыт; active connections = 0 |

У проверенного Master нет `aclose`/async-context-manager с полной такой
семантикой. `ctx.master` также сохраняет ссылку после `run()`; harness восстанавливает
предыдущее module context. Для production требуется удалить ссылки на private
certificate/credential state после join, не полагаясь на завершение всего процесса.

Spike обращается к `Proxyserver.servers` и `handler.transports`, чтобы подтвердить
реальное закрытие. Это связь с внутренним API закреплённой версии, а не готовый
стабильный transport-class контракт. При upgrade эту границу нужно проверять.

## Существенные границы интеграции

1. **Policy hook не является готовым fail-closed enforcement.** Намеренный
   `RuntimeError` в callback был обработан движком как ошибка addon, после чего
   target получил запрос и вернул `200`. Владелец policy должен явно обрабатывать
   ошибки отказом/остановкой, а не рассчитывать на propagation callback exception.
2. **Proxy response не всегда является target observation.** Два `responseheaders`
   события относились к нашим локальным `403`. Их нельзя публиковать в sitemap
   как фактические target responses. Нужны текущие нормализованные outcomes и
   отдельный учёт synthetic replies/errors.
3. **Scope/DNS/peer/lease/retry остаются контрактами Contractor.** Путь engine
   открывает соединения через собственный networking. Требуется сохранить нашу
   проверку всех DNS answers, IP pinning, peer verification, защищённых endpoints
   и lease перед первым application write; доказательство этих свойств в spike
   не заявляется. Так же не проверена эквивалентность автоматических retries.
4. **mTLS support не равен готовой credential integration.** В spike использован
   тестовый файл. Production private material, exact-origin cookie jars и
   выбранная identity остаются в существующем owner.
5. **Одна замена входного bridge не даёт весь выигрыш.** Нынешний исходящий
   transport принудительно использует HTTP/1.1, новое соединение на attempt и
   буферизованный response. HTTP/2 до target, WebSocket и streaming требуют
   изменить и эту часть пути.

## Какой код можно заменить

При использовании только frontend engine заменяется часть
`adapters/http_proxy_bridge.py` (338 строк): HTTP parser/serializer, локальный
TLS listener и proxy connection handling. Сохранится forwarding в текущий
HTTPTransport, а значит и ограничения его исходящей стороны. В этом варианте
нужно проверить чистоту единственного владельца TLS/network I/O: добавление
ещё одного proxy-hop само по себе не является упрощением.

При использовании полного engine можно рассматривать удаление собственного
HTTP/1/TLS/stream dispatch из `http_transport.py` (572 строки) и части
`http_stream.py` (115 строк). Это верхняя граница затронутых файлов, не обещанное
сокращение: часть их кода реализует нужные policy, ownership и retry guarantees
и должна сохраниться в общей реализации.

`http_policy.py` (146 строк), `http_session.py` (172 строки), shared limits,
engagement assignments, allocation cleanup/fencing, ArtifactStore и sitemap
collector сохраняют свои обязанности. Их нельзя удалить на основании наличия
похожих addons в mitmproxy.

## Рекомендация и объём

Для цели «получить продвинутые proxy-функции» рекомендую рассматривать
**минимальный встроенный engine с существующим HTTP/allocation lifecycle**.
Spike подтвердил, что активная обвязка может быть небольшой, а TLS/mTLS,
HTTP/2, connection reuse, streaming и WebSocket действительно работают.

Production implementation сначала должна доказать policy/DNS/lease/retry и
остановку под cancellation. Чистая конфигурация уже не главный вопрос. До
этих проверок сохраняется текущий production transport.

Предварительная инженерная оценка, не измеренная длительность разработки:

| Срез | Оценка для одного разработчика | Что принимается |
| --- | --- | --- |
| Встраивание с нынешним owner и сохранение действующей HTTP authority | 3–5 рабочих дней | Lifecycle, deny/DNS/peer/lease, credentials, bounds и retry regressions; проверка реально удаляемого кода |
| HTTP/2 и streaming end to end | Ещё 2–4 дня | Protocol selection, stream cancellation/body bounds, существующий HTTP result contract |
| WebSocket как пользовательская capability | Отдельный срез после появления вызывающего Workflow | Semantics сообщений, bounds, результаты и закрытие долгоживущего соединения |

Оценку 3–5 дней пересмотреть, если сохранение DNS pinning/first-write boundary
потребует fork двигателя или повторной реализации его network server. Это
условие остановки и отдельного решения, а не разрешение расширять подсистему.

Более простой вариант для текущего sitemap — добавить observation hook в
существующий HTTPTransport. Это не даст новых протоколов, зато не требует
замены engine. V410-026 может сравнить качество metadata независимо от выбора
proxy engine. Нельзя обосновать добавление mitmproxy одной лишь soft-404
фильтрацией: protocol engine не заменяет scanner verification.

## Проверки и ограничения gate

Выполнены три полных protocol/lifecycle прогона harness, совместный Runtime
import, Ruff check/format и существующие полные модули `test_threads.py`,
`test_engagement_allocation.py`, `test_podman_engine.py`.

Полный `make lint` не является подтверждённым pass: на этом host `go`
отсутствует в PATH, и gate останавливается на `shared-config-check`. Production
Python dependencies и исходники не изменялись. Linux engine packaging,
production conformance и полный release gate не проверены этой оценкой.
