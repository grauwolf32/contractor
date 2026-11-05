# Тонкий HTTPTransport поверх одного mitmproxy engine

Дата: 2026-11-03. Ветка: `codex/sitemap-source-comparison`.
Коммит прототипа: `152c7fe2127ea796e6a60491f11a4baa7f932b0f`.
Задача: [V410-029](../done/v410-029-mitmproxy-thin-transport-spike.yml).

**Получилось провести программные HTTP/HTTPS запросы и настоящий Nuclei через
один native engine, сохранив реальные policy/session primitives Contractor.**
Проверка lease непосредственно перед командой отправки тоже работает для
проверенных HTTP/1.1 и TLS сценариев. Копировать или менять network server
mitmproxy не пришлось.

Фасад программного HTTP занимает 40 строк; guard на границе I/O — 26 строк.
Это размеры двух классов прототипа, а не всей интеграции: authentication,
admission, session callbacks и lifecycle занимают отдельный код. Часть этих
обязанностей уже существует в production HTTP owner и должна быть переиспользована.

## Как повторить

Из корня checkout этой ветки:

```sh
uv run --project runtime --locked --with mitmproxy==12.3.1 \
  python scripts/experiments/mitmproxy_thin_transport.py \
  --output /tmp/contractor-mitmproxy-thin-transport.json
```

[Исполняемый прототип](../../scripts/experiments/mitmproxy_thin_transport.py)
и [JSON результатов](v410-029-mitmproxy-thin-transport.json) сохранены в репозитории.
Требуются установленный Nuclei **3.11.1**, Python 3.13 и принадлежащий host
private IPv4 interface, отличный от loopback/VPN. Проверен mitmproxy **12.3.1**
на macOS arm64. JSON закрепляет hash исходника и бинарного файла Nuclei.

Fixture слушает только собственный адрес host; resolver передаёт этот адрес
реальному HTTPRoutePolicy. В assignment разрешён точный `/32`. Production
запрет loopback не отключался и не подменялся тестовым permissive policy.
Других targets, Server, модели и зарегистрированной allocation в эксперименте нет.
Dependency overlay сохраняет production `pyproject.toml` и `uv.lock`.

## Общий исходящий путь

```text
Python request → ThinHTTPTransport → private explicit ingress ┐
Nuclei       → private config     → тот же ingress           ├→ mitmproxy → fixture target
                                                             ┘   ↑
                                      HTTPRoutePolicy + IdentitySession + I/O guard
```

HTTPX у фасада соединяется с private loopback listener и получает ответ от него.
Target TCP/TLS и HTTP обслуживает native mitmproxy networking. Старые
`HTTPTransport._request`, `ControlledBackend` и `HTTPProxyBridge` не являются
исходящим путём прототипа. Это локальный programmatic ingress, а не второй
target transport. Фасад отключает автоматические redirects и keepalive
клиента; результат ограничен 64 KiB.

Для Nuclei создан единственный локальный test template с matcher на fixture
ответ. Proxy capability передаётся через файл с mode `0600`, без токена в argv.
Отключены updates, interactsh, redirects, retries и HTTPX preflight. Каждый
завершённый scanner invocation дал один match и ровно один target request.
Следующий scanner invocation намеренно зависал для проверки stop, а не для
интерпретации отсутствия matches как успешного сканирования.

## Что проверено

Три последовательных прогона получили одинаковые результаты. Каждый target
принял 10 HTTP requests; синтетические запреты не включены в это число.

| Проверка | Результат |
| --- | --- |
| Python GET через тонкий фасад | `200`, полный fixture response |
| Python POST через HTTPS | `200`, trusted local MITM CA и включённая origin verification; Host/SNI сохранены после numeric pin |
| Bearer identity | Назначенный bearer пришёл в target; anonymous request его не получил |
| Cookie session | Set-Cookie обработан настоящим IdentitySession и применён к следующему запросу; на другой origin не перенесён |
| Запрещённый path и DELETE | Локальные `403`, target не получил HTTP |
| DNS set с разрешённым адресом и loopback | Настоящий HTTPRoutePolicy отверг весь набор; target не получил HTTP |
| Исключение в admission | Явный локальный отказ вместо fail-open callback behavior |
| Принудительно неуспешная peer verification | I/O guard не разрешил application write; target не получил HTTP |
| Настоящий Nuclei | Exit `0`, один match, один запрос через тот же engine |
| Lease истёк после TCP setup | Guard остановил отправку; target не получил HTTP |
| Lease истёк после origin TLS setup | Guard остановил отправку; target не получил HTTP |
| Stop с зависшими Python и Nuclei requests | Оба target connections увидели EOF; scanner убит и joined; engine connections и facade tasks равны нулю |

Для peer negative case использована явная fault injection в результат проверки.
Положительные случаи сравнивают реальный `peername` с разрешённым numeric IP
и port. Это не атака на настоящий DNS или сетевую маршрутизацию.

Lease cases используют injected clock и barriers в `server_connected` и
`tls_established_server`. После завершения TCP/TLS clock переводится за deadline,
затем engine продолжает работу. Проверяется счётчик действительно принятых
target HTTP requests, а не длительность ожидания. Поэтому тест не зависит
от предположения о скорости connect или TLS.

Stop отменяет и joins facade tasks и scanner invocation. Scanner cleanup
выполняет kill/wait процесса через общий `join_owned_task`. Затем закрываются
listener и native connection handlers, очищаются session credentials и proxy
capability, восстанавливаются ссылки `mitmproxy.ctx`. Два scanner-процесса
на повторение — один завершённый и один отменённый — были joined.

## Как работает проверка непосредственно перед отправкой

`Proxyserver.register_connection` позволяет обернуть верхний `handler.layer`.
Обёртка делегирует все события оригинальному layer и проверяет authority перед
возвратом команд `SendData` и `OpenConnection`. Для target sends дополнительно
требуется успешно проверенный peer. При отказе вместо передачи bytes выдаётся
`CloseConnection` клиенту, после чего native handler закрывает связанные streams.

В закреплённой версии `ConnectionHandler.server_event` исполняет `SendData`
через `writer.write` без промежуточного await. Значит, проверка находится рядом
с реальной отправкой, а не только в admission hook до потенциально долгого TLS.
Проверенные barriers подтверждают этот случай фактическим target counter.

Для DNS pinning `server_connect` вызывает текущий RequestPolicy, задаёт numeric
`server.address` и сохраняет исходный hostname в `server.sni`. TLS остаётся
реализацией TlsConfig. При ошибке до connect устанавливается `server.error`;
исключение addon не используется как единственный механизм запрета.

Связь с `handler.layer`, engine commands и порядком исполнения — внутренний API.
Fork не понадобился, но exact version и conformance на этой границе обязательны
при upgrade. Выполнение простого `request` hook не заменяет guard: при request
streaming этот hook вызывается уже после передачи тела.

## Что можно убрать из собственного protocol path

| Сейчас | При интеграции native engine |
| --- | --- |
| H11 parsing, CONNECT handling и temporary TLS listener в HTTPProxyBridge | Explicit/transparent ingress и TLS config движка |
| Per-attempt httpcore pool и ручная HTTP сериализация в HTTPTransport | Native connection management и HTTP layers движка |
| Собственная проверка response header framing в ControlledStream | Engine parser с отдельно подтверждёнными совместимыми bounds |
| Socket/TLS wrappers для каждого исходящего HTTP request | Небольшой integration guard плюс engine lifecycle |

HTTPRoutePolicy, IdentitySession, limits, lease ownership, result/evidence assembly
и allocation cleanup сохраняют свои обязанности. Shared PolicyNetworkBackend
используется также обычным HTTP tool path; удалять его вместе с controlled
backend до перевода остальных callers нельзя.

## Оставшиеся границы production интеграции

- **HTTP/2, keepalive и несколько активных flows на connection.** Прототип
  хранит request admission на client connection и намеренно работает с HTTP/1.1
  без reuse. Для multiplexing нужны состояние конкретного exchange и проверка
  reuse с текущей identity/authority. Сам HTTP/2 engine ранее проверен в V410-027,
  но это не conformance нового policy integration.
- **Точная семантика reads, connect queues и deadline.** Guard проверяет
  входящие `DataReceived` после socket read и команды до их исполнения.
  Остановка idle read и connect, ожидающего внутренних engine permits, должна
  быть связана с существующим allocation cancellation/lease owner. Проверенные
  TCP/TLS barriers подтверждают запрет application write, но не весь HTP-R138.
- **Redirects и retry after possible send.** Полного orchestration для пяти
  redirects и безопасного pre-send retry здесь нет. Нельзя включать native
  replay или повторять запрос по одному transport exception без доказательства,
  что application bytes не могли быть отправлены.
- **Bounds и streaming результата.** Здесь body limit — 64 KiB; фасад получает
  ограниченный buffer. Production truncation semantics, header limits,
  streaming evidence и устранение второй полной копии тела требуют следующего
  конкретного среза. Этот прототип не является новым замером памяти.
- **Нормализация ошибок и provenance.** Синтетические отказы и engine errors
  нужно переводить в действующие outcomes Contractor. Код прототипа различает
  собственные отказы, но не заменяет полный evidence/sitemap adapter.
- **Lifecycle startup failure и конкурентные close.** Проверены рабочий start
  и stop активного client/scanner. Partial preparation, repeated cancellation,
  неизвестный outcome запуска scanner и concurrent close должны использовать
  обычный allocation owner и его существующие проверки.

Экспериментальный Authority содержит упрощённые counters и callbacks, чтобы
проверить границу движка. Он не предлагается как второй production policy
framework. При реализации сохраняется один существующий HTTP owner; общие
admission/session операции вызываются из него и выбранных engine hooks.

## Вывод и проверки

Основное предположение proposal подтвердилось: можно оставить тонкий
programmatic facade и передать protocol/networking mitmproxy, сохранив policy
Contractor через небольшой integration layer. Главный риск оказался точнее:
потребуется поддерживать версионную связь с командной границей, но копия
сетевого сервера для проверенного среза не нужна.

Дальнейшая production-задача должна переводить конкретный HTTP tool и scanner
с существующими contract tests, после чего удалять заменённый исходящий код.
Для такого решения есть работающий прототип; полного production replacement
и новых engagement guarantees эта оценка не заявляет.

Выполнены три прогона прототипа, Ruff check/format и полные существующие модули
`test_engagement_target.py`, `test_engagement_allocation.py`,
`test_http_proxy_bridge.py`, `test_nuclei_engagement.py` — 66 tests passed.
Общий `make lint` по-прежнему останавливается на отсутствии `go` в PATH;
production Python dependencies не изменены.
