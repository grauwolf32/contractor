# mitmproxy для HTTP транспорта Runtime и sandbox

Дата: 3 ноября 2026 года. Статус: proposal; выбор production-реализации открыт.

Цель — получить развитую обработку HTTP/TLS и полезные proxy-функции без
собственной реализации протокольного движка, сохранив существующие engagement
policy, credentials, limits и lifecycle Contractor. Дополнительный сценарий —
прозрачно направлять HTTP/HTTPS из Podman sandbox через тот же контроль.

Пользователь запросил оценку встраивания Python-пакета, предложил переиспользовать
существующий sandbox/allocation lifecycle и сохранить обсуждение как proposal.
Последующее уточнение: для sandbox интересует перенаправление трафика портов
80 и 443 на bridge на сетевом уровне, без настройки proxy в каждой программе.
Варианты ниже являются предложениями агента; миграция production и удаление
Caido пока не выбраны.

## Ветка и исполняемый тест

Ветка с экспериментом: **`codex/sitemap-source-comparison`**.
На момент записи ветка локальная и не слита в `main`.

- Коммит теста: `6f9679437fcf8430207d7f8820650959bcfff199`.
- Коммит закрытия оценки: `dc13b44a5a5e4eef39e62cc588b3840365132814`.
- [Исполняемый эксперимент](../../scripts/experiments/mitmproxy_embedding.py).
- [Подробная оценка](../../tasks/evidence/v410-027-mitmproxy-embedding.md)
  и [JSON результатов](../../tasks/evidence/v410-027-mitmproxy-embedding.json).
- Завершённая задача оценки: [V410-027](../../tasks/done/v410-027-mitmproxy-embedding-assessment.yml).

Из корня checkout этой ветки:

```sh
uv run --script scripts/experiments/mitmproxy_embedding.py \
  --output /tmp/contractor-mitmproxy-embedding.json
```

Эксперимент закрепляет mitmproxy **12.3.1**, использует Python **3.13** и
локальные HTTP/TLS fixtures. Сохранённые результаты получены на Python 3.13.16,
macOS arm64. Production dependency files и transport не менялись.

## Текущий bridge и ожидаемая польза

Существующий [HTTPProxyBridge](../../runtime/src/contractor_runtime/adapters/http_proxy_bridge.py)
предоставляет scanner локальный explicit HTTP proxy с приватным токеном.
Обычные запросы и HTTPS после `CONNECT` передаются в allocation-owned
[HTTPTransport](../../runtime/src/contractor_runtime/adapters/http_transport.py).
Bridge завершает входящий TLS, а transport независимо выполняет исходящий
запрос. Это IPC-вход в наш HTTP owner; управление scope и credentials находится
за ним.

Собственный входной bridge поддерживает обычный HTTP/1, один запрос на
соединение и ограниченный буферизованный обмен. Исходящий transport также
использует HTTP/1.1 и новое соединение на attempt. mitmproxy интересен готовой
обработкой TLS/mTLS, HTTP/2, переиспользованием соединений, streaming,
WebSocket и hooks обработки сообщений.

Пакет в основном Python, но содержит зависимость от Rust extension. Его можно
импортировать в Runtime. Готового клиентского класса, эквивалентного нашему
`await transport.request(...)` со всеми engagement guarantees, он не даёт:
это событийный proxy engine. `DumpMaster` — удобная сборка Master с набором
стандартных addons для консольного запуска. Для встраивания достаточно
обычного `Master` и явно выбранных addons; UI и flow archive не обязательны.

## Что подтвердил эксперимент

Минимальная композиция — `Master`, `Proxyserver`, `NextLayer`, `TlsConfig`
и собственные test hooks. Три последовательных прогона дали одинаковые
результаты: HTTP/1, проверяемый HTTPS MITM, upstream mTLS, реальный HTTP/2
на обоих участках, reuse origin connection, SSE streaming и WebSocket echo.
Запрещённый fixture path дал локальный `403` без upstream connection.

Значимые ограничения результата:

- В fixture выбран `connection_strategy="lazy"` и `upstream_cert=False`.
  Для HTTP/2 потребовался явный выбор ALPN в hook. Одной настройки `http2=True`
  оказалось недостаточно для используемого клиента.
- Исключение из policy addon было зарегистрировано движком, после чего запрос
  прошёл к target. Authoritative hook должен явно завершать ошибку отказом.
- Событие response может относиться к синтетическому ответу прокси. Такой
  ответ нельзя выдавать за полученный от target.
- Разрешение зависимостей в disposable Runtime lock добавило 36 universal
  package records, не изменив существующие версии. Совместный import Runtime,
  ADK и движка прошёл; это не проверка полной рабочей нагрузки.
- Реальные scanners, rootless Podman redirect, Caido chain, Linux packaging
  и production DNS/peer/lease/retry conformance этим тестом не подтверждены.

## Два варианта встраивания

| Вариант | Что заменяет | Выигрыш и ограничение | Предварительная оценка |
| --- | --- | --- | --- |
| Только входной bridge | Собственные HTTP parser, TLS listener и connection handling | Готовый frontend; исходящий HTTPTransport сохраняет HTTP/1.1 и buffering | 2–3 рабочих дня, включая lifecycle, отказ при ошибках и проверку реального Nuclei |
| Полный protocol engine под существующим HTTP owner | Входной bridge и часть исходящего HTTP/TLS dispatch | Возможны HTTP/2, reuse и streaming до target; нужно сохранить сетевые гарантии Contractor | 3–5 дней на базовую интеграцию; ещё 2–4 дня на HTTP/2 и streaming end to end |

Оценки относятся к одному разработчику, не измерены разработкой и не включают
прозрачную сеть sandbox. WebSocket как пользовательская capability требует
отдельного конкретного вызывающего Workflow и контракта сообщений.

В первом варианте собственный forwarding hook вызывает нынешний HTTPTransport
и формирует response для engine. Самостоятельный outbound движка должен быть
закрыт; ошибки hook не должны приводить к прямой отправке. Во втором варианте
engine обслуживает исходящие соединения, а Contractor сохраняет authority:
canonical URL, scope, все DNS answers, IP pinning, peer verification, protected
endpoints, exact-origin credentials/cookies, limits, lease и retry boundary.
Текущие требования: [HTP-R137–149](../spec/tools/http-caido.md).

Рекомендация для развитых proxy-функций — проверить второй вариант небольшим
production conformance срезом. Более простой жизнеспособный вариант — оставить
transport и заменить frontend, если его parser/TLS действительно мешает
конкретному клиенту. Если сохранение DNS pinning или проверки перед первой
отправкой потребует fork движка либо повторной реализации его network server,
оценку и сам выбор следует пересмотреть до расширения работы.

## Переиспользование lifecycle

Proxy engine должен стать ресурсом существующего HTTP/allocation owner.
`AllocationService._close_http_or_exit` уже задаёт cleanup deadline и fences
Runtime при неподтверждённой остановке. `HTTPTransport.close` завершает HTTP
invocations и streams. Общий `core.async_tasks.join_owned_task` позволяет
дождаться cleanup при повторной отмене; эксперимент использует этот helper.

У проверенной версии `Master.shutdown()` с ожиданием `run()` оставляет listener
и активные соединения. Отключение server закрывает listener, но открытый stream
остаётся. Только явное закрытие и join connection tasks завершило stream и
обнулило active connections. Значит, небольшой engine-specific cleanup нужен;
используемые внутренние API требуют закрепления версии и проверки при upgrade.

Podman `PreparedExecution`, owner RPC и guardian отвечают за контейнеры и
surviving processes. Их перенос в in-process proxy не требуется. Существующие
deadline, join, cleanup/fencing и порядок освобождения ресурсов применимы без
нового lifecycle framework. Для sandbox bridge живёт в границах allocation
и завершается вместе с сетевым доступом контейнера, а не по случайному окончанию
одного scanner invocation.

## Полезные addons

Подключать следует функции с конкретным вызывающим сценарием:

| Функция | Возможное применение |
| --- | --- |
| ProxyAuth | Аутентификация explicit proxy клиента; не замена engagement admission |
| AntiCache и AntiComp | Управляемые условия эксперимента без cache validators или compression |
| ModifyHeaders и ModifyBody | Явные преобразования для выбранного сценария с учётом streaming |
| MapLocal и server-side replay | Локальные fixtures и воспроизводимые тестовые ответы |
| HTTP/TLS/WebSocket hooks | Нормализованные observations и выбранные protocol checks |

StickyAuth/StickyCookie не включаются автоматически: identity и cookie jars
уже принадлежат Contractor. BlockList не заменяет scope/DNS/peer policy.
Изменения запроса должны предшествовать окончательной проверке отправляемого
URL и credentials. Возможности описаны в
[документации mitmproxy](https://docs.mitmproxy.org/stable/overview/features/).

## Расход памяти

Дополнительное измерение в диалоге использовало три свежих Python-процесса
с реальным Runtime overlay и тремя циклами protocol fixtures в каждом.
Текущий RSS на macOS arm64 составил:

| Состояние процесса | RSS |
| --- | --- |
| После import Runtime | 124–140 MiB |
| После import mitmproxy | 162–183 MiB |
| После первого запуска минимального engine | 167–187 MiB |
| После трёх циклов и cleanup с GC | 173–195 MiB |

Разница после циклов относительно Runtime после import составила 49–56 MiB
для всего процесса. Приписывать её mitmproxy нельзя: такого же контрольного
прогона без него не было, а HTTP/TLS clients, fixtures и прогрев зависимостей
сами могут увеличивать RSS. Обычная allocation этим замером также не оценена.
Скрипт дополнительного измерения и исходный RSS trace не сохранены в репозитории;
JSON protocol evidence не содержит измерений памяти. Эти значения остаются
предварительными наблюдениями из диалога, а оценка добавочного расхода engine
остаётся открытой.

Для такой оценки нужны сопоставимые прогоны полного Runtime/allocation
с одинаковой работой, fixtures и числом циклов: через нынешний transport
и через mitmproxy. Сравниваются одинаковые стадии запуска, нагрузки и cleanup
в нескольких свежих процессах; отдельно учитываются current RSS и peak RSS.
До этого замер не задаёт production memory limit или цену каждой allocation.

Хранение тел и истории может доминировать: 1000 тел по 1 MiB требуют около
1 GiB только для payload. Минимальный engine поэтому предлагается использовать
без flow archive, с ограниченными bodies и streaming там, где он поддержан
всем HTTP result path.

## Caido и порядок прокси

mitmproxy и Caido совместимы. У первого есть
[upstream mode](https://docs.mitmproxy.org/stable/concepts/modes/#upstream-proxy),
у Caido — [настройка upstream proxy](https://docs.caido.io/app/guides/upstream).
Порядок определяет, какой трафик видит общий контроль:

```text
Contractor → mitmproxy → Caido → target
                        └ Replay / Automate → target

Contractor → Caido → mitmproxy → target
              └ Replay / Automate ─┘
```

В первом варианте Caido служит интерфейсом и историей трафика Contractor;
собственные Replay/Automate по этой топологии обходят стоящий перед ним engine.
Caido может изменить запрос после проверок и сам устанавливает последнее
соединение. Прежние гарантии Runtime о DNS/IP/target peer не сохраняются
автоматически. Этот вариант требует явно доверенной границы транспорта.

Второй порядок интереснее, если общий контроль должен охватывать также
Caido-generated traffic. Он требует надёжно связать запрос с разрешённым
engagement и identity; наличие upstream настройки само эту задачу не решает.
Движок mitmproxy должен быть соединён с authority Contractor. Автоматически
образовывать обратную цепочку между двумя прокси нельзя из-за петли.

При двойном HTTPS interception клиент доверяет CA первого прокси, а первый
прокси — CA второго; последний проверяет сертификат target. Добавляются TLS,
буферы и latency. Для начального опыта Caido можно использовать с отключёнными
приостановкой Intercept и автоматическими изменениями. Две записи одного
обмена в двух историях не означают два физических запроса к target.

Рекомендация агента — рассматривать Caido как опциональное рабочее место
человека и источник внешней истории. Его поиск, Replay/Automate и
[workflows](https://docs.caido.io/app/concepts/workflows_intro) не появляются
при минимальном встраивании engine. Сейчас shipped security-analysis Workflow
имеет явные Caido bindings; исключение Caido потребует отдельного изменения
реальных callers. Решение об удалении не принято.

## Прозрачное перенаправление sandbox

Уточнение пользователя: перехватывать исходящий TCP на порты **80 и 443**
на сетевой границе Podman и направлять его в bridge. Это отдельный сценарий
от explicit proxy через `HTTP_PROXY`/`HTTPS_PROXY`.

```text
Программа sandbox без proxy settings
    → TCP 80/443
    → правила redirect в доверенной сетевой границе
    → transparent listener и TLS interception
    → HTTPTransport с engagement authority
    → target
```

Текущий HTTPProxyBridge нельзя просто поставить получателем такого redirect.
На 80 обычная программа отправляет origin-form `GET /path`, а bridge ждёт
absolute URL и proxy authentication. На 443 сразу приходит TLS ClientHello,
тогда как bridge сначала разбирает HTTP `CONNECT` с токеном. Его единичный
временный сертификат также не является полноценным CA с выдачей сертификатов
для произвольных target names.

У mitmproxy есть готовый
[transparent mode](https://docs.mitmproxy.org/stable/howto/transparent/).
Ему нужны redirect и доступ к исходному адресу назначения в соответствующем
network namespace. Поэтому это содержательный довод в пользу готового engine:
можно переиспользовать HTTP authority Contractor за новым transparent frontend,
вместо разработки transparent HTTP/TLS входа самостоятельно.

Podman отвечает за неизменяемую для workload сетевую границу; proxy отвечает
за HTTP admission и forwarding. Для принудительного контроля предлагаются
следующие условия будущего среза:

- Redirect только трафика выбранного sandbox; соединения самого proxy не
  попадают обратно в него. Источник соединения связывается с allocation
  доверенной топологией, поскольку прозрачный клиент не присылает proxy token.
- Остальной egress запрещён по умолчанию либо явно разрешён отдельной
  политикой. Redirect 80/443 сам по себе не закрывает другие порты, IPv6 или
  UDP/443 с QUIC. DNS получает отдельное ограниченное правило; target admission
  всё равно выполняет собственную проверку DNS в Runtime.
- HTTPS-клиент доверяет CA прокси. Приложения с certificate pinning или своим
  trust store требуют отдельной проверки; глобальное отключение TLS verification
  не входит в предложение.
- Workload сохраняет dropped capabilities и не получает `NET_ADMIN`. Настройка
  redirect, защита правил и их cleanup принадлежат доверенному owner снаружи
  исполняемого кода. Отказ или остановка proxy не открывает прямой egress.
- Cleanup закрывает доступ, останавливает контейнер и proxy exchanges, затем
  удаляет принадлежащие allocation правила/каналы без затрагивания чужих правил.

Сейчас `podman@1` использует `--network=none`, а `podman-network@1` — фиксированный
rootless `pasta`. Proxy routes и target credentials в них не передаются;
engagement admission не допускает нынешний networked profile. Граница описана
в [EXE-R24 и EXE-R123–126](../spec/tools/code-execution.md) и
[SEC-R89](../spec/platform/security.md). Её изменение потребует отдельной задачи
и согласованного контракта; это не готовая настройка текущего профиля.

Открытый инженерный выбор — где установить redirect и сохранить original
destination при rootless Podman/pasta, не ослабляя container isolation и не
требуя неподтверждённых host privileges. Рекомендация — сначала узкий Linux
spike на настоящем поддерживаемом Podman с HTTP/HTTPS, попыткой обхода и
остановкой proxy. Если потребуется новая привилегированная network-служба,
её стоимость и ownership оцениваются отдельно. Более простой вариант для
proxy-aware программ — закрытая сеть с единственным доступным explicit proxy;
`network=none` с выделенным Unix-socket каналом также можно рассмотреть, но он
не реализует запрошенный прозрачный redirect обычных TCP-приложений.

## Связь с sitemap и последующие решения

Выбор engine независим от [сравнения transport observations и scanner reports](0052-sitemap-transport-and-report-comparison.md)
в [V410-026](../../tasks/v410-026-sitemap-source-comparison.yml). Ни proxy, ни его
response hook сами по себе не отличают реальный ресурс от soft-404 или SPA
fallback. Текущего Nuclei report importer для sitemap нет; удалять его не нужно.
Для эксперимента достаточно observation hook нынешнего transport.

Сохраняется направление пользователя от 2026-11-03: ffuf, Katana и Nuclei
максимально объединяются под существующим scan toolset. Значения лимитов
приходят из engagement; гарантировать один allocation или строить новое общее
accounting ради этого proposal не требуется. Нынешние counters allocation-local.

До выбора implementation остаются открытыми: frontend или полный engine;
первый конкретный protocol caller; необходимость Caido и порядок цепочки;
реализуемость transparent redirect на поддерживаемом rootless Podman. Ветка
выше содержит оценочный тест, а не production bridge replacement или sandbox
network integration. Эта запись не запускает их реализацию.
