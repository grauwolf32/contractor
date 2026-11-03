# Память Runtime и встроенного mitmproxy

Дата: 2026-11-03. Ветка: `codex/sitemap-source-comparison`.
Коммит теста: `544e8bab96ab3e91c02799e7515a5eba0c5f631a`.
Задача: [V410-028](../done/v410-028-runtime-proxy-memory-assessment.yml).

Наиболее полезная оптимизация под HTTP-нагрузкой — передавать большие тела
потоком и не хранить их повторно в proxy flows. Минимальная композиция addons
сама по себе почти не снижает память импорта. Для Runtime без модели отдельное
направление — исключить загрузку ADK, но нынешний startup discovery требует его
проверки; простой перенос import в функцию не даст устойчивого выигрыша.

Production исходники и зависимости этой оценкой не меняются. Полноценная
allocation, ADK invocation, Caido и transparent Podman здесь не запускаются.

## Воспроизводимость и метод

[Скрипт](../../scripts/experiments/runtime_proxy_memory.py) запускается из корня:

```sh
uv run --script scripts/experiments/runtime_proxy_memory.py \
  --output /tmp/contractor-runtime-proxy-memory.json
```

[Сохранённый JSON](v410-028-runtime-proxy-memory.json) включает 30 последовательных
дочерних процессов: три свежих процесса для каждого из семи import-сценариев
и трёх вариантов HTTP. Версии: Python 3.13.16, mitmproxy 12.3.1, ADK 2.13.0,
psutil 7.2.2; платформа macOS arm64. Скрипт записывает свой SHA-256, current RSS
и process peak RSS через OS, а также проверяет неизменность production
`pyproject.toml` и `uv.lock`. У RSS нет машинозависимого порога pass/fail.

Каждый HTTP-процесс импортирует одинаковый Runtime CLI и использует одинаковые
loopback origin и HTTPX-клиенты. Три раунда содержат по четыре одновременных
ответа по 8 MiB: 12 target requests и 96 MiB payload на процесс. Target пишет
повторяемый chunk 64 KiB; клиент читает и считает bytes без сохранения тела.
Нет TLS, compression и внешних целей. Размер 8 MiB служит усиленным локальным
сценарием, а не новым production limit.

В proxy-вариантах событие `response` удерживает четыре завершённых flow до
измерения; direct-клиенты ждут на такой же границе после чтения ответа. Это
позволяет измерить состояние с четырьмя одновременно удерживаемыми ответами.
Измерение считает `raw_content` в самих flows; оно не держит их сильными
ссылками ради статистики. После завершения раундов закрываются clients,
listener, engine connection tasks и origin. Ссылки `mitmproxy.ctx`
восстанавливаются, затем выполняется GC. Weak references подтвердили ноль
живых proxy flows после cleanup во всех девяти HTTP-процессах.

Это контроль одинаковой HTTP-работы поверх импортированного Runtime. Direct
вариант использует HTTPX, а proxy — исходящий networking mitmproxy; текущий
engagement HTTPTransport в этом тесте не участвует. Значения нельзя называть
добавочным расходом целой allocation или подтверждением production authority.

## Начальная стоимость импортов

Каждая строка получена в трёх отдельных процессах:

| Граница import | Текущий RSS |
| --- | --- |
| Скрипт с measurement helpers и psutil | Около 31 MiB |
| Только модуль HTTPTransport | 57.25–57.30 MiB |
| Только модуль model-free ToolWorker | 72.17–72.31 MiB |
| AllocationService | 126.03–126.75 MiB |
| Runtime CLI | 125.77–126.38 MiB |
| Минимальные модули mitmproxy в отдельном процессе | 80.94–81.11 MiB |
| Модуль DumpMaster в отдельном процессе | 82.52–82.67 MiB |
| Runtime CLI, затем минимальные модули mitmproxy | 164.52–165.12 MiB |

В последней строке **попарная** прибавка внутри каждого процесса от импорта
mitmproxy равна 38.25–38.64 MiB. Она относится к import, а не запуску engine,
трафику или allocation. Старые +49–56 MiB после произвольного protocol fixture
не используются как оценка цены mitmproxy.

Малый набор активных addons не означает малый набор загруженных модулей:
`Master` импортирует `addons.termlog`, что выполняет package initializer
`mitmproxy.addons`. В закреплённой версии этот initializer заранее импортирует
все 31 стандартный addon. Minimal import загрузил 866 модулей, DumpMaster —
917. Поэтому выигрыш только от отказа от DumpMaster на import здесь около
1.5–1.7 MiB; размеры активных экземпляров и их runtime effects этим сравнением
не измерены. Сокращение зависимостей потребовало бы изменения import graph
самого пакета. Fork или monkeypatch не предлагается без отдельного измерения
и подтверждённого выигрыша.

## Буферизация и streaming на одинаковой нагрузке

Использован `Master` с `Proxyserver`, `NextLayer`, `TlsConfig` и measurement hook,
без `View`, UI, архива flows или сохранения сообщений в нашем hook. Для обоих
proxy-вариантов задан `body_size_limit="16m"` и `store_streamed_bodies=False`.
Различие — `stream_large_bodies=None` против `stream_large_bodies="1m"`.

| Состояние | Direct | Proxy с buffering | Proxy со streaming |
| --- | --- | --- | --- |
| До запросов, после setup | 128.33–128.38 MiB | 170.16–170.36 MiB | 170.20–170.27 MiB |
| Четыре ответа на границе измерения, все раунды | 129.00–129.06 MiB | 240.16–241.56 MiB | 171.56–173.23 MiB |
| После cleanup и GC | 129.00–129.06 MiB | 240.83–241.56 MiB | 171.58–173.23 MiB |
| Тела в proxy flows на границе измерения | — | Ровно 32 MiB | 0 bytes |
| Живые proxy flows после cleanup | — | 0 | 0 |

В этом сценарии streaming снижает наблюдаемый RSS примерно на 67–70 MiB.
Разница больше 32 MiB payload: RSS учитывает также обработку и временные
выделения. Точную долю Python/native allocator здесь не разделяли.
Высокий RSS после cleanup у buffered варианта сохранялся при нуле живых flows;
одного RSS недостаточно, чтобы заявить утечку. Три раунда также не доказывают
стабильность долгоживущего процесса под произвольной нагрузкой.

Streaming исключает чтение полного тела из flow и его произвольное изменение
после получения. Scanners продолжают получать тело потоком, но потребителям
полного evidence нужна своя ограниченная политика хранения. Эти свойства
описаны в [документации mitmproxy](https://docs.mitmproxy.org/stable/overview/features/#streaming).

## Что оптимизировать в Contractor

1. **Ограничить число одновременно буферизуемых bodies и их размер.** Для
   будущего engine отдельно задать общий предел активных exchanges и общий
   memory budget. Только `body_size_limit` ограничивает один body и не
   ограничивает сумму. Настройки остаются производными действующей HTTP policy;
   эксперимент не вводит новый общий engagement quota.
2. **Провести streaming через весь путь.** В нынешнем
   [HTTPTransport](../../runtime/src/contractor_runtime/adapters/http_transport.py)
   `_request` собирает `bytearray`, затем создаёт `bytes(body)`. Входной proxy
   перед этим transport не устранит его полный буфер. Выигрыш требует либо
   исходящего engine, либо изменения существующего transport/result path.
   Действующие evidence и truncation guarantees при этом сохраняются.
3. **Избегать повторного хранения полного тела.**
   [ControlledHTTPSession](../../runtime/src/contractor_runtime/toolsets/http/controlled.py)
   строит preview и evidence, а
   [RecentHTTPCaptures](../../runtime/src/contractor_runtime/toolsets/common/http_capture.py)
   хранит canonical JSON с base64 в ограниченном 64 MiB buffer. Base64 добавляет
   примерно треть к payload, transient parsing/validation создаёт ещё копии.
   При внедрении engine не следует дополнительно хранить то же тело в flow.
   Сокращать capture budget без решения о доступности evidence нельзя.
4. **Проверить model-free Runtime как отдельную оптимизацию startup.**
   [factories.py](../../runtime/src/contractor_runtime/factories.py), allocation
   service и cleanup импортируют ADK factory, Agent Skills и ADK caches заранее.
   При этом [capability discovery](../../runtime/src/contractor_runtime/capabilities.py)
   вызывает `probe` каждого зарегистрированного Worker runtime. Перенос imports
   внутрь `create` не поможет, если probe всё равно загрузит ADK. Процесс,
   объявляющий только реально обслуживаемый tool runtime, мог бы избежать этих
   импортов, но это отдельное решение об advertised capabilities и composition.
   Разница 72 против 126 MiB не является измеренной экономией такого процесса:
   сравнивались разные import boundaries, а не две готовые Runtime сборки.

Существующая очистка ADK callable caches и удаление allocation-owned sessions
уже реализованы. Повторная новая очистка поверх них не обоснована этой оценкой.
Изменение prompts или лимитов модели не уменьшает начальный RSS импортов.

## Что оптимизировать при встраивании mitmproxy

- Отложить import и создание engine до реально выбранного proxy path, если
  startup capability contract позволяет. Это экономит примерно 38 MiB импорта
  для процесса, который ещё не использовал engine. После первого использования
  модули остаются в процессе даже у следующей allocation без proxy.
- Использовать streaming threshold и `store_streamed_bodies=False`, а также
  ограничение активных запросов. Число `1m` выше — параметр fixture, не выбранный
  production default. Для SSE решение нужно принять до чтения бесконечного body.
- Не удерживать flow objects в callbacks и не подключать ненужную историю.
  Для sitemap достаточно ограниченных normalized metadata; synthetic responses
  остаются отдельными outcomes. Пример measurement hook хранит только weak refs.
- WebSocket требует отдельного byte/message budget. В текущем engine сообщения
  добавляются в `flow.websocket.messages` даже без View. Кроме того, до event
  собирается полное сообщение из frames. Очистка истории в hook ограничит
  завершённые сообщения, но сама не задаст предел незавершённого сообщения.
  Нужны проверка на настоящем долгом соединении и согласованный message limit;
  HTTP `body_size_limit` недостаточен. Это source inspection, не WS load test.
- Ограничить время жизни idle connections и caches по потребности caller.
  Не следует отключать HTTP/2 или reuse только ради импортов: они могут снизить
  число TCP/TLS соединений при той же конкурентности; это отдельный нагрузочный
  trade-off. CertStore закреплённой версии уже ограничен 100 generated entries.
- Вынести engine в дочерний процесс только при требовании возвращать его память
  ОС между использованиями. Пока child работает, отдельный interpreter может
  увеличить суммарную память. Ни выгода отдельного процесса, ни его lifecycle
  этим тестом не измерены; новый wrapper ради предполагаемой экономии не выбран.

## Рекомендованный следующий срез

Для будущего engine сначала сохранить один существующий allocation/HTTP owner
и доказать ограниченную буферизацию, streaming и полную остановку. Большой
измеренный выигрыш находится в обработке тел, а не в количестве подключённых
addons. Текущая production-реализация остаётся прежней до выбора конкретного
protocol caller и conformance по scope/DNS/peer/lease/retry.

Если целью станет уменьшение idle RSS чисто инструментальных Runtime,
сначала отдельно собрать и измерить такой настоящий процесс с корректной
регистрацией capabilities. Выполненное сравнение imports указывает направление,
но не даёт готовой оценки его памяти или цены реализации.
