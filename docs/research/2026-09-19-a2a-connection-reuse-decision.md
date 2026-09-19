# V57-005: соединения A2A внутри одного invocation

Дата: 2026-09-19. Решение: **no-change для production**. Эксперимент завершает
проверку гипотезы, но не разрешает включить keepalive одним флагом.

Переиспользование действительно убирает почти все повторные TLS handshakes при
polling. При этом оно меняет поведение при истечении сертификата и некоторых
транспортных сбоях. На локальном стенде с текущим интервалом poll 100 мс выигрыш
во времени значительно меньше выигрыша в количестве соединений. Доказательств
существенного ускорения настоящего Run или снижения production CPU нет.

Production-код и настройки не менялись. V57-004 остаётся отдельной задачей для
подробного обсуждения с пользователем.

## Что именно измерено

База эксперимента — `be33cebbf8b9b5fd833d238b2bf0547d5694f319`, Go 1.25.6,
A2A Go SDK 2.5.0, Linux amd64, Intel Core i7-7700K, 8 логических CPU.
HTTP/1.1 работает по настоящему loopback TLS 1.3 с локальным CA, клиентским
сертификатом Control Plane и привязкой Runtime к SPKI. HTTP/2 и TLS session
cache выключены; сеть вне loopback и модели не используются.

Используются настоящий `Invoker.Invoke`, JSON-RPC клиент SDK и production
`cloneBoundedHTTPClient`. Сервер управляемый: это валидные SDK-encoded ответы,
без запуска Python Runtime. Счётчики снимаются на TCP accept, проверке TLS peer,
декодированном `SendMessage` и `GetTask`, а не выводятся из числа HTTP-запросов.

Сравниваются текущий `NewMTLS`, тестовая копия его transport с
`DisableKeepAlives=true` и тестовый transport с reuse. Каждый экспериментальный
transport создаётся внутри одного `Invoke`, принадлежит одному endpoint и
principal и явно закрывает idle connections в `Destroy`. Базовая TLS
конфигурация загружается до измеряемого вызова, как в production; новая копия
выделяется для invocation. Ответы не дренируются ради keepalive.

Есть два разных workload:

- Фиксированные переходы состояния: один `SendMessage`, затем 8 `GetTask` в
  benchmark. Это сравнение одинакового объёма RPC; готовность Task здесь зависит
  от номера poll. Отдельный тест с 20 poll проверяет parity с настоящим `NewMTLS`.
- Фиксированная длительность работы: сервер становится готов через 800 мс после
  получения `SendMessage`, независимо от количества poll. Интервал — 100 мс;
  фактическое число poll может различаться. Этот тест не подменяет время работы
  Task скоростью клиента.

## Измерения

| Poll interval | Baseline, мс: медиана [min–max] | Reuse, мс: медиана [min–max] | TCP / TLS на invocation |
| --- | --- | --- | --- |
| 2ms | 38.58 [37.80–45.22] | 20.52 [20.41–23.53] | 9 → 1 |
| 100ms | 826.84 [824.44–833.75] | 809.21 [808.77–810.10] | 9 → 1 |

При 100 мс разница медиан — **17.63 мс, около 2.1%** в этом
синтетическом workload. Это не оценка ускорения настоящего Run. При 2 мс доля
transport overhead значительно выше; этот interval не является текущей настройкой.

В отдельном опыте с фиксированными 800 мс работы: baseline **822.55 мс**,
8 poll и 9 TCP/TLS; reuse **808.63 мс**, 8 poll и 1 TCP/TLS. Это одна пара,
без оценки статистической значимости. Под race detector baseline сделал
7 poll, reuse — 8: время готовности сохраняется, число наблюдений зависит
от overhead клиента. Race timings не используются для performance-вывода.

Parity-тест настоящего `NewMTLS` и тестового baseline дал одинаковые
**21 TCP, 21 TLS, 1 SendMessage, 20 GetTask**; reuse — **1 TCP, 1 TLS** при
том же числе RPC. Однократные timings parity-теста не используются как benchmark.

Benchmark запускается без race detector: 3 итерации на образец, 3 образца на
каждую пару interval/reuse. PKI и запуск сервера исключены из времени; построение
SDK client, invocation и закрытие owned transport включены. Это небольшая серия
на общей рабочей машине, а не статистически подтверждённый production benchmark.
Сырые значения каждого образца и наблюдения отдельных тестов сохранены в
[evidence](../../tasks/evidence/v57-005.json).

## Повторы POST: что доказано и что не следует из опыта

В SDK 2.5.0 обе операции — HTTP **POST**, созданные из `bytes.Buffer`, поэтому
`Request.GetBody` заполнен. Application retry loop в JSON-RPC SDK отсутствует.
Go 1.25.6 `Transport.shouldRetryRequest` может повторить запрос на использованном
соединении после ошибки без записи bytes, если тело можно восстановить. После
возможной записи повтор дополнительно зависит от replayability; для POST её
меняют `Idempotency-Key` и `X-Idempotency-Key`.

Матрица проходит через SDK и bounded wrapper. Write fault вводится поверх уже
проверенного TLS, ровно до/после одного plaintext HTTP byte. Эта ветка отдельно
выполняет hostname/SPKI verification в `DialTLSContext`; benchmark использует
обычный transport без этого wrapper. Lost-response fixture сначала полностью
декодирует запрос и записывает effect, затем закрывает socket без HTTP-ответа.

| Случай | Наблюдение |
| --- | --- |
| Первый SendMessage, zero/partial write | Ошибка, 0 декодированных effects, повторов нет |
| Reused POST, zero write, GetBody сохранён | Новый TCP/TLS, успешный повтор, 1 target effect |
| Reused POST, zero write, GetBody=nil | Ошибка, 0 effects, повторов нет |
| Reused POST, partial write одного byte, без idempotency header | Ошибка, 0 декодированных effects, повторов нет |
| Reused SendMessage, потерян ответ, без header | Ошибка, 1 effect |
| Reused SendMessage, потерян ответ, любой из двух headers | Успех после повтора, **2 effects**, одинаковые RPC ID, SHA256 тела и request ID |
| Тот же случай, GetBody=nil | Ошибка, 1 effect |
| Реальный порядок SendMessage → GetTask со сбоями poll | Исходный SendMessage остаётся единственным; возможен повтор самого GetTask |

Опасный повтор SendMessage получен **после специального read-only warmup тем же
SDK client**. В нынешнем invocation SendMessage первый на новом transport;
эксперимент не обнаружил повторную dispatch в этом порядке. Сам header тоже
не доказывает idempotency на сервере: fixture намеренно не скрывает физические
повторы за deduplication. Request ID и JSON-RPC ID не дают exactly-once effects.

Тестовый `GetBody=nil` оказался достаточен для запрета наблюдаемых скрытых
повторов непустого POST, включая zero-write retry. Это вариант строгой политики
«не повторять автоматически», а не уже внесённое исправление. Альтернатива —
явно разрешить доказанно zero-write retry для GetTask и запретить расширение
replayability headers. Выбор политики должен быть частью отдельной реализации.

## Сертификаты, allocation и завершение соединений

| Проверка | Результат и граница |
| --- | --- |
| Корректные CA/SAN, неверный SPKI | Обе стратегии отвергают peer до protected HTTP request |
| Новый ключ на прежнем endpoint после закрытия старого соединения | Старый principal отвергается; новый явно bound invocation проходит отдельную handshake |
| Сертификат уже истёк при первом соединении | Обе стратегии отклоняют запрос до HTTP |
| Client verification clock проходит NotAfter между SendMessage и poll | Baseline делает новую handshake и отказывает; reuse выполняет poll по уже установленной сессии |
| Cancel во время poll delay, headers/body первого SendMessage и активного GetTask | `planner_cancelled`; 1 SendMessage, 0 poll до его начала или 1 активный poll; все соединения закрыты |
| Deadline во время headers/body первого SendMessage и активного GetTask | `worker_deadline_exceeded`, active request отменён, все соединения закрыты |
| Следующая allocation на том же endpoint | Новый owned transport, новая handshake, точный tenant каждой invocation |
| Simulated process retirement | Старые sockets закрыты; новый запрос со старым tenant достигает HTTP, но gate отклоняет его до Worker dispatch |
| SDK Destroy и CloseIdleConnections на bounded HTTP client | Не закрывают pooled socket; явный вызов на исходном transport закрывает его |

Активный GetTask отдельно подтверждён через `httptrace.GotConn`: baseline
использует второе свежее соединение, reuse — первое повторно; при cancel/deadline
обе стороны закрывают его.

Expiry проверяется управляемым клиентским `tls.Config.Time`, без ожидания
календарного срока или изменения системных часов. Продолжение уже установленной
TLS-сессии — обычное свойство keepalive; это не обход проверки SPKI. Но оно
отличается от поведения нынешнего клиента. **Прозрачная замена с сохранением
проверки сертификата на каждом poll не подтверждена.**

Сертификат идентифицирует principal, а не процесс или allocation. Same-key
restart не делает старый tenant допустимым. Retirement fixture моделирует HTTP
gate; он не доказывает production watchdog, Registry или двухфазный release.
Закрытие transport само по себе не освобождает allocation. Эти границы взяты из
[Runtime/A2A](../spec/02-runtime-and-a2a.md),
[lifecycle](../spec/04-execution-lifecycle-and-metrics.md) и
[identity/configuration](../spec/07-runtime-labels-and-infrastructure-config.md).

## Bounded responses и условия reuse

Каждый случай отправляет два последовательных SDK-запроса. Счётчик стоит под
production bounded wrapper и измеряет bytes, которые приложение прочитало из
response body; отдельно проверяется `Close` каждого body.

| Ответ | Результат | TCP: baseline / reuse |
| --- | --- | --- |
| Полный корректный Content-Length | Decode успешен | 2 / 1 |
| Полный корректный chunked | Decode успешен | 2 / 1 |
| Content-Length > 1 MiB | Отклонён до чтения body, 0 bytes | 2 / 2 |
| Chunked JSON, для завершения которого нужен >1 MiB | Decode отклонён, ровно limit+1 bytes на ответ | 2 / 2 |
| Короткий malformed/truncated JSON | Decode отклонён; полностью прочитанное entity может сохранить socket | 2 / 1 |
| Корректный первый JSON, затем ответ намеренно не завершён | Первый объект принят без EOF; body закрыт, server context отменён | 2 / 2 |

SDK декодирует один JSON и закрывает body. Поэтому этот лимит не доказывает
полную проверку размера HTTP entity, отсутствие хвоста, лимит bytes на проводе
или RSS. Незавершённый хвост — управляемая проверка partial consumption,
а не заявление об отклонении любого oversized ответа. Поведение наблюдается
в обеих стратегиях; эксперимент не меняет существующий parser или валидации.

## Решение и возможная отдельная задача

Сейчас оставить `DisableKeepAlives=true`. Число handshakes удалось уменьшить,
но включение reuse ещё требует согласованной certificate-lifetime policy,
явного владельца transport и решения о скрытых повторах POST. Одной локальной
серии недостаточно, чтобы обосновать этот объём изменений выигрышем Run latency.

Если измерения реальных запусков позднее покажут существенные затраты на TLS,
отдельный follow-up должен содержать:

1. Один transport на invocation/principal/endpoint; SendMessage первым на новом
   соединении, отсутствие общего pool между allocation/invocation.
2. Явное решение о GetBody и idempotency headers; воспроизведение матрицы при
   обновлении Go/SDK. Не вводить blind retry семантической операции.
3. Политику срока жизни аутентифицированной сессии и её тесты, включая expiry,
   key rotation и retirement; не обещать повторную TLS verification на poll.
4. Cleanup исходного transport при success/error/cancel; active запросы
   завершаются context cancellation. SDK Destroy недостаточен.
5. Сохранение bounded reads без неограниченного drain; отдельные метрики TLS/CPU
   и Run latency на репрезентативной нагрузке.

Такой follow-up в V57-005 не создаётся и не реализуется автоматически.

## Воспроизводимость

Команды, результаты и наблюдения всех случаев — в
[tasks/evidence/v57-005.json](../../tasks/evidence/v57-005.json). Основные проверки:

```sh
go test -race -count=1 ./internal/planner/a2a -run '^TestConnectionReuseExperiment' -v
go test ./internal/planner/a2a -run '^$' -bench '^BenchmarkConnectionReuseExperiment$' -benchtime=3x -count=3
go test -race -count=1 ./internal/planner/a2a ./internal/controlplane ./internal/mtls
make test-mtls
git diff --check
```

Код опыта: [fixture и benchmark](../../internal/planner/a2a/client_connection_experiment_test.go),
[retry matrix](../../internal/planner/a2a/client_connection_retry_experiment_test.go),
[lifecycle matrix](../../internal/planner/a2a/client_connection_lifecycle_experiment_test.go).
Проверенное поведение production: [A2A client](../../internal/planner/a2a/client.go),
[mTLS](../../internal/mtls/mtls.go); SDK `a2aclient/jsonrpc.go` (`newHTTPRequest`,
`sendRequest`, `Destroy`), Go `net/http/transport.go` (`shouldRetryRequest`) и
`net/http/request.go` (`isReplayable`) в указанных выше установленных версиях.
