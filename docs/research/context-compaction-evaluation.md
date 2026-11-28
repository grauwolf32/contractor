# Context compaction: live-исследование

28 ноября 2026 года пользователь возобновил исследование. Новая **R5-серия
из 120 участников выполняется** на текущем ADK 2.15.0 и настроенной модели.
Все четыре сравнения прошли native source/task preflight до первого dispatch.
Итоговая оценка R5 ещё не опубликована; старые результаты не смешиваются с ней.
[Точная заморозка R5](../../tasks/evidence/v417-010-freeze-r5-20261128.json).

Перед R5 исправлены текущая SDK attestation и имя общего Artifact reader
(V500-001/V500-003). Предшествующая R4 сохранена целиком: два failed, один
cancelled и 117 unsubmitted. Её локальный OTLP-сборщик отклонял допустимые
Run IDs внешних Eval; формат исправлен, новая серия получает новые identities.
Все три старых Run released; timeout, который видел consumer при остановке,
сохранён отдельно от последующего авторитетного cancellation.
[Неполная R4](../../tasks/evidence/v417-010-protocol-attempt4-20261128.json).
Перед вторым блоком V500-004 исправила распознавание native compaction по
`model.phase`, исключив зависимость от auxiliary alias. Оба observer дали
одинаковые наблюдения для всех 30 уже завершённых terminal Runs; их execution
IDs и ранее выбранные records сохранены. Продолжение использует тот же native
план и ключи submission, без повторных модельных исполнений.
[Обновление observer attestation](../../tasks/evidence/v417-010-observer-phase-r5-20261128.json).

Исходный OTLP новой серии сохраняется до пересылки; работающий Phoenix получает
сохранённые spans без повторения model calls.

У текущего scorer обнаружены ограничения интерпретации. Неизменённый recorded
API answer проходит проверку; замена только POST-ответа `200` на `201` или
перенос той же tenant security на уровень документа дают
`well_formed_evidence=false`. Все три документа принимает pinned OpenAPI
validator, а видимое задание не требует `200` или только operation-level
security. Это offline counterexamples, без новых model calls. Есть и реальный
валидный API-документ с `201`; тип поля LikeC4 в нём тоже расходится с ожиданием
scorer. Замороженные правила и native assessments остаются неизменными.
Общий fail нельзя без разбора приписывать потере доменных данных при compaction.
[Воспроизведение ограничений scorer](../../tasks/evidence/v417-010-scorer-contract-counterexamples-20261128.json).

## Наблюдение 28 ноября: граница instrumentation → OTLP

После завершения первых двух сравнений сохранены точные native экспорты
60 участников. Разбор всех 30 compact off/on Runs выявил ограничение
предыдущего исправления observer: producer выставляет `model.phase`, но
allowlist в `runtime/src/contractor_runtime/adapters/otlp_http.py` удаляет этот
атрибут. Имя `contractor.worker.context_compaction` также заменяется на
`contractor.worker.error`; при этом `operation.kind=context_compaction`,
успешный status и исходные timestamps сохраняются.

В 24 из 30 Runs native diagnostics подтверждает применённую compaction.
В 21 из 30 исходные spans подтверждают успешные инструменты после завершения
такой операции. Замороженный observer показывает ноль во всех 30, поскольку
не видит ожидаемого phase. Это ошибка измерения; ноль нельзя трактовать как
отсутствие продолжения. V500-004 проверила распознавание emitted phase, но её
синтетический fixture не обнаружил потерю на реальной export-границе.
[Воспроизводимые счётчики и evidence](../../tasks/evidence/v417-010-otlp-phase-observation-20261128.json).

Замороженные assessments и параметры серии не переписываются. Supplementary
подсчёт по сохранённому `operation.kind` описывает порядок реальных действий,
не заменяет native quality check. Остальные сравнения сохраняют собственные
frozen denominator. До нового валидированного измерения выводы о пользе
`includeReasoning`, превосходстве masking и выборе production threshold
остаются недоказанными. Приоритетный follow-up — согласовать emitted telemetry
с allowlist и проверять observer через настоящий OTLP adapter, а также отделить
ошибки контракта scorer от ошибок доменных данных.

## Исторический отчёт на 16 ноября 2026

Ниже сохранены исходы и ограничения прежних попыток. Указание о паузе относится
к состоянию на 16 ноября и отменено новым запросом пользователя.


V417-009 и V466-021 приняты; V417-010 не завершена и по указанию пользователя
приостановлена до команды продолжить. R3 подтверждает source/task pins во всех
60 парах; после трёх Runs новые submissions остановлены. Сетевой выход Gateway
восстановлен 16 ноября 2026 года. Обнаруженная в попытке2 недостающая provenance
external Eval исправлена в V466-021. Модель, gates, truth и параметры не
подгонялись по held-out результатам. Рекомендаций по reasoning, оптимальному
threshold или включению пилота нет.

## Что реально выполнено

| Попытка | Замороженный denominator | Завершено | Отменено | Не запускалось | Native quality assessments |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1: ошибка isolated observer | 120 | 1 | 0 | 119 | 0 |
| 2: отсутствуют native source/task pins | 120 | 36 | 1 | 83 | 36 |
| 3: пауза по команде пользователя | 120 | 3 | 0 | 117 | 3 |

Попытки имеют разные frozen identities и не смешиваются в одну выборку.
R3 заморожена на чистом `e6268ad21` после V466-021, без изменения модели,
задач, критериев или параметров. Все 60 пар прошли проверку pins до первой
submission. Три Run завершились ошибкой естественным образом; оценки — fail,
известны 123 541 токен и USD 0.0239856091 reported. Текущий Run дождались,
не отменяя; остальные 117 участников не запускались. Эта пауза не даёт
результата повторного сравнения. Временный ключ отозван, процессы joined,
база удалена после сохранения точных native exports и частного PostgreSQL
архива. Продолжение требует команды пользователя и проверки frozen clocks
и credentials; старые исходы сохраняются.
[Заморозка R3](../../tasks/evidence/v417-010-freeze-r3-20261116.json),
[пауза и расходы](../../tasks/evidence/v417-010-protocol-paused-r3-20261116.json).

Первая остановилась до assessment: `python -I -S` исключает каталог скрипта
из `sys.path`, поэтому observer не импортировал соседний client. Исправление
явно загружает pinned sibling через `importlib.util`. Тест через настоящий
изолированный subprocess воспроизводит ошибку до исправления и проходит после;
тот же сохранённый Run проверен без нового вызова модели.
[Данные попытки1](../../tasks/evidence/v417-010-protocol-attempt1-20261116.json).

Во второй попытке все 30 terminal off/on запусков завершены и оценены.
Compact off/on имеет шесть завершённых запусков и один отменённый. Handoff и
контролируемое compact/masking сравнение в held-out серии не запускались.
Всего 13 Runs succeeded, 23 failed, один cancelled. Все 36 опубликованных
quality assessments — fail; это не подменяет неизвестные исходы остальных
84 членов. Из ошибок выполнения 22 имеют `worker_summarized`, одна —
`worker_summarization_failed`.

Для 36 членов известны **1 377 320 токенов** и **USD 0.1930442420 reported**.
У отменённого члена полные usage и цена недоступны. Это известная часть
расхода, не полный итог и не нулевая цена отменённого вызова. В первой попытке
отдельно сохранён расход USD0.0048948605. Все failed/cancelled/unrun члены
остаются в своих denominator.
[Данные попытки2](../../tasks/evidence/v417-010-protocol-attempt2-20261116.json).

## Почему native сравнение не состоялось

`resolveExternal` сохраняет локальные executable snapshots и observed pins
конфигурации. `source/tasks` добавляются native plan builder, но не external
registration. Существующий `verifiedComparisonPins` поэтому правильно
отказывает в сравнении required-equal dimensions. Все 60 native пар попытки2
помечены `pins-unverified`; итог каждого сравнения остаётся `inconclusive`.
Operator-pinned входы и конфигурация в manifest не превращают producer claims
в observed provenance сервера (EVM-R31, EVM-R135, EVM-R138).

Preflight V417-010 ошибочно разрешал такой план. Теперь consumer читает
штатные pair exclusions до первой submission и отклоняет `pins-unverified`.
Регрессия до исправления допускает запись submission, после — не делает
ни одной записи. Read-only replay сохранённого native плана отклонён после
19 GET, без model calls, results или assessments.

Численные paired intervals и сравнительный token ratio здесь не публикуются:
native пары исключены, и отдельный локальный comparison engine не обходит
это решение. Даже при полной серии её gates не могли бы дать рекомендацию.
По решению пользователя общий владелец Eval исправлен в **V466-021**:
новые bindings наблюдают проверенные локальные inputs/effective tasks по
общему с native plan builder правилу. PostgreSQL, API, wire и lint проверки
прошли. Старые immutable registrations не переписываются; последующая серия
потребует новой заморозки с сохранением связи с обеими неполными попытками.

## Зафиксированный протокол

| Сравнение | Варианты | Запланировано на попытку |
| --- | --- | ---: |
| Terminal reasoning | off / on | 30 |
| Compactor reasoning | off / on | 30 |
| Terminal continuation | terminal off / terminal → streamline | 30 |
| Контролируемое продолжение | native compact / observation masking | 30 |

Три класса: Python source review; контракт 12 API routes и dependency
projection; negative-result memory из 48 наблюдений. По пять повторов на
каждую сторону. Terminal off в двух сравнениях — независимые члены.

Попытка2 зафиксирована до первого dispatch: manifest
`sha256:5cfcf096990838527eafbe9106c200609491c0a7cd2ac34951c90977f6207c8f`,
seed417 только для порядка, concurrency1, 20 минут на члена, 24 часа на
consumer. Gates: candidate pass1, quality drop0, total-token ratio1.
[Точные pins и attestation](../../tasks/evidence/v417-010-freeze-r2-20261116.json).

Настроенная модель — `glm-5.3-flash`, upstream
`openrouter/z-ai/glm-5.3-flash`. Alias Worker, Planner и summarizer сохранены
из конфигурации `cc78cc13596ab7d7`; физическая ревизия весов недоступна.
Один существующий auxiliary alias обслуживает summarizer и compactor.
Клонированные ограниченные политики сохраняют alias, context/output caps и
температуру0.6. Runtime — Python3.13.16, ADK2.13.0, lock
`sha256:945f997f38cabe445f1f05a559a26089fdf3bf44c95b523a6d8eb71c8e235c65`;
build чистого `1f80459f36dad3901fb61d4bf612bf52852b7808`.

Общие параметры: ratio0.006 при configured context1M (soft6000),
keepRecentGroups1, maxCompactions4; template version4, неизменная общая
методологическая инструкция. Live-копия фиксирует выбранные policies/Gateway
и summarizer cumulativeBudget200000 при caller maxTotalTokens250000.
Это исследуемая конфигурация, не production-рекомендация и не оптимум.

Totals принадлежат native cumulative Eval usage. Runtime phases и monetary
activity — breakdowns, не дополнительные затраты. Byte compression не
пересчитывается в токены. SDK digests — redacted request projections, не
raw provider transport; fresh Run/revision IDs отличаются. Идентичный replay
native checkpoint не поддерживается. Пары case/repetition не означают
одинаковое состояние Runtime.

## Development и ограниченные наблюдения

До held-out выполнялись отдельные development fixtures. Soft12000 сработал
после полезных инструментов: prompt12744 →3066, но дальнейшей полезной работы
не было. При soft6000 development03 имел три успешных инструмента после
checkpoint и prompt6928 →2522. Оба development-контроля реально активировались
и верно ответили на поздний вопрос; дальнейших инструментов было пять после
native compact и16 после masking. AST подтверждал identity sink, однако
schema/hypothesis-ID промахи сохранены. Эти одиночные прогоны не доказывают
улучшения. [Development02](../../tasks/evidence/v417-010-development-20261116-02.json),
[development03 и контроли](../../tasks/evidence/v417-010-development-20261116-03.json).

В попытке2 первый compact-off source Run применил checkpoint1:
prompt6646 →2233; main33658, compaction6386, finalization1020 токенов.
Уменьшение одного prompt не является экономией полной задачи. Незавершённой
серии недостаточно для выбора includeReasoning, threshold, keepRecentGroups,
maxCompactions или пилота. Terminal arms оценивают end-to-end завершение,
а не самостоятельную полезность текста terminal summary.

Observer различает валидную пагинацию text-artifact (`truncated=true`) и
native telemetry envelope `{truncated,preview}`. Последний делает evidence
недоступным на arguments и output. Проверки используют настоящий content
encoder и protobuf projection. Успешные post-checkpoint tool spans,
source receipts и native archive reads — разные измерения. HTTP probes
capabilities не допускают; их отсутствие не доказывает решения модели.

## Инфраструктура и границы

Первый отдельный development Run15 ноября остановился на DNS/WireGuard
Gateway failure: девять физических попыток, неизвестные tokens, estimated
USD0. Его исходы не скрыты.
[Development15](../../tasks/evidence/v417-010-development-20261115.json).

По указанию пользователя16 ноября Gateway переведён на выделенный peer
существующего VPN-узла. Прежний peer, Gateway identity и mesh сохранены;
deployed и защищённые локальные profile/metadata обновлены с rollback backups.
DNS, readiness, provider HTTPS200 и native fail-closed check проходят.
Настроенная модель не заменена. Persistent recovery не является disposable
ресурсом исследования.

Hidden truth, вопросы до activation, raw answers и reasoning остаются вне
model-visible inputs/workspaces и публичного отчёта. Credentials не
публикуются. Обе попытки отменены через native lifecycle, собственные ключи
отозваны, процессы joined и owned disposable database удалена после экспорта.
623 evaluator-private файла (16 275 048 байт) сохранены в ignored
`.local/v417-010-private-evidence-20261116`; ephemeral auth/cipher keys, PKI
и workspaces удалены. V417-004 остаётся зарезервированной in_progress и выполняется LAST.
