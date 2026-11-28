# Context compaction: live-исследование

28 ноября 2026 года R5 остановлена с **63 выполненными из 120 запланированных
участников**: 31 Run succeeded, 32 failed; 62 native assessments — fail,
один выполненный участник unscored, ещё 57 не запускались. Все четыре native
сравнения остаются **inconclusive**. Полной повторной проверки handoff/masking
нет; V417-010 не закрыта. Для новой серии сначала нужны исправления измерений
[V500-005](../../tasks/v500-005-delivered-context-observer-evidence.yml) и
[V500-006](../../tasks/v500-006-context-eval-output-contract.yml).
[Все 120 исходов, расходы и native summaries](../../tasks/evidence/v417-010-results-r5-20261128.json).

Это исследование показало рабочее применение compaction и несколько конкретных
проблем на границах инструментов, telemetry и evaluator. Оно пока не позволяет
выбрать `includeReasoning`, оптимальный threshold или production-пилот.
Замороженные оценки сохранены без исправления задним числом.

## Исполнение и причина остановки R5

| Сравнение | Expected | Выполнено | Succeeded / failed | Scored | Не запускалось |
| --- | ---: | ---: | ---: | ---: | ---: |
| Terminal reasoning off/on | 30 | 30 | 2 / 28 | 30 | 0 |
| Compactor reasoning off/on | 30 | 30 | 28 / 2 | 30 | 0 |
| Terminal off / terminal handoff | 30 | 3 | 1 / 2 | 2 | 27 |
| Controlled compact / masking | 30 | 0 | 0 / 0 | 0 | 30 |

Непосредственный триггер остановки consumer воспроизведён read-only:
`verify_export` требует `otlp-http@1` у каждого diagnostic report и отвергает
`planner_executor` с `adapters: []`. Два Worker report этого Run полны и
успешно flushed; Planner spans также сохранены. Успешный Run не получил
assessment из-за ошибки evaluator. Дополнительно native Eval помечает его
80 328 наблюдаемых токенов как `partial`; даже после устранения ошибки observer
это блокирует новые submissions по правилу полного cumulative usage.
Причина неполной маркировки требует разбора общего владельца accounting;
наличие токенов в spans не превращает native counter в complete.

После остановки все четыре внешних experiment закрыты через native cancellation;
уже завершённые Run не заменялись и не повторялись ради лучшей оценки.
`make eval-context-compaction-live` завершился ошибкой неполной серии
(consumer exit 1, make exit 2). Инфраструктурные и measurement-сбои остаются
отдельными от ошибок доменных ответов.

## Что можно заключить из артефактов и traces

Во всех 30 compact off/on Runs проверен исходный OTLP и native diagnostics:

- В **24/30** применена хотя бы одна compaction: по 12/15 в каждой стороне.
  Шесть небольших исполнений не создали достаточного давления на контекст.
- В **21/30** успешные tool spans начинаются после конца успешной операции
  `operation.kind=context_compaction`. Это реальное продолжение работы.
- В OTLP всех 30 отсутствует `model.phase`. Producer выставляет этот атрибут,
  но allowlist адаптера его удаляет; имя context span заменяется на
  `contractor.worker.error`. `operation.kind`, status и timestamps сохраняются.
  Замороженный observer поэтому считает продолжение нулевым во всех 30.
  V500-004 проверила emitted phase на синтетической protobuf-проекции,
  но не обнаружила потерю через настоящий adapter.
- Native diagnostics сообщает 48 compactor calls и два archive reads.
  Доступность reasoning видна в применённых checkpoints обеих сторон;
  off не включает reasoning, on включает его с отмеченным bounded truncation.
  Последний snapshot каждого Run не является полным журналом всех checkpoints.
  В первом terminal-блоке также разобраны 14/14 auxiliary inputs каждой
  стороны: tagged reasoning отсутствует во всех off и присутствует во всех on.
  В восьми on inputs reasoningTruncated=true. Это подтверждает применение
  переключателя и его лимита, а не выигрыш от него.

[Счётчики по каждому compact Run и исходная граница потери phase](../../tasks/evidence/v417-010-otlp-phase-observation-20261128.json).
Supplementary подсчёт порядка действий не заменяет frozen native quality check.
Source receipts из обычного `read_artifact` и чтение native archive — разные
измерения. HTTP/probe capability не предоставлялась; отсутствие сетевых probes
обеспечено capabilities и не доказывает решения модели.

Среди 31 опубликованного результата все 31 — синтаксически корректный JSON,
все сохраняют constraint IDs. Все 13 опубликованных OpenAPI документов
принимает pinned `kin-openapi` validator. Во всех девяти опубликованных source
reviews проекция finding на path/line/kind совпадает с независимо разобранным
AST; точная структура finding совпадает только в пяти. Эти частичные проверки
не повышают native quality assessments до pass.

Общий `well_formed_evidence=false` часто описывает несовпадение контракта, а не
невалидный JSON или потерю сведений. Scorer напрямую требует `responses[200]`
для каждой операции и operation-level security, хотя видимое задание допускает
стандартные альтернативы. Offline замена только POST200 на POST201 или перенос
той же tenant security на уровень документа ломает scorer при сохранении
валидности OpenAPI. В live-артефактах шесть документов используют 201, два имеют
LikeC4 как list вместо ожидаемой строки. В negative-memory видимое «observed
fact IDs» не уточняет probe IDs против receipt IDs и состав набора; все девять
опубликованных ответов расходятся с точным expected observed set. Этот fail
нельзя целиком приписывать compaction.
[Независимые counterexamples](../../tasks/evidence/v417-010-scorer-contract-counterexamples-20261128.json).

Из failed attempts сохранены `worker_summarized`, `worker_summarization_failed`,
`worker_output_limit_exceeded` и `worker_compaction_failed`. Успех lifecycle
Run и успех полного quality gate учитываются отдельно. У terminal off/on
28 из 30 Run не дошли до обычного опубликованного результата; по этой серии
нельзя оценить самостоятельную полезность текста terminal summary.

## Наблюдаемые расходы и неопределённость

Для **62/63** выполненных участников полны **3 370 127 токенов**. Ещё у одного
наблюдаемы 80 328 токенов с native coverage `partial`: это не полный итог.
Для всех 63 полны counters: **495 model calls, 528 tool calls, 121 tool failure**;
native стоимость всех 63 — **USD 0.5732791227 reported**. Cached-input counters
не доступны как полная сравнимая метрика. Cost per quality-passed result
не определена: подтверждённых полных quality passes нет.

| Role | Calls | Tokens в Runtime phase breakdown | USD reported |
| --- | ---: | ---: | ---: |
| Worker main | 380 | 2 385 844 | 0.3551507646 |
| Compaction | 48 | 627 676 | 0.1367655726 |
| Finalization | 30 | 56 591 | 0.0138111793 |
| Terminal summary | 32 | 364 154 | 0.0655014092 |
| Planner handoff | 5 | phase total недоступен; spans содержат 16 190 | 0.0020501970 |

Role costs и Runtime phases — breakdowns, их не прибавляют повторно к cumulative
Eval usage. Родительские Audit и дочерние Run расходы также не суммируются.
Цена может иметь complete coverage при partial token counter.

В двух полностью выполненных блоках native metric cohorts включают все 15 пар,
в том числе execution failures:

| Сравнение | Tokens off / on | p50 / p90 tokens off | p50 / p90 tokens on | Ratio on/off, условный 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Terminal | 404 732 / 479 196 | 26 287 / 39 685 | 33 659 / 44 364 | 1.184 [1.011; 1.394] |
| Compactor | 1 011 437 / 1 418 003 | 65 994 / 99 572 | 92 986 / 149 717 | 1.402 [1.067; 1.802] |

Native duration p50/p90: terminal off 82.2/101.6 с, on 71.3/210.9 с;
compact off 141.3/179.5 с, on 177.4/285.5 с. Ratio суммарной duration:
terminal 1.514 [0.996; 2.106], compact 1.360 [1.065; 1.677].
Intervals — supplementary percentile bootstrap из 10 000 resamples существующих
native pair deltas, seed417. Они условны для этих трёх синтетических кейсов и
их повторов, не отражают неопределённость на новых задачах и не создают нового
comparison verdict. Пары — независимые end-to-end Run, не replay одинакового
checkpoint. Первые блоки частично выполнялись одновременно с Facts на общем
Gateway; duration не является изолированным performance benchmark.

В этих блоках on расходует больше токенов, но валидного подтверждения качества
недостаточно. Terminal и compactor рассматриваются отдельно: выбирать off/on
или оптимальные параметры по этим данным преждевременно. Handoff и masking
не имеют достаточных повторов для сравнительного вывода.

## Заморозка, воспроизводимость и следующие шаги

Все четыре R5 сравнения прошли native source/task preflight до первого dispatch:
120 expected members, три кейса, пять повторов на сторону, seed417,
concurrency1, 20 минут на участника и 24 часа на consumer. Все прочие finite
call/token/tool ceilings сохранены в manifest. Модель осталась настроенной
`glm-5.3-flash`, upstream `openrouter/z-ai/glm-5.3-flash`; четыре фактических
Worker/Planner/summary/compactor alias принадлежат общему frozen policy set.
Физическая ревизия весов недоступна.

Runtime: Python3.13.16, ADK2.15.0, lock
`sha256:e644f2f9bac0d9eac615deb2a425da4214781795253bc20af05e847efa3197ee`,
build `sha256:01d5753d7daf3184358a0ef393114ec292c2f6f6f9845894376499c972c9c88a`.
Server source — `d4b8c5112fef383a47ce3aae9abc1094e346c2e5`.
Policy clones сохраняют текущие configured sampling/aliases; это другая
заморозка, чем ноябрьские ограниченные policies старых попыток. Template4,
Workflow1; ratio0.006 при context1M даёт soft6000, keepRecentGroups1,
maxCompactions4. Эти значения не объявляются оптимальными.
[Полная заморозка](../../tasks/evidence/v417-010-freeze-r5-20261128.json).

Перед вторым блоком исправлен observer по emitted phase. Read-only replay
обоих observers совпал на всех 30 прежних terminal Run; execution IDs и ранее
выбранные records сохранены. Resume использовал прежние native plans и
submission keys, без повторных model calls. Первоначальный R5 consumer snapshot
с 30 завершёнными и 90 незапущенными — пересекающийся snapshot той же серии,
не дополнительная выборка. Последующий реальный export выявил описанное выше
ограничение этого исправления.
[Observer bridge](../../tasks/evidence/v417-010-observer-phase-r5-20261128.json).

Предложения по следующему шагу:

1. V500-005: проверить evaluator через реальный OTLP adapter, разделить контракты
   Worker/Planner и сохранить fail-closed при неполном native usage. Отдельно
   выяснить partial token coverage handoff; не подменять его суммой spans.
2. V500-006: объявить точный видимый output schema и независимые domain checks;
   различать JSON/schema/данные/continuation. Подтвердить исправление offline
   на контрпримерах до новой заморозки.
3. После этих prerequisites подготовить новую frozen серию. Старые identities,
   truth, scores и расходы сохранить. Выбор includeReasoning, threshold и
   production-пилота оставить открытым до валидного сравнения.

R1–R4 остаются отдельными неполными попытками. R4 от 28 ноября: два failed,
один cancelled, 117 unsubmitted из 120. Её collector отвергал допустимые
external Run IDs; проверка формата исправлена, R5 получила новые identities.
Все три R4 Run released; первоначальный consumer timeout и последующий native
cancellation сохранены отдельно.
[Evidence R4](../../tasks/evidence/v417-010-protocol-attempt4-20261128.json).

`debug` был выбран до dispatch. Для всех 63 R5 Run сохранено 1 276 исходных
spans, digests начальной redacted OTLP-проекции контекста и точные native outputs/diagnostics/costs.
Digests вычислены по сохранённым model alias/phase и GenAI input messages,
system instructions и tool definitions первого Worker span. Они меняются
вместе с Run/revision IDs; это не raw provider bytes, не controlled SDK
projection после callback и не доказательство одинакового checkpoint.
Controlled later-question/masking adapter в R5 не исполнялся. Truth, вопросы,
ответы и reasoning остаются вне публичного отчёта и model workspaces.

Собственные процессы joined; все 103 Run общего стенда released. PostgreSQL
custom archive восстановлен в проверочную базу: совпали counts и sorted row
digests всех 61 таблицы и состояния sequences. Обе базы и собственный cluster
удалены, временные ключи/PKI/workspaces удалены. Demo и общие secrets не менялись.
Существующий Phoenix оставлен работающим; 200 сохранённых packets / 1 867 spans
общего стенда получили подтверждение доставки.
[Общий cleanup](../../tasks/evidence/v500-002-shared-cleanup-20261128.json).

Offline fixture gate прошёл: 45 Python cases, 105 recorded samples и два Go
package suites. Эти проверки не доказывают пригодность live observer: найденные
разрывы export-границы сохранены как отдельные дефекты. `make lint` прошёл полностью; после финального bookkeeping также прошла
проверка task records/BACKLOG. Полный release gate не запускался.

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
