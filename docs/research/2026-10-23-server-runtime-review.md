# Server and Runtime review — 2026-10-23

## Проверенная база и статус

Первоначальный обзор был сделан на `39d89438` с незакоммиченными изменениями
V405-002. Повторная проверка выполнена на неизменяемом `main`
`dc7012e8b1dc11efb1be4505f3c31a7d743a04f4`. Последующий `725b863e` добавил
только материалы V408 и не изменил проверенный код Server/Runtime.
Описание ниже заменяет исходные утверждения и оценки этого обзора. Номера
B1–G15 сохранены для связи с задачами; номера строк в задачах относятся к
проверенной базе и требуют сверки перед реализацией.

Область: Go Server (`internal/`, `cmd/`) и Python Runtime (`runtime/src/`).
UI не проверялся; изменения публичных контрактов в будущем должны учитывать
его потребителей. Метод: четыре независимых разбора по областям, проверка
спецификаций, существующих тестов и production-путей, затем совместная сверка
и локальные диагностические пробы. Это проверка утверждений, не полный
security scan и не доказательство отсутствия других дефектов.

**V407-001…039 остаются pending. Исправлены описания, планы, границы и
критерии приёмки; исправления кода ещё не выполнялись.**

- `defect` (15): 001, 007–010, 019, 021–024, 026, 030–032, 037.
- `chore` (24): 002–006, 011–018, 020, 025, 027–029, 033–036, 038–039.
  Сюда входят оптимизации без изменения контрактов, рефакторинг,
  характеристические тесты, проектное исследование 018 и exit gate 039.
- Изменение credential lifecycle в 018 — возможное расширение поведения,
  а не доказанный дефект. Текущая задача готовит предложение; реализация
  потребует отдельного принятого решения и отдельной задачи.

Подтверждённая лишняя работа не означает измеренный production impact.
Критерии оптимизаций проверяют число операций, ограниченность памяти,
сохранение поведения и отмену; они не обещают неподтверждённые миллисекунды
или недостижимые размеры входов. Проверки PostgreSQL требуют непустой
`CONTRACTOR_TEST_DATABASE_URL` отдельной тестовой базы: пропуск не является
успешной проверкой.

## B. HTTP API и контракты

| Finding | Проверенный результат и решение | Задача |
| --- | --- | --- |
| B1 | HEAD действительно доходит до public GET handlers, включая Audit review. Ответ зависит от входа: возможен 400, а не только 200. Нужен ранний 405 после существующей auth-проверки; нельзя приписывать всем private endpoints тот же контракт. | V407-022 defect |
| B2 | Eval допускает `a!b` и `a/b`, которые общая грамматика Idempotency-Key отвергает. Объединить правило и проверить всех потребителей общей фикстурой. | V407-021 defect |
| B3 | Login читает 8192 bytes при объявленных 2048; часть Git body limits не объявлена. Согласовать фактический и публичный контракт, включая представление максимальных строк. | V407-021 defect |
| B4 | Дополнительная гидратация есть для preparing/inventory Audit. GeneratedInventory защищён фазой rounds: чтение его payload из списка не подтвердилось. Нужна пакетная гидратация с сохранением всех полей ответа, не урезание list DTO. | V407-004 chore |
| B5 | Обрыв RuntimeConfig publish body (`unexpected EOF`) превращается в 500. Классифицировать как ошибку запроса, сохранив отдельный oversized-body outcome. | V407-023 defect |
| B6 | Project lookup выполняется до проверки query, имени артефакта, media type и preconditions. Уточнить порядок валидации без изменения auth, ownership и согласованных статусов. | V407-023 defect |
| B7 | HTTP 415 для неверного media type предусмотрен OPS-R60. Унификация к 400 исключена; устаревшее описание реестра можно исправить. | V407-021, сохранение контракта |
| B8 | Разные JSON helpers сами по себе не доказывают ошибку. Обязательное объединение удалено из defect scope. | V407-023, исключено |
| B9 | Избыточные nil checks — локальный cleanup, не основание для mutation defect. Обязательная зачистка исключена. | V407-023, исключено |
| B10 | Вложенные Git deadlines не складываются; handler дополнительно владеет socket deadline. Удаление таймаутов как «трёх одинаковых ожиданий» исключено. | V407-023, исключено |
| B11 | UTF-8 byte bounds отличаются от OpenAPI maxLength в code points. Описать и проверить обе границы, включая многобайтный ввод. | V407-021 defect |
| B12 | Лимит 100 у Operations предусмотрен OPS-R168. Общая замена на 200 исключена; совпадение некоторых чисел не делает все page contracts одинаковыми. | V407-021, сохранение контракта |
| B13 | Некоторые mutation handlers буферизуют тело до проверки обязательных заголовков. Исправить конкретные пути и precedence; общий универсальный preamble не является обязательным дизайном. | V407-023 defect |

## C. Persistence

| Finding | Проверенный результат и решение | Задача |
| --- | --- | --- |
| C1 | Построчный persistPlan и statement-trigger на каждый INSERT подтверждены. 10 000 public members не помещаются в документ 1 MiB; native максимум 1000. Production preparation test на 1000 уже есть: расширить его счётчиками и rollback cases. | V407-028 chore |
| C2 | Project-only lookup имеет возможность улучшить индексный доступ. Seq scan на каждом poll не измерен: сначала EXPLAIN на реалистичном наборе, затем обоснованный predicate/index. | V407-025 chore |
| C3 | Placement SQL grammar из V405-002 отсутствует на проверенном main. Не считать дефектом main, не удалять SQL validation; проверить committed реализацию при завершении V405. | V407-024, вне текущего scope |
| C4 | CollectionReceiptIDs делает отдельные запросы на entries; результат ограничен 256. Пакетировать с сохранением ownership и лимитов. | V407-002 chore |
| C5 | Held-Audit credential lookup допускает оптимизацию. Нужность GIN/другого индекса подтвердить планом и размером данных, а не требованием «всегда index scan». | V407-025 chore |
| C6 | Предложенная two-writer artifact race не воспроизведена на PostgreSQL. Не менять ClassReferenceViolation mapping без отдельного доказательства и точного constraint. | V407-026, не подтверждено / вне scope |
| C7 | Run store возвращает неклассифицированную ссылку на отсутствующий Project. Публичный путь уже проверяет Project и возвращает 404; это store-boundary defect, не доказанный public 500. | V407-026 defect |
| C8 | Строгая валидация Run events на чтении соответствует правилу репозитория. Не удалять её на основании immutability. | V407-026, сохранение контракта |
| C9 | Stage triggers назначены на любой UPDATE. Сузить только несущественные обновления; старый тест отсутствия lifecycle event уже проходит, поэтому проверять реальные вызовы/определение триггера и существенные изменения. | V407-025 chore |

## D. Scheduler, Control Plane, credentials и telemetry

| Finding | Проверенный результат и решение | Задача |
| --- | --- | --- |
| D1 | Эксклюзивный barrier удерживается через remote delete, как требует RCF-R52. Меньшая область блокировки требует доказательства безопасности всех reference writers. Подготовить сравнение вариантов, без реализации нового поведения. | V407-018 chore / proposal |
| D2 | Control Plane создаёт Transport на вызов; A2A — на Invoke, но keep-alive отключён и при polling. Повторные TLS connections подтверждены, утечка idle connections — нет. | V407-013 chore |
| D3 | Перекрывающиеся placement resolutions есть, но не всегда ровно 2×N. GetRun/gofmt из незакоммиченного V405 отсутствуют на базе. Повторное использование ограничить попыткой; Operations revision меняется и на heartbeat и сам по себе не обосновывает cross-poll cache. | V407-014 chore |
| D4 | Go trimming повторяет полную сериализацию; Runtime private wire ограничен 1 MiB, поэтому 4 MiB report через него не проходит. Локальные Server producers/redaction рассматриваются отдельно. Python build_report включён в scope. | V407-020 chore |
| D5 | Повторная материализация credentials и сборка metrics есть на normal path; после restart восстановление необходимо. Оптимизировать с сохранением redaction и recovery. | V407-015 chore |
| D6 | deleteDirty может остаться после неоднозначного DB INSERT; Recover его не снимает. Обычная ошибка manager.Delete не воспроизводит этот путь. Перед снятием fence надо доказать завершение неоднозначной транзакции. | V407-019 defect |
| D7 | Idle claim probes растут с числом lanes. Объединение должно сохранить независимые model/tool pools, fairness и доступную конкуренцию. | V407-016 chore |
| D8 | Дубли нормализации resource sentinel можно объединить при сохранении semantics. Credential classifiers имеют разные роли; доказанной ошибки классификации ErrKeyUnavailable/ErrCrypto здесь нет. | V407-020 chore; classifier часть исключена из 019 |
| D9 | Terminal release последователен и допускает head-of-line blocking на медленном вызове. Быстрый 409 не тратит 30 секунд. Проверять управление concurrency и retry/backoff, не N×timeout без условий. | V407-016 chore |
| D10 | Operations history копируется при heartbeat; ring ограничен 255. Сократить копирование без изменения cursor overflow/snapshot semantics. | V407-020 chore |
| D11 | Startup использует copied pool config: connect timeout 5 s и query tracer 20 s уже ограничивают ожидание. Пробы опровергли бесконечное зависание. Зафиксировать владельцев бюджетов и покрыть startup тестами; не объявлять новый SLA. | V407-017 chore |
| D12 | История Stage читается на cancelling path без active Stage, не на каждом tick. Оптимизировать этот условный путь с сохранением завершения и cleanup. | V407-015 chore |
| D13 | Gateway outage препятствует startup recovery только при существующей prepared operation. Recovery-before-traffic задан RCF-R72. Restricted startup — отдельный проектный вариант, не автоматически разрешённое изменение. | V407-018 chore / proposal |

## E. Audit, Eval и Planner

| Finding | Проверенный результат и решение | Задача |
| --- | --- | --- |
| E1 | Ошибка standard loader при сверке уже сохранённых proposals превращается в permanent invalid. Сохранить transient class; permanent validation должна использовать существующие receipt outcomes. | V407-001 defect |
| E2 | Missing/invalid result artifact ошибочно получает retryable. Классифицировать по причине и стабилизировать выбор ошибки; bounded attempt policy не означает бесконечный retry. | V407-008 defect |
| E3 | Часть повреждённых pinned scan data возвращается как обычная retryable ошибка. Missing plan/report уже дают coverage gap; часть decode paths защищена раньше. ErrPermanent приводит к collection-contract-invalid receipt, не обязательно Audit failed. | V407-001 defect |
| E4 | При подготовке inventory один OpenAPI повторно разбирается и хешируется на operation. Переиспользовать подготовленный source в batch; единичный Run caller уже обрабатывает одну operation. | V407-006 chore |
| E5 | Дескрипторы читаются вместе с payload и retention выполняется отдельно. Реальные границы: 4096 evidence links на результат, 1024 retained links в receipt; не 16 384. | V407-002 chore |
| E6 | ResolveAuditProposals имеет N+1, но batch ограничен 128, а proposals на Run — 4096. Счётчики и тесты должны следовать этим границам. | V407-002 chore |
| E7 | Metadata callers читают полный FrozenPlan, GET повторяет nativeReadiness. Validate("Experiment") проверяет собранный ответ, не portable plan. Сохранить строгие read/output проверки. | V407-029 chore |
| E8 | Один progress на snapshot подтверждён, но RunOnce сразу повторяет шаг. Fixed-budget batching полезен; завершение cancel/delete зависит от асинхронных детей и barriers, поэтому обещание ≤3 passes исключено. | V407-003 chore |
| E9 | Passthrough может отклонить candidate, но оставить invokeFailure пустым. ExecutionReport ошибочно считает tool invocation успешным; остальной report не обязательно успешен или пуст. | V407-008 defect |
| E10 | Preparing retry при инфраструктурной ошибке предусмотрен EVM-R90; Duplicate при наличии draft разрешён. Новый Cancel или предел числа retries — изменение продукта, исключённое из overflow defect. | V407-030, сохранение контракта |
| E11 | Оба batch limits сейчас равны 200. Это риск будущего drift, не текущий неправильный размер. Связать/test invariant, не заявлять воспроизведённый retry loop. | V407-001, защитный invariant |
| E12 | Overflow в usage aggregation зависит от map order; расхождение воспроизведено. P3: практический объём недостижим, но обработка границы должна быть детерминированной. | V407-030 defect |
| E13 | auditpriority не интегрирован в production; Confidence не участвует в ranking. Это не неправильное ранжирование работающих Audits. Сохранить отложенное решение V64. | V407-005, documented non-goal |
| E14 | Общую claim lifecycle plumbing можно выделить, но политика wake/fairness/budgets различается. Прямой time.Now сам по себе не доказывает runtime bug. | V407-027 chore |
| E15 | Есть повторные checks/reads. Marshal fail-open не достижим через нормализацию конкретного Workflow type; transaction-local provenance read имеет самостоятельную роль и не удаляется. | V407-005 chore |

## F. Runtime core

| Finding | Проверенный результат и решение | Задача |
| --- | --- | --- |
| F1 | Повторные полные metric dumps дают квадратичное накопление работы: 100/200 records в пробе дали 10 000/40 000 сериализаций. Уменьшить повторную работу без изменения публикации состояния. | V407-011 chore |
| F2 | Session deepcopy существует, но immutable строки не означают повторное копирование всех payload bytes. Subtree contractor требуется WRK-R105; отсутствие найденного reader не разрешает удалить его. | V407-011, сохранение контракта |
| F3 | Обычный Worker передаёт authored string в ADK: `{path}` вызывает ошибку до model call. Нужен literal instruction provider. Finalizer использует фиксированный безопасный текст; нового запрета фигурных скобок не нужно. | V407-009 defect |
| F4 | TranscriptRecorder делает ненужную работу при отключённом summarizer. Ограничить её, сохранив enabled path и cleanup. | V407-011 chore |
| F5 | Ordinary Worker не применяет effective call deadline локально; probe с истёкшим deadline сделал model calls. Audit completion path уже защищён, другие budgets существуют. | V407-010 defect |
| F6 | Runtime private HTTPS закрывает соединения. Нужен bounded reuse с проверкой identity, lifecycle и cancellation, без новых retries неидемпотентных вызовов. | V407-013 chore |
| F7 | Exporter скачивает прежний payload для сравнения. ArtifactReadResult не содержит digest, revision opaque. Использовать только доказанные size/media fast paths и bounded exact-revision comparison cache; неизвестные равные размеры требуют чтения. | V407-034 chore |
| F8 | Повторные helpers не все эквивалентны: JSON fallback и UTF-8 suffix policy различаются. Объединять только эквивалентные части, сохраняя различия явно. | V407-012 chore |
| F9 | Небольшие синхронные filesystem операции есть. Переносить с joined cancellation и сохранением ownership, не обещая измеренный выигрыш. | V407-012 chore |
| F10 | Prepare удерживает allocation lock, но heartbeat работает независимо, а lease expiry сначала ставит fence. Общий heartbeat deadlock не доказан; сохранить эти гарантии при рефакторинге. | V407-012 chore |

## G. Runtime toolsets

| Finding | Проверенный результат и решение | Задача |
| --- | --- | --- |
| G1 | OpenAPI parser принимает 4 MiB/depth 128, Server — 2 MiB/64; различается подсчёт keys в node bound. Scanner bounds уже есть в LIM-SCN-39/41/42. Согласовать документные bounds и общие boundary fixtures, сохранив разные semantic validators. | V407-024 defect |
| G2 | Синхронные YAML parse/serialize/validation блокируют loop. В пробе две проверки всего документа и три PathItem, не три whole-document passes. Устранить повторную работу и offload с сохранением lock ownership. | V407-031 defect |
| G3 | Caido decode/redaction/preview выполняются inline. Adapter ограничивает весь JSON 16 MiB, raw base64 меньше 12 MiB: 32–64 MiB fake обходит настоящий вход. Re-encode проверяет canonical pad bits; эту проверку сохранить. | V407-032 defect |
| G4 | Admission повторяет decode immutable ZIP и encode растущего набора до 64 items. Кэшировать immutable input, вынести CPU с loop, сохранив atomic collector lock и реальную стоимость output encoding. | V407-033 chore |
| G5 | Snapshot читает content для binary classification и content-based cursor; stat-only listing нарушает WSP-R91/R69. Disk I/O уже threaded. Оптимизировать glob matching и доказанные export comparisons; 88 ms не повторно измерены. | V407-034 chore |
| G6 | 30 findings из одного ZIP дали 30 metadata/payload reads и anchor derivations. Переиспользовать bounded immutable bytes; AST не хранить между файлами (FND-R169). Timeout anchors не означает потерю finding. | V407-033 chore |
| G7 | Payload cache работает, но 10 страниц дали 10 полных decodes при одном download. Кэшировать decoded representation с общей memory accounting. Новый LikeC4 line_offset — отдельная capability, не исправление честно описанного long-line ограничения. | V407-035 chore |
| G8 | HTTP/Caido TypeError превращается в retryable tool error. Code analysis TypeError становится parseErrors/incomplete; filesystem error уже retryable=false. Исправлять подтверждённые error boundaries, не объявлять все случаи одинаковыми. | V407-032 и 037 defect; filesystem часть исключена из 034 |
| G9 | Document close уже отменяет validator и joins process cleanup до lock: полного timeout ждать не требуется. HTTP request serialization задана HTP-R75/R129. Для code analysis не доказано превышение shutdown budget. Сохранить гарантии; общий unlock переписыванием не предписывать. | V407-031/032/037, уточнение scope |
| G10 | `.hh` объявлен C++ для graph, но отсутствует в shallow parser map; list_symbols пропускает header. Исправить с regression на настоящем reader. | V407-037 defect |
| G11 | Repeated page canonicalization квадратична; заменить точным incremental sizing. Unsigned cursor намеренно предусмотрен FND-R137–141; common HMAC codec имеет другой контракт и не подставляется. | V407-033 chore |
| G12 | Scanner parsing и часть prepare/cleanup I/O выполняются на loop. Offload синхронную работу с joined cancellation; async publication остаётся await. | V407-007 defect |
| G13 | Worker summary 16 KiB уже описан LIM-AWC-2/AWC-R60; generic 64 KiB не доказывает drift. CWE equality gate уже существует. ID regex duplication — cleanup; LikeC4 уже пользуется common line window. | V407-024, неподтверждённые подпункты исключены |
| G14 | До 128 payload reads Memory выполняются последовательно. Внутри serialized operation допустим bounded concurrency; MEM-R49 и последовательность writes сохраняются. Готового private Worker batch endpoint нет. | V407-036 chore |
| G15 | read_openapi_document доступен custom templates и упоминается production guidance; отсутствие shipped selection не доказывает dead API. Оставить tool. Общий exception перенести для filesystem, edit_files и workspace_changes; katana existence read проверить только на реально существующем артефакте. | V407-038 chore; katana — V407-007 |

## H. Обвязка и ограничения проверки

`regexp2` имеет test-only caller; оснований удалять его нет. Локальные
брошенные worktrees не являются дефектом репозитория. Семейные gates могут
пропускать повторный Go run внутри release-verify согласно существующему
контракту; сам по себе этот механизм не является новым V407 defect.

При повторной проверке прошли Go packages `internal/evaldomain`,
`internal/scanplan`, выбранные Audit/Planner suites и целевые overlay probes.
Полные public HTTP и часть Planner suites были ограничены sandbox-запретом
localhost bind и не считаются прошедшими.

Runtime: core группа — 213 passed и три заблокированные сетевые fixtures
(отдельный повтор с явным deselect: 213 passed); Worker State/metrics —
18 passed. Toolset группы — 280 passed / 1 failed и 250 passed / 3 failed.
Они пересекаются, поэтому числа нельзя складывать. Повторяющееся падение —
существующий wall-clock parser-gap threshold 100 ms (наблюдались 126/204 ms).
Ещё два падения — обращения теста к Linux `/proc/self/fd` на macOS. Они не
скрыты и не объявлены исправлениями текущего пакета.

PostgreSQL URL не был задан: integration tests, EXPLAIN и artifact two-writer
race не выполнялись. Live models и release-verify не запускались. Число
SQL-запросов, production latency, последствия clock skew и platform-specific
сценарии ниже остаются предметом будущей проверки. Точные команды и результаты
закрывающих запусков должны быть внесены в этот отчёт задачей V407-039.

## I. Вопросы, оставленные закрывающей проверке

| ID | Текущий ответ | Что требуется для закрытия |
| --- | --- | --- |
| Q1 | Preparing/inventory list hydration делает дополнительную работу; точное число SQL queries не измерялось. | V407-004: real-store counters на реалистичной странице, включая остальные фазы. |
| Q2 | Локальный deadline ordinary Worker действительно отсутствует; удалённая отмена его не заменяет. | V407-010: deterministic expiry/cancellation tests и joined cleanup. |
| Q3 | Production reader ADK subtree не найден, но WRK-R105 требует зеркало. | V407-011: сохранить контракт и проверить bounded publication; удаление не входит в scope. |
| Q4 | Bounds отдельного Audit inventory generator не доказаны этой проверкой целиком. | V407-024/039: trace producer-to-consumer, JSON/YAML byte/depth/node cases; новый подтверждённый gap оформить отдельно, если не покрыт 024. |
| Q5 | Document close уже отменяет validator; HTTP/Memory serialization намеренна. Shutdown bound code analysis не воспроизведён. | V407-031/032/037/039: сохранить cancellation ownership; для оставшейся гипотезы дать synchronized test либо явный disposition с условием пересмотра. |
| Q6 | Client/DB clock skew в renewClaim остался гипотезой. | V407-039: deterministic skew/takeover test, затем defect task при подтверждении; не считать гипотезу исправленной другими chores. |
| Q7 | Две projectfs ошибки объясняются Linux-only `/proc` на macOS; первоначальный третий случай не подтверждён как product defect. | V407-039: Linux run и раздельная классификация тестовой переносимости, flaky threshold и runtime failures. |

V407-039 закрывает только проверенные и реализованные решения. Если отдельная
задача целиком отклонена, её архивирование с причиной и удаление соответствующей
dependency выполняются согласованно: archived/deferred не считаются completed.
Принятый риск получает владельца, механизм контроля и условие пересмотра.
Редактура этого отчёта и задач не заменяет implementation commits или release gate.
