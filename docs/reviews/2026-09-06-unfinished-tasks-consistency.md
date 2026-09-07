# Проверка непротиворечивости незавершённых задач

Исторический review плана от 2026-09-06. Приведённые ниже количества pending
относятся к моменту проверки, а не к текущей очереди. На 2026-09-07 завершены
V41-001–008 и V43-001–005, а в V39 — контракты и публикация (001/004).
Текущие статусы и приоритеты задаёт [tasks/index.yml](../../tasks/index.yml).

Проверены все 39 задач, незавершённых на исходном срезе: V32-005/006/008,
V37-001–012, V38-001, V39-001–007, V40-001–003, V41-001–008 и V43-001–005.
Во время проверки V32-005/006/008 были завершены другими изменениями workspace;
они не редактировались в рамках этого review. На итоговом срезе остаётся
36 pending задач. Завершённые задачи и implementation commits не переписывались.

Проверка включает YAML/индекс/зависимости, требования и критерии каждой задачи,
границы смежных этапов и соответствующие участки текущего кода/спецификаций.
Это проверка плана, не выполнение отложенных implementation/eval задач.

## Исправленные противоречия и существенные пробелы

| Область | Проблема | Исправление |
| --- | --- | --- |
| V43-001–003: передача evidence | Подборка refs подавалась как достаточный вход reader. Но Runtime читает только свой RunScope, а одинаковые namespace/name встречаются у разных Runs. | Контракт требует exact bytes и исходное scoped происхождение. При реализации V43-001 выбран самодостаточный ZIP: V43-002 публикует его как обычный input, V43-003 при подготовке reader-toolset размещает документы в RunScope и проверяет exact refs. Отдельная ветка Run-create не нужна. |
| V43-002/004/005 и V37-010: доказательство выполнения | Обычный `go test` может пропустить DB/process cases. В V43-004 критерии требовали запуска агентов, а команды в основном проверяли config. V37-010 не запускала пакет auditservice, где находятся запросы findings/review. | Добавлены необходимые пакеты/DB-условия, явный process smoke и планируемый `test-findings-e2e`, который обязан падать при отсутствии обязательных тестов, prerequisites или при skip. Runtime reader tests включены в итоговый gate. |
| V40-002/003 и spec 26: Audit arm | V40-003 описывала все samples как обычные evaluation Runs. Разрешённые изменения refs не учитывали AuditProfile wrappers, без которых Audit продолжал бы выбирать baseline Workflow. | Разделены Workflow/Audit пути и Project-kind правила; V40-002 готовит явные versioned wrappers. Разрешены только зависимые изменения refs при равенстве inventory, standards, evidence, execution и interaction policy. Child eval labels не являются условием membership. |
| V40-003: pilot/confirmation | После pilot предполагались исправления candidate и другая матрица, но был указан один frozen plan. | Confirmation получает отдельные experiment ID, план, каталог и бюджет. Pilot остаётся неизменным, исправленные кандидаты получают новые версии; успешный pilot не закрывает confirmation acceptance. |
| V43-004 ↔ V40: общий Skill | Прямое обновление trace reference могло изменить Skill bytes, которые в instruction-only A/B должны совпадать. | Mapping размещается в новых инструкциях либо отдельной закреплённой Skill revision. Старые experiment bundles и равенство pins сохраняются. |
| V39: лимиты и wire contract | R5 V39-003 называла 512 coverage entries агрегатным лимитом, хотя spec 25 задаёт его для каждого bounded list. V39-001 перечисляла v1alpha1 schema, но не production private-v2 schemas. | Разведены per-summary/per-list/aggregate limits; добавлены private-v2 deliverable и проверка wire contracts. |
| V41-001 / spec 26: идентификаторы | Общий Id запрещал `@`, а provider/scorer refs требовали значения вида `contractor@1`; отдельный тип не был определён. | Введён явный VersionedId с общей границей длины и точным opaque version matching. Обычные case/member/variant IDs сохраняют прежнее правило. |
| Общая постановка задач | Index говорил, что исполнителю не требуется читать specs, тогда как V41 прямо требовала прочесть нормативный формат. | Уточнено правило: цель, scope и acceptance находятся в задаче; явно названные нормативные документы — её contract inputs. Неуказанные документы и прошлый диалог не являются скрытыми prerequisites. |

Опорный код для наиболее существенной доработки V43:
[allocation-bound artifact access](../../internal/httpapi/privateartifacts/handler.go),
[fork только объявленных Run inputs](../../internal/runservice/public.go).
Для Audit/eval:
[выбор Workflow профилем](../../configs/audit-profiles/openapi-operation-trace.yaml),
[Audit findings/review queries](../../internal/auditservice/finding_review.go),
[portable transport contract](../spec/26-portable-evaluation-format.md).

## Проверенные границы остальных задач

| Задачи | Результат |
| --- | --- |
| V32-005/006/008 | API → UI → измеренный gate согласованы со spec 22; выключенная телеметрия, неизвестные значения и lifecycle разделены. В ходе review линия завершена вне этих правок. |
| V37-001–005 | Общий Dialog, retained draft, явный input review и destructive confirmations совместимы. Нормальная навигация не требует повторной confirmation. |
| V37-006/007 | Authored metadata и полный server-side поиск предшествуют UI; legacy digest и opaque versions сохраняются. |
| V37-008/009 | Repeat создаёт новый draft с исходными exact refs; preview использует declared primary. Нет замены истории или угадывания текущих heads. |
| V37-010–012 | Review остаётся Server-owned; Operations layout не дублирует V32, общий UI gate не объявляет Evals/Performance завершёнными. Исправлен test scope V37-010. |
| V38-001 ↔ V41 | V38 проектирует UI/API над общим форматом; V41 реализует CLI/evaluator. V38 не блокирует CLI, и ни одна линия не создаёт второй Scheduler или writable manifest authority. |
| V39-001–007 | Контракты → encoder/Server pinning → collector → gate → rollout examples → verification образуют DAG. Частичная коллекция invocation-local; новый Audit child Run имеет новый output binding. |
| V40-001–003 ↔ V41-001–008 | Format readiness предшествует fixture-gap mapping, плану и model eval. Offline conformance не доказывает качество; Audit mode и отдельная confirmation теперь явно учтены. |
| V43-001–005 ↔ V39/V41 | Findings tools не требуют V39, аннотаций или verifier. V43 gate использует scripted models; реальные quality eval требуют завершения V41-008 и отдельного frozen plan. |

## Оставшиеся границы готовности

- V43-001 — задача фиксации wire contract. Schema/limits и способ материализации
  закреплены в [спецификации 27](../spec/27-findings-tools-and-collections.md);
  сам review не объявляет новый
  контракт реализованным. Подготовка evidence для reader RunScope — отдельная
  существенная часть V43.
- V40 live preflight требует реально доступных exact wrapper/config/Skill pins
  и отдельного бюджета. Подготовка YAML или зелёные offline tests не заменяют
  эти prerequisites; развёртывание eval-конфигураций не выполнялось.
- Полный OpenAPI trace coverage, автоматический annotation index и verifier
  остаются отдельными последующими возможностями. V43 не обещает их перенос.

Проверено после правок:

- Все 281 task YAML и index разбираются без повторных ключей/IDs; файлы и
  index совпадают, все dependency IDs существуют, циклов нет.
- У 36 pending задач проверены milestone и ссылки tests на acceptance criteria;
  каждый критерий представлен в плане проверок. Разрешаются 23 локальные ссылки
  четырёх затронутых документов.
- `git diff --check -- tasks docs tests/eval/agent_instructions/README.md` — pass.
- `go test ./internal/config ./tests/eval/agent_instructions` — pass.

Эти проверки не исполняют отложенные тесты ещё не написанных модулей. Модельные
eval, публикация конфигураций и integration gates будущих V39/V43 не запускались.
