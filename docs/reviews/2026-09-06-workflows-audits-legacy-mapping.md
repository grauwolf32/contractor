# Соответствие Workflow и AuditProfile реализации contractor-old

Обновлено 2026-09-07: **16 версионных Workflow и 5 AuditProfile** из штатных
`configs/workflows/` и `configs/audit-profiles/`. Сравнение сделано по YAML,
инструкциям и выбранным tools текущих Workers, коду старых workflows и активным
старым task templates. Это каталог репозитория; конфигурация развёрнутого сервера
отдельно не проверялась. Примеры, e2e-конфигурации и кандидат инструкций из
`tests/eval/agent_instructions/candidate/configs/` в этот срез не входят.

«Прямое по назначению» означает перенос основной задачи и стадий, а не равенство
поведения, качества, retry/iteration policy или формата артефактов. «Частичное»
означает совпадение отдельной части старого сценария. «Новое» означает отсутствие
соответствующего готового workflow/task в старом каталоге.

Уточнение целевого назначения: **`openapi-operation-trace` должен соответствовать
старому `trace_annotation` по предметной задаче** — обходить заданный OpenAPI,
трассировать обработку запросов и собирать подтверждённые кодом находки.
Аннотации — один из возможных артефактов выбранного Workflow. Их формат и
обязательность не входят в общий контракт Audit, гипотез или findings.
Текущее ограничение `operation-resolution` означает неполный перенос этого
сценария. Варианты `trace_annotation_direct` / `trace_graph` полезны для сравнения
способов исполнения, но целевой сценарий задаётся `trace_annotation`.

Ссылки на старую реализацию предполагают соседний checkout `../contractor-old`.
Имена в таблице — каталоги workflows. В фактическом
[старом реестре](../../../contractor-old/contractor/workflows/__init__.py)
`oas_building` зарегистрирован как `oas_build`, а `oas_enrichment` — как
`oas_update`; подписи `build`/`enrich` в старых README не являются ключами этого
реестра. Название task `oas_update` при этом относится к стадии построения OAS.

Текущие **Workflow** сопоставляются так:

| Текущий Workflow | Конкретный аналог в contractor-old | Соответствие | Что совпадает и что изменилось |
| --- | --- | --- | --- |
| [openapi-from-workspace@5](../../configs/workflows/openapi_from_workspace_v5.yaml) | [oas_building](../../../contractor-old/contractor/workflows/oas_building/workflow.py): `dependency_information → project_information → oas_update → oas_validate` | Прямое по назначению | Те же четыре задачи: зависимости, устройство проекта, построение OpenAPI, проверка/исправление. Теперь отдельные Stages с `passthrough@1`, точными входами и workspace overlay. Старый `TaskRunner` использовал planner + worker; наличие старых discovery-артефактов позволяло пропускать первые стадии. |
| [openapi-from-workspace-streamline@1](../../configs/workflows/openapi_from_workspace_streamline.yaml) | `oas_building`, включая planner + worker внутри задач | Прямое по назначению; ближе по организации | Те же четыре стадии, входы, артефакты и переходы, что у passthrough-варианта; в каждой стадии `streamline@1`. Политики `project_planner@1` / `project_worker@1`, усиленная проверка OAS сохранена. |
| [openapi-from-analysis@2](../../configs/workflows/openapi_from_analysis.yaml) | `oas_building` после пропуска discovery: `oas_update → oas_validate`; при наличии seed также близок к [oas_enrichment](../../../contractor-old/contractor/workflows/oas_enrichment/workflow.py): `oas_enrich → oas_validate` | Прямое для повторного использования анализа; частичное для enrichment | Два готовых отчёта и source передаются явно, `existing_openapi` опционален. Это общий build/validate, отдельная старая задача `oas_enrich` и её политика итераций не перенесены как самостоятельный режим. |
| [likec4-from-workspace@5](../../configs/workflows/likec4_from_workspace_v5.yaml) | [likec4_building](../../../contractor-old/contractor/workflows/likec4_building/workflow.py): `dependency_information → project_information → likec4_build → likec4_validate` | Прямое по назначению | Те же четыре стадии и возможность продолжить существующую модель. Сейчас `passthrough@1`, отдельные шаблоны builder/validator, точные артефакты и экспорт overlay. Раньше обе LikeC4-задачи выполнял `likec4_builder_agent` через `TaskRunner`. |
| [likec4-from-workspace-streamline@2](../../configs/workflows/likec4_from_workspace_streamline.yaml) | Тот же `likec4_building`, включая planner + worker внутри задач | Прямое по назначению; ближе по организации | Те же четыре текущих Stages, но с `streamline@1`. Это ближайший вариант к старому планированию подзадач; протокол, бюджеты и условия завершения всё равно другие. |
| [likec4-from-analysis@3](../../configs/workflows/likec4_from_analysis_v3.yaml) | `likec4_building` после пропуска discovery: `likec4_build → likec4_validate` | Прямое по назначению | Повторное использование анализа стало явным контрактом source + два отчёта; optional `existing_likec4` задаёт seed. Чтение исходников идёт через source archive tools. |
| [taint-trace-from-workspace@2](../../configs/workflows/taint_trace_from_workspace.yaml) | Одна задача [trace_annotation/v3](../../../contractor-old/contractor/tasks/trace_annotation/v3.yml), выполняемая `trace_agent`; по tools/отсутствию LLM planner ближе всего к одной операции [trace_graph](../../../contractor-old/contractor/workflows/trace_graph/workflow.py) | Частичное: одна цель | Есть graph navigation, `@trace`/`@validate`/`@sink`, overlay/diff и отчёт о потоке данных. Target задаётся извне; нет обхода всего OpenAPI, старой агрегации по paths и структурированных `vulnerability-reports`. Findings здесь описываются в Markdown. |
| [security-analysis@2](../../configs/workflows/security_analysis.yaml) | HTTP-часть [exploitability](../../../contractor-old/contractor/workflows/exploitability/workflow.py), task `exploitability_assessment`, `exploitability_agent`; также близок по роли к [http_agent](../../../contractor-old/contractor/agents/http_agent/agent.py) | Частичное: HTTP/Caido-исследование | Один `caido_analyst` работает по objective + target + authorization_scope и пишет Markdown. Старого цикла по findings, `verification_tools`, автоматического сбора `exploit-http-chains` и code execution в этом Workflow нет. |
| [findings-review@1](../../configs/workflows/findings_review.yaml) | Анализ находок по роли близок к vuln_analytics_agent | Частичное; новый общий вход | Читает самодостаточную ZIP-подборку с exact evidence через list_findings и read_artifact, публикует Markdown. Не требует diff/OpenAPI, не воспроизводит целиком trace_postdiff или trace_verify. |
| [audit-source-check@1](../../configs/workflows/audit_source_check.yaml) | Для произвольного checklist готового старого аналога нет | Новый исполнитель checklist | Выполняет закреплённый набор items `source-checklist@1` и возвращает result ZIP через `submit_check_result`. Читает source archive; graph tools отсутствуют. |
| [audit-openapi-operation-trace@1](../../configs/workflows/audit_openapi_operation_trace.yaml) | Одна задача trace_annotation, с graph navigation как у trace_graph | Частичное: ограниченный graph-only вариант | Исполнитель openapi-operation-trace@2: 11 code-analysis tools, bounded filesystem reads и точный Audit source. Canonical result и operation-resolution; findings и аннотации намеренно не подключены. |
| [audit-openapi-operation-trace@2](../../configs/workflows/audit_openapi_operation_trace_v2.yaml) | Исследование одной операции trace_annotation и публикация уязвимостей | Частичное: graph + структурированные findings | Исполнитель профиля @3 использует finding и write_text_artifact. Сохраняет evidence/receipt refs и описание воспроизведения; нет автоматического verifier и экспорта аннотаций, coverage остаётся operation-resolution. |
| [audit-top10-source-risk@1](../../configs/workflows/audit_top10_source_risk.yaml) | Близкая предметная задача — [vuln_scan/v3](../../../contractor-old/contractor/tasks/vuln_scan/v3.yml) с `codereview_agent`; разделение по классам напоминает первый проход [vuln_sweep](../../../contractor-old/contractor/workflows/vuln_sweep/workflow.py) | Частичное по анализу риска; новый контракт | Один Worker проверяет один точный standard mapping, а не весь проект по всем классам. Может создать evidence-bound candidate finding и result ZIP. Старые nomination → dedup → trace стадии не входят. |
| [audit-asvs-source-verification@1](../../configs/workflows/audit_asvs_source_verification.yaml) | Прямого аналога нет | Новое | Проверяет одно закреплённое требование ASVS с заданным evidence contract. Старый `trace_verify` проверял уже найденную уязвимость, поэтому его нельзя считать эквивалентом этой проверки требований. |
| [artifact-copy@1](../../configs/workflows/artifact_copy.yaml) | Прямого предметного аналога нет | Новое, технический пример | Один Worker копирует текстовый вход в результат; проверяет базовый путь исполнения и артефактов. |
| [podman-python-check@1](../../configs/workflows/podman_python_check.yaml) | Прямого workflow нет; лишь общее сходство с возможностями `swe_agent` | Новое, технический пример | Конкретный сценарий: исправить `add` в `calculator.py`, выполнить `check.py`, сохранить JSON-отчёт в Podman sandbox. Это не перенос универсального SWE-сценария. |

Для discovery текущий `workspace_source_graph_analyst@1` выполняет роли старого
`swe_agent` из `dependency_information/v1` и `project_information/v1`.
OpenAPI builder/validator соответствуют `oas_builder_agent` и
`oas_linter_agent`; активные старые задачи — `oas_update/v2`, `oas_validate/v1`
и отдельно `oas_enrich/v2`. Для LikeC4 активны `likec4_build/v1` и
`likec4_validate/v2`. Это соответствие ролей; старые промпты не подключаются
автоматически к новым шаблонам.

Происхождение документных цепочек дополнительно закреплено в задачах
[V2-004](../../tasks/v2-004-openapi-workflow.yml),
[V2-006](../../tasks/v2-006-likec4-workflow.yml) и
[V2-009](../../tasks/v2-009-precomputed-analysis-variants.yml).

**AuditProfile** задаёт набор проверок и правила их исполнения через дочерние
Workflow Runs. В старом каталоге не было отдельной сущности с таким контрактом:
закреплённым inventory, task packages, execution manifest, учётом coverage/gaps
и решениями по предложениям Workers. Поэтому ниже сопоставлено назначение,
а не заявлен перенос AuditProfile один к одному.

| Текущий AuditProfile | Единица проверки и дочерний Workflow | Ближайший старый сценарий | Степень соответствия и границы |
| --- | --- | --- | --- |
| [openapi-operation-trace@2](../../configs/audit-profiles/openapi-operation-trace.yaml) | Одна операция method + path; audit-openapi-operation-trace@1, batch size 1 | Ограниченная часть trace_annotation | Graph-only вариант для сравнения: inventory, workspace, source evidence и gaps; без findings, аннотаций и полного trace coverage. |
| [openapi-operation-trace@3](../../configs/audit-profiles/openapi-operation-trace-v3.yaml) | Одна операция method + path; audit-openapi-operation-trace@2, batch size 1 | Целевой аналог trace_annotation: обход OpenAPI, трассировка и находки | Findings с exact evidence уже подключены и требуют human review. Перенос неполон: coverage остаётся operation-resolution; verifier, annotation output и предметная агрегация отдельно. Старый workflow группировал методы path, текущий — операции. |
| [source-checklist@1](../../configs/audit-profiles/source-checklist.yaml) | Пункт переданного JSON/YAML checklist; `audit-source-check@1`, batch size 2 | Прямого аналога нет; контрольные списки были внутри инструкций `trace_agent` и задач | Новый общий механизм исполнения пользовательского checklist. Внутренний checklist старого prompt не эквивалентен внешнему inventory с отдельными результатами и coverage. |
| [owasp-top10-2025-source-risk@1](../../configs/audit-profiles/owasp-top10-2025-source-risk.yaml) | 10 закреплённых mappings — по одному на категорию; `audit-top10-source-risk@1`, batch size 1 | Поиск рисков в `vuln_scan`; тематическое разделение первого прохода `vuln_sweep` | Частичное по назначению, новый способ организации. Проверяются ограниченные сценарии по представительному source; это не полный старый scan/trace pipeline и не доказательство отсутствия рисков во всей категории. Findings остаются предложениями до подтверждения человеком. |
| [owasp-asvs-5-0-l1-source-review@1](../../configs/audit-profiles/owasp-asvs-5.0-l1-source-review.yaml) | 5 выбранных требований ASVS 5.0.0 L1; `audit-asvs-source-verification@1`, batch size 1 | Прямого аналога проверки версионированных ASVS requirements в старом каталоге нет | Новый source/documentation pilot. Это именно пять требований, а не весь Level 1. Findings и решения о неприменимости требуют человека. |

У ASVS имя selector содержит `5-0`, хотя имя файла содержит `5.0`. В пилот входят
`v5.0.0-1.2.4`, `v5.0.0-1.2.5`, `v5.0.0-1.3.2`, `v5.0.0-1.5.1`,
`v5.0.0-2.1.1`. Число требований и mappings проверено по
[пакету ASVS](../../configs/audit-standards/owasp-asvs-5.0.0/standard.json);
10 mappings Top 10 — по
[пакету Top 10](../../configs/audit-standards/owasp-web-top10-2025/standard.json).

Все пять текущих профилей запрещают active checks и используют один раунд
`fixed-barrier`. У OpenAPI @2/checklist создание findings отключено.
У OpenAPI @3, Top 10 и ASVS подтверждение finding человеком — отдельное решение; оно не
заменяет автоматический второй проход `trace_verifier_agent` из старой реализации.

Ограничение OpenAPI audit следует непосредственно из
[генератора inventory](../../internal/auditdomain/openapi.go),
[инструкции Worker](../../configs/instructions/audit-openapi-operation-tracer-worker.md) и
[его allowlist tools](../../configs/agent-templates/audit_openapi_operation_tracer.yaml).
В профиле @2 граф подключён к тому же закреплённому source, который проверяет
Audit. В @1 использовался общий `audit-source-check@1` без графа.
Профиль @3 уже добавляет findings; полный предметный trace coverage остаётся доработкой.
Если выбран вариант с аннотациями, ему также нужны annotation tools и экспорт
артефактов; наличие этих возможностей в отдельном
`taint-trace-from-workspace@2` само по себе не добавляет их Audit-профилю.

Для завершения переноса `trace_annotation` нужно согласовать и реализовать
следующий контракт профиля:

1. Для каждой операции из закреплённого OpenAPI проследить entrypoint, входные
   данные, преобразования, проверки/контроли и sinks либо конечную бизнес-операцию.
   Неразрешённые участки должны оставаться явными gaps.
2. Сохранить предусмотренные Workflow артефакты исследования. Для варианта с
   аннотациями это могут быть аннотированный source и workspace state/diff;
   для другого — отчёт, граф или результаты проверок. Отдельный индекс нужен
   только потребителю, которому требуется структурированное чтение аннотаций.
3. Сохранить реализованную в @3 публикацию findings, связанных с операцией и
   exact evidence. Для дальнейшего объединения находок определить предметные
   правила с сохранением происхождения; совпадение функции не означает дубль.
4. Определить evidence/coverage contract профиля и проверить достаточность общих
   ссылок на результаты, artifacts и proposals. Если расширение общего
   task/result необходимо, оно должно переносить ссылки и происхождение,
   не поля `annotations`, `sink` или `operation` специально для trace.
   Техническую часть собирать детерминированно из доверенных inputs,
   принятых записей и точных artifact/proposal receipts.
   Graph Worker уже использует task package / execution manifest и
   `submit_check_result`. Модель должна задавать предметную оценку и объяснение;
   Runtime/Server — проверять ссылки и контракт, связывать результаты и
   публиковать артефакты. Простая замена Workflow ref на
   `taint-trace-from-workspace@2` недостаточна: у него другие входы и выходы.

**Общие понятия и граница сценария.** Ни Finding, ни гипотеза не ограничены
OpenAPI, трассировкой или конкретным методом проверки. Гипотеза формулирует
проверяемое предположение; Finding фиксирует значимый вывод с обоснованием и
состоянием оценки. Связь между ними полезна, но обязательного жизненного цикла
«гипотеза → проверка → Finding» нет. Это смысловое различие не требует сейчас
двух новых API: существующий `FindingProposal` уже содержит необязательную
гипотезу и предложения проверок.

Общий план tools теперь выделен отдельно:
[`finding` для создания, `list_findings` для анализа](2026-09-06-findings-tools-plan.md),
задачи V43-001–005 завершены. Ниже описаны предметные требования
OpenAPI-сценария, реализованное подключение и оставшиеся расширения.

| Уровень | Ответственность |
| --- | --- |
| Общие механизмы | Идентичность и происхождение задания/исполнения, subject, текст и состояние оценки, exact artifact/evidence refs, receipts, явные связи с результатами. Проверка ссылок, прав, версий и идемпотентности. |
| Workflow / AuditProfile | Способ формирования заданий, цель анализа, tools, обязательные outputs/evidence, допустимые assessments, coverage и политика review. OpenAPI operations — один из вариантов inventory. |
| Предметный tool или обработчик формата | Построение и проверка содержимого выбранного артефакта: аннотаций, графа, отчёта, лога проверки. Формат версионируется независимо; предметный parser не встраивается в общий Audit collector. |

Артефакт становится evidence для конкретного вывода через явную ссылку и
правила выбранного evidence contract. Само наличие файла, в том числе
`annotations.json`, ничего не подтверждает. Workflow может выпускать полезные
артефакты без единой гипотезы или находки; Finding может ссылаться на отчёт или
лог без аннотаций. Общий механизм не требует знания их внутреннего формата.
Проверка содержимого применяется там, где её предусматривает выбранный контракт.

Это архитектурное ограничение дальнейших доработок, а не объявление нового
универсального протокола. Сначала переиспользуются существующие Workflow
outputs, ArtifactRef, finding intake и result contracts. Расширения вводятся
по конкретному отсутствующему механизму; новый глобальный registry типов,
универсальный граф сущностей или обязательный `trace_id` не нужны заранее.

**Граница детерминированной сборки и анализа.** Отдельный toolset для находок
уже реализован:
[`security-findings@2`](../../runtime/src/contractor_runtime/toolsets/security_findings/tools.py)
с независимо выбираемыми `finding` и `list_findings`. `finding` принимает
кандидат и точные evidence refs,
регистрирует предложение и возвращает `proposal_id` / `receipt_id`.
Сервер выводит происхождение из allocation/Run, закрепляет evidence и
обрабатывает повтор идентичной отправки. Этот toolset уже выбран у Top 10 и
ASVS в версии @1; OpenAPI @3 выбирает finding из @2 и human-required review.
Ограниченный профиль @2 оставляет
`findingConfirmation: disabled`, поэтому простое добавление toolset сделает
его несовместимым с Server.

**Реализовано: прямые findings при обходе операций.** Единицей
работы остаётся операция из `openapi-operations@1`. Worker исследует её код и
сообщает от нуля до нескольких обоснованных находок в том же Run. В текущем
`finding` поля `hypothesis` и `proposed_checks` уже необязательны; отдельный
цикл гипотез, новый inventory и verifier для этого не нужны. Статус candidate
в intake означает ожидание оценки находки, а не обязательную гипотезную стадию.
Ноль находок сам по себе не доказывает безопасность или полноту трассировки.

Из старого
[`report_vulnerability`](../../../contractor-old/contractor/tools/vuln.py)
стоит перенести требования к содержанию `details`: техническое объяснение,
воспроизведение, impact и при необходимости исправление. В текущем контракте
это поле `description`. Старые mutable upsert/get/list не следует копировать
целиком: существующий intake уже даёт точное происхождение, evidence pins и
идемпотентные receipts. Отдельный toolset с дублирующим `report_vulnerability`
для этого сценария не требуется.

Предлагаемая структура `description` **в инструкции OpenAPI security review**.
Она не является обязательной схемой общего `finding`:

1. **Проблема:** краткое утверждение и затронутая операция; путь к исходнику,
   функция и место дефекта.
2. **Обоснование по коду:** достижимая цепочка до чувствительной операции,
   конкретный слабый/отсутствующий control, source evidence. Проверки в
   middleware, вызывающем коде и framework учитываются явно; отсутствие
   локального decorator само по себе не доказывает отсутствие контроля.
3. **Условия и воспроизведение:** необходимые права/состояние, последовательность
   действий или шаблон запроса, ожидаемый признак проблемы. Непроверенные
   предпосылки и недостающие детали отмечаются, а не подменяются выдуманными.
4. **Влияние:** возможный результат и границы вывода.
5. **Что проверено:** исследованные участки кода и, если есть, фактически
   выполненные проверки. Для текущего source-only сценария способ
   воспроизведения описывается как выведенный из кода, без заявления о запуске.

Недоступность живого воспроизведения не запрещает обоснованную находку по коду.
Неопределённая достижимость или неизвестный control остаются gaps; обычное
попадание данных в sink без установленного дефекта не является находкой.
`severity_suggestion` и `standard_refs` остаются отдельными доступными полями.
Если UI/eval потребуется разбирать шаги, можно определить отдельный формат
артефакта воспроизведения для использующего его сценария. В общий Finding
не добавляются обязательные поля запроса, sink, CWE или аннотаций.

Подключение OpenAPI выполнено в V43-004. Профиль @3 использует шаблон @2,
`security-findings@2.finding` и `text-artifacts@1.write_text_artifact`.
Инструкция требует сначала публиковать evidence и получить exact refs,
затем передавать `client_key` находки через `proposal_keys` в `submit_check_result`.
Сохранённое моделью описание не объявляется автоматически извлечённым source-фрагментом.
`findingConfirmation: human-required`, `activeChecks: prohibited` и operation inventory
сохранены.

Общий `findings-review@1` читает самодостаточную ZIP-подборку, которую Server
публикует по receipts, а Runtime материализует в RunScope.
[V43-005 gate](2026-09-06-findings-tools-validation.md) проверил передачу evidence,
replay, paging и обычные Run/Audit сценарии через scripted models.
Предметные quality cases — безопасный sink, control во внешнем middleware,
общая функция у операций с разными controls — требуют отдельного model eval.
[Portable format уже готов](portable-eval-format-readiness.md); следующий шаг —
fixture mapping, bindings и frozen plan в V40.

Полный trace coverage определяется предметным контрактом профиля. Индекс
аннотаций — возможное расширение отдельного Workflow, когда нужен его разбор.
Текущий `operation-resolution` не становится полным trace coverage от добавления
findings. Независимый verifier остаётся полезной опцией для дополнительной
проверки. При будущей агрегации сохраняются все затронутые операции и evidence:
совпадение функции или `file + CWE` не даёт достаточного основания сливать
находки. Не требуется добавлять общий mutable список находок в каждый Worker
ради первого подключения.

| Часть результата | Что собирает/проверяет код | Что остаётся предметной оценкой |
| --- | --- | --- |
| Назначение и происхождение | Item/operation, source revision, manifest digest, принадлежность evidence и proposal текущему исполнению | Соответствует ли найденный handler операции по смыслу |
| Аннотации | Успешные структурированные записи `taint-annotations`, exact source/overlay snapshot, экспорт state/diff; replay и конфликты | Верно ли выбраны source, transformation, control и sink; достоверна ли сама аннотация |
| Evidence | Извлечение заданного фрагмента из закреплённого source/snapshot, допустимость пути/диапазона, hash, exact artifact ref | Подтверждает ли фрагмент сформулированный вывод |
| Findings | Валидация структуры, запись через `security-findings`, evidence pins, receipts и привязка к item | Наличие уязвимости, причинная цепочка, влияние, предлагаемая severity |
| Coverage | Фиксированный набор обязательств из task, наличие нужных записей/evidence, явные gaps, правила принятия | Достаточно ли исследованы пути; эффективен ли control. Число вызовов tools или аннотаций само по себе не даёт `completed` |
| Итоговый пакет | Каноническая сериализация принятых записей, ссылки на artifacts/receipts, проверка полноты и публикация | Summary и assessment, которые модель явно представила с evidence |

Для source-сценариев полезен захват evidence с возвратом точной ссылки. Общий
collector должен принимать результат такого инструмента через обычный
artifact/evidence contract, без встроенного знания о source или аннотациях.
Сейчас [`submit_check_result`](../../runtime/src/contractor_runtime/toolsets/audit_results/v1.py)
создаёт evidence members из переданных текстовых summaries. Нельзя считать
такой summary автоматически извлечённым фрагментом исходников. При захвате после
аннотирования ссылка должна учитывать snapshot overlay, поскольку строки могли
сместиться. Для финальной сборки нужны принятые записи с определённым lifecycle;
восстанавливать их из текста диалога или диагностических metrics нельзя.

Если потребителю нужны reused-аннотации и связи общих функций с несколькими
операциями, annotation tools могут экспортировать собственный индекс,
сверенный с итоговым workspace. Его контекст и внутренние IDs остаются частью
формата этого артефакта. Для простого экспорта изменений достаточно state/diff;
достоверная атрибуция участия требует явных связей. Варианты описаны в
[предложении об аннотациях как артефакте](2026-09-06-trace-annotation-contract.md).

Детерминированное завершение Audit Worker уже запланировано в
[spec 25 / V39](../spec/25-audit-worker-finalization.md): collector → seal →
канонический ZIP → Runtime-authored WorkerResult. V39-001/004 уже реализуют
неактивные контракты и детерминированный encoder/publisher. Collector, pinning,
общий completion gate и rollout ещё не завершены. Если для артефактов сценария потребуется расширение завершения,
его следует делать поверх этого механизма, согласовав общие artifact/evidence
refs и finding receipts, без второго независимого finalizer и обязательных
annotation-полей в общем result.
Одинаковые закреплённые inputs и принятые записи должны давать одинаковый payload;
это не обещает одинаковых выводов модели, storage revisions или одинаковых
receipts в разных запусках. Автоматический replay/dedup одинаковой отправки также
не заменяет предметное объединение похожих уязвимостей.

При этом Audit отвечает за весь сценарий `trace_annotation`, а дочерний trace
Workflow — за назначенную операцию или согласованную группу операций. Выбор
группировки и planner не должен подменять предметный результат. Это зафиксированное
целевое соответствие; общий findings path реализован, остальные расширения
не объявляются готовыми только из-за подключения tools.

При обратном сравнении видны следующие **старые сценарии без готового полного
эквивалента в текущем каталоге**:

| Сценарий contractor-old | Чего не хватает для эквивалентности |
| --- | --- |
| [trace_annotation](../../../contractor-old/contractor/workflows/trace_annotation/workflow.py), [trace_annotation_direct](../../../contractor-old/contractor/workflows/trace_annotation_direct/workflow.py), [trace_graph](../../../contractor-old/contractor/workflows/trace_graph/workflow.py), [trace_graph_pathpar](../../../contractor-old/contractor/workflows/trace_graph_pathpar/workflow.py) целиком | Целевой преемник `trace_annotation` — Audit `openapi-operation-trace`; его перенос пока неполон. Graph Worker и структурированные findings уже связаны с inventory. Остаются полный trace coverage, предметная агрегация и, для annotation-варианта, экспорт/анализ аннотаций. В `pathpar` дополнительно были forks overlay по paths и последующее слияние; обычный dispatch Audit items это поведение не воспроизводит. |
| [trace_postdiff](../../../contractor-old/contractor/workflows/trace_postdiff/workflow.py) | Отдельная цепочка «аннотирование без vulnerability reporting → `vuln_analytics_agent` анализирует diff и публикует findings». |
| [trace_verify](../../../contractor-old/contractor/workflows/trace_verify/workflow.py) | Независимый статический второй проход по каждому finding с отдельным структурированным verdict. |
| [vuln_scan](../../../contractor-old/contractor/workflows/vuln_scan/workflow.py) | Самостоятельный обзор всего source: dangerous patterns + missing controls, `codereview_agent`, структурированный набор findings. Top 10 audit имеет другой объём и единицу проверки. |
| [vuln_sweep](../../../contractor-old/contractor/workflows/vuln_sweep/workflow.py) | Параллельные nominations по пяти классам → merge/dedup/cap → глубокая трассировка каждой оставшейся находки. |
| [vuln_scan_trace](../../../contractor-old/contractor/workflows/vuln_scan_trace/workflow.py), [vuln_scan_fast](../../../contractor-old/contractor/workflows/vuln_scan_fast/workflow.py) | Готовая связка широкого scan и trace по findings; у fast дополнительно discovery, dedup `(file, CWE)` и необязательный exploit. |
| [exploitability](../../../contractor-old/contractor/workflows/exploitability/workflow.py) целиком | Цикл по переданным findings, task `exploitability_assessment/v4`, структурированные verification results и автоматический сбор HTTP proof chains. Отдельный HTTP/Caido Worker покрывает лишь часть этого сценария. |
| [vuln_assess](../../../contractor-old/contractor/workflows/vuln_assess/workflow.py) | Полная композиция OpenAPI build → trace paths → объединение findings → exploitability. Наличие отдельных текущих Workflow ещё не образует такой pipeline. |
| [router](../../../contractor-old/contractor/workflows/router/workflow.py) | Готовый workflow свободного запроса с planner и маршрутизацией к SWE/OAS/trace/HTTP специалистам. `streamline` внутри фиксированной LikeC4-цепочки имеет более узкую задачу. |

Для дальнейших eval это задаёт границы сравнения: OpenAPI/LikeC4 можно сравнивать
по одинаковым входам и выходным документам, согласовав seed и discovery reports.
`taint-trace-from-workspace` нужно сравнивать с одной такой же операцией/целью
старого trace, а `security-analysis` — с сопоставимой ограниченной HTTP-задачей.
Покрытие `operation-resolution` нельзя выдавать за покрытие taint analysis.
Целевой `openapi-operation-trace` следует сравнивать со всем `trace_annotation`
на одинаковых source/OpenAPI: оценивать пути данных, controls/sinks, корректность
аннотаций, findings и явные пробелы по всем операциям. Одного успешного поиска
handlers для такого eval недостаточно.
Новым Audit-контрактам нужны собственные проверки inventory, assessments,
coverage/gaps и evidence, даже если source fixtures переиспользуются из старых
eval. Эта таблица не оценивает качество модели; новые запуски eval не выполнялись.

Продуктовое назначение и рекомендация по prod: [обзор каталога](2026-09-06-production-scenario-catalog.md).
