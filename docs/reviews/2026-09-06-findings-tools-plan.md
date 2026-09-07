# Findings: общие tools создания и чтения

Статус: V43-001–005 завершены. `security-findings@2` предоставляет независимо
выбираемые `finding` и `list_findings`; сервер публикует самодостаточные ZIP
подборки, Runtime материализует exact документы в текущем Run. Подключены
версионный OpenAPI trace producer и общий findings reader.
[Спецификация 27](../spec/27-findings-tools-and-collections.md) закрепляет контракт;
[validation review](2026-09-06-findings-tools-validation.md) фиксирует 13 process
cases, 48 Python-тестов и сценарии будущих eval. Модельные eval ещё не запускались.

## Решение

Один toolset с независимо выбираемыми операциями:

| Роль агента | Tools | Результат |
| --- | --- | --- |
| Обнаружение | `finding(...)` и предметные tools | Предложение находки с evidence и receipt |
| Анализ находок | `list_findings(...)`, `read_artifact(...)` | Анализ явно переданной подборки; подробности доступны по exact refs |
| Совмещённая роль | Обе операции, если нужны Workflow | Анализ известных находок и создание новых с сохранением происхождения |

Переиспользуем `security-findings`, добавляя чтение в новой версии toolset.
AgentTemplate выбирает конкретные tools. Чтение не означает право создания
или подтверждения находок. Отдельные `report_openapi_finding`, новый registry
сущностей и переименование всей модели данных для этой доработки не нужны.

Finding фиксирует значимый вывод с обоснованием; гипотеза — проверяемое
предположение. Текущий `FindingProposal` уже допускает необязательную гипотезу
и предлагаемые проверки. Обязательного цикла «гипотеза → проверка → Finding»
нет. Сохранение предложения и его итоговая оценка — разные действия.

## Контракт tools

`finding` сохраняет текущие аргументы: `client_key`, `title`, `description`,
`subject`, `evidence_refs`; необязательные `hypothesis`, `proposed_checks`,
`standard_refs`, `severity_suggestion`. OpenAPI operation, функция, CWE, sink,
воспроизведение и аннотации не становятся обязательными полями общего Finding.
Повтор идентичной отправки возвращает receipt; изменённый payload с тем же
ключом не становится неявным update.

Интерфейс чтения, закреплённый в V43-001:

```text
list_findings(subject_kind?, subject_key?, limit?, cursor?)
    -> items, next_cursor
```

Для первого reader источник — самодостаточный ZIP с manifest, документами
proposals и exact bytes evidence, переданный обычным входным артефактом Workflow.
Сервер готовит его из доступных
receipts исходных Runs или Audit. Reader работает с одним форматом независимо
от происхождения подборки. Подготовка списка не создаёт verification inventory
и не назначает checks; для новых поступлений выпускается новая подборка.

Страница возвращает ограниченные previews: proposal/receipt identity, title,
subject, краткое описание, наличие гипотезы, exact refs и происхождение.
Если есть решение по находке, сохраняются его состояние и точная ревизия.
У proposal обычного Run не выдумывается Audit finding ID. Полный документ и
evidence читаются через `read_artifact`; отдельный `get_finding` пока не нужен.
Cursor привязан к подборке и фильтрам. Ограничиваются число записей и байты;
частичная страница не выдаётся за полный список.

Подборка не расширяет права на перечисленные артефакты. ZIP удерживает выбранные
bytes; подготовка reader размещает их в текущем Run и сохраняет exact revisions.
Произвольные owner/project/
Audit IDs не передаются моделью для выхода за пределы заданного списка.
Новый формат не содержит обязательных OpenAPI-, annotation- или check-полей.

В текущей реализации `read_artifact` ограничен RunScope своего исполнения.
Передача JSON со ссылками исходных Runs не делает их доступными читателю.
V43-001 закрепляет отображение документов: namespace зависит от SHA-256 ZIP,
имя — от scoped identity документа. V43-002 публикует ZIP; V43-003 при подготовке
выбранного reader-toolset размещает документы через существующий ArtifactClient
и проверяет exact receipts. Обычный Run-create продолжает передавать один input.
Исходное scoped происхождение сохраняется отдельно. Одинаковые namespace/name
из двух Runs не сталкиваются. Общая политика `ReadCurrentRun` сохраняется.
Это существенная часть передачи подборки, а не новое хранилище findings.

`list_findings` не выполняет скрытый смысловой dedup. Общая функция у разных
операций может иметь разные controls. Предложение аналитика об объединении
сохраняет исходные ссылки; изменение состояния следует review contract.

## План работ

Уже есть [receipts и чтение proposals Run/Audit](../../internal/findingintake/postgres.go),
[список и оценка Audit findings](../../internal/auditservice/finding_review.go),
Runtime [`finding`](../../runtime/src/contractor_runtime/toolsets/security_findings/tools.py)
и чтение artifacts. Новое хранилище findings не требуется.

| Задача | Доработка | Критерий завершения |
| --- | --- | --- |
| [V43-001](../../tasks/v43-001-findings-access-contract.yml) | Закрепить формат подборки и `list_findings` | Обычный Run и Audit используют один контракт без обязательных гипотез и аннотаций |
| [V43-002](../../tasks/v43-002-findings-collection-publication.yml) | Сборка и публикация ZIP через существующие server receipts/review данные | Все выбранные bytes удерживаются внутри ZIP; proposals не теряются молча |
| [V43-003](../../tasks/v43-003-findings-reader-tool.yml) | Подготовка документов в RunScope, tool чтения, docstrings, pagination | Документы читаются по exact refs; reader-only не классифицируется как создание findings |
| [V43-004](../../tasks/v43-004-findings-agent-integration.yml) | Создание в OpenAPI trace и общий Workflow/шаблон анализа подборки | Один агент создаёт находки, другой читает их и выпускает отчёт |
| [V43-005](../../tasks/v43-005-findings-contract-gate.yml) | Сквозная проверка и сценарии eval | Создание → передача → чтение → анализ работают без обязательных аннотаций |

Порядок: контракт → подборка → tool → агенты → сквозная проверка.
V39 finalization не блокирует tools: используются текущие result/receipt
contracts. Модельные eval следуют после portable evaluation format / V41-008.

Аннотации, diff, граф, отчёт или лог — возможные артефакты Workflow. Их
содержимое проверяет потребитель формата, общий механизм хранит ссылки и
происхождение. Annotation index, автоматический dedup и обязательный verifier
не входят в первый этап. Шаги воспроизведения задаются security-инструкцией,
а не общим API; ожидаемый результат отделяется от фактически проверенного.
