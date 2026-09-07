# Описания tools: проверка и перенос из contractor-old

Проверено 2026-09-06 по текущему рабочему дереву и соседнему репозиторию
`/home/ruslan/src/contractor-old`.

## Что было в текущей реализации

У всех **89 callable tools** уже были непустые описания: конструкторы присваивают
`self.__doc__ = self.description`, а
[`worker/runtime.py`](../../runtime/src/contractor_runtime/worker/runtime.py) передаёт
экземпляры в `FunctionTool`. Описания содержали от одной до нескольких фраз,
но ни одно не имело разделов `Args:` и `Returns:`.

Ещё **3 Agent Skill tools** (`list_skills`, `load_skill`, `load_skill_resource`)
создают ADK declarations явно. У них уже были описания функций и параметров.

В закреплённом `google-adk==2.8.0` полный `__doc__` попадает в описание функции.
Реализация построения схемы в этой версии не извлекает `Args:` в отдельные
описания параметров. Поэтому пояснения аргументов оставлены в полном тексте
описания. Общий принцип использования docstring описан в
[документации ADK](https://adk.dev/tools/function-tools/); поведение конкретной
версии дополнительно проверено по установленному коду и созданным declarations.

## Единый формат

Все 89 описаний и 3 явных описания Agent Skill tools обновлены. Формат закреплён
в [`runtime/README.md`](../../runtime/README.md):

1. Английский язык и короткое предложение с глаголом действия: `Read`, `List`,
   `Find`, `Create`, `Replace`, `Submit`.
2. Краткое пояснение порядка вызовов, ограничений или побочных эффектов там,
   где это влияет на использование.
3. `Args:` с точными именами всех аргументов модели. Для каждого указаны смысл,
   а при необходимости — значения по умолчанию, единицы, допустимые значения,
   область пути, правила ревизий и продолжения страниц.
4. `Returns:` с фактическими полями ответа и признаками неполного результата.
5. У tools без аргументов модели раздел `Args:` отсутствует.
   Внедряемый ADK `tool_context` в описаниях не показывается.

У callable tools единственным источником текста остаётся `description`,
из которого заполняется `__doc__`. Сигнатуры, значения по умолчанию и логика
вызовов сохранены.

## Список переноса описаний

Для **51 текущего имени** в старом коде найден одноимённый callable с docstring.
Дополнительно использованы описания переименованных и близких по назначению
операций. Ниже перечислены все текущие группы и результат адаптации.

| Текущая группа | Tools | Источник и перенесённые пояснения |
| --- | --- | --- |
| [Filesystem](../../runtime/src/contractor_runtime/toolsets/filesystem/tools.py), 4 | `ls`, `glob`, `read_file`, `grep` | `fs/read_tools.py`, `fs/write_tools.py`: область поиска, примеры glob, чтение перед редактированием, отделение номеров строк от содержимого. Добавлены текущие курсоры, лимиты и режим буквального поиска. |
| [Edit files](../../runtime/src/contractor_runtime/toolsets/edit_files/tools.py), 9 | `write_file`, `append_file`, `mkdir`, `rm`, `cp`, `mv`, `insert_line`, `edit`, `replace_range` | `fs/write_tools.py`: точное совпадение текста, уникальность фрагмента, `replace_all`, включительные границы строк, необходимость перечитать файл. Описаны текущая обработка переводов строк и ответ `changed`. |
| [Workspace changes](../../runtime/src/contractor_runtime/toolsets/workspace_changes/tools.py), 3 | `changed_paths`, `diff`, `rollback_changes` | `fs/write_tools.py`: `changed_paths`, `diff`, `restore`. Адаптировано к текущему checkpoint, курсорам и ограниченному diff. |
| [Memory](../../runtime/src/contractor_runtime/toolsets/memory/tools.py), 6 | `list_memories`, `read_memory`, `write_memory`, `append_memory`, `search_memory`, `list_memory_tags` | `memory.py`, включая `list_tags`: поиск имени перед чтением, предотвращение дубликатов, замена метаданных при записи, сохранение метаданных при append, OR-поиск по тегам. Исправлен порядок выдачи и описаны нынешние ограничения имён и тегов. |
| [HTTP](../../runtime/src/contractor_runtime/toolsets/http/tools.py), 6 | `http_request`, `http_read_body`, `http_history`, `http_session_set`, `http_session_get`, `http_session_clear` | `http.py`: формы `body_type` и `auth`, слияние заголовков, cookies, чтение тела по request ID, различие символьных и байтовых смещений. Лимиты и поля ответа сверены с текущим кодом. |
| [Caido](../../runtime/src/contractor_runtime/toolsets/caido/tools.py), 10 | `caido_scope`, `caido_history`, `caido_request_detail`, `caido_replay`, `caido_automate_run`, `caido_automate_results`, `caido_sitemap`, `caido_workflow_list`, `caido_workflow_run`, `caido_workflow_findings` | `caido.py`: примеры HTTPQL, выбор ID или raw request, порядок detail → automate → results, первое вхождение target, допустимые сортировки и виды workflow. Уточнены превью, артефакты и статусы текущей реализации. |
| [Code analysis](../../runtime/src/contractor_runtime/toolsets/code_analysis/tools.py), 11 | `search_def`, `list_symbols`, `graph_summary`, `find_symbol`, `find_callers`, `find_callees`, `paths_between`, `entrypoint_paths_to`, `attack_surface`, `complexity_hotspots`, `functions_that_raise` | `code/tools.py`, `code/graph.py`: назначение структурного поиска, графовых запросов и оценки покрытия. Аргументы адаптированы к точным symbol IDs, курсорам и нынешним ограничениям обхода. |
| [Taint annotations](../../runtime/src/contractor_runtime/toolsets/taint_annotations/tools.py), 3 | `annotate_trace`, `annotate_validate`, `annotate_sink` | `code/annotations.py`: когда выбирать каждый вид аннотации, состояния аргументов, категории sink/validation, автоматический выбор комментария. Добавлен `definition_line` для разрешения неоднозначности. |
| [LikeC4](../../runtime/src/contractor_runtime/toolsets/likec4/tools.py), 6 | `load_likec4`, `write_likec4`, `read_likec4`, `append_likec4`, `replace_likec4`, `validate_likec4` | `likec4.py` содержит прежний `validate_likec4`. Перенесено назначение проверки; остальные описания написаны для нынешней сессии документа и ревизий артефактов. |
| [OpenAPI](../../runtime/src/contractor_runtime/toolsets/openapi/tools.py), 18 | `load_openapi`, `initialize_openapi`, `get_openapi_info`, `set_openapi_info`, `list_openapi_servers`, `set_openapi_servers`, `list_openapi_tags`, `set_openapi_tags`, `list_openapi_paths`, `get_openapi_path`, `upsert_openapi_path`, `remove_openapi_path`, `list_openapi_components`, `get_openapi_component`, `upsert_openapi_component`, `remove_openapi_component`, `read_openapi_document`, `validate_openapi` | `openapi/openapi.py`, `openapi/vacuum.py`: структурные объекты вместо JSON-строк, source evidence, merge, сохранение отсутствующих полей info, выбор целевого чтения вместо полного документа. Добавлены текущие sections, ревизии и поведение validators. |
| [Run artifacts](../../runtime/src/contractor_runtime/toolsets/run_artifacts/tools.py), 3 | `list_artifacts`, `read_artifact`, `write_artifact` | Близкие операции `pool_list`/`pool_read` в `artifact_pool.py`. Описания написаны для нынешних namespace/name/revision, base64 и защиты обновлений ожидаемой ревизией. |
| [Text artifacts](../../runtime/src/contractor_runtime/toolsets/text_artifacts/tools.py), 2 | `read_text_artifact`, `write_text_artifact` | `artifact_pool.py` даёт подсказку об ограниченном чтении. Текущие строки, UTF-8, namespace записи и ревизии описаны по новой реализации. |
| [Source archive](../../runtime/src/contractor_runtime/toolsets/source_analysis/tools.py), 4 | `open_source_archive`, `list_source_files`, `search_source`, `read_source` | Отдельная новая поверхность. Описаны обязательное открытие точной ZIP-ревизии, область путей внутри архива, pagination, режимы поиска и ограничения чтения. |
| [Code execution](../../runtime/src/contractor_runtime/toolsets/code_execution/tools.py), 1 | `exec_command` | Близкий `execute_bash` в `podman.py`: назначение shell-вызова, timeout и вывод. Описание составлено для нынешнего allocation container, относительного cwd и прекращения allocation при timeout/overflow. |
| [Security findings](../../runtime/src/contractor_runtime/toolsets/security_findings/tools.py), 1 | `finding` | Близкий `report_vulnerability` в `vuln.py`: стабильный ключ, описание наблюдений и влияния. Описаны текущие proposal/receipt, exact evidence refs и формы вложенных объектов. |
| [Audit results](../../runtime/src/contractor_runtime/toolsets/audit_results/v1.py), 2 | `read_audit_task`, `submit_check_result` | Отдельный новый контракт. Старые `report_verification`/`submit_verdict` имеют другую модель результата. Описаны нынешние assessments, coverage IDs, evidence, proposal keys и взаимоисключающие одиночная и пакетная формы. |

Для [Agent Skill tools](../../runtime/src/contractor_runtime/agent_skills/runtime.py)
уточнены последовательность list → load → resource, точные имена, пути
`references/...`/`assets/...` и формат ответа. Описания параметров в явной
JSON Schema согласованы с общим текстом.

## Что изменилось относительно старых подсказок

- `read_file`: прежние `file`, `offset`, `limit`, `with_line_numbers` заменены
  текущими `path`, `start_line`, `max_lines`; первая строка теперь задаётся как 1.
- `grep`: по умолчанию буквальный поиск, regex включается явно. У `search_source`
  отдельно сохранён его собственный default `case_sensitive=False`.
- `edit`: аргументы теперь `old`/`new`; пустой `old` отклоняется. Старое обещание
  создания файла через пустой шаблон не перенесено.
- `replace_range`: прежнего `preserve_trailing_newline` в сигнатуре нет.
- `rollback_changes`: описан текущий checkpoint и выбор всего workspace пустым
  путём; прежние `restore(path, recursive)` и overlay/base-семантика заменены.
- Memory: результаты сейчас упорядочены по времени обновления, последние первыми.
  Ответы содержат поля заметки и timestamps, а ревизии остаются внутри реализации.
  Используются lowercase snake_case имена и до трёх lowercase тегов.
- Code analysis: `search_def` возвращает структурные определения; обещание
  старого grep fallback удалено. Caller/callee/path tools требуют точные IDs,
  полученные из `find_symbol`. Пустая выдача трактуется с учётом coverage и лимитов.
- Caido: raw request/response выдаются ограниченными превью с артефактами.
  `caido_scope` уже поддерживает create; устаревшая фраза о будущей доступности
  создания удалена. Replay возвращает `started` без ожидания и `timeout` при
  истечении polling deadline.
- OpenAPI: текущие имена отличаются от старых (`upsert_path` →
  `upsert_openapi_path`, `get_full_openapi_schema` → `read_openapi_document`,
  `lint_openapi` → `validate_openapi`). Upsert возвращает metadata и `changed`;
  прежнее обещание diff не перенесено. Список component sections расширен.
- LikeC4: `validate_likec4` проверяет текущий документ сессии; прежнего аргумента
  `path` нет. Описаны предварительные load/write и ошибки запуска validator.
- Execution: прежние обещания Kali, набора пакетов, host network, `/project`
  read-only и автоматического сохранения файлов как артефактов не перенесены.
- Findings/Audit: старые severity `info`, confidence и verdict-поля заменены
  существующими контрактами; `finding` создаёт предложение, `submit_check_result`
  публикует результат назначенной проверки.

## Проверка

Суммарный размер описаний 89 callable tools вырос примерно с 6,8 до 41,5 тыс.
символов, то есть в 6,1 раза. Конкретный Worker получает только выбранные tools.
Ожидаемая польза — меньше неоднозначностей в аргументах и последовательности
вызовов; её влияние на качество ещё не измерено сравнительным прогоном задач.

Для всех 89 callable tools созданы реальные экземпляры с тестовыми зависимостями
и построены ADK declarations до и после правки. Проверено, что новый полный
текст попал в `description`, `Args:` соответствует аргументам модели, а остальные
части declarations совпали: имена, схемы параметров и ответов, обязательные
поля и defaults. Проверки не вызывали сами tools и не отправляли запросы во
внешние системы.

Отдельно проверены 3 Agent Skill declarations: описания доходят до ADK,
перечисленные в `Args:` параметры совпадают с явной схемой. Сравнение AST всех
16 модулей toolsets с рабочим деревом перед правкой подтвердило, что изменения
затронули только присваивания `description`. Ruff и `git diff --check` прошли.

**314 тестов прошли**: проверены ADK runtime, все затронутые группы tools и
Agent Skills с `-W error`. Запуск выполнен вне песочницы: внутри неё зависало
завершение потоков `asyncio`, что воспроизвелось также на минимальном примере
без кода проекта.
