# Compaction контекста Worker с выбором в Workflow

Статус: isolated compaction, единый выбор compactor через Workflow
executionConfig, метаданные State/report/UI и offline Eval harness реализованы
и интегрированы в main. Приёмка deterministic milestone остаётся V417-009;
живые повторные сравнения — V417-010, shared-session compaction — последняя
V417-004. Текущие нормативные гарантии задают WFL-R91–R95 и WRK-R184–R197;
остальные проектные разделы ниже сохраняют историю выбора и вопросы измерений.

Исходный анализ выполнен до реализации и изменения нормативных контрактов
на `b0ce0a3fe4d222bae903ac5c77100d90e922ee47`, с установленным
`google-adk==2.8.0`.

План уточнён после анализа ADK 2.13.0: spike V417-002 начинается после
[V418-007](../../tasks/done/v418-007-adk-213-release-gate.yml), который принимает
обновление и его Runtime-контракты. Compaction проверяется на этом точном
lock, а не повторно на исторической 2.8.0. Обновление ADK 2.13.0 выполнено.
V417-001 с terminal `includeReasoning` принята 11 ноября 2026 года независимо
от compaction.

V417-003 (31 октября 2026 года) реализует `compact@1` для isolated обычных
Workers; нормативные правила теперь в
[WFL-R91–WFL-R95](../spec/configuration/workflows.md) и
[WRK-R184–WRK-R195](../spec/execution/worker.md). Приёмка V417-003 завершена
11 ноября 2026 года. В тот же день принята V417-005: isolated Audit compaction
сохраняет trusted completion и receipts по
[AWC-R130](../spec/audits/worker-completion.md#AWC-R130).
Shared sessions (V417-004) остаются предложением.

Решение пользователя от 10 ноября 2026 года: V417-005 допускает compaction
только для Audit Workers с `isolated`. Последующее указание в тот же день
возвращает V417-004 в `in_progress` и ставит её последней в серии V417:
после V417-009, V417-010 и всех их зависимостей. Этот порядок заменяет
предыдущее откладывание до конкретного потребителя. Статус задан пользователем;
реализация shared ждёт зависимостей. V417-005 и deterministic exit V417-009
от неё не зависят. Завершённая приёмка V417-003 остаётся prerequisite.
Реализация и приёмка isolated Audit compaction зафиксированы в
[V417-005 evidence](../../tasks/evidence/v417-005-acceptance-20261111.json);
это не приёмка всего milestone или доказательство live-качества.

Результат integration spike от 26 октября 2026 года:
[V417-002 evidence](../../tasks/evidence/v417-002-adk-compaction-spike.md).
Выбран публичный request-projection callback по точным IDs/ordinals: native
включительный timestamp range поглощает оставляемых соседей с тем же временем.
Real Runner/Gateway proof покрывает два checkpoint, сохранение одного логического
владельца и fail-closed continuation. Это test-only proof, не включение `compact@1`.

Предлагается добавить продолжающую стратегию `compact@1`, которую автор
Workflow выбирает для конкретного Worker binding. При давлении на контекст
Runtime сворачивает старую историю и продолжает ту же подзадачу с теми же
инструментами, бюджетом и deadline. Управление возвращается Planner после
обычного завершения или ошибки, а не после каждого сжатия.

Начальный алгоритм — защищённое исходное задание, ограниченная рабочая
память, недавние полные взаимодействия и возможность перечитать исходную
историю. Включение reasoning во вход summarizer/compactor задаётся явно.
Исследования дают основания проверить этот вариант, но не доказывают
оптимальность его параметров для моделей и задач Contractor.

## 1. Текущее поведение и границы изменения

Сейчас [AgentTemplate](../spec/configuration/agent-templates.md) может
содержать `summarizer` с фиксированной семантикой `terminal@1`.
[Worker](../spec/execution/worker.md), WRK-R131–R152, проверяет мягкую
границу после ответа модели и завершения выбранных ею инструментов.
При достижении границы Runtime завершает обычный Runner, делает один
tool-free вызов и возвращает `WorkerResult(summarized=true)`.

Дальнейшая передача управления зависит от Planner и Workflow. В частности,
[WFL-R86](../spec/configuration/workflows.md) и
[Scheduler](../spec/execution/scheduler.md) допускают явно настроенное
`on.summarized.escalate` из `passthrough@1` в `streamline@1`. Сам summarizer
не выбирает Planner и не запускает streamline.

| Событие | Terminal summarizer | Предлагаемый compaction |
| --- | --- | --- |
| Давление на контекст | Завершает вызов Worker | Меняет активное представление истории |
| Результат модели | Текст для конечного WorkerResult | Рабочая память для следующего шага |
| Следующий участник | Planner, затем Workflow policy | Тот же Worker с теми же инструментами |
| `summarized` | `true` в доверенном результате | Не выставляется из-за compaction |
| Идентичность подзадачи | Вызов завершается | Сохраняются allocation, A2A task, invocation и subtask ID |
| Лимиты | Отдельная существующая summarizer policy | Общий бюджет вызова плюс ограниченная compactor policy |

Существующие WRK-R137 и WRK-R146 описывают отсутствие оценочного триггера
и исключение thought из terminal summary. Это пересматриваемые проектные
решения, а не основания запретить новые варианты. TPL-R48 уже требует
явного discriminator для стратегии с продолжением; `compact@1` вводит его.

## 2. Выбор в Workflow и конфигурация reasoning

### 2.1 Выбор стратегии

Добавить `contextManagement` в object-valued binding
`spec.stages.<stage>.agents.<agent>`. Это часть Workflow, а не изменение
текста AgentTemplate и не решение Planner. Один шаблон можно использовать
с terminal summarization в одном Workflow и с compaction в другом.

Фрагмент предлагаемого authoring API:

```yaml
spec:
  executionConfig:
    workers:
      compactorModelPolicy: compactor@1
  stages:
    review:
      planner: passthrough@1
      session: isolated
      agents:
        analyst:
          template: workspace_source_graph_analyst@3
          namespace: analysis
          contextManagement:
            strategy: compact@1
            instructions:
              ref: instructions/worker-compaction.md
            includeReasoning: true
            contextWindowRatio: 0.75
            keepRecentGroups: 4
            maxCompactions: 8
```

`compactor@1` и instruction file — явные версионированные ресурсы каталога.
Числа иллюстрируют конфигурацию; оптимальные значения ещё не измерены.
Полный Workflow сохраняет обычные context, result и transition contracts.

Authoring принимает закрытый union:

- `strategy: template@1`: использовать явно выбранное поведение шаблона —
  его terminal summarizer либо существующие hard budgets без summarizer;
- `strategy: compact@1`: использовать закрытую compaction-конфигурацию выше.

Отсутствие authoring-поля разрешается в `template@1`. В immutable resolved
Workflow каждый binding содержит явный выбор, а AllocationSpec — полностью
разрешённую эффективную политику. Snapshot readers не восстанавливают
отсутствующее поле и не поддерживают старую форму данных. Runtime получает
одну активную ветвь: `none`, `terminal@1` либо `compact@1`.

У `compact@1` требуется явный `includeReasoning`; остальным параметрам
можно дать документированные authoring defaults. В resolved форме все
значения явные. Переопределение `compact@1` целиком заменяет context policy
шаблона: его terminal instructions, ratio и `cumulativeBudget` не смешиваются
с compaction-настройками. Сам шаблон и его digest при этом не изменяются.

Первый вариант не вводит Workflow-wide defaults и несколько уровней merge
для поведения. Binding уже является точным местом выбора Worker; Router
может сочетать разные политики разных bindings. Общее значение на весь
Workflow можно добавить позднее при подтверждённой потребности автора,
с однозначным whole-object precedence.

`on.summarized` с compaction binding отклоняется: выбранная стратегия не
может породить этот исход. Нельзя одновременно настроить compaction и
неявный terminal fallback на его ошибке. Если нужен такой составной режим,
он требует отдельного явного контракта.

### 2.2 Reasoning как настройка самого потребителя

В существующий AgentTemplate `summarizer` добавить:

```yaml
summarizer:
  modelPolicy: summarizer@1
  instructions:
    ref: instructions/terminal-summarizer.md
  includeReasoning: true
```

Для terminal summarizer отсутствие authoring-поля разрешается в `false`;
нормализованная форма содержит обязательный boolean. Это сохраняет нынешний
поддерживаемый режим и добавляет явный новый. Для нового compaction пример
выбирает `true`, чтобы проверить сохранение незавершённых рассуждений.
Сравнение `true/false` входит в обязательный live Eval; менять default по
одной удачной сессии не следует.

Под reasoning здесь понимается только текст, который provider действительно
вернул в `reasoning_content` / `reasoning` и который
[Gateway adapter](../../runtime/src/contractor_runtime/llm/openai.py)
превратил в `Part(thought=True)`. Счётчик reasoning tokens сам по себе не
предоставляет этот текст. Отсутствующий или непрозрачный reasoning не
восстанавливается, не заменяется выдуманным объяснением и не является ошибкой.

При включении reasoning:

1. Проектор сохраняет его как отдельные помеченные входные данные с
   привязкой к ходу модели. Он не становится system instruction или
   подтверждённым наблюдением.
2. В запрос compactor он попадает как обычный текст структурированного
   документа. Простая передача нового `Part(thought=True)` не подходит:
   текущий Gateway намеренно не replay-ит такие части в исходящие messages.
3. Instructions просят сохранить полезные гипотезы, причины отвергнутых
   вариантов, неопределённость и ближайший шаг. Состоявшиеся действия,
   точные refs и наблюдения подтверждает Runtime, а не reasoning.
4. Общая граница входного документа остаётся обязательной. Task и результаты
   инструментов имеют приоритет перед reasoning; обрезка reasoning явно
   отмечается. Предлагаемые отдельные пределы: 8 KiB на часть и 64 KiB на
   входной документ. Это параметры первичного измерения, не токенные оценки.
5. Применяются те же правила удаления известных секретов и private paths,
   что и к другим входным данным потребителя.

Обычный replay `reasoning_content` в main model не меняется. В
`_content_messages` уже описаны циклы Qwen/LM Studio при таком replay.
Compactor получает reasoning как материал и возвращает обычную рабочую
память; raw thought не переподключается как новый скрытый reasoning turn.

Флаг не включает сохранение reasoning в Worker State, Planner payload,
публичный API или UI. Он также не включает telemetry content capture:
действуют отдельные OBS-R54–R60, включая существующий opt-in trusted sink
и принудительный metadata-only режим для sensitive-output toolsets.

### 2.3 Модели, pins и совместимость

Compactor — отдельный tool-free model consumer с точным ModelPolicy,
но использует тот же effective Gateway и credential, что Worker. Он не
получает domain tools, MemoryTools, Skills, workspace или Artifact client.
Менять модель работающего Run или выбирать её автоматически не требуется.

Его ModelPolicy требует `contextWindowTokens`, `maxOutputTokens`,
`maxModelCalls: 1` и конечный `maxTotalTokens`; tool/worker-call limits
запрещены. Один вызов относится к одному циклу compaction, а не ко всему
Worker invocation. Runtime ограничивает общее число циклов отдельно.

Для управления выбором через существующую модель executionConfig добавить
reference-only leaf `compactorModelPolicy`, применимый только к bindings с
`compact@1`. Стратегию, instructions, reasoning flag и предел циклов Run
override не меняет. Неприменимые `summarizerModelPolicy` и
`compactorModelPolicy` selections отклоняются, а не игнорируются.

Новые поля и точные dependencies включаются в соответствующие Workflow,
AgentTemplate и Run digests/snapshots. Runtime проверяет разрешённые refs и
тексты до построения Worker. Labels не выбирают стратегию или reasoning flag.
Generated model policies обновляются через исходники `deploy/models/`.

## 3. Предлагаемый алгоритм compaction

Активный контекст после успешного сжатия:

```text
Неизменяемая AgentTemplate instruction и действующие tool declarations
Точное текущее задание от Runtime
Одна рабочая память предыдущей истории
Недавние полные группы взаимодействий
Ограниченный Runtime reminder для продолжения
```

### 3.1 Что сохраняет код

Runtime повторно подставляет точные `subtask_id`, objective, task instructions,
parameters и named input refs из принятого запроса. Эти данные не должны
восстанавливаться моделью из summary. Выбранные инструменты, политика
доступа, их рабочие объекты и State остаются прежними.

Это защищает ограничения, которые действительно дошли до Worker. Оно не
восстанавливает требование исходного пользователя, если Planner ранее не
передал его в StageContentRequest. Полнота передачи таких требований —
отдельная проверка Planner boundary.

Trusted observations, artifact receipts, список действительно наблюдённых
refs, completion contract, counters и canonical `contractor` State не
сворачиваются и не переписываются compactor. ArtifactRef, написанный в
summary, остаётся обычным текстом и не становится output binding.

### 3.2 Что поручается модели

Отдельный вызов получает предыдущую рабочую память и новый старый префикс,
который теперь можно свернуть. Сохраняются:

- достигнутый прогресс и подтверждающие его наблюдения;
- важные пути, имена, числа и идентификаторы;
- неудачные проверки и причины не повторять их;
- открытые гипотезы и альтернативные объяснения;
- неизвестные исходы и незавершённые вопросы;
- ближайший полезный шаг.

Включённый reasoning служит дополнительным материалом для последних трёх
пунктов. «Предположено» и «подтверждено инструментом» должны различаться.
Instructions не должны учить конкретным уязвимостям из Eval.

Модель возвращает небольшой внутренний `CompactionModelResult` с одним
полем `summary: str`. Структуру checkpoint, source range, generation,
идентичности и ссылки на архив собирает код. Это внутренний результат
сжатия, не новый WorkerResult и не модельный пакет доверенных фактов.

Новая память заменяет предыдущую. Не накапливать список всех summary и не
суммаризировать весь архив заново на каждом ходе. Старые исходные события
сохраняются для выборочного чтения в пределах allocation-local лимита.

### 3.3 Недавняя история и восстановление подробностей

Группа — завершённый ответ модели вместе со всеми выбранными им tool calls
и соответствующими результатами, включая ошибки. Частичный streaming event,
незавершённый call или отделённый response нельзя переносить через границу.
`keepRecentGroups: 4` сохраняет последние четыре полные группы при условии
общего byte bound. Нельзя молча оставить половину группы; если требуемый
хвост не помещается, нужен явный отказ или изменённая проверенная политика.

Для деталей предоставить один allocation-bound read-only Runtime tool
`read_context_history`. Он доступен только при `compact@1` и читает исходные
данные этой же session policy. Аргументы — проверяемые ordinal и позиция
продолжения, не host path, произвольный session ID или чужой ArtifactRef.
Нужны pagination, ограничение ответа и явные признаки следующей страницы.

Инструмент возвращает только разрешённую проекцию истории. Reasoning
подчиняется `includeReasoning`; секреты и runtime internals исключаются.
Большой payload можно читать частями как документ с полным descriptor
исходного вызова, а не превращать отрывки в реальные protocol tool messages.
Чтение расходует обычный tool-call budget. Повторная выдача Skill content
должна учитывать действующий disclosure budget, а не открывать обход
SKL-R98–R100 через архив.

Штатные Workspace, Artifact и MemoryTools остаются источниками актуальных
данных. Архив сохраняет наблюдение на момент вызова и не утверждает, что
файл или mutable binding с тех пор не изменились.

## 4. Trigger, budgets и ограничение работы

### 4.1 Начальный trigger

Для первой реализации использовать provider-reported prompt tokens последнего
завершённого main-model вызова:

```text
boundary = min(
  floor(contextWindowTokens * contextWindowRatio),
  contextWindowTokens - maxOutputTokens,
)
```

Проверка — после завершения выбранных инструментов и до следующего model
вызова. Валидное обычное завершение имеет приоритет над compaction.
Для Audit сначала действует существующий completion boundary: сжатие не
заменяет проверку обязательных результатов и publication receipts.

Граница наблюдает прошлый запрос. Она не гарантирует, что следующий запрос
после большого tool result поместится. Byte admission и provider context
rejection остаются отдельными исходами. В v1 не вводить автоматический
повтор уже отвергнутого provider запроса с уменьшенной историей.

Если usage отсутствует, v1 не запускает compaction по оценке символов.
Оценочный trigger возможен как отдельная будущая политика с явно указанным
источником оценки и калибровкой на коде, JSON и нескольких языках. Запрет
из terminal-контракта не объявляется неизменяемым правилом нового режима.

### 4.2 Что compaction не обнуляет

`cumulativeBudget` terminal summarizer — мягкий предел потраченных токенов.
Compaction этого расхода не уменьшает. В `compact@1` этот trigger не
используется; жёсткие `maxTotalTokens`, `maxModelCalls`, `maxToolCalls` и
Worker-call deadline продолжают ограничивать весь логический вызов.

Вызовы и usage compactor входят в общий `maxModelCalls` / `maxTotalTokens`
Worker invocation и отдельно показываются как breakdown. Это предотвращает
скрытую бесплатную ёмкость. Usage не учитывается дважды. Result finalizer
по-прежнему расходует тот же общий бюджет. Последний prompt count для
context trigger обновляется только main model, не compactor/finalizer.

После compaction увеличивается context generation. Старый высокий prompt
count не может немедленно вызвать ещё одно сжатие. Для повторного цикла
нужен новый завершённый main-model ответ в новой generation и новый
сворачиваемый префикс. Если контекст сразу остаётся выше границы, а удалить
без потери защищённого задания и недавнего хвоста больше нечего, возвращается
явный `ineffective` исход вместо цикла compaction без прогресса.
Это проверка после commit и следующего измеренного ответа: уже принятый
marker, наблюдения и списанные вызовы сохраняются, а следующий semantic
dispatch/цикл запрещается. Она отличается от непригодного candidate,
который можно отвергнуть до commit без добавления marker.

Контракт последней задачи V417-004: в `shared` сохраняется session-level последнее измерение с его generation,
чтобы новый A2A invocation не забывал размер уже общей истории. Invocation
budgets при новом принятом A2A запросе меняются только по существующему
контракту; сам compaction новый запрос не создаёт.

### 4.3 Предлагаемые начальные пределы

Значения ниже нужно подтвердить измерением перед реализацией и затем
определить один раз в platform limits с общими fixtures. Это byte/group
bounds, а не утверждение о числе токенов.

| Объект | Начальное предложение |
| --- | --- |
| Циклы compaction | `maxCompactions` от 1 до 16, default 8 на invocation; не более 32 на allocation |
| Недавние группы | От 1 до 16, default 4; суммарно не более 128 KiB |
| Вход compactor | Не более 8 MiB, включая task, предыдущую память и сворачиваемый префикс |
| Принятая память | Не более 16 KiB UTF-8; отдельный `maxOutputTokens` в ModelPolicy |
| Входной reasoning | Не более 8 KiB на часть и 64 KiB суммарно, с явным truncation marker |
| Сохраняемая история | Не более 32 MiB сериализованных данных и 4096 полных групп на allocation |
| Одна страница архива | Не более 32 KiB и 8 групп либо ограниченная часть одной большой группы |
| Копии и transient input | Учитывать raw ADK events, проекции и checkpoint; не считать только размер summary |

Нельзя адаптировать нынешний tail-only TranscriptRecorder как полный архив:
он уже удаляет старые группы при превышении 512 KiB и ограничивает вложенные
значения. Для продолжающегося режима это потеря возможности восстановить
детали. Переиспользуются проверенные redaction и grouping primitives после
проверки их точности, а не его семантика хранения.

Input не обрезается молча, чтобы вызвать compactor любой ценой. Если полный
выбранный prefix не помещается, возвращается явный limit outcome; изменение
политики, предварительное offloading или несколько последовательных
compression batches требуют своих измеренных границ. Не вводить в v1
скрытый recursive map-reduce.

Группы индексируются по мере поступления. Страница архива не пересканирует
всю allocation history. Обход ADK events ограничивается объявленным ceiling;
проверяется на 4096 группах. Случай `shared` с большим числом subtasks относится
к последующей V417-004 и не блокирует isolated-поставку.
Сериализация больших проекций выполняется off event loop через существующий
joined-thread механизм, а не под lock canonical State.

## 5. Интеграция с ADK и Runtime

Рекомендуется переиспользовать формат ADK `EventCompaction` и штатную сборку
контекста, а trigger, input projection, compactor call и проверку результата
оставить в Runtime. Это продолжает существующее разделение instrumentation
и coordinator и позволяет проверить переход до следующего side effect.

Переиспользование публичного формата не означает включение штатного
controller. В ADK 2.13 summarizer может перехватить исключение и вернуть
`None`, после чего native compaction пропускается с сохранением исходной
истории. Для уже начатой операции Contractor это неуспешный исход:
нужны явная ошибка, отсутствие принятого marker и запрет следующего вызова.
[Исходный compaction v2.13.0](https://github.com/google/adk-python/blob/v2.13.0/src/google/adk/apps/compaction.py).

Проверяется точная версия и lock, принятые V418-007. Native automatic,
token-trigger и post-invocation compaction остаются выключенными;
оценка tokens самим SDK не подменяет actual-usage trigger Contractor.
Reasoning projection, общие бюджеты, completion priority, cancellation
ownership и точка commit остаются у Runtime. Публичный adapter допустим
только после проверки всех этих свойств в V417-002.

Одна логическая invocation Contractor — это текущий A2A task/subtask,
владелец State, collector, budgets и deadline. Внутри неё допускаются
несколько Runner turns ADK с разными SDK invocation IDs и повторными
callbacks. Mapping и доставка каждого continuation message в реальный
Gateway request наследуются из V418. Повторный SDK callback не начинает
заново логическую invocation и не очищает collector или счётчики.

Последовательность:

1. Instrumentation фиксирует `compaction_requested` на безопасной границе,
   когда нет незавершённых model/tool callbacks. В callback не выполняется
   вложенный LLM call.
2. Coordinator закрывает текущий async generator в той же task, удерживая
   invocation lock, deadline, counters и существующие tool instances.
3. Строит bounded immutable input и делает один tool-free compactor call.
4. Проверяет schema, размер, output-limit finish, секреты, бюджеты,
   актуальность source range и возможность безопасно применить checkpoint.
5. Коммитит один marker через SessionService. Непринятый результат не меняет
   активный контекст. Вызов, прошедший в момент cancellation/deadline,
   не даёт позднего разрешения продолжить Worker.
6. Использует принятый в V418 механизм continuation для той же логической
   invocation и session. Новый SDK turn получает bounded Runtime-authored
   continuation с точным текущим task; Runtime проверяет доставку сообщения
   в следующий запрос, не повторяя исходное действие и не сбрасывая бюджеты.
7. Следующий main-model вызов получает compacted history и продолжает работу
   с прежними инструментами.

`new_message=None` нельзя считать готовым универсальным решением: в
не-resumable ветви установленного Runner есть его явный запрет. Нужен
проверенный continuation message; включать SDK resumability и его
at-least-once tool replay только ради compaction не требуется.

Thin adapter изолирует ADK-specific event shape. Runtime хранит точные
event IDs/ordinals, generation, pending marker identity и scope
allocation/binding/session/branch/isolation. В публичном `EventCompaction`
есть `start_timestamp`, `end_timestamp`, `compacted_content`; поля
`event_ids` в нём нет. Диапазон timestamps включительный, поэтому adapter
проверяет его точное соответствие выбранным группам до mutation.
[Публичный формат](https://github.com/google/adk-python/blob/v2.13.0/src/google/adk/events/event_actions.py)
и [сборка контекста](https://github.com/google/adk-python/blob/v2.13.0/src/google/adk/flows/llm_flows/context/_compaction.py).

Fixtures покрывают совпавшие timestamps на обеих границах, оставляемые
соседние события с тем же временем, события не по порядку, предыдущие
summary, пересекающиеся markers и чужие branches/scopes. Ни один такой
случай не должен захватить лишнюю группу или разделить calls и responses.
Непредставимый диапазон отклоняется до append либо используется доказанный
в spike request-projection adapter. Private SDK helpers не становятся
публичным Contractor API.

Raw events уже остаются в allocation-local session и могут служить архивом;
не создавать автоматически вторую полную копию и новый durable store.
Archive reader обязан применять ту же политику projection. Встроенные
automatic compaction triggers не включаются параллельно с controller.

До основной реализации нужен небольшой integration spike на real ADK с fake
Gateway: checkpoint, два продолжения, calls/results, отсутствие повторного
tool side effect и сохранение State. Если публичный event mechanism не
позволяет обеспечить эти свойства на закреплённой версии, пересмотреть
adapter на уровне request projection; не обходить это молча SDK internals.

## 6. Session, State и lifecycle

| Ситуация | Поведение |
| --- | --- |
| `isolated` | Compaction и архив живут в пределах одного invocation; после его завершения conversation удаляется. Никакая новая память не протекает в следующий invocation через eligible State |
| `shared` | Сейчас `compact@1` отклоняется. Последняя задача V417-004 сохраняет предложение: compacted context и архив доступны последовательным invocations той же allocation; между bindings и allocations они не объединяются |
| Обычное завершение | Существующий finalizer и trusted result assembly; `summarized=false`, даже если ранее были compactions |
| Audit completion | Существующий trusted completion boundary и receipts; compactor не завершает проверку и не публикует результат |
| Cancel, deadline, lease loss | Остановить и join текущий compactor/continuation через действующий lifecycle; больше не запускать model/tool calls |
| Retry или escalation | Новая StageExecution и allocation по существующей политике; compaction не создаёт собственный retry engine |
| Runtime restart | Session и архив потеряны; существующий interrupted/retry path, без обещания возобновить checkpoint |
| Finalize, abort, release | Не запускают compactor. Удаляют session, summary, pending input, архивные проекции, reasoning и model client; неполный cleanup удерживает fence |

Активированные Skills, MemoryTools и canonical State не сбрасываются при
сжатии. Skill body может исчезнуть из активного контекста, но остаётся
доступным через разрешённый архив или явный повторный load с учётом budget.
Compaction не синтезирует activation calls. Memory остаётся отдельным
явным механизмом координации по MEM-R2–R4, а не скрытым backend summary.

## 7. Ошибки и неоднозначные исходы

Названия новых кодов ниже предлагаемые; перед реализацией они регистрируются
в platform errors и shared fixtures вместе со всеми Go/Python readers.

| Исход | Решение |
| --- | --- |
| Валидная summary и checkpoint | Принять один раз и продолжить ту же invocation |
| Невалидный, пустой, oversized или output-limited ответ | `worker_compaction_failed`, bounded cause; ни repair, ни второй семантический вызов внутри цикла |
| Summarizer exception или native `None`/skip после допуска compaction | Явный `worker_compaction_failed`, ноль принятых markers и последующих model/tool calls; продолжение на прежней истории не разрешено |
| Retryable Gateway failure | Существующая bounded transport attempt series; после неё `worker_compaction_failed` сохраняет retryable classification |
| Authorization, quota, invalid request, context rejection | Та же обёртка сохраняет non-retryable classification; стратегия не меняется |
| General Worker budget исчерпан | Существующий `worker_budget_exhausted`; compaction не предоставляет дополнительный бюджет |
| Лимит циклов, истории или защищённого хвоста | Предлагаемый `worker_context_limit_exceeded`, non-retryable; без бесконечных повторов детерминированно отвергаемого запроса |
| Нет полезного сокращения | `worker_compaction_failed` с cause `ineffective`; остановить локальные повторы |
| Response lost после Gateway acceptance | Возможен повтор вычисления/биллинга по существующему Gateway transport contract; не обещать exactly-once model execution |
| Append marker мог выполниться, acknowledgement потерян | Сверить тот же pending marker ID и exact range в allocation-local session; не добавлять новый marker с новой identity |
| Commit нельзя однозначно проверить | Fence Worker, вернуть session/compaction failure; не продолжать на неизвестной истории |
| Cancel или Worker deadline | Существующий cancelled/timeout outcome имеет приоритет, поздняя summary не публикуется как успех |

Если отсутствие сокращения доказано до commit, отвергнутый candidate не
добавляет marker. Если оно выявлено только по usage следующего main-model
ответа, сохраняются уже принятый checkpoint и весь учтённый расход;
останавливается следующий dispatch, а история не откатывается.

Повторный `Event.id` не доказывает идемпотентность `append_event`. При
потерянном acknowledgement нужно подтвердить наличие ровно того pending
marker либо доказать отсутствие до append; неоднозначность требует fence.
Отмена запрещает новые semantic calls и требует join локальной работы
до release, но не обещает откат уже отправленного внешнего действия.
Неопределённые эффекты сверяются через existing receipts. Разрешённые
WRK-R168 terminal cleanup/auto-export сохраняют отдельный bounded timeout.

Проверка результата не доказывает семантическую полноту summary. Схема,
точные protected inputs, доступ к архиву и Eval снижают риск потери фактов,
но Runtime не должен выдавать model interpretation за гарантию сохранения
всего смысла.

## 8. State, telemetry и UI

Текущая summarizer section монотонна и рассчитана на одну попытку. Её нельзя
сбросить в `not_requested` и переиспользовать для нескольких compactions.
Добавить закрытый numeric compaction section с текущей generation,
`idle/requested/running/applied/failed`, счётчиками attempts/applied/failed,
usage и последним bounded cause. Историю попыток в State не накапливать.
Счётчики относятся к логическим invocations и циклам compaction, а не к
числу SDK contexts/callbacks. Native skip, abort и ошибка не дают `applied`.

Отдельный metrics breakdown содержит compactor calls/tokens/time, available
before/after main prompt counts, missing-usage count, bytes до/после,
archive reads и reasoning available/included/truncated counts.
Сравнение токенов после сжатия появляется лишь после следующего main-model
ответа; заранее нельзя сообщать измеренный compression ratio.

Summary, raw reasoning и полный архив не помещаются в private State API,
ExecutionReport и публичные события. UI может показывать «Контекст сжат,
работа продолжается», опираясь на подтверждённую phase. Это не `completed`,
не новый subtask и не `summarized=true`. Content capture остаётся отдельной
настройкой observability с её существующим trust boundary.

## 9. Проверка решения исследованиями и Eval

| Источник из подборки | Что используется в проектировании |
| --- | --- |
| [The Complexity Trap](https://arxiv.org/abs/2508.21433v3) | Observation masking — обязательный baseline, сложность LLM summary сама по себе не доказывает выгоду |
| [ACON](https://arxiv.org/abs/2510.00615v3) | Корректировать compression instructions по типовым ошибкам на обучающем наборе; оценивать на отдельных задачах |
| [Lost in Context Compaction](https://arxiv.org/abs/2608.11242v2) | Отделять сохранение ограничений от сохранения основного задания; защищённые inputs собирает код |
| [Questions That Come Later](https://arxiv.org/abs/2610.27301v1) | Проверять последующие вопросы, неизвестные в момент сжатия, и точные IDs |
| [TRACE](https://arxiv.org/abs/2608.06503v2) | Сравнивать продолжения из одного состояния непосредственно до и после checkpoint |
| [Continuous Context Management](https://arxiv.org/abs/2609.35540v1) | Не выбирать постоянное сжатие на каждом ходе без подтверждения качества |
| [Bud ContextCompact 2.0](https://arxiv.org/abs/2610.25516v1) | Измерять время compactor и место для reasoning/ответа; не ограничиваться размером входа |
| [Deep Agents](https://www.langchain.com/blog/context-engineering-deep-agents-remember) | Сохранять возможность восстановить подробности после compaction |

Эти работы не дают контролируемого ответа, улучшает ли именно
`includeReasoning=true` terminal summarizer Contractor. Это отдельная
проверяемая гипотеза. Польза вероятна, когда reasoning содержит ещё не
выраженный в действиях план; вред возможен, когда он закрепляет неверную
гипотезу или вытесняет реальные наблюдения. Оба исхода входят в Eval.

Сравнивать на одинаковых задачах, configured models, wall-clock и суммарных
token/call budgets:

1. Текущий terminal summarizer с reasoning выключенным и включённым.
2. Существующий handoff в streamline, если он настроен в исходном Workflow.
3. Продолжающийся observation masking как экспериментальный baseline.
4. Предлагаемый compaction с reasoning выключенным и включённым.

До dispatch закрепить также exact ADK version, digest dependency lock и
Runtime build. Все arms используют одну принятую SDK closure: сравнение
terminal на 2.8 с compaction на 2.13 смешивает два изменения. Все внутренние
Runner turns входят в один логический member и его общий бюджет.

Измерять успешность итоговой задачи, проверенные артефакты/находки,
корректность неопределённости, сохранение ограничений и идентификаторов,
повторные действия, retrieval success, суммарные tokens/calls, wall-clock и
стоимость успешного результата. Ground truth остаётся вне сканируемого
проекта. Одна сессия не является доказательством улучшения.

Пилотные классы задач: source review с несколькими гипотезами, построение
OpenAPI/LikeC4 с отложенными зависимостями и последовательность проверок,
где нужно помнить отрицательные результаты. Причины ошибок превращаются
в общую методологию instructions, а не в имена файлов и sinks benchmark.

## 10. Поставки и критерии приёмки

Текущие статусы и evidence находятся в связанных задачах. На 11 ноября
2026 года приняты V417-001, V417-002, V417-003 и V417-005; остальные задачи
ещё не завершены. Зависимости задают технический порядок. Нормативные
контракты обновляются вместе с реализующим их кодом.

| Задача | Самостоятельный результат | Зависимости |
| --- | --- | --- |
| [V417-001](../../tasks/done/v417-001-terminal-summarizer-include-reasoning.yml) | `includeReasoning` существующего terminal summarizer: конфигурация, pins, Runtime projection и проверки обоих режимов | Нет |
| [V417-002](../../tasks/done/v417-002-adk-compaction-continuation-spike.yml) | Test-only spike на принятом ADK 2.13: два checkpoint, точные ranges, одна логическая invocation через несколько SDK turns и строгие failure outcomes | V418-007 |
| [V417-003](../../tasks/done/v417-003-isolated-worker-context-compaction.yml) | Сквозной `compact@1` для обычного isolated Worker: Workflow → snapshot/wire → coordinator, архив и reader, catalog resources, limits и cleanup | V417-002 |
| [V417-004](../../tasks/v417-004-shared-session-context-compaction.yml) | `in_progress` по указанию пользователя; выполняется последней: shared context между invocations с общими границами allocation, корректными generation и текущей task authority | V417-003, V417-009, V417-010 |
| [V417-005](../../tasks/done/v417-005-audit-worker-context-compaction.yml) | Допуск compaction в isolated Audit с сохранением completeness gate, trusted results и receipts | V417-003, V64-012 |
| [V417-006](../../tasks/done/v417-006-compactor-model-policy-selection.yml) | Reference-only `compactorModelPolicy` через существующие executionConfig selections и exact pins | V417-003 |
| [V417-007](../../tasks/v417-007-context-compaction-state-and-ui.yml) | Числовые State/metrics, public projections и правдивое отображение в UI | V417-003 |
| [V417-008](../../tasks/v417-008-context-compaction-eval-harness.yml) | Замороженные Eval fixtures/scorers и opt-in harness; без live-вызовов при реализации harness | V417-001, V417-003 |
| [V417-009](../../tasks/v417-009-context-compaction-release-gate.yml) | Интегрированная deterministic проверка isolated compaction на реальных Server/Runtime/PostgreSQL и release gate | V417-003, V417-005, V417-006, V417-007 |
| [V417-010](../../tasks/v417-010-context-compaction-live-evaluation.yml) | Повторные live Eval и измеренные рекомендации отдельно для terminal summarizer и compactor | V417-008, V417-009 |

**V417-001 независима:** `depends_on: []`, собственный milestone
`terminal-summarizer-reasoning-2026-10`. Её можно реализовать и принять без
compaction, нового strategy union, ADK spike или live Eval. Отсутствие флага
в authoring разрешается в `false`; выбор default по качеству остаётся
отдельным исследованием.

Первая реализация compaction намеренно сквозная: V417-003 вводит контракт
вместе с рабочим consumer. Она сразу включает безопасную отмену, обработку
неоднозначного append, границы памяти и восстановление истории. V417-005
допускает isolated Audit через существующий completion owner. Shared
комбинации строго отклоняются до V417-004; отдельной поставки принимаемой,
но не исполняемой конфигурации нет.

После приёмки V417-003 технически независимы задачи по model-policy selection,
метрикам/UI и Eval harness (последний также требует V417-001). Isolated Audit
compaction принята в V417-005. Реактивированная V417-004 выполняется последней,
после deterministic
exit V417-009 и quality-задачи V417-010. Пока shared не реализован, допуск
`compact@1` с `shared` остаётся запрещённым; этот отрицательный сценарий
входит в isolated deterministic exit. При последующей реализации shared
соответствующие проверки обновляются вместе с контрактом; ранняя приёмка
и измерения качества не выдаются за подтверждение shared-поведения.

У поставки два отдельных выхода: V417-009 закрывает
`worker-context-compaction-2026-10`, V417-010 —
`worker-context-compaction-quality-2026-10`. Deterministic exit не зависит
от V417-001, V417-008 или V417-010 и не доказывает улучшение качества.
Измеренный результат может рекомендовать сохранить текущий режим; он не
переключает существующие Run и не меняет их модели автоматически.

Минимальная deterministic приёмка:

- Terminal path по-прежнему отдаёт `summarized=true`; compaction дважды
  продолжает одну invocation и отдаёт обычный результат только в конце.
- Полезный tool side effect до checkpoint выполняется один раз; finalizer
  отсутствует между compactions и вызывается при обычном завершении.
- Model/tool/token counters, deadline, observed refs, Skill disclosure budget
  и canonical State не обнуляются после checkpoint.
- Exact task constraints и input refs сохраняются даже при summary,
  которая их опустила или изменила. Model-authored refs не становятся
  trusted result artifacts.
- Reasoning действительно виден потребителю при `true`, отсутствует при
  `false`, не replay-ится как `reasoning_content`, редактируется по secret
  rules и явно помечается при ограничении размера. Provider без reasoning
  успешно проходит оба режима.
- Неверная гипотеза из reasoning не становится подтверждённой finding только
  потому, что попала в summary; доменные gates продолжают работать.
- Call/response groups, несколько tool calls за ход, error results,
  reasoning-only parts, partial events и совпадающие timestamps не ломают
  последовательность SDK content.
- `isolated` не переносит compacted context или доступ к архиву в следующий
  invocation; `compact@1` с `shared` отклоняется. Сохранение compacted context
  между invocations проверяется при реализации последней задачи V417-004.
  Skills/eligible State следуют своему контракту.
- Missing usage не запускает оценочный trigger, stale pre-compaction usage
  не вызывает повторный цикл, большой свежий tool result даёт честный
  context/limit failure вместо скрытого repair.
- At-ceiling тесты охватывают 4096 групп и 32 MiB, pagination, отсутствие
  whole-archive rescans на чтение и прекращение при исчерпании границ.
- Cancellation/deadline/lease loss на каждом шаге не позволяют позднему
  compactor запустить модель, инструмент или публикацию после fence.
- Ambiguous append даёт один checkpoint или fenced failure; не создаёт
  новый receipt mechanism и дублирующиеся события.
- Real Server/Runtime process test подтверждает выбранный Workflow mode,
  private wire, terminal report и отсутствие ложного handoff в streamline.
- Audit test сохраняет completeness gate и publication receipts, включая
  незавершённую проверку, которая после compaction продолжает работу.

## 11. Затрагиваемые контракты и решения для пересмотра

| Граница | Будущие изменения |
| --- | --- |
| Workflows | Context policy binding, normalization, immutable snapshots, совместимость с `on.summarized` |
| AgentTemplates | `summarizer.includeReasoning`, revised TPL/WRK projection rules; точные digests |
| ModelPolicy и executionConfig | Compactor consumer и reference-only override, budget accounting |
| Private wire | AllocationSpec, State schema, report metrics; Go/Python shared fixtures |
| Worker runtime | Coordinator, SDK adapter, trigger epochs, archive reader, continuation и cleanup |
| Public API и UI | Новые поля read/authoring models и metrics, правдивое отображение состояния |
| Platform registries | Limits, errors, failure outcomes, allocation-local lifecycle и observability |
| Catalog и deploy | Новый compactor policy/instruction resource через источники генерации |

При реализации меняются OpenAPI, event schemas по необходимости, обе
generated client поверхности штатными генераторами, server/client validation,
config editor и cross-language fixtures в одном согласованном change set.
Нормативные документы получают требования только вместе с реализованным
поведением. В будущих задачах указываются owning spec sections и реальные
команды из текущего release graph; PostgreSQL gates проверяют реальные
snapshot/read paths, process gates — настоящий Worker continuation.

Основные точки кода для будущей реализации:
[runtime coordinator](../../runtime/src/contractor_runtime/worker/runtime.py),
[instrumentation](../../runtime/src/contractor_runtime/worker/instrumentation.py),
[budget](../../runtime/src/contractor_runtime/worker/budget.py),
[summarizer projection](../../runtime/src/contractor_runtime/worker/summarizer.py),
[session lifecycle](../../runtime/src/contractor_runtime/worker/sessions.py),
[Go config source](../../internal/config/types.go),
[resolved transition validation](../../internal/config/resolved/summary.go).

| Решение | Основание и условие пересмотра |
| --- | --- |
| Выбор в binding Workflow | Точное соответствие Worker и поддержка Router; общий default добавить при реальном повторении конфигурации |
| Summary с недавними группами и архивом | Сохраняет продолжение и доступ к деталям; менять, если masking даёт лучшее качество/стоимость в повторных Eval |
| Reasoning — явный параметр | Данные уже доступны, польза и ошибки зависят от модели/задачи; default определяется сравнением |
| Runtime controller и ADK event format | Позволяет повторно использовать существующий safe-boundary coordinator; пересмотреть после spike, если SDK adapter сложнее управляемого native integration |
| Никакого compaction на cumulative spending | Потраченные токены не уменьшаются; другой stop/handoff по расходу требует явно составной стратегии |
| Локальная недолговечная история | Совпадает с текущим allocation lifecycle; durable recovery проектировать только при отдельной потребности |
| Без обучения политики в первой версии | Сначала нужен воспроизводимый baseline; ACON/AutoCompact-подходы рассматривать после устойчивого Eval и корпуса ошибок |

Новые таблицы, Scheduler engine, общий memory service и автоматическая смена
модели для этого варианта не требуются. Главное доказательство полезности —
Worker завершает исходную задачу после нескольких compactions с меньшими
совокупными расходами или лучшим качеством при равных ограничениях.
