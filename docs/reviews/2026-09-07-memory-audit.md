Аудит Shared MemoryTools от 2026-09-07

Проверена текущая рабочая копия Contractor, базовый HEAD:
`d7e229c66349902892babeff3139e2115f57287f`. Область проверки —
`memory-tools@1`: Go Planner, Python Worker, Artifact API, конфигурация,
ограничения доступа, повторные записи и диагностические проекции.
Это аудит общей памяти агентов, а не потребления RAM или workspace storage=memory.
Исходный код и рабочая конфигурация в рамках этого аудита не изменялись.

Вердикт: основной сценарий работает и покрыт интеграционными проверками,
но найдены три воспроизводимых дефекта. Кроме того, в текущем каталоге
`configs/agent-templates` ни один из 20 шаблонов не выбирает `memory-tools@1`.
Выбор имеется в `configs/e2e/agent-templates`. Следовательно, обычные агенты,
созданные из проверенного рабочего каталога, инструментами общей памяти
не пользуются. Состояние конфигураций уже запущенного сервера не проверялось.

1. **P1 — HTTP 500 после состоявшейся записи допускает повторный append.**

   В `runtime/src/contractor_runtime/toolsets/memory.py:274–279` точный повтор
   с прежним CAS разрешён только для `ArtifactTransportError`.
   `ArtifactAPIError(500, internal_error)` немедленно превращается в
   `memory_unavailable`, `retryable=true`, без проверки текущего содержимого.
   Между тем потеря подтверждения PostgreSQL после коммита может привести
   именно к такому HTTP-ответу: обычная ошибка репозитория передаётся через
   `internal/httpapi/privateartifacts/artifact_handlers.go:159–170` и
   `internal/httpapi/privateartifacts/errors.go:61–62`.

   Воспроизведение использовало настоящий Python `ArtifactClient` и Memory
   tools, тестовый backend и инъекцию HTTP 500 после успешного PUT.
   Исходная заметка — `base`, добавляемый фрагмент — `fragment`:

   ```text
   first append: memory_unavailable retryable= True
   PUT attempts for first append: 1
   stored after error: 'base\nfragment'
   stored after advertised retry: 'base\nfragment\nfragment'
   ```

   Это локальная инъекция ошибки, а не наблюдение сбоя на рабочем сервере.
   Возможность коммита с ошибкой подтверждения отдельно проверяется штатным
   PostgreSQL-тестом `TestFilesystemLostWriteAcknowledgementKeepsCommittedBytes`.
   Штатная матрица Memory response loss моделирует транспортные исключения
   и поэтому этот случай пропускает.

   Исправление: ошибки PUT, которые не доказывают отсутствие коммита,
   включая `internal_error`/неопределённые серверные сбои, должны попадать
   в ограниченный повтор тех же байтов с тем же precondition и сверку
   текущего payload. Авторитетные отказы доступа и явный исходный CAS-конфликт
   должны сохранять свои отдельные ветки. Нужен регрессионный тест HTTP 500
   после коммита для create, replace и append.

2. **P2 — обычные артефакты могут заблокировать даже пустую memory.**

   `runtime/src/contractor_runtime/toolsets/memory.py:214–220` сначала
   получает все bindings namespace и только потом фильтрует `memory.`.
   У списка ArtifactClient есть лимит JSON-ответа 1 MiB
   (`runtime/src/contractor_runtime/artifacts.py:28,216–226`), пагинации
   в этом контракте нет. Лимит 128 заметок не ограничивает число обычных
   артефактов, которые хранятся в том же namespace.

   Воспроизведение: тестовый namespace с 7000 обычных bindings с допустимыми
   именами длиной 126 ASCII-байт и нулём заметок. Настоящий ArtifactClient
   использован с тестовым transport, соблюдающим переданный ему лимит ответа:

   ```text
   response bytes: 1120050 limit: 1048576
   7000 ordinary artifacts, 0 notes: list_memories memory_unavailable
   7000 ordinary artifacts, 0 notes: write_memory memory_unavailable
   ```

   По той же причине страдают `search_memory` и `list_memory_tags`.
   Чтение/изменение существующей заметки по имени от полного списка не зависит.
   Planner тоже сначала перечисляет весь namespace
   (`internal/memory/namespace.go:331–339`), но без HTTP-лимита, поэтому
   поведение двух адаптеров расходится. Даже в нормальном случае список
   из N заметок требует отдельного чтения каждого тела; при 128 заметках
   это 129 последовательных запросов Worker к Artifact API.

   Исправление: ограниченное перечисление bindings с фильтром имени
   на стороне существующего Artifact API/Store либо пагинация с явным
   пределом. Простое увеличение HTTP-лимита только отодвинет отказ.

3. **P2 — Planner возвращает `tags: null` для заметок без тегов.**

   `FullProjection` и `PreviewProjection` в `internal/memory/codec.go:104–116`
   копируют теги через `append([]string(nil), note.Tags...)`. При пустом
   наборе получается nil slice, который `encoding/json` кодирует как null.
   Python возвращает пустой список. По спецификации `MemoryNote.tags`
   и `MemoryPreview.tags` всегда имеют тип `list[str]`.

   Воспроизведение: `Normalize` заметки `untagged` без тегов, затем
   `json.Marshal(FullProjection(...))` и `json.Marshal(PreviewProjection(...))`.
   Обе Go-проекции содержат `"tags":null`; Python mutation response —
   `"tags":[]`. Stored canonical JSON при этом остаётся корректным.
   Дефект затрагивает обычный вызов с аргументами по умолчанию и нарушает
   одинаковый model-facing контракт двух реализаций.

   Исправление: создавать непустой по указателю slice нулевой длины
   (`append([]string{}, note.Tags...)` либо `make`+`copy`) и проверять
   сериализованный JSON обеих проекций на общей фикстуре без тегов.

Проверки, завершившиеся успешно:

- Python: 55 тестов из `test_memory.py`, `test_memory_toolset.py`,
  `test_artifacts.py`, `test_run_artifacts_toolset.py`,
  `test_text_artifacts_toolset.py`, с `-W error`.
- Python: ещё 6 проверок `-k 'memory or reserved'` в `test_adk_runtime.py`,
  `test_source_analysis_toolset.py`, `test_openapi_toolset.py`,
  `test_likec4_toolset.py`, с `-W error`.
- Go: `go test -race -count=1 -timeout=4m` для `internal/memory`,
  `internal/planner/streamline`, `internal/planner/router`,
  `internal/artifactpolicy`, `internal/httpapi/privateartifacts`,
  `internal/config`. Настроена отдельная временная PostgreSQL 17;
  PostgreSQL-тесты выполнены, а не пропущены. В JSON-отчёте —
  499 успешных test/subtest-событий, без fail/skip.
- `TestSharedMemoryMVPProcesses`: PASS, около 98 секунд. Запуск настоящих
  Go Server и двух Python Runtime с локальным сценарным LLM Gateway.
  Проверены Streamline/Router, обмен заметками, retry, последующий Stage,
  независимость нового Run, скрытие storage identities и проверяемые каналы
  диагностики/OTLP при `captureContent=false`.

Три дополнительных воспроизведения выше проверяют случаи, которых нет
в прошедших штатных тестах. Успешные штатные тесты не отменяют замечания.
Проверка реальных LLM, длительный нагрузочный прогон, аварийное завершение
боевой БД и весь набор тестов репозитория в этот аудит не входили.

Отсутствие cross-Run памяти, удаления/автокомпакции и семантического поиска
соответствует версии v1. Memory сохраняется между попытками в одном Run
и namespace; она не добавляется автоматически в prompt и не становится
результатом Stage. Общая атомарность разных concurrent creates не заявлена:
нынешний контракт опирается на последовательное исполнение. При расширении
модели конкурентности квота и назначение ordinal потребуют отдельной защиты.

По запросу пользователя замечания сопоставлены с последними задачами
V47-001/002/003. Они посвящены temperature Planner, retryability LLM Gateway
и допуску запроса summarizer в контекст модели. V47-002 тематически близка
к первому замечанию, но не покрывает неопределённый результат Artifact PUT.
Прямого покрытия найденных дефектов этими задачами нет.

Созданы отдельные задачи со статусом pending:

- V48-001 — сверка результата Memory PUT после неоднозначной HTTP-ошибки, P1.
- V48-002 — ограниченное перечисление Memory независимо от обычных артефактов, P2.
- V48-003 — одинаковая сериализация пустых тегов в Planner и Worker, P2.
- V48-004 — включение Memory в рабочие шаблоны, инструкции и активные ссылки
  Workflow/AuditProfile, P1; зависит от трёх исправлений выше.

Отсутствие Memory в рабочих шаблонах пользователь подтвердил как реальную
проблему конфигурации. Создание задач не означает, что исправления уже внесены
или Memory включена в работающем приложении.
