# Конфигурации toolsets на Run и MCP Streamable HTTP

План от 2026-09-19. Серия V61, статус: запланировано, реализация не выполнена.
20.09.2026 задачи перенесены в main и перенумерованы из V56 в V61; объём сохранён.
Статусы, зависимости, критерии готовности и команды проверки находятся в
[`tasks/index.yml`](../../tasks/index.yml) и `tasks/v61-*.yml`.

Пользователь запросил общий механизм конфигурации инструментов через
AgentTemplate и существующие bindings, выбор при запуске из UI и доступ к
произвольному MCP endpoint. Приняты ограничения: только Streamable HTTP,
одна конфигурация каждого toolset на весь Run, максимальное переиспользование
RuntimeConfig и минимальные изменения YAML.

Этот документ фиксирует решения для реализации. V61-001 переносит нормативные
контракты в существующие спецификации AgentTemplate, Runtime, RuntimeConfig и
Audit и в общие schemas/fixtures. Реализация должна следовать этим контрактам;
план и задачи не являются свидетельством готовности функции.

## Пользовательский контракт

AgentTemplate сохраняет текущую структуру и явно выбирает инструменты:

```yaml
spec:
  toolsets:
    - ref: mcp@1
      tools: [search, fetch]
```

`mcp@1` — новая реализация toolset. Endpoint и credentials задаются новым
разделом существующего RuntimeConfig:

```yaml
apiVersion: contractor/v1alpha1
kind: RuntimeConfig
metadata:
  name: knowledge-production
  version: "1"
spec:
  worker:
    toolsets:
      mcp@1:
        endpoint: https://knowledge.example.com/mcp
        credential: knowledge-access
```

Существующий runtime label `knowledge-prod` связывается с точными name, version
и digest этого документа. Run-create использует существующее поле:

```json
{"runtimeLabels": ["knowledge-prod"]}
```

Workflow и Stage agent binding сохраняют текущую форму. Новые ToolConfiguration,
configurationInputs, обязательный configBinding и toolset-поля executionConfig
не вводятся. Конфигурация и аргументы вызова различаются: существующий
`tool@1.execution.arguments` продолжает задавать literal/parameter/artifact
аргументы операции.

Ключ карты — точный `<toolset-id>@<version>`. Для native toolsets содержимое
определяется их зарегистрированной схемой; механизм не привязан к endpoint.
Допустимы собственные вложенные настройки отдельных инструментов, если это
предусмотрено схемой их toolset. Для первой поставки единственная новая
пользовательская схема — MCP; расширяемость проверяется тестовым native toolset.

На `mcp@1` приходится один endpoint на Run. Два агента могут выбирать разные
инструменты этого подключения. Несколько экземпляров одного ref, разные
endpoint одного ref по агентам и динамическая регистрация произвольных aliases
в эту поставку не входят. Разные будущие зарегистрированные toolset refs могут
использовать одну реализацию клиента.

## Descriptor, типизация и совместимость

Расширяется существующий code-owned ToolsetDescriptor: источник инструментов
`static` или `allocation`, схема/обязательность конфигурации, допустимые Worker
runtimes и общие инфраструктурные каналы при динамическом списке инструментов.
Эти свойства не дублируются в пользовательском YAML. Каждая схема регистрирует
типизированные normalization/validation и извлечение credential references.
Неизвестные refs, неподдерживаемая конфигурация и неизвестные поля отклоняются.

Для `mcp@1` обязательна MCP-конфигурация, список проверяется при allocation,
первоначально поддерживается только `adk@1`; HTTP использует существующий канал
`runtime-http-client`. Native статическая валидация, видимые имена, запрет
коллизий и специальные имена Agent Skills сохраняются. Общая доставка native
конфигурации работает и для model-free Worker без появления model credentials.

Контракты расширяются синхронно в Go, Python, JSON Schema, OpenAPI и строгих
UI-парсерах. Отсутствующие новые поля не сериализуются: старые canonical bytes,
digests, пустой built-in RuntimeConfig и статические capabilities сохраняются.
Остаётся единый private protocol `contractor/v1alpha1` из V50.

Новая обязательная конфигурация существующего native toolset требует отдельной
версии его контракта/capability: прежний static ref сам по себе не доказывает
поддержку новых settings старым runtime. Тестовый native descriptor имеет свой
ref; существующие production native toolsets в этой серии не переопределяются.

## Схема MCP и credentials

| Поле RuntimeConfig | Контракт |
| --- | --- |
| `endpoint` | Обязательный HTTP/HTTPS URL, до 2048 байт; без userinfo, query и fragment, без добавления пути клиентом |
| `credential` | Необязательный ID immutable credential вида `mcp-headers@1` |
| `caBundlePem` | Необязательный дополнительный CA, существующий предел 64 KiB; обычная проверка TLS сохраняется |
| `connectTimeoutSeconds` | Целое 1–60, default 10 |
| `requestTimeoutSeconds` | Целое 1–600, default 60; полный бюджет отдельного MCP RPC |

Более ранний enclosing deadline всегда ограничивает вложенную операцию.
Таймаут отдельного HTTP connect, полного RPC, подготовки allocation и cleanup
имеет своего владельца; настройка MCP не продлевает deadline Scheduler.
Долгоживущий входящий stream не считается бесконечным бюджетом tools/call.

Карта toolsets ограничена 128 refs и существующим общим пределом RuntimeConfig
128 KiB. Объекты нормализуются до сравнения и вычисления digest; optional
defaults материализуются одинаково во всех слоях. Нормализация URL сохраняет
значимую семантику пути и не переписывает произвольно пользовательский endpoint.

Новый credential хранит `{"headers": {"Authorization": "Bearer ..."}}`.
Переиспользуются шифрование, write-only create, idempotency и reference barrier.
Нормализация заголовков общая с существующими header credentials: не более 32,
имя до 64 байт, значение до 4096 байт, сумма значений до 16 KiB; case-insensitive
duplicates и CR/LF запрещены. Кроме hop-by-hop/proxy headers запрещены
управляемые SDK Accept, Content-Type, MCP-Session-Id, MCP-Protocol-Version и
Last-Event-ID. Авторизация отсутствует, если credential не задан.

RuntimeConfig, Run draft, provenance, descriptions и аргументы tools не содержат
plaintext. Заголовки материализуются только перед private allocation delivery.
Расширяются CHECK допустимых credential kinds, metadata validators, SQL-поиск
использующих bindings, Run/Audit holds и allocation references. Все collectors
редактирования секретов, safe repr, очистка и SDK logging учитывают новые значения
и session IDs. Ротация использует новый credential ID и новую версию config.
Интерактивный OAuth и автоматический refresh не входят в первую поставку.

## Merge, bindings и snapshot Run

Для `worker.toolsets` действуют только слои `default` и явные `runtimeLabels`
Run. Объект одного ref — атомарная настройка. В одном слое разные refs
объединяются, одинаковые нормализованные значения deduplicate, разные значения
одного ref дают стабильный конфликт с ref/path и источниками. Порядок labels
не определяет победителя. Явный Run-слой целиком заменяет значение default.

Отсутствие ref наследует значение; `ref: null` очищает его; `toolsets: {}` не
изменяет слой и нормализуется в отсутствие секции; `toolsets: null` запрещён.
Очистка обязательной конфигурации приводит к ошибке создания Run. Конфликты
проверяются по всем выбранным labels до проекции на отдельного агента.

Labels с непустой toolsets patch, включая очистку, нельзя назначать физическому
Runtime Agent. Переназначение binding уже назначенного агентам label на такой
документ также отклоняется под существующими locks/revision checks. Смешанный
config подчиняется тому же правилу. Resolver дополнительно отвергает такой
agent layer. Существующие правила proxy/telemetry не меняют выбранные Run
endpoint и credential; физический маршрут может отличаться по текущей политике.

При Run-create собирается объединение требований всех AgentTemplate workflow,
включая поздние стадии. В транзакции фиксации bindings проверяются merge,
обязательные конфигурации и credential kinds по тем же exact refs, которые
будут записаны в Run snapshot. Сетевых обращений к MCP в этой транзакции нет.
Используется существующее хранение immutable документов, без новой таблицы
конфигураций или копии bodies в Run. Credentials входят в существующие holds.

Повтор принятого idempotent запроса возвращает исходный Run до проверки текущих
aliases. Действие Repeat создаёт новый Run и сохраняет существующее review
изменившихся/удалённых bindings. Изменение default или label после Run-create
не меняет настройки уже созданного Run, его поздних стадий и replacement
allocations. Pinning endpoint не фиксирует реализацию удалённого сервера.

Audit переиспользует baseline/credential holds и проверяет требования всех
своих workflow bindings. До появления доверенной классификации generic MCP
имеет unknown external effects и несовместим с Audit-профилями, требующими
классификации/согласования действий. Это явная compatibility reason через
существующий admission, а не новый approval flow. MCP annotations и имена не
считаются доверенной классификацией. Native Audit поведение сохраняется.

## Allocation и capabilities

Общий pure resolver карты toolsets используется при Run-create и placement.
Для allocation вычисляется пересечение effective карты с refs AgentTemplate
до загрузки secrets и вычисления требований этого allocation. Дополнительные
известные настройки в Run не доставляются чужим агентам. Existing snapshot
может консервативно удерживать все выбранные credential references.

Расширяется существующий `AllocationSpec.runtimeSettings.toolsets`. В private
значении для MCP находятся endpoint, materialized secret headers, CA и timeouts.
Публичные и сохраняемые projections содержат только безопасные refs/origins.
`origins.toolsets[ref]` использует существующий формат происхождения значения.
Default/run pin refs и binding revisions сохраняются; при dedup источники
отражаются детерминированно, без создания искусственного precedence.

Предлагаемая новая capability:

```json
{"ref": "mcp@1", "tools": [], "toolDiscovery": "allocation"}
```

Отсутствующее `toolDiscovery` означает прежний `static`, где tools непустой.
Allocation mode допустим только для соответствующего серверного descriptor и
локальной factory. Wildcard names и ложный startup inventory запрещены.
Startup probe проверяет локальную готовность клиента, без пользовательского
endpoint. До реализации полного ADK пути production factory не объявляет MCP.
Placement проверяет generic capability, а preparation — реальные selected tools.
Нужные proxy adapters по-прежнему проходят существующие capability checks.

## Runtime: transport, lifecycle и Worker

Используется официальный Python MCP SDK с совместимой зафиксированной в uv.lock
версией и существующий ADK BaseTool. JSON-RPC, initialization, session headers,
Streamable HTTP JSON/streaming ответы обслуживает SDK. Legacy HTTP+SSE, stdio,
транспортный fallback и запуск серверных subprocess не реализуются.

MCP-клиент получает отдельные HTTP state/headers/cookies. При настроенном
`tool-http` route используется существующая proxy policy и host restrictions.
Интеграция с SDK не обходит эти проверки через raw underlying AsyncClient и
не позволяет закрытию MCP-клиента закрыть общий adapter allocation. Redirects
выключены; CA, запреты внутренних Contractor origins и deadline проверяются и
для запросов восстановления соединения. Существующие ограничения не заменяются
общим запретом всех private/local endpoints.

Фабрика создаёт ToolInstance и общего владельца сессии без сетевых ожиданий.
Сначала allocation принимает ownership через текущий rollback/cleanup path,
затем выполняется необязательная `prepare(deadline)` фаза, и только после неё
создаётся Worker. Эквивалентный дизайн допустим при доказанном ownership до
первого cancellable await; отдельный resource supervisor не требуется.

У каждого allocation своя сессия. Выбранные tools одного ref делят owner.
Owner входит и выходит из SDK async contexts в одной задаче, учитывает in-flight
calls, имеет идемпотентное закрытие и подчиняется существующим deadlines,
abort/release/lease-loss fencing. Неподтверждённая очистка не становится
успешной из-за подавленного SDK исключения. Частичная preparation остаётся
учтённой до подтверждённого освобождения.

Initialization и полный paginated tools/list выполняются с конечными лимитами
времени, страниц, числа tools, bytes сообщений/описаний/schema и глубины schema.
Лимиты фиксируются кодом и нормативной спецификацией до включения factory.
Проверяются missing/duplicate/reserved names, exact allowlist и допустимость
схем выбранных tools. Произвольные remote schema refs не загружаются.
Невыбранные tools не становятся model-visible. Схемы/список выбранных tools
замораживаются на allocation; list-changed не расширяет полномочия.

BaseTool использует description и JSON Schema выбранного инструмента, owner
выполняет tools/call. Native callables продолжают проходить через FunctionTool.
Сохраняются Worker callbacks, tool budgets, metrics, observation/completion
semantics и tool-free finalizer/summarizer. Native typed effects/evidence не
выводятся из удалённого имени или текста. Generic MCP под tool@1 отклоняется
до allocation; параметры model-free native tools не становятся LLM-настройками.

Результаты сохраняют isError, structuredContent и поддерживаемые content blocks
в bounded представлении. Неподдерживаемые blocks/превышение лимита имеют явный
исход, без скрытой загрузки URI, потери признака ошибки или неограниченного
base64 в контексте. Tool business error отличается от transport/protocol error.
Exact supported schema/content subset и пределы закрепляются V61-001; V61-006
добавляет fixtures и реализацию, прежде чем объявлять capability.

Setup/discovery можно ограниченно повторить в пределах preparation deadline.
После неоднозначного tools/call новый POST автоматически не отправляется.
Восстановление stream и повтор удалённой операции различаются; новая сессия
сверяет frozen selected schemas. Отмена/закрытие не обещает remote rollback.
Sampling, prompts, автоматическая загрузка resources и серверные запросы доступа
к local roots не включаются. Session IDs и credentials не попадают в logs.

## Public API, UI и наблюдаемость

Workflow detail получает вычисляемые `toolsetConfigurationRequirements`: ref,
required, объединение tools и bounded consumers (stage/agent). Label picker
получает безопасное покрытие refs (configured/cleared), `valueDigest`
нормализованного non-secret значения каждого configured ref и изменяемые секции,
с точным config ref и binding revision. Digest значения исключает metadata
документа: одинаковые settings в разных RuntimeConfig должны распознаваться как
одинаковые. Формы не загружают по полному config
для каждого label; projection/query count остаются bounded. Endpoint и secret
values для выбора/сопоставления не нужны. Источником записи остаются текущие APIs.

Форма Run показывает требуемые toolsets, инструменты/потребителей, выбранный
профиль, наследование default, отсутствие настройки и конфликт. В draft/POST
хранится только существующий runtimeLabels. Общий профиль выбирается один раз;
смешанные LLM/telemetry эффекты показываются, а не молча отбрасываются.

Создание подключения использует существующие Operations mutations: при
необходимости create credential, publish RuntimeConfig, create binding, select
label. Каждый шаг имеет устойчивую idempotency identity; после сбоя сохраняются
полученные refs и повторяется незавершённый шаг. Уже существующий общий label не
переназначается неявно. Незавершённая публикация не приводит к удалению чужих
ресурсов. Secrets не пишутся в URL/localStorage/Run draft и не читаются обратно.
Существующая глобальная Operations/auth модель сохраняется.

Первоначально используется специализированная MCP-форма в существующем UI
каркасе. Новая универсальная schema-form платформа и отдельный сетевой
preflight job/RPC не требуются. Фактическую доступность проверяет runtime при
preparation; UI показывает её честный статус, без имитации browser probe.

Диагностика различает missing/conflicting config, incompatible credential,
unsupported capability, connect/auth/schema/missing-tool ошибки, tool business
error, timeout/ambiguous call и cleanup unconfirmed. Происхождение настроек,
выбранные имена, digest выбранных схем и bounded timings/counters безопасны;
endpoint/session/credential plaintext не добавляются в метрики как labels.

## Задачи и зависимости

| ID | Результат | Новые зависимости |
| --- | --- | --- |
| V61-001 | Нормативные contracts, схемы, typed models и golden fixtures | — |
| V61-002 | RuntimeConfig normalization/merge, MCP credentials и reference lifecycle | 001 |
| V61-003 | Run/Audit pinning, allocation projection, secret materialization и origins | 002, 004 |
| V61-004 | Dynamic tool discovery, template/placement validation и Audit admission | 001 |
| V61-005 | Allocation-owned MCP transport/session, proxy integration и подготовка/cleanup | 003 |
| V61-006 | MCP ToolsetFactory, ADK BaseTool, exact selection, вызовы и accounting | 005 |
| V61-007 | Public requirements/label coverage/provenance projections | 003 |
| V61-008 | Operations и Run UI, создание/выбор подключения и draft recovery | 007 |
| V61-009 | Сквозной release gate, примеры и документация | 006, 008 |

Детальные depends_on также указывают необходимые завершённые задачи прежних
серий. Очерёдность внутри V61: 001 → (002, 004) → 003 → (005 → 006,
007 → 008) → 009. Это не меняет приоритеты других очередей в tasks/index.yml.
Все V61 задачи остаются pending до начала своей реализации.

## Критерий поставки

Контролируемый локальный MCP fixture и scripted model проводят обычный Run
через реальные Server/PostgreSQL/Scheduler/private allocation/Runtime/ADK
границы. Два агента используют одну pinned конфигурацию, разные allowlists и
отдельные сессии; третий native агент не получает MCP settings или secrets.
Изменение binding не влияет на позднюю стадию/replacement allocation исходного
Run; Repeat нового Run показывает изменение и фиксирует новый config.

Фикстуры покрывают JSON и streaming transport, pagination, missing/duplicate
tools, несовместимую schema, auth/proxy/TLS failure, oversized content,
неоднозначный call, cancellation во время setup/call, lease loss, failed prepare,
concurrent close и неподтверждённый cleanup. Браузерный сценарий проверяет
создание подключения, конфликт и восстановление между шагами публикации.
Регрессионный native Run и model-free Worker проходят со старыми настройками.

V61-009 добавляет `make test-toolset-runtime-configuration-release` с isolated
database, MCP fixture и browser prerequisites. Отсутствующий prerequisite,
невыбранный обязательный case или skipped mandatory case делает gate неуспешным.
Внешний MCP, платный LLM и production deployment для подтверждения не нужны.
Evidence точно описывает границы, исполненные cases и отсутствие утечек ресурсов.

Развёртывание: обновить schema/server, затем runtime с реализованной capability,
затем включить новые шаблоны и подключения. Новый сервер принимает старые static
registrations; старые runtime не получают allocations с неизвестными полями.
Rollback после публикации новых документов требует сохранения совместимого
reader либо явного отключения/дренирования новых запусков; старый бинарь не
объявляется совместимым с документами, которых он не понимает.

## Основные точки переиспользования

- `internal/config/descriptors.go`, `template.go`: контракты toolsets и selections.
- `internal/runtimeconfig`: immutable versions, normalization, merge, bindings,
  transaction pinning, credential references и origins.
- `internal/credentials`: encrypted write-only store и usage barriers.
- `internal/runservice`, `auditservice`: единая фиксация запуска и Audit baseline.
- `internal/controlplane`, `scheduler`: placement, allocation и materialization.
- `runtime/src/contractor_runtime/factories.py`, `allocation`: ownership/cleanup.
- `runtime/src/contractor_runtime/adapters/http_proxy.py`: HTTP route и guards.
- `runtime/src/contractor_runtime/worker`, `llm/openai.py`: ADK и JSON Schema.
- `ui/src/routes/operations`, `routes/workflows/run-form.tsx`, `run-drafts`:
  существующие CRUD, dialogs, labels и idempotent draft lifecycle.

Протокольные источники: [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
и [официальный Python SDK](https://github.com/modelcontextprotocol/python-sdk).
Версия SDK и фактически поддерживаемые protocol/schema/content возможности
фиксируются контрактными тестами; примеры из latest SDK не подменяют uv.lock.
