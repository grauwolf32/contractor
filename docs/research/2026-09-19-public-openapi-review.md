# Ревью публичного OpenAPI — 2026-09-19

Проверен `api/openapi/contractor-public-v1.yaml` и его соответствие текущему
серверу, конфигурациям и клиентам. Основа проверки:
`b779dedf285f5d6645f064d423e8873b5c382ab6`. Номера строк ниже относятся к этому
состоянию. Это результаты ревью, а не изменение нормативного контракта.
Исправления в этом документе предложены, но не реализованы.

Файл структурно корректен, однако полностью соответствующим реализации его
считать нельзя. Найдены 12 замечаний: одна проблема P1, восемь P2 и три P3.
Часть ошибок находится в обработчиках или генерации клиента. Изменение одного
YAML не устранит их. Переписывать API целиком для перечисленных исправлений
не требуется.

| ID | Приоритет | Где исправлять | Проблема |
| --- | --- | --- | --- |
| R01 | P1 | Public projection, затем схема | `AuditReport.review` теряется до HTTP-ответа |
| R02 | P2 | YAML и generated clients | В Summarizer отсутствует действующее `instructions` |
| R03 | P2 | YAML и generated clients | В telemetry export отсутствует действующее `retry` |
| R04 | P2 | Схемы RuntimeConfig | Авторский запрос и сохранённый документ описаны одним union |
| R05 | P2 | Audit pagination | Разрешённый `limit=200` превращается во внутренний недопустимый 201 |
| R06 | P2 | Схема finding decision | Не описана условная обязательность полей verdict |
| R07 | P2 | Query schemas | Запрещён фильтр по метке с пустым значением |
| R08 | P2 | Схема login и документация единиц | Символьная длина пароля расходится с байтовой политикой сервера |
| R09 | P2 | Генерация Go client | Explicit `null` превращается в отсутствие patch-поля |
| R10 | P3 | Параметры Git mutations | Не описаны требования Origin/CSRF для cookie auth |
| R11 | P3 | Схема RunStatus | Лимит 1024 попыток не обеспечен для automatic retry |
| R12 | P3 | Описание Git import | Явный create precondition объявлен обязательным, но фактически необязателен |

1. **R01 — P1: потеря review блокирует подтверждение предложенного отчёта в UI.**

   `AuditReport.review` объявлен в OpenAPI:4355. Сервис получает review для
   `status=proposed` и возвращает его в `ReportProjection`
   ([service.go](../../internal/auditservice/service.go), строки 210–215, 264).
   Однако `auditReportResponse` не содержит этого поля, а `getAuditReport`
   не переносит его в HTTP-ответ
   ([audit_handlers.go](../../internal/httpapi/public/audit_handlers.go),
   строки 192–198, 781–785).

   UI показывает `ActionReviewControls` только при
   `report.data.review?.state === "pending"`; переход с `?review=...` также
   проверяет совпадение request ID
   ([detail.tsx](../../ui/src/routes/projects/audits/detail.tsx), строки 1811, 1901–1905).
   Поэтому предложенный отчёт не получает действий подтверждения на этой странице.

   Воспроизведение: настоящий HTTP-handler с fake service, возвращающим
   ненулевой pending review, выдал `200 {"status":"proposed"}` без `review`.
   Проверка существующей схемой этого не обнаруживает: поле optional.

   Исправление: передавать review через public DTO и покрыть proposed report
   контрактным сценарием. В схеме выразить обязательность review для proposed,
   сохранив поведение остальных статусов. Ошибка находится прежде всего в
   реализации; удалять поле из OpenAPI было бы неверно.

2. **R02 — P2: закрытая схема Summarizer отвергает обычные ответы каталога.**

   `WorkerSummarizerConfigBody` в OpenAPI:6169–6176 содержит
   `additionalProperties: false`, но не содержит `instructions`.
   Public projection добавляет `{ref, digest}` при наличии настроенных
   инструкций Summarizer
   ([resources.go](../../internal/config/resources.go), строки 219–232).

   Воспроизведение: текущий `configs/` загружен настоящим `config.Load`;
   сериализованные `ConfigurationResource` проверены схемой Draft 2020-12.
   **30 из 41 версии AgentTemplate не прошли проверку** из-за
   `body.summarizer.instructions`. Остальные 11 AgentTemplate, четыре
   ModelPolicy и один Gateway прошли. Пример:
   [artifact_builder.yaml](../../configs/agent-templates/artifact_builder.yaml).

   Расхождение касается list/detail configuration responses. Строгий клиент
   отвергнет соответствующий ресурс; Go typed model не сохранит неизвестное
   поле при десериализации, TypeScript-тип его не описывает.

   Исправление: добавить optional `instructions` с существующей схемой ссылки
   на инструкции, регенерировать оба клиента. Проверять реальные публичные
   проекции репозиторных конфигураций, включая optional вложенные поля.
   Снимать `additionalProperties: false` со всего объекта не требуется.

3. **R03 — P2: telemetry export retry реализован, но запрещён схемой.**

   `WorkerTelemetryExportConfig` в OpenAPI:5158–5166 закрыт и перечисляет только
   `batchSizeBytes`, `maxAttempts`, `maxPendingSpans`, `maxPendingBytes`.
   Сервер поддерживает вложенный `retry` с настройками backoff
   ([normalize.go](../../internal/runtimeconfig/normalize.go), строки 481–507;
   [telemetry_export.go](../../internal/contracts/telemetry_export.go), строка 9).
   Сохранённый canonical document передаётся в public response.

   Воспроизведение: `PreparePublication` и `Resolve` приняли
   `export.retry={"initialBackoffMilliseconds":17,"maxBackoffMilliseconds":43}`.
   Полученный документ не прошёл `RuntimeConfigDocument`: `retry was unexpected`.
   Расхождение затрагивает запрос публикации, ответы чтения и generated DTO.

   Исправление: добавить optional retry schema с фактическими bounds/defaults;
   сохранить допустимость старых документов без этого поля. Проверить запрос
   и нормализованный ответ, поскольку нормализация добавляет defaults.

4. **R04 — P2: RuntimeConfig author/read schemas смешаны.**

   OpenAPI:5199–5203 допускает `gateway` как selector string, expanded ref или
   `null`; строки 5213–5216 допускают `llmGateway: null`. Тот же
   `RuntimeConfigDocument` используется для POST на строке 2446 и для чтения
   опубликованного документа на строке 5259.

   Реальные правила иные: оба указанных null запрещены; при публикации gateway
   должен быть строковым exact selector, после разрешения сохраняется объект
   ссылки с digest
   ([normalize.go](../../internal/runtimeconfig/normalize.go), строки 335–350).
   Это соответствует
   [спецификации RuntimeConfig](../spec/07-runtime-labels-and-infrastructure-config.md),
   строке 133.

   Воспроизведение: JSON Schema принимает каждый из worker blocks
   `{"llmGateway":null}`, `{"llmGateway":{"gateway":null}}` и
   `{"llmGateway":{"gateway":{"gatewayId":"local-litellm","version":"1","digest":"sha256:<64 hex>"}}}`.
   `PreparePublication` отвергает все три. В тесте использован полный валидный
   digest, а не сокращение из примера.

   Исправление: разделить author document и resolved document schemas,
   разделяя только действительно общие компоненты. Убрать неподдерживаемые
   null-ветки gateway. Серверный parser расширять для согласования с нынешним
   слишком широким union не требуется.

5. **R05 — P2: Audit pagination ломается на объявленной верхней границе.**

   Общий параметр `Limit` в OpenAPI:3265 разрешает 1–200. Его используют
   findings, reviews и finding provenance. Их обработчики запрашивают
   `limit + 1`, чтобы определить наличие следующей страницы
   ([audit_review_handlers.go](../../internal/httpapi/public/audit_review_handlers.go),
   строки 55, 186, 339). Но service validators запрещают больше
   `MaxFindingPageSize = 200`
   ([finding_review.go](../../internal/auditservice/finding_review.go),
   строки 908–943).

   Воспроизведение: при `?limit=200` настоящие handlers в связке с реальными
   service validators возвращают HTTP 400 для всех трёх списков. Для provenance
   в тесте использован fake lookup существующего owned finding; PostgreSQL
   для доказательства расхождения лимита не нужен.

   Исправление: согласовать публичную страницу и внутреннюю дополнительную
   запись. Учесть, что последующая batch-гидратация receipts тоже ограничена
   200 ID
   ([audit_receipt_batch.go](../../internal/findingintake/audit_receipt_batch.go),
   строка 21). Простое увеличение только одного service bound недостаточно.
   Acceptance должен проверять 199/200, наличие 201-й записи и следующий cursor.

6. **R06 — P2: не описаны условные поля решения finding.**

   `DecideAuditFindingRequest` в OpenAPI:4704–4712 требует только verdict и
   rationale. Сервер требует severity при `true_positive` и запрещает её при
   остальных verdict. `duplicateTargetId` обязателен исключительно при
   `duplicate`
   ([finding_review.go](../../internal/auditservice/finding_review.go),
   строки 967–975).

   Воспроизведение: схема принимает
   `{"verdict":"true_positive","rationale":"Observed evidence"}` и аналогичный
   запрос с `duplicate`; настоящий `DecideFinding` отвергает их до SQL.

   Исправление: описать варианты запроса либо условные правила для каждого
   verdict, включая запрет неподходящих полей. Отдельно проверить, что выбранная
   форма сохраняет полезные generated types, а не только проходит validator.

7. **R07 — P2: filter schema запрещает поддерживаемые метки с пустым значением.**

   В двух query schemas `label` — OpenAPI:761 и 1633 — стоят `minLength: 3`
   и `pattern: '^[^=]+=.+'`. Они запрещают `triaged=`. При этом
   `RunMetadataLabels` явно разрешает пустое значение, а обработчик делит
   selector по первому `=` и принимает его
   ([run_handlers.go](../../internal/httpapi/public/run_handlers.go), строки 131–144).

   Воспроизведение: `GET /v1/runs?label=triaged%3D` проходит настоящий
   authenticated handler с HTTP 200, но отклоняется OpenAPI request validator.

   Исправление: разрешить пустую правую часть в обоих query schemas; учитывать
   минимальную длину `a=`. Покрыть общий и проектный список одинаковыми cases.

8. **R08 — P2: длина пароля описана в других единицах.**

   `LoginRequest.password` в OpenAPI:6406 задаёт `minLength: 12`, считая Unicode
   code points. Auth использует предел 12–1024 UTF-8 байт
   ([password.go](../../internal/auth/password.go), строки 20–31).

   Воспроизведение: пароль `пароль` содержит шесть символов и 12 байт.
   Bootstrap принимает его, настоящий login возвращает HTTP 200 и cookie,
   а request validator отклоняет запрос.

   Исправление: убрать несовместимый символьный минимум, явно описать байтовую
   политику и сохранить точную серверную проверку. При необходимости обозначить
   byte bounds расширением схемы и клиентской проверкой. Аналогично проверить
   верхнюю границу с многобайтовыми символами. Менять действующую парольную
   политику или требовать смены существующих паролей ради схемы не нужно.

9. **R09 — P2: typed Go client теряет explicit-null patch operations.**

   Nullable-ветки telemetry/httpProxy/caido в OpenAPI:5217–5228 корректны.
   Однако generated Go `RuntimeWorkerPatch` использует `*T` с `omitempty`
   ([public.gen.go](../../internal/publicclient/generated/public.gen.go),
   строки 3629–3633). Обычный `json.Marshal` не различает отсутствующее поле
   и явный null. Конфигурация генератора:
   [oapi-codegen.yaml](../../internal/publicclient/generated/oapi-codegen.yaml).

   Воспроизведение: raw document с worker `telemetry:null`, `httpProxy:null`,
   `caido:null` и planner `telemetry:null` проходит `PreparePublication`.
   Unmarshal/marshal через generated `RuntimeConfigDocument` превращает spec
   в `{"planner":{},"worker":{}}`; повторная публикация получает
   `spec.worker cannot be empty`. При наличии соседних полей возможна потеря
   clear-операции без ошибки пустого объекта.

   Исправление: сохранить три состояния — absent/null/value — в generated Go
   request types и проверить фактические bytes typed POST. Это consumer defect:
   HTTP API поддерживает clear, TypeScript сохраняет `| null`, Go raw-body
   overload позволяет отправить корректный документ. Убирать поддерживаемые
   null из OpenAPI для упрощения Go types нельзя.

   Временная регенерация с `output-options.nullable-type: true` подтвердила,
   что закреплённый oapi-codegen распознаёт текущие `oneOf`/null и создаёт
   `nullable.Nullable[T]`. Опция глобальная: меняет и другие nullable DTO,
   добавляет зависимость `github.com/oapi-codegen/nullable`, поэтому потребуется
   проверить callsites. Сборка и round-trip временного изменённого клиента
   в рамках ревью не выполнялись.

10. **R10 — P3: Git mutations не объявляют browser mutation headers.**

    `replaceGitKey`, `deleteGitKey`, `importGitArtifact`,
    `importProjectGitArtifact` в OpenAPI:106–214 не ссылаются на
    `OptionalOrigin`/`OptionalCSRFToken`, в отличие от остальных обычных mutations.
    Общая аутентификация допускает session cookie, но unsafe request с ней
    требует оба заголовка
    ([auth_handlers.go](../../internal/httpapi/public/auth_handlers.go), строки 148–153).

    Это неполное описание запроса: внешний клиент, ориентирующийся на эти
    операции, не узнаёт необходимые заголовки и может получить 403.
    Общий UI transport уже добавляет CSRF; обхода серверной защиты не выявлено.

    Исправление: добавить те же reusable parameters и описание условной
    обязательности для cookie auth. Bearer-запросам CSRF не требуется.

11. **R11 — P3: граница RunStatus в 1024 попытки не обеспечивается.**

    OpenAPI:5612–5613 задаёт `maxItems: 1024` для attempts/transitions.
    Workflow validator допускает `retry.maxAttempts=1025`
    ([workflow.go](../../internal/config/workflow.go), строки 690, 957).
    Scheduler создаёт следующую попытку до configured maximum
    ([stage_finalization.go](../../internal/scheduler/stage_finalization.go),
    строки 204, 241). Store и public handler возвращают всю историю.
    Ограничение manual resume не ограничивает automatic retry.

    Воспроизведение: настоящий config validator принял 1025; HTTP-handler с
    fake persisted history вернул 1025 attempts, которые schema отвергла.
    Запуск 1025 реальных Scheduler attempts не выполнялся.

    Исправление в рамках согласования текущего контракта: убрать
    неподтверждённый `maxItems`. Глобальную квоту истории или отдельную пагинацию
    согласовать самостоятельно; не обрезать данные молча ради validator.

12. **R12 — P3: описание Git create precondition строже реализации.**

    OpenAPI:150,196 говорит `Requires explicit create or CAS precondition`.
    Общий parser принимает отсутствие обоих заголовков как create
    ([request.go](../../internal/httpapi/public/request.go), строки 108–109).
    Git importer допускает такой запрос для отсутствующего binding
    ([importer.go](../../internal/gitimport/importer.go), строки 166–180).

    Воспроизведение parser подтвердило отсутствие обязательного заголовка.
    У существующего binding отсутствие expected revision вызывает conflict;
    обхода CAS-update здесь нет.

    Исправление: документировать фактический create default. Если явный
    заголовок действительно нужен как продуктовая гарантия, это отдельное
    ужесточение API, требующее проверки совместимости существующих клиентов.

**Проверки и границы уверенности.** YAML разобран как YAML 1.2 без дубликатов
ключей; 338 component schemas прошли meta-schema Draft 2020-12. В файле
88 paths и 111 операций с implementation marker. Все операции сопоставлены с
регистрацией маршрутов, включая шесть archive routes, регистрируемых helper.

Успешно выполнены:

```sh
make verify-public-api
go test -count=1 ./internal/httpapi/public ./internal/publicclient/...
```

Свежая генерация закреплёнными `oapi-codegen@v2.8.0` и
`openapi-typescript@7.13.0` дала файлы, побайтно совпадающие с committed Go/TS
клиентами. Дополнительно выполнены восемь временных Go overlay-тестов с
настоящими валидаторами/обработчиками и описанными выше fake dependencies;
проверены реальные configuration projections и canonical RuntimeConfig JSON.
Временные тесты не добавлялись в production checkout. Для соответствующих
дефектов в документе отделено поведение, доказанное тестом, от следствий,
установленных чтением кода.

Ревью не включало live PostgreSQL/LLM/browser end-to-end прогон. Зелёный
`verify-public-api` подтверждает проверенные fixtures; он не доказывает
соответствие всех optional fields, веток решений и граничных значений.
Существенных подтверждённых проблем в SchedulerSettings, Credentials и
Operations enums/required arrays в рамках этого ревью не найдено.

**Предлагаемый порядок работы.** Отдельным небольшим изменением исправить R01
и добавить HTTP-проверку предложенного отчёта. Независимо можно исправлять
R05 с тестами полной страницы и следующего cursor. Затем согласовать схемы
каталога и telemetry (R02/R03), простые query/auth/decision contracts
(R06/R07/R08/R10/R12), регенерируя клиентов после изменений источника.

R04/R09 целесообразно делать вместе: author/read schemas, nullable request
types и round-trip проверки на generated bytes. Этот блок пересекается с
RuntimeConfig-контрактами будущей работы по toolsets; перед реализацией нужно
сверить ветки, затрагивающие те же схемы. R11 можно закрыть локальной правкой
документируемой границы; проектирование общей политики хранения истории
выходит за рамки исправления OpenAPI.
