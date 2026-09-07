# Незакоммиченные изменения main: review и план коммитов

## Интеграция выполнена 2026-09-07

Нужные изменения из исходного review сохранены отдельными коммитами.
Текущие документы синхронизированы с V41/V43 и каталогом 16 Workflow /
5 AuditProfile; исторические измерения и результаты оставлены с исходной датой.
Graph-only вариант закреплён отдельной задачей V44-001, OpenAPI Streamline —
V44-002; обе завершены с implementation refs.

| Логический блок | Коммит |
| --- | --- |
| Ignore для UI build directory и release symlink | `10127297` |
| Тест отдельного demo timeout 300s при общем default 180s | `dd293cc8` |
| Retry/rollback тест: по одной ревизии input и защищённого repeat request | `8475848d` |
| V32-005: Performance API, policy, migration 53, оба клиента и process decoder | `34d61223` |
| V32-006: Performance/allocations UI и ресурсы Run | `7c65167b` |
| Постановка V44-001/002 | `f2c93723` |
| V44-001: ограниченный graph-only профиль | `0eb47803` |
| V44-002: OpenAPI Streamline | `e8d4b203` |
| Browser tests текущих Project dialogs, подтверждения inputs и result cards | `a963ec27` |
| V32-008: release gates, benchmark harness и эксплуатационная документация | `62c28caf` |
| Завершение V44 с точными implementation commits | `fd83886b` |

Проверки после исправлений:

| Проверка | Результат |
| --- | --- |
| `CONTRACTOR_TEST_DATABASE_URL=... go test -race -count=1 -json ./...` | **56 пакетов passed, 2152 test/subtest записей passed, 0 failed** |
| Исключения общего Go-прогона | 8 пакетов без тестов; 4 явно условных внешних/live проверки skipped: pinned LiteLLM contract и три live model tests |
| UI Vitest на Node 24.20.0 | **306/306**, 51 файл |
| Typecheck, ESLint/Prettier, `generate:check` | Passed; typecheck/lint повторены после изменения browser tests |
| Static-server Node tests | **7/7** |
| Production browser stack | **29 сценариев**, включая Performance и реальные завершённые allocations; Go harness passed |
| `TestAuditProgramsAcrossProductionProcesses` | Passed, 363s; реальная DB, Server и Runtime, scripted models |
| Python resources/performance/allocation subset | **86 passed**, 18 deselected явным тематическим фильтром |
| `TestCrossLanguageMTLSAllocationLifecycle` | Passed; реальные Go/Python mTLS finalize/abort reports |
| `make test-config verify-public-api test-performance-matrix` | Passed |
| `go test -count=1 ./internal/config ./internal/agentskills ./tests/e2e` | Passed |
| Каталог в обеих таблицах и task/index references | Все 16 Workflow / 5 AuditProfile отражены; 283 задачи согласованы с индексом |

Toolchain: Go 1.25.6, Python 3.13.14, Node 24.20.0, PostgreSQL 17.11.
Database tests использовали отдельный временный контейнер и изолированные схемы.
Browser harness строил UI в отдельном checkout: локальная `ui/dist` ссылка
и demo release не изменялись. В этом checkout переиспользовались существующие
зависимости; автоматическая переустановка pnpm при запуске scripts была отключена
только в его локальном workspace config. Сам build и browser tests выполнены.
Необязательный screenshot OCR отсутствовал; обязательные browser assertions
и проверки secret boundaries завершились успешно.

Первый DB-прогон обнаружил устаревший счётчик артефактов после V37-008;
первый Audit process gate — отсутствующее `resources` в тестовом Run decoder.
Browser gate обнаружил старые селекторы после V37. Все эти случаи исправлены
и соответствующие проверки повторены; ошибки не скрывались расширением
таймаутов или отключением строгого JSON decoding.

Полный repository `release-verify`, live model quality eval и новый benchmark
campaign не запускались. Старые raw benchmark samples остаются историческим
измерением; переносимый eval format готов, но model quality/rollout выводов нет.

Единственное исходное изменение, оставленное вне коммитов:
`configs/server.local.yaml` → `gitAllowedRemotes: [github.com:443]`.
`ui/dist` теперь игнорируется. Временные test services относятся только
к этой проверке; рабочие demo services не перезапускались.

## Исходный review до исправлений

Следующий раздел сохраняет исходную оценку и раскладку 100 файлов на HEAD
`08e204c2`. Указания «ещё не закоммичено», найденные ошибки и отсутствие
задач для OpenAPI описывают этот исторический срез; итог приведён выше.


Проверено 2026-09-07 на `main`, HEAD
`08e204c2da198aac3314d111b3ade93cd2afb043`.
Исходный объём: **100 файлов — 65 изменённых tracked и 35 untracked**,
включая развёрнутые каталоги и одну символьную ссылку.
Index пуст; изменений задач среди них нет.

Большую часть изменений стоит сохранить в отдельных коммитах.
Основная реализация соответствует завершённым задачам V32, чьи коммиты
существуют в отдельной ветке, но ещё не входят в историю main.
Есть два отдельных OpenAPI-варианта из предыдущих запросов, оставшиеся
проектные документы и два локальных изменения.

Это review текущего дерева, а не подтверждение готовности релиза.
Исходные 100 файлов при проверке не редактировались; коммиты не создавались.

## Что исправить или уточнить перед фиксацией

1. **Навигация по документации противоречит выполненным задачам.**
   [docs/README.md](../README.md) называет Runtime reader для findings планируемым.
   [spec/README.md](../spec/README.md) объявляет portable eval и Runtime
   materialization ещё не реализованными.
   [agent-evals-proposal.md](../agent-evals-proposal.md) говорит, что реализация
   интеграции не начата. V41-001–008 и V43-001–005 уже завершены.
   Нужно обновить текущие статусы и сослаться на readiness/validation reports;
   модельные quality eval при этом по-прежнему не объявлять проведёнными.

2. **Старые таблицы каталога требуют явной даты/границ и обновления.**
   [Production catalog](2026-09-06-production-scenario-catalog.md) и
   [legacy mapping](2026-09-06-workflows-audits-legacy-mapping.md) перечисляют
   14 Workflow / 4 AuditProfile и ещё планируют первое подключение findings.
   В проверенном дереве 16 Workflow / 5 AuditProfile, включая уже закоммиченные
   `audit-openapi-operation-trace@2`, `openapi-operation-trace@3` и
   `findings-review@1`. Это число версионных конфигураций, не уникальных задач.
   Исторические результаты и raw benchmark samples сохранять как исторические;
   актуальное состояние добавлять отдельно.

3. **Общий Go gate уже сломан в HEAD.**
   `TestRepositoryLocalServerConfigTracksExecutableDefaults` падает и в
   рабочем дереве, и в отдельном чистом checkout того же HEAD:
   [конфиг](../../configs/server.local.yaml) задаёт 300s, а
   [default](../../internal/app/config.go) и
   [тест](../../internal/app/server_config_file_test.go) ожидают 180s.
   Изменение 300s уже закоммичено в `fc307c87`.
   Это отдельная корректировка выбранного default/локального override и
   соответствующего тестового контракта. Незакоммиченный Git allowlist
   не является причиной падения. Не включать этот дефект в перечень регрессий V32.

4. **Graph-only профиль должен иметь отдельное понятное назначение.**
   Незакоммиченный `openapi-operation-trace@2` использует новый graph Worker,
   но не создаёт findings. Уже закоммиченный `@3` умеет и graph, и findings.
   Ограниченный `@2` полезен как отдельный вариант operation-resolution и
   baseline для будущих сравнений; его не следует представлять как полный перенос
   `trace_annotation` или новый основной security review.
   Изменение файла заменяет каталоговую версию `@1` на `@2`: при необходимости
   новых запусков по старому exact selector старую конфигурацию нужно сохранить
   отдельно. Snapshot уже созданного Audit — другой контракт.

5. **В коммиты реализации не включать локальную среду.**
   `ui/dist` — ссылка на
   `../.local/demo/releases/performance-integrated-20260906/ui-dist`.
   Это артефакт локального развёртывания; существующие правила `dist/`
   не игнорируют такую ссылку. Рекомендуется отдельная правка ignore
   с `/ui/dist`, сохраняющая саму рабочую ссылку.
   Единственный dirty hunk `configs/server.local.yaml` добавляет
   `gitAllowedRemotes: [github.com:443]`: это локальная политика Git import,
   а не часть performance. Оставить в локальном overlay; общий default менять
   только как осознанное отдельное изменение.

## Предлагаемые коммиты и связь с задачами

| Группа | Что делает | Существующая задача | Решение |
| --- | --- | --- | --- |
| `v32-api`, 41 файл | Закрепляет collection policy allocation; согласует запрос с capability Runtime; добавляет API текущих метрик, истории и завершённых allocations; показывает те же ресурсы в Run detail; вводит миграцию 53 и обновляет оба клиента | [V32-005](../../tasks/v32-005-performance-api-and-allocation-history.yml), completed | Коммитить первым: `feat(performance): integrate metrics API and allocation history` |
| `v32-ui`, 20 файлов | Performance charts, polling/cancellation, completed allocations с pagination, состояния missing/stale/partial, ресурсы в попытках Run и static deep links | [V32-006](../../tasks/v32-006-operations-performance-ui.yml), completed | После API: `feat(ui): integrate Operations performance and allocation resources` |
| `v32-gate`, 14 файлов | Матрица acceptance, benchmark harness, browser сценарии, Make targets, эксплуатационные документы и измерения | [V32-008](../../tasks/v32-008-performance-release-gate.yml), completed | После API/UI: `test(performance): integrate release gates and benchmark evidence` |
| `graph`, 9 файлов | Graph Worker одной OpenAPI operation, tool allowlist, профиль @2, конфигурационные и process fixtures | Прямой предыдущий запрос; развитие [V25-008A](../../tasks/v25-008a-runnable-audit-programs.yml), смежно V15-007. Отдельной точной задачи нет | Отдельный коммит после фиксации назначения ограниченного варианта |
| `streamline`, 3 файла | Четыре OpenAPI стадии со Streamline planner; сохранены source/report/workspace/output handoffs и проверка семантического соответствия passthrough | Прямой предыдущий запрос. [V22-002](../../tasks/v22-002-likec4-workspace-streamline.yml) касается только LikeC4; V22-001 не включает новый Workflow | Стоит сохранить как экспериментальный вариант; оформить отдельную задачу и коммит |
| `design`, 7 файлов | Agent/eval proposal, legacy mapping, prod catalog, findings plan, annotation proposal и review задач | V39/V40/V41/V43 и предшествующие запросы | Коммитить после согласования текущих статусов; при желании разделить eval и findings/catalog |
| `shared`, 4 файла | README/index и тест каталога содержат изменения нескольких групп | См. ниже | Разделять по hunks, не переносить целиком в первый попавшийся коммит |
| `local`, 2 файла | Git network allowlist и ссылка на локальный UI release | Локальная среда; allowlist тематически связан с V35 | Исключить из указанных коммитов |

`docs/reviews/2026-09-06-findings-tools-plan.md` нужно обязательно сохранить:
его уже называют `context.proposal` все пять закоммиченных задач V43.
Сам документ уже отражает завершение V43; устарели соседние указатели и обзоры.

Точные задачи для двух дополнительных OpenAPI-вариантов лучше завести отдельно,
чтобы явно записать scope и проверки. Не заменять ими implementation commits
ранее завершённых V22/V25/V43.

## Как сохранить историю и не потерять интеграцию

V32-005/006/008 пришли из `v32-performance-ui` на `1519b7ad`.
Их последние implementation commits, записанные в задачах:

- V32-005: `78016cf29ec4dada507bdc45f465a902a9668092`.
- V32-006: `dfd30428dee1da1832f8504a41725fb3b04a7529`.
- V32-008: `18bf39cdaf33cb63ef60fd85617888447aed5c8d`.

Это коммиты реализации в другой ветке, а будущие коммиты main фиксируют
адаптацию к актуальному main. Существующие implementation refs не подменять
метаданными или integration commits.

Из 100 исходных файлов 48 побайтно совпадают с V32 branch, 25 присутствуют
там, но отличаются целиком, ещё 27 не относятся к такому совпадению.
Последняя группа включает и интеграционные дополнения V32, например
Go client и fixture возобновления Run; это не означает 27 посторонних изменений.

Перенос всей ветки вслепую или замена файлов версиями из неё непригодны:
в текущих файлах уже сохранены более новые изменения main, включая resume,
content capture и UI V37. Номер performance migration исправлен с занятого
`000052` на `000053`; это изменение необходимо оставить.

OpenAPI schema, сгенерированные Go/TypeScript clients и backend-проекции
должны попасть в один API-коммит. Новое обязательное поле policy в allocation
fixtures разных пакетов тоже относится к этому коммиту.

Общие файлы разделяются так:

- `configs/README.md`: graph-only profile и OpenAPI Streamline — каждый со
  своим конфигурационным коммитом.
- `internal/config/catalog_cleanup_test.go`: одна новая запись на каждый
  соответствующий Workflow.
- `docs/README.md`: performance navigation с V32; остальные указатели
  с документацией, после исправления статусов.
- `docs/spec/README.md`: статус spec 22 с V32; указатели 26/27 с актуализацией
  eval/findings документации.

## Проверки этого review

| Проверка | Результат |
| --- | --- |
| `git diff --check` | Прошла |
| `go test -json ./...` | 55 пакетов passed, один failed, восемь skipped; 1990 успешных test/subtest записей, одно падение, 138 skipped |
| Единственное Go-падение на чистом HEAD | Воспроизведено отдельно; timeout 300s против 180s |
| Полный UI Vitest | Первый прогон: 305 passed / 1 timeout в Catalog; повторный полный прогон: **306/306 passed** |
| Catalog Vitest отдельно | **9/9 passed** в dirty tree и **9/9 passed** на чистом HEAD; первое падение не воспроизвелось |
| TypeScript typecheck | Прошёл |
| ESLint + Prettier check | Прошли |
| Static-server Node tests | **7/7 passed** |
| Генерация клиентов из текущей OpenAPI schema | Go и TypeScript побайтно совпали с текущими generated files |
| Vite production build | Прошёл; вывод направлен в отдельный `/tmp`, локальный demo release не изменён |

UI-проверки выполнялись на Node 26.7.0; package engines требует 24.20.x,
и pnpm сообщает предупреждение. Build также сообщает о крупных chunks.
Это не подтверждает отдельный прогон на объявленной версии Node.

PostgreSQL/process/browser gates и benchmark campaign в этом review повторно
не запускались. Обычный Go-прогон пропускает DB prerequisites, а process cases
под build tags в него не входят. Исторические проверки с реальной DB и
21 browser case описаны в
[V32 integration report](2026-09-06-performance-v32-integration.md);
они относятся к срезу 2026-09-06, не заменяют проверку будущих integration commits.

Перед обозначением интеграции готовой к релизу нужно исправить baseline Go
failure и повторить требуемые задачами DB/process/browser gates на итоговых
коммитах, с объявленным toolchain. Полный benchmark повторять при изменении
измеряемого поведения или требований к актуальности измерений.

## Полная раскладка исходных 100 файлов

Обозначения групп соответствуют таблице выше. Числа не включают этот новый report.
У каждого исходного файла ровно одна строка; shared-файлы требуют разделения hunks.

| Файл | Группа | Связь с задачами |
| --- | --- | --- |
| [Makefile](../../Makefile) | `v32-gate` | V32-008 |
| [README.md](../../README.md) | `v32-gate` | V32-008 |
| [api/openapi/contractor-public-v1.yaml](../../api/openapi/contractor-public-v1.yaml) | `v32-api` | V32-005 |
| [configs/README.md](../../configs/README.md) | `shared` | Несколько групп; разделять hunks |
| [configs/agent-templates/audit_openapi_operation_tracer.yaml](../../configs/agent-templates/audit_openapi_operation_tracer.yaml) | `graph` | Отдельная задача отсутствует; развитие V25-008A |
| [configs/audit-profiles/openapi-operation-trace.yaml](../../configs/audit-profiles/openapi-operation-trace.yaml) | `graph` | Отдельная задача отсутствует; развитие V25-008A |
| [configs/instructions/audit-openapi-operation-tracer-worker.md](../../configs/instructions/audit-openapi-operation-tracer-worker.md) | `graph` | Отдельная задача отсутствует; развитие V25-008A |
| [configs/server.local.yaml](../../configs/server.local.yaml) | `local` | Локальная среда |
| [configs/workflows/audit_openapi_operation_trace.yaml](../../configs/workflows/audit_openapi_operation_trace.yaml) | `graph` | Отдельная задача отсутствует; развитие V25-008A |
| [configs/workflows/openapi_from_workspace_streamline.yaml](../../configs/workflows/openapi_from_workspace_streamline.yaml) | `streamline` | Отдельная задача отсутствует; аналог V22-002 |
| [docs/README.md](../../docs/README.md) | `shared` | Несколько групп; разделять hunks |
| [docs/agent-evals-proposal.md](../../docs/agent-evals-proposal.md) | `design` | V39 / V40 / V41 / V43 и исходный review |
| [docs/operations-performance.md](../../docs/operations-performance.md) | `v32-gate` | V32-008 |
| [docs/reviews/2026-09-06-agent-instructions.md](../../docs/reviews/2026-09-06-agent-instructions.md) | `design` | V39 / V40 / V41 / V43 и исходный review |
| [docs/reviews/2026-09-06-findings-tools-plan.md](../../docs/reviews/2026-09-06-findings-tools-plan.md) | `design` | V39 / V40 / V41 / V43 и исходный review |
| [docs/reviews/2026-09-06-performance-v32-integration.md](../../docs/reviews/2026-09-06-performance-v32-integration.md) | `v32-gate` | V32-008 |
| [docs/reviews/2026-09-06-performance-v32-release.md](../../docs/reviews/2026-09-06-performance-v32-release.md) | `v32-gate` | V32-008 |
| [docs/reviews/2026-09-06-performance-v32-release.yml](../../docs/reviews/2026-09-06-performance-v32-release.yml) | `v32-gate` | V32-008 |
| [docs/reviews/2026-09-06-production-scenario-catalog.md](../../docs/reviews/2026-09-06-production-scenario-catalog.md) | `design` | V39 / V40 / V41 / V43 и исходный review |
| [docs/reviews/2026-09-06-trace-annotation-contract.md](../../docs/reviews/2026-09-06-trace-annotation-contract.md) | `design` | V39 / V40 / V41 / V43 и исходный review |
| [docs/reviews/2026-09-06-unfinished-tasks-consistency.md](../../docs/reviews/2026-09-06-unfinished-tasks-consistency.md) | `design` | V39 / V40 / V41 / V43 и исходный review |
| [docs/reviews/2026-09-06-workflows-audits-legacy-mapping.md](../../docs/reviews/2026-09-06-workflows-audits-legacy-mapping.md) | `design` | V39 / V40 / V41 / V43 и исходный review |
| [docs/spec/22-performance-metrics-and-profiling.md](../../docs/spec/22-performance-metrics-and-profiling.md) | `v32-gate` | V32-008 |
| [docs/spec/README.md](../../docs/spec/README.md) | `shared` | Несколько групп; разделять hunks |
| [internal/agentskills/analysis_migration_test.go](../../internal/agentskills/analysis_migration_test.go) | `graph` | Отдельная задача отсутствует; развитие V25-008A |
| [internal/app/app.go](../../internal/app/app.go) | `v32-api` | V32-005 |
| [internal/app/performance_test.go](../../internal/app/performance_test.go) | `v32-api` | V32-005 |
| [internal/config/audit_graph_workflow_test.go](../../internal/config/audit_graph_workflow_test.go) | `graph` | Отдельная задача отсутствует; развитие V25-008A |
| [internal/config/audit_profile_test.go](../../internal/config/audit_profile_test.go) | `graph` | Отдельная задача отсутствует; развитие V25-008A |
| [internal/config/catalog_cleanup_test.go](../../internal/config/catalog_cleanup_test.go) | `shared` | Несколько групп; разделять hunks |
| [internal/config/code_analysis_repository_test.go](../../internal/config/code_analysis_repository_test.go) | `streamline` | Отдельная задача отсутствует; аналог V22-002 |
| [internal/config/workspace_streamline_workflow_test.go](../../internal/config/workspace_streamline_workflow_test.go) | `streamline` | Отдельная задача отсутствует; аналог V22-002 |
| [internal/contracts/performance.go](../../internal/contracts/performance.go) | `v32-api` | V32-005 |
| [internal/contracts/performance_test.go](../../internal/contracts/performance_test.go) | `v32-api` | V32-005 |
| [internal/controlplane/allocation.go](../../internal/controlplane/allocation.go) | `v32-api` | V32-005 |
| [internal/controlplane/placement.go](../../internal/controlplane/placement.go) | `v32-api` | V32-005 |
| [internal/controlplane/placement_postgres_test.go](../../internal/controlplane/placement_postgres_test.go) | `v32-api` | V32-005 |
| [internal/controlplane/registry.go](../../internal/controlplane/registry.go) | `v32-api` | V32-005 |
| [internal/controlplane/runtime_client.go](../../internal/controlplane/runtime_client.go) | `v32-api` | V32-005 |
| [internal/controlplane/runtime_client_integration_test.go](../../internal/controlplane/runtime_client_integration_test.go) | `v32-api` | V32-005 |
| [internal/controlplane/runtime_client_test.go](../../internal/controlplane/runtime_client_test.go) | `v32-api` | V32-005 |
| [internal/credentials/runtime_service_postgres_test.go](../../internal/credentials/runtime_service_postgres_test.go) | `v32-api` | V32-005 |
| [internal/findingintake/service_postgres_test.go](../../internal/findingintake/service_postgres_test.go) | `v32-api` | V32-005 |
| [internal/httpapi/public/fakes_test.go](../../internal/httpapi/public/fakes_test.go) | `v32-api` | V32-005 |
| [internal/httpapi/public/handler_test.go](../../internal/httpapi/public/handler_test.go) | `v32-api` | V32-005 |
| [internal/httpapi/public/openapi_contract_test.go](../../internal/httpapi/public/openapi_contract_test.go) | `v32-api` | V32-005 |
| [internal/httpapi/public/performance_contract_test.go](../../internal/httpapi/public/performance_contract_test.go) | `v32-api` | V32-005 |
| [internal/httpapi/public/performance_handlers.go](../../internal/httpapi/public/performance_handlers.go) | `v32-api` | V32-005 |
| [internal/httpapi/public/postgres_integration_test.go](../../internal/httpapi/public/postgres_integration_test.go) | `v32-api` | V32-005 |
| [internal/httpapi/public/router.go](../../internal/httpapi/public/router.go) | `v32-api` | V32-005 |
| [internal/httpapi/public/run_detail_batch.go](../../internal/httpapi/public/run_detail_batch.go) | `v32-api` | V32-005 |
| [internal/httpapi/public/run_detail_batch_postgres_test.go](../../internal/httpapi/public/run_detail_batch_postgres_test.go) | `v32-api` | V32-005 |
| [internal/httpapi/public/run_handlers.go](../../internal/httpapi/public/run_handlers.go) | `v32-api` | V32-005 |
| [internal/httpapi/public/types.go](../../internal/httpapi/public/types.go) | `v32-api` | V32-005 |
| [internal/performance/api.go](../../internal/performance/api.go) | `v32-api` | V32-005 |
| [internal/performance/api_test.go](../../internal/performance/api_test.go) | `v32-api` | V32-005 |
| [internal/performance/collector_test.go](../../internal/performance/collector_test.go) | `v32-gate` | V32-008 |
| [internal/persistence/migrations/000053_allocation_performance_policy.sql](../../internal/persistence/migrations/000053_allocation_performance_policy.sql) | `v32-api` | V32-005 |
| [internal/publicclient/generated/public.gen.go](../../internal/publicclient/generated/public.gen.go) | `v32-api` | V32-005 |
| [internal/runstore/allocation_store.go](../../internal/runstore/allocation_store.go) | `v32-api` | V32-005 |
| [internal/runstore/postgres_integration_test.go](../../internal/runstore/postgres_integration_test.go) | `v32-api` | V32-005 |
| [internal/runstore/report_store.go](../../internal/runstore/report_store.go) | `v32-api` | V32-005 |
| [internal/runstore/resume_store_test.go](../../internal/runstore/resume_store_test.go) | `v32-api` | V32-005 |
| [internal/runstore/types.go](../../internal/runstore/types.go) | `v32-api` | V32-005 |
| [internal/scheduler/scheduler.go](../../internal/scheduler/scheduler.go) | `v32-api` | V32-005 |
| [internal/scheduler/scheduler_test.go](../../internal/scheduler/scheduler_test.go) | `v32-api` | V32-005 |
| [internal/telemetry/allocation_resources.go](../../internal/telemetry/allocation_resources.go) | `v32-api` | V32-005 |
| [internal/telemetry/policy_test.go](../../internal/telemetry/policy_test.go) | `v32-api` | V32-005 |
| [internal/telemetry/postgres_integration_test.go](../../internal/telemetry/postgres_integration_test.go) | `v32-api` | V32-005 |
| [internal/telemetry/repository.go](../../internal/telemetry/repository.go) | `v32-api` | V32-005 |
| [tests/e2e/audit_programs_process_test.go](../../tests/e2e/audit_programs_process_test.go) | `graph` | Отдельная задача отсутствует; развитие V25-008A |
| [tests/e2e/audit_programs_test.go](../../tests/e2e/audit_programs_test.go) | `graph` | Отдельная задача отсутствует; развитие V25-008A |
| [tests/e2e/performance_matrix.yml](../../tests/e2e/performance_matrix.yml) | `v32-gate` | V32-008 |
| [tests/e2e/performance_matrix_test.go](../../tests/e2e/performance_matrix_test.go) | `v32-gate` | V32-008 |
| [tools/performancebench/main.go](../../tools/performancebench/main.go) | `v32-gate` | V32-008 |
| [tools/performancebench/main_test.go](../../tools/performancebench/main_test.go) | `v32-gate` | V32-008 |
| [ui/dist](../../ui/dist) | `local` | Локальная среда |
| [ui/e2e/performance.spec.ts](../../ui/e2e/performance.spec.ts) | `v32-gate` | V32-008 |
| [ui/e2e/stack.spec.ts](../../ui/e2e/stack.spec.ts) | `v32-gate` | V32-008 |
| [ui/server/static-server.mjs](../../ui/server/static-server.mjs) | `v32-ui` | V32-006 |
| [ui/server/static-server.test.mjs](../../ui/server/static-server.test.mjs) | `v32-ui` | V32-006 |
| [ui/src/api/generated/public.ts](../../ui/src/api/generated/public.ts) | `v32-api` | V32-005 |
| [ui/src/api/performance.test.ts](../../ui/src/api/performance.test.ts) | `v32-ui` | V32-006 |
| [ui/src/api/performance.ts](../../ui/src/api/performance.ts) | `v32-ui` | V32-006 |
| [ui/src/api/query-keys.ts](../../ui/src/api/query-keys.ts) | `v32-ui` | V32-006 |
| [ui/src/api/runs.ts](../../ui/src/api/runs.ts) | `v32-ui` | V32-006 |
| [ui/src/app/router.tsx](../../ui/src/app/router.tsx) | `v32-ui` | V32-006 |
| [ui/src/routes/operations/allocations/completed.tsx](../../ui/src/routes/operations/allocations/completed.tsx) | `v32-ui` | V32-006 |
| [ui/src/routes/operations/allocations/index.tsx](../../ui/src/routes/operations/allocations/index.tsx) | `v32-ui` | V32-006 |
| [ui/src/routes/operations/allocations/tabs.tsx](../../ui/src/routes/operations/allocations/tabs.tsx) | `v32-ui` | V32-006 |
| [ui/src/routes/operations/layout.tsx](../../ui/src/routes/operations/layout.tsx) | `v32-ui` | V32-006 |
| [ui/src/routes/operations/performance/chart.tsx](../../ui/src/routes/operations/performance/chart.tsx) | `v32-ui` | V32-006 |
| [ui/src/routes/operations/performance/freshness.ts](../../ui/src/routes/operations/performance/freshness.ts) | `v32-ui` | V32-006 |
| [ui/src/routes/operations/performance/index.tsx](../../ui/src/routes/operations/performance/index.tsx) | `v32-ui` | V32-006 |
| [ui/src/routes/operations/performance/performance.test.tsx](../../ui/src/routes/operations/performance/performance.test.tsx) | `v32-ui` | V32-006 |
| [ui/src/routes/operations/performance/resources.tsx](../../ui/src/routes/operations/performance/resources.tsx) | `v32-ui` | V32-006 |
| [ui/src/routes/operations/performance/series.ts](../../ui/src/routes/operations/performance/series.ts) | `v32-ui` | V32-006 |
| [ui/src/routes/runs/components.tsx](../../ui/src/routes/runs/components.tsx) | `v32-ui` | V32-006 |
| [ui/src/routes/runs/runs.test.tsx](../../ui/src/routes/runs/runs.test.tsx) | `v32-ui` | V32-006 |
| [ui/src/styles.css](../../ui/src/styles.css) | `v32-ui` | V32-006 |
