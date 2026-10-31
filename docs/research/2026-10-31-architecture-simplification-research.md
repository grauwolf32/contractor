# Исследование сложности Contractor

Дата: 2026-10-31. Базовая ревизия: 8e0be503189c5f069f5fa99eef00a8aa0c4c08d1. Область: Go Server, Python Runtime, UI, связанные контракты, схема хранения и поставляемые AuditProfile.

Главная возможность сокращения — пересмотреть обязательную обвязку вокруг исполнения. Ядро Artifact, Run и Allocation уже предоставляет значительную часть нужных гарантий. Поверх него Audit, Evals, редактор конфигураций и отдельные форматы добавляют собственные снимки, промежуточные представления, валидацию и восстановление. Вынесение одинаковых двадцати строк в helper полезно, но наиболее крупное сокращение получится при удалении некоторых из этих требований и представлений.

Наиболее убедительные первые кандидаты: повторная семантическая проверка собственного SARIF, полная локальная валидация конфигураций в Studio, две реализации sweep и многократно повторённый rollback подготовки Allocation. Более крупное направление — убрать внутренние Audit ZIP-пакеты и сократить протокол передачи материалов между раундами и Runs. Managed Evals заслуживают отдельного решения о составе продукта: сейчас это большая самостоятельная подсистема.

## Статус на 31 октября 2026

Пользователь 2026-10-31 разрешил пересматривать избыточные требования. Шаг 1 оформлен как [V425-030](/Users/ruslan/src/contractor/tasks/done/v425-030-allocation-prepare-rollback.yml) — completed и [V425-031](/Users/ruslan/src/contractor/tasks/done/v425-031-planner-local-tool-adapters.yml) — completed. Шаг 2 про полные выходные проверки SARIF — [V425-032](/Users/ruslan/src/contractor/tasks/v425-032-sarif-output-checks-in-tests.yml) — pending. Следующее решение пользователя: самостоятельного Sweep не должно быть; target сканируется в Audit через конкретный Workflow. Прежнее предложение объединить sweep-пути отменено. Новый дизайн — [0049](/Users/ruslan/src/contractor/docs/proposals/0049-audit-owned-target-scanning.md), реализация — [V425-033](/Users/ruslan/src/contractor/tasks/v425-033-remove-sweeps-use-audit-scans.yml) — in_progress. Последующие шаги остаются предложениями.

Во время исследования в рабочем checkout началась независимая работа V425-029 с pgx collectors. Её изменения не редактировались. Числа общего размера ниже пересчитаны по Git HEAD, чтобы они не зависели от этих незакоммиченных изменений. Основные функции из списка кандидатов относятся к базовой ревизии.

Уже выполненные упрощения не включены в предложения повторно: общий PostgreSQL test harness V425-027; общий механизм receipts и controller claims; общий проход claimpass; удаление части копий Audit plans и manifests в V62-018–021; общий ExactArtifact и contentdigest. Переход на pgx collectors V425-029 завершён к моменту оформления задач шага 1; при исходном исследовании он был in_progress.

## Масштаб и расположение кода

Подсчёт по отслеживаемым файлам Go, Python, TS, TSX, MJS и CSS. Считаются физические строки, включая комментарии и пустые строки. Generated, тесты, fixtures и явно выделенные test helper packages отделены. Это размер исходников, а не количество исполняемых инструкций.

| Область | Рукописные файлы вне тестов и generated | Тесты и fixtures с helpers | Generated |
| --- | ---: | ---: | ---: |
| internal | 177 127 | 263 839 | 91 236 |
| runtime | 68 689 | 96 213 | 0 |
| ui | 114 619 | 74 493 | 16 206 |
| Всего | 360 435 | 434 545 | 107 442 |

В UI 21 480 строк CSS. Поэтому 114 тысяч строк UI нельзя приравнивать к 114 тысячам строк алгоритмов. Generated Go client особенно велик; его размер не является основанием вручную переписывать клиент.

Некоторые семейства кода, без их тестов:

| Семейство | Строк | Состав |
| --- | ---: | --- |
| Audit в Go | 39 136 | auditdomain, auditstore, auditservice, auditcontroller, auditimport, auditbaseline, auditscan, auditpriority |
| Evals в Go | 12 722 | evaldomain, evalstore, evalservice, evalcoordinator |
| Evals в UI | 8 576 | routes/evals, включая CSS |
| Конфигурации в Go | 10 916 | internal/config вместе с resolved и auditprofile |
| Worker в Runtime | 7 265 | worker |
| Studio в UI | 6 619 | routes/studio, включая CSS |
| Audit results в Runtime | 2 834 | toolsets/audit_results |
| Два CLI sweep пути вместе | 1 019 | commands, remote и платформенные locks |
| Server sweep | 1 201 | scanservice, scanstore, scancontroller |

Это области влияния, а не обещание удалить столько строк. Например, 39 тысяч строк Audit включают нужные findings, review и coverage.

В [реестре отношений](/Users/ruslan/src/contractor/internal/persistence/table_ownership.txt:22) 76 таблиц и представлений, включая migration ledger. Из них 19 принадлежат Evals, 17 — auditstore, ещё 2 — findingintake. Их большая доля сама по себе не доказывает лишность; она показывает, где сосредоточена стоимость изменения модели.

Особенно показательно сопоставление поддерживаемого масштаба с поставляемыми сценариями. Во всех 14 файлах configs/audit-profiles задан maxRounds: 1; самый большой maxItemsPerRound и maxItemsTotal — 256. Например, [OpenAPI observe](/Users/ruslan/src/contractor/configs/audit-profiles/openapi_operation_observe.yaml:36). Между тем общий контракт допускает до 10 000 items в раунде и 100 000 в Audit: [реестр лимитов](/Users/ruslan/src/contractor/docs/spec/platform/limits.md:269). В доменных документах есть собственные ограничения формата, в частности 4096 items: [limits.go](/Users/ruslan/src/contractor/internal/auditdomain/limits.go:18).

Это не доказывает отсутствия пользовательских многораундовых Audit. Они поддерживаются кодом и тестами. Но готовые профили не демонстрируют необходимость всей многораундовой инфраструктуры и максимального масштаба.

## Что именно раздувает архитектуру

Увеличивается число представлений одного результата. Для одного логического задания могут потребоваться исходный inventory, canonical inventory, task document, task package, worklist, execution manifest, SQL item, execution membership, descriptor retained dependencies, результат Worker, импортированный результат, снимок раунда и материализованный вход следующего Run. Часть из них содержит разные необходимые сведения. Каждое дополнительное представление всё равно требует кода согласования, лимитов, проверки, восстановления и удаления.

Характерный пример — [finishInventory](/Users/ruslan/src/contractor/internal/auditdomain/inventory.go:28). Она строит task document и ZIP-пакет, worklist, execution manifest и coverage, а затем проверяет их согласованность через ValidateInventory. [BuildPackage](/Users/ruslan/src/contractor/internal/auditdomain/package.go:21) сама возвращает результат после ValidatePackage. Далее [ValidateInventory](/Users/ruslan/src/contractor/internal/auditdomain/inventory.go:161) снова открывает сгенерированные пакеты, декодирует task и сравнивает кодировки.

Система защищается в том числе от расхождения собственных только что построенных представлений. При сохранении всех этих контрактов проверки имеют смысл. Удаление ненужных представлений позволяет убрать и целый класс таких проверок.

При этом не следует считать отдельную сущность дублем только из-за похожего состояния. Например, Run может технически завершиться успешно, но его результат не пройти доменную приёмку. Различие execution outcome и accepted finding нужно сохранить.

## Функции с наиболее сомнительным соотношением объёма и пользы

Размер ниже — физические строки тела функции по синтаксическому дереву, включая комментарии и пустые строки. Число ветвлений означает количество узлов if, for, case и аналогичных конструкций, в том числе вложенных. Это диагностическая метрика, не формальная cyclomatic complexity и не доказательство дефекта.

| Приоритет | Функция | Строк тела | Ветвлений | Что пересмотреть |
| --- | --- | ---: | ---: | --- |
| Первый | [validateDraft](/Users/ruslan/src/contractor/ui/src/routes/studio/validation.ts:149) | 671 | 93 | Вторую ручную реализацию правил конфигурации |
| Первый | [validateAdvanced](/Users/ruslan/src/contractor/ui/src/routes/studio/advanced-validation.ts:43) | 387 | 62 | Локальное моделирование inheritance, escalation, workspace и связанных ограничений |
| Первый | [validateSemantics](/Users/ruslan/src/contractor/internal/findingexport/sarif/semantics.go:30) | 216 | 55 | Производственную перепроверку результата собственного renderer |
| Первый | [captureScanSweep](/Users/ruslan/src/contractor/internal/cli/scan_sweep_commands.go:245) | 211 | 49 | Второй механизм планирования и фиксации запуска рядом с Server sweep |
| Первый | [AllocationService.prepare](/Users/ruslan/src/contractor/runtime/src/contractor_runtime/allocation/service.py:228) | 355 | 29 | Семь почти одинаковых веток rollback |
| Архитектурный | [buildAcceptedDependenciesWithReader](/Users/ruslan/src/contractor/internal/auditservice/retained_dependencies.go:53) | 282 | 69 | Audit как отдельный resolver и менеджер переноса артефактов |
| Архитектурный | [PrepareBatch](/Users/ruslan/src/contractor/internal/auditcontroller/builder.go:346) | 223 | 38 | Сборку очередной упаковки задач перед обычным Run |
| Архитектурный | [PrepareNextRound](/Users/ruslan/src/contractor/internal/auditservice/next_round.go:26) | 212 | 36 | Полноту отдельного протокола допуска следующего раунда |
| Архитектурный | [materializeRoundResultsSlice](/Users/ruslan/src/contractor/internal/auditservice/round_results_materialize.go:60) | 185 | 41 | Поштучное переназначение refs и checkpoint передачи уже сохранённых результатов |
| Архитектурный | [_build_result_package](/Users/ruslan/src/contractor/runtime/src/contractor_runtime/toolsets/audit_results/packages.py:232) | 112 | 10 | Собственный внутренний архивный контейнер поверх ArtifactStore |

Слово «сомнительный» здесь относится к необходимости механизма. Это production-код с вызывающими путями. Функции не объявляются мёртвыми или ошибочными.

### Две большие функции в Studio

validateDraft и validateAdvanced вместе содержат 1058 строк тел. Они знают о Workflow, AgentTemplate, AuditProfile, Planner, execution overrides, compact strategy, слотах, графе, workspace и escalation. Значительная часть тех же решений принадлежит Server: [workflow validation](/Users/ruslan/src/contractor/internal/config/workflow.go:15), [execution overrides](/Users/ruslan/src/contractor/internal/config/execution_config.go:100), [resolved validation](/Users/ruslan/src/contractor/internal/config/resolved/validation.go:1).

При этом [кнопка Validate](/Users/ruslan/src/contractor/ui/src/routes/studio/index.tsx:240) сообщает только о local checks и прямо требует CLI для bundle references. [Export](/Users/ruslan/src/contractor/ui/src/routes/studio/index.tsx:164) тоже требует полной проверки bundle перед использованием. Тысяча строк не даёт полной гарантии валидности.

Рекомендация: оставить в браузере разбор YAML, базовые ограничения редактируемых контролов и отображение ошибок. Правила исполнения и связывания конфигураций убрать из ручной TS-реализации. Самый простой вариант продолжает пользоваться существующей полной проверкой через CLI. Если необходима полноценная проверка без CLI, потребуется узкая операция Server для проверки черновика; сейчас нельзя считать, что такая операция уже есть и остаётся только подключить UI.

Более сильный вариант — отказаться от визуального редактирования всех внутренних полей. Сохранить read-only граф и редактор YAML, а запуск типовых проверок оставить отдельным продуктовым формам. Вся Studio занимает 6619 строк, но часть Live UI остаётся полезной и не обязана удаляться вместе с авторингом.

Цена: меньше мгновенных локальных подсказок; при отказе от visual authoring — потеря части визуального редактирования. Полнота исполнения и проверка опубликованных конфигураций не обязаны ослабевать.

Проверка результата: реальные Workflow и AuditProfile проверяются единственным авторитетным validator; ошибки приходят с путём к полю; UI сохраняет черновик при ошибке и не объявляет непроверенный документ полностью валидным.

### Повторный renderer внутри SARIF validator

[Render](/Users/ruslan/src/contractor/internal/findingexport/sarif/render.go:25) валидирует входной snapshot, строит document, сериализует его, проверяет выходную схему и затем вызывает validateSemantics. Последняя заново выводит ожидаемые источники, файлы, индексы, severity, native properties, HTTP и flows из snapshot и сопоставляет с тем, что renderer только что построил.

Все вспомогательные проверки собраны в [semantics.go](/Users/ruslan/src/contractor/internal/findingexport/sarif/semantics.go:1), 494 строки. Например, rating определяется как в renderer, так и в expectedRating. validLocation и validFlows восстанавливают существенную часть логики проекции. Такой validator может находить ошибки renderer, но сам требует синхронного изменения с ним.

Уточнённая рекомендация 2026-10-31: перенести в тесты полные схемную и семантическую проверки собственного вывода. Уже есть [mutation tests](/Users/ruslan/src/contractor/internal/findingexport/sarif/projection_test.go:340), вызывающие validateSemantics, и проверка реального результата через pinned official schema в тестовом helper rendered. Компиляция и загрузка этой схемы также становятся тестовыми. Сохранить в production авторизацию, проверку недоверенного snapshot и evidence, доменные проверки при построении, лимиты diagnostics и выходных байтов. Внешний SARIF, если когда-либо появится его ingestion, требует отдельного рассмотрения; этот путь экспортирует собственный результат.

Цена: ошибки renderer, обнаруживаемые только полными выходными проверками, перестанут обнаруживаться каждым production-вызовом и должны ловиться тестами. Это изменение защитной гарантии, а не механический перенос без последствий. Pinned schema, независимые goldens, проверка семантики реального вывода и mutation tests сохраняются в тестах. FND-R202 должен измениться вместе с реализацией.

Потенциал: значительная часть 494 строк semantics.go и обвязка выходной схемной проверки могут уйти из production в test code. Это не означает удаления того же числа строк из всего репозитория. Выигрыш — меньше production-логики и меньше повторных обходов результата. Измерения ускорения не проводились.

### Удалить самостоятельный Sweep

Уточнение по решению пользователя 2026-10-31: скан конкретного target принадлежит Audit и выбранному Workflow. Прежняя рекомендация оставить Server sweep и свести CLI к нему отменена. Удаляются оба пути, без нового batch-ресурса.

[runScanSweep](/Users/ruslan/src/contractor/internal/cli/scan_sweep_commands.go:57) ведёт локальный manifest с create, resume, status. [scanservice](/Users/ruslan/src/contractor/internal/scanservice/service.go:115) и [scancontroller](/Users/ruslan/src/contractor/internal/scancontroller/controller.go:105) отдельно ведут долговечный sweep в PostgreSQL. У этой подсистемы шесть API-операций, две таблицы, собственные admission, cancel, purge и retention.

Причина сохранения двух моделей установлена по задачам: V406-004 ввела отдельный Server batch, а [V406-018](/Users/ruslan/src/contractor/tasks/done/v406-018-project-scan-audits.yml) уже перенесла Web scan и custom rules в Audit, явно исключив удаление прежних API/CLI. Пользовательское направление теперь отменяет это исключение.

Нужный путь уже существует: [web-scan@1](/Users/ruslan/src/contractor/configs/audit-profiles/web_scan.yaml:1) выбирает [audit-web-nuclei-scan@1](/Users/ruslan/src/contractor/configs/workflows/audit_web_nuclei_scan.yaml:1). Один target и точные settings/rules входят в baseline; Audit управляет проверкой, обычные Run и Allocation исполняют Workflow, результаты поступают в обычную Audit collection. Installed/custom rules — варианты одного check.

Удаление охватывает API/OpenAPI и generated clients, local/remote CLI и locks, scanservice/scanstore/scancontroller, startup/retention, sweep-only Artifact/receipt helpers и поля Run creation. PinnedWorkflow сохраняется: им также пользуется Eval. Старые дочерние Runs и их материалы сохраняют обычную историю; искусственные Audit для них не создаются.

Цена решения — прекращение независимого cross-Project массового запуска. Новый массовый запуск взамен не проектируется. Подробный разбор границ, миграции, основного каталога и проверок записан в [дизайне 0049](/Users/ruslan/src/contractor/docs/proposals/0049-audit-owned-target-scanning.md) и [V425-033](/Users/ruslan/src/contractor/tasks/v425-033-remove-sweeps-use-audit-scans.yml).

### Семь одинаковых rollback в Allocation prepare

В [prepare](/Users/ruslan/src/contractor/runtime/src/contractor_runtime/allocation/service.py:228) семь веток except вызывают один и тот же _rollback_prepare с большим списком аргументов, после чего классифицируют ошибку. Повторение находится внутри одной функции; межфайловый поиск клонов его не показывает.

Рекомендация: один общий путь rollback, отдельная классификация ошибок. Использовать существующий _rollback_prepare; общий framework ресурсов для этого не нужен. Семантику cancellation, cleanup_confirmed, fence и force exit сохранить явно. AsyncExitStack имеет смысл только если действительно уменьшит и подготовку, и владение ресурсами; автоматически заменять им установленный cleanup protocol не требуется.

Это локальное сокращение без изменения продуктовых требований. Из семи повторных блоков можно убрать существенную часть обвязки. Вся prepare при этом не станет маленькой: она действительно создаёт workspace, adapters, tools, Worker, sandbox и транспорт.

Проверка результата: сбои на каждом этапе подготовки, отмена в полёте, неподтверждённый cleanup, повторный prepare и штатный release. Rollback вызывается ровно один раз и заканчивается до освобождения allocation.

## Крупные изменения требований

### Убрать Audit ZIP из внутреннего обмена

[Task construction](/Users/ruslan/src/contractor/internal/auditdomain/inventory.go:61) упаковывает task.json в package с manifest. Runtime [строит result package](/Users/ruslan/src/contractor/runtime/src/contractor_runtime/toolsets/audit_results/packages.py:232) из check-results.json, evidence.json и evidence members. Server затем [разбирает и проверяет result package](/Users/ruslan/src/contractor/internal/auditdomain/result_package.go:21).

При этом ArtifactStore уже хранит payload, media type, exact revision и digest. Package повторно вводит идентичности members, их paths, sizes и digests, правила детерминированного ZIP и ограничения архива.

Рекомендация: для внутреннего task использовать обычный JSON Artifact; для результатов — JSON Artifact со ссылками на отдельные evidence Artifacts. Полезные доменные проверки membership, coverage, evidence и authority остаются. ZIP нужен на границе пользовательского импорта и экспорта, а не при каждом внутреннем переходе. Skills и source archives имеют самостоятельную причину быть архивами и этим предложением не затрагиваются.

Удаляемый класс кода: внутренний Audit archive codec, упаковка одиночных task JSON, повторное разархивирование, часть package IDs и member mapping, измерение ZIP overhead, часть связанных cross-language fixtures. Потребуются изменения Go, Runtime, приватных completion contracts и потребителей результатов.

Цена: одна ZIP-запись больше не обеспечивает атомарную публикацию набора evidence. Сначала публикуются immutable evidence, затем root result; до root набор не считается принятым. Незавершённые публикации должны очищаться обычной retention, а не новым доменным протоколом. Нельзя считать, что атомарность нескольких существующих Artifact-записей уже эквивалентна атомарности одного ZIP.

Первый демонстрационный срез: один обычный Audit check получает task JSON, записывает evidence и result JSON, проходит импорт, выводит finding и отчёт, затем корректно удаляется. Только после этого расширять форматы.

### Пересмотреть точное прогнозирование всех производных лимитов

[FindingRoundBudget](/Users/ruslan/src/contractor/internal/auditdomain/finding_round_budget.go:54) измеряет одновременно proposal inventory, canonical inventory, coverage, worklist, execution manifest и task packages. [jsonExtentOf](/Users/ruslan/src/contractor/internal/auditdomain/finding_round_budget.go:349) самостоятельно считает JSON nodes и depth. [measuredPackageOverhead](/Users/ruslan/src/contractor/internal/auditdomain/finding_round_budget.go:405) строит пробные task и round ZIP из пустого JSON, чтобы оценить размер контейнера. Для восстановления есть отдельный [budget checkpoint](/Users/ruslan/src/contractor/internal/auditdomain/finding_round_checkpoint.go:1).

Два файла бюджета и checkpoint занимают 630 строк. Это яркий пример кода, который нужен прежде всего из-за множества производных форматов. Построение пробных пакетов выполняется через sync.OnceValues, поэтому это не утверждение о постоянной повторной ZIP-работе на каждом item.

Рекомендация: сначала сократить число представлений и отказаться от внутреннего ZIP. Затем оставить счётчик items, ограничения отдельных inputs/evidence и проверку фактических сериализованных байтов root/pages. Не обещать заранее идеальное заполнение максимальной ёмкости каждого из нескольких форматов.

Более простой вариант: консервативный размер batch, ограничение metadata и явное отклонение слишком большого одиночного задания. Сложный нынешний вариант лучше использует бюджет и допускает большой раунд с точным воспроизводимым отбором. Выбор зависит от необходимости этого масштаба.

Цена: некоторые пограничные наборы придётся разбить раньше или отклонить. Нельзя просто выкинуть predictor и продолжать принимать все прежние предельные inputs. Ограниченность работы и памяти остаётся обязательной; меняется способ её достижения.

### Сократить Audit как вторую систему исполнения

Audit имеет собственные Round, Item, Execution, ExecutionItem, admission checkpoints, publication и staging seals, collection dispositions и при этом запускает обычные Runs. См. [состояния и типы](/Users/ruslan/src/contractor/internal/auditstore/types.go:38), [Execution](/Users/ruslan/src/contractor/internal/auditstore/types.go:574), [подготовленный раунд](/Users/ruslan/src/contractor/internal/auditdomain/prepared_round.go:22), [admission step](/Users/ruslan/src/contractor/internal/auditstore/internal/pgstore/admission.go:86).

Главный кандидат для пересмотра — не сами findings и review, а автономный протокол подготовки большого следующего раунда: чтение candidate sources по cutoff, budget checkpoint, публикация pages, staging SQL membership, seal, выборка части manifest и дальнейшая материализация closure в child Run. [prepareTypedNextRound](/Users/ruslan/src/contractor/internal/auditservice/typed_next_round.go:16) показывает цепочку явно.

Рекомендуемая простая модель для основного пути: immutable task Artifact, небольшой индекс задания со ссылкой на child Run и отдельная доменная приёмка результата. Техническое состояние child execution читается из Run. Review хранит решение и версию принимаемого результата. Раунд — граница состава и итогового обзора, а не отдельный универсальный механизм исполнения.

Вариант с наибольшим сокращением — ограниченный Audit: меньшие batches, без полного многораундового durable admission в первой поддерживаемой модели. С учётом всех 14 shipped profiles с одним раундом это заслуживает рассмотрения. Альтернатива сохраняет итеративный Audit, но упрощает только формат tasks и передачу refs. Она дороже в коде, зато сохраняет сценарии research/verification между раундами.

Цена сильного варианта: потеря части многораундовых сценариев, крупных custom profiles и точного возобновления промежуточного admission. Их нельзя объявить ненужными только по shipped configs. Новые направления pentest и hypothesis search могут нуждаться в итерациях; для них стоит выбрать несколько небольших обычных Runs с явными inputs либо сохранить ограниченный iterative coordinator.

Не следует буквально сливать AuditState и RunState. Технический успех, результат доменной приёмки, review и удаление имеют разные значения. Также labels не заменяют authority и связь owner с конкретным Run.

### Передавать артефакты через общий механизм

[buildAcceptedDependenciesWithReader](/Users/ruslan/src/contractor/internal/auditservice/retained_dependencies.go:53) расширяет граф preparation mappings, различает baseline, preparation output, accepted result и finding receipt, проверяет источники, удерживает artifacts, составляет новый manifest и считает бюджеты.

[materializeRoundResultsSlice](/Users/ruslan/src/contractor/internal/auditservice/round_results_materialize.go:60) затем читает accepted root, восстанавливает checkpoint, обходит pages, проверяет зависимости, создаёт bindings в RunScope и переписывает source/consumer mappings. [Run initialization](/Users/ruslan/src/contractor/internal/runservice/audit_input_initialization.go:50) отдельно удерживает незавершённый input materialization.

Здесь смешаны два вопроса: домен решает, какие материалы допустимы; ArtifactStore и Run должны предоставлять доступ к выбранным immutable материалам и сохранять их время жизни.

Рекомендация: домен выдаёт ограниченный список точных входов и provenance. Удержание, fork и выдача чтения Allocation выполняются одним существующим owner-механизмом Artifact/Run. Начать с одного реального caller — round-results — и проверить, можно ли убрать переписывание документов через уже существующий fork и input initialization. Если недостаточно, расширять их владельца, а не вводить отдельный Audit artifact store.

Более радикальная альтернатива — allocation-bound доступ к точным project revisions без создания дополнительных bindings для каждой ссылки. Это меняет scope и retention model и не является готовой небольшой заменой. Ref сам по себе не даёт право читать чужой Artifact. Нужно сохранить ограниченный набор разрешённых revisions, гарантию жизни данных и запрет чтения произвольных ссылок из model-authored JSON.

Текущие fork/retain операции не следует автоматически считать копированием payload bytes: binding и blob — разные уровни. Главная стоимость здесь — metadata, refs, authority и восстановление, а не доказанное физическое дублирование файлов.

### Пересмотреть Managed Evals как состав продукта

Evals сейчас включают dataset revisions, frozen plans, members, submissions, suboperations, tombstones, external registration, native checks, private truth, selected records, live projections, comparison generations, pair pages и chart aggregates. [memberReconciler.Reconcile](/Users/ruslan/src/contractor/internal/evalservice/driver.go:76) восстанавливает не только Run, но и создание Audit, старт Audit, отмену и удалённые execution. [CollectView](/Users/ruslan/src/contractor/internal/evalservice/collection.go:42) ведёт отдельный цикл материализации представлений. [projectComparison](/Users/ruslan/src/contractor/internal/evalstore/comparison_projection.go:40) сохраняет пары и агрегаты.

Для исследовательского A/B сравнения минимальная полезная версия может состоять из experiment manifest Artifact, списка обычных запусков с явными связями, результатов и отдельного scorer/report. Тогда нет необходимости переносить весь исследовательский lifecycle в основной Server.

Рекомендация: решить, нужны ли Contractor полноценные browser-managed Evals как самостоятельный продукт. Если нужны, нынешний объём нельзя честно назвать бессмысленным: он реализует эти возможности. Если главная цель — AppSec проверки и сравнение моделей при разработке, сильный кандидат — вынести experiment orchestration и comparison в отдельный инструмент, оставив Runs, artifacts, diagnostics и необходимые метки в ядре.

Это не предложение немедленно заменить всё «несколькими labels». Должны остаться учёт всех попыток, точные input/config digests, hidden ground truth вне проверяемого workspace и предотвращение повторного создания work после потери ответа.

Цена: текущие live comparison, selection/recollection, external control, browser authoring, human rubrics и восстановление managed experiment потребуют упрощения или переноса. Перемещение тех же 12 тысяч строк в другой процесс само по себе не уменьшает весь код; сокращение получится только при уменьшении поддерживаемого поведения.

Размер затрагиваемого Go и UI — около 21,3 тысячи строк без тестов. Это верхняя область пересмотра, а не оценка чистой экономии.

### Сократить язык настройки исполнения

[applyExecutionConfigPatch](/Users/ruslan/src/contractor/internal/config/execution_config.go:100) — сравнительно небольшие 77 строк тела, но она обслуживает многоуровневую модель: defaults Planner/Workers, stage overrides, logical agent overrides, отдельные model/summarizer/compactor policies, credentials, origin и escalation variants. У неё есть зеркальные правила в wire, Runtime и Studio.

Рекомендация: определить один resolved execution snapshot на Run и сократить authoring до Workflow, deployment settings и одного явного override на нужном уровне. Стоит отдельно проверить необходимость независимо публикуемого ExecutionConfig и каскада override на всех уровнях. Различие поведения Workflow и окружения RuntimeConfig полезно; объединять секреты и поведенческие настройки в один документ не требуется.

Простой вариант — новые Workflow revisions для существенно другой политики исполнения. Гибкий нынешний вариант позволяет менять отдельные параметры без размножения definitions, но переносит сложность в resolver и UI.

Цена: меньше динамического переопределения и автоматических escalation combinations. Перечислить использующиеся комбинации по shipped catalog и реальным сценариям до изменения контракта. Только количество строк resolver не отвечает на вопрос, какие варианты нужны.

## Подтверждённые дубли меньшего масштаба

### Два ADK tool adapters в Streamline

[plannerMemoryTool.ProcessRequest](/Users/ruslan/src/contractor/internal/planner/streamline/memory_tools.go:78) и [plannerStateTool.ProcessRequest](/Users/ruslan/src/contractor/internal/planner/streamline/state_tools.go:86) делают одинаковую упаковку declaration в request.Tools и GenerateContentConfig. Похожи также builders и выбор worker binding.

Здесь подходит небольшой общий adapter внутри streamline. Различия Run, validation и закрытых error vocabularies оставить владельцам инструментов. Комментарий к Memory объясняет, почему прямое использование стандартного ADK functiontool.New меняет ошибки; подмена на него не является заведомо эквивалентной.

### Два пути frozen Audit write

[ProjectArtifactAccess.PutImmutableProject](/Users/ruslan/src/contractor/internal/auditcontroller/artifacts.go:80) и [ServiceArtifactAccess.PutImmutableProject](/Users/ruslan/src/contractor/internal/auditimport/artifacts.go:238) повторяют ProjectScope, claim-bound artifact write, CreateFrozen и ExactContent.

Подходит общая операция в существующем владельце claimed artifact writes. Controller дополнительно переводит collision в ErrInvalidSubmission; это различие должно остаться явным. Не нужно ради двух небольших методов создавать новый repository layer.

### Повторное создание окружения toolset

Около двадцати фабрик имеют одинаковую широкую create_selected сигнатуру, require_selected_tools, извлечение metrics, ArtifactClient, secrets и сборку выбранных builders. Примеры: [RunArtifacts](/Users/ruslan/src/contractor/runtime/src/contractor_runtime/toolsets/run_artifacts/tools.py:63), [TextArtifacts](/Users/ruslan/src/contractor/runtime/src/contractor_runtime/toolsets/text_artifacts/tools.py:60), [Memory](/Users/ruslan/src/contractor/runtime/src/contractor_runtime/toolsets/memory/tools.py:93).

Проверки уже частично общие: [factory.py](/Users/ruslan/src/contractor/runtime/src/contractor_runtime/toolsets/common/factory.py:1). Подходит небольшой context object и локальная функция сборки только для действительно одинаковых artifact-backed factories. HTTP, sandbox, workspace и внешние adapters имеют самостоятельные lifecycles; их не следует насильно загонять в декларативный универсальный конструктор.

### Gateway URL

[parseConfiguredGatewayURL](/Users/ruslan/src/contractor/internal/config/llm_gateway.go:146) и [parseGatewayURL](/Users/ruslan/src/contractor/internal/contracts/llmgateway/llm_gateway.go:257) повторяют структурные требования к HTTP URL. Общая грамматика уместна в нижнем владельце llmgateway.

Нормализация authored management origin и строгая проверка уже канонического wire origin различаются: у первой может быть корневой slash, у второй требуется canonical path. Объединить общий разбор, не стирать различие write normalization и read validation.

### Каталог конфигураций

В [loader.go](/Users/ruslan/src/contractor/internal/config/loader.go:334) и [loadExecutionConfigs](/Users/ruslan/src/contractor/internal/config/execution_config_profile.go:58) повторяются discover, decode, envelope, duplicate identity, resolve, insert и source attribution.

Небольшой typed helper может уменьшить обвязку. Предложение низкого приоритета: filesystem discovery и decodeOne уже общие, а resolve действительно различается. Новый plugin framework с callbacks на каждую строку может оказаться сложнее нынешнего кода.

### Архивные primitives

[readBoundedMember](/Users/ruslan/src/contractor/internal/agentskills/validate.go:121) и [readPackageMember](/Users/ruslan/src/contractor/internal/auditdomain/package.go:322) повторяют bounded stream read, проверку лимита и дочитывание до EOF. Это кандидат на один низкоуровневый helper, если Audit archives сохраняются.

Package rules при этом различаются: допустимые paths, directories, manifest и формат. Общий архивный механизм не должен скрывать эти доменные различия. При принятии отказа от внутреннего Audit ZIP сначала удалить ненужный consumer, затем решать, осталось ли достаточно причин для abstraction.

## Что не стоит упрощать по одному лишь размеру

[CreatePublic](/Users/ruslan/src/contractor/internal/runservice/public.go:42), 208 строк тела и 43 ветвления, и [findingintake.Submit](/Users/ruslan/src/contractor/internal/findingintake/service.go:73), 200 и 39, велики, но находятся на границах авторизации и атомарной записи. Их размер не доказывает бессмысленность. То же относится к [validate_spec](/Users/ruslan/src/contractor/runtime/src/contractor_runtime/allocation/validation.py:33): Runtime должен проверять локальную доступность capabilities, lease, sandbox и authority, даже когда Server уже выбрал конфигурацию.

Самые длинные JSX-компоненты тоже не обязательно алгоритмически сложны: Inspector — 643 строки тела, StudioRoute — 642, EvalSetupForm — 568. В основном это формы и разметка. Разбиение на десять компонентов уменьшит файл, но само по себе не уменьшит поддерживаемую функциональность или объём кода.

Следует сохранить:

- Точную связь owner, Project, Run и Allocation. Произвольная ссылка на Artifact не даёт доступ.
- Fencing отменённых и завершённых allocations, lease epochs и ожидание работы перед cleanup.
- Idempotency в местах, где повтор создаёт новую работу, внешний вызов или изменение.
- Разделение найденного предположения, подтверждённого finding и решения человека.
- Контроль bytes, количества элементов, времени и работы. Пересматривать число лимитов и способ их доказательства можно; неограниченные обходы из этого не следуют.
- Строгую проверку внешних данных и данных на новой границе доверия. Кандидат на сокращение — повторная реконструкция своих гарантированных промежуточных значений.
- Валидацию реального PostgreSQL поведения и реальные проверки cancellation/recovery.

Общий claimpass уже намеренно не владеет политикой scheduling. Audit может сразу продолжать после изменения, Eval должен ждать poll interval, sweep работает с одним parent и своим drain. Сделать над ними один большой controller engine — риск получить ещё один слой конфигурации и callbacks ради сравнительно небольших loop bodies.

## Предлагаемая целевая модель

Основа остаётся небольшой по набору понятий:

    Project и владелец
        ├── immutable Artifact revisions и их удержание
        └── обычные Runs
              └── Stages и Allocations
                    └── model Worker или deterministic operation

    Audit
        ├── task refs и связь с обычными Runs
        ├── доменная приёмка результата
        └── findings, coverage и review

    Experiment
        └── состав Runs, результаты и сравнение как artifacts или внешний инструмент

Схема не означает, что в Run уже есть готовый универсальный parent/child batch primitive. Если конкретным Audit callers понадобится общая возможность, следует расширять существующий runservice по этим callers; удаляемый Sweep не служит основанием для нового batch primitive. Сначала зафиксировать минимальные поля и удаляемые пути. Создание универсальной системы jobs, effects, plugins и event sourcing не является необходимым следствием этой модели.

Для данных нужен один авторитетный источник и явно производные индексы. Техническое исполнение принадлежит Run/Allocation; domain acceptance — владельцу Audit/finding; payload, exact identity и удержание — ArtifactStore. У каждой новой таблицы и каждого нового checkpoint должна быть названа функция, которую существующий владелец не может дать проще.

## Последовательность изменений

| Шаг | Что сделать | Что должно исчезнуть | Как доказать результат |
| --- | --- | --- | --- |
| 1 | Общий rollback Allocation и локальные ADK adapters | Повторные блоки, без нового framework | Owner suites, cancellation и lifecycle regressions |
| 2 | Перенести полные проверки собственного SARIF в тесты | Production semantics.go и выходная schema validation | Pinned schema, goldens и mutations в тестах; input validation, construction и byte bounds при экспорте |
| 3 | Удалить самостоятельный Sweep и использовать существующий Audit scan | Оба sweep-пути, шесть API-операций, две таблицы и собственная orchestration | Audit UI/API/Run/Allocation, installed/custom rules, cancel/replay и миграция с сохранением child Runs |
| 4 | Сократить Studio validation | Ручной TS-дубль полного domain validator | UI показывает авторитетные ошибки, сохраняет YAML и корректно экспортирует |
| 5 | Один Audit check на JSON Artifact без ZIP | Task ZIP и result ZIP в этом вертикальном пути | Run, evidence, import, finding, report, restart и purge через реальные компоненты |
| 6 | Сократить inventory representations и бюджетирование | ZIP overhead, часть predictor и повторных decode/encode | Пограничные размеры, oversize item, bounded batch и отсутствие повторного запуска |
| 7 | Выбрать поддерживаемый Audit scale и iterative model | Излишние admission/staging/materialization пути в принятом варианте | Показательный пользовательский Audit, crash/restart, cancel и review |
| 8 | Принять состав Managed Evals и execution configuration | Только явно исключённые возможности и их полноценные вертикальные пути | Повторяемое A/B сравнение и основные AppSec сценарии без потерянных попыток |

Шаги 1–3 оформлены как V425-030, V425-031, V425-032 и V425-033. Для шага 3 действует новое решение об удалении Sweep целиком. Шаги 2–4 меняют гарантии или возможности; последующие ещё не оформленные шаги не ставятся в очередь автоматически.

Не начинать с переписывания всех контроллеров или массового разбиения больших функций. Сначала выбрать требование, которое перестаёт быть обязательным, и проследить удаление от API/UI до persisted fields и тестов. Сокращение нужно измерять по production LOC, числу authoritative representations, lifecycle states и обязательных переходов. Перенос файла или распределение прежнего кода по новым packages не считается результатом.

## Основания и пределы выводов

Проведены подсчёт отслеживаемых исходников, синтаксический просмотр больших функций, поиск точных и нормализованных межфайловых совпадений, ручной разбор перечисленных production callers и сопоставление с текущими contracts, task status и shipped profiles.

Размеры исходников пересчитаны по git archive HEAD. Размеры функций проверены Tree-sitter из установленного Runtime окружения; для одного стороннего по отношению к основным кандидатам файла UI parser сообщил неподдержанный синтаксис, поэтому полный рейтинг не трактуется как доказательство охвата всех возможных функций. Перечисленные основные кандидаты разобраны без ошибок parser. Поиск клонов служил для отбора, а не для оценки процента дублирования всего проекта.

Исторические reports использовались для понимания уже выполненных упрощений, не как доказательство нынешней реализации. Нет утверждения, что все перечисленные возможности никем не используются. Наличие больших feature families и сложных гарантий установлено кодом; их продуктовая ценность требует выбора поддерживаемых сценариев.

Production-код не менялся, поведенческие тесты и release gate в этом исследовании не запускались. Не измерялись latency, runtime memory, реальное распределение размеров пользовательских Audit или качество моделей. Поэтому здесь нет обещаний конкретного ускорения или сокращения репозитория на заданный процент.
