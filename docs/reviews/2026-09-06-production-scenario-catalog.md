# Каталог сценариев: назначение и использование в prod

Обновлено 2026-09-07: **16 версионных Workflow и 5 AuditProfile**. Включены
OpenAPI Streamline, ограниченный graph-only профиль @2 и профиль @3 с findings,
а также общий `findings-review@1`. Состояние развёрнутого сервера здесь
не проверяется. Примеры из `configs/e2e`, `configs/examples` и кандидаты из
`tests/eval` не входят в подсчёт.

Рекомендация: организовать пользовательский каталог вокруг задач, а варианты
исполнения и служебные Audit Workers показывать в настройках и деталях запуска.
«Нужен в prod» ниже означает продуктовую полезность. Готовность модели к
стабильному использованию требует eval; прохождение технических тестов само
по себе не доказывает качество анализа.

Workflow выполняет ограниченную работу и выпускает артефакты. AuditProfile
определяет программу из проверок: какие items проверить, какими Workflow, как
учесть evidence, gaps, находки и решения человека.

| Текущий Workflow | Вход → решаемая задача → результат | Нужен в prod и как показывать |
| --- | --- | --- |
| [openapi-from-workspace@5](../../configs/workflows/openapi_from_workspace_v5.yaml) | Source ZIP + optional OpenAPI seed → discovery, build, validate → OpenAPI, validation report, workspace state/diff | **Да.** Основной вариант действия «Построить OpenAPI». Сейчас `passthrough@1`. |
| [openapi-from-workspace-streamline@1](../../configs/workflows/openapi_from_workspace_streamline.yaml) | Те же входы и результат; в каждой из четырёх стадий planner декомпозирует работу для одного Worker | **Экспериментальный вариант того же действия.** Нужен для сравнения качества/стоимости и сложных задач; не отдельный продукт. До eval не назначать новым вариантом по умолчанию. |
| [openapi-from-analysis@2](../../configs/workflows/openapi_from_analysis.yaml) | Source + готовые dependency/project reports + optional seed → build/validate → OpenAPI и отчёт | **Да, расширенный режим.** Повторное использование анализа. Пользователь явно выбирает совместимые ревизии; каталог не доказывает свежесть отчётов относительно source. |
| [likec4-from-workspace@5](../../configs/workflows/likec4_from_workspace_v5.yaml) | Source + optional LikeC4 seed → discovery, build, validate → модель архитектуры, отчёт, state/diff | **Да.** Основное действие «Построить архитектуру», особенно для последующего security review. |
| [likec4-from-workspace-streamline@2](../../configs/workflows/likec4_from_workspace_streamline.yaml) | Та же архитектурная задача с planner внутри каждой стадии | **Экспериментальный вариант.** Те же правила выбора и eval, что у OpenAPI Streamline. |
| [likec4-from-analysis@3](../../configs/workflows/likec4_from_analysis_v3.yaml) | Source + два готовых отчёта + optional seed → LikeC4 и validation report | **Да, расширенный режим** повторного использования анализа. |
| [taint-trace-from-workspace@2](../../configs/workflows/taint_trace_from_workspace.yaml) | Source + конкретный target → trace с graph navigation и аннотациями → Markdown, state/diff | **Да, для точечного расследования.** Действие «Исследовать путь/обработчик». Для массового анализа API нужен Audit. Находки сейчас описываются в отчёте. |
| [security-analysis@2](../../configs/workflows/security_analysis.yaml) | Objective + target + authorization_scope + optional context → HTTP/Caido-анализ → evidence report | **Да, как инструмент специалиста.** Требует настроенного HTTP/Caido окружения; не является готовой цепочкой проверки каждого finding. |
| [findings-review@1](../../configs/workflows/findings_review.yaml) | Самодостаточная ZIP-подборка findings/evidence → анализ через list_findings и read_artifact → Markdown report | **Да, общий анализ подборки.** Подходит для разных producer-сценариев; не является автоматическим verifier или решением о подтверждении finding. |
| [audit-openapi-operation-trace@1](../../configs/workflows/audit_openapi_operation_trace.yaml) | Trusted task package + execution manifest + source → анализ одной операции с графом → canonical result ZIP | **Служебный ограниченный вариант** для профиля @2 и сравнения operation-resolution; без findings и экспорта аннотаций. |
| [audit-openapi-operation-trace@2](../../configs/workflows/audit_openapi_operation_trace_v2.yaml) | Те же закреплённые входы → анализ операции, evidence и предложения findings → canonical result ZIP | **Да, служебный** для профиля @3. Findings публикуются через общий toolset; coverage остаётся operation-resolution. |
| [audit-source-check@1](../../configs/workflows/audit_source_check.yaml) | Те же типы Audit-входов → bounded checklist batch → canonical result ZIP | **Да, служебный**, для `source-checklist`. |
| [audit-top10-source-risk@1](../../configs/workflows/audit_top10_source_risk.yaml) | Один standard mapping + source → оценка ограниченного риска и при наличии evidence предложение finding → result ZIP | **Да, служебный**, для Top 10 Audit. |
| [audit-asvs-source-verification@1](../../configs/workflows/audit_asvs_source_verification.yaml) | Одно требование ASVS + source → оценка требования и возможное предложение finding → result ZIP | **Да, служебный**, для ASVS-пилота. |
| [artifact-copy@1](../../configs/workflows/artifact_copy.yaml) | Текст → копирование → текст | **Только dev/диагностика.** Исключить из обычного пользовательского меню prod; полезен для smoke-проверок платформы. |
| [podman-python-check@1](../../configs/workflows/podman_python_check.yaml) | Специальная Python fixture → исправить `add`, запустить `check.py` → JSON | **Только dev/диагностика Podman.** Это фиксированный тестовый сценарий. |

Девять первых строк описывают пользовательские действия и их варианты,
следующие пять — служебных исполнителей Audit, последние две — диагностику.
Это рекомендация по представлению; YAML пока остаются в штатном каталоге.

| Текущий AuditProfile | Какую задачу решает | Нужен в prod / ограничение |
| --- | --- | --- |
| [openapi-operation-trace@2](../../configs/audit-profiles/openapi-operation-trace.yaml) | По закреплённым source/OpenAPI перечисляет операции и поручает каждую graph Worker; собирает evidence и gaps | **Ограниченный вариант для сравнения.** Только operation-resolution, без findings и аннотаций; не основной security review. |
| [openapi-operation-trace@3](../../configs/audit-profiles/openapi-operation-trace-v3.yaml) | Тот же operation inventory плюс предложения findings с evidence и описанием воспроизведения | **Да, API source review.** Целевой преемник trace_annotation; перенос неполон: coverage пока operation-resolution, нет автоматического verifier и annotation output. Предложения подтверждает человек; качество требует eval. |
| [source-checklist@1](../../configs/audit-profiles/source-checklist.yaml) | Выполняет заданный JSON/YAML checklist по source, до двух items в Run; учитывает отдельные результаты | **Да, расширенный пользовательский сценарий.** Полезен для внутренних требований и повторяемых проверок. Результат ограничен качеством checklist и evidence contract. |
| [owasp-top10-2025-source-risk@1](../../configs/audit-profiles/owasp-top10-2025-source-risk.yaml) | Проверяет 10 ограниченных сценариев по категориям Top 10 и создаёт предложения находок | **Да, как пилот source security review.** Не обещать исчерпывающий scan. Находки подтверждает человек. |
| [owasp-asvs-5-0-l1-source-review@1](../../configs/audit-profiles/owasp-asvs-5.0-l1-source-review.yaml) | Проверяет пять выбранных требований ASVS 5.0.0 L1 по source/documentation | **Пилот, не основной полный ASVS-продукт.** В UI явно писать «5 требований». Для широкого использования расширить выбранные требования, методы и fixtures. |

Все пять текущих профилей используют один fixed-barrier round и запрещают
active checks. Это настройка этих профилей: сам Controller уже поддерживает
дополнительные роли и ограниченные последующие раунды.

Предлагаемая пользовательская группировка:

- **Документация проекта:** OpenAPI и архитектура. Готовые отчёты — режим входов;
  passthrough/Streamline — настройка исполнения.
- **Security review:** анализ операций API, проверка checklist, Top 10 review;
  ASVS — явно обозначенный пилот.
- **Расследование:** отдельный trace target и ограниченный HTTP/Caido-анализ.
- Служебные check Workflow доступны в деталях Audit и для операторов;
  диагностические примеры — в инструментах разработки.

Из старых security workflows стоит переносить переиспользуемые этапы и
продуктовые сценарии:

| Старый сценарий | Организация существующими механизмами | Нужен ли отдельный продукт / приоритет |
| --- | --- | --- |
| `trace_verify` | Отдельный Workflow «source + exact candidate + evidence → независимая проверка → структурированный verdict». Audit может назначать его выбранным находкам и учитывать результат отдельно от обнаружения. | **Нужна общая возможность.** Следует после прямых findings в API trace; не является условием их публикации. Использовать в нескольких Audits, при необходимости дать точечный запуск из finding. Вердикт модели и решение аналитика должны оставаться различными сущностями. |
| `vuln_sweep` | Audit с закреплённым checklist классов уязвимостей, включая отсутствие controls. Служебный nomination Workflow на класс; затем dedup/cap и trace/verification кандидатов. | **Нужен source review, высокий приоритет.** Хорошо дополняет API review, поскольку может искать риски вне OpenAPI. Отдельное пользовательское название `vuln_sweep` не обязательно. |
| `trace_postdiff` | Две serial Stages внутри per-operation trace Workflow: запись аннотаций → отдельный аналитик получает точные diff/state и source → предложения findings. Audit продолжает отвечать за items и общий отчёт. | **Экспериментальная стратегия trace.** Сохранить как вариант для eval; делать основным после измерения выигрыша от разделения навигации и анализа. |
| `vuln_scan_fast` | Вариант source-review Audit: discovery → широкий scan → dedup → trace/verification → опциональная динамическая проверка. Переиспользует те же Workers и контракты кандидатов, что и sweep. | **Отдельный продукт пока не нужен.** «Быстрый» режим имеет смысл после сравнения со sweep по качеству и стоимости; одного меньшего бюджета для обещания скорости недостаточно. |
| `vuln_assess` | Пользовательский API-review сценарий: подготовить/выбрать OpenAPI → trace Audit → проверки findings → при выбранном scope динамическая проверка. Обычные Workflow выпускают документы/результаты; Audit координирует множество проверок. | **Нужен как итоговый сценарий**, после trace и verification. Не добавлять новый монолитный Worker, повторяющий всю оркестрацию. |

Возможности платформы и необходимые доработки проверены по реализации:

| Механизм | Уже есть | Что требуется для перечисленных сценариев |
| --- | --- | --- |
| Serial Stages, pinned artifacts, разные Workers | [Workflow](../spec/00-workflow-and-planner.md); достаточно для локальной цепочки annotate → analyze | Новые конкретные task/result контракты и шаблоны; передавать точный diff/state. Сам Scheduler менять ради `trace_postdiff` не требуется. |
| Audit discovery/assessment phases, child Runs, recovery, barriers | [Controller](../../internal/auditcontroller/controller.go), [role submission](../../internal/auditcontroller/builder.go) | Авторство профилей/Workers и явная передача выходных артефактов. Названия ролей в schema ещё не означают готовый предметный сценарий. |
| Итерации по предложенным проверкам | [PrepareNextRound](../../internal/auditservice/next_round.go) строит следующий immutable worklist по допустимым, ещё не использованным proposed checks | Сейчас выбирается прежний `inventory.itemWorkflowRole`. Для разных scan/trace/verify Workers определить разрешённое профилем соответствие вида проверки роли; Worker не должен сам назначать произвольный Workflow. |
| Старт проверки с готовых findings | Domain inventory существует, но [ProfileCompatibility](../../internal/auditservice/compatibility.go) отвергает `finding-candidates@1`; [start](../../internal/auditservice/start.go) не обслуживает такой initial inventory | Доработать доверенный импорт точных candidate/evidence refs и начальное построение inventory для самостоятельного verification Audit. |
| Подготовка OpenAPI перед trace inventory | Обычный build Workflow и Audit с входным OpenAPI уже есть; исходный inventory строится на старте по переданным inputs | Сначала можно явно передавать результат build в новый Audit. Для единого восстанавливаемого `vuln_assess` нужна согласованная подготовительная фаза/композиция, закрепляющая build output до создания operation inventory. Нельзя считать это готовым следствием наличия discovery role. |
| Findings, review, evidence, lineage | Intake, review, общие finding/list_findings, самодостаточные ZIP-подборки и Runtime materialization уже есть | Для sweep/verify зафиксировать candidate identity, dedup с сохранением источников, типизированный verdict и связь повторной проверки с исходным candidate. Старый dedup только по `(file, CWE)` не стоит переносить без оценки потерь. |

Прямые findings уже подключены к обходу операций в профиле @3 через
`security-findings@2`, с evidence и описанием воспроизведения.
`hypothesis` и `proposed_checks` необязательны. Затем определить полный предметный
trace coverage; выделить общий verifier; добавить source-review sweep и собрать
пользовательский полный assessment. Аннотации — один из возможных артефактов
Workflow; их экспорт и индекс добавляются по потребности выбранного сценария.
Общие Finding/гипотезы, ссылки на evidence и завершение результата не должны
зависеть от наличия аннотаций или OpenAPI operation. Verifier и annotation index
не блокируют первое подключение findings. Минимальный план и формат описания
зафиксированы в [таблице переноса](2026-09-06-workflows-audits-legacy-mapping.md).
Общие tools создания/чтения и задачи V43 вынесены в
[план findings tools](2026-09-06-findings-tools-plan.md): producer получает
`finding`, аналитик — `list_findings` и чтение evidence; совмещённая роль может
получить обе операции.
`trace_postdiff` и fast scan сравнить как стратегии на тех же fixtures. Новые
модельные eval требуют отдельного frozen plan и бюджета.
[Portable format уже прошёл offline gate](portable-eval-format-readiness.md).

Этот документ фиксирует рекомендацию и текущие границы. Реализованы graph-only
и finding-producing OpenAPI Workers, общий findings reader и OpenAPI Streamline.
V43-001–005 завершены; [offline process gate](2026-09-06-findings-tools-validation.md)
не заменяет model quality eval. Пять старых security pipelines целиком не перенесены.
Точное соответствие старому каталогу находится в
[таблице переноса](2026-09-06-workflows-audits-legacy-mapping.md).
