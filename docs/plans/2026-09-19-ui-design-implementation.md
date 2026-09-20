# Реализация дизайн-аудита UI — 19 сентября 2026

Пользователь подтвердил реализацию всех 32 замечаний [обзора](../research/2026-09-19-ui-design-review.md). Рабочая ветка: `feat/ui-design-review`. После проверок изменения интегрируются в локальный main и UI текущего стенда обновляется.

Основные сценарии: открыть результат Run и файл/Skill; подготовить точные входы; оценить покрытие Audit и принять решение по Finding; понять паузу очереди и действующую конфигурацию. Существующие точные версии, черновики, подтверждения изменений и границы хранения секретов сохраняются.

Задачи самостоятельны, каждая реализация фиксируется отдельным коммитом. `in_progress` ставится до изменения кода; `completed` — после критериев и проверок с полным `implementation_commit`. Финальная задача отвечает за общий прогон, браузерные доказательства и выпуск. Проверки запускаются с Node 24.20 и закреплённым pnpm 11.24; смена зависимостей не требуется.

| Задача | Область | Замечания |
| --- | --- | --- |
| [V58-001](../../tasks/v58-001-runtime-provenance.yml) | Restore Run viewing with current runtime provenance | D01 |
| [V58-002](../../tasks/v58-002-shared-layout-and-forms.yml) | Unify fields, responsive layouts and interaction states | D02, D03, D05, D14, D26, D27, D32 |
| [V58-003](../../tasks/v58-003-artifact-and-skill-reading.yml) | Put artifact and skill contents before secondary metadata | D04, D08, D18, D25 |
| [V58-004](../../tasks/v58-004-run-results-and-history.yml) | Make Run results primary and history compact | D06, D07 |
| [V58-005](../../tasks/v58-005-audit-coverage-and-review.yml) | Clarify Audit coverage and bring finding decisions into context | D09, D10, D11, D12, D29 |
| [V58-006](../../tasks/v58-006-home-and-projects.yml) | Focus Home and Project overview on current actionable work | D13, D15, D28 |
| [V58-007](../../tasks/v58-007-catalog-and-run-inputs.yml) | Improve catalog discovery and parameter authoring | D16, D17, D23 |
| [V58-008](../../tasks/v58-008-eval-workspaces.yml) | Put Eval execution results and distinguishing context first | D19 |
| [V58-009](../../tasks/v58-009-configuration-navigation.yml) | Expose active configuration and separate reading from publishing | D20, D21, D22, D30 |
| [V58-010](../../tasks/v58-010-operations-observation.yml) | Make resource history and current health scannable | D06, D24, D31 |
| [V58-011](../../tasks/v58-011-error-and-empty-recovery.yml) | Verify coherent recovery and empty states across views | D25, D26, D27 |
| [V58-012](../../tasks/v58-012-release-verification.yml) | Verify and deploy the complete design review implementation | Все 32, проверка и выпуск |
| [V58-013](../../tasks/v58-013-finding-text-and-run-actions.yml) | Restore full Finding descriptions and improve expanded history layouts | Замечания после выпуска: полный Markdown, sources, корзина, labels и метрики allocations |
| [V58-014](../../tasks/v58-014-configuration-navigation-and-credentials.yml) | Clarify configuration navigation and managed LLM credential inventory | Назначение ссылок, исключение текущего раздела, managed и development credentials |

Рекомендации применяются к существующим данным и контрактам. Для недоступных сведений о вердикте Eval или пользовательском назначении Workflow UI показывает отсутствие данных либо опубликованное описание; новое значение не выводится из успешности исполнения или технического имени. Полная система экспериментов и новый протокол toolset-конфигурации остаются в своих задачах.

Подробные критерии содержатся в task-файлах. До/после сравниваются те же наполненные данные стенда; для изменений логики добавляются целевые регрессионные проверки, визуальные изменения проверяются браузером. Базовые снимки находятся локально в `.local/ui-design-audit-20260919`.

Задачи V58-001–V58-012 выполнены; выпущен UI 0.3.0. [Результаты по каждому замечанию, сравнения и проверки](../research/2026-09-19-ui-design-implementation-results.md), [доказательства выпуска](../../tasks/evidence/v58-012.json).

После проверки пользователем добавлена V58-013, ветка `fix/ui-finding-run-history`: описания Finding показываются полностью в Markdown; ссылки на source берутся из точных входов Audit; раскрытые labels и allocation metrics занимают ширину строки; для корзины резервируется место с внутренними отступами. Отдельной структурированной ссылки на файл/строку в Finding нет, поэтому UI не выводит её из сгенерированного описания. V58-013 выполнена: затронутые экраны повторно проверены, UI 0.3.1 развёрнут на стенде. [Доказательства исправлений и выпуска](../../tasks/evidence/v58-013.json).
