# Корректировка генерации гипотез: инструкция исполнителю

Дата: 2026-11-05. Это план реализации, не отчёт о готовности.

Обязательный контракт:
[Гипотезы как результат обычного Workflow](../proposals/0054-workflow-hypothesis-production.md).
База исследования кода: `codex/audit-hypothesis-generation`,
`737ff0319d40658a17155edc73f4d782234dbccd`.
Исходная реализация: V64-013, `6696978451c09d90bfd1249096b2bedb804eb2dc`.
Статус этой завершённой задачи не является доказательством нового контракта.

## Порядок

1. [V64-033](../../tasks/done/v64-033-workflow-material-sets.yml): общий способ передать
   ограниченный плоский набор exact материалов и реальный caller в начальном
   запуске hypothesis Workflow. Закрыть небольшой end-to-end путь первым.
2. [V64-034](../../tasks/v64-034-workflow-hypothesis-production.yml): источники,
   созданные/выбранные во время Run, основания гипотез, общий proposal tool и
   адаптация continuation. Удалить оставшиеся `source_XX` пути.
3. Зависимые V64-008/010/011 используют новый контракт. Нельзя считать базовый
   Audit готовым лишь по старым результатам тестирования V64-013.

Обе задачи pending. Этот пакет задаёт работу; он не запускает реализацию,
не изменяет production code и не разрешает demo deployment. Выполнять по одной
задаче на commit, с её production caller и проверками. Новые prerequisite вне
этого объёма сначала оформить и объяснить, не прятать внутри реализации.

## Карта кода

| Место | Требуемое действие |
| --- | --- |
| `internal/auditservice/continuation.go` | Убрать построение `source_%02d`. Разделить решение о запуске и подготовку входов по контракту Workflow; сохранить ограниченный курсор и identity |
| `internal/auditstore/internal/pgstore/continuation*.go` | Проверить bounded seeks/непотерю хвоста и исключение собственных решений по trusted origin, а не имени output |
| `internal/runservice/audit_tasks.go`, общая сборка Run inputs, `internal/artifacts/` | Материализация exact коллекций у общего владельца; локальные bindings и descriptor view, проверка доступа, replay и deletion fences |
| `internal/contracts/auditcontract/`, `api/v1alpha1/`, shared fixtures | Выбранные основания, local handles и exact system-derived links; общая форма набора у общего владельца |
| `internal/auditservice/proposals.go` | Принимать исходные inputs и допустимые exact artifacts самого producer Run; удерживать основания, атомарно принимать hypotheses + optional work |
| `runtime/src/contractor_runtime/toolsets/audit_results/proposals.py` | Расширить существующий tool; runtime assembly, configured context role, CAS и точные источники; не добавлять direct registry writer |
| `runtime/.../security_findings/project_reader.py`, `internal/findingintake/allocation_reader.go` | Переиспользовать существующий select/publish и его revision/authority contract; не писать новый collector |
| `internal/auditcontroller/controller.go`, `tasks.go`, `internal/auditimport/tasks.go` | Одинаково классифицировать permanent errors при dispatch/continuation/collection, прекращать retry deterministic rejection |
| `configs/workflows/audit_hypothesis_generation.yaml`, `configs/audit-profiles/hypothesis_research.yaml` | Удалить 16 необязательных slots, объявить обычный набор и нужные tools; сохранить реальный shipped caller |
| `configs/agent-templates/audit_hypothesis_generator.yaml`, `configs/instructions/audit-hypothesis-generator.md` | Объяснить выбор оснований, создание предложений и отличие от публикации; не обещать немедленный Hypothesis ID |
| `docs/spec/audits/common-contract.md` ACT-R7, `execution.md` AUX-R112, `platform/limits.md` LIM-ACT-7 | Удалить магические slots и монополию Task slice на контекст; оставить один источник правил/лимитов |
| Новая `docs/spec/audits/hypotheses.md` + `docs/spec/README.md` | Перенести согласованное поведение производства гипотез из proposal после реализации; сослаться на ACT-R16 и владельцев артефактов/Findings |

Указанные пути — ориентиры исходного head. Проверить sibling paths через `rg`
перед правкой; после интеграции main расположение может измениться. Публичные
изменения требуют OpenAPI и обоих generated clients, а не ручной правки generated.

## Замечания предыдущего ревью, которые должны закрыться

- `internal/auditcontroller/controller.go:387`: service `ErrInvalid` из
  continuation не распознаётся как постоянный. Валидная по старому каталогу
  binding без `source_01` оставляет Audit active на повторных ticks. Новый
  resolver должен отклонять несовместимую конфигурацию заранее; runtime/store
  permanent rejection тоже обязан завершаться один раз. Удаление slots само
  по себе не доказывает правильность всех sibling error paths.
- `internal/auditservice/continuation.go:40`: исключение источника по
  `Outputs["proposals"]` обходится mapping `material: proposals`. Следующее
  решение может потребить свой output без внешней новой работы. Доказывать
  исправление на переименованном output/alias и на trusted continuation source.
- `internal/persistence/migrations/000008_audit_continuation_seek.sql`: на
  проверенной ветке номер совпал с уже существующей в main миграцией
  `000008_runtime_snapshot_byte_order.sql`. Перед интеграцией обновить base и
  выбрать следующий свободный номер; проверить загрузку полного ledger. Не
  менять уже применённую миграцию main и не запускать миграции demo.

## Что не следует переписывать

Сохранить common Task intake, RunService, ArtifactStore forks/publication,
ControllerClaims, execution source/cursor, keyed/natural receipts, Project
Finding ownership и qualified-key deduplication. Сохранить ACT-R16 identity/
revision, paused drain, terminal Cancel и сам факт optional continuation.
Не вводить отдельную сущность generation execution, material registry,
hypothesis event bus, автоматический graph resolver или compatibility mode.

Новые источники после запуска не означают «принять любой ref из JSON»: Server
проверяет ownership/visibility/exact revision и сохраняет обычные Artifact
holds. Генератор не получает весь Project по умолчанию. Передача manifest без
создания проверенных bindings не решает задачу доступа.

## Приёмка результата

Нужны реальные PostgreSQL + Server/Runtime доказательства:

- 17 источников через один набор; 32 проходят, 33 явных не принимаются частично;
- initial generation из Project materials без предыдущих Tasks;
- выбор Finding внутри Run становится exact основанием гипотезы;
- refuted/inconclusive вместе с точной проверявшейся гипотезой доступны анализу;
- две гипотезы, одно задание, автоматическая связь, отсутствие обязательного Finding;
- producer не специальный execution type; тот же output в исследовательском Workflow;
- response loss/restart возвращают прежние IDs; empty не вызывает новый Run;
- переименованный output не запускает собственное continuation;
- конфликт выбора, permanent error, pause/resume/cancel и удаление источника;
- история 1 000+ Tasks не превращает tick в полный scan; хвост batch не теряется.

Проверки и exact команды перечислены в задачах. Не подменять эти сценарии
прохождением старых fixtures; обновить case inventory существующего gate и
отчёты release selectors. Live model quality — отдельная работа.

## Готовый текст для передачи агенту

> Продолжи корректировку реализации V64-013 по
> `docs/proposals/0054-workflow-hypothesis-production.md` и этому плану.
> Сначала выполни V64-033, затем V64-034 после закрытия зависимости.
> Ключевой результат: обычный Workflow получает настроенные материалы через
> общий Artifact/Run механизм, может выбрать дополнительные Findings существующим
> reader и предложить гипотезы с exact основаниями. Server публикует их после
> успешного Run, автоматически назначает IDs/revisions/origin и принимает
> необязательные задания через существующий owner. Удали `source_01…source_16`
> и специальное определение всего контекста через очередные Tasks. Сохрани
> bounded continuation, receipts и lifecycle. Реализуй реальные callers и
> приёмку задач; не расширяй scope до нового research framework. Перед началом
> сверь текущие main/branch, изменения других агентов и номера миграций.
