# V57-004: сборка WorkerResult и происхождение ограничений размера

Дата: 2026-09-19. Пользователь одобрил разделение декодирования и сборки
результата, затем явно выбрал: убрать только лишний Audit-лимит, а общий
контракт размера разобрать отдельно.

## Изменение

`_build_runtime_result` остаётся общим входом для JSON от ordinary finalizer
и terminal summarizer. `_decode_model_result` проверяет размер входа и JSON;
`_validate_result_fields` проверяет текст, схему и ожидаемый subtask ID.
`_assemble_runtime_result` применяет Runtime policy и формирует WorkerResult
с request-owned identity и допустимыми artifact refs.

Trusted completion после collector/publisher gate передаёт свежие поля в
валидацию и сборку напрямую. Он больше не создаёт `json.dumps` wrapper ради
обратного `json.loads`. ID берётся из запроса; переданный completion-объект
не назначает observations, summarized или имена выходных slots. Его verified
artifact values дополняют refs, наблюдавшиеся в текущем invocation, с прежним
приоритетом опубликованной ревизии. Общая сборка сохраняет reserved bindings,
exporter-owned slots, проверку известного секрета и итоговый encoded bound.

Обе ветки передают результат свежей валидации в синхронную сборку без `await`
между ними. Существующий mutable `WorkerResult` не используется как бессрочное
доказательство корректности его полей. Unchecked model construction не введён.

Обычный tool-free LLM finalizer, exact-copy comparison, summarizer, budgets,
workspace export, cancellation и terminal State сохраняют свои места в
pipeline. Exact-copy mismatch по-прежнему проверяется после успешной сборки;
ошибки decoder по-прежнему возвращаются как WorkerFailure, включая прежний
перевод в summarizer failure codes.

## Единственное выбранное изменение принимаемых данных

Раньше `json.dumps` использовал `ensure_ascii=True`, а длина временной строки
сравнивалась с 256 KiB. Теперь trusted completion проверяется по размеру текста
и фактического результата. Прежний искусственный лимит не воспроизводится.

Проверяемый пример: строка из 44 000 символов U+007F имеет 44 000 UTF-8 bytes,
но synthetic wrapper с `subtaskId="0"` занимает 264 032 bytes. Итоговый
WorkerResult без артефактов занимает 44 145 bytes. Раньше такой typed result
отклонялся из-за wrapper; теперь проходит. Если те же данные поступают именно
как oversized JSON от модели, декодер по-прежнему отклоняет их.

Реальный Audit publisher возвращает короткий фиксированный ASCII summary,
например `Recorded and published results for 1 assigned Audit items.`
Содержимое отчёта хранится в опубликованном ZIP. Его обычный результат,
публикация и число модельных вызовов от этой правки не меняются. Измеренного
ускорения Run задача не заявляет.

## Откуда взялись 256 KiB

История репозитория показывает несколько отдельных применений числа. В
проверенных задачах, спецификациях и commit messages не найден расчёт или
benchmark, обосновывающий именно 256 KiB. Это бюджет Contractor, а не предел
ADK или A2A.

| Ограничение | Введение / перенос |
| --- | --- |
| Python result JSON | `a565c242`, MVP-012, 2026-08-29: MAX_STAGE_RESULT_JSON_BYTES |
| Go Planner result payload | `aacda4b3`, V1-006, 2026-08-30: maxStagePayloadBytes; V17-003 сохранил предел при переходе к WorkerResult |
| Workspace-exported result | `4d39a692`, V11-010, 2026-09-01: MAX_EXPORTED_RESULT_JSON_BYTES; Runtime стал использовать общую константу exporter |
| WorkerCompletion в Go/Python | `b939133f`, V17-001, 2026-09-02: отдельный bound всего completion |
| Вход LLM finalizer | `6631c606`, V21-001, 2026-09-04: предел явно записан в scope задачи и spec14 |
| Synthetic Audit wrapper | `bd36ef6b`, V39-005, 2026-09-07: повторное использование существующего decoder; отдельного решения ограничивать именно временный wrapper не найдено |
| Именованные Go result limits | `79d30099`, V45-012: разделение имён прежних констант без изменения значений |

## Что осталось ограниченным

| Данные | Текущий предел | Владелец |
| --- | --- | --- |
| Текст Worker result | 64 KiB UTF-8 | Python/Go contracts |
| JSON модельного результата | 256 KiB | Runtime model decoder |
| Вход ordinary finalizer | 256 KiB JSON | V21 / finalizer |
| Собранный WorkerResult, в том числе после workspace export | 256 KiB JSON | Runtime / exporter; отдельная проверка Planner |
| Полный WorkerCompletion | 256 KiB JSON | Python и Go wire contract |
| StageResult | 256 KiB | Go Stage contract |
| Внешний A2A response/DataPart | 1 MiB | Planner A2A client; это не разрешённый размер внутреннего completion |

Повышение общего лимита требует согласовать Runtime, exporter, Python/Go
WorkerCompletion, Planner и StageResult, включая накладные расходы envelope.
В V57-004 эти значения не менялись. Большие Audit-данные передаются через
артефакты: Audit ZIP имеет отдельный предел 16 MiB, общий Artifact transport —
64 MiB. Это другие контракты, не увеличивающие допустимый текст WorkerResult.

## Проверка

Именованные новые/усиленные регрессии, выполненные команды и результаты
записаны в [evidence](../../tasks/evidence/v57-004.json).
Тесты отдельно проверяют прежние model errors, typed handoff, fresh field
validation, снятие только synthetic bound, authority артефактов, actual output
bounds и настоящий Audit runner с continuation и verified publication.

Исходники: [Runtime](../../runtime/src/contractor_runtime/worker/runtime.py),
[Audit completion](../../runtime/src/contractor_runtime/toolsets/audit_results/completion.py),
[Python result contracts](../../runtime/src/contractor_runtime/contracts/worker.py),
[Python limits](../../runtime/src/contractor_runtime/contracts/base.py),
[Go completion](../../internal/contracts/worker_completion.go),
[Go result limits](../../internal/contracts/result_limits.go),
[exporter](../../runtime/src/contractor_runtime/projectfs/exporter.py).
