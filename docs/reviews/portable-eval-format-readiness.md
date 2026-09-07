# Portable eval format: offline readiness

Дата: **2026-09-07**. V41 прошла offline gate. Формат готов к V40-001
(сопоставление покрытия fixtures) и V40-002 (подготовка bindings и frozen plan).
Качество и стоимость новых инструкций **не измерены**. Model/gateway eval,
реальные target campaigns и публикация на внешнем сервере не запускались.

## Проверенная реализация

Playground: `3e43620ee4a34ff4bb96597c2964cbcf5c8d19d9`; реализация V41-001–007
заканчивается коммитом `2ee43504ffd8e319af4803d60107b1fe002b39b9`.
V41-008 добавляет usage guide и сквозную проверку v2.
Contractor compatibility проверена в общей рабочей копии на базе
`8baf7c5596a7be9dce6c6b334421c760af4be51a`.

[JSON evidence](portable-eval-format-readiness.json) сохраняет результаты каждого
теста/подтеста, версии зависимостей и SHA-256 каждого файла схем, реализации,
fixtures и 204 входных файлов Contractor. Идентификаторы параметризованных pytest
случаев содержат хеш параметра вместо его потенциально большого/private payload.
Исходники соответствующих assertions перечислены с точными хешами.

Общие рабочие копии содержат ранее начатые изменения других задач. Поэтому
проверки из отдельного checkout и интеграционные проверки общей копии указаны
отдельно. Дополнительные 11 тестов общей копии и изменения исходников перечислены
в `shared_worktree_integration`; они не включены в коммит V41-008.

| Проверка | Результат |
| --- | --- |
| Playground, отдельный checkout; `cd evals && uv run --extra dev pytest` | **271 passed**, 0 errors/failures/skipped |
| Playground, общая рабочая копия; та же команда | **282 passed**, 0 errors/failures/skipped |
| Contractor: `go test ./tests/eval/agent_instructions ./internal/config ./tests/eval/project_workflows ./tests/eval/audit_programs -skip '^TestLive'` | **4 packages passed, 352 теста/подтеста passed**, 0 failed/skipped |
| Ruff для добавленных файлов публикации/CLI/client/parity | Passed |

`TestLive*` исключены явным шаблоном команды. Обязательные offline случаи
выполнены, prerequisite skips не засчитаны как готовность.
Python `3.13.14`; httpx `0.28.1`, jsonschema `4.26.0`, PyYAML `6.0.3`,
pytest `8.4.2`, ruff `0.16.6`.

## Точные pins

| Объект | SHA-256 |
| --- | --- |
| Canonical spec 26 и её retained copy | `df496ffa4ef7d7eea9260c59505c49b4976dbb1e16d9613585fc73d026311f7a` |
| Набор 10 portable schemas | `9146f8d60468f72c470993b75e0e27626f579cae5f1d3fe5de3b3c8583493b88` |
| Набор файлов реализации/package/lock | `c474dd94f6ac06941a784ee04656df5531587db80aa2860db5d6316ebb0dbafc` |
| Contractor adapter build в отдельном checkout | `33b65b64a109d51dde7119c77ddb0df076b971ad05fdfd58e1f25095e09764c1` |
| Contractor compatibility inputs | `56d48887680460acbb63da3e42e54118142928a14d1e6d78266de31a24a2e49c` |

Каждый aggregate digest считается от UTF-8 JSON отображения
`relative_path → sha256:<exact file bytes>`, с сортировкой ключей и compact
separators. В evidence сохранены сами отображения. Build digest adapter/scorer
считается соответствующей реализацией, включая её declared dependencies/schema.
Все восемь scorer pins находятся в
`committed_portable_implementation.scorers`: findings, OpenAPI, LikeC4, trace-file,
taint-diff, Audit coverage, report structure и evidence support.

## Проверенные границы и сбои

| Сценарий | Воспроизводимая проверка в playground/evals/tests |
| --- | --- |
| Все схемы/examples, closed fields, unknown versions, дубли/aliases/oversize, точные bytes/ref pins | `test_formats.py`, retained `fixtures/portable-v2` |
| v1 bytes и чтение старых результатов; явная конверсия без придуманной v2 membership | `test_compat.py`, legacy catalog/result/scorer tests |
| Один неизменный Case v2 и один pin `openapi-coverage@2` через Contractor и recorded | `test_readiness.py`: равные output refs и Judgment, hidden sentinel отсутствует в запросах и JSON/Markdown projections |
| Public Workflow/Audit API, exact revisions, пагинация, все attempts, partial/unknown usage | `test_adapters.py`, `test_contractor_v2.py` |
| Lost response/crash до и после квитанции, exact replay, competing writer, неизменные deadline/token high-water marks | `test_experiment_recovery.py`, включая отдельные процессы и восемь членов |
| Изолированные target handles, потерянный create/cleanup, запрет replacement после deadline | `test_experiment_recovery.py` |
| Required error/fail/incomplete, safe/unsafe controls, N/A denominator, invalid scorer, rescore с сохранением Result | `test_assessments.py` |
| Missing/error validator и exact retained oracle receipt, evidence predicates | `test_assessments.py`; CLI doubles и записанные receipts |
| Полный denominator, receipt conflicts, missing records, unknown/partial costs, pairing scopes, frozen gates | `test_comparison.py` |
| Owner isolation, sentinel exclusion, create-only exact bytes/media, lost record/index reply, stale CAS | `test_publication.py`, Contractor API fixtures и local publisher |
| Удаление до проверки/индекса, квитанция успешной записи до verification, deleted Run evidence | `test_publication.py`: unavailability сохраняется, proof не пересоздаётся |

Worked example: A end-to-end **3/4**, B **2/4**; conditional quality **3/3** и
**2/3**. Сопоставимы три terminal pairs, две quality pairs и две token pairs.
B имеет 130 известных total tokens на **2/4** expected members. Вывод
`inconclusive`. Failed execution costs остаются в сумме; неизвестные значения
остаются null. Это проверка арифметики на записанных наблюдениях.

## Практические ограничения и V40 handoff

- Contractor и recorded — execution adapters; Contractor/local — независимые
  publishers. Live target/oracle integration в portable runner пока не поставлена:
  реализованы интерфейсы и recorded lifecycle. Реальный Caido/HTTP/CAGE требует
  отдельных bindings, service pins и разрешённого бюджета.
- Нет подтверждённых pins реальной модели/sampling, runtime/container/ADK closure,
  полного набора tool declarations/docstrings, bytes Skill packages и service
  reset/session state. Read-only preflight сообщает недоступность требуемых pins;
  authoring `variants.json` не подменяет resolved execution configuration.
- Validator wrappers фиксируют entrypoint hash. Для воспроизводимого live сравнения
  нужно отдельно зафиксировать внешние зависимости CLI/runtime. Recorded oracle
  receipt доказывает только соответствие выбранному result/submission/session.
- Evidence support проверяет заданные anchors/predicates; произвольную причинную
  корректность finding/path предстоит оценивать fixtures и adjudication в V40.
- Один локальный writer; документы до 1 MiB, до 10 000 planned members,
  1 024 pairs в detail list, 128 selected publication records в одном batch.
  Conflicting observations требуют явного разбора receipts. Распределённый writer,
  автоматический best/latest, significance и универсальный scalar отсутствуют.
- Assessment доступна через Python API; generic assessment CLI не добавлялась.
  Публикуется безопасная проекция с отдельным digest. Доступность evidence —
  timestamped snapshot; локальный private bundle остаётся authority для recovery.

Usage guide: `playground-v2/docs/portable-evals.md`, с отдельными guides для
compatibility, adapters, experiments, assessments, comparisons и publication.
[План instruction eval](../../tests/eval/agent_instructions/README.md) обновлён.
V40-001/002/003 сохраняют зависимость от V41-008. V40-003 требует frozen pilot и
отдельного confirmation plan с явными limits; предложенные в задаче числа сами
по себе не являются состоявшимся запуском. Instruction candidates остаются opt-in,
существующие production selectors и работающие Audits этим gate не изменяются.
