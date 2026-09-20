# V60-010 — Configuration attribution и границы delivered Evals

Review выполнен 20.09.2026 в `review/v60-deep-review`, от базы
`8edeabf21194f5bba8dd258d53a072af713c630e`. Подтверждённых production-дефектов
в этом срезе не найдено. Исправлены устаревшие delivery statements specs 06,
17 и 26: V37 UI и V38 managed Evals уже имеют completed task evidence.
Это синхронизация документации, не изменение execution или evaluation contracts.
Команды, время и пределы результата записаны в [evidence](../../tasks/evidence/v60-010.json).

## Нормативная база и source trace

Проверены specs 00, 01, 05, 07, 16, 17, 26, 28, 29, task histories V37,
V38, V40-002, V41, V55 и V61; shared Runtime wire gate и real PostgreSQL
configuration gate повторно использованы из [V60-006](2026-09-20-v60-006-runtime-review.md).
В этих подсистемах между gates нет новых production-изменений V60-010.

| Граница | Проверенный код и результат |
| --- | --- |
| Author publication → immutable catalog | `internal/config/manager.go` нормализует кандидат, проверяет весь union до mutation, различает idempotent publication и digest conflict; durable create-only link не перезаписывает существующую версию. `current.Store(next)` происходит после durable publication. Тесты покрывают restart/crash recovery, concurrent writer/reader, symlink/external change и secret-free audit. |
| RuntimeConfig authoring | `internal/runtimeconfig/normalize.go` отклоняет malformed Unicode, duplicate keys, unknown fields и reserved built-in identity. Авторский digest формируется до разрешения изменяемых Gateway references. Лимит manifest именован, разные ограничения transport/result/authoring не объединены произвольно. |
| Run creation → pinned model/credentials | `runservice/public.go` вызывает `PinRuntimeLabels` внутри той же transaction, что создаёт Run, сравнивает ожидаемые runtime/Skills digests и сохраняет snapshot. `runstore/store.go` требует специальный transaction-bound constructor. `runtimeconfig/run_snapshot.go` locks bindings и разрешает exact version/digest с transaction-bound credential lookup; Run хранит references/provenance, не secret bytes. |
| Pinning → allocation | `controlplane/placement.go` читает exact Run pins, а Agent labels закрепляет при placement. `pinReservations` locks label bindings/principals, повторно сверяет revision, compatible adapters и optimistic resolved selection перед записью. Agent labels не выдаются за Run-time snapshot. |
| Effective route → private allocation settings | `runtimeconfig/resolver.go` применяет заданный порядок default/workflow/Run labels/override/escalation/Agent labels и clone ownership; Planner не получает Worker-only layer. `scheduler/execution_preparation.go` берёт resolved reservation, проверяет model-free selection, затем раскрывает exact credential через selected Gateway; HTTP origin передаётся только HTTP Worker. Проверены omission/clear, same-layer conflict/dedup, immutable snapshots, model-free route и real DB credential locks. |
| Portable authoring → frozen experiment | Playground `experiments.py` фиксирует exact dataset/binding/scorer refs, source/config pins, matrix и порядок members; unsupported/blocked members сохраняют denominator. Preflight read-only; неизвестные pins остаются unavailable. Frozen preparation повторно воспроизводима, explicit test запрещает prepare/submit/reconcile/cancel. |
| Result → scorer/comparison | Playground `assessments.py` и comparison tests связывают scorer implementation и input evidence digests, требуют явного partial opt-in, различают execution failure/quality/error/missing outputs и не превращают отсутствующие наблюдения в success. Изменённые refs/pins и closure extras отклоняются. Private expected data исключены из execution projections. |
| Managed client | Tests используют `httpx.MockTransport`, проверяют native experiment authority и independent producer protocol. Generic labels и exported portable data не создают membership автоматически. Offline gate не доказывает реальные credentials/provider/model quality. |

Проверка не установила оснований объединять ModelPolicy, RuntimeConfig,
Workflow snapshot и Credential в один mutable объект: это разные владельцы и
моменты фиксации. Повторная validation на publication/Run/placement boundaries
защищает разные источники данных. Разделение здесь сохранено.

## Delivered, draft и отдельные владельцы

| Направление | Наблюдаемый статус и граница |
| --- | --- |
| V37-010/011/012 | Completed в Contractor main с implementation hashes в task files: Audit review, Operations layout, connected UI gate. Spec 06 больше не называет их pending. |
| V38-007/008/010 | Completed Contractor setup/comparison/release gate. Это recorded deterministic delivery; V60-010 не повторяет browser suite и не приписывает ей live quality. |
| V38-009 | Client implementation `d1f7e970e3db12a231bffca565748d1293cbed6d` находится в независимом `playground-v2`; при refresh он уже ancestor merged main `d2ffd39ff4c214cf514225a8fd097824ac676c86`. Нельзя судить о его delivery только по Contractor Git history. |
| V41 portable format | Delivered и offline-verified; исходный handoff spec digest в Playground не менялся. Исправление статуса в Contractor spec 26 не меняет copied wire/format contract. |
| V40-002 | Остаётся `in_progress`, `live_ready: false`; task и оба active worktrees не изменены. Есть recorded frozen 24-member control, но нет strict live matrix с полным набором pins. Новая проверка подтверждает offline preparation, не закрывает A1/A2. |
| V55 | V55-006/007/008 completed: scanner release, OpenAPI request set, deterministic scan planner. V55-009 ranking и V55-010 Katana pending. Review не заменяет их работу. |
| Spec 28 | Structured finding analysis/SARIF остаётся Draft/not implemented. Existing finding/collection/code-analysis contracts не считаются реализацией будущего export service. |
| V61 | Task files на review/main line остаются pending. Отдельная completed feature branch не означает интеграцию в эту baseline; OAuth/MCP/toolset implementation не входит в V60-010. |

У V40-002 сохранены явные blockers: unpublished exact wrappers/candidates;
агрегация шести programs и общего budget; unsupported source-inspection
capability; reviewed source/target/checklist mapping; trusted normalization,
media conversion и observation/review receipts; aggregation трёх A1 items
через неизменённые `batchSize=2` child Runs. Live model/revision, sampling,
tools/docstrings, Skills, effective budgets и runtime build pins unavailable.
Предложенные 90 minutes / 1 inflight / 2 million observed tokens остаются
preparation parameters; observed tokens не объявлены hard spend cap.

## Реальная verification

- `make test-config` — PASS, **22.042 s**, **428 Go PASS events**, без skips;
  config validator: 33 workflows, 41 agent templates, 4 model policies,
  1 Gateway, 0 execution configs, 11 Audit profiles, 52 instructions.
- `make verify-wire-contracts` — фактически выполнен в общей V60-006 команде
  `make verify-wire-contracts test-wire-cross-language`; **8.784 s**,
  Python 193 + 114 passes, Go 540 PASS events, без skips.
- `go test -race -count=1 ./internal/runtimeconfig` — shared V60-006 gate:
  **25.787 s**, **84 Go PASS events**, без skips; disposable PostgreSQL17.
- `go test -count=1 ./tests/eval/agent_instructions` — PASS, **0.647 s**,
  **10 Go PASS events**, без skips; exact wrapper/execution contracts сохранены.
- Playground offline selection (formats, adapters, recovery, assessments,
  comparison, instruction preparation, managed client) — **237 passed**,
  **46.892 s** wall time, без skips и реальных model/provider requests.
  После gate в чужом Playground worktree появились concurrent managed-client
  edits. Поэтому тот же набор отдельно выполнен на tracked archive exact
  `d2ffd39f`: **237 passed**, **56.148 s**, без skips. Архив создан
  внутри `.local/v60-review`, исходный worktree не изменялся.
- `git diff --check` — PASS после записи report/evidence.

Go PASS events включают parent tests и subtests; это не число независимых
leaf scenarios. Результат не утверждает whole-project correctness, live
model quality, provider availability, deployment readiness или завершение V40.
