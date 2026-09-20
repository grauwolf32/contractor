# V60-006 — Runtime, completion, wire и ownership инструментов

База: `8edeabf21194f5bba8dd258d53a072af713c630e`, отдельный worktree
`review/v60-deep-review`, 20.09.2026. Проверка обнаружила один подтверждённый
P2: Go/Python по-разному считали необязательный JSON escaping в Worker result.
Коррекция вынесена в [V60-026](../../tasks/v60-026-worker-result-json-size-parity.yml)
с [отдельным разбором](2026-09-20-v60-026-worker-result-size-parity.md) и
межъязыковой матрицей. Иные production-дефекты в проверенном объёме не установлены.

Фактически выполненные команды, длительности, первоначальные ошибки и skips
хранятся в [evidence](../../tasks/evidence/v60-006.json). Этот документ не является
доказательством корректности всего проекта. Обязательные gates этого review пройдены; implementation и completion
metadata фиксируются отдельными коммитами.

## Прочитанные решения и нормативная база

Прочитаны V21-001 (`6631c606209240e2713d51c34c343b4714903303`), V57-004
(`7f553b2295051b8ad88a2cea98972727b41afaf8`), V57-005
(`6f2bc0968a439cba944cbce0fab160a13a37c89d`) и их исследования. Рабочие contracts:
specs 01, 02, 07, 10–15, 21, 25, 29. Требования scoped-сценариев сверены с
[source plan](../plans/2026-09-20-project-review.md).

- Ordinary finalizer остаётся обязательным tool-free exact-copy serializer:
  один вызов в том же Worker budget. Это принятое compatibility-решение V21,
  а не найденная лишняя модельная операция.
- V57-004 убрала только synthetic Audit JSON round trip. Декодирование actual
  model output, request-owned subtask, Runtime-owned observations/artifacts и
  реальные caps сохранены. V60-026 выравнивает способ подсчёта bytes, не числа.
- V57-005 сохраняет production A2A transport без reuse: POST retry и проверка
  certificate lifetime меняют semantics. Повторный performance experiment в
  этой проверке не проводился; существующие fault/lifecycle эксперименты
  действительно прошли в `go test -race .../planner/a2a`.
- Optional summarizer имеет свой pinned policy/budget и structured terminal
  path; V47-005 не требует восстановить снятую byte-to-token эвристику.

## Прослеженные границы и выполненные сценарии

| Граница | Source trace и проверка |
| --- | --- |
| Allocation / A2A / subtask | `allocation/service.py` проверяет snapshot, slot и spec до construction; `a2a_server.py` выбирает только активное allocation приложение и проверяет tenant. `worker/runtime.py` сериализует invocation и принимает identity из StageContentRequest. SDK round-trip/stale allocation, concurrent invocation, mismatched finalizer subtask и Go changed Task/context cases прошли. |
| Частичная preparation / release | Созданные tools передаются rollback owner до дальнейшей validation; unconfirmed close сохраняет fencing. `test_partial_tool_preparation_retains_cleanup_ownership`, `test_unconfirmed_terminal_tool_cleanup_fences_slot_and_requests_process_exit`, `test_release_timeout_retains_one_cleanup_task_and_keeps_loop_responsive` прошли. |
| Lease и stop | Watchdog использует подтверждённый monotonic lease; поздний ACK не оживляет старую allocation. `test_response_partition_expires_worker_and_late_ack_cannot_revive`, replayed ACK и unconfirmed-stop cases прошли. |
| Sessions / cleanup | `worker/sessions.py` владеет IDs до cancellable mutations, переносит только допустимый state, закрывает allocation-local app/user maps. Isolated/shared, ambiguous create/delete, bounded carry и cleanup-failure tests прошли. Cancel during cleanup не публикует success. |
| Finalizer / tool interleaving | `worker/finalizer.py` запрещает второй вызов; `runtime.py` проверяет equality и identity, callbacks считают общий budget. `test_adk_worker_does_not_start_required_finalizer_without_model_budget`, tool/token-boundary side-effect tests и `test_abort_cancels_active_result_finalizer` прошли с scripted model, без реальной модели. |
| Summarizer / Audit completion | Summarizer вызывается один раз на safe tool boundary; stop не запускает его с idle. Audit collection/publication и cancellation сохраняют ownership до cleanup. Полные runtime tests и summarizer-hardening прошли; отдельный реальный Go Artifact API bridge учитывается общим Audit gate root. |
| Typed result / artifacts | `_decode_model_result` остаётся на model-authored paths; typed Audit values заново валидируются, `_assemble_runtime_result` не доверяет model refs. Проверены fresh mutated fields, stale refs, reserved bindings, exact receipt order, workspace export-before-terminal и actual encoded bounds. |
| Go/Python / defaults / pins | Closed strict models и golden fixtures проверяют types/extras, omitted optional fields, exact refs и digests. Дополнительно выполнен весь `internal/runtimeconfig` под race: omission/clear/escalation, atomic Caido precedence, same-layer permutation/conflict/dedup, exact snapshot clone и real PostgreSQL binding locks. |
| Filesystem / stale snapshot | `projectfs/paths.py` нормализует NFC relative POSIX paths; local/storage tests проверяют traversal/symlink и isolation. `WorkspaceOperationGuard` сохраняет владельца syscall после caller cancellation. Taint annotation применяет изменение через `update_text` с сравнением current/source; cross-tool race не перезаписывает победивший edit. |
| Subprocess / graph | `toolsets/common/process.py` владеет spawn, process group, pipes и повторной cancellation; cleanup не зависит от живого leader. Реальные local subprocess tests проверяют fork descendants, proxied/unproxied validator cancel/close, bounded stdout/stderr, stripped environment, child timeout и failed reap fencing. Trailmark lifecycle tests проверяют stale snapshot rebuild и cleanup mirrors. |
| HTTP / Caido | `http/tools.py` проверяет target на каждой redirect/retry, не переносит auth/cookies на другой origin, закрывает intermediate responses и фиксирует cookies только после полной операции. Реальный локальный HTTP/Caido allocation→tools→finalize→release→reuse test и redaction/concurrency tests прошли. |
| Model-free tool@1 | `worker/tool_runtime.py` использует create-only receipt до запуска; ambiguous publication/start исключает rescan. Replay на новой allocation сохраняет exact report и нулевые новые model/tool calls. Real A2A + controlled scanner, timeout/cancel, changed input и report failures прошли. |
| Real Podman | Из 27 opt-in skipped cases общего pytest все выполнены дополнительно: supervisor/owner/execution/probe gate — 52 passes; release/deployment — 6 passes. Проверены реальные rootless processes, cgroup limits, fork/double-fork/session descendants, expiry/runtime death, owner recovery и removal. |

## Подтверждённый finding

**V60-026, P2.** `WorkerCompletion.Validate` пересчитывал размер через HTML-escaped
Go `json.Marshal`, тогда как Python считает compact UTF-8 JSON. Документ в
65 769 bytes с результатом из 65 536 `<`, `>` или `&` Python принимал,
Go отвергал после увеличения временной копии до 393 449 bytes. У Pipeline
были ещё два аналогичных result-size consumers в Planner.

До исправления shared Go matrix падала на трёх HTML случаях и U+2028 около
максимального envelope. После исправления 18 shared Go/Python cases совпадают,
включая actual control overflow и 262 144/262 145-byte границу. Исходные три
документа повторно приняты Go. Numeric caps и approved finalizer/transport
semantics не менялись.

## Выполненные gates и ограничения

`make test-runtime-hardening`: **2389 passed, 34 skipped** на исходном runtime.
После добавления 18 shared cases изменённый contract test-файл полностью
выполнен wire gate. Runtime production-код в этой коррекции не менялся.
`make verify-wire-contracts test-wire-cross-language`: Python **193** и **114**
passes; `make test-worker-summarizer-hardening`: Python **92** passes, Go race
packages и matrix прошли. `make test-worker-session-modes-hardening`: Python
**193 + 114 + 266** passes и Go packages без skips, wall time **274.703 s**.
Все обязательные команды V60-006 завершены. Shared `make test-audit-completion-e2e`
на source `45b31abfd848739a0a1b32d7027028989a911a25` также прошёл: **173 Go cases,
331 Runtime cases**, без selected skips. Матрица и JUnit проверены повторно;
`executed.json`, Go log и Runtime reports сохранены с SHA-256 в evidence.

34 skips общего pytest разобраны явно:

- 27 real Podman opt-ins выполнены дополнительно без skips. Первая попытка
  с образом `contractor-sandbox` была неверна: required supervisor label
  отсутствовал. Она дала 12 failures/40 passes и сохранена в evidence.
  Повтор использовал проверенный preinstalled immutable digest
  `localhost/contractor-supervisor@sha256:faea9fa41ec757ad3c18bf3a947409692f2fcc374a87c1d209185175b0d215ca`:
  52 + 6 passes, без pull/build и без вмешательства в чужие resources.
- 1 Audit Artifact API bridge выполнен общим `make test-audit-completion-e2e`
  на том же worktree. `TestAuditCompletionRuntimeZIPImporter` реально запускает
  этот Python test через Go Artifact API, включая cancellation и process-loss;
  его nested pytest report проверяет отсутствие skips/failures.
- 3 live-model Gateway cases не запускались: это явный запрет scope.
- 3 optional sqlmap-binary integration cases относятся к отдельному V55
  scanner release; в данном review выполнялись controlled subprocess/tool
  contracts. Наличие/безопасность новой версии реального sqlmap не утверждается.

Live production endpoints, реальные модели, deployments, OAuth/MCP/V61 и
снятие общей policy ограничений не входят в результат этой проверки.
