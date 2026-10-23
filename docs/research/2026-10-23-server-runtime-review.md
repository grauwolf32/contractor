# Server and Runtime review — 2026-10-23

**Revision reviewed:** `39d89438` (branch `codex/run-placement-requirements`,
same tree as `main` `cba266ed` except the nuclei scan work and the
uncommitted V405-002 placement changes). File:line references are as of that
revision; re-verify on current `main` before fixing.

**Scope:** the Go Server (`internal/`, `cmd/`) and the Python Runtime Agent
(`runtime/src/`). The UI was excluded at the user's request.

**Method:** six parallel read-only reviewers by area (HTTP API and contracts;
persistence; Scheduler, Control Plane and credentials; Audit, Eval and
Planner domain; Runtime core; Runtime toolsets), checked against AGENTS.md
section 3. The top findings were re-verified by hand. No Go toolchain was
available, so Go code was read, not built. For the Runtime: `ruff check src`
clean, `lint-imports` 5 contracts kept, `pytest` 4219 passed / 108 failed /
6 errors / 39 skipped, the failures being macOS environment (Podman AF_UNIX,
no `go` on PATH, Linux-only parametrizations); three `test_projectfs_*`
failures deserve a Linux run.

**Outcome:** about 75 findings, no exploitable security defect found. The
biggest classes are misclassified failure outcomes in Audit collection and
Planner result verification, O(N²) Worker State publication, event-loop
blocking in the openapi, Caido and audit_results toolsets, one TLS handshake
per Server↔Runtime request, and six-fold duplication of the Idempotency-Key
grammar.

## Status (2026-10-23)

All findings are open. Tasks V407-001 … V407-038 own them (milestone
`server-runtime-review-2026-10`); V407-039 closes the review and must record
here, per finding, the closing task or the decision that declined it.

| Section | Findings | Tasks |
| --- | --- | --- |
| B HTTP API and contracts | B1–B13 | V407-004 (B4), V407-021 (B2, B3, B7, B11, B12), V407-022 (B1), V407-023 (B5, B6, B8, B9, B10, B13) |
| C Persistence | C1–C9 | V407-028 (C1), V407-025 (C2, C5, C9), V407-024 (C3), V407-002 (C4), V407-026 (C6, C7, C8) |
| D Scheduler, Control Plane, credentials | D1–D13 | V407-018 (D1, D13), V407-013 (D2), V407-014 (D3), V407-020 (D4, D8 sentinel, D10), V407-015 (D5, D12), V407-019 (D6, D8 classifiers), V407-016 (D7, D9), V407-017 (D11) |
| E Audit, Eval, Planner | E1–E15 | V407-001 (E1, E3, E11), V407-008 (E2, E9), V407-006 (E4), V407-002 (E5, E6), V407-029 (E7), V407-003 (E8), V407-030 (E10, E12), V407-005 (E13, E15), V407-027 (E14) |
| F Runtime core | F1–F10 | V407-011 (F1, F2, F4), V407-009 (F3), V407-010 (F5), V407-013 (F6), V407-034 (F7), V407-012 (F8, F9, F10) |
| G Runtime toolsets | G1–G15 | V407-024 (G1, G13), V407-031 (G2, G9 openapi/likec4), V407-032 (G3, G8 http/caido, G9 http), V407-033 (G4, G6, G11), V407-034 (G5, G8 filesystem), V407-035 (G7), V407-037 (G8, G9 code_analysis, G10), V407-007 (G12, G15 katana), V407-036 (G14), V407-038 (G15) |
| H Tooling | — | No task: `regexp2` is a legitimate test-only dependency; stale `.claude/worktrees` are local |
| I Open questions | 7 | V407-039 |

The findings below describe the reviewed revision.

---


## A. Самое важное (сводный топ)

| # | Где | Что | Класс |
|---|-----|-----|-------|
| 1 | `internal/auditimport/check_results.go:240-244`, `standards.go:60-66` | Транзиентная ошибка чтения эталона (`ReadProjectExact`) при повторной попытке превращается в постоянный `finding-proposal-standard-reference-invalid` и сжигает попытку | bug, high |
| 2 | `runtime/.../worker/state.py:236-252`, `instrumentation.py:665` (9 вызовов) | Полная публикация Worker State после каждого tool/model колбэка: `model_dump` до 1000 записей + два `deepcopy` до 4 MiB на event loop, O(N²) за инвокацию | perf, high |
| 3 | `runtime/.../toolsets/openapi/tools.py:50-52` vs `internal/scanplan/types.go:7-10` | Лимиты Runtime (4 MiB / глубина 128) шире серверных (2 MiB / 64): документ валиден для инструмента и отвергается сервером | drift, high |
| 4 | `runtime/.../toolsets/openapi/tools.py` | Весь парсинг/валидация/дамп YAML до 4 MiB выполняется на event loop под локом сессии, дважды | bug, high |
| 5 | `internal/credentials/lifecycle.go:321-329` → `executeDelete` → `manager.Delete` | Эксклюзивный lifecycle-барьер удерживается через сетевой вызов LiteLLM (до 2 мин); блокирует подготовку Worker'ов, placement и создание Run'ов | bug/perf, medium |
| 6 | `internal/controlplane/runtime_client.go:438-453`, `internal/planner/a2a/client.go:139-144` | Новый `http.Transport` с `DisableKeepAlives` на каждый запрос; A2A-опрос каждые 100 мс = ~10 mTLS-хендшейков/с на Worker | perf, medium |
| 7 | `internal/planner/result.go:56-63`, `streamline/tools.go:437-442` | Любая ошибка `Inspect`, включая `ErrArtifactNotFound`/`ErrInvalidName`, помечается retryable; Scheduler (`stage_finalization.go:146-151`) те же ошибки считает постоянными | bug/drift, medium |
| 8 | `internal/auditimport/scan_recovery.go`, `auditscan/result.go:82-97`, `preparation.go:23-26` | Ошибки декодирования закреплённых данных возвращаются «сырыми», контроллер повторяет их каждый тик бесконечно; Audit не падает и не завершается | bug, medium |
| 9 | `internal/httpapi/public/audit_review_handlers.go` (6 GET) | HEAD обслуживается (полное чтение, пустое тело) вместо 405, в отличие от всех соседей | bug/contract, medium |
| 10 | Idempotency-Key: `public/request.go:25-37`, `evaldomain/mutation.go:18-25`, ещё три regex-копии | Шесть определений; Eval уже принимает `a!b`, `a/b`, которые контракт и другие стора отвергают | drift, medium |

---

## B. Server: HTTP API и контракты

1. **HEAD на GET-маршрутах Audit review** (bug, medium). `audit_review_handlers.go:47,122,178,303,424,438`, `event_handlers.go:12` (HEAD на `/v1/events/ws` → 400). Go 1.22 `ServeMux` матчит GET-паттерн и для HEAD. Тест `TestQueryRoutesDoNotImplicitlyServeHEAD` проверяет 12 ручных путей. Фикс: один отказ HEAD в `register` (`routes.go:~205`) для всех GET, удалить ~70 `rejectHead` по хендлерам, тест по `routes()`.
2. **Idempotency-Key шесть раз** (drift, medium). См. топ-10. Фикс: `contracts.ValidIdempotencyKey`, общая фикстура, расширить `TestPublicIdentifierGrammarsHaveOneGoDefinition`.
3. **Лимит тела login** (drift, medium). `auth_handlers.go:22` = 8192, OpenAPI `x-max-request-body-bytes: 2048`; при 2048 пароль максимальной длины не помещается. `git_import_handlers.go:55` и `git_key_handlers.go:48` (64 KiB) лимит не декларируют. Нужен контрактный тест operationId → константа.
4. **`listProjectAudits`/`listOwnerAudits` N+1** (perf, medium). `audit_handlers.go:245-251` → `auditDetailReadModel` на каждый элемент: ~6 запросов + чтение inventory-артефакта. Страница 200 строк ≈ 1200 запросов. Фикс: summary-проекция в списке, детали в `getAudit`; тест на 200 строк со счётчиком запросов.
5. **RuntimeConfig publish: обрыв тела → 500** (bug, low-medium). `runtime_configuration_handlers.go:556-571` возвращает неклассифицированную ошибку; сосед `decodeJSONBounded` отвечает `invalid_request`.
6. **`putProjectArtifact` читает Project до валидации** (low). `project_artifact_handlers.go:47-78` против `artifact_handlers.go:438`.
7. **415 vs 400 на неверный media type** (drift, low). `scheduler_settings_handlers.go:40`, `model_selection_handlers.go:45` отдают 415, остальные 400; текст реестра `apierror/registry.go` устарел.
8. **Три техники декодирования массивов/объектов** (simpl, low). `types_run_project.go:38-105`, `types_configuration.go:228`.
9. **Мёртвые nil-проверки** `model_selection_handlers.go:24,55` (`requireServices` уже отвергает nil).
10. **Git import: три одинаковых таймаута** `git_import_handlers.go:21`, `gitimport/importer.go:84`, `gitimport/client.go:214`.
11. **Байты vs code points**: `run_handlers.go:339` (4096 байт) и `audit_artifact_handlers.go:17` против `maxLength` в OpenAPI.
12. **Отдельный лимит страницы** `performance_handlers.go:38` (`> 100`) мимо `pagelimit`; аналогично `evaldomain.MaxPageSize`, `FindingComparisonMaximumLimit`.
13. **Заголовки проверяются после буферизации тела** в 5 хендлерах (`run_handlers.go:242-258` и др.); общий «preamble» хелпер убрал бы ~15 копий.

Крепко: auth middleware (constant-time compare, CSRF/Origin, лимитер логина), HMAC-курсоры с привязкой к фильтрам, ZIP/Git импорт (ZIP64/метод/флаги, `..`, симлинки, submodules, LFS, бомбы, redirect off), transfer-слоты.

## C. Server: хранилища и миграции

1. **`evalstore/experiments.go:296-330` persistPlan** (perf, medium). До 10 000 `INSERT` по одному в транзакции под `FOR SHARE` Project и `FOR UPDATE` эксперимента; на каждую строку триггеры `contractor_eval_member_bound` и `contractor_eval_members_inserted`. Тест на реальном размере (`comparison_pages_postgres_test.go`) сидит bulk-insert'ом мимо продакшен-пути. Фикс: `INSERT … SELECT FROM unnest(...)`.
2. **`projectlifecycle/controller_sql.go:9-24` и `controller.go:390-394`** (perf, medium). Фильтр `workflow_runs` только по `project_id`, индекса с ведущим `project_id` нет; при дренировании Project seq scan каждую секунду. Фикс: добавить `owner_id = $2` или частичный индекс.
3. **Грамматика placement-labels дважды** (drift, medium). `migrations/000031_runtime_placement_labels.sql:3-24` (PL/pgSQL CHECK) vs `contracts/placement_labels.go:40-65`; SQL-копия не тестируется фикстурой `placement-label-cases.json`. Это ваша текущая ветка.
4. **`pgstore/finding_collection.go:19-43` CollectionReceiptIDs** (perf, low-medium). Два запроса на finding, до 512 round trips.
5. **`pgstore/store.go` listHeldAuditIDsByCredential** (perf, low). `baseline_snapshot ? $1` без GIN-индекса.
6. **`artifacts/postgres_write_sql.go:37-52`** (bug, low, plausible). Гонка при смене filesystem key: FK 23503 → `ErrInvalidScope` вместо конфликта.
7. **`runstore/store.go:165-174` insertRun** (low). Несуществующий Project → сырое `ClassReferenceViolation`, соседи маппят в `ErrNotFound`.
8. **`runstore/run_event_store.go:469-475`** (perf, low). Повторная строгая валидация иммутабельных событий на каждое чтение страницы.
9. **Триггеры `stage_executions`** (`000001_baseline.sql:2166-2170`) срабатывают на любой UPDATE; `000012` починил это для `workflow_runs`.

Крепко: один классификатор SQLSTATE, `KeyedMutation`/`ControllerClaims` переиспользуются, порядок локов Run/Stage, `SKIP LOCKED` без сортировки, purge артефактов батчами.

## D. Server: Scheduler, Control Plane, Credentials, Telemetry

1. **Lifecycle-барьер через сетевой вызов** (bug/perf, medium). `credentials/lifecycle.go:321-329` → `executeDelete` → `manager.Delete`. Spec RCF-R52 это предписывает, так что нужна правка спеки и кода: держать write-сторону только вокруг проверок и prepared-intent, дальше опираться на строковый fence (`HasPreparedDelete`).
2. **Transport на каждый запрос** (perf, medium). `controlplane/runtime_client.go:438-453`, `planner/a2a/client.go:139-144`. Один bound Transport на RuntimeAgentID с keep-alive.
3. **Per-poll работа для ждущего model Stage** (perf, medium-low). `controlplane/placement.go` ~95-136, 195-224: `candidateEdges` дважды со свежим кэшем, `refreshCandidatePrincipals` на каждый опрос. В незакоммиченном диффе `pinReservations` читает полный Run (со snapshot Workflow) ради сравнения `RequiredPlacementLabels`; строка `if err != nil { return err }` не отформатирована gofmt.
4. **`telemetry/policy.go:58-70,130-145`** (perf, low-medium). Усечение отчёта: по одной записи за итерацию с полной перекодировкой ~1 MiB.
5. **`scheduler/reports.go:60-66,195-225`** (perf, low). Повторная расшифровка всех credential'ов при завершении Stage ради списка редакции; `rebuildStageMetrics` трижды.
6. **Sticky `deleteDirty`** (bug, low-medium). `credentials/lifecycle.go:182,435`: после одной неоднозначной вставки все Run'ы блокируются до успешного повторного Delete или рестарта; `Recover` флаг не сбрасывает.
7. **Idle polling по числу lane'ов** `scheduler/supervisor.go:57-66,100-110`.
8. **Дубли**: `sanitizeRuntimeResources` (`controlplane/runtime_client.go:391-406`) = `normalizeAllocationResources` (`telemetry/repository.go:141-157`); три классификатора credential-ошибок (`execution_preparation.go:444`, `placement.go:809`, `credentials/composite.go`) уже расходятся по `ErrKeyUnavailable`/`ErrCrypto`.
9. **Серийный terminal release** `scheduler/allocation_release.go:92-118`: N застрявших Stage × 30 с.
10. **Копия истории на каждый heartbeat** `controlplane/operations.go:415-441`.
11. **Стартовый lease без tolerance** `app/single_active.go:216-229`: зависший PostgreSQL на старте вешает процесс.
12. **`executeCancelling` сканирует всю историю Stage** `scheduler/run_execution.go:175-190`.
13. **Recover при старте роняет весь Server** (accepted risk без владельца, 3.9). `app/composition_credentials.go:126`: недоступный LiteLLM → crash-loop, недоступны Audits/UI/Run'ы без managed credential.

Крепко: claim lifecycle (renewal на 1/3 lease, cleanup context, CAS release), Registry write-gate, Gateway recovery lock order, lease session, shutdown ordering, strictjson на приватной границе.

## E. Server: Audit / Eval / Planner

1. **Транзиентная ошибка → постоянная invalid** (bug, high). `auditimport/check_results.go:240-244`. Фикс: пробрасывать ошибки loader'а как есть, оборачивать в `invalidCheckResults` только результат `validateProposalStandardRefs`; тест с падением чтения на второй попытке.
2. **`planner/result.go:56-63` retryable для детерминированных ошибок** (bug, medium). Stage с `on.interrupted: retry` переповторяет баг Worker'а до `MaxAttempts`. Код `result_artifact_unavailable` не зарегистрирован в `docs/spec/platform/errors.md`; итерация по map даёт недетерминированное имя артефакта.
3. **Scan recovery без `ErrPermanent`** (bug, medium). `auditimport/scan_recovery.go:30-51,79-86`, `scan_retention.go:26-36`, `auditscan/result.go:82-97`, `evidence.go:63-64,112-113`, `preparation.go:23-26`. `internal/auditscan` без тестов.
4. **`scanplan/audit_settings.go:166-202` PrepareOperation** (perf, medium). Полный парс OpenAPI + SHA-256 по 2 MiB на каждую из 1000 операций; O(ops × размер) и при подготовке Audit, и при каждом Run.
5. **`auditimport/check_results.go:134,306-337`** (perf, medium). `ReadRunExact` с полным payload ради дескриптора; `RetainRunExact` в отдельной транзакции с `FOR UPDATE` на `audits` на каждый источник: до 16 384 чтений и транзакций за тик.
6. **`findingintake/postgres.go:257-312` ResolveAuditProposals N+1** (perf, medium): до ~32k statements за Collect, хотя батчевые `hydrateAuditReceiptBatch` уже есть.
7. **`evalservice/selected_views.go:61,166`, `public.go:84-136,224`, `review.go:78`** (perf, medium). Полный `FrozenPlan` (до 1 MiB) вместо `FrozenPlanMetadata`; `Get` пересчитывает Readiness и гоняет `evaldomain.Validate` на каждый GET, который UI опрашивает.
8. **Контроллер Audit: одна мутация за проход** (perf, low-medium). `auditcontroller/controller.go:360-389,580-611`, `settlement.go:76-90`: отмена/удаление Audit с N Run'ами = N полных `GetReconcileSnapshot`.
9. **`planner/passthrough.go:150-156`** (bug, low). При отклонении кандидата не выставляется `invokeFailure`; ExecutionReport говорит «успех» при abort.
10. **Eval в `preparing` нельзя отменить** (lifecycle, low). `evaldomain/lifecycle.go:97-100`; неклассифицированная ошибка подготовки повторяется каждый тик без предела.
11. **`pagelimit.MaxRows` (200) vs `MaxAuditReceiptBatchSize` (200)** — два литерала, при расхождении бесконечный retry (`auditimport/importer.go:318,369`).
12. **Переполнение int64 в `evaldomain/usage.go:129-147`** зависит от порядка map (теоретически).
13. **`internal/auditpriority` без продакшен-вызова** (3.7); `Confidence` парсится и игнорируется.
14. **Дубли контроллерных циклов** `evalcoordinator/coordinator.go:112-153` vs `auditcontroller/controller.go:21-77`; `time.Now()` напрямую в `evalservice/audit_adapter.go:185`, `workflow_adapter.go:108`, `planner/passthrough.go:73,99,113`.
15. **Мёртвый код**: `auditimport/importer.go:277-280`, `check_results.go:325-328`, `auditservice/finding_comparisons.go:277-279,593,606`; `findingintake` читает один StageExecution трижды за Submit; `runservice/audit.go:234-236` игнорирует ошибки `json.Marshal` (при сбое `nil == nil` принимает несовпадающий Workflow).

Вопрос по текущей ветке: `runservice.CreateAudit` (`audit.go:72-86`) не выставляет `RequiredPlacementLabels` для Run'ов, созданных Audit'ом. Так задумано для V405-002?

## F. Runtime: ядро

1. **Публикация State на каждый колбэк** (perf, high). `worker/state.py:236-252`, `instrumentation.py:665-682`, `telemetry/metrics.py:425-450`. Фикс: публиковать метрики текущей инвокации, кэшировать проекцию с инвалидацией в `record_*`, либо coarse-кэденс.
2. **Полный снапшот в ADK-сессию** (perf, medium). `instrumentation.py:706-714`, `runtime.py:1169-1192`. ADK 2.8.0 `InMemorySessionService` делает `deepcopy` сессии на каждый `run_async`; в shared-режиме копируются все накопленные снапшоты. Если никто не читает `session.state["contractor"]`, зеркало можно убрать.
3. **Bare `{identifier}` в инструкции Worker'а** (bug latent, medium). `worker/runtime.py:215`, `finalizer.py:96` передают строку; summarizer (`summarizer.py:232-234`) уже использует `lambda _: text`. Ничего не валидирует фигурные скобки в `configs/instructions`.
4. **TranscriptRecorder без summarizer'а** (perf, medium). `runtime.py:616-650`: redaction + `json.dumps` каждого события впустую.
5. **Deadline применяется только с `completion`** (bug, low-medium). `runtime.py:621-632`: обычный Worker полагается на удалённый `CancelTask`.
6. **Новое TLS-соединение на каждый приватный вызов** (perf, low-medium). `_https.py:120-150`, `artifacts.py:465`, `control_client.py:170` (`Connection: close`).
7. **`projectfs/exporter.py:124-131`** скачивает старый payload до 64 MiB ради побайтового сравнения.
8. **Дубли**: четыре exception-chain walker'а в `runtime.py:1362-1478`; `_start_span`/`_end_span` в `runtime.py` и `instrumentation.py`; `_consume_background_task` ×3; `_truncate_utf8` ×2 (уже расходятся: одна добавляет `…`).
9. **Мелкий sync I/O на loop**: `workspace.py:57-59`, `projectfs/provider.py:118-122`, `agent_skills/runtime.py:573-583`.
10. **`allocation/service.py:184-380` prepare() под общим локом** с обходами в `expire_control_lease`/`reconcile_drain`; стоит флаг «prepare in progress».

Крепко: lifecycle аллокации (rollback, shielded cleanup, кэш терминального ответа, `_force_exit(70)`), Gateway-клиент (bounded retries, Retry-After cap, без тел провайдера), projectfs (`O_NOFOLLOW`, dir_fd, NFC), bounded HTTP-тела.

## G. Runtime: тулсеты

1. **openapi лимиты шире серверных** (drift, high). `openapi/tools.py:50-52` vs `scanplan/types.go:7-10`; нет записи в `docs/spec/platform/limits.md`, нет фикстуры в `api/testdata`.
2. **openapi парсинг на loop под локом** (bug, high). Ни одного `to_thread_until_done` в `openapi/`; валидация трижды за мутацию (`_mutate`, `_serialize`, `upsert_path` ~356). Лучше всего чинить в `common/document_session.py`, likec4 унаследует.
3. **Caido до 64 MiB на loop** (bug, medium). `caido/tools.py:471-475,606-607,944`: `_blob()` (decode + re-encode сравнение), `_CaidoRedaction.message()`, `_raw_preview()` под `self._lock`.
4. **audit_results: re-decode пакета и сборка ZIP под локом** (perf, medium). `collector.py:306-307`, `encoding.py:59-61`, `packages.py:324-333`; тот же decode повторяется в `completion.py:122,131`, `collector.py:78-79`, `publication.py:109`. O(items²).
5. **`ls`/`glob` читают и хэшируют все файлы** (perf, medium). `filesystem/tools.py:143,169` → `snapshot()` → `scan(contents=True)`; `glob` 88 мс на 20k путей на loop. `scan(contents=False)` уже есть.
6. **Каждый `finding` заново качает и парсит source ZIP** (perf, medium). `security_findings/anchors.py:47-59`: 30 findings = ~240 MiB трафика и 30 парсов; есть прецедент `common/artifact_read_cache.py`.
7. **Пагинация больших артефактов O(size) на страницу** `text_artifacts/tools.py:156-160`, `run_artifacts/tools.py:208`.
8. **Инфраструктурные баги замаскированы под retryable tool-ошибки** (bug, medium). `http/tools.py:721-724,990-993`, `caido/tools.py:1131-1132,2359-2360`, `code_analysis/tools.py:1024-1025`, `filesystem/tools.py:321-322,331-332`: `except Exception` → retryable. Ловить конкретные семейства (`httpx.HTTPError`, `CaidoClientError`, tree-sitter), остальное отдавать `_safe_tool_response`.
9. **Валидаторы держат лок сессии через 30-с subprocess** `openapi/tools.py` ~461-475, `likec4/tools.py:261-265`; аналогично `code_analysis/tools.py:309-340` (до 120 с), `_HTTPSession.request` (до 120 с). Если намеренно, зафиксировать как accepted risk.
10. **`.hh` не парсится** (bug, low). `code_analysis/languages.py:92` vs `EXTENSION_LANGUAGES :51-86`.
11. **`list_findings` O(page²) канонизация** `security_findings/reader.py:243-247`; свой cursor-кодек вместо `common/cursors.py`.
12. **Разбор вывода сканеров на loop** `scan/tools.py:124`, `ffuf.py:100-155`, `katana.py:80-226`.
13. **Дубли грамматик**: идентификатор `^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$` в 4 файлах + Go; `MAX_SUMMARY_BYTES` 16 KiB vs Go 64 KiB; line-window в `text_artifacts`, `likec4` и `common/line_window.py`.
14. **memory `_load_all`: 1 + N последовательных чтений** `memory/tools.py:211-237`.
15. **Мёртвое**: `read_openapi_document` не выбран ни одним манифестом; цепочка `edit_files → filesystem` ради одного класса исключения; `katana prepare` качает 128 KiB ради проверки существования.

Крепко: пути (`normalize_project_path`, `O_NOFOLLOW`, ZIP-члены), SSRF/egress (проверка каждого hop'а, strip auth off-origin, argv без shell), CAS-записи, HMAC-курсоры, `_safe_tool_response` не пропускает трейсбеки, нет `assert`.

## H. Обвязка (проверено лично)

- `github.com/dlclark/regexp2` в `go.mod` используется только `internal/contracts/fixture_index_test.go`.
- В `.claude/worktrees/` около десяти брошенных worktree прошлых агентов (не в git).
- 176 make-целей в `make/*.mk`; семейные `test-*-e2e` цели в `release-verify` пропускают Go-сьюты (`RELEASE_CONSOLIDATED=1`), что AGENTS уже отмечает.

## I. Открытые вопросы (нужен запуск кода или решение)

- Реальное число запросов `listProjectAudits` на страницу preparing-аудитов (счётчик на PostgreSQL).
- Достаточно ли серверного `CancelTask` по дедлайну для обычных Worker'ов, или Runtime должен сам применять `StageContentRequest.deadline` (WRK-R6).
- Читает ли кто-то `session.state["contractor"]` из ADK.
- Применяет ли `openapi-operations@1` генератор Audit (`auditdomain/canonical.go:73`) те же 2 MiB / глубина 64.
- Намеренны ли долгие удержания локов валидаторами openapi/likec4 и `_CodeAnalysisSession`.
- `renewClaim` сравнивает клиентское время с `clock_timestamp()` БД; при skew > `claimDuration/3` lane может считать claim своим после takeover.
- Три падения `test_projectfs_*` на macOS: перепроверить на Linux.
