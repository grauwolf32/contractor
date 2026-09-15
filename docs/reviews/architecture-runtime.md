# Ревью архитектуры: Runtime

Дата: 2026-09-15. Источник: `docs/spec/artitecture.likec4`.
Проверены все 11 перечисленных ниже views: связи сопоставлены с реализацией,
ошибочными переходами и соответствующими контрактами спецификации.

## Исправленные проблемы

### P2 — потеря владельцев инструментов при частично неудачной подготовке

**Views:** `runtimeOverview`, `runtimeLifecycle`; затрагивает teardown всех
выбранных Toolsets, в том числе `runtimeCodeAnalysis` и `workerArtifacts`.

`AllocationService.prepare` получал словарь инструментов только после успешного
завершения `_create_tools`. Если второй Toolset выбрасывал исключение или
`CancelledError`, инструменты первого уже существовали, но rollback видел
пустой словарь. Аналогично терялись инструменты фабрики, вернувшей неверный
набор или имя. Слот мог вернуться в idle без вызова их `close()`.

**Исправление:** `_create_tools` сразу передаёт каждый возвращённый инструмент
в словарь, принадлежащий `prepare`, до проверки фабрики и следующего `await`.
Некорректные совпадающие имена получают отдельные внутренние записи только
для cleanup, поэтому один владелец не затирает другого. Все обычные ошибки,
отмена и неподтверждённый cleanup используют существующий rollback/fence.

**Код:** `runtime/src/contractor_runtime/allocation/service.py`.
**Регрессия:** `test_partial_tool_preparation_retains_cleanup_ownership` в
`runtime/tests/test_allocation.py`: exception, cancellation, неверный набор с
коллизией и неверное имя инструмента. До исправления все четыре случая
показывали `close_calls == 0`; после исправления каждый возвращённый ресурс
закрывается, workspace удаляется, слот остаётся idle только после cleanup.

### P2 — последняя ошибка валидации адаптера превращалась в успешный prepare

**Views:** `runtimeAdapters`, `workerTelemetry`, `runtimeLifecycle`.

`AllocationAdapterHost.create` регистрировал адаптер для cleanup до проверки
его ref, metrics и typed handles. Исключение перехватывалось, но дальнейшее
решение опиралось только на `len(hosted) == len(selected)`. Ошибка единственного
или последнего адаптера поэтому пропускалась: например, OTLP без обязательного
instrumentation handle принимался и Worker начинал работу без настроенной
телеметрии.

**Исправление:** успешный возврат находится в `try/except/else`; любое
исключение подготовки обязательно закрывает все уже зарегистрированные
адаптеры и возвращает `AdapterPreparationError`. Размер коллекции больше не
используется как доказательство успешной валидации.

**Код:** `runtime/src/contractor_runtime/adapters/host.py`.
**Регрессия:** `test_invalid_last_adapter_is_rejected_and_closed_before_sandbox`
в `runtime/tests/test_adapter_host.py`: один OTLP и OTLP после HTTP proxy. До
исправления оба случая не выдавали ошибку; теперь prepare прекращается до
sandbox, cleanup идёт в обратном порядке, слот остаётся idle.

P1 в рассмотренных runtime-сценариях не подтверждены.

## Проверка каждого view

| View | Сценарии и corner cases | Результат и реализация |
| --- | --- | --- |
| `runtimeOverview` | Один allocation на процесс; повторный prepare; иной allocation; неудача между созданием ресурсов и публикацией WorkerHandle. | Владение разделено правильно. Исправлена потеря частично созданных Toolsets. `allocation/service.py`, `allocation/context.py`, `state.py`, `server.py`. |
| `runtimeLifecycle` | Prepare с тем же ID и изменённым fingerprint; finalize/abort с другим operation ID; потеря ответа release; отмена cleanup; неостановившийся Worker; истечение control lease. | Повторные операции возвращают сохранённый результат только при совпадении identity. Release очищает ресурсы, но idle появляется после heartbeat confirmation. Неопределённый stop/cleanup сохраняет fence и требует выхода процесса. Исправлены оба дефекта подготовки. `allocation/service.py`, `allocation/cleanup.py`, `lease.py`, `control_client.py`. |
| `workerSessions` | Shared/isolated mode; второй одновременный A2A запрос; отмена при создании/удалении сессии; oversized/non-JSON carried state; точность finalizer; summarizer против hard budget. | Основная сессия принадлежит invocation; конкурентный запрос получает `worker_busy`. Isolated переносит ограниченный state без `contractor` и `temp:`; общий и пользовательский ADK state удаляется при close. Finalizer делает один вызов без tools и проверяет неизменность текста. Summarizer использует отдельную политику, а hard failures сохраняют приоритет. `worker/runtime.py`, `worker/sessions.py`, `worker/finalizer.py`, `worker/summarizer.py`, `worker/budget.py`, `a2a_server.py`. Новых подтверждённых дефектов нет. |
| `runtimeFilesystems` | Внешнее изменение local workspace; binary/symlink/special files; ZIP traversal/повторные пути/распаковка сверх лимита; timeout выполняющейся операции и отмена ожидающей; overlay export. | Local direct читает актуальный диск. WorkspaceOperationGuard удерживает владельца после timeout/отмены уже начатого I/O; отменённый ожидающий не начинает mutation. Cleanup присоединяется к тому же владельцу. Hydration проверяет exact artifact/media type и ZIP bounds. Memory и overlay используют собственные модели, overlay export остаётся ограниченным. `projectfs/local_direct.py`, `local_io.py`, `operation_guard.py`, `hydrate.py`, `overlay.py`, `exporter.py`. Новых подтверждённых дефектов нет. |
| `runtimeCodeAnalysis` | Устаревший graph symbol ID; edit между parse и annotation commit; зависший/упавший child; malformed frame; BrokenPipe после write; отмена создания mirror; непроверенный reap. | Snapshot определяет graph identity, узкие handles разделяют чтение и запись. Annotation сравнивает актуальные bytes внутри `update_text` и не затирает выигравший edit. Trailmark сериализует child/mirror, ограничивает протокол, применяет TERM→KILL/reap и сохраняет mirror ownership при неопределённом stop. Replay разрешён только до передачи запроса. Частичная подготовка теперь также закрывает созданные analysis tools. `toolsets/code_analysis/tools.py`, `trailmark_host.py`, `trailmark_child.py`, `toolsets/taint_annotations/tools.py`. |
| `runtimeAdapters` | Несовпавший ref/handle; последний адаптер с ошибкой; недоступный proxy; partial prepare; отсутствующий selected channel; redirect cookies; flush/close timeout. | Исправлено принятие невалидного последнего адаптера. Каналы передаются только явно выбранным потребителям; Worker получает только model/telemetry handles. Private Artifact/control transport отделён от worker proxy, внешние HTTP handles проверяют redirect/body limits. Flush best-effort, обязательный close остаётся под lifecycle deadline. `adapters/host.py`, `http_proxy.py`, `caido_graphql.py`, `artifacts.py`, `control_client.py`, `toolsets/http/tools.py`. |
| `workerArtifacts` | Чтение чужого namespace; stale allocation/write fence; CAS conflict; hidden memory/skills/body bindings; потеря ответа при append; observed refs из предыдущего invocation. | Allocation ID фиксирован в ArtifactClient URL; окончательная authority остаётся на Server. Generic tools скрывают purpose-specific bindings, MemoryTools сериализует операции и использует exact revision CAS. Неопределённый append не превращается в безусловное повторное добавление. Provenance собирается по invocation cursor и очищается при завершении. Исправлен общий teardown при частичном prepare. `artifacts.py`, `toolsets/run_artifacts/tools.py`, `toolsets/memory/tools.py`, `toolsets/common/artifact_visibility.py`, `worker/runtime.py`. Server grant lifecycle проверяется отдельным ревью. |
| `agentSkills` | Другой exact ref/digest/name; ZIP path escape/duplicate/scripts; превышение stored/expanded/disclosure bytes; запрос невыбранного ресурса; ошибки загрузки/удаления; бинарные ресурсы. | Runtime загружает только resolved allocation packages, сверяет exact ref, digest, media type и package contents. Native ADK доступен через три разрешённых tools, без registry/script execution. Disclosure budget общий для allocation; бинарные attachment разрешения связаны с invocation. Cleanup удаляет извлечённые файлы и ссылки native ADK; его failure не делает слот reusable. `agent_skills/package.py`, `agent_skills/runtime.py`, `worker/factory.py`. Новых подтверждённых дефектов runtime loader нет; server pinning входит в отдельное ревью. |
| `podmanSandbox` | Prepare после hydration; symlink cwd; конкурентный edit/command; timeout ожидающего; потерянный command receipt; потомки после shell exit; EOF/lease expiry; orphan другого owner; failed release. | Opt-in capability требует полного probe. Команда использует exact container ID, ограниченный argv/окружение и общий workspace guard. Неопределённый outcome отвергает дальнейшее execution и требует подтверждённого stop; remove контейнера предшествует удалению bind workspace. Owner/guardian authority отделена от workload; liveness не продлевает подтверждённую lease. `sandbox/podman/lifecycle.py`, `executor.py`, `command.py`, `owner.py`, `guardian.py`, `lifecycle_backend.py`, `allocation/service.py`. Offline/fake coverage проходит; реальные host gates в этом проходе не включались. |
| `auditWorkerCompletion` | Неполный набор; третье напоминание; бюджет на последнем допустимом вызове; конкурентный submit/seal; повтор CAS revision; отмена после write; потеря ответа; существующий ZIP с другими bytes; stale grants. | Collector принадлежит одному invocation; reminders ограничены двумя и не сбрасывают обычные бюджеты. Seal запрещает дальнейшие submissions. Publisher использует create-only CAS, не больше двух writes/two read-backs, сравнивает exact bytes и немедленно прекращает публикацию при authority/fence rejection. Pending cancellation проверяется даже после inline receipt. WorkerResult строится без LLM serializer. `worker/completion.py`, `worker/runtime.py`, `toolsets/audit_results/completion.py`, `collector.py`, `publication.py`, `encoding.py`. Publication корректно не объявлена importer acceptance; V39-007 остаётся самостоятельным release gate. |
| `workerTelemetry` | Overflow во время POST; новые spans при отправке; слишком большой span; sub-threshold tail; partial collector rejection; retry/Retry-After; истечение final flush deadline; close. | Один sender, in-flight prefix остаётся учтён в лимитах очереди, retry повторяет те же bytes. Удаляется только отправленный prefix; новые spans остаются. Подпороговый tail ждёт enqueue/финального flush согласно view. Ошибки экспорта меняют diagnostics, а execution report сохраняет собственные counters. Исправлено принятие OTLP без instrumentation. `adapters/otlp_http.py`, `otlp_retry.py`, `host.py`, `worker/instrumentation.py`, `telemetry/execution.py`. |

## Рефакторинг и границы

- Сделаны два локальных изменения ownership/control flow, необходимые для
  исправлений. Новый слой lifecycle и изменение публичного протокола не нужны.
- Сохранено разделение ordinary finalizer, optional summarizer и deterministic
  Audit completion: их объединение смешало бы разные authority и budget rules.
- Сохранена общая точка workspace ownership для commands/edits/snapshots;
  перенос cleanup в отдельные tool callbacks ослабил бы этот контракт.
- Сопутствующая неточность: текст `docs/spec/13-taint-annotations.md` утверждал,
  что между Toolsets нет общего lock. Это справедливо для private annotation
  session lock, но workspace mutation уже имеет общий ownership guard.
  Формулировка уточнена в основном ревью: общий workspace guard сохранён.

## Проверки

Из `runtime/`:

```sh
.venv/bin/python -m pytest tests/test_allocation.py -k partial_tool_preparation --tb=short
.venv/bin/python -m pytest tests/test_adapter_host.py -k invalid_last_adapter --tb=short
.venv/bin/ruff check src/contractor_runtime/allocation/service.py src/contractor_runtime/adapters/host.py tests/test_allocation.py tests/test_adapter_host.py
.venv/bin/python -m pytest --tb=short -r s
```

- Новые regression tests проверены на исходном коде: четыре failure по
  отсутствию tool close и два failure из-за отсутствующего adapter rejection.
- После исправлений: **1788 passed, 30 skipped**, полный финальный прогон
  занял 52.77 s. Ruff: **All checks passed**.
- 3 skipped требуют настроенный live LLM Gateway.
- 27 skipped — явно включаемые real rootless Podman capability, deployment,
  command execution, owner, release и supervisor gates. Наличие Podman на хосте
  само по себе не включает эти тесты. Исторические утверждения diagram о
  release-verified V30/V31 не выдаются здесь за свежий прогон реального sandbox.
- V39-007 importer/restart release gate не подменяется runtime unit/integration
  suite: принятие результатов Audit остаётся независимой обязанностью Server.
