# План ревью Contractor — V60

Запрошено пользователем после исправления публичного OpenAPI. Начальная точка —
`3ebe07cb` в локальном `main`: V59-001–005 завершены, параллельная UI-доработка
V58-013 включена. [Результаты V59](../research/2026-09-20-public-openapi-corrections-results.md)
сохраняются отдельно от новых замечаний. Этот план запускает ревью; он не
объявляет всю систему проверенной по результатам первого прохода.

Текущий результат: углублённые задачи V60-005–010 и все подтверждённые
исправления V60 завершены. [Итоговый отчёт](../research/2026-09-20-v60-deep-review-results.md)
содержит результаты проверок и их границы; [первый проход](../research/2026-09-20-project-review-first-pass.md)
сохранён отдельно. Проверки относятся к указанным в evidence исходным коммитам.
Live-model качество и production deployment остаются вне рамок этого ревью.

## Цель и порядок

Найти воспроизводимые нарушения пользовательских сценариев и действующих
контрактов, затем исправить их небольшими проверяемыми задачами. Отдельно
оценить дорогие или сложные участки, когда есть измерения. Размер файла,
количество валидаций или наличие двух языков сами по себе не считаются дефектом.

1. **Зафиксировать основания и выполнить первый проход — V60-001.** Составить
   карту требований, текущих задач и проверок. Для наиболее рискованных границ
   прочитать код, историю решений и отрицательные тесты; воспроизвести кандидаты.
2. **Исправить подтверждённые локальные дефекты — V60-002–004.** Не ждать конца
   большого ревью, если причина, контракт и проверка уже известны. Каждый
   исправленный дефект получает регрессию и отдельный implementation commit.
   Фактический интеграционный прогон дополнительно выявил устаревшее имя
   обязательного Runtime-теста — V60-011. Независимое ревью обнаружило отдельную
   invalidation-ошибку при удалении Run — V60-012; она исправлена после
   PostgreSQL-проверок import/delete/purge и восстановления report finalization.
3. **Углубить проверки по сценариям — V60-005–010.** Выполнить приведённые ниже
   fault, database, transport и browser проверки. Перед началом каждой задачи
   обновить базовый commit и проверить пересечения с параллельной работой.
4. **Свести результаты после углублённых проходов.** Для каждой области указать
   выполненные сценарии, оставшиеся пробелы, исправления и измеренные расходы.
   Обновить единый отчёт; завершение одной области не закрывает остальные.

Первый проход и его исправления не зависят от live-model evals. Процессные
проверки используют fake Gateway и изолированный PostgreSQL. Реальные модели,
изменение production-конфигурации и деплой не входят в этот план.

## Основания и параллельная работа

Нормативная точка входа — [каталог спецификаций](../spec/README.md). В каждой
задаче перечислены конкретные контрактные документы. При расхождении prose,
тестов и реализации сначала выясняется история принятого решения, затем
фиксируется, какой контракт исправляется и почему.

На исходном commit найдено 349 task-файлов: 337 завершены (`complete` или
`completed`), восемь pending, два in_progress и два archived. Это снимок до
добавления V60, а не актуализируемая вручную статистика продукта.

- V40-002 — активная instruction-eval работа; V40-003 зависит от неё.
  Ревью читает действующие format/binding/attribution контракты, не запускает
  параллельные эксперименты и не создаёт второй eval runner.
- V55-004 — активный sqlmap prepared-request путь. Остальные pending V55
  остаются собственным scan backlog; не маскируем их отсутствие под regression.
- В `main` V38-001 ещё pending, но отдельный worktree
  `v38-evals-experience` на `b0bfb40c` уже содержит completed V38-001–005
  и in_progress V38-006. Это активная независимая реализация managed Evals;
  V60 не перехватывает её код или UI и не создаёт дублирующий backlog.
- V57-003 — отдельный незакоммиченный decision draft в основном workspace.
  V57-004 уже разделила Audit completion и убрала промежуточный Audit JSON
  limit; V57-005 обоснованно сохранила production transport без reuse.
- Toolset-план уже перенесён в отдельный worktree
  `v56-toolset-runtime-configuration` на `7dff70fa`: V56-001–009 pending.
  Незакоммиченные архитектурные и эти toolset-планы не являются поставленным
  поведением. Их перемещение или изменение другим исполнителем не откатывается
  при интеграции V60. На этапе V60-012 основной `main` получил toolset-план
  под ID V61-001–009 (`c41a5989`); этот planning commit сохранён при интеграции.

ADK-финализатор проверяется с учётом V21-001 и последующей истории: tool-using
agent не обеспечивает нужный structured output для используемого модельного
пути, поэтому отдельный tool-free вызов имеет основание. Его удаление, изменение
общих wire limits, scheduler lease guarantees или recovery semantics требует
отдельного contract decision и стратегии перехода; это не автоматический quick win.

## Карта углублённого ревью

| Область / задача | Что читать | Какие сценарии проверить | Условие завершения |
| --- | --- | --- | --- |
| Execution lifecycle / V60-005 | specs 00, 04, 18, 20; `internal/scheduler`, control, persistence | cancel vs success, finalizing vs aborting, claim loss, crash между durable states, queue pause/admission, concurrency=1 и saturation | Для каждого сценария известен победивший durable transition; старые вызовы не создают недопустимых outputs. Ограничение до обнаружения claim loss описано отдельно. |
| Runtime/A2A / V60-006 | specs 01, 02, 07, 14, 15, 21, 25, 29; Runtime contracts, runner, supervisor, invoker | stale task/session/subtask, cancel во время tool/finalizer, allocation expiry, partial cleanup, mandatory finalizer vs optional summarizer, tool-only worker | Сохранены identity, terminal result и cleanup guarantees; ошибки транспорта отделены от ошибок результата; нет фонового процесса после подтверждённого release. |
| Artifacts и persistence / V60-007 | specs 03, 08, 09, 10, 13, 17, 18, 23, 24; artifactstore, blobstore, migrations | owner/project/run authority, stale CAS, удерживаемый exact-ref, deletion vs admission, corruption, restart, файловый backend, минимальный пул | Реальные DB/filesystem проверки подтверждают атомарность и сохранность; admission fences не обходятся конкурентной записью; нет вложенного ожидания того же ресурса. |
| Audit / V60-008 | specs 19, 25, 27; auditservice/store/coordinator, findings intake | proposed→approve/reject, duplicate/reopen, receipt retention, revisions при list/read, multiround restart, review TTL, intake replay | API/UI показывают согласованное состояние, stale decisions отвергаются, replay не повторяет settlement, опубликованный report соответствует зафиксированной ревизии. |
| Пользовательские пути / V60-008 | spec 06, `ui-user-stories.md`, UI routes/API helpers, CLI | login→Project→input→Run→cancel/retry→result; Audit→finding→review→report; empty/error/reconnect, длинный текст, unicode, readonly owner | Browser выполняет сценарий с настоящим server и fake Gateway; DOM assertions проверяют действие и результат, а не только отсутствие ошибки. CLI отправляет ожидаемые bytes. |
| Auth и секреты / V60-009 | spec 06; auth, public/private middleware, Git SSH/runtime credentials | malformed body, UTF-8/escaping, cookie/bearer distinction, Origin/CSRF, duplicate headers, lockout/retry, TLS identities, secret projections/logs | Требуемые права проверяются до мутации; поддерживаемый ввод достижим через HTTP; секреты отсутствуют в проверяемых публичных ответах и диагностике. |
| Configuration и cross-language / V60-006, V60-010 | specs 00, 01, 07, 16, 26, 29; configs, runtimeconfig, allocation DTOs, generated clients | pinned vs mutable refs, defaults/clear/absence, existing stored versions, string/number/bool mismatches, rejected extras, effective model/credential route | Общие wire-cases проверены настоящими Go/Python readers; публикация и resolve сохраняют принятую семантику; отказ происходит на правильной границе. |
| Tools/workspace / V60-006 | specs 10–13, 21, 27, 29; filesystem, HTTP/Caido, code analysis, scanners | path traversal/symlink, network timeout, tool subprocess timeout, output cap, changed/stale snapshots, cancellation, wrong allocation | Capability/ownership isolation выдержана, timeout освобождает ресурсы; изменения источника атомарны, а не частично применены. Активный V55 scope не перехватывается. |
| Установка, миграции, восстановление / V60-007, V60-009 | deployment/testing guides, CLI migrate, manifests, schema guards | fresh disposable install, supported upgrade, newer schema refusal, interrupted migration, missing secret, restore instructions | Есть выполненный повторяемый сценарий и допустимое состояние после отказа. Наличие SQL-файла само по себе не доказывает upgrade/recovery. |
| CI и тестовая достоверность / V60-009 | Makefile, CI, gate scripts, test matrices | required tests действительно начались/прошли; skip/fail/missing evidence, secret redaction, nonzero subprocess, env guards, pinned generation | Gate не принимает пропуски и не отклоняет валидный результат из-за обработки лога. Обязательные process/browser tests видны в результате; recorded evidence соответствует commit и toolchain. |
| Производительность / V60-009 | spec 22; metrics, pool usage, bounded reads, runtime memory | минимальный пул, конкурентные страницы, cold/warm allocation, workspace digest/hash/parse, repeated data transfer | Сначала baseline RSS/latency/query count, затем сравнение на одинаковых данных. Без измерений оптимизация остаётся гипотезой. |
| Evals и продуктовый backlog / V60-010 | specs 05, 17, 26, 28, 30, task files V38/V40/V55 | implemented/draft distinction, frozen plan, model attribution, missing evidence, determinism/scorer inputs | Проверены control-plane contracts и связь task→evidence; live quality не выводится из offline fixtures. Новая задача не дублирует активную. |

Specs 05 и 28 отдельно проверяются как границы незавершённого продукта:
их draft-поведение не объявляется реализованным и не превращается автоматически
в дефект текущего релиза. Карта охватывает спецификации 00–30, включая
[29 — Tool Workers](../spec/29-tool-workers.md) и
[30 — managed Evals](../spec/30-managed-evals.md).

## Проверки по этапам

Команды ниже — выбранные точки входа, не заявление об уже выполненных тестах.
Перед запуском проверяются их фактические зависимости в текущем Makefile.
Результаты записываются отдельно для unit/fixture, real DB, process, browser
и live-model уровней. Пустой результат `-run` и skipped case не считаются PASS.

| Этап | Основные команды / метод | Условия |
| --- | --- | --- |
| Baseline | `git status`, inventory task YAML, `git log -S`/`git blame`, чтение specs и существующих отрицательных tests | Зафиксировать commit, dirty paths, active task owners; не повторять исправленные или отозванные рекомендации. |
| Execution | `make test-faults`, `make test-lease-integration`, `make test-lifecycle-controls-hardening`, `make test-scheduler-concurrency-hardening` | Disposable PostgreSQL; named cases подтверждают cancel/restart/lease окна, а не только счастливый путь. |
| Runtime/contracts | `make verify-wire-contracts`, `make test-wire-cross-language`, `make test-runtime-hardening`, `make test-worker-session-modes-hardening`, `make test-worker-summarizer-hardening` | Locked Python env, fake LLM adapters; targeted tool tests по карте выше. Podman checks только при подтверждённых prerequisites. |
| Storage | `make test-artifact-integration`, `make test-artifact-blob-backends`, `make test-git-artifacts`, targeted PostgreSQL pool/revision/deletion tests | Уникальные схемы и временные каталоги; закрытие пула/очистка после теста; upgrade/recovery отдельно от CRUD. |
| Audit/product | `make test-audits-hardening`, `make test-audit-completion-e2e`, `make test-findings-e2e`, `make test-audits-browser`, `make test-lifecycle-controls-browser` | Сначала проверить, что wrappers не скрывают skip или ошибки разбора evidence; настоящий server + fake Gateway для process/browser. |
| API/auth/UI | `make verify-public-api`, `make verify-public-api-postgres`, focused auth/HTTP tests, UI typecheck/lint/Vitest/build | Schema-only success дополняется фактическим HTTP и generated request body. PostgreSQL gate обязан исполняться. |
| Delivery | `make release-verify` и отдельные upgrade/restore сценарии | Только после готовности env и профильных исправлений; записать длительности и какие gated suites реально выполнились. Полный release gate не заменяет restore test. |
| Evals/backlog | чтение portable fixtures/format + согласованные offline checks существующих harness | Не запускать эксперимент и не занимать V40/V55 без изменения назначения. |

## Правила регистрации замечаний

Каждое замечание содержит: ID, приоритет, пользовательский эффект, исходный
commit, точные функции/строки, нарушенное требование, историю решения,
воспроизведение, минимальную корректировку, регрессию и границы доказательства.

Статусы: `confirmed` — воспроизведено; `hypothesis` — нужна конкретная проверка;
`accepted` — подтверждённое ограничение дизайна; `rejected` — подозрение не
подтвердилось; `fixed` — есть regression, implementation hash и verification.
Отсутствие теста сначала означает пробел проверки, а не доказанную ошибку.

Приоритет отражает эффект: P1 — блокировка важного сценария, потеря/ошибочная
публикация данных или нарушенная authority; P2 — ограниченный воспроизводимый
дефект; P3 — вводящая в заблуждение документация без установленной поломки.
Task priority использует принятую в `tasks/index.yml` policy follow-up P2;
приоритет finding и task могут отличаться по этой причине.

Для исправления: task `in_progress` до правок; сначала воспроизведение на старом
коде, затем минимальное изменение и meaningful regression. Acceptance включает
смежные ограничения, чтобы устранение одного cap не открыло объект или не
обошло owner/CAS. Каждый implementation commit отделён от completion metadata.
После слияния чужого кода повторяются затронутые проверки; документационные
изменения сами по себе не требуют нового полного runtime/eval прогона.

## Первый проход и очередь исправлений

| Finding | Что обнаружено | Задача |
| --- | --- | --- |
| PR-01 | Login raw body 2 KiB не вмещает поддерживаемый 1024-байтовый пароль после JSON escaping; прямой Login успешен, HTTP даёт 400. | [V60-002](../../tasks/v60-002-login-json-body-bound.yml): обоснованный 8 KiB raw cap и граничные HTTP-тесты. |
| PR-02 | Provenance handler может подписать новые данные старыми Audit/finding revisions при конкурентной мутации. | [V60-003](../../tasks/v60-003-provenance-consistent-reads.yml): revision fence и regression interleaving. |
| PR-03 | Provenance hydration запрашивает receipt из пула, пока прежние rows удерживают connection; подтверждено на PostgreSQL с pool=1. | V60-003: закрыть rows перед hydration и закрепить воспроизведение с настоящим PostgreSQL. |
| PR-04 | Audit completion gate редактирует пароль во всей JSON-строке. CI password `contractor` портит Package, `pass` портит Action; валидные результаты не проходят gate. | [V60-004](../../tasks/v60-004-audit-gate-event-redaction.yml): отделить identity/status от redacted diagnostics и проверить их композицию. |
| PR-05 | После V57-004 обязательная Runtime matrix ссылается на прежнее имя усиленного теста; реальный gate отвергает его отсутствие после 331 успешного Python-теста. | [V60-011](../../tasks/v60-011-audit-gate-runtime-matrix.yml): сохранить требуемое поведение под действующим именем, проверять объявления тестов и выполнить полный gate. |
| PR-06 | Удаление source Run меняет retained provenance без повышения Audit revision; прежние pins остаются допустимыми. | [V60-012](../../tasks/v60-012-run-deletion-audit-revisions.yml), завершена: атомарная invalidation всех затронутых Audits; import/delete/purge и immutable report retry проверены на PostgreSQL. |

V60-012 закрыта локальным исправлением: Run lock при import, упорядоченные
Audit locks при deletion и атомарный revision bump. Для `finalizing` сохранён
report timestamp; наивный timestamp bump воспроизвёл immutable collision.
[Решение и проверки](../research/2026-09-20-run-deletion-audit-revisions.md)
показывают, почему переписывание подсистемы не потребовалось.

Проверка V60-008 подтвердила коллизию при импорте одного receipt в два Audits.
V60-027 устраняет её аддитивной миграцией 62: новые finding и assessment
получают identity в пределах Audit, а прежние ID, история и повторный импорт
в тот же Audit сохраняются. Проверены реальные PostgreSQL upgrade/replay,
независимые review decisions, retention и удаление источника.

Наблюдения первого прохода сохранены в [исходном отчёте](../research/2026-09-20-project-review-first-pass.md).
Результаты углублённых проверок, исходные неуспешные прогоны, исправления и
финальные обязательные gates сведены в [итоговом отчёте V60](../research/2026-09-20-v60-deep-review-results.md).
Он отдельно указывает доказанные сценарии и ограничения каждого уровня проверки.
