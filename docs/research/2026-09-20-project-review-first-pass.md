# Ревью проекта: первый проход — 2026-09-20

Основание — [план V60](../plans/2026-09-20-project-review.md), исходный commit
`3ebe07cb` после завершения V59. Исследование делалось на объединённом V59/V58
коде; completion metadata не меняла рассматриваемые runtime paths.

Первый проход охватил три направления: execution/runtime, data/Audit/product
и auth/quality/operations. Первоначально воспроизведены четыре дефекта;
интеграционный прогон и независимая проверка добавили ещё два. Итого —
**шесть подтверждённых замечаний**. Это не завершённое ревью всех подсистем.
Углублённые проверки V60-005–010 остаются отдельной очередью.

**Результат реализации: PR-01–06 исправлены и проверены.** PR-06 закрыта
в [V60-012 с проверкой report retry](2026-09-20-run-deletion-audit-revisions.md).
[Общая проверка первоначальной интеграции](../../tasks/evidence/v60-integration.json)
содержит original implementation hashes, фактические команды и оставшиеся
границы. Ниже сохранено описание дефектов в момент их обнаружения.

[Evidence первого прохода](../../tasks/evidence/v60-001.json) содержит точные
команды выполненных проверок, исходные результаты probes, snapshot параллельных
веток и проверки task graph. Строки исходников ниже относятся к состоянию до
исправлений PR-01–04, а не к последующим сдвигам строк.

## Подтверждённые дефекты

### PR-01 — P2: допустимый пароль недостижим через HTTP login

`internal/httpapi/public/auth_handlers.go:17,40` ограничивал raw JSON 2048 байтами;
`internal/auth/password.go:28` разрешает пароль 12–1024 UTF-8 байт. Эти правила
одновременно записаны в spec 06, раздел authentication, но не согласованы
после JSON escaping.

Воспроизведение использовало настоящий auth service и HTTP-handler. Для всех
вариантов HashPassword и прямой `Service.Login` успешны; длина пароля — 1024
UTF-8 байта. HTTP с компактной JSON-сериализацией даёт:

| Пароль | Размер JSON с тестовым username | До исправления |
| --- | --- | --- |
| ASCII без escaping | 1058 байт | 200 |
| Backslash или quote | 2082 байта | 400 |
| U+0001, сериализованный как `\u0001` | 6178 байт | 400 |

Предельная стандартная компактная сериализация для 64-символьного разрешённого
ASCII username и 1024 однобайтных control characters занимает
`29 + 64 + 6 × 1024 = 6237` байт. Поэтому [V60-002](../../tasks/v60-002-login-json-body-bound.yml)
увеличивает только raw-body cap до обоснованных 8 KiB и сохраняет decoded
password bound. Это не обещание принимать неограниченный whitespace в JSON.
Регрессия должна проверять также overflow, malformed/unknown fields, origin
и rate-limit safeguards.

### PR-02 — P2: provenance возвращает новые данные под старыми revisions

`internal/httpapi/public/audit_review_handlers.go:311–370` получает Audit/finding
revisions, затем вызывает `ListFindingProvenance` и возвращает исходный envelope.
Сервис заново читает finding в `internal/auditservice/finding_review.go:403`.
Конкурентная мутация между чтениями может изменить и список, и `SupportsCurrent`.
Финальной проверки revisions в этом пути не было.

Требование spec 19, Finding review/provenance pagination, и V25-011 R8:
конфликт должен предотвращать смешивание истории. Детерминированный test с
настоящим handler и fake interleaving получил `200`, revisions `9/4` и новую
current assessment после мутации до `10/5`. Ожидаемого `409` не было.

Это доказательство HTTP orchestration при заданном interleaving; оно не
выдаётся за измерение вероятности PostgreSQL race. Исправление входит в
[V60-003](../../tasks/v60-003-provenance-consistent-reads.yml): проверка согласованности
охватывает чтение и гидратацию, сохраняя имеющийся cursor contract.

### PR-03 — P2: provenance ждёт второе соединение, удерживая первое

`finding_review.go:500–532` держит `pool.Query` rows, а внутри `rows.Next()`
вызывает receipt hydration. `findingintake/postgres.go:209` для неё запрашивает
ещё одно соединение из того же пула. Первое pgxpool освобождает после
закрытия/исчерпания rows.

Настоящий PostgreSQL, отдельная удаляемая test schema и одинаковый retained
finding дали: `MaxConns=2` — PASS примерно за 0,03 с; `MaxConns=1` — deadline
примерно через 1,99 с с ошибкой `authorize Audit finding receipt: context
deadline exceeded`. Контрольный `GetFinding` проходит на обоих пулах. После
возврата все соединения освобождены: это вложенное ожидание, не утечка.

V60-003 должна сначала закрывать rows, затем гидратировать receipts. Regression
проверяет реальный минимальный pool, owner/exact identity, retention и сохранение
V59 page=200. Аналогичный эффект при одновременном насыщении большого пула
логически возможен, но первый probe этого отдельно не измерял.

### PR-04 — P1: redaction ломает обязательный Audit release gate

`scripts/test-audit-completion-e2e.py:118` применяет `redact` ко всей Go JSON event
строке до сохранения и проверки. `redact:104` заменяет каждое вхождение DB password.
В CI этот пароль — `contractor`, поэтому меняется каждый Package
`github.com/grauwolf32/contractor/...`. Пароль `pass` аналогично меняет Action.

Offline reproduction строит полный report из настоящей mandatory matrix:
неизменённые события проходят все 54 required cases; после текущей redaction
с CI password verifier объявляет обязательные cases отсутствующими. С длинным
паролем, не совпадающим с metadata, та же матрица проходит. Это детерминированный
дефект обработки evidence; полный GitHub Actions run в этом исследовании не
исполнялся.

[V60-004](../../tasks/v60-004-audit-gate-event-redaction.yml) должна сохранить
framework identity/status и редактировать диагностические данные. Нельзя
исправлять это ослаблением required cases или сменой пароля CI. Тест должен
соединять subprocess capture и verifier, включая отсутствие секретов в логах,
skip/fail/missing/malformed и nonzero exit.

## Дополнительные подтверждённые замечания

### Дополнение PR-05 — P1: обязательная matrix отстала от V57-004

Реальный `make test-audit-completion-e2e` выполнил 331 Python-тест успешно,
затем отверг отсутствие обязательного
`test_artifact_observations_survive_a_reminder_and_join_verified_publication`.
История `7f553b22` показывает: тест переименован в
`test_audit_typed_assembly_preserves_reminder_publication_and_refs_without_model_decode`,
его прежние assertions сохранены и усилены, а matrix не обновлена.

Это пропуск интеграции V57-004. [V60-011](../../tasks/v60-011-audit-gate-runtime-matrix.yml)
сохраняет обязательный минимум `1` у действующего усиленного теста. Новый
offline AST-check сравнивает matrix с реальными объявлениями Python-тестов:
он падает на старой записи и проходит после её исправления. Синтетические
JUnit-пробы сами копировали имена из matrix, поэтому такой drift не обнаруживали.
AST не доказывает collection/execution: это по-прежнему обязанность полного gate.

### Дополнение PR-06 — P2: удаление Run не инвалидирует Audit revision

`internal/runstore/delete_store.go:98,120–133` меняет execution tombstone и receipt
retention без изменения Audit revision. Spec 19:1571–1576 требует продвижения
ревизии для projection-visible execution/receipt/provenance изменений.

Проба с настоящим PostgreSQL использовала production `ImportIntoAudit`, затем
`DeleteReleasedTerminalRun` между чтениями provenance. Прежние caller pins
приняты, хотя `Origin.RunDeleted` изменился `false → true`; Audit revision
осталась `2`, finding revision — `1`. Это отдельная upstream invalidation-ошибка:
сравнение корректно фиксируемых ревизий в V60-003 не может обнаружить мутацию,
которая сама не продвигает ревизию.

[V60-012](../../tasks/v60-012-run-deletion-audit-revisions.yml) завершена:
managed execution, нативные receipts и все destination Audit holds включены
в атомарную invalidation. Порядок блокировок согласован с import/purge; rollback,
HTTP pins/cursor и report retry проверены на PostgreSQL. `Attempt.RunDeleted`
не приравнивается к `Origin.RunDeleted`: эти поля могут относиться к разным Run.

Причина отдельной задачи конкретна: report importer записывает immutable
`report.json`/`report.md` до revision CAS, а байты содержат `Audit.UpdatedAt`.
По коду `auditimport/report.go:223,262–325` и `auditimport/artifacts.go:166–167`,
наивный revision/timestamp bump между записью и CAS способен оставить прежнее
immutable имя с несовместимыми байтами при повторе. Это вывод из пути кода,
на момент первого прохода ещё не проверенный fault-пробой. В V60-012 проба
воспроизвела immutable collision. Принято и проверено правило: deletion в
`finalizing` повышает revision, сохраняя report timestamp и bytes для retry.
Pending report review сравнивает subject с сохранённым candidate, поэтому
нельзя вводить для него новую проверку равенства текущей Audit revision.

## Принятые решения и отклонённые подозрения

| Путь | Проверенное основание | Классификация / следующий шаг |
| --- | --- | --- |
| Claim ID отсутствует в terminal SQL | Историческое решение `7ca8daa6`, тогдашний `docs/reviews/architecture-orchestration.md:43–45`, ограничивает защиту моментом обнаружения ErrClaimLost. Spec 20 предусматривает один активный Server/Scheduler. | **accepted**. Тестировать окно до обнаружения можно как проверку принятого ограничения; не вводить multi-server fencing под видом локального исправления. |
| Cleanup `_stop` содержит await без собственного timeout | Внешний allocation owner ограничивает всю операцию deadline; failed cleanup удерживает Runtime fenced и приводит к force-exit. | Подозрение локального отсутствия timeout **rejected**. Нужна дополнительная process-проверка отсутствия оставшихся дочерних ресурсов после настоящего exit. |
| Project delete может обойти Artifact write admission | Актуальный CAS находится в SQL; migration 39 admission triggers берут FOR SHARE, исключение cleanup ограничено точным Audit namespace. | Поверхностный handler TOCTOU **rejected**. Следующий этап — управляемая DB-гонка write/update/publication vs delete и rollback. |
| Повторный Run не перезаписывает Project output | Spec 17 требует create-only publication; Scheduler сохраняет успешный Run при нефатальной ошибке Project publication. | **accepted**. Проверить конкурентную публикацию и repeat при удалённом/изменённом exact source. |
| Дополнительный tool-free finalizer | V21-001 подтверждает ADK/tools/structured-output ограничение; V57-004 сохранила ordinary completion. | **accepted**. Замена принадлежит отдельному decision scope V57-003; не дублировать прежнюю отозванную рекомендацию. |
| Нет A2A connection reuse | V57-005 измерила TLS effect и сохранила transport из-за retry/expiry/certificate различий и недоказанного production выигрыша. | **accepted** до новых измерений и согласованной семантики. |
| Разное JSON escaping в Go/Python на общем wire cap | Разные serializers могут считать bytes по-разному; межъязыковой отрицательный сценарий пока не выполнен. | **hypothesis**, проверять общей fixture matrix в V60-006. Общий cap сейчас не менять. |
| SARIF/structured finding analysis отсутствуют | Spec 28 прямо обозначена draft/not implemented. | **existing backlog**, не regression поставленного контракта. |

## Что проверено и чего ещё нет

Execution/runtime проход прочитал durable finalizing recovery, cancel/claim
cleanup, shielded session cleanup, two-phase allocation release, trusted result
assembly и Go A2A correlation. Выбранные offline проверки прошли: семь Scheduler
tests, четыре группы A2A, десять cancellation/session tests и 29 decoder/assembler
tests. Это не полный fault/process suite.

Data проход прочитал Artifact CAS/deletion fences, provenance/retention,
Project output publication и UI repeat. Новые отрицательные probes для PR-02/03
падают на исходном коде, как ожидается при воспроизведении дефектов. PostgreSQL
probe использовал только выделенную тестовую схему, которая удалена после теста.

Quality проход воспроизвёл PR-01 настоящим HTTP и PR-04 локальной композицией
report/redactor/verifier, прочитал CI/gate composition и migration fences.
Полный release gate, fresh install, restore, browser journeys и live models
не запускались этим первым проходом. Пройденные в V59 435 UI и полные Go tests
не переименовываются в новые результаты V60.

Финальные исправления, исходные implementation hashes и реально выполненные
regressions записываются в task-файлах V60-002–004 и evidence первого прохода.
Шесть задач V60-005–010 остаются pending до отдельного углублённого исполнения.

## Реализованные исправления и общая проверка

| Задача | Статус / результат |
| --- | --- |
| V60-001 | План и первый проход завершены; принятое решение отделено от доказанного дефекта и пробела проверки. |
| V60-002 | Завершена: 8 KiB login body, 19 новых HTTP subcases; максимальный username/escaping, Content-Length/chunked, 8192/8193, decoded bounds и safeguards. |
| V60-003 | Завершена: optional internal pins передаются из HTTP, before/after revisions читаются owner-scoped JOIN; rows закрываются до batch hydration. PostgreSQL pool=1 проверяет 201 уникальный receipt, повторные records, missing receipt и concurrent revision changes. |
| V60-004 | Завершена: Go event identity/status сохраняются, diagnostics редактируются после JSON parsing, arbitrary extra fields не сохраняются. 14 subprocess composition regressions дополняют существующие required tests. |
| V60-011 | Завершена: mandatory Runtime test обновлён без ослабления assertions/minimum, offline declaration check выявляет stale names. Полный gate прошёл с PostgreSQL password, совпадающим с CI. |
| V60-012 | Завершена: Run→Audit lock order и атомарный revision bump; finalizing timestamp сохраняет immutable retry. Проверены concurrent import/purge, rollback, terminal Audit, pending review и реальные HTTP stale pins/cursors. |

На объединённом коде прошли `go test -count=1 ./...`, `go vet ./...`, affected
Go packages с настоящим PostgreSQL и `-race`, public API gate и обязательный
pagination gate. `make test-audit-completion-e2e` прошёл **158 Go cases и
331 Runtime tests без selected skips**. Начальный отказ из-за старого имени
теста сохранён в [evidence V60-011](../../tasks/evidence/v60-011.json), а не скрыт
успешным повтором. Это полный Audit completion gate, не полный `release-verify`.

V60 не меняет UI или generated OpenAPI clients. Результаты 435 UI-тестов,
typecheck/lint/build и byte-reproducible generation относятся к отдельно
зафиксированной V59-проверке. Локальные PostgreSQL-прогоны используют собственный
disposable контейнер и уникальные схемы; live-сервисы и модели не затрагиваются.

Дополнительный прогон V60-012: **173 Go cases и 331 Runtime tests без selected
skips** в Audit completion gate, пять затронутых пакетов с PostgreSQL и `-race`,
public pagination gate. [Evidence](../../tasks/evidence/v60-012.json) отделяет
этот прогон от первоначальных 158 Go cases выше.

## Отдельное наблюдение для V60-008

Во время V60-012 temporary PostgreSQL probe проверила повторный импорт одного
receipt: в тот же Audit replay успешен, во второй совместимый Audit —
`audit_findings_pkey`, SQLSTATE `23505`. Транзакция откатывается, holds остаются
A=1/B=0. Migration 041 строит глобальный `finding_id` только из receipt ID;
таблица holds при этом имеет ключ `(receipt_id, audit_id)`.

Это подтверждённое ограничение реализации. Явное принятое требование именно
для одного receipt в нескольких Audit не найдено: существующая множественность
Audits и compatible import ещё не определяют этот edge case полностью.
[V60-008](../../tasks/v60-008-audit-product-journeys-review.yml) должна уточнить
контракт, ожидаемый HTTP outcome и безопасную identity/migration strategy,
затем оформить correction task. V60-012 проверяет несколько destination Audits
для разных receipts одного Run и не заявляет исправление этой коллизии.
