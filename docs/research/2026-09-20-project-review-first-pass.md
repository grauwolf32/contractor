# Ревью проекта: первый проход — 2026-09-20

Основание — [план V60](../plans/2026-09-20-project-review.md), исходный commit
`3ebe07cb` после завершения V59. Исследование делалось на объединённом V59/V58
коде; completion metadata не меняла рассматриваемые runtime paths.

Первый проход охватил три направления: execution/runtime, data/Audit/product
и auth/quality/operations. Найдены **четыре воспроизведённых дефекта**, сгруппированные
в три correction tasks. Это не завершённое ревью всех подсистем. Углублённые
проверки V60-005–010 остаются отдельной очередью.

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

## Подозрения, которые не следует выдавать за дефекты

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
