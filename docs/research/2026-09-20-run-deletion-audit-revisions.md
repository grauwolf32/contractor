# V60-012: удаление Run и ревизии Audit

Исправление PR-06 из [первого прохода ревью](2026-09-20-project-review-first-pass.md).
[Задача](../../tasks/v60-012-run-deletion-audit-revisions.yml) содержит acceptance;
фактические команды и результаты сохраняются в
[evidence](../../tasks/evidence/v60-012.json).

## Дефект и границы изменения

Удаление Run меняло retained provenance (`runDeleted: false → true`), оставляя
ревизию Audit прежней. Поэтому корректные revision fences из V60-003 не могли
обнаружить изменение: сохранённые pins и подписанный cursor продолжали давать
`200`, а ETag не менялся. Это воспроизведено на настоящих PostgreSQL, HTTP handler
и `ImportIntoAudit` / `DeleteReleasedTerminalRun`.

Исправление локальное, без миграции или переписывания подсистемы. Транзакция
удаления повышает ревизию каждого затронутого owner Audit ровно один раз:

- Audit связанного managed execution;
- Audit нативных proposal receipts;
- каждый destination Audit, удерживающий импортированный proposal из этого Run.

Учитываются и terminal Audits. Несколько receipts и пересекающиеся связи не
создают несколько повышений одной ревизии. Audits без этих связей не изменяются.
Проверки terminal state, release и завершённой managed collection сохраняются.
Ошибочная или откатившаяся операция не оставляет повышения ревизии или tombstone.
Повтор после успешного удаления возвращает обычный `ErrNotFound`.

Finding revision остаётся ревизией его оценки и решения. Доступность исходного
Run инвалидирует enclosing Audit revision; она не создаёт новую оценку finding.
Поля `Origin.RunDeleted` и `Attempt.RunDeleted` по-прежнему вычисляются для своих
Run IDs — автоматического копирования между ними нет.

## Согласование конкурентных операций

Поздний `UPDATE audits` без предварительных блокировок недостаточен: импорт
может добавить destination после выборки, а Audit purge берёт Audit раньше
receipt/retention и Artifact locks. Новый порядок согласован с этими путями:

| Операция | Порядок |
| --- | --- |
| ImportIntoAudit | Source Run `FOR KEY SHARE` → owner/project-bound Audit `FOR UPDATE` → receipt/retention → Artifact import/hold |
| DeleteReleasedTerminalRun | Source Run `FOR UPDATE` → затронутые Audits по Audit ID `FOR UPDATE` → execution/retention → Artifact purge → Run DELETE и Audit revision UPDATE |
| Audit purge | Audit → Artifact purge → удаление Audit и каскадное освобождение holds/retention |

`FOR KEY SHARE` сохраняет source Run на время импорта и конфликтует с его
удалением. Импорты могут одновременно удерживать этот lock; обычные обновления
неидентифицирующих Run-полей не требуют эксклюзивной сериализации всех импортов.

Если import победил, удаление ждёт его commit и включает новый Audit в выборку.
Если deletion победило, import ждёт и затем получает `ErrNotFound`. Rollback
deletion освобождает исходный Run для успешного импорта. Отмена ожидающего
импорта не оставляет частичного hold.

Нативные receipts не открывают отдельную гонку: production Submit проверяет
allocation write grant, который закрывается до `release_completed_at`.
Managed collector создаёт proposal holds до durable collection receipt;
удаление требует эту receipt. Текущий Project controller вызывает одиночное
удаление Run вне удерживаемой Project-транзакции. Существующий ограниченный
retry определённых PostgreSQL aborts сохранён.

Это обоснование текущих путей и проверенных interleavings. Оно не утверждает,
что произвольный будущий SQL или произвольный порядок внешних транзакций не
может создать deadlock. Новый путь создания hold должен соблюдать этот порядок.

## Почему при finalizing сохраняется updatedAt

Report importer сначала пишет immutable `report.json` и `report.md`, затем
выполняет `CommitReport` / `ProposeReport` с проверкой Audit revision. В машинном
отчёте `generatedFrom` берётся из `Audit.UpdatedAt`.

Наивное одновременное повышение revision и timestamp между записью файла и CAS
отклоняет stale commit, но при повторе создаёт другие bytes под уже занятым
immutable именем. Отдельная fault-проба воспроизводит
`ErrArtifactIntegrity: Audit immutable binding collision`; принудительная
перезапись отчёта нарушила бы действующий контракт.

Принято узкое правило: когда Audit находится в `finalizing`, именно удаление
связанного Run повышает Audit revision, сохраняя `updatedAt`. Удаление меняет
доступность Run, которая отсутствует в report payload; durable collection
receipts, assessments и evidence остаются. Повтор с новой ревизией воспроизводит
прежние bytes и exact refs, после чего обязательный CAS проходит.

Для других Audit states `updatedAt` повышается обычным образом. Авторитетный
change token — revision/ETag. Это правило зафиксировано в
[spec 19](../spec/19-audits.md), а атомарность удаления — в
[spec 18](../spec/18-run-and-workspace-lifecycle-controls.md).

Pending human review уже имеет frozen subject revision/digest и точные report
refs. Изменение текущей Audit revision не отменяет этот subject. Approve/reject,
повтор proposal после потерянного ответа и повтор review decision сохраняют
прежнюю authority. Дополнительная проверка равенства current Audit revision
старому subject здесь была бы ошибкой.

## Проверки

| Граница | Что проверено |
| --- | --- |
| Исходный HTTP дефект | На baseline Audit A остаётся rev 3, B — rev 2; прежние pins/cursor дают `200`. После исправления A/B получают +1, старые pins/cursor и мутация посреди чтения дают `409`; ETag обновляется. |
| Retained provenance | Чтение через отдельный pool с одной connection; fresh envelope содержит новую ревизию, исходные refs/receipt IDs, корректные Run IDs, exact proposal/evidence bytes. Не связанные Audits того же и другого owner не меняются. |
| Import ↔ Run deletion | Оба победителя, rollback и отмена ожидающего import с последующим retry. Реальные service calls; ожидание доказано через `pg_blocking_pids` и текст выполняемого SQL. |
| Run deletion ↔ Audit purge | Оба порядка; retained Project Artifact и source Run используют одну версию. Проверяется ожидание Audit lock до retention/artifact locks и завершение обоих удалений. |
| Report finalization | Automatic/human-required × deletion перед machine write, после machine write, после summary write. Stale CAS отклоняется, новый экземпляр importer повторяет те же bytes/refs. |
| Pending report review | Approve и reject после удаления; точный frozen subject и proposal/decision replay. |
| Managed deletion | Реальное создание и collection Run, transaction-backed rollback с восстановлением Run/Audit/tombstone/input Artifact, repeat без второго bump; uncollected terminal Run отклоняется без мутации. |
| Native receipt retention | Storage-boundary seed двух immutable native receipts одного collected execution; после завершения Audit — +1 revision и два discarded tombstones с неизменной receipt identity. Это проверка persisted rows, а не имитация Runtime Submit. |
| Отрицательная report-проба | Принудительное изменение finalizing timestamp после записи machine report воспроизводит immutable collision. Защита immutable bytes не ослаблена. |

Новые report/managed/native regression cases внесены в обязательную Go matrix
Audit completion gate. Он прошёл: **173 Go cases и 331 Runtime tests без
selected skips**. Общий PostgreSQL `-race` прогон пяти затронутых пакетов и
`make verify-public-api-postgres` также прошли. После этого точечными `-race`
прогонами дополнительно проверены terminal Audit, ровно один bump и неизменность
finding revision; production-код с момента общего прогона не менялся.
Полные команды и пределы проверок сохранены в evidence. PostgreSQL — отдельный
одноразовый локальный контейнер, схемы изолированы; Runtime gate использует
существующие offline fixtures. Live-model evals, активные Evals/toolset/sqlmap
работы и production deployment в это изменение не входят.
