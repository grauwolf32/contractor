# Research

[Documentation index](../README.md)

Dated reviews that open or deferred tasks still cite as context, and the
backlog of planned experiments. They are
non-normative: accepted contracts belong in the [specification](../spec/README.md)
and implementation status in the [task files](../../tasks/). A historical report
does not establish current behaviour. Reports of finished work move to [historical documents](../history/README.md)
once no open task needs them.

Each report starts with its date, the revision it reviewed, its scope and
outcome, then a dated *Status* section that says what is still open and which
task owns it. The findings below that section describe the reviewed revision.

| Report | Cited by |
| --- | --- |
| [Исследование сложности Contractor — 2026-10-31](2026-10-31-architecture-simplification-research.md) | Шаг 1 и удаление Sweep выполнены (V425-030/031/033, предпосылка V425-034); SARIF (V425-032) запланирован; остальные сокращения остаются предложениями |
| [Database schema review — 2026-10-03](2026-10-03-database-schema-review.md) | The [schema reduction plan](../plans/2026-10-04-schema-reduction.md), V365-016 and the [database ERD](../spec/database-erd.html) |
| [Code review of main — 2026-10-04](2026-10-04-code-review.md) | V341-001, V347-001 |
| [Maintenance backlog — 2026-10-09](2026-10-09-maintenance-backlog.md) | V261-001, V341-001, V347-001 |
| [`main` branch protection request](2026-10-09-main-protection.json) | V347-001 |
| [v2 research hypotheses — 2026-10-23](2026-10-23-v2-research-hypotheses.md) | Experiment planning; no task yet |
| [Server and Runtime review — 2026-10-23](2026-10-23-server-runtime-review.md) | V407-001 … V407-038 completed except V407-036 (archived); disposition ledger recorded; follow-ups V414-001 … V414-027 open; exit V407-039 in progress until `make release-verify` passes |
| [SQL and schema review — 2026-10-23](2026-10-23-sql-review.md) | V409-001 … V409-013 (milestone `sql-review-2026-10`) |

Evaluation methodology lives in the [portable evaluation specification](../spec/evals/format.md),
the [instruction-evaluation fixtures](../../tests/eval/agent_instructions/README.md)
and the [live-model evaluation commands](../testing/live-models.md).
