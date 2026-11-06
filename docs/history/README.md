# Historical documents

[Documentation index](../README.md)

Completed proposals and superseded plans/reviews are preserved here so their
evidence remains available independently of Git history. Current contracts live
in the [specification](../spec/README.md); current delivery status lives in the
[task records](../../tasks/). The original document paths remain short links so
existing task references continue to resolve.

| Historical record | Original path |
| --- | --- |
| [Implemented model-free Stage lanes](proposals/0040-model-free-stage-lanes.md) | `docs/proposals/0040-model-free-stage-lanes.md` |
| [Implemented web scanning and rule sweeps](proposals/0041-web-nuclei-scanning-and-rule-sweeps.md) | `docs/proposals/0041-web-nuclei-scanning-and-rule-sweeps.md` |
| [Implemented Runtime placement labels](proposals/0043-runtime-placement-labels.md) | `docs/proposals/0043-runtime-placement-labels.md` |
| [Target scanning through Audit](proposals/0049-audit-owned-target-scanning.md) | `docs/proposals/0049-audit-owned-target-scanning.md` |
| [Workflow hypothesis production](proposals/0054-workflow-hypothesis-production.md) | `docs/proposals/0054-workflow-hypothesis-production.md` |
| [Hypothesis Workflow correction plan](plans/2026-11-05-hypothesis-workflow-correction.md) | `docs/plans/2026-11-05-hypothesis-workflow-correction.md` |
| [Superseded hypothesis search plan](plans/2026-10-24-audit-hypothesis-search.md) | `docs/plans/2026-10-24-audit-hypothesis-search.md` |
| [Evals experience design](evals-experience-design.md) | `docs/evals-experience-design.md` |
| [Superseded Audit prioritization design](proposals/0034-audit-check-prioritization.md) | `docs/proposals/0034-audit-check-prioritization.md` |
| [Superseded V64 integration plan](plans/2026-09-20-audit-check-prioritization.md) | `docs/plans/2026-09-20-audit-check-prioritization.md` |
| [Review of the superseded V64 integration plan](research/2026-09-20-audit-check-prioritization-review.md) | `docs/research/2026-09-20-audit-check-prioritization-review.md` |
| [Completed test gate cost review](research/2026-10-23-test-gate-cost-review.md) | `docs/research/2026-10-23-test-gate-cost-review.md` |

The one-time Go ADK 1.6.0 to 2.6.0 characterization harness has been removed from
active `scripts/` discovery. Its source and unit tests remain in the pre-cleanup
revision `f098676f3ff496e614ccedbcb0a77d076ec23822`, at
`scripts/check_go_adk_migration.py` and `scripts/test_check_go_adk_migration.py`.
The recorded characterization evidence and ongoing Planner regression tests
remain in the repository. Preserve the old revision through an archive ref or
bundle before any later history squash.
