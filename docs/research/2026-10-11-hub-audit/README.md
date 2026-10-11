# Hub audit, 2026-10-11 (non-normative)

Raw reports of the eight read-only agents that audited the layer-map hubs on
`main @ 02bbdcbf`. Line numbers refer to that commit. The consolidated report is
the artifact https://claude.ai/artifact/63wB3aD3Y2LQ4Vy8ZHeEjq.

| Report | Hub |
| --- | --- |
| [public-api.md](public-api.md) | `internal/httpapi/*`, OpenAPI, `publicclient`, `ui/src/api` |
| [audits.md](audits.md) | Audit packages |
| [dispatch.md](dispatch.md) | `scheduler`, `planner/*`, `controlplane`, `gatewayrecovery` |
| [exec-state.md](exec-state.md) | `runstore`, `runtimeconfig`, `credentials`, `memory`, `rundeletion` |
| [config.md](config.md) | `config`, `configload`, catalog bundles |
| [artifacts.md](artifacts.md) | `artifacts` and the Runtime artifact client |
| [contracts.md](contracts.md) | `internal/contracts`, `api/v1alpha1`, Python wire models |
| [persistence.md](persistence.md) | `persistence`, migrations, SQL ownership |

Follow-up work is tracked as tasks, not here:

- V367-001: migrations folded into one baseline (done).
- V367-002..013: the twelve fix-first defects (done, merged as 1a4f2f45).
- V368-001..035 (milestone `hub-audit-defects-2026-10`): remaining defects and
  the findings the fix agents reported but did not fix.
- V369-001..007 (`hub-audit-guards-2026-10`): gates and fitness tests that keep
  the defect classes from returning.
- V370-001..030 (`hub-audit-structure-2026-10`): structure, simplification and
  scaling work.

Tasks V368-004 and V368-027 need a decision from the user before they start.
