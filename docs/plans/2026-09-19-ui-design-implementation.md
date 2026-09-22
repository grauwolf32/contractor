# UI design audit implementation — September 19, 2026

The user approved implementation of all 32 findings from the [review](../research/2026-09-19-ui-design-review.md). Working branch: `feat/ui-design-review`. After verification, the changes will be integrated into local main and the UI of the current environment will be updated.

Primary journeys: open a Run result and a file/Skill; prepare exact inputs; assess Audit coverage and decide on a Finding; understand a paused queue and the effective configuration. Existing exact versions, drafts, change confirmations and secret-storage boundaries are preserved.

Tasks are independent, with each implementation recorded in a separate commit. Set `in_progress` before changing code; set `completed` after meeting the criteria and passing checks, with the full `implementation_commit`. The final task owns the combined verification, browser evidence and release. Checks use Node 24.20 and pinned pnpm 11.24; no dependency changes are needed.

| Task | Area | Findings |
| --- | --- | --- |
| [V58-001](../../tasks/v58/v58-001-runtime-provenance.yml) | Restore Run viewing with current runtime provenance | D01 |
| [V58-002](../../tasks/v58/v58-002-shared-layout-and-forms.yml) | Unify fields, responsive layouts and interaction states | D02, D03, D05, D14, D26, D27, D32 |
| [V58-003](../../tasks/v58/v58-003-artifact-and-skill-reading.yml) | Put artifact and skill contents before secondary metadata | D04, D08, D18, D25 |
| [V58-004](../../tasks/v58/v58-004-run-results-and-history.yml) | Make Run results primary and history compact | D06, D07 |
| [V58-005](../../tasks/v58/v58-005-audit-coverage-and-review.yml) | Clarify Audit coverage and bring finding decisions into context | D09, D10, D11, D12, D29 |
| [V58-006](../../tasks/v58/v58-006-home-and-projects.yml) | Focus Home and Project overview on current actionable work | D13, D15, D28 |
| [V58-007](../../tasks/v58/v58-007-catalog-and-run-inputs.yml) | Improve catalog discovery and parameter authoring | D16, D17, D23 |
| [V58-008](../../tasks/v58/v58-008-eval-workspaces.yml) | Put Eval execution results and distinguishing context first | D19 |
| [V58-009](../../tasks/v58/v58-009-configuration-navigation.yml) | Expose active configuration and separate reading from publishing | D20, D21, D22, D30 |
| [V58-010](../../tasks/v58/v58-010-operations-observation.yml) | Make resource history and current health scannable | D06, D24, D31 |
| [V58-011](../../tasks/v58/v58-011-error-and-empty-recovery.yml) | Verify coherent recovery and empty states across views | D25, D26, D27 |
| [V58-012](../../tasks/v58/v58-012-release-verification.yml) | Verify and deploy the complete design review implementation | All 32; verification and release |
| [V58-013](../../tasks/v58/v58-013-finding-text-and-run-actions.yml) | Restore full Finding descriptions and improve expanded history layouts | Post-release findings: full Markdown, sources, delete control, labels and allocation metrics |
| [V58-014](../../tasks/v58/v58-014-configuration-navigation-and-credentials.yml) | Remove redundant configuration navigation and clarify credential inventory | Remove the shared panel; contextual links; managed and development credentials |

Recommendations apply to existing data and contracts. When an Eval verdict or a Workflow's user-facing purpose is unavailable, the UI shows missing data or the published description; it does not infer a new value from execution success or a technical name. The full experiment system and the new toolset configuration protocol remain in their respective tasks.

Detailed criteria are in the task files. Before/after comparisons use the same populated environment data; logic changes receive focused regression checks, and visual changes are checked in the browser. Baseline screenshots are stored locally in `.local/ui-design-audit-20260919`.

Tasks V58-001–V58-012 are complete; UI 0.3.0 has been released. [Per-finding results, comparisons and checks](../research/2026-09-19-ui-design-implementation-results.md), [release evidence](../../tasks/evidence/v58-012.json).

Following user review, V58-013 was added on branch `fix/ui-finding-run-history`: Finding descriptions are displayed in full Markdown; source links come from exact Audit inputs; expanded labels and allocation metrics use the full row width; space with internal padding is reserved for the delete control. Findings have no separate structured file/line reference, so the UI does not infer one from generated descriptions. V58-013 is complete: affected screens were rechecked and UI 0.3.1 was deployed to the environment. [Fix and release evidence](../../tasks/evidence/v58-013.json).

V58-014 is complete following the user's clarification: the shared navigation panel was removed from all five screens; contextual links and the explanation of managed/development credentials were retained. UI 0.3.3 was released. [Evidence](../../tasks/evidence/v58-014.json).
