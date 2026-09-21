# UI design review follow-up implementation — September 22, 2026

The user approved recording the ten findings of the [follow-up review](../research/2026-09-22-ui-design-review-2.md) as tasks and starting with the copy and layout wave. Work happens on `main`; each task is a separate commit. Set `in_progress` before changing code; set `completed` after meeting the criteria and passing checks, with the full `implementation_commit`. V66-011 owns combined verification, screenshot comparison and the demo release.

| Task | Area | Findings |
| --- | --- | --- |
| [V66-001](../../tasks/v66-001-plain-language-copy.yml) | Replace internal-model vocabulary and explanatory copy with plain labels | E01 |
| [V66-002](../../tasks/v66-002-refresh-controls.yml) | Demote manual refresh controls where data refreshes itself | E02 |
| [V66-003](../../tasks/v66-003-compact-headers.yml) | Compact project and audit headers and move delete into the overflow menu | E03 |
| [V66-004](../../tasks/v66-004-coverage-checks-merge.yml) | Fold audit Checks into Coverage rows | E04 |
| [V66-005](../../tasks/v66-005-configuration-hub.yml) | Gather execution configuration under one Operations section | E05 |
| [V66-006](../../tasks/v66-006-runtime-agent-identity.yml) | Name Runtime Agents and de-emphasise offline identities | E06 |
| [V66-007](../../tasks/v66-007-legacy-eval-workspace.yml) | Simplify legacy evaluation workspaces | E07 |
| [V66-008](../../tasks/v66-008-bounded-audit-collections.yml) | Bound audit collection loading and stop polling terminal audits | E08 |
| [V66-009](../../tasks/v66-009-one-form-pattern.yml) | Use one pattern for create flows | E09 |
| [V66-010](../../tasks/v66-010-dates-markdown-bundle.yml) | Relative dates, scaled Markdown headings and smaller main bundle | E10 |
| [V66-011](../../tasks/v66-011-release-verification.yml) | Verify and release the second design review implementation | E01, E02, E03, E04, E05, E06, E07, E08, E09, E10 |

Delivery order: wave 1 = V66-003, V66-002, V66-001 (headers, refresh controls, copy); wave 2 = V66-008, V66-004, V66-006; wave 3 = V66-005, V66-007, V66-009, V66-010; verification = V66-011. Baseline screenshots for comparison are the 2026-09-21 capture; heights and request errors are compared per state.
