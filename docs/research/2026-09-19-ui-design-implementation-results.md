# Contractor UI design audit implementation — September 19, 2026

The current release is **UI 0.3.3**, commit `e981099436b55528c6d5f85873ac8bbb692ad852`, [environment](http://127.0.0.1:4173). [V58-014](../../tasks/v58-014-configuration-navigation-and-credentials.yml) is complete; [evidence](../../tasks/evidence/v58-014.json).

Following the user's clarification, the shared navigation panel was removed from Runtime configuration, the LLM configurations list and detail view, Credentials and Settings. Links to Runtime service credentials and Git SSH keys remain in the credentials page description. Form styles are imported directly and preserve expandable optional sections. The page explicitly distinguishes managed credentials from development tokens: the environment has 0 managed credentials, separate Worker/Planner tokens are configured, and there are 2 Runtime credentials.

UI 0.3.3 checks: 14 Operations/Settings tests, lint/format, both TypeScript projects and production build; 15 screenshots before and 15 after release at 1440/390/320 px, six contextual-link navigations and RuntimeConfig form-style checks. No JavaScript/API errors, horizontal overflow or submitted changes occurred. Queue, scheduler, credentials, 147 Runs, Runtime identities, backend PID and the main release link were identical before and after the update.

Release **UI 0.3.1**, commit `c030303ab81a80665836ccc484dae7e45e3ed504`, [environment](http://127.0.0.1:4173). User feedback after the first release was addressed in [V58-013](../../tasks/v58-013-finding-text-and-run-actions.yml); [evidence](../../tasks/evidence/v58-013.json), [gallery of corrected screens](../../.local/ui-design-implementation/phase-013/index.html).

- Findings in both lists display complete, safely rendered Markdown; the decision link remains before the description and does not overlap text while scrolling.
- Sources links open the exact revision of an Audit input artifact and return to the original Finding. Findings have no structured file/line reference; one is not inferred from the text.
- The Completed Runs delete control fits inside its cell with padding. Expanded labels use the full row width; ordinary keys are no longer broken into pieces.
- Expanded allocations span the entire row; metrics are distributed across columns according to available width. At 1920 px, the first card changed from 341 × 677 to 1189 × 230 px, retaining all six values.

For UI 0.3.1, **47 focused tests**, lint/format, both TypeScript projects and production build passed. Seven states were checked at 1440/390/320 px: 21 screenshots before and 21 after the update. Separate measurements of the delete control, labels and allocations cover 1920/1440/1024/768/390/320 px; two Finding → Sources → Finding round trips were checked. No JavaScript/API errors, document overflow or submitted changes occurred. The browser used an injected session and real authenticated REST reads.

The UI service was restarted for release. The queue remained paused at revision 29; scheduler, 147 Runs, four Runtime identities, Server/Runtime/proxy PIDs and the main release link were unchanged before/after. Local changes from other tasks were preserved. No Server change is required.

## Original UI 0.3.0 release

The figures below describe the first full walkthrough; subsequent D06/D10 refinements were implemented in V58-013 above.

All 32 findings D01–D32 were addressed in tasks V58-001–V58-012. UI **0.3.0** was integrated into local `main` and deployed to the [current environment](http://127.0.0.1:4173). Release code revision: `a894fba6f1b0621e42d7f651a95b0e908ed3c785`. Full implementation commits are retained in the task files; release confirmation is stored separately from the code.

[Task plan](../plans/2026-09-19-ui-design-implementation.md) · [original audit](2026-09-19-ui-design-review.md) · [machine-readable evidence](../../tasks/evidence/v58-012.json) · [gallery of all states](../../.local/ui-design-implementation/phase-012/index.html) · [CSV element inventory](../../.local/ui-design-implementation/phase-012/elements.csv).

Checks: **413 UI tests in 60 files**, **9 UI service tests**, ESLint/Prettier, both TypeScript projects, generated API contract verification and production build passed. The repeat run also corrected waiting for asynchronous rendering in the catalog pagination test.

The browser checked **105 states**, with one screenshot each at 1440 and 390 px, plus **8 critical states at 320 px**: **218 first-viewport screenshots** in total, additional full-page screenshots and 16878 semantic elements in the inventory. After release, **16 states were rechecked on the running UI**. There were no unhandled JavaScript errors, scenario errors, horizontal page overflow or submitted changes. Separate browser checks covered read recovery after a 503 while preserving the filter, keyboard handling and focus return in nested dialogs, mobile navigation, profile selection and enabling dependent telemetry fields.

The walkthrough used an injected session and REST reads with valid access; it did not test password login or authenticated WebSocket operation. Not all states are available in the populated environment: the absence of active work, pending decisions and LLM credentials was checked as an empty state; forms and mutations also have fixture coverage. The browsing limit for the large Nextcloud ZIP remains: a 422 response now gives the exact reason and a Download action, with one expected response at each width.

Only the UI service was restarted during release. The Server, two Runtime and LAN proxy PIDs, main release link, queue, scheduler, Run list and Runtime identities were unchanged before/after. Existing local task changes were preserved. The new UI also includes the updated UI static server with an HTML 404 page; these changes need no separate API Server update.

The comparison uses the same populated lists and key journeys. Page height depends on record count, so these are measurements of this particular environment, not a user study.

| Screen and width | Before | After |
| --- | ---: | ---: |
| runs-completed · 1440 px · page height | 15449 px | 3787 px |
| runs-completed · 390 px · page height | 23283 px | 7708 px |
| operations-allocations-completed · 1440 px · page height | 15673 px | 4193 px |
| operations-allocations-completed · 390 px · page height | 31415 px | 15366 px |
| audit-checks · 1440 px · page height | 16641 px | 5762 px |
| audit-checks · 390 px · page height | 26316 px | 7600 px |
| run-1-succeeded · 1440 px · primary heading position | 925 px | 584 px |
| run-1-succeeded · 390 px · primary heading position | 1379 px | 746 px |
| eval-detail · 1440 px · primary heading position | 1996 px | 328 px |
| eval-detail · 390 px · primary heading position | 3419 px | 409 px |

| Finding | Implementation | Tasks |
| --- | --- | --- |
| D01 | Run viewing with caido provenance is restored; failure of an optional projection no longer hides results. | [V58-001](../../tasks/v58-001-runtime-provenance.yml) |
| D02 | The New Audit form fits at 390 and 320 px; long versions and links do not stretch the page. | [V58-002](../../tasks/v58-002-shared-layout-and-forms.yml) |
| D03 | ZIP file text retains contrast on hover and selection; file rows are 44 px high. | [V58-002](../../tasks/v58-002-shared-layout-and-forms.yml) |
| D04 | Bounded Skill YAML frontmatter is shown as metadata; the main Markdown is read separately, and Source preserves the original. | [V58-003](../../tasks/v58-003-artifact-and-skill-reading.yml) |
| D05 | Page and portal-dialog fields share sizes, backgrounds, labels and focus styles; the Agent version selector was checked. | [V58-002](../../tasks/v58-002-shared-layout-and-forms.yml), [V58-012](../../tasks/v58-012-release-verification.yml) |
| D06 | Run and allocation history is compact; identifiers, labels and metrics are available in expanded details. | [V58-004](../../tasks/v58-004-run-results-and-history.yml), [V58-010](../../tasks/v58-010-operations-observation.yml) |
| D07 | Run results appear before rerun controls; successful execution offers Read results, with metrics collapsed. | [V58-004](../../tasks/v58-004-run-results-and-history.yml), [V58-011](../../tasks/v58-011-error-and-empty-recovery.yml) |
| D08 | Content viewing appears before updates; version and provenance are available through Details and Versions in every scope. | [V58-003](../../tasks/v58-003-artifact-and-skill-reading.yml) |
| D09 | Audit shows Partial coverage, the proportion of completed checks, candidates and decisions separately from execution status. | [V58-005](../../tasks/v58-005-audit-coverage-and-review.yml) |
| D10 | A Finding summary and a link to the decision appear before the long text; unreviewed Findings come first, with a verdict filter. | [V58-005](../../tasks/v58-005-audit-coverage-and-review.yml) |
| D11 | Checks, Coverage and child Runs use retained method/path values or Check N; Runs link to the exact check. | [V58-005](../../tasks/v58-005-audit-coverage-and-review.yml), [V58-012](../../tasks/v58-012-release-verification.yml) |
| D12 | Profiles within a family are ordered by numeric version; the latest loaded compatible version is suggested, with older versions available. | [V58-005](../../tasks/v58-005-audit-coverage-and-review.yml) |
| D13 | Home explicitly shows a paused queue; healthy Runtime and free slots do not imply permission to start. | [V58-006](../../tasks/v58-006-home-and-projects.yml) |
| D14 | Headers, forms and Settings are more compact; repeated decorative blocks and duplicate Audit stop reasons are removed. | [V58-002](../../tasks/v58-002-shared-layout-and-forms.yml), [V58-009](../../tasks/v58-009-configuration-navigation.yml), [V58-012](../../tasks/v58-012-release-verification.yml) |
| D15 | Project overview shows next actions, recent results, sources and the number of unreviewed Findings, with an explicit counting scope. | [V58-006](../../tasks/v58-006-home-and-projects.yml) |
| D16 | Workflow uses its published description or explicitly states that none is available; matching input formats do not imply task readiness. | [V58-007](../../tasks/v58-007-catalog-and-run-inputs.yml) |
| D17 | Agent versions are grouped on the current result page and available through a selector; search and cursors are preserved. | [V58-007](../../tasks/v58-007-catalog-and-run-inputs.yml), [V58-012](../../tasks/v58-012-release-verification.yml) |
| D18 | Skills show their purpose and file count; opening a package preserves Catalog/Skills context. | [V58-003](../../tasks/v58-003-artifact-and-skill-reading.yml) |
| D19 | Evals show the latest execution and case/leg/sample; Runs appear before settings. A quality verdict is not inferred from succeeded. | [V58-008](../../tasks/v58-008-eval-workspaces.yml) |
| D20 | RuntimeConfig shows effective labels/default; the duplicate digest is removed, Built-in replaces the placeholder date, and unchanged Rebind is disabled. | [V58-009](../../tasks/v58-009-configuration-navigation.yml) |
| D21 | Cross-links were added between Runtime defaults, Models/Gateways, Credentials/Budgets and Scheduler/Git. | [V58-009](../../tasks/v58-009-configuration-navigation.yml) |
| D22 | Upload, create and clone open on demand; disabled RuntimeConfig sections hide dependent fields, while drafts are preserved. | [V58-003](../../tasks/v58-003-artifact-and-skill-reading.yml), [V58-009](../../tasks/v58-009-configuration-navigation.yml), [V58-011](../../tasks/v58-011-error-and-empty-recovery.yml) |
| D23 | Objective and authorization scope are multiline; labels are human-readable, while exact keys and wire values are preserved. | [V58-007](../../tasks/v58-007-catalog-and-run-inputs.yml) |
| D24 | Current CPU/RAM/GPU/DB readings and charts appear before detailed counters; units, freshness and collection gaps are preserved. | [V58-010](../../tasks/v58-010-operations-observation.yml) |
| D25 | Errors name the operation and offer a read retry; writes are not retried automatically. ZIP-limit errors explain browsing bounds; 404 offers navigation. | [V58-003](../../tasks/v58-003-artifact-and-skill-reading.yml), [V58-011](../../tasks/v58-011-error-and-empty-recovery.yml) |
| D26 | Audit and Operations use a labeled mobile section selector; responsive lists were checked on desktop/mobile and at the critical 320 px width. | [V58-002](../../tasks/v58-002-shared-layout-and-forms.yml), [V58-011](../../tasks/v58-011-error-and-empty-recovery.yml) |
| D27 | Primary actions are emphasized for each journey; Delete remains neutral until interaction, confirmation stays explicit, and focus management was checked. | [V58-002](../../tasks/v58-002-shared-layout-and-forms.yml), [V58-005](../../tasks/v58-005-audit-coverage-and-review.yml), [V58-011](../../tasks/v58-011-error-and-empty-recovery.yml) |
| D28 | Project cards are more compact: duplicate type/icon elements were removed, IDs shortened with full values available, and dates show elapsed time. | [V58-006](../../tasks/v58-006-home-and-projects.yml) |
| D29 | A readable report Summary appears before exports; the UUID heading is smaller, with a link to coverage gaps. | [V58-005](../../tasks/v58-005-audit-coverage-and-review.yml), [V58-012](../../tasks/v58-012-release-verification.yml) |
| D30 | Published configuration is separate from new-version creation; Clone opens a prepared editable draft. | [V58-009](../../tasks/v58-009-configuration-navigation.yml) |
| D31 | Runtime Agents are ordered online-first with a connection-status filter; available labels/capabilities are used, and missing hostnames are not invented. | [V58-010](../../tasks/v58-010-operations-observation.yml) |
| D32 | Login and the shell read the version from package.json; dates have context, GiB/TiB are readable, and accents and fields are consistent. | [V58-002](../../tasks/v58-002-shared-layout-and-forms.yml), [V58-006](../../tasks/v58-006-home-and-projects.yml), [V58-010](../../tasks/v58-010-operations-observation.yml), [V58-012](../../tasks/v58-012-release-verification.yml) |
