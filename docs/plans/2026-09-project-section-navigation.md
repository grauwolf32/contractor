# Project section navigation

The Project workspace put Audits, metadata, inputs, Workflow discovery and a
50-row Run history on one page. On the demo Nextcloud project this measured
17,317 px at 1440 px wide and 40,528 px at 390 px. Locating a result or changing
inputs required scrolling through unrelated work.

## Implementation

1. Keep one Project header and navigation; give Overview, Artifacts, Workflows,
   Runs, Audits, Findings and Settings independent routes. Keep deep links and
   migrate existing root anchors without losing queries or return context.
2. Build Overview from bounded collections: pending Audit decisions, outputs of
   three successful Runs, five recent Runs, three materials and three Audits.
   Display primary roles only from the exact Workflow contract. Do not infer
   totals or source roles from these samples.
3. Put upload and Git import behind Add artifact. Reuse the Catalog drawer for
   exact Project Workflow setup, maintaining ProjectScope and session drafts.
4. Reduce Run history to 25 compact rows per page with Server lifecycle/state
   filters, Audit origin links and optional full identity/labels. Keep query,
   pagination and selected Workflow versions in navigation state and URLs.
5. Retain existing Audit, Findings and Eval behavior. Update specs, route serving
   and browser journeys; deploy an immutable UI release and its static server.

## Acceptance

- New Projects can add material, review exact inputs and launch a Project Run.
- Existing Projects can locate successful outputs, inspect an Audit decision,
  filter Run history and return from details without losing their context.
- Closing and reopening setup, including after changing sections, preserves the
  exact Workflow draft. Nested upload/Git dialogs preserve focus and scope.
- Old links and direct reloads work. Unopened sections do not fetch their full
  inventories. Errors are local and partial collections are not shown as totals.
- Desktop and 390/320 px layouts have no horizontal document overflow.
- Verification includes unit/integration tests, fixture browser journeys,
  read-only review of live demo data, typecheck, lint, static-server tests and a
  production build. Live analysis and Audit mutations are not needed for rollout.

Global Run text search and broader Evals redesign are outside this change.

## Verified result

The same demo Project Overview measures 1,637 px on desktop and 2,441 px at
390 px. The Run list is a separate 25-row page. All seven sections were checked
at 1440, 390 and 320 px without horizontal document overflow or JavaScript errors.

Validation passed: 355 UI tests, seven static-server tests, 18 fixture browser
scenarios, typecheck, ESLint and formatting. Five read-only live-data journeys
cover legacy links, filtered/paginated Run return, exact Artifact return, latest
and earlier Workflow versions across sections, and mobile navigation. Fixture
journeys additionally verify ProjectScope launch, nested Git/upload focus,
session draft retention, Run repeat and Audit lifecycle controls.
