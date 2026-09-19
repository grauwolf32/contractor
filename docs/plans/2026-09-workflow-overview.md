# Workflow detail overview — 2026-09-19

Approved design A: a readable Workflow overview with a separate Run setup drawer.
Based on `main` after fast-forward merging `feat/ui-user-story-usability` at
`b1e0dac2`. Implementation branch: `feat/workflow-overview-drawer`.

## Scope and implementation

1. Replace the repeated Catalog/detail headings with breadcrumbs, the shared
   Workflow identity and an exact version selector. Reuse complete inventory and
   numeric version ordering; preserve the explicit route version and return
   context. Keep refresh in the overflow menu.
2. Show input/output slots, required flags, human-readable formats plus MIME,
   published primary flags and stage objectives before setup. Mark the entry
   stage; a multi-stage list does not claim execution order. Keep exact Agents,
   instruction references, resolved execution and transitions in a disclosure.
3. Open the existing standalone Run form in the shared accessible Dialog, aligned
   right on desktop and full-width on mobile. Inputs precede required parameters;
   optional parameters and advanced settings are disclosed on demand. Keep
   Project forms' existing layout. Preserve the `#workflow-run-setup` repeat-run
   destination, version/scope-isolated drafts, exact input review and idempotency.
4. Keep nested upload/import/discard dialogs isolated, restore focus on closing,
   and prevent panel dismissal while submission is pending. Include summaries in
   the focus loop and exclude controls hidden by closed disclosures.
5. Validate contract/version browsing, close/reopen/version-switch draft recovery,
   nested focus, repeat-run review and response-loss retry. Check desktop/mobile
   against live read-only data, then commit and deploy an immutable UI build.

## Validation

Pre-deployment checks passed:

- 57 unit/component files, 353 tests; targeted Workflow/Project regressions after
  the final input ordering adjustment also pass.
- TypeScript including E2E sources, ESLint across `src`, changed-file Prettier,
  production build and `git diff --check`.
- Three fixture browser scenarios: overview/drawer/draft recovery, reviewed exact
  repeat with conflict preservation, and Audit-managed Run return.
- Live read-only review of 41 routes at 1440px and 390px: no JavaScript errors,
  horizontal page overflow, unnamed visible buttons or attempted writes. Seven
  navigation/keyboard journeys pass. Workflow overview and nested-drawer checks
  additionally pass at 320px.

Browser mutation tests use fixtures; the live stand review uses a fixture session
and authenticated REST reads with writes blocked. The immutable release manifest
records deployment health, asset verification and post-deployment browser checks.
Deployment only restarts the UI service and verifies the existing Server, Runtime
processes and Scheduler state are retained.
