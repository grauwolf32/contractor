# V37 checkpoint — paused at the user's request

Work branch: `feat/v37-ui-journeys`. No merge or deployment was performed.

## V37-010 — completed

Implementation: `1b45fc52` (full hash recorded in the task file).

Added an owner-scoped Audit workspace snapshot, revision-fenced finding/review
pages with whole-filter counts, revision-bound cursors, exact review reads and
report review context. The UI now has summaries, bounded filters, exact subject
navigation, return context, explicit stale/unavailable states and readable Audit
identity with copy. Updated the Audit specification and generated public client.

Passed:

- `go test ./internal/auditservice ./internal/auditstore ./internal/httpapi/public`
- The same packages with `-count=1` and a disposable PostgreSQL database;
  new filter/count/owner/revision queries actually ran.
- `make verify-public-api`
- `vitest run src/routes/projects/audits`: 25 tests.
- `make ui-typecheck`, `make ui-build`.
- `playwright test e2e/audits.spec.ts`: 4 tests, including 1440px/390px review
  navigation, stale-context recovery and dismissed destructive confirmation.

ESLint has no errors; the new `queue.tsx` has a fast-refresh warning because it
exports both components and a hook. It can be split before the final V37 gate.

## V37-011 — in progress, checkpoint only

Implemented, not yet accepted:

- Operations readiness before collapsed snapshot/connection diagnostics.
- Separate current and proposed binding versions.
- Preserve the selected proposal across an authoritative CAS reload, including
  a concurrent rebind to another version. Disable rebind until revision reload.
- Correct scope copy: new Run snapshots and future Agent-label allocation
  resolution; already prepared allocations keep their pinned settings.
- Explicit publication proposal summary; existing credential clearing remains.
- New `ui/e2e/operations-forms.spec.ts` for desktop/mobile, keyboard, denied
  credential creation, secret clearing, publication conflict, binding CAS and
  missing Operations capability.

Operations unit tests passed (24 tests in 5 files). Typecheck and build passed.
The browser gate is **not passing yet**: the new 1440px case cannot find the
Execution readiness heading. Inspect its fixture/API response and browser error
context first. The test run was stopped on the user's request. The complete
browser command still needs a successful run:

```
cd ui
corepack pnpm exec playwright test e2e/operations-forms.spec.ts \
  e2e/scheduler-settings.spec.ts e2e/git-artifacts.spec.ts \
  e2e/responsive-layout.spec.ts
```

Finish relevant specification updates and checks before recording V37-011 as
completed. Keep its implementation commit distinct from this WIP checkpoint.

## V37-012 — not started

After V37-011, complete the connected journey verification task and its dated
before/after report. Use the task's full UI/Playwright gates and real
`make test-ui-stack` integration with disposable PostgreSQL and fixture model
services, not live models or the shared demo. Update `docs/spec/ui-user-stories.md`.
Do not mark deferred V38 work as complete or change model limits.

Read-only inspection found that `ui/e2e/stack.spec.ts` still contains older
selectors for Runtime Agents tables, the long Project page, Workflow launch and
inline RuntimeConfig forms. Update the harness to current navigation while
preserving its real backend, ownership and secret-boundary assertions. Add a
connected local-file upload → exact Run → output read → repeat-draft journey.
`tests/ui-stack/stack_test.go` already starts real Server/UI/Runtime and a fixture
model gateway; no live model dependency is needed.

## Local verification environment

Logs are retained under ignored `.local/v37-ui-journeys/`. Successful V37-010
logs have `*-final.log` names; PostgreSQL success is in `pg-second.log`.
The disposable PostgreSQL container `contractor-v37-tests` was stopped and
removed. The isolated UI on `127.0.0.1:43173` and the running Playwright process
were stopped. The shared demo was not restarted or mutated.

Recreate the disposable database and update the connection URL in the ignored
`.local/v37-ui-journeys/env.sh` before resuming database tests. Source that file
from the repository root to use the local Node 24 binary and explicit browser
base/API URLs. Start the isolated UI **after building**, and restart it after a
new build: its retained asset/index state can otherwise request removed hashes.
Always supply the isolated browser base URL; the default points at the demo.
