# V37 connected journey verification — 2026-09-19

Branch: `feat/v37-ui-journeys`. Scope: V37-001…V37-012 and
[US-01…09, US-11](../spec/ui-user-stories.md) within their V37 boundaries.
Implementation hashes are recorded in the individual task files. This report
supersedes the [paused checkpoint](2026-09-19-v37-checkpoint.md).

This is automated scenario verification and an implementation review. No study
with new or experienced users was conducted. There is no measured before/after
success rate, task duration or overall UX score. The task baselines and the
[earlier all-view review](2026-09-ui-user-story-review.md) provide qualitative
context only; test durations are not user-performance measurements.

## Before and verified behavior

| Journey | Baseline problem | Verified behavior and evidence |
| --- | --- | --- |
| Prepare a Project and choose a Workflow | Long Project page; input preparation and launch compete for attention | Separate Project sections; contextual upload; shared Workflow cards; latest numeric default with deliberate older selection. `project-workspace.spec.ts`, `catalog.spec.ts`, real stack selecting 7 → 5 explicitly. |
| Fill and resume Run setup | Leaving setup risks reconstructing inputs; missing material requires another route | Same-tab drafts retain parameters and exact input refs across close, navigation and return. Local upload binds the returned exact revision to the originating slot. `run-drafts.spec.ts` and the real stack. |
| Keyboard and nested input forms | Child forms can capture parent submission or lose focus on close | Enter/Tab/Shift+Tab/Escape stay in the active Git dialog; parent and background are inert; Escape restores trigger focus. Dismissing Git or local-file setup sends no mutation and retains the parent Run draft. `dialogs.spec.ts`, `run-drafts.spec.ts`, `git-artifacts.spec.ts`. |
| Lost launch response | Retrying can create a second Run or lose the original submission identity | A simulated network failure retains the exact body/key after leaving setup. Only explicit retry sends the second request. The real Server returns the original Run ID for an identical accepted request. |
| Result → next Run | Result inspection requires an Artifact detour; repeating means rebuilding setup | Primary preview loads in place. Server repeat-draft supplies exact retained inputs; ordinary failed Runs require review; an existing local draft wins over a new repeat proposal. Audit-managed Runs return to their owning Audit. `run-repeat.spec.ts` and the real stack. |
| Pending Audit review | Counts, evidence and decisions lack a common bounded context | Server workspace/counts and revision-bound finding/review pages; exact evidence → decision → original filtered queue. A changed revision requires refresh. Report acceptance is beside the exact report. Dismissed cancellation sends no request; accepted writes carry the expected CAS revision. `audits.spec.ts` plus PostgreSQL tests. |
| Operator observation → edit | Readiness competes with diagnostics and large inline forms | Readiness first; diagnostics collapsed; explicit publication/credential/binding dialogs; current and proposed versions shown separately. CAS reload preserves the proposal, denied credential creation clears the secret, and capability guards hide unauthorized forms. `operations-forms.spec.ts` and the real stack. |
| Safe file inspection | Rich documents can trigger unwanted remote content or navigation | Markdown, diff, OpenAPI and LikeC4 render locally at both viewports. Script/remote-image checks and external-request capture pass. Primary text preview remains on the Run route. `artifact-preview.spec.ts`, `run-repeat.spec.ts`. |

Visual inspection of the generated mobile screenshots also found an inherited
`Hide details` pseudo-label colliding with the Workflow technical disclosure
heading. The overview now suppresses that redundant label in both closed and
open states; the existing mobile screenshot journey was rerun after rebuilding.

## Observed actions and navigation

These are counts from the scripted interactions, not usability-study estimates.
A control activation or file selection counts as one action; typing individual
characters is not counted. Setup preceding each named start point is excluded.

| Start → outcome | Observed actions | Observed result |
| --- | --- | --- |
| Open Run setup → local file bound | Upload action, choose file, submit: **3** | **0** route changes; exact returned revision selected and confirmed; parent parameter retained. |
| Filled standalone setup → leave and resume | Close, Artifacts, browser Back, Configure: **4** on desktop; opening Menu adds **1** on mobile | Parameter and exact input restored; browser local/session storage remains empty in the standalone fixture. |
| Primary result card → rendered text | Preview result: **1** | **0** route changes; **2** exact reads (metadata and bytes), no eager byte read. |
| Terminal ordinary Run → repeat setup | Configure another Run: **1** | Exact Workflow/version and original input populated; **0** Run-create requests until review and submission. |
| Pending review queue → evidence → queue | Review finding, return link: **2** | State filter, page-two cursor and Audit revision retained. |
| Audit → dismiss cancellation | Cancel, Keep Audit unchanged: **2** | **0** cancellation requests; the preceding review decision remains the only mutation. |
| Ambiguous launch → explicit retry | First submit, later Retry exact request: **2** submissions | Same exact body and idempotency key; navigation alone sends **0** extra submissions. |

## Evidence environments

**Browser fixtures:** production UI build served by `node ui/server/index.mjs`
on `http://127.0.0.1:43173`. Each test installs its own API routes. The separate
fixture API origin is `http://127.0.0.3:8080`; some tests intentionally fixture
the UI origin. No shared-demo mutations or restarts were performed.

After building, start the isolated fixture UI in a separate terminal:

```sh
CONTRACTOR_UI_HOST=127.0.0.1 CONTRACTOR_UI_PORT=43173 \
CONTRACTOR_UI_API_BASE_URL=http://127.0.0.3:8080 node ui/server/index.mjs
```

Restart this process after another build so its retained index matches the
new asset filenames. Always pass the explicit isolated Playwright base URL.

Chromium/Playwright 1.62.1; desktop 1440×1000 and mobile 390×844. Existing 320px
layout regressions remain included. Run drafts, primary previews, failed-Run
repeat, nested keyboard dialogs, Audit review and Operations forms run explicitly
at both acceptance viewports. This is not a Safari/Firefox or touch-device study.

**Real integration:** disposable PostgreSQL 17.11 in Podman, a fresh isolated
schema per test, actual Go Server, Python Runtime, production Node UI, TLS browser
proxies and deterministic local model/credential-manager fixtures. No live model
service is used. Node 24.20.0, pnpm 11.24.0, Go 1.25.6, Python 3.13.14.

`make test-ui-stack` runs only `e2e/stack.spec.ts`; fixture-only browser scenarios
run in the separate browser gate. The real connected path is:

1. Catalog → exact `streamline-copy@1` → upload local text into the missing slot.
2. Set parameters/metadata → close → Artifacts → Back → retained exact draft.
3. Create Run → replay the same accepted request → same Run ID.
4. Observe execution/Operations, reconnect events and restart the isolated UI.
5. Server reports success → preview primary output → download identical bytes.
6. Prepare an exact repeat draft with review still required, without another Run.
7. Create a Project, upload ZIP/OpenAPI material and explicitly choose
   `openapi-from-workspace@5` from the family default of 7; complete the fixture
   workflow and read its published result.
8. Exercise configuration publication, credential lifecycle and Agent labels;
   verify Server/UI restarts and sign-out.

After browser sign-out, the harness restarts its isolated Server with another
local-auth owner over the same database. Both created Runs retain the original
owner; the new owner receives 404 for Run detail, repeat-draft and artifact
listing, sees an empty Run list, and cannot read the uploaded UserScope artifact.
The database contains exactly **2** Runs: accepted-request replay and preparing a
repeat draft created none. Existing CSRF/CORS and secret checks remain active:
browser requests/responses, WebSockets, storage, cookies and traces are inspected.

The new Audit workspace/finding/review reads are exercised separately against
PostgreSQL by `internal/auditservice`, `internal/auditstore` and public HTTP tests:
owner denial, whole-filter counts, current-round status, exact review identity,
revision fences and stale cursors. Browser Audit data is explicitly synthetic;
this report does not claim an integrated end-to-end live-model Audit.

## Commands and results

Run from the repository root with the local Node binary on `PATH` and
`CONTRACTOR_TEST_DATABASE_URL` pointing to the disposable database. The commands
were executed on the final UI implementation; static checks were repeated after
the final browser-test additions.

```sh
make ui-typecheck ui-lint ui-test ui-build

export CONTRACTOR_UI_E2E_BASE_URL=http://127.0.0.1:43173
export CONTRACTOR_UI_E2E_API_URL=http://127.0.0.3:8080
cd ui
corepack pnpm exec playwright test \
  e2e/run-drafts.spec.ts e2e/dialogs.spec.ts e2e/run-repeat.spec.ts \
  e2e/operations-forms.spec.ts e2e/catalog.spec.ts \
  e2e/project-workspace.spec.ts e2e/artifact-preview.spec.ts \
  e2e/audits.spec.ts e2e/responsive-layout.spec.ts e2e/git-artifacts.spec.ts
cd ..

make test-ui-stack
make verify-public-api
go test -count=1 ./internal/auditservice ./internal/auditstore ./internal/httpapi/public
```

| Gate | Outcome |
| --- | --- |
| UI typecheck, ESLint, Prettier, production build | Passed; Audit queue hook split removes the mixed-export Fast Refresh warning. Existing large preview-bundle build advisory remains. |
| UI unit tests | **57 files / 379 tests passed**. |
| Required fixture browser suite | **36 tests passed**. |
| Real `make test-ui-stack` | **Passed**; Server/Runtime workflows, exact reads, idempotency, ownership and secret boundaries. |
| Public API contract verification | **Passed**. |
| Audit service/store/public HTTP with PostgreSQL and `-count=1` | **All three packages passed**, without skipping database cases. |

Successful command logs are retained locally under ignored
`.local/v37-ui-journeys/journey-*.log`. The fixture suite produces screenshots
such as `audit-coverage-desktop.png`, `audit-coverage-mobile.png`,
`catalog-agent-desktop.png`, `catalog-agent-mobile.png` and
`workflow-technical-mobile.png` beneath `ui/test-results/ui-stack/`.
Playwright regenerates this output directory on subsequent runs. Real-stack
evidence is checked inside its temporary directory and removed by test cleanup;
the committed assertions and commands are the reproducible record.

## Remaining limits

- Same-tab draft continuity excludes reload, another tab, sign-out and another
  owner. This preserves the in-memory storage boundary.
- Scripted fixtures cover error/conflict/ambiguous states; they do not establish
  real-user discoverability or production-scale responsiveness.
- US-10 experiment comparison remains V38. Cross-version prompt comparison is
  also outside V37. Performance remains under V32-006; this gate does not close
  or redefine that milestone.
- The optional participant study remains unperformed. It should separately
  measure success without hints, time, backtracking and incorrect input choices
  with new and experienced users before setting numerical UX targets.
- No model budgets were tightened. No merge or demo deployment is part of this
  verification run.
