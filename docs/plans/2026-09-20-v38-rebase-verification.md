# V38 rebase verification — 2026-09-20

`feat/v38-managed-evals-ui` is rebased onto main `b162df2452a68fab2ea622c6dcb428f0cdd8e187`.
All seven feature commits transferred without conflicts or corrective product
changes. `git range-diff` marks every patch equivalent. The original branch is
retained at `backup/v38-managed-evals-ui-7db1b014` (`7db1b014881989b65f8823b1a88977f79f58ba03`).

## Commit mapping

| Before rebase | After rebase | Change |
| --- | --- | --- |
| `851dae1e` | `98ad7b81` | feat(evals): add native setup, comparison review and deterministic release gate |
| `04a1f4ae` | `a343ffd3` | test(evals): document release checks and refresh shared browser journey |
| `38afad0b` | `a9237f77` | fix(ui): align Evals controls with shared form styles |
| `2c1fb13e` | `f3d00ccd` | test(evals): isolate native and external release journeys |
| `d0e37d1a` | `a5aaa119` | fix(evals): select one overview chart on narrow screens |
| `5f9a75a5` | `d2a843d3` | test(evals): allow full setup journey under concurrent load |
| `7db1b014` | `dfcc96a1` | docs(evals): close V38 with verified native and external evidence |

The feature tip before this verification-metadata commit is `dfcc96a1bbce4927d577d137027206e77dd38561`.
Completed Contractor V38 task references use the corresponding new hashes;
Playground's independent repository was not rebased. The
[original delivery record](2026-09-20-managed-evals-ui.md) retains its historical
commits and measured evidence.

## Preservation of main changes

The initial target was `5fad87a3d4e783f3322bb72755c4195f717d1f14`. Of its 135 changed paths since the
feature's previous integrated base `033cccd4`, 133 remain byte-for-byte equal to
main. The two intersections are `ui/server/static-server.mjs` and its tests:
Evals adds its explicit client routes while keeping main's single-decoding path
validation, traversal rejection and API-like route exclusions. The final static
server suite passes all 11 tests.

Session/CSRF generation fences, Runtime cancellation ownership and process
cleanup, transaction-bound credential reads, manual escalation resumption and
migration 59→60, terminal Audit projection refresh, credential-picker pagination,
raw HTTP query preservation, Scan Planner and the Audit/Runtime readability
refactor are retained. Their source files are unchanged from main.

## Verification

Required process checks ran on `30a05fa6b8ad36b974d19314b001d6360d257c11`, with a clean worktree at invocation.
While checks were running, main received its V60-025 completion metadata. A
second rebase includes that commit. Comparing the tested and final trees shows
only documentation/task metadata changes; API, configuration, production source,
Runtime, UI and test files are identical.

| Check | Result |
| --- | --- |
| Evals domain/store/service/coordinator, PostgreSQL and race detector | 213 tests/subtests passed; no skips |
| Evals public API, PostgreSQL and race detector | 13 tests/subtests passed; no skips |
| Process package | Five tests passed, including Gateway helpers and the three real journeys below; no skips |
| Supplemental Go race/PostgreSQL | 14 packages, 814 tests/subtests passed; no skips |
| Runtime ownership/process/source/HTTP/validator/allocation regressions | 278 passed |
| Full UI | 463 tests in 67 files passed with two workers |
| Static Node UI service | 11 passed; no skips |
| Mocked Evals browser journeys | 10 passed at 390px/1280px |
| TypeScript, ESLint/Prettier, production UI build, Go vet and public API verification | Passed |
| Managed schema parity with Playground | Byte-for-byte equal |

Supplemental Go packages cover Control Plane, RuntimeConfig, credentials, app,
RunStore, Scheduler, PostgreSQL persistence and every Planner package. Runtime
checks include repeated cancellation, workspace/process cleanup ownership and
raw HTTP query preservation. The UI suite includes stale 200/401 CSRF/session
responses, terminal Audit refresh and complete credential-picker pagination.

| Real process journey | Seconds |
| --- | ---: |
| Operations | 101.53 |
| Native Workflow and Audit | 509.01 |
| Independent external Workflow and Audit | 481.78 |

Both modes retain two eight-member experiments, ordinary Run/Audit association,
exact evidence and four pairs per experiment. Native setup/reload/Start/server
restart/review/comparison and 52 external lost-response replays pass. Foreign
owners cannot read 12 experiment/evidence routes per mode. Private truth is
excluded from model requests and exports. The test stack has no Playground
runtime dependency and uses a deterministic loopback Gateway.

## Commands and retained artifacts

The [release gate guide](../testing/evals-release-gate.md) describes database,
Runtime and browser setup. This verification used Go 1.25.6, Runtime Python
3.13.14, standard-library producer Python 3.14.7, Node 24.20.0, pnpm 11.24.0 and
PostgreSQL 17.11 in a dedicated disposable container.

```sh
make test-evals
cd ui
corepack pnpm typecheck
corepack pnpm lint
corepack pnpm test --run --maxWorkers=2
node --test server/*.test.mjs
corepack pnpm build
CONTRACTOR_UI_E2E_BASE_URL=http://127.0.0.1:4191 \
  corepack pnpm exec playwright test e2e/evals-setup.spec.ts \
  e2e/evals-comparison.spec.ts e2e/evals-skills.spec.ts
```

The standalone browser command used a separately started Vite service.
Exact supplemental commands are recorded in `verification.json`; complete Go
JSON events and Runtime logs are retained in the canonical checkout's
`.local/evidence/v38-rebase-5fad87a3`. This directory contains `gate.json`, all
three gate JSONL files, `upstream-go.jsonl`, `ui.log`, `runtime.log`, `browser.log`,
`contract.log`, `vet.log`, both range-diffs, native/external reports and browser
traces/screenshots. `verification.json` records the final base, feature tip and
full commit mapping. Production deployment and live model evaluation were not
part of this rebase verification.
