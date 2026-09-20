# V38-007–010 implementation

Initial base: main `22c464c1`. Worktree: `feat/v38-managed-evals-ui`.
Committed main through `033cccd4` was integrated during verification; unrelated
uncommitted work in the canonical checkout remains separate.
Scope: complete native setup, comparison/review, the independent Playground
client and deterministic release acceptance from spec 30.

## Preflight

- [x] Verify V38-001–006 completion against task evidence and main history.
- [x] Reconcile spec 30, experience design, public DTOs and legacy UI routes.
- [x] Identify the missing safe readiness projection: native preparation verifies
  equality pins, but the public setup view does not expose their coverage.
- [x] Re-run existing Eval integration and legacy UI checks.
- [x] Describe required additive API changes and the independent client boundary.

## Delivery

- [x] V38-007: dataset authoring/import, exact Workflow/Audit variants, persisted
  drafts, preparation/readiness and recoverable explicit controls.
- [x] V38-008: experiment discovery, paired comparisons, attributable review,
  charts, exports and evidence navigation, preserving legacy workspaces.
- [x] V38-009: optional public API client in the canonical sibling Playground
  repository, with durable recovery and existing portable compatibility.
- [x] V38-010: real isolated Server/database/browser and independent HTTP-client
  gates, fault coverage and recorded evidence. No live models or shared deploy.

## Verification record

Core Contractor implementation: `851dae1ee77baff7c2abc7423a2a76a737a114fb`.
Shared browser/reproduction follow-up: `04a1f4ae02a83ffbbe9dbc198a205f165fe511bd`.
Main integration: `26f9e13b31f34b8019d5f40c1e7de898e5e7b50d`.
Playground implementation: `8f57b08ff168111adcac1bed190f98c025ac67d3`,
branch `feat/v38-managed-evals-client`, based on `d97ddc5`.
Initial member-projection recovery: `d1f7e970e3db12a231bffca565748d1293cbed6d`.

The UI separates dataset editing, variant mapping, assessment, readiness,
command recovery, comparison, charts, inventory and human review into focused
modules. Bounds and supported comparison policies are named. Saved drafts use
Server CAS; browser storage retains command/correlation metadata without cases,
private rubrics or review text. Start confirms the originally reviewed revision.

The public API additions are optional setup/list/readiness/inventory metadata:
registered check schemas, safe pin origins/equality status, per-arm eligibility,
list summaries and exact execution Project IDs. List A/B ordering follows the
comparison identity. No private binding/expected contents become generic views.
OpenAPI, generated Go/TypeScript clients and the copied Playground schema agree.

Playground keeps HTTP, journal, portable projection, collection, private scoring,
dispatch and CLI concerns separate. Durable intent precedes each mutation;
replay uses the original body/key/CAS. Frozen member bindings, comparison policy,
execution order and inventory pins are checked. Only supported ordinary artifact
mappings are translated; unsupported bindings fail explicitly. Existing portable
index/shard authority is preserved.

Environment: Linux amd64, Go 1.25.6, Runtime/Playground Python 3.13.14,
standard-library producer Python 3.14.7, Node 24.20.0, pnpm 11.24.0,
PostgreSQL 17.11. Chromium uses Playwright's Ubuntu 24.04 fallback on this host.
The local Node binary and disposable PostgreSQL container are task-specific;
no shared demo, live target, paid model or default production policy was changed.

Verified release checks:

- `corepack pnpm exec vitest run`: 64 files, 446 tests passed; static UI-server
  tests: 9 passed. Both TypeScript configurations, ESLint and Prettier passed.
- `corepack pnpm exec playwright test e2e/evals-setup.spec.ts
  e2e/evals-comparison.spec.ts e2e/evals-skills.spec.ts`: 10 passed at 390/1280px.
  These use mocked API fixtures; screenshots/traces are separate from process
  evidence. Dialog focus and direct production routes were fixed during review.
- Full Eval domain/store/service/coordinator race gate: 213 tests/subtests,
  no skips. Public API Eval gate: 13 tests/subtests, no skips. The full service
  package intentionally includes fault tests whose names do not contain `Eval`.
- The 10,000-member fixture measured 50 pair pages, two SQL queries per page,
  maximum response 632,876 bytes and 20 metric bins. Stage accounting, missing
  observations, exact cohorts and stale cursor/fence cases pass.
- `make verify-public-api`, `go vet ./...`, repeated OpenAPI/Go/TypeScript
  generation and `git diff --check`: passed. Schema copies compare byte-for-byte.
- Playground: `uv run --extra dev pytest tests/test_managed_evals.py`: 21 passed;
  `uv run --extra dev pytest`: 454 passed; new Python module/test Ruff checks pass.
  A newly registered external experiment waits for its first member projection
  before dispatching. This delay is covered by a separate regression fixture.
- Running `scripts/test-managed-evals.py` without database configuration returns
  exit 1. Required database/process checks cannot silently skip.

Commands, artifacts and the distinction between process, PostgreSQL and mocked
browser assertions are in [the release gate](../testing/evals-release-gate.md).

## Final delivery and process evidence

All V38 tasks are completed. Primary completion commits:

| Task | Repository | Commit |
| --- | --- | --- |
| V38-007 | Contractor | `5f9a75a5b8d0ac68aa67e4c0f07902c2e2594f1b` |
| V38-008 | Contractor | `d0e37d1acfbc5abe14d343f3484a4a297c1c313d` |
| V38-009 | Playground | `d1f7e970e3db12a231bffca565748d1293cbed6d` |
| V38-010 | Contractor | `2c1fb13e185f54bd5c2d3d384033f7ec068b988d` |

Shared form/button styling: `38afad0b622a6464ed887154c7297ca26912594a`.
Playground guide clarification: `4a4e81d5846a6dc7aae75756d3bcc8e5cbb702d0`.
The complete process gate ran on product code `d0e37d1acfbc5abe14d343f3484a4a297c1c313d`,
with a clean worktree at invocation. The later V38-007 commit only gives the
complete UI setup test a local 15-second timeout under concurrent test load;
it changes no product or process-harness code. Final UI checks include it.

`make test-evals` passed with 213 domain/store/service/coordinator tests and
subtests, 13 public API tests/subtests, and five process-package tests, without
skips. The latter contains two deterministic Gateway tests and these three
real journeys:

| Journey | Elapsed seconds | Observed result |
| --- | ---: | --- |
| Operations | 115.97 | Existing production UI/Runtime, configuration and secret-boundary journey passed |
| Native Workflow + Audit | 508.95 | Two eight-member experiments; explicit UI Start, lost response, Server/browser restart, exact plan/deadline, review and comparison |
| Independent external Workflow + Audit | 484.97 | Two eight-member experiments; standard-library HTTP producer, 52 lost-response replays, full inventory, attributed assessments and explicit finalization |

Each mode verifies exact one-member/one-ordinary-execution association and
rejects foreign-owner reads on 12 experiment/evidence routes. Real Audits have
two check children; other roles, retries, overlaps and missing reports are
covered by the PostgreSQL accounting fixtures. Private truth is absent from
Worker requests and exported evidence. The Runtime cannot import Playground;
the independent producer runs with `python3 -I -S`.

`make ui-typecheck ui-lint ui-test ui-build` passed: 64 files, 446 tests.
The ten mocked browser scenarios pass, including keyboard switching of the
single mobile Overview chart and simultaneous desktop charts. Static UI-server
tests (nine), public API verification, generated-client parity and Go vet pass.
The optional Playground suite passes 454 tests, including 21 managed tests.

Recorded artifacts are in the canonical Contractor checkout under
`.local/evidence/v38-release`: `gate.json`, all three Go JSONL logs,
`native-evidence.json`, `external-evidence.json`, and real browser traces and
390px/1280px screenshots under `browser/native` and `browser/external`.
Mocked browser artifacts are separate in `.local/evidence/v38-mocked-final`.
Initial failing/debug runs are retained separately; they are not release proof.

The final metadata check validates all 393 indexed tasks, the dependency graph,
V38 acceptance/test mappings, local documentation links and unique numbered
specifications. No V38 task remains pending. Reproduction and the limits of
these deterministic checks are in the [release gate](../testing/evals-release-gate.md).
