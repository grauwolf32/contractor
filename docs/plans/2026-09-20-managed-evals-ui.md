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
- [ ] V38-010: real isolated Server/database/browser and independent HTTP-client
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
standard-library producer Python 3.14, Node 24.20.0, pnpm 11.24.0,
PostgreSQL 17.11. Chromium uses Playwright's Ubuntu 24.04 fallback on this host.
The local Node binary and disposable PostgreSQL container are task-specific;
no shared demo, live target, paid model or default production policy was changed.

Verified so far:

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

Final real process matrix and completion metadata are pending verification.
Commands, artifacts and the distinction between process, PostgreSQL and mocked
browser assertions are in [the release gate](../testing/evals-release-gate.md).
