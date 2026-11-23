# Code review of main — 2026-10-04

[Research index](README.md)

- **Date:** 2026-10-04; status rechecked 2026-10-23 at `af9fd522`.
- **Scope:** the last 500 commits on `main`, `126a2e56^..347b790e`: tasks
  V158-001..V277-001 including V179-001..009, PRs #423–#536, 1,030 files,
  +35k/−15.7k lines, all landed on 2026-10-03/04. Line numbers in the task
  files refer to `347b790e`.
- **Method:** each task was checked against its goal, acceptance criteria and
  actual diff, and the current code of each area beyond the diff. Findings were
  reproduced with tests on a disposable PostgreSQL where feasible.
- **Outcome:** 86 tasks: V278-001..V340-001, V347-001, V348-001 and the
  follow-ups V341-001..V346-001 and V349-001..V363-001. Each task file holds
  its location, failure scenario and acceptance criteria.

## Status on 2026-10-23

- 84 of the 86 tasks are completed and merged into `main`.
- **V341-001** (in progress): CI probe-duration measurements are still needed
  to justify or reduce the factory and Trailmark probe timeouts. The
  deterministic Audit fixture asserts exact attempts again.
- **V347-001** (pending): `main` is still unprotected. The prepared request is
  [2026-10-09-main-protection.json](2026-10-09-main-protection.json); see the
  [maintenance backlog](2026-10-09-maintenance-backlog.md#remaining-remote-acceptance).
- The test-infrastructure lessons below became AGENTS.md §3.5 and §3.6.

## Summary

- Local gates at `347b790e` were green: Go build, vet, gofmt, staticcheck
  U1000, `go test` with PostgreSQL packages, Ruff, pytest (3,276 passed,
  39 skipped), UI generate:check, lint, typecheck, vitest and build. One pytest
  test was flaky under load (V316-001).
- GitHub CI `release-verify` had never passed since it was created on
  2026-05-29, and `main` was not protected, so every merge in the range was
  ungated (V334-001, V347-001). After the review, run 37186027201 on
  `bc7d5f37` passed in 85m43s once #537 raised the job timeout to 150 minutes
  (V155-001).
- 28 of 127 completed tasks in the range were partial or regressed: V159,
  V161, V162, V173, V174, V177, V179-002/005/008/009, V180, V188, V194, V205,
  V210, V211, V221, V234, V235 (broke V165), V237, V240, V250, V253, V258,
  V260, V262, V265 and V267.
- Verified correct: the behaviour-preserving refactors V270–V277 (AST and SQL
  compared before and after), the Go part of V179 (all eight drift defects
  fixed) and the earlier review fixes V163, V164 and V166–V168.

## Findings

| Group | Count | Tasks by area |
| --- | ---: | --- |
| High | 5 | Audit V278, V280 · Artifacts V286 · Control Plane V288 · Runtime V309 |
| Medium | 16 | Audit V281, V283 · Control Plane V289 · Planner V293 · Evals V298, V299 · UI V305, V306 · Runtime V313–V315, V317, V320, V322 · Contracts V323 · Git/config V329 |
| Low | 32 | Audit V279, V282, V284, V285 · Artifacts V287 · Control Plane V290–V292 · Planner V294–V297 · Evals V300, V301 · UI V307, V308 · Runtime V310–V312, V316, V318, V319, V321 · Contracts V324–V328 · Git/config V330–V333 |
| CI, gates, bookkeeping | 9 | CI V334–V340 · repository setting V347 · task records V348 |
| Follow-ups | 24 | Evals V302–V304 · found while fixing V341–V346, V349–V363 |

All IDs end in `-001`; the files are in [tasks/done](../../tasks/done/), except
V341-001 and V347-001 in [tasks](../../tasks/).

## Decisions taken while fixing

- V309-001 scrubs only credentials the Runtime injected and secrets it holds;
  values set by the model or other proxy users stay visible. The remaining
  Caido search oracle over stored requests is an accepted risk (see the task).
- V323-001 keeps `+` in configuration versions (Audit standard versions use
  it) and aligns Go, Python, schemas, OpenAPI and the UI to one grammar with
  shared fixtures.
- V304-001 showed that PR #536 misread a quadratic per-row trigger as a slow
  CI runner: a 20k-row insert spent 9.4 of 10.7 s in the trigger and now takes
  about 1 s.

Measured effects: schema-scoped migration locks (V338-001) halved the
PostgreSQL test packages' wall time and removed the parallel-load timeouts
(`internal/artifacts` 240–600 s → 44 s); trusted plan reads (V302-001) cut a
1,000-member Eval tick from 3.3 s to 0.2–0.6 s; view-generation pruning
(V303-001) keeps one generation instead of ten per lifecycle.

## Test-infrastructure lessons

- Fakes hid real defects: the importer batch test passed only because its fake
  store skipped `validateCollect` (V280-001); V240 tests used a copied
  predicate (V281-001); V211 tests mocked vacuum (V318-001); the Eval UI
  fixture still modelled pre-V235 revisions (V299-001).
- Wall-clock assertions were fragile: V221's 50 ms loop-gap check (V316-001),
  V240's 8.8 s-of-10 s collection test (V281-001) and the race/non-race budget
  split from #534 (V337-001).
- PostgreSQL tests migrated 76 versions per isolated schema behind one
  database-wide advisory lock, so parallel packages serialized (V338-001).

## Limits

Not run: the full `make release-verify`, Playwright browser suites, e2e tests
that need Podman or real scanners, and a global `-race` pass. CI job logs are
admin-only, so the cause of the CI failure 71 minutes into `347b790e` was not
established. Findings marked plausible in their task files were not
reproduced.
