# Test gate cost review — 2026-10-23

**Revision reviewed:** `main` `dc7012e8` (Makefile and `make/*.mk`, `.github/workflows/ci.yml`, `scripts/select_ci_stages.py`, the Go and Runtime test trees).

**Scope:** why `make verify`, `make test` and `make release-verify` take long, and whether the time buys the checks the project needs. Measured: the Runtime suite on macOS (`uv run pytest --durations`). Not measured: Go and e2e stage durations (no Go toolchain or PostgreSQL in the review environment); CI `reports-<stage>` artifacts hold them.

**Outcome:** the checks are the right ones (real PostgreSQL, shared fixtures, process tests, race pass); the cost comes from how often each is repeated and from per-test setup. The Runtime suite is not the problem.

## Status (2026-10-23)

All findings open. Tasks V408-001 … V408-007 (milestone `test-gate-cost-2026-10`); V408-007 records the before/after stage durations and closes the review.

| Finding | Task |
| --- | --- |
| T1 per-test schema and migrations, 34 helper copies | V408-002 |
| T2 serial race stages (`-p 1`), almost no `t.Parallel` | V408-003 |
| T3 e2e builds the Server and PKI per test | V408-004 |
| T4 family-target guard gaps, 20 unguarded pytest runs, unused `test-runtime-hardening` | V408-005 |
| T5 `-count=1` on pure packages | V408-006 |
| Baseline and final measurement | V408-001, V408-007 |

## Findings

**T1. Every PostgreSQL-backed Go test creates a schema and applies all 31 migrations.** 34 test files each carry their own copy of an `isolated*Pool` helper (`CREATE SCHEMA` + `persistencepostgres.ApplyMigrations`, baseline 250 KB), called from about 341 test sites (`isolatedArtifactPool` 48, `isolatedRunStorePool` 46, `isolatedSchedulerPool` 35, `isolatedAuditServicePool` 33, `isolatedPublicPool` 26, …). `make release-verify` runs the Go packages three times (unit without race, two race stages, integration-tagged tests again under race), so one gate applies the migration set on the order of a thousand times. Only 2 of 171 files with these helpers use `t.Parallel()`; helpers pin `MaxConns = 1` or `2`.

**T2. Race stages are serial.** `test-release-go-race`, `test-release-go-race-discovered` and `test-release-integration` use `go test -p 1 -race`. The commit that added the discovered stage (V340-001) recorded 204 s for 41 packages at `-p 2`. With `-p 1` and no in-package parallelism the race stages are the slowest Go stages (CI timeouts 45–50 min).

**T3. Each process e2e test builds and provisions everything itself.** 19 tagged tests in `tests/e2e`; each runs `go build ./cmd/contractor-server` (21 files call it), `migrate`, generates a CA and two leaves, starts the Server and the Python Runtime, with per-test budgets of 110 s to 7 min; shard timeouts 80 min × 3.

**T4. Family targets repeat work inside `release-verify-families`.** `run-family-test` skips Go suites only where it is applied. Unguarded `go test` lines: `make/features.mk:15,40` (audit hardening, integration race), `make/platform.mk:22,25,83,119,134,138,139` and `make/toolsets.mk:84,107,108`. All 20 `uv run pytest` lines in family targets run unguarded: `tests/test_allocation.py` 7 times, `tests/test_lease_watchdog.py` 5, `tests/test_adk_runtime.py` (3048 lines) 3, `tests/test_adapter_host.py` 3. `test-runtime-hardening` (`pytest -W error tests`) exists in `make/core.mk:18` but no stage calls it, so `-W error` coverage is the union of those subsets.

**T5. The Go test cache is disabled almost everywhere.** 84 of 93 `go test` invocations in `make/*.mk` pass `-count=1`, including pure packages (`internal/config` 163 tests, `internal/scanplan` 84). `test-hardening-matrices` (untagged `tests/e2e`, 35 tests) runs with `-count=1` on every `make test`.

**Runtime suite (measured).** 4217 passed, 110 failed (macOS: Podman AF_UNIX, no `go`, Linux-only cases), 39 skipped, 6 errors in 149 s. Slowest: `test_scan_katana_live.py` 25 s total (runs the installed katana), Podman recovery tests with real 3–5 s waits, `test_metrics.py` 2.8 s. 23 `time.sleep(60)` calls live in child-process fixtures that are killed early. Nothing structural to fix.

**Observations kept out of scope.** 410 tests set contexts with 10 s or longer timeouts while injected clocks exist in most packages; `make release-verify` locally runs the twelve stages sequentially although CI runs them on separate runners, so a local `release-verify` costs the sum of all stage times.
