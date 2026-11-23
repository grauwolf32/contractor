# Test gate cost review — 2026-10-23

[Historical documents](../README.md). Retained design/review evidence; current
contracts belong in the specification and current status in the task records.

**Revision reviewed:** `main` `dc7012e8` (Makefile and `make/*.mk`, `.github/workflows/ci.yml`, `scripts/select_ci_stages.py`, the Go and Runtime test trees).

**Scope:** why `make verify`, `make test` and `make release-verify` take long, and whether the time buys the checks the project needs. Measured: the Runtime suite on macOS (`uv run pytest --durations`). Not measured: Go and e2e stage durations (no Go toolchain or PostgreSQL in the review environment); CI `reports-<stage>` artifacts hold them.

**Outcome:** the checks are the right ones (real PostgreSQL, shared fixtures, process tests, race pass); the cost comes from how often each is repeated and from per-test setup. The Runtime suite is not the problem.

## Status (2026-10-24)

V408-003 (bounded race package scheduling), V408-004 (package-lifetime Server
builds) and V408-005 (consolidated family runs with coverage guards) are
implemented and merged. V408-008 and V408-009 repaired verification fixtures.

At the user's explicit request, the remaining V408-001, V408-002, V408-006 and
V408-007 are archived. No further V408 optimisation or timing campaign is
scheduled. Their original scope and unmet acceptance remain in the archived
task files; this disposition does not claim they were completed. The extra
three consecutive Linux CI race runs from V408-007 were not performed and
are no longer scheduled. Existing gates, parallelism, deadlines and the
requirement for evidence before increasing parallelism or reducing budgets
remain unchanged. Other independently tracked correctness work is unaffected.

| Finding | Disposition |
| --- | --- |
| T1 per-test schema and migrations, 34 helper copies | [V408-002](../../../tasks/done/v408-002-shared-postgres-test-schema.yml): archived; retain per-test isolation |
| T2 serial race stages (`-p 1`), almost no `t.Parallel` | [V408-003](../../../tasks/done/v408-003-parallel-race-stages.yml): package scheduling implemented; in-package parallelism excluded |
| T3 e2e builds the Server and PKI per test | [V408-004](../../../tasks/done/v408-004-e2e-build-once.yml): binary reuse implemented; PKI and deadlines remain per test |
| T4 family-target guard gaps and repeated runs | [V408-005](../../../tasks/done/v408-005-family-guard-gaps.yml): implemented and verified |
| T5 `-count=1` on pure packages | [V408-006](../../../tasks/done/v408-006-go-test-cache.yml): archived; no additional cache classifier |
| Baseline and final measurement | [V408-001](../../../tasks/done/v408-001-gate-duration-baseline.yml), [V408-007](../../../tasks/done/v408-007-gate-duration-after.yml): archived at user request |

## Findings

**T1. Every PostgreSQL-backed Go test creates a schema and applies all 31 migrations.** 34 test files each carry their own copy of an `isolated*Pool` helper (`CREATE SCHEMA` + `persistencepostgres.ApplyMigrations`, baseline 250 KB), called from about 341 test sites (`isolatedArtifactPool` 48, `isolatedRunStorePool` 46, `isolatedSchedulerPool` 35, `isolatedAuditServicePool` 33, `isolatedPublicPool` 26, …). `make release-verify` runs the Go packages three times (unit without race, two race stages, integration-tagged tests again under race), so one gate applies the migration set on the order of a thousand times. Only 2 of 171 files with these helpers use `t.Parallel()`; helpers pin `MaxConns = 1` or `2`.

**T2. Race stages are serial.** `test-release-go-race`, `test-release-go-race-discovered` and `test-release-integration` use `go test -p 1 -race`. The commit that added the discovered stage (V340-001) recorded 204 s for 41 packages at `-p 2`. With `-p 1` and no in-package parallelism the race stages are the slowest Go stages (CI timeouts 45–50 min).

**T3. Each process e2e test builds and provisions everything itself.** 19 tagged tests in `tests/e2e`; each runs `go build ./cmd/contractor-server` (21 files call it), `migrate`, generates a CA and two leaves, starts the Server and the Python Runtime, with per-test budgets of 110 s to 7 min; shard timeouts 80 min × 3.

**T4. Family targets repeat work inside `release-verify-families`.** `run-family-test` skips Go suites only where it is applied. Unguarded `go test` lines: `make/features.mk:15,40` (audit hardening, integration race), `make/platform.mk:22,25,83,119,134,138,139` and `make/toolsets.mk:84,107,108`. All 20 `uv run pytest` lines in family targets run unguarded: `tests/test_allocation.py` 7 times, `tests/test_lease_watchdog.py` 5, `tests/test_adk_runtime.py` (3048 lines) 3, `tests/test_adapter_host.py` 3. `test-runtime-hardening` (`pytest -W error tests`) exists in `make/core.mk:18` but no stage calls it, so `-W error` coverage is the union of those subsets.

**T5. The Go test cache is disabled almost everywhere.** 84 of 93 `go test` invocations in `make/*.mk` pass `-count=1`, including pure packages (`internal/config` 163 tests, `internal/scanplan` 84). `test-hardening-matrices` (untagged `tests/e2e`, 35 tests) runs with `-count=1` on every `make test`.

**Runtime suite (measured).** 4217 passed, 110 failed (macOS: Podman AF_UNIX, no `go`, Linux-only cases), 39 skipped, 6 errors in 149 s. Slowest: `test_scan_katana_live.py` 25 s total (runs the installed katana), Podman recovery tests with real 3–5 s waits, `test_metrics.py` 2.8 s. 23 `time.sleep(60)` calls live in child-process fixtures that are killed early. Nothing structural to fix.

**Observations kept out of scope.** 410 tests set contexts with 10 s or longer timeouts while injected clocks exist in most packages; `make release-verify` locally runs the twelve stages sequentially although CI runs them on separate runners, so a local `release-verify` costs the sum of all stage times.

## V408-003 scope decision (2026-10-23)

The first step is package scheduling only: all three aggregate race targets
use `RACE_PACKAGE_PARALLELISM`, default `2`; `1` restores serial execution.
The package lists, race detector, uncached execution, timeouts and integration
selection stay the same. The release-graph guard accepts positive integer
parallelism and still checks the complete package and test inventory.

This does not depend on V408-002: current helpers already create a distinct
schema for each test. No additional PostgreSQL test uses `t.Parallel` in this
change. The proposed single schema per package plus `TRUNCATE` between tests
in V408-002 cannot support simultaneous tests sharing that schema. Revisit
in-package parallelism only after choosing separate schemas for simultaneous
tests or another isolation design that covers multi-connection tests.

The original T1 summary overstates the pool cap: the runstore, artifacts and
scheduler helpers do not explicitly set `MaxConns` on their ordinary pools.
Connection accounting must include admin pools, additional working pools and
direct connections. Schema separation also does not isolate database-scoped
advisory locks. Keep the default at two packages until representative CI runs
show both connection/memory headroom and an elapsed-time improvement.

The complete baseline and repeated Linux CI comparison were originally assigned
to V408-001 and V408-007. Both are now archived at user request; local
measurements alone do not establish CI speed or absence of flakes.

### Local package-parallelism measurement

Implementation: `70d9cb11`, based on `main` `725b863e`.

Measured on macOS arm64, 18 logical CPUs and 36 GiB RAM, with Go 1.26.9
and a dedicated PostgreSQL 17 cluster. The test database uses UTF-8;
`max_connections=100`. The order was one `-p 2` sweep followed by one
`-p 1` sweep, reusing the Go build cache. Both kept `-race -count=1`,
the same package/test selection and the same timeouts. Each target was
invoked separately, including its locked Runtime setup and parser prefetch.
An initial SQL_ASCII setup attempt was discarded before this comparison.

| Target | Serial `-p 1` | Parallel `-p 2` | Reduction | Peak connections, serial / parallel |
| --- | ---: | ---: | ---: | ---: |
| `test-release-go-race` | 617.2 s | 308.3 s | 50.1% | 6 / 6 |
| `test-release-go-race-discovered` | 607.5 s | 351.9 s | 42.1% | 9 / 4 |
| `test-release-integration` | 269.4 s | 160.4 s | 40.5% | 5 / 7 |
| Sum of these local targets | 1494.1 s | 820.5 s | 45.1% | |

All six completed target invocations passed. An optional third sweep was
stopped after this successful comparison; no partial invocation is included
in the table. Connections were sampled approximately
every 0.2 seconds across all client backends of the dedicated cluster,
excluding the sampler connection; short spikes can fall between samples.
No run exhausted the connection limit. This is an observed peak, not a
hard per-package pool bound.

The observed process-tree RSS peak was 1584.4 MiB.
That sampler ran every 0.5 seconds from the discovered parallel stage
through the serial sweep. It excludes PostgreSQL and can count shared
pages repeatedly. It is a local observation, not a CI memory budget.

This is one exploratory local pair on a shared development machine.
It establishes neither a minimum CI speedup nor absence of flakes.
CI runs these stages as separate jobs, so their sum is not full CI wall
time; process/browser stage durations are unchanged by this change.
The full baseline and repeated Linux CI comparison originally assigned to
V408-001 and V408-007 are now archived at user request. Do not increase the default or lower CI timeouts from these
local numbers alone.

Verification: all three race targets on real PostgreSQL at `-p 2` and
`-p 1`; full `make lint`; 120 script unit tests in the Runtime environment;
release-graph checks with the default and the serial override. The graph
retains 98 raced packages and 106 discovered integration tests. The
parallel integration log contains no skipped tests.

Raw per-target logs, the JSON timing/connection summaries and the RSS
summary remain outside git under `/private/tmp/contractor-v408-*`.


## V408-005 implementation and verification (2026-10-23)

The consolidated family dry run previously contained 12 visible Go commands,
17 pytest commands, two gate-script invocations and the shared-fixture runner.
The updated family graph contains no Go, pytest or test-script executions.
Direct family targets retain their focused suites. The guard also compares
compiled test inventories across direct families and canonical stages, retaining
race coverage and explicit filesystem-backend modes. All 46 Go selections
captured before the change remain covered.

The unit stage runs the complete Runtime suite once with `-W error`, captures
JUnit and verifies the existing Audit and findings minimum case counts and
no-skip requirements. The intentionally malformed duplicate ZIP fixture now
explicitly expects zipfile's warning instead of failing during fixture creation.
Shared-fixture validation consumes canonical Go events and actual Go/Python read
records without rerunning tests. The tagged configuration fixtures run once in
the unit stage; untagged e2e gate tests join its ordinary Go pass. Audit completion
selection joins the canonical integration pass, and the four findings process
tests join process shard C. Report capture replaces stale files, propagates child
failure, redacts database credentials and joins the owned process group on
cancellation. Reports remain ignored local artifacts under `reports/`.

The graph retains 99 race packages, 24 process tests, seven tagged configuration
fixtures, 47 tagged e2e tests (eight opt-in), 159 integration-selection names and
both Artifact backend modes. These counts describe selection, not a green full
release gate. No CI speedup or full release pass is claimed from dry-run counts.

The complete Linux Runtime gate passed with 4375 tests and 55 existing optional
skips; mandatory Audit/findings scopes had no skips. Script regressions passed
(131 tests), including transitive unwrapped commands, lost tags/race/backend
coverage, filtered Runtime runs, missing cases, skips, stale reports, subprocess
failure, secret redaction and cancellation. Shared evidence verified 249
fixtures, 229 Go shared tests and 184 Python shared tests.

Broader checks also exposed existing failures at base `e4480ace`, in unchanged
Go paths: the namespace-constant source guard; SQL table ownership for the new
sweep CLI; the public paging inventory missing `listScanSweeps` and
`listScanSweepMembers`; and the skills process test expecting a fixed catalog
list. Resource-related lease/performance failures from overlapping local sweeps
passed when both complete packages were rerun serially on a separate database.
At initial verification these broader failures were retained and V408-005
was left in progress; the closure follow-up below records the final result. The original worktree's Make fragment
was repeatedly overwritten externally, so final work continues in the dedicated
`v408-005-family-tests` worktree. Raw verification logs remain outside git under
`/private/tmp/v408-*.log`.


### V408-005 closure after main integration

The current main already contains the sweep-page fixture repair V405-006.
Two additional test-only blockers were repaired independently: V408-008 derives
the Skills process expectation from its staged SeedPlan and follows bounded
two-item public pages; V408-009 supplies explicit empty labels and placement
requirements in the completion importer baseline. Production validation remains
strict.

All four commands listed by V408-005 passed: `make lint`,
`make test-runtime-hardening`, `make test-release-integration` and
`make test-release-process-e2e-c`. The Runtime result is 4375 passed and 55
existing optional skips. All 13 process-shard tests passed, including the four
migrated findings tests, with the original mandatory-case verifier. Integration
passed under `-race -p 2` with the original Audit-case verifier. Current shared
read evidence covers 249 fixtures, 231 Go shared tests and 184 Python shared
tests; the complete Runtime suite was not rerun for that evidence check.

The merged graph retains 99 race packages, 25 process tests, seven tagged
configuration fixtures, 49 tagged e2e tests (eight opt-in), 159 integration
selection names and both blob backend modes. The 131 script regressions and
compiled coverage guard passed. No speedup is inferred from these counts.

The pre-existing namespace-literal and SQL-ownership source guards were
reconfirmed as failures in unchanged production paths. Their owners remain the
artifactpolicy and persistence maintainers; `make test-go` and their package
suites detect them. Reopen these separately before claiming a green full release
gate. They are not Runtime-suite, duplicate-execution or coverage defects fixed
by V408-005. This follow-up claims the listed task checks, not a complete
`release-verify` or CI run. Logs remain under `/private/tmp/v408-close-*.log`.


## V408-004: package-lifetime Server builds (2026-10-24)

The accepted scope is binary reuse only. The original shared CA, deadline
reductions and shard rebalancing are excluded. V408-001 is no longer a
prerequisite for this narrower change. The broader CI timing and repeated
Linux stability checks originally assigned to V408-007 were later archived
at user request; they are not completed verification.

The ordinary process selection previously invoked `go build` 25 times. The
four browser stack scenarios each built again. Both harnesses now request a
lazy package-owned executable from `tests/internal/serverbuildtest`; `sync.Once`
shares one build and its error across callers. `TestMain` removes it after all
tests and their cleanup finish. Listing tests and configuration-only selections
allocate no executable. No binary survives into another test-package run.
Separate CI runners each still build once. The two opt-in scratch-container
paths keep a separate `CGO_ENABLED=0` build and copy that executable into each
test's own build context. Migrations, databases, PKI, Runtime processes,
assertions, deadlines and shard selections are unchanged. A source guard rejects
Server build commands in either harness tree; CI path selection includes all
process and browser consumers when the shared build helper changes.

### Local build-overhead measurement

Measured on macOS arm64 (18 logical CPUs, 36 GiB RAM), Go 1.26.9, based on
`main` `64e2f214745ea3e4c571ec9d070056bb6a1ca766`. The production Server
sources are unchanged. The Go build cache was warm. The benchmark compares
25 builds into distinct output paths with 25 requests to one fresh package
cache, including its one real build and cleanup. Each strategy ran three
one-iteration samples; repeated-build samples ran before reuse samples.

```shell
go test -run '^$' -bench '^BenchmarkServerBuild' -benchtime=1x -count=3 ./tests/internal/serverbuildtest
```

| Sample | 25 separate builds | One build reused 25 times |
| --- | ---: | ---: |
| 1 | 52.904 s | 1.501 s |
| 2 | 40.351 s | 2.584 s |
| 3 | 36.167 s | 1.526 s |
| Median | 40.351 s | 1.526 s |

The difference between medians is **38.825 seconds**, or **96.2% of the build
portion** for 25 scenarios. This is a local measurement on a shared development
machine with other checks running, not a guaranteed full-suite saving. It does
not establish full process-suite or CI wall-time speedup; there is no matched
before/after full-suite run. The browser suite avoids three repeated builds
when invoked together, while the three process and two browser CI shards each
retain their own build. Raw benchmark output remains outside git at
`/private/tmp/v408-004-build-benchmark.log`.


### V408-004 verification and scope limits

Implementation: `ba6fc0a50142dd3af03aa6db53ebe606c48e8e3d` (Server source unchanged).

- `go test -race -count=1 ./tests/internal/serverbuildtest` passed all five
  regressions: concurrent callers, binary lifetime, sticky failures/partial-file
  cleanup, lazy allocation and preserved compiler arguments/CGO modes.
- The complete untagged `go test -count=1 ./tests/e2e` passed. A temporary
  injected per-test Server build made the new source guard fail as expected;
  the injection was removed. Both tagged packages also listed/compiled without
  building the Server. All seven `make test-release-config-fixtures` cases
  also passed with zero Server builds.
- `make test-e2e` passed all **25** ordinary process tests on real PostgreSQL,
  including the existing findings report verifier, with **one** build marker.
- The final complete `make test-ui-stack` passed all **nine** top-level tests
  (four production stack scenarios and five harness checks), with **one** build
  marker across all four real stack scenarios.
- `make lint` passed with the shared-helper CI mapping regression and compiled
  release coverage guard. The graph retains 100 raced packages and the same
  25 process tests, seven configuration fixtures, 49 tagged e2e tests and eight
  opt-in tests. PKI, deadlines and stage timeouts are unchanged.

The successful local package durations were 2547.610 seconds
(process) and 1164.326 seconds (browser); these include actual
scenario execution and ran with other work on the same host. They are not
matched before/after full-suite measurements.

The first complete browser invocation had one Playwright click timeout at
`ui/e2e/stack.spec.ts:802` (`All workflows`); the widgets and both managed-Eval
scenarios passed in that invocation. The unchanged original-main operations
scenario then passed, and the complete unmodified retry on this branch passed.
No UI assertion or timeout was changed to obtain that result. The UI test
maintainers own recurrence; the full browser gate detects it, and the same
failure recurring should reopen an independent UI investigation. This records
an observed intermittent failure, not evidence of three-run CI stability.

The opt-in container/scanner scenarios were compiled, and CGO mode preservation
was unit-tested; their external-service acceptance runs were not performed.
No full release gate or remote CI run is claimed. Raw logs remain outside git
under `/private/tmp/v408-004-*.log`.
