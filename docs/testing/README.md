# Testing

[Documentation index](../README.md) · [Development setup](../development.md)

Run commands from the repository root after installing the locked Runtime and
UI dependencies. Choose a check by the boundary you changed:

| Check | Command | Prerequisites |
| --- | --- | --- |
| Formatting, language tests, builds, shipped configuration and UI checks | `make verify` | Go, Python/uv, Node/Corepack |
| PostgreSQL repositories | `make test-postgres` | Test database |
| Live advisory scans of the Go commands and the Runtime lock | `make advisories` | Go, uv and access to the Go and PyPI advisory services |
| Server/Runtime process integration | `make test-e2e` | Test database and locked Runtime environment |
| Real scanner process and browser integration | `make test-scan-e2e` | Test database, Chromium, and host `nuclei`, `naabu`, `sqlmap`, `ffuf`, `katana` |
| Separate Node UI with the real Go/Python stack | `make test-ui-stack` | Test database and Chromium with host libraries |
| Basic scan/sweep widgets through the real API | `make test-web-scan-ui-stack` | Test database and locked Runtime/UI; no Studio or managed Eval scenarios |
| API-mocked browser journeys | `make ui-browser-mocked` | Node/Corepack and Chromium with host libraries |
| Native and external managed Evals | `make test-evals`; [evidence and reproduction](evals-release-gate.md) | Disposable test database, locked Runtime/UI and Chromium |
| Generic iterative Audit store/scale/recovery matrix | `make test-audit-iterative-postgres` | Dedicated disposable PostgreSQL and locked Runtime |
| Two-Round iterative Audit journey and direct/default bypass | `make test-audit-iterative-e2e` | Dedicated disposable PostgreSQL and real Server/Runtime; scripted local provider |
| Aggregate deterministic release gate used by CI | `make release-verify` | Go, Python/uv, Node/Corepack, Chromium with host libraries and a test database; no advisory service, host scanners or Podman |

The aggregate targets and their exact dependencies are defined in the
[Makefile](../../Makefile), which includes the grouped target files under
[`make/`](../../make). Targets needing a test database declare
`require-database`, and those reaching Python declare `runtime-venv`; the
latter prepares locked Python dependencies and prefetched
versioned parser assets, so capability probes and finding publication do not
rely on a cold grammar download. Both are prerequisites, so make prepares each
at most once per invocation.
Gate definitions describe what a command checks; completed results are
recorded in the corresponding task files.

## Temporary directory on macOS

Go's `t.TempDir()` uses `TMPDIR` as given. The default macOS `TMPDIR` lies
under `/var/folders`, and `/var`, like `/tmp`, is a symlink into `/private`.
The filesystem Artifact blob store refuses a directory with a symlink
component, so every test that roots one in a temporary directory fails there
with `invalid artifact blob directory: artifact blob storage I/O failed`. The
[Makefile](../../Makefile) therefore exports the symlink-free form of `TMPDIR`
(`/tmp` when it is unset) to every recipe; on Linux, where neither has a
symlink, this changes nothing. A `go test` run outside make on macOS, such as
a task's focused test command, needs the same resolution in its shell first:

```shell
export TMPDIR="$(cd "${TMPDIR:-/tmp}" && pwd -P)"
```

Do not work around the failure by relaxing the blob store's symlink check; a
symlinked blob directory stays refused in production.

## Runtime suite on macOS

The ordinary Runtime targets Linux and macOS, while the allocation-scoped
Podman backend needs Linux host capabilities
([OPS-R1](../spec/operations/operations.md)). Unit tests of that backend's
owner, supervisor and allocation protocol open `AF_UNIX` `SOCK_SEQPACKET`
socket pairs or call `pidfd_open`, which macOS lacks, so they carry the
`linux_podman` marker of
[`runtime/tests/platform_markers.py`](../../runtime/tests/platform_markers.py):
`make test-runtime` reports them as skipped with the OPS-R1 reason on any
host other than Linux, and nothing skips them on Linux. The real-engine
Podman suites skip on every host unless their explicit gate variable is set.

Every other Runtime test must pass on both. A test that needs the path of an
open descriptor uses `descriptor_path` of
[`runtime/tests/descriptor_paths.py`](../../runtime/tests/descriptor_paths.py)
rather than `/proc/self/fd`, runs fake executables with the test interpreter
rather than `/usr/bin/python3` (a launcher on macOS that adds its own
environment variables), and compares resource limits with
`resource.RLIM_INFINITY` rather than `-1`. These macOS failures are open
defects, each owned by a task; any other failure on macOS is a new defect:

- V425-001: `test_projectfs_local_io.py::test_limits_deadline_names_and_binary_classification`
  (an NFD opaque leaf is readable through its NFC name on APFS).
- V425-002: `test_code_analysis_graph.py::test_graph_collection_stops_at_encoded_result_ceiling_with_resumable_cursor`
  (the Trailmark mirror exceeds the 1,024-byte macOS `PATH_MAX`).

## Release gate stages

`make release-verify` runs the stages below in this order, cheapest first, so
a lint, unit or UI failure is reported before the long suites start. Every
stage is also a make target. [CI](../../.github/workflows/ci.yml) selects
stages from the files changed by a PR. A `v*` release tag or a manual
`workflow_dispatch` run on any branch selects the full gate. The selector
is [checked with mapping tests](../../scripts/test_select_ci_stages.py).
Selected jobs run concurrently against PostgreSQL 17; one failure does not
cancel the others. `pr-verify` requires every selected PR job to pass, while
`release-verify` requires every stage on a tag or manual run. Each stage
uses `make -k` and uploads its log and available evidence as a
`reports-<stage>` artifact even when it fails. Locally,
`make -k release-verify` reports every failing stage in one run.

PR selection always includes lint. Server and Runtime changes add unit,
integration and focused cross-process capability checks; UI changes add UI
checks and the affected browser stacks.
Changes to `tests/e2e` add all three process shards. Changes to the shared
Server build helper under `tests/internal/serverbuildtest` add unit, all three
process shards and both browser shards. Edits confined to
`ui/e2e/stack.spec.ts` run the focused operations stack; other `ui/e2e`
edits select the operations browser shard (both browser shards for the
managed-Evals stack). Shared API and configuration changes select Go, UI,
browser and integration checks. Release-only race and family passes remain
available through a tag or a manual full-gate run before a PR is merged.

| Stage | Runs |
| --- | --- |
| `release-verify-lint` | `make lint build test-config`: the generated Go public client check, gofmt, vet, staticcheck, the test-only export check (`tools/testonlyexports`, with its vet and tests), the release-graph guard and its tests, the database ERD freshness check, Ruff, the Runtime import-layer contracts, the command builds, the configuration package tests and validation of the shipped catalog (`config validate --root ./configs`) |
| `release-verify-unit` | Every Go package (including untagged e2e gate tests), the seven tagged configuration fixtures, and one full Runtime run with `pytest -W error tests`; shared fixture reads and required Audit/findings cases are verified from these reports |
| `release-verify-ui` | `make ui-verify`: generated-type check, lint, typecheck, unit and server tests, and the production build |
| `release-verify-families` | The feature families' UI and matrix checks; Go and Runtime execution belongs to the canonical stages |
| `release-verify-browser-a` | The API-mocked browser journeys, production operations browser stack, and native managed Evals |
| `release-verify-browser-b` | External managed Evals in the production browser stack |
| `release-verify-race` | The deduplicated Go race pass over the platform packages named in `make/release.mk` |
| `release-verify-race-discovered` | The Go race pass over every other package with tests, discovered by [`scripts/release_race_packages.py`](../../scripts/release_race_packages.py), so a new package is raced automatically |
| `release-verify-integration` | Every PostgreSQL-only integration-tagged Go test and the Audit completion/iterative selections under the race detector, with their required cases verified from one canonical report, a pass without it for packages whose tests relax a budget under the race detector, such as the 10-second finding-collection deadline, and every package whose tests take their Artifact blob backend from `CONTRACTOR_TEST_ARTIFACT_BACKEND`, run whole on the filesystem backend (`make test-artifact-filesystem`) |
| `release-verify-process-a` | The first shard of the canonical process e2e tests, including all iterative Audit cases and their required-case report check |
| `release-verify-process-b` | The second shard of the canonical process e2e tests |
| `release-verify-process-c` | The third shard of the canonical process e2e tests, including the four findings process tests and their required-case report check |

The process and browser shards run on separate CI runners. `make test-e2e`
still runs all canonical process tests in one local command, and `make test-ui-stack`
still runs the full browser stack locally. The release-graph guard checks that
each selected process and browser test runs exactly once.

The three aggregate race targets (`test-release-go-race`,
`test-release-go-race-discovered`, `test-release-integration`) run at most two
package binaries concurrently by default. Set `RACE_PACKAGE_PARALLELISM=1`
when diagnosing contention or reproducing serial execution, for example
`make test-release-go-race RACE_PACKAGE_PARALLELISM=1`. This setting changes
package scheduling only; tests inside each package keep their existing
scheduling. Before increasing it, measure peak PostgreSQL connections
(including admin pools and direct connections), memory and elapsed time on
the runner. PostgreSQL advisory locks also require checking their database
scope; separate schemas alone do not isolate every resource.

The serial sitemap publication-volume fixture runs without race instrumentation
in the complete ordinary Go unit pass. Its 257 publications, more than 256 MiB
of cumulative encoded content and production 15-second operation deadline stay
unchanged. Race instrumentation can consume that deadline on the first large
publication, so this fixture lives in a `!race` test file. The package still runs
whole under the race detector, including concurrent writers, receipts, retention
holds and lifecycle fences. The release graph requires its complete non-race
owner pass as well.

Inside the stages, family targets skip their own Go and Runtime suites and browser stack
in favor of the race, integration, process and browser passes; running a
family target directly still runs its focused suites. Every Go package with
tests runs under the race detector in exactly one of the two race stages, or
in the integration stage for tagged tests, unless
[`scripts/release_race_packages.py`](../../scripts/release_race_packages.py)
lists it as an exception with the reason. The graph guard compares every
direct family Go selection with the canonical stages, using compiled test
inventories to retain selected names, race coverage and backend modes. It
rejects Go, pytest and test-script execution anywhere in the consolidated
family dependency graph, and requires exactly one full Runtime run with
warnings as errors. The Audit completion, iterative Audit, findings and shared-fixture validators
consume canonical reports in `reports/`; direct family commands retain their
focused runs. The shared read journal is initialized once before either language writes,
including when consolidation is inherited from a target rather than an
environment flag. Report files are replaced on each run; missing, skipped or
incomplete required evidence fails the gate. Optional skips outside those
focused scopes retain their existing policy.
`make lint` checks the stage order, the CI jobs and this table against the
Makefile, the race coverage of every package with tests, and the release graph
against the original process-test inventory, and discovers every
integration-tagged test. It also requires the lint stage to keep the shipped
catalog validation, the ERD check and the test-only export check, and every
package whose tests read `CONTRACTOR_TEST_ARTIFACT_BACKEND` to run whole on
the filesystem backend in some stage. New names
enter the release pass automatically; tool-dependent exceptions must name an
opt-in gate and reason in
[`scripts/release_integration_tests.py`](../../scripts/release_integration_tests.py).
The guard also lists the real tests with `go test -list`: every `-run`
alternative in release-verify and in the opt-in gates must select an existing
test, and every e2e-tagged test must run in release-verify unless
[`scripts/check_release_verify_graph.py`](../../scripts/check_release_verify_graph.py)
allowlists it with its opt-in target and the reason (real scanners or Podman).

## Iterative Audit acceptance

`make test-audit-iterative-postgres` runs the exact store, scale and recovery
selection in [`audit_iterative_matrix.yml`](../../tests/e2e/audit_iterative_matrix.yml)
(each case appears once, with its boundary and acceptance links)
with the race detector. It includes the 4,096-item Round inside a 100,000-item
Audit, 32-dependency closures, bounded capture/admission/materialization and
populated deletion. Fixtures use real PostgreSQL and deterministic work/query
counters; elapsed time does not substitute for a row or byte limit.

`make test-audit-iterative-e2e` runs the same matrix's production-process
selection. The composed journey has an ordinary check, its settled-Round
assessment, a receipt-derived typed later check and a second assessment.
Assertions compare exact source, retained and consumer bytes and attribution
across producer deletion and Server restart. Separate cases retain the
more-than-128-page local snapshot reads and the direct model-free/default Audit
bypass proofs. These cases need no installed scanner, Chromium, hypothesis
registry, paid provider or live-model quality assumption.

Both direct targets execute their focused harness unit checks once per make
invocation. Missing prerequisites, a zero/missing/skipped required selection,
changed source/matrix, command failure or an incomplete report fails the gate.
The Audit gates use one strict Go-event verifier and one process-group capture
owner in `scripts/release_test_reports.py`; completion keeps its Runtime JUnit
policy and composition keeps its browser checks. Family root scopes remain
explicit, so an unlisted importer case cannot hide a selected skip.

Fresh Audit completion, composition and iterative execution evidence requires a clean checkout and records the
actual source revision, matrix path/digest, report digest and executed case
identities, including focused descendants. It does not copy acceptance prose
from the matrix. Report-only verification checks the supplied stream without
executing tests or publishing fresh source evidence. Historical task evidence
retains its original schema and source pins.
Independent finite command deadlines and signal cleanup stop and join the
owned process group. The harness does not retry a failing whole suite.
Use a separate disposable database for each concurrent lane; sharing a schema
does not isolate database-scoped Server leases.

Each direct gate writes a unique directory under `.local/audit-iterative-gate/`
with a bounded redacted report and `result.json`. The result records source
revision/dirty state, matrix and report digests, platform capabilities and the
actual required identities. It contains no database URL, credentials or target
bodies. A failed or interrupted attempt cannot reuse earlier passing evidence.
The final delivery record is [`v62-013.json`](../../tasks/evidence/v62-013.json).

Inside `release-verify`, these family targets skip their own suites. The
canonical integration report and process shard A report are verified through
`release_test_reports.py verify-iterative-postgres` and
`verify-iterative-process`; lint's existing unittest discovery runs the harness
checks. Snapshot and typed-routing processes belong to the normal release
shard. The broader scanner/browser composition gate keeps its separate
infrastructure requirements.

## Advisory scans

`make advisories` runs `govulncheck` over the production Go commands and audits
the production Runtime graph: `uv export --locked` fails when `uv.lock` no
longer matches `pyproject.toml`, development-only packages are excluded, and
`pip-audit` is installed with its whole dependency closure from the
hash-pinned
[`scripts/pip-audit-requirements.txt`](../../scripts/pip-audit-requirements.txt).
The audit first requires the scanner to report a known-vulnerable pin, so a
scanner that silently reports nothing cannot pass, and then fails on any
published advisory for the lock. A scan's verdict changes whenever upstream
publishes an advisory, without a code change, so CI runs the scans in the
separate `advisories` job on release tags and manual runs, outside the
`release-verify` check. The
release gate queries no advisory service; apart from downloading its pinned
tools and locked dependencies it needs no network. All CI jobs run on the
pinned `ubuntu-24.04` runner image.

## Database and process tests

Provide an explicit disposable test database whose user may create and drop
schemas and databases (`CREATEDB`). Most tests and the process tests create
isolated schemas in it and clean them up; some create databases of their own
on the same server and drop them afterwards. PostgreSQL itself is supplied by
the caller. Database tests skip only while `CONTRACTOR_TEST_DATABASE_URL` is
unset: once it is set, an unreachable server or a missing privilege fails
them, and `make lint` rejects helpers that skip after a failed connection.

```shell
export CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable'
make test-e2e
```

### Read budgets

A test that budgets the rows or blocks a statement reads measures it with
[`explaintest`](../../internal/persistence/explaintest), whose counts include
the index entries of row versions VACUUM has not removed yet. VACUUM keeps
every version that an older snapshot in the same database may still see, and
`go test` runs packages in parallel against the one test database, so any
other test's open transaction can keep the superseded versions of a fixture
that updated or deleted rows. A budget must not absorb them with a fixed
slack. Such a fixture is built with `explaintest.Database` instead, in a
database of its own whose VACUUM no other database's session holds back, and
with autovacuum off for its tables. That helper also holds a snapshot open in
the shared test database for the whole test, so the budgets are always
measured while another session's snapshot is older than the fixture. A
fixture that only inserts rows, or a budget that already counts every version
its fixture writes, may stay in a schema of the shared database.

The process harness starts the actual Go Server and Python Runtime entry points,
creates temporary mTLS identities, and uses a deterministic loopback model
Gateway. It verifies scheduling, artifacts and cleanup without a live model.
Each browser process stack builds and serves one private temporary UI snapshot.
A developer build or another gate may replace ui/dist without invalidating a
running stack. The scanner browser likewise builds privately, or copies an
explicitly requested prebuilt snapshot before starting Node.

Each process-test package builds the Server lazily once and retains its binary
until `TestMain` finishes all tests and their cleanup. Test enumeration and
configuration-only selections do not build it. Separate CI shards have separate
binaries. The scratch-container opt-in tests reuse their own `CGO_ENABLED=0`
binary and copy it into each isolated container build context. PKI, databases,
scenario state and deadlines remain owned by each test. To measure the build
portion of 25 scenarios with a warm Go build cache, run:

```shell
go test -run '^$' -bench '^BenchmarkServerBuild' -benchtime=1x -count=3 ./tests/internal/serverbuildtest
```

This compares repeated builds with one cached executable; it does not measure
full process-suite or CI wall time.

For a release, use the same environment:

```shell
make release-verify
```

## Focused checks

| Area | Entry point |
| --- | --- |
| Server/Runtime wire contracts and every shared fixture under `api/testdata` | `make test-wire-cross-language` (test database and locked Runtime environment); [private contracts](../../api/v1alpha1/README.md) |
| Project workspaces, artifact handoffs and output publication | `make test-project-workspaces-release`; [Project guide](../guides/project-workflows.md) |
| Heterogeneous Runtime capability placement | `make test-capability-e2e`; [Runtime configuration](../operations/runtime-configuration.md) |
| Cloud/local placement and inherited Audit labels | `make test-runtime-placement-e2e`; real Server/Runtime processes and PostgreSQL Audit companions, with deterministic tool observations and no model/scanner installation |
| Runtime configuration, labels, browser stack, lifecycle, Scheduler, Memory and HTTP/Caido | [Release gates and diagnosis](release-gates.md) |
| Agent Skill packaging, seeding, updates and cleanup | [Agent Skills](../guides/agent-skills.md) |
| Worker session modes and lockstep rollout | [Runtime upgrades](../operations/runtime-configuration.md#worker-session-mode-upgrade) |
| Audit programs | `make test-audit-program-library-e2e`; [Audit walkthrough](../guides/audits.md) |
| Findings producer/collection/reader | `make test-findings-e2e`; [contract](../spec/findings/findings.md) |
| Controlled HTTP identity preflight | `make test-engagement-http-process`; [independent process, socket and accounting evidence](engagement-http-conformance.md) |
| Performance collection and profiling | [Operations performance](../operations/performance.md) |
| Podman execution | [Podman provisioning and verification](../../deploy/podman/README.md) |
| Runtime host provisioning (Ansible role) | `make test-ansible` (also in `make lint`; needs `uv`); [Ansible guide](../../deploy/ansible/README.md) |
| PostgreSQL/filesystem artifact payloads | [Blob backend verification](../operations/artifact-blob-storage.md#release-verification) |
| Project sitemap operation authority, shared-map views and exact context | `make test-project-sitemap-postgres` and `make test-project-sitemap-tools-e2e`; Runtime view/cleanup/size cases in `runtime/tests/test_sitemap_*.py`; [sitemap contract](../spec/artifacts/project-sitemap.md) |
| Retained Artifact history, exact-reference holds and history errors | `make test-artifact-history-postgres` and `make test-artifact-history-e2e` (test database; the second also the locked Runtime environment); [retention contract](../spec/artifacts/retention-and-deletion.md) |
| Git import | [Git artifacts](../guides/git-artifacts.md) |
| Backup/restore with exact artifact bytes and CAS | `make test-backup-restore`; needs a PostgreSQL role with CREATE/DROP DATABASE, `psql`, and `pg_dump` and `pg_restore` of the server's major version or newer, which the target checks before it creates a database; [offline backup and restore](../operations/artifact-blob-storage.md#offline-backup-and-restore) |

Podman, Git, backup/restore and blob-backend checks have their own host/image requirements;
consult their guides before running those targets.

A fixture that tests of more than one side (Go, the Runtime, the UI) read
lives under `api/testdata`. A test that reads one is named `TestShared*` in Go,
where it reads through `internal/contracts/contractstest`, and
`test_shared_*` in the Runtime, where a test module that reads one while it is
imported (to parametrize its tests) holds only such tests; any other test or
module that reads one fails. A Go `Fuzz*` target may seed its corpus from one,
and a Runtime helper module may read one at import to build test inputs only
when `runtime/tests/shared_fixtures.py` lists it with its reason; neither
counts as a reader. A process test (`tests/e2e`) uses a copy under
`tests/e2e/testdata` that a shared test keeps identical.
`api/testdata/readers.json` declares the sides that read each fixture, and
`make test-wire-cross-language` runs every shared test and holds the observed
readers to it ([`scripts/shared_fixture_gate.py`](../../scripts/shared_fixture_gate.py)).
`make lint` runs only the gate's source checks: no other Go path into
`api/testdata`, no `TestShared*` test behind a build constraint, and no file in
another `testdata` directory that tests of two sides name by its literal path,
unless the gate's `OUTSIDE_SHARED` names it with its reason.

## Live-model and research evaluations

[Live-model evaluation](live-models.md) covers Gateway dialect checks, Router,
Worker finalization, summarization and project-workflow quality. These checks
call a real model and require a separately chosen budget and configuration.

For versioned datasets and reproducible comparisons, use the
[portable evaluation specification](../spec/evals/format.md),
[fixture catalog](../../configs/evals/README.md) and
[instruction-evaluation guide](../../tests/eval/agent_instructions/README.md).
Offline conformance checks do not measure model quality.

`make test-nuclei-web-e2e` requires PostgreSQL and runs a deterministic scanner
protocol fixture through the actual shipped Workflow, Server, Scheduler and
Runtime. It also launches a seven-Project client sweep and drops one committed
create acknowledgement to prove replay and stable membership. It is selected
by the consolidated release process group C without host scanner requirements.
`make test-nuclei-web-installed` is the explicit real Nuclei engine smoke for
all individual knobs, exact package/template provenance, invalid admission and
changed-package refusal.
