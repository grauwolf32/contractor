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
| API-mocked browser journeys | `make ui-browser-mocked` | Node/Corepack and Chromium with host libraries |
| Native and external managed Evals | `make test-evals`; [evidence and reproduction](evals-release-gate.md) | Disposable test database, locked Runtime/UI and Chromium |
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
Changes to `tests/e2e` add all three process shards. Edits confined to
`ui/e2e/stack.spec.ts` run the focused operations stack; other `ui/e2e`
edits select the operations browser shard (both browser shards for the
managed-Evals stack). Shared API and configuration changes select Go, UI,
browser and integration checks. Release-only race and family passes remain
available through a tag or a manual full-gate run before a PR is merged.

| Stage | Runs |
| --- | --- |
| `release-verify-lint` | `make lint build test-config`: the generated Go public client check, gofmt, vet, staticcheck, the test-only export check (`tools/testonlyexports`, with its vet and tests), the release-graph guard and its tests, the database ERD freshness check, Ruff, the Runtime import-layer contracts, the command builds, the configuration package tests and validation of the shipped catalog (`config validate --root ./configs`) |
| `release-verify-unit` | `make test`: the hardening matrices, every Go package (PostgreSQL-backed tests included when the test URL is set) and the Runtime suite |
| `release-verify-ui` | `make ui-verify`: generated-type check, lint, typecheck, unit and server tests, and the production build |
| `release-verify-families` | The feature families' Runtime, UI and matrix checks, and the Audit completion and findings process gates |
| `release-verify-browser-a` | The API-mocked browser journeys, production operations browser stack, and native managed Evals |
| `release-verify-browser-b` | External managed Evals in the production browser stack |
| `release-verify-race` | The deduplicated Go race pass over the platform packages named in `make/release.mk` |
| `release-verify-race-discovered` | The Go race pass over every other package with tests, discovered by [`scripts/release_race_packages.py`](../../scripts/release_race_packages.py), so a new package is raced automatically |
| `release-verify-integration` | Every PostgreSQL-only integration-tagged Go test under the race detector, a pass without it for packages whose tests relax a budget under the race detector, such as the 10-second finding-collection deadline, and every package whose tests take their Artifact blob backend from `CONTRACTOR_TEST_ARTIFACT_BACKEND`, run whole on the filesystem backend (`make test-artifact-filesystem`) |
| `release-verify-process-a` | The first shard of the 19 process e2e tests |
| `release-verify-process-b` | The second shard of the 19 process e2e tests |
| `release-verify-process-c` | The third shard of the 19 process e2e tests |

The process and browser shards run on separate CI runners. `make test-e2e`
still runs all 19 process tests in one local command, and `make test-ui-stack`
still runs the full browser stack locally. The release-graph guard checks that
each selected process and browser test runs exactly once.

Inside the stages, family targets skip their own Go suites and browser stack
in favor of the race, integration, process and browser passes; running a
family target directly still runs its focused suites. Every Go package with
tests runs under the race detector in exactly one of the two race stages, or
in the integration stage for tagged tests, unless
[`scripts/release_race_packages.py`](../../scripts/release_race_packages.py)
lists it as an exception with the reason. Existing family
integration commands continue to run their focused untagged race checks.
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
schemas. Process tests create isolated schemas and clean them up; PostgreSQL
itself is supplied by the caller. Database tests skip only while
`CONTRACTOR_TEST_DATABASE_URL` is unset: once it is set, an unreachable server
fails them, and `make lint` rejects helpers that skip after a failed connection.

```shell
export CONTRACTOR_TEST_DATABASE_URL='postgres://contractor:password@127.0.0.1:5432/contractor_test?sslmode=disable'
make test-e2e
```

The process harness starts the actual Go Server and Python Runtime entry points,
creates temporary mTLS identities, and uses a deterministic loopback model
Gateway. It verifies scheduling, artifacts and cleanup without a live model.

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
| Runtime configuration, labels, browser stack, lifecycle, Scheduler, Memory and HTTP/Caido | [Release gates and diagnosis](release-gates.md) |
| Agent Skill packaging, seeding, updates and cleanup | [Agent Skills](../guides/agent-skills.md) |
| Worker session modes and lockstep rollout | [Runtime upgrades](../operations/runtime-configuration.md#worker-session-mode-upgrade) |
| Audit programs | `make test-audit-program-library-e2e`; [Audit walkthrough](../guides/audits.md) |
| Findings producer/collection/reader | `make test-findings-e2e`; [contract](../spec/findings/findings.md) |
| Performance collection and profiling | [Operations performance](../operations/performance.md) |
| Podman execution | [Podman provisioning and verification](../../deploy/podman/README.md) |
| Runtime host provisioning (Ansible role) | `make test-ansible` (also in `make lint`; needs `uv`); [Ansible guide](../../deploy/ansible/README.md) |
| PostgreSQL/filesystem artifact payloads | [Blob backend verification](../operations/artifact-blob-storage.md#release-verification) |
| Git import | [Git artifacts](../guides/git-artifacts.md) |
| Backup/restore with exact artifact bytes and CAS | `make test-backup-restore`; needs a PostgreSQL role with CREATE/DROP DATABASE, `psql`, and `pg_dump` and `pg_restore` of the server's major version or newer, which the target checks before it creates a database; [offline backup and restore](../operations/artifact-blob-storage.md#offline-backup-and-restore) |

Podman, Git, backup/restore and blob-backend checks have their own host/image requirements;
consult their guides before running those targets.

A test that reads a fixture under `api/testdata` is named `TestShared*` in Go,
where it reads through `internal/contracts/contractstest`, and
`test_shared_*` in the Runtime, where a test module that reads one while it is
imported (to parametrize its tests) holds only such tests; any other test or
module that reads one fails. A Runtime helper module may read one at import to
build test inputs only when `runtime/tests/shared_fixtures.py` lists it with
its reason.
`api/testdata/readers.json` declares the sides that read each fixture, and
`make test-wire-cross-language` runs every shared test and holds the observed
readers to it ([`scripts/shared_fixture_gate.py`](../../scripts/shared_fixture_gate.py)).
`make lint` runs only the gate's source checks: no other Go path into
`api/testdata`, and no `TestShared*` test behind a build constraint.

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
