# Testing

[Documentation index](../README.md) · [Development setup](../development.md)

Run commands from the repository root after installing the locked Runtime and
UI dependencies. Choose a check by the boundary you changed:

| Check | Command | Prerequisites |
| --- | --- | --- |
| Formatting, language tests, builds and UI checks | `make verify` | Go, Python/uv, Node/Corepack |
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
`require-database`, and those reaching Python declare `runtime-venv`; both are
prerequisites, so make prepares each at most once per invocation.
Gate definitions describe what a command checks; completed results are
recorded in the corresponding task files.

## Release gate stages

`make release-verify` runs the stages below in this order, cheapest first, so
a lint, unit or UI failure is reported before the long suites start. Every
stage is also a make target. [CI](../../.github/workflows/ci.yml) selects
stages from the files changed by a PR. A `v*` release tag or a manual
`workflow_dispatch` run on any branch selects the full gate. The selector
is [checked with its mapping tests](../../scripts/select_ci_stages.py).
Selected jobs run concurrently against PostgreSQL 17; one failure does not
cancel the others. `pr-verify` requires every selected PR job to pass, while
`release-verify` requires every stage on a tag or manual run. Each stage
uses `make -k` and uploads its log and available evidence as a
`reports-<stage>` artifact even when it fails. Locally,
`make -k release-verify` reports every failing stage in one run.

PR selection always includes lint. Server and Runtime changes add unit and
integration checks; UI changes add UI checks and the affected browser stacks.
Changes to `tests/e2e` add both process shards. Edits confined to
`ui/e2e/stack.spec.ts` run the focused operations stack; other `ui/e2e`
edits select the operations browser shard (both browser shards for the
managed-Evals stack). Shared API and configuration changes select Go, UI,
browser and integration checks. Release-only race and family passes remain
available through a tag or a manual full-gate run before a PR is merged.

| Stage | Runs |
| --- | --- |
| `release-verify-lint` | `make lint build`: gofmt, vet, staticcheck, the release-graph guard and its tests, Ruff, and the command builds |
| `release-verify-unit` | `make test`: the hardening matrices, every Go package (PostgreSQL-backed tests included when the test URL is set) and the Runtime suite |
| `release-verify-ui` | `make ui-verify`: generated-type check, lint, typecheck, unit and server tests, and the production build |
| `release-verify-families` | The feature families' Runtime, UI and matrix checks, and the Audit completion and findings process gates |
| `release-verify-browser-a` | The API-mocked browser journeys, production operations browser stack, and native managed Evals |
| `release-verify-browser-b` | External managed Evals in the production browser stack |
| `release-verify-race` | The deduplicated Go race pass over the platform packages named in `make/release.mk` |
| `release-verify-race-discovered` | The Go race pass over every other package with tests, discovered by [`scripts/release_race_packages.py`](../../scripts/release_race_packages.py), so a new package is raced automatically |
| `release-verify-integration` | Every PostgreSQL-only integration-tagged Go test under the race detector, and a pass without it for packages whose tests relax a budget under the race detector, such as the 10-second finding-collection deadline |
| `release-verify-process-a` | The first shard of the 19 process e2e tests |
| `release-verify-process-b` | The second shard of the 19 process e2e tests |

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
integration-tagged test. New names
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
| Project workspaces, artifact handoffs and output publication | `make test-project-workspaces-release`; [Project guide](../guides/project-workflows.md) |
| Heterogeneous Runtime capability placement | `make test-capability-e2e`; [Runtime configuration](../operations/runtime-configuration.md) |
| Runtime configuration, labels, browser stack, lifecycle, Scheduler, Memory and HTTP/Caido | [Release gates and diagnosis](release-gates.md) |
| Agent Skill packaging, seeding, updates and cleanup | [Agent Skills](../guides/agent-skills.md) |
| Worker session modes and lockstep rollout | [Runtime upgrades](../operations/runtime-configuration.md#worker-session-mode-upgrade) |
| Audit programs | `make test-audit-program-library-e2e`; [Audit walkthrough](../guides/audits.md) |
| Findings producer/collection/reader | `make test-findings-e2e`; [contract](../spec/27-findings-tools-and-collections.md) |
| Performance collection and profiling | [Operations performance](../operations/performance.md) |
| Podman execution | [Podman provisioning and verification](../../deploy/podman/README.md) |
| PostgreSQL/filesystem artifact payloads | [Blob backend verification](../operations/artifact-blob-storage.md#release-verification) |
| Git import | [Git artifacts](../guides/git-artifacts.md) |
| Backup/restore with exact artifact bytes and CAS | `make test-backup-restore` | PostgreSQL role with CREATE/DROP DATABASE, `pg_dump`, `pg_restore` |

Podman, Git, backup/restore and blob-backend checks have their own host/image requirements;
consult their guides before running those targets.

## Live-model and research evaluations

[Live-model evaluation](live-models.md) covers Gateway dialect checks, Router,
Worker finalization, summarization and project-workflow quality. These checks
call a real model and require a separately chosen budget and configuration.

For versioned datasets and reproducible comparisons, use the
[portable evaluation specification](../spec/26-portable-evaluation-format.md),
[fixture catalog](../../configs/evals/README.md) and
[instruction-evaluation guide](../../tests/eval/agent_instructions/README.md).
Offline conformance checks do not measure model quality.
