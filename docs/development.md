# Local development

[Documentation index](README.md)

This guide covers dependency setup and the first local checks. To run the
application, continue with [the local stack](guides/local-stack.md). For
integration and release checks, use [the testing guide](testing/README.md).
Implementation status and recorded evidence live in [tasks/index.yml](../tasks/index.yml).

## Prerequisites

| Component | Repository requirement |
| --- | --- |
| Host | Linux or macOS for Server/ordinary Runtime; Linux for the CI stack and Podman sandbox; see [deployment](deployment.md) |
| Go | 1.25 or newer; see [go.mod](../go.mod) |
| Python | 3.13 and `uv`; see [runtime/pyproject.toml](../runtime/pyproject.toml) |
| UI | Node 24.20, Corepack 0.36 and pnpm 11.24; see [ui/package.json](../ui/package.json) |
| PostgreSQL | Required to run Server and database/process tests; CI uses PostgreSQL 17 |
| Shell examples | `curl`, `jq` and Git; OpenSSL for the optional Gateway key setup |

A local application stack also needs an OpenAI-compatible LLM Gateway. The
deterministic process tests supply their own fake Gateway. Source-analysis
Workflows need additional [Runtime capabilities](guides/project-workflows.md#runtime-workspace-requirements).

Run commands from the repository root unless a block explicitly changes directory.

## Install dependencies

```shell
uv sync --project runtime --locked
npm install --global corepack@0.36.0
make ui-install
```

Install Chromium when you need browser process tests:

```shell
make ui-browser-install
```

If Playwright reports missing host libraries, install them using the same
setup as [CI](../.github/workflows/ci.yml).

## Build the commands

Build the CLI and auxiliary commands from the repository root and add them to
the current shell's search path:

```shell
go build -o ./bin/ ./cmd/...
export PATH="$PWD/bin:$PATH"
contractor --help
```

Rebuild after changing Go code. The guides use the installed `contractor`
command: `contractor server run`, `contractor server migrate` and
`contractor pki`. The build also produces the `contractor-skill` packaging
helper and the standalone `contractor-server` used by container examples and
offline blob cleanup, which is not exposed by the unified CLI.

## Validate and test

Validate the executable configuration before starting Server:

```shell
contractor server config validate --root ./configs
make verify
```

`make verify` runs formatting/lint checks, Go and Python tests, builds the
commands, and generates, checks, tests and builds the UI. It does not require a
running PostgreSQL database; database tests need the explicit test URL described
in the [testing guide](testing/README.md).

To work on one component:

| Task | Command |
| --- | --- |
| Go tests | `make test-go` |
| Runtime tests | `make test-runtime` |
| UI checks and build | `make ui-verify` |
| Build commands and compile Runtime Python | `make build` |
| Regenerate public Go client | `make generate-public-client` |
| Regenerate public UI types | `make ui-generate` |

For architecture changes, `make verify-architecture` validates the LikeC4 model
(it is part of `release-verify`). With the LikeC4 CLI installed:

```shell
likec4 validate docs/spec
likec4 start docs/spec
```

Continue with [deployment](deployment.md), the [user guides](guides/README.md)
or [testing](testing/README.md).
