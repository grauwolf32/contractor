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
| Go | 1.26.8 or newer; [go.mod](../go.mod) selects 1.26.8 automatically with `GOTOOLCHAIN=auto` |
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

Validation reads the sibling `managed-configs/` root without creating it;
pass `--managed-root` to match a different Server path.

`make verify` runs formatting/lint checks, Go and Python tests, builds the
commands, and generates, checks, tests and builds the UI. It does not require a
running PostgreSQL database; database tests need the explicit test URL described
in the [testing guide](testing/README.md). The CI gate, `make release-verify`,
runs the same checks first and then its heavier
[stages](testing/README.md#release-gate-stages); CI runs each stage as a
separate job.

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

## Layer checks

Two checks keep code dependencies pointing down the layer stack: a package or
module may import its own layer or a lower one. Both run in CI.

| Code | Layer table and known violations | Run locally |
| --- | --- | --- |
| Go Server | `layers` and `knownViolations` in [internal/archtest/layers_test.go](../internal/archtest/layers_test.go) | `go test ./internal/archtest` (part of `make test-go`) |
| Python Runtime | `[tool.importlinter]` contracts in [runtime/pyproject.toml](../runtime/pyproject.toml) | `cd runtime && uv run lint-imports` (part of `make lint`) |

The Runtime also keeps its toolsets independent: a toolset may share code
with another only through `toolsets.common`. Runtime imports under
`if TYPE_CHECKING:` are ignored. A failure is one of three kinds:

- **Upward import.** Invert the dependency (the lower package declares an
  interface or callback that the higher layer, or the composition root,
  supplies), or move the shared type down into the lower layer. Only if the
  dependency is genuinely intended, move a package to another layer or add a
  known-violation entry with its reason and the intended fix.
- **Unclassified package, module or toolset.** Add it to exactly one layer.
- **Stale known violation** (`the import no longer exists` in Go,
  `No matches for ignored import` in Python). The violation was fixed: delete
  the entry, so the list only shrinks.

Continue with [deployment](deployment.md), the [user guides](guides/README.md)
or [testing](testing/README.md).
