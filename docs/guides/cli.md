# Contractor CLI

`cmd/contractor` is the command-line entry point for the public API and local
Server administration. It uses the same versioned `/v1` contract as the UI.

The examples assume `contractor` is on `PATH`. For a source build, follow
[development setup](../development.md#build-the-commands).

## Connect to a Server

The CLI authenticates with a bearer token. Put the token in a file readable by
the current user and save a named context:

```shell
contractor context add local \
  --server http://127.0.0.1:8080 \
  --token-file ~/.config/contractor/local.token \
  --use
contractor check
```

Contexts are stored in `~/.config/contractor/config.json` with mode `0600`.
The token itself is not copied into that file. Set `CONTRACTOR_CLI_CONFIG` to
move the context file. A command can bypass contexts with global flags:

```shell
CONTRACTOR_API_TOKEN="$TOKEN" \
  contractor --server https://contractor.example --ca-file ./ca.crt workflow list
```

The token is taken from an explicit `--token-file` first, then from the
selected context's token file, and only then from `CONTRACTOR_API_TOKEN`.

Global flags must precede the command. Resource-specific flags may appear
before or after positional arguments. Use `--` before positional arguments
starting with a hyphen, for example `source push -- -source`.
The output modes are `table` (default),
`json`, and `name`:

```shell
contractor --output json run get run_123
contractor --output name workflow list
```

`--timeout` (or `CONTRACTOR_TIMEOUT`, default `30s`) bounds each request. For
`source push`, `artifact put`/`get` and `run output` it instead bounds each
period without progress, from connecting through the last body byte, so a large
Artifact may take longer on a slow link while a stalled transfer still fails.

Cleartext HTTP is accepted by default only for IP-literal loopback origins.
Use `--allow-http` explicitly for another development origin. Redirects and a
Server with an incompatible `X-Contractor-API-Version` are rejected.

## Workflows, Projects, Artifacts, and Runs

The common UI flows have direct commands:

```shell
contractor workflow list
contractor workflow get openapi-from-workspace@7

PROJECT_ID="$(contractor --output name project create example --kind project)"
contractor project get "$PROJECT_ID"

SOURCE_REF="$(contractor --output name source push ./my-repository \
  --project "$PROJECT_ID" --name source)"

RUN_ID="$(contractor --output name run create openapi-from-workspace@7 \
  --project "$PROJECT_ID" \
  --artifact "source=$SOURCE_REF" \
  --param 'objective=Document the implemented public HTTP API' \
  --label 'team=platform')"
contractor run watch "$RUN_ID" --wait-timeout 30m
contractor run output "$RUN_ID" openapi --to ./openapi.yaml
```

This example requires the [local workspace and validator capabilities](project-workflows.md#runtime-workspace-requirements).
Use the [configuration catalog](../../configs/README.md) to choose another Workflow.

`source push` creates a deterministic `application/zip` Artifact. In a Git
working tree it includes tracked files, working-tree changes, and untracked
non-ignored files. `.gitignore` and an optional `.contractorignore` are
honored. `--include-ignored` disables the Git ignore filter;
`.contractorignore` still applies. The root `.contractorignore` uses
`.gitignore` syntax but is evaluated on its own: it also excludes tracked or
force-added files, and no `.gitignore` rule or negation can re-include a path
it excludes. Submodules and nested Git repositories are skipped, and the
command reports how many. Symbolic links and special files are rejected, never
followed; the error names the first one, which `.contractorignore` can
exclude. The bundle matches the runtime source limits: at most 10,000 files,
4 MiB per file, 64 MiB expanded, and 512-byte paths, and the final ZIP is
bounded by the Server's 64 MiB Artifact limit. If the binding exists, the
command reads its current revision and performs a CAS update.

Artifact commands work with UserScope by default. Select ProjectScope or
RunScope with `--project` or `--run`:

```shell
contractor artifact list --project "$PROJECT_ID"
contractor artifact metadata projects/source --project "$PROJECT_ID"
contractor artifact versions projects/source --project "$PROJECT_ID"
contractor artifact get projects/source@rev_123 --project "$PROJECT_ID" --to source.zip

contractor artifact put documents/brief --file brief.md --type text/markdown --create
contractor artifact put documents/brief --file brief.md --type text/markdown --if-match rev_123
```

Artifact downloads and `run output` refuse to overwrite an existing destination
unless `--force` is supplied. Forced replacement is atomic and preserves the
existing regular file's permission bits; a new file created with `--force` is
private (`0600`). Symlink and other non-regular destinations are rejected.

Run creation flags cover normal Workflow inputs. For execution-config patches
or another complete request, pass the public API JSON body directly:

```shell
contractor run create --request ./create-run.json --idempotency-key retry-42
```

Queue and operational views are available through `queue list`, `queue
status`, `queue pause`, `queue resume`, `ops snapshot`, `ops agents`, `ops
principals`, and `ops allocations`.

## Run a local Server

The unified binary delegates Server administration to the existing Server
implementation:

```shell
contractor server migrate
contractor server config validate --root ./configs
contractor server run --config .local/server.yaml
contractor server auth hash-password
```

Server flags and `CONTRACTOR_DATABASE_URL` behave the same as with
`cmd/contractor-server`. `migrate` also accepts `--statement-timeout` and
`--lock-timeout`; see [migration timeouts](../deployment.md#start-and-verify).

Create the configuration, login and certificates using the
[deployment guide](../deployment.md) before starting Server.

## Generate Runtime certificates

The CLI exposes the repository's local PKI generator. Initialize one deployment
CA, issue the Control Plane identity, and issue a unique certificate for every
Runtime process:

```shell
contractor pki init-ca
contractor pki issue-control-plane
contractor pki issue-runtime --name runtime-local
contractor pki issue-runtime --name runtime-analysis \
  --dns runtime-analysis.internal --ip 10.20.0.12
```

The default root is `.local/pki`. Runtime files are written as
`.local/pki/agents/<name>.{crt,key}`, private keys use mode `0600`, and existing
material is preserved unless `--force` is supplied. These commands are for a
deployment whose CA key is available locally; connecting to a remote Server
does not grant access to its CA.
