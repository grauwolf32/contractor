# Contractor CLI

`cmd/contractor` is the command-line entry point for the public API and local
Server administration. It uses the same versioned `/v1` contract as the UI.

Build it from the repository root:

```shell
go build -o ./bin/contractor ./cmd/contractor
```

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

Global flags must precede the command. Resource-specific flags may appear
before or after positional arguments. The output modes are `table` (default),
`json`, and `name`:

```shell
contractor --output json run get run_123
contractor --output name workflow list
```

Cleartext HTTP is accepted by default only for IP-literal loopback origins.
Use `--allow-http` explicitly for another development origin. Redirects and a
Server with an incompatible `X-Contractor-API-Version` are rejected.

## Workflows, Projects, Artifacts, and Runs

The common UI flows have direct commands:

```shell
contractor workflow list
contractor workflow get security-review@3

PROJECT_ID="$(contractor --output name project create example --kind project)"
contractor project get "$PROJECT_ID"

SOURCE_REF="$(contractor --output name source push ./my-repository \
  --project "$PROJECT_ID" --name source)"

RUN_ID="$(contractor --output name run create security-review@3 \
  --project "$PROJECT_ID" \
  --artifact "source=$SOURCE_REF" \
  --param 'objective=Review authentication boundaries' \
  --label 'team=platform')"
contractor run watch "$RUN_ID" --wait-timeout 30m
contractor run output "$RUN_ID" report --to ./report.md
```

`source push` creates a deterministic `application/zip` Artifact. In a Git
working tree it includes tracked files, working-tree changes, and untracked
non-ignored files. `.gitignore` and an optional `.contractorignore` are
honored. `--include-ignored` disables the Git ignore filter;
`.contractorignore` still applies. Symbolic and special files are rejected,
and the final ZIP is bounded by the Server's 64 MiB Artifact limit. If the
binding exists, the command reads its current revision and performs a CAS
update.

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
contractor server run --config-root ./configs/e2e
contractor server auth hash-password
```

Server flags and `CONTRACTOR_DATABASE_URL` behave the same as with
`cmd/contractor-server`.

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
