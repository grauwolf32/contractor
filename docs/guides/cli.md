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
`source push`, `artifact put`/`get`, `audit findings export`/`download`, `audit materials get`, `audit tasks result` and `run output` it instead bounds each
period without progress, from connecting through the last body byte, so a large
Artifact may take longer on a slow link while a stalled transfer still fails.

Cleartext HTTP is accepted by default only for IP-literal loopback origins.
Use `--allow-http` explicitly for another development origin. Redirects and a
Server with an incompatible `X-Contractor-API-Version` are rejected.

## Audits, hypotheses and retained results

Audit commands use the same owner-authorized public API. Mutations require an
explicit `--idempotency-key`; edits and lifecycle actions also require the
revision read from the resource. A retry keeps the original key, request and
revision. Request files are strict JSON: unknown fields, duplicate keys and
metadata larger than 1 MiB are rejected before submission.

```shell
contractor audit profiles
contractor audit list --project "$PROJECT_ID" --limit 50
contractor audit create "$PROJECT_ID" --request ./audit.json --idempotency-key create-audit
contractor audit get "$AUDIT_ID"
contractor audit start "$AUDIT_ID" --revision 1 --idempotency-key start-audit
contractor audit pause "$AUDIT_ID" --revision 2 --idempotency-key pause-audit
contractor audit resume "$AUDIT_ID" --revision 3 --idempotency-key resume-audit
contractor audit workspace "$AUDIT_ID"
```

`audit.json` has the `CreateAuditRequest` shape: a selected `profile` with name
and version, exact Project Artifact `inputs`, `scope`, `runtimeLabels` and
`labels`. Start and resume accept `--deadline-seconds`; omission preserves the
owning operation's default, and zero removes the allowance. Use the actual
revision returned by the Server for each subsequent mutation.

```shell
contractor --output json audit tasks list "$AUDIT_ID" --limit 50
contractor audit tasks get "$AUDIT_ID" "$TASK_ID"
contractor audit tasks attempts "$AUDIT_ID" "$TASK_ID"
contractor audit tasks result "$AUDIT_ID" "$TASK_ID" --revision "$RESULT_REVISION" --to result.json
contractor audit hypotheses list "$AUDIT_ID"
contractor audit hypotheses get "$AUDIT_ID" "$HYPOTHESIS_ID" --revision 1
contractor audit tasks list "$AUDIT_ID" --hypothesis "$HYPOTHESIS_ID" --state settled
```

Lists return one page. Follow `page.nextCursor` with `--cursor` while `hasMore`
is true, including an empty filtered page. Task and hypothesis pages expose
accepted work and immutable assertions. A saved proposal appears as material;
it does not itself mean hypotheses were registered. Hypothesis grounds and
selected exact sources remain separate from automatically assigned origin.
Task results preserve supported, refuted or inconclusive conclusions and their
tested conditions. Run completion and workspace coordination counts do not
imply a supported assertion or a problem finding.

`tasks create AUDIT --request FILE` accepts `AuditTaskCreationRequest`;
`hypotheses create AUDIT` and `hypotheses edit AUDIT HYPOTHESIS` accept
`AuditHypothesisContent`. All three require `--revision` of the **Audit** and
`--idempotency-key`. The Server assigns hypothesis identity, revision and origin.

```shell
contractor audit materials list "$AUDIT_ID"
contractor audit materials get "$AUDIT_ID" "$LOGICAL_KEY" --revision "$MATERIAL_REVISION" --to material.bin
contractor audit materials publish "$PROJECT_ID" --request publication.json --idempotency-key publish-material
contractor audit findings list "$PROJECT_ID"
contractor audit findings get "$PROJECT_ID" "$FINDING_ID"
contractor audit findings sources "$PROJECT_ID" --finding "$FINDING_ID" --limit 64
contractor audit findings source "$PROJECT_ID" "$RECEIPT_ID"
contractor audit findings history "$PROJECT_ID" "$RECEIPT_ID" --limit 64
contractor audit events "$AUDIT_ID"
contractor audit report "$AUDIT_ID"
```

`publication.json` contains `auditId`, an exact `source` Artifact reference and
a revision-free `target` reference. Omit `--revision` for create-only publication;
provide the exact current **destination Artifact** revision to replace it by CAS.
The Project receipt replays the same publication without repeating execution.
Finding edits use `findings edit PROJECT FINDING --request FILE`, the exact
Finding `--revision` and an idempotency key. Project Finding sources remain
readable independently of the source Audit lifecycle.

`audit cancel AUDIT` requires its revision and key. `audit delete AUDIT` also
requires `--yes`. Exact material/result reads use Artifact transfer timeouts;
`--force` explicitly permits replacing a local output file.

## Workflows, Projects, Artifacts, and Runs

The common UI flows have direct commands:

```shell
contractor workflow list
contractor workflow get openapi-from-workspace@1

PROJECT_ID="$(contractor --output name project create example --kind project)"
contractor project get "$PROJECT_ID"

SOURCE_REF="$(contractor --output name source push ./my-repository \
  --project "$PROJECT_ID" --name source)"

RUN_ID="$(contractor --output name run create openapi-from-workspace@1 \
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
bounded by the Server's 64 MiB Artifact limit. A file that changes size while
it is packaged, such as an active log, fails the push; exclude it with
`.contractorignore` or retry once it is stable. Source paths must be relative
and cannot contain `:`, `\\`, or empty, `.` or `..` components. Names are
stored NFC-normalized, so names that collide after normalization, including a
file and a directory, are rejected. Exclude a non-portable path with
`.contractorignore`. The bundle must contain at least one file that the
runtime source tools can read: strict UTF-8 content, no binary extension such
as `.png` or `.zip`, and no parent directory they ignore, such as
`node_modules`, `vendor`, `build` or `dist`. If the binding exists, the command
reads its current revision and performs a CAS update.

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
`--help` after `server`, a command group such as `server config`, or a
subcommand prints its usage and exits successfully.

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

## Export Audit findings

Write an explicit selection of current finding revisions to `selection.json`:

```json
[{"findingId":"finding-1","revision":3}]
```

Create a frozen Audit-local SARIF export and download it:

```shell
contractor audit findings export audit-1 --selection selection.json \
  --client-key report-1 --to report.sarif
```

By default the file contains one selected representative per known duplicate
group. Add `--include-duplicates` to include every selected finding. The
snapshot always preserves all selected identities and independent reviews.
The same client key replays the original bytes after later triage or source
Run deletion; changing the selection or duplicate option requires a new key.
Publication output includes the export ID, counts and exact descriptors.
`--to -` writes only file bytes to stdout; existing files require `--force`.

Download an existing export or inspect its frozen inputs and diagnostics:

```shell
contractor audit findings download audit-1 export-ID --to report.sarif
contractor audit findings download audit-1 export-ID --part snapshot --to snapshot.json
contractor audit findings download audit-1 export-ID --part diagnostics --to diagnostics.json
```

Use the complete `export-<64 lowercase hex digits>` ID returned by publication
in place of `export-ID`. These downloads check Audit ownership and never
refresh live findings or regenerate content. Exports follow Audit deletion.
