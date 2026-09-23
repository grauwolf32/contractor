# Model-free scan fixtures

This standalone catalog contains `nuclei-target@1`, `naabu-host@1`,
`sqlmap-request@1`, `ffuf-wordlist@1` and `katana-discovery@1` with `passthrough@1` planning and
one `tool@1` Worker each.
It also includes `request-set-scan@1` and `target-scan-plan@1` using the
deterministic `scan-plan@1` Planner; see [scan plan configuration](SCAN_PLAN.md).
Select `configs/scan` as the Server configuration directory, or copy the
templates, workflows and
`instructions/scan.md` into an existing operator catalog. The empty catalog
directories are required by the configuration loader; no model configuration
or model credentials are needed.

All seven workflows publish the `report` output as `application/json`.
Katana also publishes reusable `targets` as `text/vnd.contractor.target-list`.
Nuclei and naabu accept one required string parameter, `target`.
Nuclei expects an HTTP(S) URL;
naabu expects a hostname or IP. Ports and rate limits are explicit literals
in the templates. Nuclei requires an installed templates directory.

Runtime must advertise `tool@1`, `local-workdir@1` and the selected `scan@1`
operation. Missing binaries prevent placement on that Runtime. See
[scanner provisioning](../../runtime/README.md#cli-scanners).

Targets must pass the Runtime
[target policy](../../docs/spec/11-http-and-caido-tools.md#target-policy):
loopback, link-local and private hosts return `scan_target_denied` unless the
Runtime was started with a matching `--private-target-network`, such as
`127.0.0.0/8` for a target on the same host. Runtime service endpoints and
cloud metadata addresses are never scanned.

Reports contain the scanner observation, exact input artifact refs and an
input digest. An empty or truncated report does not establish that a target
is clean. Repeated delivery within one StageExecution reuses its durable
receipt; an unknown outcome requires an explicit new execution to scan again.

## SQLMap request

`sqlmap-request@1` requires one input artifact, `request`, with media type
`application/json` or `application/vnd.contractor.http-request+json`; it has
no string parameters. `sqlmap-scan@1` binds that artifact to the typed
`request_ref` argument, sets level/risk to 1 and a Worker timeout of 300 seconds,
and publishes the `report` output. Runtime must have a working `sqlmap` binary
on `PATH`. A configured `tool-http` or `tool-subprocess` proxy route returns
`scan_proxy_unsupported`.

Save this example as `request.json`, replacing the URL with the intended
target and supplying the headers/body that the endpoint needs:

```json
{
  "schemaVersion": 1,
  "method": "POST",
  "url": "https://app.example.test/search",
  "headers": [
    {"name": "Content-Type", "value": "application/x-www-form-urlencoded; charset=utf-8"}
  ],
  "body": "query=example&page=1",
  "testParameters": ["query"]
}
```

The artifact contains exactly one prepared request with all six fields
required. `testParameters` explicitly selects the names sqlmap may test.
Headers are an array of `{name, value}` objects; an empty array and empty
body string are allowed. Supported methods are `GET`, `HEAD`, `POST`, `PUT`,
`PATCH`, `DELETE` and `OPTIONS`. Artifact/body limits are 256/64 KiB.
Only the supported UTF-8 text subset is accepted; binary bodies, duplicate
headers, unsupported framing/encoding headers and scanner injection markers
fail before launch. See the
[full schema and limits](../../docs/spec/01-agent-template.md#sqlmap-http-request-artifacts).

With the [CLI connected to the Server](../../docs/guides/cli.md#connect-to-a-server),
upload the request and bind its exact revision to a new Run:

```shell
REQUEST_REF="$(contractor --output name artifact put requests/sqlmap \
  --file request.json --type application/vnd.contractor.http-request+json --create)"
RUN_ID="$(contractor --output name run create sqlmap-request@1 \
  --artifact "request=$REQUEST_REF")"
contractor run watch "$RUN_ID" --wait-timeout 10m
contractor run output "$RUN_ID" report --to report.json
```

`--create` requires a new artifact binding. To update one, use `--if-match`
with its current revision, or choose a new artifact name. The Run pins the
uploaded revision, and the Worker resolves its exact Run input ref through
the Artifact API. Runtime writes a private `request.http`, passes it through
sqlmap `-r` with the chosen method and parameter names, and removes temporary
files on completion, timeout or cancellation.

The JSON report's `inputArtifacts` and `observation.requestArtifact` preserve
the exact request refs. In request mode `observation.stdout` and `stderr` are
empty and `diagnosticsRedacted` is `true`. Only recognized fixed technique
labels appear in `injectionTechniques`; `injectionOutcome` is `reported` when
there is such evidence and `unknown` otherwise. A successful process exit
with no technique labels does not mean the target is free of SQL injection.

## Reusable ffuf wordlists

`ffuf-wordlist@1` requires a `target` URL and one uploaded `wordlist` Artifact.
The input accepts `text/vnd.contractor.wordlist` or `text/plain` and is pinned
at an exact revision when the Run starts. `ffuf-scan@1` binds `url` to the
`target` parameter and `wordlist_ref` to that Artifact; no host file path is
accepted. The target must contain `FUZZ` in its path or query, never its
hostname. Literal settings schedule at most 10 payloads per second, match all
HTTP status codes, and allow 240 seconds for the tool; the Worker deadline is
300 seconds to leave time for cleanup and report publication. FFUF uses one
thread and may retry a failed request once outside its payload rate limiter.

Runtime must advertise `scan_ffuf`, independently probed with `ffuf -V`; other
scanner binaries are not required for this Workflow. Provision the executable
on the Runtime service's `PATH` and restart the Runtime after installation.
The same direct-routing limitation applies: a configured tool proxy route
returns `scan_proxy_unsupported`, without silently bypassing the proxy. Runtime
never installs scanners during a Run.

A small example list is available at [examples/paths.txt](examples/paths.txt).
Upload it once, then bind the returned exact revision to the Workflow:

```shell
WORDLIST_REF="$(contractor --output name artifact put lists/paths \
  --file configs/scan/examples/paths.txt --type text/vnd.contractor.wordlist --create)"
RUN_ID="$(contractor --output name run create ffuf-wordlist@1 \
  --param 'target=https://app.example.test/FUZZ' \
  --artifact "wordlist=$WORDLIST_REF")"
contractor run watch "$RUN_ID" --wait-timeout 10m
contractor run output "$RUN_ID" report --to report.json
```

Replace the example target with the intended target. `--create` requires a new
Artifact binding; update an existing binding with `--if-match` or choose another
name. Uploading a new revision does not change a Run that has already pinned
its input. The same list can be reused by later Runs or Project input bindings.

In the UI, upload the file in Artifacts or Project artifacts and select its
wordlist type. Open `ffuf-wordlist@1`, supply the `target`, and choose the list's
exact revision in the required `wordlist` input, or upload directly in that
input slot. Start the Run and open its `report` output when available. The
wordlist preview and report are ordinary Artifact views; the report includes
`inputArtifacts.wordlist` and `observation.wordlistArtifact` so the actual Run
input revision remains visible.

Runtime accepts at most 1 MiB, 10,000 payloads and 4096 UTF-8 bytes per payload.
LF/CRLF line endings are supported; spaces, `#`, duplicates and empty payloads
are preserved. See the
[wordlist contract](../../docs/spec/03-artifact-plane.md#scanner-wordlist-artifacts)
for byte validation and final-line semantics. Runtime privately materializes
the list and removes temporary files after completion, timeout or cancellation.
For different matching or filtering policies, publish another template version
with literal `match_status`, `filter_status`, `filter_size`, `filter_words` or
`filter_lines` settings; these are not free-form command arguments.

Reports expose bounded matched responses, including payloads, with
`wordlistEntries`, `payloadsAttempted`, `requestErrors`, `scanComplete` and
`resultsTruncated`. Check `status`, `errorCode` and `exitCode` alongside those
fields: transport errors or missing final progress fail the scan even when ffuf
exits zero. An empty match list does not establish that a target is clean, and
truncated matches are not a complete result set. Raw stdout/stderr are redacted.

Missing binaries prevent Worker placement and are visible in Runtime capability
and Operations diagnostics. Invalid wordlists/targets fail before scanner
launch. Cancellation interrupts the Run and cleans up the process and scratch
files; output limits and report-publication errors remain technical failures.
A failed or interrupted Run may have no published `report` output: inspect its
Stage/Operations diagnostics and any retained scanner report Artifact instead
of treating a missing report as an empty successful scan.

## Bounded Katana discovery

`katana-discovery@1` takes one required HTTP(S) `target` parameter and uses
`katana-discovery@1`'s `scan_katana` Worker. Install the pinned Katana executable
on the Runtime service's `PATH` and restart the Runtime; its independent
`katana -version` probe controls only `scan_katana` availability. See the
[Katana provisioning contract](../../runtime/README.md#katana-discovery).

The fixture fixes depth at 2, the page budget at 100, the rate at 10 per second
and the scanner deadline at 60 seconds; the Worker deadline is 90 seconds to
allow report publication. Discovery stays within the seed's exact origin
(scheme, hostname and effective port), with redirects and headless browsing
disabled. It does not launch follow-up scans or claim complete coverage.

```shell
DISCOVERY_RUN="$(contractor --output name run create katana-discovery@1 \
  --param 'target=https://app.example.test/')"
contractor run watch "$DISCOVERY_RUN" --wait-timeout 2m
contractor run output "$DISCOVERY_RUN" report --to discovery-report.json
contractor run output "$DISCOVERY_RUN" targets --to discovered-targets.txt
```

The UTF-8 TargetList contains sorted, deduplicated same-origin HTTP(S) URLs,
one per line, limited to 100 targets / 128 KiB. The report records the seed,
origin, limits, per-URL source provenance and the exact `targetsArtifact`
revision and content digest. `discoveryComplete` remains `false`: a bounded
Katana crawl cannot prove that an application has no more reachable content.
Check the report's coverage and incomplete reasons before reusing the list.
An empty discovery does not publish a runnable empty TargetList or succeed as
a scan input. Failed or cancelled discovery can leave retained diagnostic
Artifacts without successful Workflow outputs.

To scan these URLs later, explicitly upload the exported list and select its
exact revision in a new `target-scan-plan@1` Run:

```shell
TARGETS_REF="$(contractor --output name artifact put discovery/targets \
  --file discovered-targets.txt --type text/vnd.contractor.target-list --create)"
SCAN_RUN="$(contractor --output name run create target-scan-plan@1 \
  --artifact "targets=$TARGETS_REF")"
contractor run watch "$SCAN_RUN" --wait-timeout 35m
contractor run output "$SCAN_RUN" report --to scan-report.json
```

Use a new Artifact name or `--if-match` when updating an existing list. This
second Run applies its own explicit [scan-plan budgets](SCAN_PLAN.md); creating
the discovery Run alone never requests nuclei, naabu, SQLMap or ffuf work.
RequestSet generation, authenticated crawling, arbitrary headers/cookies,
JavaScript execution and browser installation are outside this fixture.

The dedicated real-process gate requires Katana 1.7.x on `PATH`, the Runtime
virtual environment and a disposable PostgreSQL URL. It starts isolated Server
and Runtime processes, checks missing capability behavior, publishes both
outputs from a loopback HTML crawl and feeds the actual TargetList bytes into
the pure scan-plan builder. Cross-origin links and redirects must issue no
requests; no subsequent scanner is executed. To retain report, targets and plan
evidence, set `CONTRACTOR_KATANA_EVIDENCE_DIR`:

```shell
CONTRACTOR_KATANA_EVIDENCE_DIR="$PWD/.local/evidence/katana" \
go test -tags=e2e -count=1 -timeout=6m ./tests/e2e \
  -run '^TestKatanaDiscoveryAcrossProductionProcesses$'
```

## Release verification

Apply Server migrations before using these Workflows. Migration `000061` permits
the model-free allocation provenance emitted by `tool@1`; without it the older
database constraint rejects placement even when Runtime advertises the scanner.

The real-process gate requires a disposable PostgreSQL database in
`CONTRACTOR_TEST_DATABASE_URL`, installed Runtime dependencies (`cd runtime &&
uv sync --locked`), and nuclei, naabu, SQLMap and ffuf executables on `PATH`. It starts the
production Server and Runtime with local TLS identities, an isolated database,
a private nuclei template and loopback HTTP/TCP targets. It does not require a
model service or scan an external target.

```shell
go test ./internal/config ./tests/e2e
go test -tags=e2e -count=1 -timeout=6m ./tests/e2e \
  -run '^TestScanToolsAcrossProductionProcesses$'
```

For the browser gate, install UI dependencies and Chromium with `make ui-install
ui-browser-install`, then use the repository's pinned Node version:

```shell
make ui-typecheck ui-test ui-build
CONTRACTOR_SCAN_BROWSER=1 \
CONTRACTOR_SCAN_UI_PREBUILT=1 \
CONTRACTOR_SCAN_EVIDENCE_DIR="$PWD/.local/evidence/scan-release" \
go test -tags=e2e -count=1 -timeout=6m ./tests/e2e \
  -run '^TestScanToolsAcrossProductionProcesses$'
```

The browser logs in through the real session API, creates a Project, uploads and
previews a wordlist, confirms its exact revision, starts ffuf and opens the report.
It saves four screenshots and `browser-evidence.json` with the selected/input/output
refs and observed execution outcome. Set the evidence directory explicitly to keep
these files after the temporary stack is removed. Backend scenarios also verify
missing capabilities, revision reuse after an upload update, invalid input,
cancellation and report-publication failure.
