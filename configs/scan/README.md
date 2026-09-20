# Model-free scan fixtures

This standalone catalog contains `nuclei-target@1`, `naabu-host@1` and
`sqlmap-request@1` with `passthrough@1` planning and one `tool@1` Worker each.
Select `configs/scan` as the Server configuration directory, or copy the
templates, workflows and
`instructions/scan.md` into an existing operator catalog. The empty catalog
directories are required by the configuration loader; no model configuration
or model credentials are needed.

All three workflows publish the `report` output as `application/json`.
Nuclei and naabu accept one required string parameter, `target`.
Nuclei expects an HTTP(S) URL;
naabu expects a hostname or IP. Ports and rate limits are explicit literals
in the templates. Nuclei requires an installed templates directory.

Runtime must advertise `tool@1`, `local-workdir@1` and the selected `scan@1`
operation. Missing binaries prevent placement on that Runtime. See
[scanner provisioning](../../runtime/README.md#cli-scanners).

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
on `PATH`. A configured subprocess proxy returns `scan_proxy_unsupported`.

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

These are focused fixtures. ffuf wordlists and the combined user journey are
subsequent V55 tasks.
