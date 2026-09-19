# Model-free scan fixtures

This standalone catalog contains `nuclei-target@1` and `naabu-host@1` with
`passthrough@1` planning and one `tool@1` Worker each. Select `configs/scan` as
the Server configuration directory, or copy the two templates, workflows and
`instructions/scan.md` into an existing operator catalog. The empty catalog
directories are required by the configuration loader; no model configuration
or model credentials are needed.

Both workflows accept one required string parameter, `target`, and publish
the `report` output as `application/json`. Nuclei expects an HTTP(S) URL;
naabu expects a hostname or IP. Ports and rate limits are explicit literals
in the templates. Nuclei requires an installed templates directory.

Runtime must advertise `tool@1`, `local-workdir@1` and the selected `scan@1`
operation. Missing binaries prevent placement on that Runtime. See
[scanner provisioning](../../runtime/README.md#cli-scanners).

Reports contain the scanner observation, exact input artifact refs and an
input digest. An empty or truncated report does not establish that a target
is clean. Repeated delivery within one StageExecution reuses its durable
receipt; an unknown outcome requires an explicit new execution to scan again.

These are focused fixtures. SQLMap request artifacts, ffuf wordlists and the
combined user journey are subsequent V55 tasks.
