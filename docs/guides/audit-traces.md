# Audit trace export and tool analytics

Use Python 3.10+ from the repository root. No packages, Contractor token or
database access are needed. The selected Audit must have retained debug traces
in Phoenix (normally enabled by its pinned `debug` Runtime label).

```sh
python3 scripts/audit_trace_report.py audit_example
```

The default source is `http://127.0.0.1:6006`, Phoenix project `default`.
The command prints the report location under `reports/audit-traces/`. To choose
another source or destination:

```sh
python3 scripts/audit_trace_report.py audit_example \
  --phoenix-url http://127.0.0.1:6006 \
  --project default \
  --output reports/my-audit
```

The output directory must not exist. Authenticated Phoenix instances can use a
bearer key in `PHOENIX_API_KEY`, or another environment variable named by
`--api-key-env`. The key is never printed or written to the report. Redirects
are refused; use the collector's final base URL rather than `/v1/traces`.

The script first finds Audit-labelled spans, then downloads their **whole
traces**, following all cursors and deduplicating trace/span identities.
Worker model/tool spans themselves do not carry the Audit label. The label
provides diagnostic correlation, not Contractor authorization. This reads
Phoenix's existing REST API; it does not change an Audit, Run or RuntimeConfig.

| File | Contents |
| --- | --- |
| `spans.jsonl` | Complete selected Phoenix span records, including retained model and tool content, one JSON object per line |
| `report.md` | Readable summary, calls/errors/cancellations, tool duration sum/mean/p50/p95/max, error groups, slow calls and known model tokens |
| `summary.json` | All deterministic aggregates and per-trace lifecycle counter comparisons |
| `tools.csv` | All per-tool aggregates |
| `tool-calls.csv` | One row per observed Worker tool call, with trace/span/task references and outcome |
| `errors.csv` | One row per failed/cancelled tool call, error code, exception type and bounded captured error message |
| `traces.csv` | Per-trace observed versus lifecycle-reported call counts and observed time range |
| `manifest.json` | Source type, Audit/project, export timestamps, input bytes, duplicate count and export state |

To analyse the archive again without a running Phoenix:

```sh
python3 scripts/audit_trace_report.py audit_example \
  --input reports/my-audit/spans.jsonl \
  --output reports/my-audit-replayed
```

Replay accepts the same native Phoenix span JSONL format and checks that every
trace is labelled for the selected Audit. It recomputes the analytics; it cannot
establish whether its input archive came from a completed export. Inspect the
source manifest when using a partial archive.

Counts describe **observed retained spans**, not authoritative Audit totals.
The script counts `contractor.worker.tool` spans, including calls whose Phoenix
kind is `UNKNOWN`; it does not count declarations in model messages or Planner
dispatches. Errors include failed/rejected/unavailable outcomes, cancellations
are separate, and error rates exclude cancellations. Missing tool output
does not change a recorded outcome. Error codes come from retained structured
results or span attributes; exception types remain separate. An absent code
means unavailable diagnostic detail. Error groups also include the bounded
message, so tools returning only a string error still have useful frequency
groups. Percentiles use nearest rank. Duration
sums include parallel work and do not represent elapsed Audit time.

Input/output/cache token aggregates sum only reported per-model-span values,
with measured-call counts. Missing measurements remain null. Cache-read input
is a subset of input tokens and is never added to the input total.

Worker lifecycle `counts.tool_calls` / `counts.model_calls` are compared with
observed spans within each trace. A mismatch indicates a discrepancy; matching
counts cannot rule out an entirely missing trace. Expiry, collector/exporter
loss and unfinished flushes can all produce partial data. For active Audits,
rerun into a fresh directory after Workers flush; an active Worker without a
retained labelled lifecycle span cannot yet be discovered. Enabling debug now
does not create traces for an existing Run.

Default bounds are 10 spans per page, 100,000 unique spans, 10,000 traces,
1 GiB total input bytes, 16 MiB per response/JSONL line, a 30-second socket
timeout and a 900-second traversal budget checked between requests and read
chunks. Captured content is streamed to disk; only compact measurements are
retained for analysis. Raise `--max-spans`, `--max-traces`, `--max-bytes` or
`--deadline` for larger exports; lower `--page-size` if a response is too large.
`--timeout` controls socket waits. No automatic retry or polling occurs.

Exit status is `0` for a successful export with traces (even if tools failed),
`1` for export/analysis failure, `2` for invalid arguments, `3` when no labelled
traces exist and `130` for interruption. No-trace reports show unavailable call
counts. A failure retains `state: incomplete` in the manifest and any partial
JSONL; it never prints captured content or backend error bodies. A completed
local export has `state: exported`; that does not promise complete upstream
telemetry. Existing reports are never overwritten.

Output directory/file permissions are private (`0700`/`0600`). The archive and
error CSV retain content already present in the configured debug sink. The
exporter does not add secret discovery or redaction beyond the Runtime/Server
capture policy. Aggregate tables exclude prompts, arguments and full results;
error groups include bounded captured error messages. CSV
text is escaped against spreadsheet formula interpretation.

Phoenix's [official span client reference](https://arize-phoenix.readthedocs.io/projects/client/api/spans.html)
describes project span selection and pagination. The local collector exposes
its exact REST contract at `/openapi.json`.
