# Contractor Runtime Agent

The Runtime Agent is the Python, single-slot execution process for Contractor
v2. It registers a fresh process identity with the Go Control Plane, maintains
confirmed sequenced heartbeats, and exposes one private mTLS listener. Worker,
Google ADK, A2A 1.0 JSON-RPC, and artifact tools run together in that process;
each Runtime Agent has exactly one allocation slot.

```shell
uv sync
CONTRACTOR_CONTROL_PLANE_URL=https://localhost:8443 \
CONTRACTOR_ADVERTISED_CONTROL_URL=https://localhost:9443 \
CONTRACTOR_ADVERTISED_A2A_URL=https://localhost:9443 \
CONTRACTOR_CA_FILE=../.local/pki/ca.crt \
CONTRACTOR_CERTIFICATE_FILE=../.local/pki/agents/agent-local.crt \
CONTRACTOR_PRIVATE_KEY_FILE=../.local/pki/agents/agent-local.key \
uv run contractor-runtime --listen 127.0.0.1:9443
```

Project filesystem Toolsets are disabled unless the process has an explicit
workspace provider. For disposable local storage, add
`--workspace-storage local --workspace-work-root /var/lib/contractor/workspaces`;
for an allocation-isolated fsspec tree, add `--workspace-storage memory`.
`CONTRACTOR_WORKSPACE_STORAGE`, `CONTRACTOR_WORKSPACE_WORK_ROOT` and the four
`CONTRACTOR_WORKSPACE_MAX_*` variables provide the equivalent immutable startup
configuration. Physical roots are never registered with Control Plane.

Model-selected HTTP requests and scanner targets pass one
[target policy](../docs/spec/11-http-and-caido-tools.md#target-policy). Runtime
service endpoints and cloud metadata addresses are always refused. Private
networks (RFC 1918, `100.64.0.0/10`, IPv6 unique local) and public addresses
are allowed. Loopback and link-local addresses are refused unless they are the
allocation's project HTTP target or fall inside an additional allowed network.
For same-host targets or local evaluations, allow them explicitly at startup,
for example `--allowed-target-network 127.0.0.0/8` (repeatable) or
`CONTRACTOR_ALLOWED_TARGET_NETWORKS=127.0.0.0/8,::1/128`. Values are strict CIDR
networks, at most 64; the setting is immutable for the process. Requests routed
through a `tool-http` forward proxy, such as a Caido instance on the same host,
use the same policy; loopback then refers to the proxy's host.

For `direct` on a local provider, the allocation's `run_workdir` is authoritative
on disk. Completed external writes, creates, renames and deletes are visible to
the next tool call without refresh, including same-size changes with restored
timestamps. `read_file` acquires only the requested text; complete snapshots and
derived analysis acquire the current bounded managed-text projection. Existing
symlinks, hard links, special files, files over the per-file limit, unreadable
entries and non-NFC names are listed as opaque binary-like leaves: never opened
or followed, and any operation touching them fails with
`workspace_type_conflict`. An oversized external tree can fail a complete
acquisition without preventing an otherwise bounded read.

Filesystem calls share an operation-ownership guard and run blocking I/O off the
event loop. Cancellation fences uncertain work but does not release its ownership:
allocation release must join it before removing the content tree. There are no
transactions against arbitrary concurrent host writers and no stale-memory
rollback after an uncertain disk mutation. Workspace API limits are not disk
quotas on external processes. Memory/direct and both overlay variants keep their
managed-view semantics; direct never automatically exports state or a diff.

From the repository root, `make test-local-direct-workspace` runs the focused
filesystem release checks, including an independent writer process and real
allocation cleanup retries. `make test-project-workspaces-e2e` additionally
requires `CONTRACTOR_TEST_DATABASE_URL` for isolated PostgreSQL process tests.
V30-004 passed these gates and `make verify` on 2026-09-06; evidence is recorded in
[`tasks/v30-004-local-direct-release-gate.yml`](../tasks/v30-004-local-direct-release-gate.yml).
Local/direct storage alone grants no command-execution authority. The opt-in
Podman backend and `code-execution@1` now require complete positive startup
probes; see [Podman policy](PODMAN.md).

Sandbox implementation lives in `src/contractor_runtime/sandbox/`:
`contracts.py` defines the execution interface, `lifecycle.py` defines allocation
lifecycle hooks, and `podman/` contains the backend, startup probes, settings,
ownership, and supervisor processes. The `owner` and `guardian` subprocesses run
as `contractor_runtime.sandbox.podman.owner` and
`contractor_runtime.sandbox.podman.guardian` modules.

Built-in toolsets have individual packages under `src/contractor_runtime/toolsets/`.
Factories and callable tools live in `tools.py` or explicitly versioned modules,
with supporting modules alongside them: `memory/codec.py`, `openapi/models.py`,
`security_findings/reader.py` and `collection.py`, and the code-analysis language,
identifier, and Trailmark modules in `code_analysis/`. Shared artifact visibility
rules live in `common/artifact_visibility.py`. Package initializers remain free
of eager imports so the codec and isolated child processes load independently.

Audit result handling lives in `src/contractor_runtime/toolsets/audit_results/`.
`v2.py` implements the only registered Audit toolset, `audit-results@2`.
`arguments.py` defines its model-facing argument schemas; `contracts.py`,
`packages.py`, `collector.py`, `encoding.py`, and `publication.py` handle validation,
collection and canonical result publication. `completion.py` implements the trusted completion
lifecycle invoked by the Runtime after model output, including completeness
checks and publication of the final artifact. It implements the shared
`worker/completion.py` boundary; the ADK runner binds a trusted completion
implementation and its required tools without depending on Audit tool names.

The rest of the Runtime source follows these boundaries:

| Package | Responsibility |
| --- | --- |
| `llm/` | Gateway client, OpenAI request/response adaptation, model construction, and token usage. |
| `telemetry/` | Allocation and invocation counters, process resource sampling, and execution content policy. |
| `worker/` | ADK orchestration and instrumentation, worker construction, budgets, sessions, state, summarization, and finalization. |
| `contracts/` | Strict wire models grouped into registration, allocation, workspace, worker, artifacts, settings, and reports, plus encoding/decoding. Existing imports are exported by `contracts/__init__.py`. |
| `allocation/` | Slot lifecycle service, admission validation, resource context, final reports, and cleanup primitives. The service owns preparation, cancellation, and release ordering. |
| `toolsets/common/` | Shared tool metrics interface, artifact client helpers, and artifact visibility rules. |

Completion decisions and bounded failures belong to `worker/completion.py`.
Concrete completion implementations own request validation, result publication,
and error classification. A tool binding must cover all required tools under
one prepared owner and match the allocation's trusted completion contract.

## Local Podman workflow

The sample `podman-python-check@2` uses `podman_python_fixer@2`: read and fix a
small Python source, execute its offline checker, and explicitly publish
`builder/check_report` as the ordinary JSON workflow output `report`. It uses
local **direct** storage, not an overlay, and does not automatically export files.

First follow [image and host provisioning](../deploy/podman/README.md), including
the real-host gates. From the repository root, after provisioning the existing
Control Plane and agent mTLS certificates, run:

```sh
uv sync --project runtime --locked
mkdir -p -m 700 .local/podman-runtime/scratch .local/podman-runtime/project
export CONTRACTOR_CONTROL_PLANE_URL=https://localhost:8443
export CONTRACTOR_ADVERTISED_CONTROL_URL=https://localhost:9443
export CONTRACTOR_ADVERTISED_A2A_URL=https://localhost:9443
export CONTRACTOR_CA_FILE="$PWD/.local/pki/ca.crt"
export CONTRACTOR_CERTIFICATE_FILE="$PWD/.local/pki/agents/agent-local.crt"
export CONTRACTOR_PRIVATE_KEY_FILE="$PWD/.local/pki/agents/agent-local.key"
export CONTRACTOR_WORK_ROOT="$PWD/.local/podman-runtime/scratch"
export CONTRACTOR_WORKSPACE_STORAGE=local
export CONTRACTOR_WORKSPACE_WORK_ROOT="$PWD/.local/podman-runtime/project"
export CONTRACTOR_PODMAN_ENABLED=true
export CONTRACTOR_PODMAN_IMAGE="${CONTRACTOR_TEST_PODMAN_IMAGE:?set the preinstalled digest reference}"
export CONTRACTOR_PODMAN_OWNER=local-podman-runtime
export CONTRACTOR_PODMAN_CPUS=2
export CONTRACTOR_PODMAN_MEMORY_BYTES=2147483648
export CONTRACTOR_PODMAN_PIDS=256
export CONTRACTOR_PODMAN_TMPFS_BYTES=268435456
export CONTRACTOR_PODMAN_COMMAND_MAX_SECONDS=300
export CONTRACTOR_PODMAN_PREPARE_MAX_SECONDS=30
export CONTRACTOR_PODMAN_STOP_GRACE_SECONDS=5
export CONTRACTOR_PODMAN_PREVIEW_BYTES=32768
export CONTRACTOR_PODMAN_OUTPUT_MAX_BYTES=1048576
export CONTRACTOR_SHUTDOWN_GRACE_SECONDS=30
runtime/.venv/bin/contractor-runtime --listen 127.0.0.1:9443
```

Run under the nonzero UID that owns the Podman image store and both dedicated
roots. The certificate must identify the agent authorized by your Control Plane;
adjust URLs and certificate paths to your existing local setup. This command
does not launch the Control Plane, provision credentials or enable Podman for
other agents. The corresponding CLI flags are `--podman-enabled true`,
`--podman-image`, `--podman-owner`, and `--podman-<setting-with-hyphens>`.

Before submitting the workflow, check the Runtime Agent in Operations (or
`GET /v1/operations/runtime-agents`): it must be registered and available with
`podman@1`, `code-execution@1` / `exec_command`, and local/direct workspace
capacity. Startup logs report `Runtime Podman effective policy verified` only
after cleanup, with the verified image digest and limits as structured
`podman*` JSON fields. Each capability probe line carries `capabilityRef`,
`capabilityKind`, `probeOutcome` (`available`, `unavailable`, `failed`,
`timeout` or `total_timeout`) and `durationMs`; the JSON formatter emits only
these reviewed extra fields. Enabled settings alone, a listening port, or `systemctl active`
are not positive capabilities. Missing optional prerequisites omit the paired
capabilities; unconfirmed recovery/cleanup prevents registration entirely.

Create the example source ZIP without an extra enclosing directory:

```sh
sample_bundle_dir=$(mktemp -d /tmp/contractor-podman-source.XXXXXX)
(cd configs/fixtures/podman-python-check &&
  python3 -m zipfile -c "$sample_bundle_dir/source.zip" calculator.py check.py)
printf '%s\n' "$sample_bundle_dir/source.zip"
```

Start/reload the Server with the `configs` catalog, select `podman-python-check@2`
in the Run form and upload this ZIP as `source` (`application/zip`). The Server
resolves the exact input Artifact revision before allocation. The ordinary
`worker@1` model policy uses the configured `local-litellm@1` gateway and
`development-worker` credential; those are host-side deployment prerequisites,
not container credentials. Runtime talks to that shared Gateway directly through
its `openai-compatible@1` endpoint using a narrow HTTPX client. Neither the
OpenAI nor LiteLLM Python SDK is installed in the Runtime dependency set.
The allocation-owned client makes at most one initial attempt plus three
transport retries, with the complete series bounded by the normal
request timeout plus 60 seconds. A user-triggered Run uses that real model. For
a reproducible check **without a model service or live credentials**, run from
the repository root:

```sh
make test-podman-workflow
```

The gate requires `CONTRACTOR_TEST_PODMAN_IMAGE`, starts the real Runtime CLI and
mTLS listener, verifies registration after real probes, runs the sample through
ADK and allocation preparation/finalization/release, and checks the exact output
ArtifactRef. Model responses and remote Control Plane/Artifact peers are
deterministic in-process stand-ins; this is not a full Server/PostgreSQL/LLM test.
The ordinary suite tests the same selected tools with a fake command transport
that never launches host code. Real-host skips do not satisfy the gate.

Only an explicit `write_artifact` persists output before the Server's write
fence. A successful command or `report.json` on disk is insufficient. `direct`
edits, the checker and any unuploaded files disappear after confirmed container
removal; export source explicitly if another workflow needs it. `read_file`
returns numbered lines: encode the underlying file content, not its display
wrapper, when publishing an artifact.

First-version limits: no memory/overlay execution, container networking or
package downloads, PTY, detached/background jobs, host engine socket, arbitrary
mounts or skill-script mounts. Agent Skills remain script-free. The image root
is read-only, `/tmp` is bounded tmpfs, and the bind has **no disk quota**; operator
filesystem capacity/quotas are separate from the workspace API's read/edit limits.

The listener requires both a deployment-CA client certificate and the reserved
Control Plane URI SAN before HTTP dispatch. Readiness remains false until the
listener is accepting and registration has succeeded.

## Source archives

`source-analysis@1` opens an exact `application/zip` Run artifact inside the
current allocation's `local-workdir@1`. ZIP members remain POSIX-relative and
the Runtime rejects traversal, links, special/encrypted files, duplicate
normalized names, and bounded-archive violations before replacing an already
opened tree. The initial limits are 10,000 entries, 64 MiB declared
uncompressed total, and 4 MiB per file; the outer Artifact API still limits the
compressed payload to 64 MiB.

The model-visible interface is read-only: `open_source_archive`,
`list_source_files`, `search_source`, and `read_source`. Dependency/VCS/build
trees and known binary formats are omitted. Search scans at most 32 MiB of
validated UTF-8 files and returns bounded source-relative file/line evidence.
Allocation release removes the materialized tree with the rest of the
workspace. Create archives with project contents at the ZIP root when possible;
a containing directory is safe but remains part of every reported path.

## LikeC4 documents

`likec4@1` keeps one single-file DSL document as a CAS-versioned
`text/vnd.likec4` Run artifact. The model can load/copy a seed, perform a
bounded whole write, page reads, append exact text, replace exact fragments,
and validate the current revision. It cannot select a host path, executable,
package runner, CLI flag, or temporary directory, and it never writes into the
materialized source tree.

Install the LikeC4 CLI directly on every Runtime Agent host and make the
`likec4` executable available on the service process `PATH`; for example, an
administrator may manage a pinned global npm installation. Confirm the exact
deployment with `command -v likec4` and `likec4 --version`. The Runtime does not
fall back to `npx`, `pnpx`, or `bunx` and never downloads packages while a
Worker is running.

Validation uses one allocation-local temporary project and the fixed command
shape `likec4 validate --json --no-layout --file <managed-file>
<managed-project>`, with no shell and a 30-second timeout. The temporary project
is removed after every outcome. Includes and multi-file projects, layout,
rendering, export, and the development server are outside the `likec4@1`
contract.

## CLI scanners

`scan@1` exposes `scan_nuclei`, `scan_sqlmap`, `scan_naabu`, `scan_ffuf` and
`scan_katana`.
Select the exact operations in an AgentTemplate:

```yaml
toolsets:
  - ref: scan@1
    tools: [scan_nuclei, scan_sqlmap, scan_naabu, scan_ffuf, scan_katana]
```

Provision the executables on the Runtime service's `PATH` before startup.
Each binary is checked independently with `nuclei -version`, `sqlmap --version`,
`naabu -version`, `ffuf -V` or `katana -version`, under a two-second deadline. Missing, non-executable,
failing or timed-out binaries omit only their own tool from advertised
capabilities. If all five are unavailable, `scan@1` is absent. Installation
alone is insufficient: the version command must exit successfully. Restart
the Runtime after changing installed scanners to refresh its frozen snapshot.

Nuclei uses operator-provisioned templates in `NUCLEI_TEMPLATES_DIR` (default
`~/nuclei-templates`, resolved at Runtime startup). Binary capability does not
prove template availability: a missing directory returns
`nuclei_templates_unavailable`. Calls select HTTP templates using IDs, tags and
severity; automatic updates/downloads, redirects and external OAST callbacks
are disabled. SQLMap uses batch mode and a fresh per-call session for injection
detection. It accepts either an exact `request_ref` for one prepared HTTP
request or the existing URL, parameter selection, POST data and cookie
arguments; both modes accept level and risk. Naabu uses TCP CONNECT scanning
on one hostname/IP and explicit ports.
See upstream CLI references for [nuclei](https://github.com/projectdiscovery/nuclei),
[sqlmap](https://github.com/sqlmapproject/sqlmap/wiki/usage) and
[naabu](https://github.com/projectdiscovery/naabu).

Calls use fixed argument arrays without a shell, a private allocation-local
temporary directory and a minimal child environment. Calls within one toolset
are serialized. The maximum deadline is 3600 seconds; timeout, cancellation,
allocation close and output overflow terminate the process group before scratch
cleanup. A descendant that detached into another session cannot hold a call
open: Runtime closes its pipe ends instead of waiting for their EOF. Combined
process output is capped at 1 MiB, previews at 32 KiB per stream, and JSONL
results at 100 records / 128 KiB. Every result carries process
status, error code, exit code and truncation information. A completed process
does not certify a clean target; partial results and SQLMap diagnostics require
interpretation. Direct tool calls return observations; `tool@1` Workers persist
them as reports. Neither path automatically publishes security findings.

SQLMap request artifacts use UTF-8 JSON with media type `application/json` or
`application/vnd.contractor.http-request+json`. The six required fields are
`schemaVersion: 1`, `method`, `url`, `headers`, `body` and a nonempty explicit
`testParameters` list. The artifact is limited to 256 KiB and its text body to
64 KiB. Runtime validates the supported request subset before launch; the
[request contract](../docs/spec/01-agent-template.md#sqlmap-http-request-artifacts)
defines methods, header/parameter limits and rejected representations.
`request_ref` must contain a revision and cannot be combined with nonempty
`url`, `parameter`, `data` or `cookie`. No additional generic Artifact tool is
needed to read the request through the allocation's private Artifact client.

The request is materialized as private mode-`0600` `request.http` and passed to
sqlmap with `-r`, its explicit method, UTF-8 encoding and `-p` parameter names.
Request mode uses `--skip-waf` to disable automatic WAF probes that add random
query parameters outside that selection.
The request line retains the absolute HTTP(S) URL and explicit port. Input
retrieval shares the scan deadline, and request/output files are removed after
completion or cancellation. Request-mode observations suppress stdout/stderr
with `diagnosticsRedacted: true`, retain the exact `requestArtifact` ref and
expose only recognized `injectionTechniques`. `injectionOutcome` is `reported`
when those labels appear and `unknown` otherwise; it is not a clean-scan flag.

ffuf uses `scan_ffuf(url, wordlist_ref, ...)` with `FUZZ` in the URL path/query
and an exact uploaded `text/vnd.contractor.wordlist` or `text/plain` revision.
The bounded UTF-8 list is validated and privately materialized without trimming
payloads or dropping duplicates/comments/blank entries. The
[wordlist and ffuf contract](../docs/spec/03-artifact-plane.md#scanner-wordlist-artifacts)
defines newline semantics, limits and result fields. Calls use GET, one thread,
`rate=10` by default and `timeout_seconds=300`; `match_status` defaults to `all`.
Optional `filter_status`, `filter_size`, `filter_words` and `filter_lines` accept
bounded numeric ranges. No arbitrary commands, host paths or recursion are exposed.
The rate and `payloadsAttempted` counter describe scheduled wordlist entries;
ffuf may retry a failed HTTP request once outside its payload rate limiter.
The reserved `FFUFHASH` marker is rejected to avoid implicit substitutions.

ffuf observations contain up to 100 structured matched responses / 128 KiB,
the exact `wordlistArtifact` and explicit `resultsTruncated`/`scanComplete` flags.
Raw diagnostics are suppressed. Final progress and request errors are checked
even when the process exits zero; incomplete scans and transport failures are
not successful empty results. `tool@1` publishes the ordinary artifact report.

These fixed scanner processes run on the Runtime host and need network access.
They do not use the offline Podman execution sandbox. The initial implementation
rejects calls with `scan_proxy_unsupported` when a `tool-http` or
`tool-subprocess` proxy route is assigned, because scanners cannot use the
tool HTTP route; it never silently falls back to direct routing. Before launch,
the target host is resolved and every address must pass the target policy for
every selected port; otherwise the call fails with `scan_target_denied`, or
`scan_target_unresolved` when the host does not resolve. The scanner resolves
the name again itself, so an answer that changes after the check (DNS
rebinding) is not pinned; restrict the Runtime's network namespace when that
matters. Scanner arguments and output are excluded from tool metrics. Install
binaries/templates as deployment dependencies; the Runtime does not install
them during allocation.

To add a scanner, implement a `ScanTool` adapter (or `JSONLinesScanTool` for
JSONL output) and register its class in `SCANNERS`. The adapter declares the
exported name, executable, version arguments and typed callable; `prepare`
creates private per-call files and `observation` decodes output. Shared process
launch, deadlines, cleanup and metrics do not branch on scanner names. Update
the Go toolset descriptor and descriptor-parity fixture with the new operation.
The registry is trusted Runtime configuration, never invocation input.

The model-free `tool@1` Worker invokes one selected typed callable from pinned
parameter, ArtifactRef and literal bindings, without a ModelPolicy or gateway.
It publishes a bounded JSON report through the Artifact API and returns its
exact revision in WorkerCompletion. Durable invocation receipts prevent an
automatic rescan after cancellation, response loss or an unknown outcome.
See the [Worker contract](../docs/spec/29-tool-workers.md) and the standalone
[nuclei/naabu/SQLMap Workflow fixtures](../configs/scan/README.md). The
`sqlmap-request@1` fixture binds its required `request` artifact to
`sqlmap-scan@1`, publishes a `report` and uses a 300-second Worker timeout
without model configuration or credentials.

### Katana discovery

Provision [Katana v1.7.0](https://github.com/projectdiscovery/katana/releases/tag/v1.7.0)
on the Runtime service's `PATH` before startup. This version provides the native
maximum-pages control used by the adapter; older versions without that control
are unsupported. The capability probe accepts the 1.7.x family, tested with 1.7.0, and
omits `scan_katana` for other versions until their command contract is checked.
Provisioning remains an operator action: Runtime neither
downloads Katana nor installs a browser. Confirm `katana -version` under the
service environment, then restart Runtime to refresh its advertised capability.
An unavailable Katana executable disables only `scan_katana`, leaving other
scanner operations independently available.

`scan_katana(url, max_depth=2, max_pages=100, rate_limit=10, timeout_seconds=60)`
accepts one HTTP(S) seed. Bounds are depth 1–5, pages 1–1000, rate 1–1000 and
deadline 1–3600 seconds. The adapter applies an anchored exact-origin scope,
disables redirects, uses the ordinary non-headless crawler, and passes the
native page limit. Headers/cookies, arbitrary flags, local files, external
scope overrides and browser execution are not invocation inputs. The usual
`scan_proxy_unsupported` rule and shared deadline/process-group cleanup apply.

The adapter exports only supported same-origin URLs as a sorted, deduplicated
UTF-8 `text/vnd.contractor.target-list` Artifact named `targets` in the Worker's
namespace. The export is capped at 100 targets / 128 KiB and uses create-only
publication. The report carries its exact revision, content digest, seed and
origin, configured limits, per-URL source provenance and incomplete coverage
reasons. Raw Katana request/response bodies are omitted. Empty discovery and
Artifact publication failure are explicit failures, not reusable empty inputs.
`discoveryComplete` is always `false`: successful bounded discovery is a sample,
not evidence that the application has been exhaustively crawled.

The model-free `katana-discovery@1` fixture publishes both `report` and `targets`.
Its durable Worker receipt retains the exact output refs, so successful replay
does not recrawl or republish the TargetList. See the [discovery example and
manual scan handoff](../configs/scan/README.md#bounded-katana-discovery). It never
dispatches another scan automatically. RequestSet export is not implemented.

Combined workflows follow in the
[ScanTools implementation plan](../docs/plans/2026-09-19-scan-tools.md).

## Tool descriptions

The [tool-description contract](../docs/spec/01-agent-template.md#model-visible-tool-descriptions)
owns wording, argument/return documentation and ADK declaration behavior.
Callable implementations in `src/contractor_runtime/toolsets/` expose their
single `description` value as `__doc__`; native Skill tools construct explicit
declarations.
