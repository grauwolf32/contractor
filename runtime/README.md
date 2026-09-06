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

For `direct` on a local provider, the allocation's `run_workdir` is authoritative
on disk. Completed external writes, creates, renames and deletes are visible to
the next tool call without refresh, including same-size changes with restored
timestamps. `read_file` acquires only the requested text; complete snapshots and
derived analysis acquire the current bounded managed-text projection. Existing
symlinks, hard links and special files are rejected. An oversized external tree
can fail a complete acquisition without preventing an otherwise bounded read.

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

## Local Podman workflow

The sample `podman-python-check@1` uses `podman_python_fixer@1`: read and fix a
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
after cleanup. Enabled settings alone, a listening port, or `systemctl active`
are not positive capabilities. Missing optional prerequisites omit the paired
capabilities; unconfirmed recovery/cleanup prevents registration entirely.

Create the example source ZIP without an extra enclosing directory:

```sh
sample_bundle_dir=$(mktemp -d /tmp/contractor-podman-source.XXXXXX)
(cd configs/fixtures/podman-python-check &&
  python3 -m zipfile -c "$sample_bundle_dir/source.zip" calculator.py check.py)
printf '%s\n' "$sample_bundle_dir/source.zip"
```

Start/reload the Server with the `configs` catalog, select `podman-python-check@1`
in the Run form and upload this ZIP as `source` (`application/zip`). The Server
resolves the exact input Artifact revision before allocation. The ordinary
`worker@1` model policy uses the configured `local-litellm@1` gateway and
`development-worker` credential; those are host-side deployment prerequisites,
not container credentials. Runtime talks to that shared Gateway directly through
its `openai-compatible@1` endpoint and does not load the LiteLLM Python SDK in
each agent process. The allocation-owned OpenAI client makes at most one initial
attempt plus three SDK retries, with the complete series bounded by the normal
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

## Tool descriptions

Tool descriptions are part of the model-facing API. Callable tools in
`src/contractor_runtime/toolsets/` keep a single description in `description`,
which their constructors expose as `__doc__` for ADK `FunctionTool`. Keep it in
English using a short action summary, relevant usage constraints, `Args:` for
every model-supplied argument, and `Returns:` for the actual response fields.
Omit `Args:` for tools without model-supplied arguments and omit injected
`tool_context`. Document defaults, units, allowed values, path scope, revision
requirements and pagination where relevant. Avoid internal implementation terms
and claims about behavior that the current code does not provide.

In the pinned ADK 2.8.0, the full docstring becomes the function description;
`Args:` text is not extracted into individual parameter descriptions. Keep
argument documentation in the full description. Agent Skill tools use explicit
ADK declarations and follow the same wording conventions. The
[migration inventory](../docs/reviews/2026-09-06-tool-docstrings.md) records which
guidance was adapted from `contractor-old` and which contracts have changed.
