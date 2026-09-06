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
V30-004 also requires `make verify`; the release status is recorded in
[`tasks/v30-004-local-direct-release-gate.yml`](../tasks/v30-004-local-direct-release-gate.yml).
These changes expose no command-execution Toolset or Podman capability.

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
compressed payload to 16 MiB.

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
