# 10 — Run workspaces and filesystem tools

Status: **Working agreement**

This document owns the first workspace contract for source-oriented Workers.
It deliberately separates three things:

- Workflow declares which Run artifacts form a logical workspace and which
  workspace results are expected;
- Control Plane pins those artifacts and sends one immutable construction
  manifest to each allocation;
- Runtime Agent privately chooses a disposable local or in-memory physical
  implementation and exposes only explicitly selected filesystem tools.

The Server never mounts an operator project directory. A Worker never receives
an `AgentTemplate`, backend path, storage option or infrastructure credential as
model-visible input.

## Concepts and ownership

- **workspace source** — an exact RunScope ZIP artifact used to hydrate a
  subtree of `run_workdir`;
- **workspace state** — a cumulative text-only overlay artifact which, together
  with the exact sources, reconstructs the effective tree;
- **workspace mode** — Workflow-selected semantics: `direct` or `overlay`;
- **workspace storage** — Runtime startup choice: `local` or `memory`;
- **workspace session** — one allocation-private hydrated tree and its tool
  handles;
- **checkpoint** — the effective tree from which the current invocation's diff
  and rollback are calculated;
- **auto export** — Runtime-owned persistence of overlay state and human diff
  into declared Stage result slots before a graceful terminal A2A response.

`AllocationWorkspace` in `contractor_runtime.workspace` remains the sandbox
scratch directory used by skills and existing tools. A project workspace is a
separate session. Local storage owns a private `run_workdir` below its
configured provider root; memory storage uses an allocation-isolated fsspec
instance. A future executor receives only the project content directory, not
the provider's ownership markers or the allocation's general scratch.

## Workflow contract

Workspace configuration is adjacent to artifact context in a Stage:

```yaml
stages:
  edit:
    context:
      artifacts:
        backend:
          namespace: inputs
          name: backend
          required: true
        previous_state:
          namespace: analysis
          name: workspace_state
          required: false
      workspace:
        mode: overlay
        sources:
          - artifact: backend
            target: backend
        state:
          artifact: previous_state
        export:
          state: workspace_state
          diff: workspace_diff
    result:
      artifacts:
        workspace_state:
          required: true
          mediaTypes: [application/vnd.contractor.workspace-overlay+json]
        workspace_diff:
          required: true
          mediaTypes: [text/x-diff]
```

Workflow artifact references are logical and contain no revision. When a Stage
is made runnable, Scheduler creates immutable `StageContext`, resolves every
present context binding to an exact RunScope `ArtifactRef`, and stores those
pins with the StageExecution. Each concrete `AllocationSpec` carries its own
copy of the workspace projection using those exact refs.

### Workspace schema

`context.workspace` is optional. When absent, filesystem Toolsets are invalid
for that Stage.

```text
WorkspaceContext
  mode: direct | overlay
  sources: 1..32 WorkspaceSource
  state?: WorkspaceStateInput
  export?: WorkspaceExport

WorkspaceSource
  artifact: context artifact alias
  target: normalized relative directory, or "" for the workspace root

WorkspaceStateInput
  artifact: context artifact alias

WorkspaceExport
  state: Stage result slot name
  diff: Stage result slot name
```

Rules:

1. Every source alias and optional state alias exists in
   `context.artifacts`. Source aliases must be required. State requiredness is
   inherited from its artifact entry, so an optional absent state stays absent
   from AllocationSpec.
2. Source targets are UTF-8 NFC relative POSIX directories. They contain no
   empty, `.`, `..`, backslash, NUL/control, URI, Windows drive or UNC
   component; at most 32 components and 1024 UTF-8 bytes are allowed.
3. Targets are unique and non-overlapping. `backend` and `backend/api` cannot
   both be targets. Archive structure below each target is preserved; v1 has no
   `stripComponents` or include/exclude rules.
4. `state` is allowed for either mode. In `direct` it initializes the private
   tree; no later overlay state is exported automatically.
5. `export` is allowed only for `overlay`, must contain both fields, and each
   value must name a distinct Stage result slot. The state slot accepts exactly
   `application/vnd.contractor.workspace-overlay+json`; the diff slot accepts
   exactly `text/x-diff`.
6. `workspace-changes@1` requires `mode: overlay`. `filesystem@1` and
   `edit-files@1` work in either mode. A workspace without these Toolsets is
   valid; a filesystem Toolset without a workspace is not.
7. Auto-export result slots are reserved. They omit ordinary `from` bindings and
   cannot be provided or replaced by model text or ordinary tool observations.

Multiple sources create one logical root, for example:

```text
run_workdir/
  backend/        <- inputs/backend@r7
  frontend/       <- inputs/frontend@r3
  libs/shared/    <- inputs/shared@r11
```

A Router Stage gives each logical Worker its own physical copy of the same
exact initial source/state refs. Workers do not share live filesystem state.
Changes from Worker A become visible to Worker B only if a later StageExecution
explicitly consumes A's persisted overlay state.

## AllocationSpec projection

The private v2 allocation manifest adds one optional `workspace` value:

```json
{
  "mode": "overlay",
  "sources": [
    {
      "artifact": {
        "namespace": "inputs",
        "name": "backend",
        "revision": "sha256:..."
      },
      "target": "backend"
    }
  ],
  "state": {
    "artifact": {
      "namespace": "analysis",
      "name": "workspace_state",
      "revision": "sha256:..."
    }
  },
  "export": {
    "state": "workspace_state",
    "diff": "workspace_diff"
  }
}
```

It is construction data, not task content. `StageContentRequest` continues to
carry objective, Planner instructions, string parameters and exact context
artifact refs. Workspace refs are duplicated intentionally in the private
construction manifest so Runtime can be fully ready before exposing the A2A
Worker.

Every source in `AllocationSpec.workspace` is exact. Optional state is either
absent or exact. Export destinations are logical Stage result slot names and
therefore have no revision before Runtime writes them. The allocation
namespace determines the physical bindings, for example
`editor/workspace_state@r8` and `editor/workspace_diff@r8`.

## Runtime startup configuration and capabilities

The physical implementation is immutable process configuration:

```yaml
workspace:
  storage: local
  workRoot: /var/lib/contractor/workspaces
  limits:
    maxFiles: 50000
    maxExpandedBytes: 2147483648
    maxManagedTextBytes: 268435456
    maxFileBytes: 16777216
```

or:

```yaml
workspace:
  storage: memory
  limits:
    maxFiles: 10000
    maxExpandedBytes: 268435456
    maxManagedTextBytes: 67108864
    maxFileBytes: 4194304
```

`workRoot` is required only for local storage. It is absolute, dedicated,
non-root and not sent over the wire. Limits must be positive and may be lowered
or raised by the operator within implementation safety ceilings; Workflow,
labels and model input cannot override them.

Runtime probes this immutable capability before registration and reports:

```json
{
  "workspaceCapabilities": {
    "storage": "local",
    "modes": ["direct", "overlay"],
    "limits": {
      "maxFiles": 50000,
      "maxExpandedBytes": 2147483648,
      "maxManagedTextBytes": 268435456,
      "maxFileBytes": 16777216
    }
  }
}
```

The capability snapshot is positive and frozen after registration. `storage`
is operational diagnostics; placement requires the requested mode and exact
Toolset/tool capabilities. Scheduler does not prefer local over memory.
Runtime repeats the compatibility check during prepare.

Operations exposes this frozen positive capability (storage, supported modes
and limits), but never workspace refs, paths or content. It therefore makes a
temporary workspace-mode placement wait diagnosable without weakening
allocation isolation.

## Secure hydration

Prepare performs, in order:

1. validate the complete `AllocationSpec` and local capability;
2. create an allocation-private workspace session;
3. read exact source artifacts through the existing allocation Artifact API;
4. require `application/zip`, scan and extract each archive under its target;
5. if supplied, read, validate and apply the cumulative overlay state;
6. establish the effective initial tree and, for overlay, its checkpoint;
7. construct selected Toolsets and Worker;
8. return ready only after every step succeeds.

There is no new Artifact endpoint or credential.

ZIP hydration rejects the whole allocation for:

- absolute, empty, dot, parent, backslash, NUL/control, drive or UNC names;
- duplicate normalized paths and file/directory type conflicts;
- symlink, hard-link, device, socket, FIFO or other special entries;
- a target escape or overlap after prefixing;
- declared/observed size mismatch, compression bomb, file/count/depth/expanded
  byte bound, or deadline exhaustion.

`maxFiles` bounds the complete normalized managed tree: regular files,
explicit directories and directories implied by nested member names all count.
This keeps a one-member archive with an extremely deep path from expanding
beyond the capability advertised by the Runtime.

Local storage retains ordinary binary regular files. Memory storage skips
binary files and counts their bytes toward expanded input but not managed text.
A managed text file is strict UTF-8 with no NUL. In overlay mode only the text
projection participates in overlay operations, state, diff and digests;
reading or mutating an unmanaged binary path through text tools returns
`binary_file_unsupported`.

## Direct and overlay semantics

### Direct

`direct` mutates only the disposable allocation-private working copy. Local
means a private directory below Runtime `workRoot`, never an operator checkout;
memory means an isolated fsspec tree. Changes are visible to future subprocesses
inside the same allocation but disappear on release. Direct mode has no
automatic diff, rollback or export. A Worker that needs persistence must write
an ordinary Run artifact explicitly.

#### Local direct: disk is authoritative

Design amendment, implemented and release-verified on 2026-09-06 in V30-001
through V30-004. The filesystem prerequisite for the execution sandbox in
[21](21-podman-sandbox.md) is satisfied; Podman itself is not enabled by this gate.

Implementation is decomposed into V30-001 through V30-004 in the
[task catalog](../../tasks/index.yml); V30-004 is the prerequisite release gate.
Executable acceptance coverage is recorded as LD1 through LD10 in
[`project_workspace_matrix.yml`](../../tests/e2e/project_workspace_matrix.yml).
`make test-local-direct-workspace` runs the focused gate; V30-004 additionally
requires workspace process E2E and repository-wide `make verify`.

| Mode | Storage | Authoritative effective state |
|---|---|---|
| `direct` | `local` | Current files and directories on disk in the allocation's `run_workdir` |
| `direct` | `memory` | Allocation-private managed tree with its memory backend |
| `overlay` | `local` | Managed overlay view over the hydrated local base |
| `overlay` | `memory` | Managed overlay view over the hydrated memory base |

For `direct + local`, hydration and optional state import initialize disk.
After prepare, the original archive, imported state and retained memory cannot
substitute for current files. Creates, writes, renames, deletes, empty-directory
changes and text/binary transitions completed by another process before an
operation begins must be visible to that operation. No model-visible refresh,
new allocation, executor callback or filesystem watcher is required.

YAML/wire shapes, Toolset refs and model-visible signatures remain unchanged.
Direct still has no automatic diff, rollback or export. Memory storage and
both overlay variants retain their existing semantics.

#### Reads, edits and snapshots

Tools retain narrow `WorkspaceReader` and `WorkspaceWriter` handles. The local
direct implementation obtains the current state from disk:

- `read_text` reads the requested regular file with bounded I/O, without
  requiring unrelated source texts to be loaded;
- listing and globbing use current entries and types;
- line edits, appends and exact replacements read the current target under
  the workspace operation lock before applying the transformation;
- copy, move, removal and directory creation preflight current source and
  destination state; they never replay a delta from the hydration tree or
  resurrect an externally deleted file from retained memory;
- snapshots, observation metadata, search and analysis obtain the current
  managed-text projection and its existing canonical digest.

A `WorkspaceSnapshot` is an immutable value for one operation, not a second
writable filesystem or an overlay checkpoint. Search and analysis can finish
against that value while later edits occur. The next call acquires a fresh
view. Digest-bound cursors reject a changed projection rather than combining
pages. The digest continues to describe the existing managed-text projection,
not all binary contents, permissions or timestamps.

Source contents may be retained while acquiring or consuming a bounded
snapshot. A persistent source-content cache cannot determine freshness:
remembered size/mtime, successful hydration and invalidation only after known
exec calls are insufficient. The first implementation acquires snapshots from
disk on demand. Derived analysis caches may be reused only after acquiring
the current snapshot digest; [12](12-code-analysis-tools.md) owns those caches.

Complete snapshots respect entry/file/byte limits and operation deadlines;
they never report a partial tree as complete. Potentially blocking scans,
reads and mutations run outside the asyncio event loop. Cancellation does not
release ownership of an in-flight mutation or let cleanup race that mutation.

#### Concurrency and failures

An allocation-owned coordination boundary serializes Workspace API reads,
mutations and snapshot acquisition. It covers the entire read/transform/write
sequence. Analysis releases it after acquiring its immutable snapshot.

Completed external changes are supported. Arbitrary concurrent host writers
do not participate in this boundary: there is no transactional whole-tree
snapshot or lost-update guarantee against them. Observed disappearance,
replacement or type changes that prevent completing an operation produce a
bounded safe failure, never a fallback to cached contents or an infinite
rescan. Coherent multi-file acquisition requires writers to be quiescent or
explicitly coordinated. The implemented executor in [21](21-podman-sandbox.md)
uses this same coordination boundary through trusted descendant completion.
Cancellation does not hand off the workspace before confirmed stop/removal.
The real command/edit/snapshot evidence is part of `make test-podman-release`.

Argument/path/type/match/quota validation precedes mutation. Single-file text
replacement is atomic. This amendment does not introduce crash-atomic
multi-file transactions. Recovery after physical I/O failure is limited to
the operation's own effects and cannot overwrite unrelated external changes.
An unconfirmed mutation or cleanup leaves the workspace unavailable/fenced;
an old memory tree is not evidence that disk state has been restored.

#### On-disk validation and limits

Archive validation does not validate later external changes. Operations
recheck paths and file types on access, including both sides of copy/move and
recursive removal. All access stays rooted in the allocation content
directory. Descriptor-relative access or an equivalent mechanism must prevent
path replacement from redirecting an operation outside that root;
`resolve`-then-open or string-prefix checks alone are insufficient.

Existing symlinks and multiply linked regular files fail preflight. They are
not followed or interpreted as another host path. FIFOs, sockets and devices
are rejected without blocking reads. Invalid or colliding normalized on-disk
names produce a safe error, not silent renaming or omission from a complete
snapshot. Ordinary binary files remain on disk and are classified as binary;
text tools retain `binary_file_unsupported` for unsupported access.

Existing workspace limits apply to current-state acquisition and mutation
preflight, not just archive input. Complete acquisition checks entry count,
regular-file bytes and managed-text bytes; reads stay bounded if a file grows
during acquisition. A violation yields `workspace_limit_exceeded`, not stale
data, hidden partial success or automatic deletion of external outputs.
Build/dependency files are not silently excluded from the complete workspace
contract. Scope-limited reads can remain available within their own bounds
without constructing a complete snapshot.

Workspace API limits do not impose disk quotas on external processes. Capacity
controls for command writes belong to the executor deployment under [21].

For local direct mode, text replacement is an atomic rename relative to opened
directory descriptors. Parent traversal does not follow symlinks, a replaced
final symlink is itself replaced rather than followed, and a failed commit
removes its private temporary file. This is defense in depth for filesystem
races; another same-UID process with unrestricted access to Runtime `workRoot`
is inside the Runtime host trust boundary. Executors must be confined to
explicitly exposed content directories and their approved mounts.

The Podman executor in [21](21-podman-sandbox.md) requires `local + direct` and
cannot be combined with Contractor `overlay`. This restriction is about the
workspace mode, not the container engine's internal image-layer storage.

### Overlay

`overlay` presents a copy-on-write text view over the hydrated private base.
The lower stays unchanged. The upper contains canonical text writes, explicit
directories and tombstones. All mutations and change operations serialize on
one session lock and are exception/cancellation atomic.

At prepare:

```text
sources S -> apply imported cumulative state I -> checkpoint B
```

When `state` is absent, `I` is the canonical empty overlay and `B = S`. An
unchanged invocation still exports a valid cumulative identity state `S -> S`
with an empty operation list and an empty checkpoint diff `S -> S` when the
Workflow declares `export`. These artifacts are explicit internal Workflow
lineage, not evidence that a Worker edited the tree. V1 defines no merge or
join operation; a future join must consume exact state artifacts explicitly
and define conflict semantics separately.

During one A2A invocation the Worker produces effective tree `F`:

- `diff` and `changed_paths` describe `B -> F`;
- `rollback_changes` restores `B`, not original sources `S`;
- exported overlay state describes cumulative `S -> F`;
- exported human diff describes the current invocation `B -> F`;
- after a graceful terminal invocation and successful export, `F` becomes the
  next checkpoint for another sequential A2A task on the same allocation.

There is no model-visible or automatic host `materialize`. Persistence is an
artifact export. Trusted analysis tools may build private analysis inputs
from an immutable snapshot. An execution sandbox sharing the writable project
workspace requires local direct mode under [21](21-podman-sandbox.md).

## Cumulative overlay artifact

Media type:
`application/vnd.contractor.workspace-overlay+json`.

The canonical versioned JSON is:

```json
{
  "apiVersion": "contractor.workspace/v1",
  "kind": "WorkspaceOverlay",
  "baseWorkspaceDigest": "sha256:...",
  "resultWorkspaceDigest": "sha256:...",
  "operations": [
    {"op": "create_directory", "path": "backend/new"},
    {"op": "write_file", "path": "backend/new/readme.md", "text": "..."},
    {"op": "delete_path", "path": "backend/old.txt"}
  ]
}
```

Operations are sorted into one deterministic canonical representation and use
relative normalized paths. `write_file.text` is strict UTF-8 without NUL; it
does not encode arbitrary bytes. Metadata, modes, ownership, timestamps,
symlinks and binary patches are not represented. A future binary patch, if
needed, is a separate artifact/media type.

`baseWorkspaceDigest` hashes the canonical managed-text projection of exact
hydrated sources `S`; `resultWorkspaceDigest` hashes the reconstructed managed
text projection after operations. Imported state must match the actual base
and recompute its result digest before it is applied. State does not contain a
previous Artifact revision: revision is an ArtifactStore concern hidden from
the model.

One state artifact is self-contained relative to sources. Given `S` and state
revision 8, revision 7 is unnecessary.

Human diff is deterministic UTF-8 unified diff with media type `text/x-diff`,
relative paths, LF syntax and bounded context. It is for analysis and review,
not authoritative reconstruction.

## Model-visible Toolsets

AgentTemplate selects an exact subset by normal Toolset allowlist. Tool
implementations receive narrow `WorkspaceReader`, `WorkspaceWriter` and
`WorkspaceChanges` handles, never raw fsspec objects or host paths.

`openapi@1` declares optional read access for provenance checks. When a project
workspace is present, every `evidence_files` path is validated against one
`WorkspaceReader` snapshot—the same logical tree exposed through
`filesystem@1`. It must not inspect `AllocationWorkspace` scratch in that mode.
Without a project workspace, the Toolset retains the legacy
`source-analysis@1` compatibility path below allocation-private
`scratch/source`. The two sources are never combined and project-workspace
presence always takes precedence, including for a missing path.

The read-only `code-analysis@1` Toolset consumes the same `WorkspaceReader`
snapshot boundary. Its portable Tree-sitter operations and local-only
allocation child used for Trailmark graph analysis are specified in
[12](12-code-analysis-tools.md); neither receives the provider's physical root.

### `filesystem@1`

| Tool | Contract |
|---|---|
| `ls(path="", cursor="", limit=100)` | Sorted immediate entries with type and text size; bounded cursor pagination. |
| `glob(pattern, cursor="", limit=100)` | Sorted relative path matches; bounded scan and pagination. |
| `read_file(path, start_line=1, max_lines=200)` | Bounded text lines with total/next metadata and preserved newline information. |
| `grep(pattern, path="", glob="**/*", cursor="", limit=100)` | Bounded literal/regex text matches with line and truncated excerpt. |

### `edit-files@1`

| Tool | Contract |
|---|---|
| `write_file(path, content)` | Replace/create one text file atomically. |
| `append_file(path, content)` | Append text preserving existing newline style. |
| `mkdir(path, parents=false)` | Create a directory. |
| `rm(path, recursive=false)` | Remove a file or explicit tree. |
| `cp(source, destination, recursive=false)` | Copy within the one workspace root. |
| `mv(source, destination)` | Move within the one workspace root. |
| `insert_line(path, line, content)` | Insert at a 1-based line boundary. |
| `edit(path, old, new, replace_all=false)` | Require exactly one match unless `replace_all`; no fuzzy edits. |
| `replace_range(path, start_line, end_line, content)` | Replace an inclusive 1-based line range. |

### `workspace-changes@1`

| Tool | Contract |
|---|---|
| `changed_paths(cursor="", limit=100)` | Sorted created/modified/deleted paths since checkpoint. |
| `diff(path="", cursor="", max_bytes=65536)` | Bounded deterministic `B -> F` unified diff. |
| `rollback_changes(path="")` | Restore checkpoint for one path/subtree or the whole workspace. |

All paths are relative to the single `run_workdir` root. Leading `/`, `..`,
backslash, URI, drive/UNC, NUL/control and paths above configured limits are
rejected. Tools never expose a list of host roots because there is exactly one
logical root.

Read responses are bounded; truncation is explicit and resumable. Mutations
preflight resulting file/count/managed-byte limits before changing state.
Existing CRLF style is retained by line-oriented edits. `edit` cannot silently
choose among multiple matches.

## Auto export and A2A ordering

Overlay state has an independent 16 MiB ceiling. The exporter passes the
minimum of that ceiling and the generic Artifact transport limit (currently
64 MiB) to export preparation. Increasing transport capacity must not make
small overlay exports invalid or implicitly raise overlay decode budgets.

For overlay with `export`, Runtime augments every successful normal
`WorkerResult` after semantic output and deterministic observations have been
validated and before its `WorkerCompletion` becomes terminal over A2A:

1. snapshot `F` under the workspace lock;
2. encode and locally validate cumulative state `S -> F` and diff `B -> F`;
3. write state and diff to `AllocationSpec.namespace` through the current
   allocation Artifact API;
4. inject both exact returned refs into their reserved result slots after the
   ordinary observed-ref projection;
5. validate the final Worker result/completion; only then publish the terminal
   A2A response;
6. advance checkpoint to `F` after both writes have succeeded.

Both artifacts bind the same `resultWorkspaceDigest`; diff metadata may carry
the digest outside the diff text. If either write fails, no terminal success is
returned and checkpoint does not advance. A partial artifact revision is
harmless because Scheduler never selects it. Temporary Artifact API failures
produce retryable `workspace_export_failed`; invalid state/quota/fence failures
are classified by their stable cause.

No export occurs for `WorkerFailure`, `INPUT_REQUIRED` (the allocation
continues), lost lease, forced abort/cancel, crash, missing/unsafe structured
semantic output or invocation cancellation. This ordering guarantees artifact
writes finish before Scheduler observes terminal A2A state and enters its
write-fenced `finalizing`/`aborting` transition.

## Lifecycle and cleanup

- partial prepare closes clients and deletes the private session;
- graceful finalize stops A2A/model work and flushes metrics but retains
  workspace data until release for idempotent finalization/replay;
- release closes handles and deletes the private local directory or memory
  namespace, then makes the one Runtime slot idle;
- abort or lease loss blocks new workspace operations, cancels work, performs
  no export and deletes the session;
- a cleanup failure fences/drains that Runtime process and is reported safely;
  it must not block Scheduler progress for other Runtime Agents;
- startup deletes only stale directories bearing a valid Contractor ownership
  marker immediately below the exact configured `workRoot`. It never performs
  broad or marker-free recursive cleanup.

Potentially blocking local filesystem removal never runs on the Runtime
Agent's asyncio event-loop thread. Release owns one allocation-wide cleanup
task covering Toolset close, project workspace disposal, scratch SandboxProfile
cleanup and RuntimeAdapter rollback. The task is awaited under one finite
cleanup deadline. If that deadline expires, the request returns a retryable
cleanup failure and the slot remains fenced, while the still-owned task is
retained; a later retry joins that exact task instead of starting a concurrent
second removal of the same tree. No allocation can reuse the slot until the
task completes and release is authoritatively acknowledged. Abort and lease
loss use the same bounded non-blocking primitives and retain their fail-closed
process-exit rule when teardown cannot be confirmed.

## Stable errors and telemetry

Initial stable errors include:

- `workspace_required`;
- `workspace_mode_unsupported`;
- `workspace_capacity_exceeded` (retryable placement/resource failure);
- `workspace_source_invalid` (malformed or malicious archive, non-retryable);
- `workspace_state_invalid` (non-retryable);
- `workspace_path_invalid`;
- `workspace_not_found`;
- `workspace_type_conflict`;
- `binary_file_unsupported`;
- `workspace_limit_exceeded`;
- `workspace_operation_unsupported`;
- `workspace_unavailable` (bounded filesystem acquisition, I/O or mutation failure);
- `workspace_export_failed`.

Transient Artifact transport and disk availability errors are retryable;
malformed ZIP/state and deterministic quota violations are not. Errors sent to
Server/model contain stable code, retryability and bounded safe context only.

Metrics may include storage kind, mode, source count, file/byte counts,
operation name/outcome/duration, changed count and exported byte counts. They
must not contain file contents, search patterns, diffs, archive names, host
paths, credentials or arbitrary exceptions.

## Initial acceptance

1. A Runtime without workspace configuration continues to run existing
   artifact workflows and reports no workspace capability.
2. Go and Python strict fixtures round-trip the same WorkspaceContext,
   AllocationSpec projection and capability snapshot; unknown fields fail.
3. Scheduler pins exact source/state refs and never sends a logical mutable
   binding in AllocationSpec.
4. Local and memory sessions hydrate safe ZIP fixtures deterministically and
   remain isolated across two allocations.
5. Direct and overlay read/edit behavior is backend-independent; overlay lower
   data is unchanged and rollback restores the invocation checkpoint.
6. Exported cumulative state alone plus sources reconstructs the final text
   tree and validates both digests; `text/x-diff` describes only the current
   checkpoint delta.
7. Router Workers cannot see one another's unexported changes.
8. Traversal, link/special entry, archive bomb, binary text edit, quota,
   cancellation, lease loss and cleanup fault suites fail closed with no path
   escape or retained content leak.

### Local direct amendment acceptance (release verified)

These cases use actual files modified outside WorkspaceWriter while the same
allocation remains alive, and cover both narrow handles and Toolset consumers.

1. External writes, creates, renames, deletes and empty-directory changes are
   visible on the next applicable call, without a refresh or executor signal.
   A same-size rewrite with restored mtime still returns the new contents.
2. Exact edit, append, copy and move use current contents. Unrelated external
   changes remain untouched; deleted files are never resurrected from memory.
3. Text/binary transitions update classification. Snapshots and observation
   digests reflect the current managed-text projection.
4. Search and analysis see external edits on their next call. Changed digests
   invalidate caches/cursors; previously returned snapshots remain immutable.
5. Single-file reads do not load unrelated source texts. Complete acquisitions
   fail safely on limits without fallback to the last valid snapshot.
6. Root/parent/leaf symlinks, hard links, special files, invalid names and
   adversarial path replacement cannot escape the root, leak outside contents
   or block the event loop indefinitely.
7. Workspace API calls preserve operation ordering. Cancellation cannot race
   release against an owned filesystem mutation. Detected external races fail
   safely without claiming protection from arbitrary concurrent writers.
8. Validation failure leaves files unchanged. Injected I/O/recovery failures
   report failure and fence on unconfirmed teardown; retained memory cannot
   be used as proof that disk state was restored.
9. Direct state import reaches disk before readiness; later external changes
   take precedence. Release removes the private copy. Persistence remains
   explicit ordinary Artifact writes.
10. Memory/direct and both overlay variants retain existing parity,
    checkpoint, export and isolation behavior. This amendment alone adds no
    execution capability or model-visible tool.

## Deliberately deferred

- automatic synchronization between Workers inside one StageExecution;
- remote/shared filesystem backends and Kubernetes volume integration;
- archive types other than ZIP and source filters/stripComponents;
- binary overlay patches, metadata and symlink preservation;
- model-visible materialization into an operator checkout;
- execution implementation, owned separately by [21](21-podman-sandbox.md);
- cross-Run workspace cache, fork/merge and conflict resolution;
- UI browsing/editing of workspace contents.

## Invariants

1. Workflow chooses semantics; Runtime startup chooses physical storage.
2. Every hydrated input in AllocationSpec is an exact RunScope ArtifactRef.
3. Every workspace is private to one allocation, including Router siblings.
4. No model input can select a host path, storage backend or infrastructure
   credential.
5. Direct mode changes only a disposable copy; overlay changes only its upper.
6. Overlay persistence uses ordinary artifacts before terminal A2A response.
7. Cumulative state is self-contained relative to exact sources and contains
   text only.
8. Release/abort/lease loss cannot leave a reusable slot with a live workspace.
9. Local direct disk state is authoritative; snapshots are derived values,
   never a fallback for failed current-state acquisition.
