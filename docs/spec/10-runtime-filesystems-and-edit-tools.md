# 10 — Runtime filesystems and Edit tools

Status: **Working agreement**

Depends on: [01](01-agent-template.md), [02](02-runtime-and-a2a.md) and
[04](04-execution-lifecycle-and-metrics.md)

## Purpose and boundary

Contractor Runtime Agent may expose a bounded project filesystem to one
allocation's Worker. The filesystem implementation, its host roots, copy-on-
write state and write authority belong entirely to the Python Runtime Agent.
They are not Workflow data, Control Plane configuration or Server state.

The Server:

- never receives or stores a host path;
- never creates, copies, diffs, rolls back or materializes a workspace;
- has no filesystem endpoint, database table, Artifact type or allocation wire
  field;
- sees only the ordinary exact Toolset refs and positive tool subsets already
  used for capability-aware placement;
- retains only the static Toolset/tool-name descriptors needed to validate an
  AgentTemplate allowlist.

Runtime Agent reads one optional local workspace configuration before startup
capability discovery. That immutable process configuration determines which
roots exist, which filesystem implementation backs each root and whether host
writes are allowed. A Runtime without workspace configuration simply omits the
filesystem Toolsets from its positive capability snapshot and continues to
serve artifact-only AgentTemplates.

`fsspec` is the initial Python implementation boundary. It is not a Contractor
wire type, persistence format or promise that every arbitrary fsspec backend is
safe. Contractor owns the narrower path, isolation, quota, overlay and error
contracts in this document and adapts selected fsspec implementations to them.

## Concepts

The terms in this document are distinct:

- **allocation scratch** is the private directory created by the existing
  `local-workdir@1` SandboxProfile for skills, validators, temporary files and
  lifecycle cleanup;
- **workspace mount** is one operator-configured project directory presented to
  the Worker below a logical virtual name;
- **WorkspaceSession** is one allocation's effective view of all configured
  mounts;
- **lower layer** is a rooted local filesystem or an isolated in-memory
  filesystem;
- **overlay upper** is one allocation-private in-memory change set over a lower
  layer;
- **materialize** applies an overlay upper to its lower layer; it does not mean
  “write a Workflow artifact”;
- **Edit tools** are model-visible text and path mutations implemented against
  a narrow Workspace writer contract rather than a concrete filesystem class.

Allocation scratch and project workspace may both use local disk internally,
but they have different authority and lifecycle. `local-workdir@1` remains the
mandatory allocation scratch SandboxProfile and does not imply access to any
project directory. Project mounts are never placed below its cleanup root.

## Runtime-local configuration

Runtime accepts one optional `--workspace-config` path, with equivalent
`CONTRACTOR_WORKSPACE_CONFIG`. The supplied value must already be an absolute
path; Runtime does not expand `~` or environment variables. It validates and
canonicalizes the path locally and requires an existing regular,
non-symlinked file. The document is strict YAML with duplicate keys rejected:

```yaml
schemaVersion: contractor.runtime-workspace/v1

mounts:
  project:
    path: /home/operator/src/payment-service
    mode: overlay-local
    hostWrite: true

  reference:
    path: /srv/reference/framework
    mode: memory
```

`mounts` contains between one and eight entries. A mount name matches
`[a-z][a-z0-9_-]{0,63}`. The Worker sees the example roots as `/project` and
`/reference`; it never sees either `path`. A configured path must be absolute,
clean, not a filesystem root, an existing directory and not itself a symlink.
Canonical host roots must be distinct, must not contain one another and must
not overlap Runtime's allocation work root. Unknown fields and environment-
variable or tilde interpolation inside YAML are invalid.

`mode` is exactly one of:

| Mode | Lower layer | Worker mutations | Host source mutation |
|---|---|---|---|
| `local` | rooted host directory | direct only when `hostWrite: true` | immediate |
| `memory` | fresh isolated text snapshot | direct to allocation memory | never |
| `overlay-local` | rooted host directory | allocation-private upper | only explicit materialize when `hostWrite: true` |
| `overlay-memory` | fresh isolated text snapshot | allocation-private upper | never |

`hostWrite` defaults to `false`. It is valid only for `local` and
`overlay-local`. For `local`, false means a read-only effective view and true
permits direct changes. For `overlay-local`, upper-layer edits are available in
both cases, but materialization into the host directory is available only when
true. `memory` and `overlay-memory` reject `hostWrite: true` because the host
directory is only a snapshot source.

The configuration has no includeable files, remote URL, credentials, Python
module, fsspec protocol string, arbitrary storage options or command. Initial
limits and ignore rules are registered Runtime code rather than user-supplied
executable configuration. A future schema may allow bounded declarative
filters without reinterpreting this version.

Runtime loads and validates the complete configuration before building its
FactoryRegistry. Configuration errors fail process startup. The exact file and
its resolved mount descriptors are frozen for `instance_id`; there is no hot
reload. Updating a path, mode or `hostWrite` value requires restarting Runtime,
which produces a new `instance_id` and positive capability snapshot.

The host contents of a `local` lower may legitimately change while the process
is registered, including through an authorized Worker or an operator. That is
data mutation, not capability mutation. An in-memory source snapshot, by
contrast, is captured once before first registration and reused as an
immutable baseline from which every allocation receives a fresh private copy.

## Workspace service contracts

Runtime owns the following conceptual interfaces. Exact Python protocol names
may differ, but their authority separation is normative:

```python
class WorkspaceReader(Protocol):
    def roots(self) -> tuple[WorkspaceRoot, ...]: ...
    def stat(self, path: VirtualPath) -> WorkspaceEntry: ...
    def list(self, path: VirtualPath) -> Sequence[WorkspaceEntry]: ...
    def glob(self, pattern: str, *, root: VirtualPath) -> ScanPage: ...
    def read_text(self, path: VirtualPath) -> str: ...


class WorkspaceWriter(WorkspaceReader, Protocol):
    def write_text(self, path: VirtualPath, content: str) -> Mutation: ...
    def mkdir(self, path: VirtualPath, *, parents: bool) -> Mutation: ...
    def remove(self, path: VirtualPath, *, recursive: bool) -> Mutation: ...
    def copy(self, source: VirtualPath, target: VirtualPath, *, recursive: bool) -> Mutation: ...
    def move(self, source: VirtualPath, target: VirtualPath, *, recursive: bool) -> Mutation: ...


class WorkspaceChangeSet(Protocol):
    def changed_paths(self, root: VirtualPath) -> ChangeSummary: ...
    def diff(self, root: VirtualPath, *, context_lines: int) -> DiffPage: ...
    def rollback(self, root: VirtualPath, *, recursive: bool) -> ChangeSummary: ...


class WorkspaceMaterializer(Protocol):
    def materialize(self, root: VirtualPath) -> MaterializeResult: ...
```

A `WorkspaceSession` contains a `WorkspaceReader` for every configured mount
and exposes only the additional interfaces its effective mode supports. Toolset
factories receive narrowed handles. A read-only Toolset never receives a
writer; an Edit Toolset never receives a materializer merely because both are
implemented by the same object. No model-facing tool receives an absolute
`pathlib.Path`, raw lower filesystem, overlay state dictionary or arbitrary
fsspec URL.

This inversion is the connection between Edit tools and the four modes:

```text
filesystem@1 read tools -> WorkspaceReader
edit-files@1 tools      -> WorkspaceWriter
workspace-changes@1     -> WorkspaceChangeSet / optional WorkspaceMaterializer
                                  |
                                  +-> local, memory or overlay implementation
```

Edit algorithms therefore contain no `isinstance(OverlayFS)` branch and do not
wrap a supplied filesystem implicitly. Backend selection happens once while
Runtime builds the WorkspaceSession. Overlay-only functions live in the
separate change Toolset rather than appearing and failing late on a direct
filesystem.

The underlying selected objects implement or adapt `fsspec.AbstractFileSystem`
so existing source/code tools can be ported incrementally. Contractor code
must not call `fsspec.filesystem()` with a protocol or storage options derived
from Workflow, model or Control Plane input.

## Allocation lifecycle

Runtime constructs a fresh WorkspaceSession after reserving its slot and before
building selected Toolsets:

1. validate that every selected filesystem tool is present in the process's
   frozen positive capability snapshot;
2. clone the startup text baseline for every memory-backed mount;
3. construct one rooted local view for every local-backed mount;
4. create an empty private upper for each overlay mount;
5. bind narrowed reader/writer/change handles to selected Toolsets;
6. create the Worker only after every selected tool has been constructed.

Preparation is all-or-nothing. Failure destroys every already-created memory
copy and upper and creates no ready Worker or Planner. The workspace contains
no allocation grant, RuntimeSettings secret or database credential.

One Runtime process has one slot, but model runtimes may schedule concurrent
tool calls. One WorkspaceSession serializes all mutations and materialization.
Reads observe either the complete state before a mutation or the complete state
after it. Separate Runtime processes are not coordinated by this lock; an
overlay-local materializer uses the conflict and journal contract below.

Finalization and abort stop model/tool execution before Runtime closes the
WorkspaceSession. Release discards every memory filesystem and overlay upper,
closes handles and removes materialization staging/journal state only after any
required recovery action. It does not delete configured host roots. Direct
local writes and successfully materialized host writes survive allocation
release by design; memory and unmaterialized overlay edits do not.

Lease loss follows the same self-termination rule as any Worker. Runtime does
not automatically materialize pending changes during finalize, abort, lease
loss or release. Materialization is an explicit selected tool call; absence of
that call is never inferred as success.

## Virtual paths and confinement

The model-visible namespace is POSIX-like and rooted at `/`. Each mount appears
as exactly one first-level directory. `/project/src/app.py` maps into the
configured `project` host root or private snapshot; `/etc/passwd` means an
unconfigured virtual mount named `etc` and is not a host absolute path.

Every tool path is normalized by one shared validator before backend dispatch:

- input must be a string containing valid Unicode with no NUL or control
  characters;
- backslash, URI scheme, Windows drive/UNC syntax and host `file:` paths are
  rejected rather than interpreted;
- a missing leading slash is added for convenience;
- repeated `/` and `.` components normalize deterministically;
- `..` is rejected even when lexical normalization would remain within a
  mount;
- the normalized path is Unicode NFC, has at most 32 components and at most
  1,024 UTF-8 bytes;
- the first component must name a configured mount;
- public results contain only normalized virtual paths.

The same validator applies to paths, glob roots and both operands of copy/move.
Glob syntax is path-aware: `*`, `?` and character classes do not cross `/`,
while `**` crosses zero or more components. Parent traversal is invalid rather
than an empty successful scan. Results are unique and lexicographically sorted.

Local adapters canonicalize their configured root once, route every operation
through the selected mount and verify component containment without following
symlinks. Directory walk/list/glob never enumerate or descend into symlinks.
Direct reads and mutations reject a symlink, a multiply linked regular file or
a special file at any component. Rejecting `st_nlink > 1` prevents an in-root
name from becoming an alias to an inode also named outside the configured
tree. Errors never contain the host root. Runtime does not delegate an unknown
method to raw `LocalFileSystem`, because such delegation could bypass path
validation.

For allocation-owned scratch, canonical containment is sufficient for
lifecycle cleanup but is not reused as project confinement. For project local
access, implementations use descriptor-relative/no-follow operations where the
platform provides them and verify the opened object's type. A positive host-
write capability is not advertised when the Runtime platform cannot enforce
the registered no-symlink contract.

This is a tool boundary, not an OS sandbox. A separately selected arbitrary
shell, Python executor or unsafe native tool running in the same process could
use the Runtime process's ambient filesystem permissions. No such executor is
part of these Toolsets. Strong isolation against hostile in-process code still
requires a future container/process SandboxProfile.

## Local and in-memory lower layers

### Rooted local filesystem

The rooted local adapter exposes the real regular files and directories below
one configured host root. Read tools accept only UTF-8 text and report binary
or undecodable files as unsupported; listing may still report such a regular
file's path and size. The adapter never exposes owner/group IDs, device data,
host timestamps or the host path to the model.

With `hostWrite: false`, no writer is created. With `hostWrite: true`, direct
Edit operations modify the real rooted directory. Whole-file text writes use a
same-directory temporary regular file, bounded bytes, flush and atomic replace;
they never truncate the target before validation succeeds. Cross-mount move is
invalid. Direct local mode has no diff or rollback contract: callers wanting
those properties must use `overlay-local`.

Bounded multi-entry direct operations preflight the complete affected set and
use staged targets/backups to roll back an ordinary in-process error. Once such
an operation enters its commit/rollback critical section, coroutine
cancellation is observed only after that section finishes. This is not a
crash-safe tree transaction: process/host failure can retain a completed or
partially applied direct operation, and Runtime performs no direct-local
recovery journal. An operator requiring review, rollback and restart recovery
uses `overlay-local` plus explicit materialization instead.

### Isolated in-memory filesystem

Before registration, Runtime imports each memory-backed host root into an
immutable bounded baseline. It includes regular text files and safe
directories only. A file is text when its bytes are valid UTF-8 and contain no
NUL. Known binary/archive/media extensions may be skipped before reading, but
content validation remains authoritative. Symlinks, hard links, sockets,
devices and other special entries are never imported.

The initial count/size/depth bounds apply independently to each mount:

| Quantity | Bound |
|---|---:|
| visited entries | 100,000 |
| imported regular files | 10,000 |
| one imported file | 4 MiB |
| total imported text bytes | 64 MiB |
| directory depth | 50 |
| complete configured import phase | 30 seconds total |

Version-control metadata, Contractor state, virtual environments, dependency
trees, caches, build outputs and common binary formats use one fixed Runtime
ignore set. The import summary records only counts/bytes by skipped reason and
safe virtual paths for bounded diagnostics; no skipped content is retained.
Hitting a scan/time/byte limit fails that mount's capability probe rather than
silently presenting an incomplete project as complete.

Every allocation clones the immutable baseline into an allocation-owned store.
Empty directories are preserved. The clone has the same reader/writer behavior
as the local adapter, except writes never touch the host and the entire state is
destroyed on release.

The built-in `fsspec.implementations.memory.MemoryFileSystem` uses global class-
level storage shared by instances. Runtime therefore must not expose an
unprefixed stock instance as an allocation boundary. The implementation uses an
instance-owned filesystem or an unguessable allocation prefix behind a wrapper
that makes every other prefix unreachable, disables fsspec instance caching
and proves teardown/isolation in tests.

## Overlay filesystem

An overlay composes one lower `WorkspaceReader` with an allocation-private
in-memory upper. The effective view follows these rules:

- reads prefer an upper file and otherwise read the lower;
- upper directories merge with lower directories;
- a tombstone hides a lower path and every descendant;
- a write creates missing upper parents and removes covering tombstones;
- writing through a file component and file/directory type conflicts fail;
- copying or moving reads the effective view and writes only the upper;
- recursive removal records the smallest sufficient tombstone and never
  mutates the lower;
- no-op writes whose bytes equal the lower are removed from the change set;
- one re-entrant/session lock serializes upper state.

The upper is bounded by the same 10,000-file, 4-MiB-per-file and 64-MiB text
limits as an in-memory lower. Limits apply to the resulting upper/effective
state before committing one mutation. A failed or cancelled mutation leaves
the previous effective view intact.

`changed_paths(root)` returns sorted `added`, `modified`, `deleted` and
`typeChanged` virtual paths plus counts. Empty overlay-only directories are
included only when explicitly created and still empty. It never returns lower
hashes or implementation state.

`diff(root, context_lines)` visits the sorted changed-path set. UTF-8 data with
no NUL receives deterministic unified diff output with virtual `a/...` and
`b/...` labels. Binary data, if present in a local lower, receives a bounded
`Binary files differ` marker. Creation, deletion and file/directory type
changes have explicit headers. No changes returns an empty diff. Model-visible
output is capped while returning total UTF-8 bytes and `truncated`; truncation
is never represented as a complete diff.

`rollback(root, recursive)` discards matching upper files, directories and
tombstones and reveals the current lower view. It works for both modified lower
paths and newly created upper-only paths; the lower path need not exist. Root
`/mount` with recursive true resets that mount's complete upper atomically.
Rollback never mutates the lower.

Overlay state is not serialized into WorkflowRun or Server state. A future
artifact export may persist a patch or tree through the existing Artifact API,
but it is separate from the runtime-only filesystem contract.

## Materialization

Materialization applies one overlay mount's current upper to its lower. It is
available for `overlay-memory` and for `overlay-local` only when
`hostWrite: true`. It is never automatic and is model-visible only when the
AgentTemplate explicitly selects `materialize_changes` from
`workspace-changes@1`.

For an in-memory lower, materialization is atomic under the session lock: build
the new lower state, replace the old state, clear the upper and establish a new
baseline. The effective bytes do not change, but subsequent changes/diff are
relative to the new lower. State remains allocation-local and disappears on
release.

For a local lower, every first mutation captures the lower path's type and
SHA-256 content hash, or an explicit absent marker. Before the first host write,
materialization rechecks every affected path and relevant parent. Any mismatch
returns `workspace_base_conflict`, writes nothing and retains the upper.

After successful preflight, Runtime creates a bounded journal and staged files
below its dedicated work root, never inside the configured project. The journal
contains the mount/config digest, normalized virtual operations, expected
fingerprints, staged file hashes and enough backed-up lower bytes/metadata to
restore affected regular files. It contains no RuntimeSettings secret. Runtime
then applies deterministic operations using no-follow descriptor-relative
access and per-file atomic replacement. Directories are created before files;
deletions occur after replacements from deepest to shallowest.

The first local-materialization bounds are:

| Quantity | Bound |
|---|---:|
| affected virtual paths | 10,000 |
| one backed-up lower regular file | 16 MiB |
| total staged replacement bytes | 64 MiB |
| total backed-up lower bytes | 128 MiB |
| encoded journal metadata | 8 MiB |
| one materialize call before outer Stage deadline | 60 seconds |

Exceeding a bound returns `workspace_limit_exceeded` before the first lower
write and retains the upper. One single-slot Runtime can create at most one
live journal. Startup scans at most eight journal directory entries and 256 MiB
of journal/staging data; overflow, an unexpected second live journal or an
unknown entry fails readiness as `workspace_recovery_required`. Recovery shares
the complete 30-second startup deadline and records resumable rollback progress
before yielding/failing, so a later restart never guesses which operation was
already restored.

A multi-file host filesystem has no portable atomic tree commit. Contractor
therefore promises crash recovery rather than false transaction atomicity:

- an ordinary operation error triggers immediate journal rollback;
- if rollback succeeds, the upper remains and materialization returns a safe
  failure with no intended lower change;
- if process/host failure interrupts application or rollback, the journal is
  retained;
- before the Runtime can register again, startup recovery finishes rollback or
  fails readiness and leaves the process unavailable;
- only after applied paths and parent directories are durably flushed does
  Runtime mark the journal committed, clear the upper, delete the journal and
  report success;
- response loss after commit is safe: a repeated call observes no pending
  changes and returns a successful no-op.

No other allocation can run in that Runtime process during materialization.
External processes editing the same host root remain possible; fingerprint
conflicts prevent known stale overwrites but cannot turn a general host
filesystem into a distributed transaction. An operator must not configure the
same writable root in multiple Runtime processes unless an external exclusive
ownership mechanism exists.

Materializing to a local lower changes operator-owned host data and survives
Run cancellation and allocation release. That authority requires both
`hostWrite: true` in trusted Runtime startup configuration and explicit
AgentTemplate selection of `materialize_changes`. Runtime labels, Workflow
parameters, Stage instructions, skills and model arguments can supply neither.

## Model-visible Toolsets

### Read tools: `filesystem@1`

The exact Toolset exports:

| Tool | Contract |
|---|---|
| `list_roots()` | Return mount names, virtual roots, effective `readOnly`/`overlay`/`materialize` flags and import summary counts; never host paths. |
| `ls(path, offset=0, limit=100)` | List immediate entries with honest page metadata. |
| `glob(pattern, path="/mount", offset=0, limit=100)` | Return sorted bounded matching entries and scan truncation metadata. |
| `read_file(path, start_line=1, max_lines=200, with_line_numbers=false)` | Read a strict UTF-8 line window with byte/line truncation metadata. |
| `grep(pattern, path="/mount", regex=false, case_sensitive=false, max_results=100)` | Bounded fixed-string or timeout-bounded regex search with virtual path, one-based line and excerpt. |

One list page contains at most 200 entries. One visible read/diff contains at
most 128 KiB and 400 lines. One grep scans at most 32 MiB or 100,000 files for
at most two seconds; each regex line match has a short timeout. Every bounded
result says whether it is truncated and how to narrow or continue it.

### Edit tools: `edit-files@1`

The exact Toolset exports the ported, backend-independent operations:

| Tool | Contract |
|---|---|
| `write_file(path, content)` | Create or atomically replace strict UTF-8 text. |
| `append_file(path, content)` | Append text, creating a missing file. |
| `mkdir(path, create_parents=true, exist_ok=true)` | Create a directory. |
| `rm(path, recursive=false)` | Remove an existing path; non-empty directories require recursion. |
| `cp(source, target, recursive=false)` | Copy within one mount; directories require recursion. |
| `mv(source, target, recursive=false)` | Move within one mount and reject self/descendant targets. |
| `insert_line(path, content, anchor, where="before", occurrence=1)` | Insert at one exact anchor occurrence while preserving LF/CRLF. |
| `edit(path, old_string, new_string, replace_all=false)` | Perform unambiguous literal replacement; empty old text creates only a missing file. |
| `replace_range(path, start_line, end_line, content, preserve_trailing_newline=true)` | One-based inclusive replace, insertion or deletion with validated bounds. |

Existing non-UTF-8 or NUL-containing files cannot be rewritten through text
Edit tools. `edit` without `replace_all` requires exactly one match; multiple or
zero matches are errors and leave bytes unchanged. A no-op result is successful
but explicitly reports `changed: false`. Line operations preserve the file's
dominant CRLF/LF style. All mutations validate the resulting per-file and
workspace quota before side effects. One tool-supplied text value is at most
1 MiB and the resulting file at most 4 MiB.

The Toolset is positively advertised when at least one configured mount has an
effective writer: a memory mount, an overlay upper or a `local` mount with
`hostWrite: true`. A write targeting another read-only mount returns a bounded
`workspace_read_only` error. `list_roots` lets the model distinguish mounts;
the Server does not need a path-level capability matrix.

### Overlay tools: `workspace-changes@1`

The exact Toolset exports:

- `changed_paths(path="/mount")`;
- `diff(path="/mount", context_lines=3)`;
- `rollback_changes(path="/mount", recursive=true)`;
- `materialize_changes(path="/mount")`.

The first three are advertised when at least one overlay mount exists.
`materialize_changes` is advertised when at least one configured overlay can
materialize into its lower. A call against an incompatible mount returns the
bounded `workspace_operation_unsupported` error without lower mutation.

Tool names are explicit AgentTemplate allowlists under [01]. No filesystem
tool is injected merely because Runtime has a configured mount. Selecting a
tool controls model visibility but cannot override mount-local read-only or
host-write authority. Toolset construction uses the process's already-probed
WorkspaceProvider; the model cannot choose a backend or host root.

## Capability discovery and placement

Workspace configuration is consumed while building the Runtime's local
FactoryRegistry. Before registration Runtime performs bounded probes:

- validate/canonicalize each mount and prove confinement/read behavior;
- capture and validate memory baselines;
- for configured host write, prove that the platform exposes the required
  descriptor-relative/no-follow primitives and that the root can be opened
  with the requested authority, without creating a probe entry in the operator
  project;
- construct and discard an overlay and exercise write/diff/rollback;
- recover any retained local materialization journal before declaring host
  write/materialize capability.

The complete project import shares the existing 30-second Runtime capability
discovery deadline. Individual filesystem Toolset probes reuse the prepared
provider and perform no second tree scan.

The resulting positive subsets use the existing registration fields:

```text
filesystem@1        -> available read operations
edit-files@1        -> available Edit operations
workspace-changes@1 -> available change/materialize operations
```

There is no fifth workspace capability dimension and no backend/mount payload
in registration. The Server performs its existing exact Toolset/tool subset
matching. Runtime revalidates the selected tools against its frozen snapshot
before opening a WorkspaceSession.

Because physical mount identity is intentionally not sent to Control Plane,
this Runtime-only first increment has one deployment constraint: every live
Runtime Agent eligible for the same filesystem-using AgentTemplate must expose
equivalent logical mount names, project content, effective modes and write
authority, or only one such Runtime may be connected. Otherwise generic
capability placement could correctly find the required tools but select the
wrong physical project or mutation semantics. Run-scoped source artifacts and
mount affinity would require a separate Server/allocation contract and are not
smuggled into this increment.

Labels remain unrelated. They may configure LLM/telemetry/proxy adapters under
[07](07-runtime-labels-and-infrastructure-config.md), but cannot add a mount,
change a filesystem mode, enable `hostWrite`, select an Edit tool or
materialize pending changes.

## Errors, metrics and observations

Stable safe Runtime error classes include at least:

- `workspace_path_invalid`;
- `workspace_root_unknown`;
- `workspace_not_found`;
- `workspace_read_only`;
- `workspace_binary_unsupported`;
- `workspace_limit_exceeded`;
- `workspace_operation_unsupported`;
- `workspace_base_conflict`;
- `workspace_materialize_failed`;
- `workspace_recovery_required`.

Model-visible diagnostics may contain a validated virtual path, mount name,
limit and safe operation. They never contain host paths, raw OS/fsspec errors,
file contents, journal paths or configuration YAML.

Every selected filesystem tool uses the existing allocation State metrics.
Retained arguments/results contain operation, normalized virtual paths,
booleans, counts, byte sizes, truncation, duration and stable error code.
Search patterns, Edit old/new/content strings, file bytes and diff text are not
retained; metrics record their UTF-8 length and SHA-256 only where correlation
is useful. `ExecutionReport` aggregates calls/errors/bytes without embedding
workspace content. Overlay pending-change counts and materialization outcome may
be included, but upper state and lower hashes may not.

Close/finalize errors never prevent Runtime from reporting accumulated metrics.
An unfinished host journal fences startup/slot reuse until recovery; ordinary
discard of a memory or overlay upper remains bounded best-effort cleanup.

## Decisions from `contractor-old`

The old implementation and tests are behavioral references, not an implicit
dependency. The new Runtime preserves:

- fsspec-compatible reader/writer operations and virtual `/` paths;
- rooted containment, hidden host paths and bounded path-aware globbing;
- read/list/grep pagination and honest truncation;
- exact-replace Edit semantics, CRLF preservation and non-UTF-8 refusal;
- overlay read-through, tombstones, empty directories, copy/move/type-conflict
  behavior, deterministic changed paths and unified diff;
- in-memory snapshot and overlay allocation isolation tests.

It deliberately does not preserve:

- implicit wrapping of every writable filesystem in an overlay;
- `isinstance(MemoryOverlayFileSystem)` checks inside Edit tools;
- the stock global `MemoryFileSystem` as an isolation boundary;
- arbitrary `__getattr__` delegation to a lower filesystem;
- readable in-root symlinks or silent traversal-as-not-found behavior;
- unbounded generic-backend glob fallback;
- full overlay state/patch JSON as a Server or Workflow contract;
- the specialized longest-output fork merge algorithm;
- GitLabFS, source call graphs, filetype formatting or interaction-tracking
  tools as part of this filesystem increment.

The old overlay had save/load and snapshot helpers but no crash-safe host
materialization contract. The journal/conflict rules above are new and must not
be inferred from the old `save()` patch implementation.

## Initial acceptance

1. Runtime with no workspace config registers and executes existing artifact-
   only templates unchanged.
2. Startup rejects escaping/overlapping/symlink roots and sends no host path to
   Control Plane, Agent Card, tool output, metrics or logs intended for Server.
3. Memory mounts import only complete bounded UTF-8 trees, and two allocations
   cannot observe each other's mutations.
4. Local, memory, overlay-local and overlay-memory expose the same read/Edit
   behavior wherever their declared writer contract overlaps.
5. Parent traversal, backslashes, URI/drive paths, symlinks, special files and
   replacement races cannot escape a configured mount.
6. Overlay writes never modify the lower before explicit materialization;
   changed paths, diff and rollback cover add/modify/delete/type changes.
7. Stale local lower data makes materialization fail before writes; injected
   operation failure or process restart restores from the journal before the
   Runtime can register.
8. Direct local write and host materialization require both trusted
   `hostWrite: true` and explicit model-visible Edit/materialize selection.
9. Read/edit/search/diff limits and cancellation leave a complete previous
   state and return honest truncation/failure metadata.
10. Existing `contractor-old` filesystem/Edit regression fixtures are ported
    selectively and pass against every compatible new backend without backend
    checks in tool code.

## Deliberately deferred

- Server-managed paths, Workspace API, database state or UI;
- Run/Workflow/Stage-selected roots, source Artifact auto-mounting or mount
  affinity in allocation protocol;
- dynamic/hot workspace reconfiguration after registration;
- Kubernetes volumes, containers, FUSE and remote arbitrary fsspec protocols;
- model-visible binary reads/writes, permissions, ownership, links or special
  files;
- shared writable roots across Runtime processes and distributed locking;
- general atomic multi-file host transactions beyond the recovery journal;
- automatic materialization on success/finalize/abort;
- overlay patch/tree Artifact export, fork/merge and three-way merge;
- source graph/parser migration and generic command execution.

## Invariants

1. Host roots, filesystem modes and write authority originate only in trusted
   Runtime startup configuration and never cross the private Runtime wire.
2. AgentTemplate selects tools, not host paths or backend implementations.
3. Every model-visible path is confined to a named virtual mount.
4. Memory and overlay state belongs to one allocation and is destroyed on
   release.
5. Overlay mutation never changes its lower without explicit materialization.
6. Edit tools depend on WorkspaceWriter, not a concrete fsspec class.
7. Diff/rollback/materialize are exposed only through an available
   WorkspaceChangeSet/Materializer.
8. `hostWrite: false` cannot be overridden by a tool argument, label,
   instruction, skill or allocation setting.
9. Runtime reports only ordinary exact Toolset/tool capabilities; Server owns
   no filesystem state.
10. Workspace tools do not turn Runtime's in-process privilege boundary into an
    OS sandbox claim.
