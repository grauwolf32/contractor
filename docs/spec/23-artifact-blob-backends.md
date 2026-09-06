# 23 — Server Artifact blob backends

Status: **Implemented — PostgreSQL/filesystem; S3 deferred**

Depends on: [03](03-artifact-plane.md), [06](06-server-ui-and-operations.md),
[18](18-run-and-workspace-lifecycle-controls.md), [19](19-audits.md).

## Scope and deployment choices

Artifact metadata, exact revisions, CAS, grants, pins, lineage and lifecycle
authority remain in PostgreSQL. The Server selects one payload backend at
startup for its installation, independently of Worker workspace storage.
All payloads use that backend; there is no size-based routing or automatic
selection based on whether a directory happens to be writable.

| Backend | Payload location | Local storage requirement |
| --- | --- | --- |
| `postgresql` | Existing PostgreSQL database; default | No writable blob directory, temporary upload file or PVC |
| `filesystem` | Explicit Server-owned path | Writable directory; ephemeral storage is allowed |
| `s3` | S3-compatible object storage, future implementation | Details deferred |

Kubernetes without PVC is an explicitly supported deployment. With
`postgresql`, requests use bounded memory and payload persistence belongs to
PostgreSQL. Alternatively, `filesystem` can use an `emptyDir`, including
memory-backed storage. Losing that volume may lose Artifact bytes even though
metadata remains in PostgreSQL; this is an accepted limitation of the simple
filesystem mode. A persistent disk/PVC is optional for operators who need it.
S3 is a planned third option; credentials, buckets, multipart operations and
consistency/recovery details are intentionally not designed in this increment.

## Startup configuration

| Flag | Environment | Default |
| --- | --- | --- |
| `--artifact-blob-backend` | `CONTRACTOR_ARTIFACT_BLOB_BACKEND` | `postgresql` |
| `--artifact-blob-path` | `CONTRACTOR_ARTIFACT_BLOB_PATH` | unset |

CLI overrides environment. Backend selection is not an Operations setting,
Workflow field, Runtime label or per-request option. Changing it requires a
restart. Unknown backends fail validation; `s3` returns an explicit unsupported
backend error until implemented, with no fallback to PostgreSQL or filesystem.

`filesystem` requires an absolute dedicated path and verifies actual bounded
create/write/publish/read/delete operations before serving. A read-only or
inaccessible path fails startup. `postgresql` rejects a supplied blob path and
performs no filesystem initialization for artifacts. Safe public errors never
expose storage paths. No filesystem probe decides the backend implicitly.

Persist the installation's backend kind in PostgreSQL and check it before
serving or catalog seeding. Existing installations with payloads are PostgreSQL
installations. A mismatch fails explicitly; restarting with another backend
does not copy or reinterpret existing blobs. Migration of populated stores and
online backend switching are deferred. A fresh installation may choose either
implemented backend. Filesystem path relocation requires operator-managed data
movement if existing bytes should survive; an empty replacement path is allowed
with the documented missing-content behavior below.

All Server replicas sharing a registry must use the same backend. PostgreSQL
supports replicas with no shared local volume. Filesystem replicas must see
the same underlying directory with compatible atomic publication semantics;
identical path strings in separate pods are insufficient. The initial tested
filesystem deployment uses one Server replica, and per-pod ephemeral volumes
must not be used as separate stores behind a shared multi-replica registry.

## Storage boundary and compatibility

Extract a small internal blob interface for storing complete immutable bytes,
opening them and deleting an exact physical object. The PostgreSQL adapter
continues to participate in registry transactions; the filesystem adapter
stores only a physical object key, digest and size in PostgreSQL. Preserve
database-enforced size/digest validation for inline PostgreSQL payloads and
validate filesystem payload size/digest in the Server. Do not duplicate file
payloads in a `bytea` column.

The public/private Artifact APIs, ArtifactRef, namespaces, exact revisions and
scope permissions are unchanged. Physical keys and backend settings are not
client-controlled. Metadata lists and forks need no payload read or copy. Audit
imports, proposal/evidence receipts, Skill forks, ordinary writes and Project
output publication must all use the same boundary; there is no direct-SQL
payload path that silently bypasses filesystem selection.

The generic payload ceiling is 64 MiB (67,108,864 bytes), including source ZIPs.
Package, model-context and workspace expansion limits remain independent.

## Memory and request handling

Receiving a file and retaining it are separate operations. The download/upload
path must not require a temporary local file in PostgreSQL mode: process bytes
through bounded memory or streaming interfaces into the chosen backend. A
planned Git importer in [24](24-git-artifacts.md) honors the same boundary:
in-memory snapshots, owner SSH-key Settings and Workflow/Project import UI
are separate V35 work, with no local checkout.

Check declared sizes early and enforce the actual byte count while reading,
including unknown Content-Length. Reject 64 MiB + 1 before publishing metadata.
Honor cancellation and avoid copies created solely to adapt one internal
interface to another. The first implementation may buffer one complete payload
for hash verification, but must not claim constant-memory streaming while
public/internal APIs still materialize `[]byte` or decoded buffers.

Bound simultaneous full-payload transfers per Server process; the initial
budget is four active operations, acquired before body buffering or blob reads.
Saturation fails promptly with a retryable capacity error instead of accepting
an unbounded queue of bodies. Nested trusted calls share the outer operation's
budget rather than reacquiring it. Use the same budget for public/private
transfers and trusted full-content reads/writes; metadata-only queries and
forks do not consume it. Account for driver/serialization buffers when sizing
container memory; four 64 MiB payloads are not a 256 MiB process-memory cap.

Kubernetes examples require no PVC. Read-only root filesystem with PostgreSQL
blob storage must work without a blob/tmp volume. Existing mutable managed
configuration is a separate concern: an example may mount a memory-backed
`emptyDir` at its configured managed root, explicitly documenting that such
configuration is also ephemeral. This is not an Artifact storage requirement.
Memory-backed volumes count towards container memory and are not durable; see
the [Kubernetes volume contract](https://kubernetes.io/docs/concepts/storage/volumes/#emptydir).

## Simple filesystem write and failure contract

1. Validate authority, request bounds and preconditions as early as possible.
2. Write complete bytes under a unique staging name inside the blob root while
   computing size/digest. Close and atomically publish a complete immutable
   object in the same filesystem. Never expose a staging file to readers.
3. In the authoritative short PostgreSQL transaction, recheck CAS and write
   fences, attach the exact object to the blob/version, and publish bindings.
   No transaction or pool connection is held during bulk file transfer.
4. If the registry transaction fails or its outcome is ambiguous, leave the
   candidate for offline cleanup. Never eagerly unlink a candidate after a lost
   commit acknowledgement. Unused deduplication candidates are best-effort
   removed only after definite commit. Existing CAS response-loss behavior
   remains unchanged.

Object keys are Server-generated and include a unique physical generation;
logical artifact names and URLs never become filesystem paths. Content digest
remains the deduplication identity, but a later blob with the same bytes must
not reuse a physical key that an earlier deletion may still target. Concurrent
writers attach only a complete verified winning object; losing files are
best-effort removed. Scope forks continue to reuse the existing immutable blob.
Use rooted filesystem operations and reject symlink/special-file escapes.

A process/pod/storage failure can leave an orphan or lose bytes. No distributed
transaction, durable upload-intent queue, automatic reconstruction, background
reconciliation service or power-loss durability guarantee is required here.
Atomic visibility during normal operation is still required. Missing or corrupt
referenced content produces an explicit non-success Artifact error, never an
empty successful response, silent revision replacement or automatic Git refetch.
Do not rewrite prior Run outcomes, finding ratings or historical reports; new
consumers see the storage failure. Re-upload creates a new revision under normal
CAS and cannot silently repair an old exact reference. Any targeted repair
protocol is deferred.

## Deletion and manual orphan cleanup

Registry retention checks and scope/version removal remain transactional and
reference-safe. For filesystem blobs, capture unreferenced exact object keys in
the transaction, commit logical deletion, then attempt file unlink. Unlink
failure is logged with a bounded reason and does not block completed Run,
Project or Audit deletion. This deliberately permits physical orphans;
PostgreSQL blob deletion retains its existing transactional semantics.

Only objects with no surviving version/pin/lineage dependency may be selected.
Coordinate opening a retained object with registry deletion so a live read
either obtains its file handle or returns an explicit missing-content error;
never deliver a knowingly partial or mismatched successful payload. Cleanup
always targets exact physical generations, not every file sharing a digest.

Provide an operator-only offline cleanup command with a dry-run default and an
explicit apply option. All writers/Server processes using that registry/root
must be stopped for apply; online cleanup is unsupported. Compare generated
object keys with the complete registry reference set, remove only unreferenced
objects and abandoned staging files, and report missing referenced objects
without deleting their metadata. Scan in bounded batches, avoid loading file
payloads and reject symlinks/path escapes. It is acceptable for interrupted
cleanup to require rerunning the command; no durable cleanup queue is needed.

## Verification and delivery

- Run the same Artifact contract tests against PostgreSQL and filesystem,
  including exact 64 MiB read/write, oversize refusal, CAS, write fences, scope
  isolation, forks, Audit/Skill retention and concurrent last-reference purge.
- Inject failure before file publication, before/after registry commit and
  before unlink; referenced successful bytes are exact or explicitly missing,
  and orphans are removable through the offline command.
- Demonstrate Kubernetes-compatible PostgreSQL startup with no PVC and a
  read-only root; restart the Server against the same DB and read prior bytes.
- Demonstrate filesystem with an ephemeral directory; after replacing it,
  missing payloads are reported truthfully and no automatic recovery is claimed.
- Validate backend mismatch, unsupported S3, unwritable root, traversal/symlink
  attacks, transfer saturation/cancellation and same-digest delete/recreate races.

Tasks V34-001 through V34-005 implement this increment. V34-006 separately fixes
the existing exporter/overlay limit mismatch after the generic 64 MiB increase;
it preserves the independent 16 MiB overlay budget. S3, Git import, automatic
store migration, online orphan sweeping and stronger crash durability remain
outside those tasks. Git import is specified separately in
[24](24-git-artifacts.md) and planned as V35-001 through V35-005.
