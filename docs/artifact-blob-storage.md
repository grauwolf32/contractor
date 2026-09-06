# Artifact blob storage

Artifact revisions, permissions, pins and lineage always live in PostgreSQL.
Select payload storage when starting `contractor-server serve`:

```shell
contractor-server serve --artifact-blob-backend=postgresql
contractor-server serve --artifact-blob-backend=filesystem --artifact-blob-path=/var/lib/contractor/blobs
```

Equivalent environment variables are `CONTRACTOR_ARTIFACT_BLOB_BACKEND` and
`CONTRACTOR_ARTIFACT_BLOB_PATH`. CLI flags win. PostgreSQL is the default and
does not accept a blob path or use local temporary upload files. Filesystem
requires a dedicated writable absolute path. S3 is reserved for a future
backend; selecting `s3` currently fails explicitly.

Run database migrations before startup. The installation records its selected
backend; changing flags does not migrate an existing store. Filesystem bytes
are not duplicated in PostgreSQL. Metadata-only queries and scope forks do not
copy payloads. Both adapters support payloads up to 64 MiB; Skills, Audit
packages and overlay state have independent smaller processing limits.

Kubernetes without PVC is supported. PostgreSQL mode retains payloads in the
database across Server replacement. Filesystem may use an ephemeral `emptyDir`;
volume loss then leaves metadata whose bytes are unavailable. Reads return
`artifact_content_missing`, not a successful empty file. Corruption returns
`artifact_content_corrupt`. There is no automatic Git refetch or repair.
Filesystem deployments with separate per-pod directories must use one Server
replica. Replicas require a genuinely shared underlying blob directory.

The Server admits four concurrent full-payload transfers. Saturation returns
503 with retryable `artifact_transfer_capacity`. Memory also includes request,
driver and serialization buffers; four slots are not a total process RAM cap.
PostgreSQL mode needs no local artifact scratch directory.

Filesystem publication makes a complete file visible before committing its
reference. Failed publication or unlink may leave unreferenced files. Logical
Run/Project/Audit deletion can complete even if physical unlink failed.

To clean orphan files, first stop every Server and other writer sharing this
registry and blob root. Preview the result:

```shell
contractor-server blobs cleanup --artifact-blob-path=/var/lib/contractor/blobs
```

The command reads `CONTRACTOR_DATABASE_URL`, or accepts `--database-url`. Dry-run
is the default and creates no probe or temporary files. To remove the reported
orphans and abandoned staging files while all writers remain stopped:

```shell
contractor-server blobs cleanup --artifact-blob-path=/var/lib/contractor/blobs --apply --offline
```

`--offline` acknowledges the shutdown requirement; it does not stop servers for
you. Online apply is unsupported. JSON output counts referenced, missing,
orphan, staging, removed and ignored entries. Missing referenced objects keep
their metadata. The command does not read payload bytes or delete unknown file
names. Interrupted cleanup can be rerun offline. No automatic reconciliation
service, persistent cleanup queue or populated-store migration is provided.

## Deployment without PVC

[PostgreSQL deployment](../deploy/artifact-blobs/postgresql.yaml) runs with a
read-only root and no Artifact or `/tmp` volume. `/managed` is an independent
64 MiB memory `emptyDir` for existing managed configuration publication; it is
not payload storage. Immutable operator configs are baked into `/configs`.
Prepare a static `CGO_ENABLED=0 go build -o contractor-server
./cmd/contractor-server` binary and your config tree in a build context, then
build with [Containerfile](../deploy/artifact-blobs/Containerfile).

Before deploying, provide `contractor-database` (key `url`) and
`contractor-server-credentials` (keys `local-auth.yaml`, `credential-master-key`,
`ca.crt`, `control-plane.crt`, `control-plane.key`). Generate local authentication
and PKI with the existing Server configuration tools; issue the Control Plane
certificate for the advertised private Service DNS name. Replace the browser
origin and image, and run `contractor-server migrate` against the same database
before starting the Deployment. These examples assume root-owned, mode 0400
projected secret files mounted individually with `subPath` so owner-only auth
and master-key paths are regular files, not Secret-volume symlinks. Restart
pods after secret changes; these mounts do not rotate in place. Capabilities are dropped and privilege escalation is
disabled. A non-root deployment needs secrets owned by its UID, as required by
the existing owner-only credential checks.

The [filesystem patch](../deploy/artifact-blobs/filesystem.patch.yaml) adds an
explicit `/blobs` disk-backed `emptyDir` and uses `Recreate` with one replica.
Render it with `kubectl kustomize deploy/artifact-blobs` and apply it to a
**fresh filesystem installation**, not a populated PostgreSQL store. The
PostgreSQL example is used directly without this filesystem kustomization. A container restart within the same pod can retain `emptyDir`; pod
replacement loses it. Disk `emptyDir` capacity counts against node ephemeral
storage. Using `medium: Memory` instead also charges blob files to container
memory. A PVC may replace this volume if persistence is desired, but is not
required. S3 is not implemented.

## Release verification

Run `CONTRACTOR_TEST_DATABASE_URL=... make test-artifact-blob-backends` with
Podman and the locked Runtime environment available. Tests use isolated
PostgreSQL schemas and temporary containers/images; they do not touch demo data.
The container test builds the real static Server in a scratch image, disables
Podman's implicit writable tmpfs mounts, and uses only read-only bootstrap and
secret files plus `/managed` (and `/blobs` for filesystem).

| Contract | Verification |
| --- | --- |
| 64 MiB exact bytes, read-only root, no PVC, PostgreSQL restart | `TestArtifactBlobBackendsContainers/postgresql` uploads, verifies SHA-256, replaces the container and reads the exact revision |
| Real public/private API and backend propagation | Both container cases run the Python Runtime's Artifact copy workflow over allocation-scoped mTLS, then read the exact public output |
| Ephemeral volume loss | Filesystem container replacement returns `artifact_content_missing` while metadata remains readable |
| Four-transfer saturation and cancellation | Four incomplete 64 MiB HTTP uploads hold all slots; another upload or output download gets 503, cancel frees a slot, remaining uploads commit; Run metadata stays available |
| Oversize, integrity, generation keys, symlink escape, startup configuration | Adapter/config tests plus container Content-Length rejection |
| CAS, forks, outputs, Run/Project/Skill retention | Shared PostgreSQL integration cases run with both backend contexts |
| Audit direct verification/evidence | `internal/findingintake` integration suite runs with both backend contexts |
| Commit/rollback/ambiguous outcomes and failed unlink | `TestFilesystemRegistryAndCommitCleanup`, `TestFilesystemLostWriteAcknowledgementKeepsCommittedBytes`, `TestFilesystemFailedUnlinkLeavesCleanableOrphan`, PostgreSQL commit-effect tests |
| Offline cleanup and missing references | `TestFilesystemOfflineCleanupPreservesReferences` verifies dry-run, apply, repeated apply and symlink refusal |

Measured on Linux with PostgreSQL 17, Go 1.25.6 and the real Server binary
(2026-09-06): process peak RSS (`/proc/<pid>/status`, `VmHWM`) reached 1,065,884 KiB
(~1,041 MiB) for PostgreSQL and 560,860 KiB (~548 MiB) for filesystem
in the final gate run. Each case
includes an exact-size upload/download followed by four nearly complete
64 MiB uploads, cancellation of one and completion of three. Payloads are
identical to exercise deduplication. These are observed process peaks, not
memory guarantees: Go body growth, GC and PostgreSQL driver buffers contribute;
PostgreSQL's separate process memory and filesystem tmpfs pages are additional.
The example's 2 GiB limit leaves headroom for this workload; measure your own
concurrency, payload diversity, managed configurations and other Server work.
