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
