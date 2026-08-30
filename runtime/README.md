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
