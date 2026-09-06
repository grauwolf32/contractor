# 24 — Git repositories as Artifact inputs

Status: **Implemented and verified (V35-001 through V35-005)**

Depends on: [03](03-artifact-plane.md), [06](06-server-ui-and-operations.md),
[17](17-projects-and-queue.md), [23](23-artifact-blob-backends.md).

## User flow and scope

An authenticated user can import a Git repository when selecting a Workflow
input or adding a Project artifact. The dialog accepts a repository URL, an
optional branch/tag and an ordinary Artifact namespace/name. Settings contains
one optional SSH private key per owner for private repositories. Anonymous
HTTPS repositories work without a key; SSH uses the importing owner's key.
Private HTTPS tokens, multiple named keys, repository browsing and Git writes
are deferred.

Personal Settings lives at `/settings`, linked beside the signed-in user.
It is available without the Operations capability; Operations Settings links
to it. The same Git-import dialog is used by standalone/Project Workflow
inputs and the Project artifact library. Existing bindings require reviewing
and confirming their exact revision before replacement. Closing the dialog
cancels its request and restores focus without closing the parent Run form.
Successful imports select only the requested input and show the recorded
commit; metadata detail pages also display Git provenance.

Import materializes a source ZIP and publishes an ordinary exact ArtifactRef.
A Workflow receives that ref through existing input selection/forking; it does
not clone on execution. Project import writes directly to the owned
ProjectScope. Standalone Workflow input import writes to the owner's UserScope;
Project-bound input selection uses that ProjectScope. Importing never starts a
Run or changes a Workflow definition. Closing a completed import dialog does
not delete its artifact. There is no scheduled refresh or automatic refetch
when filesystem blob content is lost.

Existing ASCII Artifact-name validation applies, including its length and
reserved-namespace rules; names with spaces are rejected. Repository filenames
retain the existing safe UTF-8 archive-path rules. A suggested Artifact name is
editable and validated before import, never silently rewritten on the Server.
Existing bindings require explicit CAS replacement. The UI assigns the exact
returned ref to the selected input slot and preserves every other input.

## Snapshot and provenance

The first slice accepts anonymous `https://host/path`,
`ssh://user@host[:port]/path`, and conventional `git@host:path` URLs. Normalize
the latter to an SSH URL, preserving its home-relative path as `/~/path`;
an explicitly absolute SCP path remains absolute. Reject passwords, URL query/fragment credentials,
local paths, `file://`, `git://`, remote helpers and unsupported schemes.
Transport credentials never enter the normalized repository URL.

An omitted ref resolves the remote default branch. An explicit ref selects a
branch or tag; fully qualified `refs/heads/...` and `refs/tags/...` are supported.
Ambiguous short names fail with a safe error. Arbitrary commit-ID fetches are
deferred; every successful import nevertheless records its full resolved
commit object ID. Resolve once and archive exactly that commit, including when
the named branch advances during the operation. Unsupported repository/object
formats fail explicitly; never reinterpret an object ID.

The archive contains the complete tracked regular-file tree at the selected
commit, in stable path order, with normalized timestamps and ordinary file
modes. It excludes the Git object database and transport configuration. There
is no working tree, hook, filter, script or build execution. Do not apply host
Git configuration, `.gitignore` rules to already tracked files, or host-specific
source-directory bundling rules. Reject unsafe paths, symlink entries,
submodules and recognizable Git LFS pointer entries with explicit unsupported
content errors; do not silently omit them or claim their targets were fetched.
LFS object download, recursive submodules and sparse/subdirectory imports are
outside the first slice. Empty source trees are rejected.

Persist an immutable Git-source record attached to the Artifact **version**,
not to the mutable name or content digest alone:

- normalized repository URL;
- requested ref (null for default branch);
- full resolved commit object ID;
- Server import timestamp.

Content deduplication may share bytes between imports with different origins;
it must not merge their provenance. Creating the Artifact revision and its
Git-source record is one registry transaction. Existing forks reuse the exact
version or preserve an explicit origin lineage edge; subsequent imports never
rewrite earlier provenance. Purging the final retained version can remove its
source record under the existing lifecycle rules.

Expose optional `gitSource` on authorized Artifact metadata/version responses;
absence means ordinary upload. Preserve existing byte and list contracts.
Authorized scripts can trace a Run's exact input through existing forks to the
repository and commit, even after branch movement or key deletion. Repository
URLs can disclose private project names, so source metadata follows Artifact
ownership checks and is not emitted in global logs or metrics. This adds source
provenance, not a new finding/verification attribution model.

## Settings and SSH trust

Settings offers replace/remove for one owner-scoped Git SSH key. Accept
unencrypted OpenSSH/PEM private keys supported by the chosen implementation
(at least Ed25519 and RSA). Bound key input to 32 KiB and validate it before
replacing the previous value. Encrypted/passphrase-protected keys return an
explicit unsupported-format response in this slice; never prompt a background
process for a passphrase.

Reuse the Server credential master-key mechanism from [06]. Store the key
only as authenticated ciphertext in PostgreSQL, with authenticated owner,
purpose and key-generation identity. Include Git-key rows in startup master-key
requirements and integrity checks. Do not reuse LLM credential records or send
the key in allocation RuntimeSettings. No private-key readback API, browser
storage, Artifact payload, Workflow field, log or telemetry field may contain
it. Responses expose only configured state, public-key fingerprint/type and
update time. A successful replacement clears the UI secret input.

An admitted import captures one key generation. Concurrent replacement/removal
affects subsequent imports; an already admitted import may finish with its
captured key. No long SQL lock is held during network transfer. Deleting a key
does not invalidate existing ZIP artifacts or provenance. No automatic
fallback to another owner, host SSH agent or machine key is allowed.

SSH host verification uses an operator-managed read-only `known_hosts` file,
selected with `--git-known-hosts-file` (environment
`CONTRACTOR_GIT_KNOWN_HOSTS_FILE`). Unknown or changed host keys fail closed;
there is no automatic trust-on-first-use prompt. Missing configuration makes
SSH import unavailable, while anonymous HTTPS can still work. This optional
configuration mount is compatible with a read-only root and requires no PVC.

## Network and resource limits

Outbound Git targets must match an explicit operator allowlist of exact
`host:port` pairs, configured by repeated `--git-allowed-remote` flags or a
comma-separated `CONTRACTOR_GIT_ALLOWED_REMOTES`. CLI overrides environment.
ServerConfig YAML also accepts `spec.gitAllowedRemotes` and
`spec.gitKnownHostsFile` (relative file paths resolve beside the YAML).
Precedence is defaults, YAML, environment, CLI; a CLI allowlist replaces the
lower-priority list.
The default empty allowlist disables Git network import without disabling the
Server or existing artifacts. HTTPS uses normal CA verification, SSH uses the
configured host-key verifier. Disable HTTP redirects in the first slice.
Resolve and validate the destination used for each actual connection; disallow
loopback, link-local, unspecified and multicast addresses. Corporate private
addresses are permitted only through the explicitly allowlisted hostname/port.
Tests may opt into isolated loopback fixtures through a test-only injection;
there is no production disable-verification flag.

Use an in-process Git reader with in-memory object storage and bounded decode;
no `git` subprocess, temporary checkout, pack file, SSH key file, shell or
credential helper is required. V35-002 must prove those properties against a
real repository before choosing the dependency. Never silently fall back to
local disk. Read-only operator trust/CA files are configuration, not scratch
storage. An inability to meet the bounds blocks that task's completion.

Initial per-process admission is **one active Git import**, also holding one
of the existing four Artifact transfer slots from before fetch until
publication/response. Reject saturation promptly without a queued request or
buffered clone. Nested Artifact operations share that slot. Other ordinary
Artifact transfers can use the remaining capacity. Fetching and ZIP creation
hold no database connection or authoritative mutation lock.

| Limit | First slice |
| --- | --- |
| Import request JSON | 64 KiB |
| SSH private-key input | 32 KiB |
| Whole import deadline, including resolution/fetch/ZIP/publication | 120 seconds |
| Cumulative received Git protocol/pack bytes | 128 MiB |
| Cumulative decoded Git object bytes, including delta results | 256 MiB |
| Published ZIP payload | 64 MiB |
| Source ZIP entries / expanded bytes / one file | 10,000 / 64 MiB / 4 MiB |

The last row preserves current source-analysis budgets rather than assuming
that a 64 MiB transport artifact is always consumable as a workspace. Validate
existing path and archive rules before publication. Bound ref advertisements,
object counts, delta depth and parser allocations as well as final output;
V35-002 records concrete decoder caps and adversarial evidence in this document.
Cancel network and decoding on request cancellation/deadline and release both
admission slots. Report actual process peak memory, including object storage,
ZIP buffers, driver copies and GC; these limits are not a total RAM guarantee.

Use a shallow selected-ref fetch where supported. If the remote cannot provide
a bounded selected-ref snapshot, fail explicitly instead of fetching unlimited
history. Large Git history can therefore hit an import budget even when the
final tree is small; return the relevant limit without publishing a partial
artifact.

### Reader implementation and evidence (V35-002)

The reader uses `go-git/v5 v5.19.2` protocol, pack-scanner and delta primitives.
It does not use `Clone`, an on-disk repository, the high-level pack parser or
host Git configuration. The scanner exposes declared object sizes before
inflation; the importer checks them before allocating fixed buffers. Delta
base/result sizes are checked before patching. SHA-1 pack checksums and decoded
object identities are verified; SHA-256 repositories are rejected explicitly.
The client requires shallow negotiation and requests depth one for the chosen
advertised ref. Annotated tags are peeled from fetched objects to a commit.

Additional parser caps are 4 MiB per advertisement/ACK negotiation, 10,000
advertised references including peeled entries, 50,000 packed objects,
64 MiB per decoded object or delta result, delta depth 50, and annotated-tag
depth eight. Decoded accounting includes delta programs and intermediate
results, even after their buffers become reclaimable. All traversed tree
entries count toward 10,000, including directories; paths are at most 512 UTF-8
bytes. Git internals, special modes, control characters and paths incompatible
with Runtime source validation are refused. The ZIP uses sorted regular-file
paths, mode 0644 and a fixed 1980 timestamp.

On 2026-09-06, `go test -race -tags=integration -count=1 ./internal/gitimport`
passed with native Git fixture servers over verified HTTPS and authenticated
SSH. The test advances the branch between advertisement and fetch and checks
the original commit, annotated tags, deterministic ZIP bytes, wrong owner and
changed host keys. Adversarial tests cover transport refusal, redirects, TLS,
advertisement/object/delta caps, malformed packs, unsafe content and cancellation.
A static probe also completed both imports in an unprivileged Podman container
with `--read-only --read-only-tmpfs=false`, no writable checkout or temporary
directory, and only read-only fixture mounts. The 13-file fixture contains about
1.6 MiB expanded content and yields a 5,747-byte ZIP; the probe's measured peak
RSS was 17,276 KiB. This small transport fixture is not a maximum-size Server
RAM estimate: publication/driver copies and near-budget imports remain part
of the V35-005 release measurements.

## Public API and publication

These authenticated public routes use existing browser CSRF protection
on mutations and owner/Project authorization:

| Route | Contract |
| --- | --- |
| `GET /v1/settings/git-key` | Configured state and non-secret key metadata |
| `PUT /v1/settings/git-key` | Validate and replace the owner's key; body `privateKey` |
| `DELETE /v1/settings/git-key` | Remove the owner's configured key |
| `POST /v1/artifacts/{namespace}/{name}/git-import` | Import into UserScope |
| `POST /v1/projects/{projectId}/artifacts/{namespace}/{name}/git-import` | Import into owned ProjectScope |

Import JSON contains `repositoryUrl` and optional `ref`. No request can choose
another owner, credential, backend or output filesystem path. Use existing
Artifact create/CAS headers (`If-None-Match: *` or exact `If-Match`), validate
authority/preconditions before fetch and recheck at publication. Return 201 for
create or 200 for replacement with the normal write fields plus `gitSource`.
No new durable job, broker, progress stream or resumable clone is required;
the dialog shows a cancellable pending request. Align Server/client/proxy
timeouts with the 120-second operation bound.

After a complete bounded ZIP exists, prepare its selected blob backend before
taking the short registry transaction; commit its revision and provenance
atomically while rechecking ownership, Project lifecycle and CAS. Apply V34's
filesystem orphan/ambiguous-commit rules. A rejected or interrupted import
cannot publish partial content. If cancellation races with a successful commit,
the complete artifact may remain; a lost response is resolved through metadata
inspection, not an automatic mutation retry. Duplicate create/CAS retries do
not advance the binding twice.

Use bounded public error codes distinguishing invalid URL/ref/key, unavailable
key/trust configuration, refused remote/host identity, repository/ref access,
unsupported tree content, exceeded import budgets, timeout and capacity.
Preserve existing ownership-not-found, CAS and blob error semantics. Do not
return remote protocol diagnostics, credential content, local paths or URLs
containing authentication material.

## Delivery and verification

1. **V35-001:** SSH-key storage, Settings API and strict public/startup contracts.
2. **V35-002:** Bounded in-memory Git transport and deterministic snapshot ZIP.
3. **V35-003:** User/Project import API, atomic provenance and Artifact integration.
4. **V35-004:** Settings key editor and shared Workflow/Project import dialog.
5. **V35-005:** Real Git/SSH/HTTPS, PostgreSQL, container and browser release gate.

The gate must demonstrate public HTTPS and private SSH imports, key replacement
and removal, unknown/changed host keys, forbidden destinations/redirects,
branch movement and exact commit provenance, CAS races, malformed/oversized
packs, cancellation and saturated capacity. Verify selected ZIPs with the
existing Runtime source consumers. Test the actual Server with read-only root,
PostgreSQL blobs and no writable Git/blob/tmp directory or PVC, then repeat
Artifact publication with filesystem storage. Measure memory and verify that
key/canary material appears nowhere in responses, artifacts or captured logs.

The reproducible gate, deployment instructions, failure matrix and measured
memory are documented in [Git artifacts](../git-artifacts.md).

Git provenance is stored in `artifact_git_sources`, keyed by `artifact_versions.version_id`;
metadata queries join it without reading blobs. The ordinary exact input fork
reuses that version. The import service checks preconditions before fetch, then
locks owned Project lifecycle and publishes the revision/source together in a
short transaction. Git and shared transfer admission remain held through the
metadata response. Failure responses never include remote diagnostics, and
interrupted responses instruct clients to inspect metadata before retrying.
