# Git artifacts

Git import creates an ordinary immutable ZIP Artifact. It is available in
Workflow input selection and the Project artifact library. The selected input
uses the returned exact revision; a Run never fetches the repository again.
The ZIP contains tracked regular files from one resolved commit. Its metadata
retains the repository, requested branch/tag, full commit and import time,
including after the artifact is forked into a Run input.

## Configuration

Run database migrations before starting the Server. Configure exact allowed
remote hosts and ports; an empty list disables imports:

```shell
contractor-server serve \
  --git-allowed-remote=github.com:443 \
  --git-allowed-remote=github.com:22 \
  --git-known-hosts-file=/etc/contractor/git_known_hosts \
  --artifact-blob-backend=postgresql
```

Equivalent environment variables are `CONTRACTOR_GIT_ALLOWED_REMOTES`
(comma-separated `host:port`) and `CONTRACTOR_GIT_KNOWN_HOSTS_FILE`.
TLS uses the system CA trust store. A scratch image needs a read-only CA bundle;
set `SSL_CERT_FILE` to its path when appropriate. Provision SSH `known_hosts`
through the operator's trusted configuration process. There is no interactive
host acceptance. Trust changes take effect on subsequent imports.

Each authenticated owner can add, replace or remove one unencrypted Ed25519 or
RSA SSH private key in Personal Settings (`/settings`), without Operations
access. The Server encrypts it with the existing credential master key. The
master-key file uses the existing owner-only, base64-encoded 32-byte format;
keep it stable across restarts. Settings returns only configured state, key
type, fingerprint and update time. An import already holding a signer can
finish after key replacement/removal. Later imports use the new state.

Anonymous HTTPS needs no owner key. SSH uses only that owner's configured key;
there is no SSH agent, password prompt, external credential helper, HTTP proxy
or redirect following. Allowlisted private network addresses are supported;
actual loopback, link-local, unspecified and multicast destinations are refused.
Use the explicit SSH URL for non-default ports. SCP syntax `git@host:repo.git`
keeps its relative path when normalized to `ssh://git@host:22/~/repo.git`.

## Storage, resources and recovery

The Server fetches Git objects and creates the ZIP in memory. Its production
image needs neither a Git executable nor writable Git, checkout, key or `/tmp`
directories. Use the existing [deployment without PVC](artifact-blob-storage.md#deployment-without-pvc)
with read-only CA/known_hosts/master-key mounts. The managed configuration
`emptyDir` is independent of artifact storage. PostgreSQL is the default blob
backend; filesystem requires an explicit writable blob path and can use an
ephemeral `emptyDir`. S3 remains a future backend with details deferred.

The operation deadline is 120 seconds, including publication. Set ingress,
reverse-proxy and client request timeouts above that bound (for example, 130
seconds) so the Server can return its bounded error. One Git import per Server
also occupies one of the four shared full-payload transfer slots, through the
metadata response. Capacity returns 503 immediately; imports are not queued.
These admission limits are per process, not a cluster-wide semaphore or RAM cap.

The reader accepts SHA-1 repositories, shallow fetch and supported pack v2/v3
objects. Receive bytes are capped at 128 MiB and cumulative decoded objects,
delta programs and delta results at 256 MiB. Each object is at most 64 MiB;
there are at most 50,000 objects, 50 delta levels and 10,000 advertised refs.
The snapshot permits at most 10,000 entries, 64 MiB expanded content, 4 MiB per
file and 512 UTF-8 bytes per path. The generated ZIP also has a 64 MiB cap.
Symlinks, submodules, LFS pointers, unsafe paths and an empty file tree fail the
whole import. Files retain their contents, use stable ordering/timestamps and
ordinary 0644 permissions. Private HTTPS tokens, arbitrary commit-ID fetches,
sparse checkout, history browsing, automatic refresh and Git writes are deferred.

Creation uses `If-None-Match: *`; replacement requires explicit consent and an
exact `If-Match` revision. Ownership and preconditions are checked before fetch
and again in the publication transaction. Cancelling, a timeout or a lost
response can race with a successful commit. Inspect artifact metadata before
retrying; the UI never retries a mutation automatically. Publication does not
leave a partial successful snapshot. Filesystem orphan cleanup and lost-file
responses follow the existing blob-storage contract; no automatic Git refetch
or recovery is provided.

## Release verification

Install the locked UI/Runtime dependencies and Chromium (`make ui-install
ui-browser-install`). Native Git is used only by fixtures. Podman and a real
PostgreSQL test database are required; release tests fail if they are absent.
Use an account allowed to create/drop isolated schemas. Run the gates
sequentially; concurrent suites in one database contend on its migration lock:

```shell
CONTRACTOR_TEST_DATABASE_URL=postgres://... GOFLAGS=-p=1 \
  make test-git-artifacts test-artifact-blob-backends
```

| Contract | Concrete verification |
| --- | --- |
| Real HTTPS, annotated tags, branch movement during fetch | `TestRealHTTPSPinnedCommitAndTags` |
| Owner SSH authentication, absolute/relative repository paths and changed host key | `TestRealSSHOwnerKeyAndStrictHostTrust` |
| Owner isolation, encryption purpose/generation, replacement/removal and startup verification | `TestGitKeyEnvelopeAuthenticatesOwnerPurposeAndGeneration`, `TestGitKeyPostgresOwnerIsolationReplacementAndStartupVerification` |
| Safe Settings response, owner and CSRF boundary | `TestGitKeyPublicAPIUsesAuthenticatedOwnerAndNeverReturnsSecrets` |
| URL/DNS allowlist, forbidden destinations, TLS and redirects | `TestRemotePolicy`, `TestHTTPSRejectsRedirectAndUntrustedTLS` and production container imports |
| Malformed/checksummed/truncated packs, object/delta/count/depth/advertisement limits | `TestPackBudgetsAndDelta`, `TestDeltaDepthAndAdvertisementLimits`, `TestDecodedPackCumulativeBudget` |
| Network cancellation, deterministic complete ZIP, unsupported files, entry/expanded/path/file limits | `TestNetworkCancellation`, `TestSnapshotContentAndDeterminism`, `TestSnapshotEntryAndExpandedLimits` |
| Both backends, SQL released during fetch, CAS, Project deletion race, transfer admission through response and cancellation cleanup | `TestGitImportPublicationAndAdmission` |
| Atomic source rollback, immutable version provenance, deduplicated bytes with distinct origins and exact Run input forks | `TestPostgresGitProvenanceRetainsExactRunInputAndRollsBack` |
| Production Server with read-only root and no Git/tmp/blob write path in PostgreSQL mode; filesystem publication | `TestGitArtifactsProductionContainers` plus `TestRealGitReadOnlyContainer` |
| Real memory-workspace Runtime consumes imported Project ZIP, edits/exports diff and retains exact source after branch/key change | Both production container cases |
| Near-cap ZIP with three concurrent ordinary uploads, second Git import rejection, Runtime source-analysis acceptance and key canaries absent from responses/archives/logs | Both production container cases |
| Personal Settings, explicit replacement CAS, exact Workflow selection, retained drafts, mobile cancel and focus restoration | `git-artifacts.test.tsx` and five `git-artifacts.spec.ts` scenarios; browser gate serves an isolated production UI build |
| Ordinary Artifact limits, storage loss, commit ambiguity, cleanup and public/private consumers | `make test-artifact-blob-backends` |

On Linux, Go 1.25.6 and PostgreSQL 17 (2026-09-06), the production-container
measurement imported a 66,082,811-byte ZIP with 16 incompressible files totalling
63 MiB plus one small tracked file. Three ordinary uploads each held 64 MiB
minus one byte concurrently. Kernel process peak RSS (`VmHWM`) in the final Git
gate was 881,980 KiB (about 861 MiB) for PostgreSQL and 744,052 KiB (about 727 MiB)
for filesystem.
The measurement includes actual Git object storage, ZIP buffers, HTTP handling
and database-driver copies, following the small end-to-end Workflow run.

These are observed process peaks for this fixture, not upper bounds for every
allowed pack or application workload. PostgreSQL's separate process memory and
filesystem tmpfs pages are additional. The existing deployment's 2 GiB Server
limit is a starting point for this measured workload; verify headroom against
your own object/delta distribution, concurrent payloads and other Server work.
