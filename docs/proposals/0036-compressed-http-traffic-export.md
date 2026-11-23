# Compressed export of complete HTTP traffic

Proposal 0036 · Status: **Deferred idea; not approved for implementation.**

On 2026-10-13 the user considered retaining every HTTP exchange in compressed
storage, with a downloadable ZIP and individual artifacts for exchanges
selected as evidence. The user then chose the smaller
[finding evidence scope](../spec/tools/http-caido.md): retain
only finding evidence in ordinary artifacts and inspect other proxied traffic
in Caido. This proposal preserves the complete-traffic export idea for a future
decision. It creates no implementation commitment or new runtime capability.

## User outcome

A user could download the HTTP traffic of a Run for offline analysis, while
the normal materials list showed only selected evidence. Each retained
exchange would contain the actual outgoing request and the received response
or an explicit failure outcome. A request would be mandatory; a response would
be optional. Redirects and retries would retain their order and identities.

The recent 128-exchange Runtime buffer would continue to serve tool reads and
finding selection. Its eviction would not delete already committed traffic
or evidence artifacts.

## Proposed shape

- Store bounded compressed parts through the existing
  [ArtifactStore](../spec/artifacts/artifact-store.md) and
  [blob boundary](../spec/artifacts/blob-storage.md). Scope, authorization,
  revisions, deletion and payload backend selection stay with that subsystem.
- Give each part an immutable identity and describe parts, exchange identities,
  ordering and completeness in a manifest. Part names and manifest refs must
  not become one visible materials row per HTTP call.
- Generate a downloadable ZIP from committed parts. JSON metadata and raw
  request/response files would allow offline inspection; the exact interchange
  format is a future contract decision.
- When an exchange is selected for a finding, publish an ordinary bounded
  exchange artifact and attach its exact ref using the existing evidence
  mechanism. The finding remains usable without opening the complete export.
- Show selected evidence and a traffic-download action in the UI. A full
  traffic browser or independent save-evidence tool is a separate decision.

## Guarantees that must be resolved before implementation

| Boundary | Required decision |
| --- | --- |
| Durability | A ZIP written only at Run completion does not preserve traffic when the Runtime crashes. Define when a prepared request becomes durable, whether sending waits for that commit, and when received response bytes are committed. |
| Completeness | Record sent, unknown, cancelled and incomplete outcomes accurately. A prepared request does not prove delivery; a crash between response receipt and its commit cannot be called a complete capture. |
| Publication failure | Define permanent rejection, bounded retry and lost-acknowledgement reconciliation through existing artifact identities and CAS. Never replay target traffic to repair storage. |
| Size and work | Bound bytes per exchange, compressed part, expanded part, manifest, upload and download, and the number of parts processed per operation. Each stored payload must fit the existing 64 MiB artifact ceiling; compression alone is not a byte bound. |
| Finalization | Decide who joins capture and publication before invocation completion, allocation fencing, cancellation and release. No archive writer may outlive its allocation. |
| Credentials | Apply the owning HTTP/Caido capture rules to previews, individual evidence and compressed bytes; an export must not bypass secret handling. |
| Lifecycle | Reuse ordinary scope retention, finding evidence pins and reference-safe purge. A selected evidence artifact must not depend on a mutable external Caido ID. |

## Alternatives and decision

- **Finding evidence plus Caido history:** selected for the current work. It
  covers durable evidence without introducing a second complete traffic store.
  Direct traffic appears in Caido only when routed through it.
- **An ordinary artifact for every request and response:** provides direct
  access, but adds collection noise and retention costs proportional to all
  traffic. It is outside the selected scope.
- **One final ZIP:** simple to produce, but insufficient for capture across
  Runtime failure and potentially larger than one artifact.
- **Compressed parts plus manifest:** the candidate preserved here. Its
  durability, publication and download contracts need further design.

## Reopening

Owner: the HTTP tools area, together with the ArtifactStore and UI owners for
their boundaries. Reopen only after explicit user direction and evidence that
Caido history plus selected finding evidence does not cover an actual offline
analysis or retention need. Define the guarantees and budgets above before
creating implementation tasks. The current Runtime and UI work does not
depend on this proposal.
