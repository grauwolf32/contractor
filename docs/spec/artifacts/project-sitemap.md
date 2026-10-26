# Project sitemap

Requirement prefix: `SMT` · [Specification index](../README.md)

## Purpose and scope

The sitemap owns shared, safe HTTP surface metadata in a Project: concrete
members, observed contexts, declarations, grouping, member annotations and
reversible suppression. It is research context. It does not establish Audit
coverage, physical request counts, retained HTTP exchanges or verified findings.

[ArtifactStore](artifact-store.md) owns scope authorization, immutable revisions,
CAS, transfer admission, retained history and exact input forks. [Projects](projects.md)
owns the owner boundary; [retention and deletion](retention-and-deletion.md)
owns fencing, scope purge and retention dependencies. A producer Run is provenance,
not the owner of Project metadata.

## Model

The ordinary `ProjectScope(project_id): sitemap/map` binding holds a strict
`project-sitemap@1` document with projection `sitemap-safe-projection@1` and
media type `application/vnd.contractor.project-sitemap+json`. Its groups contain
bounded concrete members, summaries, examples and declarations. The document
also contains sources, source-owned grouping rules, fixed member suppression
selections and bounded omission aggregates.

A group key is the safe normalized origin, HTTP method, concrete or justified
template route and explicitly configured structural query values. A member key
uses the concrete path and structural selection instead of the group route.
Keys use SHA-256 over length-prefixed UTF-8 components, so component boundaries
are unambiguous. Removed credential values never participate in these hashes.

Every contribution records trusted source kind, source mode, deployment and,
when known, producer Run, credential revision and session generation. Context
keys use the trusted engagement plus its local identity ID, or the ordinary
Run plus its local identity ID. Empty identity is a Run-local unknown context.
Matching identity labels across engagements or Runs do not establish equivalence.

A source records its accepted generation, exact source descriptor when available
and code-computed grouping policy digest. An automatic transport source can be
shared by sequential Runs; each member summary still retains the real producer
and context. A source registry entry is not a retention hold.

## Requirements

**SMT-R1.** The authenticated endpoint must derive Project ownership and the
producer's authority. Mutation JSON must not select owner, Project, producer,
source mode or observed-event authority. The public owner source is the fixed service-assigned `owner` identity.
Public owner commands may declare and
edit metadata but must not manufacture HTTP observations. Generic Project
Artifact writes must reject the sitemap namespace; only the trusted domain
reducer publishes it.

**SMT-R2.** A missing current binding means an empty map. An unavailable exact
revision must remain unavailable. Malformed, unsupported, duplicate-key,
case-aliased, unknown-member, omitted-required or null-required documents must
fail permanently and must never be replaced implicitly. The owning typed
contract and public schemas define the document and mutation fields.

**SMT-R3.** Safe ingress projection must run before persisted identities,
examples and request digests. It must remove URL userinfo, query values,
fragments and sensitive parameter names; headers and body values are not map
input. Known credential values, repeated percent encoding and opaque or
credential-bearing paths must not produce concrete members. Unrepresentable
observations create bounded reason aggregates without removed values or their
hashes. Safe paths preserve case, trailing slash and encoded reserved characters.
The HTTP URL and host grammar is the shared prepared-request grammar.

**SMT-R4.** Structural query selectors must have explicitly configured finite
allowed values. A name alone does not authorize retaining its incoming values.
Projection must still reject a known secret in an allowed value. Unknown or
conflicting selections must create an omission rather than a fabricated member.
Selector multiplicity and value order remain distinct. Free-form notes and tags
pass the same known-secret and sensitive-content checks; omitted metadata has
an explicit gap.

**SMT-R5.** Observations must be bounded sets of source, context, shape and
outcome summaries with minimum and maximum timestamps. A timeout or transport
error must not acquire a response status. Overflow must preserve accepted
members and report discarded membership or contexts without inventing a shared
principal. Representative examples must not serve as the member inventory.

**SMT-R6.** A manual edit, declaration, remove or restore must name its expected
map revision, with `absent` for conditional creation. Suppression and annotation
commands must name an exact bounded member selection. Repeated traffic must not
remove suppression. A grouping change must preserve retained member identity,
metadata and selection, and must conflict when a structural change would make
an existing suppression ambiguous. Unknown discarded membership must remain a
regrouping gap. Example selection may shrink deterministically within its bound;
retained concrete members and evidence must not be dropped to make room.

**SMT-R7.** A source replacement must compare its expected generation,
predecessor exact descriptor and policy digest, together with stable source and
deployment attribution. Creation requires absence. Code assigns the next
generation and computes the new policy digest. It must preserve independent
observations and reject stale source intent; retry must never refresh that
intent automatically. Exact concrete declarations precede templates; ambiguous
template matches remain concrete and explicitly ambiguous.

**SMT-R8.** The service must publish through the ordinary caller-owned
ArtifactStore transaction and CAS. Its keyed mutation must use the shared
mutation receipt owner in the same transaction. An append-compatible CAS race
may reapply the same batch or source transition only within LIM-SMT-8. A manual
revision conflict, source-precondition conflict and protected-history refusal
are permanent. Transaction-conflict and maintenance outcomes retain their
common classification and original operation deadline.

**SMT-R9.** Receipt replay must confirm the originally accepted exact ref,
even after its unprotected bytes were evicted. It must create no publication,
charge or hold and must never substitute current. A bounded safe selector policy
stored in that same receipt must make replay independent of later grouping
changes without hashing discarded secret values. The receipt's policy is not
another map or retained-history mechanism.

**SMT-R10.** The map must consume ordinary automatic per-binding ArtifactStore
history policy, admission, maintenance and shared errors. It must not register
an artifact family, own retention SQL or accounting, retain all predecessors or
impose a cumulative producer-Run publication quota. The generic RunScope lifetime
insert contract applies only to RunScope, not to a Project write with producer
Run provenance.

**SMT-R11.** Durable selected context and proof must use existing authorized
exact-reference retention or input-fork lifecycles, atomically against history
eviction. Reading metadata, a CAS predecessor and a provenance descriptor must
not pin a revision. A lost exact-selection race requires explicit refresh or
reselection; cached bytes cannot recreate the source or authorize latest.

**SMT-R12.** Project deletion must fence publication under the existing shared
Project admission lock. Deleting a producer Run must not erase self-contained
Project metadata. Project purge must remove the domain's receipts through the
shared bounded receipt helper before removing the Project; no independent
receipt journal, claim, deletion controller or evidence lifetime is introduced.

**SMT-R13.** Allocation access requires an explicit pinned `sitemap-access@1`
policy in the AgentTemplate and exact selected `sitemap@1` operations. The
policy fixes `sourceMode` (`blackbox` or `source-assisted`), an optional deployment
identity and explicitly authorized OpenAPI input names. Authority resolves each
name to its immutable original Run-creation input fork, never the Worker-editable
current binding. Advancing a working copy cannot declassify another source. Reader, editor and
importer authority derives from the selected operations. Only the policy's
explicit `collectObservations` setting grants the separate observation writer;
reader selection alone grants no collection. Project and owner come from the
allocation's durable Run. A request may not select another Project or relabel
its source mode. Manual editor mutations cannot select a Project artifact; exact source selection belongs to authorized import adapters. Unknown policy versions and operations fail validation.

**SMT-R14.** A blackbox view must be assembled from allowed contributions before
returning any document bytes. Source-assisted annotations, declarations, paths,
parameter names and grouping effects must be excluded. An OpenAPI contribution
requires an explicitly authorized original exact Run input revision; another Run's import is not a
grant. Allowed observations may recover their concrete route after excluded
templates are removed. If a structural selection cannot be reconstructed safely,
omit it with a gap. A mixed raw map must never be materialized into a blackbox Run.

**SMT-R15.** Runtime keeps one immutable encoded safe view per allocation,
bounded by LIM-SMT-12. Parsing and indexing run in joined threads. A cursor binds
the view identity, original exact revision and filters. Every page checks current
Server allocation, Project and source-mode authority; local bytes confer no
authority. Publication or source eviction does not change an existing view.
Refresh, absence, replacement and release invalidate old cursors with
`sitemap_cursor_expired`. There is no view TTL or implicit Artifact hold.

**SMT-R16.** A context snapshot must contain only the selected authorized safe
projection, original map revision, projection policy and authorized exact inputs.
The Server must atomically acquire ordinary exact-reference protection before
accepting it. Needed permitted original Run input bytes are frozen with ordinary
exact forks in the same transaction; foreign provenance alone grants no byte
read. Derived context has its own bytes and must not claim byte identity
with the full map. A lost eviction race returns exact-source unavailability;
Runtime memory and disk caches cannot revive the source. A snapshot does not
change an existing Audit baseline or grant arbitrary foreign-Run reads.

**SMT-R17.** Allocation fencing and release must join mutations, view construction
and materialization through the existing allocation gate. Clearing/replacing a
view joins its indexing work before freeing it. Ordinary HTTP/scanner operations
have no sitemap publication dependency; allocation composition may connect an
optional observation subscriber to a separate collector. Explicit sitemap tools
report their own operation failures. Automatic collection failure belongs to Run
owner status and deterministic completion, and never changes or repeats target
transport.

Explicit allocation tools return `status: completed` on success and `status: failed`
with a classified error on a definite API rejection, matching ordinary `tool@1`
completion. Transport/protocol bugs remain Python exceptions. Telemetry for these
tools records outcome metadata only. Known credentials, including short and
percent-encoded forms, are rejected before hashing or sending explicit tool
arguments. Import capability is advertised only when its format dispatcher is
installed. The private command wrappers carry the private API version; the
domain command schemas are generated from the public owning schema.

## Interfaces

The public owner operations are `GET /v1/projects/{projectId}/sitemap` and
`POST /v1/projects/{projectId}/sitemap/mutations` in the
[public OpenAPI contract](../../../api/openapi/contractor-public-v1.yaml).
The read accepts an optional exact revision; mutations require `Idempotency-Key`.
Commands are `declare`, `annotate`, `remove`, `restore` and `replace-source`.
The trusted service also accepts observation batches from an authenticated
producer context. No command accepts a replacement map document.

The CLI uses the generated public client:

```text
contractor project sitemap get PROJECT [--revision REVISION]
contractor project sitemap mutate PROJECT --file command.json --idempotency-key KEY
```

An owner can select the exact sitemap binding as an ordinary declared input
when creating a Project Run whose consumer policy permits the complete map.
A narrower source-mode consumer requires an authorized projection. The existing
input fork retains the selected source;
a live read alone creates no such dependency.

## Limits, failures and errors

Source, producer and context identifiers reuse the shared ResourceId grammar
(LIM-API-19); manual owner mutation cannot choose a transport source identity.
The numeric contract is LIM-SMT-1 through LIM-SMT-11 in the
[limits registry](../platform/limits.md). Common artifact history and maintenance
bounds remain LIM-ART-9 through LIM-ART-11; they are not copied into the domain.

FAIL-SMT-1 in the [failure registry](../platform/failure-model.md) owns mutation
outcomes. Invalid input uses `invalid_request`; stale edits/source intent use
`conflict`; changed keyed intent uses `idempotency_conflict`. Common history
outcomes remain `artifact_history_protected` and
`artifact_history_maintenance_pending`, including their existing `Retry-After`.
Storage transaction conflicts remain retryable `storage_transaction_conflict`.
Foreign Projects use owner-safe `not_found`; exact unavailable artifacts preserve
the existing public/private not-found mapping and distinct content-missing error.

## Security

The owner may read the complete safe map. Source-mode metadata remains trusted
provenance, including for manual notes and rules, for narrower consumer views.
Public commands cannot self-classify imported facts as observed traffic.
The projection is conservative: an opaque concrete path can be omitted even
when it was an ordinary application identifier. The explicit gap, and any
independent selected declaration, remain available; omission is not absence.
The HTTP tools area owns collection completeness and reopens this choice if
retaining such identifiers becomes necessary under a stronger classification
policy.

## Invariants

The reducer operates on a private copy and leaves accepted content unchanged on
failure. ArtifactStore owns atomic pointer/accounting/hold effects. Context IDs
and member IDs survive route regrouping; a changed structural selection creates
an explicitly distinct member and cannot silently widen a suppression.
