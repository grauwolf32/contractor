# Project sitemap

Requirement prefix: `SMT` · [Specification index](../README.md)

## Purpose and scope

The sitemap owns shared HTTP surface metadata in a Project: concrete
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

A group key is the normalized origin, HTTP method, concrete or justified
template route and explicitly configured structural query values. A member key
uses the concrete path and structural selection instead of the group route.
Keys use SHA-256 over length-prefixed UTF-8 components, so component boundaries
are unambiguous. URL userinfo, fragments and nonstructural query values do not
participate in these hashes because they are outside the metadata model.

Every contribution records trusted source kind, deployment and,
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
producer's authority. Mutation JSON must not select owner, Project, producer or
observed-event authority. The public owner source is the fixed service-assigned
`owner` identity. Public owner commands may declare and edit metadata but must not manufacture HTTP observations. Generic Project
Artifact writes must reject the sitemap namespace; only the trusted domain
reducer publishes it.

**SMT-R2.** A missing current binding means an empty map. An unavailable exact
revision must remain unavailable. Malformed, unsupported, duplicate-key,
case-aliased, unknown-member, omitted-required or null-required documents must
fail permanently and must never be replaced implicitly. The owning typed
contract and public schemas define the document and mutation fields.

**SMT-R3.** Ingress normalization must run before persisted identities,
examples and request digests. The HTTP URL and host grammar is the shared
prepared-request grammar. Normalize origins and preserve concrete path case,
trailing slash and encoded reserved characters. URL userinfo, fragments and
nonstructural query values are outside the map metadata model; headers and body
values are not map input. Valid host labels, paths and parameter names must not
be rejected or omitted because they contain credential-like words, look opaque,
use percent encoding or match known credential strings. The sitemap must not
receive matching-secret lists or implement secret detection. Syntax and byte
bounds remain mandatory; unrepresentable structural selections use bounded
reason aggregates without fabricated concrete identities.

**SMT-R4.** Structural query selectors must have explicitly configured finite
allowed values. A name alone does not authorize retaining its incoming values.
Unknown or conflicting selections must create an omission rather than a
fabricated member. Selector multiplicity and value order remain distinct.
Allowed structural values, free-form notes and tags must retain valid content
within their syntax and byte bounds, without keyword, opaque-token or
known-secret filtering.

Rationale (user decision, 2026-10-27): sitemap metadata normalization owns
structure and bounded representation. Secret matching at this boundary discarded
useful paths, names and descriptions and required duplicate matching logic and
transient secret transport. Preserve the existing `sitemap-safe-projection@1`
identifier; it names this metadata projection, not a content-redaction guarantee.
General credential handling and telemetry redaction remain with their existing
owners. The sitemap area owns this separation; reconsider it only for a concrete,
explicitly requested metadata-disclosure policy.

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
Manual declarations and observations use the same concrete member identity and
route/structural-selector placement, regardless of arrival order. A manual
declaration retains its concrete path and structural selection separately from
the group's template route, so its provenance survives source replacement and
regrouping. Annotation and suppression selections continue to name that same
member.

**SMT-R7.** A source replacement must compare its expected generation,
predecessor exact descriptor and policy digest, together with stable source and
deployment attribution. Creation requires absence. Code assigns the next
generation and computes the new policy digest. It must preserve independent
observations and reject stale source intent; retry must never refresh that
intent automatically. Exact concrete source rules precede templates; ambiguous template matches
remain concrete and explicitly ambiguous both on ingress and when a source
replacement regroups already retained members. Manual declaration metadata does
not override a source's grouping rule: it identifies a concrete member while
observations and declarations share the same placement policy.

**SMT-R8.** The service must publish through the ordinary caller-owned
ArtifactStore transaction and CAS. Its keyed mutation must use the shared
mutation receipt owner in the same transaction. An append-compatible CAS race
may reapply the same batch or source transition only within LIM-SMT-8. A manual
revision conflict, source-precondition conflict and protected-history refusal
are permanent. Transaction-conflict and maintenance outcomes retain their
common classification and original operation deadline.

**SMT-R9.** Receipt replay must confirm the originally accepted exact ref,
even after its unprotected bytes were evicted. It must create no publication,
charge or hold and must never substitute current. A bounded selector policy
stored in that same receipt must make replay independent of later grouping
changes without hashing excluded URL components. It must preserve structural
values for declarations and observations, including selector multiplicity and
order, and unconditional omission under conflicting policies even when a
selector is absent. Changed accepted structural intent must conflict on replay.
The receipt's policy is not another map or retained-history mechanism.

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
policy supplies an optional deployment identity, explicit `collectObservations`
permission and optional `snapshotInputs` names for retaining original Run input
files in a context snapshot. Reader, editor, importer and snapshot authority
derive from the selected operations. Reader selection alone grants no collection.
Project and owner come from the allocation's durable Run. Every request checks
current allocation, Project and operation authority; a request cannot select
another Project or manufacture producer identity. Manual editor mutations cannot
select a Project artifact; exact source selection belongs to authorized import
adapters. Unknown policy versions, operations and obsolete members fail strict
validation. Access checks return no input descriptors or view policy digest and
perform no input-lineage or input-payload lookup.

**SMT-R14.** Every permitted sitemap reader sees the same complete Project map
at a selected revision, with the same grouping and contribution provenance.
Missing or incomplete source files do not hide accumulated route metadata.
There is no source-mode classification, consumer-specific source filtering or
regrouping. Observed traffic and declarations remain distinct: metadata from
source code, OpenAPI, manual edits or another Run does not establish current
reachability, a current HTTP response, complete coverage or verified proof.
An owner's explicit ordinary Run input selection also supplies the complete
selected map. Reading metadata through either path grants no permission to read
arbitrary referenced artifact bytes.

Rationale (user decision, 2026-10-28): a shared Project map accumulates useful
knowledge from every available source; absent code is missing input, not a
reason to conceal routes already learned elsewhere. The earlier dedicated-tool
blackbox/source-assisted split added filtering, identities and repeated input
resolution without a concrete consumer requiring that distinction. Project,
operation and exact-file authorization provide the required boundaries. The
sitemap area owns this decision; reconsider a separate context restriction only
for an explicitly requested consumer, such as a deliberately blind experiment,
with a defined purpose and enforcement boundary. It is not a default sitemap
mode.

**SMT-R15.** Runtime keeps one immutable encoded authorized view per allocation,
bounded by LIM-SMT-12. Parsing and indexing run in joined threads. A cursor binds
the view identity, original exact revision and filters. Every page checks current
Server allocation, Project and the requested operation permission; local bytes
confer no authority. Neither a view policy digest nor a policy-copy precondition
is part of the page contract. Publication or source eviction does not change an
existing view.
Refresh, absence, replacement and release invalidate old cursors with
`sitemap_cursor_expired`. There is no view TTL or implicit Artifact hold.
The existing `list_sitemap_entries` operation accepts `scope="entries"` by
default and `scope="metadata"` for independently paged sources, rules and gaps,
including maps with no entries. Metadata uses the same view and page bounds,
checks current authority and binds its scope into cursors; entry filters are
invalid in metadata scope. Entry reads contain only their own parts and removal
markers, without repeating global sources, rules or gaps. Complete Source
metadata remains discoverable after its rules are cleared so a later editor can
supply the exact source precondition.

**SMT-R16.** A context snapshot selects bounded group/member IDs from an exact
map revision and contains that selected metadata, the original revision and any
retained exact input descriptors. It contains no allocation policy copy or view
policy digest, and selection has no expected-policy precondition. The Server
checks current snapshot permission and rereads the exact map inside the snapshot
transaction; Runtime view bytes are not a source for materialization.

Only in that transaction, the at-most-eight configured `snapshotInputs` names
resolve through a bounded batch to immutable original Run-creation input forks,
never Worker-editable current bindings. Query work is independent of the
containing Run's artifact count. Only needed permitted original input bytes are
frozen with ordinary exact forks. A missing optional referenced file preserves
selected metadata and records the existing `source-unavailable` gap; with
`requireReferencedInputs: true`, a missing needed file fails the snapshot.
Unneeded missing input names do not invalidate metadata reads or snapshots.
Foreign provenance alone grants no byte read.

The derived frozen context, required forks and ordinary exact-reference
protection commit atomically or roll back together. A lost map-revision eviction
race returns exact-source unavailability; Runtime memory and disk caches cannot
revive the source or substitute latest. Derived context has its own bytes and
must not claim byte identity with the full map. A snapshot preserves the selected
members' concrete manual declarations even when their members belong to template
groups; companion declaration groups add no unselected members or unrelated
declarations. A snapshot does not change an existing Audit baseline or grant
arbitrary foreign-Run reads.

**SMT-R17.** Allocation fencing and release must join mutations, view construction
and materialization through the existing allocation gate. Clearing/replacing a
view joins its indexing work before freeing it. Ordinary HTTP/scanner operations
have no sitemap publication dependency; allocation composition may connect an
optional observation subscriber to a separate collector. Explicit sitemap tools
report their own operation failures. Automatic collection is best effort. Its failures produce technical logs,
without changing target transport, Worker results, Run state or Runtime reports.

Explicit allocation tools return `status: completed` on success and `status: failed`
with a classified error on a definite API rejection, matching ordinary `tool@1`
completion. Transport/protocol bugs remain Python exceptions. Telemetry for these
tools records outcome metadata only under the existing general telemetry
redaction policy. Explicit tool arguments use the syntax and bounds of SMT-R3
and SMT-R4, without sitemap-specific secret matching. Import capability is
advertised only when its format dispatcher is installed. The private command wrappers carry the private API version; the
domain command schemas are generated from the public owning schema.

**SMT-R18.** Allocation composition connects selected HTTP capture to an optional
transport-neutral observation sink. It receives minimal metadata for actual
attempts, including each physical retry/redirect and negative outcome, before
evidence-buffer admission/eviction. It never follows a Location. Submission does
not await network or storage I/O. Subscriber errors cannot replace a target
result; caller cancellation is preserved and joined. HTTP tools own no collector
flush, retry or cleanup. An invalid target response status is represented as a
transport-error observation with status zero, while the HTTP tool retains its
original protocol error. It must not invalidate otherwise valid observations
in the same batch.

**SMT-R19.** Runtime retains only a bounded private in-memory batch of attempt
metadata. Observation frames and private collection requests contain no
matching-secret list; no separate secret-count limit applies. No headers, bodies,
persistent queue, URL identity or receipt digest are constructed there. One
background private batch request invokes the existing Go normalizer before
sitemap identities, receipt hashing and Artifact publication; the existing
reducer owns coalescing. Server code assigns Project,
source, Run and deployment from the allocation; Runtime's session generation
distinguishes unknown sessions without claiming an authenticated principal.

**SMT-R20.** Pending and in-flight batches obey LIM-SMT-14; count, byte and time
boundaries trigger publication. Overflow may drop metadata with bounded technical
logging. An ambiguous publication repeats only its identical input/key within
one original three-request/15-second allowance. Definite protected-history
refusal is permanent. A Server-exhausted transaction conflict receives no new
Runtime transaction retry allowance. Refusal or budget exhaustion disables
further automatic publication for the allocation; later reads and drain cannot
renew that allowance. `Retry-After` applies only inside the remaining budget.
A fresh live view attempts the caller's local flush and then reads committed
data, including when automatic publication failed. Existing paged views remain
immutable. Explicit read authorization/storage errors retain their own contract.
Lock acquisition and the explicit read each have their own 15-second allowance;
the automatic checkpoint retains its separate existing flush budget. Exhausting
that checkpoint must not consume the explicit read allowance. Caller cancellation
still joins publication and view construction before propagating.

**SMT-R21.** Composition attempts a bounded infrastructure checkpoint before
returning the Worker result, while allocation writes are still permitted.
Server finalization fences writes before calling Runtime finalize, so finalize
is never the first flush boundary. Composition closes HTTP producers before
joining collector work at finalize, abort, fence, release and prepare rollback.
No publisher survives release. Automatic failures, exhausted publication,
overflow and crash may lose the bounded private batch; they produce no Run or
Runtime-report collection state, generic map-completeness marker or required
completion policy. Explicit imports, mutations and snapshots report their own
operation failures. The HTTP-tools area owns this best-effort limitation;
reopen the design only when a concrete consumer requires durable zero-loss
collection and supplies the decision that depends on it.

## Interfaces

The public owner operations are `GET /v1/projects/{projectId}/sitemap` and
`POST /v1/projects/{projectId}/sitemap/mutations` in the
[public OpenAPI contract](../../../api/openapi/contractor-public-v1.yaml).
The read accepts an optional exact revision; mutations require `Idempotency-Key`.
Commands are `declare`, `annotate`, `remove`, `restore` and `replace-source`.
The trusted service also accepts observation batches from an authenticated
producer context. No command accepts a replacement map document.

The allocation collector uses
`POST /private/v1/allocations/{allocationId}/sitemap/observations` with
`{apiVersion, input}` and `Idempotency-Key`. Transient input contains only bounded
attempt metadata. The existing mTLS allocation gate and
[private wire schema](../../../api/v1alpha1/project-sitemap.schema.json) apply.
No Project selector, source classification or raw headers/body are accepted.
Explicit allocation tools retain their separate read/mutation/snapshot operations.

The CLI uses the generated public client:

```text
contractor project sitemap get PROJECT [--revision REVISION]
contractor project sitemap mutate PROJECT --file command.json --idempotency-key KEY
```

An owner can select the exact sitemap binding as an ordinary declared input
when creating a Project Run. This deliberate selection and permitted dedicated
sitemap reads expose the complete selected map. Dedicated snapshots select
entries from an exact map revision and retain only explicitly permitted needed
input files. The existing input fork retains the selected source; a live read
alone creates no such dependency.

## Limits, failures and errors

Source, producer and context identifiers reuse the shared ResourceId grammar
(LIM-API-19); manual owner mutation cannot choose a transport source identity.
The numeric contract is LIM-SMT-1 through LIM-SMT-14 in the
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

The owner may read the complete map and explicitly select it as ordinary Run
context. Dedicated operations retain current Project/allocation grants and
fencing, while artifact bytes retain their own exact-read authority. Source kind,
producer, deployment and observation context remain trusted provenance; public
commands cannot classify imported facts as observed traffic. Sitemap normalization
does not classify or redact metadata content. General credential storage, HTTP evidence
redaction and telemetry policies retain their existing owners and guarantees.

## Invariants

The reducer operates on a private copy and leaves accepted content unchanged on
failure. ArtifactStore owns atomic pointer/accounting/hold effects. Context IDs
and member IDs survive route regrouping; a changed structural selection creates
an explicitly distinct member and cannot silently widen a suppression.
