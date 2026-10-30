# Project sitemap

Requirement prefix: `SMT` · [Specification index](../README.md)

## Purpose and scope

The sitemap owns shared HTTP surface metadata in a Project: concrete
members, observed behavior, declarations, grouping, member annotations and
reversible suppression. It is research context. It does not establish Audit
coverage, physical request counts, retained HTTP exchanges or verified findings.

[ArtifactStore](artifact-store.md) owns scope authorization, immutable revisions,
CAS, transfer admission, retained history and exact input forks. [Projects](projects.md)
owns the owner boundary; [retention and deletion](retention-and-deletion.md)
owns fencing, scope purge and retention dependencies. Runs authorize operations and
retain execution history; they do not own or identify accumulated Project metadata.

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

Every contribution records trusted source ID, source kind and optional deployment.
Source kinds use the shared name grammar (1–128 ASCII characters), rather than a
provider registry. `manual` and `openapi` are reserved for domain editing and
declarations; any other valid label may identify an observation producer. The
label is provenance data and must not select a Server parsing or repair algorithm.
Metadata provenance carries no producer Run, credential revision or session
generation; summaries carry no principal/context identifier. An observation summary is the set of source and
deployment, outcome, response status and parameter names for a concrete member,
with the minimum and maximum observation timestamps. Matching observations from
later Runs or HTTP sessions extend that time range without adding another summary.
This aggregation makes no assertion about authentication, principal equivalence
or which execution produced a particular observation. Exact execution history and
HTTP authentication/session state remain with their existing owners.

A source records its accepted generation, exact source descriptor when available
and code-computed grouping policy digest. An automatic transport source is stable
across sequential Runs that use the same configured source and deployment. A
source registry entry is not a retention hold. An exact evidence reference keeps
its Run ID solely to identify the retained file through the ordinary Artifact API.

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

**SMT-R5.** Observations must be bounded sets of source, deployment, parameter-name
shape and outcome summaries with minimum and maximum timestamps. Run and session
changes alone must not create another summary; no principal attribution or context
quota exists. A timeout or transport error must not acquire a response status.
Overflow must preserve accepted members and summaries and report discarded
membership or distinct summaries. Already retained summaries continue to update
their time ranges when the bound is full. Representative examples must not serve
as the member inventory.

Rationale (user decision, 2026-10-29): execution attribution consumed the same
sixteen-summary allowance as useful HTTP behavior. Repeated Runs could fill that
allowance with identical results and suppress a later new status. Remove this
attribution from the map instead of adding a second Run-history mechanism. HTTP
credentials and sessions remain execution concerns; sitemap metadata proves no
identity or authentication state. The sitemap area owns this boundary; reconsider
principal attribution only for a concrete consumer with trusted identity inputs
and a decision that needs them. No current caller supplies those inputs.

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
The receipt's policy is not another map or retained-history mechanism. Receipt
intent separately retains the authenticated producer Run identity: removing Run
attribution from published metadata must not let another Run replay its mutation.

**SMT-R10.** The map must consume ordinary automatic per-binding ArtifactStore
history policy, admission, maintenance and shared errors. It must not register
an artifact family, own retention SQL or accounting, retain all predecessors or
impose a cumulative producer-Run publication quota. The generic RunScope lifetime
insert contract applies only to RunScope, not to a Project write performed by a
Run.

**SMT-R11.** Durable selected context and proof must use existing authorized
exact-reference retention or input-fork lifecycles, atomically against history
eviction. Reading metadata, a CAS predecessor and a provenance descriptor must
not pin a revision. A lost exact-selection race requires explicit refresh or
reselection; cached bytes cannot recreate the source or authorize latest.

**SMT-R12.** Project deletion must fence publication under the existing shared
Project admission lock. Deleting a producer Run must not erase self-contained
Project metadata. Project purge must remove the domain's receipts through the
shared bounded Project receipt helper before removing the Project; no independent
receipt journal, claim, deletion controller or evidence lifetime is introduced.
Sitemap uses the common Project mutation key scope and lifecycle; persistence
and Project cleanup must contain no sitemap-specific receipt family or branch.

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

**SMT-R15.** Runtime downloads the map through the ordinary Artifact client and
explicit Project read target (ART-R135), then keeps one immutable encoded view per allocation,
bounded by LIM-SMT-12. Parsing and indexing run in joined threads. A cursor binds
the view identity, original exact revision and filters. Every page checks current
Server allocation, Project and the requested operation permission through the
common authority-only Artifact check; local bytes
confer no authority. Neither a view policy digest nor a policy-copy precondition
is part of the page contract. Publication or source eviction does not change an
existing view.
Refresh, absence, replacement and release invalidate old cursors with
`sitemap_cursor_expired`. There is no view TTL, disk persistence or implicit Artifact hold. The view keeps
one encoded payload plus byte-span/index metadata, not a second decoded document
graph. Commands do not edit or invalidate that view; only an explicit refresh
loads the later committed map. A missing current binding becomes the domain
empty map after authorized artifact absence; an exact missing revision remains
unavailable. Generic artifact transport never interprets map contents.
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
resolve through RunStore's original-input operation and ArtifactStore's bounded
fork-revision descriptors to immutable original Run-creation input forks,
never Worker-editable current bindings. Query work is independent of the
containing Run's artifact count. Sitemap owns no artifact-table SQL. Only needed permitted original input bytes are
frozen with ordinary exact forks. If a referenced file has no matching permitted
original input, preserve selected metadata and record the existing
`source-unavailable` gap; `requireReferencedInputs: true` makes this fail the
snapshot. Once a needed exact input is resolved, missing or corrupt stored
content and storage I/O errors remain ordinary ArtifactStore errors and roll
back the snapshot, regardless of that flag. Optional selection must not turn
storage failure into a different accepted result on replay. This distinction is
owned by the sitemap snapshot contract; reconsider it only together with an
explicit change to frozen-context replay semantics. Preserve omission details
for selected sources and whole-map gaps in the selected context.
Unneeded missing input names do not invalidate metadata reads or snapshots.
Foreign provenance alone grants no byte read.
Report evidence may match a permitted original input by its exact content
descriptor (digest, media type and size), even when the producer Run/ref differs.
The snapshot rebinds it to that consumer input and deduplicates equivalent
locators before ordinary retention. It never reads the producer's scope to
resolve a provenance mention.

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

**SMT-R18.** Allocation composition connects selected HTTP and Caido capture to an optional
transport-neutral observation sink. It receives minimal metadata for actual
attempts, including each physical retry/redirect and negative outcome, before
evidence-buffer admission/eviction. It never follows a Location. Submission does
not await network or storage I/O. Subscriber errors cannot replace a target
result; caller cancellation is preserved and joined. HTTP and Caido tools own no collector
flush, retry or cleanup. An invalid target response status is represented as a
transport-error observation with status zero, while the HTTP tool retains its
original protocol error. It must not invalidate otherwise valid observations
in the same batch.

**SMT-R19.** Runtime retains one bounded private in-memory queue of attempt
metadata and already published report references. Observation frames and private collection requests contain no
matching-secret list; no separate secret-count limit applies. No headers, bodies,
persistent queue, URL identity or receipt digest are constructed there. One
background private batch request carries normalized observations and bounded
omission facts. Runtime tool adapters own provider formats, timestamp conversion,
row selection and malformed-provider-row handling. Server strictly validates the
common batch and applies only source-independent domain projection, grouping,
identities, receipt hashing and coalescing. Server code derives Project and
deployment from the allocation and stable source identity from the supplied
producer label and deployment. Run identity remains in operation authority and receipt
intent, outside published metadata. Collection frames contain no session generation,
credential revision or principal/context identifier; HTTP tools add no tracking
state solely for sitemap. Actual authentication, cookies, sessions and request tags
remain owned by the HTTP transport.

**SMT-R20.** Pending and in-flight batches obey LIM-SMT-14; count, byte and time
boundaries trigger publication. Overflow may drop metadata with bounded technical
logging. An ambiguous publication repeats only its identical input/key within
one original three-request/15-second allowance. Definite protected-history
refusal is permanent. A Server-exhausted transaction conflict receives no new
Runtime transaction retry allowance. Refusal or budget exhaustion disables
further automatic publication for the allocation; later reads and drain cannot
renew that allowance. A definite invalid or unavailable source report drops only
that report with technical diagnostics; subsequent valid contributions may continue.
`Retry-After` applies only inside the remaining budget.
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
is never the first flush boundary. Composition closes producers before
joining collector work at finalize, abort, fence, release and prepare rollback.
No publisher survives release. Automatic failures, exhausted publication,
overflow and crash may lose the bounded private batch; they produce no Run or
Runtime-report collection state, generic map-completeness marker or required
completion policy. Explicit imports, mutations and snapshots report their own
operation failures. The HTTP-tools area owns this best-effort limitation;
reopen the design only when a concrete consumer requires durable zero-loss
collection and supplies the decision that depends on it.

**SMT-R22.** Structured imports accept an explicit `ffuf-report@1` or
`katana-report@1` format and an exact current-Run ArtifactRef at the Runtime tool
boundary. Runtime reads exact bytes through the ordinary ArtifactClient and
performs bounded parsing off the event loop. Server knows neither report formats
nor provider payloads; it receives the same normalized batch as any other producer.
There is no import-specific file store, raw-file API or Server report parser. The
`application/json` schemaVersion 1 tool report is limited to 512 KiB by SCN-R41;
its retained results obey the owning 100-record and 128-KiB limits. The projected
command must also fit the existing LIM-SMT-7 byte bound, including JSON escaping.
One admitted report is one bounded reducer operation. Unknown formats, mismatched tools, malformed
envelopes or invalid normalized records fail permanently without publication.
No import executes a scanner, follows a redirect or substitutes a latest revision.

**SMT-R23.** Runtime prepares one normalized batch before publication retries.
The existing Project mutation receipt binds its projected domain intent, omission
facts, optional exact evidence ref and authenticated Run. Replaying that same
prepared batch returns the accepted map ref without rereading source metadata or
map bytes, even after source or result eviction. Changed accepted intent, evidence
or Run under the same key conflicts. A fresh explicit tool invocation reads its
selected exact input again and fails if it is unavailable; it does not query a
second receipt API or reconstruct the source. Source identity remains stable by producer label and
deployment; importing Runs and report revisions do not create summary variants.
Existing bounded member evidence stores exact report locators without acquiring
holds. Server resolves an optional evidence ref through current-Run Artifact
metadata inside the new mutation transaction, without reading or interpreting
payload bytes. Ordinary namespace and scope rules apply. Evidence overflow does not discard behavior. Snapshot authorization and
retention continue to follow SMT-R16. Reports contain no exchange timestamp, so
their immutable Artifact revision publication time supplies first/last; these
times describe reports, not the exact time of individual HTTP requests.
Reports with no retained members can contribute source gaps but have no member
evidence locator; the caller retains its selected exact report ref independently.

**SMT-R24.** ffuf contributes accepted GET matches and their response statuses;
it does not establish route existence beyond a returned response or distinguish
wildcard/soft-404 matches. Katana contributes only observed same-origin GET
responses. Its bounded crawl, extension filter, response-size limit and
unverified exhaustion remain inherent limitations; detailed per-report facts
remain in the exact source report. The map preserves known discarded-result
counts as `members`, invalid/skipped observations as `projection`, request-error
counts as `request-errors`, and known execution incompleteness as `incomplete`.
`report-truncated` counts reports declaring truncation or output overflow, not
an invented count of discarded rows. `evidence` counts omitted exact report
locators. These are bounded source facts, not a collector completion status or
physical-traffic counters; replaying one accepted key never adds them again.

**SMT-R25.** The generic tool Worker notifies an optional report subscriber only
after ordinary report publication, including a valid report of a failed tool.
Composition selects supported scanner reports and enqueues exact refs without
reading their bytes. One pending queue and one publisher serve HTTP, Caido and
reports within LIM-SMT-14. A pending report ref is read and preprocessed by its
Runtime adapter before publication; every Server request contains one normalized
source batch. Automatic publication requires collection/write
authority and the publishing allocation namespace; explicit import requires the
selected import operation and ordinary current-Run read authority.

Selected Caido history pages, request details and completed replay entries may
contribute only metadata already returned by those operations, before model
output redaction or evidence-buffer eviction. No additional provider request is
made for collection. Native sitemap nodes lacking methods and other projections
lacking factual origin/time do not manufacture observations. A missing response
is an `attempt` with status zero. Runtime normalizes valid timestamps; unsupported or
invalid selected rows become bounded `projection` gaps without rejecting other
valid rows. Caido provider IDs never become invented durable ArtifactRefs. Server
rejects an invalid normalized batch uniformly, regardless of producer label.

Rationale (user decision, 2026-10-29): replacing Caido or adding another tool must
require a Runtime adapter, without adding a Server provider enum, endpoint,
format parser or special case. The sitemap domain owns shared grouping and
storage invariants; Runtime tools own all source preprocessing. Reconsider this
boundary only for an explicitly selected producer outside Runtime with a defined
trust and transport contract.

## OpenAPI contract operations

**SMT-R26.** OpenAPI is an API contract artifact, not a provider observation.
`sitemap-openapi-join@1` is a Server operation that joins exact current-Run
`sitemap` and `openapi` Stage context inputs in a Project Run. It requires string
arguments `origin` (explicit absolute deployment base URL) and `contract` (stable
contract name), with optional `deployment` identity. Source identity derives
from contract/deployment, independently of producer Run and artifact binding.
The common Stage declaration permits `sitemap` and `report` results;
only declared slots are materialized, and `result.artifacts: {}` creates neither.
The bounded Stage summary always gives comparison counts and, for apply, the
actual Project publication ref. No Runtime, Planner, model, provider call,
target traffic or remote reference request occurs. `internal/operations/sitemapjoin`
is an adapter; `internal/projectsitemap` owns computation and publication.

**SMT-R27.** Join reuses the existing Go OpenAPI decoder/version policy and
local structural reference resolver, including local reference/sibling overlay
budgets. Explicit `origin` supplies the base path; OpenAPI `servers` does not
choose or guess a deployment. Invalid origin, malformed/unsupported selected
structure, reference cycles, remote/missing references and exceeded limits fail
permanently and atomically; no partial source replacement is accepted. Schemas,
request/response bodies and examples are not interpreted as traffic. Parameters
merge path-item and operation declarations by location/name with operation
overrides; declared statuses support exact codes, `nXX` and `default`.

**SMT-R28.** Computation uses the ordinary source reducer and its common
concrete-before-template and ambiguity rules. Only retained concrete members
are regrouped; annotations, evidence, suppression selections and existing
omission gaps retain SMT-R5/R6/R7 semantics. Comparison uses only the selected
contract's rules, even if other contracts affect the shared grouping. It counts
retained observed members and declared method/path operations, not requests.
A declaration alone never creates observed behavior or coverage. Parameter
comparison uses the union of retained observed query names; it is not per-request
validation or a required-parameter verdict. Declarations without a retained match, unmatched
observations, ambiguous members/declarations, query-name differences and undeclared response
statuses have bounded report entries. Observed media types are unavailable in
the metadata model and are explicitly reported as a limitation. Limits and
omitted counts are LIM-SMT-15; retained-members-only and reachability limitations
are always explicit. A declaration with only ambiguous matches is counted separately
as unresolved, never as declared-only.

**SMT-R29.** Pure join creates ordinary Run artifacts without changing the
live Project map. `sitemap-apply-openapi@1` is a separate explicit operation with
the same bindings, applying only the computed source change to the current map.
It preserves concurrent observations but compares the selected predecessor's
source generation/exact descriptor/policy and manual rules, annotations and
suppression state. A conflict never refreshes the selected intent. Its output
contains the map actually published; its comparison basis remains the pinned
input map. An optional JSON report names exact input descriptors, comparison and
actual Project publication ref. No report or patch copy is mandatory.

**SMT-R30.** Artifact reads, ordinary output creation and optional live
publication share one transaction with the existing Project keyed mutation
receipt, keyed by StageExecution identity. Replay precedes input/content reads,
keeps exact returned refs and never republishes or substitutes latest bytes.
Run cancellation, expired/replaced claim, Stage progression and Project deletion
fence publication. All effects roll back together on any definite error. Common
Artifact limits, exact holds, history eviction, cleanup and Project receipt
purge apply; descriptors and receipts acquire no additional retention hold.
New attempts retain original pinned inputs and source intent. The first slice
requires a Project Run, including pure join, to use its existing receipt and
lifecycle owner; standalone execution is not silently inferred.

## Interfaces

Owner reads use the ordinary Project Artifact raw/metadata endpoints for
`sitemap/map`, with optional exact revision and the common owner-read lifecycle
policy. There is no public sitemap GET. The CLI preserves its sitemap get command
as a domain decoder over ordinary Artifact bytes and ETag; a missing current map
becomes empty only after confirming that the Project exists and is active. An
exact miss and an inaccessible/missing Project never become empty.

The public owner mutation is `POST /v1/projects/{projectId}/sitemap/mutations`
in the [public OpenAPI contract](../../../api/openapi/contractor-public-v1.yaml).
Commands are `declare`, `annotate`, `remove`, `restore` and `replace-source` and
require `Idempotency-Key`. Manual editor commands carry source rules and optional
exact source preconditions, but no new source/rule identity or source artifact:
the authenticated domain handler assigns those identities. Stored documents and
expected source descriptors retain their identities. No command accepts a
replacement map document.

The allocation collector uses
`POST /private/v1/allocations/{allocationId}/sitemap/observations` with
`{apiVersion, input, automatic}` and `Idempotency-Key`. Transient input contains a
Runtime-selected `kind`, required `observations` (zero to 100), optional `gaps`
(at most 16 distinct reasons with positive bounded counts) and optional exact
`artifact` evidence ref. Empty batches register the source without manufacturing
an observation. Existing mTLS allocation fencing and the
[private wire schema](../../../api/v1alpha1/project-sitemap.schema.json) apply.
Explicit allocation commands retain their mutation/snapshot operations.
There is no separate Server report-import endpoint. The required `automatic` transport flag belongs to
composition and is absent from model tool arguments. All store-reaching routes
preserve retryable `503 storage_transaction_conflict`.
The model-facing `import_sitemap_source(format, artifact_ref, idempotency_key)`
uses ordinary scalar and Artifact argument bindings, including in `tool@1`
Workflows; it does not require nested argument construction by the Worker.

Allocation reads use the common Artifact raw/metadata API with `scope=project`
and the exact selected operation (ART-R135). Before every cached list/read page,
Runtime uses the same binding's common `/access` operation; that check reads no
artifact bytes, metadata or lineage. There is no private sitemap GET/access or
sitemap-specific revision header. Grouping commands derive their manual source
identity on the Server without a preceding metadata endpoint. Snapshot POST
independently checks its selected operation, write permission and allocation fence.

The CLI uses the generated public client:

```text
contractor project sitemap get PROJECT [--revision REVISION]
contractor project sitemap mutate PROJECT --file command.json --idempotency-key KEY
```

An owner can select the exact sitemap binding as an ordinary declared input
when creating a Project Run. This deliberate selection and permitted allocation
artifact reads expose the complete selected map. Dedicated snapshots select
entries from an exact map revision and retain only explicitly permitted needed
input files. The existing input fork retains the selected source; a live read
alone creates no such dependency.

## Limits, failures and errors

Source and deployment identifiers reuse the shared ResourceId grammar
(LIM-API-19); manual owner mutation cannot choose a transport source identity.
The numeric contract is LIM-SMT-1 through LIM-SMT-15 in the
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
fencing, while artifact bytes retain their own exact-read authority. Source ID,
kind and deployment remain trusted provenance; public commands cannot classify
imported facts as observed traffic. Summaries contain no principal or authentication
claim. Sitemap normalization does not classify or redact metadata content. General
credential storage, HTTP evidence redaction and telemetry policies retain their
existing owners and guarantees.

## Invariants

The reducer operates on a private copy and leaves accepted content unchanged on
failure. ArtifactStore owns atomic pointer/accounting/hold effects. Member IDs
and their retained summaries survive route regrouping; a changed structural
selection creates an explicitly distinct member and cannot silently widen a
suppression.
