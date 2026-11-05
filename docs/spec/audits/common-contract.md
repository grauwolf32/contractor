# Common Audit document contract

Requirement prefix: `ACT` · [Specification index](../README.md)

## Purpose and scope

This document owns the strict JSON program/task/hypothesis/context/conclusion/result
contract, its strict validation and the accepted Task boundary. Execution remains with the ordinary
[Run](../execution/runs.md), [Scheduler](../execution/scheduler.md), existing
Audit service/controller and [ArtifactStore](../artifacts/artifact-store.md).
An offline-valid document grants no access, registers no executable profile and
creates no Run. The executable AuditProfile contains this Program; its pinned ordinary Workflow
closures are retained by the Audit owner. Offline validation alone does not
register a profile or translate another format.

## Concepts

A program names permitted Workflows, selected material input slots, initial
ordinary work, optional hypothesis seeds, optional continuation and finite budgets.
A task is accepted work with selected exact input descriptors and an optional
system-pinned hypothesis revision. A conclusion is authored content; a result is
an ordinary immutable JSON output assembled with trusted per-item provenance.
A finding is an evidenced problem, subject to the existing
[finding intake](../findings/findings.md), and is optional.

## Model

[The single JSON Schema](../../../api/audit/v1/contract.schema.json) defines the
closed documents and reuses the existing shared wire name, ID, media type and
exact ArtifactRef definitions. No model-authored final package or internal ZIP
is required. A hypothesis ID is opaque and independent of its owner's identity;
its revision remains exact when ownership changes. The positive revision counter
uses the existing exact-integer bound (LIM-AUD-22), independently of the per-Audit
revision-count budget; fractional or exponent JSON tokens cannot alias an integer
revision during decoding. There is no owner prefix in
that ID and no second Project hypothesis registry.

| Document | Authority and content |
| --- | --- |
| `program` | Authored selected Workflow aliases, input contracts, initial work and named hypothesis seeds, continuation and limits/policy |
| `hypothesis` | System identity/revision and immutable statement, scope and grounds |
| `task` | Accepted Audit/task identity, selected Workflow alias, exact input descriptors/parameters and optional hypothesis pin |
| `context` | Trusted Audit/Run, exact Workflow identity/digest, bounded item-key/task membership and objective, optional exact readable hypothesis snapshot and admitted evidence descriptors |
| `proposal-context` | Server-local pinned program and exact selected material handles used to validate producer choices; not a model-authored program or second execution state |
| `proposals` | [Local hypotheses, exact grounds and optional work](hypotheses.md#HG-R6); no persisted IDs or automatic admission |
| `proposal_context` | Frozen Run-local research handles, original input identities, allowed Program and explicitly selected observations |
| `proposal_binding` | Frozen configured context/output refs for the current producer Run, owned by its trusted materializer |
| `prepared-work` | Deterministic work selections and exact ordinary inputs; no assigned Task identity or permission |
| `prepared-work-context` | Server-local pinned program, trusted producer Run and admitted exact material descriptors |
| `conclusions` | Model-authored per-item content, conditions, evidence and limitations; no provenance fields |
| `result` | Ordinary output with system-derived origin on each item; no Run status or mandatory finding |
| `control` | Offline-valid control transition; no live state change or authorization |
| `mutation` | Bounded slice of task/hypothesis records; no duplicate record identity |

## Requirements

**ACT-R1.** Validation rejects duplicate/unknown/case-aliased members, nulls,
trailing values, invalid Unicode/NUL, unsupported schemas and over-bound work
before returning any canonical document. The closed rejection is
`audit_document_invalid`; content is never included in a diagnostic.

**ACT-R2.** The real configuration Snapshot resolves every selected exact
Workflow and validates required/unknown input and parameter slots, required
outputs and compatible canonical media types. The same catalog checks apply to
each effective initial-task input after selected-material overrides: a required
Workflow slot must select a required Audit input; optional Workflow slots may
select optional inputs. An input mapping selects an
explicit Audit material, an explicitly selected `task-input`, trusted task context
or prior-results JSON; no arbitrary
object paths, per-provider parsing, second Workflow graph or automatic
cross-Workflow dependencies are interpreted by this validator. Workflow closures
are pinned by the accepted task owner, not by model output.

A `task-input` mapping has no `name` or future dependency. It declares a slot
whose already available material must be selected explicitly at task acceptance.
An unused checking binding therefore needs no fabricated initial Audit input.
Initial work must select an available Audit input for each required `task-input`;
the real Workflow owns requiredness and media compatibility. These selections
cannot overwrite system-owned task-context or prior-results inputs.

A hypothesis-bound initial or proposed task must select a Workflow binding with
a JSON `task-context` input and required JSON `result` output. The trusted
context carries the task objective and complete exact hypothesis statement,
scope, grounds and source provenance; its result origin retains the stable
ID/revision. A context key/ID alone cannot stand in for the assertion.

**ACT-R3.** An initial task names an allowed Workflow and optionally a named
seed. Unknown names and duplicate work keys are invalid. Initial work may be
exploratory, checking, supporting or generation work; these purposes introduce
no execution entity types or mandatory phases. An authored program never supplies
a persisted hypothesis ID/revision pair.

**ACT-R4.** A checking conclusion has supported/refuted/inconclusive meaning
only within its tested conditions. Supported/refuted needs nonempty admitted
evidence and tested conditions; inconclusive needs an explicit limitation.
Exploratory work requires neither a dummy hypothesis nor verdict. Technical Run
success/failure is separate and never proves or refutes a claim. A negative probe
cannot globally refute an existential bug claim beyond the stated conditions.

**ACT-R5.** The assembler derives Audit/task/Run and exact hypothesis provenance
from caller-supplied trusted context. It requires complete, unique per-item
membership and selected evidence membership. It rejects latest-revision
substitution, foreign provenance/evidence and model-authored origin. In a batch,
one conclusion is never copied to every hypothesis. Optional finding receipt
associations also come from trusted per-item context, never model-authored IDs. A later hypothesis revision
or Run attempt never rewrites an accepted result.

The common Workflow resolver derives a Runtime completion target from selected
`audit-results@2` tools and declared JSON context/result slots. RunService pins
that target inside Run creation; Scheduler revalidates the target and exact
context before allocation. Runtime refuses those tools without the contract.
PostgreSQL protects immutable Run completion and exact allocation-to-Run
identity. It does not interpret Audit roles or duplicate Workflow selection
rules to infer a required completion target.

**ACT-R6.** Neither verdict polarity nor a completed Run creates a finding.
Supporting a bug claim or refuting a positive property may justify a problem only
under the owning finding evidence rules. Findings use existing intake receipts,
qualified deduplication and independent source contributions. A negative recheck
remains a linked ordinary result; it does not erase or automatically resolve a
historical problem. A generation input can include these results and gaps.

**ACT-R7.** Continuation is an optional ordinary allowed Workflow, evaluated at
the drained Task barrier before normal Audit completion. Accepted decision,
source boundary and exact inputs retain their identities through existing
Task/Run receipts and claims. New accepted work or an explicit owner decision
can trigger a later call; unchanged ticks and valid empty output cannot.
[Hypothesis production](hypotheses.md#HG-R13) owns bounded source selection,
origin-based self-trigger protection, configured research inputs and limits.
[The proposal contract](hypotheses.md#HG-R8) owns successful-Run publication,
hypotheses-only/work-only/empty decisions, source validation and atomic intake.
Pause preserves accepted continuation without dispatch; Resume uses the same
work. Cancellation/deletion permits collection without semantic admission.

**ACT-R8.** The control contract is soft pause/drain, Resume of accepted work,
and terminal Cancel. Admission and Run/HTTP/model timeouts retain separate owners.
The admission clock freezes while paused; Resume uses its remaining allowance
or an explicit replacement bounded deadline. Deadline exhaustion closes new
admission and drains, rather than inventing a semantic result or cancelling work.
Collection remains permitted under the owner's fence while paused/cancelling.

| Intent | Guarantee |
| --- | --- |
| Start | Pin Project/baseline, selected program/closures, exact inputs, scope, operation policy and budgets |
| Pause | Close new Run submissions; admitted work drains and its exact outputs are collected |
| Resume | Resume the same accepted inputs/results and remaining work after owner/Project/scope checks |
| Cancel | Close submission, cancel children, join terminal collection/cleanup, then close with partial results; terminal Audit cannot Resume |
| Deadline | Pause new admission with `deadline_exhausted`, retaining truthful progress and already admitted work |
| Delete | Fence new work, join children/readers, then bounded owner-local purge; independent Project publications/findings survive |

**ACT-R9.** Formal approval requests, waiting/expiry/renewal, approval-required
policy and report acceptance are absent from the common contract. Ordinary
scope, allowed/prohibited operations and access checks remain. Historical
consent cannot become an active permission; no compatibility reader, special
policy-wait state or old-policy Resume is defined.
The schema removal requires the earlier FND-R215 cutover to have archived and
removed executable old records. Remaining requests, decisions, waiting states
or approval-bearing items reject migration atomically without changing their
bytes. The removal neither repeats archival nor interprets archived facts.

**ACT-R10.** Numeric limits are LIM-ACT-1 through LIM-ACT-5. Existing tighter
owner bounds still apply. Each initial or generated hypothesis seed must fit
32 KiB in its complete RFC 8785 canonical UTF-8 encoding, including statement,
scope, grounds, member names, punctuation and JSON escaping. The check also
applies to seeds in pinned proposal-context programs. This is a necessary early
content bound, without guessed identity metadata; after trusted ID, revision and
origin are assigned, the complete hypothesis document must independently fit
32 KiB. Full hypothesis snapshots in contexts and mutation records retain that
complete-document bound; unrelated result text retains its own limits.
Mutation limits are slice limits, not permission to
materialize or rescan a whole containing collection. Lists retain existing
row/byte pagination bounds. Replays reconcile the original identity through
existing keyed/natural receipts; they never repeat target activity merely to
repair an acknowledgement. A deterministic invalid result is permanent.

An allowed Workflow may set `max_attempts` to preserve a tighter authored
ceiling than the program-wide limit. It is a positive integer no larger than
the program's `max_attempts`; reservation and retry settlement use the minimum
of this ceiling and the effective Audit attempt budget. This changes neither
the cumulative Run budget nor the identity of an accepted Task.

**ACT-R11.** Exact accepted input/output refs retain their ordinary ownership,
revision, digest and lineage. Publication to Project is explicit, create-only or
CAS replacement through the existing Artifact owner, with its original receipt.
No ref embedded in JSON grants authority or recursively imports dependencies.
Source Audit deletion preserves independent Project findings/publications and
necessary evidence and readable hypothesis snapshots; it does not create a second
live registry. Reading alone creates no hold.

**ACT-R12.** The CLI accepts `config validate --audit-document <JSON>` after
loading the real catalog. Conclusions/results, proposals and prepared-work selections additionally require the exact
`--audit-context <JSON>` supplied by the trusted caller (per-item context for results; a server-local pinned program and selected-material
context for proposals). Local context files are
validation inputs, not proof of live authority. There are no database writes,
credentials, target calls, grants or implicit execution effects. Other JSON
Artifacts and external archive/export contracts keep their own formats.

**ACT-R13.** A `contractor.audit.prepared-work.v1` document is an ordinary
immutable preparation output with `work` and `gaps`. Each selection has a unique
producer-local `key`, allowed Workflow alias, objective and explicit input-slot
to scoped exact Artifact descriptors. It contains no Audit/task/hypothesis IDs,
Run state, approvals or dependency expressions. An empty list is valid; its
actual domain gaps remain visible and no work is invented.

The concrete OpenAPI producer is the ordinary operation in
[SOP-R10](../execution/server-operations.md). A `prepared-work-context` retains
the pinned program, trusted producer Run ID and an admitted exact source set.
All descriptors belong to that producer Run; duplicate references, changed
metadata, foreign selections, unknown aliases/slots and system-context overrides
are rejected. The catalog additionally checks actual required slots and media
types. Source-set comparisons are indexed rather than an item-by-material search.
Producer encoding may validate shape without that context; the CLI and typed
admission decoder require it. Offline validation grants no live access.

The common accepting owner derives provenance from the accepted producer Run
and exact output, assigns Task identities and applies current control/scope and
budget fences. Local keys are not global identities. Slices retain the same
producer, exact list and entry identities through the existing receipt owner;
response loss cannot create fresh logical work. Paused intake may accept without
submitting Runs. Cancellation/deletion may collect the output without accepting
more work. Live Audit acceptance remains the owner's execution responsibility;
this document's validation creates no task rows or controller.

An authored binding selects a preparation output with the logical output key
`work`, which names a required ordinary JSON Workflow output. Other material
outputs are retained without inspecting their contents for work. The collecting
owner validates that declared list and keeps its producer execution collecting
while bounded intake is pending; this protects its explicitly selected sources
from Run purge. Source identity, count and accepted offset live on the existing
execution row, and Task acceptance advances that offset with its existing
mutation receipt. Closing intake retains the original list and partial offset.
Proposal generation and authored continuation use their own explicit contract;
the coordinator does not infer either from an ordinary material's bytes.

Bounds are LIM-ACT-5, with tighter ordinary Artifact/source/domain bounds still
applying. Every selected material is explicitly retained before source deletion
when an accepting consumer needs it. Publishing the list alone protects only its
own bytes; no generic recursive dependency import or retention is implied.

**ACT-R14.** Accepted common Tasks have only `ready`, `selected` and `settled`
coordination states. Execution intent, technical outcome, attempts and collection
remain the existing Run/execution/membership relations; common work creates no
Round, inventory manifest or task archive. Reserving compatible work assigns one
stable submission identity under the ordinary claim, Scheduler lane windows,
Project/control/deadline and Run-budget fences. The ordinary RunService derives
the Workflow, parameters, exact inputs and configuration from that locked pin;
trusted task context uses the actual Run identity and its frozen input forks.
Recovery resumes a reserved intent before checking whether another reservation
is affordable. Pause preserves an unbound reservation for Resume. Cancellation
settles undispatched Tasks with explicit gaps in bounded slices and drains
selected attempts through terminal observation and the existing collection
receipt.

The reconcile snapshot probes no more than 64 ready index keys per allowed
Workflow (at most 16 Workflows), then hydrates no more than 64 Tasks and 1 MiB
of Task metadata. Closed lane windows exclude their aliases before selection,
so a full model lane cannot hide ready tool work. A sparse queue does not read
settled Task history. An active Audit completes only after any eligible configured continuation is settled or explicitly limited, its initial work is
accepted or explicitly limited, its Tasks settle and its execution window
drains; completion uses the same Audit revision/control fence as acceptance.
An empty paused Audit never completes automatically. Successful supporting
outputs are retained as ordinary bytes; only a declared logical `result` is
normalized against trusted context. Permanent exact-data/contract rejections
commit a technical gap once; ambiguous storage/read failures keep the original
identity and retry through its existing receipt. Available partial outputs of a
failed Run remain retained without inventing a semantic conclusion.

**ACT-R15.** The public Audit profile summary carries the exact common input
contracts, limits and operation policy; the detail additionally carries its
Program. It exposes no inferred inventory, fixed execution role or compatibility
mode. Public Audit detail/list/start and lifecycle responses expose committed
owner state, Task count and iteration ordinal. Start returns the Audit without
creating a Round or claiming that initial Task intake has already happened.
Task lists are bounded keyset pages, and each Task's technical attempts remain
ordinary Run associations. An explicit owner-authorized task request is accepted
through the same operation that admits declared preparation work.

A saved Task result is readable through the ordinary Audit artifact endpoint
with logical key `task/{task_id}/result` and its exact revision. The read resolves
only that owned Task's accepted Audit material, verifies exact metadata and the
owner revision fence, and creates no new receipt or retention. Producer Run
deletion does not remove the accepted bytes. An obsolete result revision is not
substituted with a current revision. Existing execution-output links retain
their own exact receipt authority.

The minimal check view reads current Task pages and retained materials, with
explicit limitations and gaps; Run state and Finding state remain separate. It
does not claim coverage or a security verdict from successful Run completion.
Activity uses the existing paged Audit event history.

`GET /v1/audits/{auditId}/materials` lists only accepted receipt links as exact
scoped material descriptors and logical keys, without payload/provenance hydration.
It seeks by an internal immutable ordinal, so even a 512-byte logical key fits
the existing 512-byte signed cursor bound. Candidate selection reads at most
`limit+1` links (maximum 201); encoded metadata is bounded to 1 MiB including
framing. The ordinal is traversal metadata, not a public artifact identity.
`GET /v1/audits/{auditId}/tasks/{taskId}/attempts` reads only the selected Task's
indexed membership history (at most the configured attempt ceiling, never more
than ten), with retained Run ID, deletion marker and technical outcome/disposition.
Both lists authorize the owner, bind continuation to the resource and Audit revision, recheck the
owner fence after the read, and reject concurrent changes with `409`. A deleted
Run remains provenance and need not be a live navigable resource.

The public Round `items` and `coverage` operations are removed. Coverage is
read from the fixed hypothesis denominator, pending authored seeds, immutable
hypothesis revisions and their indexed Task/result history (ACT-R16–ACT-R17).
The workspace summary reads persisted Task/submitted-Run/outstanding-Run counts
from one Audit row; it makes no aggregate claim about supported hypotheses,
security outcomes or Finding counts. Findings and their sources retain their
Project owner and bounded independent reads.

**ACT-R16.** The Audit owns hypothesis identities and append-only exact revisions.
Initial authored seeds receive opaque deterministic IDs; manual commands receive
owner-assigned IDs. Public writers provide source key, statement, scope and
grounds, never identity, revision or origin. The admission origin remains immutable
across edits; it records how the assertion entered this Audit. An edit appends a
revision and never changes an accepted Task's full frozen hypothesis snapshot.
Creating/editing an assertion together with its check uses the existing Task
mutation receipt and one transaction; refusal commits neither. Standalone edits
use the existing keyed Audit mutation family. Concurrent edits use the Audit
revision precondition. Replaying a lost acknowledgement reads the receipt before
mutable hypothesis heads.

The assertion editor captures the Audit revision with the source content when
opened. Both Save and Save and check retain that precondition through polling.
A changed source revision or a definite precondition refusal preserves the
unsaved text and blocks another write until the user explicitly reloads the
current assertion. A new assertion may explicitly adopt the refreshed Audit
revision after a refusal. An ambiguous acknowledgement retains the original
request body, key and precondition until its retry resolves.

Registry pages expose checking Workflow aliases from the stored Program pin, so
continuation does not depend on a changed or removed live catalog entry. Only
bindings that can use already available Audit inputs are offered by this form.
Current assertions use ordinal keyset pages bounded by rows and 1 MiB; exact
revision reads are explicit. Task history can filter one hypothesis ID through
its indexed accepted references. A normalized batch result remains one exact
Run artifact with separate trusted per-item origins. The UI displays only the
member whose origin names the selected Task; downloading returns the original
batch. No semantic conclusion fans out to other hypotheses.

**ACT-R17.** A Program may select `seed_source` from one required checklist input
or one exact curated standard version. Start freezes the interpreted selection
and its ordinary Audit-owned source artifact. `contractor.audit.assertions.v1`
is a strict finite JSON/YAML source: ordered entries contain key, version,
statement, scope, grounds, applicability, sorted allowed methods and sorted
required evidence. Its owning shared schema defines the shape; the ordinary
Audit domain owns YAML parsing and standard mappings. No model generates fixed
seeds. Selection is at most 128 assertions / 1 MiB of frozen metadata. A standard
entry retains its original version, applicability, mappings, allowed methods and
evidence sufficiency contract. Unmapped selected requirements stay unchecked in
the original denominator, without fabricated work.

Seed admission precedes Task admission and charges the ordinary pinned hypothesis
budgets. Each slice has at most 128 primary records / 1 MiB; an explicit pending
fixed count exposes incomplete seed admission. Additional assertions, edits,
split checks, exclusions, unavailable evidence and generated work never change
the frozen fixed denominator. Registry usage is cumulative: revisions consume
bytes and revision budget even when no longer current.

## Interfaces

`internal/contracts/auditcontract.Decode` returns one canonical JSON document
or no document and an error. The configuration catalog validator validates program
bindings against real resolved Workflows. Python conformance tests independently
validate the same public document schema and frozen per-item cases; they do not
advertise a Runtime completion capability.

## Limits, failures and errors

See LIM-ACT-1 through LIM-ACT-5, FAIL-ACT-1 / FAIL-ACT-2 and LIFE-ACT-1 / LIFE-ACT-2 in the platform
registries. Existing stable API error/transaction conflict rules remain with their
owners. The public API publishes the same Program schema and the existing
Audit/task/material projections; offline validation performs no HTTP mutation.

## Non-goals

This boundary adds no controller, collector, ledger, queue, capacity mechanism,
Project hypothesis registry or compatibility execution path. Actual Run execution,
Artifact retention and database mutations retain their existing owning APIs.
