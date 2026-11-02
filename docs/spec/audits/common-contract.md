# Common Audit document contract

Requirement prefix: `ACT` · [Specification index](../README.md)

## Purpose and scope

This document owns the strict JSON program/task/hypothesis/context/conclusion/result
contract and its offline validation. Execution remains with the ordinary
[Run](../execution/runs.md), [Scheduler](../execution/scheduler.md), existing
Audit service/controller and [ArtifactStore](../artifacts/artifact-store.md).
An offline-valid document grants no access, registers no executable profile and
creates no Run. The current executable AuditProfile contract remains in
[audits](audits.md); accepting these documents offline does not translate it.

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
| `proposals` | Bounded model-proposed named seeds/work using allowed bindings; no persisted IDs, scope permission or automatic admission |
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

**ACT-R6.** Neither verdict polarity nor a completed Run creates a finding.
Supporting a bug claim or refuting a positive property may justify a problem only
under the owning finding evidence rules. Findings use existing intake receipts,
qualified deduplication and independent source contributions. A negative recheck
remains a linked ordinary result; it does not erase or automatically resolve a
historical problem. A generation input can include these results and gaps.

**ACT-R7.** Continuation is an optional ordinary allowed Workflow. Its accepted
source set is a frozen selection of new ordinary results/materials, findings and
hypotheses. One source set/producer decision has one durable identity, reconciled
through existing receipts. A continuation response does not itself trigger
another invocation: another call requires new accepted sources or an explicit
new decision, within pinned budgets. No model call occurs on every controller tick. A proposals document is an
ordinary output carrying named seeds and work only; its combined record count
obeys the mutation slice bound. Explicitly empty `hypotheses` and `work`
collections are a successful no-new-proposals output. Generation is optional;
the consumer settles this decision without creating hypotheses/tasks or
calling the generator again for the same sources. Missing or malformed output
is invalid, and hypothesis statements remain nonblank. Validation against the pinned program rejects
unknown Workflow/hypothesis aliases, collisions and authored system identities.
Optional work input selections name only available material handles from the
trusted frozen source set. They cannot overwrite task-context/prior-results
inputs or select foreign/incompatible artifacts. Initial selections name only
already selected Audit input slots. The owner resolves accepted choices to exact
ordinary Task input descriptors, without a graph of future output dependencies.

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

Bounds are LIM-ACT-5, with tighter ordinary Artifact/source/domain bounds still
applying. Every selected material is explicitly retained before source deletion
when an accepting consumer needs it. Publishing the list alone protects only its
own bytes; no generic recursive dependency import or retention is implied.

## Interfaces

`internal/contracts/auditcontract.Decode` returns one canonical JSON document
or no document and an error. The configuration catalog validator validates program
bindings against real resolved Workflows. Python conformance tests independently
validate the same public document schema and frozen per-item cases; they do not
advertise a Runtime completion capability.

## Limits, failures and errors

See LIM-ACT-1 through LIM-ACT-5, FAIL-ACT-1 / FAIL-ACT-2 and LIFE-ACT-1 / LIFE-ACT-2 in the platform
registries. Existing stable API error/transaction conflict rules remain with their
owners. Offline validation introduces no public HTTP operation or response field.

## Non-goals

This boundary adds no controller, collector, ledger, queue, capacity mechanism,
Project hypothesis registry or compatibility execution path. Actual Run execution,
Artifact retention and database mutations retain their existing owning APIs.
