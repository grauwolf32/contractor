# 34 — Contextual Audit checklist prioritization

Status: **Draft integration target; pure core implemented ([V64-000](../../tasks/v64/v64-000-audit-priority-core.yml)); Audit capability not enabled.**

This document owns the proposed behavior; the
[plan](../plans/2026-09-20-audit-check-prioritization.md) owns task sequencing.
Existing profiles and [Audit contracts](19-audits.md) remain unchanged until
this opt-in capability is implemented and advertised.

## 1. User outcome and boundary

An Audit receives an exact checklist and optional service description and
existing findings. Each checklist item gets an independent model evaluation
against the same frozen context and rubric. Server selects the highest-priority
items for the current pass and retains every remaining item, its evaluation and
its deferral explanation. A pass means selection of one immutable Audit Round.

`topN` defaults to 10 and MUST be an integer from 10 to 1,000. For `R` remaining
candidates, a successful selection contains exactly `min(topN, R)` distinct
checklist item identities. Ten findings, scanner invocations, attempts or child
Runs do not satisfy this rule. If fewer than ten candidates remain, select all
and expose `fewer_candidates_remaining`. If none remain, close selection with
`no_remaining_candidates`; create neither a ranking Run nor an empty Round.

V1 supports a supplied `custom-checklist` inventory. Findings are context,
not additional candidates. Source analysis, generated OpenAPI, standard-package
expansion, discovery-driven candidate insertion, arbitrary model-authored
checks and autonomous pentest are separate capabilities. Scans and code-analysis
checks can both execute selected items through existing profile-owned Workflows.

This is an Audit selection capability. `scan-plan@1` remains the model-free
planner of concrete scanner jobs. The narrower optional scanner ranking in
[V55-009](../../tasks/v55/v55-009-scan-candidate-ranking.yml) is independent.

## 2. Ownership and prerequisite contracts

| Owner | Responsibility |
| --- | --- |
| Checklist importer | Finite full inventory, stable identities and task packages |
| Audit Controller | Frozen context/cycle, ordinary ranking Run, selection acceptance, rounds and coverage |
| `audit-priority@1` Planner | Independent bounded model calls, strict verdict validation and durable evaluation journal |
| Pure selector | Reproducible ordering and selection from complete validated verdicts |
| Scheduler | Ranking/check Run admission, global queue, concurrency, deadlines and claims |
| Existing check Workers | Execute accepted checks and publish ordinary Audit results |

Reuse the shared preparation contracts, persistence foundations and control
paths for Audit work before a Round exists
([19 §10.1](19-audits.md#101-preparation-inventory-and-controls-before-the-first-round)).
Prioritization adds a `prioritization` execution role and phase at the initial
and later round boundary; it does not duplicate the `prepare` producer or
redefine its once-per-Audit semantics. Contract work MUST name and test these
extension points. The prioritization feature itself does not require
generated-input, OpenAPI, retained-dependency or routing inputs. A later
prepared-checklist bridge must explicitly depend on prepared inventory
([19 §4.4](19-audits.md#44-preparation-contract)) instead of implying
that it already works.

Verification: integration depends on the preparation contract, store and
controller tasks
([V62-001](../../tasks/v62/v62-001-audit-composition-contracts.yml),
[V62-002](../../tasks/v62/v62-002-audit-preparation-store.yml),
[V62-003](../../tasks/v62/v62-003-audit-preparation-controller.yml)).

A new opt-in profile declares a pinned ranking Workflow, its resolved model
route and finite ranking policy, plus existing check roles, limits and interaction
policy. The authoring field `prioritization` is reserved; the current profile
schema rejects it. Unknown/incomplete capabilities fail compatibility
checks. No existing profile is opted in by default; current non-priority
execution and review behavior remains supported.

This specification follows the Audit composition
[schema policy](../plans/2026-09-20-audit-workflow-composition.md#current-schema-policy).
Prioritization MUST use the explicit inventory `source`/`settings` mappings of
[19 §4.4](19-audits.md#44-preparation-contract) and reject
obsolete fields and persisted profile snapshots without compatibility readers,
aliases or implicit conversion. Repository profiles, fixtures, public contracts
and clients move together to that schema. Canonical bytes/digests may change
with the schema; deterministic current-schema round trips remain required.
Accepted baseline, Run and Artifact snapshots stay immutable, and current-schema
Audits retain recovery, retention and deletion guarantees. Storage upgrade tests
use current-schema snapshot fixtures; preserving historical schema readability
is not an acceptance requirement.

## 3. Full inventory, candidates and pass identity

The full inventory is retained at Audit scope through protected ProjectScope
artifacts and durable Audit relations. A candidate ID binds inventory digest,
checklist item key and version. Keys are validated for uniqueness by the trusted
importer; matching titles never merge distinct items. Original source ref,
version, task package and configured Workflow role remain exact and immutable.

Candidates are not pre-created as executable AuditItems. Acceptance atomically
binds the selected candidates to a new Round and creates their AuditItems.
Deferred candidates consume inventory/retention space, but do not consume
`maxItemsTotal`, `maxItemsPerRound`, check attempts or a Round barrier slot.
The original complete inventory remains the coverage denominator.

The opt-in path MUST separate inventory capacity from execution capacity in
profile validation, preview, start and retention. A 100-item checklist with
`topN: 10` and `maxItemsPerRound: 10` is valid when other budgets suffice. Existing
non-priority profiles keep their full-inventory admission checks.

Remaining candidates are inventory members never admitted into an accepted
Round. Once admitted, their ordinary retries/review/settlement remain owned by
that Round; a failed or inconclusive check does not silently re-enter ranking.
Rechecking settled candidates and introducing new checklist versions are outside
v1. New findings affect context, not candidate membership.

A cycle has a stable Audit-local ID and ordinal independent of its child Run's
Stage retry identity. It binds the remaining-candidate digest, source inventory,
context ref/digest, rubric/prompt version, policy and exact resolved model route.
There is at most one active cycle per Audit and one accepted Round per cycle.
Both successful model outputs and selection outcomes are immutable history.

The profile declares `prioritization.topN` (default 10) and `maxTopN`
(default 1,000, range 10–1,000, no smaller than its default). Draft/Start accepts
an optional `prioritization: {topN: N}` override; omission uses the profile
default. Reject unknown override fields, non-integers, N below 10 or above the
profile ceiling. Preview applies the same resolution. The effective value is
pinned in the baseline, cycle and selection digest and included in start
idempotency comparison. Reusing an idempotency key with a different topN conflicts.
An active/paused Audit cannot edit it; resume/deadline changes preserve it. A new
Audit can choose a new value. Admission checks the actual `min(topN, R)` against
execution limits, including the all-if-fewer case.

## 4. Context snapshot and independent input

Before a cycle starts, retain one bounded snapshot containing:

- Optional exact service-description artifact, or an explicit absent marker.
- Optional supplied findings collection, with exact source revisions; finding
  IDs, proposal versus confirmed status, analyst decision revision if available,
  summary and evidence pointers remain distinguishable.
- At later passes, accepted check results and Audit findings settled before the
  prior Round barrier, with exact receipt/result/analyst revisions as of capture.
- Original scope/profile references and available task metadata; absence of a
  description or findings is a valid input, not an assumed safe service.

The source set is declared by the profile and supplied inputs. Do not search
unrelated Project artifacts, use mutable `latest` bindings or silently import
all Project findings. Audit findings included in later context are read at a
consistent revision; a concurrent review update causes retry of snapshot capture,
not a mixture of old and new rows. Source Run deletion does not remove retained
context or break provenance; Audit deletion releases the corresponding holds.

V1 sends the same complete bounded common context to every item. It does not
add per-item model retrieval or summaries from other evaluations. Each request
contains system rubric, common context and exactly one checklist statement,
applicability text, allowed methods and required evidence. No sibling verdict,
conversation transcript, mutable memory or previous evaluation is carried over.
Common context may be prepared once and reused as immutable bytes.

Accept service description as UTF-8 text and supplied findings through the
existing strict findings-collection codec; the context artifact stores a bounded
projection and exact source lineage. Raw HTTP bodies, authentication headers,
cookies and credential settings are not model input fields. Narrative text can
contain private data: only owner-selected scoped input is sent to the pinned
model route, and public events/diagnostics never echo it. Descriptions and finding
text are untrusted data, not instructions or authority to change the candidate set.

Oversized descriptions/collections or context that cannot fit the complete
projection produce `priority_context_limit_exceeded` before model dispatch.
There is no silent truncation or arbitrary subset of findings in v1. Snapshot
replacement during an active pass is not supported. Newly arriving results or
review changes are eligible only for a subsequent cycle.

## 5. Verdict and comparison rubric

Model output is one strict object, not prose surrounding JSON:

```json
{
  "item_key": "AUTHZ-07",
  "priority": "high",
  "confidence": "medium",
  "rationale": "Existing cross-tenant access reports make this check useful.",
  "evidence_ids": ["finding-3"],
  "missing_context": ["The tenant isolation design is not supplied."]
}
```

Server supplies cycle/candidate IDs, context and input digests, model-policy
provenance, call identity and usage. The model only echoes the assigned item key;
foreign/duplicate/extra item output invalidates that evaluation. Unknown fields,
unknown evidence IDs, invalid enums, duplicate JSON keys, oversized values, tool
calls and truncated/incomplete provider output are rejected. Evidence IDs refer
only to the exact context entries supplied to that item. Free-text reasoning is
an explanation, not accepted evidence of a vulnerability or applicability.

One versioned rubric applies to every item and pass of the pinned profile:

| Priority | Anchored interpretation |
| --- | --- |
| critical | Available service facts point to a potentially severe, directly relevant issue needing urgent verification |
| high | Relevant sensitive behavior or supporting observations make this check materially useful now |
| medium | Applicable/general checklist coverage with no stronger prioritizing evidence; the baseline when context is absent |
| low | Concrete context supports lower usefulness in this pass; explain that fact |

Relevance, potential impact, supporting observations and what a new check adds
to existing results drive the assessment. Finding severity is an input fact,
never overwritten by a priority verdict. Confidence (`low | medium | high`) is
reported separately and does not change sorting. Missing context alone MUST NOT
lower priority below the rubric's baseline; applicability is advisory and cannot
remove a candidate or create `not-applicable` coverage.

Sort by priority descending (`critical`, `high`, `medium`, `low`), then canonical
candidate ID ascending. Same input, accepted verdicts and policy reproduce the
same order and selection digest. Model calls themselves are not asserted to be
deterministic. Tests cover both mechanics and scripted rubric examples; they do
not claim live-model prioritization quality.

### Pure core boundary

`internal/auditpriority` has no gateway, storage, Scheduler or Worker access.
Its schema and fixtures are in `api/audit-priority/v1`. The model object requires
all six fields above with exact case and non-null values; empty arrays are `[]`.
Unknown/duplicate fields, trailing data, invalid UTF-8 and unpaired escaped
surrogates fail. A legitimately encoded U+FFFD remains valid text. Item/evidence
IDs are ASCII `[A-Za-z0-9][A-Za-z0-9._:-]*`, at most 160 bytes. Rationale and
missing-context strings are nonblank UTF-8 without NUL. Evidence references and
missing-context strings must be unique within their respective arrays. Both raw
response bytes and canonical programmatic verdicts respect the 8 KiB bound;
the other response limits in section 6 apply before selection as well.

Candidate identity is `priority-candidate-` followed by the lowercase SHA-256
hex digest of RFC 8785 canonical JSON containing `schema` equal to
`contractor.audit.priority-candidate-id.v1`, `inventory_digest`, `item_key` and
`item_version`. Digests use `sha256:` plus 64 lowercase hex characters. Item
version is nonblank UTF-8 without NUL, at most 160 bytes for this opt-in core;
current checklist version limits are unchanged. Duplicate keys, including two
versions of the same key, are rejected. The pure pool has schema
`contractor.audit.priority-candidates.v1`, the source inventory digest and
candidate-ID-sorted identity rows; its canonical digest binds exact remaining
membership. This identity pool is not the metadata-bearing Run input.

The caller binds each verdict to the cycle ID, inventory/pool/context/policy/
prompt/model-configuration digests and effective topN. Selection requires exact
agreement on the entire binding, all candidates exactly once and evidence
membership in the caller-supplied retained-context whitelist (at most 100 IDs).
The caller is responsible for proving that the whitelist and input digests
actually belong to that retained context and accepted model Run. Pure validation
does not authenticate them or prove receipt/journal authority.

The internal calculation has schema `contractor.audit.priority-selection.v1`,
the complete binding, ordered candidate/verdict rows, one-based rank, selected
boolean, row reason code, selected/deferred counts and overall reason code.
Deferred rows use `deferred_top_n`; selected rows have an empty reason code.
Rationale, original priority, rank and the binding's topN allow a truthful cutoff
explanation without overwriting priority. Fewer-than-topN and empty remaining
pools use `fewer_candidates_remaining` and `no_remaining_candidates` respectively;
the Audit layer must still reject an empty original inventory. Canonical result
bytes are bounded by 16 MiB. Results detach all mutable slices from inputs;
confidence and input response order never break priority ties.

`Selection.Validate` and `MarshalSelection` check internal consistency only.
They are not the accepted `contractor.audit.priorities.v1` Run output or an
admission receipt. Persisted provenance, accepted output/journal comparison and
atomic budget/selection admission (sections 6–8) are required before the result
can authorize a Round. The pure package registers no planner or profile
capability and changes no existing API or canonical contract.

Verification: the pure core is recorded in
[V64-000](../../tasks/v64/v64-000-audit-priority-core.yml).

## 6. Evaluation execution, bounds and accounting

One cycle creates one ordinary audit-managed ranking WorkflowRun through the
trusted Run service. Its new `audit-priority@1` Stage has exactly zero logical
Workers and direct, tool-free access to the existing model gateway adapter.
No Runtime allocation, ADK agent conversation, model tool loop, Worker finalizer
or separate external ranking service is required. Each item is one independent
logical and physical model call. V1 evaluates sequentially in candidate-ID order;
bounded parallel evaluation is a later optimization, not a new Scheduler pool.

The Workflow declares required exact RunScope inputs `candidates` and `context`
and one required output `priorities` with media type
`application/vnd.contractor.audit-priorities+json`, schema
`contractor.audit.priorities.v1`. The candidate input includes only trusted
checklist metadata/identity; complete per-item tasks and target credentials stay
outside the ranking Run's input set. The typed model adapter reads only these
named authorized inputs and the pinned rubric, not arbitrary ProjectScope data.

For a complete cycle, the Planner creates a bounded canonical result containing
cycle/candidate/context/policy digests and exactly one journal-backed verdict ref
per remaining candidate, using a deterministic create-only RunScope binding.
It returns that exact declared artifact through normal StageResult completion;
Scheduler finalizes and freezes the Workflow output. Only the Audit-associated
successful terminal Run/version and its frozen output can produce a complete
ranking receipt. Audit collection validates full membership and equality with
the accepted per-item journal, retains the output in protected Audit-managed
storage and commits the receipt once. A model-authored success flag, arbitrary
artifact, journal row alone or mutable output binding cannot authorize selection.
Incomplete/failed Runs keep their partial journal for diagnosis but have no
complete ranking receipt. Lost completion/collection acknowledgements replay
against the same Run/output/receipt; check dispatch never precedes receipt
acceptance. Child Run deletion is fenced until required output retention/receipt
collection completes, using existing Audit child-Run deletion rules.

This requires explicit support, not an empty-Agent loophole: configuration,
resolved snapshots, execution policy, preparation, allocation reservation,
recovery, termination, execution reports and readiness MUST support this one
planner-only shape. Existing streamline/router/passthrough/scan planner cardinality
rules remain unchanged. A running priority Stage with no allocations is valid;
other running Stages still detect lost control-plane state. The ranking Run
uses a global queue/concurrency slot and honours queue pause even with no Runtime
connected. Model credentials stay in memory and use existing pinned resolution.

| Bound | V1 rule |
| --- | --- |
| Full candidate inventory | 1–1,000 unique items; empty inventory is explicit and does not call a model |
| `topN` | Default 10, range 10–1,000 |
| Context artifact | At most 128 KiB canonical JSON; model common-context projection at most 24 KiB UTF-8 |
| Service description | At most 8 KiB UTF-8 within common context |
| Finding/result context entries | At most 100 combined, complete bounded projection or explicit rejection |
| Per-item model request | At most 48 KiB encoded request including rubric/common context/item |
| Per-item response | At most 8 KiB, rationale at most 2 KiB, 16 evidence IDs and 16 missing-context strings of at most 256 bytes each |
| Model calls | At most one per remaining item per cycle, no automatic retry |
| Per-call time | Profile value 1–120 seconds, bounded by remaining Run/Stage/cycle deadline |
| Output tokens | Profile value 128–2,048, also within pinned ModelPolicy |
| Cycle model/token/time budget | Explicit positive finite policy; all calls and provider-reported total tokens accounted |
| Stored ranking result | At most 16 MiB, separate paginated projections; no model input contains this full artifact |

The 1,000-item bound deliberately fits the current ModelPolicy call limit. It
is separate from the existing broader inventory ceiling. Reject overflow before
ranking instead of evaluating an invisible prefix. An enabled profile must
provide enough call budget for its current candidate count; zero-call or absent
model access is an explicit ranking failure, not a low priority score.

Before starting ranking, atomically create a cycle-owned count reservation for
one ranking Run, `min(topN, R)` initial check Runs and that many item slots. A
count reservation does not create or consume AuditItems. It reduces available
capacity for competing admissions. Converting the accepted selection atomically
assigns these counts to its exact candidate/Run identities without charging twice;
ordinary submission converts its own reserved slot to consumed, rather than
requiring another unreserved slot. Duplicate acknowledgement/reconciliation is
idempotent. Pre-acceptance failure/cancel releases only unused reservations;
submitted ranking/check Runs and unknown external calls remain charged.

Reserve ranking Run overhead and capacity for the intended check subset before
starting ranking. Count ranking Runs in `maxSubmittedRuns`; selected distinct
items count in item budgets. Reserve a conservative one-check-Run-per-selected-item
allowance; batching may release excess after the Round settles. This can reject
an otherwise batchable but underfunded profile and is an explicit v1 bound.
Configured discovery/assessment and mandatory follow-up work must also fit
reserved capacity; v1 profile omits discovery and assessment roles. Retries use
remaining unreserved Audit capacity and never consume capacity already reserved
for other selected initial check attempts. No existing hard network request or
cost ceiling is implied by this accounting.

Before the first call, construct and validate every candidate's bounded request
and context-window fit, calculate all admission estimates, and reserve their
whole-pool sum within the cycle token policy. An oversized last candidate,
insufficient call count or a known insufficient whole-pool token budget produces
zero gateway calls. The current policy must also cover the complete declared
request count. This avoids spending on a pool already known to be unfinishable.
Before each call, persist its intent and transfer its existing reservation under
the current Scheduler claim, without a second charge. Input/output bounds,
accumulated actual usage and outstanding reservations govern dispatch. The pinned `utf8-bytes-v1` admission estimate is the encoded request byte
length plus configured maximum output tokens. It deliberately over-reserves
ordinary text input and is a local scheduling estimate, not a provider billing
guarantee. Require the estimate to fit both the remaining token budget and the
configured context window; no new tokenizer service is required. Record provider input/output/total usage separately, including
unknown usage after a lost response. Unknown usage retains its reservation;
known overruns are recorded honestly and stop further calls. Limits are enforced
before dispatch where knowable; provider billing is not guaranteed by a local
estimate. Responses never grant authority to increase a budget.

## 7. Persistence, failure and recovery

Persist independent states per evaluation: `pending`, `started`, `succeeded`,
`invalid`, `failed`, `unknown`, `cancelled`, `not_attempted`. A persisted intent
is required before a provider request. One accepted verdict is immutable for
`(cycle_id, candidate_id)`; it includes exact context/item/prompt/model digests,
resolved model reference/version when returned, usage and bounded rationale.
Persisted call state is fenced by both the current ranking Run claim and the
Audit's active cycle association. A stale process cannot settle a new cycle.

Known completed verdicts are reused after process restart. A committed `started`
intent with no durable response becomes `unknown`; it is never automatically
sent again. Lost acknowledgement of a verdict commit is resolved by reading the
exact journal row. A request that may not have reached the provider still keeps
its conservative unknown accounting. An ordinary Run retry/escalation cannot
reset the cycle journal or mint a new cycle; v1 ranking Workflow has no retry or
escalation path. A new Audit is an explicit new budget and evaluation scope.

For v1, **all remaining candidates must have valid verdicts before ranked
selection**. The first invalid/failed/unknown evaluation stops new calls, records
remaining entries as `not_attempted`, and leaves the cycle `ranking_incomplete`.
Retain successful verdicts for diagnosis. Do not mix them with invented low
scores, silently select a partial top-N, or label deterministic fallback as the
highest model priorities. Automatic fallback modes and manual reranking of a
failed cycle are deferred. Existing non-priority Audit profiles remain available
for deterministic execution without ranking.

Cancellation, Run/Stage/cycle deadline, Project/Audit deletion and lost
ownership stop new calls and check dispatch. Queue pause prevents new Run/Stage admissions; an admitted
ranking Stage may drain under the existing queue contract. Audit pause and Audit admission-deadline exhaustion prevent new child Runs and
Round acceptance while an admitted ranking Run may complete and its results may
be retained. Audit admission deadline is the existing pause gate: it does not
shorten the ranking Stage deadline or terminate an admitted model call. Existing
resume may extend or disable that admission deadline. Resume reuses the same cycle, context and outcomes.
Late output after cancellation/deletion is non-authoritative. These controls
also work before any Round exists, through the shared preparation-phase
lifecycle ([19 §10.1](19-audits.md#101-preparation-inventory-and-controls-before-the-first-round)).

## 8. Selection, readiness and subsequent passes

Only a complete validated ranking receipt over the exact remaining set is
selectable.
Server rechecks complete membership, output digests, expected Audit revision,
cycle ownership and budgets in a fenced acceptance operation. Stage immutable selection bytes first, then commit their exact retained
reference/ownership, Round/Items and candidate-consumption records in one
acceptance transaction. Staging an artifact alone cannot authorize work; replay
recovers equal bytes and garbage collection can remove unaccepted staging refs.

Selection contains every remaining candidate's rank, priority, confidence,
verdict reference and disposition. Selected items receive immutable ordinals in
rank order. Deferred items retain priority and `deferred_top_n` with their rank,
configured limit and selected count. For example, item ranked 11 may stay `high`
while deferred behind ten other items. There is no score modification to justify
its exclusion from this pass.

Readiness is separate: an item requiring existing approval can be selected and
then await approval. It is not removed from the top-N, and lower-ranked work is
not backfilled. Missing check capability follows existing preparation failure;
it cannot silently replace a selected candidate. Successful selection is not a
promise of ten successful checks: execution may later fail, time out or be
cancelled, with distinct selected/started/completed counts.

If budgets cannot admit exactly `min(topN, R)`, record
`selection_blocked_budget` and accept no smaller Round. Preflight catches known
shortfalls before model calls; acceptance rechecks remaining capacity. Context
failure, incomplete ranking or blocked selection stops automatic progression
with an explicit gap/stop reason and releases unused reservations. There is no
indefinite hidden retry loop. Terminal mapping is explicit, using the shared no-Round cleanup/report path:

| Cause | Audit outcome after child Runs drain |
| --- | --- |
| Empty original checklist | `failed`, `empty_inventory`; no Round and no model calls |
| Invalid/oversized context, inaccessible pinned model route, incomplete ranking | `failed` with the concrete priority failure code; retain all known inventory/results |
| Full selected subset cannot fit admission budget | `failed`, `selection_blocked_budget`; no undersized Round |
| Cancel | `cancelled`, existing cancellation reason |
| Audit admission deadline | `paused`, `deadline_exhausted`; running Runs drain, existing resume can extend/disable the deadline |
| Ranking Run/Stage/cycle deadline | `failed`, `priority_ranking_deadline_exceeded`; incomplete journal retained |
| No remaining candidates after a settled Round | Normal existing report-finalization/review path to `completed` |
| Configured maxRounds reached after a settled Round | Same bounded finalization path, with remaining candidates explicitly not-tested |
| Pause | `paused`, nonterminal; resume uses the same snapshot/journal |

A terminal failed/cancelled Audit still exposes a retained coverage/stop snapshot;
this diagnostic projection is not an owner-accepted successful report. Any earlier
accepted report/result revisions remain immutable. No review decision is invented
merely to close a ranking failure.

After every selected item settles, an enabled later pass may rank remaining
candidates against a fresh context snapshot, within existing `maxRounds`, total
items, submissions and deadlines. The supplied default profile pins one Round. A separate named/versioned
example profile pins maxRounds=3 for the multi-pass journey; neither creation
nor resume silently edits the profile round limit. Deferred-checklist
continuation is independent of finding confirmation and does not enter the
finding-proposal selection path. When enabled, remaining baseline candidates
have their own deterministic next-pass path. Finding proposals never become
new candidates implicitly. No remaining candidates means normal bounded closure;
remaining candidates at final closure stay visible as untested coverage.

## 9. Public API, UI, coverage and retention

Expose candidate identity, source, latest pass evaluation and immutable history,
rank/priority/confidence/rationale, selection disposition/reason, selected Round
and ordinary execution/approval state through bounded revision-consistent pages.
All cursors bind the Audit/cycle snapshot. Counts declare their scope:

- Audit totals count unique original candidates: inventory, ever evaluated,
  ever selected, remaining never-selected, ever started, settled and candidates
  with an accepted check result. Re-evaluation never increments a unique total.
- Cycle totals count its frozen remaining pool: evaluation outcomes, selected,
  deferred, awaiting approval, started, settled and accepted-result candidates.
- `started` requires an actual check invocation; `settled` includes explicit
  rejection/cancellation without invocation. An accepted result remains separate
  from technical completion and from a passing semantic finding assessment.

For 25 original candidates and two passes of 10, Audit totals are 20 selected
and five remaining; the second cycle evaluates 15, selects ten and defers five.
An approval rejection can increase settled while leaving started/accepted-result
unchanged. API labels use these exact meanings rather than ambiguous completed
or evaluated counters. Use optional exact
input selectors for description and findings; omission is meaningful. The Draft/Start
form exposes the bounded topN override defined in section 3 and shows the pinned
value read-only after start.

Audit coverage and machine/human reports include every original candidate across
all passes. Deferred items are `not-tested`; their presence cannot shrink the
denominator, become `not-applicable`, or imply a passing check. Earlier accepted
results keep their ordinary semantic status even after later rounds. Priority,
finding severity, analyst judgement and execution status remain separate.
No-Round failure/cancellation still reports the known inventory and its gaps.

Retain exact context, verdict, selection and result/decision references across
permitted child/source Run deletion. Subsequent analyst edits cannot rewrite a
published selection or report. Owner isolation, active holds, protected bindings,
CAS, deletion and safe error/redaction rules follow existing Audit/Artifact
contracts. Public events contain bounded phase/count/error codes; rationale and
context are accessed through authorized artifact/detail reads and rendered as
untrusted text. Reports and UI must show incomplete ranking distinctly from a
fully ranked top-N.

## 10. Required verification

The implementation gate uses scripted model responses, disposable PostgreSQL,
production Server/Scheduler boundaries and controlled check Workers. No paid
model or external target is required. Mandatory cases include:

- Counts 0/3/9/10/11/100; `topN=9` rejection, `topN=12`, and all-if-fewer.
- 100 inventory items with maxItemsPerRound 10; full denominator and exactly ten
  accepted items, with no deferred item budget consumption.
- Identical scores, stable tie order, high-priority position 11 retaining `high`,
  absent optional context, and advisory non-applicability retaining membership.
- One malformed/foreign-ID/tool-call/truncated verdict prevents ranked selection;
  no-context alone does not lower priority and no failure becomes a fabricated score. Oversized late-candidate input or
  insufficient known whole-pool token budget gives zero model calls.
- Check capacity three with desired ten gives explicit blocked selection, no
  three-item Round; ranking overhead and retry/batch reservations are accounted. Eleven submission
  slots permit one ranking Run plus ten one-item checks, without double charging
  on acceptance/replay or loss of reservations to another retry.
- Selected high-priority item awaiting approval; no substitution or weakened gate.
- Restart at intent/send/result/selection boundaries, lost commit acknowledgements,
  two Controllers/claims, whole-Run retry attempts and actual provider call counts.
  Reject tampered aggregate output, wrong Run/cycle/source revisions and output
  collection/deletion races before selection.
- Pause/cancel/deadline/delete before Round1 and between rounds; Audit admission
  expiry drains ranking then resume reuses its receipt, while Run/Stage timeout
  yields incomplete ranking; unknown usage,
  ownership loss, source Run deletion, stale pagination and context review races.
- Ranking without any Runtime connected still respects global queue/concurrency;
  existing Worker-backed Stages retain allocation-loss detection.
- Two passes of 10 from 25 leave five not-tested after maxRounds=2, use refreshed
  context, keep prior verdicts, and work with finding confirmation disabled.
- Browser upload/configuration/ranking/progress/selection/deferred explanation/
  incomplete recovery and full-inventory report against production public API.
- Current-schema non-priority profiles retain execution/review behavior and
  deterministic canonicalization; obsolete profile fields and snapshots are
  rejected. Current public clients and storage upgrade/recovery fixtures pass
  without compatibility readers. Tests explicitly distinguish planning mechanics
  from model quality.

The release task must register discoverable mandatory process/browser cases,
fail on missing prerequisites/zero selected tests/skips, and retain actual
commands, revisions and results. Creating this document or task files is not
implementation or test evidence.
