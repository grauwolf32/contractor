# 30 — Independent managed Evals and external producers

Status: **Implemented and deterministically verified ([V38-010](../../tasks/v38-010-eval-release-gate.yml)): native setup, comparison/review, optional Playground client and independent process acceptance.**

[User guide](../guides/evals.md) · [Release checks](../testing/evals-release-gate.md)

The [embedded data catalog](../../api/evals/v1/README.md) and
[conformance fixtures](../../api/testdata/evals/README.md) define the closed DTOs
and portable identity mapping. The [EvalStore](../../internal/evalstore/README.md)
persists private records, expected membership, receipts and Project deletion fences.
Native coordination and public endpoints run in Contractor Server; browser setup
and optional external clients use the same protocol.

[Product journey](../evals-experience-design.md) ·
[Portable evaluation format](26-portable-evaluation-format.md) ·
[Projects](17-projects-and-queue.md) · [Audits](19-audits.md) ·
[Lifecycle](18-run-and-workspace-lifecycle-controls.md)

This document owns Contractor's browser-managed experiment service and its
public producer API. Spec 26 remains the owner of portable document identities,
member/pair identity, hidden-truth separation, scoring and comparison semantics.
Its CLI-only publication/recovery boundary is preserved. This is an
explicit additional server authority for new managed experiments, not an upgrade
of an old private CLI journal inferred from labels or public projections.

## Independence and ownership

Contractor has no dependency on a Playground process, Python package, repository,
filesystem, hostname or callback. It builds/starts without Playground installed.
Any authorized client can push datasets, author/start native experiments, or drive
external experiments and submit assessments through the same public API.

| Data / operation | Authority |
| --- | --- |
| Dataset import provenance | Attributed producer assertion; exact imported visible bytes have their own digest |
| Native dataset revision, draft, frozen plan and expected members | Contractor EvalStore in its PostgreSQL database |
| Native experiment dispatch, pause, resume, cancel | Contractor EvalCoordinator under a durable claim |
| External plan and runner journal | External evaluator; server stores its registered safe manifest and source digest |
| Managed member submission, verified Run/Audit association | Common Contractor submission service used by either driver |
| Run/Stage/Worker execution and Audit child ownership | Existing Run/Audit services, Scheduler, Control Plane and Runtime |
| Native registered check / human assessment | Registered evaluator or authenticated reviewer, over an exact result |
| External assessment | Declared producer; validation of shape/provenance is not independent quality verification |
| View generation, complete denominators, pagination | Rebuildable Contractor read model from selected immutable records |

`controlMode` is `server | external`, immutable at creation. It is not selected by
`eval.*` labels. In external mode the native coordinator never dispatches; UI does
not pretend it can resume an unavailable producer. The driver can finalize or
cancel remaining submissions through the generic protocol, without a server-to-
producer request. Another client cannot take over using label matches alone.

The service targets the existing single-VM topology. The coordinator
is a Go Server component with PostgreSQL claims, not a new Worker scheduler. It
limits its own outstanding submissions according to the frozen experiment and
uses the ordinary queue/global concurrency gates. No separate Playground daemon
or arbitrary plugin process is a deployment requirement.

## Entities and portable identity

An evaluation Project contains datasets and experiments, with normal owner and
active/deleting semantics. Use the existing owner principal from authentication;
clients cannot supply an effective ownerId. All referenced Projects/Runs/Audits/
artifacts must be authorized under that principal.

- **Dataset revision:** immutable exact visible case definitions and a source
  identity/revision/digest. Case fields follow case/v2's task, inputs, capabilities
  and output roles. Imports omit private truth by default and retain the original
  private case digest only as producer provenance; the visible revision has a
  different digest. No remote source URL is fetched automatically.
- **Draft:** mutable name, exact dataset/case selection, A/B bindings, input and
  parameter mapping, evaluation policy, repetitions, order and budgets. Saved
  under revision CAS. Editing invalidates prior readiness.
- **Prepared native plan:** the exact portable plan/v1 and reachable case/suite/
  binding resources, materialized by the Go implementation from the draft. Hidden
  review/check material occupies the private evaluator partition, never the Run
  input projection. There is one authoritative frozen plan, no second editable
  public manifest. All later records name its digest.
- **External registration:** a schema-versioned safe plan projection plus exact
  source plan digest, explicit comparison policy/budgets and exact executable
  bindings/mappings supplied for managed submissions. It cannot reconstruct the
  external private plan or truth. Freeze registration once; identify all expected
  members before submitting. Portable input/case digests that cannot be verified
  locally are labelled producer-supplied, not observed equality pins.
- **Member:** identity is spec 26 SHA-256 of
  `[experiment_id,suite_id,case_id,sample,variant_id]`. All expected members are
  retained, including unsupported, blocked and unsubmitted. Sample is one-based.
- **Submission:** immutable member operation, request digest, stable key and
  verified execution receipt; no new sample on retries or response loss.
- **Result/Assessment/Comparison:** exact immutable revisions with explicit
  selection, using spec 26 meanings. A superseding record preserves its predecessor.

New native experiments use a server-allocated spec-26-compatible ID. External
registrations may request their existing portable experiment ID; uniqueness is
`(owner, project, experiment ID)`, with source plan digest immutably bound. A
conflicting digest is 409, not replacement. Public API lookup IDs are opaque
server IDs, separate from the portable experiment ID. A pair ID is the canonical
hash of `[experiment_id,suite_id,case_id,sample]`, not a Run ID.

Managed resources have `revision`/ETag for selection and lifecycle CAS. Content
hashes are SHA-256 over exact retained bytes. Secret values do not enter public
DTOs, logs, cursors or error text. Objects/arrays remain bounded by spec 26;
DTO schema validation is closed, with explicitly versioned extensions only.

## Dataset and variant selection

Native UI supports authoring visible cases with exact uploaded/imported input
artifacts and explicit output/check contracts, or selecting a registered immutable
dataset revision. Creating a subset or changing visible inputs produces a new
revision with provenance; it does not reuse the original case's digest or claim
that its private oracle remains applicable. Selected cases are identical in A/B.

The executable catalog consumes current Workflow/AuditProfile and supported
execution override contracts. The first managed slice uses one exact binding per
arm and matching binding kind; mixed per-case program dispatch remains separate.
AgentTemplate changes require an explicit executable Workflow version/wrapper.
No arbitrary Python scorer, document-named module or shell command is imported.

Registered native evaluators initially include required artifact/media/schema
checks using allowlisted existing validators and `human-review@1`. A validator
reports the stated structural property, not overall semantic correctness. A
native human rubric/check configuration is versioned and pinned separately from
visible inputs. External semantic/domain scorers are supported by attributed
assessment ingestion. No automatic LLM judge is introduced.

Preparation uses read-only resolution and existing capability metadata. It records
available, unavailable and producer-supplied pins honestly. Unknown required-equal
pins block readiness; a policy alias is not an observed model revision. Unsupported
members can remain in a valid manifest as ineligible with reasons; they are never
silently dropped or submitted. Arbitrary target provisioning and private Playground
oracle protocols are not native capabilities of this slice.

### Workflow and Audit parity

Both executable kinds are required in native and external control modes. Kind is
chosen for the experiment; A/B use either two exact Workflow bindings or two exact
AuditProfile bindings. Mixed Workflow-versus-Audit comparisons are outside this
slice and fail preparation/registration explicitly. The common denominator is one
member per `(case,sample,variant)`, not the number of underlying Runs or Audit items.

| Concern | Workflow member | Audit member |
| --- | --- | --- |
| Executable selection | Exact Workflow with resolved dependent versions/settings | Exact AuditProfile with resolved roles, inventory/standards and execution settings |
| Submission receipt | One ordinary Run | One ordinary Audit, with separate durable create/start receipts |
| Workspace | Exact visible case inputs forked into RunScope | Separate owned execution Project of kind `project` per member; the evaluation Project holds only its explicit association |
| Internal work | Stage/Worker attempts and retries | Audit rounds, check/discovery/assessment Runs, retries and child attempts |
| Evaluation progress | One terminal Run contributes one terminal member | One terminal Audit contributes one terminal member, regardless of child/item counts |
| Evidence navigation | Exact Run outputs and diagnostics | Exact Audit report/coverage/findings plus bounded child Run/item drill-down |

The existing Audit service rejects evaluation Projects as execution workspaces;
managed Evals preserves that boundary rather than changing Project kind or using
a special Audit creation bypass. Input, inventory, standards, worklist, importer
and trusted child-Run authority remain in [19](19-audits.md). Private eval truth
never becomes a trusted Audit checklist/task manifest. Audit `completed` maps to
technical execution success, not a passing quality assessment; the selected
external/native/human checks still determine assessment.

For either kind, token measures aggregate one authoritative cumulative snapshot
per unique Run/stage execution across all included roles/retries. For Audit,
resolve all owned child executions through authoritative Audit associations,
not only item labels or the first item page; discovery/assessment work counts too.
The bounded managed member execution-list endpoint exposes these authoritative
associations to external clients as well as the UI. An Audit item list alone is
not a complete execution inventory.
Never add parent totals to child totals or sum successive polling snapshots.
Missing children, truncated reports or unavailable finalizer/role accounting make
the corresponding scope partial/unavailable. Child counts do not change the
expected member or scored denominator. Cached tokens remain a subset of inputs.

The model-free `passthrough@1` Planner reports explicit zero model calls and token
counters. For retained reports that omitted those counters, collection derives
the zeros only from the immutable Stage's exact `passthrough@1` identity. Missing
Worker or model-backed Planner counters remain incomplete. Recollection creates
new result revisions; an existing selected result changes only through explicit
selection, preserving result and assessment history.

`wall_ms` uses the parent execution's `createdAt` to confirmed `finishedAt` for
both Run and Audit, matching the existing Contractor provider interval. It
includes queue/hold time but excludes evaluator preparation/scoring/publication.
Audit wall time is not the sum of child Run durations because children may overlap.
Unavailable timestamps are not replaced by guessed child bounds. Any separately
reported active/service time must have a distinct scope and cannot share the same
comparison series. Progress-chart time is experiment wall time and is labelled
separately from per-member execution duration.

Conformance fixtures and the release gate include all four kind/control combinations,
an Audit with multiple children/rounds/retries, overlapping child durations,
missing child usage and duplicate observation replay. UI labels and drill-down
follow the selected kind; they never suggest an Audit is just one child Run.

## Execution control and recovery

Keep lifecycle distinct from result quality:

`draft → preparing → ready → running → settling → finished`.
A confirmed preparation/configuration failure returns to draft with a retained
safe diagnostic and no Run/Audit. Unknown infrastructure failures keep the prepare
command pending in `preparing`, expose `eval_preparation_unavailable` with `wait`,
and retry through the coordinator. The original internal cause reaches coordinator
logging even when persisting the diagnostic or failed command succeeds. Successful
retry clears the transient diagnostic.
Frozen ready plans cannot be edited. Duplicate is native-only and creates a new
native draft with new identity; it requires another Prepare and Start. An external
producer creates a new portable invocation/registration itself, since Contractor
cannot regenerate its private plan. Unsupported commands return an explicit mode
or state conflict before effects.

Native controls additionally use `pausing → paused → running` and
`cancelling → cancelled`. `interrupted` means the coordinator needs recoverable
operator attention; an ordinary process restart normally reacquires the claim
and reconciles automatically. Finished can have `pass`, `regressions` or
`inconclusive` conclusion; awaiting assessment is separate from execution progress.
Finished means accepted executions have drained, even if human review is still
pending; later assessments can change the comparison without restarting execution.
External registration starts ready; the first accepted submission starts its clock
and moves it to running. Finalize moves it through settling to finished. The API
records `lastProducerActivityAt` on accepted producer operations. An inactive
producer is a freshness observation, not proof of failure or completion; Run/Audit
states remain authoritative. There is no inferred heartbeat or server callback.

Every mutating request uses a stable `Idempotency-Key`; mutable state transitions
also require `If-Match`. Persist command intent before side effects and return a
command receipt. Same key/body replays the receipt even after revision advances,
including a retry that races the original request; changed body, plan digest or
requested revision under that key is 409. Validate
ownership/current deletion fence before accepting a new command. Missing CAS is
428, stale CAS 412. A timed-out response is not an unaccepted command.

Native coordinator claims use an expiring owner/epoch lease. Durable intents and
execution keys, rather than a lease alone, prevent duplicate side effects. A
replacement first reconciles all uncertain submissions. Dispatch verifies the
current control state/plan and records its intent atomically; a stale holder
cannot commit transitions. Any already committed intent can still need remote
reconciliation after Pause/Cancel was requested and remains counted as outstanding.

- Start initializes deadline and accounting once and dispatches in frozen order.
- Pause fences new intents, lets accepted work settle and resolves uncertain
  submissions. Paused requires no outstanding submission/execution. A closed
  browser does not pause an experiment.
- Resume uses the same plan, IDs, deadline and token high-water marks. It never
  resets limits or reruns a terminal member. Only paused/interrupted native
  experiments are resumable; expired remaining budget completes pending members
  as not submitted with the budget reason.
- Cancel fences new intents, requests normal Run/Audit cancellation and remains
  cancelling until confirmed drain. Unknown outcomes stay unknown; pending
  members retain their original denominator and not-submitted reason.
- A terminal attempt is not resampled. Duplicate/new experiment is explicit.
  Reassessment can run without new model execution.

Concurrency counts whole active members, including uncertain submissions and an
Audit's parent lifetime; its child Runs remain the Audit controller's work. Global
Run concurrency and Audit limits remain unchanged. Token thresholds account for
known cumulative usage once, include retries/planners/workers/finalizers when
available, and expose missing scope. A reached threshold stops new work and
requests cancellation according to the frozen policy; active work may overshoot.
Deadline and token enforcement remains active through `settling` and recovery
until all accepted executions have drained. Entering `settling` only closes new
admission; it does not release the frozen allowance for active Runs/Audits.
Pause does not extend wall time. Limits are explicit, not new global defaults.

### Common managed submissions

`POST .../members/{memberId}/submissions` resolves the frozen member recipe and
creates its ordinary Run/Audit through existing domain services. The native
coordinator calls the same service internally. External callers choose when to
submit, within eligibility, deadline, total/concurrency limits and finalization
fences; they cannot replace the member recipe in the request. The first external
submission initializes the immutable deadline/accounting exactly once under the
same transaction that checks admission; concurrent requests cannot exceed the
frozen in-flight allowance. A later retry never resets that clock.

Persist the exact creation intent/key before mutation; replay returns the same
Run ID or Audit ID. Audit create and start use distinct durable suboperations and
retain both receipts. A crash between them resumes the same Audit. No synthetic
child Run, checklist truth substitution or bypass of importer authority is allowed.
Normal Run creation retains immutable input/Workflow/Runtime snapshots and labels.
Verified member association is durable before it is exposed as successful.

Case input/output mappings name the case role as key and executable slot as
value. Unmapped input roles retain their names; collisions are invalid. Variant
parameters override case task parameters; `$task.objective` substitutes the
visible objective. Workflow variants use ordinary execution overrides. Audit
variants select exact AuditProfiles and cannot supply execution overrides until
the ordinary Audit service supports them.

Managed submission keys are scoped by experiment lookup identity and member ID,
including distinct keys for Audit creation and start. This prevents collisions
between Projects sharing a portable invocation ID without changing the frozen
portable CLI key format. Local binding snapshots retained during external
registration are separate from the producer's attributed binding hashes.

An external runner may finalize only after its intended submissions have been
accounted for and all accepted operations are reconciled. Finalization fences
new submissions, represents the rest as not submitted and drains/observes accepted
work; it cannot claim remote termination. API cancellation of an external managed
experiment also fences future submissions and cancels known accepted executions,
but must not claim that an arbitrary external process/target has been stopped.
The UI reserves experiment control for native mode; external clients can use these
protocol controls and UI retains normal individual Run/Audit actions.

## Public API contract

Endpoints use the same
public authentication, cookie CSRF, origin, request-ID, error-envelope and method/
HEAD rejection conventions as existing handlers. No browser-to-Playground calls.
JSON uses camelCase DTO fields; retained portable documents keep their own schema.

| Method/path | Request / result |
| --- | --- |
| `GET /v1/eval-capabilities` | Safe modes, check IDs, import versions and paginated exact Workflow/AuditProfile selectors; optional `kind` filter; no host paths or tokens |
| `GET /v1/projects/{projectId}/eval-datasets` | Paginated dataset identities/revisions, safe provenance and case counts |
| `POST /v1/projects/{projectId}/eval-datasets` | `datasetId`, `name`, optional safe `source`, visible `cases`, private check/rubric partition; creates immutable revision |
| `GET /v1/projects/{projectId}/eval-datasets/{datasetId}/revisions/{revision}/cases` | Bounded visible case projections for setup; never private expected data |
| `POST /v1/projects/{projectId}/eval-experiments` | `name`, `controlMode`, native `draft` or external `registration`; allocates experiment lookup ID and draft/registration receipt |
| `GET /v1/eval-experiments` | Owner-scoped paginated experiment summaries, filter `projectId`, `state`, `datasetId`, `controlMode` |
| `GET /v1/eval-experiments/{id}` | Lifecycle, plan identity, control mode, capabilities, complete summary, selected view snapshot and ETag |
| `PATCH /v1/eval-experiments/{id}` | Full replacement `name,draft` under CAS; only mutable native drafts |
| `POST /v1/eval-experiments/{id}/commands` | `kind`, expected `planSha256` where frozen; kinds `prepare,start,pause,resume,cancel,finalize,duplicate`; 202 receipt or 201 new draft for duplicate |
| `GET /v1/eval-experiments/{id}/commands/{commandId}` | Durable accepted/running/completed/failed receipt and safe reason |
| `POST /v1/eval-experiments/{id}/members/{memberId}/submissions` | `planSha256`; managed replay-safe submission receipt; server mode rejects external dispatch |
| `GET /v1/eval-experiments/{id}/members` | Expected member page including zero/one verified receipt, state, result/assessment selection and conflicts; accepts a bound chart bin filter token |
| `GET /v1/eval-experiments/{id}/members/{memberId}/executions` | Owner-scoped paginated parent/child execution refs with kind, role/round, exact association, observed status and inventory completeness; all Audit roles, no private inputs |
| `POST /v1/eval-experiments/{id}/members/{memberId}/results` | Versioned external collection/result with exact execution/artifact refs; cannot override authoritative execution state or server usage |
| `GET /v1/eval-experiments/{id}/members/{memberId}/review` | Owner-only bounded review context: exact selected result, pinned rubric/check settings and evidence availability; omitted from execution and safe reports |
| `POST /v1/eval-experiments/{id}/members/{memberId}/assessments` | Registered native-check request, human decision or external record; exact result digest, check pins, actor/producer and predecessor |
| `POST /v1/eval-experiments/{id}/selections` | CAS selection of exact result/assessment refs; creates new view generation, never edits old records |
| `GET /v1/eval-experiments/{id}/pairs` | Snapshot-bound case/sample pairs and per-dimension completeness; full summary attached; accepts a bound chart bin filter token |
| `GET /v1/eval-experiments/{id}/pairs/{pairId}` | Exact A/B selected records, safe artifact/Run/Audit links and exclusion reasons |
| `GET /v1/eval-experiments/{id}/charts/{chart}` | Bounded selected-snapshot chart data; chart is `quality`, `tokens`, `duration`, `pair-deltas` or `progress`; scope, units and coverage are explicit |
| `GET /v1/eval-experiments/{id}/report` | Safe versioned JSON or Markdown export of one complete selected view snapshot |
| `DELETE /v1/eval-experiments/{id}` | CAS, durable fence/drain/purge receipt; see retention |

Experiment collection rows use `ExperimentSummary`: identity, name, Project,
control/execution kind, state, revision, expected count and update time. They do
not include draft/setup or private case documents. Create, draft update,
duplicate and deletion return a stable `ExperimentReceipt` with experiment ID,
accepted revision and state. Read the detail separately for its current state;
an exact replay never silently substitutes a later revision for that receipt.
Command POST returns its accepted receipt, while command GET observes completion.
Submission POST acknowledges the durable request; the member view exposes an
execution only after the common submission service verifies its association.
Retained owner-scoped mutation receipts remain replayable after experiment purge
until the owning Project is purged.

Dataset revisions are append-only; another POST with the same datasetId and a new
idempotency key creates a new revision. A missing body/field, unknown key, duplicate
case/variant/member, invalid digest, invalid role mapping or incompatible media is
422 before side effects.

### Native setup and start example

The following abbreviated case IDs/artifact refs stand for already registered
exact revisions; hashes in real requests are full SHA-256 values.

```http
POST /v1/projects/evaluation-1/eval-experiments
Idempotency-Key: ui-create-trace-1
Content-Type: application/json

{
  "name": "Trace instructions",
  "controlMode": "server",
  "draft": {
    "dataset": {"id": "trace-small", "revision": "r1"},
    "caseIds": ["unsafe-query", "safe-query"],
    "variants": [
      {"id": "a", "kind": "workflow", "selector": "trace-baseline@1", "executionConfig": {}},
      {"id": "b", "kind": "workflow", "selector": "trace-candidate@1", "executionConfig": {}}
    ],
    "repetitions": 2,
    "order": {"kind": "alternating"},
    "checks": [{"id": "evidence-review", "evaluator": "human-review@1", "rubricRevision": "r1", "required": true}],
    "comparison": {"baseline": "a", "candidate": "b", "gates": {"minCandidateEndToEndPass": 1, "maxQualityDrop": 0}, "requiredEqual": ["source", "tasks", "scorers"], "allowedDifferences": ["instructions"]},
    "budgets": {"maxMembers": 8, "maxInFlight": 1, "wallMs": 5400000, "maxObservedTotalTokens": null}
  }
}
```

201 returns `experimentId`, `portableExperimentId`, `state: draft`, `revision: 1`,
`expectedMembers: 8`, `controlMode: server` and its ETag. These example budgets are
user-entered, not defaults or permission to execute a live model. An actual strict
instruction experiment also requires the declared model/tools/Skills/runtime pins;
the example's small requiredEqual set is not proof those dimensions match.

```http
POST /v1/eval-experiments/experiment-1/commands
If-Match: "1"
Idempotency-Key: ui-prepare-trace-1
Content-Type: application/json

{"kind":"prepare"}
```

202 returns a command ID. GET of the completed command returns the prepared plan
SHA-256, new experiment revision and ready/blocked diagnostics. Prepare submits
zero Runs/Audits. Start names the exact reviewed hash and latest revision:

```http
POST /v1/eval-experiments/experiment-1/commands
If-Match: "2"
Idempotency-Key: ui-start-trace-1
Content-Type: application/json

{"kind":"start","planSha256":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
```

A stale hash/revision returns 409/412 without execution. Accepted 202 includes
`commandId`, `state: accepted`, `experimentRevision` and command resource link.
Replay with the same key/body returns that receipt, not another launch. Recovery
polls the command, then reads the experiment; the browser does not generate a new
key on timeout. Mutable controls use the same request/receipt pattern.

### External registration and assessment example

An external client registers with `controlMode: external` and a versioned
`contractor.eval-registration/v1` envelope containing a safe plan manifest,
`sourcePlanSha256`, safe source identity, exact member binding recipes and frozen
comparison/budgets. The manifest uses spec 26's existing portable public projection
shape (historically named `playground.public-projection/v1`); that schema name
identifies a format, not a required producer or a Playground server dependency.
Native private plan fields cannot be guessed from that projection; required mappings must be
supplied and validated. Registration reports unsupported/unavailable dimensions.
`source.system: playground` is display provenance, never special routing.

The client submits one registered member through the common submission endpoint
with its plan digest and a stable key. The response returns a verified Run/Audit
association. After collecting a result, it can post an assessment such as:

```json
{
  "schemaVersion": "contractor.eval-assessment-input/v1",
  "source": {"kind": "external", "producerId": "local-evaluator", "recordSha256": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
  "resultSha256": "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "checks": [{"id": "evidence-review", "evaluator": "instruction-review@1", "implementationSha256": "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd", "status": "fail", "reason": "Unsupported causal claim", "evidenceRefs": []}],
  "previousAssessmentSha256": null
}
```

The actor is derived from authentication. The server verifies registered check
identity/pins, member/result/plan association, evidence ownership and the declared
schema; missing required evidence marks evidence completeness as incomplete,
regardless of the submitted check verdict. The check verdict remains attributed to the external producer. It cannot rewrite Run state,
replace an assessment in place or become a server-verified semantic judgment.

## Assessments, selection and read model

Evaluation partition data is never included in source packaging or model inputs.
Native rubrics/expected check settings live in owner-scoped private eval storage,
not ordinary browsable Project artifact namespaces inherited by execution.
Native collection resolves the frozen binding's output roles through registered
normalizers and records exact raw evidence, completeness and observed usage. It
can retain useful partial outputs from failed executions; it cannot fill missing
observations from private expectations. Audit bindings create one isolated
execution Project of kind `project` per member and pin their visible inputs, so findings and
published outputs from another case/arm cannot become implicit inputs.
Native validators run in explicitly allowed bounded paths over exact artifacts;
external check records do not trigger code import or a callback. Human review
requires an explicit actor, rubric pin and exact result, with text/evidence bounds.

Registered required-check policy follows spec 26 precedence: error, fail,
incomplete, pass. Missing required evidence or checks cannot become a pass.
The first native collected result and its first registered-check assessment are
selected through recorded system CAS transitions. Later revisions never silently
replace the selected record. External clients select exact ingested records
explicitly. Human UI Save and use first persists the immutable review, then posts
the exact selection under CAS; if selection conflicts, the review is retained and
the user must inspect the newer evidence. Neither step reruns execution.
Reassess creates a new immutable revision. New selections use CAS and preserve
old reports. Result records distinguish server-observed execution/usage from
external collection claims; contradictory execution state is rejected. Source
provenance alone is not trust or an observed execution pin.

Read models are keyed by `(owner,experiment,plan digest,selection revision)` and
built from the complete expected member set. A candidate view generation becomes
visible atomically only when its rows and summaries are complete. Interrupted
projection leaves the last complete view visible with an explicit freshness flag.
It is rebuildable; changing selected authority still requires CAS.

Return expected/eligible/unsupported/blocked/submitted/terminal/missing/conflicting,
collection-complete/scored/quality-passed counts per arm and suite. End-to-end pass
uses expected denominator; conditional scored quality carries its own denominator.
Technical failures retain known cost/time and remain in compatible pairs. Cost
and latency compare only complete matching scopes, with included/excluded pair
counts and reasons. Compute p50/p90 by spec 26 nearest rank. Unknown is null;
partial totals disclose coverage and cannot establish total cost savings.

Membership conflicts and missing required evidence make the conclusion
inconclusive. A known regression still appears even when overall evidence is
incomplete. Strict conclusions require declared gates and verified required-equal
pins. Native setup explicitly selects `minCandidateEndToEndPass` and
`maxQualityDrop` in [0,1]; the reviewed initial preset is 1 and 0. It may also
select `maxTotalTokensRatio` >= 0. These map to the existing portable experiment
extension `playground:comparison-gates` with snake_case keys. They are frozen in
the referenced experiment, not added to plan/v1's closed `comparison` object.
All expected members must have terminal, collection-complete and scored evidence
before a conclusion. Candidate end-to-end pass below its gate or baseline minus
candidate conditional quality above the allowed drop is a regression. A selected
token-ratio gate additionally requires full compatible coverage and a nonzero
baseline total; otherwise the result is inconclusive. A complete result satisfying
all declared gates passes; UI says Meets declared gates, not universal quality
approval. Per-pair regressions remain visible even if aggregate thresholds pass.
An unsupported external gate extension remains an attributed producer claim and
cannot yield a native verdict.
No best-of-N or success-only denominator replaces individual attempts.

Managed member execution lists use indexed authoritative Run/Audit associations,
not label search or item-to-Run guesses. Rows retain unavailable/deleted refs with
an explicit gap. Cursors bind to owner, member and an execution-inventory revision;
new child execution associations invalidate the cursor for an explicit reload.
`inventoryComplete` requires a closed/drained parent dispatch inventory and
resolved accepted child intents; observing one page or a terminal child cannot
establish completeness. An external collector follows every page before claiming
complete scope; each HTTP query remains bounded. The optional Playground client
consumes this projection only through the public API.

### Chart projections

Charts are a projection of the same complete selected view, never a browser scan
or aggregation of the first member page. GET
`/v1/eval-experiments/{id}/charts/{chart}?viewSnapshot=...` accepts an optional
suite filter and a metric/measurement-scope selector appropriate to that chart.
Unknown chart/options and incompatible combinations fail closed. Responses carry
`viewSnapshot`, freshness, filters, scope, units, included/expected/excluded counts,
exclusion reasons and the unfiltered experiment summary. Changing a view uses the
same 409 reload contract as pairs; a filtered series is labelled explicitly.

- **Quality:** end-to-end passed/expected and separately scored/expected counts
  per arm. A conditional scored-quality mode uses its explicitly named denominator.
  Technical success, check success and external producer provenance remain distinct.
- **Tokens/duration:** distributions over complete matching pairs for the chosen
  dimension and verified required-equal pins. Technical failures remain included
  when measured. Server materialization computes exact p50/p90 by nearest rank,
  with at most 20 shared equal-width bins spanning the pooled observed range.
  Bins are left-closed/right-open except the final inclusive upper edge; all-equal
  values use one bin. Return bin counts, exact boundaries and deterministic filter
  tokens so drill-down uses exactly the displayed cohort, including missing-data
  and scope exclusions. Zero/one-value cohorts are unavailable/single-point views,
  not interpolated densities. Separate measurement scopes are never pooled.
- **Pair differences:** B minus A in the selected raw unit, one case/sample per
  row; never divide by a zero baseline or imply a quality verdict from cost alone.
  Page through at most 100 exact pair IDs/values using the existing cursor rules.
  Sort by absolute difference with frozen pair order as a tie-breaker, or frozen
  order when requested. Show total comparable pairs and quality-regression flags.
  These row pages are explicitly paginated; they are not the whole-experiment
  aggregate plotted above them.
- **Progress:** observed terminal-member counts per arm against fixed expected
  counts, with separate current assessment coverage. Persist server observation
  timestamps/counts during view publication; do not reconstruct intermediate
  samples from browser polls or producer-provided wall clocks. Bound the display
  to 200 time buckets, retaining the last observed value per bucket, and return
  interval, observation range and aggregation metadata. Use elapsed experiment
  wall time, including pauses; absence of observations is a gap, not zero or proof
  of stalled execution. A latest complete snapshot supplies the current endpoint;
  unknown historical progression remains unavailable.

Bin filter tokens are opaque and owner/experiment/snapshot/suite/scope bound.
Member/pair list endpoints accept them as `binFilter` for bounded drill-down; changing their
context rejects the request rather than broadening access or membership. Exact
percentiles come from retained complete cohorts, not rounded histogram bins.
Known all-arm partial totals stay separate from comparable-pair totals; neither
can establish complete experiment savings when coverage is incomplete.

No historical trend across different experiments is part of this slice. An empty
metric suppresses its plot while retaining a short coverage explanation. A and B
keep fixed blue/orange identities with text/shape/line alternatives; status colours
must not override variant identity. Every plot offers an equivalent bounded data
table and accessible detail controls. p50/p90 are variation summaries, not error
bars or claims of statistical significance. Existing retained portable documents
and comparison rules remain unchanged.

### Pagination and response examples

All lists use keyset pagination: default 25, maximum 100; oversized limit is 422,
not silent truncation. Format member/document bounds remain spec 26. Cursor is
opaque/signed and bound to owner, resource, filters and sort. Experiment member/
pair pages additionally bind to `viewSnapshot`. Mutable experiment/dataset lists
bind to an owner collection revision; a mutation invalidates the cursor and asks
for an explicit reload instead of silently skipping or repeating moved rows.
Sort is deterministic: experiments newest first by immutable creation time plus
lookup ID (not updated time, which usage ticks move without a collection
revision change); members/pairs by frozen suite order, case order, sample,
variant/member ID. Filters cannot alter
the unfiltered `experimentSummary`; any `filteredCount` is labelled separately.

```http
GET /v1/eval-experiments/experiment-1/pairs?viewSnapshot=view-7&limit=1&filter=unresolved
```

```json
{
  "viewSnapshot": "view-7",
  "freshness": "current",
  "experimentSummary": {
    "expected": {"a": 4, "b": 4},
    "terminal": {"a": 4, "b": 3},
    "qualityPassed": {"a": 3, "b": 2},
    "completeTokenPairs": 2,
    "conclusion": "inconclusive"
  },
  "filteredCount": 2,
  "items": [{"pairId": "opaque-pair-id", "caseId": "unsafe-query", "sample": 2, "a": {"execution": "succeeded", "assessment": "pass"}, "b": {"execution": "not_submitted", "assessment": "incomplete"}}],
  "page": {"hasMore": true, "nextCursor": "opaque-cursor"}
}
```

The shortened example illustrates the projection, not a full schema fixture.
Subsequent pages retain the same full summary. A changed view yields 409
`eval_view_changed` with the new snapshot ID; preserve filters and explicitly
restart pagination. Missing/deleted view data is not a guessed partial total.
No request scans all historical Runs or deserializes arbitrary artifact listings.
Use indexed bounded rows and one materialized summary read; a 10,000-member
fixture verifies query counts and response bounds.

## Authorization, retention and errors

Standard bearer/cookie owner authentication applies. Unknown or foreign resources
return owner-safe 404. Neither labels nor caller-supplied producer/owner strings
grant access. The ordinary private Worker API has no dataset, control or
assessment mutation capability. All new eval endpoints reject HEAD consistently.
No secret is returned by capabilities, catalogs or provenance/error projections.

Dataset revisions referenced by frozen plans are retained until those experiments
are purged. Experiment DELETE records a durable deletion fence, blocks new
submissions/assessments, cancels and drains accepted executions, then removes its
private data and public views according to existing artifact/Run retention rules.
It never deletes an unrelated Run merely sharing labels. Referenced completed
Run/Audit records keep their normal owner deletion behavior; subsequent evidence
loss makes comparisons incomplete and invalidates affected materialized views.

Project deletion must fence eval operations in the same durable Project lifecycle
before draining. The existing deletion barrier is extended to native/external
managed submission intents and referenced Audit execution Projects. Audit child
projects are exact registered dependencies, not inferred from labels; ownership
and deletion eligibility are rechecked. An external producer returning after the
fence receives 409 and cannot recreate deleted attempts. Native private eval
storage is not left behind after Project purge. Lifecycle uncertainty blocks purge
rather than claiming external work was undone.

Reuse standard error envelopes. Stable additional reasons include
`eval_not_ready`, `eval_preparation_unavailable`, `eval_pin_mismatch`, `eval_external_control`,
`eval_view_changed`, `eval_member_conflict`, `eval_evidence_unavailable`,
`eval_producer_stale`, `eval_budget_exhausted` and `eval_project_deleting`.
Errors include safe recovery actions, not raw source/credential details. Public
input/conflict/auth status mapping is tested with the existing API conventions.

## Compatibility and external clients

Spec 26 and existing Playground frozen bundles keep their exact bytes and local
recovery authority. The optional client can import visible datasets, create native
experiments, or register external matrices and use the managed submission API.
It retains hidden truth and scorer code locally, collects exact outputs through
normal public artifact APIs and sends attributed results/assessments. No reverse
connection, SQL access or internal Go imports are required.

Its previous direct Run/Audit mode remains usable and labelled legacy/unassociated
unless a separate verified migration is implemented. Importing a report does not
silently authorize native takeover, invent a missing journal or resolve ambiguous
receipt membership. Such takeover/migration is outside this slice. External
assessment provenance is visible beside native/human judgments.

This contract is producer-neutral. An independent minimal HTTP client must pass
the same external registration/submission/replay/assessment tests without importing
Playground. Native UI process tests run with Playground entirely absent. These
are the release tests for the selected independence boundary.
