# Audit Worker completion contracts

Status: **Implemented opt-in through V39-006; release gate V39-007 pending.**
V39-001–006 implement contracts, Server pinning and capability-aware placement,
incremental collection, deterministic ZIP publication, common Runtime completion,
bounded diagnostics and versioned examples. The required PostgreSQL/restart and
real Runtime-to-importer release gate remains V39-007.
See [contract validation](../reviews/2026-09-07-audit-completion-contracts.md),
[Server pinning](../reviews/2026-09-07-audit-completion-pinning.md),
[collector](../reviews/2026-09-07-audit-result-collector.md),
[publication](../reviews/2026-09-07-audit-result-publication.md),
[Runtime integration](../reviews/2026-09-07-audit-completion-runtime.md) and
[diagnostics](../reviews/2026-09-07-audit-completion-diagnostics.md) evidence.
Existing deployed Runs and immutable snapshots keep their current behavior until
an explicitly versioned configuration selects this contract.

This document extends [14](14-worker-results-and-live-state.md) and
[19](19-audits.md). Where those documents describe mandatory LLM serialization
or one-shot Audit submission, this document defines the explicit opt-in
exception. It does not replace ordinary Workflow finalization.

## 1. One completion boundary, two strategies

Every Worker has one Runtime-owned semantic completion boundary, reached while
the invocation/session can still continue and before WorkerCompletion or
terminal invocation State is published. This is not allocation drain/finalize.
The boundary returns exactly one of:

- `continue`: a bounded Runtime-authored reminder; resume the same invocation;
- `complete`: a trusted WorkerResult ready for normal lifecycle processing;
- `fail`: a bounded WorkerFailure, with no fabricated successful result.

The shared boundary lives in `worker/completion.py`; the Audit implementation
lives with `toolsets/audit_results`. Worker orchestration binds a prepared
completion implementation to the identical trusted contract and all its required
tools under one owner. Missing, mixed or forged owners fail preparation. The
common runner does not infer Audit behavior from tool names or import an
Audit-specific completion policy. This organization does not enable additional
publicly selectable completion strategies.

| Selected contract | Completion behavior |
|---|---|
| Omitted / ordinary | Existing terminal text → one-shot tool-free LLM serializer → trusted projection. Existing optional summarizer remains unchanged. No mandatory Audit tool check. |
| `audit-check-results@1` | Check Runtime's validated per-item collection → seal → deterministic ZIP publication → Runtime-authored WorkerResult. No LLM serializer. |

For Audit-check, success requires a valid accepted submission for **every**
assigned item. A proposed tool call, an unsuccessful call, a textual claim of
completion, or one successful submission in a larger batch cannot satisfy it.
The old MandatoryToolCallback is an inspiration for bounded reminders only:
its proposal-based, agent-instance-lifetime call tracking is not sufficient.

This is a condition for successful completion, not an exactly-once tool-call
or eventual-success guarantee. A model may never submit valid data; bounded
continuation then fails the invocation. Duplicate calls can occur, and accepted
collection, published artifact and importer-accepted Audit coverage remain
three different facts.

## 2. Trusted activation and compatibility

The activation source is an immutable AuditProfile check-workflow binding:

```yaml
workflows:
  check:
    kind: check
    ref: audit-source-check@2
    # Existing inputs, parameters and outputs mappings remain required.
    workerCompletion:
      kind: audit-check-results@1
      stage: check
      agent: checker
```

`stage` and `agent` select exactly one logical Worker in the resolved Workflow,
not a physical Runtime. Version 1 requires that selected stage to use
`passthrough@1` with one Worker responsible for the entire assigned task set.
Its declared result binding must feed the check binding's canonical logical
`result` output. Other stages/Workers are not implicitly enrolled. Subdividing
the batch among Streamline/Router subtask invocations is outside this version.

The selected AgentTemplate must explicitly select `audit-results@2` with both
`read_audit_task` and `submit_check_result`. It must not select legacy
`audit-results@1` simultaneously. Version 1 rejects an optional terminal
summarizer on this Worker: a tool-free summary cannot replace missing check
submissions. This restriction does not affect ordinary Worker summarization.

Validate these constraints against every reachable escalation variant of the
selected stage, including replacement AgentTemplates and result mappings.
Changing an execution configuration must not remove the gate, select @1, add a
summarizer or change the pinned task/result ownership. Server authoring
descriptors must register @2 separately from Runtime capability advertisement.

Only the trusted Audit Run-creation service may activate this binding, after
checking actual AuditExecution ownership and `kind: check`. Discovery,
assessment and other role kinds cannot select it. Ordinary public Run creation
cannot set it, including when it uses the same Workflow or spoofed Audit labels.
Run labels, task text, tool names and model function calls are never activation
signals. Selecting `audit-results@2` without a trusted contract is a preparation
error, not an implicit change of ordinary completion behavior.

The Server snapshots the choice with the child Run and propagates it through
Scheduler/Control Plane into optional `AllocationSpec.completionContract`:

```yaml
completionContract:
  kind: audit-check-results@1
  task: {namespace: inputs, name: task, revision: exact-task-revision}
  executionManifest: {namespace: inputs, name: execution_manifest, revision: exact-manifest-revision}
  resultArtifact: {namespace: audit-check, name: result}
```

Refs and the versionless output binding are derived from the resolved input,
Stage and output mappings; they are not additional operator/model authorities.
They must agree with AllocationSpec grants and the StageContentRequest result
binding. Runtime validates exact task/manifest identity before starting work.
No new Project/Audit mutation or foreign-scope artifact authority is granted.

Runtime advertises `capabilities.completionContracts` containing supported
contract IDs. Omission means no new contracts. Placement requires both the
contract and selected toolset capability; direct prepare also rejects unknown
or unsupported contracts. Retry/placement preserves the pinned choice and must
never downgrade silently to ordinary completion on an older Runtime.

`audit-results@1`, omitted contracts and old snapshots retain their existing
behavior. New toolset, Workflow, AgentTemplate and AuditProfile versions opt in
together after capable Server/Runtime deployment. Do not mutate existing
versions, bindings or running Audits as part of implementing this feature.

## 3. Invocation-local result collection

`audit-results@2` keeps the read tool and changes the submit tool into a bounded
collector. For incremental submissions its model-facing shape is:

```text
submit_check_result(
  item_key, assessment, summary, completed, gaps,
  evidence=[], proposal_keys=[], expected_revision=None
)
```

`item_key` must come from the trusted task set; it may be omitted only for a
single-item assignment. Subject, requested coverage and execution digest are
derived from that task, never from model arguments. Invocation identity for
proposal selection is injected by Runtime. `assessment` uses the existing
closed vocabulary, not synonyms such as `pass`.

Both tools use the invocation's validated snapshot of the exact contract input
refs. Reusing @1's reads of current `inputs/task` or `inputs/execution_manifest`
is insufficient: later alias changes must not change the assignment or what
read_audit_task presents as authoritative.

The compatible `results=[...]` convenience form accepts one complete ordered
batch only, without scalar fields, item_key or expected_revision. It validates
all members atomically before accepting any. Incremental scalar calls may
arrive in any order, including parallel calls; a lock serializes acceptance.

When the batch form follows incremental submissions, already recorded identical
members are replays and absent members are creates. Any changed existing member
conflicts and leaves the entire collection unchanged. Corrections use scalar
calls with expected_revision; the batch form does not silently replace items.

Each accepted item has an invocation-local positive revision, starting at 1:

- absent expected_revision creates an unrecorded item;
- identical canonical content repeated without a revision returns the existing
  receipt without incrementing it;
- different content without the current expected_revision conflicts;
- an explicit update with the current revision atomically replaces the item
  and increments its revision; stale revisions fail without mutation;
- a retry of an already applied explicit update returns its receipt only when
  expected revision and content match that immediately preceding update.

A receipt reports `status: recorded`, item key/revision(s), accepted count,
total count, missing item keys, and completeness. It is not an ArtifactRef and
does not claim publication or Audit acceptance. Update histories are not
retained as unbounded content; retain only current data and bounded replay
identity. Repeating or correcting a call still consumes normal tool budget.

Validate before changing the collection: identifiers and membership, closed
assessment vocabulary, bounded text/arrays, requested/completed coverage,
proposal-key shape and the task-local evidence rules available to Runtime.
In particular, conclusive checklist results must include each required evidence
kind; `completed` alone is not evidence. Errors identify the field/item and
missing kinds without copying raw source, summaries or credentials. Evidence
retention and cross-Run/proposal/profile acceptance remain the trusted importer's
responsibility; local acceptance is not a substitute for that validation.

Task-local validation explicitly includes the pinned standard evidence
contract's allowed assessments, evidence kinds and minimum/maximum counts
(minimum applies to conclusive results), and the operation-task rule that
`not-tested` cannot carry completed coverage. Shared Go/Python fixtures must
cover these rules as well as checklist evidence. Preserve the importer's
distinction between invalid input and a valid result whose gaps yield
inconclusive coverage; do not require every accepted result to be conclusive.

Use the existing limits: at most 64 assigned items, 16 KiB per summary, 512
coverage values per bounded list, 128 proposal keys per item (the existing importer
limit), 256 evidence records across the whole batch,
8 MiB canonical collected-data budget, and 16 MiB final package. Charge
replacements by prospective total size, not just the new item's size. Validate
prospective encoded member/package bounds before accepting submissions so a
locally accepted complete set is serializable. Invalid calls consume no stored
item revision and never erase an earlier valid result.

The publisher owns the pure canonical encoder/size calculation used by the
collector for prospective validation, including per-member limits and ZIP
overhead. Do not maintain a second approximate encoder in the collector.

Collection state belongs to `(allocation, invocation, pinned task-set digest)`.
Even shared ADK sessions do not carry accepted results into another invocation.
It is outside model-editable ADK State and public/live-state raw content, and
cannot be restored from conversation, metrics or prior tool-call names.
Cancellation, failure, release and process loss discard unsealed partial data.
There is no partial durable AuditItem completion or cross-invocation resume.

## 4. Completion gate and bounded continuation

At each normal model finish, including empty/meaningless terminal text, the
common completion boundary dispatches to the pinned strategy. Audit-check
consults accepted collector state, not terminal prose. If items are missing,
Runtime supplies a deterministic reminder naming missing trusted item keys and
instructing the model to use submit_check_result/read_audit_task. No untrusted
tool error text is promoted into control instructions.

Allow at most two reminders per invocation, independently of item count. They
continue the same invocation/session with the same ModelPolicy and counters;
no reset, hidden model call, second A2A invocation or increased deadline is
allowed. The reminder is bounded by the existing 64-item identifier limits.
The model may truthfully submit blocked/inconclusive/not-tested where the
existing task contract permits; Runtime never invents those entries itself.

Separate one logical Worker invocation from repeated ADK runner turns. A
continuation must not repeat begin_invocation/prepare_invocation, reset the
instrumentation reducer or tool ordinals, clear observed refs, or erase the
collector. Final State and cleanup occur once. The current before_run_callback
consumes a one-use preparation token; simply wrapping runner.run_async in a
loop is not sufficient. Before freezing the interfaces in V39-001, demonstrate
the continuation seam with the pinned real ADK Runner, production
instrumentation and a scripted model; V39-005 supplies the production integration.

After the reminder allowance is exhausted, return retryable
`audit_result_incomplete` with safe counts/missing identifiers. Existing hard
budget, deadline, cancellation and sandbox failures take precedence and do not
grant another reminder or trigger publication. A failed or cancelled main
invocation cannot be converted into success just because its collection is full.
Stage/Audit policies, not this flag, decide whether a new attempt is submitted.

Consuming the last permitted model/tool call is not itself a budget failure.
A normally completed full collection may publish without another call; actual
budget exceptions/overshoot still fail. If another model or tool call would
exceed its limit, continuation cannot obtain extra capacity.

Once a normal completion has a full valid collection, atomically seal it.
Further submissions/updates are rejected. The deterministic publication phase
may not return control to the model or modify the sealed contents.

## 5. Deterministic publication and trusted result

The common finalizer's Audit strategy, not the model-visible tool, builds the
existing `check-results` package. Keep the importer/wire package formats:
manifest.json, check-results.json, optional evidence.json and evidence members.
Use trusted task order regardless of submission order, stable member/evidence
IDs, canonical JSON, fixed ZIP timestamps/permissions and stored compression.
For identical sealed values and pinned manifest bytes the resulting ZIP is
byte-identical; wall clock, random IDs and submission/update order cannot affect
it. Proposal invocation identities remain meaningful inputs, so this does not
promise identical bytes across different invocations selecting proposals.

Write only the Server-declared output binding using create-only CAS. On a
conflict or ambiguous transport outcome, a bounded exact-content read-back may
recognize the same bytes/media type at that binding and return its exact
revision. Different content is `audit_result_publication_conflict`; never
overwrite it. Retry only identical sealed bytes within the existing deadline
and a finite policy (at most one additional write); no model repair/re-execution.
If success cannot be established, return `audit_result_publication_failed`.
Size/input-contract errors are non-retryable, transport errors are retryable,
and a conflicting existing output is non-retryable for this contract.

The finite policy permits at most two writes and two read-backs in total, all
under the invocation deadline and with the package-size response bound. A
create conflict permits comparison, not overwrite; retryable transport errors
permit reconciliation. Authentication, authorization and write-fence rejection
are terminal for this publication and do not become transport retries. Translate
these failures at the Audit boundary so generic exception handling does not
record a programmatic publisher failure as an LLM error.

The result binding is Run-scoped, not Stage-attempt-scoped. If a failed attempt
already wrote it, a later attempt in the same Run must collect and seal its own
complete set before comparing bytes. Equal bytes may be recognized; different
bytes fail with `audit_result_publication_conflict`, including changed proposal invocation
identities. Existing bytes never seed the new collector. A fresh Audit-level
retry creates a new child Run and therefore a fresh result binding. Version 1
does not promise transparent recovery by Stage retry after publication; opt-in
examples end the child Run on failure/interruption and leave retry selection to
the existing Audit policy. No automatic revision overwrite is introduced.

After verified publication, Runtime constructs WorkerResult directly: the
request's authoritative subtask ID, a bounded deterministic completion summary,
the finalizer's verified exact artifact receipt, normal trusted observations,
and `summarized=false`. There is no LLM serializer, invented tool call, or model
token charge for this work. Exact receipts from this trusted finalizer are an
explicit extension of tool-observed artifact projection, not an invitation to
accept model-authored artifact refs.

The common lifecycle path still applies cancellation/write fencing, terminal
State, cleanup and WorkerCompletion publication. If cancellation wins after the
artifact write, the artifact may remain for diagnostics but cannot imply Run
success or accepted Audit coverage. Scheduler/Planner validate Stage outputs;
the Audit importer independently validates and atomically accepts the full set.

## 6. Diagnostics and release requirements

Expose bounded completion facts through existing typed report/diagnostic
mechanisms: contract kind, phase (collecting/sealed/publishing/published/failed),
accepted/total counts, reminder count and stable failure code. Persist terminal
facts so failed Runs can distinguish incomplete collection, publication failure
and ordinary LLM serialization failure. Missing item IDs belong in bounded
diagnostics, not metric labels; no evidence/source/summary bodies are added to
reports. Existing opt-in model-content telemetry policy remains unchanged.

`ExecutionReport.completion` and the live allocation metrics' optional
`completion` carry the latest invocation's facts, not a history of item data.
The object has `kind`, `phase`, `acceptedCount` (0..64), `totalCount` (1..64),
`reminderCount` (0..2), and a bounded `failureCode` for failed completion.
Sealed/publishing/published phases require all assigned results. Ordinary
Workers and historical reports omit the field. Consumers discard unknown
optional kinds/phases; malformed known diagnostics are rejected. The optional
object is bounded to 4 KiB at the wire boundary. Public attempt diagnostics
retain these facts through the existing report projection and explicitly
distinguish publication from Audit evidence acceptance.

Phase publication only synchronizes facts; it charges no model/tool call or
token usage. Invalid submit calls remain real failed tool calls. Reminders use
the existing invocation budget. Cancellation and session cleanup failure after
the ZIP was written change completion diagnostics to `failed`, preserving the
recorded count and leaving the existing output intact. Publishing these late
facts does not complete the invocation a second time.

The opt-in repository example is `source-checklist@3` → `audit-source-check@4`
→ `audit_source_checker@4`, with a separate bounded
`audit_completion_worker@1` ModelPolicy. It selects `audit-results@2`, one
`passthrough@1` Worker and no target summarizer. The completion-specific
instruction explains scalar `item_key` submissions, `expected_revision`,
local `recorded` receipts, truthful gaps, and Runtime-owned ZIP publication.
Both failed and interrupted transitions end the child Run; Audit's existing
attempt policy decides whether to create a fresh child Run. A same-Run retry
cannot overwrite different bytes or different invocation-owned proposal IDs.

Rollout is capability-first: install compatible Server/Runtime/report readers,
verify the release gate, then explicitly select the new AuditProfile for a new
Audit. Adding these files does not rebind a running Audit. Older profile,
Workflow and template versions retain their completion protocol. There is no
durable recovery of a partially collected invocation, target summarizer, or
multi-Worker completion strategy in this version. These examples are not live
deployment or certification evidence.

Release tests must include ordinary/no-contract regression, spoofed activation,
unsupported Runtime placement, per-item validation/correction/replay, parallel
submissions, reversed submission order producing identical bytes, no-call and
partial batches, invalid calls not satisfying the gate, shared-session reuse,
hard budgets and summarizer configuration rejection, cancellation at each
boundary, dropped write replies, conflicting artifacts, and process loss.
End-to-end tests must pass a runtime-produced ZIP through the real Go importer,
including the missing-required-evidence regression. No live model or demo
mutation is required for deterministic acceptance tests.

Provide a dedicated release target that requires an isolated PostgreSQL test
database and executes the new persistence/restart and importer bridge tests
without skips. Plain go test ./... and make verify are not sufficient evidence:
existing database tests skip when CONTRACTOR_TEST_DATABASE_URL is absent.
Include same-Run versus new-Run retry, escalation variants, mixed batch/scalar
replays, exact-budget completion, input-alias changes, and real ADK continuation.

## 7. Non-goals

This increment adds no mandatory tool policy to ordinary workflows, general
callback framework, public completion override, durable partial-check store,
batch decomposition across Workers, automatic acceptance of Audit evidence,
global replacement of the ordinary LLM finalizer, or automatic demo rollout.
