# Audit checklist prioritization — specification and task review

Date: 2026-09-20. Scope: proposed [specification 34](../spec/34-audit-check-prioritization.md),
[V64 plan](../plans/2026-09-20-audit-check-prioritization.md) and V64-001–011.
Implementation inspection base: `cd2f377b`. This is a design/task review, not
implementation verification or evidence of deployed functionality.

## Method

Two independent read-only reviews examined architecture/lifecycle/dependencies
and user semantics/model-budget/coverage behavior. The author checked the task
DAG, integration boundaries, task acceptance ownership and documentation links.
Review used the actual Audit importer/start/preview, immutable Round dispatch,
next-round selection, report/coverage, Scheduler allocation/recovery, ModelPolicy
bounds and existing admission-deadline contract. V62 prerequisites were verified
to be pending, rather than treated as available code.

The user requirements used as acceptance anchors were optional service/finding
context; independent per-item verdicts; at least ten highest-priority checklist
checks per pass (all if fewer remain); preserved explanations for the rest; and
separate specification/tasks before implementation.

## Findings and corrections

| ID | Severity | Finding | Resolution and verification owner |
| --- | --- | --- | --- |
| R1 | P1 | Draft treated Audit deadline as terminal model-call failure, contrary to the existing admission-pause contract. | Spec34 sections 7–8 preserve draining and extend/disable/resume for Audit admission expiry; Run/Stage/cycle timeouts remain distinct. V64-005/007/009/011 require pause/resume and no repeated verdict tests. |
| R2 | P1 | A runnable profile could be published before full-inventory report/coverage support. | V64-010 now depends on V64-008; plan/index ordering and capability gating agree. |
| R3 | P1 | Verdict task requested concurrent requests while the first-version spec required sequential calls. | V64-005 specifies one in-flight request, no sibling history and first-failure stop; concurrency is deferred. |
| R4 | P2 | Complete per-item journal lacked an explicit ordinary Run output/receipt acceptance boundary. | Spec34 section 6 declares exact candidates/context inputs and versioned priorities output, Scheduler freezing, associated terminal Run/version import and journal comparison. V64-005/007 verify tampered/cross-cycle outputs and collection/deletion races. |
| R5 | P1 | Pre-ranking and post-ranking reservation descriptions risked double charging or loss of check capacity. | Cycle-owned count reservations are persisted before scoring, then converted to exact accepted identities and consumed once. V64-002/007 test eleven Run slots for one ranking plus ten checks, replay, batching and unused release. |
| R6 | P2 | UI promised editable topN but profile/baseline override semantics were absent. | Spec34 section 3 defines bounded Draft/Start prioritization.topN, profile default/ceiling, baseline/cycle/selection pins, immutable resume and changed-request idempotency conflicts. V64-001/007/008/009 own fixtures and UI behavior. |
| R7 | P2 | One-round default and multi-pass example appeared to describe the same immutable profile. | Default profile pins one Round; a separately named/versioned example pins three. V64-010 verifies 25 candidates as 10/10/5, and a two-Round bound retaining five untested. |
| R8 | P2 | Known oversized late-item input or insufficient whole-pool token allowance could waste earlier model calls. | Spec34 section 6 requires all-request size/window/call-count validation and whole-pool token reservation before the first call. V64-005/011 require zero calls for known impossible pools. |
| R9 | P2 | Aggregate counts could double-count repeated evaluations or equate settled with executed. | Spec34 section 9 separates unique Audit totals from cycle totals and accepted results from started/settled. V64-008/011 cover 25 candidates/two passes and rejection without invocation. |

## Architecture checks retained in the final design

- Full inventory has its own bound and provenance; deferred candidates are not
  admitted AuditItems. A 100-candidate inventory with a ten-item Round limit is
  a mandatory case, including preview/config validation.
- Selection is exact `min(topN, remaining)`, with a minimum configurable topN
  of ten. Approval wait or insufficient execution budget cannot secretly shrink
  membership or substitute lower-priority checks.
- Deferred high priority remains high; coverage stays not-tested. Finding
  severity, applicability advice, selection and check results are distinct.
- All remaining candidates use one frozen context, prompt/rubric/model policy.
  No title-based deduplication, shared conversation, implicit finding insertion
  or moving context is accepted.
- The narrow zero-Worker Planner path includes Scheduler reservation/recovery,
  model policy, metrics and global queue handling. Real ranking without any
  registered Runtime is a mandatory process gate, not a mocked unit claim.
- Durable intent plus per-item outcomes prevents silent re-invocation after an
  unknown model outcome. Run completion and Audit collection are separate checks.
- Deferred baseline continuation is independent of finding-confirmation mode;
  current proposal-only next-round behavior is not reused accidentally.
- Shared pre-Round foundations are explicit dependencies on V62-001–003. The
  new capability does not duplicate preparation or depend on generated OpenAPI,
  the autonomous pentest draft, or scanner-candidate ranking.
- Whole-inventory coverage spans all rounds, including no-Round failure, with
  retained context/decision/output lineage after permitted source Run deletion.

## Deliberate first-version tradeoffs

The first version blocks ranked admission if any item has no valid verdict.
It retains partial evidence and reports incomplete ranking instead of claiming
that a fallback list contains the highest model priorities. This improves the
clarity of the minimum-ten contract at the cost of availability for large pools.
An automatic fallback or manual reranking policy requires a separate explicit
contract, not an implementation guess.

The bounded slice supports 1,000 candidates, at most 100 combined context entries,
a complete bounded common-context projection and sequential model calls. Oversize
input fails explicitly. The byte-based token estimate is local admission
accounting, not a guarantee about provider billing; actual/unknown usage remains
separate. Reserving one initial check Run per selected item is conservative when
batching would use fewer Runs, and is documented rather than hidden.

Scripted responses test input isolation, validation, deterministic selection and
recovery. They cannot establish the quality or calibration of a real model's
priority judgments. A live-quality evaluation is outside this specification task.

## Planning validation and status

Validation covers YAML parsing with duplicate-key rejection, task/index identity
and unique registration, all dependency IDs and DAG acyclicity, pending/null-commit
status, acceptance-to-test coverage, dependency-layer ordering, document links,
release-task transitive closure and whitespace checks. Runtime/model/database/
browser implementation gates are defined as future tasks and are not reported
as executed by this review.

Final review status: **passed for specification and task planning**. Both
independent review passes were addressed; the architecture reviewer confirmed
all five findings closed. The author verified the final whole-pool zero-call
criterion in V64-005 and scoped-counter criterion in V64-008 after task alignment.
No unresolved must-fix issue remains in the reviewed design.

Planning validation passed for 11 pending tasks, 48 acceptance criteria with
explicit test coverage, and all ten prerequisite V64 tasks reachable from the
release task. YAML duplicate-key checks, task registration/DAG/layers, existing
context/document references and git diff --check passed. The local validation
manifest is `.local/evidence/v64/planning-validation.json`; it records planning
file hashes only, not product verification. Implementation remains pending for
every V64 task; V62-001–003 are still required foundations.
