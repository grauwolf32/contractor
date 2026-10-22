# Audit checklist prioritization review — 2026-09-20

[Research index](README.md)

- **Date:** 2026-09-20; status rechecked 2026-10-23 at `af9fd522`.
- **Reviewed:** the [contextual checklist prioritization proposal](../proposals/0034-audit-check-prioritization.md),
  the [V64 plan](../plans/2026-09-20-audit-check-prioritization.md) and
  V64-001–011, against code at `cd2f377b`.
- **Kind:** design and task review. It does not verify an implementation.
- **Outcome:** passed for specification and task planning; no must-fix issue
  left open.

## Status on 2026-10-23

- V64-000 (independent verdict contracts and the deterministic selection core)
  is completed. V64-001–011 were deferred on 2026-10-21; reactivation needs an
  explicit user decision after V62 is delivered and Audit measurements show
  that model ranking beats fixed ordering (see `deferral` in each task).
- The prerequisites V62-001–003 were pending at review time and are now
  completed, together with V62-004 and V62-009–012. V62-005–008 are deferred.
  Recheck the findings below against the delivered Audit preparation code
  before V64 resumes.
- The proposal no longer numbers its sections; the table below cites its
  headings instead.
- The planning manifest `.local/evidence/v64/planning-validation.json` was a
  local file and is not in the repository.

## Method

Two independent read-only reviews examined architecture, lifecycle and
dependencies, and user semantics, model budget and coverage. The author checked
the task DAG, integration boundaries, acceptance ownership and document links.
The review read the actual Audit importer, start and preview, immutable Round
dispatch, next-Round selection, report and coverage, Scheduler allocation and
recovery, ModelPolicy bounds and the admission-deadline contract.

Acceptance anchors taken from the user requirements: optional service and
finding context; an independent verdict per item; at least ten
highest-priority checks per pass (all of them if fewer remain); explanations
retained for the rest; specification and tasks before implementation.

## Findings

| ID | Severity | Finding | Resolution |
| --- | --- | --- | --- |
| R1 | P1 | The draft treated the Audit deadline as a terminal model-call failure, contrary to the admission-pause contract. | *Persistence, failure and recovery* and *Selection, readiness and subsequent passes* keep draining and extend/disable/resume on admission expiry; Run, Stage and cycle timeouts stay distinct. V64-005/007/009/011 test pause/resume without repeated verdicts. |
| R2 | P1 | A runnable profile could be published before full-inventory report and coverage support. | V64-010 depends on V64-008; capability gating agrees. |
| R3 | P1 | The verdict task asked for concurrent requests while the spec required sequential calls. | V64-005: one request in flight, no sibling history, stop at the first failure. Concurrency is deferred. |
| R4 | P2 | The per-item journal lacked an ordinary Run output and receipt boundary. | *Ranking Run* declares exact candidate and context inputs, a versioned priorities output frozen by the Scheduler, terminal Run import and journal comparison. V64-005/007 test tampered and cross-cycle outputs and collection/deletion races. |
| R5 | P1 | Pre- and post-ranking reservations could double-charge or lose check capacity. | Cycle-owned count reservations persist before scoring, then convert to accepted identities and are consumed once (*Capacity reservation*). V64-002/007 test eleven Run slots for one ranking plus ten checks, replay, batching and release of unused slots. |
| R6 | P2 | The UI promised an editable `topN` without profile or baseline override rules. | *Pass size (`topN`)* defines bounded Draft/Start `prioritization.topN`, profile default and ceiling, pins, immutable resume and idempotency conflicts. V64-001/007/008/009. |
| R7 | P2 | A one-Round default and a multi-pass example described the same immutable profile. | The default profile pins one Round; a separately versioned example pins three. V64-010 checks 25 candidates as 10/10/5, and a two-Round bound leaving five untested. |
| R8 | P2 | An oversized late item or an insufficient whole-pool token allowance could waste earlier model calls. | *Token admission and usage*: every request is validated for size, window and call count, and the whole-pool token estimate is reserved before the first call. V64-005/011 require zero calls for a known-impossible pool. |
| R9 | P2 | Aggregate counts could double-count repeated evaluations or equate settled with executed. | *Public API, UI, coverage and retention* separates unique Audit totals from cycle totals, and accepted from started or settled. V64-008/011. |

## Design constraints the review confirmed

- The full inventory has its own bound and provenance; deferred candidates are
  not admitted AuditItems. A 100-candidate inventory with a ten-item Round
  limit is a mandatory case, including preview and config validation.
- Selection is exactly `min(topN, remaining)`, with a minimum `topN` of ten.
  Approval waits or a short execution budget never shrink the membership or
  substitute lower-priority checks.
- A deferred high-priority check stays high, and its coverage stays not-tested.
  Finding severity, applicability advice, selection and check results are
  distinct.
- All remaining candidates share one frozen context, prompt, rubric and model
  policy. No title-based deduplication, shared conversation, implicit finding
  insertion or moving context.
- The zero-Worker Planner path goes through Scheduler reservation and recovery,
  model policy, metrics and the global queue. Real ranking without any
  registered Runtime is a mandatory process gate.
- Durable intent plus per-item outcomes prevent silent re-invocation after an
  unknown model outcome. Run completion and Audit collection are separate
  checks.
- Deferred baseline continuation is independent of the finding-confirmation
  mode; the proposal-only next-Round behaviour is not reused by accident.
- Whole-inventory coverage spans all Rounds, including a failure before any
  Round, and keeps context, decision and output lineage after a permitted
  source Run deletion.

## Accepted first-version trade-offs

- Ranked admission blocks if any item lacks a valid verdict. Partial evidence
  is kept and ranking is reported incomplete rather than presenting a fallback
  list as the model's top priorities. An automatic fallback or manual reranking
  needs its own contract.
- Bounds: 1,000 candidates, at most 100 combined context entries, sequential
  model calls; oversize input fails explicitly. The byte-based token estimate
  is local admission accounting, not a statement about provider billing.
- One check Run is reserved per selected item, which over-reserves when
  batching would use fewer Runs.
- Scripted responses test isolation, validation, selection and recovery; they
  say nothing about the quality of real model rankings. A live-quality
  evaluation is out of scope.
