# V60-008: Audit consistency and Project/Run/finding journeys

Review started from `8edeabf21194f5bba8dd258d53a072af713c630e` in the isolated
`review/v60-deep-review` worktree. The scope and required commands are recorded
in [V60-008](../../tasks/v60-008-audit-product-journeys-review.yml). No existing
user Audit, live model, production service or external scan target was used.

## Contract and historical basis

The review follows specs [06](../spec/06-server-ui-and-operations.md),
[17](../spec/17-projects-and-queue.md),
[18](../spec/18-run-and-workspace-lifecycle-controls.md),
[19](../spec/19-audits.md), [25](../spec/25-audit-worker-finalization.md),
[27](../spec/27-findings-tools-and-collections.md) and
[US-01–07](../spec/ui-user-stories.md).

Relevant accepted changes were read before evaluating current behavior:

- `7a618fb1`: retain frozen report reviews and support the complete public
  200-item page boundary; review is permitted again after report acceptance.
- `85845aca`: revision-bound provenance reads, with database rows closed before
  related receipt hydration; one-connection operation must remain possible.
- `dca420ba` / V60-012: source Run deletion atomically advances every affected
  Audit revision. Finding assessment/review identity stays intact. Finalizing
  report bytes retain their timestamp so a refreshed CAS can publish the same
  exact candidate; a pending report review retains its original frozen subject.
- `d0f6252e` and `2015edfc`: Controller phase selection and write preparation
  were separated without changing durable admission/collection barriers.

The reviewed paths are the Controller's round progression and collection,
AuditStore's acceptance/claims/report transactions, FindingIntake's exact
receipt import and direct verification, AuditService's review and collection
publication, public pagination/provenance handlers, and the Project/Run/Audit UI
journeys named below.

## Confirmed defects and corrections

### V60-027 — same receipt in multiple Audits (P2)

The previously documented limitation was reproduced against real PostgreSQL:
import of one exact ordinary-Run receipt into Audit A succeeded; the same import
into compatible Audit B failed with `audit_findings_pkey`, SQLSTATE `23505`.
Migration 041 formed a globally unique finding ID from receipt ID alone. Direct
verification also formed a global assessment ID from receipt alone and its
replay lookup could encounter the other Audit's assessment.

The explicit contract decision is to allow the same exact receipt in several
compatible Audits of its owner. Each Audit independently owns its finding,
review, assessment and retention. Same-Audit replay reuses existing records.
Ownership, Project compatibility, frozen-report fences and exact-content checks
remain in force. The existing public handler returns `201 Created` for each new
destination import and `200 OK` with `replayed: true` for same-Audit replay; no
new endpoint or success status is introduced. Spec 19 now records this decision.

[Correction V60-027](../../tasks/v60-027-audit-scoped-finding-import.yml) adds
migration 062 to change only future finding admission. Its opaque ID hashes a
JSON array of `(audit_id, receipt_id)` with SHA-256. Historical migration 041 and
all existing IDs/history remain unchanged. New direct-assessment IDs include
Audit identity; legacy receipt-only assessment IDs replay only in their owning
Audit. The compatibility lookup still rejects changed semantic assessment,
result digest or contract digest. Existing artifact names already live inside
an Audit namespace and require no format change.

Regression coverage proves:

- exact receipt imported/replayed independently into two Audits;
- separate direct-result/contract retention, two assessments, no extra links or
  byte-budget charge on replay;
- source Run deletion preserves both Audit holds; purging one Audit preserves
  the other's proposal and receipt;
- TP in one Audit and FP in the other retain independent immutable decisions;
  a finding ID cannot be read through the other Audit;
- an actual pre-062 database reproduces the collision, then normal migration
  preserves byte-equivalent legacy finding/assessment/review/decision/event
  history and admits the shared receipt in additional Audits;
- legacy assessment replay preserves the existing row, rejects changed content,
  and cannot block a new assessment in another Audit;
- public pagination, hydration and deletion tests obtain opaque finding IDs
  from their admitted records instead of reproducing the old naming formula.

Implementation: `f72102a1`. Follow-up `8c426a74` removes a trailing blank line
from migration 062; this changes its fresh-schema checksum, not SQL semantics.
Exact commands and original failure are in
[evidence V60-027](../../tasks/evidence/v60-027.json).

### V60-028 — stale revision in the hardening fixture (P2)

The initial `make test-audits-hardening` failed in
`TestPostgresControllerCollectsAndPublishesExactAuditReport`: after source Run
deletion the fixture attempted a transition with the pre-deletion Audit
revision. This run exited 2 after 121.700 s with 328 passing Go events and
no skipped Go cases. V60-012 intentionally invalidates that revision. The correction
asserts `ErrPrecondition` for the stale attempt, refreshes the Audit, verifies
one revision increment, and retries. Production behavior was preserved.
[evidence V60-028](../../tasks/evidence/v60-028.json) retains the focused proof
and failed/full gate results.

### V60-031 — browser aliases omitted required journey files (P2)

Inspection of the actual browser subprocess found a verification omission that
successful aliases alone could not reveal. `efa7d5da` narrowed the production
stack invocation to `e2e/stack.spec.ts`; its V37 report ran fixture journeys as a
separate manual command. The existing Audit/lifecycle/Scheduler/Performance
browser aliases still depended only on `test-ui-stack`, without that separate
fixture invocation. An actual before-selection command listed just **1 test in
1 file** for the operations stack. Independently managed Evals runs did not
supply the omitted journeys.

[V60-031](../../tasks/v60-031-audit-lifecycle-browser-gate-selection.yml)
restores an explicit bounded set: real `stack.spec.ts`, and route-fixture
`audits`, `lifecycle-controls`, `project-workspace`, `run-repeat`,
`scheduler-settings` and `performance`. Actual after-selection lists **20 tests
in 7 files**. A JSON-report validator requires executed passing cases in every
selected file and rejects skips, failures/flakes, empty or incomplete reports.
The corrected operations harness executed all **20 cases with zero skips,
unexpected outcomes or flakes** (111.68 s including its process setup). Its
retained JSON report is hashed in V60-031 evidence. The full prerequisite passed
in 1107.48 s with 21 Go pass events, zero skips, and separate native/external
Evals journeys (511.67/481.69 s). Separate scan/Git gates retain their existing
boundaries.

The earlier full release compiled source `9cfd09cc`. Its old UI package
passed in 1095.579 s (real operations 101.30 s, native Evals 511.98 s, external
Evals 482.29 s), but that result does not prove execution of the omitted fixture
journeys. The corrected full browser prerequisite and aliases passed on
`45b31abf`, whose relevant delta consists only of test changes `7afb3809`
(V60-031) and `45b31abf` (V60-032). Unaffected release results can be reused;
both browser aliases now have proof from the corrected full browser run.
[evidence V60-031](../../tasks/evidence/v60-031.json) records this source delta
and the actual affected-gate result.

### V60-032 — stale selectors exposed by the restored browser selection (P2)

A separate preflight on a temporary copy of the same built UI executed all
19 route-fixture cases: 16 passed and 3 failed, with zero skips/flakes
(53.764 s). The omitted cases had stale assertions: Audit text matched both a
badge and its excerpt; lifecycle controls used the previous Project action
label; Performance expected GPU temperature/Power counters outside the
collapsed Detailed counters view introduced by `f5ae4a4f` / V58-010.

[V60-032](../../tasks/v60-032-browser-fixture-selectors.yml) narrows the Audit
badge assertion, uses the current accessible Project action, and explicitly
opens Detailed counters for the retained metric checks. Existing chart,
mobile, exact evidence and destructive-action assertions remain. These are
fixture corrections to delivered UI behavior; application code is unchanged.
Its preflight and the corrected full browser prerequisite supply execution
proof in [evidence V60-032](../../tasks/evidence/v60-032.json).

### V60-034 — replacement catalog prerequisites for the restart journey (P2)

The later full Audit-program process run exposed two stale replacement-fixture
assumptions: deleting instructions still used by version-3 Workflows, then
retaining version-2 profiles that depended on the removed current standards.
[V60-034](2026-09-20-v60-034-audit-catalog-fixture-review.md) preserves the shared
instructions and removes both generations of dependent profiles. Current
standard deletion and every historical Audit assertion remain intact.

The complete `make test-audits-process` target subsequently passed at
`e49c65230b185c0daf575a202a22e4508bb7ed4b`: the Audit restart journey took
363.73 s, heterogeneous placement 46.02 s, and scheduler concurrency 65.13 s.
The package passed in 474.894 s with three Go PASS events and zero skips.
Its actual restart now verifies retained baselines, coverage, reports and
finding backtraces while current profiles and standards are unavailable.
Original failed attempts remain in [V60-034 evidence](../../tasks/evidence/v60-034.json).

### V60-035 — per-read snapshot time compared as durable review state (P2)

The required Findings process gate completed producer and reader execution, then
failed its whole-page equality check. A diagnostic repeat reproduced the failure
in 38.293 s: only `asOf` changed between the two responses. `auditRevision`,
`items`, `page` and `total` were identical. The original assertion predates
`1b45fc52`, which added revision-bound pages with an SQL statement timestamp for
each read, as documented by spec19.

[V60-035](../../tasks/v60-035-findings-read-snapshot-comparison.yml) validates
both nonzero RFC3339 timestamps, then compares every remaining response field
using the existing equality check. The producer, reader, exact collection,
no-new-proposal and Gateway assertions are unchanged. No product mutation or
Gateway defect was observed. Full required gate results are recorded below.

## Scenario analysis and evidence levels

| Scenario | Representative verification and implication |
| --- | --- |
| Review TTL and decision replay | `TestAuditFindingReviewHistoryAndDeletedRunProvenance`: injected time crosses TTL; expiry persists despite rejected decision. Same key replays immutable decision; changed subject and stale revision fail. |
| Frozen report review | `TestAuditReportAcceptanceUsesFrozenCandidate` and importer report-deletion regressions: exact candidate survives source deletion, acceptance publishes its original links, later finding review does not rewrite the report. |
| Multiple rounds and recovery | `TestPostgresAcceptNextRoundIsAtomicReplaySafeAndConsumeOnce` races next-round acceptance in real SQL; one insertion/consume fence survives replay and rejects reuse. Controller replacement reconstructs the durable dispatch window in `TestPostgresControllersConvergeWithoutDuplicateExecutionAttempts`. The separate `test-audits-process` restart journey subsequently passed under [V60-034](2026-09-20-v60-034-audit-catalog-fixture-review.md), preserving pinned history after catalog replacement. This is not a process kill at a round boundary. |
| Retained findings and collections | Intake/deletion regressions plus `TestFindingCollectionPublicationRunRetentionAndReplay` and `TestFindingCollectionAuditContributionsAndPinnedReview`: exact bytes, contributing receipts, pinned review revisions, missing/foreign sources and source deletion retain their distinct outcomes. |
| Provenance and complete pages | `TestFindingProvenanceConsistentReadsWithSingleConnection`, `TestPublicAuditPaginationBoundary`, `TestPublicAuditSourceRunDeletionInvalidatesReadContexts`: 199/200 boundaries, stale cursors and one-connection hydration; deletion invalidates the enclosing Audit revision. |
| Programmatic Worker completion | Dedicated Audit completion gate checks its required Go/Python matrix, real Artifact API, deterministic sealed result publication and incomplete/invalid/transport cases; absent or skipped mandatory cases fail the gate. |
| Findings producer/reader | Dedicated findings gate checks real producer/reader processes, receipt replay, source-deletion snapshots, changed selection conflicts, exact evidence/pagination, inaccessible input, explicit empty success and invalid archives; required Go cases/Python multiplicities reject skips. |
| Project/input/Run/result | `ui/e2e/stack.spec.ts`: real independently served Go/Node stack, Project upload, exact input setup, Worker execution, exact output download and Project publication. |
| Repeat and unavailable exact sources | `run-repeat.spec.ts` uses browser requests with route fixtures for reviewed exact repeat drafts, conflict preservation, primary preview and Audit-managed return to its owner. `project-workspace.spec.ts` covers scoped setup and constrained layouts. |
| Audit/review/report UI | `audits.spec.ts` drives browser UI with route fixtures: exact baseline, coverage gaps, finding decisions, retained evidence, report review, stale/error outcomes and destructive confirmations on desktop/mobile. |
| Lifecycle UI | `lifecycle-controls.spec.ts` drives browser controls without manual reload, using route fixtures. Real process lifecycle coverage is a separate release prerequisite, not inferred from mocked responses. |

Rejected hypotheses are recorded separately from defects: persisted review
expiry is committed before returning its precondition error; waiting for human
review does not retain an allocation or transaction; Run deletion's revision
invalidation is intentional and must not be bypassed by stale fixtures.
No new defect was confirmed in those reviewed paths.

## Executed checks

Toolchain: Go 1.25.6 with race detection where prescribed, Python 3.13.14 in the
locked Runtime environment, Node 24.20.0/pnpm 11.24.0, disposable PostgreSQL 17
with isolated schemas. `GOFLAGS=-p=2 -v`, `GOMAXPROCS=2`. Connection credentials
are excluded from durable evidence.

All five required targets passed after the bounded corrections, with zero
selected skips. Counts include subtests and shared matrices, not unique
cross-command scenarios.

| Check | Actual result |
| --- | --- |
| Exact V60-027 broad PostgreSQL/race command | 644 Go pass events, 182.006 s; migration preservation separately passed on final migration bytes. |
| `make test-audit-completion-e2e` | 173 Go cases and 331 Runtime cases passed; the gate verifies required cases and rejects skips. |
| `make test-audits-hardening` | 193 Go pass events and 34 UI tests/3 files; its separate matrix prerequisite passed 139 Go events. |
| `make test-findings-e2e` | 13 required process cases and 48 Runtime cases, 190.699 s, exit 0. |
| Supplemental `make test-audits-process` / V60-034 | 3 process tests, 474.894 s Go package time, exit 0 and zero skips; actual catalog replacement/restart and retained history. |
| `make test-audits-browser test-lifecycle-controls-browser` through corrected `test-ui-stack` | 21 Go pass events, 20 validated Chromium cases/7 files, plus real native/external Evals; 1107.48 s, exit 0. |

The dedicated Audit completion/hardening targets ran on `45b31abf` and finished
successfully before the downstream Findings comparison failure; their immutable
log sections are retained. No individual target wall time is invented for that
shared 298.83 s invocation. The corrected full browser gate also ran on
`45b31abf`. The full Findings rerun passed on `7100eb41`; its only relevant delta
is V60-035's test assertion. Intervening V60-033/034 changes affect separate
process fixtures. Production code is identical between those historical
`45b31abf` and `7100eb41` gate sources. The supplemental restart target is
separately attributed to integrated source `e49c6523`; no unchanged-production
claim is made across that later main integration.

The V60-035 rerun retained all producer/reader assertions and all five retention
and five reader-boundary subcases. The original failure and diagnostic timestamp
difference remain in evidence; they are not replaced by the final successful log.

Detailed commands, exit codes, counts, skip classification, timings and log
hashes are retained in [V60-008 evidence](../../tasks/evidence/v60-008.json).
`test-audits-browser` and `test-lifecycle-controls-browser` share `test-ui-stack`;
a single invocation runs that prerequisite once and satisfies both aliases.
`test-audit-completion-e2e` is executed explicitly in the dedicated Audit run.
The initial review checkpoint did not execute the broader Audit-program
restart target. The later complete target is now recorded separately under
V60-034 above; this target does not itself establish overall release success,
which is recorded separately under V60-009.
Shared prerequisites are not counted as repeated independent proof.

## Remaining boundaries

This is representative verification of the specified journeys, not exhaustive
crash injection at every SQL boundary or evidence of live-model quality.
Multiround SQL acceptance and Controller reconstruction are separate
complementary checks; this review does not claim a Server process kill at a
round boundary. The separately executed Audit-program restart gate proves its
specified catalog replacement and historical-read journey. Browser route
fixtures demonstrate UI behavior; only explicitly real-stack/process cases
demonstrate backend wiring.

Task records and the [V60-010 delivery snapshot](2026-09-20-v60-010-configuration-evaluation-review.md)
distinguish, at that snapshot's reviewed commit, delivered V38/V55 work from
pending paired instruction evaluation
(V40-002), V55-009/010, V61 and separate SARIF work. Passing these review gates
does not complete those pending tasks or prove integration of a separate
external client repository.
