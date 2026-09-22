# Contractor review plan — V60

Requested by the user after the public OpenAPI corrections. Starting point:
`3ebe07cb` in local `main`: V59-001–005 are complete, and the concurrent UI
refinement V58-013 is included. [V59 results](../research/2026-09-20-public-openapi-corrections-results.md)
remain separate from new findings. This plan initiates the review; it does not
declare the entire system verified on the basis of the first pass.

Current outcome: the deeper reviews V60-005–010 and all confirmed V60
corrections are complete. The [completion report](../research/2026-09-20-v60-deep-review-results.md)
records executed checks and their limits; the [first pass](../research/2026-09-20-project-review-first-pass.md)
is retained separately. Verification is attributed to the source commits listed
in task evidence. Live-model quality and production deployment remain outside
this review.

## Goal and sequence

Find reproducible violations of user journeys and existing contracts, then fix
them through small, verifiable tasks. Assess expensive or complex areas separately
when measurements are available. File size, validation count or the presence of
two languages are not defects in themselves.

1. **Establish the baseline and perform the first pass — V60-001.** Map requirements,
   current tasks and checks. For the highest-risk boundaries, read code, decision
   history and negative tests; reproduce candidate findings.
2. **Fix confirmed local defects — V60-002–004.** Do not wait for the full review
   when the cause, contract and verification are already known. Each fix gets a
   regression and a separate implementation commit. The actual integration run
   additionally exposed an outdated mandatory Runtime test name — V60-011.
   Independent review found a separate invalidation defect on Run deletion —
   V60-012; it was fixed after PostgreSQL import/delete/purge checks and report
   finalization recovery checks.
3. **Deepen scenario checks — V60-005–010.** Perform the fault, database, transport
   and browser checks below. Before each task, refresh the baseline commit and
   check for overlap with concurrent work.
4. **Consolidate results after the deeper passes.** For each area, record executed
   scenarios, remaining gaps, fixes and measured costs. Update the shared report;
   completing one area does not close the others.

The first pass and its fixes do not depend on live-model evals. Process checks
use a fake Gateway and isolated PostgreSQL. Real models, production configuration
changes and deployment are outside this plan.

## Basis and concurrent work

The normative entry point is the [specification catalog](../spec/README.md).
Each task lists its specific contract documents. When prose, tests and
implementation disagree, first establish the accepted decision's history, then
record which contract is being corrected and why.

The initial commit had 349 task files: 337 complete (`complete` or `completed`),
eight pending, two in_progress and two archived. This is a snapshot before V60
was added, not manually maintained product statistics.

- V40-002 is a prepared instruction experiment with `live_ready: false`;
  recorded controls and wrappers are in main, but live pins/dispatch/mapping
  prerequisites remain open. No fresh work was found in the corresponding
  Contractor and Playground worktrees as of 2026-09-20. V40-003 depends on it.
- V55-004 is formally in_progress, but neither a prepared-request implementation
  nor current edits were found in accessible worktrees. This observation alone
  does not reassign the task; V55-005/006 remain the nearest scan scope.
- V38-001–005 are complete and integrated into main (`45277817`, `64a14716`).
  Current V38-006 development is in `v38-006-comparison` on recent main;
  the older `v38-evals-experience` retains earlier drafts of that same stage.
  V60 does not create a duplicate eval runner or take over the active implementation.
- V57-003 is a separate decision draft; after reassessing its necessity on
  2026-09-20, it was archived as a disproportionate standalone investigation.
  The ordinary finalizer remains active; its necessity remains a question in
  the review. V57-004 already separated Audit completion and removed the
  intermediate Audit JSON limit; V57-005 justified retaining production
  transport without reuse.
- The toolset series was renumbered V56 → V61 (`c41a5989`). In working branch
  `v61-toolset-runtime-configuration` at `b3bb9ea7`, V61-001/002/004 are complete
  and V61-003 is under development. These changes are not yet in main, whose
  task files remain pending. Before integration, reconcile migration 000056,
  already used by Evals, and shared API/credentials files.

This updates the initial snapshot rather than describing the historical state
at V60's start. The [full open-backlog snapshot](2026-09-20-open-task-review.md)
lists all 29 tasks, confirmed activity, limitations and the order of independent work.

Review the ADK finalizer in light of V21-001 and subsequent history: a tool-using
agent does not provide the required structured output on the model path in use,
so a separate tool-free call has a rationale. Removing it, changing shared wire
limits, Scheduler lease guarantees or recovery semantics requires a separate
contract decision and transition strategy; it is not an automatic quick win.

## Deeper review map

| Area / task | Read | Scenarios to verify | Completion condition |
| --- | --- | --- | --- |
| Execution lifecycle / V60-005 | specs 00, 04, 18, 20; `internal/scheduler`, control, persistence | cancel vs success, finalizing vs aborting, claim loss, crashes between durable states, queue pause/admission, concurrency=1 and saturation | Each scenario has an identified winning durable transition; stale calls cannot create invalid outputs. The limitation before claim-loss detection is documented separately. |
| Runtime/A2A / V60-006 | specs 01, 02, 07, 14, 15, 21, 25, 29; Runtime contracts, runner, supervisor, invoker | stale task/session/subtask, cancellation during tool/finalizer execution, allocation expiry, partial cleanup, mandatory finalizer vs optional summarizer, tool-only worker | Identity, terminal-result and cleanup guarantees hold; transport failures are distinct from result failures; no background process remains after confirmed release. |
| Artifacts and persistence / V60-007 | specs 03, 08, 09, 10, 13, 17, 18, 23, 24; artifactstore, blobstore, migrations | owner/project/run authority, stale CAS, retained exact ref, deletion vs admission, corruption, restart, filesystem backend, minimal pool | Real DB/filesystem checks demonstrate atomicity and preservation; concurrent writes cannot bypass admission fences; no nested wait on the same resource. |
| Audit / V60-008 | specs 19, 25, 27; auditservice/store/coordinator, findings intake | proposed→approve/reject, duplicate/reopen, receipt retention, revisions during list/read, multiround restart, review TTL, intake replay | API/UI show consistent state, stale decisions are rejected, replay does not repeat settlement, and the published report matches its frozen revision. |
| User journeys / V60-008 | spec 06, `ui-user-stories.md`, UI routes/API helpers, CLI | login→Project→input→Run→cancel/retry→result; Audit→finding→review→report; empty/error/reconnect, long text, Unicode, read-only owner | Browser completes the journey with a real server and fake Gateway; DOM assertions check action and outcome, not merely absence of errors. CLI sends the expected bytes. |
| Auth and secrets / V60-009 | spec 06; auth, public/private middleware, Git SSH/runtime credentials | malformed body, UTF-8/escaping, cookie/bearer distinction, Origin/CSRF, duplicate headers, lockout/retry, TLS identities, secret projections/logs | Required permissions are checked before mutation; supported input is reachable through HTTP; secrets are absent from the checked public responses and diagnostics. |
| Configuration and cross-language / V60-006, V60-010 | specs 00, 01, 07, 16, 26, 29; configs, runtimeconfig, allocation DTOs, generated clients | pinned vs mutable refs, defaults/clear/absence, existing stored versions, string/number/bool mismatches, rejected extras, effective model/credential route | Shared wire cases are checked with real Go/Python readers; publication and resolution preserve accepted semantics; rejection occurs at the correct boundary. |
| Tools/workspace / V60-006 | specs 10–13, 21, 27, 29; filesystem, HTTP/Caido, code analysis, scanners | path traversal/symlink, network timeout, tool subprocess timeout, output cap, changed/stale snapshots, cancellation, wrong allocation | Capability/ownership isolation holds; timeouts release resources; source changes are atomic rather than partially applied. Active V55 scope is not taken over. |
| Installation, migrations, recovery / V60-007, V60-009 | deployment/testing guides, CLI migrate, manifests, schema guards | fresh disposable install, supported upgrade, refusal of a newer schema, interrupted migration, missing secret, restore instructions | An executed, repeatable scenario demonstrates an acceptable state after failure. An SQL file alone does not prove upgrade/recovery. |
| CI and test credibility / V60-009 | Makefile, CI, gate scripts, test matrices | required tests actually started/passed; skip/fail/missing evidence, secret redaction, nonzero subprocess, env guards, pinned generation | Gate neither accepts omissions nor rejects valid results because of log processing. Required process/browser tests are visible in the result; recorded evidence matches the commit and toolchain. |
| Performance / V60-009 | spec 22; metrics, pool usage, bounded reads, runtime memory | minimal pool, concurrent pages, cold/warm allocation, workspace digest/hash/parse, repeated data transfer | Establish baseline RSS/latency/query count, then compare on identical data. Without measurements, optimization remains a hypothesis. |
| Evals and product backlog / V60-010 | specs 05, 17, 26, 28, 30, V38/V40/V55 task files | implemented/draft distinction, frozen plan, model attribution, missing evidence, determinism/scorer inputs | Control-plane contracts and task→evidence links are verified; offline fixtures do not establish live quality. New tasks do not duplicate active ones. |

Specs 05 and 28 are reviewed separately as boundaries of unfinished product
work: their draft behavior is neither declared implemented nor automatically
treated as a defect in the current release. The map covers specifications
00–30, including [29 — Tool Workers](../spec/29-tool-workers.md) and
[30 — managed Evals](../spec/30-managed-evals.md).

## Checks by stage

The commands below are selected entry points, not claims that tests have already
run. Check their actual dependencies in the current Makefile before execution.
Record results separately for unit/fixture, real DB, process, browser and live-model
levels. An empty `-run` result or a skipped case does not count as PASS.

| Stage | Main commands / method | Conditions |
| --- | --- | --- |
| Baseline | `git status`, task YAML inventory, `git log -S`/`git blame`, specs and existing negative tests | Record commit, dirty paths and active task owners; do not repeat fixed or withdrawn recommendations. |
| Execution | `make test-faults`, `make test-lease-integration`, `make test-lifecycle-controls-hardening`, `make test-scheduler-concurrency-hardening` | Disposable PostgreSQL; named cases establish cancel/restart/lease windows, not just the happy path. |
| Runtime/contracts | `make verify-wire-contracts`, `make test-wire-cross-language`, `make test-runtime-hardening`, `make test-worker-session-modes-hardening`, `make test-worker-summarizer-hardening` | Locked Python env, fake LLM adapters; targeted tool tests from the map above. Podman checks only with confirmed prerequisites. |
| Storage | `make test-artifact-integration`, `make test-artifact-blob-backends`, `make test-git-artifacts`, targeted PostgreSQL pool/revision/deletion tests | Unique schemas and temporary directories; pool closure/cleanup after tests; upgrade/recovery separate from CRUD. |
| Audit/product | `make test-audits-hardening`, `make test-audit-completion-e2e`, `make test-findings-e2e`, `make test-audits-browser`, `make test-lifecycle-controls-browser` | First ensure wrappers do not hide skips or evidence-parsing errors; real server + fake Gateway for process/browser checks. |
| API/auth/UI | `make verify-public-api`, `make verify-public-api-postgres`, focused auth/HTTP tests, UI typecheck/lint/Vitest/build | Schema-only success is supplemented by actual HTTP and generated request bodies. The PostgreSQL gate must execute. |
| Delivery | `make release-verify` and separate upgrade/restore scenarios | Only after the environment and relevant fixes are ready; record durations and which gated suites actually ran. A full release gate does not replace a restore test. |
| Evals/backlog | Read portable fixtures/format and run agreed offline checks of existing harnesses | Do not launch an experiment or take over V40/V55 without changing the assignment. |

## Finding registration rules

Each finding records: ID, priority, user impact, baseline commit, exact
functions/lines, violated requirement, decision history, reproduction, minimal
correction, regression and limits of the evidence.

Statuses: `confirmed` — reproduced; `hypothesis` — requires a specific check;
`accepted` — confirmed design limitation; `rejected` — suspicion not confirmed;
`fixed` — has a regression, implementation hash and verification.
A missing test initially represents a verification gap, not a proven defect.

Priority reflects impact: P1 — blocked important journey, lost/incorrectly
published data or broken authority; P2 — bounded reproducible defect;
P3 — misleading documentation without an established failure.
Task priority follows the follow-up P2 policy in `tasks/index.yml`;
finding and task priorities may therefore differ.

For a fix: set the task to `in_progress` before edits; reproduce on old code,
then apply the minimal change and a meaningful regression. Acceptance includes
adjacent constraints so removing one cap cannot open a closed object or bypass
owner/CAS. Each implementation commit is separate from completion metadata.
After merging others' code, rerun affected checks; documentation changes alone
do not require another full Runtime/eval run.

## First pass and fix queue

| Finding | Discovery | Task |
| --- | --- | --- |
| PR-01 | The 2 KiB login raw-body cap cannot fit a supported 1024-byte password after JSON escaping; direct Login succeeds, HTTP returns 400. | [V60-002](../../tasks/v60/v60-002-login-json-body-bound.yml): justified 8 KiB raw cap and boundary HTTP tests. |
| PR-02 | Provenance handler can label new data with old Audit/finding revisions during concurrent mutation. | [V60-003](../../tasks/v60/v60-003-provenance-consistent-reads.yml): revision fence and regression interleaving. |
| PR-03 | Provenance hydration requests a receipt from the pool while prior rows hold a connection; confirmed on PostgreSQL with pool=1. | V60-003: close rows before hydration and retain reproduction with real PostgreSQL. |
| PR-04 | Audit completion gate redacts the password across the entire JSON line. CI password `contractor` corrupts Package; `pass` corrupts Action; valid results fail the gate. | [V60-004](../../tasks/v60/v60-004-audit-gate-event-redaction.yml): separate identity/status from redacted diagnostics and check their composition. |
| PR-05 | After V57-004, the mandatory Runtime matrix refers to the old name of a strengthened test; the real gate rejects its absence after 331 successful Python tests. | [V60-011](../../tasks/v60/v60-011-audit-gate-runtime-matrix.yml): preserve required behavior under the current name, check test declarations and run the full gate. |
| PR-06 | Deleting a source Run changes retained provenance without incrementing Audit revision; old pins remain valid. | [V60-012](../../tasks/v60/v60-012-run-deletion-audit-revisions.yml), complete: atomic invalidation of all affected Audits; import/delete/purge and immutable report retry verified on PostgreSQL. |

V60-012 was closed with a local fix: Run lock during import, ordered Audit locks
during deletion and an atomic revision bump. The report timestamp is preserved
for `finalizing`; a naive timestamp bump reproduced an immutable collision.
The [decision and checks](../research/2026-09-20-run-deletion-audit-revisions.md)
explain why no subsystem rewrite was needed.

V60-008 confirmed the collision when one receipt is imported into two Audits.
V60-027 fixes it with additive migration 62: new finding and assessment
identities are scoped to their Audit, while legacy IDs, history and same-Audit
replay remain unchanged. Actual PostgreSQL checks cover upgrade/replay,
independent review decisions, retention and source deletion.

First-pass observations remain in the [original report](../research/2026-09-20-project-review-first-pass.md).
The [V60 completion report](../research/2026-09-20-v60-deep-review-results.md) records
the deeper checks, original failed attempts, corrections and final mandatory
gates, with the demonstrated scenarios and limits of each verification level.
