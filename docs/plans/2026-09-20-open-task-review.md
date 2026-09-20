# Open Contractor tasks — 2026-09-20

Snapshot around 08:42 MSK: `main` at `15974cba`, with working branches and
uncommitted plans checked separately. Evidence comes from task YAML, code,
history, changes in all accessible worktrees and the two relevant Playground
checkouts. This is a backlog-state review, with no implementations, tests or
live evaluations launched. Local machine-readable inventory:
`.local/open-task-review-2026-09-20.json`.

The main worktree contains **381 tasks: 349 completed, 26 pending,
3 in_progress, 3 archived**. The active open backlog has **29 tasks**.
Archived V57-003 is still held in uncommitted planning and is included in these
counts. Completion in a separate branch below does not change the status of
code delivered to `main`.

## Work actually in progress

| Stream | Actual state | Evidence |
| --- | --- | --- |
| **V38-006 — Eval results and comparisons** | Implementation is underway on recent `main`, not yet integrated | Worktree `v38-006-comparison`, HEAD `15974cba`; staged changes and new collection/assessment/review/comparison/SQL files. Changes were being added during the review. |
| **V61-003 — toolset config in Run/Audit and allocation** | Implementation is underway; earlier stages are ready only in the branch | Worktree `v61-toolset-runtime-configuration`, HEAD `b3bb9ea7`; V61-001/002/004 completed, V61-003 in_progress, with changing pinning, resolver, placement and secret-materialization code. |
| **V40-002 — instruction experiment preparation** | Formally in_progress; preparation is complete, but exact live pins and related capabilities block acceptance | Contractor `9b1613ab` and Playground `d97ddc55` are integrated into their respective main branches. Both task worktrees are clean; no fresh changes for this task were found. |
| **V55-004 — sqlmap with a complete HTTP request** | Formally in_progress; implementation and active development were not confirmed in accessible checkouts | `b05b1a6b` changed the status on September 19. SQLMapTool still accepts URL/data/cookie; the request artifact, `-r` and `test_scan_http_request.py` are absent. |

The absence of a local diff does not prove that no work is taking place elsewhere.
V40-002 and V55-004 were neither reassigned nor moved to pending. However, they
should not be presented as two confirmed active coding streams.

The old `v38-evals-experience` contains earlier drafts of the same V38-006 work;
it has no unique commits relative to main. It is not another independent Evals
implementation. The old, clean `v56-toolset-runtime-configuration` contains the
plan before renumbering; the current series is V61.

## Evals and instruction evaluation — 7 tasks

**V38-001–005 are complete and in main**: contracts, DTOs, storage, coordinator
and public setup/control API. Subsequent review corrected the core and verified
its integration with V60-012. [Results](2026-09-20-managed-evals-core-review.md).

| Task | Remaining work | Relevance and dependency |
| --- | --- | --- |
| [V38-006](../../tasks/v38-006-eval-assessment-comparison.yml) | Collection, attributed assessments, review, CAS selection, full snapshots, pairs, charts/report/inventory API and checks | Active development; foundation for the remaining Evals features. |
| [V38-007](../../tasks/v38-007-eval-setup-ui.yml) | Dataset authoring and the full experiment preparation/start UI | Relevant after 006. A ready setup backend does not replace this UI. |
| [V38-008](../../tasks/v38-008-eval-comparison-ui.yml) | Comparison UI, review, evidence, charts and exports | After 007. The existing Evals workspace does not cover the full journey. |
| [V38-009](../../tasks/v38-009-playground-eval-client.yml) | Playground as a managed Evals API client | Optional extension after 006. Not a duplicate of the portable/direct V41 runner; native Contractor does not depend on this client. |
| [V38-010](../../tasks/v38-010-eval-release-gate.yml) | Full native/external process/browser gate: restart, ownership, completeness, deletion | Needed; currently depends on 008 and 009. Reconsider the optional Playground dependency before scheduling release. |
| [V40-002](../../tasks/v40-002-agent-instruction-paired-runner.yml) | Valid model/runtime/tools/Skills/budget pins, six-program dispatch, input/capability mapping, normalization and receipt collection | Prepared wrappers and the recorded 24-member matrix are preserved. Strict live readiness remains false. |
| [V40-003](../../tasks/v40-003-agent-instruction-eval-decision.yml) | Live pilot, independent confirmation and adopt/revise/reject/inconclusive decisions | After 002; offline fixtures and managed Evals do not establish instruction quality. |

V40-002 has a scope-organization problem: generic dispatch/normalization is
required for acceptance, but generic runner implementation is explicitly excluded.
Before resuming, resolve these prerequisites and update the frozen experiment's
assumptions. Older candidates intentionally retain the previous completion
contract; mixing them with the new implementation under the old experiment
label is not acceptable. Publishing wrappers alone does not remove the blockers.

V38-010's mandatory dependency on 009 is broader than the product's runtime
dependency: its scope requires an independent standard HTTP producer, while
Playground evidence is described as a separate optional gate. If native release
is needed before the client, separating these gates is reasonable. This review
did not change dependencies/acceptance or remove external API verification.

## Scans — 7 tasks

V55-001–003 have delivered an extensible scan toolset and model-free Worker.
The complete first user-facing set is not yet finished.

| Task | Remaining work | Relevance and dependency |
| --- | --- | --- |
| [V55-004](../../tasks/v55-004-sqlmap-http-request.yml) | Complete HTTP request artifact, exact method/headers/body, private `sqlmap -r`, provenance and cleanup | Needed; first clarify the previous assignment's status. `in_progress` alone does not mean ready. |
| [V55-005](../../tasks/v55-005-ffuf-wordlist-artifacts.yml) | ffuf with a validated wordlist from an exact ArtifactRef | Needed for the first set; dependencies are complete. Shared scan files require coordination with 004. |
| [V55-006](../../tasks/v55-006-scan-workflows-release.yml) | sqlmap/ffuf Workflows, input/wordlist upload, a connected user journey and E2E | After 004 and 005. Nuclei/naabu fixtures from 003 deliver only part of this outcome. |
| [V55-007](../../tasks/v55-007-openapi-request-set.yml) | Deterministic RequestSet from OpenAPI | Relevant extension after 006, not a prerequisite for the first manual scans. |
| [V55-008](../../tasks/v55-008-deterministic-scan-planner.yml) | Persisted job plan, constraints and recovery | After 007. `internal/scanplan` is still absent. |
| [V55-009](../../tasks/v55-009-scan-candidate-ranking.yml) | Optional LLM ranking of already permitted candidates | After 008; evaluate usefulness against the ready deterministic path. Not mandatory technical debt. |
| [V55-010](../../tasks/v55-010-katana-scan-adapter.yml) | Katana as a bounded discovery adapter | After 007; a deferred extension, with no adapter implemented yet. |

Practical path to the first result: **004 + 005 → 006**. Parallel implementation
of 004/005 requires separating edits to `scan/tools.py`, the factory and tests.

## Toolset configuration and MCP — 9 records in main

This is V61; the former V56 designation no longer represents a separate backlog.
All nine tasks remain pending in main. Three are complete in the branch:

| Task | Deliverable | Actual state |
| --- | --- | --- |
| [V61-001](../../tasks/v61-001-toolset-configuration-contracts.yml) | Contracts, typed settings and discovery schema | Completed in branch, implementation `323b54eb`; not in main. |
| [V61-002](../../tasks/v61-002-toolset-runtime-config-store.yml) | RuntimeConfig merge, MCP credentials and lifecycle references | Completed in branch, `761068d1`; not in main. |
| [V61-004](../../tasks/v61-004-allocation-tool-discovery.yml) | Allocation discovery, placement and Audit admission | Completed in branch, `9d74c38b`; not in main. |
| [V61-003](../../tasks/v61-003-run-toolset-pinning-allocation.yml) | Immutable Run/Audit pins, allocation setting selection and secret delivery | Active development. |
| [V61-005](../../tasks/v61-005-mcp-session-lifecycle.yml) | MCP Streamable HTTP sessions, ownership, proxy and cleanup | After 003. The real MCP transport is still absent. |
| [V61-007](../../tasks/v61-007-toolset-configuration-read-api.yml) | Public requirements, coverage and provenance | After 003; can proceed alongside 005. |
| [V61-006](../../tasks/v61-006-mcp-tools-adk.yml) | MCP tools in ADK and shared accounting | After 005. Fake factories in discovery tests do not establish production MCP readiness. |
| [V61-008](../../tasks/v61-008-toolset-connections-ui.yml) | Connection selection/creation UI and draft recovery | After 007. |
| [V61-009](../../tasks/v61-009-toolset-configuration-release.yml) | Final DB/Runtime/process/browser gate | After 006 and 008. |

Before integration, update the branch to current main. The common base is
currently `c41a5989`; subsequent changes overlap in OpenAPI,
`runtime_repository.go`, public types, the generated client, UI RuntimeConfig
and spec 19.

There is a **specific migration-version collision**: main contains
`000056_managed_evals.sql`, while V61 has `000056_mcp_runtime_credentials.sql`.
`loadMigrations` rejects two files with the same version. V38-006 is also already
creating `000059_eval_selected_views.sql`. Coordinate the new V61 migration
number when updating the branch; an already applied main migration must not be
renamed. This is a future integration issue, not a current main failure.

## Deeper review — 6 tasks

V60-001–004, V60-011 and V60-012 are complete; six first-pass defects have been
fixed. The following tasks are review areas, not six proven new defects.
Their pending status remains appropriate.

| Task | Actual remaining scope | How to avoid repeating completed work |
| --- | --- | --- |
| [V60-005](../../tasks/v60-005-execution-recovery-review.yml) | Cancel/success/restart/claim-loss and queue-admission interleavings | Start with durable state transitions; V61 is changing Scheduler execution preparation, so coordinate that area. |
| [V60-006](../../tasks/v60-006-runtime-contract-tools-review.yml) | Identity, cancellation/cleanup, cross-language contracts and tools | Strong overlap with V61 Runtime/private contracts. A complete pass is better after integrating 003. Do not automatically resume V57-003. |
| [V60-007](../../tasks/v60-007-artifact-persistence-review.yml) | Artifact CAS/ownership, blobs, corruption/backend parity, migrations/recovery | Storage/read/CAS can be checked separately. Use completed V60-003/012 retention/pool/delete cases as evidence; Run holds/migrations overlap with V61 and Evals. |
| [V60-008](../../tasks/v60-008-audit-product-journeys-review.yml) | TTL/review, multiround, full browser journeys and importing the same receipt into multiple Audits | A specific identity collision was confirmed during V60-012. Start with its contract and a bounded correction; do not duplicate verified report-delete/replay cases. |
| [V60-009](../../tasks/v60-009-operations-security-gates-review.yml) | Auth, secrets, real install/restore and credible release gates | Cookie/auth/mTLS can be checked separately. Review shared credentials/schema/generated API and the final gate after active integrations. Do not repeat V60-004/011 as new investigations. |
| [V60-010](../../tasks/v60-010-configuration-evaluation-backlog-review.yml) | End-to-end configuration attribution and exact eval inputs→evidence links | This backlog review covers only the informational portion. The technical pass is needed after V61 pinning and V38 collection; do not create a parallel eval runner. |

Migration internals/atomicity belong to V60-007; operator startup/restore and
mandatory release gates belong to V60-009. This avoids two identical passes.
Existing verification results retain their original commits and are not relabeled
as new evidence.

## Excluded from the active queue

- **V53-001/002** — Audit budget/reasoning-only incident investigations,
  archived at the user's decision. Their problems are not declared fixed.
- **V57-003** — archived ordinary-completion investigation; V57-004 already
  delivered typed assembly. The original finalizer is preserved; no new
  fake-model environment is needed.
- **V57-001/002/004/005, V58, V59, V38-001–005 and V60 fixes** do not become
  new pending work because old branches or draft summaries remain.
- **V54 pentest and SARIF/spec 28** are separate draft directions. Their missing
  implementation does not automatically turn them into more mandatory tasks.

## Recommended order of attention

1. Finish active V38-006 and V61-003; before merging V61, coordinate migrations
   and shared API/credentials changes. This does not assign a second implementation.
2. For our relatively independent work, take a narrow V60-008 slice: the contract
   for importing one receipt into different Audits and its specific fix.
   The Artifact storage/CAS portion of V60-007 is an alternative.
3. For the nearest scan deliverable, clarify the V55-004 handoff and execute
   V55-005 → V55-006. Keep V55-009/010 as extensions for now.
4. Perform full V60-006/010 passes after active changes to their boundaries.
   Return V40 to execution after separately resolving live prerequisites.

This review did not change other tasks' statuses, dependencies, budgets or
current assignments. Outdated summaries were corrected; test results in task
files were read as historical evidence and are not claimed as rerun.
