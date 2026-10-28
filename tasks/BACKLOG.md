# Backlog

Generated from the task files by `make tasks-backlog`; do not edit by hand.
`make lint` fails when this file is stale. New tasks start from
[templates/](templates/); the task policy is in [index.yml](index.yml) and
the epics are in [epics.yml](epics.yml).

119 open tasks: 5 in progress, 103 pending, 11 deferred. 1187 finished: 1177 completed, 10 archived.

## In progress

| Task | Epic | Kind | Priority | Title |
| --- | --- | --- | --- | --- |
| [V40-002](v40-002-agent-instruction-paired-runner.yml) | evals | feature | P1 | Prepare instruction experiment bindings and a frozen plan after format readiness |
| [V261-001](v261-001-stabilize-release-gate-process-tests.yml) | quality | chore | P1 | Restore the deterministic release process gate |
| [V341-001](v341-001-investigate-ci-tolerance-bumps.yml) | quality | chore | P2 | Investigate tolerances raised to unblock CI |
| [V407-039](v407-039-close-server-runtime-review.yml) | quality | chore | P3 | Close the validated Server and Runtime review with explicit dispositions and verification |
| [V410-004](v410-004-sitemap-structured-imports.yml) | scan-tools | feature | P1 | Import bounded scanner and selected Caido observations |

## Pending

### Run execution core (`execution-core`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-014](v371-014-scheduler-stage-history-reads.yml) | defect | P3 | — | Count escalation attempts and check the first tick without reading Stage snapshots |
| [V372-007](v372-007-terminal-stage-state-set.yml) | chore | P3 | — | Define the terminal StageExecution states once and hold Go SQL and OpenAPI to them |
| [V372-028](v372-028-runstore-strict-snapshot-reads.yml) | chore | P3 | — | Decode persisted Run, Stage and allocation snapshots strictly in runstore |
| [V414-014](v414-014-standby-shutdown-race.yml) | defect | P3 | — | Exit a standby Server cleanly when shutdown races its readiness listener |

### Planners and the Worker model boundary (`planner-worker`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-004](v371-004-worker-completion-exact-integers.yml) | defect | P3 | — | Bound WorkerCompletion integers to the range A2A carries exactly |
| [V371-005](v371-005-worker-state-read-cost-guard.yml) | chore | P3 | — | Guard the Worker State read cost with a deterministic bound, make its benchmark runnable and re-measure it |
| [V371-006](v371-006-workspace-observation-descriptor.yml) | chore | P3 | — | Take the Worker tools that observe the workspace from the shared Toolset descriptor |
| [V414-007](v414-007-planner-session-state-limit.yml) | chore | P3 | — | Register the 2 MiB Planner session state bound and the other unregistered stored-size CHECKs |
| [V417-001](v417-001-terminal-summarizer-include-reasoning.yml) | feature | P1 | — | Make terminal summarizer reasoning input explicitly configurable |
| [V417-003](v417-003-isolated-worker-context-compaction.yml) | feature | P1 | — | Continue an isolated Worker through workflow-selected context compaction |
| [V417-004](v417-004-shared-session-context-compaction.yml) | feature | P1 | V417-003 | Preserve compacted context across shared-session invocations |
| [V417-006](v417-006-compactor-model-policy-selection.yml) | feature | P1 | V417-003 | Resolve reference-only compactor model policy overrides |
| [V417-007](v417-007-context-compaction-state-and-ui.yml) | feature | P1 | V417-003 | Report compaction progress and costs without exposing context content |
| [V420-002](v420-002-planner-compaction-vertical-slice.yml) | feature | P1 | — | Implement opt-in bounded compaction for Streamline and Router Planners |
| [V420-003](v420-003-planner-compaction-fault-scale-conformance.yml) | chore | P1 | V420-002 | Verify Planner compaction persistence cancellation and scale boundaries |
| [V420-004](v420-004-planner-compaction-diagnostics.yml) | feature | P1 | V420-002 | Expose bounded truthful Planner compaction diagnostics |

### Runtime labels, configuration and credentials (`runtime-config`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-006](v372-006-runtime-label-set-rule.yml) | chore | P3 | — | Give the Runtime label set rule one Go definition and hold the UI and the Runtime to it |

### HTTP and Caido tools (`http-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V386-004](v386-004-caido-catalog-cardinality.yml) | defect | P2 | — | Keep Caido catalog selection usable above one hundred entries |
| [V410-008](v410-008-sitemap-caido-history.yml) | feature | P2 | V410-004, V410-007 | Import selected Caido history incrementally with durable checkpoints |
| [V410-009](v410-009-sitemap-browser-collection.yml) | feature | P2 | V410-007, V412-001 | Collect managed Playwright network observations into the sitemap |
| [V411-004](v411-004-controlled-http-conformance.yml) | chore | P1 | V411-003 | Verify controlled HTTP transport against an independent denial and fault matrix |
| [V411-005](v411-005-identity-sessions-http-login.yml) | feature | P1 | V411-004 | Isolate target identities and execute bounded private HTTP login recipes |
| [V411-009](v411-009-http-recipes-predicates.yml) | feature | P1 | V411-005, V411-007, V411-008 | Execute frozen HTTP recipes and evaluate trusted three-valued predicates |
| [V412-001](v412-001-managed-pentest-browser.yml) | feature | P2 | V411-007, V411-008, V411-010 | Add a managed browser with isolated contexts and enforced target transport |
| [V412-002](v412-002-browser-login-replay-recipes.yml) | feature | P2 | V412-001, V411-009 | Execute closed browser login and replay recipes with trusted markers |
| [V412-004](v412-004-trusted-pentest-callback-proof.yml) | feature | P2 | V411-009, V411-012, V411-020 | Integrate trusted Interactsh callback evidence and its release gate |

### Scanner tool Workers and scan planning (`scan-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V55-009](v55-009-scan-candidate-ranking.yml) | feature | P2 | — | Optionally rank existing scan candidates with an LLM |
| [V412-005](v412-005-controlled-scanner-pentest-integration.yml) | feature | P2 | V411-020, V410-004 | Gate controlled ffuf execution through the pentest transport |

### Findings tools and collections (`findings`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-011](v371-011-proposal-quota-limits.yml) | chore | P3 | — | Register the per-proposal evidence quota and hold both proposal quotas to the registry |
| [V372-023](v372-023-findings-tool-error-codes.yml) | chore | P3 | — | Name the findings error codes the open question lists and register the findings tool codes |
| [V397-006](v397-006-finding-analysis-resolver.yml) | feature | P2 | — | Resolve optional finding analysis against exact retained evidence |
| [V397-007](v397-007-finding-export-snapshot.yml) | feature | P2 | V397-006 | Freeze explicit finding selections and independent Audit results |
| [V397-008](v397-008-generic-sarif-renderer.yml) | feature | P2 | V397-007 | Render deterministic validated generic SARIF from frozen snapshots |
| [V397-009](v397-009-sarif-export-api-ui.yml) | feature | P2 | V397-008 | Publish and download generic SARIF through API CLI and UI |
| [V411-007](v411-007-trusted-http-capture.yml) | feature | P1 | V411-004, V411-005, V411-006 | Expose safe HTTP evidence only through authenticated capture receipts |
| [V411-011](v411-011-project-findings-reader.yml) | feature | P1 | — | Expose safe live Project findings to allocated pentesters |

### Workflow catalog and document workflows (`workflows`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-013](v372-013-planner-facts-one-source.yml) | chore | P3 | — | Take Planner refs and facts from the descriptor registry in Go, the composition and Studio |
| [V372-014](v372-014-execution-config-patch-strict-decode.yml) | chore | P3 | — | Decode executionConfig patch members with strictjson |
| [V372-015](v372-015-config-split-leftovers.yml) | chore | P3 | — | Stop locals from shadowing imported packages and drop config's copy of clone.Map |
| [V410-006](v410-006-sitemap-workflow-composition.yml) | feature | P1 | V410-004, V410-005 | Ship ordinary sitemap discovery and reconciliation Workflows |
| [V411-014](v411-014-pentest-discovery-surface.yml) | feature | P1 | V411-007, V410-006 | Discover URL-only targets and retain bounded source-aware surfaces |
| [V411-015](v411-015-http-pentest-workflow-presets.yml) | feature | P1 | V411-012, V411-013, V411-014, V411-011, V411-010, V411-016 | Publish exact HTTP API pentest Workflows and source-mode presets |
| [V412-003](v412-003-browser-pentest-profiles-and-gate.yml) | feature | P2 | V412-002, V411-020, V410-009 | Publish optional browser programs behind their independent release gate |

### Projects, artifacts and lifecycle controls (`projects-artifacts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V410-005](v410-005-sitemap-openapi-reconciliation.yml) | feature | P1 | V410-004 | Reconcile exact OpenAPI declarations with live sitemap members |

### Audits (`audits`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V64-001](v64-001-audit-priority-contracts.yml) | feature | P2 | — | Specify optional Audit research contracts and deterministic bypass |
| [V64-002](v64-002-audit-priority-store.yml) | feature | P2 | V64-001, V64-012 | Persist research cycles and exact selection decisions |
| [V64-003](v64-003-planner-only-priority-execution.yml) | feature | P2 | V64-002 | Execute resumable research phases through ordinary Workflow Runs |
| [V64-004](v64-004-audit-priority-context.yml) | feature | P2 | V64-012, V64-003 | Assemble bounded exact research context for each experiment |
| [V64-005](v64-005-independent-priority-verdicts.yml) | feature | P2 | V64-003, V64-004, V64-006 | Add optional model evaluation of bounded experiment shortlists |
| [V64-006](v64-006-audit-priority-selection.yml) | feature | P2 | V64-002, V64-004, V64-013 | Select feasible experiments deterministically and admit exact work |
| [V64-007](v64-007-audit-priority-controller.yml) | feature | P2 | V64-003, V64-006, V64-013, V64-014, V64-015 | Close the deterministic hypothesis feedback loop across Audit rounds |
| [V64-008](v64-008-audit-priority-api-coverage.yml) | feature | P2 | V64-007 | Expose paginated research state and separate coverage through the public API |
| [V64-009](v64-009-audit-priority-ui.yml) | feature | P2 | V64-008 | Show hypotheses, selection reasons and bounded lineage in the Audit UI |
| [V64-010](v64-010-audit-priority-profile.yml) | feature | P2 | V64-007, V64-008 | Publish opt-in guide and material research profiles with deterministic selection |
| [V64-011](v64-011-audit-priority-release.yml) | chore | P2 | V64-009, V64-010, V64-015 | Verify deterministic hypothesis search and unchanged direct execution end to end |
| [V64-012](v64-012-audit-hypothesis-registry.yml) | feature | P2 | V64-001 | Store Audit hypotheses and sparse provenance with trusted ingestion |
| [V64-013](v64-013-audit-hypothesis-generation.yml) | feature | P2 | V64-003, V64-004, V64-012 | Generate cold hypotheses and accept bounded concrete experiment plans |
| [V64-014](v64-014-audit-hypothesis-assessments.yml) | feature | P2 | V64-003, V64-004, V64-012, V64-006 | Import scoped research assessments and preserve conflicting contributions |
| [V64-015](v64-015-audit-research-frontier.yml) | feature | P2 | V64-012, V64-014, V64-006 | Maintain indexed research readiness and bounded priority refresh |
| [V64-016](v64-016-audit-research-ranking-evaluation.yml) | chore | P2 | V64-005, V64-011 | Measure optional research ranking against deterministic baselines |
| [V64-017](v64-017-audit-attack-tree-projection.yml) | feature | P2 | V64-014, V64-008, V64-009 | Project optional attack trees over exact research evidence |
| [V64-018](v64-018-audit-research-pentest-integration.yml) | feature | P2 | V64-011, V411-012, V411-015, V411-016, V411-020 | Connect optional hypothesis research to pentest proof and reports |
| [V411-003](v411-003-action-journal-request-permits.yml) | feature | P1 | — | Fence and execute scoped HTTP preflight through the durable Audit action ledger |
| [V411-006](v411-006-cumulative-budget-finalization.yml) | feature | P1 | V411-003 | Enforce cumulative pentest budgets and finalize at the hard active-time cap |
| [V411-008](v411-008-stateful-actions-cleanup.yml) | feature | P1 | V411-003, V411-004, V411-006, V411-007 | Serialize stateful target checks and retain unresolved cleanup effects |
| [V411-010](v411-010-pentest-program-availability.yml) | feature | P1 | — | Pin optional pentest programs and nonblocking finding review |
| [V411-012](v411-012-pentest-admission-rounds.yml) | feature | P1 | V411-009, V411-010 | Admit typed pentest plans and derive exact later-round replay |
| [V411-016](v411-016-pentest-proof-report.yml) | feature | P1 | V411-012, V411-014 | Assemble retained pentest proof and incomplete-coverage reports |
| [V414-020](v414-020-preparation-phase-one-rule.yml) | chore | P3 | — | Give the pre-Round preparation phase set one definition in Go and one in the UI |
| [V417-005](v417-005-audit-worker-context-compaction.yml) | feature | P1 | V417-003, V417-004 | Admit compaction without bypassing trusted Audit completion |

### Evaluations (`evals`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V40-003](v40-003-agent-instruction-eval-decision.yml) | feature | P1 | V40-002 | Evaluate instruction candidates and record evidence-based adoption decisions |
| [V411-019](v411-019-pentest-target-fixtures-and-proof-scorers.yml) | chore | P1 | — | Build paired pentest fixtures and independent proof scorers |
| [V411-021](v411-021-pentest-frozen-live-quality-evaluation.yml) | chore | P2 | V411-020 | Evaluate pentest quality against a frozen repeated-sample baseline |
| [V414-022](v414-022-eval-purge-guard-lexer.yml) | chore | P3 | — | Read trigger functions with the explaintest lexer in the Eval purge coverage guard and check rewrite rules |
| [V417-008](v417-008-context-compaction-eval-harness.yml) | chore | P2 | V417-001, V417-003 | Build frozen context-compaction evaluation fixtures and an opt-in harness |
| [V417-010](v417-010-context-compaction-live-evaluation.yml) | chore | P2 | V417-008, V417-009 | Measure compaction and reasoning choices with repeated frozen live experiments |

### Podman sandbox (`sandbox`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V411-013](v411-013-pentest-sandbox-integration.yml) | feature | P1 | V411-006 | Integrate bounded sandbox investigations and tool installation |

### Web UI (`ui`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V403-003](v403-003-openapi-preview-at-1mib.yml) | defect | P3 | — | Keep a 1 MiB OpenAPI preview from blocking the page |
| [V411-018](v411-018-pentest-ui-journey.yml) | feature | P1 | V411-015, V411-017 | Build the autonomous HTTP pentest setup and evidence journey |

### Public API and wire contracts (`api-contracts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-020](v372-020-error-code-vocabularies.yml) | chore | P3 | — | Hold every vocabulary that carries error code names to the registry |
| [V372-025](v372-025-typed-openapi-enums.yml) | chore | P3 | — | Give every enum of the public OpenAPI document a type |
| [V372-026](v372-026-request-digests-in-services.yml) | chore | P3 | — | Move the Project, Audit and review request digests out of the public handlers |
| [V411-017](v411-017-pentest-api-cli.yml) | feature | P1 | V411-010, V411-011, V411-012, V411-016, V411-006 | Expose pentest drafts progress surfaces and proof through API and CLI |
| [V414-009](v414-009-machine-value-byte-bounds.yml) | chore | P3 | — | Declare the byte bounds of machine values in the public API and hold OpenAPI, Go and the UI to them |

### PostgreSQL persistence and schema (`persistence`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-008](v372-008-upward-table-reads.yml) | chore | P3 | — | Remove the table reads that point up the layer stack or name who owns each one |
| [V372-009](v372-009-ownership-check-blind-spots.yml) | chore | P3 | — | Close the blind spots of the table ownership check |
| [V372-010](v372-010-erd-usage-from-ownership-scan.yml) | chore | P3 | — | Take the ERD's code-usage lists from the ownership scan and document the new-table workflow |
| [V414-021](v414-021-migration-file-deadline.yml) | defect | P3 | — | Decide the client deadline of a migration file: per statement or a migration budget of its own |
| [V414-024](v414-024-plan-tests-exact-reads.yml) | chore | P3 | — | Measure the V407-025 drain and held-credential row budgets with explaintest Analysis.Reads |

### Quality, CI and process (`quality`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V347-001](v347-001-protect-main-branch.yml) | chore | P1 | — | Protect main with a required check |
| [V371-015](v371-015-memory-claim-paused-skip.yml) | chore | P3 | — | Model the paused-owner claim skip in the Scheduler's memory store doubles |
| [V371-021](v371-021-postgres-test-wall-clock.yml) | chore | P3 | — | Replace wall-clock budgets in the Scheduler and runstore PostgreSQL tests |
| [V371-022](v371-022-capability-probe-test-budget.yml) | chore | P3 | — | Remove the 10 ms probe budget from the capability discovery tests |
| [V372-016](v372-016-wall-clock-test-budgets-wave-2.yml) | chore | P3 | — | Replace wall-clock budgets in the finding intake, Audit, Eval and persistence tests and the result publication test |
| [V372-024](v372-024-audit-item-attempts-fake.yml) | chore | P3 | — | Make the Audit item attempt fake answer like the store and keep the item list schema-valid without it |
| [V372-029](v372-029-shared-fixture-gate-scope.yml) | chore | P3 | — | Extend the shared fixture gate to every cross-language fixture |
| [V372-030](v372-030-shrink-test-only-export-baseline.yml) | chore | P3 | — | Resolve the exported identifiers that only tests use and empty the test-only export baseline |
| [V401-007](v401-007-runtime-suite-on-macos.yml) | chore | P3 | — | Make the Runtime suite pass on macOS: declare the Linux-only Podman tests, make the other tests portable and file the macOS defects |
| [V410-007](v410-007-sitemap-release-gate.yml) | chore | P1 | V410-006 | Verify the first sitemap slice across scale and lifecycle boundaries |
| [V411-020](v411-020-pentest-http-release-gate.yml) | chore | P1 | V411-003, V411-004, V411-005, V411-006, V411-007, V411-008, V411-009, V411-010, V411-011, V411-012, V411-013, V411-014, V411-015, V411-016, V411-017, V411-018, V411-019, V410-007 | Gate the first HTTP API pentest delivery through real processes |
| [V414-027](v414-027-close-review-followups.yml) | chore | P3 | V414-007, V414-009, V414-014, V414-020, V414-021, V414-022, V414-024 | Close the V407/V409 review follow-ups |
| [V417-009](v417-009-context-compaction-release-gate.yml) | chore | P1 | V417-003, V417-004, V417-005, V417-006, V417-007 | Verify context compaction across real processes and close the deterministic milestone |
| [V420-005](v420-005-planner-compaction-release-gate.yml) | chore | P1 | V420-003, V420-004 | Accept Planner compaction through real-process and integrated release gates |

## Deferred

| Task | Epic | Since | Reactivate when |
| --- | --- | --- | --- |
| [V61-001](v61-001-toolset-configuration-contracts.yml) | runtime-config | 2026-10-11 | Explicit user decision that external MCP tools are needed (for example for autonomous pentest Audits, proposal 0033); then port the branch onto main rather than reimplementing, renumber its migration and rerun the V61-009 release gate. |
| [V61-002](v61-002-toolset-runtime-config-store.yml) | runtime-config | 2026-10-11 | Explicit user decision that external MCP tools are needed (for example for autonomous pentest Audits, proposal 0033); then port the branch onto main rather than reimplementing, renumber its migration and rerun the V61-009 release gate. |
| [V61-003](v61-003-run-toolset-pinning-allocation.yml) | runtime-config | 2026-10-11 | Explicit user decision that external MCP tools are needed (for example for autonomous pentest Audits, proposal 0033); then port the branch onto main rather than reimplementing, renumber its migration and rerun the V61-009 release gate. |
| [V61-004](v61-004-allocation-tool-discovery.yml) | runtime-config | 2026-10-11 | Explicit user decision that external MCP tools are needed (for example for autonomous pentest Audits, proposal 0033); then port the branch onto main rather than reimplementing, renumber its migration and rerun the V61-009 release gate. |
| [V61-005](v61-005-mcp-session-lifecycle.yml) | runtime-config | 2026-10-11 | Explicit user decision that external MCP tools are needed (for example for autonomous pentest Audits, proposal 0033); then port the branch onto main rather than reimplementing, renumber its migration and rerun the V61-009 release gate. |
| [V61-006](v61-006-mcp-tools-adk.yml) | runtime-config | 2026-10-11 | Explicit user decision that external MCP tools are needed (for example for autonomous pentest Audits, proposal 0033); then port the branch onto main rather than reimplementing, renumber its migration and rerun the V61-009 release gate. |
| [V61-007](v61-007-toolset-configuration-read-api.yml) | runtime-config | 2026-10-11 | Explicit user decision that external MCP tools are needed (for example for autonomous pentest Audits, proposal 0033); then port the branch onto main rather than reimplementing, renumber its migration and rerun the V61-009 release gate. |
| [V61-008](v61-008-toolset-connections-ui.yml) | runtime-config | 2026-10-11 | Explicit user decision that external MCP tools are needed (for example for autonomous pentest Audits, proposal 0033); then port the branch onto main rather than reimplementing, renumber its migration and rerun the V61-009 release gate. |
| [V61-009](v61-009-toolset-configuration-release.yml) | runtime-config | 2026-10-11 | Explicit user decision that external MCP tools are needed (for example for autonomous pentest Audits, proposal 0033); then port the branch onto main rather than reimplementing, renumber its migration and rerun the V61-009 release gate. |
| [V62-008](v62-008-project-analysis-preparation.yml) | audits | 2026-09-20 | Explicit user approval of standalone analysis-report reuse and the additional two-Workflow chain. |
| [V365-016](v365-016-fold-planner-execution-reports.yml) | persistence | 2026-10-04 | Explicit user approval of making Planner reports a mutable slot on stage_metrics. |

## Epics

All tasks of each epic by kind, and how many later tasks name one of its
tasks in `introduced_by`.

| Epic | Open | Feature | Defect | Chore | Later fixes caused |
| --- | ---: | ---: | ---: | ---: | ---: |
| Run execution core (`execution-core`) | 4 | 41 | 56 | 20 | 84 |
| Planners and the Worker model boundary (`planner-worker`) | 12 | 34 | 47 | 28 | 64 |
| Runtime labels, configuration and credentials (`runtime-config`) | 10 | 26 | 13 | 4 | 43 |
| Shared Memory tools (`memory`) | 0 | 6 | 9 | 2 | 12 |
| Agent Skills (`agent-skills`) | 0 | 11 | 4 | 2 | 4 |
| Run workspaces and project filesystem (`workspaces`) | 0 | 16 | 14 | 5 | 16 |
| Code analysis and taint annotations (`code-analysis`) | 0 | 14 | 14 | 2 | 19 |
| HTTP and Caido tools (`http-tools`) | 9 | 20 | 22 | 6 | 19 |
| Scanner tool Workers and scan planning (`scan-tools`) | 3 | 16 | 9 | 9 | 18 |
| Findings tools and collections (`findings`) | 8 | 17 | 7 | 8 | 12 |
| Workflow catalog and document workflows (`workflows`) | 7 | 18 | 15 | 12 | 19 |
| Projects, artifacts and lifecycle controls (`projects-artifacts`) | 1 | 36 | 33 | 9 | 52 |
| Audits (`audits`) | 27 | 67 | 82 | 32 | 113 |
| Evaluations (`evals`) | 7 | 21 | 39 | 15 | 48 |
| Podman sandbox (`sandbox`) | 1 | 10 | 2 | 1 | 3 |
| Web UI (`ui`) | 2 | 25 | 66 | 14 | 58 |
| Operations, authentication, CLI and deployment (`operations`) | 0 | 18 | 43 | 7 | 18 |
| Public API and wire contracts (`api-contracts`) | 5 | 2 | 37 | 22 | 12 |
| PostgreSQL persistence and schema (`persistence`) | 6 | 0 | 24 | 44 | 13 |
| Quality, CI and process (`quality`) | 17 | 1 | 27 | 102 | 23 |
