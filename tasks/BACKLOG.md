# Backlog

Generated from the task files by `make tasks-backlog`; do not edit by hand.
`make lint` fails when this file is stale. New tasks start from
[templates/](templates/); the task policy is in [index.yml](index.yml) and
the epics are in [epics.yml](epics.yml).

119 open tasks: 10 in progress, 84 pending, 25 deferred. 1082 finished: 1072 completed, 10 archived.

## In progress

| Task | Epic | Kind | Priority | Title |
| --- | --- | --- | --- | --- |
| [V40-002](v40-002-agent-instruction-paired-runner.yml) | evals | feature | P1 | Prepare instruction experiment bindings and a frozen plan after format readiness |
| [V261-001](v261-001-stabilize-release-gate-process-tests.yml) | quality | chore | P1 | Restore the deterministic release process gate |
| [V341-001](v341-001-investigate-ci-tolerance-bumps.yml) | quality | chore | P2 | Investigate tolerances raised to unblock CI |
| [V407-039](v407-039-close-server-runtime-review.yml) | quality | chore | P3 | Close the validated Server and Runtime review with explicit dispositions and verification |
| [V410-010](v410-010-artifact-history-retention.yml) | projects-artifacts | feature | P1 | Apply automatic bounded history retention to every logical Artifact |
| [V411-001](v411-001-target-credential-revisions.yml) | runtime-config | feature | P1 | Pin immutable target credential revisions in the existing encrypted store |
| [V414-002](v414-002-eval-token-budget-overflow.yml) | evals | defect | P2 | Keep the Eval token budget and stage token summaries defined when token counters exceed int64 |
| [V414-019](v414-019-held-inbox-page-bound.yml) | audits | defect | P2 | Bound the held Audit inbox page by the page, not by the Audit's holds |
| [V414-031](v414-031-lock-wait-query-truncation.yml) | quality | defect | P3 | Identify a waiting statement within the query prefix PostgreSQL records |
| [V414-033](v414-033-pre-release-api-changes.yml) | api-contracts | chore | P3 | Allow incompatible /v1 changes until the first release |

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

### Runtime labels, configuration and credentials (`runtime-config`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-006](v372-006-runtime-label-set-rule.yml) | chore | P3 | — | Give the Runtime label set rule one Go definition and hold the UI and the Runtime to it |

### HTTP and Caido tools (`http-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V386-004](v386-004-caido-catalog-cardinality.yml) | defect | P2 | — | Keep Caido catalog selection usable above one hundred entries |
| [V410-003](v410-003-http-sitemap-collection.yml) | feature | P1 | V410-002 | Collect HTTP observations into the live Project sitemap |
| [V410-008](v410-008-sitemap-caido-history.yml) | feature | P2 | V410-004, V410-007 | Import selected Caido history incrementally with durable checkpoints |
| [V410-009](v410-009-sitemap-browser-collection.yml) | feature | P2 | V410-002, V410-007, V412-001 | Collect managed Playwright network observations into the sitemap |
| [V411-004](v411-004-controlled-http-conformance.yml) | chore | P1 | V411-003 | Verify controlled HTTP transport against an independent denial and fault matrix |
| [V411-005](v411-005-identity-sessions-http-login.yml) | feature | P1 | V411-001, V411-004 | Isolate target identities and execute bounded private HTTP login recipes |
| [V411-009](v411-009-http-recipes-predicates.yml) | feature | P1 | V411-005, V411-007, V411-008 | Execute frozen HTTP recipes and evaluate trusted three-valued predicates |
| [V412-001](v412-001-managed-pentest-browser.yml) | feature | P2 | V411-007, V411-008, V411-010 | Add a managed browser with isolated contexts and enforced target transport |
| [V412-002](v412-002-browser-login-replay-recipes.yml) | feature | P2 | V412-001, V411-009 | Execute closed browser login and replay recipes with trusted markers |
| [V412-004](v412-004-trusted-pentest-callback-proof.yml) | feature | P2 | V411-009, V411-012, V411-020 | Integrate trusted Interactsh callback evidence and its release gate |

### Scanner tool Workers and scan planning (`scan-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V55-009](v55-009-scan-candidate-ranking.yml) | feature | P2 | — | Optionally rank existing scan candidates with an LLM |
| [V410-004](v410-004-sitemap-structured-imports.yml) | feature | P1 | V410-002 | Import bounded scanner and selected Caido observations |
| [V412-005](v412-005-controlled-scanner-pentest-integration.yml) | feature | P2 | V411-020, V410-004 | Gate controlled ffuf execution through the pentest transport |
| [V414-004](v414-004-scan-audit-result-failure-classes.yml) | defect | P2 | — | Classify Audit scan result building failures by cause instead of as invalid input |

### Findings tools and collections (`findings`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-010](v371-010-discarded-receipt-projection.yml) | defect | P2 | — | Drop the report projection of a proposal receipt discarded by Run deletion |
| [V371-011](v371-011-proposal-quota-limits.yml) | chore | P3 | — | Register the per-proposal evidence quota and hold both proposal quotas to the registry |
| [V372-023](v372-023-findings-tool-error-codes.yml) | chore | P3 | — | Name the findings error codes the open question lists and register the findings tool codes |
| [V397-006](v397-006-finding-analysis-resolver.yml) | feature | P2 | — | Resolve optional finding analysis against exact retained evidence |
| [V397-007](v397-007-finding-export-snapshot.yml) | feature | P2 | V397-006 | Freeze explicit finding selections and independent Audit results |
| [V397-008](v397-008-generic-sarif-renderer.yml) | feature | P2 | V397-007 | Render deterministic validated generic SARIF from frozen snapshots |
| [V397-009](v397-009-sarif-export-api-ui.yml) | feature | P2 | V397-008 | Publish and download generic SARIF through API CLI and UI |
| [V411-007](v411-007-trusted-http-capture.yml) | feature | P1 | V411-004, V411-005, V411-006 | Expose safe HTTP evidence only through authenticated capture receipts |
| [V411-011](v411-011-project-findings-reader.yml) | feature | P1 | V411-002 | Expose safe live Project findings to allocated pentesters |

### Workflow catalog and document workflows (`workflows`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-013](v372-013-planner-facts-one-source.yml) | chore | P3 | — | Take Planner refs and facts from the descriptor registry in Go, the composition and Studio |
| [V372-014](v372-014-execution-config-patch-strict-decode.yml) | chore | P3 | — | Decode executionConfig patch members with strictjson |
| [V372-015](v372-015-config-split-leftovers.yml) | chore | P3 | — | Stop locals from shadowing imported packages and drop config's copy of clone.Map |
| [V410-006](v410-006-sitemap-workflow-composition.yml) | feature | P1 | V410-003, V410-004, V410-005 | Ship ordinary sitemap discovery and reconciliation Workflows |
| [V411-014](v411-014-pentest-discovery-surface.yml) | feature | P1 | V411-007, V410-003, V410-006 | Discover URL-only targets and retain bounded source-aware surfaces |
| [V411-015](v411-015-http-pentest-workflow-presets.yml) | feature | P1 | V411-012, V411-013, V411-014, V411-011, V411-010, V411-016 | Publish exact HTTP API pentest Workflows and source-mode presets |
| [V412-003](v412-003-browser-pentest-profiles-and-gate.yml) | feature | P2 | V412-002, V411-020, V410-009 | Publish optional browser programs behind their independent release gate |

### Projects, artifacts and lifecycle controls (`projects-artifacts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V410-001](v410-001-project-sitemap-store.yml) | feature | P1 | V410-010 | Publish a bounded Project sitemap through one authorized reducer |
| [V410-002](v410-002-project-sitemap-toolset.yml) | feature | P1 | V410-001 | Expose authorized live sitemap tools to Project allocations |
| [V410-005](v410-005-sitemap-openapi-reconciliation.yml) | feature | P1 | V410-002 | Reconcile exact OpenAPI declarations with live sitemap members |
| [V413-001](v413-001-runtime-artifact-disk-cache.yml) | feature | P2 | — | Reuse authorized immutable archives in a bounded Runtime disk cache |

### Audits (`audits`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V411-002](v411-002-engagement-baseline-preflight.yml) | feature | P1 | V411-001 | Validate and pin a typed pentest engagement at Audit start |
| [V411-003](v411-003-action-journal-request-permits.yml) | feature | P1 | V411-002 | Fence and execute scoped HTTP preflight through the durable Audit action ledger |
| [V411-006](v411-006-cumulative-budget-finalization.yml) | feature | P1 | V411-002, V411-003 | Enforce cumulative pentest budgets and finalize at the hard active-time cap |
| [V411-008](v411-008-stateful-actions-cleanup.yml) | feature | P1 | V411-003, V411-004, V411-006, V411-007 | Serialize stateful target checks and retain unresolved cleanup effects |
| [V411-010](v411-010-pentest-program-availability.yml) | feature | P1 | V411-002 | Pin optional pentest programs and nonblocking finding review |
| [V411-012](v411-012-pentest-admission-rounds.yml) | feature | P1 | V411-009, V411-010, V62-005, V62-006, V62-007 | Admit typed pentest plans and derive exact later-round replay |
| [V411-016](v411-016-pentest-proof-report.yml) | feature | P1 | V411-012, V411-014 | Assemble retained pentest proof and incomplete-coverage reports |
| [V414-001](v414-001-openapi-inventory-string-bound.yml) | defect | P2 | — | Give openapi-operations@1 the string bounds of the shared OpenAPI document rule |
| [V414-020](v414-020-preparation-phase-one-rule.yml) | chore | P3 | — | Give the pre-Round preparation phase set one definition in Go and one in the UI |

### Evaluations (`evals`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V40-003](v40-003-agent-instruction-eval-decision.yml) | feature | P1 | V40-002 | Evaluate instruction candidates and record evidence-based adoption decisions |
| [V411-019](v411-019-pentest-target-fixtures-and-proof-scorers.yml) | chore | P1 | V411-002 | Build paired pentest fixtures and independent proof scorers |
| [V411-021](v411-021-pentest-frozen-live-quality-evaluation.yml) | chore | P2 | V411-020 | Evaluate pentest quality against a frozen repeated-sample baseline |
| [V414-022](v414-022-eval-purge-guard-lexer.yml) | chore | P3 | — | Read trigger functions with the explaintest lexer in the Eval purge coverage guard and check rewrite rules |

### Podman sandbox (`sandbox`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V411-013](v411-013-pentest-sandbox-integration.yml) | feature | P1 | V411-002, V411-006 | Integrate bounded sandbox investigations and tool installation |

### Web UI (`ui`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V403-003](v403-003-openapi-preview-at-1mib.yml) | defect | P3 | — | Keep a 1 MiB OpenAPI preview from blocking the page |
| [V411-018](v411-018-pentest-ui-journey.yml) | feature | P1 | V411-015, V411-017 | Build the autonomous HTTP pentest setup and evidence journey |

### Operations, authentication, CLI and deployment (`operations`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V414-025](v414-025-allocation-history-page-reads.yml) | defect | P2 | — | Bound allocation-history pages by page size at any cursor depth and Run filter |

### Public API and wire contracts (`api-contracts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-020](v372-020-error-code-vocabularies.yml) | chore | P3 | — | Hold every vocabulary that carries error code names to the registry |
| [V372-025](v372-025-typed-openapi-enums.yml) | chore | P3 | — | Give every enum of the public OpenAPI document a type |
| [V372-026](v372-026-request-digests-in-services.yml) | chore | P3 | — | Move the Project, Audit and review request digests out of the public handlers |
| [V411-017](v411-017-pentest-api-cli.yml) | feature | P1 | V411-010, V411-011, V411-012, V411-016, V411-006 | Expose pentest drafts progress surfaces and proof through API and CLI |
| [V414-009](v414-009-machine-value-byte-bounds.yml) | chore | P3 | — | Declare the byte bounds of machine values in the public API and hold OpenAPI, Go and the UI to them |
| [V414-032](v414-032-nul-escape-outside-evals.yml) | defect | P2 | — | Reject the NUL escape before Audit and finding writes reach a jsonb cast |

### PostgreSQL persistence and schema (`persistence`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-008](v372-008-upward-table-reads.yml) | chore | P3 | — | Remove the table reads that point up the layer stack or name who owns each one |
| [V372-009](v372-009-ownership-check-blind-spots.yml) | chore | P3 | — | Close the blind spots of the table ownership check |
| [V372-010](v372-010-erd-usage-from-ownership-scan.yml) | chore | P3 | — | Take the ERD's code-usage lists from the ownership scan and document the new-table workflow |
| [V414-012](v414-012-classify-closed-connection.yml) | defect | P2 | — | Classify a closed PostgreSQL connection, including a lost COMMIT reply, as unavailable |
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
| [V411-020](v411-020-pentest-http-release-gate.yml) | chore | P1 | V411-001, V411-002, V411-003, V411-004, V411-005, V411-006, V411-007, V411-008, V411-009, V411-010, V411-011, V411-012, V411-013, V411-014, V411-015, V411-016, V411-017, V411-018, V411-019, V410-007 | Gate the first HTTP API pentest delivery through real processes |
| [V414-027](v414-027-close-review-followups.yml) | chore | P3 | V414-001, V414-002, V414-004, V414-007, V414-009, V414-012, V414-014, V414-019, V414-020, V414-021, V414-022, V414-024, V414-025, V414-031, V414-032, V414-033 | Close the V407/V409 review follow-ups |

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
| [V62-005](v62-005-audit-round-result-snapshots.yml) | audits | 2026-09-20 | Explicit user approval after the first scan slice demonstrates useful results; define the snapshot contract in this task before execution. |
| [V62-006](v62-006-audit-retained-dependencies.yml) | audits | 2026-09-20 | Explicit user approval of an actual later-round consumer and its exact dependency contract. |
| [V62-007](v62-007-audit-check-routing.yml) | audits | 2026-09-20 | Explicit user approval of iterative checks after dependency and snapshot contracts are reviewed. |
| [V62-008](v62-008-project-analysis-preparation.yml) | audits | 2026-09-20 | Explicit user approval of standalone analysis-report reuse and the additional two-Workflow chain. |
| [V64-001](v64-001-audit-priority-contracts.yml) | audits | 2026-10-21 | Explicit user decision after cookies and V62 are delivered and representative audit measurements show that model-based prioritization improves useful-result yield or time/cost over fixed ordering. |
| [V64-002](v64-002-audit-priority-store.yml) | audits | 2026-10-21 | Explicit user decision after cookies and V62 are delivered and representative audit measurements show that model-based prioritization improves useful-result yield or time/cost over fixed ordering. |
| [V64-003](v64-003-planner-only-priority-execution.yml) | audits | 2026-10-21 | Explicit user decision after cookies and V62 are delivered and representative audit measurements show that model-based prioritization improves useful-result yield or time/cost over fixed ordering. |
| [V64-004](v64-004-audit-priority-context.yml) | audits | 2026-10-21 | Explicit user decision after cookies and V62 are delivered and representative audit measurements show that model-based prioritization improves useful-result yield or time/cost over fixed ordering. |
| [V64-005](v64-005-independent-priority-verdicts.yml) | audits | 2026-10-21 | Explicit user decision after cookies and V62 are delivered and representative audit measurements show that model-based prioritization improves useful-result yield or time/cost over fixed ordering. |
| [V64-006](v64-006-audit-priority-selection.yml) | audits | 2026-10-21 | Explicit user decision after cookies and V62 are delivered and representative audit measurements show that model-based prioritization improves useful-result yield or time/cost over fixed ordering. |
| [V64-007](v64-007-audit-priority-controller.yml) | audits | 2026-10-21 | Explicit user decision after cookies and V62 are delivered and representative audit measurements show that model-based prioritization improves useful-result yield or time/cost over fixed ordering. |
| [V64-008](v64-008-audit-priority-api-coverage.yml) | audits | 2026-10-21 | Explicit user decision after cookies and V62 are delivered and representative audit measurements show that model-based prioritization improves useful-result yield or time/cost over fixed ordering. |
| [V64-009](v64-009-audit-priority-ui.yml) | audits | 2026-10-21 | Explicit user decision after cookies and V62 are delivered and representative audit measurements show that model-based prioritization improves useful-result yield or time/cost over fixed ordering. |
| [V64-010](v64-010-audit-priority-profile.yml) | audits | 2026-10-21 | Explicit user decision after cookies and V62 are delivered and representative audit measurements show that model-based prioritization improves useful-result yield or time/cost over fixed ordering. |
| [V64-011](v64-011-audit-priority-release.yml) | audits | 2026-10-21 | Explicit user decision after cookies and V62 are delivered and representative audit measurements show that model-based prioritization improves useful-result yield or time/cost over fixed ordering. |
| [V365-016](v365-016-fold-planner-execution-reports.yml) | persistence | 2026-10-04 | Explicit user approval of making Planner reports a mutable slot on stage_metrics. |

## Epics

All tasks of each epic by kind, and how many later tasks name one of its
tasks in `introduced_by`.

| Epic | Open | Feature | Defect | Chore | Later fixes caused |
| --- | ---: | ---: | ---: | ---: | ---: |
| Run execution core (`execution-core`) | 4 | 41 | 54 | 18 | 80 |
| Planners and the Worker model boundary (`planner-worker`) | 4 | 27 | 42 | 17 | 61 |
| Runtime labels, configuration and credentials (`runtime-config`) | 11 | 26 | 13 | 4 | 41 |
| Shared Memory tools (`memory`) | 0 | 6 | 9 | 2 | 12 |
| Agent Skills (`agent-skills`) | 0 | 11 | 2 | 1 | 2 |
| Run workspaces and project filesystem (`workspaces`) | 0 | 16 | 13 | 5 | 16 |
| Code analysis and taint annotations (`code-analysis`) | 0 | 14 | 14 | 2 | 19 |
| HTTP and Caido tools (`http-tools`) | 10 | 19 | 15 | 5 | 16 |
| Scanner tool Workers and scan planning (`scan-tools`) | 4 | 16 | 6 | 7 | 12 |
| Findings tools and collections (`findings`) | 9 | 16 | 7 | 8 | 12 |
| Workflow catalog and document workflows (`workflows`) | 7 | 18 | 14 | 12 | 18 |
| Projects, artifacts and lifecycle controls (`projects-artifacts`) | 5 | 35 | 33 | 6 | 38 |
| Audits (`audits`) | 25 | 60 | 78 | 23 | 101 |
| Evaluations (`evals`) | 6 | 21 | 37 | 11 | 45 |
| Podman sandbox (`sandbox`) | 1 | 10 | 2 | 1 | 3 |
| Web UI (`ui`) | 2 | 25 | 61 | 13 | 55 |
| Operations, authentication, CLI and deployment (`operations`) | 1 | 16 | 37 | 6 | 13 |
| Public API and wire contracts (`api-contracts`) | 7 | 2 | 37 | 22 | 11 |
| PostgreSQL persistence and schema (`persistence`) | 7 | 0 | 24 | 42 | 13 |
| Quality, CI and process (`quality`) | 16 | 1 | 26 | 92 | 23 |
