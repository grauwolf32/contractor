# Backlog

Generated from the task files by `make tasks-backlog`; do not edit by hand.
`make lint` fails when this file is stale. New tasks start from
[templates/](templates/); the task policy is in [index.yml](index.yml) and
the epics are in [epics.yml](epics.yml).

98 open tasks: 6 in progress, 67 pending, 25 deferred. 1060 finished: 1052 completed, 8 archived.

## In progress

| Task | Epic | Kind | Priority | Title |
| --- | --- | --- | --- | --- |
| [V40-002](v40-002-agent-instruction-paired-runner.yml) | evals | feature | P1 | Prepare instruction experiment bindings and a frozen plan after format readiness |
| [V261-001](v261-001-stabilize-release-gate-process-tests.yml) | quality | chore | P1 | Restore the deterministic release process gate |
| [V341-001](v341-001-investigate-ci-tolerance-bumps.yml) | quality | chore | P2 | Investigate tolerances raised to unblock CI |
| [V398-001](v398-001-runtime-completion-capability-derivation.yml) | audits | chore | P3 | Derive the Runtime's completion-contract capability from one rule |
| [V406-007](v406-007-web-scan-release-gate.yml) | scan-tools | chore | P2 | Gate the complete web scan and rule sweep journeys |
| [V406-016](v406-016-release-report-initialization.yml) | quality | defect | P2 | Initialize shared release evidence before either language writes it |

## Pending

### Run execution core (`execution-core`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-014](v371-014-scheduler-stage-history-reads.yml) | defect | P3 | — | Read only the Stage history fields escalation counting and cancellation cleanup use |
| [V372-007](v372-007-terminal-stage-state-set.yml) | chore | P3 | — | Define the terminal StageExecution states once and hold SQL and the UI to them |
| [V372-028](v372-028-runstore-strict-snapshot-reads.yml) | chore | P3 | — | Decode persisted Run, Stage and allocation snapshots strictly in runstore |
| [V410-010](v410-010-credential-recovery-previous-owner.yml) | defect | P2 | — | Keep a previous lease owner's credential delete intent from committing behind the new owner's startup recovery |
| [V410-014](v410-014-standby-shutdown-race.yml) | defect | P3 | — | Exit a standby Server cleanly when shutdown races its readiness listener |

### Planners and the Worker model boundary (`planner-worker`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-004](v371-004-worker-completion-exact-integers.yml) | defect | P3 | — | Bound WorkerCompletion integers to the range A2A carries exactly |
| [V371-005](v371-005-worker-state-read-cost-guard.yml) | chore | P3 | — | Guard the Worker State read cost with a deterministic bound and re-measure it |
| [V371-006](v371-006-workspace-observation-descriptor.yml) | chore | P3 | — | Take the Worker tools that observe the workspace from the shared Toolset descriptor |
| [V410-006](v410-006-planner-session-write-outcomes.yml) | defect | P2 | — | Report one Planner outcome when the session write after a result fails |
| [V410-007](v410-007-planner-session-state-limit.yml) | chore | P3 | — | Register the 2 MiB Planner session state bound and the other unregistered stored-size CHECKs |
| [V410-013](v410-013-adk-callable-cache-release.yml) | defect | P2 | — | Stop ADK's module-level callable caches from keeping a released allocation's tools alive |

### Runtime labels, configuration and credentials (`runtime-config`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-006](v372-006-runtime-label-set-rule.yml) | chore | P3 | — | Give the Runtime label set rule one Go definition and hold the UI and the Runtime to it |

### Run workspaces and project filesystem (`workspaces`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V410-005](v410-005-source-archive-matching-off-loop.yml) | defect | P2 | — | Match and sort source-archive list and search paths off the event loop |

### HTTP and Caido tools (`http-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V386-004](v386-004-caido-catalog-cardinality.yml) | defect | P2 | — | Keep Caido catalog selection usable above one hundred entries |

### Scanner tool Workers and scan planning (`scan-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V55-009](v55-009-scan-candidate-ranking.yml) | feature | P2 | — | Optionally rank existing scan candidates with an LLM |
| [V410-004](v410-004-scan-audit-result-failure-classes.yml) | defect | P2 | — | Classify Audit scan result building failures by cause instead of as invalid input |

### Findings tools and collections (`findings`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-010](v371-010-discarded-receipt-projection.yml) | defect | P2 | — | Drop the report projection of a proposal receipt discarded by Run deletion |
| [V371-011](v371-011-proposal-quota-limits.yml) | chore | P3 | — | Register the per-Run proposal and per-proposal evidence quotas in the specification |
| [V372-023](v372-023-findings-tool-error-codes.yml) | chore | P3 | — | Name the findings error codes the open question lists and register the findings tool codes |
| [V397-006](v397-006-finding-analysis-resolver.yml) | feature | P2 | — | Resolve optional finding analysis against exact retained evidence |
| [V397-007](v397-007-finding-export-snapshot.yml) | feature | P2 | V397-006 | Freeze explicit finding selections and independent Audit results |
| [V397-008](v397-008-generic-sarif-renderer.yml) | feature | P2 | V397-007 | Render deterministic validated generic SARIF from frozen snapshots |
| [V397-009](v397-009-sarif-export-api-ui.yml) | feature | P2 | V397-008 | Publish and download generic SARIF through API CLI and UI |

### Workflow catalog and document workflows (`workflows`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-013](v372-013-planner-facts-one-source.yml) | chore | P3 | — | Take Planner refs and facts from the descriptor registry in Go, the composition and Studio |
| [V372-014](v372-014-execution-config-patch-strict-decode.yml) | chore | P3 | — | Decode executionConfig patch members with strictjson |
| [V372-015](v372-015-config-split-leftovers.yml) | chore | P3 | — | Stop locals from shadowing the resolved package and drop config's copy of clone.Map |

### Audits (`audits`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-009](v371-009-audit-list-preparation-batch.yml) | defect | P3 | — | Read the preparation state of a whole Audit page in bounded statements |
| [V410-001](v410-001-openapi-inventory-string-bound.yml) | defect | P2 | — | Give openapi-operations@1 the string bounds of the shared OpenAPI document rule |
| [V410-018](v410-018-audit-detail-snapshot-read.yml) | defect | P2 | — | Read the Audit detail projection in one snapshot instead of answering an undeclared 409 |
| [V410-019](v410-019-held-inbox-page-bound.yml) | defect | P2 | — | Bound the held Audit inbox page by the page, not by the Audit's holds |
| [V410-020](v410-020-preparation-phase-one-rule.yml) | chore | P3 | — | Give the pre-Round preparation phase set one definition in Go and one in the UI |

### Evaluations (`evals`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V40-003](v40-003-agent-instruction-eval-decision.yml) | feature | P1 | V40-002 | Evaluate instruction candidates and record evidence-based adoption decisions |
| [V410-002](v410-002-eval-token-budget-overflow.yml) | defect | P2 | — | Keep the Eval token budget and stage token summaries defined when token counters exceed int64 |
| [V410-017](v410-017-eval-json-nul-escape.yml) | defect | P1 | — | Reject the NUL escape in Eval JSON so stored documents never break jsonb reads |
| [V410-022](v410-022-eval-purge-guard-lexer.yml) | chore | P3 | — | Read trigger functions with the explaintest lexer in the Eval purge coverage guard and check rewrite rules |
| [V410-026](v410-026-eval-summary-generic-plan.yml) | defect | P2 | V410-017 | Bound Eval experiment summary pages by page size in generic plans |

### Web UI (`ui`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V403-003](v403-003-openapi-preview-at-1mib.yml) | defect | P3 | — | Keep a 1 MiB OpenAPI preview from blocking the page |

### Operations, authentication, CLI and deployment (`operations`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V410-011](v410-011-secret-holders-all-verbs.yml) | defect | P1 | — | Redact every Go secret holder for every fmt verb, slog and unexported-field nesting |
| [V410-016](v410-016-otlp-transport-family.yml) | defect | P2 | — | Retry OTLP batches after TLS and OS transport failures |
| [V410-023](v410-023-go-telemetry-secret-matching.yml) | defect | P2 | — | Match secrets at the Go telemetry boundary by the shared 16-byte rule, longest first |
| [V410-025](v410-025-allocation-history-page-reads.yml) | defect | P2 | — | Bound allocation-history pages by page size at any cursor depth and Run filter |

### Public API and wire contracts (`api-contracts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-016](v371-016-idempotency-key-grammar.yml) | defect | P3 | — | Give the Idempotency-Key grammar one Go definition |
| [V372-020](v372-020-error-code-vocabularies.yml) | chore | P3 | — | Hold every vocabulary that carries error code names to the registry |
| [V372-025](v372-025-typed-openapi-enums.yml) | chore | P3 | — | Give every enum of the public OpenAPI document a type |
| [V372-026](v372-026-request-digests-in-services.yml) | chore | P3 | — | Move the Project, Audit and review request digests out of the public handlers |
| [V410-003](v410-003-openapi-yaml-resolution-rule.yml) | defect | P2 | — | Read and write OpenAPI YAML under one scalar and key rule in openapi@1, scan preparation and the Audit inventory |
| [V410-009](v410-009-machine-value-byte-bounds.yml) | chore | P3 | — | Declare the byte bounds of machine values in the public API and hold OpenAPI, Go and the UI to them |
| [V410-015](v410-015-credential-token-field-grammar.yml) | defect | P2 | — | Accept only sendable HTTP field values as Project target and infrastructure credential tokens |

### PostgreSQL persistence and schema (`persistence`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-008](v372-008-upward-table-reads.yml) | chore | P3 | — | Remove the table reads that point up the layer stack or name who owns each one |
| [V372-009](v372-009-ownership-check-blind-spots.yml) | chore | P3 | — | Close the blind spots of the table ownership check |
| [V372-010](v372-010-erd-usage-from-ownership-scan.yml) | chore | P3 | — | Take the ERD's code-usage lists from the ownership scan and document the new-table workflow |
| [V409-013](v409-013-verify-sql-review-on-database.yml) | chore | P2 | — | Verify retained SQL-review changes and record evidence for every accepted or declined claim |
| [V410-008](v410-008-stored-size-check-classification.yml) | defect | P2 | — | Classify stored-size CHECK rejections as invalid input at every store writer |
| [V410-012](v410-012-classify-closed-connection.yml) | defect | P2 | — | Classify a closed PostgreSQL connection, including a lost COMMIT reply, as unavailable |
| [V410-021](v410-021-migration-file-deadline.yml) | defect | P3 | — | Decide the client deadline of a migration file: per statement or a migration budget of its own |
| [V410-024](v410-024-plan-tests-exact-reads.yml) | chore | P3 | — | Measure the V407-025 drain and held-credential row budgets with explaintest Analysis.Reads |

### Quality, CI and process (`quality`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V347-001](v347-001-protect-main-branch.yml) | chore | P1 | — | Protect main with a required check |
| [V371-015](v371-015-memory-claim-paused-skip.yml) | chore | P3 | — | Model the paused-owner claim skip in the Scheduler's memory store doubles |
| [V371-021](v371-021-postgres-test-wall-clock.yml) | chore | P3 | — | Replace wall-clock budgets in the Scheduler and runstore PostgreSQL tests |
| [V371-022](v371-022-capability-probe-test-budget.yml) | chore | P3 | — | Remove the 10 ms probe budget from the capability discovery tests |
| [V372-016](v372-016-wall-clock-test-budgets-wave-2.yml) | chore | P3 | — | Replace wall-clock budgets in the finding intake, Audit, Eval and persistence tests and the result publication test |
| [V372-024](v372-024-audit-item-attempts-fake.yml) | chore | P3 | — | Make the Audit item attempt fake answer like the store and keep the item list schema-valid without it |
| [V372-029](v372-029-shared-fixture-gate-scope.yml) | chore | P3 | — | Extend the shared fixture gate to every cross-language fixture and keep slow store suites out of it |
| [V372-030](v372-030-shrink-test-only-export-baseline.yml) | chore | P3 | — | Resolve the exported identifiers that only tests use and empty the test-only export baseline |
| [V401-007](v401-007-runtime-suite-on-macos.yml) | chore | P3 | — | Find out why about a hundred Runtime tests fail on a macOS host and make each one pass or declare its platform |
| [V407-039](v407-039-close-server-runtime-review.yml) | chore | P3 | — | Close the validated Server and Runtime review with explicit dispositions and verification |
| [V410-027](v410-027-close-review-followups.yml) | chore | P3 | V410-001, V410-002, V410-003, V410-004, V410-005, V410-006, V410-007, V410-008, V410-009, V410-010, V410-011, V410-012, V410-013, V410-014, V410-015, V410-016, V410-017, V410-018, V410-019, V410-020, V410-021, V410-022, V410-023, V410-024, V410-025, V410-026 | Close the V407/V409 review follow-ups |

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
| Run execution core (`execution-core`) | 5 | 41 | 54 | 18 | 80 |
| Planners and the Worker model boundary (`planner-worker`) | 6 | 27 | 42 | 17 | 61 |
| Runtime labels, configuration and credentials (`runtime-config`) | 10 | 25 | 13 | 4 | 41 |
| Shared Memory tools (`memory`) | 0 | 6 | 9 | 2 | 12 |
| Agent Skills (`agent-skills`) | 0 | 11 | 2 | 1 | 2 |
| Run workspaces and project filesystem (`workspaces`) | 1 | 16 | 13 | 5 | 16 |
| Code analysis and taint annotations (`code-analysis`) | 0 | 14 | 14 | 2 | 19 |
| HTTP and Caido tools (`http-tools`) | 1 | 11 | 15 | 4 | 16 |
| Scanner tool Workers and scan planning (`scan-tools`) | 3 | 14 | 6 | 7 | 12 |
| Findings tools and collections (`findings`) | 7 | 14 | 6 | 8 | 11 |
| Workflow catalog and document workflows (`workflows`) | 3 | 14 | 14 | 12 | 18 |
| Projects, artifacts and lifecycle controls (`projects-artifacts`) | 0 | 30 | 33 | 6 | 38 |
| Audits (`audits`) | 21 | 53 | 76 | 23 | 98 |
| Evaluations (`evals`) | 6 | 21 | 37 | 9 | 45 |
| Podman sandbox (`sandbox`) | 0 | 9 | 2 | 1 | 3 |
| Web UI (`ui`) | 1 | 24 | 61 | 13 | 55 |
| Operations, authentication, CLI and deployment (`operations`) | 4 | 16 | 37 | 6 | 13 |
| Public API and wire contracts (`api-contracts`) | 7 | 1 | 36 | 21 | 10 |
| PostgreSQL persistence and schema (`persistence`) | 9 | 0 | 24 | 42 | 13 |
| Quality, CI and process (`quality`) | 14 | 1 | 25 | 90 | 23 |
