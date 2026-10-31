# Backlog

Generated from the task files by `make tasks-backlog`; do not edit by hand.
`make lint` fails when this file is stale. New tasks start from
[templates/](templates/); the task policy is in [index.yml](index.yml) and
the epics are in [epics.yml](epics.yml).

80 open tasks: 10 in progress, 59 pending, 11 deferred. 1286 finished: 1271 completed, 15 archived.

## In progress

| Task | Epic | Kind | Priority | Title |
| --- | --- | --- | --- | --- |
| [V40-002](v40-002-agent-instruction-paired-runner.yml) | evals | feature | P1 | Prepare instruction experiment bindings and a frozen plan after format readiness |
| [V261-001](v261-001-stabilize-release-gate-process-tests.yml) | quality | chore | P1 | Restore the deterministic release process gate |
| [V341-001](v341-001-investigate-ci-tolerance-bumps.yml) | quality | chore | P2 | Investigate tolerances raised to unblock CI |
| [V401-007](v401-007-runtime-suite-on-macos.yml) | quality | chore | P3 | Make the Runtime suite pass on macOS: declare the Linux-only Podman tests, make the other tests portable and file the macOS defects |
| [V407-039](v407-039-close-server-runtime-review.yml) | quality | chore | P3 | Close the validated Server and Runtime review with explicit dispositions and verification |
| [V417-001](v417-001-terminal-summarizer-include-reasoning.yml) | planner-worker | feature | P1 | Make terminal summarizer reasoning input explicitly configurable |
| [V417-003](v417-003-isolated-worker-context-compaction.yml) | planner-worker | feature | P1 | Continue an isolated Worker through workflow-selected context compaction |
| [V420-002](v420-002-planner-compaction-vertical-slice.yml) | planner-worker | feature | P1 | Implement opt-in bounded compaction for Streamline and Router Planners |
| [V420-006](v420-006-shipped-planner-compaction.yml) | planner-worker | feature | P1 | Enable shared Planner compaction and reuse native YAML in shipped Workflows |
| [V425-034](v425-034-allocation-prepare-rollback.yml) | execution-core | chore | P3 | Consolidate failed Allocation preparation rollback |

## Pending

### Run execution core (`execution-core`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V425-016](v425-016-run-event-history-bound.yml) | chore | P3 | — | Bound or detect the per-Run event history and delete it in bounded batches |

### Planners and the Worker model boundary (`planner-worker`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V417-004](v417-004-shared-session-context-compaction.yml) | feature | P1 | V417-003 | Preserve compacted context across shared-session invocations |
| [V417-006](v417-006-compactor-model-policy-selection.yml) | feature | P1 | V417-003 | Resolve reference-only compactor model policy overrides |
| [V417-007](v417-007-context-compaction-state-and-ui.yml) | feature | P1 | V417-003 | Report compaction progress and costs without exposing context content |
| [V420-003](v420-003-planner-compaction-fault-scale-conformance.yml) | chore | P1 | V420-002 | Verify Planner compaction persistence cancellation and scale boundaries |
| [V420-004](v420-004-planner-compaction-diagnostics.yml) | feature | P1 | V420-002 | Expose bounded truthful Planner compaction diagnostics |

### Run workspaces and project filesystem (`workspaces`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V425-001](v425-001-apfs-opaque-leaf-normalization.yml) | defect | P3 | — | Keep an NFD opaque leaf unreadable through its NFC name on a normalization-insensitive volume |

### Code analysis and taint annotations (`code-analysis`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V425-002](v425-002-trailmark-mirror-path-max.yml) | defect | P3 | — | Build the Trailmark graph for workspace paths longer than the host PATH_MAX |

### HTTP and Caido tools (`http-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V410-008](v410-008-sitemap-caido-history.yml) | feature | P2 | V410-007 | Import selected Caido history incrementally with durable checkpoints |
| [V410-009](v410-009-sitemap-browser-collection.yml) | feature | P2 | V410-007, V412-001 | Collect managed Playwright network observations into the sitemap |
| [V412-001](v412-001-managed-pentest-browser.yml) | feature | P2 | — | Add allocation-owned browser investigation with explicit network authority |
| [V412-004](v412-004-trusted-pentest-callback-proof.yml) | feature | P2 | V411-015, V411-016 | Use existing Interactsh tools as bounded HTTP investigation evidence |

### Scanner tool Workers and scan planning (`scan-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V55-009](v55-009-scan-candidate-ranking.yml) | feature | P2 | — | Optionally rank existing scan candidates with an LLM |
| [V412-005](v412-005-controlled-scanner-pentest-integration.yml) | feature | P2 | — | Run ffuf through the existing allocation-local engagement authority |

### Findings tools and collections (`findings`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V411-007](v411-007-trusted-http-capture.yml) | defect | P1 | — | Retain actual engagement HTTP exchanges through ordinary Audit evidence |
| [V411-011](v411-011-project-findings-reader.yml) | feature | P1 | V411-015 | Read live Project findings through allocation-authorized existing mechanisms |

### Workflow catalog and document workflows (`workflows`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V410-006](v410-006-sitemap-workflow-composition.yml) | feature | P1 | — | Ship ordinary sitemap discovery and reconciliation Workflows |
| [V411-014](v411-014-pentest-discovery-surface.yml) | feature | P1 | V411-015, V410-006 | Discover URL-only targets using HTTP tools and the shared sitemap |
| [V411-015](v411-015-http-pentest-workflow-presets.yml) | feature | P1 | V411-007, V411-019 | Ship an ordinary HTTP pentest Workflow with an early process demonstration |
| [V412-003](v412-003-browser-pentest-profiles-and-gate.yml) | feature | P2 | V412-001, V410-009, V411-020 | Compose optional browser investigation with truthful availability and results |

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
| [V64-018](v64-018-audit-research-pentest-integration.yml) | feature | P2 | V64-011, V411-015, V411-016, V411-020 | Connect optional hypothesis research to ordinary HTTP checks and reports |
| [V411-010](v411-010-pentest-program-availability.yml) | feature | P1 | V411-015 | Allow nonblocking analyst review for automated HTTP Audits |
| [V411-016](v411-016-pentest-proof-report.yml) | feature | P1 | V411-015, V411-014, V411-010 | Show HTTP engagement context and gaps in ordinary retained Audit reports |
| [V417-005](v417-005-audit-worker-context-compaction.yml) | feature | P1 | V417-003, V417-004 | Admit compaction without bypassing trusted Audit completion |
| [V425-017](v425-017-machine-report-ceiling-test.yml) | chore | P3 | — | Drive the cumulative 64 MiB machine report check to its rejection in a test |

### Evaluations (`evals`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V40-003](v40-003-agent-instruction-eval-decision.yml) | feature | P1 | V40-002 | Evaluate instruction candidates and record evidence-based adoption decisions |
| [V411-019](v411-019-pentest-target-fixtures-and-proof-scorers.yml) | chore | P1 | — | Build small paired HTTP targets and independent result oracles |
| [V411-021](v411-021-pentest-frozen-live-quality-evaluation.yml) | chore | P2 | V411-020 | Evaluate HTTP investigation quality with frozen repeated comparisons |
| [V417-008](v417-008-context-compaction-eval-harness.yml) | chore | P2 | V417-001, V417-003 | Build frozen context-compaction evaluation fixtures and an opt-in harness |
| [V417-010](v417-010-context-compaction-live-evaluation.yml) | chore | P2 | V417-008, V417-009 | Measure compaction and reasoning choices with repeated frozen live experiments |

### Podman sandbox (`sandbox`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V411-013](v411-013-pentest-sandbox-integration.yml) | feature | P1 | V411-015 | Compose the existing networked sandbox into HTTP investigations |

### Web UI (`ui`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V411-018](v411-018-pentest-ui-journey.yml) | feature | P1 | V411-015, V411-017, V411-010 | Build the ordinary HTTP Audit setup and evidence UI journey |

### Public API and wire contracts (`api-contracts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V411-017](v411-017-pentest-api-cli.yml) | feature | P1 | V411-015, V411-016, V411-010 | Expose HTTP engagement setup and results through existing API and CLI |

### PostgreSQL persistence and schema (`persistence`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V425-005](v425-005-stored-size-bounds-agree.yml) | defect | P2 | — | Make Go validation, stored-size CHECKs and the limits registry agree on every remaining column |
| [V425-015](v425-015-optional-filter-generic-plans.yml) | chore | P3 | — | Measure the generic plans of optional-filter predicates and bound the ones that scan |

### Quality, CI and process (`quality`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V347-001](v347-001-protect-main-branch.yml) | chore | P1 | — | Protect main with a required check |
| [V410-007](v410-007-sitemap-release-gate.yml) | chore | P1 | V410-006 | Verify the first sitemap slice across scale and lifecycle boundaries |
| [V411-020](v411-020-pentest-http-release-gate.yml) | chore | P1 | V411-007, V411-010, V411-011, V411-013, V411-014, V411-015, V411-016, V411-017, V411-018, V411-019, V410-007 | Gate the composed ordinary HTTP pentest through real processes |
| [V417-009](v417-009-context-compaction-release-gate.yml) | chore | P1 | V417-003, V417-004, V417-005, V417-006, V417-007 | Verify context compaction across real processes and close the deterministic milestone |
| [V420-005](v420-005-planner-compaction-release-gate.yml) | chore | P1 | V420-006, V420-003, V420-004 | Accept Planner compaction through real-process and integrated release gates |
| [V425-013](v425-013-go-test-wall-clock-wave-3.yml) | chore | P3 | — | Replace the remaining wall-clock waits in Go PostgreSQL tests and restore the listener budget coverage |
| [V425-014](v425-014-runtime-tests-fail-on-macos.yml) | chore | P3 | — | Make the Runtime process-liveness assertions able to fail on macOS |

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
| Run execution core (`execution-core`) | 2 | 47 | 56 | 22 | 84 |
| Planners and the Worker model boundary (`planner-worker`) | 9 | 35 | 47 | 29 | 68 |
| Runtime labels, configuration and credentials (`runtime-config`) | 9 | 26 | 13 | 5 | 44 |
| Shared Memory tools (`memory`) | 0 | 6 | 9 | 2 | 12 |
| Agent Skills (`agent-skills`) | 0 | 11 | 4 | 2 | 4 |
| Run workspaces and project filesystem (`workspaces`) | 1 | 16 | 15 | 5 | 16 |
| Code analysis and taint annotations (`code-analysis`) | 1 | 14 | 15 | 2 | 20 |
| HTTP and Caido tools (`http-tools`) | 4 | 22 | 27 | 9 | 25 |
| Scanner tool Workers and scan planning (`scan-tools`) | 2 | 17 | 10 | 9 | 19 |
| Findings tools and collections (`findings`) | 2 | 16 | 9 | 9 | 14 |
| Workflow catalog and document workflows (`workflows`) | 4 | 18 | 15 | 12 | 19 |
| Projects, artifacts and lifecycle controls (`projects-artifacts`) | 0 | 36 | 37 | 11 | 60 |
| Audits (`audits`) | 23 | 67 | 82 | 35 | 121 |
| Evaluations (`evals`) | 6 | 21 | 39 | 15 | 49 |
| Podman sandbox (`sandbox`) | 1 | 10 | 2 | 1 | 3 |
| Web UI (`ui`) | 1 | 25 | 66 | 16 | 59 |
| Operations, authentication, CLI and deployment (`operations`) | 0 | 18 | 45 | 8 | 20 |
| Public API and wire contracts (`api-contracts`) | 1 | 2 | 38 | 25 | 16 |
| PostgreSQL persistence and schema (`persistence`) | 3 | 0 | 27 | 45 | 15 |
| Quality, CI and process (`quality`) | 11 | 1 | 28 | 112 | 26 |
