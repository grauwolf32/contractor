# Backlog

Generated from the task files by `make tasks-backlog`; do not edit by hand.
`make lint` fails when this file is stale. New tasks start from
[templates/](templates/); the task policy is in [index.yml](index.yml) and
the epics are in [epics.yml](epics.yml).

68 open tasks: 11 in progress, 44 pending, 13 deferred. 1669 finished: 1651 completed, 18 archived.

## In progress

| Task | Epic | Kind | Priority | Title |
| --- | --- | --- | --- | --- |
| [V40-002](v40-002-agent-instruction-paired-runner.yml) | evals | feature | P1 | Prepare instruction experiment bindings and a frozen plan after format readiness |
| [V261-001](v261-001-stabilize-release-gate-process-tests.yml) | quality | chore | P1 | Restore the deterministic release process gate |
| [V341-001](v341-001-investigate-ci-tolerance-bumps.yml) | quality | chore | P2 | Investigate tolerances raised to unblock CI |
| [V401-007](v401-007-runtime-suite-on-macos.yml) | quality | chore | P3 | Make the Runtime suite pass on macOS: declare the Linux-only Podman tests, make the other tests portable and file the macOS defects |
| [V407-039](v407-039-close-server-runtime-review.yml) | quality | chore | P3 | Close the validated Server and Runtime review with explicit dispositions and verification |
| [V411-015](v411-015-http-pentest-workflow-presets.yml) | workflows | feature | P1 | Demonstrate an ordinary HTTP Workflow on the lean Audit path |
| [V417-004](v417-004-shared-session-context-compaction.yml) | planner-worker | feature | P1 | Preserve compacted context across shared-session invocations |
| [V417-010](v417-010-context-compaction-live-evaluation.yml) | evals | chore | P2 | Measure compaction and reasoning choices with repeated frozen live experiments |
| [V420-006](v420-006-shipped-planner-compaction.yml) | planner-worker | feature | P1 | Accept shipped Planner compaction and native YAML reuse |
| [V425-034](v425-034-allocation-prepare-rollback.yml) | execution-core | chore | P3 | Consolidate failed Allocation preparation rollback |
| [V478-003](v478-003-audit-submit-result.yml) | audits | feature | P1 | Submit a thin attributed result over accumulated observations |

## Pending

### Run execution core (`execution-core`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V425-016](v425-016-run-event-history-bound.yml) | chore | P3 | — | Bound or detect the per-Run event history and delete it in bounded batches |
| [V469-002](v469-002-opencode-audit-runtime.yml) | feature | P1 | V469-001 | Run OpenCode in one retained Audit environment through existing A2A |

### Planners and the Worker model boundary (`planner-worker`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V420-003](v420-003-planner-compaction-fault-scale-conformance.yml) | chore | P1 | — | Verify Planner compaction persistence cancellation and scale boundaries |
| [V420-004](v420-004-planner-compaction-diagnostics.yml) | feature | P1 | — | Expose bounded truthful Planner compaction diagnostics |
| [V469-001](v469-001-opencode-audit-prototype.yml) | chore | P1 | — | Prove two ordinary Runs can reuse one OpenCode Audit container |

### HTTP and Caido tools (`http-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V410-008](v410-008-sitemap-caido-history.yml) | feature | P2 | V410-007 | Import selected Caido history incrementally with durable checkpoints |
| [V410-009](v410-009-sitemap-browser-collection.yml) | feature | P2 | V410-007, V412-001 | Collect managed Playwright network observations into the sitemap |
| [V410-037](v410-037-allocation-openvpn-bridge-egress.yml) | feature | P2 | — | Route bridge egress through an allocation-owned OpenVPN tunnel |
| [V412-001](v412-001-managed-pentest-browser.yml) | feature | P2 | — | Add allocation-owned browser investigation with explicit network authority |
| [V412-004](v412-004-trusted-pentest-callback-proof.yml) | feature | P2 | V411-015, V411-016 | Use existing Interactsh tools as bounded HTTP investigation evidence |

### Scanner tool Workers and scan planning (`scan-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V55-009](v55-009-scan-candidate-ranking.yml) | feature | P2 | — | Optionally rank existing scan candidates with an LLM |
| [V410-026](v410-026-sitemap-source-comparison.yml) | chore | P2 | — | Compare transport observations and scanner report projections for sitemap |

### Workflow catalog and document workflows (`workflows`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V410-006](v410-006-sitemap-workflow-composition.yml) | feature | P1 | — | Ship standalone sitemap discovery and reconciliation Workflows |
| [V410-025](v410-025-sitemap-audit-composition.yml) | feature | P1 | V410-007 | Compose exact sitemap artifacts through ordinary Audit tasks |
| [V411-014](v411-014-pentest-discovery-surface.yml) | feature | P1 | V411-015, V410-025 | Discover URL-only targets using HTTP tools and the shared sitemap |
| [V412-003](v412-003-browser-pentest-profiles-and-gate.yml) | feature | P2 | V412-001, V410-009, V411-020 | Compose optional browser investigation with truthful availability and results |
| [V425-043](v425-043-resolve-historical-run-model-policy-snapshots.yml) | defect | P2 | — | Resolve historical Run snapshots after Workflow-owned model selection |

### Audits (`audits`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V64-005](v64-005-independent-priority-verdicts.yml) | feature | P2 | — | Optionally rank the common Audit ready-task shortlist |
| [V64-011](v64-011-audit-priority-release.yml) | feature | P1 | V411-015, V425-017 | Gate the basic Audit research scenario and complexity reduction |
| [V64-015](v64-015-audit-research-frontier.yml) | feature | P2 | V64-011 | Measure whether a larger Audit frontier needs incremental indexing |
| [V64-016](v64-016-audit-research-ranking-evaluation.yml) | chore | P2 | V64-005, V64-011 | Evaluate optional Audit ranking against the deterministic baseline |
| [V64-017](v64-017-audit-attack-tree-projection.yml) | feature | P2 | V64-011 | Project optional attack trees over retained research conclusions |
| [V64-018](v64-018-audit-research-pentest-integration.yml) | feature | P2 | V64-011, V411-020 | Compose optional hypothesis generation with ordinary HTTP Audit tasks |
| [V411-016](v411-016-pentest-proof-report.yml) | feature | P1 | V478-005, V411-015, V411-014 | Show HTTP engagement context and gaps in ordinary retained Audit reports |
| [V425-017](v425-017-machine-report-ceiling-test.yml) | chore | P3 | — | Drive the cumulative 64 MiB machine report check to its rejection in a test |
| [V469-004](v469-004-opencode-audit-workflow.yml) | feature | P1 | V469-003 | Ship the minimal OpenCode Audit workflow and environment view |
| [V478-004](v478-004-audit-partial-stop-results.yml) | feature | P1 | V478-003 | Preserve partial Audit results without inventing coverage at stop |
| [V478-005](v478-005-audit-results-acceptance.yml) | feature | P1 | V478-003, V478-004 | Expose honest Audit results and gate partial-result lifecycle |

### Evaluations (`evals`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V40-003](v40-003-agent-instruction-eval-decision.yml) | feature | P1 | V40-002 | Evaluate instruction candidates and record evidence-based adoption decisions |
| [V411-021](v411-021-pentest-frozen-live-quality-evaluation.yml) | chore | P2 | V411-020 | Evaluate HTTP investigation quality with frozen repeated comparisons |

### Podman sandbox (`sandbox`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V411-013](v411-013-pentest-sandbox-integration.yml) | feature | P1 | V411-015 | Compose the existing networked sandbox into HTTP investigations |
| [V469-003](v469-003-opencode-audit-recovery.yml) | chore | P1 | V469-002 | Verify retained environment recovery and deletion across owner failures |

### Web UI (`ui`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V411-018](v411-018-pentest-ui-journey.yml) | feature | P1 | V478-005, V411-015, V411-017 | Build the ordinary HTTP Audit setup and evidence UI journey |

### Operations, authentication, CLI and deployment (`operations`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V473-003](v473-003-model-recovery-diagnostics.yml) | feature | P1 | — | Explain model recovery costs and verify zero and partial fact extraction |

### Public API and wire contracts (`api-contracts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V411-017](v411-017-pentest-api-cli.yml) | feature | P1 | V411-015, V411-016 | Expose HTTP engagement setup and results through existing API and CLI |

### PostgreSQL persistence and schema (`persistence`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V425-005](v425-005-stored-size-bounds-agree.yml) | defect | P2 | — | Make Go validation, stored-size CHECKs and the limits registry agree on every remaining column |
| [V425-015](v425-015-optional-filter-generic-plans.yml) | chore | P3 | — | Measure the generic plans of optional-filter predicates and bound the ones that scan |

### Quality, CI and process (`quality`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V347-001](v347-001-protect-main-branch.yml) | chore | P1 | — | Protect main with a required check |
| [V410-007](v410-007-sitemap-release-gate.yml) | chore | P1 | V410-006 | Verify the standalone sitemap slice across scale and lifecycle boundaries |
| [V411-020](v411-020-pentest-http-release-gate.yml) | chore | P1 | V478-005, V64-011, V411-013, V411-014, V411-016, V411-017, V411-018, V410-025 | Gate the composed ordinary HTTP pentest through real processes |
| [V420-005](v420-005-planner-compaction-release-gate.yml) | chore | P1 | V420-006, V420-003, V420-004 | Accept Planner compaction through real-process and integrated release gates |
| [V425-014](v425-014-runtime-tests-fail-on-macos.yml) | chore | P3 | — | Make the Runtime process-liveness assertions able to fail on macOS |
| [V433-005](v433-005-runtime-resource-bounds-gate.yml) | chore | P3 | — | Verify the Runtime resource-bound fixes and reconcile the review dispositions |
| [V469-005](v469-005-opencode-audit-release.yml) | chore | P1 | V469-004 | Accept the minimal OpenCode Audit environment through real release gates |

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
| [V425-001](v425-001-apfs-opaque-leaf-normalization.yml) | workspaces | 2026-11-02 | Explicit user decision to require separate macOS filename hardening; first rescope its acceptance and dedicated coverage rather than treating this Linux fixture as a required macOS test. |
| [V425-002](v425-002-trailmark-mirror-path-max.yml) | code-analysis | 2026-11-02 | Explicit user decision to require separate macOS long-path support; first rescope its acceptance and dedicated coverage rather than treating this Linux fixture as a required macOS test. |

## Epics

All tasks of each epic by kind, and how many later tasks name one of its
tasks in `introduced_by`.

| Epic | Open | Feature | Defect | Chore | Later fixes caused |
| --- | ---: | ---: | ---: | ---: | ---: |
| Run execution core (`execution-core`) | 3 | 50 | 72 | 40 | 108 |
| Planners and the Worker model boundary (`planner-worker`) | 5 | 49 | 60 | 48 | 97 |
| Runtime labels, configuration and credentials (`runtime-config`) | 9 | 26 | 16 | 5 | 48 |
| Shared Memory tools (`memory`) | 0 | 7 | 9 | 3 | 15 |
| Agent Skills (`agent-skills`) | 0 | 11 | 4 | 2 | 4 |
| Run workspaces and project filesystem (`workspaces`) | 1 | 16 | 16 | 7 | 17 |
| Code analysis and taint annotations (`code-analysis`) | 1 | 14 | 15 | 2 | 20 |
| HTTP and Caido tools (`http-tools`) | 5 | 26 | 38 | 16 | 39 |
| Scanner tool Workers and scan planning (`scan-tools`) | 2 | 21 | 13 | 13 | 30 |
| Findings tools and collections (`findings`) | 0 | 22 | 15 | 13 | 29 |
| Workflow catalog and document workflows (`workflows`) | 6 | 21 | 17 | 16 | 27 |
| Projects, artifacts and lifecycle controls (`projects-artifacts`) | 0 | 41 | 41 | 21 | 76 |
| Audits (`audits`) | 13 | 77 | 101 | 42 | 175 |
| Evaluations (`evals`) | 4 | 23 | 47 | 21 | 55 |
| Podman sandbox (`sandbox`) | 2 | 10 | 2 | 2 | 6 |
| Web UI (`ui`) | 1 | 56 | 81 | 34 | 80 |
| Operations, authentication, CLI and deployment (`operations`) | 1 | 23 | 55 | 16 | 35 |
| Public API and wire contracts (`api-contracts`) | 1 | 2 | 43 | 30 | 21 |
| PostgreSQL persistence and schema (`persistence`) | 3 | 0 | 28 | 50 | 18 |
| Quality, CI and process (`quality`) | 11 | 1 | 34 | 153 | 32 |
