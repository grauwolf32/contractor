# Backlog

Generated from the task files by `make tasks-backlog`; do not edit by hand.
`make lint` fails when this file is stale. New tasks start from
[templates/](templates/); the task policy is in [index.yml](index.yml) and
the epics are in [epics.yml](epics.yml).

83 open tasks: 3 in progress, 66 pending, 14 deferred. 813 finished: 810 completed, 3 archived.

## In progress

| Task | Epic | Kind | Priority | Title |
| --- | --- | --- | --- | --- |
| [V40-002](v40-002-agent-instruction-paired-runner.yml) | evals | feature | P1 | Prepare instruction experiment bindings and a frozen plan after format readiness |
| [V261-001](v261-001-stabilize-release-gate-process-tests.yml) | quality | chore | P1 | Restore the deterministic release process gate |
| [V341-001](v341-001-investigate-ci-tolerance-bumps.yml) | quality | chore | P2 | Investigate tolerances raised to unblock CI |

## Pending

### Run execution core (`execution-core`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V368-017](v368-017-registration-permanent-failure.yml) | defect | P1 | — | Stop retrying a Runtime registration the Server rejected permanently |
| [V368-034](v368-034-gateway-recovery-followups.yml) | chore | P3 | — | Share the gateway recovery participant protocol and index the probe holder |
| [V370-001](v370-001-run-stage-transition-owner.yml) | chore | P1 | V369-001 | Give Run and Stage state transitions one owner |
| [V370-011](v370-011-scheduler-controlplane-surface.yml) | chore | P3 | — | Slim the Scheduler and Control Plane surfaces |
| [V370-012](v370-012-runstore-runtimeconfig-cleanup.yml) | chore | P3 | — | Merge Run creation paths and split the Runtime principal registry |
| [V370-019](v370-019-scheduler-polling-cost.yml) | defect | P1 | — | Cut steady-state Scheduler, A2A and gateway polling |
| [V370-020](v370-020-scheduler-claim-and-tick-reads.yml) | defect | P2 | — | Bound the Scheduler claim and tick reads |

### Planners and the Worker model boundary (`planner-worker`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V368-025](v368-025-a2a-numbers-and-duplicates.yml) | defect | P2 | V368-002 | Decode A2A Worker results without float64 and with duplicate detection |
| [V370-026](v370-026-worker-state-read-cost.yml) | defect | P3 | V368-002 | Cut the CPU cost of reading Worker State |

### Shared Memory tools (`memory`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V368-018](v368-018-memory-reservation-drift.yml) | defect | P1 | — | Use one reserved-Memory-binding rule in Go and Python |

### Scanner tool Workers and scan planning (`scan-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V55-009](v55-009-scan-candidate-ranking.yml) | feature | P2 | — | Optionally rank existing scan candidates with an LLM |

### Findings tools and collections (`findings`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V368-009](v368-009-finding-receipt-retention.yml) | defect | P1 | — | Delete finding proposal receipts with their Project |
| [V370-029](v370-029-collection-publication-batches.yml) | defect | P3 | — | Batch the finding collection publication reads |

### Workflow catalog and document workflows (`workflows`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V368-014](v368-014-snapshot-size-on-write.yml) | defect | P1 | — | Reject oversized Workflow snapshots when the Run is created |
| [V368-030](v368-030-llm-gateway-clone.yml) | defect | P3 | — | Deep-copy LLMGateway configuration in snapshot accessors |
| [V369-006](v369-006-toolset-parity-fixture.yml) | chore | P2 | — | Extend the toolset parity fixture to the whole descriptor |
| [V370-006](v370-006-config-model-split.yml) | chore | P2 | — | Separate the configuration data model from the catalog loader |
| [V370-015](v370-015-simplify-config.yml) | chore | P3 | — | Remove duplicated code in config and catalog bundles |
| [V370-030](v370-030-catalog-hot-paths.yml) | defect | P3 | — | Cache catalog fingerprints and decoded Workflow snapshots |

### Projects, artifacts and lifecycle controls (`projects-artifacts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V368-022](v368-022-artifact-request-validation.yml) | defect | P2 | — | Tighten Artifact namespace and revision validation |
| [V370-016](v370-016-simplify-artifacts.yml) | chore | P3 | — | Simplify the Artifact service and Runtime client |
| [V370-024](v370-024-artifact-purge-bound.yml) | defect | P2 | — | Bound Artifact purge and the number of bindings per Run |
| [V370-025](v370-025-artifact-read-efficiency.yml) | defect | P3 | — | Avoid full Artifact materialization and unindexed listing filters |

### Audits (`audits`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V62-010](v62-010-audit-composition-api.yml) | feature | P1 | — | Expose Audit preparation and exact composition provenance through the API |
| [V62-011](v62-011-audit-composition-ui.yml) | feature | P1 | V62-010 | Show preparation and generated coverage in the Audit user journey |
| [V62-012](v62-012-audit-composition-release.yml) | feature | P1 | V62-011 | Verify preparation and OpenAPI scans across production boundaries |
| [V64-001](v64-001-audit-priority-contracts.yml) | feature | P1 | — | Specify opt-in Audit checklist prioritization and exact top-N contracts |
| [V64-002](v64-002-audit-priority-store.yml) | feature | P1 | V64-001 | Persist Audit candidate inventory, ranking cycles and atomic admissions |
| [V64-003](v64-003-planner-only-priority-execution.yml) | feature | P1 | V64-001 | Support ordinary zero-Worker Runs for the audit-priority planner |
| [V64-004](v64-004-audit-priority-context.yml) | feature | P1 | V64-002 | Freeze bounded service and finding context for each priority cycle |
| [V64-005](v64-005-independent-priority-verdicts.yml) | feature | P1 | V64-003, V64-004 | Produce durable independent model verdicts for every remaining candidate |
| [V64-006](v64-006-audit-priority-selection.yml) | feature | P1 | V64-002 | Select exact top-N candidates and reject incomplete or undersized admission |
| [V64-007](v64-007-audit-priority-controller.yml) | feature | P1 | V64-005, V64-006 | Run prioritization before each Audit pass and advance deferred checklist work |
| [V64-008](v64-008-audit-priority-api-coverage.yml) | feature | P1 | V64-007 | Expose complete priority inventory, pass history and truthful Audit coverage |
| [V64-009](v64-009-audit-priority-ui.yml) | feature | P1 | V64-008 | Show Audit ranking progress, top-N selection and deferred checklist reasons |
| [V64-010](v64-010-audit-priority-profile.yml) | feature | P1 | V64-008 | Publish a runnable prioritized custom-checklist Audit profile and guide |
| [V64-011](v64-011-audit-priority-release.yml) | feature | P1 | V64-009, V64-010 | Verify contextual checklist prioritization across process and browser boundaries |
| [V368-006](v368-006-freeze-audit-artifacts.yml) | defect | P1 | — | Write Audit manifests, tasks and worklists frozen through one path |
| [V368-023](v368-023-audit-api-bounds.yml) | defect | P2 | — | Bound the Audit start replay and validate action review keys |
| [V370-004](v370-004-split-auditstore.yml) | chore | P2 | — | Split auditstore along its seams |
| [V370-014](v370-014-simplify-audit.yml) | chore | P3 | — | Remove duplicated code in the Audit packages |
| [V370-022](v370-022-batch-audit-reads.yml) | defect | P2 | — | Batch the Audit controller and owner finding reads |
| [V370-027](v370-027-audit-write-and-read-cost.yml) | defect | P3 | — | Reduce Audit start and event listing costs |

### Evaluations (`evals`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V40-003](v40-003-agent-instruction-eval-decision.yml) | feature | P1 | V40-002 | Evaluate instruction candidates and record evidence-based adoption decisions |

### Operations, authentication, CLI and deployment (`operations`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V368-020](v368-020-telemetry-export-defaults.yml) | defect | P2 | — | Let the Server pin telemetry export settings |

### Public API and wire contracts (`api-contracts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V368-002](v368-002-strict-wire-decoding.yml) | defect | P1 | — | Reject wire messages that omit required fields in Go |
| [V368-008](v368-008-go-python-text-rules.yml) | defect | P1 | — | Share the workspace path, blank-text and byte-count rules between Go and Python |
| [V368-032](v368-032-remove-legacy-digest-paths.yml) | chore | P3 | — | Remove backward-compatibility digest code |
| [V369-002](v369-002-wire-gate-all-fixtures.yml) | chore | P1 | — | Run every shared fixture on both sides and mutate the schemas |
| [V369-004](v369-004-route-table-and-operation-coverage.yml) | chore | P2 | — | Register routes from one table and validate every operation |
| [V370-005](v370-005-contracts-kernel-domain.yml) | chore | P2 | — | Move domain logic out of the contracts kernel |
| [V370-009](v370-009-context-handlers.yml) | chore | P2 | — | Split the public API handler by context with required dependencies |
| [V370-010](v370-010-domain-error-kinds.yml) | chore | P2 | — | Classify domain errors through one error kind and a code registry |
| [V370-017](v370-017-simplify-contracts.yml) | chore | P3 | — | Remove orphan schemas and lenient paths in the wire contracts |
| [V370-021](v370-021-live-event-polling.yml) | defect | P2 | — | Poll Run events once per Run instead of once per subscription |

### PostgreSQL persistence and schema (`persistence`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V368-026](v368-026-run-snapshot-json-paths.yml) | defect | P2 | — | Pin the JSON paths SQL reads from Go-encoded snapshots |
| [V369-001](v369-001-table-ownership-test.yml) | chore | P1 | — | Assign every table an owner package and test SQL against it |
| [V369-005](v369-005-sql-placement-concatenation.yml) | chore | P2 | — | Make the SQL placement check see concatenated statements |
| [V370-002](v370-002-cross-context-writes.yml) | chore | P2 | V369-001 | Move the remaining cross-context table writes behind their owners |
| [V370-007](v370-007-one-receipt-and-claim-mechanism.yml) | chore | P2 | — | Use one receipt and claim mechanism |
| [V370-028](v370-028-run-event-trigger-cost.yml) | defect | P3 | — | Skip trigger work for Run event appends |

### Quality, CI and process (`quality`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V347-001](v347-001-protect-main-branch.yml) | chore | P1 | — | Protect main with a required check |
| [V368-033](v368-033-test-fidelity-followups.yml) | chore | P3 | — | Close test gaps the fix batch left |
| [V369-003](v369-003-release-gate-coverage.yml) | chore | P1 | — | Add the missing checks to the release gate |
| [V369-007](v369-007-unused-export-lint.yml) | chore | P3 | V370-008 | Fail lint on exported identifiers used only by tests |
| [V370-008](v370-008-remove-test-only-exports.yml) | chore | P3 | — | Remove exported surface that only tests use |

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
| [V365-016](v365-016-fold-planner-execution-reports.yml) | persistence | 2026-10-04 | Explicit user approval of making Planner reports a mutable slot on stage_metrics. |

## Epics

All tasks of each epic by kind, and how many later tasks name one of its
tasks in `introduced_by`.

| Epic | Open | Feature | Defect | Chore | Later fixes caused |
| --- | ---: | ---: | ---: | ---: | ---: |
| Run execution core (`execution-core`) | 7 | 36 | 44 | 9 | 62 |
| Planners and the Worker model boundary (`planner-worker`) | 2 | 23 | 34 | 7 | 42 |
| Runtime labels, configuration and credentials (`runtime-config`) | 9 | 25 | 11 | 2 | 36 |
| Shared Memory tools (`memory`) | 1 | 6 | 8 | 1 | 11 |
| Agent Skills (`agent-skills`) | 0 | 10 | 2 | 1 | 1 |
| Run workspaces and project filesystem (`workspaces`) | 0 | 16 | 12 | 4 | 15 |
| Code analysis and taint annotations (`code-analysis`) | 0 | 13 | 12 | 2 | 17 |
| HTTP and Caido tools (`http-tools`) | 0 | 8 | 11 | 1 | 10 |
| Scanner tool Workers and scan planning (`scan-tools`) | 1 | 10 | 4 | 1 | 7 |
| Findings tools and collections (`findings`) | 2 | 7 | 4 | 1 | 6 |
| Workflow catalog and document workflows (`workflows`) | 6 | 13 | 13 | 8 | 11 |
| Projects, artifacts and lifecycle controls (`projects-artifacts`) | 4 | 27 | 31 | 3 | 33 |
| Audits (`audits`) | 24 | 49 | 64 | 12 | 71 |
| Evaluations (`evals`) | 2 | 21 | 26 | 4 | 33 |
| Podman sandbox (`sandbox`) | 0 | 8 | 1 | 1 | 2 |
| Web UI (`ui`) | 0 | 19 | 51 | 10 | 47 |
| Operations, authentication, CLI and deployment (`operations`) | 1 | 8 | 30 | 4 | 6 |
| Public API and wire contracts (`api-contracts`) | 10 | 1 | 24 | 16 | 6 |
| PostgreSQL persistence and schema (`persistence`) | 7 | 0 | 17 | 29 | 4 |
| Quality, CI and process (`quality`) | 7 | 1 | 15 | 65 | 15 |
