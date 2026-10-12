# Backlog

Generated from the task files by `make tasks-backlog`; do not edit by hand.
`make lint` fails when this file is stale. New tasks start from
[templates/](templates/); the task policy is in [index.yml](index.yml) and
the epics are in [epics.yml](epics.yml).

52 open tasks: 4 in progress, 34 pending, 14 deferred. 848 finished: 845 completed, 3 archived.

## In progress

| Task | Epic | Kind | Priority | Title |
| --- | --- | --- | --- | --- |
| [V40-002](v40-002-agent-instruction-paired-runner.yml) | evals | feature | P1 | Prepare instruction experiment bindings and a frozen plan after format readiness |
| [V261-001](v261-001-stabilize-release-gate-process-tests.yml) | quality | chore | P1 | Restore the deterministic release process gate |
| [V341-001](v341-001-investigate-ci-tolerance-bumps.yml) | quality | chore | P2 | Investigate tolerances raised to unblock CI |
| [V370-006](v370-006-config-model-split.yml) | workflows | chore | P2 | Separate the configuration data model from the catalog loader |

## Pending

### Run execution core (`execution-core`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V370-001](v370-001-run-stage-transition-owner.yml) | chore | P1 | V369-001 | Give Run and Stage state transitions one owner |
| [V370-011](v370-011-scheduler-controlplane-surface.yml) | chore | P3 | — | Slim the Scheduler and Control Plane surfaces |
| [V370-012](v370-012-runstore-runtimeconfig-cleanup.yml) | chore | P3 | — | Merge Run creation paths and split the Runtime principal registry |

### Scanner tool Workers and scan planning (`scan-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V55-009](v55-009-scan-candidate-ranking.yml) | feature | P2 | — | Optionally rank existing scan candidates with an LLM |

### Projects, artifacts and lifecycle controls (`projects-artifacts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V370-016](v370-016-simplify-artifacts.yml) | chore | P3 | — | Simplify the Artifact service and Runtime client |

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
| [V370-004](v370-004-split-auditstore.yml) | chore | P2 | — | Split auditstore along its seams |

### Evaluations (`evals`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V40-003](v40-003-agent-instruction-eval-decision.yml) | feature | P1 | V40-002 | Evaluate instruction candidates and record evidence-based adoption decisions |

### Public API and wire contracts (`api-contracts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V369-002](v369-002-wire-gate-all-fixtures.yml) | chore | P1 | — | Run every shared fixture on both sides and mutate the schemas |
| [V369-004](v369-004-route-table-and-operation-coverage.yml) | chore | P2 | — | Register routes from one table and validate every operation |
| [V370-005](v370-005-contracts-kernel-domain.yml) | chore | P2 | — | Move domain logic out of the contracts kernel |
| [V370-009](v370-009-context-handlers.yml) | chore | P2 | — | Split the public API handler by context with required dependencies |
| [V370-010](v370-010-domain-error-kinds.yml) | chore | P2 | — | Classify domain errors through one error kind and a code registry |
| [V370-017](v370-017-simplify-contracts.yml) | chore | P3 | — | Remove orphan schemas and lenient paths in the wire contracts |

### PostgreSQL persistence and schema (`persistence`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V369-001](v369-001-table-ownership-test.yml) | chore | P1 | — | Assign every table an owner package and test SQL against it |
| [V369-005](v369-005-sql-placement-concatenation.yml) | chore | P2 | — | Make the SQL placement check see concatenated statements |
| [V370-002](v370-002-cross-context-writes.yml) | chore | P2 | V369-001 | Move the remaining cross-context table writes behind their owners |
| [V370-007](v370-007-one-receipt-and-claim-mechanism.yml) | chore | P2 | — | Use one receipt and claim mechanism |

### Quality, CI and process (`quality`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V347-001](v347-001-protect-main-branch.yml) | chore | P1 | — | Protect main with a required check |
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
| Run execution core (`execution-core`) | 3 | 36 | 44 | 9 | 62 |
| Planners and the Worker model boundary (`planner-worker`) | 0 | 23 | 34 | 7 | 42 |
| Runtime labels, configuration and credentials (`runtime-config`) | 9 | 25 | 11 | 2 | 37 |
| Shared Memory tools (`memory`) | 0 | 6 | 8 | 1 | 11 |
| Agent Skills (`agent-skills`) | 0 | 10 | 2 | 1 | 1 |
| Run workspaces and project filesystem (`workspaces`) | 0 | 16 | 12 | 4 | 15 |
| Code analysis and taint annotations (`code-analysis`) | 0 | 13 | 12 | 2 | 17 |
| HTTP and Caido tools (`http-tools`) | 0 | 8 | 11 | 1 | 10 |
| Scanner tool Workers and scan planning (`scan-tools`) | 1 | 10 | 4 | 1 | 7 |
| Findings tools and collections (`findings`) | 0 | 7 | 4 | 1 | 7 |
| Workflow catalog and document workflows (`workflows`) | 1 | 13 | 13 | 8 | 11 |
| Projects, artifacts and lifecycle controls (`projects-artifacts`) | 1 | 27 | 31 | 3 | 34 |
| Audits (`audits`) | 19 | 49 | 64 | 12 | 73 |
| Evaluations (`evals`) | 2 | 21 | 27 | 4 | 33 |
| Podman sandbox (`sandbox`) | 0 | 8 | 1 | 1 | 2 |
| Web UI (`ui`) | 0 | 19 | 51 | 10 | 48 |
| Operations, authentication, CLI and deployment (`operations`) | 0 | 8 | 30 | 4 | 6 |
| Public API and wire contracts (`api-contracts`) | 6 | 1 | 24 | 16 | 7 |
| PostgreSQL persistence and schema (`persistence`) | 5 | 0 | 17 | 29 | 4 |
| Quality, CI and process (`quality`) | 5 | 1 | 18 | 65 | 15 |
