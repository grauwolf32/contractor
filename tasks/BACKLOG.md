# Backlog

Generated from the task files by `make tasks-backlog`; do not edit by hand.
`make lint` fails when this file is stale. New tasks start from
[templates/](templates/); the task policy is in [index.yml](index.yml) and
the epics are in [epics.yml](epics.yml).

75 open tasks: 3 in progress, 58 pending, 14 deferred. 849 finished: 846 completed, 3 archived.

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
| [V370-001](v370-001-run-stage-transition-owner.yml) | chore | P1 | — | Give Run and Stage state transitions one owner |
| [V370-011](v370-011-scheduler-controlplane-surface.yml) | chore | P3 | — | Slim the Scheduler and Control Plane surfaces |
| [V370-012](v370-012-runstore-runtimeconfig-cleanup.yml) | chore | P3 | — | Merge Run creation paths and split the Runtime principal registry |
| [V371-014](v371-014-scheduler-stage-history-reads.yml) | defect | P3 | — | Read only the Stage history fields escalation counting and cancellation cleanup use |

### Planners and the Worker model boundary (`planner-worker`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-002](v371-002-worker-state-endpoint-contract.yml) | defect | P2 | — | Serve a schema-valid Worker State and classify its failures like the Server |
| [V371-004](v371-004-worker-completion-exact-integers.yml) | defect | P3 | — | Bound WorkerCompletion integers to the range A2A carries exactly |
| [V371-005](v371-005-worker-state-read-cost-guard.yml) | chore | P3 | — | Guard the Worker State read cost with a deterministic bound and re-measure it |
| [V371-006](v371-006-workspace-observation-descriptor.yml) | chore | P3 | — | Take the Worker tools that observe the workspace from the shared Toolset descriptor |

### Scanner tool Workers and scan planning (`scan-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V55-009](v55-009-scan-candidate-ranking.yml) | feature | P2 | — | Optionally rank existing scan candidates with an LLM |

### Findings tools and collections (`findings`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-010](v371-010-discarded-receipt-projection.yml) | defect | P2 | — | Drop the report projection of a proposal receipt discarded by Run deletion |
| [V371-011](v371-011-proposal-quota-limits.yml) | chore | P3 | — | Register the per-Run proposal and per-proposal evidence quotas in the specification |

### Workflow catalog and document workflows (`workflows`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V370-006](v370-006-config-model-split.yml) | chore | P2 | — | Separate the configuration data model from the catalog loader |

### Projects, artifacts and lifecycle controls (`projects-artifacts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V370-016](v370-016-simplify-artifacts.yml) | chore | P3 | — | Simplify the Artifact service and Runtime client |
| [V371-019](v371-019-audit-namespace-prefix-rule.yml) | chore | P3 | — | Use artifactpolicy's Audit namespace rule in the Audit namespace purge |

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
| [V371-007](v371-007-fence-audit-artifact-writes.yml) | defect | P2 | — | Fence the Audit Controller's artifact writes against a purge by a newer claim |
| [V371-008](v371-008-bounded-audit-deletion-steps.yml) | defect | P2 | — | Run the Audit deletion phase in bounded steps under its own budget |
| [V371-009](v371-009-audit-list-preparation-batch.yml) | defect | P3 | — | Read the preparation state of a whole Audit page in bounded statements |

### Evaluations (`evals`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V40-003](v40-003-agent-instruction-eval-decision.yml) | feature | P1 | V40-002 | Evaluate instruction candidates and record evidence-based adoption decisions |
| [V371-012](v371-012-eval-projection-lock-order.yml) | defect | P2 | — | Stop Eval member collection and Audit collection from deadlocking |
| [V371-013](v371-013-eval-bounded-json-walker.yml) | chore | P3 | — | Read Eval JSON documents with strictjson.DecodeBounded |

### Web UI (`ui`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-020](v371-020-ui-resource-id-grammar.yml) | defect | P3 | — | Give the UI one ResourceId and Revision grammar and check revisions before If-Match |

### Public API and wire contracts (`api-contracts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V369-002](v369-002-wire-gate-all-fixtures.yml) | chore | P1 | — | Run every shared fixture on both sides and mutate the schemas |
| [V369-004](v369-004-route-table-and-operation-coverage.yml) | chore | P2 | — | Register routes from one table and validate every operation |
| [V370-005](v370-005-contracts-kernel-domain.yml) | chore | P2 | — | Move domain logic out of the contracts kernel |
| [V370-009](v370-009-context-handlers.yml) | chore | P2 | — | Split the public API handler by context with required dependencies |
| [V370-010](v370-010-domain-error-kinds.yml) | chore | P2 | — | Classify domain errors through one error kind and a code registry |
| [V370-017](v370-017-simplify-contracts.yml) | chore | P3 | — | Remove orphan schemas and lenient paths in the wire contracts |
| [V371-003](v371-003-reject-optional-member-null.yml) | defect | P3 | V371-002 | Reject explicit null for optional private-wire members and null map values |
| [V371-016](v371-016-idempotency-key-grammar.yml) | defect | P3 | — | Give the Idempotency-Key grammar one Go definition |
| [V371-017](v371-017-strong-etag-grammar.yml) | defect | P3 | — | Accept only RFC 9110 strong entity tags in If-Match |
| [V371-018](v371-018-repeat-draft-description.yml) | chore | P3 | — | Remove the historical-request sentence from the repeat-draft OpenAPI description |

### PostgreSQL persistence and schema (`persistence`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V369-005](v369-005-sql-placement-concatenation.yml) | chore | P2 | — | Make the SQL placement check see concatenated statements |
| [V370-002](v370-002-cross-context-writes.yml) | chore | P2 | — | Move the remaining cross-context table writes behind their owners |
| [V370-007](v370-007-one-receipt-and-claim-mechanism.yml) | chore | P2 | — | Use one receipt and claim mechanism |

### Quality, CI and process (`quality`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V347-001](v347-001-protect-main-branch.yml) | chore | P1 | — | Protect main with a required check |
| [V369-007](v369-007-unused-export-lint.yml) | chore | P3 | V370-008 | Fail lint on exported identifiers used only by tests |
| [V370-008](v370-008-remove-test-only-exports.yml) | chore | P3 | — | Remove exported surface that only tests use |
| [V371-015](v371-015-memory-claim-paused-skip.yml) | chore | P3 | — | Model the paused-owner claim skip in the Scheduler's memory store doubles |
| [V371-021](v371-021-postgres-test-wall-clock.yml) | chore | P3 | — | Replace wall-clock budgets in the Scheduler and runstore PostgreSQL tests |
| [V371-022](v371-022-capability-probe-test-budget.yml) | chore | P3 | — | Remove the 10 ms probe budget from the capability discovery tests |
| [V371-023](v371-023-audit-route-test-timeouts.yml) | chore | P3 | — | Keep the Audit route UI tests within their timeout under load |
| [V371-024](v371-024-browser-gate-ci-artifacts.yml) | chore | P3 | — | Keep browser gate screenshots and traces where CI uploads them |
| [V371-025](v371-025-pnpm-root-invocation.yml) | chore | P3 | — | Document a pnpm invocation that works from the repository root |

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
| Run execution core (`execution-core`) | 4 | 36 | 45 | 9 | 63 |
| Planners and the Worker model boundary (`planner-worker`) | 4 | 23 | 36 | 9 | 44 |
| Runtime labels, configuration and credentials (`runtime-config`) | 9 | 25 | 11 | 2 | 37 |
| Shared Memory tools (`memory`) | 0 | 6 | 8 | 1 | 11 |
| Agent Skills (`agent-skills`) | 0 | 10 | 2 | 1 | 1 |
| Run workspaces and project filesystem (`workspaces`) | 0 | 16 | 12 | 4 | 15 |
| Code analysis and taint annotations (`code-analysis`) | 0 | 13 | 12 | 2 | 17 |
| HTTP and Caido tools (`http-tools`) | 0 | 8 | 11 | 1 | 10 |
| Scanner tool Workers and scan planning (`scan-tools`) | 1 | 10 | 4 | 1 | 7 |
| Findings tools and collections (`findings`) | 2 | 7 | 5 | 2 | 7 |
| Workflow catalog and document workflows (`workflows`) | 1 | 13 | 13 | 8 | 11 |
| Projects, artifacts and lifecycle controls (`projects-artifacts`) | 2 | 27 | 31 | 4 | 34 |
| Audits (`audits`) | 22 | 49 | 67 | 12 | 75 |
| Evaluations (`evals`) | 4 | 21 | 28 | 5 | 33 |
| Podman sandbox (`sandbox`) | 0 | 8 | 1 | 1 | 2 |
| Web UI (`ui`) | 1 | 19 | 52 | 10 | 48 |
| Operations, authentication, CLI and deployment (`operations`) | 0 | 8 | 30 | 4 | 6 |
| Public API and wire contracts (`api-contracts`) | 10 | 1 | 27 | 17 | 7 |
| PostgreSQL persistence and schema (`persistence`) | 4 | 0 | 17 | 29 | 4 |
| Quality, CI and process (`quality`) | 11 | 1 | 18 | 71 | 15 |
