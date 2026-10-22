# Backlog

Generated from the task files by `make tasks-backlog`; do not edit by hand.
`make lint` fails when this file is stale. New tasks start from
[templates/](templates/); the task policy is in [index.yml](index.yml) and
the epics are in [epics.yml](epics.yml).

74 open tasks: 4 in progress, 45 pending, 25 deferred. 927 finished: 924 completed, 3 archived.

## In progress

| Task | Epic | Kind | Priority | Title |
| --- | --- | --- | --- | --- |
| [V40-002](v40-002-agent-instruction-paired-runner.yml) | evals | feature | P1 | Prepare instruction experiment bindings and a frozen plan after format readiness |
| [V261-001](v261-001-stabilize-release-gate-process-tests.yml) | quality | chore | P1 | Restore the deterministic release process gate |
| [V341-001](v341-001-investigate-ci-tolerance-bumps.yml) | quality | chore | P2 | Investigate tolerances raised to unblock CI |
| [V392-002](v392-002-ffuf-transport-fixture.yml) | scan-tools | chore | P3 | Make scanner subprocess fixtures deterministic |

## Pending

### Run execution core (`execution-core`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-014](v371-014-scheduler-stage-history-reads.yml) | defect | P3 | — | Read only the Stage history fields escalation counting and cancellation cleanup use |
| [V372-007](v372-007-terminal-stage-state-set.yml) | chore | P3 | — | Define the terminal StageExecution states once and hold SQL and the UI to them |
| [V372-028](v372-028-runstore-strict-snapshot-reads.yml) | chore | P3 | — | Decode persisted Run, Stage and allocation snapshots strictly in runstore |

### Planners and the Worker model boundary (`planner-worker`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-004](v371-004-worker-completion-exact-integers.yml) | defect | P3 | — | Bound WorkerCompletion integers to the range A2A carries exactly |
| [V371-005](v371-005-worker-state-read-cost-guard.yml) | chore | P3 | — | Guard the Worker State read cost with a deterministic bound and re-measure it |
| [V371-006](v371-006-workspace-observation-descriptor.yml) | chore | P3 | — | Take the Worker tools that observe the workspace from the shared Toolset descriptor |

### Runtime labels, configuration and credentials (`runtime-config`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-006](v372-006-runtime-label-set-rule.yml) | chore | P3 | — | Give the Runtime label set rule one Go definition and hold the UI and the Runtime to it |
| [V372-021](v372-021-credential-master-key-unavailable.yml) | defect | P2 | — | Answer an LLM or Runtime credential operation without the master key with a registered non-retryable code |

### HTTP and Caido tools (`http-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V382-001](v382-001-interactsh-worker-tools.yml) | feature | P1 | — | Add configurable Interactsh tools for Worker callback evidence |
| [V386-004](v386-004-caido-catalog-cardinality.yml) | defect | P2 | — | Keep Caido catalog selection usable above one hundred entries |

### Scanner tool Workers and scan planning (`scan-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V55-009](v55-009-scan-candidate-ranking.yml) | feature | P2 | — | Optionally rank existing scan candidates with an LLM |

### Findings tools and collections (`findings`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-010](v371-010-discarded-receipt-projection.yml) | defect | P2 | — | Drop the report projection of a proposal receipt discarded by Run deletion |
| [V371-011](v371-011-proposal-quota-limits.yml) | chore | P3 | — | Register the per-Run proposal and per-proposal evidence quotas in the specification |
| [V372-018](v372-018-finding-proposal-namespace-constant.yml) | chore | P3 | — | Name the finding-proposals Namespace through artifactpolicy in finding intake |
| [V372-023](v372-023-findings-tool-error-codes.yml) | chore | P3 | — | Name the findings error codes the open question lists and register the findings tool codes |

### Workflow catalog and document workflows (`workflows`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-013](v372-013-planner-facts-one-source.yml) | chore | P3 | — | Take Planner refs and facts from the descriptor registry in Go, the composition and Studio |
| [V372-014](v372-014-execution-config-patch-strict-decode.yml) | chore | P3 | — | Decode executionConfig patch members with strictjson |
| [V372-015](v372-015-config-split-leftovers.yml) | chore | P3 | — | Stop locals from shadowing the resolved package and drop config's copy of clone.Map |

### Projects, artifacts and lifecycle controls (`projects-artifacts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-019](v371-019-audit-namespace-prefix-rule.yml) | chore | P3 | — | Use artifactpolicy's Audit namespace rule in the Audit namespace purge |

### Audits (`audits`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-009](v371-009-audit-list-preparation-batch.yml) | defect | P3 | — | Read the preparation state of a whole Audit page in bounded statements |

### Evaluations (`evals`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V40-003](v40-003-agent-instruction-eval-decision.yml) | feature | P1 | V40-002 | Evaluate instruction candidates and record evidence-based adoption decisions |
| [V371-013](v371-013-eval-bounded-json-walker.yml) | chore | P3 | — | Read Eval JSON documents with strictjson.DecodeBounded |

### Web UI (`ui`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-005](v372-005-issues-decision-focus-race.yml) | defect | P2 | — | Move focus to the next possible issue reliably after a decision |

### Public API and wire contracts (`api-contracts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-016](v371-016-idempotency-key-grammar.yml) | defect | P3 | — | Give the Idempotency-Key grammar one Go definition |
| [V371-017](v371-017-strong-etag-grammar.yml) | defect | P3 | — | Accept only RFC 9110 strong entity tags in If-Match |
| [V371-018](v371-018-repeat-draft-description.yml) | chore | P3 | — | Remove the historical-request sentence from the repeat-draft OpenAPI description |
| [V372-020](v372-020-error-code-vocabularies.yml) | chore | P3 | — | Hold every vocabulary that carries error code names to the registry |
| [V372-022](v372-022-standby-public-api-envelope.yml) | defect | P3 | — | Answer the public API on a standby Server with the error envelope |
| [V372-025](v372-025-typed-openapi-enums.yml) | chore | P3 | — | Give every enum of the public OpenAPI document a type |
| [V372-026](v372-026-request-digests-in-services.yml) | chore | P3 | — | Move the Project, Audit and review request digests out of the public handlers |

### PostgreSQL persistence and schema (`persistence`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-008](v372-008-upward-table-reads.yml) | chore | P3 | — | Remove the table reads that point up the layer stack or name who owns each one |
| [V372-009](v372-009-ownership-check-blind-spots.yml) | chore | P3 | — | Close the blind spots of the table ownership check |
| [V372-010](v372-010-erd-usage-from-ownership-scan.yml) | chore | P3 | — | Take the ERD's code-usage lists from the ownership scan and document the new-table workflow |

### Quality, CI and process (`quality`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V347-001](v347-001-protect-main-branch.yml) | chore | P1 | — | Protect main with a required check |
| [V371-015](v371-015-memory-claim-paused-skip.yml) | chore | P3 | — | Model the paused-owner claim skip in the Scheduler's memory store doubles |
| [V371-021](v371-021-postgres-test-wall-clock.yml) | chore | P3 | — | Replace wall-clock budgets in the Scheduler and runstore PostgreSQL tests |
| [V371-022](v371-022-capability-probe-test-budget.yml) | chore | P3 | — | Remove the 10 ms probe budget from the capability discovery tests |
| [V371-023](v371-023-audit-route-test-timeouts.yml) | chore | P3 | — | Keep the Audit route UI tests within their timeout under load |
| [V371-024](v371-024-browser-gate-ci-artifacts.yml) | chore | P3 | — | Keep browser gate screenshots and traces where CI uploads them |
| [V371-025](v371-025-pnpm-root-invocation.yml) | chore | P3 | — | Document a pnpm invocation that works from the repository root |
| [V372-016](v372-016-wall-clock-test-budgets-wave-2.yml) | chore | P3 | — | Replace wall-clock budgets in the finding intake, Audit, Eval and persistence tests and the result publication test |
| [V372-019](v372-019-restore-rehearsal-pg-dump-version.yml) | chore | P3 | — | Check the pg_dump and pg_restore version before the backup and restore rehearsal |
| [V372-024](v372-024-audit-item-attempts-fake.yml) | chore | P3 | — | Make the Audit item attempt fake answer like the store and keep the item list schema-valid without it |
| [V372-029](v372-029-shared-fixture-gate-scope.yml) | chore | P3 | — | Extend the shared fixture gate to every cross-language fixture and keep slow store suites out of it |
| [V372-030](v372-030-shrink-test-only-export-baseline.yml) | chore | P3 | — | Resolve the exported identifiers that only tests use and empty the test-only export baseline |

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
| Run execution core (`execution-core`) | 3 | 36 | 49 | 11 | 64 |
| Planners and the Worker model boundary (`planner-worker`) | 3 | 27 | 37 | 14 | 51 |
| Runtime labels, configuration and credentials (`runtime-config`) | 11 | 25 | 12 | 3 | 37 |
| Shared Memory tools (`memory`) | 0 | 6 | 8 | 1 | 11 |
| Agent Skills (`agent-skills`) | 0 | 10 | 2 | 1 | 1 |
| Run workspaces and project filesystem (`workspaces`) | 0 | 16 | 12 | 4 | 15 |
| Code analysis and taint annotations (`code-analysis`) | 0 | 13 | 12 | 2 | 17 |
| HTTP and Caido tools (`http-tools`) | 2 | 11 | 14 | 4 | 15 |
| Scanner tool Workers and scan planning (`scan-tools`) | 2 | 10 | 4 | 2 | 8 |
| Findings tools and collections (`findings`) | 4 | 7 | 5 | 4 | 7 |
| Workflow catalog and document workflows (`workflows`) | 3 | 14 | 13 | 12 | 12 |
| Projects, artifacts and lifecycle controls (`projects-artifacts`) | 1 | 30 | 32 | 4 | 34 |
| Audits (`audits`) | 16 | 50 | 69 | 15 | 80 |
| Evaluations (`evals`) | 3 | 21 | 29 | 5 | 33 |
| Podman sandbox (`sandbox`) | 0 | 9 | 1 | 1 | 2 |
| Web UI (`ui`) | 1 | 20 | 54 | 11 | 49 |
| Operations, authentication, CLI and deployment (`operations`) | 0 | 11 | 32 | 5 | 9 |
| Public API and wire contracts (`api-contracts`) | 7 | 1 | 29 | 20 | 8 |
| PostgreSQL persistence and schema (`persistence`) | 4 | 0 | 19 | 33 | 8 |
| Quality, CI and process (`quality`) | 14 | 1 | 20 | 78 | 16 |
