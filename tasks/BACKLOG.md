# Backlog

Generated from the task files by `make tasks-backlog`; do not edit by hand.
`make lint` fails when this file is stale. New tasks start from
[templates/](templates/); the task policy is in [index.yml](index.yml) and
the epics are in [epics.yml](epics.yml).

156 open tasks: 6 in progress, 125 pending, 25 deferred. 1009 finished: 1002 completed, 7 archived.

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
| [V407-013](v407-013-mtls-connection-reuse.yml) | chore | P2 | — | Reuse verified private TLS connections with bounded lifecycle ownership |
| [V407-014](v407-014-placement-per-poll-cost.yml) | chore | P2 | — | Reuse overlapping placement resolution work within a waiting attempt |
| [V407-015](v407-015-scheduler-stage-end-cost.yml) | chore | P3 | — | Reduce redundant Stage-end work while preserving recovery and redaction |
| [V407-016](v407-016-scheduler-idle-and-release-pacing.yml) | chore | P3 | — | Reduce idle claim probes and bound terminal-release head-of-line work |
| [V407-017](v407-017-startup-lease-tolerance.yml) | chore | P3 | — | Document and test inherited startup lease connection and query budgets |
| [V409-003](v409-003-runstore-lock-order-and-cas.yml) | defect | P2 | — | Lock the owning Run before the Stage in StartPlanner |
| [V409-010](v409-010-byte-bounded-run-reads.yml) | chore | P2 | — | Bound internal Run event fetches by bytes while preserving complete replay |
| [V409-011](v409-011-execution-store-n1-and-duplicates.yml) | chore | P3 | — | Reduce execution and Eval query round trips while preserving lock and recovery contracts |

### Planners and the Worker model boundary (`planner-worker`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-004](v371-004-worker-completion-exact-integers.yml) | defect | P3 | — | Bound WorkerCompletion integers to the range A2A carries exactly |
| [V371-005](v371-005-worker-state-read-cost-guard.yml) | chore | P3 | — | Guard the Worker State read cost with a deterministic bound and re-measure it |
| [V371-006](v371-006-workspace-observation-descriptor.yml) | chore | P3 | — | Take the Worker tools that observe the workspace from the shared Toolset descriptor |
| [V407-008](v407-008-planner-result-verification-classes.yml) | defect | P1 | — | Align Planner artifact-failure classes and report rejected invocations correctly |
| [V407-009](v407-009-worker-instruction-provider.yml) | defect | P1 | — | Pass authored Worker instructions literally to ADK |
| [V407-010](v407-010-worker-local-deadline.yml) | defect | P2 | — | Enforce the effective Worker-call deadline for ordinary ADK invocations |
| [V407-011](v407-011-worker-state-publication-cost.yml) | chore | P2 | — | Reduce repeated Worker metric serialization and unused transcript recording |
| [V407-012](v407-012-runtime-core-duplicates.yml) | chore | P3 | — | Consolidate equivalent Runtime helpers and offload prepare filesystem work |

### Runtime labels, configuration and credentials (`runtime-config`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-006](v372-006-runtime-label-set-rule.yml) | chore | P3 | — | Give the Runtime label set rule one Go definition and hold the UI and the Runtime to it |
| [V407-018](v407-018-credential-delete-lock-scope.yml) | chore | P2 | — | Document credential lifecycle availability tradeoffs and a bounded redesign proposal |
| [V407-019](v407-019-credential-recovery-fence-and-classifier.yml) | defect | P2 | — | Reconcile a dirty credential deletion fence against durable operation state |
| [V411-001](v411-001-target-credential-revisions.yml) | feature | P1 | — | Pin immutable target credential revisions in the existing encrypted store |

### Shared Memory tools (`memory`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V407-036](v407-036-memory-concurrent-reads.yml) | chore | P3 | — | Load Memory note payloads concurrently within the existing serialized operation |

### Run workspaces and project filesystem (`workspaces`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V407-034](v407-034-workspace-listing-and-export-cost.yml) | chore | P2 | — | Reduce repeated workspace glob work and export comparisons without changing snapshot semantics |

### Code analysis and taint annotations (`code-analysis`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V407-037](v407-037-code-analysis-hh-and-errors.yml) | defect | P2 | — | Parse .hh headers as C++ and distinguish parser implementation failures from source parse errors |

### HTTP and Caido tools (`http-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V386-004](v386-004-caido-catalog-cardinality.yml) | defect | P2 | — | Keep Caido catalog selection usable above one hundred entries |
| [V407-032](v407-032-http-caido-loop-and-error-classes.yml) | defect | P2 | — | Offload bounded Caido payload processing and preserve unexpected HTTP/Caido failures |
| [V410-003](v410-003-http-sitemap-collection.yml) | feature | P1 | V410-002 | Collect HTTP observations into the live Project sitemap |
| [V410-008](v410-008-sitemap-caido-history.yml) | feature | P2 | V410-004, V410-007 | Import selected Caido history incrementally with durable checkpoints |
| [V410-009](v410-009-sitemap-browser-collection.yml) | feature | P2 | V410-002, V410-007, V412-001 | Collect managed Playwright network observations into the sitemap |
| [V411-004](v411-004-controlled-http-conformance.yml) | chore | P1 | V411-003 | Verify controlled HTTP transport against an independent denial and fault matrix |
| [V411-005](v411-005-identity-sessions-http-login.yml) | feature | P1 | V411-001, V411-004 | Isolate target identities and execute bounded private HTTP login recipes |
| [V411-009](v411-009-http-recipes-predicates.yml) | feature | P1 | V411-005, V411-007, V411-008 | Execute frozen HTTP recipes and evaluate trusted three-valued predicates |
| [V412-001](v412-001-managed-pentest-browser.yml) | feature | P2 | V411-007, V411-008, V411-010 | Add a managed browser with isolated contexts and enforced target transport |
| [V412-002](v412-002-browser-login-replay-recipes.yml) | feature | P2 | V412-001, V411-009 | Execute closed browser login and replay recipes with trusted markers |
| [V412-004](v412-004-trusted-pentest-callback-proof.yml) | feature | P2 | V411-009, V411-012, V411-020 | Integrate independently attributable callback evidence and its release gate |

### Scanner tool Workers and scan planning (`scan-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V55-009](v55-009-scan-candidate-ranking.yml) | feature | P2 | — | Optionally rank existing scan candidates with an LLM |
| [V407-006](v407-006-scanplan-prepare-once.yml) | chore | P2 | — | Reuse one parsed OpenAPI source across an Audit inventory operation batch |
| [V407-007](v407-007-scan-output-off-loop.yml) | defect | P3 | — | Offload scanner parsing and filesystem preparation with joined cancellation |
| [V410-004](v410-004-sitemap-structured-imports.yml) | feature | P1 | V410-002 | Import bounded scanner and selected Caido observations |
| [V412-005](v412-005-controlled-scanner-pentest-integration.yml) | feature | P2 | V411-020, V410-004 | Gate controlled ffuf execution through the pentest transport |

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
| [V407-033](v407-033-findings-toolsets-cost.yml) | chore | P2 | — | Reuse immutable Audit inputs and finding source bytes, and assemble finding pages linearly |
| [V409-001](v409-001-finding-correlation-project-lock.yml) | defect | P2 | — | Retry definite transaction conflicts in Audit finding import and retention |
| [V409-002](v409-002-finding-inbox-index.yml) | chore | P2 | — | Bound Audit finding inbox reads to native and held receipts |
| [V411-007](v411-007-trusted-http-capture.yml) | feature | P1 | V411-004, V411-005, V411-006 | Expose safe HTTP evidence only through authenticated capture receipts |
| [V411-011](v411-011-project-findings-reader.yml) | feature | P1 | V411-002 | Expose safe live Project findings to allocated pentesters |

### Workflow catalog and document workflows (`workflows`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-013](v372-013-planner-facts-one-source.yml) | chore | P3 | — | Take Planner refs and facts from the descriptor registry in Go, the composition and Studio |
| [V372-014](v372-014-execution-config-patch-strict-decode.yml) | chore | P3 | — | Decode executionConfig patch members with strictjson |
| [V372-015](v372-015-config-split-leftovers.yml) | chore | P3 | — | Stop locals from shadowing the resolved package and drop config's copy of clone.Map |
| [V407-031](v407-031-document-session-off-loop.yml) | defect | P1 | — | Offload OpenAPI document preparation and avoid repeated validation while preserving session ownership |
| [V410-006](v410-006-sitemap-workflow-composition.yml) | feature | P1 | V410-003, V410-004, V410-005 | Ship ordinary sitemap discovery and reconciliation Workflows |
| [V411-014](v411-014-pentest-discovery-surface.yml) | feature | P1 | V411-007, V410-003, V410-006 | Discover URL-only targets and retain bounded source-aware surfaces |
| [V411-015](v411-015-http-pentest-workflow-presets.yml) | feature | P1 | V411-012, V411-013, V411-014, V411-011, V411-010, V411-016 | Publish exact HTTP API pentest Workflows and source-mode presets |
| [V412-003](v412-003-browser-pentest-profiles-and-gate.yml) | feature | P2 | V412-002, V411-020, V410-009 | Publish optional browser programs behind their independent release gate |

### Projects, artifacts and lifecycle controls (`projects-artifacts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V407-035](v407-035-artifact-reader-paging-cost.yml) | chore | P3 | — | Cache bounded text decoding for Artifact pages without changing reader interfaces |
| [V410-001](v410-001-project-sitemap-store.yml) | feature | P1 | — | Publish a bounded Project sitemap through one authorized reducer |
| [V410-002](v410-002-project-sitemap-toolset.yml) | feature | P1 | V410-001 | Expose authorized live sitemap tools to Project allocations |
| [V410-005](v410-005-sitemap-openapi-reconciliation.yml) | feature | P1 | V410-002 | Reconcile exact OpenAPI declarations with live sitemap members |

### Audits (`audits`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-009](v371-009-audit-list-preparation-batch.yml) | defect | P3 | — | Read the preparation state of a whole Audit page in bounded statements |
| [V407-001](v407-001-audit-collection-failure-classes.yml) | defect | P1 | — | Preserve transient Audit collection errors and settle invalid pinned data through existing receipts |
| [V407-002](v407-002-audit-collection-store-work.yml) | chore | P2 | — | Batch Audit collection lookups and retention within existing row and byte limits |
| [V407-003](v407-003-audit-controller-batch-per-pass.yml) | chore | P2 | — | Batch Audit drain mutations under a fixed per-step work budget |
| [V407-004](v407-004-audit-list-summary-projection.yml) | chore | P2 | — | Batch preparing Audit list hydration without changing the response contract |
| [V407-005](v407-005-audit-fail-closed-and-dead-code.yml) | chore | P3 | — | Remove redundant Audit checks and share safe intake preparation reads |
| [V409-004](v409-004-audit-read-statements-indexes.yml) | chore | P2 | — | Reduce avoidable Audit read work while preserving exact totals and expiry |
| [V409-005](v409-005-audit-review-write-guarantees.yml) | defect | P2 | — | Reject review decision writes when the locked request is not pending |
| [V411-002](v411-002-engagement-baseline-preflight.yml) | feature | P1 | V411-001 | Validate and pin a typed pentest engagement at Audit start |
| [V411-003](v411-003-action-journal-request-permits.yml) | feature | P1 | V411-002 | Fence and execute scoped HTTP preflight through the durable Audit action ledger |
| [V411-006](v411-006-cumulative-budget-finalization.yml) | feature | P1 | V411-002, V411-003 | Enforce cumulative pentest budgets and finalize at the hard active-time cap |
| [V411-008](v411-008-stateful-actions-cleanup.yml) | feature | P1 | V411-003, V411-004, V411-006, V411-007 | Serialize stateful target checks and retain unresolved cleanup effects |
| [V411-010](v411-010-pentest-program-availability.yml) | feature | P1 | V411-002 | Pin optional pentest programs and nonblocking finding review |
| [V411-012](v411-012-pentest-admission-rounds.yml) | feature | P1 | V411-009, V411-010, V62-005, V62-006, V62-007 | Admit typed pentest plans and derive exact later-round replay |
| [V411-016](v411-016-pentest-proof-report.yml) | feature | P1 | V411-012, V411-014 | Assemble retained pentest proof and incomplete-coverage reports |

### Evaluations (`evals`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V40-003](v40-003-agent-instruction-eval-decision.yml) | feature | P1 | V40-002 | Evaluate instruction candidates and record evidence-based adoption decisions |
| [V407-028](v407-028-eval-member-bulk-freeze.yml) | chore | P2 | — | Freeze admitted Eval members with one bounded set-based insert |
| [V407-029](v407-029-eval-reads-plan-metadata.yml) | chore | P2 | — | Use bounded Eval plan metadata and precomputed readiness for polled views |
| [V407-030](v407-030-eval-preparing-lifecycle-and-usage.yml) | defect | P3 | — | Make Eval usage aggregation deterministic when counters overflow |
| [V409-009](v409-009-eval-store-bounds.yml) | defect | P2 | — | Purge drained Eval data in bounded, restartable transactions |
| [V411-019](v411-019-pentest-target-fixtures-and-proof-scorers.yml) | chore | P1 | V411-002 | Build paired pentest fixtures and independent proof scorers |
| [V411-021](v411-021-pentest-frozen-live-quality-evaluation.yml) | chore | P2 | V411-020 | Evaluate pentest quality against a frozen repeated-sample baseline |

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
| [V407-020](v407-020-telemetry-trim-and-history.yml) | chore | P2 | — | Reduce bounded telemetry trimming and operations-history copying |

### Public API and wire contracts (`api-contracts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-016](v371-016-idempotency-key-grammar.yml) | defect | P3 | — | Give the Idempotency-Key grammar one Go definition |
| [V372-020](v372-020-error-code-vocabularies.yml) | chore | P3 | — | Hold every vocabulary that carries error code names to the registry |
| [V372-025](v372-025-typed-openapi-enums.yml) | chore | P3 | — | Give every enum of the public OpenAPI document a type |
| [V372-026](v372-026-request-digests-in-services.yml) | chore | P3 | — | Move the Project, Audit and review request digests out of the public handlers |
| [V407-021](v407-021-public-api-grammars-one-definition.yml) | defect | P1 | — | Align Idempotency-Key, request body bounds and UTF-8 string limits with the public contract |
| [V407-022](v407-022-reject-head-centrally.yml) | defect | P2 | — | Reject implicit HEAD before dispatching public GET handlers |
| [V407-023](v407-023-mutation-handler-preamble.yml) | defect | P3 | — | Classify aborted RuntimeConfig bodies and reject invalid mutation metadata before expensive work |
| [V407-024](v407-024-cross-language-limits-and-grammars.yml) | defect | P1 | — | Align OpenAPI authoring document bounds with downstream scan preparation |
| [V411-017](v411-017-pentest-api-cli.yml) | feature | P1 | V411-010, V411-011, V411-012, V411-016, V411-006 | Expose pentest drafts progress surfaces and proof through API and CLI |

### PostgreSQL persistence and schema (`persistence`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-008](v372-008-upward-table-reads.yml) | chore | P3 | — | Remove the table reads that point up the layer stack or name who owns each one |
| [V372-009](v372-009-ownership-check-blind-spots.yml) | chore | P3 | — | Close the blind spots of the table ownership check |
| [V372-010](v372-010-erd-usage-from-ownership-scan.yml) | chore | P3 | — | Take the ERD's code-usage lists from the ownership scan and document the new-table workflow |
| [V407-025](v407-025-persistence-indexes-and-triggers.yml) | chore | P2 | — | Reduce Project-drain and held-Audit query work and scope Stage triggers |
| [V407-026](v407-026-store-error-classes-and-event-validation.yml) | defect | P3 | — | Classify missing Project references at the Run-store boundary |
| [V407-027](v407-027-shared-controller-claim-loop.yml) | chore | P3 | — | Share controller claim-lifecycle plumbing without merging distinct scheduling policies |
| [V409-006](v409-006-trigger-column-scoping.yml) | chore | P2 | — | Scope unnecessary trigger work while preserving admission and projection effects |
| [V409-007](v409-007-dead-and-redundant-indexes.yml) | chore | P2 | — | Consolidate indexes only where reader and constraint evidence permits it |
| [V409-008](v409-008-state-set-schema-tests.yml) | chore | P3 | — | Extend migrated-schema state-set parity tests to the remaining reviewed owners |
| [V409-012](v409-012-migration-and-sql-hygiene.yml) | chore | P3 | — | Test the revision-count and controller-claim backfills and document migration budgets |
| [V409-013](v409-013-verify-sql-review-on-database.yml) | chore | P2 | V409-001, V409-002, V409-003, V409-004, V409-005, V409-006, V409-007, V409-008, V409-009, V409-010, V409-011, V409-012 | Verify retained SQL-review changes and record evidence for every accepted or declined claim |

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
| [V407-038](v407-038-runtime-dead-surface.yml) | chore | P3 | — | Move the shared filesystem tool error type into toolsets.common |
| [V407-039](v407-039-close-server-runtime-review.yml) | chore | P3 | V407-001, V407-002, V407-003, V407-004, V407-005, V407-006, V407-007, V407-008, V407-009, V407-010, V407-011, V407-012, V407-013, V407-014, V407-015, V407-016, V407-017, V407-018, V407-019, V407-020, V407-021, V407-022, V407-023, V407-024, V407-025, V407-026, V407-027, V407-028, V407-029, V407-030, V407-031, V407-032, V407-033, V407-034, V407-035, V407-036, V407-037, V407-038 | Close the validated Server and Runtime review with explicit dispositions and verification |
| [V410-007](v410-007-sitemap-release-gate.yml) | chore | P1 | V410-006 | Verify the first sitemap slice across scale and lifecycle boundaries |
| [V411-020](v411-020-pentest-http-release-gate.yml) | chore | P1 | V411-001, V411-002, V411-003, V411-004, V411-005, V411-006, V411-007, V411-008, V411-009, V411-010, V411-011, V411-012, V411-013, V411-014, V411-015, V411-016, V411-017, V411-018, V411-019, V410-007 | Gate the first HTTP API pentest delivery through real processes |

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
| Run execution core (`execution-core`) | 11 | 41 | 52 | 18 | 74 |
| Planners and the Worker model boundary (`planner-worker`) | 8 | 27 | 40 | 16 | 59 |
| Runtime labels, configuration and credentials (`runtime-config`) | 13 | 26 | 13 | 4 | 40 |
| Shared Memory tools (`memory`) | 1 | 6 | 9 | 2 | 12 |
| Agent Skills (`agent-skills`) | 0 | 11 | 2 | 1 | 2 |
| Run workspaces and project filesystem (`workspaces`) | 1 | 16 | 12 | 5 | 16 |
| Code analysis and taint annotations (`code-analysis`) | 1 | 14 | 14 | 2 | 19 |
| HTTP and Caido tools (`http-tools`) | 11 | 19 | 15 | 5 | 16 |
| Scanner tool Workers and scan planning (`scan-tools`) | 6 | 16 | 5 | 7 | 11 |
| Findings tools and collections (`findings`) | 12 | 16 | 6 | 8 | 11 |
| Workflow catalog and document workflows (`workflows`) | 8 | 18 | 14 | 12 | 15 |
| Projects, artifacts and lifecycle controls (`projects-artifacts`) | 4 | 33 | 33 | 6 | 37 |
| Audits (`audits`) | 31 | 60 | 73 | 22 | 93 |
| Evaluations (`evals`) | 8 | 21 | 34 | 9 | 40 |
| Podman sandbox (`sandbox`) | 1 | 10 | 2 | 1 | 3 |
| Web UI (`ui`) | 2 | 25 | 61 | 13 | 55 |
| Operations, authentication, CLI and deployment (`operations`) | 1 | 16 | 33 | 6 | 11 |
| Public API and wire contracts (`api-contracts`) | 9 | 2 | 34 | 20 | 11 |
| PostgreSQL persistence and schema (`persistence`) | 12 | 0 | 21 | 41 | 10 |
| Quality, CI and process (`quality`) | 16 | 1 | 25 | 91 | 22 |
