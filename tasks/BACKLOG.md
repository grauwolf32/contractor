# Backlog

Generated from the task files by `make tasks-backlog`; do not edit by hand.
`make lint` fails when this file is stale. New tasks start from
[templates/](templates/); the task policy is in [index.yml](index.yml) and
the epics are in [epics.yml](epics.yml).

121 open tasks: 5 in progress, 91 pending, 25 deferred. 981 finished: 978 completed, 3 archived.

## In progress

| Task | Epic | Kind | Priority | Title |
| --- | --- | --- | --- | --- |
| [V40-002](v40-002-agent-instruction-paired-runner.yml) | evals | feature | P1 | Prepare instruction experiment bindings and a frozen plan after format readiness |
| [V261-001](v261-001-stabilize-release-gate-process-tests.yml) | quality | chore | P1 | Restore the deterministic release process gate |
| [V341-001](v341-001-investigate-ci-tolerance-bumps.yml) | quality | chore | P2 | Investigate tolerances raised to unblock CI |
| [V398-001](v398-001-runtime-completion-capability-derivation.yml) | audits | chore | P3 | Derive the Runtime's completion-contract capability from one rule |
| [V406-004](v406-004-durable-scope-sweeps.yml) | scan-tools | feature | P2 | Coordinate durable bounded Nuclei sweeps over Project scopes |

## Pending

### Run execution core (`execution-core`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-014](v371-014-scheduler-stage-history-reads.yml) | defect | P3 | — | Read only the Stage history fields escalation counting and cancellation cleanup use |
| [V372-007](v372-007-terminal-stage-state-set.yml) | chore | P3 | — | Define the terminal StageExecution states once and hold SQL and the UI to them |
| [V372-028](v372-028-runstore-strict-snapshot-reads.yml) | chore | P3 | — | Decode persisted Run, Stage and allocation snapshots strictly in runstore |
| [V405-002](v405-002-run-placement-requirements.yml) | feature | P2 | — | Pin Run placement requirements and enforce dynamic Runtime matching |
| [V405-005](v405-005-placement-process-release-gate.yml) | feature | P2 | V405-002, V405-003, V405-004 | Verify cloud and local placement across real processes and fold proposal 0043 |
| [V407-013](v407-013-mtls-connection-reuse.yml) | defect | P1 | — | Reuse verified mTLS connections between Server and Runtime instead of one handshake per request |
| [V407-014](v407-014-placement-per-poll-cost.yml) | defect | P2 | V405-002 | Bound per-poll placement work for a capacity-waiting model Stage |
| [V407-015](v407-015-scheduler-stage-end-cost.yml) | defect | P2 | — | Stop re-decrypting Worker credentials and rebuilding metrics at Stage end, and scope cancel scans to pending work |
| [V407-016](v407-016-scheduler-idle-and-release-pacing.yml) | defect | P3 | — | Probe idle lanes once per class and release stuck terminal Stages concurrently |
| [V407-017](v407-017-startup-lease-tolerance.yml) | defect | P2 | — | Bound the single-active Server's startup lease dial and first acquire by the lease tolerance |

### Planners and the Worker model boundary (`planner-worker`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-004](v371-004-worker-completion-exact-integers.yml) | defect | P3 | — | Bound WorkerCompletion integers to the range A2A carries exactly |
| [V371-005](v371-005-worker-state-read-cost-guard.yml) | chore | P3 | — | Guard the Worker State read cost with a deterministic bound and re-measure it |
| [V371-006](v371-006-workspace-observation-descriptor.yml) | chore | P3 | — | Take the Worker tools that observe the workspace from the shared Toolset descriptor |
| [V407-008](v407-008-planner-result-verification-classes.yml) | defect | P1 | — | Classify Planner result verification failures once: contract violations are permanent, storage faults retryable |
| [V407-009](v407-009-worker-instruction-provider.yml) | defect | P1 | — | Pass Worker and finalizer instructions to ADK through a provider and reject bare braces at validation |
| [V407-010](v407-010-worker-local-deadline.yml) | defect | P2 | — | Enforce the Stage deadline locally for every Worker invocation, not only Audit completions |
| [V407-011](v407-011-worker-state-publication-cost.yml) | defect | P1 | — | Make Worker State publication cost proportional to the change, not to the invocation history |
| [V407-012](v407-012-runtime-core-duplicates.yml) | chore | P3 | — | Fold duplicated Runtime core helpers and move small prepare-path disk writes off the loop |

### Runtime labels, configuration and credentials (`runtime-config`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-006](v372-006-runtime-label-set-rule.yml) | chore | P3 | — | Give the Runtime label set rule one Go definition and hold the UI and the Runtime to it |
| [V407-018](v407-018-credential-delete-lock-scope.yml) | defect | P1 | — | Release the credential lifecycle barrier before the Gateway delete call and bound startup recovery's blast radius |
| [V407-019](v407-019-credential-recovery-fence-and-classifier.yml) | defect | P2 | — | Clear the credential recovery fence from durable state and classify credential errors in one place |

### Shared Memory tools (`memory`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V407-036](v407-036-memory-concurrent-reads.yml) | defect | P3 | — | Read Memory notes with bounded concurrency instead of serially under the lock |

### Run workspaces and project filesystem (`workspaces`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V407-034](v407-034-workspace-listing-and-export-cost.yml) | defect | P2 | — | List and glob the workspace without reading file contents, compare exports by metadata, and narrow filesystem tool exception handling |

### Code analysis and taint annotations (`code-analysis`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V407-037](v407-037-code-analysis-hh-and-errors.yml) | defect | P2 | — | Parse .hh headers as C++ and stop masking parser defects as incomplete coverage |

### HTTP and Caido tools (`http-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V386-004](v386-004-caido-catalog-cardinality.yml) | defect | P2 | — | Keep Caido catalog selection usable above one hundred entries |
| [V407-032](v407-032-http-caido-loop-and-error-classes.yml) | defect | P2 | — | Decode and redact Caido payloads off the loop, and let only transport errors become retryable HTTP and Caido tool errors |

### Scanner tool Workers and scan planning (`scan-tools`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V55-009](v55-009-scan-candidate-ranking.yml) | feature | P2 | — | Optionally rank existing scan candidates with an LLM |
| [V406-007](v406-007-web-scan-release-gate.yml) | chore | P2 | V406-004, V406-005, V406-006 | Gate the complete web scan and rule sweep journeys |
| [V407-006](v407-006-scanplan-prepare-once.yml) | defect | P2 | — | Parse and digest an Audit OpenAPI source once per settings document, not once per operation |
| [V407-007](v407-007-scan-output-off-loop.yml) | defect | P3 | — | Parse scanner output and prepare scan inputs off the Runtime event loop |

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
| [V407-033](v407-033-findings-toolsets-cost.yml) | defect | P2 | — | Bound finding toolset cost: encode audit results off the collector lock, cache the exact source ZIP per allocation, and page list_findings linearly |

### Workflow catalog and document workflows (`workflows`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-013](v372-013-planner-facts-one-source.yml) | chore | P3 | — | Take Planner refs and facts from the descriptor registry in Go, the composition and Studio |
| [V372-014](v372-014-execution-config-patch-strict-decode.yml) | chore | P3 | — | Decode executionConfig patch members with strictjson |
| [V372-015](v372-015-config-split-leftovers.yml) | chore | P3 | — | Stop locals from shadowing the resolved package and drop config's copy of clone.Map |
| [V407-031](v407-031-document-session-off-loop.yml) | defect | P1 | — | Run OpenAPI and LikeC4 document parsing, validation and serialization off the event loop and validate once per mutation |

### Projects, artifacts and lifecycle controls (`projects-artifacts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V407-035](v407-035-artifact-reader-paging-cost.yml) | defect | P3 | — | Page large text and Run artifacts from a cached decoded view instead of re-decoding the payload per page |

### Audits (`audits`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-009](v371-009-audit-list-preparation-batch.yml) | defect | P3 | — | Read the preparation state of a whole Audit page in bounded statements |
| [V405-003](v405-003-audit-placement-propagation.yml) | feature | P2 | V405-002 | Pin Audit placement selection and propagate it to every child Run |
| [V407-001](v407-001-audit-collection-failure-classes.yml) | defect | P1 | — | Classify Audit collection failures: transient reads stay retryable, pinned-data decoding is permanent |
| [V407-002](v407-002-audit-collection-store-work.yml) | defect | P2 | — | Bound Audit result collection store work per tick: batch receipt resolution, metadata-only evidence reads, one retention transaction |
| [V407-003](v407-003-audit-controller-batch-per-pass.yml) | defect | P2 | — | Settle every terminal, cancelled or deleted child Run of an Audit within one controller pass |
| [V407-004](v407-004-audit-list-summary-projection.yml) | defect | P2 | — | Serve Audit list pages from a summary projection instead of a per-row detail read model |
| [V407-005](v407-005-audit-fail-closed-and-dead-code.yml) | defect | P3 | — | Fail closed on preparation Workflow comparison and remove unreachable Audit domain code |

### Evaluations (`evals`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V40-003](v40-003-agent-instruction-eval-decision.yml) | feature | P1 | V40-002 | Evaluate instruction candidates and record evidence-based adoption decisions |
| [V407-028](v407-028-eval-member-bulk-freeze.yml) | defect | P2 | — | Freeze Eval members with one set-based insert instead of one statement per member |
| [V407-029](v407-029-eval-reads-plan-metadata.yml) | defect | P2 | — | Read Eval plan metadata instead of the frozen document and stop re-validating the plan on every GET |
| [V407-030](v407-030-eval-preparing-lifecycle-and-usage.yml) | defect | P3 | — | Let a preparing Eval be cancelled or fall back after bounded attempts, and make usage totals order-independent |

### Web UI (`ui`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V403-003](v403-003-openapi-preview-at-1mib.yml) | defect | P3 | — | Keep a 1 MiB OpenAPI preview from blocking the page |
| [V405-004](v405-004-placement-label-picker-ui.yml) | feature | P2 | V405-002, V405-003 | Select Runtime placement labels in Workflow and Audit launch forms |
| [V406-005](v406-005-web-scan-settings-ui.yml) | feature | P2 | — | Launch a web scan with rule settings and owner rulepack selection |
| [V406-006](v406-006-scope-sweep-ui-cli.yml) | feature | P2 | V406-004, V406-005 | Launch and follow durable rule sweeps from UI and CLI |

### Operations, authentication, CLI and deployment (`operations`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V407-020](v407-020-telemetry-trim-and-history.yml) | defect | P2 | — | Trim telemetry reports by prefix sum, ring-buffer the operations history and keep one resource sentinel |

### Public API and wire contracts (`api-contracts`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V371-016](v371-016-idempotency-key-grammar.yml) | defect | P3 | — | Give the Idempotency-Key grammar one Go definition |
| [V372-020](v372-020-error-code-vocabularies.yml) | chore | P3 | — | Hold every vocabulary that carries error code names to the registry |
| [V372-025](v372-025-typed-openapi-enums.yml) | chore | P3 | — | Give every enum of the public OpenAPI document a type |
| [V372-026](v372-026-request-digests-in-services.yml) | chore | P3 | — | Move the Project, Audit and review request digests out of the public handlers |
| [V407-021](v407-021-public-api-grammars-one-definition.yml) | defect | P1 | — | Give every public API grammar and bound one definition: Idempotency-Key, body limits, media-type status, string units, page sizes |
| [V407-022](v407-022-reject-head-centrally.yml) | defect | P2 | — | Reject HEAD once at route registration for every GET operation |
| [V407-023](v407-023-mutation-handler-preamble.yml) | defect | P3 | — | Share one mutation preamble and one bounded JSON decoder across public handlers |
| [V407-024](v407-024-cross-language-limits-and-grammars.yml) | defect | P1 | V405-002 | Align Runtime limits and grammars with the Server through shared fixtures: openapi@1 bounds, finding identifier grammar, summary bytes, placement-label SQL |

### PostgreSQL persistence and schema (`persistence`)

| Task | Kind | Priority | Waiting for | Title |
| --- | --- | --- | --- | --- |
| [V372-008](v372-008-upward-table-reads.yml) | chore | P3 | — | Remove the table reads that point up the layer stack or name who owns each one |
| [V372-009](v372-009-ownership-check-blind-spots.yml) | chore | P3 | — | Close the blind spots of the table ownership check |
| [V372-010](v372-010-erd-usage-from-ownership-scan.yml) | chore | P3 | — | Take the ERD's code-usage lists from the ownership scan and document the new-table workflow |
| [V407-025](v407-025-persistence-indexes-and-triggers.yml) | defect | P2 | — | Index Project drain and held-Audit credential queries and scope stage_executions triggers to the columns they guard |
| [V407-026](v407-026-store-error-classes-and-event-validation.yml) | defect | P3 | — | Classify a missing Project on Run insert as not-found, treat the artifact-write key race as a conflict, and validate immutable Run events once |
| [V407-027](v407-027-shared-controller-claim-loop.yml) | chore | P3 | — | Share one controller claim loop with an injected clock between the Audit controller and the Eval coordinator |
| [V408-002](v408-002-shared-postgres-test-schema.yml) | chore | P2 | V408-001 | Apply migrations once per package: one shared PostgreSQL test helper with per-test truncation |

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
| [V407-038](v407-038-runtime-dead-surface.yml) | chore | P3 | — | Remove unselected Runtime tool surface and the edit_files to filesystem import chain |
| [V407-039](v407-039-close-server-runtime-review.yml) | chore | P3 | V407-001, V407-002, V407-003, V407-004, V407-005, V407-006, V407-007, V407-008, V407-009, V407-010, V407-011, V407-012, V407-013, V407-014, V407-015, V407-016, V407-017, V407-018, V407-019, V407-020, V407-021, V407-022, V407-023, V407-024, V407-025, V407-026, V407-027, V407-028, V407-029, V407-030, V407-031, V407-032, V407-033, V407-034, V407-035, V407-036, V407-037, V407-038 | Close the 2026-10-23 Server and Runtime review: every finding fixed or declined with a recorded reason |
| [V408-001](v408-001-gate-duration-baseline.yml) | chore | P2 | — | Record the baseline duration of every release stage and local test target |
| [V408-003](v408-003-parallel-race-stages.yml) | chore | P2 | V408-002 | Run race stages with package parallelism and mark schema-isolated tests parallel |
| [V408-004](v408-004-e2e-build-once.yml) | chore | P2 | V408-001 | Build the Server binary and the test PKI once per e2e package and right-size per-test budgets |
| [V408-005](v408-005-family-guard-gaps.yml) | chore | P2 | — | Close the family-target guard gaps and run the Runtime suite once with warnings as errors |
| [V408-006](v408-006-go-test-cache.yml) | chore | P3 | — | Let pure Go packages use the test cache: -count=1 only for database and process tests |
| [V408-007](v408-007-gate-duration-after.yml) | chore | P3 | V408-001, V408-002, V408-003, V408-004, V408-005, V408-006 | Record the release gate durations after the optimisations and close the test gate cost review |

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
| Run execution core (`execution-core`) | 10 | 41 | 55 | 11 | 72 |
| Planners and the Worker model boundary (`planner-worker`) | 8 | 27 | 41 | 15 | 57 |
| Runtime labels, configuration and credentials (`runtime-config`) | 12 | 25 | 14 | 3 | 39 |
| Shared Memory tools (`memory`) | 1 | 6 | 10 | 1 | 12 |
| Agent Skills (`agent-skills`) | 0 | 11 | 2 | 1 | 1 |
| Run workspaces and project filesystem (`workspaces`) | 1 | 16 | 13 | 4 | 16 |
| Code analysis and taint annotations (`code-analysis`) | 1 | 14 | 14 | 2 | 19 |
| HTTP and Caido tools (`http-tools`) | 2 | 11 | 15 | 4 | 16 |
| Scanner tool Workers and scan planning (`scan-tools`) | 5 | 14 | 6 | 4 | 9 |
| Findings tools and collections (`findings`) | 8 | 14 | 6 | 6 | 9 |
| Workflow catalog and document workflows (`workflows`) | 4 | 14 | 14 | 12 | 15 |
| Projects, artifacts and lifecycle controls (`projects-artifacts`) | 1 | 30 | 34 | 5 | 36 |
| Audits (`audits`) | 23 | 52 | 75 | 17 | 87 |
| Evaluations (`evals`) | 5 | 21 | 35 | 5 | 39 |
| Podman sandbox (`sandbox`) | 0 | 9 | 2 | 1 | 3 |
| Web UI (`ui`) | 4 | 25 | 58 | 12 | 53 |
| Operations, authentication, CLI and deployment (`operations`) | 1 | 15 | 34 | 5 | 10 |
| Public API and wire contracts (`api-contracts`) | 8 | 1 | 34 | 20 | 11 |
| PostgreSQL persistence and schema (`persistence`) | 8 | 0 | 22 | 35 | 9 |
| Quality, CI and process (`quality`) | 19 | 1 | 21 | 87 | 20 |
