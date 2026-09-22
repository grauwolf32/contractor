# Platform families: Runtime configuration and labels, project
# workspaces, lifecycle controls, scheduling and performance.

.PHONY: test-runtime-wire test-runtime-principals test-runtime-label-placement \
	test-runtime-configuration-matrix test-runtime-configuration-hardening \
	test-runtime-configuration-e2e test-run-metadata-labels-matrix \
	test-run-metadata-labels-hardening test-run-metadata-labels-e2e \
	test-runtime-labels-e2e test-project-workspaces-matrix \
	test-local-direct-workspace test-project-workspaces-hardening \
	test-project-workspaces-e2e test-project-workspaces-release \
	test-lifecycle-controls-matrix test-lifecycle-controls-hardening \
	test-lifecycle-controls-e2e test-lifecycle-controls-browser \
	test-lifecycle-controls-release test-scheduler-concurrency-matrix \
	test-scheduler-concurrency-hardening test-scheduler-concurrency-process \
	test-scheduler-concurrency-browser test-scheduler-concurrency-e2e \
	test-performance-matrix test-performance-go test-performance-runtime \
	test-performance-postgres test-performance-browser \
	test-performance-metrics benchmark-performance test-git-artifacts \
	test-artifact-blob-backends

test-runtime-wire:
	go test ./internal/contracts/... ./internal/controlplane/...
	cd runtime && uv run pytest tests/test_contracts.py tests/test_capabilities.py tests/test_settings.py tests/test_state.py tests/test_control_client.py

test-runtime-principals: require-database
	go test -count=1 ./internal/runtimeconfig ./internal/persistence/postgres

test-runtime-label-placement: require-database
	go test -count=1 ./internal/controlplane ./internal/credentials ./internal/runstore ./internal/runtimeconfig

test-runtime-configuration-matrix: test-hardening-matrices

test-runtime-configuration-hardening: test-runtime-configuration-matrix
	go test -race -count=1 ./internal/controlplane/... ./internal/credentials/... ./internal/httpapi/... ./internal/mtls/... ./internal/runstore/... ./internal/runtimeconfig/... ./internal/scheduler/... ./internal/telemetry/... ./tests/integration/lease
	cd runtime && uv run pytest -W error tests/test_adapter_host.py tests/test_http_proxy_adapter.py tests/test_lease_watchdog.py tests/test_otlp_adapter.py

test-runtime-configuration-e2e: test-runtime-configuration-hardening test-runtime-labels-e2e test-ui-stack

test-run-metadata-labels-matrix: test-hardening-matrices

test-run-metadata-labels-hardening: test-run-metadata-labels-matrix
	go test -race -count=1 ./internal/contracts/... ./internal/controlplane/... ./internal/httpapi/public ./internal/persistence/postgres ./internal/runstore ./internal/scheduler/... ./internal/telemetry/...
	cd runtime && uv run pytest -W error tests/test_contracts.py tests/test_app.py tests/test_allocation.py tests/test_adapter_host.py tests/test_otlp_adapter.py
	cd ui && corepack pnpm test --run src/api/run-metadata-labels.test.ts src/api/workflows.test.ts src/run-drafts/idempotency.test.ts src/routes/workflows/workflows.test.tsx src/routes/runs/runs.test.tsx

test-run-metadata-labels-e2e: test-run-metadata-labels-hardening require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=5m ./tests/e2e -run '^TestRunMetadataLabelsAcrossProcesses$$'

test-runtime-labels-e2e: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=4m ./tests/e2e -run '^TestLabelDrivenRuntimeConfigurationAcrossProcesses$$'

test-project-workspaces-matrix: test-hardening-matrices

test-local-direct-workspace:
	go test -count=1 ./tests/e2e -run ProjectWorkspace
	cd runtime && uv run pytest -W error tests/test_projectfs_local_io.py tests/test_projectfs_operation_guard.py tests/test_projectfs_local_direct.py tests/test_local_direct_tool_consumers.py tests/test_local_direct_faults.py tests/test_projectfs_security.py tests/test_projectfs_concurrency.py tests/test_projectfs_storage.py tests/test_projectfs_zip.py tests/test_workspace_provider.py tests/test_workspace_state.py tests/test_projectfs_overlay.py tests/test_projectfs_edit_parity.py tests/test_projectfs_legacy_parity.py tests/test_workspace_auto_export.py tests/test_workspace_process_e2e.py tests/test_allocation.py tests/test_lease_watchdog.py tests/test_filesystem_toolset.py tests/test_edit_files_toolset.py tests/test_filesystem_observations.py tests/test_code_analysis_shallow.py tests/test_code_analysis_graph.py tests/test_taint_annotations.py tests/test_openapi_toolset.py

test-project-workspaces-hardening: test-project-workspaces-matrix require-database
	go test -race -count=1 ./internal/projectstore/... ./internal/artifacts/... ./internal/runstore/... ./internal/scheduler/... ./internal/httpapi/public
	cd runtime && uv run pytest -W error tests/test_http_toolset.py -k project_authorization_is_exact_origin_hidden_and_erased
	cd ui && corepack pnpm test --run src/api/projects.test.ts src/api/project-artifacts.test.ts src/api/queue.test.ts src/routes/projects/projects.test.tsx src/routes/projects/recommendations.test.ts src/routes/queue.test.tsx

test-project-workspaces-e2e: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=6m ./tests/e2e -run '^TestProjectWorkspaceLifecycleAcrossProductionProcesses$$'
	cd runtime && uv run pytest -W error tests/test_workspace_process_e2e.py

test-project-workspaces-release: test-project-workspaces-hardening test-project-workspaces-e2e test-ui-stack

test-lifecycle-controls-matrix: test-hardening-matrices

test-lifecycle-controls-hardening: test-lifecycle-controls-matrix require-database
	CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -race -count=1 ./internal/persistence/postgres ./internal/runstore ./internal/scheduler ./internal/artifacts ./internal/projectlifecycle ./internal/httpapi/public -run '^(TestTerminalRunPurgeMigrationTracksPinOwnershipAndKeepsBypassScoped|TestProjectDeletionMigrationIsDurableClaimedAndFenced|TestOwnerQueueControlMigrationIsDurableAndRevisionProtected|TestPostgresOwnerQueueControlSerializesWithStageAdmission|TestPostgresQueuePauseAllowsTerminalResultCommit|TestSchedulerOwnerQueuePauseDefersInitialAdmissionUntilResume|TestSchedulerOwnerQueuePauseDrainsCurrentStageWithoutAdmittingNext|TestSchedulerOwnerQueuePauseDoesNotBlockCancellation|TestPostgresIntegrationDeletesReleasedTerminalRunWithoutSharedArtifacts|TestPostgresIntegrationRunDeletionParticipatesInCallerTransaction|TestProjectDeletionCancelsDrainsPurgesAndRetainsSharedResources|TestProjectDeleteIsCASDurableIdempotentAndFencesMutations|TestDeleteRunRequiresOwnedReleasedTerminalRun|TestUserArtifactNamespaceExclusionPrecedesPagination)$$'
	cd ui && corepack pnpm test --run src/api/projects.test.ts src/api/project-artifacts.test.ts src/api/queue.test.ts src/routes/projects/projects.test.tsx src/routes/queue.test.tsx src/routes/runs/runs.test.tsx

test-lifecycle-controls-e2e: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=8m ./tests/e2e -run '^(TestLocalGoToPythonArtifactCopy|TestProjectWorkspaceLifecycleAcrossProductionProcesses)$$'

# test-ui-stack serves the built Node UI independently from the Go API and
# includes ui/e2e/lifecycle-controls.spec.ts in the production browser suite.
test-lifecycle-controls-browser: test-ui-stack

test-lifecycle-controls-release: test-lifecycle-controls-hardening test-lifecycle-controls-e2e test-lifecycle-controls-browser

test-scheduler-concurrency-matrix: test-hardening-matrices

test-scheduler-concurrency-hardening: test-scheduler-concurrency-matrix require-database
	CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -tags=integration -race -count=1 ./internal/persistence/postgres ./internal/settingsstore ./internal/runstore ./internal/scheduler ./internal/projectlifecycle -run '^(TestSchedulerSettingsMigrationIsNarrowSeededAndProtected|TestPostgresSchedulerSettingsSeedCASRestartAndFailClosed|TestPostgresSchedulerSupervisorBoundsRealRunClaims|TestPostgresSchedulerSupervisorResizesAndDrainsDurableLanes|TestPostgresClaimRunnableRunRotatesAfterDeferredRelease|TestPostgresQueuePauseAllowsTerminalResultCommit|TestPostgresCancelAndSuccessRaceSerializesOnRunRow|TestProjectDeletionCancelsDrainsPurgesAndRetainsSharedResources|TestSchedulerLeaseLossInterruptsPlannerAndStartsBoundedAbort|TestTerminalReleaseFailureDoesNotBlockClaimPath|TestSchedulerSupervisorFailsClosedAndShutdownReleasesClaim)$$'

test-scheduler-concurrency-process: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=4m ./tests/e2e -run '^TestSchedulerConcurrencyAcrossProductionProcesses$$'

# The browser process builds and serves the Node UI independently from the Go API.
test-scheduler-concurrency-browser: test-ui-stack

test-scheduler-concurrency-e2e: test-scheduler-concurrency-hardening test-scheduler-concurrency-process test-scheduler-concurrency-browser

test-performance-matrix:
	go test -count=1 ./tests/e2e -run '^(TestPerformanceMetricsMatrixIsComplete|TestPerformanceReleaseEvidenceIsComplete)$$'

test-performance-go: test-performance-matrix
	go test -race -count=1 ./internal/performance ./internal/profiling ./internal/app ./internal/controlplane ./internal/httpapi/public ./tools/performancebench -run '^(Test.*Performance.*|Test.*Profiling.*|TestHTTP.*|TestDiagnostics.*|TestDatabaseRates.*|TestDiagnostic.*|TestMinuteValidation.*|TestObserveOwnsFrames|TestRecordBoundsAndHTTPDimensions|TestSampleRejectsUnsafeOrUnboundedRecords|TestHandler.*|TestTimedAndSnapshotCapacityDoesNotQueue|TestCPUAndTraceOutputIsReadableByGoTools|TestListenRejectsUnsafeAddressesAndOccupiedPort|TestServerServesLoopbackAndStopsWithItsContext|TestServerShutdownCancelsAnActiveCapture|TestNewWithListenerRejectsNonLoopbackSocket|TestRuntimeResourcesAreIsolatedFromLifecycleTruth|TestBenchmarkOptionsAndSummaryAreBounded|TestCollectionMeasurementReportsFixedStateAndLogicalIO)$$'

test-performance-runtime: runtime-venv
	cd runtime && uv run pytest -W error tests/test_resource_metrics.py tests/test_performance_contracts.py tests/test_allocation.py -k 'resource or performance'
	go test -tags=integration -count=1 ./internal/controlplane -run '^TestCrossLanguageMTLSAllocationLifecycle$$'

test-performance-postgres: require-database
	CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -race -count=1 ./internal/performance ./internal/telemetry ./internal/controlplane -run '^(TestPostgresDiagnosticsIsolationVisibilityAndOptionalPrivileges|TestPostgresDiagnosticBudgetsAndRecovery|TestPostgresDisabledStatisticsAndIndependentSizeFailure|TestPostgresHistoryIdempotencyTTLBoundsAndPlans|TestAllocationResourceHistoryUsesTerminalIdentityAndPinnedPolicy|TestPlacementPerformanceCollectionPolicyDoesNotFilterCandidates)$$'

# The production browser process proves the current metrics page and the same
# terminal Runtime observation in Run detail and durable post-release history.
test-performance-browser: test-ui-stack

test-performance-metrics: test-performance-go test-performance-runtime test-performance-postgres test-performance-browser

benchmark-performance:
	go test -run '^$$' -bench '^(BenchmarkHTTPInstrumentation|BenchmarkPerformanceCollect|BenchmarkPerformanceMinute)$$' -benchmem -count=5 ./internal/performance
	go run ./tools/performancebench -repetitions=5 -http-requests=50000 -collection-cycles=240 -profile-seconds=1

test-git-artifacts: require-database runtime-venv
	@command -v git >/dev/null || (echo "native git is required for fixtures" >&2; exit 1)
	@command -v podman >/dev/null || (echo "podman is required" >&2; exit 1)
	go test -race -tags=integration -count=1 ./internal/credentials ./internal/gitimport ./internal/artifacts ./internal/httpapi/public ./internal/app ./internal/persistence/postgres
	cd runtime && uv run pytest -q tests/test_source_analysis_toolset.py tests/test_workspace_auto_export.py tests/test_projectfs_overlay.py
	$(MAKE) ui-typecheck ui-lint ui-test verify-public-api
	node ui/server/git-artifacts-gate.mjs
	go test -tags=e2e -count=1 -timeout=12m ./tests/e2e -run '^TestGitArtifactsProductionContainers$$' -v

test-artifact-blob-backends: require-database
	@command -v podman >/dev/null || (echo "podman is required" >&2; exit 1)
	go test -race -tags=integration -count=1 ./internal/artifacts ./internal/app ./internal/persistence/postgres ./internal/httpapi/public ./internal/httpapi/privateartifacts ./internal/auditstore ./internal/findingintake ./internal/projectlifecycle
	CONTRACTOR_TEST_ARTIFACT_BACKEND=filesystem go test -race -count=1 ./internal/artifacts -run 'TestPostgresIntegration(64MiB|InputFork|ProjectInput|ProjectScope|Publishes|DeletesReleased|ArtifactCAS|SkillFork)'
	CONTRACTOR_TEST_ARTIFACT_BACKEND=filesystem go test -race -tags=integration -count=1 ./internal/findingintake
	cd runtime && uv run pytest -q tests/test_artifacts.py tests/test_workspace_auto_export.py tests/test_projectfs_overlay.py
	go test -tags=e2e -count=1 -timeout=8m ./tests/e2e -run '^TestArtifactBlobBackendsContainers$$' -v
