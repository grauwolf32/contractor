# The release gate runs in stages, cheapest first, so a lint, unit or UI
# failure surfaces within minutes instead of after the long suites. CI runs
# every stage as its own job (.github/workflows/ci.yml) and make
# release-verify runs them in this order; add -k to see every failing stage.
# Family targets keep their focused Go suites when invoked directly. Inside a
# stage RELEASE_CONSOLIDATED replaces them with the unions below, with each
# selected test run once.

RELEASE_STAGES := release-verify-lint release-verify-unit release-verify-ui \
	release-verify-families release-verify-browser-a release-verify-browser-b release-verify-race \
	release-verify-race-discovered release-verify-integration \
	release-verify-process-a release-verify-process-b

.PHONY: $(RELEASE_STAGES) test-release-go-race test-release-go-race-discovered \
	test-release-process-e2e test-release-process-e2e-a test-release-process-e2e-b \
	test-release-integration test-release-non-race \
	test-release-ui-stack-a test-release-ui-stack-b

release-verify $(RELEASE_STAGES): RELEASE_CONSOLIDATED := 1
release-verify: $(RELEASE_STAGES)

# The first three stages are exactly make verify.
release-verify-lint: lint build

release-verify-unit: test

release-verify-ui: ui-verify

release-verify-families: ui-install verify-public-api-postgres \
	test-runtime-configuration-e2e test-run-metadata-labels-e2e \
	test-shared-memory-hardening test-agent-skills-hardening \
	test-http-caido-hardening test-code-analysis-e2e test-taint-annotations-e2e \
	test-worker-observations-e2e test-worker-summarizer-e2e \
	test-worker-session-modes-e2e test-project-workspaces-release \
	test-lifecycle-controls-release test-scheduler-concurrency-e2e \
	test-audit-program-library-e2e test-audit-completion-e2e \
	test-findings-e2e test-performance-metrics

release-verify-browser-a: ui-browser-mocked test-release-ui-stack-a

release-verify-browser-b: test-release-ui-stack-b

release-verify-race: test-release-go-race

release-verify-race-discovered: test-release-go-race-discovered

release-verify-integration: test-release-integration test-release-non-race

release-verify-process-a: test-release-process-e2e-a

release-verify-process-b: test-release-process-e2e-b

RELEASE_RACE_PATTERNS := \
	./cmd/contractor-skill/... \
	./internal/agentskills/... ./internal/app/... ./internal/artifactpolicy/... \
	./internal/artifacts/... ./internal/config/... ./internal/contracts/... \
	./internal/controlplane/... ./internal/credentials/... ./internal/httpapi/... \
	./internal/memory/... ./internal/mtls/... ./internal/performance \
	./internal/persistence/postgres ./internal/planner/... ./internal/profiling \
	./internal/projectlifecycle ./internal/projectstore/... ./internal/runstore/... \
	./internal/runtimeconfig/... ./internal/scheduler/... ./internal/telemetry/... \
	./tests/integration/lease ./tools/performancebench

test-release-go-race: require-database runtime-venv
	go test -p 1 -race -count=1 -timeout=20m $$(go list $(RELEASE_RACE_PATTERNS) | sort -u)

# Every other package with tests, discovered so new packages are raced too;
# exceptions and their reasons live in scripts/release_race_packages.py.
test-release-go-race-discovered: require-database runtime-venv
	go test -p 1 -race -count=1 -timeout=30m $$(python3 scripts/release_race_packages.py --exclude $(RELEASE_RACE_PATTERNS))

# Discover exact names from integration-tagged files. This adds complete tagged
# coverage without replacing family commands that also race-test untagged cases.
RELEASE_INTEGRATION_PACKAGES := $(shell python3 scripts/release_integration_tests.py --packages)
RELEASE_INTEGRATION_TESTS := $(shell python3 scripts/release_integration_tests.py --regex)

test-release-integration: require-database runtime-venv
	go test -p 1 -race -tags=integration -v -count=1 -timeout=20m $(RELEASE_INTEGRATION_PACKAGES) -run '$(RELEASE_INTEGRATION_TESTS)'

# Packages with race-constrained test files relax a budget under the race
# detector (findingintake doubles its 10-second production collection
# deadline). This pass runs them without -race, enforcing the real budget.
RELEASE_NON_RACE_PACKAGES := ./internal/findingintake

test-release-non-race: require-database
	go test -tags=integration -count=1 -timeout=10m $(RELEASE_NON_RACE_PACKAGES)

RELEASE_E2E_TESTS_A := TestAgentSkillsMVPProcesses|TestAuditProgramCatalogReplacementRestartsServer|TestAuditProgramsAcrossProductionProcesses|TestCodeAnalysisAcrossHeterogeneousRuntimeProcesses|TestGatewayRecoveryCancellationAndPermanentErrorAcrossProcesses|TestGatewayRecoveryKeepsThreeQueuedRunsAcrossProcesses|TestHTTPAndCaidoAcrossHeterogeneousRuntimeProcesses|TestHeterogeneousRuntimeCapabilityPlacement|TestLabelDrivenRuntimeConfigurationAcrossProcesses|TestLocalGoToPythonArtifactCopy
RELEASE_E2E_TESTS_B := TestProductionMemoryTemplatesAcrossProcesses|TestProjectWorkspaceLifecycleAcrossProductionProcesses|TestRoutingAndEscalationProductionBoundaries|TestRunMetadataLabelsAcrossProcesses|TestSchedulerConcurrencyAcrossProductionProcesses|TestSharedMemoryMVPProcesses|TestTaintAnnotationsAcrossRealRuntimeProcess|TestWorkerSessionModesAcrossProductionProcesses|TestWorkerSummarizerProductionBoundaries
RELEASE_E2E_TESTS := $(RELEASE_E2E_TESTS_A)|$(RELEASE_E2E_TESTS_B)

test-release-process-e2e: require-database runtime-venv
	go test -json -tags=e2e -count=1 -timeout=50m ./tests/e2e -run '^($(RELEASE_E2E_TESTS))$$'

test-release-process-e2e-a: require-database runtime-venv
	go test -json -tags=e2e -count=1 -timeout=50m ./tests/e2e -run '^($(RELEASE_E2E_TESTS_A))$$'

test-release-process-e2e-b: require-database runtime-venv
	go test -json -tags=e2e -count=1 -timeout=50m ./tests/e2e -run '^($(RELEASE_E2E_TESTS_B))$$'

# The two browser stages run independently in CI. Each tagged stack test runs
# once. Keep the small helper tests in the first shard too.
RELEASE_UI_STACK_TESTS_A := TestBrowserOperationsStack|TestManagedEvalsNativeStack|TestBrowserReportRequiresExecutedSelection|TestManagedEvalGatewayHasIndependentHistories|TestUIStackConfigurationClosure|TestModelGatewayFindsNamedInputAfterParameterBlock
test-release-ui-stack-a: ui-install ui-browser-install require-database runtime-venv
	$(UI_STACK_TEST_FLAGS) ./tests/ui-stack -run '^($(RELEASE_UI_STACK_TESTS_A))$$'

test-release-ui-stack-b: ui-install ui-browser-install require-database runtime-venv
	$(UI_STACK_TEST_FLAGS) ./tests/ui-stack -run '^TestManagedEvalsExternalStack$$'
