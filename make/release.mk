# The release gate runs in stages, cheapest first, so a lint, unit or UI
# failure surfaces within minutes instead of after the long suites. CI runs
# every stage as its own job (.github/workflows/ci.yml) and make
# release-verify runs them in this order; add -k to see every failing stage.
# Family targets keep their focused Go suites when invoked directly. Inside a
# stage RELEASE_CONSOLIDATED replaces them with the unions below, run once.

RELEASE_STAGES := release-verify-lint release-verify-unit release-verify-ui \
	release-verify-families release-verify-browser release-verify-race \
	release-verify-integration release-verify-process

.PHONY: $(RELEASE_STAGES) test-release-go-race test-release-process-e2e \
	test-release-integration test-release-non-race test-release-ui-stack \
	test-runtime-dependencies-audit

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
	test-findings-e2e test-performance-metrics test-runtime-dependencies-audit

release-verify-browser: ui-browser-mocked test-release-ui-stack

release-verify-race: test-release-go-race

release-verify-integration: test-release-integration test-release-non-race

release-verify-process: test-release-process-e2e

# Audit exactly the frozen production Runtime graph. The scanner runs outside
# the shipped Runtime environment and is pinned for repeatable CI behavior.
test-runtime-dependencies-audit:
	python3 scripts/audit_runtime_dependencies.py

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

RELEASE_E2E_TESTS := TestAgentSkillsMVPProcesses|TestAuditProgramCatalogReplacementRestartsServer|TestAuditProgramsAcrossProductionProcesses|TestCodeAnalysisAcrossHeterogeneousRuntimeProcesses|TestGatewayRecoveryCancellationAndPermanentErrorAcrossProcesses|TestGatewayRecoveryKeepsThreeQueuedRunsAcrossProcesses|TestHTTPAndCaidoAcrossHeterogeneousRuntimeProcesses|TestHeterogeneousRuntimeCapabilityPlacement|TestLabelDrivenRuntimeConfigurationAcrossProcesses|TestLocalGoToPythonArtifactCopy|TestProductionMemoryTemplatesAcrossProcesses|TestProjectWorkspaceLifecycleAcrossProductionProcesses|TestRoutingAndEscalationProductionBoundaries|TestRunMetadataLabelsAcrossProcesses|TestSchedulerConcurrencyAcrossProductionProcesses|TestSharedMemoryMVPProcesses|TestTaintAnnotationsAcrossRealRuntimeProcess|TestWorkerSessionModesAcrossProductionProcesses|TestWorkerSummarizerProductionBoundaries

test-release-process-e2e: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=50m ./tests/e2e -run '^($(RELEASE_E2E_TESTS))$$'

# Several families depend on test-ui-stack; the browser stage runs it once.
test-release-ui-stack: ui-install ui-browser-install require-database runtime-venv
	$(UI_STACK_TEST)
