# The release gate reuses each family's non-Go checks but runs Go race and
# process suites once across their union. Target-specific variables propagate
# to prerequisites; direct family invocations keep their original recipes.

.PHONY: test-release-go-race test-release-process-e2e test-release-integration test-runtime-dependencies-audit

release-verify: RELEASE_CONSOLIDATED := 1
release-verify: test-release-go-race test-release-process-e2e test-release-integration test-runtime-dependencies-audit

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

RELEASE_E2E_TESTS := TestAgentSkillsMVPProcesses|TestAuditProgramsAcrossProductionProcesses|TestCodeAnalysisAcrossHeterogeneousRuntimeProcesses|TestHTTPAndCaidoAcrossHeterogeneousRuntimeProcesses|TestHeterogeneousRuntimeCapabilityPlacement|TestLabelDrivenRuntimeConfigurationAcrossProcesses|TestLocalGoToPythonArtifactCopy|TestProductionMemoryTemplatesAcrossProcesses|TestProjectWorkspaceLifecycleAcrossProductionProcesses|TestRoutingAndEscalationProductionBoundaries|TestRunMetadataLabelsAcrossProcesses|TestSchedulerConcurrencyAcrossProductionProcesses|TestSharedMemoryMVPProcesses|TestTaintAnnotationsAcrossRealRuntimeProcess|TestWorkerSessionModesAcrossProductionProcesses|TestWorkerSummarizerProductionBoundaries

test-release-process-e2e: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=50m ./tests/e2e -run '^($(RELEASE_E2E_TESTS))$$'
