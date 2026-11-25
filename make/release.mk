# The release gate runs in stages, cheapest first, so a lint, unit or UI
# failure surfaces within minutes instead of after the long suites. CI runs
# selected stages for PRs and every stage for release tags or manual dispatch.
# make release-verify runs them in this order; add -k to see every failure.
# Family targets keep their focused Go suites when invoked directly. Inside a
# stage RELEASE_CONSOLIDATED replaces them with the unions below, with each
# selected test run once.

RELEASE_STAGES := release-verify-lint release-verify-unit release-verify-ui \
	release-verify-families release-verify-browser-a release-verify-browser-b release-verify-race \
	release-verify-race-discovered release-verify-integration \
	release-verify-process-a release-verify-process-b release-verify-process-c

.PHONY: $(RELEASE_STAGES) test-release-go-race test-release-go-race-discovered \
	test-release-process-e2e test-release-process-e2e-a test-release-process-e2e-b test-release-process-e2e-c \
	test-release-integration test-release-non-race \
	test-release-ui-stack-a test-release-ui-stack-b

release-verify $(RELEASE_STAGES): RELEASE_CONSOLIDATED := 1
release-verify: $(RELEASE_STAGES)

# The first three stages are exactly make verify.
release-verify-lint: lint build test-config

release-verify-unit: test test-release-config-fixtures test-release-shared-fixtures

release-verify-ui: ui-verify

release-verify-families: ui-install verify-public-api-postgres \
	test-runtime-configuration-e2e test-run-metadata-labels-e2e \
	test-shared-memory-hardening test-agent-skills-hardening \
	test-http-caido-hardening test-code-analysis-e2e test-taint-annotations-e2e \
	test-worker-observations-e2e test-worker-summarizer-e2e \
	test-worker-session-modes-e2e test-worker-compaction-e2e test-project-workspaces-release \
	test-lifecycle-controls-release test-scheduler-concurrency-e2e \
	test-audit-program-library-e2e test-audit-completion-e2e test-audit-basic-e2e \
	test-audit-iterative-postgres test-audit-iterative-e2e \
	test-findings-e2e test-engagement-http-process test-performance-metrics test-nuclei-web-e2e \
	test-artifact-history-postgres test-artifact-history-e2e test-project-sitemap-postgres test-project-sitemap-tools-e2e

release-verify-browser-a: ui-browser-mocked test-release-ui-stack-a test-audit-research-ui

release-verify-browser-b: test-release-ui-stack-b

release-verify-race: test-release-go-race

release-verify-race-discovered: test-release-go-race-discovered

release-verify-integration: test-release-integration test-release-non-race test-artifact-filesystem

release-verify-process-a: test-release-process-e2e-a

release-verify-process-b: test-release-process-e2e-b

release-verify-process-c: test-release-process-e2e-c

RELEASE_RACE_PATTERNS := \
	./cmd/contractor-skill/... \
	./internal/agentskills/... ./internal/app/... ./internal/artifacts/policy/... \
	./internal/artifacts/... ./internal/config/... ./internal/contracts/... \
	./internal/controlplane/... ./internal/credentials/... ./internal/httpapi/... \
	./internal/memory/... ./internal/mtls/... ./internal/performance \
	./internal/persistence/postgres ./internal/planner/... ./internal/profiling \
	./internal/projectlifecycle ./internal/projectstore/... ./internal/runstore/... \
	./internal/runtimeconfig/... ./internal/scheduler/... ./internal/telemetry/... \
	./tests/integration/lease ./tools/performancebench

# Keep database-backed package binaries bounded independently of the runner's
# CPU count. Existing tests keep their own scheduling and isolated schemas.
# Use RACE_PACKAGE_PARALLELISM=1 for serial diagnosis; raise it only after
# measuring database connections and memory on the runner.
RACE_PACKAGE_PARALLELISM ?= 2

test-release-go-race: require-database runtime-venv
	go test -p $(RACE_PACKAGE_PARALLELISM) -race -count=1 -timeout=20m $$(go list $(RELEASE_RACE_PATTERNS) | sort -u)

# Every other package with tests, discovered so new packages are raced too;
# exceptions and their reasons live in scripts/release_race_packages.py.
test-release-go-race-discovered: require-database runtime-venv
	go test -p $(RACE_PACKAGE_PARALLELISM) -race -count=1 -timeout=30m $$(python3 scripts/release_race_packages.py --exclude $(RELEASE_RACE_PATTERNS))

# Discover exact names from integration-tagged files. This adds complete tagged
# coverage, including the former Audit completion gate selection.
RELEASE_INTEGRATION_PACKAGES := $(shell python3 scripts/release_integration_tests.py --packages)
RELEASE_INTEGRATION_TESTS := $(shell python3 scripts/release_integration_tests.py --regex)

test-release-integration: require-database runtime-venv
	python3 scripts/release_test_reports.py run --report "$(RELEASE_REPORTS)/go-integration.jsonl" --timeout-seconds 2640 -- go test -json -p $(RACE_PACKAGE_PARALLELISM) -race -tags=integration -v -count=1 -timeout=20m $(RELEASE_INTEGRATION_PACKAGES) -run '$(RELEASE_INTEGRATION_TESTS)'
	python3 scripts/release_test_reports.py verify-audit --report "$(RELEASE_REPORTS)/go-integration.jsonl"
	python3 scripts/release_test_reports.py verify-iterative-postgres --report "$(RELEASE_REPORTS)/go-integration.jsonl"

# Packages with race-constrained test files relax a budget under the race
# detector (findingintake doubles its 10-second production collection
# deadline). This pass runs them without -race, enforcing the real budget.
RELEASE_NON_RACE_PACKAGES := ./internal/findingintake

test-release-non-race: require-database
	go test -tags=integration -count=1 -timeout=10m $(RELEASE_NON_RACE_PACKAGES)

# Shards are balanced from uncached CI test durations; all three run on
# separate runners. Keep the full local test-e2e command below.
RELEASE_ITERATIVE_TESTS := $(shell python3 scripts/release_test_reports.py iterative-pattern)
RELEASE_E2E_TESTS_A := TestRuntimeManagedMaterialVersionsAcrossProcesses|TestRuntimeManagedSitemapEditsAcrossProcesses|TestRuntimeManagedArtifactWritesAcrossProcesses|TestArtifactImagesAcrossProcesses|TestArtifactFilePublicationAcrossProcesses|TestWorkflowMaterialSetsAcrossProductionProcesses|TestAuditHypothesisContinuationAcrossProductionProcesses|TestAuditFixedHypothesesAcrossProductionProcesses|TestAuditIncrementalEvidenceAcrossProcesses|TestAuditPartialJudgmentsAcrossProcesses|TestAuditCompactionPreservesTrustedCompletionAcrossProcesses|TestAuditCompactionLeaseLossAcrossProcesses|TestAuditWorkspaceAcrossProductionProcesses|TestMemoryMVPProcesses|TestHTTPAndCaidoAcrossHeterogeneousRuntimeProcesses|TestGatewayRecoveryKeepsThreeQueuedRunsAcrossProcesses|TestLocalGoToPythonArtifactCopy|TestArtifactHistoryAcrossProductionProcesses|TestProjectSitemapToolsAcrossProductionProcesses|TestProjectSitemapHTTPCollectionAcrossProductionProcesses|TestProjectSitemapReportImportsAcrossProductionProcesses|TestProjectSitemapServerOperationsAcrossProductionProcesses|TestOpenAPIPreparedWorkServerOperation|$(RELEASE_ITERATIVE_TESTS)
RELEASE_HTTP_TESTS := $(shell python3 scripts/test-engagement-http-process.py --process-names)
RELEASE_E2E_TESTS_B := $(RELEASE_HTTP_TESTS)|TestWorkerSummarizerProductionBoundaries|TestRoutingAndEscalationProductionBoundaries|TestLabelDrivenRuntimeConfigurationAcrossProcesses|TestRuntimeAdapterSelectionAllowsAdditionalCapabilities|TestRunMetadataLabelsAcrossProcesses|TestSchedulerConcurrencyAcrossProductionProcesses|TestTaintAnnotationsAcrossRealRuntimeProcess|TestAuditProgramCatalogReplacementRestartsServer
RELEASE_FINDINGS_TESTS := $(shell python3 scripts/release_test_reports.py findings-pattern)
RELEASE_E2E_TESTS_C := TestRuntimeGuidanceAcrossProductionProcesses|TestNucleiWebWithDeterministicScannerAcrossProcesses|TestAgentSkillsMVPProcesses|TestProjectWorkspaceLifecycleAcrossProductionProcesses|TestProductionMemoryTemplatesAcrossProcesses|TestCodeAnalysisAcrossHeterogeneousRuntimeProcesses|TestWorkerSessionModesAcrossProductionProcesses|TestWorkerCompactionContinuesIsolatedWorker|TestHeterogeneousRuntimeCapabilityPlacement|TestGatewayRecoveryCancellationAndPermanentErrorAcrossProcesses|$(RELEASE_FINDINGS_TESTS)|TestRuntimePlacementAcrossProductionProcesses|TestRuntimePlacementScannerFixture|TestRuntimeAllocationSlotsAcrossProductionProcesses
RELEASE_E2E_TESTS := $(RELEASE_E2E_TESTS_A)|$(RELEASE_E2E_TESTS_B)|$(RELEASE_E2E_TESTS_C)

test-release-process-e2e: require-database runtime-venv
	python3 scripts/release_test_reports.py run --report "$(RELEASE_REPORTS)/go-process.jsonl" --timeout-seconds 4800 -- go test -json -tags=e2e -count=1 -timeout=75m ./tests/e2e -run '^($(RELEASE_E2E_TESTS))$$'
	python3 scripts/release_test_reports.py verify-compaction-process --report "$(RELEASE_REPORTS)/go-process.jsonl"
	python3 scripts/release_test_reports.py verify-findings --report "$(RELEASE_REPORTS)/go-process.jsonl"
	python3 scripts/test-audit-basic-e2e.py --verify-report "$(RELEASE_REPORTS)/go-process.jsonl"
	python3 scripts/release_test_reports.py verify-iterative-process --report "$(RELEASE_REPORTS)/go-process.jsonl"
	python3 scripts/test-audit-composition-e2e.py --case-set workspace --verify-only "$(RELEASE_REPORTS)/go-process.jsonl" --boundary process
	python3 scripts/test-engagement-http-process.py --verify-process-report "$(RELEASE_REPORTS)/go-process.jsonl"

test-release-process-e2e-a: require-database runtime-venv
	python3 scripts/release_test_reports.py run --report "$(RELEASE_REPORTS)/go-process-a.jsonl" --timeout-seconds 3300 -- go test -json -tags=e2e -count=1 -timeout=50m ./tests/e2e -run '^($(RELEASE_E2E_TESTS_A))$$'
	python3 scripts/release_test_reports.py verify-compaction-process --report "$(RELEASE_REPORTS)/go-process-a.jsonl" --scope '^($(RELEASE_E2E_TESTS_A))$$'
	python3 scripts/test-audit-basic-e2e.py --verify-report "$(RELEASE_REPORTS)/go-process-a.jsonl"
	python3 scripts/release_test_reports.py verify-iterative-process --report "$(RELEASE_REPORTS)/go-process-a.jsonl"
	python3 scripts/test-audit-composition-e2e.py --case-set workspace --verify-only "$(RELEASE_REPORTS)/go-process-a.jsonl" --boundary process

test-release-process-e2e-b: require-database runtime-venv
	python3 scripts/release_test_reports.py run --report "$(RELEASE_REPORTS)/go-process-b.jsonl" --timeout-seconds 3300 -- go test -json -tags=e2e -count=1 -timeout=50m ./tests/e2e -run '^($(RELEASE_E2E_TESTS_B))$$'
	python3 scripts/test-engagement-http-process.py --verify-process-report "$(RELEASE_REPORTS)/go-process-b.jsonl"

test-release-process-e2e-c: require-database runtime-venv
	python3 scripts/release_test_reports.py run --report "$(RELEASE_REPORTS)/go-process-c.jsonl" -- go test -json -tags=e2e -count=1 -timeout=50m ./tests/e2e -run '^($(RELEASE_E2E_TESTS_C))$$'
	python3 scripts/release_test_reports.py verify-compaction-process --report "$(RELEASE_REPORTS)/go-process-c.jsonl" --scope '^($(RELEASE_E2E_TESTS_C))$$'
	python3 scripts/release_test_reports.py verify-findings --report "$(RELEASE_REPORTS)/go-process-c.jsonl"

# The two browser stages run independently in CI. Each tagged stack test runs
# once. Keep the small helper tests in the first shard too.
RELEASE_UI_STACK_TESTS_A := TestUIProcessDatabaseIsolation|TestBrowserUIBuildSurvivesSiblingRebuild|TestBrowserOperationsStack|TestBrowserScanWidgetsStack|TestManagedEvalsNativeStack|TestBrowserReportRequiresExecutedSelection|TestManagedEvalGatewayHasIndependentHistories|TestManagedEvalAssertionsUseCurrentDomainContract|TestManagedEvalGatewayUsesCurrentCheckerContract|TestUIStackConfigurationClosure|TestModelGatewayFindsNamedInputAfterParameterBlock|TestBrowserGatewayGeneratedModelRoutes
test-release-ui-stack-a: ui-install ui-browser-install require-database runtime-venv
	$(UI_STACK_TEST_FLAGS) ./tests/ui-stack -run '^($(RELEASE_UI_STACK_TESTS_A))$$'

test-release-ui-stack-b: ui-install ui-browser-install require-database runtime-venv
	$(UI_STACK_TEST_FLAGS) ./tests/ui-stack -run '^TestManagedEvalsExternalStack$$'
