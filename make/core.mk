# The base suites: Go and Runtime unit tests, the wire contracts both
# languages share, and the Postgres-backed integration tests.

.PHONY: test-go test-runtime test-runtime-hardening test-hardening-matrices \
	test-contracts verify-wire-contracts test-wire-cross-language \
	test-config test-postgres test-runtime-config-postgres \
	test-runtime-credentials test-litellm-contract test-mtls \
	test-control-integration test-artifact-integration \
	test-lease-integration test-streamline test-faults test-e2e \
	test-capability-e2e

test-go: test-hardening-matrices runtime-venv
	go test $$(go list ./... | grep -v '/tests/e2e$$')

test-runtime:
	cd runtime && uv run pytest

test-runtime-hardening:
	cd runtime && uv run pytest -W error tests

test-hardening-matrices: runtime-venv
	go test -count=1 ./tests/e2e

test-contracts:
	go test ./internal/contracts/...
	cd runtime && uv run pytest tests/test_contracts.py

verify-wire-contracts:
	go test ./internal/contracts/...
	cd runtime && uv run pytest tests/test_contracts.py tests/test_performance_contracts.py

test-wire-cross-language:
	go test ./internal/contracts/... ./internal/config/... -run 'Golden|SharedPython|ResolvedSkills'
	cd runtime && uv run pytest tests/test_contracts.py tests/test_performance_contracts.py -k 'golden or digest or resolved_skills'

test-config:
	go test ./internal/config/...
	go run ./cmd/contractor-server config validate --root ./configs

test-postgres: require-database
	go test -count=1 ./internal/persistence/postgres ./internal/credentials ./internal/runtimeconfig ./internal/runstore ./internal/artifacts ./internal/httpapi/public ./internal/planner/session ./internal/scheduler ./internal/telemetry

test-runtime-config-postgres: require-database
	go test -count=1 ./internal/runtimeconfig ./internal/persistence/postgres

test-runtime-credentials: require-database
	go test -count=1 ./internal/credentials ./internal/runtimeconfig ./internal/persistence/postgres

test-litellm-contract:
	deploy/litellm/test-contract.sh

test-mtls:
	go test -count=1 ./internal/mtls/... ./cmd/contractor-pki/...
	cd runtime && uv run pytest tests/test_mtls.py

test-control-integration:
	go test -tags=integration -count=1 ./internal/controlplane -run TestCrossLanguageMTLSAllocationLifecycle

test-artifact-integration:
	go test -tags=integration -count=1 ./internal/httpapi/privateartifacts -run TestCrossLanguagePrivateArtifactLifecycle

test-lease-integration:
	go test -race -count=1 ./tests/integration/lease
	go test -race -count=1 ./internal/controlplane/... ./internal/scheduler/... -run 'Lease|Reconcile|Partition'
	cd runtime && uv run pytest tests/test_lease_watchdog.py

test-streamline: require-database
	go test -count=1 ./internal/planner/streamline ./internal/planner/session ./tests/integration/streamline

test-faults: require-database
	go test -race -count=1 ./tests/faults ./tests/integration/lease ./internal/requestid ./internal/controlplane ./internal/httpapi/privateartifacts ./internal/mtls ./internal/config ./internal/planner/...
	CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -race -count=1 ./internal/artifacts ./internal/httpapi/public ./internal/runstore ./internal/scheduler ./internal/telemetry
	cd runtime && uv run pytest -W error tests

test-e2e: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=28m ./tests/e2e -run '^(TestLocalGoToPythonArtifactCopy|TestRoutingAndEscalationProductionBoundaries|TestHeterogeneousRuntimeCapabilityPlacement|TestLabelDrivenRuntimeConfigurationAcrossProcesses|TestRunMetadataLabelsAcrossProcesses|TestSharedMemoryMVPProcesses|TestHTTPAndCaidoAcrossHeterogeneousRuntimeProcesses|TestCodeAnalysisAcrossHeterogeneousRuntimeProcesses|TestTaintAnnotationsAcrossRealRuntimeProcess|TestWorkerSummarizerProductionBoundaries|TestWorkerSessionModesAcrossProductionProcesses|TestProjectWorkspaceLifecycleAcrossProductionProcesses|TestSchedulerConcurrencyAcrossProductionProcesses)$$'

test-capability-e2e: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=3m ./tests/e2e -run '^TestHeterogeneousRuntimeCapabilityPlacement$$'
