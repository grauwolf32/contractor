# The base suites: Go and Runtime unit tests, the wire contracts both
# languages share, and the Postgres-backed integration tests.

.PHONY: test-go test-runtime test-runtime-hardening test-hardening-matrices \
	test-contracts verify-wire-contracts test-wire-cross-language \
	test-config test-postgres test-runtime-config-postgres \
	test-runtime-credentials test-litellm-contract test-mtls \
	test-control-integration test-artifact-integration test-backup-restore \
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
	go test -tags=e2e -v -count=1 ./tests/e2e -run '^(TestCodeAnalysisE2EConfigurationLoads|TestProductionMemoryConfigurationStaging|TestProjectWorkerBudgetMatchesPinnedPolicy|TestDomainGatewayFindsNamedInputAfterParameterBlock|TestDomainGatewayScriptedModelFailureAdvancesWithoutFixtureFailure|TestRuntimeWorkRootEmptyAllowsPersistentOwnerLock)$$'

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

test-backup-restore: require-database
	@command -v pg_dump >/dev/null && command -v pg_restore >/dev/null || (echo "pg_dump and pg_restore are required" >&2; exit 1)
	go test -tags=integration -count=1 -timeout=10m ./tests/integration/restore -run '^TestBackupRestorePreservesExactArtifactsAndCAS$$'

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

# The same process pass as release-verify's process stage.
test-e2e: test-release-process-e2e

test-capability-e2e: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=3m ./tests/e2e -run '^TestHeterogeneousRuntimeCapabilityPlacement$$'
