# The base suites: Go and Runtime unit tests, the wire contracts both
# languages share, and the Postgres-backed integration tests.

.PHONY: test-go test-runtime test-runtime-hardening test-hardening-matrices \
	test-wire-cross-language \
	test-config test-postgres test-runtime-config-postgres \
	test-runtime-credentials test-litellm-contract test-mtls \
	test-control-integration test-artifact-integration test-backup-restore \
	test-lease-integration test-streamline test-faults test-e2e \
	test-capability-e2e

.PHONY: test-release-reports test-release-config-fixtures test-release-shared-fixtures

RELEASE_REPORTS := $(CURDIR)/reports
SHARED_FIXTURE_LOG := $(RELEASE_REPORTS)/shared-fixture-reads.log
CONFIG_FIXTURE_TESTS := TestAuditProgramsE2EConfigurationLoads|TestAuditIterativeFixtureConfigurationLoads|TestCodeAnalysisE2EConfigurationLoads|TestProductionMemoryConfigurationStaging|TestProjectWorkerBudgetMatchesPinnedPolicy|TestDomainGatewayFindsNamedInputAfterParameterBlock|TestDomainGatewayScriptedModelFailureAdvancesWithoutFixtureFailure|TestRuntimeWorkRootEmptyAllowsPersistentOwnerLock

test-release-reports:
	mkdir -p "$(RELEASE_REPORTS)"
	: > "$(SHARED_FIXTURE_LOG)"

# Prerequisites are parsed before a caller's target-specific consolidation
# flag exists. Initialize unconditionally, once, before either language writes.
test-go: test-hardening-matrices runtime-venv test-release-reports
	$(if $(filter 1,$(RELEASE_CONSOLIDATED)),CONTRACTOR_SHARED_FIXTURE_LOG="$(SHARED_FIXTURE_LOG)" python3 scripts/release_test_reports.py run --report "$(RELEASE_REPORTS)/go-unit.jsonl" -- go test -json -count=1 ./...,go test $$(go list ./... | grep -v '/tests/e2e$$'))

test-runtime: test-runtime-hardening

test-runtime-hardening: runtime-venv test-release-reports
	cd runtime && CONTRACTOR_SHARED_FIXTURE_LOG="$(SHARED_FIXTURE_LOG)" uv run pytest -W error tests --junitxml="$(RELEASE_REPORTS)/runtime.xml"
	python3 scripts/release_test_reports.py verify-runtime --report "$(RELEASE_REPORTS)/runtime.xml"

test-release-config-fixtures: runtime-venv
	go test -tags=e2e -v -count=1 ./tests/e2e -run '^($(CONFIG_FIXTURE_TESTS))$$'

test-release-shared-fixtures: RELEASE_CONSOLIDATED := 1
test-release-shared-fixtures: test-go test-runtime-hardening
	python3 scripts/shared_fixture_gate.py --go-report "$(RELEASE_REPORTS)/go-unit.jsonl" --runtime-log "$(SHARED_FIXTURE_LOG)"
	python3 scripts/release_test_reports.py verify-engagement-accounting --report "$(RELEASE_REPORTS)/go-unit.jsonl"

test-hardening-matrices: runtime-venv
	$(call run-family-test,go test -count=1 ./tests/e2e)
	$(call run-family-test,go test -tags=e2e -v -count=1 ./tests/e2e -run '^($(CONFIG_FIXTURE_TESTS))$$')

# The one wire contract gate: the whole contracts packages and Runtime contract
# models, then every TestShared*/test_shared_* test, which are the tests that
# read api/testdata, with each fixture held to the sides its readers manifest
# declares (scripts/shared_fixture_gate.py). Some shared tests use PostgreSQL;
# a skipped one fails the gate.
test-wire-cross-language: require-database runtime-venv
	$(call run-family-test,go test ./internal/contracts/...)
	$(call run-family-test,cd runtime && uv run pytest tests/test_contracts.py tests/test_performance_contracts.py)
	$(call run-family-test,python3 scripts/shared_fixture_gate.py)

test-config:
	go test ./internal/config/... ./internal/configload/...
	go run ./cmd/contractor-server config validate --root ./configs

test-postgres: require-database
	go test -count=1 ./internal/persistence/postgres ./internal/credentials ./internal/agentprincipal ./internal/runtimeconfig ./internal/runstore ./internal/artifacts/... ./internal/httpapi/public ./internal/planner/session ./internal/scheduler ./internal/telemetry

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

# pg_dump refuses a server of a newer major version, so the client tools are
# checked against SHOW server_version_num before the rehearsal creates a
# database (scripts/check_postgres_client_version.py).
test-backup-restore: require-database
	python3 scripts/check_postgres_client_version.py
	go test -tags=integration -count=1 -timeout=10m ./tests/integration/restore -run '^TestBackupRestorePreservesExactArtifactsAndCAS$$'

test-lease-integration:
	go test -race -count=1 ./tests/integration/lease
	go test -race -count=1 ./internal/controlplane/... ./internal/scheduler/... -run 'Lease|Reconcile|Partition'
	cd runtime && uv run pytest tests/test_lease_watchdog.py

test-streamline: require-database
	go test -count=1 ./internal/planner/streamline ./internal/planner/session ./tests/integration/streamline

test-faults: require-database
	go test -race -count=1 ./tests/faults ./tests/integration/lease ./internal/helpers/requestid ./internal/controlplane ./internal/httpapi/privateartifacts ./internal/mtls ./internal/config/... ./internal/planner/...
	CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -race -count=1 ./internal/artifacts/... ./internal/httpapi/public ./internal/runstore ./internal/scheduler ./internal/telemetry
	cd runtime && uv run pytest -W error tests

# The same process pass as release-verify's process stage.
test-e2e: test-release-process-e2e

test-capability-e2e: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=3m ./tests/e2e -run '^TestHeterogeneousRuntimeCapabilityPlacement$$'

.PHONY: test-interactsh-protocol
test-interactsh-protocol: runtime-venv
	sh scripts/test_interactsh_protocol.sh
