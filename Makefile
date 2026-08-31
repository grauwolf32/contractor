.PHONY: fmt lint test-go test-runtime test-runtime-hardening test-contracts test-config verify-public-api test-postgres test-runtime-config-postgres test-runtime-credentials test-litellm-contract test-mtls test-control-integration test-artifact-integration test-lease-integration test-streamline test-faults test-e2e test-capability-e2e test-project-workflows test-project-workflows-live test-live-routing test-ui-stack ui-install ui-browser-install ui-generate ui-generate-check ui-format ui-lint ui-typecheck ui-test ui-build ui-verify run-local test build verify

fmt:
	gofmt -w cmd internal tests
	cd runtime && uv run ruff format .

lint:
	test -z "$$(gofmt -l cmd internal tests)"
	go vet ./...
	cd runtime && uv run ruff check .
	cd runtime && uv run ruff format --check .

test-go:
	go test ./...

test-runtime:
	cd runtime && uv run pytest

test-runtime-hardening:
	cd runtime && uv run pytest -W error tests

test-contracts:
	go test ./internal/contracts/...
	cd runtime && uv run pytest tests/test_contracts.py

test-config:
	go test ./internal/config/...
	go run ./cmd/contractor-server config validate --root ./configs

verify-public-api:
	go test -count=1 ./internal/httpapi/public -run '^(TestPublicOpenAPIContractIsValidAndPolicySafe|TestPublicEventSchemaIsClosedAndExamplesValidate|TestImplementedPublicHandlersConformToOpenAPI|TestPublicOpenAPIPathsAreRepositoryRelative)$$'

test-postgres:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -count=1 ./internal/persistence/postgres ./internal/credentials ./internal/runtimeconfig ./internal/runstore ./internal/artifacts ./internal/httpapi/public ./internal/planner/session ./internal/scheduler ./internal/telemetry

test-runtime-config-postgres:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -count=1 ./internal/runtimeconfig ./internal/persistence/postgres

test-runtime-credentials:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
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

test-streamline:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -count=1 ./internal/planner/streamline ./internal/planner/session ./tests/integration/streamline

test-faults:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -race -count=1 ./tests/faults ./tests/integration/lease ./internal/requestid ./internal/controlplane ./internal/httpapi/privateartifacts ./internal/mtls ./internal/config ./internal/planner/...
	CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -race -count=1 ./internal/artifacts ./internal/httpapi/public ./internal/runstore ./internal/scheduler ./internal/telemetry
	cd runtime && uv run pytest -W error tests

test-e2e:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=6m ./tests/e2e -run '^(TestLocalGoToPythonArtifactCopy|TestRoutingAndEscalationProductionBoundaries|TestHeterogeneousRuntimeCapabilityPlacement)$$'

test-capability-e2e:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=3m ./tests/e2e -run '^TestHeterogeneousRuntimeCapabilityPlacement$$'

test-project-workflows:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=4m ./tests/e2e -run '^TestProjectWorkflowsFromSource$$'

test-project-workflows-live:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	@test -n "$$CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL" || (echo "CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL is required" >&2; exit 1)
	@test -n "$$CONTRACTOR_WORKFLOWS_LIVE_MODEL" || (echo "CONTRACTOR_WORKFLOWS_LIVE_MODEL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -count=1 -timeout=65m ./tests/eval/project_workflows -run '^TestLiveProjectWorkflows$$'

test-live-routing:
	@test -n "$$CONTRACTOR_LIVE_LLM_URL" || (echo "CONTRACTOR_LIVE_LLM_URL is required" >&2; exit 1)
	@test -n "$$CONTRACTOR_LIVE_LLM_MODEL" || (echo "CONTRACTOR_LIVE_LLM_MODEL is required" >&2; exit 1)
	go test -v -count=1 -timeout=3m ./tests/integration/streamline -run '^TestLiveRouterWorkflow$$'

test-ui-stack:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	$(MAKE) ui-install
	$(MAKE) ui-browser-install
	go test -tags=e2e -count=1 -timeout=6m ./tests/ui-stack

ui-install:
	cd ui && corepack pnpm install --frozen-lockfile

ui-browser-install:
	cd ui && corepack pnpm exec playwright install chromium

ui-generate:
	cd ui && corepack pnpm generate

ui-generate-check:
	cd ui && corepack pnpm generate:check

ui-format:
	cd ui && corepack pnpm format

ui-lint:
	cd ui && corepack pnpm lint

ui-typecheck:
	cd ui && corepack pnpm typecheck

ui-test:
	cd ui && corepack pnpm test --run
	cd ui && corepack pnpm test:server

ui-build:
	cd ui && corepack pnpm build

ui-verify:
	cd ui && corepack pnpm install --frozen-lockfile
	cd ui && corepack pnpm generate:check
	cd ui && corepack pnpm lint
	cd ui && corepack pnpm typecheck
	cd ui && corepack pnpm test --run
	cd ui && corepack pnpm test:server
	cd ui && corepack pnpm build

run-local:
	go run ./cmd/contractor-server migrate
	go run ./cmd/contractor-server serve --config-root ./configs/e2e

test: test-go test-runtime

build:
	go build ./cmd/...
	cd runtime && uv run python -m compileall -q src

verify: lint test build ui-verify
