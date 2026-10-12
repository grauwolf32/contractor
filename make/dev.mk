# Formatting, linting, code generation and the checks that hold the
# public API and the architecture model to the specifications.

.PHONY: fmt lint generate-public-client generate-public-client-check verify-architecture docs-erd \
	docs-error-registry verify-public-api verify-public-api-postgres run-local tasks-backlog

fmt:
	gofmt -w cmd internal tests tools
	cd runtime && uv run ruff format .

# tools/testonlyexports is its own module (it pins golang.org/x/tools), so the
# root go vet does not reach it; it fails on an exported identifier that only
# tests reference, beyond the entries of its baseline (AGENTS.md 3.7).
lint: generate-public-client-check
	test -z "$$(gofmt -l cmd internal tests tools)"
	go vet ./...
	go run honnef.co/go/tools/cmd/staticcheck@v0.8.1 -checks U1000 -tests -tags=e2e,integration ./...
	go -C tools/testonlyexports vet ./...
	go -C tools/testonlyexports test -count=1 ./...
	go -C tools/testonlyexports run . -tags=integration,e2e -baseline tools/testonlyexports/baseline.txt "$(CURDIR)"
	python3 -m unittest discover -s scripts -p 'test_*.py'
	python3 scripts/check_release_verify_graph.py
	cd runtime && uv run python ../scripts/check_tasks.py
	python3 scripts/check_spec.py
	python3 scripts/shared_fixture_gate.py --sources-only
	python3 scripts/check_database_erd.py
	cd runtime && uv run ruff check .
	cd runtime && uv run ruff format --check .
	cd runtime && uv run lint-imports --no-cache --no-logo

# Regenerates tasks/BACKLOG.md from the task files; make lint fails when it is stale.
tasks-backlog:
	cd runtime && uv run python ../scripts/check_tasks.py --write-backlog

# The Go public client used by the CLI is generated from the committed OpenAPI
# document. oapi-codegen is pinned in its own tool modfile, so it builds offline
# from the module cache with the dependency versions it was released with and
# without changing the Server's go.mod.
public_client = go tool -modfile=tools/oapi-codegen.mod oapi-codegen \
	-config internal/publicclient/generated/oapi-codegen.yaml -o $(1) \
	api/openapi/contractor-public-v1.yaml

generate-public-client:
	$(call public_client,internal/publicclient/generated/public.gen.go)

# Regenerates the client into a temporary file and fails when it differs from
# the committed one, like ui-generate-check for the UI client.
generate-public-client-check:
	@tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
	$(call public_client,"$$tmp/public.gen.go") && \
	if ! cmp -s "$$tmp/public.gen.go" internal/publicclient/generated/public.gen.go; then \
		echo 'internal/publicclient/generated/public.gen.go is stale; run make generate-public-client' >&2; \
		exit 1; \
	fi

verify-architecture:
	npx --yes likec4@1.56.0 validate docs/spec

# Regenerates docs/spec/database-erd.html from a scratch database migrated to
# the current schema; see scripts/erd/README.md. make lint fails while the page
# lags the migrations (scripts/check_database_erd.py).
docs-erd:
	npm ci --prefix scripts/erd --no-audit --no-fund
	python3 scripts/erd/build_erd.py

# Regenerates the public error code table of docs/spec/platform/errors.md from
# the Server's error registry in internal/apierror; its tests fail while the
# table lags the registry.
docs-error-registry:
	go test -count=1 ./internal/apierror -run '^TestRegistryTableIsCurrent$$' -update

# verify-public-api checks the error registry against errors.md, the public
# contracts and the browser client, and validates a response of every public
# operation without a database: the selected tests must cover each operation,
# or TestMain fails.
verify-public-api:
	go test -count=1 ./internal/apierror
	CONTRACTOR_TEST_DATABASE_URL= CONTRACTOR_REQUIRE_OPERATION_COVERAGE=1 go test -count=1 ./internal/httpapi/public -run '^(TestErrorEnvelopesComeFromTheRegistry|TestPublicOpenAPIContractIsValidAndPolicySafe|TestPublicEventSchemaIsClosedAndExamplesValidate|TestRouteTableMatchesOpenAPI|TestImplementedPublicHandlersConformToOpenAPI|TestProjectRunHandlersConformToOpenAPI|TestPublicOperationsConformToOpenAPI|TestPublicOpenAPIPathsAreRepositoryRelative|TestPerformancePublicContracts|TestPublicAuditReportPreservesProposedReview|TestSharedPublicAuditPreparationContracts|TestPublicConfigurationProjectionContracts|TestPublicRuntimeConfigAuthorAndReadContracts|TestPublicRuntimeConfigPublicationResolvesGatewayAndPreservesClears|TestSharedPublicRequestAndHistoryContracts|TestAgentInstructionsAuthenticatedExactCatalogRead|TestArtifactArchiveScopesAndExactRevision|TestTransactionConflictResponsesConformToOpenAPI|TestRunCreateKeepsWorkflowCredentialLookupClass|TestRunRepeatDraftDoesNotReportLookupOutageAsUnavailableSelection|TestCreateRunRejectsAnOversizedWorkflowSnapshotBeforeAcceptance)$$'

verify-public-api-postgres: require-database
	$(call run-family-test,go test -race -count=1 ./internal/httpapi/public -run '^TestPublicAuditPaginationBoundary$$')

run-local:
	go run ./cmd/contractor-server migrate
	go run ./cmd/contractor-server serve --operator-config-root ./configs/e2e
