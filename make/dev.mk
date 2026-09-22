# Formatting, linting, code generation and the checks that hold the
# public API and the architecture model to the specifications.

.PHONY: fmt lint generate-public-client verify-architecture verify-public-api \
	verify-public-api-postgres run-local

fmt:
	gofmt -w cmd internal tests
	cd runtime && uv run ruff format .

lint:
	test -z "$$(gofmt -l cmd internal tests)"
	go vet ./...
	cd runtime && uv run ruff check .
	cd runtime && uv run ruff format --check .

generate-public-client:
	go generate ./internal/publicclient/generated

verify-architecture:
	npx --yes likec4@1.56.0 validate docs/spec

verify-public-api:
	go test -count=1 ./internal/httpapi/public -run '^(TestPublicOpenAPIContractIsValidAndPolicySafe|TestPublicEventSchemaIsClosedAndExamplesValidate|TestImplementedPublicHandlersConformToOpenAPI|TestProjectRunHandlersConformToOpenAPI|TestPublicOpenAPIPathsAreRepositoryRelative|TestPerformancePublicContracts|TestPublicAuditReportPreservesProposedReview|TestPublicAuditPreparationContracts|TestPublicConfigurationProjectionContracts|TestPublicRuntimeConfigAuthorAndReadContracts|TestPublicRuntimeConfigPublicationResolvesGatewayAndPreservesClears|TestPublicRequestAndHistoryContracts)$$'

verify-public-api-postgres: require-database
	go test -race -count=1 ./internal/httpapi/public -run '^TestPublicAuditPaginationBoundary$$'

run-local:
	go run ./cmd/contractor-server migrate
	go run ./cmd/contractor-server serve --operator-config-root ./configs/e2e
