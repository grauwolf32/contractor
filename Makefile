.PHONY: fmt lint test-go test-runtime test-contracts test-config test-postgres test-mtls test-control-integration test-artifact-integration test build verify

fmt:
	gofmt -w cmd internal
	cd runtime && uv run ruff format .

lint:
	test -z "$$(gofmt -l cmd internal)"
	go vet ./...
	cd runtime && uv run ruff check .
	cd runtime && uv run ruff format --check .

test-go:
	go test ./...

test-runtime:
	cd runtime && uv run pytest

test-contracts:
	go test ./internal/contracts/...
	cd runtime && uv run pytest tests/test_contracts.py

test-config:
	go test ./internal/config/...
	go run ./cmd/contractor-server config validate --root ./configs

test-postgres:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -count=1 ./internal/persistence/postgres ./internal/runstore ./internal/artifacts ./internal/httpapi/public ./internal/planner/session ./internal/scheduler

test-mtls:
	go test -count=1 ./internal/mtls/... ./cmd/contractor-pki/...
	cd runtime && uv run pytest tests/test_mtls.py

test-control-integration:
	go test -tags=integration -count=1 ./internal/controlplane -run TestCrossLanguageMTLSAllocationLifecycle

test-artifact-integration:
	go test -tags=integration -count=1 ./internal/httpapi/privateartifacts -run TestCrossLanguagePrivateArtifactLifecycle

test: test-go test-runtime

build:
	go build ./cmd/...
	cd runtime && uv run python -m compileall -q src

verify: lint test build
