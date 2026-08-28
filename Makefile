.PHONY: fmt lint test-go test-runtime test-contracts test-config test build verify

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

test: test-go test-runtime

build:
	go build ./cmd/...
	cd runtime && uv run python -m compileall -q src

verify: lint test build
