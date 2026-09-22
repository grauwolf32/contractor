# Contractor build and verification entry points.
#
# The targets live in make/*.mk, grouped by what they exercise. Two
# prerequisites are shared by many of them:
#
#   require-database  fails fast when CONTRACTOR_TEST_DATABASE_URL is unset
#   runtime-venv      prepares runtime/.venv from the lock file
#
# Both are phony, so make runs each at most once per invocation however many
# targets in the chain ask for it. Declare them as prerequisites rather than
# repeating the command in a recipe.

.PHONY: test build verify release-verify

.PHONY: require-database runtime-venv

require-database:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)

# Go tests that drive the Runtime shell out to runtime/.venv/bin/python, so
# every suite reaching Python depends on this, test-go included.
runtime-venv:
	cd runtime && uv sync --locked

include make/dev.mk make/ui.mk make/podman.mk make/core.mk make/toolsets.mk make/platform.mk make/features.mk make/live.mk

test: test-go test-runtime

build:
	go build ./cmd/...
	cd runtime && uv run python -m compileall -q src

verify: lint test build ui-verify

release-verify: verify verify-public-api-postgres test-runtime-configuration-e2e test-run-metadata-labels-e2e test-shared-memory-hardening test-agent-skills-hardening test-http-caido-hardening test-code-analysis-e2e test-taint-annotations-e2e test-worker-observations-e2e test-worker-summarizer-e2e test-worker-session-modes-e2e test-project-workspaces-release test-lifecycle-controls-release test-scheduler-concurrency-e2e test-audit-program-library-e2e test-audit-completion-e2e test-performance-metrics
