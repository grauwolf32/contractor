# Contractor build and verification entry points.
#
# The targets live in make/*.mk, grouped by what they exercise. Two
# prerequisites are shared by many of them:
#
#   require-database  fails fast when CONTRACTOR_TEST_DATABASE_URL is unset
#   runtime-venv      prepares locked Runtime dependencies and cached parsers
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
	cd runtime && uv run python -c 'from contractor_runtime.toolsets.code_analysis.languages import PARSER_NAMES; from tree_sitter_language_pack import prefetch; prefetch(sorted(set(PARSER_NAMES.values())))'

# Family targets retain their focused Go suites when invoked directly. The
# release gate runs their union once through dedicated aggregate targets.
run-family-test = $(if $(filter 1,$(RELEASE_CONSOLIDATED)),:,$(1))

include make/dev.mk make/ui.mk make/podman.mk make/core.mk make/toolsets.mk make/platform.mk make/features.mk make/live.mk make/release.mk make/advisories.mk

test: test-go test-runtime

build:
	go build ./cmd/...
	cd runtime && uv run python -m compileall -q src

verify: lint test build ui-verify test-config

# release-verify runs verify first and then the heavier stages; its stages
# and their order are defined in make/release.mk.
