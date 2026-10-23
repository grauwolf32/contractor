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
	cd runtime && uv run python scripts/prefetch_parsers.py

# Go's t.TempDir and os.TempDir return $TMPDIR as given. On macOS the default
# TMPDIR lies under /var/folders, and /var (like /tmp) is a symlink into
# /private, while the filesystem blob store refuses a directory with a symlink
# component (docs/spec/artifacts/blob-storage.md). Every recipe therefore gets
# the symlink-free spelling of the same directory; where it has no symlink, as
# on the Linux CI runners, nothing changes. A directory that cannot be resolved
# is passed on unchanged so its own error surfaces. docs/testing/README.md says
# what a direct `go test` on macOS needs instead.
CANONICAL_TMPDIR := $(realpath $(or $(TMPDIR),/tmp))
ifneq ($(CANONICAL_TMPDIR),)
export TMPDIR := $(CANONICAL_TMPDIR)
endif

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
