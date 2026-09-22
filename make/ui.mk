# The React UI and its Node static service. Every target runs inside
# ui/ against the pinned pnpm toolchain.

.PHONY: ui-install ui-browser-install ui-generate ui-generate-check ui-format \
	ui-lint ui-typecheck ui-test ui-build ui-verify

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
