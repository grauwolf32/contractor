# Suites that need something outside this repository: a live LLM
# Gateway, or the production Node UI served next to the Go API.

.PHONY: test-project-workflows test-project-workflows-live test-live-routing \
	test-ui-stack

# Backward-compatible name retained for local scripts.
test-project-workflows: test-project-workspaces-e2e

test-project-workflows-live: require-database runtime-venv
	@test -n "$$CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL" || (echo "CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL is required" >&2; exit 1)
	@test -n "$$CONTRACTOR_WORKFLOWS_LIVE_MODEL" || (echo "CONTRACTOR_WORKFLOWS_LIVE_MODEL is required" >&2; exit 1)
	go test -count=1 -timeout=65m ./tests/eval/project_workflows -run '^TestLiveProjectWorkflows$$'

test-live-routing:
	@test -n "$$CONTRACTOR_LIVE_LLM_URL" || (echo "CONTRACTOR_LIVE_LLM_URL is required" >&2; exit 1)
	@test -n "$$CONTRACTOR_LIVE_LLM_MODEL" || (echo "CONTRACTOR_LIVE_LLM_MODEL is required" >&2; exit 1)
	go test -v -count=1 -timeout=3m ./tests/integration/streamline -run '^TestLiveRouterWorkflow$$'

test-ui-stack: ui-install ui-browser-install require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=26m ./tests/ui-stack
