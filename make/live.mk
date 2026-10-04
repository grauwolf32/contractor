# Suites that need something outside this repository: a live LLM
# Gateway, or the production Node UI served next to the Go API.

.PHONY: test-project-workflows-live test-live-routing \
	test-ui-stack test-ui-stack-operations

test-project-workflows-live: require-database runtime-venv
	@test -n "$$CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL" || (echo "CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL is required" >&2; exit 1)
	@test -n "$$CONTRACTOR_WORKFLOWS_LIVE_MODEL" || (echo "CONTRACTOR_WORKFLOWS_LIVE_MODEL is required" >&2; exit 1)
	go test -count=1 -timeout=65m ./tests/eval/project_workflows -run '^TestLiveProjectWorkflows$$'

test-live-routing:
	@test -n "$$CONTRACTOR_LIVE_LLM_URL" || (echo "CONTRACTOR_LIVE_LLM_URL is required" >&2; exit 1)
	@test -n "$$CONTRACTOR_LIVE_LLM_MODEL" || (echo "CONTRACTOR_LIVE_LLM_MODEL is required" >&2; exit 1)
	go test -v -count=1 -timeout=3m ./tests/integration/streamline -run '^TestLiveRouterWorkflow$$'

UI_STACK_TEST_FLAGS := go test -json -tags=e2e -count=1 -timeout=26m
UI_STACK_TEST := $(UI_STACK_TEST_FLAGS) ./tests/ui-stack

test-ui-stack: ui-install ui-browser-install require-database runtime-venv
	$(call run-family-test,$(UI_STACK_TEST))

# Focused PR check for the operations browser journey. The full release gate
# continues to run every browser stack test through its two shards.
test-ui-stack-operations: ui-install ui-browser-install require-database runtime-venv
	$(UI_STACK_TEST_FLAGS) ./tests/ui-stack -run '^TestBrowserOperationsStack$$'
