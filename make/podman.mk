# The rootless Podman sandbox. These need a preinstalled
# digest-pinned image and a local podman, so they stay opt-in.

.PHONY: test-podman-prerequisites test-podman-matrix test-podman-unit \
	test-podman-e2e test-podman-release test-podman-workflow \
	test-podman-supervisor

test-podman-prerequisites: require-database
	@test -n "$$CONTRACTOR_TEST_PODMAN_IMAGE" || (echo "CONTRACTOR_TEST_PODMAN_IMAGE must name a preinstalled digest-pinned image" >&2; exit 1)
	@command -v podman >/dev/null || (echo "local rootless podman is required" >&2; exit 1)

test-podman-matrix:
	go test -count=1 ./tests/e2e -run PodmanSandboxMatrix

test-podman-unit:
	go test -count=1 ./internal/config ./internal/controlplane ./internal/scheduler
	cd runtime && uv run pytest -W error tests/test_podman*.py tests/test_code_execution_toolset.py

test-podman-e2e: test-podman-prerequisites
	cd runtime && CONTRACTOR_RUN_PODMAN_SUPERVISOR_GATE=1 uv run pytest -W error tests/test_podman_release_integration.py
	go test -tags=e2e -count=1 -timeout=4m ./tests/e2e -run '^TestPodmanSandboxAcrossProductionProcesses$$'
	$(MAKE) test-project-workspaces-e2e

test-podman-release: test-podman-prerequisites
	$(MAKE) test-podman-matrix
	$(MAKE) test-podman-unit
	$(MAKE) test-local-direct-workspace
	$(MAKE) test-podman-supervisor
	$(MAKE) test-podman-workflow
	$(MAKE) test-podman-e2e

test-podman-workflow:
	@test -n "$$CONTRACTOR_TEST_PODMAN_IMAGE" || (echo "CONTRACTOR_TEST_PODMAN_IMAGE must name a preinstalled digest-pinned image" >&2; exit 1)
	go test -count=1 ./internal/config
	cd runtime && CONTRACTOR_RUN_PODMAN_WORKFLOW_GATE=1 uv run pytest -W error tests/test_podman_workflow.py tests/test_agent_skill_package.py tests/test_podman_deployment_integration.py

test-podman-supervisor:
	@test -n "$$CONTRACTOR_TEST_PODMAN_IMAGE" || (echo "CONTRACTOR_TEST_PODMAN_IMAGE must name a preinstalled digest-pinned image" >&2; exit 1)
	cd runtime && CONTRACTOR_RUN_PODMAN_SUPERVISOR_GATE=1 uv run pytest -W error tests/test_podman_supervisor_integration.py tests/test_podman_owner_integration.py tests/test_podman_execution_integration.py tests/test_podman_capabilities.py
