# Per-toolset and per-Worker families. Each follows the same four
# phases: matrix, runtime, hardening, then e2e across real processes.

.PHONY: test-code-analysis-matrix test-code-analysis-runtime \
	test-code-analysis-hardening test-code-analysis-e2e \
	test-taint-annotations-matrix test-taint-annotations-runtime \
	test-taint-annotations-hardening test-taint-annotations-e2e \
	test-worker-observations-matrix test-worker-observations-runtime \
	test-worker-observations-hardening test-worker-observations-e2e \
	test-worker-summarizer-matrix test-worker-summarizer-runtime \
	test-worker-summarizer-hardening test-worker-summarizer-e2e \
	test-worker-session-modes-runtime test-worker-session-modes-hardening \
	test-worker-session-modes-e2e test-memory-contracts \
	test-agent-skill-contract test-agent-skills-matrix \
	test-agent-skills-runtime-hardening test-agent-skills-races \
	test-agent-skills-mvp test-agent-skills-hardening \
	test-migrated-agent-skills test-migrated-agent-skills-analysis \
	test-migrated-agent-skills-live test-shared-memory-matrix \
	test-shared-memory-faults test-shared-memory-hardening \
	test-http-caido-matrix test-http-caido-runtime \
	test-http-caido-architecture test-http-caido-hardening \
	test-shared-memory-e2e test-production-memory-e2e test-http-caido-e2e

test-code-analysis-matrix: test-hardening-matrices

test-code-analysis-runtime:
	cd runtime && uv run pytest -W error tests/test_code_analysis_shallow.py tests/test_code_analysis_graph.py tests/test_code_analysis_traversal.py tests/test_trailmark_child.py tests/test_code_analysis_child_lifecycle.py tests/test_trailmark_probe.py tests/test_capabilities.py

test-code-analysis-hardening: test-code-analysis-matrix
	cd runtime && uv run pytest -W error tests/test_code_analysis_concurrency.py tests/test_code_analysis_faults.py tests/test_code_analysis_redaction.py tests/test_allocation.py tests/test_abort.py tests/test_lease_watchdog.py tests/test_code_analysis_child_lifecycle.py

test-code-analysis-e2e: test-code-analysis-hardening require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=6m ./tests/e2e -run '^TestCodeAnalysisAcrossHeterogeneousRuntimeProcesses$$'

test-taint-annotations-matrix: test-hardening-matrices

test-taint-annotations-runtime:
	cd runtime && uv run pytest -W error tests/test_taint_annotations.py tests/test_capabilities.py

test-taint-annotations-hardening: test-taint-annotations-matrix test-taint-annotations-runtime
	cd runtime && uv run pytest -W error tests/test_taint_annotations_hardening.py tests/test_allocation.py tests/test_abort.py tests/test_lease_watchdog.py

test-taint-annotations-e2e: test-taint-annotations-hardening require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=6m ./tests/e2e -run '^TestTaintAnnotationsAcrossRealRuntimeProcess$$'

test-worker-observations-matrix: test-hardening-matrices

test-worker-observations-runtime:
	cd runtime && uv run pytest -W error tests/test_adk_runtime.py tests/test_a2a_server.py tests/test_instrumentation.py tests/test_worker_state.py tests/test_metrics.py tests/test_agent_state_endpoint.py tests/test_observations.py tests/test_filesystem_observations.py tests/test_allocation.py

test-worker-observations-hardening: test-worker-observations-matrix test-worker-observations-runtime
	go test -race -count=1 ./internal/config/... ./internal/contracts/... ./internal/controlplane/... ./internal/planner/...

test-worker-observations-e2e: test-worker-observations-hardening require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=8m ./tests/e2e -run '^TestRoutingAndEscalationProductionBoundaries$$'

test-worker-summarizer-matrix: test-hardening-matrices

test-worker-summarizer-runtime:
	cd runtime && uv run pytest -W error tests/test_adk_runtime.py tests/test_worker_summarizer.py tests/test_token_usage.py tests/test_instrumentation.py tests/test_worker_state.py tests/test_metrics.py

test-worker-summarizer-hardening: test-worker-summarizer-matrix test-worker-summarizer-runtime
	go test -race -count=1 ./internal/config/... ./internal/contracts/... ./internal/controlplane/... ./internal/telemetry/...

test-worker-summarizer-e2e: test-worker-summarizer-hardening require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=8m ./tests/e2e -run '^TestWorkerSummarizerProductionBoundaries$$'

test-worker-session-modes-runtime:
	cd runtime && uv run pytest -W error tests/test_session_lifecycle.py tests/test_contracts.py tests/test_a2a_server.py tests/test_adk_runtime.py tests/test_allocation.py

test-worker-session-modes-hardening: verify-wire-contracts test-wire-cross-language test-worker-session-modes-runtime
	go test -race -count=1 ./internal/config/... ./internal/contracts/... ./internal/controlplane/... ./internal/scheduler/... ./internal/httpapi/public/...

test-worker-session-modes-e2e: test-worker-session-modes-hardening require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=5m ./tests/e2e -run '^TestWorkerSessionModesAcrossProductionProcesses$$'

test-memory-contracts:
	go test ./internal/memory/...
	cd runtime && uv run pytest tests/test_memory.py

test-agent-skill-contract:
	go test -race ./internal/agentskills/... ./cmd/contractor-skill/...
	cd runtime && uv run pytest tests/test_agent_skill_package.py

test-agent-skills-matrix: test-hardening-matrices

test-agent-skills-runtime-hardening:
	cd runtime && uv run pytest -W error tests/test_agent_skill_package.py tests/test_agent_skill_toolset.py tests/test_agent_skill_lifecycle.py tests/test_run_artifacts_toolset.py

test-agent-skills-races: require-database
	go test -race -count=1 -timeout=4m ./internal/agentskills/... ./internal/artifacts/... ./internal/app/... ./internal/httpapi/privateartifacts/... ./internal/httpapi/public/... ./internal/runstore/... ./internal/scheduler/...

test-agent-skills-mvp: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=5m ./tests/e2e -run '^TestAgentSkillsMVPProcesses$$'

test-agent-skills-hardening: test-agent-skills-matrix test-agent-skill-contract test-migrated-agent-skills test-agent-skills-runtime-hardening test-agent-skills-races test-agent-skills-mvp

test-migrated-agent-skills: test-migrated-agent-skills-analysis test-migrated-agent-skills-live
	go test -count=1 ./internal/agentskills/... -run '^(TestRepositoryLikeC4SkillMigrationIsCompleteAndDeterministic|TestMigratedAgentSkillsAreDeterministicAndWorkerFacing)$$'
	go test -count=1 ./internal/config/... -run '^(TestRepositoryLikeC4SkillTemplateVersionBoundary|TestRepositoryLiveSkillCompatibilityBoundary)$$'

test-migrated-agent-skills-analysis:
	go test -count=1 ./internal/agentskills/... -run '^TestMigratedAnalysisSkill'

test-migrated-agent-skills-live:
	go test -count=1 ./internal/agentskills/... -run '^TestMigratedLiveSkill'
	go test -count=1 ./internal/config/... -run 'SkillCompatibility'

test-shared-memory-matrix: test-hardening-matrices

test-shared-memory-faults: require-database
	CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -race -count=1 ./internal/artifactpolicy/... ./internal/config/... ./internal/memory/... ./internal/httpapi/privateartifacts/... ./internal/scheduler/...
	cd runtime && uv run pytest -W error tests/test_memory_toolset.py -k 'response_loss or changed or serialized or unexpected_tool_failure or reconciliation or purpose_reserved'

test-shared-memory-hardening: test-shared-memory-matrix test-memory-contracts test-shared-memory-faults test-shared-memory-e2e test-production-memory-e2e

test-http-caido-matrix: test-hardening-matrices

test-http-caido-runtime:
	cd runtime && uv run pytest -W error tests/test_http_caido_security.py tests/test_http_caido_concurrency.py tests/test_http_caido_redaction.py tests/test_http_toolset.py tests/test_http_toolset_lifecycle.py tests/test_http_proxy_adapter.py tests/test_caido_adapter.py tests/test_caido_graphql_fixtures.py tests/test_caido_read_tools.py tests/test_caido_action_tools.py tests/test_adapter_host.py

test-http-caido-architecture: verify-architecture

test-http-caido-hardening: test-http-caido-matrix test-http-caido-runtime test-http-caido-architecture test-http-caido-e2e

test-shared-memory-e2e: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=5m ./tests/e2e -run '^TestSharedMemoryMVPProcesses$$'

test-production-memory-e2e: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=6m ./tests/e2e -run '^TestProductionMemoryTemplatesAcrossProcesses$$'

test-http-caido-e2e: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=6m ./tests/e2e -run '^TestHTTPAndCaidoAcrossHeterogeneousRuntimeProcesses$$'
