.PHONY: fmt lint test-go test-runtime test-runtime-hardening test-hardening-matrices test-code-analysis-matrix test-code-analysis-runtime test-code-analysis-hardening test-code-analysis-e2e test-taint-annotations-matrix test-taint-annotations-runtime test-taint-annotations-hardening test-taint-annotations-e2e test-worker-observations-matrix test-worker-observations-runtime test-worker-observations-hardening test-worker-observations-e2e test-worker-summarizer-matrix test-worker-summarizer-runtime test-worker-summarizer-hardening test-worker-summarizer-e2e test-worker-session-modes-runtime test-worker-session-modes-hardening test-worker-session-modes-e2e test-contracts verify-wire-contracts test-wire-cross-language test-memory-contracts test-agent-skill-contract test-agent-skills-matrix test-agent-skills-runtime-hardening test-agent-skills-races test-agent-skills-mvp test-agent-skills-hardening test-migrated-agent-skills test-migrated-agent-skills-analysis test-migrated-agent-skills-live test-shared-memory-matrix test-shared-memory-faults test-shared-memory-hardening test-runtime-wire-v2 test-runtime-principals test-runtime-label-placement test-runtime-configuration-matrix test-runtime-configuration-hardening test-runtime-configuration-e2e test-run-metadata-labels-matrix test-run-metadata-labels-hardening test-run-metadata-labels-e2e test-http-caido-matrix test-http-caido-runtime test-http-caido-architecture test-http-caido-hardening test-config verify-public-api test-postgres test-runtime-config-postgres test-runtime-credentials test-litellm-contract test-mtls test-control-integration test-artifact-integration test-lease-integration test-streamline test-faults test-e2e test-capability-e2e test-runtime-labels-e2e test-shared-memory-e2e test-http-caido-e2e test-project-workflows test-project-workspaces-matrix test-project-workspaces-hardening test-project-workspaces-e2e test-project-workspaces-release test-project-workflows-live test-live-routing test-ui-stack ui-install ui-browser-install ui-generate ui-generate-check ui-format ui-lint ui-typecheck ui-test ui-build ui-verify run-local test build verify release-verify

fmt:
	gofmt -w cmd internal tests
	cd runtime && uv run ruff format .

lint:
	test -z "$$(gofmt -l cmd internal tests)"
	go vet ./...
	cd runtime && uv run ruff check .
	cd runtime && uv run ruff format --check .

test-go: test-hardening-matrices
	go test $$(go list ./... | grep -v '/tests/e2e$$')

test-runtime:
	cd runtime && uv run pytest

test-runtime-hardening:
	cd runtime && uv run pytest -W error tests

test-hardening-matrices:
	go test -count=1 ./tests/e2e

test-code-analysis-matrix: test-hardening-matrices

test-code-analysis-runtime:
	cd runtime && uv run pytest -W error tests/test_code_analysis_shallow.py tests/test_code_analysis_graph.py tests/test_code_analysis_traversal.py tests/test_trailmark_child.py tests/test_code_analysis_child_lifecycle.py tests/test_trailmark_probe.py tests/test_capabilities.py

test-code-analysis-hardening: test-code-analysis-matrix
	cd runtime && uv run pytest -W error tests/test_code_analysis_concurrency.py tests/test_code_analysis_faults.py tests/test_code_analysis_redaction.py tests/test_allocation.py tests/test_abort.py tests/test_lease_watchdog.py tests/test_code_analysis_child_lifecycle.py

test-code-analysis-e2e: test-code-analysis-hardening
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=6m ./tests/e2e -run '^TestCodeAnalysisAcrossHeterogeneousRuntimeProcesses$$'

test-taint-annotations-matrix: test-hardening-matrices

test-taint-annotations-runtime:
	cd runtime && uv run pytest -W error tests/test_taint_annotations.py tests/test_capabilities.py

test-taint-annotations-hardening: test-taint-annotations-matrix test-taint-annotations-runtime
	cd runtime && uv run pytest -W error tests/test_taint_annotations_hardening.py tests/test_allocation.py tests/test_abort.py tests/test_lease_watchdog.py

test-taint-annotations-e2e: test-taint-annotations-hardening
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=6m ./tests/e2e -run '^TestTaintAnnotationsAcrossRealRuntimeProcess$$'

test-worker-observations-matrix: test-hardening-matrices

test-worker-observations-runtime:
	cd runtime && uv run pytest -W error tests/test_adk_runtime.py tests/test_a2a_server.py tests/test_instrumentation.py tests/test_worker_state.py tests/test_metrics.py tests/test_agent_state_endpoint.py tests/test_observations.py tests/test_filesystem_observations.py tests/test_allocation.py

test-worker-observations-hardening: test-worker-observations-matrix test-worker-observations-runtime
	go test -race -count=1 ./internal/config/... ./internal/contracts/... ./internal/controlplane/... ./internal/planner/...

test-worker-observations-e2e: test-worker-observations-hardening
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=8m ./tests/e2e -run '^TestRoutingAndEscalationProductionBoundaries$$'

test-worker-summarizer-matrix: test-hardening-matrices

test-worker-summarizer-runtime:
	cd runtime && uv run pytest -W error tests/test_adk_runtime.py tests/test_worker_summarizer.py tests/test_token_usage.py tests/test_instrumentation.py tests/test_worker_state.py tests/test_metrics.py

test-worker-summarizer-hardening: test-worker-summarizer-matrix test-worker-summarizer-runtime
	go test -race -count=1 ./internal/config/... ./internal/contracts/... ./internal/controlplane/... ./internal/telemetry/...

test-worker-summarizer-e2e: test-worker-summarizer-hardening
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=8m ./tests/e2e -run '^TestWorkerSummarizerProductionBoundaries$$'

test-worker-session-modes-runtime:
	cd runtime && uv run pytest -W error tests/test_session_lifecycle.py tests/test_contracts.py tests/test_a2a_server.py tests/test_adk_runtime.py tests/test_allocation.py

test-worker-session-modes-hardening: verify-wire-contracts test-wire-cross-language test-worker-session-modes-runtime
	go test -race -count=1 ./internal/config/... ./internal/contracts/... ./internal/controlplane/... ./internal/scheduler/... ./internal/httpapi/public/...

test-worker-session-modes-e2e: test-worker-session-modes-hardening
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=5m ./tests/e2e -run '^TestWorkerSessionModesAcrossProductionProcesses$$'

test-contracts:
	go test ./internal/contracts/...
	cd runtime && uv run pytest tests/test_contracts.py

verify-wire-contracts:
	go test ./internal/contracts/...
	cd runtime && uv run pytest tests/test_contracts.py

test-wire-cross-language:
	go test ./internal/contracts/... ./internal/config/... -run 'Golden|SharedPython|ResolvedSkills'
	cd runtime && uv run pytest tests/test_contracts.py -k 'golden or digest or resolved_skills'

test-memory-contracts:
	go test ./internal/memory/...
	cd runtime && uv run pytest tests/test_memory.py

test-agent-skill-contract:
	go test -race ./internal/agentskills/... ./cmd/contractor-skill/...
	cd runtime && uv run pytest tests/test_agent_skill_package.py

test-agent-skills-matrix: test-hardening-matrices

test-agent-skills-runtime-hardening:
	cd runtime && uv run pytest -W error tests/test_agent_skill_package.py tests/test_agent_skill_toolset.py tests/test_agent_skill_lifecycle.py tests/test_run_artifacts_toolset.py

test-agent-skills-races:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -race -count=1 -timeout=4m ./internal/agentskills/... ./internal/artifacts/... ./internal/app/... ./internal/httpapi/privateartifacts/... ./internal/httpapi/public/... ./internal/runstore/... ./internal/scheduler/...

test-agent-skills-mvp:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
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

test-shared-memory-faults:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -race -count=1 ./internal/artifactpolicy/... ./internal/config/... ./internal/memory/... ./internal/httpapi/privateartifacts/... ./internal/scheduler/...
	cd runtime && uv run pytest -W error tests/test_memory_toolset.py -k 'response_loss or changed or serialized or unexpected_tool_failure or reconciliation or purpose_reserved'

test-shared-memory-hardening: test-shared-memory-matrix test-memory-contracts test-shared-memory-faults test-shared-memory-e2e

test-runtime-wire-v2:
	go test ./internal/contracts/... ./internal/controlplane/...
	cd runtime && uv run pytest tests/test_contracts.py tests/test_capabilities.py tests/test_settings.py tests/test_state.py tests/test_control_client.py

test-runtime-principals:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -count=1 ./internal/runtimeconfig ./internal/persistence/postgres

test-runtime-label-placement:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -count=1 ./internal/controlplane ./internal/credentials ./internal/runstore ./internal/runtimeconfig

test-runtime-configuration-matrix: test-hardening-matrices

test-runtime-configuration-hardening: test-runtime-configuration-matrix
	go test -race -count=1 ./internal/controlplane/... ./internal/credentials/... ./internal/httpapi/... ./internal/mtls/... ./internal/runstore/... ./internal/runtimeconfig/... ./internal/scheduler/... ./internal/telemetry/... ./tests/integration/lease
	cd runtime && uv run pytest -W error tests/test_adapter_host.py tests/test_http_proxy_adapter.py tests/test_lease_watchdog.py tests/test_otlp_adapter.py

test-runtime-configuration-e2e: test-runtime-configuration-hardening test-runtime-labels-e2e test-ui-stack

test-run-metadata-labels-matrix: test-hardening-matrices

test-run-metadata-labels-hardening: test-run-metadata-labels-matrix
	go test -race -count=1 ./internal/contracts/... ./internal/controlplane/... ./internal/httpapi/public ./internal/persistence/postgres ./internal/runstore ./internal/scheduler/... ./internal/telemetry/...
	cd runtime && uv run pytest -W error tests/test_contracts.py tests/test_app.py tests/test_allocation.py tests/test_adapter_host.py tests/test_otlp_adapter.py
	cd ui && corepack pnpm test --run src/api/run-metadata-labels.test.ts src/api/workflows.test.ts src/run-drafts/idempotency.test.ts src/routes/workflows/workflows.test.tsx src/routes/runs/runs.test.tsx

test-run-metadata-labels-e2e: test-run-metadata-labels-hardening
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=5m ./tests/e2e -run '^TestRunMetadataLabelsAcrossProcesses$$'

test-http-caido-matrix: test-hardening-matrices

test-http-caido-runtime:
	cd runtime && uv run pytest -W error tests/test_http_caido_security.py tests/test_http_caido_concurrency.py tests/test_http_caido_redaction.py tests/test_http_toolset.py tests/test_http_toolset_lifecycle.py tests/test_http_proxy_adapter.py tests/test_caido_adapter.py tests/test_caido_graphql_fixtures.py tests/test_caido_read_tools.py tests/test_caido_action_tools.py tests/test_adapter_host.py

test-http-caido-architecture:
	npx --yes likec4@1.56.0 validate docs/spec

test-http-caido-hardening: test-http-caido-matrix test-http-caido-runtime test-http-caido-architecture test-http-caido-e2e

test-config:
	go test ./internal/config/...
	go run ./cmd/contractor-server config validate --root ./configs

verify-public-api:
	go test -count=1 ./internal/httpapi/public -run '^(TestPublicOpenAPIContractIsValidAndPolicySafe|TestPublicEventSchemaIsClosedAndExamplesValidate|TestImplementedPublicHandlersConformToOpenAPI|TestPublicOpenAPIPathsAreRepositoryRelative)$$'

test-postgres:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -count=1 ./internal/persistence/postgres ./internal/credentials ./internal/runtimeconfig ./internal/runstore ./internal/artifacts ./internal/httpapi/public ./internal/planner/session ./internal/scheduler ./internal/telemetry

test-runtime-config-postgres:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -count=1 ./internal/runtimeconfig ./internal/persistence/postgres

test-runtime-credentials:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -count=1 ./internal/credentials ./internal/runtimeconfig ./internal/persistence/postgres

test-litellm-contract:
	deploy/litellm/test-contract.sh

test-mtls:
	go test -count=1 ./internal/mtls/... ./cmd/contractor-pki/...
	cd runtime && uv run pytest tests/test_mtls.py

test-control-integration:
	go test -tags=integration -count=1 ./internal/controlplane -run TestCrossLanguageMTLSAllocationLifecycle

test-artifact-integration:
	go test -tags=integration -count=1 ./internal/httpapi/privateartifacts -run TestCrossLanguagePrivateArtifactLifecycle

test-lease-integration:
	go test -race -count=1 ./tests/integration/lease
	go test -race -count=1 ./internal/controlplane/... ./internal/scheduler/... -run 'Lease|Reconcile|Partition'
	cd runtime && uv run pytest tests/test_lease_watchdog.py

test-streamline:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -count=1 ./internal/planner/streamline ./internal/planner/session ./tests/integration/streamline

test-faults:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -race -count=1 ./tests/faults ./tests/integration/lease ./internal/requestid ./internal/controlplane ./internal/httpapi/privateartifacts ./internal/mtls ./internal/config ./internal/planner/...
	CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -race -count=1 ./internal/artifacts ./internal/httpapi/public ./internal/runstore ./internal/scheduler ./internal/telemetry
	cd runtime && uv run pytest -W error tests

test-e2e:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=24m ./tests/e2e -run '^(TestLocalGoToPythonArtifactCopy|TestRoutingAndEscalationProductionBoundaries|TestHeterogeneousRuntimeCapabilityPlacement|TestLabelDrivenRuntimeConfigurationAcrossProcesses|TestRunMetadataLabelsAcrossProcesses|TestSharedMemoryMVPProcesses|TestHTTPAndCaidoAcrossHeterogeneousRuntimeProcesses|TestCodeAnalysisAcrossHeterogeneousRuntimeProcesses|TestTaintAnnotationsAcrossRealRuntimeProcess|TestWorkerSummarizerProductionBoundaries|TestWorkerSessionModesAcrossProductionProcesses|TestProjectWorkspaceLifecycleAcrossProductionProcesses)$$'

test-capability-e2e:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=3m ./tests/e2e -run '^TestHeterogeneousRuntimeCapabilityPlacement$$'

test-runtime-labels-e2e:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=4m ./tests/e2e -run '^TestLabelDrivenRuntimeConfigurationAcrossProcesses$$'

test-shared-memory-e2e:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=5m ./tests/e2e -run '^TestSharedMemoryMVPProcesses$$'

test-http-caido-e2e:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=6m ./tests/e2e -run '^TestHTTPAndCaidoAcrossHeterogeneousRuntimeProcesses$$'

test-project-workspaces-matrix: test-hardening-matrices

test-project-workspaces-hardening: test-project-workspaces-matrix
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	go test -race -count=1 ./internal/projectstore/... ./internal/artifacts/... ./internal/runstore/... ./internal/scheduler/... ./internal/httpapi/public
	cd runtime && uv run pytest -W error tests/test_http_toolset.py -k project_authorization_is_exact_origin_hidden_and_erased
	cd ui && corepack pnpm test --run src/api/projects.test.ts src/api/project-artifacts.test.ts src/api/queue.test.ts src/routes/projects/projects.test.tsx src/routes/projects/recommendations.test.ts src/routes/queue.test.tsx

test-project-workspaces-e2e:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=6m ./tests/e2e -run '^TestProjectWorkspaceLifecycleAcrossProductionProcesses$$'

test-project-workspaces-release: test-project-workspaces-hardening test-project-workspaces-e2e test-ui-stack

.PHONY: test-lifecycle-controls-matrix test-lifecycle-controls-hardening test-lifecycle-controls-e2e test-lifecycle-controls-browser test-lifecycle-controls-release

test-lifecycle-controls-matrix: test-hardening-matrices

test-lifecycle-controls-hardening: test-lifecycle-controls-matrix
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -race -count=1 ./internal/persistence/postgres ./internal/runstore ./internal/scheduler ./internal/artifacts ./internal/projectlifecycle ./internal/httpapi/public -run '^(TestTerminalRunPurgeMigrationTracksPinOwnershipAndKeepsBypassScoped|TestProjectDeletionMigrationIsDurableClaimedAndFenced|TestOwnerQueueControlMigrationIsDurableAndRevisionProtected|TestPostgresOwnerQueueControlSerializesWithStageAdmission|TestPostgresQueuePauseAllowsTerminalResultCommit|TestSchedulerOwnerQueuePauseDefersInitialAdmissionUntilResume|TestSchedulerOwnerQueuePauseDrainsCurrentStageWithoutAdmittingNext|TestSchedulerOwnerQueuePauseDoesNotBlockCancellation|TestPostgresIntegrationDeletesReleasedTerminalRunWithoutSharedArtifacts|TestPostgresIntegrationRunDeletionParticipatesInCallerTransaction|TestProjectDeletionCancelsDrainsPurgesAndRetainsSharedResources|TestProjectDeleteIsCASDurableIdempotentAndFencesMutations|TestDeleteRunRequiresOwnedReleasedTerminalRun|TestUserArtifactNamespaceExclusionPrecedesPagination)$$'
	cd ui && corepack pnpm test --run src/api/projects.test.ts src/api/project-artifacts.test.ts src/api/queue.test.ts src/routes/projects/projects.test.tsx src/routes/queue.test.tsx src/routes/runs/runs.test.tsx

test-lifecycle-controls-e2e:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=8m ./tests/e2e -run '^(TestLocalGoToPythonArtifactCopy|TestProjectWorkspaceLifecycleAcrossProductionProcesses)$$'

# test-ui-stack serves the built Node UI independently from the Go API and
# includes ui/e2e/lifecycle-controls.spec.ts in the production browser suite.
test-lifecycle-controls-browser: test-ui-stack

test-lifecycle-controls-release: test-lifecycle-controls-hardening test-lifecycle-controls-e2e test-lifecycle-controls-browser

# Backward-compatible name retained for local scripts.
test-project-workflows: test-project-workspaces-e2e

test-project-workflows-live:
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	@test -n "$$CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL" || (echo "CONTRACTOR_WORKFLOWS_LIVE_GATEWAY_URL is required" >&2; exit 1)
	@test -n "$$CONTRACTOR_WORKFLOWS_LIVE_MODEL" || (echo "CONTRACTOR_WORKFLOWS_LIVE_MODEL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -count=1 -timeout=65m ./tests/eval/project_workflows -run '^TestLiveProjectWorkflows$$'

test-live-routing:
	@test -n "$$CONTRACTOR_LIVE_LLM_URL" || (echo "CONTRACTOR_LIVE_LLM_URL is required" >&2; exit 1)
	@test -n "$$CONTRACTOR_LIVE_LLM_MODEL" || (echo "CONTRACTOR_LIVE_LLM_MODEL is required" >&2; exit 1)
	go test -v -count=1 -timeout=3m ./tests/integration/streamline -run '^TestLiveRouterWorkflow$$'

test-ui-stack: ui-install ui-browser-install
	@test -n "$$CONTRACTOR_TEST_DATABASE_URL" || (echo "CONTRACTOR_TEST_DATABASE_URL is required" >&2; exit 1)
	cd runtime && uv sync --locked
	go test -tags=e2e -count=1 -timeout=6m ./tests/ui-stack

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

run-local:
	go run ./cmd/contractor-server migrate
	go run ./cmd/contractor-server serve --config-root ./configs/e2e

test: test-go test-runtime

build:
	go build ./cmd/...
	cd runtime && uv run python -m compileall -q src

verify: lint test build ui-verify

release-verify: verify test-runtime-configuration-e2e test-run-metadata-labels-e2e test-shared-memory-hardening test-agent-skills-hardening test-http-caido-hardening test-code-analysis-e2e test-taint-annotations-e2e test-worker-observations-e2e test-worker-summarizer-e2e test-worker-session-modes-e2e test-project-workspaces-release test-lifecycle-controls-release
