# Product families: audits, audit programs, findings intake and
# evaluations.

.PHONY: test-audits-matrix test-audits-hardening test-audits-process \
	test-top10-audit-e2e test-asvs-audit-e2e test-audits-browser \
	test-audits-e2e test-audit-program-library-matrix \
	test-audit-program-library-hardening test-audit-program-library-process \
	test-audit-program-library-browser test-audit-program-library-e2e \
	test-audit-completion-e2e test-evals test-findings-e2e \
	test-openapi-audit-scan-e2e test-audit-composition-e2e \
	test-audit-iterative-harness test-audit-iterative-postgres test-audit-iterative-e2e \
	test-artifact-history-postgres test-artifact-history-e2e

.PHONY: test-project-sitemap-postgres
test-project-sitemap-postgres: require-database
	$(call run-family-test,python3 scripts/test-project-sitemap-postgres.py)

test-audits-matrix: test-hardening-matrices

test-audits-hardening: test-audits-matrix require-database
	$(call run-family-test,CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -tags=integration -race -count=1 ./internal/auditdomain ./internal/auditstore/internal/pgstore ./internal/auditservice ./internal/auditcontroller ./internal/auditimport ./internal/findingintake ./internal/scheduler ./internal/httpapi/public -run '^(Test.*Audit.*|TestController.*|TestPostgresController.*|TestPostgresControllers.*|TestPostgresDispatch.*|TestPostgresFindingReceipt.*|TestOpenAPI.*|TestChecklist.*|TestImporter.*|TestRoleDispositionRetryabilityIsExplicit|TestReadRoundExecutionManifestUsesExactValidatedWorklistPackage)$$')
	cd ui && corepack pnpm test --run src/api/audits.test.ts src/api/audit-report.test.ts src/routes/projects/audits/audits.test.tsx

test-audits-process: require-database runtime-venv
	$(call run-family-test,go test -tags=e2e -count=1 -timeout=15m ./tests/e2e -run '^(TestAuditProgramsAcrossProductionProcesses|TestAuditProgramCatalogReplacementRestartsServer|TestHeterogeneousRuntimeCapabilityPlacement|TestSchedulerConcurrencyAcrossProductionProcesses|TestLocalGoToPythonArtifactCopy)$$')

test-top10-audit-e2e: require-database runtime-venv
	$(call run-family-test,go test -count=1 ./tests/eval/audit_programs -run '^TestTop10')
	$(call run-family-test,go test -tags=e2e -count=1 -timeout=15m ./tests/e2e -run '^TestAuditProgramsAcrossProductionProcesses$$')

test-asvs-audit-e2e: require-database runtime-venv
	$(call run-family-test,go test -count=1 ./tests/eval/audit_programs -run '^TestASVS')
	$(call run-family-test,go test -tags=e2e -count=1 -timeout=15m ./tests/e2e -run '^TestAuditProgramsAcrossProductionProcesses$$')

# The browser process is built and served independently from the Go API. The
# suite includes both mocked Audit contract flows and the production stack.
test-audits-browser: test-ui-stack

test-audits-e2e: test-audits-hardening test-audits-process test-audits-browser

test-audit-program-library-matrix:
	$(call run-family-test,go test -count=1 ./tests/e2e -run '^TestAuditProgramLibraryMatrixIsComplete$$')

test-audit-program-library-hardening: test-audits-hardening test-audit-program-library-matrix require-database
	$(call run-family-test,go test -count=1 ./tests/eval/audit_programs)
	$(call run-family-test,CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -tags=integration -race -count=1 ./internal/auditstandards ./internal/auditcontroller ./internal/auditservice -run '^(TestCatalogPostgresConcurrentSeedAndExactRetention|TestPostgresControllerCollectsAndPublishesExactAuditReport|TestPostgresControllerDeletesActiveAuditWhileOwnerQueuePaused|TestAuditReportAcceptanceUsesFrozenCandidate)$$')

test-audit-program-library-process: test-audits-process

test-audit-program-library-browser: test-audits-browser

test-audit-program-library-e2e: test-audit-program-library-hardening test-audit-program-library-process test-audit-program-library-browser

test-audit-completion-e2e:
	@$(call run-family-test,python3 scripts/test-audit-completion-e2e.py)

test-evals:
	$(call run-family-test,python3 scripts/test-managed-evals.py)

test-findings-e2e:
	$(call run-family-test,python3 scripts/test-findings-e2e.py)

test-openapi-audit-scan-e2e:
	$(call run-family-test,python3 scripts/test-openapi-audit-scan-e2e.py)

test-audit-composition-e2e:
	$(call run-family-test,python3 scripts/test-audit-composition-e2e.py)

# Focused direct gates share their harness checks once per make invocation.
# Consolidated release verifies the same cases from canonical reports; its
# lint stage already executes these harness unit suites.
test-audit-iterative-harness: runtime-venv
	$(call run-family-test,cd runtime && uv run python -m unittest discover -s ../scripts -p 'test_audit_iterative_gate.py' -v)
	$(call run-family-test,cd runtime && uv run python -m unittest discover -s ../scripts -p 'test_release_test_reports.py' -v)

test-audit-iterative-postgres: require-database runtime-venv test-audit-iterative-harness
	$(call run-family-test,python3 scripts/test-audit-iterative.py --boundary postgres)

test-audit-iterative-e2e: require-database runtime-venv test-audit-iterative-harness
	$(call run-family-test,python3 scripts/test-audit-iterative.py --boundary process)

# Automatic retained history of every Artifact binding: the store's admission,
# eviction and holds, the migration, every writer adapter and the shared
# fixture readers against PostgreSQL, then the Server and Runtime processes
# with a configured limit.
test-artifact-history-postgres: require-database
	$(call run-family-test,CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -tags=integration -count=1 ./internal/artifacts ./internal/persistence/postgres ./internal/memory ./internal/httpapi/public ./internal/httpapi/privateartifacts ./internal/findingintake ./internal/app -run '^(TestPostgresHistory.*|TestPostgresRunRevisionLimitIsNotRefundedByHistoryRetention|TestSharedArtifactHistory.*|TestSharedPostgresArtifactHistory.*|TestPostgresPlannerMemoryCommitsHistoryMaintenanceAfterRollback|TestPostgresPublicArtifactHistoryOutcomes|TestPostgresGitImportCommitsHistoryMaintenanceAfterItsRollback|TestPostgresReceiptCitingARemovedRevisionIsAccessDenied|TestPostgresCollectionRejectsOnlyUnretainableProposals|TestArtifactHistoryConfig.*)$$')

test-artifact-history-e2e: require-database runtime-venv
	$(call run-family-test,go test -tags=e2e -count=1 -timeout=10m ./tests/e2e -run '^TestArtifactHistoryAcrossProductionProcesses$$')
