# Product families: audits, audit programs, findings intake and
# evaluations.

.PHONY: test-audits-matrix test-audits-hardening test-audits-process \
	test-top10-audit-e2e test-asvs-audit-e2e test-audits-browser \
	test-audits-e2e test-audit-program-library-matrix \
	test-audit-program-library-hardening test-audit-program-library-process \
	test-audit-program-library-browser test-audit-program-library-e2e \
	test-audit-completion-e2e test-evals test-findings-e2e \
	test-openapi-audit-scan-e2e

test-audits-matrix: test-hardening-matrices

test-audits-hardening: test-audits-matrix require-database
	CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -tags=integration -race -count=1 ./internal/auditdomain ./internal/auditstore ./internal/auditservice ./internal/auditcontroller ./internal/auditimport ./internal/findingintake ./internal/scheduler ./internal/httpapi/public -run '^(Test.*Audit.*|TestPostgresController.*|TestPostgresControllers.*|TestOpenAPI.*|TestChecklist.*|TestImporter.*|TestRoleDispositionRetryabilityIsExplicit|TestReadRoundExecutionManifestUsesExactValidatedWorklistPackage)$$'
	cd ui && corepack pnpm test --run src/api/audits.test.ts src/api/audit-report.test.ts src/routes/projects/audits/audits.test.tsx

test-audits-process: require-database runtime-venv
	go test -tags=e2e -count=1 -timeout=15m ./tests/e2e -run '^(TestAuditProgramsAcrossProductionProcesses|TestHeterogeneousRuntimeCapabilityPlacement|TestSchedulerConcurrencyAcrossProductionProcesses)$$'

test-top10-audit-e2e: require-database runtime-venv
	go test -count=1 ./tests/eval/audit_programs -run '^TestTop10'
	go test -tags=e2e -count=1 -timeout=15m ./tests/e2e -run '^TestAuditProgramsAcrossProductionProcesses$$'

test-asvs-audit-e2e: require-database runtime-venv
	go test -count=1 ./tests/eval/audit_programs -run '^TestASVS'
	go test -tags=e2e -count=1 -timeout=15m ./tests/e2e -run '^TestAuditProgramsAcrossProductionProcesses$$'

# The browser process is built and served independently from the Go API. The
# suite includes both mocked Audit contract flows and the production stack.
test-audits-browser: test-ui-stack

test-audits-e2e: test-audits-hardening test-audits-process test-audits-browser

test-audit-program-library-matrix:
	go test -count=1 ./tests/e2e -run '^TestAuditProgramLibraryMatrixIsComplete$$'

test-audit-program-library-hardening: test-audits-hardening test-audit-program-library-matrix require-database
	go test -count=1 ./tests/eval/audit_programs
	CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -tags=integration -race -count=1 ./internal/auditstandards ./internal/auditcontroller ./internal/auditservice -run '^(TestCatalogPostgresConcurrentSeedAndExactRetention|TestPostgresControllerCollectsAndPublishesExactAuditReport|TestPostgresControllerDeletesActiveAuditWhileOwnerQueuePaused|TestAuditReportAcceptanceUsesFrozenCandidate)$$'

test-audit-program-library-process: test-audits-process

test-audit-program-library-browser: test-audits-browser

test-audit-program-library-e2e: test-audit-program-library-hardening test-audit-program-library-process test-audit-program-library-browser

test-audit-completion-e2e:
	@python3 scripts/test-audit-completion-e2e.py

test-evals:
	python3 scripts/test-managed-evals.py

test-findings-e2e:
	python3 scripts/test-findings-e2e.py

test-openapi-audit-scan-e2e:
	python3 scripts/test-openapi-audit-scan-e2e.py
