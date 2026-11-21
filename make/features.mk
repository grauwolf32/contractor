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
	$(call run-family-test,go test -tags=integration -race -count=1 ./internal/auditdomain ./internal/auditstore/... ./internal/auditservice ./internal/auditcontroller ./internal/auditimport ./internal/findingintake ./internal/runservice)
	cd ui && corepack pnpm test --run src/api/audits.test.ts src/routes/projects/audits/check-page.test.tsx src/routes/projects/audits/audits.test.tsx

test-audits-process: test-audit-basic-e2e

# Assertion-specific execution gates return with the rewritten profiles in V64-012.
test-top10-audit-e2e test-asvs-audit-e2e:
	@echo 'Assertion Audit profiles are temporarily absent; V64-012 restores their execution gates.' >&2
	@exit 1

# The browser process is built and served independently from the Go API. The
# suite includes both mocked Audit contract flows and the production stack.
test-audits-browser: test-ui-stack

test-audits-e2e: test-audits-hardening test-audits-process test-audits-browser

test-audit-program-library-matrix:
	$(call run-family-test,go test -count=1 ./tests/e2e -run '^TestAuditProgramLibraryMatrixIsComplete$$')

test-audit-program-library-hardening: test-audits-hardening test-audit-program-library-matrix require-database
	$(call run-family-test,go test -count=1 ./tests/eval/audit_programs)
	$(call run-family-test,CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -tags=integration -race -count=1 ./internal/auditstandards)

test-audit-program-library-process: test-audits-process

test-audit-program-library-browser: test-audits-browser

test-audit-program-library-e2e: test-audit-program-library-hardening test-audit-program-library-process test-audit-program-library-browser

.PHONY: test-audit-basic-e2e test-audit-common-programs-e2e
test-audit-basic-e2e: require-database runtime-venv
	@$(call run-family-test,python3 scripts/test-audit-basic-e2e.py)

test-audit-common-programs-e2e: require-database runtime-venv
	@$(call run-family-test,python3 scripts/test-audit-basic-e2e.py --case-set programs --output-dir .local/audit-common-programs-gate)

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
# eviction and holds, every writer adapter and the shared fixture readers against PostgreSQL, then the Server and Runtime processes
# with a configured limit.
test-artifact-history-postgres: require-database
	$(call run-family-test,CONTRACTOR_TEST_DATABASE_URL="$$CONTRACTOR_TEST_DATABASE_URL" go test -tags=integration -count=1 ./internal/artifacts/... ./internal/memory ./internal/httpapi/public ./internal/httpapi/privateartifacts ./internal/findingintake ./internal/app -run '^(TestPostgresHistory.*|TestPostgresRunRevisionLimitIsNotRefundedByHistoryRetention|TestSharedArtifactHistory.*|TestPostgresPlannerMemoryCommitsHistoryMaintenanceAfterRollback|TestPostgresPublicArtifactHistoryOutcomes|TestPostgresGitImportCommitsHistoryMaintenanceAfterItsRollback|TestPostgresReceiptCitingARemovedRevisionIsAccessDenied|TestPostgresCollectionRejectsOnlyUnretainableProposals|TestArtifactHistoryConfig.*)$$')

test-artifact-history-e2e: require-database runtime-venv
	$(call run-family-test,go test -tags=e2e -count=1 -timeout=10m ./tests/e2e -run '^TestArtifactHistoryAcrossProductionProcesses$$')

# Offline recorded providers and private scorers; the Go catalog suite is
# already in test-go during consolidated release verification.
.PHONY: test-context-compaction-eval-fixtures
test: test-context-compaction-eval-fixtures
test-context-compaction-eval-fixtures: runtime-venv
	cd runtime && uv run python ../tests/eval/context_compaction/gate.py
	$(call run-family-test,go test ./tests/eval/context_compaction/...)

# The common Findings gate includes the Project facts process and rejects skips.
.PHONY: test-project-facts-e2e
test-project-facts-e2e: test-findings-e2e
	$(call run-family-test,go test -count=1 ./tests/eval/project_facts)

.PHONY: test-memory-snapshot-e2e
test-memory-snapshot-e2e: test-findings-e2e
