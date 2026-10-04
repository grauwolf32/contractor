# Code review of main — 2026-10-04

Scope: the last 500 commits on `main` (range `126a2e56^..347b790e`, tasks
V158-001..V277-001 including V179-001..009, PRs #423-#536, 1030 files,
+35k/-15.7k lines, all landed on 2026-10-03/04). Two later commits (#537) did
not touch any finding. Each task was checked against its own goal and
acceptance criteria and against the actual diff; the current code of every
area was reviewed beyond the diffs. Findings were reproduced with tests on a
disposable PostgreSQL where feasible; line numbers refer to `347b790e`.

## Summary

- Local gates at `347b790e` are green (Go build/vet/gofmt/staticcheck U1000,
  `go test` including PostgreSQL packages, ruff, pytest 3276 passed / 39
  skipped, UI generate:check/lint/typecheck/vitest/build). One pytest test is
  flaky under load (V316-001).
- GitHub CI `release-verify` had never passed since it was created on
  2026-05-29 and `main` is not protected, so every merge in the range was
  ungated by CI (V334-001, V347-001). After the review, run 37186027201 on
  `bc7d5f37` passed in 85m43s once #537 raised the job timeout to 150 minutes
  (recorded by V155-001).
- 28 of 127 completed tasks in the range were partial or introduced a
  regression: V159, V161, V162, V173, V174, V177, V179-002/005/008/009, V180,
  V188, V194, V205, V210, V211, V221, V234, V235 (broke V165), V237, V240,
  V250, V253, V258, V260, V262, V265 and V267. V261 remains in progress.
- Verified correct: the behaviour-preserving refactors V270-V277 (AST and SQL
  comparison before/after), the Go part of V179 (all eight drift defects
  fixed) and the earlier review fixes V163, V164 and V166-V168.

Each finding is recorded as a task with its location, failure scenario and
acceptance criteria.

## Fix status

All defect and gate tasks V278-001..V340-001 are completed on branch
`review/2026-10-04-fixes`, one implementation commit per task with a
regression test that fails without the fix. Sixteen work areas were fixed in
parallel branches and merged without conflicts beyond one spec paragraph.

Decisions taken while fixing:

- V309-001 scrubs only credentials the Runtime injected and secrets it holds;
  values set by the model or other proxy users stay visible. The remaining
  Caido search oracle over stored requests is an accepted risk (see the task).
- V323-001 keeps `+` in configuration versions (spec 00 and Audit standard
  versions use it) and aligns Go, Python, schemas, OpenAPI and the UI to one
  grammar with shared fixtures.
- V304-001 confirmed that PR #536 misread a quadratic per-row trigger as a
  slow CI runner: a 20k-row insert spent 9.4 of 10.7 s in the trigger and now
  takes about 1 s.

Measured effects: schema-scoped migration locks (V338-001) halve the
PostgreSQL test packages' wall time and remove the parallel-load timeouts
(`internal/artifacts` 240-600 s -> 44 s); trusted plan reads (V302-001) cut a
1,000-member Eval tick from 3.3 s to 0.2-0.6 s; view-generation pruning
(V303-001) keeps one generation instead of ten per lifecycle.

Follow-ups found while fixing are recorded as V349-001..V363-001; V341-001
to V347-001 remain open (V347-001 is a repository setting for the owner).

## High defects

| Task | Area | Finding |
| --- | --- | --- |
| V278-001 | Audit | [Purge Audits that hold review decisions or collected finding assessments](../../tasks/v278-001-purge-audits-with-reviews-and-assessments.yml) |
| V280-001 | Audit | [Collect co-batched verification checks of one finding proposal](../../tasks/v280-001-collect-co-batched-verification-checks.yml) |
| V286-001 | Artifacts | [Release artifact transfer slots before writing archive previews](../../tasks/v286-001-release-transfer-slots-before-archive-writes.yml) |
| V288-001 | Control plane | [Tolerate transient Control Plane lease-session stalls](../../tasks/v288-001-tolerate-transient-lease-session-stalls.yml) |
| V309-001 | Runtime | [Scrub Runtime-injected target credentials from Caido results](../../tasks/v309-001-scrub-target-credentials-from-caido.yml) |

## Medium defects

| Task | Area | Finding |
| --- | --- | --- |
| V281-001 | Audit | [Bound finding-collection transactions by work](../../tasks/v281-001-bound-finding-collection-transactions.yml) |
| V283-001 | Audit | [Respect inventory limits when selecting the next Audit Round](../../tasks/v283-001-respect-inventory-limits-in-next-round.yml) |
| V289-001 | Control plane | [Scope allocation-loss interrupts to the owning Stage](../../tasks/v289-001-scope-allocation-loss-to-owning-stage.yml) |
| V293-001 | Planner | [Report permanent Planner Gateway failures as non-retryable](../../tasks/v293-001-non-retryable-permanent-planner-gateway-failures.yml) |
| V298-001 | Evals | [Keep Eval admission and publication running when one member fails](../../tasks/v298-001-eval-member-errors-do-not-block-admission.yml) |
| V299-001 | Evals | [Clear Eval stop intents the server rules out with 409](../../tasks/v299-001-clear-stop-intents-rejected-with-409.yml) |
| V305-001 | UI | [Retry failed Operations snapshot reads with backoff](../../tasks/v305-001-back-off-failed-operations-snapshots.yml) |
| V306-001 | UI | [Show artifact version-upload conflicts across reconciles](../../tasks/v306-001-show-artifact-upload-conflicts.yml) |
| V313-001 | Runtime | [Name C++ declarations that return references](../../tasks/v313-001-name-cpp-reference-declarations.yml) |
| V314-001 | Runtime | [Resolve C-family header language once for shallow, graph and taint](../../tasks/v314-001-single-header-language-resolver.yml) |
| V315-001 | Runtime | [Keep Trailmark proxy node IDs stable for root packages](../../tasks/v315-001-stable-trailmark-proxy-node-ids.yml) |
| V317-001 | Runtime | [Validate CWE standard references before storing finding proposals](../../tasks/v317-001-validate-cwe-standard-refs-at-submission.yml) |
| V320-001 | Runtime | [Compute overlay diffs off the event loop](../../tasks/v320-001-overlay-diff-off-event-loop.yml) |
| V322-001 | Runtime | [Account for the Runtime work-root lock in Podman e2e and diagnostics](../../tasks/v322-001-work-root-lock-in-podman-e2e-and-errors.yml) |
| V323-001 | Contracts | [One configuration ID and version grammar across languages](../../tasks/v323-001-single-config-version-grammar.yml) |
| V329-001 | Git/config | [Report untrusted SSH hosts that offer no pinned key type](../../tasks/v329-001-untrusted-ssh-host-without-pinned-type.yml) |

## Low defects

| Task | Area | Finding |
| --- | --- | --- |
| V279-001 | Audit | [Close report reviews when a Project deletes its Audits](../../tasks/v279-001-close-report-reviews-on-project-deletion.yml) |
| V282-001 | Audit | [Isolate deterministic retention failures to one proposal](../../tasks/v282-001-isolate-retention-failures-per-proposal.yml) |
| V284-001 | Audit | [Classify next-Round preparation failures](../../tasks/v284-001-classify-next-round-preparation-failures.yml) |
| V285-001 | Audit | [Renew expired item reviews on Resume through auditstore](../../tasks/v285-001-renew-resume-item-reviews-through-auditstore.yml) |
| V287-001 | Artifacts | [Separate storage budgets from transfer socket deadlines](../../tasks/v287-001-separate-storage-budget-from-socket-deadlines.yml) |
| V290-001 | Control plane | [Release pinned placement when a paused owner blocks admission](../../tasks/v290-001-release-placement-on-paused-admission.yml) |
| V291-001 | Control plane | [Classify concurrent label removal during rebind as a precondition failure](../../tasks/v291-001-classify-rebind-label-race.yml) |
| V292-001 | Control plane | [Bound local mTLS leaf expiry by the CA and warn on CA expiry](../../tasks/v292-001-bound-leaf-expiry-by-ca.yml) |
| V294-001 | Planner | [Keep model replies when Gateway recovery bookkeeping fails](../../tasks/v294-001-keep-replies-when-recovery-bookkeeping-fails.yml) |
| V295-001 | Planner | [Give scan-planner completion writes the shared budget and retry](../../tasks/v295-001-scan-planner-completion-budget.yml) |
| V296-001 | Planner | [Update Memory notes without loading the whole notebook](../../tasks/v296-001-update-memory-notes-without-full-load.yml) |
| V297-001 | Planner | [Remove legacy Planner session token decoding](../../tasks/v297-001-remove-legacy-session-token-decode.yml) |
| V300-001 | Evals | [Use a stable timestamp in the paused Eval claim branch](../../tasks/v300-001-stable-timestamp-in-paused-eval-claims.yml) |
| V301-001 | Evals | [Make the batch-admission race test detect overfill](../../tasks/v301-001-detect-batch-admission-overfill.yml) |
| V307-001 | UI | [Correct paged inventory states, library links and triage codes](../../tasks/v307-001-correct-paged-states-links-and-triage.yml) |
| V308-001 | UI | [Remove UI compatibility paths and duplicated helpers](../../tasks/v308-001-remove-ui-compatibility-paths.yml) |
| V310-001 | Runtime | [One host-syntax rule for HTTP, Caido and scanners](../../tasks/v310-001-shared-host-syntax-rule.yml) |
| V311-001 | Runtime | [One redaction policy for metrics and results](../../tasks/v311-001-single-redaction-policy.yml) |
| V312-001 | Runtime | [Scan Agent Card object keys for long secrets](../../tasks/v312-001-scan-agent-card-keys-for-secrets.yml) |
| V316-001 | Runtime | [Replace the V221 wall-clock stall test with structural checks](../../tasks/v316-001-structural-search-page-test.yml) |
| V318-001 | Runtime | [Reject OpenAPI 3.0 Path Item references with guidance](../../tasks/v318-001-reject-openapi30-path-item-refs.yml) |
| V319-001 | Runtime | [Bound OpenAPI seed nesting and keep findings errors non-retryable in preparation](../../tasks/v319-001-openapi-depth-and-findings-preparation-errors.yml) |
| V321-001 | Runtime | [Validate overlay delete and mkdir by delta off the event loop](../../tasks/v321-001-overlay-delete-mkdir-delta-validation.yml) |
| V324-001 | Contracts | [Cap media types at one length everywhere](../../tasks/v324-001-single-media-type-length.yml) |
| V325-001 | Contracts | [Restore Python coverage of every fixture-index message type](../../tasks/v325-001-python-fixture-index-coverage.yml) |
| V326-001 | Contracts | [Restore error wording damaged by the slices.Contains replacement](../../tasks/v326-001-restore-contains-wording.yml) |
| V327-001 | Contracts | [Check Eval state sets and plan-size estimates across Go, OpenAPI and UI](../../tasks/v327-001-eval-state-and-plan-size-conformance.yml) |
| V328-001 | Contracts | [Document transaction-retry 503 responses for public operations](../../tasks/v328-001-document-transaction-retry-503.yml) |
| V330-001 | Git/config | [Fail configuration reloads when a managed subtree is missing](../../tasks/v330-001-fail-reload-on-missing-managed-subtree.yml) |
| V331-001 | Git/config | [Enforce source bundle member sizes while writing](../../tasks/v331-001-enforce-source-member-sizes-while-writing.yml) |
| V332-001 | Git/config | [Reject source bundles the Runtime cannot open](../../tasks/v332-001-reject-runtime-unopenable-bundles.yml) |
| V333-001 | Git/config | [Normalize configuration roots and accept group-level help](../../tasks/v333-001-normalize-config-roots-and-group-help.yml) |

## CI, gates and bookkeeping

| Task | Area | Finding |
| --- | --- | --- |
| V334-001 | CI | [Keep a verdict for every main push and run fast gates first](../../tasks/v334-001-ci-verdict-per-push-and-fast-first-gate.yml) |
| V335-001 | CI | [Gate e2e tests by real selection and existence](../../tasks/v335-001-e2e-gate-real-selection.yml) |
| V336-001 | CI | [Fail Postgres-backed tests when the configured database is unreachable](../../tasks/v336-001-fail-db-tests-when-database-unreachable.yml) |
| V337-001 | CI | [Enforce the production collection budget outside the race detector](../../tasks/v337-001-non-race-collection-budget.yml) |
| V338-001 | CI | [Scope migration advisory locks to the target schema](../../tasks/v338-001-schema-scoped-migration-lock.yml) |
| V339-001 | CI | [Move live advisory scans out of the deterministic gate](../../tasks/v339-001-advisory-scans-outside-deterministic-gate.yml) |
| V340-001 | CI | [Run Eval and fault race suites in release verification](../../tasks/v340-001-race-coverage-for-evals-and-faults.yml) |
| V347-001 | Follow-up | [Protect main with a required check](../../tasks/v347-001-protect-main-branch.yml) |
| V348-001 | Bookkeeping | [Repair the task index and task records](../../tasks/v348-001-repair-task-index-and-records.yml) |

## Follow-ups

| Task | Area | Item |
| --- | --- | --- |
| V302-001 | Evals | [Stop re-validating frozen Eval plans on every read](../../tasks/v302-001-trust-stored-frozen-eval-plans.yml) |
| V303-001 | Evals | [Bound Eval view generation growth](../../tasks/v303-001-bound-eval-view-generations.yml) |
| V304-001 | Evals | [Measure and fix per-row eval_collections upserts on bulk experiment inserts](../../tasks/v304-001-measure-eval-collection-trigger-cost.yml) |
| V341-001 | Follow-up | [Investigate tolerances raised to unblock CI](../../tasks/v341-001-investigate-ci-tolerance-bumps.yml) |
| V342-001 | Follow-up | [Replace production asserts in the Runtime](../../tasks/v342-001-replace-runtime-production-asserts.yml) |
| V343-001 | Follow-up | [Move remaining raw Audit review writes into auditstore](../../tasks/v343-001-audit-review-writes-into-auditstore.yml) |
| V344-001 | Follow-up | [Deduplicate PKI CLI command tables](../../tasks/v344-001-deduplicate-pki-cli-commands.yml) |
| V345-001 | Follow-up | [Remove fsspec and test-only overlay APIs from projectfs](../../tasks/v345-001-projectfs-dead-surface.yml) |
| V346-001 | Follow-up | [Bound tree-sitter parse stalls for large files](../../tasks/v346-001-bound-tree-sitter-parse-stalls.yml) |
| V349-001 | Follow-up | [Bound OpenAPI mutation nesting before deep copies](../../tasks/v349-001-bound-openapi-mutation-nesting.yml) |
| V350-001 | Follow-up | [Index Eval claim candidates and batch collection updates](../../tasks/v350-001-eval-claims-and-update-trigger.yml) |
| V351-001 | Follow-up | [Read Eval plan digests without fetching whole documents](../../tasks/v351-001-lighter-eval-plan-reads.yml) |
| V352-001 | Follow-up | [Reject Runtime-unopenable Git snapshots and normalize dot config roots](../../tasks/v352-001-git-import-openability-and-root-dot.yml) |
| V353-001 | Follow-up | [Bound overlay diff CPU under the session lock](../../tasks/v353-001-bound-overlay-diff-cpu.yml) |
| V354-001 | Follow-up | [Back off failed Operations snapshot reads triggered by live events](../../tasks/v354-001-back-off-live-event-snapshot-reads.yml) |
| V355-001 | Follow-up | [Index the RESTRICT keys of Audit finding assessments](../../tasks/v355-001-index-assessment-restrict-keys.yml) |
| V356-001 | Follow-up | [Accept underscore hosts in Go scan planning](../../tasks/v356-001-underscore-hosts-in-scan-planning.yml) |
| V357-001 | Follow-up | [Apply the shared redaction policy to OTLP span attributes](../../tasks/v357-001-otlp-redaction-policy.yml) |
| V358-001 | Follow-up | [Align finding-verification association limits](../../tasks/v358-001-align-verification-association-limits.yml) |
| V359-001 | Follow-up | [Isolate unreadable proposal documents during collection reads](../../tasks/v359-001-isolate-unreadable-proposal-documents.yml) |
| V360-001 | Follow-up | [Label the new Audit stop codes in the UI](../../tasks/v360-001-label-new-audit-stop-codes.yml) |
| V361-001 | Follow-up | [Bound receipt hydration batches by bytes](../../tasks/v361-001-bound-receipt-hydration-batches.yml) |
| V362-001 | Follow-up | [Align Eval selector, media and 503 contracts with the shared rules](../../tasks/v362-001-eval-contracts-follow-config-grammar.yml) |
| V363-001 | Follow-up | [Use exact configuration-name patterns in UI route pre-checks](../../tasks/v363-001-exact-config-name-route-checks.yml) |

## Test infrastructure notes

- Fakes hid real defects: the importer batch test passes only because its fake
  store skips `validateCollect` (V280-001); V240 tests use a copied predicate
  (V281-001); V211 tests mock vacuum (V318-001); the Eval UI fixture still
  models pre-V235 revisions (V299-001).
- Wall-clock assertions are fragile: V221's 50 ms loop-gap check (V316-001),
  V240's 8.8 s-of-10 s collection test (V281-001) and the race/non-race budget
  split from #534 (V337-001).
- PostgreSQL tests migrate 76 versions per isolated schema behind one
  database-wide advisory lock, so parallel packages serialize (V338-001).

## Limits

Not run: the full `make release-verify`, Playwright browser suites, e2e tests
that need Podman or real scanners, and a global `-race` pass. CI job logs are
admin-only, so the cause of the latest CI failure (71 minutes into `347b790e`)
was not established. Findings marked plausible in their task files were not
reproduced.
