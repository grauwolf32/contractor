# Independent review fixes — verification results

All 11 reproduced findings in the [independent review](2026-09-20-independent-code-review.md)
have permanent corrections and regression tests. Each task has its own implementation
commit, followed by separate verification metadata. Existing unrelated edits were preserved.

| Finding | Task | Implementation | Change |
| --- | --- | --- | --- |
| CR-01 | [V60-013](../../tasks/v60/v60-013-runtime-operation-ownership.yml) | `9b37df76fe1e` | Retain Runtime operation ownership until cancellation cleanup completes |
| CR-02 | [V60-014](../../tasks/v60/v60-014-worker-handle-deep-copy.yml) | `bcdfd661c15b` | Detach WorkerHandle AgentCard copies without shared mutation |
| CR-03 | [V60-015](../../tasks/v60/v60-015-transaction-bound-credential-reads.yml) | `6cebf5a02734` | Bind placement and RuntimeConfig rebind credential reads to their transaction |
| CR-04 | [V60-016](../../tasks/v60/v60-016-resume-escalated-attempts.yml) | `74d3ede369ec` | Resume escalated Run attempts without reusing automatic escalation identity |
| CR-05 | [V60-017](../../tasks/v60/v60-017-http-raw-query-preservation.yml) | `0dd5bb1041ce` | Preserve caller-provided raw HTTP query bytes |
| CR-06 | [V60-018](../../tasks/v60/v60-018-session-generation-fence.yml) | `c7018285cc89` | Fence stale session responses from newer authentication state |
| CR-07 | [V60-019](../../tasks/v60/v60-019-audit-terminal-projection-refresh.yml) | `8d73ef8edce8` | Refresh Audit projections at their terminal transition |
| CR-08 | [V60-020](../../tasks/v60/v60-020-http-credential-picker-pages.yml) | `18a48d448014` | Make later Runtime credential pages selectable in HTTP target editor |
| CR-09 | [V60-021](../../tasks/v60/v60-021-encoded-client-route-identities.yml) | `cf21c0763994` | Serve encoded valid client-route identities on direct navigation |
| CR-10 | [V60-022](../../tasks/v60/v60-022-download-file-permissions.yml) | `e7f679a493e6` | Preserve private file permissions on forced CLI downloads |
| CR-11 | [V60-023](../../tasks/v60/v60-023-cli-positional-separator.yml) | `033cccd4be01` | Retain positional protection after CLI flag reordering |

The Runtime fix retains filesystem ownership through cancellation and joins validator
process groups before cleanup. Scan, LikeC4 and OpenAPI reuse one bounded asynchronous
process runner. Proxy execution retains its private environment and temporary CA lifecycle.

Credential validation now uses the current PostgreSQL transaction while retaining
credential lifecycle barriers. The regressions run with one available connection, including
a two-connection pool whose other connection is held by a listener. They cover managed LLM
and Runtime credentials, placement pins, Rebind and the public management service.

Migration `000060_manual_escalation_resumption.sql` gives manual continuation an immutable
source identity and ties it to its receipt through a deferred foreign key. Manual attempts
retain their source's exact configuration and ordinal; only original automatic attempts
consume escalation budget positions. Tests cover both escalation variants, repeated resume,
response-loss replay, restart, invalid lineage, missing-receipt rollback, schema upgrade 59→60
without changes to historical fields, and terminal Run deletion with cyclic foreign keys.

Audit coverage, items and report queries receive a final refresh when polling stops. Their
query keys remain stable and explicit revision pins remain enforced. Session generation
checks fence stale 200/401 responses. The credential picker follows every page and handles
later-page failures without presenting a partial inventory as complete.

## Verification

Verified source revision: `1ffc30d566153e70eab4e9fba6a0428b523574f5` (implementation
commits are listed above). Source hashes confirm that Runtime/UI checks cover the final
code. Commands and raw local logs are retained under `.local/review-fixes/`; each task
also has a committed evidence summary under `tasks/evidence/v60-*.json`.

| Check | Result |
| --- | --- |
| Full Go suite with disposable PostgreSQL, `go test -json -count=1 -p 2 ./...` | 62 packages passed; 3507 passing test/subtest events, 5 gated tests skipped; 11 packages have no tests |
| Focused Go race checks | Planner packages plus Control Plane, credentials, RuntimeConfig, app, RunStore, Scheduler, public API and PostgreSQL passed |
| Full Runtime, `python -m pytest -W error tests` | 2388 passed, 34 skipped |
| Full UI, `corepack pnpm test --run --maxWorkers=2` | 451 tests in 65 files passed |
| Static server, `node --test server/*.test.mjs` | 11 passed |
| TypeScript, ESLint/Prettier, Ruff lint/format, Go vet, diff whitespace | Passed |

Go skips are the pinned LiteLLM contract, local NVIDIA GPU and three live workflow/Gateway
checks. Python skips are explicit live Gateway, real rootless/Podman capability gates,
the separately launched Go Artifact API completion test, and three optional sqlmap tests.
Database regressions used the disposable PostgreSQL database and ran without DSN skips.

The initial full UI run had one five-second timeout in the existing Workflow form test
(450 tests passed). The unchanged suite passed with two workers; the timeout limit was
retained. This resource-sensitive result is preserved in `ui-full.log` and
`ui-full-retry.log`.

Toolchain: Go 1.25.6, Python 3.13.14, Node 24.20.0 and PostgreSQL 17.11.
The disposable database was stopped after verification. Verification used local fake
models and subprocess fixtures; production deployment and live external targets are
outside this corrective wave.

## Follow-up

[V60-025](../../tasks/v60/v60-025-review-readability-followups.yml) was completed in a subsequent
step: Audit sections and Runtime batch orchestration were separated, and RuntimeConfig
merge assignment became type-checked. See the [readability results](2026-09-20-review-readability-results.md).
Earlier review-track tasks V60-005–V60-010 retain their own status and acceptance requirements.
