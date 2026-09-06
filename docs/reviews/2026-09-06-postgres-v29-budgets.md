# V29-007: database budgets and integrated verification

## Policy

Production `postgres.OpenPool` now installs finite server settings and pgx
operation contexts. The limits apply to database work, not the lifetime of an
HTTP request, artifact download or WebSocket.

| Work | Pool acquire | Client query | Server statement | Lock wait | Idle transaction | Whole operation |
| --- | --- | --- | --- | --- | --- | --- |
| Ordinary repositories | 2 s | 20 s | 15 s | 2 s | 30 s | Existing caller deadline |
| Migrator | 5 s | 125 s | 120 s | 10 s | 60 s | 15 min |
| Physical Artifact collection | 2 s | 65 s | 60 s | 5 s | 30 s | 2 min |

The ordinary defaults can be changed through `PoolOptions.Budgets`; zero fields
select defaults, not infinity. Durations must be 1 ms–24 h, and lock < statement
< query. These explicit pool settings take precedence over the corresponding
connection-string settings. No new environment variables or HTTP-wide timeout
were introduced. Existing shorter caller deadlines continue to win.

Acquire and Query/QueryRow/Exec contexts are bounded using pgx's supported
tracing hooks, whose returned context is used for the operation. Their timers
are cancelled at the respective end hook; a Query timer lives through row
consumption/Close. All current repository SQL uses these methods. Direct raw
`PgConn` operations, or future CopyFrom/SendBatch callers, must supply their own
bounded context or extend this policy; they are not current repository paths.
The startup Ping also has an explicit client budget.

`ApplyMigrations` selects the migration context before opening its transaction.
`InTx` applies maintenance server settings with transaction-local `set_config`.
The physical purger selects the cleanup context and applies the same local
policy to its existing owning transaction. Commit/rollback restores pool
defaults, including when the maintenance operation fails. This does not turn
timeouts into retryable conflicts: only definite 40001/40P01 aborts retain the
V29-006 retry policy. Transport/commit ambiguity is not automatically replayed.
Rollback still uses a fresh, cancellation-independent context bounded to 5 s.

LISTEN acquisition and its short LISTEN/UNLISTEN commands are bounded. The
subsequent `WaitForNotification` deliberately uses the listener's original
lifetime context. It is outside a transaction, so the idle-transaction timeout
does not terminate an idle subscription. Notification cancellation and
UNLISTEN/release remain bounded during shutdown. HTTP/WebSocket settings and
streaming behavior are unchanged.

## Safe pool diagnostics

The existing process `slog.Logger` receives a rate-limited warning for a failed
acquire or an acquire taking at least 100 ms. At most one warning per second
per pool includes outcome, wait milliseconds, max/total/acquired/idle connection
counts and cumulative empty/cancelled acquire counts/acquire duration.
No SQL, parameters, DSN, connection config or raw exception is logged.

## Isolated PostgreSQL evidence

The verification database is a disposable **PostgreSQL 17.11** container,
matching the version of the original demo review. Tests use isolated schemas;
the demo database, its settings and running services are untouched.

`TestPostgresOrdinaryWaitBudgetsAndRecovery` uses deliberately short configured
budgets and invokes ordinary operations without caller deadlines:

| Probe | Observed outcome |
| --- | --- |
| Pool exhausted | DeadlineExceeded after about 0.11 s; cancelled-acquire counter and safe pressure log; capacity reusable |
| Statement stalled | 57014 after about 0.31 s; one attempt, writes rolled back |
| Row locked by another transaction | 55P03 after about 0.08 s; one attempt, writes rolled back |
| Server statement timeout disabled locally | Client DeadlineExceeded after about 0.53 s; rollback and pool recovery |
| Shorter caller deadline | Cancellation identity preserved, pending writes rolled back |
| Abandoned idle server transaction | Backend reaped after about 0.23 s; uncommitted write gone and connection replaced |

Numbers are representative observations, not latency promises. Tests assert
error identity, rollback, capacity recovery and generous finite upper bounds.

Maintenance tests execute a 650 ms statement successfully on a pool whose
ordinary client/server limits are 500/300 ms. They verify local settings and
their restoration on both commit and injected rollback, apply/replay all
migrations, and cancel a blocked migration advisory-lock acquisition before
successfully running the migrator again. A real Run event listener stays alive
beyond all the short test budgets, receives NOTIFY and releases on shutdown.

## PG-01–PG-07 regression mapping

| Finding | Repeatable production-path fixture |
| --- | --- |
| PG-01: nested pool acquisition | `TestTransactionLookupPinsRuntimeConfigWithoutAnotherPoolConnection`, now using production OpenPool with two connections, one reserved for LISTEN |
| PG-02: serialized pins | `TestPostgresBindingPinsShareLocksAndExcludeWriters` and cancelled-pin rollback |
| PG-03: orphan blob | Concurrent shared-blob purge, rollback, identical write and retaining-publication tests |
| PG-04: scans/sort | 50,000-row `TestPostgresArtifactSelectiveQueryPlans`, migration replay and exact pagination |
| PG-05: N+1 | `TestPostgresRunDetailFixedBatchQueries`: 13 queries for 1/5/30 stages, byte-identical responses |
| PG-06: lost SQLSTATE/recovery | Real binding rebind and post-create rollback tests, now using production OpenPool; deadlock and finite-retry tests |
| PG-07: unbounded waits | Budget, maintenance, cancellation and listener tests above |

See [the V29 access-path and query-count evidence](2026-09-06-postgres-v29.md)
and [the original baseline](2026-09-05-postgres-review.md).

## Initial aggregate gate

The required PostgreSQL race suite passed, plus credentials, Project lifecycle
and public event tests. `make verify` passed Go formatting, vet and Ruff after
format-only corrections in the existing Runtime artifact/finding clients, but
stopped at `TestMigratedAnalysisSkillsAreCompleteDeterministicAndClosed`:
`audit_asvs_source_verifier.yaml unexpectedly selects skills/trace`.
That is part of the separate ASVS work and was not changed under V29-007.
Keep this task `in_progress` until aggregate verification is green.

The remaining aggregate targets were also run independently:
`make test-runtime build ui-verify` passed (Runtime: 835 passed / 3 skipped;
UI: 227 tests plus 6 server tests; Go and UI builds passed). UI tooling warns
that installed Node 26 differs from the declared Node 24.20.x engine, but its
checks pass. No ASVS whitelist or verification assertion was weakened.

## ASVS assignment follow-up

With explicit user approval, commit
`57a4b9037cd3f1eecbcd4f11afc4f7dcc8e11c78` adds only
`audit_asvs_source_verifier.yaml → trace` to the expected assignment inventory.
The guard still rejects every unlisted assignment and requires each expected
assignment to exist. The original failing test now passes.

The main worktree's subsequent `make verify` stops on Ruff errors in the
separate in-progress `projectfs` implementation; those files were untouched.
To isolate submitted code from ongoing work, verification was repeated in a
clean detached worktree at the exact commit above:

- Required PostgreSQL race suite plus credentials, Project lifecycle and
  public events: passed on a separate PostgreSQL 17.11 container after readiness
  was confirmed. An initial attempt started before container readiness and was
  rerun; no database behavior was changed to accommodate it.
- Full Go tests, vet, Ruff, Runtime tests (835 passed / 3 skipped), Go build,
  UI generation check, lint and typecheck: passed.
- UI tests: 225 passed, one failed. The committed Workflow route test expects
  `eval.id=eval-ui-01`; the existing UI renders `eval.id:eval-ui-01`.
  The matching expectation correction already exists among the main worktree's
  uncommitted UI changes and was not staged as part of this ASVS fix.

The ASVS blocker is resolved, but the complete aggregate command has not yet
passed. V29-007 remains `in_progress`; no claim is made that either the clean
commit or the combined development tree currently passes `make verify`.
