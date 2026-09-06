# PostgreSQL performance collection and retained history

V32-003 adds the optional PostgreSQL observer and history repository. Public
history endpoints and their UI belong to V32-005/V32-006; this change does not
expose an additional HTTP endpoint.

## Isolation and failure behavior

With `performance_metrics` enabled, the server runs one serial diagnostic worker
alongside the independent 15-second HTTP/process/working-pool sampler. It creates
a lazy additional pool using the configured database DSN, with `MaxConns=1`,
`MinConns=0`, `MinIdleConns=0` and `application_name=contractor-performance`.
DSN warm-pool settings cannot increase this pool. Construction does not ping;
connection or authentication failures become unavailable observations, not an
optional-feature startup error. The main server's ordinary DB startup checks
remain unchanged.

Statistics run initially and on minute boundaries; database size runs initially
and no sooner than 300 seconds after its preceding attempt. Missed ticks are
skipped, not replayed. There is no immediate retry of a failed operation, no
overlapping cycle and no transaction held between cycles. Cancellation stops the
worker and closes its pool. Diagnostic SQL never acquires the working pool.

| Operation | Acquire/connect | Statement | Query/transaction | Lock |
| --- | --- | --- | --- | --- |
| Statistics | 250 ms | 750 ms | 1 s | 100 ms |
| Size, history write/cleanup | 250 ms | 1500 ms | 2 s | 100 ms |

All work in a cycle shares a five-second deadline; earlier caller deadlines win.
The existing PostgreSQL budget tracer bounds acquisition and queries. Overrides
use transaction-local settings. Commit/rollback restores pool defaults; if a
deadline prevents confirmed rollback, pgx discards that connection. The default
diagnostic idle-in-transaction timeout is two seconds.

When disabled, the feature creates no diagnostic pool, sampler, queue, worker or
cleanup timer. A `HistoryRepository` can still read retained data through the
working pool with normal request budgets and expiry filtering.

## What observations mean

One statistics query combines current-database `pg_stat_database`,
`pg_stat_activity` and `pg_stat_user_tables` aggregates. It collects transaction,
deadlock, temporary-file and buffer counters; visible client states and Lock
waits; longest open transaction and idle-in-transaction duration; estimated live
and dead tuples, vacuum counters/timestamps and visible autovacuum workers.
The observer excludes its own backend. No SQL text, session/user/application
identifiers or per-table names are selected into the payload. There are no
application-table scans, `pg_stat_statements`, manual vacuum or automatic grants.

Ordinary roles can have hidden activity fields for other users. Such rows are
counted as hidden and activity is explicitly partial (`permission_denied`), not
silently treated as idle. Visible counts are lower bounds in that case. With
`track_counts=off`, unavailable cumulative/table statistics are omitted and the
reason is `statistics_disabled`; visible activity remains useful. Disabled
activity tracking is also reported explicitly. Background workers with no client
state are not, by themselves, mistaken for a permission failure.

Operators who need visibility across roles may explicitly grant the optional
predefined role to the server's login, after reviewing its wider visibility:

```sql
GRANT pg_read_all_stats TO contractor_server;
```

Replace `contractor_server` with the actual deployment role. The collector never
executes this grant. PostgreSQL documents the visibility rules and estimated,
asynchronously updated statistics in its
[statistics reference](https://www.postgresql.org/docs/17/monitoring-stats.html).

Rates use differences over actual elapsed observation time, not a fixed nominal
60-second divisor. First observations, changed `stats_reset`, decreasing counters
or missing fields omit rates with an explicit reason. Two absent reset timestamps
are treated as unchanged; decreasing counters still invalidate the baseline.
After a failed read, the next rate spans the actual interval since the previous
successful observation. Missing intervals remain partial. Database counters
include other applications and the monitoring queries themselves. Buffer hit
ratios describe PostgreSQL shared buffers, not OS cache or physical disk I/O.
Live/dead tuple estimates are not an exact bloat diagnosis.

`pg_database_size(current_database())` has independent freshness. A size failure
does not invalidate successful statistics. Cached values retain their original
observation/coverage timestamps; failed attempts update only attempt/status and
remove stale rates. Reusing a cached observation never counts as a new measurement.

## History, bounds and aggregation

Migration `000049_performance_minutes.sql` creates an operational-only table,
keyed by `(server_generation, minute_start)`, with schema version 1, 0–60 seconds
of coverage, a JSON byte payload of at most 32 KiB and expiry exactly 168 hours
after minute start. It has no foreign keys or deletion coupling to Runs, Projects,
reports or execution accounting. Separate range and expiry indexes support reads
and bounded cleanup.

The sampler builds at most one completed/partial minute per observation and
enqueues without database I/O. There are at most ten immutable pending records;
overflow drops the oldest pending record and increments the cumulative loss
counter recorded in diagnostics and subsequent minutes. Missing sample minutes
are counted separately and are not filled with synthetic zeroes. The serial
worker snapshots a batch and acknowledges exact successfully written keys, so
concurrent enqueues or overflow cannot cause new records to be removed. Pending
storage is at most 320 KiB; a writer may temporarily retain one additional batch
of at most 320 KiB while SQL is in flight. Shutdown does not perform an unbounded
final flush: this operational queue is intentionally best-effort, not durable.

Once per cycle, one transaction inserts at most ten records idempotently
(`ON CONFLICT DO NOTHING`) and deletes at most 1000 expired rows using an ordered,
`SKIP LOCKED` expiry pass. Failure retains pending records until a later cycle;
overflow is still bounded. Already expired pending records are not reinserted.
Independent minute timers may make a completed minute visible one cycle later.
When collection is off, physical cleanup stops, but reads always reject expired
rows regardless of whether cleanup has run.

`HistoryRepository.Read` supports steps of 1 minute, 5 minutes or 1 hour, ranges
of at most seven days and at most 1000 output points across all generations.
Oversized requests return `ErrHistoryRange`, never a silently truncated series.
Rows stream into bounded accumulators (at most 1000 × step-in-minutes input rows);
payloads and final numeric/size bounds are validated. No history read starts the
optional collector. Restart generations remain separate points; gaps remain gaps.

Aggregation merges request counts and fixed histograms before deriving quantiles,
adds CPU time and measured durations before computing cores, and retains gauge
last/min/max with actual observation counts. Cumulative pool/GC values and sparse
DB values are the latest observations with original timestamps. The existing
15-second aggregator conservatively omits HTTP/CPU windows that straddle a nominal
minute boundary: it does not invent a proportional count split. Timer jitter and
startup can therefore produce partial minutes with less than 60 seconds of
coverage. Consumers must show coverage/status, not assume every point is complete.

## Verification

Deterministic tests exercise cadence, no catch-up, resets, independent size
failure, deadline/cancellation, detached snapshots, a ten-record overflow queue,
exact acknowledgement during concurrent enqueues, nonblocking HTTP/process
sampling and generation-aware aggregation. App tests check explicit disabled
construction and lazy enabled construction.

Real PostgreSQL tests require `CONTRACTOR_TEST_DATABASE_URL`. They use isolated
schemas, and the optional visibility fixture requires a test superuser to create
and drop its own login. They verify normal/all-stats roles, idle transactions,
pool isolation, 750/1500 ms statement timeouts, 100 ms lock timeout, 250 ms acquire
timeout, earlier cancellation, reconnect, restored settings/idle transactions,
idempotent writes and DB constraints. A three-generation seven-day fixture checks
range/expiry query plans, hourly aggregation and the 1000-row cleanup bound.

```sh
go test -race -count=1 ./internal/performance ./internal/app
test -n "$CONTRACTOR_TEST_DATABASE_URL" && go test -race -count=1 ./internal/performance ./internal/persistence/postgres
```
