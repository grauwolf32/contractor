# 22 — Operations performance metrics and Go profiling

Status: **Draft specification — implementation pending**

Depends on: [02](02-runtime-and-a2a.md),
[04](04-execution-lifecycle-and-metrics.md),
[06](06-server-ui-and-operations.md),
[18](18-run-and-workspace-lifecycle-controls.md).

## Purpose and ownership

Operations displays current Server/PostgreSQL performance, a bounded history,
and resource summaries for completed allocations. Contractor owns collection,
storage and presentation. No Prometheus, Grafana, exporter service or external
time-series database is required.

Server collects its own process/HTTP/pool measurements continuously while the
feature is enabled. Runtime collects process resources only during an explicitly
requested allocation and returns aggregates in the existing finalize/abort
response. Go profiling is a separate, opt-in diagnostic facility.

This document owns performance collection and profiling. Execution reports,
model/tool/token accounting, execution budgets, finalization and report retention
remain owned by [04]. Performance telemetry cannot affect placement eligibility,
semantic outcomes, lease confirmation, write fencing or release eligibility.

The first implementation targets Linux, including the existing single-host
deployment. Unsupported process measurements are unknown with a bounded reason;
they never prevent startup or execution. Host-wide CPU, container/cgroup limits,
child-process resources, Podman resources, Python profiling, live Runtime polling,
per-allocation time series, SQL statement profiling and external exports are
outside this increment.

## Startup switches

| Flag | Environment | Default | Meaning |
| --- | --- | --- | --- |
| `--performance-metrics` | `CONTRACTOR_PERFORMANCE_METRICS` | `true` | Enable new Server/DB collection, performance history writing and requests for allocation resource summaries |
| `--pprof` | `CONTRACTOR_PPROF` | `false` | Enable the separate Go diagnostic listener |
| `--pprof-listen` | `CONTRACTOR_PPROF_LISTEN` | `127.0.0.1:6060` | Numeric loopback IP and TCP port for that listener |

Boolean environment values accept `true` or `false`. Explicit CLI values take
precedence; invalid effective configuration fails startup with a safe error.
Switches are immutable for one Server process; changing them requires restart.
There is no writable Operations setting or browser action for these switches.
They are not Workflow, AgentTemplate or RuntimeConfig authoring fields.

With `--performance-metrics=false`:

- do not install the new HTTP accounting middleware, allocate sample/history
  buffers, start sampler/writer/retention timers or open the diagnostic DB pool;
- omit performance collection requests from newly prepared allocations;
- return explicit `enabled: false` from the current-performance API, with no
  manufactured current measurements;
- keep existing retained history available through explicitly requested bounded
  reads; disabling does not delete data or expose expired rows;
- pause this feature's physical time-series retention cleanup until re-enabled.
  Ordinary Run/Project deletion and existing execution-report cleanup continue;
- preserve existing execution metrics, budgets, OTLP trace adapters, logging,
  health/readiness behavior and immutable report ingestion.

A previously prepared allocation retains its pinned collection instruction even
if Server restarts with metrics disabled. Recovery may accept that allocation's
final report through the existing report path; there is no new remote command
to toggle collection halfway through an allocation. The API/UI distinguish
current collection policy from the policy recorded for a historical allocation.

`--pprof` is independent: all four metrics/profiling boolean combinations work.
Disabling either facility does not disable Go's ordinary runtime/GC bookkeeping
or the runtime's default heap sampling, and does not promise zero total
diagnostic overhead.

## Collection schedule and bounded records

| Data | Schedule | Storage |
| --- | --- | --- |
| HTTP accounting | Each admitted HTTP request and completion | Fixed in-memory counters and histograms |
| Server process, HTTP snapshot, working pgx pool | Every 15 seconds | Last 60 minutes in memory |
| PostgreSQL activity and cumulative statistics | Every 60 seconds | Latest typed snapshot and minute history |
| PostgreSQL database size | Every 5 minutes | Latest typed snapshot and minute history |
| Server/DB history write | Every minute | Minute aggregates, 7-day retention |
| Allocation process resources | At start/end and every 15 seconds while owned | Constant-size allocation accumulator and final report |

Take initial measurements after initialization without delaying readiness. The
first CPU/rate measurement needs two observations; an absent baseline is unknown.
Use monotonic elapsed time for rates/durations and UTC timestamps for display and
history. Skipped ticks are not replayed in a catch-up burst.

Every sample group carries its own `observedAt`, last attempt/result, coverage
and bounded status (`ok`, `partial`, `unavailable`). Never relabel a cached
5-minute database-size value as a fresh 15-second measurement. The UI derives
staleness from the last successful observation and the group's declared interval;
two missed intervals mark it stale. Error codes never contain SQL, DSNs,
credentials, filesystem paths or raw driver messages.

Initial limits: at most 240 completed 15-second frames plus one active frame,
32 KiB encoded per frame/minute record, and 8 MiB for retained live frames.
Prefer compact typed records. Exceeding a byte/count limit evicts oldest live
frames or omits optional details with explicit partial coverage; core counters
must not silently become zero. No unbounded labels or per-request records.

## HTTP, process and pool semantics

Instrument both public and private Server HTTP listeners. Use fixed dimensions:
`surface` (`public`, `private`), a fixed HTTP-method allowlist with `other`, and
status class (`1xx` through `5xx`, or `no_response`). Histogram totals are per
surface. Route-specific breakdowns are deferred; raw paths, query strings,
user/Run/allocation IDs and exception text never become dimensions.

Exclude health/readiness, performance-read APIs and the profiling listener from
the workload HTTP aggregates. In-flight requests increment at entry and decrement
exactly once on completion/panic/disconnect. Ordinary response count, status,
duration and histogram are recorded at completion, including implicit HTTP 200
and safe handling of failures before headers. Preserve streaming, flushing,
HTTP hijacking, cancellation and WebSocket behavior. WebSocket upgrades count
as handshakes; connection lifetime is not ordinary request latency. HTTP errors
mean 5xx/no-response; 4xx are visible separately, not treated as server failures.

HTTP histogram bucket upper bounds in seconds are
`0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, +Inf`.
Retain count/sum/buckets and derive approximate p50/p95/p99 when reading.
An estimate in the overflow bucket is displayed as `>60s`, not a fabricated
finite quantile. Empty intervals have zero completed requests and unknown
latency quantiles. Merge buckets/counts, never average percentile values.

Process readings include user/system CPU seconds, RSS bytes, Go heap live bytes,
goroutine count, completed GC cycle count and cumulative GC pause histogram.
Use selected runtime counters and inexpensive OS process readings; collection
does not force GC or build heap/stack profiles. CPU utilization is
`delta(user + system CPU seconds) / monotonic elapsed seconds`, displayed in
used CPU cores. It is not host load or a percentage of a container CPU limit.

Read the application's existing `pgxpool.Stat()` in memory: acquired/idle/total/
max connections, successful acquisition count/time, waits on an empty pool and
cancelled acquisitions. A mean successful acquisition time is not a p95 and
does not include cancelled attempts. SQL latency/error instrumentation for all
application statements is deferred; pool, probe and HTTP latency remain distinct.

Readiness remains an independent dependency check: `/healthz` reports process
liveness; `/readyz` uses a bounded check of the working DB connection path and
returns 503 when it is unusable. Its total DB wait is at most one second and
does not depend on metrics being enabled or on a cached minute-old DB sample.

## PostgreSQL collector

Use one additional pool with `MaxConns=1`, `MinConns=0`, the existing database
connection configuration and a fixed `application_name` identifying performance
maintenance. It is separate from the work pool and from LISTEN connections.
Collector startup/reconnect failure marks DB metrics unavailable; it must not
make an otherwise working Server fail startup. Retry only on a subsequent
scheduled cycle. No extension, superuser, server setting change or automatic
role grant is required.

Every minute execute fixed aggregate SELECTs restricted to `current_database()`:

1. `pg_stat_database`: `xact_commit`, `xact_rollback`, `deadlocks`, `temp_files`,
   `temp_bytes`, `blks_read`, `blks_hit` and `stats_reset`.
2. `pg_stat_activity`: client connections by state, count waiting with
   `wait_event_type = 'Lock'`, longest open transaction and longest idle-in-
   transaction age. Exclude the collector's own backend. Do not fetch query text.
3. `pg_stat_user_tables`: aggregate estimated live/dead tuple counts and available
   manual/autovacuum activity timestamps/counts. These are estimates, not a bloat
   measurement or a declaration that the database needs manual vacuuming.

Activity/lock values are point observations; short waits between samples can be
missed. Database counters are cumulative and include other users of the same
database as well as Contractor's own monitoring/history traffic. Compute rates
from actual observed intervals; `stats_reset` changes or counter decreases
invalidate the delta. Buffer hits describe PostgreSQL's buffer cache, not the
OS cache; rollbacks are not a count of application errors. Do not enable
`track_io_timing`, `pg_stat_statements` or other extra instrumentation.

Every five minutes execute separately:

```sql
SELECT pg_database_size(current_database());
```

This measures database files, including tables/indexes/TOAST, across its
tablespaces; it is neither free filesystem space nor total cluster/WAL usage.
PostgreSQL inspects directory/file metadata, so it is not a constant-time
counter read. It must not gate the minute collector or be recomputed on API reads.

The initial DB budgets are: connection/acquisition at most 250 ms per attempt,
lock timeout 100 ms, statement timeout 750 ms for statistics and 1500 ms for
size/history maintenance, client query deadline 1 s and 2 s respectively,
and at most 5 s for a complete minute collection cycle. Earlier caller/shutdown
deadlines win. Reuse the database budget/cleanup mechanisms from V29-007;
transaction-local overrides cannot leak into the working pool. End every
transaction before the next cycle so cached PostgreSQL snapshots do not persist.
Size and maintenance operations are separately scheduled, serial and bounded;
they never create a queue of collector queries or immediate retries.

Ordinary DB roles can see restricted session details only for their own/member
roles. Detect visibility/configuration limitations and mark affected aggregates
partial/unknown instead of counting hidden state as zero. Document the optional
operator grant of `pg_read_all_stats` for full session visibility; never issue
it automatically. Missing privileges or disabled PostgreSQL statistics do not
disable independent HTTP/process/pool metrics.

## Time-series history and Operations API

Use a dedicated forward-only migration for `performance_minutes`, separate from
Stage execution truth. Key rows by `(server_generation, minute_start)` with
schema version, observation coverage, typed bounded aggregate payload and expiry.
A Server boot creates a new generation; counters from different boots are never
subtracted. A partial final minute may be written during bounded shutdown.

The writer uses the single performance DB pool, writes minute batches with
idempotent keys and keeps at most ten pending minute records. On overflow drop
oldest unwritten minutes and increment a loss diagnostic. Request handlers and
samplers never wait for the writer. Cleanup runs at most once a minute, deletes
at most 1,000 expired performance rows within the maintenance deadline and never
uses an unbounded DELETE/VACUUM. Expired data is excluded from reads regardless
of physical cleanup timing. Existing Run/Project deletion rules remove linked
allocation reports independently of this table.

Counts and histograms aggregate events; CPU uses CPU-time/elapsed-time deltas;
gauges preserve last/min/max and observed coverage. Sparse DB observations keep
their actual timestamps and are not counted repeatedly as new measurements.
Minute boundaries, partial windows and missing samples must have deterministic
tested behavior. Restart, writer loss and disabled periods produce gaps. Fine
history is volatile; minute history survives restart. An unavailable DB still
allows current/in-memory performance reads.

Proposed read-only endpoints under existing authenticated Operations authority:

- `GET /v1/operations/performance`: enabled state, generation, intervals,
  current groups, freshness and collector/writer diagnostics; no SQL on this path.
- `GET /v1/operations/performance/history?from=...&to=...&step=15s|1m|5m|1h`:
  bounded series, coverage and generation boundaries. `15s` is limited to the
  current process's last hour; durable queries are limited to seven days and
  at most 1,000 points. Reject excessive ranges/point counts explicitly.
- `GET /v1/operations/allocation-history`: durable terminal-allocation resource
  summaries with optional exact Run filter, default page size 50, maximum 100,
  descending `(finished_at, allocation_id)` keyset cursor and a pinned upper
  boundary for subsequent pages. Here `finished_at` is the owning StageExecution's
  durable terminal timestamp, available even without a Runtime report; a late
  report cannot reorder the history. Resource interval timestamps remain separate.
  Apply existing owner authorization in SQL.

New DTOs follow public API conventions and generated TypeScript contracts.
Current metrics use their own timestamps, not Control Plane registry revisions;
sampling must not advance live-registry cursors or emit heartbeat-like events.
Bounded history queries use normal authorized request DB budgets. They remain
available when collection is off, without starting the diagnostic pool.

## Allocation request and final resources

Extend private v2 registration with optional
`supportedPerformanceMetricsVersions: [1]`; omission/empty means unsupported.
Advertise version 1 only when the Runtime collector is implemented. A Server
must not filter placement based on this optional diagnostic capability.

When collection is enabled and the chosen Runtime supports version 1, add this
optional top-level field to its Server-generated `AllocationSpecV2`:

```json
"performanceMetrics": {"version": 1, "intervalSeconds": 15}
```

Omission disables sampling. Persist the Server decision (`requested`, `disabled`
or `unsupported`) alongside the allocation so final-report attribution and
restart recovery do not infer it from current process flags. Duplicate prepare
uses the same decision. It is operational metadata, not model-visible context
or secret-bearing adapter configuration.

Strict Go/Python DTOs, JSON schemas and fixtures must change together. New
Servers accept old registrations and omit the new request for them; new Runtimes
accept old prepare payloads with collection off. Old Servers may reject the new
registration field: document Server-first upgrade/coordinated rollback, not
transparent bidirectional compatibility or fallback that hides a protocol error.

Start measuring after accepting allocation ownership, before expensive prepare
work. Stop and freeze after bounded Worker/adapter teardown immediately before
the cached finalize/abort response is built. Failed prepare follows existing
cleanup/report delivery rules; metrics add no new required response or lifecycle.
Release occurs after the final report and is outside the measured interval.
Idle time before/after allocation and a later allocation never enter its sample.

Add optional `runtime.resources` to the existing `RuntimeReport`:

| Field | Meaning |
| --- | --- |
| `version` | `1` |
| `scope` | `runtime_process` |
| `status` | `complete`, `partial`, or `unavailable` |
| `reason` | Optional bounded enum: `unsupported_platform`, `read_failed`, `sampling_gap`, `counter_reset`, `invalid_report` |
| `durationSeconds` | Monotonic duration of this collection interval |
| `cpuUserSeconds`, `cpuSystemSeconds` | End-minus-start process CPU counters |
| `rssStartBytes`, `rssEndBytes` | Successful boundary RSS observations, if available |
| `rssPeakObservedBytes` | Maximum successful RSS observation inside this interval |
| `rssSampleCount` | Number of successful RSS observations, including boundaries |
| `maxSampleGapSeconds` | Largest uncovered interval, including missing boundaries |

Optional unknown numeric values are absent, not zero; values are finite and
nonnegative. CPU average is derived from available CPU deltas/duration, never
division by zero. Reject negative/NaN/infinite/overflowing values and inconsistent
sample/boundary/peak combinations at the telemetry boundary. `complete` means
both boundaries and scheduled measurements succeeded without an uncovered gap
greater than twice the 15-second interval; it never asserts the true RSS peak
was captured. CPU remains a boundary delta despite skipped intermediate samples.

Resource fields cover the entire Runtime process, including its service work
and retained memory from earlier allocations. They exclude subprocesses and
containers, including the implemented [21](21-podman-sandbox.md) sandboxes. The UI must
label that scope and call the RSS value an observed peak. Do not use the
process-lifetime RSS high-water mark as an allocation-local peak, trace Python
object allocations or force garbage collection.

Keep a constant-size accumulator and one cancellable sampler, with no overlapping
reads or leaked sampler after prepare failure, abort, watchdog drain or release.
Measurement failure changes only resource completeness, not Worker report
completeness or semantic outcomes. If the Runtime is lost before reporting,
Server records unknown resources; there is no periodic delivery or crash recovery
of this accumulator. No sampling happens while idle or when the request is absent.

Server trusts allocation identity/policy from its own envelope and validates the
optional block separately. Malformed/unrequested resources cannot poison an
otherwise valid report: omit the block, retain a safe diagnostic and continue
normal finalization. Resource data remains within the existing 1 MiB final-report
limit; no raw samples are returned. Duplicate finalization/abort returns the same
cached report and persistence never double-counts it.

Use existing 30-day terminal execution-report retention for allocation summaries.
History includes terminal allocation records even when their report is missing;
show `disabled`, `unsupported`, `pending`, `available`, `partial` or `unavailable`
from trusted policy and available reports, rather than silently omitting failures.
Old allocations without pinned policy are `unavailable` with a legacy reason.
Late reports follow [04]'s existing related-telemetry rules; no frozen Stage
outcome or StageMetrics is rewritten to populate history.

## Operations presentation

Add `/operations/performance` with Server, HTTP and Database cards/charts, time
ranges and explicit last-update, partial/stale/disabled states. Poll current
snapshots every 15 seconds only while visible, coalesce requests and cancel them
on navigation. Historical reads occur on range changes, with bounded refresh for
a range following the present. Browser tabs do not multiply collection work.

Add a Completed/history view alongside the existing current allocations page.
The live registry still removes released allocations. History comes from durable
Run/allocation/report joins, uses existing indexes plus narrowly required new
indexes, and shows Run link, logical Worker, outcome, duration, CPU seconds/cores,
observed RSS peak and completeness. It has no execution mutation controls and
never exposes payloads, SQL, secrets or raw profiles. Show resources in Run
attempt details through the same safe projection, without creating another store.

## Go profiling

`--pprof=false` creates no diagnostic listener and starts no CPU/trace/block/
mutex collection. Enabling it exposes a dedicated explicit HTTP mux on the
configured numeric loopback address only. Reject wildcard/non-loopback addresses
and invalid ports; an explicitly enabled listener that cannot bind fails startup
cleanly. Stop it through the same bounded Server shutdown lifecycle.

Public/private application muxes never serve `/debug/pprof/*`, regardless of
settings. Do not serve `http.DefaultServeMux`: importing `net/http/pprof` registers
global handlers as a side effect. Register only an explicit allowlist on the
diagnostic mux; a generic Index prefix must not bypass it. In particular, do not
expose `cmdline`, which may contain command-line credentials.

Allow an index, symbol lookup, CPU `profile`, `heap`, `allocs`, `goroutine`,
`threadcreate` and execution `trace`. CPU sampling starts only on request:
default 30 s, allowed 1..60 s. Trace default is 1 s, allowed 1..10 s. Validate
duration parameters on every supported timed/delta endpoint; reject invalid,
duplicate and out-of-range values rather than silently falling back. CPU/trace
requests share one active-session slot with no waiting queue; a conflict returns
409. Permit at most one additional instantaneous/delta profile request, also
without a waiting queue. Cancellation and shutdown release slots and stop any
active recording within the existing shutdown deadline.

Block and mutex profiling remain off; this version adds no sampling-rate flags
for them and does not alter `runtime.MemProfileRate`. Ordinary performance
collection never invokes these handlers or forces GC. Profile requests may be
expensive (including an explicitly requested heap GC), so their cost is measured
separately from background metrics and from merely enabling the idle listener.

Profiles stream to the diagnostic client; they are not stored in PostgreSQL,
execution reports or served through Operations. Document local use and SSH
tunnelling, including a bounded example:

```sh
contractor-server serve --performance-metrics=false --pprof=true
go tool pprof 'http://127.0.0.1:6060/debug/pprof/profile?seconds=30'
```

## Verification and delivery

Implement through V32-001..V32-008 in [the task index](../../tasks/index.yml):

| Task | Deliverable | Dependencies |
| --- | --- | --- |
| [V32-001](../../tasks/v32-001-performance-contracts-and-settings.yml) | Startup and public/private measurement contracts | Existing report, private-v2 and Operations contracts |
| [V32-002](../../tasks/v32-002-server-performance-collection.yml) | Server HTTP/process/pool collectors | V32-001 |
| [V32-003](../../tasks/v32-003-postgres-performance-history.yml) | PostgreSQL collector and minute storage | V32-002, V29-007 |
| [V32-004](../../tasks/v32-004-allocation-resource-sampling.yml) | Runtime allocation accumulator and final resources | V32-001, existing lifecycle hardening |
| [V32-005](../../tasks/v32-005-performance-api-and-allocation-history.yml) | Pinned allocation policy, ingestion and read APIs | V32-003, V32-004 |
| [V32-006](../../tasks/v32-006-operations-performance-ui.yml) | Operations charts and completed allocation history | V32-005, existing Operations UI |
| [V32-007](../../tasks/v32-007-opt-in-go-profiling.yml) | Separate bounded Go diagnostic listener | V32-001 |
| [V32-008](../../tasks/v32-008-performance-release-gate.yml) | Fault/integration/browser gate and overhead evidence | V32-006, V32-007 |

Required acceptance:

1. All four startup-switch combinations work; metrics off creates no new
   collectors, metric queues/timers or diagnostic DB connections; pprof off
   creates no listener. Existing execution accounting/budgets still work.
2. Fake-clock tests prove 15 s / 60 s / 300 s cadence, bounded memory/queues,
   non-overlap, truthful missing/zero/reset semantics and minute aggregation.
3. HTTP accounting preserves streaming/WebSockets, concurrent totals and
   quantiles while avoiding unbounded dimensions and observer-traffic feedback.
4. Disposable PostgreSQL tests cover limited roles, hidden activity, lock waits,
   counter resets, database outage, budgets, retention, restart and query plans.
5. Allocation tests cover short/long/sequential runs, unsupported/disabled
   collection, partial reads, failure/loss, retries, cleanup and old/new contracts.
6. Authenticated API/browser tests cover charts, disabled/stale/gap states,
   bounded queries, owner isolation and durable history after authoritative release.
7. Profiling tests cover loopback-only exposure, no public/private/cmdline route,
   finite durations/concurrency, valid readable output and cancellation/shutdown.
8. A reproducible enabled/disabled load comparison records CPU time, allocations,
   RSS, throughput, p95 latency and DB query/write counts, with toolchain/hardware/
   workload/repetitions. Measure idle pprof and active profiling separately.
   Establish before/after evidence, not an unmeasured universal overhead percentage.

## Reference behavior

- [PostgreSQL statistics](https://www.postgresql.org/docs/current/monitoring-stats.html)
  define cumulative versus current observations and visibility restrictions.
- [PostgreSQL database-size functions](https://www.postgresql.org/docs/current/functions-admin.html#FUNCTIONS-ADMIN-DBSIZE)
  and [their implementation](https://doxygen.postgresql.org/dbsize_8c_source.html)
  explain the separately scheduled file-size query.
- [Go HTTP profiling](https://pkg.go.dev/net/http/pprof) defines the standard
  diagnostic transport and on-demand profiles;
  [Go diagnostics](https://go.dev/doc/diagnostics) describes profiling overhead.
