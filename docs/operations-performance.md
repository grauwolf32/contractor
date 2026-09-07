# Operations performance

Contractor Server includes bounded operational observations for a small
single-VM deployment. The feature needs no external metrics service and is
enabled by default. It is observational only: it cannot change placement,
budgets, Workflow outcomes, finalization, write fencing or allocation release.

## Startup configuration

The settings are process-level and require a Server restart:

| YAML `ServerConfig.spec` | CLI | Environment | Default |
| --- | --- | --- | --- |
| `performanceMetrics` | `--performance-metrics` | `CONTRACTOR_PERFORMANCE_METRICS` | `true` |
| `pprof` | `--pprof` | `CONTRACTOR_PPROF` | `false` |
| `pprofListen` | `--pprof-listen` | `CONTRACTOR_PPROF_LISTEN` | `127.0.0.1:6060` |

For example:

```yaml
apiVersion: contractor/v1alpha1
kind: ServerConfig
spec:
  performanceMetrics: true
  pprof: false
  pprofListen: 127.0.0.1:6060
```

CLI values override environment values and the configuration file. The metrics
and profiling switches are independent. With metrics disabled, Server does not
install performance HTTP middleware, allocate its history ring or writer queue,
start collection timers, open the diagnostic PostgreSQL pool or ask new Runtime
allocations for resource measurements. Health/readiness, execution metrics,
budget enforcement and report ingestion continue normally. Retained history is
still readable until expiry.

`pprof` owns a different loopback-only listener. Enabling that listener does not
start CPU or trace recording. See [Go profiling](go-profiling.md) for its exact
routes, capture bounds and SSH-tunnel use.

## What Operations shows

Authenticated operators can open `/operations/performance` for current Server,
HTTP, working-pool and PostgreSQL observations, plus bounded charts. Current
process/HTTP/pool frames are sampled every 15 seconds and kept in memory for at
most one hour (240 frames and 8 MiB). PostgreSQL statistics are sampled every 60
seconds, database size every 300 seconds, and minute aggregates expire after
seven days. History queries return at most 1,000 points.

`/operations/allocations/completed` shows terminal allocation resource summaries
after authoritative release. The same projection appears on the owning Run's
Stage attempt. Reports use the existing 30-day execution-report retention and
are owner-scoped in PostgreSQL. The values cover the entire Runtime Agent
process during that allocation, not child processes, Podman containers, host
load or a guaranteed true RSS peak. Sequential allocations on one Runtime are
measured independently, but retained Runtime process memory can still affect a
later allocation's boundary RSS.

HTTP metrics use only fixed public/private, method and status-class dimensions.
They retain no URL, Run ID, user ID, header, SQL or error text. Performance,
health, readiness and profiling requests are excluded so an open Operations tab
does not create observer feedback.

## Reading missing data

Missing is never displayed as zero. Each group has its own observation time,
coverage and `ok`, `partial` or `unavailable` state. Two missed group intervals
make a value stale in the UI. Restarts create a new generation and charts do not
join lines across generations. A DB outage, missing privileges, disabled
PostgreSQL statistics, counter reset, skipped sample or writer overflow is shown
as a bounded reason or gap while unrelated groups continue.

The diagnostic PostgreSQL connection is lazy and independent of the working
pool. A normal database role is supported, but it may not see other sessions;
those connection values are partial rather than falsely zero. An operator may
grant `pg_read_all_stats` for fuller visibility, but Contractor never grants it.
Database size is allocated database content, not free disk space. Dead tuples
are estimates, not a bloat verdict.

If Runtime resource capability is absent, history says `unsupported`. Metrics
disabled for an allocation says `disabled`; Runtime loss or a missing report is
`unavailable`. Resource-report failure cannot change a successful or failed
execution outcome.

## Upgrade and rollback

Upgrade Server before Runtime Agents. New Server accepts old Runtime
registrations, records them as resource-metrics unsupported and omits the
optional allocation request. After Runtimes advertise version 1, Server requests
the pinned resource policy. An old Server may reject the new registration field,
so rollback across the protocol change must be coordinated; do not rely on a
silent bidirectional fallback.

Disabling metrics after restart affects only newly prepared allocations. A
recovered existing allocation retains its pinned request policy and may still
return its final resource block. Disabling collection does not immediately
delete seven-day history or 30-day allocation reports.

## Verification and measurement

Run the focused gate against a disposable PostgreSQL database:

```sh
CONTRACTOR_TEST_DATABASE_URL='postgres://...' make test-performance-metrics
make benchmark-performance
```

The benchmark command compares identical isolated metrics-off/on and pprof
off/idle/active workloads. Its figures are host-specific evidence, not a
universal overhead guarantee. The recorded V32 run and raw samples are in the
[release review](reviews/2026-09-06-performance-v32-release.md).
