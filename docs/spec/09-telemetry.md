# 09 — Telemetry

Status: **Draft**  
Depends on: [02](02-domain-contracts.md), [04](04-database-runtime.md), [07](07-agent-runtime-and-a2a.md)

## Design

Telemetry is a derived observability view, not another correctness store. The
default design avoids persisting the same activity once as ADK session history
and again as a row-per-event Contractor trace.

Sources have an explicit precedence:

1. RunState, run/task/Attempt records, the budget/model-invocation ledgers,
   outbox, artifact metadata and the append-only security audit log provide
   authoritative lifecycle, model-resolution, usage and audit facts.
2. ADK-backed adapters keep their detailed invocation history in ADK sessions.
   Contractor reads a known session through an authorized, redacting,
   paginating adapter over the supported session-service API or converts selected
   callbacks into aggregates; it does not copy every ADK event into a second
   PostgreSQL table.
3. Non-ADK strategies may emit framework-neutral progress/diagnostic events
   correlated to the boundary Attempt. These remain best effort.
4. OpenTelemetry spans are sampled and sent through the process OpenTelemetry
   SDK/exporter to an external Collector/backend when detailed tracing is
   required. Raw high-cardinality spans are not stored in the Contractor
   database by default.

PostgreSQL projection and OpenTelemetry export are separate boundaries:

```text
domain/adapter signals → bounded TelemetryProjector → PostgreSQL rollups/samples
instrumentation         → OTel SDK batch processor → external OTel Collector
```

The projector buffer owns no database connection. Its sink writes bounded
rollups, indexes or deliberately retained samples in short transactions through
the process-wide `AsyncEngine`; it does not become the default sink for every
model token, tool event or span. ADK built-in instrumentation exports through
the configured OTel path rather than being copied through a Contractor
row-per-event callback. A selected ADK callback may feed a bounded aggregate or
explicitly retained sample, but not a second raw event stream.

Derivation from one ADK session is incremental and bounded. The adapter stores a
stable public cursor/checkpoint when the session API supplies one. Without such
a cursor it uses callback-maintained aggregates or performs at most one capped
terminal scan, records the scanned bound/checkpoint and marks truncated detail;
it never repeatedly rescans the full session. Direct ADK-table queries remain
forbidden.

Security audit is not telemetry. `AuditRepository` appends immutable,
tenant/principal-scoped records for the security-sensitive facts required by
spec 12. Audit records have their own authorization, integrity and retention
policy and are never sampled or dropped by telemetry backpressure. A state
change and its required audit fact share a transaction when they have the same
durable owner; otherwise the operation's documented fail-closed/audit-recovery
policy applies.

## Event identity

Every persisted diagnostic sample has a globally unique `event_id`, timestamp,
`trace_id`/`span_id`, source process and optional `run_id`, `task_id`,
`attempt_id`, `agent_id`, Planner strategy identity, Worker strategy identity
and ADK invocation/session IDs when present.
The sink enforces unique `event_id` for idempotent batch retry.

Rollups use a deterministic source/window/metric key and a source checkpoint so
replaying an input batch cannot increment a metric twice. Evaluation tooling
groups child Runs externally and uses durable run results, usage ledgers and
evaluation artifacts; it never depends on sampled or dropped telemetry.

## Requirements

- **TEL-001** — Queue capacity MUST be bounded by event count and/or bytes.
- **TEL-002** — Execution producers MUST NOT await PostgreSQL on every event or
  synchronously duplicate ADK/session events. Enqueue latency has a small
  configured upper bound.
- **TEL-003** — Backpressure policy MUST prefer dropping sampled/debug records
  before lifecycle/error records and MUST increment a dropped-events metric.
- **TEL-004** — Replaying a sample or aggregate batch MUST NOT create a
  duplicate sample or double-count a rollup.
- **TEL-005** — A PostgreSQL projection flush uses the shared `AsyncEngine`, a
  short transaction and a configurable batch size/time interval.
- **TEL-006** — Database outage MUST not create unbounded memory growth or stop
  run correctness. Recovery retries use bounded exponential backoff.
- **TEL-007** — Prompt, response and tool argument content capture is disabled
  by default. Explicit enablement still applies secret/PII redaction to both
  retained samples and ADK-session diagnostic reads.
- **TEL-008** — Control Plane observability queries MUST combine authoritative
  lifecycle data with optional ADK-session detail and retained diagnostic
  samples. Session detail MUST be tenant/run authorized, normalized to v2
  diagnostics, redacted before return and separately size-limited. Potentially
  large detail collections are paginated by run, task, attempt, trace and time
  range; raw ADK DTOs MUST NOT cross the API boundary.
- **TEL-009** — Retention deletion MUST be chunked and MUST NOT delete run
  state or execution artifacts.
- **TEL-010** — Shutdown performs one bounded flush and reports any remaining
  dropped records without delaying process exit indefinitely.
- **TEL-011** — Normalized lifecycle and usage projection schemas MUST be
  identical for ADK and non-ADK Worker strategies. Framework-specific
  identifiers are optional diagnostic attributes, never required correlation
  keys. The authoritative accepted usage remains the Server Attempt/budget
  ledger value, not a telemetry row.
- **TEL-012** — ADK session derivation MUST use a stable public cursor/checkpoint
  when available. Otherwise it MUST use callback aggregates or one capped
  terminal scan per session and expose truncation; periodic full-session rescans
  and direct ADK-table queries are forbidden.
- **TEL-013** — ADK events MUST be accessed only through documented ADK service
  or callback APIs and MUST NOT be copied row-for-row into Contractor telemetry
  by default. Direct queries against ADK-owned tables are forbidden.
- **TEL-014** — Run/task/Attempt transitions, accepted usage, model invocation
  provenance, artifact facts and security audit facts MUST be queried from their
  durable repositories or a rebuildable projection with those declared inputs,
  not inferred solely from lossy telemetry.
- **TEL-015** — Detailed diagnostic timelines are best effort. Queries MUST
  identify missing/dropped detail while still returning the complete
  authoritative lifecycle assembled from correctness records.
- **TEL-016** — Retained raw samples, aggregate dimensions and ADK diagnostic
  lookups MUST have explicit size/cardinality limits and retention independent
  of RunState and business artifacts.
- **TEL-017** — PostgreSQL telemetry projection MUST use bounded concurrency and
  yield to correctness traffic. Pool pressure drops/defers optional detail
  before it delays RunState, result, cancellation or artifact commits.
- **TEL-018** — Raw/sampled spans MUST use the configured OTel SDK/exporter and
  external Collector path. `TelemetryProjector` MUST persist only bounded
  aggregates, indexes and explicitly retained samples, and MUST NOT receive a
  row-for-row mirror of ADK sessions or OTel spans.
- **TEL-019** — `AuditRepository` MUST be append-only to runtime roles, retain
  actor/action/target/outcome/correlation/provenance and integrity metadata, and
  enforce tenant/operator authorization and an explicit retention policy.
  Audit writes MUST NOT enter the lossy telemetry queue.
- **TEL-020** — Observability queries MUST assemble authoritative lifecycle from
  RunState/run, Attempt, budget/model-invocation, artifact and audit owners.
  Removing telemetry projections, OTel data or ADK diagnostic detail MUST leave
  that lifecycle, requested/resolved model evidence and accepted usage unchanged.

## Acceptance

1. With PostgreSQL unavailable, a sustained event stream stays within the
   configured memory bound and Worker execution completes or fails normally.
2. Retrying the same telemetry batch produces one retained sample per event ID
   and does not increment any rollup twice.
3. A complete authoritative run lifecycle is reconstructed from domain records;
   available diagnostics are joined by correlation IDs through public adapters
   and explicitly report any sampled/dropped gaps.
4. Default telemetry contains no configured canary secret from prompts, tool
   inputs, environment variables or errors.
5. Graceful shutdown flushes available records within its deadline.
6. One ADK and one non-ADK strategy run expose the same normalized
   lifecycle/usage fields by durable run/Attempt identity without
   framework-specific queries.
7. Generating a large ADK event fixture creates no row-for-row copy in the
   Contractor telemetry table; only configured aggregates/samples grow, within
   declared bounds.
8. Deleting retained diagnostic samples or making ADK session detail
   unavailable does not change run status, accepted usage or audit results.
9. A telemetry/ADK-event load fixture remains within its database-write and
   connection budgets while correctness transactions meet their acquisition
   bound.
10. A raw ADK/OTel span fixture reaches the configured external Collector while
    Contractor PostgreSQL grows only by configured aggregate/sample bounds.
11. An unauthorized caller cannot request ADK session detail; an authorized
    response is paginated, uses normalized v2 DTOs and contains no configured
    canary secret after redaction.
12. Run, Attempt, budget/model-invocation, artifact and audit repositories
    reconstruct the full authoritative lifecycle and resolved-model evidence
    after Contractor telemetry rows and external OTel data are deleted.
13. Repeated diagnostic polling of a large ADK session advances a stable public
    cursor without rereading old pages; when the service exposes no cursor, one
    capped terminal scan or callback aggregate is used and truncated detail is
    reported explicitly.
