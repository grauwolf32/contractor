# 11 — LLM Proxy contract

Status: **Draft**  
Depends on: [02](02-domain-contracts.md), [03](03-adk-integration.md)

## Boundary

LLM Proxy is external infrastructure shared by Server and Agents. Contractor
implements only its client configuration, request metadata, policy and health
checks. Provider credentials, routing and model serving belong to the Proxy.

## Requirements

- **LLM-001** — All production model calls from Planner and Worker strategies,
  including Agent-internal planning, MUST use the configured Proxy base URL.
  Direct provider endpoints are forbidden.
- **LLM-002** — Contractor configuration uses stable model aliases rather than
  provider-specific model IDs in domain contracts.
- **LLM-003** — Requests MUST propagate trace context and the non-secret
  `run_id`, `task_id`, `attempt_id`, caller role and budget class as supported
  metadata.
- **LLM-004** — Every enabled model-backed Server/Agent strategy MUST
  authenticate to Proxy without receiving underlying provider credentials.
- **LLM-005** — Connect, response, stream-idle and total invocation deadlines
  MUST be bounded and aligned with the enclosing run/attempt deadline.
- **LLM-006** — Retry policy MUST distinguish pre-request/connect failures from
  ambiguous or partially streamed responses. Retries consume the run budget.
- **LLM-007** — Usage and cost data returned by Proxy MUST be normalized into
  the authoritative Planner/Attempt budget accounting path. Provider-incurred
  charge and fence-dependent accepted-result usage are distinct: a lost, failed
  or stale Attempt still settles or conservatively charges its invocation
  evidence/reservation, but that usage is not attributed to a replacement
  Attempt or used to accept the stale result. Accepted Worker totals are
  cross-checked against `WorkerResult.usage`; telemetry may project only derived
  aggregates or deliberately retained diagnostics and is not an accounting
  owner.
- **LLM-008** — Prompts, responses, auth headers and provider errors MUST pass
  through configured redaction before application logs or telemetry storage.
- **LLM-009** — Proxy outage MUST produce a stable retryable error code; policy,
  not the model adapter, decides whether to create a new Attempt.
- **LLM-010** — Every Planner/Worker model call MUST carry a stable model
  invocation ID and idempotency metadata. Contractor MUST still reserve and
  reconcile budget because the Proxy may not guarantee billing deduplication.
- **LLM-011** — Model calls made while a Worker alternates planning and working
  MUST carry the enclosing Attempt identity, consume its Worker budget and be
  aggregated into its terminal `WorkerResult.usage`.
- **LLM-012** — A Worker strategy MUST receive only an Attempt-scoped model
  client. It MUST reject new calls after local cancellation or the effective
  WorkerJob deadline and MUST interrupt an active stream within the cancellation
  grace period. Server-side fence loss still invalidates result/usage acceptance
  even when a partitioned Agent has not yet received cancellation.
- **LLM-013** — Before sending, the process MUST append/update a durable model
  invocation record containing stable invocation ID, caller/run/task/Attempt,
  requested model alias and policy/budget identity. Completion MUST add the
  Proxy-resolved provider/model identity, opaque provider request/reference when
  returned, terminal or `ambiguous` status and normalized usage/cost. Prompt and
  response content are not required in this ledger.
- **LLM-014** — Model invocation records and the Server Attempt/budget ledger are
  authoritative evidence for model selection and accounting. Telemetry/session
  detail may correlate by invocation ID but MUST NOT be the only source. An
  ambiguous invocation remains explicit and is reconciled conservatively; it is
  never rewritten as zero usage because telemetry is absent.
- **LLM-015** — Agent may append/update only invocation records scoped to its
  authenticated Attempt and cannot mark usage accepted or settled. Server reads
  those records, validates terminal WorkerResult aggregates and alone commits
  accepted Attempt/budget settlement.
- **LLM-016** — A new Agent model invocation MUST check the Server-owned
  Attempt operation-start gate and insert its immutable request identity in the
  same short transaction before contacting Proxy. Loss/cancellation/fencing
  seals or revokes the gate before overlapping retry. A non-open gate forbids
  new Proxy I/O but permits bounded monotonic terminal/reconciliation
  observations for an exact pre-existing invocation.
- **LLM-017** — Invocation request headers and scope MUST be immutable after
  creation; status, provider identity/reference and usage observations advance
  only through versioned monotonic transitions. Server settlement MUST pin the
  exact evidence version/hash it read and MUST NOT convert absent or ambiguous
  evidence to zero.
- **LLM-018** — Registration replacement MUST reject the old generation's new
  invocations even if it copies current registration fields. A distinct
  persistence capability or exact record-scoped recovery grant may only
  finalize/reconcile a named existing invocation. Exact-reference status/read/
  cancel I/O declared by that reconciliation protocol is allowed, but another
  billed/model request and usage acceptance/settlement are not.
- **LLM-019** — Every Attempt-scoped invocation record MUST contain an immutable
  `max_charge_reservation` in the normalized dimensions enforced by its
  `WorkerJob` budget. It covers the maximum provider-billable exposure of every
  transmission permitted under that invocation ID, including configured
  retries after an ambiguous response. Before any Proxy I/O, one transaction
  MUST both pass the Attempt operation-start gate and reserve that maximum
  in the Agent-owned invocation-reservation subledger beneath the immutable
  Server-reserved WorkerJob budget; Agent receives no write access to the Server
  budget row. The transaction MUST serialize concurrent callers and reject the
  invocation unless, in every dimension, settled/observed charges plus all
  unreleased invocation reservations (including open, terminal-but-unsettled and
  ambiguous records) plus the new maximum remain within the Attempt reservation.
  A conclusively reconciled invocation replaces its held maximum with its actual
  charge and releases only the proven-unused remainder. An open, absent or
  ambiguous result retains its complete maximum until authoritative
  reconciliation, or policy settles it at that maximum; timeout, Agent loss,
  cancellation and telemetry absence MUST NOT release it.

## Acceptance

1. A network test fails if either process contacts a provider endpoint directly.
2. Correlation metadata appears in a fake Proxy request and authoritative usage
   accounting; when enabled, an optional aggregate/span carries the same IDs.
3. Deadline and cancellation stop a streaming response within the configured
   grace period.
4. Retry and usage accounting never exceed the configured run budget silently.
5. Canary provider secrets do not appear in logs, artifacts or telemetry.
6. Repeating a call after an ambiguous disconnect preserves invocation identity
   and charges or reconciles the run budget conservatively.
7. Multiple calls from a Worker strategy that internally alternates planning
   and working remain correlated to and bounded by its one boundary Attempt
   reservation.
8. After cancellation or deadline, the scoped model client rejects a later call
   and stops an active stream; loss of Server connectivity cannot make a stale
   result or its usage settle under a replacement fence.
9. A completed fake-Proxy call records requested alias, exact resolved
   provider/model identity, stable invocation ID/provider reference, terminal
   status and normalized usage; an injected disconnect records `ambiguous` and
   follows conservative reconciliation.
10. Deleting Contractor telemetry, external OTel data and optional ADK session
    detail leaves accepted usage and the exact requested/resolved model evidence
    queryable from the invocation and budget ledgers.
11. Agent database credentials cannot alter another Agent's invocation records
    or Server settlement state; Server detects a WorkerResult aggregate that
    disagrees with its scoped invocation evidence.
12. Proxy bills an invocation and Agent is lost before `WorkerResult`. Server
    seals or revokes the operation-start gate and charges/reconciles the exact
    invocation snapshot without accepting stale result usage or attributing it
    to the replacement Attempt.
13. A response arriving during the bounded post-terminal window can seal the
    exact existing invocation monotonically; a new invocation after the gate
    becomes non-open sends no Proxy request, and a forged current-generation row
    value cannot substitute for persistence-capability proof.
14. Two or more concurrent calls whose maximum charges cannot all fit the
    remaining Attempt reservation are forced through the same budget guard:
    only the fitting set commits invocation reservations and reaches the fake
    Proxy, while every rejected call sends no request. If a committed call is
    made ambiguous, its full maximum remains held and blocks otherwise
    over-budget calls until authoritative reconciliation or maximum-charge
    settlement; the test covers a retry whose possible duplicate provider
    charge was included in that immutable maximum.
