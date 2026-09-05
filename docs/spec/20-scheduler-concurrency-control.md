# 20 — Workflow Scheduler concurrency control

Status: **Working agreement**

Depends on: [00](00-workflow-and-planner.md),
[02](02-runtime-and-a2a.md), [06](06-server-ui-and-operations.md),
[18](18-run-and-workspace-lifecycle-controls.md), and
[19](19-audits.md)

## 1. Purpose and boundary

Contractor exposes one durable Operations setting that limits how many
different WorkflowRuns the supported Server process may actively progress at
the same time. The setting is global because Runtime Agent capacity and the
Workflow Scheduler are deployment resources, not owner- or Project-local
resources.

```text
ordinary Run creation -----\
Project Run creation -------+--> ordinary WorkflowRun queue
Audit child Run creation ---/               |
                                             v
                              Workflow Scheduler execution lanes
                                             |
                                  maxConcurrentRuns
                                             |
                                             v
                              Runtime Agent placement/capacity
```

`maxConcurrentRuns` is the only Contractor setting for concurrent WorkflowRun
execution. It applies identically to standalone, Project, Evaluation, and Audit
child Runs. It does not introduce a second queue, an Audit execution pool, or
special scheduling based on Run labels.

The limit is not the number of records whose durable state is `running`.
Waiting Runs remain `running` while they lack Scheduler or compatible Runtime
capacity. The limit instead bounds concurrent Scheduler execution lanes, each
of which owns at most one durable Run claim while performing one bounded
progression attempt.

Reaching the limit does not reject ordinary Run creation. Newly created Runs
remain in the existing Queue read model until a lane and compatible Runtime
capacity are available; the setting is execution control, not a queue-length
quota.

One Run still has at most one active StageExecution. Concurrency in this
document is across different WorkflowRuns; it does not permit concurrent
Stages inside one Run or concurrent Tasks inside one Runtime Agent.

## 2. Durable setting

The authoritative singleton setting is:

```text
maxConcurrentRuns: integer, 1..32, default 1
revision:          positive monotonic integer
updatedAt:         UTC timestamp
```

The next forward-only PostgreSQL migration creates and seeds exactly one
scheduler-settings row with `maxConcurrentRuns = 1` and `revision = 1`. The
default preserves the current serial production behavior after upgrade. A
database check enforces the range, and a missing or invalid singleton is a
storage invariant failure rather than an invitation to guess a default at
runtime.

The upper bound of 32 is the deliberately finite single-host first increment,
not an estimate derived from the current number of agents. Raising it later
requires an explicit contract and load/recovery gate rather than an unbounded
integer accepted by old Servers.

The persistence model is a dedicated typed `scheduler_settings` singleton, not
a generic string/JSON settings bag. `Operations / Settings` is a UI grouping;
future settings retain their own typed validation, storage and API resources.

The setting is deployment state. It is not part of Workflow, RuntimeConfig,
ExecutionConfig, owner Queue control, Project, Run, or Audit snapshots. A Run
therefore does not pin the value that happened to exist when it was created.
Changing it affects future Scheduler admission for every owner and every
nonterminal Run.

`0` is deliberately invalid. Owner Queue Pause is the explicit way to stop new
normal Stage admission for that owner, while cancellation, recovery,
finalization, and release must continue to make progress.

## 3. Public Operations API

The singleton has a resource-oriented API:

```http
GET /v1/operations/settings/scheduler
PUT /v1/operations/settings/scheduler
```

There is no action endpoint, Run identifier, `force` query parameter, `POST`,
or `DELETE` form of this resource.

An authorized `GET` returns `200 OK`, `Cache-Control: no-store`, and a strong
ETag containing the decimal revision:

```json
{
  "maxConcurrentRuns": 2,
  "revision": "4",
  "updatedAt": "2026-09-05T12:00:00Z"
}
```

`PUT` is a complete idempotent replacement of the editable settings resource.
It requires `Content-Type: application/json`, the Operations capability, normal
browser CSRF protection, and exactly one strong `If-Match` revision:

```http
PUT /v1/operations/settings/scheduler
If-Match: "4"
Content-Type: application/json

{"maxConcurrentRuns": 3}
```

Unknown or missing fields, a value outside `1..32`, a weak/multiple/malformed
ETag, or an unsupported media type fail without mutation. A well-formed stale
revision returns `412 Precondition Failed`. A successful change returns the
complete new resource and ETag. Replacing the resource with its existing value
under the current revision returns that same resource without advancing the
revision or emitting a false change notification.

Authentication failures use `401`, missing Operations capability uses `403`,
an unsupported request media type uses `415`, malformed JSON/schema/header or
an out-of-range value uses `400`, and a valid but stale revision uses `412`.
These responses use the existing bounded public error envelope.

The mutation writes the ordinary safe administrative audit record with the
actor and old/new numeric values. Neither the public response nor logs expose
credentials or execution payloads.

## 4. Scheduler execution lanes

At startup the Scheduler reads the durable setting before admitting work. Its
production loop is a supervisor with at most `maxConcurrentRuns` execution
lanes. Each lane repeatedly:

1. claims at most one runnable WorkflowRun through PostgreSQL;
2. registers that Run for in-process cancellation;
3. renews the exact claim while progressing it;
4. executes one bounded Run/Stage lifecycle attempt;
5. releases the claim before selecting more work.

The existing `FOR UPDATE SKIP LOCKED` claim boundary remains authoritative:
different lanes cannot claim the same Run, and a claim cannot be silently
stolen before expiry. The claim ordering continues to prioritize cancellation
and rotate capacity-deferred work. A paused or capacity-ineligible Run releases
its lane after a bounded admission/placement attempt so it cannot indefinitely
occupy all execution capacity.

`RunOnce` remains a deterministic single-claim compatibility primitive for
focused tests and recovery tools and may perform one maintenance tick before
that claim. Production lanes use the extracted claim/progression primitive;
the supervisor owns maintenance, lane creation and resizing and never wraps
concurrent calls around the compatibility method.

All objects shared across lanes must be proven concurrency-safe. In particular,
Planner factories, allocation coordination, Artifact and credential services,
Run cancellation registration, clocks/ID generation, and test doubles may not
rely on the old production-loop serialization. PostgreSQL transaction and
claim boundaries remain the source of durable ordering.

## 5. Dynamic changes and restart

The database value is the desired limit. The Server applies a committed update
to its in-process supervisor and sends a wake hint; a bounded periodic refresh
also reconciles the value so correctness never depends on delivery of that
hint. Restart reads the same durable row before new work is admitted.

An increase starts additional lanes promptly. A decrease is a drain operation:

- no active Run or Planner call is cancelled merely because the limit fell;
- current lanes finish their bounded claim attempt and release it normally;
- the supervisor starts no replacement attempt while active lanes exceed the
  new limit;
- once drained, concurrency remains at or below the new value.

If a settings update commits immediately before process failure, the restarted
Server still observes it. The HTTP response does not promise that a newly added
lane has already acquired Runtime capacity before returning.

## 6. Pause, cancellation, and maintenance

Owner Queue Pause and the global concurrency setting solve different problems:

- Queue Pause controls normal Stage admission for one owner;
- `maxConcurrentRuns` bounds Scheduler work across all owners;
- Runtime placement decides whether the selected Stage's exact allocation set
  can run on currently eligible agents.

Paused owners must not prevent another owner from using available lanes.
Already admitted work drains under the existing Queue contract. Cancellation
of a Run currently owned by a lane interrupts that lane through the existing
in-process cancellation path; other cancelling Runs retain claim priority.

Allocation-loss polling, terminal allocation-release recovery, expired
telemetry cleanup, and comparable safety/recovery maintenance are not semantic
Workflow execution and do not consume an execution lane. They run through
bounded dedicated maintenance loops so lowering the limit or pausing an owner
cannot disable cleanup. Their control calls may overlap lane work without
allowing another StageExecution for the same Run.

Project deletion continues to durably request cancellation for every
nonterminal member Run. It neither changes nor bypasses the global limit, and
all cancellation/release phases must converge at every valid setting.

## 7. Runtime capacity

Each Runtime Agent process still supplies exactly one slot. One WorkflowRun
counts as one Scheduler lane even when its selected Stage needs several named
Workers and therefore several Runtime Agent slots. Consequently the setting is
an upper bound, not a promise of utilization:

```text
actual concurrent Runs <= maxConcurrentRuns
actual concurrent Runs is additionally constrained by compatible Agent slots
```

With two compatible idle agents and `maxConcurrentRuns = 2`, two independent
one-Worker Runs may execute together. A single two-Worker Stage may instead use
both agents while consuming one Run lane. Adding or removing Runtime Agents
does not automatically rewrite the configured limit.

## 8. Audit dispatch integration

AuditProfile has no execution-concurrency field. In particular,
`maxActiveRuns` is removed from its authoring and resolved contracts. Audit
child executions are ordinary WorkflowRuns, so their actual execution is
already governed by the global Scheduler setting.

The Audit Controller must nevertheless avoid eagerly materializing every item
as a queued Run. It uses a Server-owned per-Audit dispatch look-ahead equal to
the current `maxConcurrentRuns`. This is internal producer backpressure, not a
second execution limit and not reproducible Audit policy. It counts both
reserved submission intents and associated nonterminal child Runs so two
Controller instances cannot overfill the window through a crash or race.

The settings row and an Audit dispatch reservation have a defined PostgreSQL
ordering. If the global value decreases below an Audit's current outstanding
count, existing child Runs are not cancelled; that Audit submits no additional
Run until the count drains below the new window. An increase may wake Audit
reconciliation, while periodic reconciliation remains authoritative.

`maxSubmittedRuns` remains a pinned Audit budget over all child Run attempts
across rounds and retries. It bounds accidental or adversarial total work; it
does not control concurrent execution. `batchSize` continues to count logical
items carried by one ordinary child Run.

## 9. Operations UI and live invalidation

Operations navigation adds `Settings` at `/operations/settings`. The first
card is `Workflow scheduling` and contains:

- a numeric `Maximum concurrent Workflow Runs` input constrained to `1..32`;
- the saved value and unsaved-change state;
- Save and reset actions with pending, success, and safe error feedback;
- concise copy distinguishing this global bound from owner Queue Pause and
  Runtime Agent capacity.

The first increment does not derive a current-utilization gauge from allocation
count or the number of durable `running` records because neither equals active
Scheduler lanes.

A committed change emits a typed Operations invalidation with resource
`schedulerSettings` and no resource ID. The event remains a process-local hint;
the UI refetches the authoritative singleton. A revision conflict refetches and
shows the newly saved value instead of silently overwriting another operator's
change.

## 10. Supported topology and non-goals

The supported deployment has one active Server/Scheduler process and one or
more Runtime Agent processes. PostgreSQL claims make accidental duplicate
claiming safe, but two independently active Scheduler processes would each
apply their local lane count. A distributed global semaphore and multiple
active Control Plane replicas remain a separate feature.

This increment does not add:

- per-owner, per-Project, per-Workflow, or per-Audit execution quotas;
- user-visible queue positions, priority editing, or fairness guarantees;
- automatic scaling or automatic limit changes based on Runtime Agent count;
- concurrent StageExecutions within one WorkflowRun;
- Runtime Agent multi-slot processes;
- manual force-release, force-run, or force-cancel Operations actions.

## 11. Acceptance invariants

1. Upgrade and restart preserve effective serial behavior until an operator
   changes the seeded value from one.
2. Outside the documented decrease-drain interval, no more than configured `N`
   distinct WorkflowRuns hold Scheduler execution lanes, including
   initialization and cancellation work. During drain, no new claim starts at
   or above the lower desired value.
3. No Run ever has two lanes or two active StageExecutions.
4. Increasing the value admits waiting work without restart; decreasing it
   drains without cancelling active Runs or starting replacement work above the
   new limit.
5. Queue Pause remains owner-scoped, and cancellation, Project deletion,
   terminal release, allocation-loss recovery, and cleanup converge at every
   valid limit.
6. Runtime incompatibility or insufficient slots can lower actual concurrency
   but cannot make a deferred Run monopolize all lanes.
7. Audit child Runs consume the same global lanes as every other Run.
   AuditProfile contains no concurrency knob; its internal dispatch look-ahead
   follows the current global setting and its total submitted-Run budget remains
   independent.
8. The settings API is Operations-authorized, strict, CSRF-protected, CAS-safe,
   restart-durable, audited, and represented by the exact OpenAPI contract.
9. The independently deployed UI handles initial load, invalid input, pending
   update, stale revision, reconnect invalidation, Server error, and refresh
   without guessing an effective value.
