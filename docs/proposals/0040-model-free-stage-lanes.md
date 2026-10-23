# Separate lanes for model-free Stages

Proposal 0040 · Status: **Accepted on 2026-10-23
([Decisions](#decisions)); in progress. V400-001 implements sections 1 to 3
and the `maxConcurrentToolRuns` setting, V400-002 section 4; V400-003 and
V400-004 remain ([Proposed tasks](#proposed-tasks)). The specification already
states the implemented part (SCH-R175, SCH-R176, CTL-R155).**

Recorded on 2026-10-23 and checked against `main @ 63aa62b4`. Scanner Stages
make no model calls, yet each one holds one of the `maxConcurrentRuns`
Scheduler lanes that, in practice, pace model work. This proposal gives Stages
without a model a separate lane pool, so idle Runtime Agents can run scans
while model work stays bounded. It is a design recommendation, not an enabled
capability or an implementation commitment.

## User outcome

An operator sets how many model conversations run at once, independently of
scanner work. An Audit with many OpenAPI scan items keeps its check Runs
moving: scans use Runtime Agents the model work is not using, instead of
waiting in the same queue as the model checks or making them wait.

Scans never take Runtime capacity that a waiting model Stage could use, and
the parallel load on a scanned target stays bounded by an explicit setting.

## Current boundary

- **One limit.** `maxConcurrentRuns` (1..32, default 1, LIM-OPS-7) is the
  number of Scheduler execution lanes; one Run holds at most one lane while
  it progresses one Stage attempt ([scheduler](../spec/execution/scheduler.md),
  SCH-R115, SCH-R116, SCH-R136; `internal/scheduler/supervisor.go`). SCH-R128
  makes it the only concurrency setting, with "no second queue, no Audit
  execution pool and no special scheduling based on Run labels".
- **What it protects.** SCH-R127 justifies the limit by Runtime Agent and
  Scheduler capacity. Runtime capacity is already enforced separately: each
  Runtime Agent supplies one slot and a Stage without enough compatible slots
  stays `preparing` (SCH-R19, SCH-R139). Contractor does not bound concurrent
  model calls itself; only the Gateway's `tpmLimit`, `rpmLimit` and
  `maxParallelRequests` do (RCF-R63). The lane count is therefore the only
  Contractor control that keeps a local model from being overloaded.
- **Scans.** A scan Stage uses the model-free `scan-plan@1` Planner
  (SCN-R170) and `tool@1` Workers, whose allocations carry no LLM Gateway
  (SCN-R21). The shipped Audit scan Workflows
  (`configs/workflows/audit_openapi_nuclei_scan.yaml`,
  `audit_openapi_sqlmap_scan.yaml`) have one `scanner` Worker, one job and a
  300-second job bound. Each still takes a lane.
- **Audit pacing.** The Audit Controller keeps at most `maxConcurrentRuns`
  outstanding child Runs per Audit (AUX-R3, LIM-AUX-1;
  `internal/auditcontroller/dispatch_selection.go`), counted as one total.
- **Deferral.** A Run that releases its lane without progress is marked
  `scheduler_deferred` and ranks behind ready work for 30 seconds (SCH-R123,
  LIM-SCH-8). One boolean covers three causes: owner queue pause
  (`supervisor.go`, `ErrQueuePaused`), insufficient Runtime capacity
  (`stage_preparation.go`) and a blocked model route
  (`allocation_reservations.go`).
- **Placement errors.** Control Plane returns one `ErrInsufficientCapacity`
  both when compatible Agents are busy and when no registered Agent is
  compatible (`internal/controlplane/errors.go`).

With `maxConcurrentRuns = 2` and an Audit of 50 OpenAPI operations, scan and
check child Runs share two lanes. While two scans run, the model is idle; while
two checks run, idle Runtime Agents cannot start scans.

## Design

### 1. Lane class of a Stage

A StageExecution has lane class `tool` when both hold:

- its Planner strategy uses no model client: today `passthrough@1` and
  `scan-plan@1`;
- every Worker in its resolved execution config is model-free, as already
  recorded by the resolved runtime config (`ModelFree` in
  `internal/runtimeconfig/resolver.go`).

Every other StageExecution has class `model`, including a `passthrough@1`
Stage with an ADK Worker and a Router Stage with one `tool@1` Worker. The
class is computed once, from the pinned resolved config, when the
StageExecution is created. A retry or escalated execution is a new
StageExecution and is classified on its own; SCN-R24 already prevents an
escalation from adding a model to a tool Worker.

Model use is declared once: the Planner registry states whether each strategy
needs a model client, and one function derives the class from that declaration
and the Workers' `ModelFree` values. The Scheduler and the Audit Controller
both call that function; neither keeps its own list (AGENTS.md §3.1). Labels
and Workflow names play no part.

### 2. Two lane pools

The Scheduler settings singleton gains `maxConcurrentToolRuns`, an integer
from 1 to 32 with default 2, in the same `scheduler_settings` row and under the
same revision, CAS update, wake hint, periodic refresh and restart rules as
`maxConcurrentRuns` (SCH-R133 to SCH-R138, OPS-R47 to OPS-R50). On the wire it
is a new field of the existing Scheduler settings resource; a write that omits
it keeps the stored value, so the `/v1` change is additive.

`maxConcurrentRuns` keeps its name and becomes the size of the `model` pool:
the number of Runs that may progress a model Stage at once. The supervisor
runs both pools with the same lane loop. Each pool resizes, drains on a
decrease and bounds its distinct Runs exactly as SCH-R134 to SCH-R136 do for
the single pool today. The exclusion set of SCH-R120 is bounded by the sum of
both pools (at most 64).

Actual concurrency of the `tool` pool is `min(maxConcurrentToolRuns, free
compatible Runtime slots)`: capacity deferral (SCH-R19) already supplies the
second bound, so "as many scans as there are free workers" needs no new
mechanism.

### 3. Run claims per pool

`workflow_runs` gains `scheduler_lane_class` (`model` or `tool`), set for every
non-terminal Run to the class of the StageExecution the Scheduler progresses
next: the active one, otherwise the first Stage at Run creation or the target
of the committed transition decision (LIM-SCH-12). It is written in the same
transaction that creates the StageExecution or commits that decision.

The claim indexes `workflow_runs_claim_idx` and
`workflow_runs_deferred_claim_idx` gain the class as their leading key, and
each pool's claim reads only its own class. The claim still reads candidates
in index order without sorting (SCH-R164).

A lane progresses Stages of its own class only. When a transition creates a
Stage of the other class, the lane releases the claim without deferral, and
the other pool claims the Run in its normal order. Cancellation, finalizing,
aborting, allocation-loss handling and recovery run in the pool of the Stage
they concern. Owner queue pause applies to both pools (SCH-R148).

### 4. Model Stages first on shared Runtime Agents

Runtime Agents may serve both pools; dedicated scan Agents are optional. When
there is no free Runtime Agent for both, the conflict on a shared Agent is
resolved in favour of the model Stage. Without this rule, each slot that frees
up goes to the next scan, and a model Stage that needs several slots at once
(SCH-R23) can wait indefinitely.

The rule, as implemented by V400-002 (SCH-R176, CTL-R155): when placement of a
`model` Stage finds no complete matching, Control Plane checks whether one
exists among all live Agents, idle or allocated. If it does, every live Agent
the Stage could use is held for a hold window of five Scheduler poll
intervals; tool placement does not reserve a held Agent, and model placement
ignores holds. A tool Stage that fits on an Agent no waiting model Stage can
use, such as a dedicated scan Agent, is placed as usual.

- It holds back only placement of new tool Stages; running tool Stages,
  cancellation, finalization, release and recovery continue.
- Each failed attempt of the waiting Stage extends the hold. A Stage that is
  placed, cancelled, aborted, past its deadline, paused, denied by a blocked
  model route or left without a free model lane stops extending it, and the
  hold lapses within the window.
- A model Stage that could not be placed even with every allocation released
  holds nothing.
- Every waiting model Stage holds the Agents it could use, not only the oldest
  one: that is the decision taken literally, and it needs no ordering. Waiting
  model Stages compete among themselves as they do today (SCH-R121, SCH-R123).
- A hold is process-local Registry state, like a reservation: one deadline per
  Agent entry, cleared by a Server restart until the next failed attempt.

The first design recorded a deferral cause on `workflow_runs` and read the
oldest capacity-waiting model Run with SQL before each tool placement. It was
dropped: a durable cause outlives the attempts that justify it, so a model Run
waiting for a free model lane, when no lane could use an Agent, would still
keep scans off idle Agents, and it needed a migration, an index and a second
derivation of the Stage's placement requirements.

Accepted risks: Operations shows a held Agent as idle while a scan waits for
capacity; and a model Stage that needs several Agents keeps the ones that free
up idle until the rest free up, bounded by its deadline, so scans on shared
Agents may wait longer and reach their own deadline. Reopen the first when
operators need to see why a scan waits on an idle Agent, the second if scans
miss deadlines in practice.

Operators who want physical separation can still have it without code:
install scanners only in a dedicated Runtime image. An Agent advertises only
the scanner tools whose version commands succeed (SCN-R54), so
capability matching keeps scan Stages on those Agents.

### 5. Audit dispatch windows per class

The per-Audit dispatch look-ahead is split by class. A role whose pinned
Workflow has only `tool` Stages is paced by `maxConcurrentToolRuns`; every
other role is paced by `maxConcurrentRuns`. The Controller counts outstanding
child Runs per class instead of one total (`OutstandingRunCount`). The window
remains producer backpressure, not an execution limit (AUX-R5), so a Workflow
with mixed Stages counts as `model`. Without this split no Audit submits more
than `maxConcurrentRuns` child Runs at a time, and the `tool` pool idles.

### 6. Load on the scanned target

Scans send active traffic to the target. Per-job rate limits such as the ffuf
`rate` (LIM-SCN-28) do not combine across parallel jobs, so the `tool` pool
size multiplies the load on a target that several scans share.

**Accepted risk** (the user, 2026-10-23): no per-target limit or
synchronization is added. Parallel scans of one target are bounded only by
`maxConcurrentToolRuns`, default 2. Reopen this when the default is raised
above 4 or when scans may target fragile or production systems. The fix then
is a per-host concurrency limit enforced at tool placement, not an
AuditProfile knob (SCH-R142).

## Failure, lifecycle and bounds

- An invalid or missing `maxConcurrentToolRuns` is a storage invariant
  failure, as for `maxConcurrentRuns` (OPS-R50, LIFE-OPS-5); the Server does
  not guess a default at runtime.
- Lowering either value drains that pool only; no Run or Planner call is
  cancelled because a limit fell (SCH-R135).
- A Run is in at most one pool and holds at most one lane (LIM-SCH-7). A class
  change between Stages releases and re-claims; it never holds two lanes.
- A Run whose stored class disagrees with the class recomputed from its
  StageExecution is a storage invariant failure, not a hint to correct
  silently.
- Lane work per tick stays bounded by the pool sizes, and a hold of section 4
  is one deadline per Agent entry, computed from the live Agents of one
  failed model placement.

## Alternatives

| Option | Why not |
| --- | --- |
| Exempt tool Runs from the limit, bounded only by free slots | No cap on Server lanes or on load on the target; scans can starve model Stages of Runtime slots |
| Classify the Run, not the Stage | A Workflow can scan and then triage with a model; the Run would either hold a model lane while scanning or bypass the model limit later |
| Semaphore on concurrent model calls instead of Run lanes | Precise for the GPU, but many interleaved conversations evict prefix/KV caches on local models and stretch each Stage towards `workerRequestTimeout` and its deadline; bounding whole conversations suits 27b–80b models better. It can be added later as an extra guard |
| Weighted single pool (tool Runs count 0 up to a cap) | The same behaviour as two pools, harder to explain and to drain |
| Require dedicated scan Agents | Works today through capabilities, but forces a second image on every deployment; section 4 protects shared Agents instead |

## Decision record

- **Guarantee.** At most `maxConcurrentRuns` Runs progress model Stages and at
  most `maxConcurrentToolRuns` Runs progress model-free Stages at once. Tool
  placement never reserves a Runtime Agent that a model Stage waiting for
  allocated capacity could use.
- **Evidence.** SCH-R127, SCH-R128 and SCH-R139 and the code cited above; no
  load measurement yet.
- **Reversal condition.** Return to one pool if Server lane overhead at the
  combined maximum fails the load gate in *Verification*, or if Contractor
  gains a model-call admission limit that makes the Run lane limit redundant.

## Decisions

Taken by the user on 2026-10-23:

1. `maxConcurrentToolRuns` ranges over 1..32 with default 2.
2. Runtime Agents may be shared. When no free Runtime Agent serves both, a
   conflict on a shared Agent is resolved in favour of the model Stage
   (section 4). Dedicated scan Agents remain an operator option.
3. No per-target limit or synchronization; accepted as a risk (section 6).

## Specification and contract changes

- [scheduler](../spec/execution/scheduler.md): SCH-R115, SCH-R116, SCH-R120 to
  SCH-R123, SCH-R127 to SCH-R129, SCH-R134 to SCH-R142, SCH-R164; LIM-SCH-6.
- [operations](../spec/operations/operations.md): OPS-R46 to OPS-R52 and the
  Scheduler settings fields; a new limit beside LIM-OPS-7.
- [Audit execution](../spec/audits/execution.md): AUX-R3 to AUX-R7, invariant
  6, LIM-AUX-1.
- [planners](../spec/execution/planners.md): each strategy declares whether it
  needs a model client.
- [control plane](../spec/execution/control-plane.md): CTL-R155 model Stage
  holds, CTL-R73, CTL-R84 and LIM-CTL-10.
- OpenAPI Scheduler settings resource; the Operations settings page in the UI.
- Migration: `scheduler_settings.max_concurrent_tool_runs`,
  `workflow_runs.scheduler_lane_class` and the rebuilt claim indexes.

## Verification

All on real PostgreSQL unless stated otherwise:

- the class function over Planner strategies and Worker kinds, including mixed
  Stages, retries and escalations;
- two pools at once: each respects its own limit, a scan-then-triage Run moves
  between pools without holding two lanes, and cancellation reaches a Run in
  either pool;
- decrease drain, restart and CAS for both settings;
- section 4: a model Stage waiting on allocated compatible Agents keeps tool
  placement off the Agents it could use, and a released Agent goes to it, not
  to the next scan; a tool Stage still lands on an Agent that Stage cannot
  use; a model Stage with no complete matching among live Agents holds
  nothing; a hold lapses with its window;
- Audit dispatch keeps separate outstanding counts per class and never
  overfills either window across two Controllers (AUX-R4);
- a load gate for claim, renewal and connection-pool cost at 32 + 32 lanes,
  since OPS-R50 requires one before raising the lane bound.

## Proposed tasks

1. V400-001, Server: the lane class, the second pool, the settings field,
   migration, OpenAPI and specification changes. The setting has no caller
   without the pool, so they land together (AGENTS.md §3.7).
2. V400-002, Server: the section 4 rule, model Stage holds in Control Plane.
3. V400-003, UI: the `maxConcurrentToolRuns` field on the Operations settings
   page.
4. V400-004, Audit Controller: per-class dispatch windows and outstanding
   counts.
