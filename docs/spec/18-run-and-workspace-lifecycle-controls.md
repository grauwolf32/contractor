# 18 — Run and workspace lifecycle controls

Status: **Working agreement**

Depends on: [03](03-artifact-plane.md),
[04](04-execution-lifecycle-and-metrics.md),
[06](06-server-ui-and-operations.md) and
[17](17-projects-and-queue.md)

## Purpose and boundary

This document owns the operator-facing lifecycle controls for the ordinary
owner-scoped WorkflowRun history and Project workspaces: one consolidated Runs
surface, a durable queue admission gate, explicit terminal Run deletion and
recoverable Project deletion.

None of these controls creates a second execution state machine. WorkflowRun,
StageExecution and allocation state remain owned by Scheduler and Control
Plane. Destructive cleanup may start only after those authorities have reached
their existing safe terminal/released boundaries.

## One Runs surface

The top-level UI has one **Runs** destination with two views:

```text
Runs
  Queue       default; initializing | running | cancelling
  Completed   succeeded | failed | cancelled
```

Queue keeps the existing oldest-first `/v1/queue` read projection, optional
Project display context, live invalidation and polling fallback. Completed uses
the ordinary newest-first `/v1/runs` collection with an additive
`lifecycle=terminal` filter. The filter is applied by Server before keyset
pagination; filtering one browser page is incorrect. Exact `state` and metadata
label filters may be combined with the lifecycle filter by intersection, and
all filters are bound into the cursor kind.

The two tabs deliberately need not use the same endpoint: their ordering and
display projections are different even though both read `workflow_runs`.
`/queue` remains a client-side compatibility redirect to the Queue view of
`/runs`; Queue is no longer a separate top-level navigation item.

## Durable owner queue gate

Queue pause is an owner-scoped durable admission gate, not a process-local
toggle and not a new WorkflowRun state:

```text
OwnerQueueControl
  owner_id       primary key, derived from authentication
  paused         boolean
  revision       opaque concurrency token
  updated_at
```

Absence of a row means running. `GET /v1/queue/control` returns the effective
state. `PUT /v1/queue/control` accepts exactly `{"paused": true|false}` and a
strong `If-Match` for an existing representation. Repeating the same desired
state is idempotent. The Queue tab renders one authoritative **Pause queue** or
**Resume queue** action and refetches after mutation.

Pausing has drain semantics:

- Server may continue accepting new Runs; they remain visible and wait.
- A semantic Stage already admitted before the pause linearization point may
  finish normally, including finalization and release.
- Scheduler must not admit the owner's next normal Stage after pause commits.
- Cancellation, abort, recovery of an already admitted Stage, terminal
  finalization, output publication and allocation release continue while
  paused. A Project deletion request can therefore always make progress.
- Resume makes waiting Runs eligible after a restart without rewriting them.

The pause update and normal Stage admission share one database serialization
boundary. A process that observed an older state cannot begin a new semantic
Stage after the pause transaction commits. Pause does not suspend an in-flight
HTTP or model call and does not promise an exact queue position.

## Terminal Run hard deletion

`DELETE /v1/runs/{run_id}` permanently removes an owned Run. It is allowed only
when the Run is terminal (`succeeded`, `failed` or `cancelled`) and every
allocation belonging to it has completed release. Run list/detail responses
expose a Server-derived `deletable` boolean so the UI shows a trash action only
at that safe boundary. A stale attempt receives a conflict and changes
nothing; a foreign or missing Run remains the ordinary owner-safe `404`.

Deletion is one controlled transaction over all Run-owned durable state,
including:

- the complete RunScope: input forks, intermediate bindings, declared outputs,
  revisions, pins and lineage;
- Stage executions, plans, attempts, candidate/accepted results, allocation
  records and release metadata;
- Run events, metrics, labels, cancellation state, resolved execution data,
  publication receipts and create-idempotency records;
- the WorkflowRun row itself.

Source UserScope and ProjectScope artifacts selected as Run inputs are not
Run-owned and remain. A ProjectScope output previously published from the Run
is also an independent Project artifact and remains when only the Run is
deleted; the Run-owned publication receipt and cross-scope provenance edge are
removed. Deleting the Project later removes that ProjectScope output.

Artifact content may be deduplicated. Cleanup removes an immutable version or
blob only when no binding, retained revision, pin, lineage edge or other scope
outside the purge set still references it. A physically shared blob retained
for another artifact is not a surviving Run artifact. The controlled purge is
the only exception to normal create-only/immutable Artifact APIs and is never
available to a Runtime or generic Artifact endpoint.

The browser asks for confirmation immediately before DELETE and removes the
row only after Server success. Hard deletion intentionally removes the original
Run-create idempotency record; a later create with the same key may therefore
create a new Run.

## Recoverable Project deletion

A Project has an additive lifecycle state:

```text
active | deleting
```

`DELETE /v1/projects/{project_id}` requires the current Project `If-Match` and
returns `202 Accepted` after atomically changing `active` to `deleting` and
durably scheduling cleanup. Reconciliation of the same already-deleting
Project returns its current deleting representation. The UI requires explicit
confirmation by typing the Project name, warns that Runs and artifacts are
permanent, then polls the ordinary Project representation until it becomes
owner-safe `404`.

Once deleting, Server rejects Project metadata/target changes, artifact
writes and new Project Runs. Reads may continue so progress is visible. A
restart-safe controller repeatedly performs these phases:

1. request cancellation for every nonterminal Project Run using the ordinary
   Scheduler cancellation transition;
2. wait until all Project Runs are terminal and all their allocations have
   completed release;
3. invoke the terminal Run purge for every Project Run, including Runs that
   were already terminal when deletion began;
4. purge every binding, retained revision, pin and lineage edge in the complete
   ProjectScope, then garbage-collect only globally unreferenced versions and
   blobs;
5. remove Project create-idempotency data and the Project row.

The operation is monotonic, idempotent and retryable after a crash at every
phase. Queue pause never blocks its cancellation, drain, release or cleanup
work. UserScope artifacts, global Skills and referenced RuntimeCredential
records are outside Project ownership and are retained; deleting the Project
only removes their Project/Run references. No response claims completion while
an allocation or ProjectScope artifact is still retained.

## Failure and concurrency behavior

| Situation | Result |
|---|---|
| Delete a nonterminal or not-yet-released Run | conflict; no partial cleanup |
| Run becomes terminal while DELETE races | one locked snapshot decides; retry after refetch |
| Shared artifact blob exists outside purge set | logical Run/Project data is removed; blob remains referenced |
| Pause races with normal Stage admission | the shared serialization boundary orders drain versus wait |
| Cancel or release fails during Project deletion | Project stays `deleting`; controller retries |
| Server restarts during Project deletion | durable phase is resumed; no Run is recreated |
| Project DELETE response is lost | retry observes the existing `deleting` operation |
| Foreign Project or Run identifier | owner-safe `404` with no existence disclosure |

## Invariants

1. Queue and Completed are views of ordinary WorkflowRuns, not new entities.
2. Pause is owner-scoped, durable and blocks only new normal semantic work.
3. Cancellation, recovery, finalization and release cannot be paused.
4. A Run cannot be deleted before terminal allocation release completes.
5. Run deletion removes all RunScope and execution state but never deletes an
   independently retained source or ProjectScope artifact.
6. Project deletion is durable and cannot finish before every owned Run and the
   entire ProjectScope have been purged.
7. Artifact garbage collection is reference-safe across all scopes.
8. Only authenticated public lifecycle endpoints and trusted Server recovery
   code can enter the controlled deletion path.

