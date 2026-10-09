# Managed Eval persistence

This package owns PostgreSQL authority and bounded execution metric reads.
`evaldomain.Lifecycle` defines typed command targets, observed transitions,
admission, budget enforcement and command recovery; store methods apply those
rules under locks. `evalservice` owns orchestration, with separate Workflow/Audit
adapters and a shared durable operation mechanism.
Every multi-statement mutation uses `NewTxStore(pgx.Tx)` in a caller-owned
transaction. Returning an error requires the caller to roll back. Reads and
single-statement claim operations use `NewPostgresStore(DBTX)`.

Lock order is Project admission row, experiment row, then controller claim.
An Audit member's owned workspace is locked before the experiment as well.
Keys are serialized within owner/Project/resource/operation scope; an existing
receipt replays before revision or deletion checks. New effects recheck the
Project fence and immutable control mode. Only native dispatch supplies a claim
at admission. Reconciliation of accepted native/external operations requires a
current claim; an expired holder cannot acknowledge an operation after replacement.

`Admit` commits the member intent and outstanding count before any execution.
Audit Project/create/start and Workflow creation have distinct immutable keys and
retained suboperation requests. A transport failure leaves an intent unresolved.
`BindExecution` verifies ordinary Run/Audit service identity and exact workspace;
labels never establish membership. `Settle` observes authoritative terminal state
and resolves every suboperation before decrementing outstanding once. A cancelled
intent with no possible execution creation can settle without inventing a Run.

Datasets and plans retain byte payloads with database-computed SHA-256. Only
`Freeze` output reaches the immutable plan row, so `FrozenPlan` restores the
stored bytes and generated digest without validating the schema again.
`FrozenPlanMetadata` reads only plan attribution and setup for admission and
per-member collection; `PlanResource` likewise restores immutable validated
resources without repeating their schema validation. External
registration has its own stored document digest, separate from `Plan.SHA256`,
which preserves the attributed source plan identity used by the producer. Native
`Plan.SHA256` is the exact portable plan digest. Preparation in V38-004 owns
catalog resolution and verification of the reachable portable resource graph.
Private datasets/drafts/plans never enter indexed collection reads or ordinary
artifact namespaces. The service must choose explicit public projections.

The database freezes expected membership with a deferred completeness check and
bounded unique ordinals. There is no independently editable public manifest.
Project deletion fences new work in the existing lifecycle transaction. The
ordinary Project drain waits for unresolved intents and exact Audit workspaces.
Deleting a child workspace cancels dispatch while retaining its parent experiment.
Purge removes only Eval private state after drain; it does not delete Runs/Audits.
Mutation receipts survive an experiment purge until its Project is purged, so a
retry cannot resurrect the experiment. Receipts have distinct dataset,
experiment, command and submission types. Readers require the current typed
fields and reject the old overloaded `{id, revision, state}` shape, mixed types,
unknown fields and trailing JSON. They never infer fields or rewrite immutable
receipt bytes. Invalid receipts fail replay at the response boundary without
repeating the already accepted mutation; no dataset revision is stored in a
state field. Current typed receipts retain their original identity and revision
across state changes and experiment purge.
Referenced dataset revisions are protected by foreign keys.

V38-006 separates immutable records, selection history, mutable dirty-member
projections and immutable published generations. Run/Audit, child inventory,
metric and exact artifact-deletion triggers invalidate affected members. A bulk
matrix insert invalidates the experiment queue once per statement. The public
experiment `revision` is the owner's authority CAS token: draft edits, commands,
deletion, explicit user selection and Project deletion fences advance it. Native
plan preparation, coordinator transitions, token observations, admission, settlement,
first native selection, external producer activity and execution tombstones update
`updated_at` without consuming that token. A submission receipt reports the
current authority revision; selected-view and collection revisions remain separate.

Collection revalidates selected evidence and publishes every expected member,
pair, suite summary and chart aggregate atomically. Failure leaves the last
complete generation visible as stale. A source revision belongs to the snapshot
identity, so recovered evidence cannot revive an older progress timestamp. A
publication with an unchanged content digest (selected documents, comparison,
pins, selection revision) keeps the current generation; a new generation deletes
the superseded ones, which `contractor_eval_view_immutable` permits only for
generations the queue no longer names. Selected-page reads use keyset pagination
over the current generation and never scan historical executions/artifacts.
Histograms retain exact cohort percentiles; progress reads choose real
observations in at most 200 buckets.

Execution inventories join authoritative Audit associations across roles, rounds
and retries. Public inventories paginate the full set; bounded native collection
retains an explicit gap if its per-member limit is exceeded. Deleting an execution
or artifact keeps prior record attribution and invalidates current completeness.

Run against a disposable database (no live campaigns):

```sh
test -n "$CONTRACTOR_TEST_DATABASE_URL" &&
  go test -race -count=1 ./internal/evalstore ./internal/persistence/postgres
```

Integration tests use isolated random schemas. They cover owner separation,
concurrent CAS/admission, replay across revisions/deletion, exact immutable bytes,
blocked stale claims, both Project deletion race orders, Audit child dependencies,
private-data retention, bounded pages and collection revision changes.
