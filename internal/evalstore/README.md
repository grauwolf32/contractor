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

Datasets and plans retain byte payloads with database-computed SHA-256. External
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
retry cannot resurrect the experiment. New receipts have distinct dataset,
experiment, command and submission types. Legacy immutable receipt bytes are
converted only when read; no dataset revision is stored in a new state field.
Referenced dataset revisions are protected by foreign keys.

V38-006 separates immutable records, selection history, mutable dirty-member
projections and immutable published generations. Run/Audit, child inventory,
metric and exact artifact-deletion triggers invalidate affected members. A bulk
matrix insert invalidates the experiment queue once per statement. Observation
revisions do not consume the user's authority CAS revision.

Collection revalidates selected evidence and publishes every expected member,
pair, suite summary and chart aggregate atomically. Failure leaves the last
complete generation visible as stale. A source revision belongs to the snapshot
identity, so recovered evidence cannot revive an older progress timestamp.
Selected-page reads use keyset pagination over retained generations and never
scan historical executions/artifacts. Histograms retain exact cohort percentiles;
progress reads choose real observations in at most 200 buckets.

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
