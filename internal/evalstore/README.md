# Managed Eval persistence

V38-003 supplies PostgreSQL authority, not runtime dispatch or HTTP routes.
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
retry cannot resurrect the experiment. Referenced dataset revisions are protected
by foreign keys. Tables for immutable evidence, selection, view generations and
progress are reserved for the V38-006 reducer.

Run against a disposable database (no live campaigns):

```sh
test -n "$CONTRACTOR_TEST_DATABASE_URL" &&
  go test -race -count=1 ./internal/evalstore ./internal/persistence/postgres
```

Integration tests use isolated random schemas. They cover owner separation,
concurrent CAS/admission, replay across revisions/deletion, exact immutable bytes,
blocked stale claims, both Project deletion race orders, Audit child dependencies,
private-data retention, bounded pages and collection revision changes.
