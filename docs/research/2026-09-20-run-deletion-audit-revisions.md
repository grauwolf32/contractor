# V60-012: Run deletion and Audit revisions

Fix for PR-06 from the [first review pass](2026-09-20-project-review-first-pass.md).
The [task](../../tasks/v60/v60-012-run-deletion-audit-revisions.yml) defines acceptance;
actual commands and results are retained in the
[evidence](../../tasks/evidence/v60-012.json).

## Defect and scope of the change

Deleting a Run changed retained provenance (`runDeleted: false → true`) without
changing the Audit revision. The correct revision fences from V60-003 therefore
could not detect the change: saved pins and a signed cursor continued to return
`200`, and the ETag did not change. This was reproduced with real PostgreSQL,
the HTTP handler and `ImportIntoAudit` / `DeleteReleasedTerminalRun`.

The fix is local, with no migration or subsystem rewrite. The deletion
transaction increments each affected owner's Audit revision exactly once:

- the Audit of a linked managed execution;
- the Audit of native proposal receipts;
- every destination Audit retaining a proposal imported from that Run.

Terminal Audits are included. Multiple receipts and overlapping relationships
do not cause multiple increments of the same revision. Unrelated Audits remain
unchanged. Terminal-state, release and completed managed-collection checks are
preserved. Failed or rolled-back operations leave neither a revision increment
nor a tombstone. A retry after successful deletion returns the usual `ErrNotFound`.

Finding revision remains the revision of its assessment and decision. Source
Run availability invalidates the enclosing Audit revision; it does not create
a new finding assessment. `Origin.RunDeleted` and `Attempt.RunDeleted` are still
computed for their own Run IDs, with no automatic copying between them.

## Coordinating concurrent operations

A late `UPDATE audits` without prior locks is insufficient: an import can add a
destination after selection, and Audit purge locks the Audit before
receipt/retention and Artifact locks. The new order aligns with these paths:

| Operation | Order |
| --- | --- |
| ImportIntoAudit | Source Run `FOR KEY SHARE` → owner/project-bound Audit `FOR UPDATE` → receipt/retention → Artifact import/hold |
| DeleteReleasedTerminalRun | Source Run `FOR UPDATE` → affected Audits in Audit ID order `FOR UPDATE` → execution/retention → Artifact purge → Run DELETE and Audit revision UPDATE |
| Audit purge | Audit → Artifact purge → Audit deletion and cascading hold/retention release |

`FOR KEY SHARE` preserves the source Run during import and conflicts with its
deletion. Imports can hold this lock concurrently; ordinary updates to
non-identity Run fields do not require exclusive serialization of all imports.

If import wins, deletion waits for its commit and includes the new Audit in
selection. If deletion wins, import waits and then receives `ErrNotFound`.
A deletion rollback releases the source Run for successful import. Cancelling
a waiting import leaves no partial hold.

Native receipts do not introduce a separate race: production Submit checks the
allocation write grant, which closes before `release_completed_at`.
The managed collector creates proposal holds before the durable collection
receipt; deletion requires that receipt. The current Project controller deletes
individual Runs outside a held Project transaction. Existing bounded retries
for selected PostgreSQL aborts are preserved.

This explains current paths and verified interleavings. It does not claim that
arbitrary future SQL or arbitrary external transaction ordering cannot cause
a deadlock. Any new hold-creation path must follow this order.

## Why updatedAt is preserved during finalizing

The report importer first writes immutable `report.json` and `report.md`, then
calls `CommitReport` / `ProposeReport` with an Audit revision check. In the
machine report, `generatedFrom` comes from `Audit.UpdatedAt`.

Naively incrementing both revision and timestamp between the file write and CAS
rejects the stale commit, but a retry produces different bytes under an already
occupied immutable name. A separate fault probe reproduces
`ErrArtifactIntegrity: Audit immutable binding collision`; forcibly overwriting
the report would violate the existing contract.

The adopted rule is narrow: when an Audit is `finalizing`, deletion of a linked
Run increments the Audit revision while preserving `updatedAt`. Deletion changes
Run availability, which is absent from the report payload; durable collection
receipts, assessments and evidence remain. Retrying with the new revision
reproduces the previous bytes and exact refs, allowing the required CAS to pass.

For other Audit states, `updatedAt` advances normally. The authoritative change
token is revision/ETag. This rule is recorded in
[spec 19](../spec/19-audits.md), and deletion atomicity in
[spec 18](../spec/18-run-and-workspace-lifecycle-controls.md).

Pending human review already has a frozen subject revision/digest and exact
report refs. Changing the current Audit revision does not invalidate that
subject. Approve/reject, proposal replay after a lost response and review-decision
replay retain their existing authority. An additional equality check between
the current Audit revision and the old subject would be incorrect here.

## Checks

| Boundary | Verified behavior |
| --- | --- |
| Original HTTP defect | On the baseline, Audit A remains at rev 3 and B at rev 2; old pins/cursor return `200`. After the fix, A/B each gain +1; old pins/cursor and mutation during a read produce `409`; ETag changes. |
| Retained provenance | Reading through a separate single-connection pool; a fresh envelope contains the new revision, original refs/receipt IDs, correct Run IDs and exact proposal/evidence bytes. Unrelated Audits of the same or a different owner remain unchanged. |
| Import ↔ Run deletion | Both winners, rollback and cancellation of a waiting import followed by retry. Real service calls; blocking demonstrated through `pg_blocking_pids` and the executing SQL text. |
| Run deletion ↔ Audit purge | Both orders; the retained Project Artifact and source Run use one version. Checks verify waiting on the Audit lock before retention/artifact locks and completion of both deletions. |
| Report finalization | Automatic/human-required × deletion before machine write, after machine write, after summary write. Stale CAS is rejected; a new importer instance retries the same bytes/refs. |
| Pending report review | Approve and reject after deletion; exact frozen subject and proposal/decision replay. |
| Managed deletion | Real Run creation and collection; transaction-backed rollback restores Run/Audit/tombstone/input Artifact; retry causes no second bump; an uncollected terminal Run is rejected without mutation. |
| Native receipt retention | Storage-boundary seeding of two immutable native receipts for one collected execution; after Audit completion, +1 revision and two discarded tombstones with unchanged receipt identity. This tests persisted rows, not a simulated Runtime Submit. |
| Negative report probe | Forcing a finalizing timestamp change after the machine report write reproduces the immutable collision. Protection of immutable bytes is not weakened. |

New report/managed/native regression cases were added to the mandatory Go
matrix of the Audit completion gate. It passed: **173 Go cases and 331 Runtime
tests with no selected skips**. The combined PostgreSQL `-race` run across five
affected packages and `make verify-public-api-postgres` also passed. Subsequent
focused `-race` runs additionally checked terminal Audits, exactly one bump and
unchanged finding revision; production code did not change after the combined run.
Full commands and verification limits are retained in the evidence. PostgreSQL
used a separate disposable local container with isolated schemas; the Runtime
gate uses existing offline fixtures. Live-model evals, active Evals/toolset/sqlmap
work and production deployment are outside this change.
