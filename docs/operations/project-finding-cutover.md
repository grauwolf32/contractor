# Project Finding cutover

Historical procedure executed on 2026-11-02 with the repaired chain at
`5aa77fee8d6ef8c88780c1352275c78346b6e18d`. The subsequent baseline fold removes
that upgrade chain and its fixtures from the runtime; new databases install the
current baseline directly. The demo manifest keeps the execution evidence.

Stop new work, finish current Runs, wait for allocation release and collect every
Audit execution using the existing deployment. Preserve required exact Audit
results and materials as Project bindings before Run deletion. Take a consistent
PostgreSQL/filesystem/configuration backup with all writers stopped and verify
its restoration in a disposable database.

After restoring that backup for rehearsal, retire only old Audit-managed Runs
using the matching pre-cutover Server through its existing `DELETE /v1/runs/{run_id}` owner operation, or its
`rundeletion.Deleter` integration. It enforces terminal state, allocation release
and the durable collection receipt, tombstones the original execution and purges
the RunScope transactionally. Ordinary Runs and independent Project bindings stay.
An active, unreleased or uncollected Run must be drained before retrying.

The initial Project Finding migration fence additionally rejects any retained
Run-to-AuditExecution FK with SQLSTATE 23514 and an actionable hint. A terminal
state alone does not remove that FK. Run retirement completes as a separate
normal lifecycle operation; the migration then archives opaque Audit facts and
results and removes their executable rows atomically. A migration failure rolls
back its archive/schema changes, and the ledger prevents replay after success.
Deferred reference checks finish inside that transaction before its DDL phase;
foreign keys remain enforced throughout.

Rehearse the complete sequence on the verified restore before applying it to the
stopped demo. Record backups, retired Run IDs, unchanged ordinary Run and Project
material signatures, applied versions/checksums and final readiness in the release
manifest. The database is never reset. A later baseline fold changes the ledger
only after installed-schema equivalence and business-data preservation are proven;
the runtime migrator retains its strict checksum/drift behavior.
