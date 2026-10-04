package runstore

// SQL statements for queue_control_store.go.

// updateOwnerQueueControlSQL sets an owner's queue paused flag while holding
// the owner's transaction-scoped advisory lock (queue admission takes it
// shared). Expected revision $3 = 0 creates the row at revision 1; $3 > 0 is a
// revision CAS that bumps it only when paused changes, else returns the row
// as is. No row means a failed precondition.
// Used by PostgresStore.UpdateOwnerQueueControl.
var updateOwnerQueueControlSQL = `
WITH queue_lock AS MATERIALIZED (
    SELECT pg_advisory_xact_lock(
        hashtextextended('owner-queue-control:' || $1::text, 0)
    )
), inserted AS (
    INSERT INTO owner_queue_controls (owner_id, paused, revision)
    SELECT $1, $2, 1
    FROM queue_lock WHERE $3::bigint = 0
    ON CONFLICT DO NOTHING
    RETURNING owner_id, paused, revision, updated_at
), updated AS (
    UPDATE owner_queue_controls AS control
    SET paused = $2,
        revision = control.revision + 1,
        updated_at = GREATEST(clock_timestamp(), control.updated_at + interval '1 microsecond')
    FROM queue_lock
    WHERE control.owner_id = $1
      AND control.revision = $3
      AND $3::bigint > 0
      AND control.paused IS DISTINCT FROM $2
    RETURNING control.owner_id, control.paused, control.revision, control.updated_at
), unchanged AS (
    SELECT control.owner_id, control.paused, control.revision, control.updated_at
    FROM owner_queue_controls AS control CROSS JOIN queue_lock
    WHERE control.owner_id = $1
      AND control.revision = $3
      AND $3::bigint > 0
      AND control.paused IS NOT DISTINCT FROM $2
    FOR UPDATE OF control
)
SELECT owner_id, paused, revision, updated_at FROM inserted
UNION ALL
SELECT owner_id, paused, revision, updated_at FROM updated
UNION ALL
SELECT owner_id, paused, revision, updated_at FROM unchanged
LIMIT 1`
