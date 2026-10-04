package auditstore

// SQL statements for review_event.go.

// appendReviewEventSQL bumps the revision and event sequence of Audit $1 and
// appends an event of kind $2 for entity $3 with JSON summary $4, returning its
// sequence number. Used by PostgresStore.AppendReviewEvent.
var appendReviewEventSQL = `
WITH advanced AS (
    UPDATE audits
       SET revision = revision + 1, next_event_sequence = next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
     WHERE audit_id = $1
    RETURNING audit_id, next_event_sequence
), recorded AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, summary)
    SELECT audit_id, next_event_sequence - 1, $2, $3, $4::jsonb
      FROM advanced
    RETURNING sequence_number
)
SELECT sequence_number FROM recorded`
