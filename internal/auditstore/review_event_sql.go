package auditstore

// SQL statements for review_event.go.

// reviewEvents are the JSON event array $2, in array order.
var reviewEvents = auditEvents(auditEventArray("$2::jsonb"))

// appendReviewEventsSQL records the JSON event array $2 on Audit $1 in array
// order. One UPDATE reserves and appends the whole batch atomically, including
// validation failures. Returns 1, or 0 when the Audit does not exist.
// Used by PostgresStore.AppendReviewEvents and its single-event wrapper.
var appendReviewEventsSQL = `
WITH advanced AS (
    UPDATE audits AS audit
       SET ` + reviewEvents.advance() + `,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
     WHERE audit.audit_id = $1
    RETURNING ` + reviewEvents.appended() + `
)
SELECT count(*) FROM advanced`
