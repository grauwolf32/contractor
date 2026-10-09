package auditstore

// SQL statements for review_event.go.

// appendReviewEventsSQL reserves N sequence numbers on Audit $1 for the JSON
// event array $2. Array order defines contiguous sequences; one UPDATE and
// INSERT make the whole allocation atomic, including validation failures.
// Used by PostgresStore.AppendReviewEvents and its single-event wrapper.
var appendReviewEventsSQL = `
WITH advanced AS (
    UPDATE audits
       SET revision = revision + jsonb_array_length($2::jsonb),
           next_event_sequence = next_event_sequence + jsonb_array_length($2::jsonb),
           updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
     WHERE audit_id = $1
    RETURNING audit_id, next_event_sequence
), recorded AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, entity_revision, summary)
    SELECT audit_id, next_event_sequence - jsonb_array_length($2::jsonb) + event.position - 1,
           event.value->>'kind', event.value->>'entityId',
           (event.value->>'entityRevision')::bigint, event.value->'summary'
      FROM advanced CROSS JOIN jsonb_array_elements($2::jsonb) WITH ORDINALITY AS event(value, position)
    RETURNING sequence_number
)
SELECT count(*) FROM recorded`
