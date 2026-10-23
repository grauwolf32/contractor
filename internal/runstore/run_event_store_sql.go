package runstore

// SQL statements for run_event_store.go.

// listRunEventsSQL reads one page of Run $1's events after sequence $2
// (API-R110). It walks the primary key one event at a time, so it measures
// only the events it returns and at most one more: it continues from an event
// only while that event is within both the row bound $3 and, counting it, the
// data budget $4 of persisted bytes, measured as octet_length(data::text),
// the measure of the per-event CHECK. Every walked event is a result row; the
// returned flag marks the contiguous prefix within both bounds, and a walked
// event outside it is the look-ahead that shows a committed successor, whose
// data is withheld. The walk runs in one snapshot and never measures an event
// beyond the look-ahead, however long the retained history is. Used by
// PostgresStore.ListRunEvents.
var listRunEventsSQL = `
WITH RECURSIVE walk AS (
    (
        SELECT run_id, sequence_number, event_id, event_schema_version, kind, data, occurred_at,
               octet_length(data::text)::bigint AS cumulative_data_bytes, 1 AS position
        FROM workflow_run_events
        WHERE run_id = $1 AND sequence_number > $2
        ORDER BY sequence_number
        LIMIT 1
    )
    UNION ALL
    SELECT following.run_id, following.sequence_number, following.event_id,
           following.event_schema_version, following.kind, following.data, following.occurred_at,
           walk.cumulative_data_bytes + octet_length(following.data::text), walk.position + 1
    FROM walk
    CROSS JOIN LATERAL (
        SELECT run_id, sequence_number, event_id, event_schema_version, kind, data, occurred_at
        FROM workflow_run_events
        WHERE run_id = $1 AND sequence_number > walk.sequence_number
        ORDER BY sequence_number
        LIMIT 1
    ) AS following
    WHERE walk.position <= $3 AND walk.cumulative_data_bytes <= $4
)
SELECT run_id, sequence_number, event_id, event_schema_version, kind, occurred_at,
       position <= $3 AND cumulative_data_bytes <= $4 AS returned,
       CASE WHEN position <= $3 AND cumulative_data_bytes <= $4 THEN data END AS data
FROM walk
ORDER BY sequence_number`
