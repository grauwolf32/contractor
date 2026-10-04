package auditstore

import (
	"context"
	"encoding/json"
)

// ReviewEventParams describes an event emitted by a human review mutation.
type ReviewEventParams struct {
	AuditID  string
	Kind     string
	EntityID string
	Summary  map[string]any
}

// AppendReviewEvent advances the Audit revision and records the matching
// review event in one statement. Callers can use a transaction-scoped store
// to keep the event atomic with the review request or decision it describes.
func (s *PostgresStore) AppendReviewEvent(ctx context.Context, params ReviewEventParams) error {
	summary, err := json.Marshal(params.Summary)
	if err != nil {
		return err
	}
	var sequence int64
	return s.db.QueryRow(ctx, `
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
SELECT sequence_number FROM recorded`,
		params.AuditID, params.Kind, params.EntityID, summary,
	).Scan(&sequence)
}
