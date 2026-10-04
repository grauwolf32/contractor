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
	return s.db.QueryRow(ctx, appendReviewEventSQL,
		params.AuditID, params.Kind, params.EntityID, summary,
	).Scan(&sequence)
}
