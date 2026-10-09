package auditstore

import (
	"context"
	"encoding/json"

	"github.com/jackc/pgx/v5"
)

// ReviewEventParams describes an event emitted by a human review mutation.
type ReviewEventParams struct {
	AuditID        string
	Kind           string
	EntityID       string
	EntityRevision *uint64
	Summary        map[string]any
}

// AppendReviewEvent advances the Audit revision and records the matching
// review event in one statement. Callers can use a transaction-scoped store
// to keep the event atomic with the review request or decision it describes.
func (s *PostgresStore) AppendReviewEvent(ctx context.Context, params ReviewEventParams) error {
	return s.AppendReviewEvents(ctx, []ReviewEventParams{params})
}

// AppendReviewEvents reserves a contiguous sequence range and records all
// events atomically. Every event advances the Audit revision once, preserving
// the single-event contract; a rejected event rolls back the entire statement.
func (s *PostgresStore) AppendReviewEvents(ctx context.Context, events []ReviewEventParams) error {
	if len(events) == 0 {
		return nil
	}
	type event struct {
		Kind           string         `json:"kind"`
		EntityID       string         `json:"entityId"`
		EntityRevision *uint64        `json:"entityRevision"`
		Summary        map[string]any `json:"summary"`
	}
	auditID := events[0].AuditID
	input := make([]event, len(events))
	for i, value := range events {
		if value.AuditID != auditID {
			return ErrInvalid
		}
		input[i] = event{value.Kind, value.EntityID, value.EntityRevision, value.Summary}
	}
	encoded, err := json.Marshal(input)
	if err != nil {
		return err
	}
	var count int
	if err := s.db.QueryRow(ctx, appendReviewEventsSQL, auditID, encoded).Scan(&count); err != nil {
		return err
	}
	if count == 0 {
		return pgx.ErrNoRows
	}
	return nil
}
