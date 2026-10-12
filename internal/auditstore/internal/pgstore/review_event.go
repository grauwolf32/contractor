package pgstore

import (
	"context"
	"encoding/json"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

// AppendReviewEvent advances the Audit revision and records the matching
// review event in one statement. Callers can use a transaction-scoped store
// to keep the event atomic with the review request or decision it describes.
func (s *ReviewStore) AppendReviewEvent(ctx context.Context, params auditstore.ReviewEventParams) error {
	return s.appendReviewEvents(ctx, []auditstore.ReviewEventParams{params})
}

// appendReviewEvents records review events for ReviewStore.AppendReviewEvents
// and for the item review renewals of the Controller and of resume.
func (s *store) appendReviewEvents(ctx context.Context, events []auditstore.ReviewEventParams) error {
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
			return auditstore.ErrInvalid
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
