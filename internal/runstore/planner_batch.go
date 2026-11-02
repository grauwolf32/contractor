package runstore

import (
	"context"
	"fmt"
)

// ReadPlannerSessions consumes one session at a time from a single query,
// without retaining every raw state beside the consumer's projections.
// Authorization and full state/identity validation belong to the consumer,
// which must not issue database calls while this query owns its connection.
// Missing sessions are omitted. Any read or consumer error stops the batch
// and closes the rows; callers discard their partially assembled projections.
func (s *PostgresStore) ReadPlannerSessions(ctx context.Context, ids []string, consume func(PlannerSession) error) error {
	if consume == nil {
		return invalidf("Planner session consumer is required")
	}
	for _, id := range ids {
		if err := validateOpaque("sessionID", id); err != nil {
			return err
		}
	}
	if len(ids) == 0 {
		return nil
	}
	rows, err := s.db.Query(ctx, `SELECT `+plannerSessionColumns+`
FROM planner_sessions WHERE session_id = ANY($1::text[])`, ids)
	if err != nil {
		return fmt.Errorf("list Planner sessions: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		session, err := scanPlannerSession(rows)
		if err != nil {
			return fmt.Errorf("scan Planner session: %w", err)
		}
		if err := consume(session); err != nil {
			return err
		}
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate Planner sessions: %w", err)
	}
	return nil
}
