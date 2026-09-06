package runstore

import (
	"context"
	"fmt"
)

// GetPlannerSessions reads a set in one query. Authorization and complete
// execution/invocation identity checks belong to the consuming service.
func (s *PostgresStore) GetPlannerSessions(ctx context.Context, ids []string) (map[string]PlannerSession, error) {
	result := make(map[string]PlannerSession, len(ids))
	for _, id := range ids {
		if err := validateOpaque("sessionID", id); err != nil {
			return nil, err
		}
	}
	if len(ids) == 0 {
		return result, nil
	}
	rows, err := s.db.Query(ctx, `SELECT session_id, stage_execution_id, invocation_id,
state_schema_version, state, next_event_sequence, created_at, updated_at
FROM planner_sessions WHERE session_id = ANY($1::text[])`, ids)
	if err != nil {
		return nil, fmt.Errorf("list Planner sessions: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var session PlannerSession
		if err := rows.Scan(&session.SessionID, &session.StageExecutionID, &session.InvocationID,
			&session.StateSchemaVersion, &session.State, &session.NextEventSequence, &session.CreatedAt, &session.UpdatedAt); err != nil {
			return nil, fmt.Errorf("scan Planner session: %w", err)
		}
		if err := validateJSONObject("persisted Planner state", session.State); err != nil {
			return nil, err
		}
		result[session.SessionID] = session
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Planner sessions: %w", err)
	}
	return result, nil
}
