package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *PostgresStore) AppendPlannerEvent(ctx context.Context, params AppendPlannerEventParams) error {
	for field, value := range map[string]string{
		"eventID":               params.EventID,
		"sessionID":             params.SessionID,
		"eventSchemaVersion":    params.EventSchemaVersion,
		"newStateSchemaVersion": params.NewStateSchemaVersion,
	} {
		if err := validateOpaque(field, value); err != nil {
			return err
		}
	}
	if params.SequenceNumber <= 0 {
		return invalidf("Planner event sequenceNumber must be positive")
	}
	if err := validateJSONObject("Planner event", params.Event); err != nil {
		return err
	}
	if err := validateJSONObject("Planner state", params.NewState); err != nil {
		return err
	}
	var sessionID string
	err := s.db.QueryRow(ctx, `
WITH inserted AS (
    INSERT INTO planner_events (
        event_id, session_id, sequence_number, event_schema_version, event
    ) VALUES ($1, $2, $3, $4, $5::jsonb)
    RETURNING session_id
)
UPDATE planner_sessions AS session
SET state_schema_version = $6,
    state = $7::jsonb,
    updated_at = clock_timestamp()
FROM inserted
WHERE session.session_id = inserted.session_id
RETURNING session.session_id`,
		params.EventID, params.SessionID, params.SequenceNumber,
		params.EventSchemaVersion, []byte(params.Event),
		params.NewStateSchemaVersion, []byte(params.NewState),
	).Scan(&sessionID)
	if err != nil {
		sqlState := persistencepostgres.SQLState(err)
		if sqlState == "23505" {
			return fmt.Errorf("append Planner event %q: %w", params.EventID, ErrConflict)
		}
		if sqlState == "23503" || errors.Is(err, pgx.ErrNoRows) {
			return fmt.Errorf("append Planner event for session %q: %w", params.SessionID, ErrNotFound)
		}
		return fmt.Errorf("append Planner event %q: %w", params.EventID, err)
	}
	return nil
}

func (s *PostgresStore) GetPlannerSession(ctx context.Context, sessionID string) (PlannerSession, error) {
	if err := validateOpaque("sessionID", sessionID); err != nil {
		return PlannerSession{}, err
	}
	var result PlannerSession
	var state []byte
	err := s.db.QueryRow(ctx, `
SELECT session_id, stage_execution_id, invocation_id, state_schema_version, state,
       created_at, updated_at
FROM planner_sessions
WHERE session_id = $1`, sessionID).Scan(
		&result.SessionID, &result.StageExecutionID, &result.InvocationID,
		&result.StateSchemaVersion, &state, &result.CreatedAt, &result.UpdatedAt,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return PlannerSession{}, fmt.Errorf("get Planner session %q: %w", sessionID, ErrNotFound)
	}
	if err != nil {
		return PlannerSession{}, fmt.Errorf("get Planner session %q: %w", sessionID, err)
	}
	result.State = append(json.RawMessage(nil), state...)
	if err := validateJSONObject("persisted Planner state", result.State); err != nil {
		return PlannerSession{}, err
	}
	return result, nil
}

func (s *PostgresStore) ListPlannerEvents(
	ctx context.Context,
	sessionID string,
	afterSequence int64,
) ([]PlannerEvent, error) {
	if err := validateOpaque("sessionID", sessionID); err != nil {
		return nil, err
	}
	if afterSequence < 0 {
		return nil, invalidf("afterSequence must be non-negative")
	}
	rows, err := s.db.Query(ctx, `
SELECT event_id, session_id, sequence_number, event_schema_version, event, created_at
FROM planner_events
WHERE session_id = $1 AND sequence_number > $2
ORDER BY sequence_number`, sessionID, afterSequence)
	if err != nil {
		return nil, fmt.Errorf("list Planner events for session %q: %w", sessionID, err)
	}
	defer rows.Close()
	var result []PlannerEvent
	for rows.Next() {
		var event PlannerEvent
		var payload []byte
		if err := rows.Scan(
			&event.EventID, &event.SessionID, &event.SequenceNumber,
			&event.EventSchemaVersion, &payload, &event.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("scan Planner event for session %q: %w", sessionID, err)
		}
		event.Event = append(json.RawMessage(nil), payload...)
		if err := validateJSONObject("persisted Planner event", event.Event); err != nil {
			return nil, err
		}
		result = append(result, event)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Planner events for session %q: %w", sessionID, err)
	}
	return result, nil
}
