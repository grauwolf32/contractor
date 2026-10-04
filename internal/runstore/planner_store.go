package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *PostgresStore) AppendPlannerEvent(ctx context.Context, params AppendPlannerEventParams) error {
	if params.SchedulerClaimID != "" {
		if err := validateOpaque("schedulerClaimID", params.SchedulerClaimID); err != nil {
			return err
		}
	}
	for field, value := range map[string]string{
		"eventID":               params.EventID,
		"sessionID":             params.SessionID,
		"stageExecutionID":      params.StageExecutionID,
		"invocationID":          params.InvocationID,
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
	if err := validateRunEventAppend(params.RunEvent); err != nil {
		return err
	}
	if params.EventSchemaVersion != contracts.APIVersion ||
		params.NewStateSchemaVersion != contracts.APIVersion {
		return invalidf("Planner event or state schema version is unsupported")
	}
	if len(params.Event) > maxRunEventDataBytes || len(params.NewState) > maxRunEventDataBytes {
		return invalidf("Planner event or state exceeds its bounded contract")
	}
	if err := validatePlannerRunEventIdentity(
		params.RunEvent, params.StageExecutionID, params.SessionID, params.InvocationID,
	); err != nil {
		return err
	}
	if params.RunEvent.EventID != params.EventID ||
		params.RunEvent.EventSchemaVersion != params.EventSchemaVersion {
		return invalidf("Planner event and WorkflowRun event identities must match")
	}
	var sessionExists bool
	var appended bool
	err := s.db.QueryRow(ctx, appendPlannerEventSQL,
		params.EventID, params.SessionID, params.SequenceNumber,
		params.EventSchemaVersion, []byte(params.Event),
		params.NewStateSchemaVersion, []byte(params.NewState),
		params.RunEvent.EventID, params.RunEvent.EventSchemaVersion,
		params.RunEvent.Kind, []byte(params.RunEvent.Data),
		params.StageExecutionID, params.InvocationID,
		params.SchedulerClaimID,
	).Scan(&sessionExists, &appended)
	if err != nil {
		sqlState := persistencepostgres.SQLState(err)
		if sqlState == persistencepostgres.SQLStateUniqueViolation {
			return fmt.Errorf("append Planner event %q: %w", params.EventID, ErrConflict)
		}
		if sqlState == persistencepostgres.SQLStateForeignKeyViolation || errors.Is(err, pgx.ErrNoRows) {
			return fmt.Errorf("append Planner event for session %q: %w", params.SessionID, ErrNotFound)
		}
		return fmt.Errorf("append Planner event %q: %w", params.EventID, err)
	}
	if !sessionExists {
		return fmt.Errorf("append Planner event for session %q: %w", params.SessionID, ErrNotFound)
	}
	if !appended {
		return fmt.Errorf("append Planner event %q: %w", params.EventID, ErrConflict)
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
       next_event_sequence, created_at, updated_at
FROM planner_sessions
WHERE session_id = $1`, sessionID).Scan(
		&result.SessionID, &result.StageExecutionID, &result.InvocationID,
		&result.StateSchemaVersion, &state, &result.NextEventSequence,
		&result.CreatedAt, &result.UpdatedAt,
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
SELECT event_id, session_id, sequence_number, event_schema_version, event,
       run_id, run_event_sequence, created_at
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
			&event.EventSchemaVersion, &payload,
			&event.RunID, &event.RunEventSequence, &event.CreatedAt,
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
