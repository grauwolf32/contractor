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
		"sessionID":             params.SessionID,
		"stageExecutionID":      params.StageExecutionID,
		"invocationID":          params.InvocationID,
		"newStateSchemaVersion": params.NewStateSchemaVersion,
	} {
		if err := validateOpaque(field, value); err != nil {
			return err
		}
	}
	if params.SequenceNumber <= 0 {
		return invalidf("Planner event sequenceNumber must be positive")
	}
	if err := validateJSONObject("Planner state", params.NewState); err != nil {
		return err
	}
	if err := validateRunEventAppend(params.RunEvent); err != nil {
		return err
	}
	if params.RunEvent.EventSchemaVersion != contracts.APIVersion ||
		params.NewStateSchemaVersion != contracts.APIVersion {
		return invalidf("Planner event or state schema version is unsupported")
	}
	if len(params.NewState) > maxRunEventDataBytes {
		return invalidf("Planner state exceeds its bounded contract")
	}
	if err := validatePlannerRunEventIdentity(
		params.RunEvent, params.StageExecutionID, params.SessionID, params.InvocationID,
	); err != nil {
		return err
	}
	var sessionExists bool
	var appended bool
	err := s.db.QueryRow(ctx, appendPlannerEventSQL,
		params.SessionID, params.SequenceNumber,
		params.NewStateSchemaVersion, []byte(params.NewState),
		params.RunEvent.EventID, params.RunEvent.EventSchemaVersion,
		params.RunEvent.Kind, []byte(params.RunEvent.Data),
		params.StageExecutionID, params.InvocationID,
		params.SchedulerClaimID,
	).Scan(&sessionExists, &appended)
	if err != nil {
		sqlState := persistencepostgres.SQLState(err)
		if sqlState == persistencepostgres.SQLStateUniqueViolation {
			return fmt.Errorf("append Planner event %q: %w", params.RunEvent.EventID, ErrConflict)
		}
		if sqlState == persistencepostgres.SQLStateForeignKeyViolation || errors.Is(err, pgx.ErrNoRows) {
			return fmt.Errorf("append Planner event for session %q: %w", params.SessionID, ErrNotFound)
		}
		return fmt.Errorf("append Planner event %q: %w", params.RunEvent.EventID, err)
	}
	if !sessionExists {
		return fmt.Errorf("append Planner event for session %q: %w", params.SessionID, ErrNotFound)
	}
	if !appended {
		return fmt.Errorf("append Planner event %q: %w", params.RunEvent.EventID, ErrConflict)
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
