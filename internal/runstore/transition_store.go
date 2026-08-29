package runstore

import (
	"context"
	"errors"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

const stageTransitionDecisionColumns = `
source_execution_id, run_id, action, target_stage_name, target_execution_id, decided_at`

func (s *PostgresStore) RecordStageTransitionDecision(
	ctx context.Context,
	params RecordStageTransitionDecisionParams,
) (StageTransitionDecision, error) {
	if err := validateStageTransitionDecision(params); err != nil {
		return StageTransitionDecision{}, err
	}
	decision, err := scanStageTransitionDecision(s.db.QueryRow(ctx, `
INSERT INTO stage_transition_decisions (
    source_execution_id, run_id, action, target_stage_name, target_execution_id
) VALUES ($1, $2, $3, $4, $5)
RETURNING `+stageTransitionDecisionColumns,
		params.SourceExecutionID, params.RunID, params.Action,
		params.TargetStageName, params.TargetExecutionID,
	))
	if err != nil {
		switch persistencepostgres.SQLState(err) {
		case "23505":
			return StageTransitionDecision{}, fmt.Errorf(
				"record transition decision for StageExecution %q: %w",
				params.SourceExecutionID, ErrConflict,
			)
		case "23503":
			return StageTransitionDecision{}, fmt.Errorf(
				"record transition decision for StageExecution %q: %w",
				params.SourceExecutionID, ErrNotFound,
			)
		default:
			return StageTransitionDecision{}, fmt.Errorf(
				"record transition decision for StageExecution %q: %w",
				params.SourceExecutionID, err,
			)
		}
	}
	return decision, nil
}

func (s *PostgresStore) GetStageTransitionDecision(
	ctx context.Context,
	sourceExecutionID string,
) (StageTransitionDecision, error) {
	if err := validateOpaque("sourceExecutionID", sourceExecutionID); err != nil {
		return StageTransitionDecision{}, err
	}
	decision, err := scanStageTransitionDecision(s.db.QueryRow(ctx, `
SELECT `+stageTransitionDecisionColumns+`
FROM stage_transition_decisions
WHERE source_execution_id = $1`, sourceExecutionID))
	if errors.Is(err, pgx.ErrNoRows) {
		return StageTransitionDecision{}, fmt.Errorf(
			"get transition decision for StageExecution %q: %w", sourceExecutionID, ErrNotFound,
		)
	}
	if err != nil {
		return StageTransitionDecision{}, fmt.Errorf(
			"get transition decision for StageExecution %q: %w", sourceExecutionID, err,
		)
	}
	return decision, nil
}

func (s *PostgresStore) ListStageTransitionDecisions(
	ctx context.Context,
	runID string,
) ([]StageTransitionDecision, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT `+stageTransitionDecisionColumns+`
FROM stage_transition_decisions
WHERE run_id = $1
ORDER BY decided_at, source_execution_id`, runID)
	if err != nil {
		return nil, fmt.Errorf("list transition decisions for WorkflowRun %q: %w", runID, err)
	}
	defer rows.Close()
	result := make([]StageTransitionDecision, 0)
	for rows.Next() {
		decision, scanErr := scanStageTransitionDecision(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan transition decision for WorkflowRun %q: %w", runID, scanErr)
		}
		result = append(result, decision)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate transition decisions for WorkflowRun %q: %w", runID, err)
	}
	return result, nil
}

type transitionDecisionScanner interface {
	Scan(...any) error
}

func scanStageTransitionDecision(row transitionDecisionScanner) (StageTransitionDecision, error) {
	var result StageTransitionDecision
	err := row.Scan(
		&result.SourceExecutionID,
		&result.RunID,
		&result.Action,
		&result.TargetStageName,
		&result.TargetExecutionID,
		&result.DecidedAt,
	)
	return result, err
}

func validateStageTransitionDecision(params RecordStageTransitionDecisionParams) error {
	if err := validateOpaque("sourceExecutionID", params.SourceExecutionID); err != nil {
		return err
	}
	if err := validateOpaque("runID", params.RunID); err != nil {
		return err
	}
	switch params.Action {
	case StageTransitionNext, StageTransitionRetry:
		if params.TargetStageName == nil || params.TargetExecutionID == nil {
			return invalidf("%s transition requires a target Stage and StageExecution", params.Action)
		}
		if err := validateOpaque("targetStageName", *params.TargetStageName); err != nil {
			return err
		}
		return validateOpaque("targetExecutionID", *params.TargetExecutionID)
	case StageTransitionSucceed, StageTransitionFail:
		if params.TargetStageName != nil || params.TargetExecutionID != nil {
			return invalidf("%s transition must not identify a target", params.Action)
		}
		return nil
	default:
		return invalidf("unknown Stage transition action %q", params.Action)
	}
}
