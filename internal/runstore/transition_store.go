package runstore

import (
	"context"
	"errors"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

const stageTransitionDecisionColumns = `
stage_execution_id, run_id, transition_action, transition_target_stage_name, transition_target_execution_id,
transition_escalation_ordinal, transition_escalation_exhausted, transition_decided_at`

func (s *PostgresStore) RecordStageTransitionDecision(
	ctx context.Context,
	params RecordStageTransitionDecisionParams,
) (StageTransitionDecision, error) {
	if err := validateStageTransitionDecision(params); err != nil {
		return StageTransitionDecision{}, err
	}
	decision, err := scanStageTransitionDecision(s.db.QueryRow(ctx, recordStageTransitionDecisionSQL,
		params.SourceExecutionID, params.RunID, params.Action,
		params.TargetStageName, params.TargetExecutionID,
		params.EscalationOrdinal, params.EscalationExhausted,
	))
	if errors.Is(err, pgx.ErrNoRows) {
		var exists bool
		if readErr := s.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM stage_executions WHERE stage_execution_id=$1 AND run_id=$2)`, params.SourceExecutionID, params.RunID).Scan(&exists); readErr != nil {
			return StageTransitionDecision{}, fmt.Errorf("read transition source: %w", readErr)
		}
		if !exists {
			return StageTransitionDecision{}, ErrNotFound
		}
		return StageTransitionDecision{}, ErrConflict
	}
	if err != nil {
		switch persistencepostgres.Classify(err) {
		case persistencepostgres.ClassDuplicate:
			return StageTransitionDecision{}, fmt.Errorf(
				"record transition decision for StageExecution %q: %w",
				params.SourceExecutionID, ErrConflict,
			)
		case persistencepostgres.ClassReferenceViolation:
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

func (s *PostgresStore) ListStageTransitionDecisions(
	ctx context.Context,
	runID string,
) ([]StageTransitionDecision, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT `+stageTransitionDecisionColumns+`
FROM stage_executions
WHERE run_id = $1 AND transition_action IS NOT NULL
ORDER BY transition_decided_at, stage_execution_id`, runID)
	if err != nil {
		return nil, fmt.Errorf("list transition decisions for WorkflowRun %q: %w", runID, err)
	}
	result, err := pgx.CollectRows(rows, func(row pgx.CollectableRow) (StageTransitionDecision, error) {
		return scanStageTransitionDecision(row)
	})
	if err != nil {
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
		&result.EscalationOrdinal,
		&result.EscalationExhausted,
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
	case StageTransitionEscalate:
		if params.TargetStageName == nil || params.TargetExecutionID == nil {
			return invalidf("%s transition requires a target Stage and StageExecution", params.Action)
		}
		if params.EscalationOrdinal == nil || *params.EscalationOrdinal <= 0 || params.EscalationExhausted {
			return invalidf("escalate transition requires a positive ordinal and must not be exhausted")
		}
		if err := validateOpaque("targetStageName", *params.TargetStageName); err != nil {
			return err
		}
		return validateOpaque("targetExecutionID", *params.TargetExecutionID)
	case StageTransitionNext:
		if params.TargetStageName == nil || params.TargetExecutionID == nil {
			return invalidf("next transition requires a target Stage and StageExecution")
		}
		if params.EscalationExhausted {
			if params.EscalationOrdinal == nil || *params.EscalationOrdinal <= 0 {
				return invalidf("exhausted escalation transition requires a positive ordinal")
			}
		} else if params.EscalationOrdinal != nil {
			return invalidf("ordinary next transition must not have an escalation ordinal")
		}
		if err := validateOpaque("targetStageName", *params.TargetStageName); err != nil {
			return err
		}
		return validateOpaque("targetExecutionID", *params.TargetExecutionID)
	case StageTransitionRetry:
		if params.TargetStageName == nil || params.TargetExecutionID == nil ||
			params.EscalationOrdinal != nil || params.EscalationExhausted {
			return invalidf("retry transition requires a target and no escalation metadata")
		}
		if err := validateOpaque("targetStageName", *params.TargetStageName); err != nil {
			return err
		}
		return validateOpaque("targetExecutionID", *params.TargetExecutionID)
	case StageTransitionSucceed:
		if params.TargetStageName != nil || params.TargetExecutionID != nil {
			return invalidf("%s transition must not identify a target", params.Action)
		}
		if params.EscalationOrdinal != nil || params.EscalationExhausted {
			return invalidf("succeed transition must not have escalation metadata")
		}
		return nil
	case StageTransitionFail:
		if params.TargetStageName != nil || params.TargetExecutionID != nil {
			return invalidf("fail transition must not identify a target")
		}
		if params.EscalationExhausted {
			if params.EscalationOrdinal == nil || *params.EscalationOrdinal <= 0 {
				return invalidf("exhausted escalation transition requires a positive ordinal")
			}
		} else if params.EscalationOrdinal != nil {
			return invalidf("ordinary fail transition must not have an escalation ordinal")
		}
		return nil
	default:
		return invalidf("unknown Stage transition action %q", params.Action)
	}
}

// Validate checks the complete durable decision shape before a transactional
// persistence implementation performs any mutation.
func (p RecordStageTransitionDecisionParams) Validate() error {
	return validateStageTransitionDecision(p)
}
