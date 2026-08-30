package runstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *PostgresStore) CreateStageExecution(
	ctx context.Context,
	params CreateStageExecutionParams,
) (StageExecution, error) {
	if params.ExecutionConfigVariant == "" {
		params.ExecutionConfigVariant = StageExecutionConfigBase
	}
	if err := validateCreateStageExecution(params); err != nil {
		return StageExecution{}, err
	}
	contextSnapshot := params.StageContext
	if contextSnapshot.Parameters == nil {
		contextSnapshot.Parameters = map[string]string{}
	}
	if contextSnapshot.Artifacts == nil {
		contextSnapshot.Artifacts = map[string]PinnedContextArtifact{}
	}
	encodedContext, err := json.Marshal(contextSnapshot)
	if err != nil {
		return StageExecution{}, fmt.Errorf("create StageExecution: encode StageContext: %w", err)
	}
	result, err := scanStageExecution(s.db.QueryRow(ctx, `
INSERT INTO stage_executions (
    stage_execution_id, run_id, stage_name, attempt, previous_execution_id,
    execution_config_variant, escalation_ordinal,
    stage_spec_schema_version, stage_spec_snapshot,
    stage_context_schema_version, stage_context_snapshot,
    state, state_reason_code, state_reason_message
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10, $11::jsonb, 'preparing', 'created', '')
RETURNING `+stageExecutionColumns,
		params.StageExecutionID, params.RunID, params.StageName, params.Attempt,
		params.PreviousExecutionID, params.ExecutionConfigVariant, params.EscalationOrdinal,
		params.StageSpecSchemaVersion, []byte(params.StageSpecSnapshot),
		params.StageContextSchemaVersion, encodedContext,
	))
	if err != nil {
		sqlState := persistencepostgres.SQLState(err)
		if sqlState == "23505" {
			return StageExecution{}, fmt.Errorf("create StageExecution %q: %w", params.StageExecutionID, ErrConflict)
		}
		if sqlState == "23503" {
			return StageExecution{}, fmt.Errorf("create StageExecution %q: %w", params.StageExecutionID, ErrNotFound)
		}
		return StageExecution{}, fmt.Errorf("create StageExecution %q: %w", params.StageExecutionID, err)
	}
	return result, nil
}

func (s *PostgresStore) GetStageExecution(ctx context.Context, stageExecutionID string) (StageExecution, error) {
	if err := validateOpaque("stageExecutionID", stageExecutionID); err != nil {
		return StageExecution{}, err
	}
	result, err := scanStageExecution(s.db.QueryRow(ctx,
		`SELECT `+stageExecutionColumns+` FROM stage_executions WHERE stage_execution_id = $1`,
		stageExecutionID,
	))
	if errors.Is(err, pgx.ErrNoRows) {
		return StageExecution{}, fmt.Errorf("get StageExecution %q: %w", stageExecutionID, ErrNotFound)
	}
	if err != nil {
		return StageExecution{}, fmt.Errorf("get StageExecution %q: %w", stageExecutionID, err)
	}
	return result, nil
}

// ListTerminalStageExecutionsWithAllocations supports best-effort release
// recovery after the semantic terminal transaction committed. Stage allocation
// rows remain immutable provenance, so callers determine liveness through the
// volatile Control Plane registry before issuing an idempotent release.
func (s *PostgresStore) ListTerminalStageExecutionsWithAllocations(
	ctx context.Context,
) ([]StageExecution, error) {
	rows, err := s.db.Query(ctx, `
SELECT `+stageExecutionColumns+`
FROM stage_executions AS execution
WHERE execution.state IN ('succeeded', 'failed', 'interrupted', 'cancelled')
  AND EXISTS (
      SELECT 1 FROM stage_allocations AS allocation
      WHERE allocation.stage_execution_id = execution.stage_execution_id
  )
ORDER BY execution.terminal_at, execution.stage_execution_id`)
	if err != nil {
		return nil, fmt.Errorf("list terminal StageExecutions with allocations: %w", err)
	}
	defer rows.Close()
	result := make([]StageExecution, 0)
	for rows.Next() {
		execution, err := scanStageExecution(rows)
		if err != nil {
			return nil, fmt.Errorf("scan terminal StageExecution with allocations: %w", err)
		}
		result = append(result, execution)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate terminal StageExecutions with allocations: %w", err)
	}
	return result, nil
}

func (s *PostgresStore) ListStageExecutions(ctx context.Context, runID string) ([]StageExecution, error) {
	if err := validateOpaque("runID", runID); err != nil {
		return nil, err
	}
	rows, err := s.db.Query(ctx, `
SELECT `+stageExecutionColumns+`
FROM stage_executions
WHERE run_id = $1
ORDER BY created_at, stage_execution_id`, runID)
	if err != nil {
		return nil, fmt.Errorf("list StageExecutions for WorkflowRun %q: %w", runID, err)
	}
	defer rows.Close()
	var result []StageExecution
	for rows.Next() {
		execution, scanErr := scanStageExecution(rows)
		if scanErr != nil {
			return nil, fmt.Errorf("scan StageExecution for WorkflowRun %q: %w", runID, scanErr)
		}
		result = append(result, execution)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate StageExecutions for WorkflowRun %q: %w", runID, err)
	}
	return result, nil
}

func (s *PostgresStore) StartPlanner(ctx context.Context, params StartPlannerParams) error {
	for field, value := range map[string]string{
		"stageExecutionID": params.StageExecutionID, "sessionID": params.SessionID,
		"invocationID": params.InvocationID, "stateSchemaVersion": params.StateSchemaVersion,
	} {
		if err := validateOpaque(field, value); err != nil {
			return err
		}
	}
	if err := validateJSONObject("initial Planner state", params.InitialState); err != nil {
		return err
	}
	if err := validateReason(params.Reason); err != nil {
		return err
	}
	var sessionID string
	err := s.db.QueryRow(ctx, `
WITH transitioned AS (
    UPDATE stage_executions
    SET state = 'running',
        state_reason_code = $6,
        state_reason_message = $7,
        planner_session_id = $2,
        planner_invocation_id = $3,
        planner_started_at = clock_timestamp(),
        updated_at = clock_timestamp()
    WHERE stage_execution_id = $1 AND state = 'preparing'
    RETURNING stage_execution_id
)
INSERT INTO planner_sessions (
    session_id, stage_execution_id, invocation_id, state_schema_version, state
)
SELECT $2, stage_execution_id, $3, $4, $5::jsonb
FROM transitioned
RETURNING session_id`,
		params.StageExecutionID, params.SessionID, params.InvocationID,
		params.StateSchemaVersion, []byte(params.InitialState), params.Reason.Code, params.Reason.Message,
	).Scan(&sessionID)
	if errors.Is(err, pgx.ErrNoRows) {
		return &StateConflictError{Resource: "StageExecution", ID: params.StageExecutionID, Expected: string(StagePreparing)}
	}
	if err != nil {
		if persistencepostgres.SQLState(err) == "23505" {
			return fmt.Errorf("start Planner for StageExecution %q: %w", params.StageExecutionID, ErrConflict)
		}
		return fmt.Errorf("start Planner for StageExecution %q: %w", params.StageExecutionID, err)
	}
	return nil
}

func (s *PostgresStore) EnterFinalizing(ctx context.Context, params EnterFinalizingParams) error {
	for field, value := range map[string]string{
		"stageExecutionID":    params.StageExecutionID,
		"resultSchemaVersion": params.ResultSchemaVersion,
		"finalizationID":      params.FinalizationID,
	} {
		if err := validateOpaque(field, value); err != nil {
			return err
		}
	}
	candidate := normalizeStageResult(params.Candidate)
	if err := candidate.Validate(); err != nil {
		return fmt.Errorf("%w: candidate StageResult: %v", ErrInvalid, err)
	}
	if params.Deadline.IsZero() {
		return invalidf("finalization deadline is required")
	}
	if err := validateReason(params.Reason); err != nil {
		return err
	}
	encoded, err := json.Marshal(candidate)
	if err != nil {
		return fmt.Errorf("enter finalizing: encode candidate StageResult: %w", err)
	}
	tag, err := s.db.Exec(ctx, `
UPDATE stage_executions
SET state = 'finalizing',
    state_reason_code = $6,
    state_reason_message = $7,
    candidate_result_schema_version = $2,
    candidate_stage_result = $3::jsonb,
    finalization_id = $4,
    finalization_deadline = $5,
    updated_at = clock_timestamp()
WHERE stage_execution_id = $1 AND state = 'running'`,
		params.StageExecutionID, params.ResultSchemaVersion, encoded,
		params.FinalizationID, params.Deadline, params.Reason.Code, params.Reason.Message,
	)
	return transitionResult("enter finalizing", params.StageExecutionID, StageRunning, tag.RowsAffected(), err)
}

func (s *PostgresStore) CompleteStageResult(
	ctx context.Context,
	stageExecutionID string,
	resultSchemaVersion string,
	result contracts.StageContentResult,
) error {
	if err := validateOpaque("stageExecutionID", stageExecutionID); err != nil {
		return err
	}
	if err := validateOpaque("resultSchemaVersion", resultSchemaVersion); err != nil {
		return err
	}
	result = normalizeStageResult(result)
	if err := result.Validate(); err != nil {
		return fmt.Errorf("%w: accepted StageResult: %v", ErrInvalid, err)
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return fmt.Errorf("complete StageResult: encode result: %w", err)
	}
	tag, err := s.db.Exec(ctx, `
UPDATE stage_executions
SET state = $4,
    state_reason_code = 'result_accepted',
    state_reason_message = '',
    accepted_result_schema_version = $2,
    accepted_stage_result = $3::jsonb,
    terminal_at = clock_timestamp(),
    updated_at = clock_timestamp()
WHERE stage_execution_id = $1
  AND state = 'finalizing'
  AND candidate_result_schema_version = $2
  AND candidate_stage_result = $3::jsonb`,
		stageExecutionID, resultSchemaVersion, encoded, result.Outcome,
	)
	return transitionResult("complete StageResult", stageExecutionID, StageFinalizing, tag.RowsAffected(), err)
}

func (s *PostgresStore) EnterAborting(ctx context.Context, params EnterAbortingParams) error {
	for field, value := range map[string]string{
		"stageExecutionID":         params.StageExecutionID,
		"terminationSchemaVersion": params.TerminationSchemaVersion,
		"abortID":                  params.AbortID,
	} {
		if err := validateOpaque(field, value); err != nil {
			return err
		}
	}
	if params.ExpectedState != StagePreparing && params.ExpectedState != StageRunning {
		return invalidf("aborting requires expected state preparing or running")
	}
	expectedPhase := TerminationPhase(params.ExpectedState)
	if params.Termination.Phase != expectedPhase {
		return invalidf("StageTermination phase %q does not match expected state %q", params.Termination.Phase, params.ExpectedState)
	}
	if err := params.Termination.Validate(); err != nil {
		return err
	}
	if params.Deadline.IsZero() {
		return invalidf("abort deadline is required")
	}
	if err := validateReason(params.Reason); err != nil {
		return err
	}
	encoded, err := json.Marshal(params.Termination)
	if err != nil {
		return fmt.Errorf("enter aborting: encode StageTermination: %w", err)
	}
	tag, err := s.db.Exec(ctx, `
UPDATE stage_executions
SET state = 'aborting',
    state_reason_code = $7,
    state_reason_message = $8,
    termination_schema_version = $3,
    stage_termination = $4::jsonb,
    abort_id = $5,
    abort_deadline = $6,
    updated_at = clock_timestamp()
WHERE stage_execution_id = $1 AND state = $2`,
		params.StageExecutionID, params.ExpectedState, params.TerminationSchemaVersion,
		encoded, params.AbortID, params.Deadline, params.Reason.Code, params.Reason.Message,
	)
	return transitionResult("enter aborting", params.StageExecutionID, params.ExpectedState, tag.RowsAffected(), err)
}

func (s *PostgresStore) CompleteStageTermination(ctx context.Context, stageExecutionID string) error {
	if err := validateOpaque("stageExecutionID", stageExecutionID); err != nil {
		return err
	}
	tag, err := s.db.Exec(ctx, `
UPDATE stage_executions
SET state = stage_termination->>'outcome',
    state_reason_code = 'termination_committed',
    state_reason_message = '',
    terminal_at = clock_timestamp(),
    updated_at = clock_timestamp()
WHERE stage_execution_id = $1 AND state = 'aborting'`, stageExecutionID)
	return transitionResult("complete StageTermination", stageExecutionID, StageAborting, tag.RowsAffected(), err)
}

func transitionResult(operation, id string, expected StageExecutionState, rows int64, err error) error {
	if err != nil {
		return fmt.Errorf("%s for StageExecution %q: %w", operation, id, err)
	}
	if rows != 1 {
		return &StateConflictError{Resource: "StageExecution", ID: id, Expected: string(expected)}
	}
	return nil
}

func validateCreateStageExecution(params CreateStageExecutionParams) error {
	for field, value := range map[string]string{
		"stageExecutionID": params.StageExecutionID, "runID": params.RunID,
		"stageName": params.StageName, "stageSpecSchemaVersion": params.StageSpecSchemaVersion,
		"stageContextSchemaVersion": params.StageContextSchemaVersion,
	} {
		if err := validateOpaque(field, value); err != nil {
			return err
		}
	}
	if params.Attempt <= 0 {
		return invalidf("attempt must be positive")
	}
	if params.Attempt == 1 && params.PreviousExecutionID != nil || params.Attempt > 1 && params.PreviousExecutionID == nil {
		return invalidf("attempt and previousExecutionID have inconsistent lineage")
	}
	if params.PreviousExecutionID != nil {
		if err := validateOpaque("previousExecutionID", *params.PreviousExecutionID); err != nil {
			return err
		}
	}
	variant := params.ExecutionConfigVariant
	if variant == "" {
		variant = StageExecutionConfigBase
	}
	switch variant {
	case StageExecutionConfigBase:
		if params.EscalationOrdinal != nil {
			return invalidf("base executionConfig variant must not have an escalation ordinal")
		}
	case StageExecutionConfigFailedEscalation, StageExecutionConfigInterruptedEscalation:
		if params.EscalationOrdinal == nil || *params.EscalationOrdinal <= 0 {
			return invalidf("escalation executionConfig variant requires a positive ordinal")
		}
	default:
		return invalidf("unknown executionConfig variant %q", variant)
	}
	if err := validateJSONObject("stageSpecSnapshot", params.StageSpecSnapshot); err != nil {
		return err
	}
	return params.StageContext.Validate()
}

func validateNamespace(value string) error {
	if err := validateOpaque("namespace", value); err != nil {
		return err
	}
	if strings.Contains(value, "/") {
		return invalidf("namespace must not contain slash")
	}
	return nil
}

func normalizeStageResult(result contracts.StageContentResult) contracts.StageContentResult {
	if result.Artifacts == nil {
		result.Artifacts = map[string]contracts.ArtifactRef{}
	}
	return result
}
