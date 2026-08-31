package runstore

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/grauwolf32/contractor/internal/contracts"
)

const stageExecutionColumns = `
stage_execution_id, run_id, stage_name, attempt, previous_execution_id,
execution_config_variant, escalation_ordinal,
stage_spec_schema_version, stage_spec_snapshot,
stage_context_schema_version, stage_context_snapshot,
state, state_reason_code, state_reason_message,
planner_session_id, planner_invocation_id,
candidate_result_schema_version, candidate_stage_result,
accepted_result_schema_version, accepted_stage_result,
termination_schema_version, stage_termination,
finalization_id, finalization_deadline, abort_id, abort_deadline,
created_at, updated_at, planner_started_at, terminal_at`

func prefixedStageExecutionColumns(alias string) string {
	parts := strings.Split(stageExecutionColumns, ",")
	for index, part := range parts {
		parts[index] = alias + "." + strings.TrimSpace(part)
	}
	return strings.Join(parts, ", ")
}

func scanStageExecution(row rowScanner) (StageExecution, error) {
	var result StageExecution
	var stageSpec []byte
	var stageContext []byte
	var state string
	var candidate []byte
	var accepted []byte
	var termination []byte
	if err := row.Scan(
		&result.StageExecutionID, &result.RunID, &result.StageName, &result.Attempt,
		&result.PreviousExecutionID, &result.ExecutionConfigVariant, &result.EscalationOrdinal,
		&result.StageSpecSchemaVersion, &stageSpec,
		&result.StageContextSchemaVersion, &stageContext,
		&state, &result.StateReason.Code, &result.StateReason.Message,
		&result.PlannerSessionID, &result.PlannerInvocationID,
		&result.CandidateResultSchemaVersion, &candidate,
		&result.AcceptedResultSchemaVersion, &accepted,
		&result.TerminationSchemaVersion, &termination,
		&result.FinalizationID, &result.FinalizationDeadline, &result.AbortID, &result.AbortDeadline,
		&result.CreatedAt, &result.UpdatedAt, &result.PlannerStartedAt, &result.TerminalAt,
	); err != nil {
		return StageExecution{}, err
	}
	result.State = StageExecutionState(state)
	result.StageSpecSnapshot = append(json.RawMessage(nil), stageSpec...)
	if err := validateJSONObject("persisted stageSpecSnapshot", result.StageSpecSnapshot); err != nil {
		return StageExecution{}, err
	}
	if err := json.Unmarshal(stageContext, &result.StageContext); err != nil {
		return StageExecution{}, fmt.Errorf("decode persisted StageContext: %w", err)
	}
	if err := result.StageContext.Validate(); err != nil {
		return StageExecution{}, fmt.Errorf("decode persisted StageContext: %w", err)
	}
	if candidate != nil {
		var decoded contracts.StageContentResult
		if err := json.Unmarshal(candidate, &decoded); err != nil {
			return StageExecution{}, fmt.Errorf("decode candidate StageResult: %w", err)
		}
		if err := decoded.Validate(); err != nil {
			return StageExecution{}, fmt.Errorf("validate candidate StageResult: %w", err)
		}
		result.CandidateResult = &decoded
	}
	if accepted != nil {
		var decoded contracts.StageContentResult
		if err := json.Unmarshal(accepted, &decoded); err != nil {
			return StageExecution{}, fmt.Errorf("decode accepted StageResult: %w", err)
		}
		if err := decoded.Validate(); err != nil {
			return StageExecution{}, fmt.Errorf("validate accepted StageResult: %w", err)
		}
		result.AcceptedResult = &decoded
	}
	if termination != nil {
		var decoded StageTermination
		if err := json.Unmarshal(termination, &decoded); err != nil {
			return StageExecution{}, fmt.Errorf("decode StageTermination: %w", err)
		}
		if err := decoded.Validate(); err != nil {
			return StageExecution{}, fmt.Errorf("validate StageTermination: %w", err)
		}
		result.Termination = &decoded
	}
	return result, nil
}
