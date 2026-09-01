package runstore

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

const workflowRunColumns = `
run_id, owner_id, workflow_name, workflow_version,
workflow_schema_version, workflow_snapshot, parameters,
runtime_labels, runtime_config_snapshot, skill_snapshot,
state, state_reason_code, state_reason_message,
cancellation_schema_version, run_cancellation,
scheduler_claim_id, scheduler_claimed_at, scheduler_claim_expires_at,
created_at, updated_at, started_at, finished_at`

type rowScanner interface {
	Scan(...any) error
}

func scanWorkflowRun(row rowScanner) (WorkflowRun, error) {
	var result WorkflowRun
	var snapshot []byte
	var parameters []byte
	var runtimeConfig []byte
	var skillSnapshot []byte
	var state string
	var cancellation []byte
	var claimID *string
	var claimedAt *time.Time
	var claimExpiresAt *time.Time
	if err := row.Scan(
		&result.RunID, &result.OwnerID, &result.WorkflowName, &result.WorkflowVersion,
		&result.WorkflowSchemaVersion, &snapshot, &parameters,
		&result.RuntimeLabels, &runtimeConfig, &skillSnapshot,
		&state, &result.StateReason.Code, &result.StateReason.Message,
		&result.CancellationSchemaVersion, &cancellation,
		&claimID, &claimedAt, &claimExpiresAt,
		&result.CreatedAt, &result.UpdatedAt, &result.StartedAt, &result.FinishedAt,
	); err != nil {
		return WorkflowRun{}, err
	}
	result.State = WorkflowRunState(state)
	if cancellation != nil {
		var decoded WorkflowRunCancellation
		if err := json.Unmarshal(cancellation, &decoded); err != nil {
			return WorkflowRun{}, fmt.Errorf("decode WorkflowRun cancellation: %w", err)
		}
		if err := decoded.Validate(); err != nil {
			return WorkflowRun{}, fmt.Errorf("validate WorkflowRun cancellation: %w", err)
		}
		result.Cancellation = &decoded
	}
	result.WorkflowSnapshot = append(json.RawMessage(nil), snapshot...)
	if err := json.Unmarshal(parameters, &result.Parameters); err != nil {
		return WorkflowRun{}, fmt.Errorf("decode WorkflowRun parameters: %w", err)
	}
	if err := json.Unmarshal(runtimeConfig, &result.RuntimeConfig); err != nil {
		return WorkflowRun{}, fmt.Errorf("decode WorkflowRun RuntimeConfig snapshot: %w", err)
	}
	if err := result.RuntimeConfig.Validate(); err != nil {
		return WorkflowRun{}, fmt.Errorf("validate WorkflowRun RuntimeConfig snapshot: %w", err)
	}
	if err := json.Unmarshal(skillSnapshot, &result.SkillSnapshot); err != nil {
		return WorkflowRun{}, fmt.Errorf("decode WorkflowRun Skill snapshot: %w", err)
	}
	if err := validateRunSkillSnapshot(result.SkillSnapshot, false); err != nil {
		return WorkflowRun{}, fmt.Errorf("validate WorkflowRun Skill snapshot: %w", err)
	}
	if labels := result.RuntimeConfig.ExplicitLabels(); !equalRunLabels(labels, result.RuntimeLabels) {
		return WorkflowRun{}, fmt.Errorf("validate WorkflowRun RuntimeConfig labels: projection mismatch")
	}
	result.RuntimeLabels = append([]string{}, result.RuntimeLabels...)
	if claimID != nil {
		if claimedAt == nil || claimExpiresAt == nil {
			return WorkflowRun{}, fmt.Errorf("decode WorkflowRun claim: incomplete persisted claim")
		}
		result.SchedulerClaim = &SchedulerClaim{
			ClaimID: *claimID, ClaimedAt: *claimedAt, ExpiresAt: *claimExpiresAt,
		}
	}
	return result, nil
}

func equalRunLabels(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func prefixedWorkflowRunColumns(alias string) string {
	parts := strings.Split(workflowRunColumns, ",")
	for index, part := range parts {
		parts[index] = alias + "." + strings.TrimSpace(part)
	}
	return strings.Join(parts, ", ")
}
