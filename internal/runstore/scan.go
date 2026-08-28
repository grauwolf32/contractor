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
state, state_reason_code, state_reason_message,
scheduler_claim_id, scheduler_claimed_at, scheduler_claim_expires_at,
created_at, updated_at, started_at, finished_at`

type rowScanner interface {
	Scan(...any) error
}

func scanWorkflowRun(row rowScanner) (WorkflowRun, error) {
	var result WorkflowRun
	var snapshot []byte
	var parameters []byte
	var state string
	var claimID *string
	var claimedAt *time.Time
	var claimExpiresAt *time.Time
	if err := row.Scan(
		&result.RunID, &result.OwnerID, &result.WorkflowName, &result.WorkflowVersion,
		&result.WorkflowSchemaVersion, &snapshot, &parameters,
		&state, &result.StateReason.Code, &result.StateReason.Message,
		&claimID, &claimedAt, &claimExpiresAt,
		&result.CreatedAt, &result.UpdatedAt, &result.StartedAt, &result.FinishedAt,
	); err != nil {
		return WorkflowRun{}, err
	}
	result.State = WorkflowRunState(state)
	result.WorkflowSnapshot = append(json.RawMessage(nil), snapshot...)
	if err := json.Unmarshal(parameters, &result.Parameters); err != nil {
		return WorkflowRun{}, fmt.Errorf("decode WorkflowRun parameters: %w", err)
	}
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

func prefixedWorkflowRunColumns(alias string) string {
	parts := strings.Split(workflowRunColumns, ",")
	for index, part := range parts {
		parts[index] = alias + "." + strings.TrimSpace(part)
	}
	return strings.Join(parts, ", ")
}
