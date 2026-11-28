package runstore

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

const workflowRunColumns = `
run_id, owner_id, project_id, workflow_name, workflow_version,
workflow_schema_version, workflow_snapshot, parameters,
runtime_labels, runtime_config_snapshot, project_http_target_snapshot, skill_snapshot,
publication_mode, audit_execution_id, audit_submission_key, audit_completion,
state, state_reason_code, state_reason_message,
cancellation_schema_version, run_cancellation,
scheduler_claim_id, scheduler_claimed_at, scheduler_claim_expires_at, scheduler_lane_class,
created_at, updated_at, started_at, finished_at, metadata_labels, required_placement_labels, engagement_credential_ids`

type rowScanner interface {
	Scan(...any) error
}

func scanWorkflowRun(row rowScanner) (WorkflowRun, error) {
	var result WorkflowRun
	var snapshot []byte
	var parameters []byte
	var runtimeConfig []byte
	var projectHTTPTarget []byte
	var skillSnapshot []byte
	var auditCompletion []byte
	var state string
	var cancellation []byte
	var claimID *string
	var claimedAt *time.Time
	var claimExpiresAt *time.Time
	var metadataLabels []byte
	if err := row.Scan(
		&result.RunID, &result.OwnerID, &result.ProjectID, &result.WorkflowName, &result.WorkflowVersion,
		&result.WorkflowSchemaVersion, &snapshot, &parameters,
		&result.RuntimeLabels, &runtimeConfig, &projectHTTPTarget, &skillSnapshot,
		&result.PublicationMode, &result.AuditExecutionID, &result.AuditSubmissionKey, &auditCompletion,
		&state, &result.StateReason.Code, &result.StateReason.Message,
		&result.CancellationSchemaVersion, &cancellation,
		&claimID, &claimedAt, &claimExpiresAt, &result.SchedulerLaneClass,
		&result.CreatedAt, &result.UpdatedAt, &result.StartedAt, &result.FinishedAt, &metadataLabels,
		&result.RequiredPlacementLabels, &result.EngagementCredentialIDs,
	); err != nil {
		return WorkflowRun{}, err
	}
	if result.EngagementCredentialIDs == nil || validateEngagementCredentialIDs(result.EngagementCredentialIDs) != nil {
		return WorkflowRun{}, fmt.Errorf("invalid stored engagement credential dependencies")
	}
	if auditCompletion != nil {
		if err := strictjson.Decode(auditCompletion, &result.AuditCompletion); err != nil {
			return WorkflowRun{}, fmt.Errorf("decode Audit completion: %w", err)
		}
		if result.AuditCompletion == nil || result.AuditCompletion.Validate() != nil || (result.AuditCompletion.Finding == nil) != (result.PublicationMode == PublicationAuditManaged) || (result.AuditCompletion.Finding != nil && result.ProjectID == nil) {
			return WorkflowRun{}, fmt.Errorf("invalid stored Audit completion authority")
		}
	}
	result.State = WorkflowRunState(state)
	if result.RequiredPlacementLabels == nil || contracts.ValidatePlacementLabels(result.RequiredPlacementLabels) != nil {
		return WorkflowRun{}, fmt.Errorf("invalid persisted Run placement requirements")
	}
	if !result.SchedulerLaneClass.Valid() {
		return WorkflowRun{}, fmt.Errorf("validate WorkflowRun Scheduler lane class: invalid value")
	}
	if !result.PublicationMode.Valid() {
		return WorkflowRun{}, fmt.Errorf("validate WorkflowRun publication mode: invalid value")
	}
	if result.PublicationMode == PublicationOrdinary &&
		(result.AuditExecutionID != nil || result.AuditSubmissionKey != nil) {
		return WorkflowRun{}, fmt.Errorf("validate WorkflowRun Audit association: ordinary Run carries Audit authority")
	}
	if result.PublicationMode == PublicationAuditManaged &&
		(result.AuditExecutionID == nil || result.AuditSubmissionKey == nil) {
		return WorkflowRun{}, fmt.Errorf("validate WorkflowRun Audit association: managed Run is incomplete")
	}
	if cancellation != nil {
		var decoded WorkflowRunCancellation
		if err := strictjson.Decode(cancellation, &decoded); err != nil {
			return WorkflowRun{}, fmt.Errorf("decode WorkflowRun cancellation: %w", err)
		}
		if err := decoded.Validate(); err != nil {
			return WorkflowRun{}, fmt.Errorf("validate WorkflowRun cancellation: %w", err)
		}
		result.Cancellation = &decoded
	}
	result.WorkflowSnapshot = append(json.RawMessage(nil), snapshot...)
	if err := strictjson.Decode(parameters, &result.Parameters); err != nil {
		return WorkflowRun{}, fmt.Errorf("decode WorkflowRun parameters: %w", err)
	}
	if err := strictjson.Decode(runtimeConfig, &result.RuntimeConfig); err != nil {
		return WorkflowRun{}, fmt.Errorf("decode WorkflowRun RuntimeConfig snapshot: %w", err)
	}
	if err := result.RuntimeConfig.Validate(); err != nil {
		return WorkflowRun{}, fmt.Errorf("validate WorkflowRun RuntimeConfig snapshot: %w", err)
	}
	if projectHTTPTarget != nil {
		var decoded runtimesettings.HTTPOriginTargetRef
		if err := strictjson.Decode(projectHTTPTarget, &decoded); err != nil {
			return WorkflowRun{}, fmt.Errorf("decode WorkflowRun Project HTTP target snapshot: %w", err)
		}
		if err := decoded.Validate(); err != nil {
			return WorkflowRun{}, fmt.Errorf("validate WorkflowRun Project HTTP target snapshot: %w", err)
		}
		result.ProjectHTTPTarget = &decoded
	}
	if err := strictjson.Decode(skillSnapshot, &result.SkillSnapshot); err != nil {
		return WorkflowRun{}, fmt.Errorf("decode WorkflowRun Skill snapshot: %w", err)
	}
	if err := validateRunSkillSnapshot(result.SkillSnapshot, false); err != nil {
		return WorkflowRun{}, fmt.Errorf("validate WorkflowRun Skill snapshot: %w", err)
	}
	if labels := result.RuntimeConfig.ExplicitLabels(); !slices.Equal(labels, result.RuntimeLabels) {
		return WorkflowRun{}, fmt.Errorf("validate WorkflowRun RuntimeConfig labels: projection mismatch")
	}
	result.RuntimeLabels = append([]string{}, result.RuntimeLabels...)
	result.MetadataLabels = make(RunMetadataLabels)
	if err := strictjson.Decode(metadataLabels, &result.MetadataLabels); err != nil {
		return WorkflowRun{}, fmt.Errorf("decode persisted Run metadata labels: %w", err)
	}
	if err := result.MetadataLabels.Validate(); err != nil {
		return WorkflowRun{}, fmt.Errorf("validate persisted Run metadata labels: %w", err)
	}
	derived, err := result.MetadataLabels.PlacementRequirements()
	if err != nil || !slices.Equal(derived, result.RequiredPlacementLabels) {
		return WorkflowRun{}, fmt.Errorf("persisted Run placement requirements differ from labels")
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
