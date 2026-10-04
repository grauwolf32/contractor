package public

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

func auditProfileReadModel(source auditservice.ProfileProjection, detail bool) auditProfileResponse {
	profile := source.Profile
	result := auditProfileResponse{
		Ref: profile.Ref, Mode: profile.Mode,
		Standards: append([]config.AuditStandardRef{}, profile.Standards...),
		Inputs:    profile.Inputs, Inventory: profile.Inventory,
		Execution: profile.Execution, Interaction: profile.Interaction,
		ServerCompatible:        source.Compatibility.ServerCompatible,
		RequiresInputValidation: source.Compatibility.RequiresInputValidation,
		CompatibilityReasons:    append([]auditservice.CompatibilityReason{}, source.Compatibility.Reasons...),
	}
	if detail {
		result.Workflows = make(map[string]auditProfileWorkflowResponse, len(profile.Workflows))
		for role, binding := range profile.Workflows {
			result.Workflows[role] = auditProfileWorkflowResponse{
				Kind: binding.Kind, MaxRunAttempts: binding.MaxRunAttempts, Workflow: binding.Workflow.Ref, Inputs: binding.Inputs,
				Parameters: binding.Parameters, Outputs: binding.Outputs,
			}
		}
	}
	return result
}

func auditReadModel(source auditstore.Audit) (auditResponse, error) {
	selection, err := auditservice.DecodeDraftSelection(source.InputSelection)
	if err != nil {
		return auditResponse{}, err
	}
	result := auditResponse{
		Phase:   auditdomain.AuditPhaseNotStarted,
		AuditID: source.AuditID, ProjectID: source.ProjectID,
		Profile: auditProfileIdentityResponse{
			Name: source.Profile.Name, Version: source.Profile.Version, Digest: source.Profile.Digest,
		},
		Inputs: selection.Inputs, Scope: selection.Scope,
		RuntimeLabels: append([]string{}, selection.RuntimeLabels...),
		State:         source.State, Revision: source.Revision, CurrentRoundID: source.CurrentRoundID,
		DispatchState: source.Dispatch, HoldState: source.Hold, DeadlineAt: source.DeadlineAt, PausedAt: source.PausedAt,
		Limits: source.Limits, ReservedRunCount: source.ReservedRunCount,
		SubmittedRunCount: source.SubmittedRunCount, OutstandingRunCount: source.OutstandingRunCount,
		RetainedEvidenceBytes: source.RetainedEvidenceBytes, EventSequence: source.EventSequence,
		CreatedAt: source.CreatedAt, UpdatedAt: source.UpdatedAt,
		StartedAt: source.StartedAt, FinishedAt: source.FinishedAt,
		DeletionRequestedAt: source.DeletionRequestedAt,
	}
	if source.CurrentRoundID != nil {
		result.Phase = auditdomain.AuditPhaseRounds
	}
	if source.StopReason != nil {
		result.StopReason = &auditStopReasonResponse{
			Code: source.StopReason.Code, Message: source.StopReason.Message,
		}
	}
	if len(source.BaselineSnapshot) != 0 {
		baseline, decodeErr := auditservice.DecodeBaseline(source.BaselineSnapshot)
		if decodeErr != nil {
			return auditResponse{}, decodeErr
		}
		skills := make([]auditSkillResponse, 0, len(baseline.Skills))
		for _, skill := range baseline.Skills {
			if skill.Source == nil {
				return auditResponse{}, fmt.Errorf("stored Audit baseline contains an unresolved Skill")
			}
			skills = append(skills, auditSkillResponse{
				Name: skill.Name, Source: *skill.Source,
				SourceDigest: skill.SourceDigest, SourceSize: skill.SourceSize,
			})
		}
		result.Baseline = &auditBaselineResponse{
			Inputs: baseline.Inputs, Scope: baseline.Scope,
			RuntimeLabels: baseline.RuntimeLabels,
			RuntimeConfig: auditRuntimeSnapshotResponse{
				Default: baseline.RuntimeConfig.Default, Labels: baseline.RuntimeConfig.Labels,
			},
			Skills: skills, Standards: append([]auditstandards.PinnedPackage{}, baseline.Standards...),
			ProjectHTTPTarget: baseline.ProjectHTTPTarget,
			Inventory: &auditBaselineInventoryResponse{
				SourceContentDigest:      baseline.Inventory.SourceContentDigest,
				CanonicalInventoryDigest: baseline.Inventory.CanonicalInventoryDigest,
				StandardSelection:        baseline.Inventory.StandardSelection,
				Gaps:                     append([]string{}, baseline.Inventory.Gaps...),
				Worklist:                 baseline.Inventory.Worklist,
			},
		}
	}
	return result, nil
}

func auditRoundReadModel(source auditstore.Round) auditRoundResponse {
	return auditRoundResponse{
		RoundID: source.RoundID, Ordinal: source.Ordinal, Manifest: source.Manifest,
		State: source.State, ExpectedItemCount: source.ExpectedItemCount, Revision: source.Revision,
		CreatedAt: source.CreatedAt, UpdatedAt: source.UpdatedAt,
	}
}

func auditItemReadModel(source auditstore.Item) auditItemResponse {
	return auditItemResponse{
		ItemID: source.ItemID, RoundID: source.RoundID, ItemKey: source.ItemKey,
		Ordinal: source.Ordinal, Kind: source.Kind, SubjectKey: source.SubjectKey,
		Task: source.Task, Origin: source.Origin, WorkflowRole: source.WorkflowRole, State: source.State,
		ApprovalKind: source.ApprovalKind, ApprovalDigest: source.ApprovalDigest,
		FinalDisposition: source.FinalDisposition, AcceptedResult: source.AcceptedResult,
		LastExecutionItemID: source.LastExecutionItemID,
		Attempts:            []auditstore.ItemAttempt{},
		CreatedAt:           source.CreatedAt, UpdatedAt: source.UpdatedAt,
	}
}

func writeAuditJSON(w http.ResponseWriter, status int, response auditResponse) {
	w.Header().Set("ETag", strconv.Quote(strconv.FormatUint(response.Revision, 10)))
	writeJSON(w, status, response)
}

func createAuditRequestDigest(projectID string, request createAuditRequest) (string, []string, error) {
	labels, err := runtimeconfig.NormalizeRunLabels(request.RuntimeLabels)
	if err != nil {
		return "", nil, err
	}
	canonical := struct {
		ProjectID     string                           `json:"projectId"`
		Profile       auditProfileSelectorRequest      `json:"profile"`
		Inputs        map[string]contracts.ArtifactRef `json:"inputs"`
		RuntimeLabels []string                         `json:"runtimeLabels"`
		Scope         auditservice.Scope               `json:"scope"`
	}{projectID, request.Profile, request.Inputs, labels, request.Scope}
	digest, err := contentdigest.JSON(canonical)
	if err != nil {
		return "", nil, err
	}
	return digest, labels, nil
}

// auditRequestDigest binds a start (empty action) or lifecycle mutation.
// Start digests predate the action field, so it is omitted when empty to keep
// stored start digests replayable.
func auditRequestDigest(action, auditID string, revision uint64, seconds ...*int) string {
	encoded, _ := json.Marshal(struct {
		Action          string `json:"action,omitempty"`
		AuditID         string `json:"auditId"`
		Revision        uint64 `json:"revision"`
		DeadlineSeconds *int   `json:"deadlineSeconds,omitempty"`
	}{action, auditID, revision, firstTimeLimit(seconds)})
	return contentdigest.Bytes(encoded)
}

func requireEmptyBody(w http.ResponseWriter, r *http.Request) error {
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1))
	if err != nil || len(data) != 0 {
		return fmt.Errorf("%w: request body must be empty", errInvalidRequest)
	}
	return nil
}

func pointerString(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func pointerAuditState(value *auditstore.AuditState) string {
	if value == nil {
		return ""
	}
	return string(*value)
}

func pointerItemState(value *auditstore.ItemState) string {
	if value == nil {
		return ""
	}
	return string(*value)
}

func firstTimeLimit(values []*int) *int {
	if len(values) == 0 {
		return nil
	}
	return values[0]
}

func readAuditTimeLimit(w http.ResponseWriter, r *http.Request) (*int, error) {
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 1024))
	if exceedsBodyLimit(err) {
		return nil, fmt.Errorf("%w: Audit time limit body is too large", errRequestTooLarge)
	}
	if err != nil {
		return nil, fmt.Errorf("%w: invalid Audit time limit body", errInvalidRequest)
	}
	if len(data) == 0 {
		return nil, nil
	}
	media, err := requestMediaType(r)
	if err != nil || media != "application/json" {
		return nil, errInvalidRequest
	}
	var request struct {
		DeadlineSeconds *int `json:"deadlineSeconds"`
	}
	fields, err := decodeStrictWithPresence(data, &request)
	if err != nil {
		return nil, fmt.Errorf("%w: invalid Audit time limit: %v", errInvalidRequest, err)
	}
	if fields == nil || fields.null("deadlineSeconds") {
		return nil, errInvalidRequest
	}
	if request.DeadlineSeconds != nil && (*request.DeadlineSeconds < 0 || *request.DeadlineSeconds > config.MaxAuditDeadlineSeconds) {
		return nil, fmt.Errorf("%w: Audit time limit must be 0 (unlimited) or at most %d seconds", errInvalidRequest, config.MaxAuditDeadlineSeconds)
	}
	return request.DeadlineSeconds, nil
}
