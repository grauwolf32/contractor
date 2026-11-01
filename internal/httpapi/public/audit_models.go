package public

import (
	"fmt"
	"io"
	"net/http"
	"strconv"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
)

func auditProfileReadModel(source auditservice.ProfileProjection, detail bool) auditProfileResponse {
	profile := source.Profile
	result := auditProfileResponse{
		Ref: profile.Ref, Mode: profile.Mode,
		Standards: append([]auditprofile.AuditStandardRef{}, profile.Standards...),
		Inputs:    profile.Inputs, Inventory: profile.Inventory,
		Execution: profile.Execution, Interaction: profile.Interaction,
		ServerCompatible:        source.Compatibility.ServerCompatible,
		RequiresInputValidation: source.Compatibility.RequiresInputValidation,
		CompatibilityReasons:    append([]auditservice.CompatibilityReason{}, source.Compatibility.Reasons...),
	}
	if detail {
		if len(profile.Methods) > 0 {
			result.Methods = make(map[string]auditProfileCheckMethodResponse, len(profile.Methods))
			for name, mapping := range profile.Clone().Methods {
				result.Methods[name] = auditProfileCheckMethodResponse{
					PlanSchema: mapping.PlanSchema, Method: mapping.Method, WorkflowRole: mapping.WorkflowRole,
					RequiredCapabilities: mapping.RequiredCapabilities,
				}
			}
		}
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
		Labels:                  selection.Labels.Clone(),
		RequiredPlacementLabels: append([]string{}, selection.RequiredPlacementLabels...),
		Phase:                   auditdomain.AuditPhaseNotStarted,
		AuditID:                 source.AuditID, ProjectID: source.ProjectID,
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
	if source.Phase != "" {
		result.Phase = source.Phase
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
			Labels:                  baseline.Labels.Clone(),
			RequiredPlacementLabels: append([]string{}, baseline.RequiredPlacementLabels...),
			ActiveChecks:            baseline.ActiveChecks,
			Inputs:                  baseline.Inputs, Scope: baseline.Scope,
			RuntimeLabels: baseline.RuntimeLabels,
			RuntimeConfig: auditRuntimeSnapshotResponse{
				Default: baseline.RuntimeConfig.Default, Labels: baseline.RuntimeConfig.Labels,
			},
			Skills: skills, Standards: append([]auditstandards.PinnedPackage{}, baseline.Standards...),
			ProjectHTTPTarget: baseline.ProjectHTTPTarget,
		}
		if baseline.Inventory != nil {
			result.Baseline.Inventory = &auditBaselineInventoryResponse{
				SourceContentDigest:      baseline.Inventory.SourceContentDigest,
				CanonicalInventoryDigest: baseline.Inventory.CanonicalInventoryDigest,
				StandardSelection:        baseline.Inventory.StandardSelection,
				Gaps:                     append([]string{}, baseline.Inventory.Gaps...),
				Worklist:                 baseline.Inventory.Worklist,
			}
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
		FindingIDs: append([]string{}, source.FindingIDs...),
		ItemID:     source.ItemID, RoundID: source.RoundID, ItemKey: source.ItemKey,
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

func pointerString(value *string) string {
	if value == nil {
		return ""
	}
	return *value
}

func pointerItemState(value *auditstore.ItemState) string {
	if value == nil {
		return ""
	}
	return string(*value)
}

// maximumAuditTimeLimitBodyBytes bounds the optional JSON body of Audit start
// and resume (LIM-AUD-28), which holds only deadlineSeconds.
const maximumAuditTimeLimitBodyBytes int64 = 1024

// readAuditTimeLimit reads that optional body. Its callers have already
// checked the media type of an announced body (requireOptionalJSONMediaType).
func readAuditTimeLimit(w http.ResponseWriter, r *http.Request) (*int, error) {
	data, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maximumAuditTimeLimitBodyBytes))
	if exceedsBodyLimit(err) {
		return nil, fmt.Errorf("%w: Audit time limit body is too large", errRequestTooLarge)
	}
	if err != nil {
		return nil, fmt.Errorf("%w: invalid Audit time limit body", errInvalidRequest)
	}
	if len(data) == 0 {
		return nil, nil
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
	if request.DeadlineSeconds != nil && (*request.DeadlineSeconds < 0 || *request.DeadlineSeconds > auditprofile.MaxAuditDeadlineSeconds) {
		return nil, fmt.Errorf("%w: Audit time limit must be 0 (unlimited) or at most %d seconds", errInvalidRequest, auditprofile.MaxAuditDeadlineSeconds)
	}
	return request.DeadlineSeconds, nil
}
