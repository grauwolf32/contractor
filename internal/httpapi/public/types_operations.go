package public

import (
	"errors"
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/rundeletion"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type operationsCursorResponse struct {
	Generation string `json:"generation"`
	Revision   string `json:"revision"`
}

type schedulerSettingsResponse struct {
	MaxConcurrentRuns     int       `json:"maxConcurrentRuns"`
	MaxConcurrentToolRuns int       `json:"maxConcurrentToolRuns"`
	Revision              string    `json:"revision"`
	UpdatedAt             time.Time `json:"updatedAt"`
}

// updateSchedulerSettingsRequest requires maxConcurrentRuns; an absent
// maxConcurrentToolRuns keeps the stored value, and neither may be null.
type updateSchedulerSettingsRequest struct {
	MaxConcurrentRuns     int  `json:"maxConcurrentRuns"`
	MaxConcurrentToolRuns *int `json:"maxConcurrentToolRuns,omitempty"`
}

func (r *updateSchedulerSettingsRequest) UnmarshalJSON(data []byte) error {
	type wire updateSchedulerSettingsRequest
	var decoded wire
	fields, err := decodeStrictWithPresence(data, &decoded)
	if err != nil {
		return err
	}
	if fields.missingOrNull("maxConcurrentRuns") {
		return errors.New("maxConcurrentRuns is required")
	}
	if fields.null("maxConcurrentToolRuns") {
		return errors.New("maxConcurrentToolRuns cannot be null")
	}
	*r = updateSchedulerSettingsRequest(decoded)
	return nil
}

type loginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

type authSessionResponse struct {
	Principal         auth.Principal `json:"principal"`
	CSRFToken         string         `json:"csrfToken"`
	IdleExpiresAt     time.Time      `json:"idleExpiresAt"`
	AbsoluteExpiresAt time.Time      `json:"absoluteExpiresAt"`
}

type operationsSnapshotResponse struct {
	Cursor        operationsCursorResponse               `json:"cursor"`
	RuntimeAgents []controlplane.RuntimeAgentObservation `json:"runtimeAgents"`
	Allocations   []controlplane.AllocationObservation   `json:"allocations"`
}

type runtimeAgentPageResponse struct {
	Cursor operationsCursorResponse               `json:"cursor"`
	Items  []controlplane.RuntimeAgentObservation `json:"items"`
	Page   pageInfoResponse                       `json:"page"`
}

type allocationPageResponse struct {
	Cursor operationsCursorResponse             `json:"cursor"`
	Items  []controlplane.AllocationObservation `json:"items"`
	Page   pageInfoResponse                     `json:"page"`
}

type workflowSummaryResponse struct {
	MaxSubtasks  contracts.PlannerSubtaskLimit     `json:"maxSubtasks,omitempty"`
	Ref          resolved.WorkflowRef              `json:"ref"`
	Presentation *resolved.WorkflowPresentation    `json:"presentation,omitempty"`
	EntryStage   string                            `json:"entryStage"`
	Parameters   map[string]resolved.ParameterSlot `json:"parameters"`
	Inputs       map[string]resolved.ArtifactSlot  `json:"inputs"`
	Outputs      map[string]resolved.ArtifactSlot  `json:"outputs"`
}

type workflowResourceResponse struct {
	workflowSummaryResponse
	Stages map[string]workflowStageResponse `json:"stages"`
}

type instructionsRefResponse struct {
	Ref    string `json:"ref"`
	Digest string `json:"digest"`
}

// plannerContextManagementResponse renders a Planner Stage's context
// management explicitly: strategy none, or the compact@1 thresholds.
type plannerContextManagementResponse struct {
	Strategy           string                   `json:"strategy"`
	Instructions       *instructionsRefResponse `json:"instructions,omitempty"`
	ContextWindowRatio *float64                 `json:"contextWindowRatio,omitempty"`
	KeepRecentGroups   int                      `json:"keepRecentGroups,omitempty"`
	MaxCompactions     int                      `json:"maxCompactions,omitempty"`
}

func plannerContextManagementReadModel(stage resolved.ResolvedStage) *plannerContextManagementResponse {
	if stage.Operation != nil {
		return nil
	}
	compaction := stage.PlannerContextManagement
	if compaction == nil {
		return &plannerContextManagementResponse{Strategy: resolved.PlannerContextNone}
	}
	ratio := compaction.ContextWindowRatio
	return &plannerContextManagementResponse{
		Strategy: compaction.Strategy,
		Instructions: &instructionsRefResponse{
			Ref: compaction.Instructions.Ref, Digest: compaction.Instructions.Digest,
		},
		ContextWindowRatio: &ratio,
		KeepRecentGroups:   compaction.KeepRecentGroups,
		MaxCompactions:     compaction.MaxCompactions,
	}
}

type workflowAgentBindingResponse struct {
	Template          contracts.AgentTemplateRef        `json:"template"`
	Namespace         string                            `json:"namespace"`
	Skills            []contracts.ArtifactRef           `json:"skills"`
	ContextManagement workflowContextManagementResponse `json:"contextManagement"`
}

// workflowContextManagementResponse is the reference-only public projection
// of one binding's explicit context selection (WFL-R91): compaction members
// are present exactly for compact@1, and instruction text is not repeated.
type workflowContextManagementResponse struct {
	Strategy           contracts.WorkerContextStrategy `json:"strategy"`
	Instructions       *instructionsRefResponse        `json:"instructions,omitempty"`
	IncludeReasoning   *bool                           `json:"includeReasoning,omitempty"`
	ContextWindowRatio *float64                        `json:"contextWindowRatio,omitempty"`
	KeepRecentGroups   *int                            `json:"keepRecentGroups,omitempty"`
	MaxCompactions     *int                            `json:"maxCompactions,omitempty"`
}

func newWorkflowContextManagementResponse(value contracts.WorkerContextManagement) workflowContextManagementResponse {
	result := workflowContextManagementResponse{Strategy: value.Strategy}
	if compaction := value.WorkerCompaction; compaction != nil {
		includeReasoning := compaction.IncludeReasoning
		ratio := compaction.ContextWindowRatio
		keep := compaction.KeepRecentGroups
		maximum := compaction.MaxCompactions
		result.Instructions = &instructionsRefResponse{Ref: compaction.Instructions.Ref, Digest: compaction.Instructions.Digest}
		result.IncludeReasoning = &includeReasoning
		result.ContextWindowRatio = &ratio
		result.KeepRecentGroups = &keep
		result.MaxCompactions = &maximum
	}
	return result
}

type workflowStageResponse struct {
	Objective    string                          `json:"objective"`
	Instructions instructionsRefResponse         `json:"instructions,omitzero"`
	Operation    *resolved.ServerOperationConfig `json:"operation,omitempty"`
	Planner      resolved.PlannerRef             `json:"planner,omitzero"`
	// PlannerContextManagement is present for every Planner Stage.
	PlannerContextManagement *plannerContextManagementResponse       `json:"plannerContextManagement,omitempty"`
	Session                  contracts.WorkerSessionMode             `json:"session"`
	Agents                   map[string]workflowAgentBindingResponse `json:"agents"`
	ExecutionConfig          resolvedStageExecutionConfigResponse    `json:"executionConfig"`
	ContextArtifacts         map[string]resolved.ContextArtifact     `json:"contextArtifacts"`
	ResultArtifacts          map[string]resolved.ArtifactSlot        `json:"resultArtifacts"`
	WorkflowOutputs          map[string]string                       `json:"workflowOutputs"`
	On                       workflowTransitionsResponse             `json:"on"`
}

type resolvedStageExecutionConfigResponse struct {
	Planner *consumerExecutionConfigRefsResponse           `json:"planner,omitempty"`
	Agents  map[string]consumerExecutionConfigRefsResponse `json:"agents"`
}

type workflowTransitionsResponse struct {
	Succeeded   workflowTransitionResponse  `json:"succeeded"`
	Failed      workflowTransitionResponse  `json:"failed"`
	Interrupted workflowTransitionResponse  `json:"interrupted"`
	Summarized  *workflowTransitionResponse `json:"summarized,omitempty"`
}

type workflowTransitionResponse struct {
	PlannerContextManagement *plannerContextManagementResponse `json:"plannerContextManagement,omitempty"`
	Kind                     resolved.TransitionKind           `json:"kind"`
	NextStage                string                            `json:"nextStage,omitempty"`
	MaxAttempts              int                               `json:"maxAttempts,omitempty"`
	Planner                  *resolved.PlannerRef              `json:"planner,omitempty"`
	DeadlineSeconds          int                               `json:"deadlineSeconds,omitempty"`
	ExecutionConfig          *workflowEscalationConfigResponse `json:"executionConfig,omitempty"`
	Then                     *workflowTransitionResponse       `json:"then,omitempty"`
}

type workflowEscalationConfigResponse struct {
	Effective resolvedStageExecutionConfigResponse `json:"effective"`
}

type stageTransitionResponse struct {
	SourceExecutionID   string                         `json:"sourceExecutionId"`
	Action              runstore.StageTransitionAction `json:"action"`
	TargetStage         *string                        `json:"targetStage,omitempty"`
	TargetExecutionID   *string                        `json:"targetExecutionId,omitempty"`
	EscalationOrdinal   *int                           `json:"escalationOrdinal,omitempty"`
	EscalationExhausted bool                           `json:"escalationExhausted"`
	DecidedAt           time.Time                      `json:"decidedAt"`
}

type errorResponse struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
	RequestID string `json:"requestId"`
	Details   any    `json:"details,omitempty"`
}

type runtimeCredentialInUseDetailsResponse struct {
	Kind          string   `json:"kind"`
	BindingLabels []string `json:"bindingLabels"`
	ProjectIDs    []string `json:"projectIds"`
	RunIDs        []string `json:"runIds"`
	AuditIDs      []string `json:"auditIds,omitempty"`
	AllocationIDs []string `json:"allocationIds"`
}

type runtimeLabelInUseDetailsResponse struct {
	Kind            string   `json:"kind"`
	RuntimeAgentIDs []string `json:"runtimeAgentIds"`
}

type credentialInUseDetailsResponse struct {
	Kind          string   `json:"kind"`
	RunIDs        []string `json:"runIds"`
	AuditIDs      []string `json:"auditIds,omitempty"`
	BindingLabels []string `json:"bindingLabels,omitempty"`
}

type auditUnsupportedDetailsResponse struct {
	Kind    string                             `json:"kind"`
	Reasons []auditservice.CompatibilityReason `json:"reasons"`
}

type runNotDeletableDetailsResponse struct {
	Kind   string             `json:"kind"`
	Reason rundeletion.Reason `json:"reason"`
}
