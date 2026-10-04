package public

import (
	"time"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/auth"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type operationsCursorResponse struct {
	Generation string `json:"generation"`
	Revision   string `json:"revision"`
}

type schedulerSettingsResponse struct {
	MaxConcurrentRuns int       `json:"maxConcurrentRuns"`
	Revision          string    `json:"revision"`
	UpdatedAt         time.Time `json:"updatedAt"`
}

type updateSchedulerSettingsRequest struct {
	MaxConcurrentRuns int `json:"maxConcurrentRuns"`
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
	Ref          config.WorkflowRef              `json:"ref"`
	Presentation *config.WorkflowPresentation    `json:"presentation,omitempty"`
	EntryStage   string                          `json:"entryStage"`
	Parameters   map[string]config.ParameterSlot `json:"parameters"`
	Inputs       map[string]config.ArtifactSlot  `json:"inputs"`
	Outputs      map[string]config.ArtifactSlot  `json:"outputs"`
}

type workflowResourceResponse struct {
	workflowSummaryResponse
	Stages map[string]workflowStageResponse `json:"stages"`
}

type instructionsRefResponse struct {
	Ref    string `json:"ref"`
	Digest string `json:"digest"`
}

type workflowAgentBindingResponse struct {
	Template  contracts.AgentTemplateRef `json:"template"`
	Namespace string                     `json:"namespace"`
	Skills    []contracts.ArtifactRef    `json:"skills"`
}

type workflowStageResponse struct {
	Objective        string                                  `json:"objective"`
	Instructions     instructionsRefResponse                 `json:"instructions"`
	Planner          config.PlannerRef                       `json:"planner"`
	Session          contracts.WorkerSessionMode             `json:"session"`
	Agents           map[string]workflowAgentBindingResponse `json:"agents"`
	ExecutionConfig  resolvedStageExecutionConfigResponse    `json:"executionConfig"`
	ContextArtifacts map[string]config.ContextArtifact       `json:"contextArtifacts"`
	ResultArtifacts  map[string]config.ArtifactSlot          `json:"resultArtifacts"`
	WorkflowOutputs  map[string]string                       `json:"workflowOutputs"`
	On               workflowTransitionsResponse             `json:"on"`
}

type resolvedStageExecutionConfigResponse struct {
	Planner *consumerExecutionConfigRefsResponse           `json:"planner,omitempty"`
	Agents  map[string]consumerExecutionConfigRefsResponse `json:"agents"`
}

type workflowTransitionsResponse struct {
	Succeeded   workflowTransitionResponse `json:"succeeded"`
	Failed      workflowTransitionResponse `json:"failed"`
	Interrupted workflowTransitionResponse `json:"interrupted"`
}

type workflowTransitionResponse struct {
	Kind            config.TransitionKind             `json:"kind"`
	NextStage       string                            `json:"nextStage,omitempty"`
	MaxAttempts     int                               `json:"maxAttempts,omitempty"`
	ExecutionConfig *workflowEscalationConfigResponse `json:"executionConfig,omitempty"`
	Then            *workflowTransitionResponse       `json:"then,omitempty"`
}

type workflowEscalationConfigResponse struct {
	Ref       *config.ExecutionConfigRef           `json:"ref,omitempty"`
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
	Kind   string                         `json:"kind"`
	Reason runstore.RunNotDeletableReason `json:"reason"`
}
