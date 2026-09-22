package public

// Workflow catalog responses: a Workflow, its Stages, agent bindings and
// the transitions between them.

import (
	"time"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

type workflowPageResponse struct {
	Items []workflowSummaryResponse `json:"items"`
	Page  pageInfoResponse          `json:"page"`
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
