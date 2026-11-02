package resolved

import (
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
)

// WorkflowRef identifies one exact Workflow configuration document.
type WorkflowRef struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// WorkflowPresentation is optional, authored Catalog metadata. It is part of
// the immutable resolved Workflow snapshot but has no execution semantics.
type WorkflowPresentation struct {
	DisplayName string `json:"displayName"`
	Description string `json:"description"`
}

// PlannerRef identifies one registered PlannerFactory implementation.
type PlannerRef struct {
	PlannerID string `json:"plannerId"`
	Version   string `json:"version"`
}

// Selector returns the exact <id>@<version> selector of r, the form the
// descriptor registry keys and the Planner selector constants use.
func (r PlannerRef) Selector() string {
	return r.PlannerID + "@" + r.Version
}

// ParameterSlot is the v1alpha1 string parameter contract.
type ParameterSlot struct {
	Required bool `json:"required"`
}

// ArtifactBinding identifies one versionless RunScope binding selected by
// trusted Workflow configuration.
type ArtifactBinding struct {
	Namespace string `json:"namespace"`
	Name      string `json:"name"`
}

// ArtifactSlot declares one versioned artifact input, output, or Stage result.
// From is present only on Stage result slots and lets Runtime map trusted tool
// observations to a result name without asking the Worker model to serialize
// Contractor transport data.
type ArtifactSlot struct {
	SourceScope string           `json:"sourceScope,omitempty"`
	Required    bool             `json:"required"`
	MediaTypes  []string         `json:"mediaTypes"`
	From        *ArtifactBinding `json:"from,omitempty"`
	Primary     bool             `json:"primary,omitempty"`
}

// ContextArtifact identifies a logical RunScope binding to pin for a Stage.
type ContextArtifact struct {
	Namespace string `json:"namespace"`
	Name      string `json:"name"`
	Required  bool   `json:"required"`
}

type StageContext struct {
	Artifacts map[string]ContextArtifact `json:"artifacts"`
	Workspace *WorkspaceContext          `json:"workspace,omitempty"`
}

type WorkspaceSource struct {
	Artifact string `json:"artifact"`
	Target   string `json:"target"`
}

type WorkspaceStateInput struct {
	Artifact string `json:"artifact"`
}

type WorkspaceExport struct {
	State string `json:"state"`
	Diff  string `json:"diff"`
}

type WorkspaceContext struct {
	Mode    contracts.WorkspaceMode `json:"mode"`
	Sources []WorkspaceSource       `json:"sources"`
	State   *WorkspaceStateInput    `json:"state,omitempty"`
	Export  *WorkspaceExport        `json:"export,omitempty"`
}

type StageResultContract struct {
	Artifacts map[string]ArtifactSlot `json:"artifacts"`
}

// ResolvedAgentBinding contains both the immutable template selected by a
// Workflow and the logical namespace assigned to that Stage binding.
type ResolvedAgentBinding struct {
	Template  contracts.ResolvedAgentTemplate `json:"template"`
	Namespace string                          `json:"namespace"`
	// ContextManagement is the explicit Workflow selection (WFL-R91); a
	// snapshot reader never infers it.
	ContextManagement contracts.WorkerContextManagement `json:"contextManagement"`
}

// ExecutionConfigOrigins records which authoring layer selected each resolved
// field. Values are safe provenance strings, never secret material.
type ExecutionConfigOrigins struct {
	ModelPolicy           string `json:"modelPolicy,omitempty"`
	SummarizerModelPolicy string `json:"summarizerModelPolicy,omitempty"`
	CompactorModelPolicy  string `json:"compactorModelPolicy,omitempty"`
	LLMGateway            string `json:"llmGateway,omitempty"`
	Credential            string `json:"credential,omitempty"`
}

// ResolvedConsumerExecutionConfig is one complete immutable model-access
// selection. Credential is a non-secret identity; token bytes never enter a
// Workflow or Run snapshot.
type ResolvedConsumerExecutionConfig struct {
	SummarizerModelPolicy *contracts.ResolvedModelPolicy `json:"summarizerModelPolicy,omitempty"`
	// CompactorModelPolicy is the tool-free compactor of a compact@1 Planner
	// (PLN-R125); a Worker selection never has one.
	CompactorModelPolicy *contracts.ResolvedModelPolicy       `json:"compactorModelPolicy,omitempty"`
	ModelPolicy          contracts.ResolvedModelPolicy        `json:"modelPolicy,omitzero"`
	LLMGateway           *llmgateway.ResolvedLLMGatewayConfig `json:"llmGateway,omitempty"`
	Credential           *llmgateway.LLMCredentialRef         `json:"credential,omitempty"`
	Origins              ExecutionConfigOrigins               `json:"origins"`
}

type ResolvedStageExecutionConfig struct {
	Planner *ResolvedConsumerExecutionConfig           `json:"planner,omitempty"`
	Agents  map[string]ResolvedConsumerExecutionConfig `json:"agents"`
}

// ResolvedCredentialOverride preserves the escalation patch's selected/clear
// distinction. A nil *ResolvedCredentialOverride means inherit the lower
// layer; Clear and Ref are mutually exclusive.
type ResolvedCredentialOverride struct {
	Clear bool                         `json:"clear"`
	Ref   *llmgateway.LLMCredentialRef `json:"ref,omitempty"`
}

// ResolvedExecutionSelectionOverride is a partial, dependency-resolved
// selection. Nil fields inherit the Run's base Stage configuration.
type ResolvedExecutionSelectionOverride struct {
	SummarizerModelPolicy *contracts.ResolvedModelPolicy       `json:"summarizerModelPolicy,omitempty"`
	CompactorModelPolicy  *contracts.ResolvedModelPolicy       `json:"compactorModelPolicy,omitempty"`
	ModelPolicy           *contracts.ResolvedModelPolicy       `json:"modelPolicy,omitempty"`
	LLMGateway            *llmgateway.ResolvedLLMGatewayConfig `json:"llmGateway,omitempty"`
	Credential            *ResolvedCredentialOverride          `json:"credential,omitempty"`
}

type ResolvedStageExecutionConfigOverride struct {
	Planner *ResolvedExecutionSelectionOverride           `json:"planner,omitempty"`
	Agents  map[string]ResolvedExecutionSelectionOverride `json:"agents,omitempty"`
}

// ResolvedEscalationExecutionConfig pins the inline patch and its fully
// effective result for one consuming Stage.
type ResolvedEscalationExecutionConfig struct {
	Override  ResolvedStageExecutionConfigOverride `json:"override"`
	Effective ResolvedStageExecutionConfig         `json:"effective"`
}

type TransitionKind string

const (
	TransitionNext     TransitionKind = "next"
	TransitionRetry    TransitionKind = "retry"
	TransitionEscalate TransitionKind = "escalate"
	TransitionSucceed  TransitionKind = "succeed"
	TransitionFail     TransitionKind = "fail"
)

// TransitionAction is a validated tagged union. NextStage, Retry, and Escalate
// are set only for their matching kinds; succeed/fail carry no payload.
type TransitionAction struct {
	Kind      TransitionKind      `json:"kind"`
	NextStage string              `json:"nextStage,omitempty"`
	Retry     *RetryTransition    `json:"retry,omitempty"`
	Escalate  *EscalateTransition `json:"escalate,omitempty"`
}

type RetryTransition struct {
	MaxAttempts int              `json:"maxAttempts"`
	Then        TransitionAction `json:"then"`
}

type EscalateTransition struct {
	PlannerContextManagement *PlannerContextManagement         `json:"plannerContextManagement,omitempty"`
	MaxAttempts              int                               `json:"maxAttempts"`
	Planner                  *PlannerRef                       `json:"planner,omitempty"`
	DeadlineSeconds          int                               `json:"deadlineSeconds,omitempty"`
	ExecutionConfig          ResolvedEscalationExecutionConfig `json:"executionConfig"`
	Then                     TransitionAction                  `json:"then,omitzero"`
}

type StageTransitions struct {
	Succeeded   TransitionAction  `json:"succeeded"`
	Failed      TransitionAction  `json:"failed"`
	Interrupted TransitionAction  `json:"interrupted"`
	Summarized  *TransitionAction `json:"summarized,omitempty"`
}

type ResolvedStage struct {
	Operation     *ServerOperationConfig                   `json:"operation,omitempty"`
	ModelDefaults map[string]contracts.ResolvedModelPolicy `json:"modelDefaults,omitempty"`
	MaxSubtasks   contracts.PlannerSubtaskLimit            `json:"maxSubtasks,omitempty"`
	Objective     string                                   `json:"objective"`
	Instructions  contracts.ResolvedInstructions           `json:"instructions,omitzero"`
	Planner       PlannerRef                               `json:"planner,omitzero"`
	// PlannerContextManagement is nil for none and the complete compact@1
	// branch otherwise (PLN-R125).
	PlannerContextManagement *PlannerContextManagement       `json:"plannerContextManagement,omitempty"`
	ScanPlan                 *scan.ScanPlanPolicy            `json:"scanPlan,omitempty"`
	AuditScan                *AuditScanConfig                `json:"auditScan,omitempty"`
	Session                  contracts.WorkerSessionMode     `json:"session"`
	Agents                   map[string]ResolvedAgentBinding `json:"agents"`
	ExecutionConfig          ResolvedStageExecutionConfig    `json:"executionConfig"`
	Context                  StageContext                    `json:"context"`
	Result                   StageResultContract             `json:"result"`
	WorkflowOutputs          map[string]string               `json:"workflowOutputs"`
	On                       StageTransitions                `json:"on"`
}

// ResolvedWorkflow is the complete immutable-in-snapshot Workflow definition.
// Workflow digests are intentionally not part of v1alpha1; Runs persist this
// complete value as their execution authority.
type ResolvedWorkflow struct {
	MaxSubtasks  contracts.PlannerSubtaskLimit `json:"maxSubtasks,omitempty"`
	Ref          WorkflowRef                   `json:"ref"`
	AuditTask    *AuditTaskExecution           `json:"auditTask,omitempty"`
	Presentation *WorkflowPresentation         `json:"presentation,omitempty"`
	Parameters   map[string]ParameterSlot      `json:"parameters"`
	Inputs       map[string]ArtifactSlot       `json:"inputs"`
	Outputs      map[string]ArtifactSlot       `json:"outputs"`
	EntryStage   string                        `json:"entryStage"`
	Stages       map[string]ResolvedStage      `json:"stages"`
}
