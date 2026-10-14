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
	Required   bool             `json:"required"`
	MediaTypes []string         `json:"mediaTypes"`
	From       *ArtifactBinding `json:"from,omitempty"`
	Primary    bool             `json:"primary,omitempty"`
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
}

// ExecutionConfigOrigins records which authoring layer selected each resolved
// field. Values are safe provenance strings, never secret material.
type ExecutionConfigOrigins struct {
	ModelPolicy string `json:"modelPolicy,omitempty"`
	LLMGateway  string `json:"llmGateway,omitempty"`
	Credential  string `json:"credential,omitempty"`
}

// ResolvedConsumerExecutionConfig is one complete immutable model-access
// selection. Credential is a non-secret identity; token bytes never enter a
// Workflow or Run snapshot.
type ResolvedConsumerExecutionConfig struct {
	ModelPolicy contracts.ResolvedModelPolicy        `json:"modelPolicy,omitzero"`
	LLMGateway  *llmgateway.ResolvedLLMGatewayConfig `json:"llmGateway,omitempty"`
	Credential  *llmgateway.LLMCredentialRef         `json:"credential,omitempty"`
	Origins     ExecutionConfigOrigins               `json:"origins"`
}

type ResolvedStageExecutionConfig struct {
	Planner *ResolvedConsumerExecutionConfig           `json:"planner,omitempty"`
	Agents  map[string]ResolvedConsumerExecutionConfig `json:"agents"`
}

// ExecutionConfigRef identifies one immutable Stage-local escalation profile.
type ExecutionConfigRef struct {
	ConfigID string `json:"configId"`
	Version  string `json:"version"`
	Digest   string `json:"digest"`
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
	ModelPolicy *contracts.ResolvedModelPolicy       `json:"modelPolicy,omitempty"`
	LLMGateway  *llmgateway.ResolvedLLMGatewayConfig `json:"llmGateway,omitempty"`
	Credential  *ResolvedCredentialOverride          `json:"credential,omitempty"`
}

type ResolvedStageExecutionConfigOverride struct {
	Planner *ResolvedExecutionSelectionOverride           `json:"planner,omitempty"`
	Agents  map[string]ResolvedExecutionSelectionOverride `json:"agents,omitempty"`
}

// ResolvedExecutionConfigProfile is the safe immutable catalog value. It
// contains exact dependency bodies but never credential token bytes.
type ResolvedExecutionConfigProfile struct {
	Ref      ExecutionConfigRef                   `json:"ref"`
	Override ResolvedStageExecutionConfigOverride `json:"override"`
}

// ResolvedEscalationExecutionConfig pins both the declared patch and its fully
// effective result for one consuming Stage. Ref is present only for a named
// profile; inline declarations use the same Override representation.
type ResolvedEscalationExecutionConfig struct {
	Ref       *ExecutionConfigRef                  `json:"ref,omitempty"`
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
	MaxAttempts     int                               `json:"maxAttempts"`
	Planner         *PlannerRef                       `json:"planner,omitempty"`
	DeadlineSeconds int                               `json:"deadlineSeconds,omitempty"`
	ExecutionConfig ResolvedEscalationExecutionConfig `json:"executionConfig"`
	Then            TransitionAction                  `json:"then"`
}

type StageTransitions struct {
	Succeeded   TransitionAction  `json:"succeeded"`
	Failed      TransitionAction  `json:"failed"`
	Interrupted TransitionAction  `json:"interrupted"`
	Summarized  *TransitionAction `json:"summarized,omitempty"`
}

type ResolvedStage struct {
	Objective       string                          `json:"objective"`
	Instructions    contracts.ResolvedInstructions  `json:"instructions"`
	Planner         PlannerRef                      `json:"planner"`
	ScanPlan        *scan.ScanPlanPolicy            `json:"scanPlan,omitempty"`
	AuditScan       *AuditScanConfig                `json:"auditScan,omitempty"`
	Session         contracts.WorkerSessionMode     `json:"session"`
	Agents          map[string]ResolvedAgentBinding `json:"agents"`
	ExecutionConfig ResolvedStageExecutionConfig    `json:"executionConfig"`
	Context         StageContext                    `json:"context"`
	Result          StageResultContract             `json:"result"`
	WorkflowOutputs map[string]string               `json:"workflowOutputs"`
	On              StageTransitions                `json:"on"`
}

// ResolvedWorkflow is the complete immutable-in-snapshot Workflow definition.
// Workflow digests are intentionally not part of v1alpha1; Runs persist this
// complete value as their execution authority.
type ResolvedWorkflow struct {
	Ref          WorkflowRef              `json:"ref"`
	AuditTask    *AuditTaskExecution      `json:"auditTask,omitempty"`
	Presentation *WorkflowPresentation    `json:"presentation,omitempty"`
	Parameters   map[string]ParameterSlot `json:"parameters"`
	Inputs       map[string]ArtifactSlot  `json:"inputs"`
	Outputs      map[string]ArtifactSlot  `json:"outputs"`
	EntryStage   string                   `json:"entryStage"`
	Stages       map[string]ResolvedStage `json:"stages"`
}
