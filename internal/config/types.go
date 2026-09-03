package config

import (
	"github.com/grauwolf32/contractor/internal/contracts"
	"go.yaml.in/yaml/v4"
)

const (
	modelPolicyKind      = "ModelPolicy"
	llmGatewayConfigKind = "LLMGatewayConfig"
	executionConfigKind  = "ExecutionConfig"
	agentTemplateKind    = "AgentTemplate"
	workflowKind         = "Workflow"
)

// Selector is an exact, versioned configuration or code descriptor lookup.
type Selector struct {
	ID      string
	Version string
}

func (s Selector) String() string { return s.ID + "@" + s.Version }

// WorkflowRef identifies one exact Workflow configuration document.
type WorkflowRef struct {
	Name    string `json:"name"`
	Version string `json:"version"`
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
	Mode    contracts.WorkspaceModeV2 `json:"mode"`
	Sources []WorkspaceSource         `json:"sources"`
	State   *WorkspaceStateInput      `json:"state,omitempty"`
	Export  *WorkspaceExport          `json:"export,omitempty"`
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
	ModelPolicy string `json:"modelPolicy"`
	LLMGateway  string `json:"llmGateway,omitempty"`
	Credential  string `json:"credential,omitempty"`
}

// ResolvedConsumerExecutionConfig is one complete immutable model-access
// selection. Credential is a non-secret identity; token bytes never enter a
// Workflow or Run snapshot.
type ResolvedConsumerExecutionConfig struct {
	ModelPolicy contracts.ResolvedModelPolicy       `json:"modelPolicy"`
	LLMGateway  *contracts.ResolvedLLMGatewayConfig `json:"llmGateway,omitempty"`
	Credential  *contracts.LLMCredentialRef         `json:"credential,omitempty"`
	Origins     ExecutionConfigOrigins              `json:"origins"`
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
	Clear bool                        `json:"clear"`
	Ref   *contracts.LLMCredentialRef `json:"ref,omitempty"`
}

// ResolvedExecutionSelectionOverride is a partial, dependency-resolved
// selection. Nil fields inherit the Run's base Stage configuration.
type ResolvedExecutionSelectionOverride struct {
	ModelPolicy *contracts.ResolvedModelPolicy      `json:"modelPolicy,omitempty"`
	LLMGateway  *contracts.ResolvedLLMGatewayConfig `json:"llmGateway,omitempty"`
	Credential  *ResolvedCredentialOverride         `json:"credential,omitempty"`
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
	ExecutionConfig ResolvedEscalationExecutionConfig `json:"executionConfig"`
	Then            TransitionAction                  `json:"then"`
}

type StageTransitions struct {
	Succeeded   TransitionAction `json:"succeeded"`
	Failed      TransitionAction `json:"failed"`
	Interrupted TransitionAction `json:"interrupted"`
}

type ResolvedStage struct {
	Objective       string                          `json:"objective"`
	Instructions    contracts.ResolvedInstructions  `json:"instructions"`
	Planner         PlannerRef                      `json:"planner"`
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
	Ref        WorkflowRef              `json:"ref"`
	Parameters map[string]ParameterSlot `json:"parameters"`
	Inputs     map[string]ArtifactSlot  `json:"inputs"`
	Outputs    map[string]ArtifactSlot  `json:"outputs"`
	EntryStage string                   `json:"entryStage"`
	Stages     map[string]ResolvedStage `json:"stages"`
}

// Counts summarizes a successfully published configuration snapshot.
type Counts struct {
	Workflows        int
	AgentTemplates   int
	ModelPolicies    int
	LLMGateways      int
	ExecutionConfigs int
	Instructions     int
}

type metadataSource struct {
	Name    string `yaml:"name"`
	Version string `yaml:"version"`
}

type instructionsRefSource struct {
	Ref string `yaml:"ref"`
}

type modelPolicyDocument struct {
	APIVersion string                 `yaml:"apiVersion"`
	Kind       string                 `yaml:"kind"`
	Metadata   *metadataSource        `yaml:"metadata"`
	Spec       *modelPolicySpecSource `yaml:"spec"`
}

type modelPolicySpecSource struct {
	Model               string   `yaml:"model"`
	ContextWindowTokens *int     `yaml:"contextWindowTokens,omitempty"`
	MaxOutputTokens     *int     `yaml:"maxOutputTokens,omitempty"`
	MaxModelCalls       *int     `yaml:"maxModelCalls,omitempty"`
	MaxToolCalls        *int     `yaml:"maxToolCalls,omitempty"`
	MaxWorkerCalls      *int     `yaml:"maxWorkerCalls,omitempty"`
	MaxTotalTokens      *int     `yaml:"maxTotalTokens,omitempty"`
	Temperature         *float64 `yaml:"temperature,omitempty"`
}

type llmGatewayConfigDocument struct {
	APIVersion string                      `yaml:"apiVersion"`
	Kind       string                      `yaml:"kind"`
	Metadata   *metadataSource             `yaml:"metadata"`
	Spec       *llmGatewayConfigSpecSource `yaml:"spec"`
}

type llmGatewayConfigSpecSource struct {
	Protocol          string                          `yaml:"protocol"`
	URL               string                          `yaml:"url"`
	CredentialManager *llmCredentialManagerSpecSource `yaml:"credentialManager,omitempty"`
}

type llmCredentialManagerSpecSource struct {
	Implementation string `yaml:"implementation"`
	ManagementURL  string `yaml:"managementUrl"`
}

type executionConfigDocument struct {
	APIVersion string                            `yaml:"apiVersion"`
	Kind       string                            `yaml:"kind"`
	Metadata   *metadataSource                   `yaml:"metadata"`
	Spec       *executionConfigProfileSpecSource `yaml:"spec"`
}

type executionConfigProfileSpecSource struct {
	Planner        *executionSelectionSource            `yaml:"planner,omitempty"`
	Agents         *map[string]executionSelectionSource `yaml:"agents,omitempty"`
	plannerPresent bool
	agentsPresent  bool
}

type agentTemplateDocument struct {
	APIVersion string                   `yaml:"apiVersion"`
	Kind       string                   `yaml:"kind"`
	Metadata   *metadataSource          `yaml:"metadata"`
	Spec       *agentTemplateSpecSource `yaml:"spec"`
}

type agentTemplateSpecSource struct {
	Description    string                    `yaml:"description"`
	Runtime        string                    `yaml:"runtime"`
	Instructions   *instructionsRefSource    `yaml:"instructions"`
	ModelPolicy    string                    `yaml:"modelPolicy"`
	Summarizer     *workerSummarizerSource   `yaml:"summarizer,omitempty"`
	Toolsets       *[]toolsetSelectionSource `yaml:"toolsets"`
	Skills         *[]artifactRefSource      `yaml:"skills,omitempty"`
	SandboxProfile string                    `yaml:"sandboxProfile"`
}

type workerSummarizerSource struct {
	ModelPolicy        string   `yaml:"modelPolicy"`
	ContextWindowRatio *float64 `yaml:"contextWindowRatio,omitempty"`
	CumulativeBudget   *int     `yaml:"cumulativeBudget,omitempty"`
}

type artifactRefSource struct {
	Namespace string  `yaml:"namespace"`
	Name      string  `yaml:"name"`
	Revision  *string `yaml:"revision,omitempty"`
}

type toolsetSelectionSource struct {
	Ref   string   `yaml:"ref"`
	Tools []string `yaml:"tools"`
}

type workflowDocument struct {
	APIVersion string              `yaml:"apiVersion"`
	Kind       string              `yaml:"kind"`
	Metadata   *metadataSource     `yaml:"metadata"`
	Spec       *workflowSpecSource `yaml:"spec"`
}

type workflowSpecSource struct {
	Parameters      *map[string]parameterSlotSource `yaml:"parameters"`
	Inputs          *map[string]artifactSlotSource  `yaml:"inputs"`
	Outputs         *map[string]artifactSlotSource  `yaml:"outputs"`
	ExecutionConfig *workflowExecutionConfigSource  `yaml:"executionConfig,omitempty"`
	EntryStage      string                          `yaml:"entryStage"`
	Stages          *map[string]stageSource         `yaml:"stages"`
}

type executionSelectionSource struct {
	ModelPolicy yaml.Node `yaml:"modelPolicy,omitempty"`
	LLMGateway  yaml.Node `yaml:"llmGateway,omitempty"`
	Credential  yaml.Node `yaml:"credential,omitempty"`
}

type workflowExecutionConfigSource struct {
	Planner *executionSelectionSource             `yaml:"planner,omitempty"`
	Workers *executionSelectionSource             `yaml:"workers,omitempty"`
	Stages  map[string]stageExecutionConfigSource `yaml:"stages,omitempty"`
}

type stageExecutionConfigSource struct {
	Planner *executionSelectionSource           `yaml:"planner,omitempty"`
	Agents  map[string]executionSelectionSource `yaml:"agents,omitempty"`
}

type parameterSlotSource struct {
	Required *bool `yaml:"required"`
}

type artifactSlotSource struct {
	Required   *bool                  `yaml:"required"`
	MediaTypes []string               `yaml:"mediaTypes"`
	From       *artifactBindingSource `yaml:"from,omitempty"`
}

type artifactBindingSource struct {
	Namespace string `yaml:"namespace"`
	Name      string `yaml:"name"`
}

type stageSource struct {
	Objective       string                        `yaml:"objective"`
	Instructions    *instructionsRefSource        `yaml:"instructions"`
	Planner         string                        `yaml:"planner"`
	Agents          map[string]agentBindingSource `yaml:"agents"`
	Context         *stageContextSource           `yaml:"context,omitempty"`
	Result          *stageResultSource            `yaml:"result,omitempty"`
	WorkflowOutputs map[string]string             `yaml:"workflowOutputs,omitempty"`
	On              *stageTransitionsSource       `yaml:"on"`
}

type agentBindingSource struct {
	Template  string  `yaml:"template"`
	Namespace *string `yaml:"namespace,omitempty"`
}

type stageContextSource struct {
	Artifacts *map[string]contextArtifactSource `yaml:"artifacts"`
	Workspace *workspaceContextSource           `yaml:"workspace,omitempty"`
}

type workspaceContextSource struct {
	Mode    string               `yaml:"mode"`
	Sources []workspaceSource    `yaml:"sources"`
	State   *workspaceStateInput `yaml:"state,omitempty"`
	Export  *workspaceExport     `yaml:"export,omitempty"`
}

type workspaceSource struct {
	Artifact string `yaml:"artifact"`
	Target   string `yaml:"target"`
}

type workspaceStateInput struct {
	Artifact string `yaml:"artifact"`
}

type workspaceExport struct {
	State string `yaml:"state"`
	Diff  string `yaml:"diff"`
}

type contextArtifactSource struct {
	Namespace string `yaml:"namespace"`
	Name      string `yaml:"name"`
	Required  *bool  `yaml:"required"`
}

type stageResultSource struct {
	Artifacts *map[string]artifactSlotSource `yaml:"artifacts"`
}

type stageTransitionsSource struct {
	Succeeded   *transitionActionSource `yaml:"succeeded"`
	Failed      *transitionActionSource `yaml:"failed"`
	Interrupted *transitionActionSource `yaml:"interrupted"`
}

type transitionActionSource struct {
	Next     *string            `yaml:"next,omitempty"`
	Retry    *retrySource       `yaml:"retry,omitempty"`
	Escalate *escalateSource    `yaml:"escalate,omitempty"`
	Succeed  *emptyObjectSource `yaml:"succeed,omitempty"`
	Fail     *emptyObjectSource `yaml:"fail,omitempty"`
}

type retrySource struct {
	MaxAttempts int                     `yaml:"maxAttempts"`
	Then        *transitionActionSource `yaml:"then"`
}

type escalateSource struct {
	MaxAttempts     int                              `yaml:"maxAttempts"`
	ExecutionConfig *escalationExecutionConfigSource `yaml:"executionConfig"`
	Then            *transitionActionSource          `yaml:"then"`
}

type escalationExecutionConfigSource struct {
	Ref            yaml.Node                            `yaml:"ref,omitempty"`
	Planner        *executionSelectionSource            `yaml:"planner,omitempty"`
	Agents         *map[string]executionSelectionSource `yaml:"agents,omitempty"`
	refPresent     bool
	plannerPresent bool
	agentsPresent  bool
}

type emptyObjectSource struct{}
