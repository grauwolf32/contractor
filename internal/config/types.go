package config

import "github.com/grauwolf32/contractor/internal/contracts"

const (
	modelPolicyKind   = "ModelPolicy"
	agentTemplateKind = "AgentTemplate"
	workflowKind      = "Workflow"
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

// ArtifactSlot declares one versioned artifact input, output, or Stage result.
type ArtifactSlot struct {
	Required   bool     `json:"required"`
	MediaTypes []string `json:"mediaTypes"`
}

// ContextArtifact identifies a logical RunScope binding to pin for a Stage.
type ContextArtifact struct {
	Namespace string `json:"namespace"`
	Name      string `json:"name"`
	Required  bool   `json:"required"`
}

type StageContext struct {
	Artifacts map[string]ContextArtifact `json:"artifacts"`
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

type TransitionKind string

const (
	TransitionNext    TransitionKind = "next"
	TransitionRetry   TransitionKind = "retry"
	TransitionSucceed TransitionKind = "succeed"
	TransitionFail    TransitionKind = "fail"
)

// TransitionAction is a validated tagged union. NextStage is set only for
// next; Retry is set only for retry; succeed/fail carry no payload.
type TransitionAction struct {
	Kind      TransitionKind   `json:"kind"`
	NextStage string           `json:"nextStage,omitempty"`
	Retry     *RetryTransition `json:"retry,omitempty"`
}

type RetryTransition struct {
	MaxAttempts int              `json:"maxAttempts"`
	Then        TransitionAction `json:"then"`
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
	Workflows      int
	AgentTemplates int
	ModelPolicies  int
	Instructions   int
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
	Model           string   `yaml:"model"`
	MaxOutputTokens int      `yaml:"maxOutputTokens"`
	Temperature     *float64 `yaml:"temperature,omitempty"`
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
	Toolsets       *[]toolsetSelectionSource `yaml:"toolsets"`
	SandboxProfile string                    `yaml:"sandboxProfile"`
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
	Parameters *map[string]parameterSlotSource `yaml:"parameters"`
	Inputs     *map[string]artifactSlotSource  `yaml:"inputs"`
	Outputs    *map[string]artifactSlotSource  `yaml:"outputs"`
	EntryStage string                          `yaml:"entryStage"`
	Stages     *map[string]stageSource         `yaml:"stages"`
}

type parameterSlotSource struct {
	Required *bool `yaml:"required"`
}

type artifactSlotSource struct {
	Required   *bool    `yaml:"required"`
	MediaTypes []string `yaml:"mediaTypes"`
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
	Next    *string            `yaml:"next,omitempty"`
	Retry   *retrySource       `yaml:"retry,omitempty"`
	Succeed *emptyObjectSource `yaml:"succeed,omitempty"`
	Fail    *emptyObjectSource `yaml:"fail,omitempty"`
}

type retrySource struct {
	MaxAttempts int                     `yaml:"maxAttempts"`
	Then        *transitionActionSource `yaml:"then"`
}

type emptyObjectSource struct{}
