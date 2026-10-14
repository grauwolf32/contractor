package config

import (
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
	"go.yaml.in/yaml/v4"
)

const (
	modelPolicyKind      = "ModelPolicy"
	llmGatewayConfigKind = "LLMGatewayConfig"
	executionConfigKind  = "ExecutionConfig"
	agentTemplateKind    = "AgentTemplate"
	workflowKind         = "Workflow"
)

// Counts summarizes a successfully published configuration snapshot.
type Counts struct {
	Workflows        int
	AgentTemplates   int
	ModelPolicies    int
	LLMGateways      int
	ExecutionConfigs int
	AuditProfiles    int
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
	Protocol          string                             `yaml:"protocol"`
	URL               string                             `yaml:"url"`
	CredentialManager *llmCredentialManagerSpecSource    `yaml:"credentialManager,omitempty"`
	FailureSignatures *llmGatewayFailureSignaturesSource `yaml:"failureSignatures,omitempty"`
}

type llmGatewayFailureSignaturesSource struct {
	ModelUnavailable []llmGatewayFailureSignatureSource `yaml:"modelUnavailable,omitempty"`
	PermanentCodes   []string                           `yaml:"permanentCodes,omitempty"`
}

type llmGatewayFailureSignatureSource struct {
	Status         int    `yaml:"status"`
	MessageEquals  string `yaml:"messageEquals,omitempty"`
	LiteLLMWrapped string `yaml:"litellmWrapped,omitempty"`
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
	Description    string                         `yaml:"description"`
	Runtime        string                         `yaml:"runtime"`
	Instructions   *instructionsRefSource         `yaml:"instructions"`
	ModelPolicy    string                         `yaml:"modelPolicy"`
	Execution      *contracts.ToolExecutionConfig `yaml:"execution,omitempty"`
	Summarizer     *workerSummarizerSource        `yaml:"summarizer,omitempty"`
	Toolsets       *[]toolsetSelectionSource      `yaml:"toolsets"`
	Skills         *[]artifactRefSource           `yaml:"skills,omitempty"`
	SandboxProfile string                         `yaml:"sandboxProfile"`
}

type workerSummarizerSource struct {
	Instructions       *instructionsRefSource `yaml:"instructions,omitempty"`
	ModelPolicy        string                 `yaml:"modelPolicy"`
	ContextWindowRatio *float64               `yaml:"contextWindowRatio,omitempty"`
	CumulativeBudget   *int                   `yaml:"cumulativeBudget,omitempty"`
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
	AuditTask       *resolved.AuditTaskExecution    `yaml:"auditTask,omitempty"`
	Presentation    *workflowPresentationSource     `yaml:"presentation,omitempty"`
	Parameters      *map[string]parameterSlotSource `yaml:"parameters"`
	Inputs          *map[string]artifactSlotSource  `yaml:"inputs"`
	Outputs         *map[string]artifactSlotSource  `yaml:"outputs"`
	ExecutionConfig *workflowExecutionConfigSource  `yaml:"executionConfig,omitempty"`
	EntryStage      string                          `yaml:"entryStage"`
	Stages          *map[string]stageSource         `yaml:"stages"`
}

type workflowPresentationSource struct {
	DisplayName string `yaml:"displayName"`
	Description string `yaml:"description"`
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
	Primary    *bool                  `yaml:"primary,omitempty"`
}

type artifactBindingSource struct {
	Namespace string `yaml:"namespace"`
	Name      string `yaml:"name"`
}

type stageSource struct {
	Objective       string                        `yaml:"objective"`
	Instructions    *instructionsRefSource        `yaml:"instructions"`
	Planner         string                        `yaml:"planner"`
	ScanPlan        *scan.ScanPlanPolicy          `yaml:"scanPlan,omitempty"`
	AuditScan       *resolved.AuditScanConfig     `yaml:"auditScan,omitempty"`
	Session         yaml.Node                     `yaml:"session,omitempty"`
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
	Succeeded         *transitionActionSource `yaml:"succeeded"`
	Failed            *transitionActionSource `yaml:"failed"`
	Interrupted       *transitionActionSource `yaml:"interrupted"`
	Summarized        *transitionActionSource `yaml:"summarized,omitempty"`
	summarizedPresent bool
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
	Planner         *string                          `yaml:"planner,omitempty"`
	DeadlineSeconds *int                             `yaml:"deadlineSeconds,omitempty"`
	ExecutionConfig *escalationExecutionConfigSource `yaml:"executionConfig"`
	Then            *transitionActionSource          `yaml:"then"`
	plannerPresent  bool
	deadlinePresent bool
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
