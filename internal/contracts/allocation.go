package contracts

import (
	"math"
	"strings"
	"time"
)

type AgentTemplateRef struct {
	TemplateID string `json:"templateId"`
	Version    string `json:"version"`
	Digest     string `json:"digest"`
}

type WorkerRuntimeRef struct {
	RuntimeID string `json:"runtimeId"`
	Version   string `json:"version"`
}

type ModelPolicyRef struct {
	PolicyID string `json:"policyId"`
	Version  string `json:"version"`
	Digest   string `json:"digest"`
}

type ToolsetRef struct {
	ToolsetID string `json:"toolsetId"`
	Version   string `json:"version"`
}

type SandboxProfileRef struct {
	SandboxProfileID string `json:"sandboxProfileId"`
	Version          string `json:"version"`
}

type ResolvedInstructions struct {
	Ref    string `json:"ref"`
	Digest string `json:"digest"`
	Text   string `json:"text"`
}

type ResolvedModelPolicy struct {
	Ref             ModelPolicyRef `json:"ref"`
	Model           string         `json:"model"`
	MaxOutputTokens int            `json:"maxOutputTokens"`
	Temperature     *float64       `json:"temperature,omitempty"`
}

type ToolsetSelection struct {
	Ref   ToolsetRef `json:"ref"`
	Tools []string   `json:"tools"`
}

type ResolvedAgentTemplate struct {
	Ref            AgentTemplateRef     `json:"ref"`
	Description    string               `json:"description"`
	Runtime        WorkerRuntimeRef     `json:"runtime"`
	Instructions   ResolvedInstructions `json:"instructions"`
	ModelPolicy    ResolvedModelPolicy  `json:"modelPolicy"`
	Toolsets       []ToolsetSelection   `json:"toolsets"`
	SandboxProfile SandboxProfileRef    `json:"sandboxProfile"`
}

func (t ResolvedAgentTemplate) Validate() error { return validateResolvedAgentTemplate(t) }

type RuntimeSettings struct {
	LLMGatewayURL         string       `json:"llmGatewayUrl"`
	LLMGatewayToken       SecretString `json:"llmGatewayToken"`
	ArtifactAPIURL        string       `json:"artifactApiUrl"`
	RequestTimeoutSeconds int          `json:"requestTimeoutSeconds"`
}

type AllocationSpec struct {
	APIVersion       string                `json:"apiVersion"`
	AllocationID     string                `json:"allocationId"`
	RunID            string                `json:"runId"`
	StageExecutionID string                `json:"stageExecutionId"`
	LogicalAgentName string                `json:"logicalAgentName"`
	Namespace        string                `json:"namespace"`
	AgentTemplate    ResolvedAgentTemplate `json:"agentTemplate"`
	RuntimeSettings  RuntimeSettings       `json:"runtimeSettings"`
}

func (s AllocationSpec) Validate() error {
	if err := validateAPIVersion(s.APIVersion); err != nil {
		return err
	}
	for field, value := range map[string]string{
		"allocationId": s.AllocationID, "runId": s.RunID,
		"stageExecutionId": s.StageExecutionID, "logicalAgentName": s.LogicalAgentName,
		"namespace": s.Namespace,
	} {
		if err := validateOpaqueID(field, value); err != nil {
			return err
		}
	}
	if strings.Contains(s.Namespace, "/") {
		return invalidf("namespace must not contain slash")
	}
	if err := validateResolvedAgentTemplate(s.AgentTemplate); err != nil {
		return err
	}
	return validateRuntimeSettings(s.RuntimeSettings)
}

type PrepareAllocationRequest struct {
	APIVersion string         `json:"apiVersion"`
	Spec       AllocationSpec `json:"spec"`
}

func (r PrepareAllocationRequest) Validate() error {
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	return r.Spec.Validate()
}

type WorkerHandle struct {
	AllocationID     string           `json:"allocationId"`
	AgentTemplateRef AgentTemplateRef `json:"agentTemplateRef"`
	WorkerRuntimeRef WorkerRuntimeRef `json:"workerRuntimeRef"`
	AgentCard        map[string]any   `json:"agentCard"`
	LeaseExpiresAt   time.Time        `json:"leaseExpiresAt"`
}

type PrepareAllocationResponse struct {
	APIVersion   string       `json:"apiVersion"`
	WorkerHandle WorkerHandle `json:"workerHandle"`
}

func (r PrepareAllocationResponse) Validate() error {
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if err := validateOpaqueID("workerHandle.allocationId", r.WorkerHandle.AllocationID); err != nil {
		return err
	}
	if err := validateTemplateRef(r.WorkerHandle.AgentTemplateRef); err != nil {
		return err
	}
	if err := validateRuntimeRef(r.WorkerHandle.WorkerRuntimeRef); err != nil {
		return err
	}
	if len(r.WorkerHandle.AgentCard) == 0 || r.WorkerHandle.LeaseExpiresAt.IsZero() {
		return invalidf("workerHandle Agent Card and leaseExpiresAt are required")
	}
	return nil
}

type FinalizeAllocationRequest struct {
	APIVersion     string    `json:"apiVersion"`
	AllocationID   string    `json:"allocationId"`
	FinalizationID string    `json:"finalizationId"`
	Deadline       time.Time `json:"deadline"`
}

func (r FinalizeAllocationRequest) Validate() error {
	return validateLifecycleRequest(r.APIVersion, r.AllocationID, "finalizationId", r.FinalizationID, r.Deadline)
}

type TerminationError struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Retryable bool   `json:"retryable"`
}

type AbortAllocationRequest struct {
	APIVersion   string           `json:"apiVersion"`
	AllocationID string           `json:"allocationId"`
	AbortID      string           `json:"abortId"`
	Reason       TerminationError `json:"reason"`
	Deadline     time.Time        `json:"deadline"`
}

func (r AbortAllocationRequest) Validate() error {
	if err := validateLifecycleRequest(r.APIVersion, r.AllocationID, "abortId", r.AbortID, r.Deadline); err != nil {
		return err
	}
	return validateTerminationError(r.Reason)
}

type ReleaseAllocationRequest struct {
	APIVersion   string `json:"apiVersion"`
	AllocationID string `json:"allocationId"`
}

func (r ReleaseAllocationRequest) Validate() error {
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	return validateOpaqueID("allocationId", r.AllocationID)
}

type ExecutionReport struct {
	AllocationID string             `json:"allocationId"`
	StartedAt    time.Time          `json:"startedAt"`
	FinishedAt   time.Time          `json:"finishedAt"`
	Complete     bool               `json:"complete"`
	Counters     map[string]int64   `json:"counters"`
	Errors       []TerminationError `json:"errors"`
	Truncated    bool               `json:"truncated"`
}

type AllocationFinalResponse struct {
	APIVersion string          `json:"apiVersion"`
	Report     ExecutionReport `json:"report"`
}

func (r AllocationFinalResponse) Validate() error {
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	if err := validateOpaqueID("report.allocationId", r.Report.AllocationID); err != nil {
		return err
	}
	if r.Report.StartedAt.IsZero() || r.Report.FinishedAt.IsZero() || r.Report.FinishedAt.Before(r.Report.StartedAt) {
		return invalidf("report timestamps are invalid")
	}
	for key, value := range r.Report.Counters {
		if key == "" || value < 0 {
			return invalidf("report counters must have non-empty names and non-negative values")
		}
	}
	for _, item := range r.Report.Errors {
		if err := validateTerminationError(item); err != nil {
			return err
		}
	}
	return nil
}

func validateResolvedAgentTemplate(template ResolvedAgentTemplate) error {
	if err := validateTemplateRef(template.Ref); err != nil {
		return err
	}
	if strings.TrimSpace(template.Description) == "" {
		return invalidf("agentTemplate.description must not be empty")
	}
	if err := validateRuntimeRef(template.Runtime); err != nil {
		return err
	}
	if strings.TrimSpace(template.Instructions.Ref) == "" || strings.TrimSpace(template.Instructions.Text) == "" {
		return invalidf("agentTemplate instructions ref/text must not be empty")
	}
	if err := validateDigest("agentTemplate.instructions.digest", template.Instructions.Digest); err != nil {
		return err
	}
	if err := validateModelPolicy(template.ModelPolicy); err != nil {
		return err
	}
	seenToolsets := make(map[string]struct{})
	seenTools := make(map[string]struct{})
	for _, selection := range template.Toolsets {
		selector := selection.Ref.ToolsetID + "@" + selection.Ref.Version
		if err := validateSelector("agentTemplate.toolsets.ref", selector); err != nil {
			return err
		}
		if _, exists := seenToolsets[selector]; exists {
			return invalidf("duplicate AgentTemplate toolset %q", selector)
		}
		seenToolsets[selector] = struct{}{}
		if len(selection.Tools) == 0 {
			return invalidf("AgentTemplate toolset %q has no selected tools", selector)
		}
		for _, tool := range selection.Tools {
			if err := validateOpaqueID("selected tool", tool); err != nil {
				return err
			}
			if _, exists := seenTools[tool]; exists {
				return invalidf("duplicate model-visible tool name %q", tool)
			}
			seenTools[tool] = struct{}{}
		}
	}
	return validateSelector(
		"agentTemplate.sandboxProfile",
		template.SandboxProfile.SandboxProfileID+"@"+template.SandboxProfile.Version,
	)
}

func validateTemplateRef(ref AgentTemplateRef) error {
	if err := validateSelector("agentTemplateRef", ref.TemplateID+"@"+ref.Version); err != nil {
		return err
	}
	return validateDigest("agentTemplateRef.digest", ref.Digest)
}

func validateRuntimeRef(ref WorkerRuntimeRef) error {
	return validateSelector("workerRuntimeRef", ref.RuntimeID+"@"+ref.Version)
}

func validateModelPolicy(policy ResolvedModelPolicy) error {
	if err := validateSelector("modelPolicyRef", policy.Ref.PolicyID+"@"+policy.Ref.Version); err != nil {
		return err
	}
	if err := validateDigest("modelPolicyRef.digest", policy.Ref.Digest); err != nil {
		return err
	}
	if strings.TrimSpace(policy.Model) == "" || policy.MaxOutputTokens <= 0 {
		return invalidf("modelPolicy model and positive maxOutputTokens are required")
	}
	if policy.Temperature != nil {
		temperature := *policy.Temperature
		if temperature < 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
			return invalidf("modelPolicy temperature must be finite and non-negative")
		}
	}
	return nil
}

func validateRuntimeSettings(settings RuntimeSettings) error {
	if err := validateURL("runtimeSettings.llmGatewayUrl", settings.LLMGatewayURL); err != nil {
		return err
	}
	if settings.LLMGatewayToken.Reveal() == "" {
		return invalidf("runtimeSettings.llmGatewayToken must not be empty")
	}
	if err := validateURL("runtimeSettings.artifactApiUrl", settings.ArtifactAPIURL); err != nil {
		return err
	}
	if settings.RequestTimeoutSeconds <= 0 {
		return invalidf("runtimeSettings.requestTimeoutSeconds must be positive")
	}
	return nil
}

func validateLifecycleRequest(apiVersion, allocationID, idField, idValue string, deadline time.Time) error {
	if err := validateAPIVersion(apiVersion); err != nil {
		return err
	}
	if err := validateOpaqueID("allocationId", allocationID); err != nil {
		return err
	}
	if err := validateOpaqueID(idField, idValue); err != nil {
		return err
	}
	if deadline.IsZero() {
		return invalidf("deadline must not be zero")
	}
	return nil
}

func validateTerminationError(value TerminationError) error {
	if strings.TrimSpace(value.Code) == "" || strings.TrimSpace(value.Message) == "" {
		return invalidf("termination error code/message must not be empty")
	}
	return nil
}
