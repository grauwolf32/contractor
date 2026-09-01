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

func (r AgentTemplateRef) ValidateRef() error { return validateTemplateRef(r) }

type WorkerRuntimeRef struct {
	RuntimeID string `json:"runtimeId"`
	Version   string `json:"version"`
}

type ModelPolicyRef struct {
	PolicyID string `json:"policyId"`
	Version  string `json:"version"`
	Digest   string `json:"digest"`
}

func (r ModelPolicyRef) ValidateRef() error {
	if err := validateSelector("modelPolicyRef", r.PolicyID+"@"+r.Version); err != nil {
		return err
	}
	return validateDigest("modelPolicyRef.digest", r.Digest)
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
	MaxOutputTokens int            `json:"maxOutputTokens,omitempty"`
	MaxModelCalls   int            `json:"maxModelCalls,omitempty"`
	MaxToolCalls    int            `json:"maxToolCalls,omitempty"`
	MaxWorkerCalls  int            `json:"maxWorkerCalls,omitempty"`
	MaxTotalTokens  int            `json:"maxTotalTokens,omitempty"`
	Temperature     *float64       `json:"temperature,omitempty"`
}

func (p ResolvedModelPolicy) Validate() error { return validateModelPolicy(p) }

func (p ResolvedModelPolicy) ValidateForWorker(hasTools bool) error {
	return validateWorkerModelPolicy(p, hasTools)
}

func (p ResolvedModelPolicy) ValidateForPlanner() error {
	if err := validateModelPolicy(p); err != nil {
		return err
	}
	if p.MaxOutputTokens <= 0 {
		return invalidf("Planner modelPolicy requires maxOutputTokens")
	}
	if p.MaxModelCalls <= 0 {
		return invalidf("Planner modelPolicy requires maxModelCalls")
	}
	if p.MaxWorkerCalls <= 0 {
		return invalidf("Planner modelPolicy requires maxWorkerCalls")
	}
	if p.MaxTotalTokens <= 0 {
		return invalidf("Planner modelPolicy requires maxTotalTokens")
	}
	if p.MaxToolCalls != 0 {
		return invalidf("Planner modelPolicy must omit maxToolCalls")
	}
	return nil
}

const (
	MaxWorkerModelCalls   = 1_000
	MaxWorkerToolCalls    = 10_000
	MaxPlannerWorkerCalls = 10_000
	MaxWorkerTotalTokens  = 100_000_000
)

var nativeSkillToolNames = map[string]struct{}{
	"list_skills": {}, "load_skill": {}, "load_skill_resource": {},
}

func IsNativeSkillToolName(name string) bool {
	_, exists := nativeSkillToolNames[name]
	return exists
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
	Skills         []ArtifactRef        `json:"skills,omitempty"`
	SandboxProfile SandboxProfileRef    `json:"sandboxProfile"`
}

func (t ResolvedAgentTemplate) Validate() error { return validateResolvedAgentTemplate(t) }

type RuntimeSettings struct {
	LLMGatewayURL         string       `json:"llmGatewayUrl"`
	LLMGatewayToken       SecretString `json:"llmGatewayToken"`
	ArtifactAPIURL        string       `json:"artifactApiUrl"`
	RequestTimeoutSeconds int          `json:"requestTimeoutSeconds"`
}

// WorkerExecutionSettings is an in-process preparation value. AllocationSpec
// carries its two members as separate wire fields so effective policy never
// mutates the digest-bearing AgentTemplate.
type WorkerExecutionSettings struct {
	ModelPolicy     ResolvedModelPolicy
	RuntimeSettings RuntimeSettings
}

type AllocationSpec struct {
	APIVersion       string                `json:"apiVersion"`
	AllocationID     string                `json:"allocationId"`
	RunID            string                `json:"runId"`
	StageExecutionID string                `json:"stageExecutionId"`
	LogicalAgentName string                `json:"logicalAgentName"`
	Namespace        string                `json:"namespace"`
	LeaseExpiresAt   time.Time             `json:"leaseExpiresAt"`
	AgentTemplate    ResolvedAgentTemplate `json:"agentTemplate"`
	ResolvedSkills   []ResolvedSkill       `json:"resolvedSkills"`
	ModelPolicy      ResolvedModelPolicy   `json:"modelPolicy"`
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
	if s.LeaseExpiresAt.IsZero() {
		return invalidf("leaseExpiresAt must not be zero")
	}
	if err := validateResolvedAgentTemplate(s.AgentTemplate); err != nil {
		return err
	}
	if err := ValidateResolvedSkills(s.AgentTemplate, s.ResolvedSkills); err != nil {
		return err
	}
	if err := validateWorkerModelPolicy(s.ModelPolicy, len(s.AgentTemplate.Toolsets) > 0 || len(s.AgentTemplate.Skills) > 0); err != nil {
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
	AllocationID string `json:"allocationId"`
	// RuntimeAgentID is Server-owned routing/authentication metadata. It is
	// never accepted from or emitted to the Runtime private wire response.
	RuntimeAgentID   string           `json:"-"`
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

type AllocationFinalResponse struct {
	APIVersion string                `json:"apiVersion"`
	Report     AllocationFinalReport `json:"report"`
}

func (r AllocationFinalResponse) Validate() error {
	if err := validateAPIVersion(r.APIVersion); err != nil {
		return err
	}
	return r.Report.Validate()
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
	if err := validateWorkerModelPolicy(template.ModelPolicy, len(template.Toolsets) > 0 || len(template.Skills) > 0); err != nil {
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
			if len(template.Skills) > 0 && IsNativeSkillToolName(tool) {
				return invalidf("model-visible tool name %q is reserved by Agent Skills", tool)
			}
			seenTools[tool] = struct{}{}
		}
	}
	if len(template.Skills) > MaxAgentTemplateSkills {
		return invalidf("AgentTemplate may select at most %d skills", MaxAgentTemplateSkills)
	}
	previousSkill := ""
	for _, skill := range template.Skills {
		if err := skill.ValidateAgentSkillRef(); err != nil {
			return err
		}
		if previousSkill != "" && skill.Name <= previousSkill {
			return invalidf("AgentTemplate skills must be sorted and unique")
		}
		previousSkill = skill.Name
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
	if strings.TrimSpace(policy.Model) == "" {
		return invalidf("modelPolicy model is required")
	}
	if policy.MaxOutputTokens < 0 {
		return invalidf("modelPolicy maxOutputTokens must be positive when present")
	}
	if policy.MaxModelCalls < 0 || policy.MaxModelCalls > MaxWorkerModelCalls {
		return invalidf("modelPolicy maxModelCalls must be between 1 and %d when present", MaxWorkerModelCalls)
	}
	if policy.MaxToolCalls < 0 || policy.MaxToolCalls > MaxWorkerToolCalls {
		return invalidf("modelPolicy maxToolCalls must be between 1 and %d when present", MaxWorkerToolCalls)
	}
	if policy.MaxWorkerCalls < 0 || policy.MaxWorkerCalls > MaxPlannerWorkerCalls {
		return invalidf("modelPolicy maxWorkerCalls must be between 1 and %d when present", MaxPlannerWorkerCalls)
	}
	if policy.MaxTotalTokens < 0 || policy.MaxTotalTokens > MaxWorkerTotalTokens {
		return invalidf("modelPolicy maxTotalTokens must be between 1 and %d when present", MaxWorkerTotalTokens)
	}
	if policy.Temperature != nil {
		temperature := *policy.Temperature
		if temperature < 0 || math.IsNaN(temperature) || math.IsInf(temperature, 0) {
			return invalidf("modelPolicy temperature must be finite and non-negative")
		}
	}
	return nil
}

func validateWorkerModelPolicy(policy ResolvedModelPolicy, hasTools bool) error {
	if err := validateModelPolicy(policy); err != nil {
		return err
	}
	if policy.MaxOutputTokens <= 0 {
		return invalidf("Worker modelPolicy requires maxOutputTokens")
	}
	if policy.MaxModelCalls <= 0 {
		return invalidf("Worker modelPolicy requires maxModelCalls")
	}
	if policy.MaxTotalTokens <= 0 {
		return invalidf("Worker modelPolicy requires maxTotalTokens")
	}
	if hasTools && policy.MaxToolCalls <= 0 {
		return invalidf("tool-using Worker modelPolicy requires maxToolCalls")
	}
	if policy.MaxWorkerCalls != 0 {
		return invalidf("Worker modelPolicy must omit maxWorkerCalls")
	}
	return nil
}

func validateRuntimeSettings(settings RuntimeSettings) error {
	if err := validateURL("runtimeSettings.llmGatewayUrl", settings.LLMGatewayURL); err != nil {
		return err
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
