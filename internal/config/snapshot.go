package config

import (
	"fmt"
	"maps"
	"slices"
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
)

// Snapshot is an immutable, dependency-resolved view of one successful load.
// Its maps are private and every accessor returns a deep copy.
type Snapshot struct {
	workflowBindings map[string]AgentTemplateWorkflowBindings
	workflows        map[string]ResolvedWorkflow
	templates        map[string]contracts.ResolvedAgentTemplate
	policies         map[string]contracts.ResolvedModelPolicy
	gateways         map[string]llmgateway.ResolvedLLMGatewayConfig
	executionConfigs map[string]ResolvedExecutionConfigProfile
	auditProfiles    map[string]ResolvedAuditProfile
	instructions     map[string]contracts.ResolvedInstructions
	sources          map[string]ConfigurationSource
}

func newSnapshot(
	workflows map[string]ResolvedWorkflow,
	templates map[string]contracts.ResolvedAgentTemplate,
	policies map[string]contracts.ResolvedModelPolicy,
	gateways map[string]llmgateway.ResolvedLLMGatewayConfig,
	executionConfigs map[string]ResolvedExecutionConfigProfile,
	auditProfiles map[string]ResolvedAuditProfile,
	instructions map[string]contracts.ResolvedInstructions,
	sources map[string]ConfigurationSource,
) *Snapshot {
	result := &Snapshot{
		workflows:        make(map[string]ResolvedWorkflow, len(workflows)),
		templates:        make(map[string]contracts.ResolvedAgentTemplate, len(templates)),
		policies:         make(map[string]contracts.ResolvedModelPolicy, len(policies)),
		gateways:         make(map[string]llmgateway.ResolvedLLMGatewayConfig, len(gateways)),
		executionConfigs: make(map[string]ResolvedExecutionConfigProfile, len(executionConfigs)),
		auditProfiles:    make(map[string]ResolvedAuditProfile, len(auditProfiles)),
		instructions:     make(map[string]contracts.ResolvedInstructions, len(instructions)),
		sources:          make(map[string]ConfigurationSource, len(sources)),
	}
	for key, workflow := range workflows {
		result.workflows[key] = cloneWorkflow(workflow)
	}
	for key, template := range templates {
		result.templates[key] = template.Clone()
	}
	for key, policy := range policies {
		result.policies[key] = policy.Clone()
	}
	for key, gateway := range gateways {
		result.gateways[key] = cloneLLMGatewayConfig(gateway)
	}
	for key, profile := range executionConfigs {
		result.executionConfigs[key] = cloneExecutionConfigProfile(profile)
	}
	for key, profile := range auditProfiles {
		result.auditProfiles[key] = cloneAuditProfile(profile)
	}
	for key, instructions := range instructions {
		result.instructions[key] = instructions
	}
	for key, source := range sources {
		result.sources[key] = source
	}
	result.buildWorkflowBindingIndex()
	return result
}

func (s *Snapshot) Counts() Counts {
	return Counts{
		Workflows: len(s.workflows), AgentTemplates: len(s.templates),
		ModelPolicies: len(s.policies), LLMGateways: len(s.gateways),
		ExecutionConfigs: len(s.executionConfigs),
		AuditProfiles:    len(s.auditProfiles),
		Instructions:     len(s.instructions),
	}
}

// AuditProfile resolves one exact name@version and returns a caller-owned,
// fully dependency-resolved Server-side profile.
func (s *Snapshot) AuditProfile(raw string) (ResolvedAuditProfile, error) {
	selector, err := ParseSelector(raw)
	if err != nil {
		return ResolvedAuditProfile{}, err
	}
	profile, ok := s.auditProfiles[selector.String()]
	if !ok {
		return ResolvedAuditProfile{}, fmt.Errorf("unknown AuditProfile %q", selector)
	}
	return cloneAuditProfile(profile), nil
}

// AuditProfiles returns every profile sorted by exact ref and deeply detached
// from the immutable Snapshot.
func (s *Snapshot) AuditProfiles() []ResolvedAuditProfile {
	keys := slices.Sorted(maps.Keys(s.auditProfiles))
	result := make([]ResolvedAuditProfile, 0, len(keys))
	for _, key := range keys {
		result = append(result, cloneAuditProfile(s.auditProfiles[key]))
	}
	return result
}

// LLMGateway resolves one exact id@version and returns a caller-owned non-secret copy.
func (s *Snapshot) LLMGateway(raw string) (llmgateway.ResolvedLLMGatewayConfig, error) {
	selector, err := ParseSelector(raw)
	if err != nil {
		return llmgateway.ResolvedLLMGatewayConfig{}, err
	}
	gateway, ok := s.gateways[selector.String()]
	if !ok {
		return llmgateway.ResolvedLLMGatewayConfig{}, fmt.Errorf("unknown LLMGatewayConfig %q", selector)
	}
	return cloneLLMGatewayConfig(gateway), nil
}

// Workflow resolves one exact name@version and returns a caller-owned copy.
func (s *Snapshot) Workflow(raw string) (ResolvedWorkflow, error) {
	selector, err := ParseSelector(raw)
	if err != nil {
		return ResolvedWorkflow{}, err
	}
	workflow, ok := s.workflows[selector.String()]
	if !ok {
		return ResolvedWorkflow{}, fmt.Errorf("unknown Workflow %q", selector)
	}
	return cloneWorkflow(workflow), nil
}

// Workflows returns every published Workflow sorted by exact ref. Every value
// is a caller-owned deep copy; mutating the result cannot change the published
// configuration snapshot.
func (s *Snapshot) Workflows() []ResolvedWorkflow {
	keys := make([]string, 0, len(s.workflows))
	for key := range s.workflows {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := make([]ResolvedWorkflow, 0, len(keys))
	for _, key := range keys {
		result = append(result, cloneWorkflow(s.workflows[key]))
	}
	return result
}

// AgentTemplate resolves one exact id@version and returns a caller-owned copy.
// Cross-package conformance tests use it to inspect pinned template bodies;
// production execution reads templates through resolved Workflows.
func (s *Snapshot) AgentTemplate(raw string) (contracts.ResolvedAgentTemplate, error) {
	selector, err := ParseSelector(raw)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, err
	}
	template, ok := s.templates[selector.String()]
	if !ok {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("unknown AgentTemplate %q", selector)
	}
	return template.Clone(), nil
}

// ModelPolicy resolves one exact id@version and returns a caller-owned copy.
func (s *Snapshot) ModelPolicy(raw string) (contracts.ResolvedModelPolicy, error) {
	selector, err := ParseSelector(raw)
	if err != nil {
		return contracts.ResolvedModelPolicy{}, err
	}
	policy, ok := s.policies[selector.String()]
	if !ok {
		return contracts.ResolvedModelPolicy{}, fmt.Errorf("unknown ModelPolicy %q", selector)
	}
	return policy.Clone(), nil
}

// Instructions returns the exact text dependency pinned by a normalized ref.
// Repository conformance tests use it to verify packaged instruction sources;
// production execution reads the text pinned into resolved Workflows.
func (s *Snapshot) Instructions(raw string) (contracts.ResolvedInstructions, error) {
	normalized, err := validateInstructionRef(raw)
	if err != nil {
		return contracts.ResolvedInstructions{}, err
	}
	instructions, ok := s.instructions[normalized]
	if !ok {
		return contracts.ResolvedInstructions{}, fmt.Errorf("unknown resolved instruction %q", normalized)
	}
	return instructions, nil
}

func cloneLLMGatewayConfig(
	source llmgateway.ResolvedLLMGatewayConfig,
) llmgateway.ResolvedLLMGatewayConfig {
	result := source
	if source.CredentialManager != nil {
		manager := *source.CredentialManager
		result.CredentialManager = &manager
	}
	return result
}

func cloneWorkflow(source ResolvedWorkflow) ResolvedWorkflow {
	result := source
	result.AuditTask = cloneAuditTask(source.AuditTask)
	if source.Presentation != nil {
		presentation := *source.Presentation
		result.Presentation = &presentation
	}
	result.Parameters = make(map[string]ParameterSlot, len(source.Parameters))
	for name, slot := range source.Parameters {
		result.Parameters[name] = slot
	}
	result.Inputs = CloneArtifactSlots(source.Inputs)
	result.Outputs = CloneArtifactSlots(source.Outputs)
	result.Stages = make(map[string]ResolvedStage, len(source.Stages))
	for name, stage := range source.Stages {
		result.Stages[name] = cloneStage(stage)
	}
	return result
}

// CloneArtifactSlots returns a never-nil copy whose slots share no media types
// or source references with source.
func CloneArtifactSlots(source map[string]ArtifactSlot) map[string]ArtifactSlot {
	result := make(map[string]ArtifactSlot, len(source))
	for name, slot := range source {
		slot.MediaTypes = append([]string(nil), slot.MediaTypes...)
		if slot.From != nil {
			from := *slot.From
			slot.From = &from
		}
		result[name] = slot
	}
	return result
}

func cloneStage(source ResolvedStage) ResolvedStage {
	result := source
	result.ScanPlan = cloneScanPlanPolicy(source.ScanPlan)
	result.AuditScan = cloneAuditScan(source.AuditScan)
	result.Agents = make(map[string]ResolvedAgentBinding, len(source.Agents))
	for name, binding := range source.Agents {
		binding.Template = binding.Template.Clone()
		result.Agents[name] = binding
	}
	result.ExecutionConfig = cloneStageExecutionConfig(source.ExecutionConfig)
	result.Context.Artifacts = make(map[string]ContextArtifact, len(source.Context.Artifacts))
	for name, artifact := range source.Context.Artifacts {
		result.Context.Artifacts[name] = artifact
	}
	if source.Context.Workspace != nil {
		workspace := *source.Context.Workspace
		workspace.Sources = append([]WorkspaceSource(nil), source.Context.Workspace.Sources...)
		if source.Context.Workspace.State != nil {
			state := *source.Context.Workspace.State
			workspace.State = &state
		}
		if source.Context.Workspace.Export != nil {
			export := *source.Context.Workspace.Export
			workspace.Export = &export
		}
		result.Context.Workspace = &workspace
	}
	result.Result.Artifacts = CloneArtifactSlots(source.Result.Artifacts)
	result.WorkflowOutputs = make(map[string]string, len(source.WorkflowOutputs))
	for output, artifact := range source.WorkflowOutputs {
		result.WorkflowOutputs[output] = artifact
	}
	result.On = StageTransitions{
		Succeeded:   cloneTransition(source.On.Succeeded),
		Failed:      cloneTransition(source.On.Failed),
		Interrupted: cloneTransition(source.On.Interrupted),
	}
	return result
}

func cloneStageExecutionConfig(source ResolvedStageExecutionConfig) ResolvedStageExecutionConfig {
	result := ResolvedStageExecutionConfig{
		Agents: make(map[string]ResolvedConsumerExecutionConfig, len(source.Agents)),
	}
	if source.Planner != nil {
		planner := cloneConsumerExecutionConfig(*source.Planner)
		result.Planner = &planner
	}
	for name, selection := range source.Agents {
		result.Agents[name] = cloneConsumerExecutionConfig(selection)
	}
	return result
}

func cloneConsumerExecutionConfig(source ResolvedConsumerExecutionConfig) ResolvedConsumerExecutionConfig {
	result := source
	result.ModelPolicy = source.ModelPolicy.Clone()
	if source.LLMGateway != nil {
		gateway := cloneLLMGatewayConfig(*source.LLMGateway)
		result.LLMGateway = &gateway
	}
	if source.Credential != nil {
		credential := *source.Credential
		result.Credential = &credential
	}
	return result
}

func cloneExecutionConfigProfile(source ResolvedExecutionConfigProfile) ResolvedExecutionConfigProfile {
	result := source
	result.Override = cloneStageExecutionConfigOverride(source.Override)
	return result
}

func cloneStageExecutionConfigOverride(
	source ResolvedStageExecutionConfigOverride,
) ResolvedStageExecutionConfigOverride {
	result := ResolvedStageExecutionConfigOverride{}
	if source.Planner != nil {
		planner := cloneExecutionSelectionOverride(*source.Planner)
		result.Planner = &planner
	}
	if source.Agents != nil {
		result.Agents = make(map[string]ResolvedExecutionSelectionOverride, len(source.Agents))
		for name, selection := range source.Agents {
			result.Agents[name] = cloneExecutionSelectionOverride(selection)
		}
	}
	return result
}

func cloneExecutionSelectionOverride(
	source ResolvedExecutionSelectionOverride,
) ResolvedExecutionSelectionOverride {
	result := source
	if source.ModelPolicy != nil {
		policy := source.ModelPolicy.Clone()
		result.ModelPolicy = &policy
	}
	if source.LLMGateway != nil {
		gateway := cloneLLMGatewayConfig(*source.LLMGateway)
		result.LLMGateway = &gateway
	}
	if source.Credential != nil {
		credential := *source.Credential
		if source.Credential.Ref != nil {
			ref := *source.Credential.Ref
			credential.Ref = &ref
		}
		result.Credential = &credential
	}
	return result
}

func cloneTransition(source TransitionAction) TransitionAction {
	result := source
	if source.Retry != nil {
		result.Retry = &RetryTransition{
			MaxAttempts: source.Retry.MaxAttempts,
			Then:        cloneTransition(source.Retry.Then),
		}
	}
	if source.Escalate != nil {
		result.Escalate = &EscalateTransition{
			MaxAttempts:     source.Escalate.MaxAttempts,
			ExecutionConfig: cloneEscalationExecutionConfig(source.Escalate.ExecutionConfig),
			Then:            cloneTransition(source.Escalate.Then),
		}
	}
	return result
}

func cloneEscalationExecutionConfig(
	source ResolvedEscalationExecutionConfig,
) ResolvedEscalationExecutionConfig {
	result := source
	if source.Ref != nil {
		ref := *source.Ref
		result.Ref = &ref
	}
	result.Override = cloneStageExecutionConfigOverride(source.Override)
	result.Effective = cloneStageExecutionConfig(source.Effective)
	return result
}
