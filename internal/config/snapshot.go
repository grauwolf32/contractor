package config

import (
	"fmt"
	"sort"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// Snapshot is an immutable, dependency-resolved view of one successful load.
// Its maps are private and every accessor returns a deep copy.
type Snapshot struct {
	workflows    map[string]ResolvedWorkflow
	templates    map[string]contracts.ResolvedAgentTemplate
	policies     map[string]contracts.ResolvedModelPolicy
	gateways     map[string]contracts.ResolvedLLMGatewayConfig
	instructions map[string]contracts.ResolvedInstructions
}

func newSnapshot(
	workflows map[string]ResolvedWorkflow,
	templates map[string]contracts.ResolvedAgentTemplate,
	policies map[string]contracts.ResolvedModelPolicy,
	gateways map[string]contracts.ResolvedLLMGatewayConfig,
	instructions map[string]contracts.ResolvedInstructions,
) *Snapshot {
	result := &Snapshot{
		workflows:    make(map[string]ResolvedWorkflow, len(workflows)),
		templates:    make(map[string]contracts.ResolvedAgentTemplate, len(templates)),
		policies:     make(map[string]contracts.ResolvedModelPolicy, len(policies)),
		gateways:     make(map[string]contracts.ResolvedLLMGatewayConfig, len(gateways)),
		instructions: make(map[string]contracts.ResolvedInstructions, len(instructions)),
	}
	for key, workflow := range workflows {
		result.workflows[key] = cloneWorkflow(workflow)
	}
	for key, template := range templates {
		result.templates[key] = cloneAgentTemplate(template)
	}
	for key, policy := range policies {
		result.policies[key] = cloneModelPolicy(policy)
	}
	for key, gateway := range gateways {
		result.gateways[key] = cloneLLMGatewayConfig(gateway)
	}
	for key, instructions := range instructions {
		result.instructions[key] = instructions
	}
	return result
}

func (s *Snapshot) Counts() Counts {
	return Counts{
		Workflows: len(s.workflows), AgentTemplates: len(s.templates),
		ModelPolicies: len(s.policies), LLMGateways: len(s.gateways),
		Instructions: len(s.instructions),
	}
}

// LLMGateway resolves one exact id@version and returns a caller-owned non-secret copy.
func (s *Snapshot) LLMGateway(raw string) (contracts.ResolvedLLMGatewayConfig, error) {
	selector, err := ParseSelector(raw)
	if err != nil {
		return contracts.ResolvedLLMGatewayConfig{}, err
	}
	gateway, ok := s.gateways[selector.String()]
	if !ok {
		return contracts.ResolvedLLMGatewayConfig{}, fmt.Errorf("unknown LLMGatewayConfig %q", selector)
	}
	return cloneLLMGatewayConfig(gateway), nil
}

// LLMGateways returns every published non-secret Gateway body sorted by exact ref.
func (s *Snapshot) LLMGateways() []contracts.ResolvedLLMGatewayConfig {
	keys := make([]string, 0, len(s.gateways))
	for key := range s.gateways {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := make([]contracts.ResolvedLLMGatewayConfig, 0, len(keys))
	for _, key := range keys {
		result = append(result, cloneLLMGatewayConfig(s.gateways[key]))
	}
	return result
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

// AgentTemplate resolves one exact id@version and returns a caller-owned copy.
func (s *Snapshot) AgentTemplate(raw string) (contracts.ResolvedAgentTemplate, error) {
	selector, err := ParseSelector(raw)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, err
	}
	template, ok := s.templates[selector.String()]
	if !ok {
		return contracts.ResolvedAgentTemplate{}, fmt.Errorf("unknown AgentTemplate %q", selector)
	}
	return cloneAgentTemplate(template), nil
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
	return cloneModelPolicy(policy), nil
}

// Instructions returns the exact text dependency pinned by a normalized ref.
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

func cloneModelPolicy(source contracts.ResolvedModelPolicy) contracts.ResolvedModelPolicy {
	result := source
	result.Temperature = cloneFloat(source.Temperature)
	return result
}

func cloneLLMGatewayConfig(
	source contracts.ResolvedLLMGatewayConfig,
) contracts.ResolvedLLMGatewayConfig {
	result := source
	if source.CredentialManager != nil {
		manager := *source.CredentialManager
		result.CredentialManager = &manager
	}
	return result
}

func cloneAgentTemplate(source contracts.ResolvedAgentTemplate) contracts.ResolvedAgentTemplate {
	result := source
	result.ModelPolicy = cloneModelPolicy(source.ModelPolicy)
	result.Toolsets = make([]contracts.ToolsetSelection, len(source.Toolsets))
	for index, toolset := range source.Toolsets {
		result.Toolsets[index] = toolset
		result.Toolsets[index].Tools = append([]string(nil), toolset.Tools...)
	}
	return result
}

func cloneWorkflow(source ResolvedWorkflow) ResolvedWorkflow {
	result := source
	result.Parameters = make(map[string]ParameterSlot, len(source.Parameters))
	for name, slot := range source.Parameters {
		result.Parameters[name] = slot
	}
	result.Inputs = cloneArtifactSlots(source.Inputs)
	result.Outputs = cloneArtifactSlots(source.Outputs)
	result.Stages = make(map[string]ResolvedStage, len(source.Stages))
	for name, stage := range source.Stages {
		result.Stages[name] = cloneStage(stage)
	}
	return result
}

func cloneArtifactSlots(source map[string]ArtifactSlot) map[string]ArtifactSlot {
	result := make(map[string]ArtifactSlot, len(source))
	for name, slot := range source {
		slot.MediaTypes = append([]string(nil), slot.MediaTypes...)
		result[name] = slot
	}
	return result
}

func cloneStage(source ResolvedStage) ResolvedStage {
	result := source
	result.Agents = make(map[string]ResolvedAgentBinding, len(source.Agents))
	for name, binding := range source.Agents {
		binding.Template = cloneAgentTemplate(binding.Template)
		result.Agents[name] = binding
	}
	result.Context.Artifacts = make(map[string]ContextArtifact, len(source.Context.Artifacts))
	for name, artifact := range source.Context.Artifacts {
		result.Context.Artifacts[name] = artifact
	}
	result.Result.Artifacts = cloneArtifactSlots(source.Result.Artifacts)
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

func cloneTransition(source TransitionAction) TransitionAction {
	result := source
	if source.Retry != nil {
		result.Retry = &RetryTransition{
			MaxAttempts: source.Retry.MaxAttempts,
			Then:        cloneTransition(source.Retry.Then),
		}
	}
	return result
}
