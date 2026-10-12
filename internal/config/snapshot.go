package config

import (
	"fmt"
	"maps"
	"slices"
	"sort"

	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/llmgateway"
)

// Snapshot is an immutable, dependency-resolved view of one successful load.
// Its maps are private and every accessor returns a deep copy.
type Snapshot struct {
	workflowBindings map[string]AgentTemplateWorkflowBindings
	workflows        map[string]resolved.ResolvedWorkflow
	templates        map[string]contracts.ResolvedAgentTemplate
	policies         map[string]contracts.ResolvedModelPolicy
	gateways         map[string]llmgateway.ResolvedLLMGatewayConfig
	executionConfigs map[string]resolved.ResolvedExecutionConfigProfile
	auditProfiles    map[string]auditprofile.ResolvedAuditProfile
	instructions     map[string]contracts.ResolvedInstructions
	sources          map[string]ConfigurationSource

	// Discovery views and fingerprints are derived once in newSnapshot.
	workflowDiscovery         WorkflowDiscovery
	workflowFingerprint       catalogFingerprint
	configurationFingerprints map[ConfigurationKind]catalogFingerprint
}

func newSnapshot(
	workflows map[string]resolved.ResolvedWorkflow,
	templates map[string]contracts.ResolvedAgentTemplate,
	policies map[string]contracts.ResolvedModelPolicy,
	gateways map[string]llmgateway.ResolvedLLMGatewayConfig,
	executionConfigs map[string]resolved.ResolvedExecutionConfigProfile,
	auditProfiles map[string]auditprofile.ResolvedAuditProfile,
	instructions map[string]contracts.ResolvedInstructions,
	sources map[string]ConfigurationSource,
) *Snapshot {
	result := &Snapshot{
		workflows:        make(map[string]resolved.ResolvedWorkflow, len(workflows)),
		templates:        make(map[string]contracts.ResolvedAgentTemplate, len(templates)),
		policies:         make(map[string]contracts.ResolvedModelPolicy, len(policies)),
		gateways:         make(map[string]llmgateway.ResolvedLLMGatewayConfig, len(gateways)),
		executionConfigs: make(map[string]resolved.ResolvedExecutionConfigProfile, len(executionConfigs)),
		auditProfiles:    make(map[string]auditprofile.ResolvedAuditProfile, len(auditProfiles)),
		instructions:     make(map[string]contracts.ResolvedInstructions, len(instructions)),
		sources:          make(map[string]ConfigurationSource, len(sources)),
	}
	for key, workflow := range workflows {
		result.workflows[key] = workflow.Clone()
	}
	for key, template := range templates {
		result.templates[key] = template.Clone()
	}
	for key, policy := range policies {
		result.policies[key] = policy.Clone()
	}
	for key, gateway := range gateways {
		result.gateways[key] = gateway.Clone()
	}
	for key, profile := range executionConfigs {
		result.executionConfigs[key] = profile.Clone()
	}
	for key, profile := range auditProfiles {
		result.auditProfiles[key] = profile.Clone()
	}
	for key, instructions := range instructions {
		result.instructions[key] = instructions
	}
	for key, source := range sources {
		result.sources[key] = source
	}
	result.buildWorkflowBindingIndex(result.buildDiscovery())
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
func (s *Snapshot) AuditProfile(raw string) (auditprofile.ResolvedAuditProfile, error) {
	selector, err := resolved.ParseSelector(raw)
	if err != nil {
		return auditprofile.ResolvedAuditProfile{}, err
	}
	profile, ok := s.auditProfiles[selector.String()]
	if !ok {
		return auditprofile.ResolvedAuditProfile{}, fmt.Errorf("unknown AuditProfile %q", selector)
	}
	return profile.Clone(), nil
}

// AuditProfiles returns every profile sorted by exact ref and deeply detached
// from the immutable Snapshot.
func (s *Snapshot) AuditProfiles() []auditprofile.ResolvedAuditProfile {
	keys := slices.Sorted(maps.Keys(s.auditProfiles))
	result := make([]auditprofile.ResolvedAuditProfile, 0, len(keys))
	for _, key := range keys {
		result = append(result, s.auditProfiles[key].Clone())
	}
	return result
}

// LLMGateway resolves one exact id@version and returns a caller-owned non-secret copy.
func (s *Snapshot) LLMGateway(raw string) (llmgateway.ResolvedLLMGatewayConfig, error) {
	selector, err := resolved.ParseSelector(raw)
	if err != nil {
		return llmgateway.ResolvedLLMGatewayConfig{}, err
	}
	gateway, ok := s.gateways[selector.String()]
	if !ok {
		return llmgateway.ResolvedLLMGatewayConfig{}, fmt.Errorf("unknown LLMGatewayConfig %q", selector)
	}
	return gateway.Clone(), nil
}

// Workflow resolves one exact name@version and returns a caller-owned copy.
func (s *Snapshot) Workflow(raw string) (resolved.ResolvedWorkflow, error) {
	selector, err := resolved.ParseSelector(raw)
	if err != nil {
		return resolved.ResolvedWorkflow{}, err
	}
	workflow, ok := s.workflows[selector.String()]
	if !ok {
		return resolved.ResolvedWorkflow{}, fmt.Errorf("unknown Workflow %q", selector)
	}
	return workflow.Clone(), nil
}

// Workflows returns every published Workflow sorted by exact ref. Every value
// is a caller-owned deep copy; mutating the result cannot change the published
// configuration snapshot.
func (s *Snapshot) Workflows() []resolved.ResolvedWorkflow {
	keys := make([]string, 0, len(s.workflows))
	for key := range s.workflows {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	result := make([]resolved.ResolvedWorkflow, 0, len(keys))
	for _, key := range keys {
		result = append(result, s.workflows[key].Clone())
	}
	return result
}

// AgentTemplate resolves one exact id@version and returns a caller-owned copy.
// Cross-package conformance tests use it to inspect pinned template bodies;
// production execution reads templates through resolved Workflows.
func (s *Snapshot) AgentTemplate(raw string) (contracts.ResolvedAgentTemplate, error) {
	selector, err := resolved.ParseSelector(raw)
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
	selector, err := resolved.ParseSelector(raw)
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
