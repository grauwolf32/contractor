package config

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// Instructions returns the exact text dependency pinned by a normalized ref,
// so conformance tests can verify packaged instruction sources; production
// execution reads the text pinned into resolved Workflows.
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

// Core catalog tests also inspect unbound templates. Cross-package tests read
// templates through published Workflow bindings with configloadtest.AgentTemplate.
func (s *Snapshot) agentTemplateForTest(raw string) (contracts.ResolvedAgentTemplate, error) {
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
