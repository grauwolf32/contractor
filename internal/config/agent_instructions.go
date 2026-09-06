package config

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// AgentInstructions exposes only the base instructions of an exact template.
// It contains no per-invocation context or resolved deployment credentials.
type AgentInstructions struct {
	Template     contracts.AgentTemplateRef     `json:"template"`
	Instructions contracts.ResolvedInstructions `json:"instructions"`
}

func (s *Snapshot) AgentInstructions(raw string) (AgentInstructions, error) {
	selector, err := ParseSelector(raw)
	if err != nil {
		return AgentInstructions{}, err
	}
	template, ok := s.templates[selector.String()]
	if !ok {
		return AgentInstructions{}, fmt.Errorf("%w: AgentTemplate %s", ErrConfigurationNotFound, selector)
	}
	return AgentInstructions{Template: template.Ref, Instructions: template.Instructions}, nil
}

func (m *Manager) AgentInstructions(raw string) (AgentInstructions, error) {
	return m.Snapshot().AgentInstructions(raw)
}
