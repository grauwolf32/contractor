// Package configloadtest reads resolved Workflow bindings for cross-package tests.
package configloadtest

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// AgentTemplate returns a detached template from a published Workflow binding.
// Unbound templates are deliberately unavailable here. Core loader tests inspect
// them in-package; execution tests use the same resolved bindings as production.
func AgentTemplate(snapshot interface {
	Workflows() []resolved.ResolvedWorkflow
}, raw string) (contracts.ResolvedAgentTemplate, error) {
	selector, err := resolved.ParseSelector(raw)
	if err != nil {
		return contracts.ResolvedAgentTemplate{}, err
	}
	for _, workflow := range snapshot.Workflows() {
		for _, stage := range workflow.Stages {
			for _, binding := range stage.Agents {
				ref := binding.Template.Ref
				if ref.TemplateID == selector.ID && ref.Version == selector.Version {
					return binding.Template, nil
				}
			}
		}
	}
	return contracts.ResolvedAgentTemplate{}, fmt.Errorf("AgentTemplate %q has no Workflow binding", raw)
}
