package config

import (
	"fmt"
	"slices"

	"github.com/grauwolf32/contractor/internal/auditdomain"
)

// The reader resolves this ordinary Workflow input once during allocation
// preparation. Writer-only selections have no collection input requirement.
func validateFindingsReaderInput(workflow ResolvedWorkflow) error {
	for stageName, stage := range workflow.Stages {
		for agentName, agent := range stage.Agents {
			for _, selected := range agent.Template.Toolsets {
				if selected.Ref.ToolsetID != "security-findings" || selected.Ref.Version != "2" ||
					!slices.Contains(selected.Tools, "list_findings") {
					continue
				}
				input, ok := workflow.Inputs["findings"]
				if !ok || !input.Required || !slices.Contains(input.MediaTypes, auditdomain.FindingCollectionMediaType) {
					return fmt.Errorf("Stage %q Agent %q security-findings@2 list_findings requires the required Workflow input findings with media type %s", stageName, agentName, auditdomain.FindingCollectionMediaType)
				}
			}
		}
	}
	return nil
}
