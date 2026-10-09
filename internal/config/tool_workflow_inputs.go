package config

import (
	"fmt"
	"maps"
	"slices"
	"sync"
)

// registeredToolsets is the Toolset descriptor table that Workflow graph
// validation consults. Persisted and Run-pinned Workflows are revalidated
// without a loader, so the check reads the registered table, as
// RequiredRuntimeAdaptersForTemplate does, rather than a per-load copy.
var registeredToolsets = sync.OnceValue(func() map[string]ToolsetDescriptor {
	return MVPDescriptors().Toolsets
})

// validateToolWorkflowInputs enforces each selected tool's declared Workflow
// input requirement. The runtime resolves that ordinary Workflow input once
// during allocation preparation, so it must be required and accept the
// declared media type. Tools without a requirement add nothing.
func validateToolWorkflowInputs(workflow ResolvedWorkflow, toolsets map[string]ToolsetDescriptor) error {
	for _, stageName := range slices.Sorted(maps.Keys(workflow.Stages)) {
		stage := workflow.Stages[stageName]
		for _, agentName := range slices.Sorted(maps.Keys(stage.Agents)) {
			for _, selected := range stage.Agents[agentName].Template.Toolsets {
				selector := selected.Ref.ToolsetID + "@" + selected.Ref.Version
				requirements := toolsets[selector].RequiredWorkflowInputs
				for _, tool := range selected.Tools {
					requirement, ok := requirements[tool]
					if !ok {
						continue
					}
					input, declared := workflow.Inputs[requirement.Input]
					if !declared || !input.Required || !slices.Contains(input.MediaTypes, requirement.MediaType) {
						return fmt.Errorf(
							"Stage %q Agent %q %s %s requires the required Workflow input %s with media type %s",
							stageName, agentName, selector, tool, requirement.Input, requirement.MediaType,
						)
					}
				}
			}
		}
	}
	return nil
}
