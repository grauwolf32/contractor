package resolved

import (
	"fmt"
	"maps"
	"slices"
)

// validateToolWorkflowInputs enforces each selected tool's declared Workflow
// input requirement. The runtime resolves that ordinary Workflow input once
// during allocation preparation, so it must be required and accept the
// declared media type. Tools without a requirement add nothing. Persisted and
// Run-pinned Workflows are checked against the same registry as a catalog
// load; toolsets is that registry's table, or a test's table for the rule.
func validateToolWorkflowInputs(workflow ResolvedWorkflow, toolsets map[string]toolsetDescriptor) error {
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
