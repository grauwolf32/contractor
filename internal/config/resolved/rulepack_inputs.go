package resolved

import "slices"

// NucleiRulepackInputs returns the Workflow input slots consumed by explicit
// scan_nuclei rulepack arguments, through their Stage context bindings. The
// caller owns source scope, exact identity and metadata admission.
func NucleiRulepackInputs(workflow ResolvedWorkflow) []string {
	names := make(map[string]bool)
	for _, stage := range workflow.Stages {
		for _, agent := range stage.Agents {
			execution := agent.Template.Execution
			if !agent.Template.IsToolWorker() || execution == nil || execution.Tool != "scan_nuclei" {
				continue
			}
			argument, ok := execution.Arguments["rulepack_ref"]
			if !ok || argument.Source != "artifact" {
				continue
			}
			context, ok := stage.Context.Artifacts[argument.Name]
			if ok && context.Namespace == "inputs" {
				names[context.Name] = true
			}
		}
	}
	result := make([]string, 0, len(names))
	for name := range names {
		result = append(result, name)
	}
	slices.Sort(result)
	return result
}
