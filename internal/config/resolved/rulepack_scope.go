package resolved

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
)

func (s *ArtifactSlot) UnmarshalJSON(data []byte) error {
	type plain ArtifactSlot
	var value plain
	fields, err := contracts.DecodeStrictObject(data, &value)
	if err != nil {
		return err
	}
	if fields["sourceScope"] != nil && value.SourceScope != "user" {
		return fmt.Errorf("sourceScope must select user")
	}
	*s = ArtifactSlot(value)
	return nil
}

// validateRulepackSourceScopes keeps the UserScope exception narrow at every
// authored and retained Workflow boundary. Runtime never receives UserScope authority.
func validateRulepackSourceScopes(workflow ResolvedWorkflow) error {
	for _, slot := range workflow.Outputs {
		if slot.SourceScope != "" {
			return fmt.Errorf("outputs cannot select sourceScope")
		}
	}
	for _, stage := range workflow.Stages {
		for _, slot := range stage.Result.Artifacts {
			if slot.SourceScope != "" {
				return fmt.Errorf("results cannot select sourceScope")
			}
		}
	}
	for name, slot := range workflow.Inputs {
		if slot.SourceScope == "" {
			continue
		}
		if slot.SourceScope != "user" || name != "rulepack" || slot.Required || len(slot.MediaTypes) == 0 || len(slot.MediaTypes) > 2 {
			return fmt.Errorf("sourceScope user is reserved for optional scanner rulepacks")
		}
		for _, media := range slot.MediaTypes {
			if media != "application/yaml" && media != scan.NucleiRulepackMediaType {
				return fmt.Errorf("rulepack sourceScope has unsupported media type")
			}
		}
		consumer := false
		for _, stage := range workflow.Stages {
			for _, agent := range stage.Agents {
				execution := agent.Template.Execution
				if !agent.Template.IsToolWorker() || execution == nil || execution.Tool != "scan_nuclei" {
					continue
				}
				argument, ok := execution.Arguments["rulepack_ref"]
				context, present := stage.Context.Artifacts[argument.Name]
				if ok && present && argument.Source == "artifact" && argument.Optional && !context.Required && context.Namespace == "inputs" && context.Name == name {
					consumer = true
				}
			}
		}
		if !consumer {
			return fmt.Errorf("UserScope rulepack requires an optional Nuclei input binding")
		}
	}
	return nil
}
