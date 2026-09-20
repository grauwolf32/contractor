package config

import (
	"fmt"
	"sort"
)

func validateAuditWorkflowDependencies(
	workflows map[string]ResolvedAuditWorkflowBinding,
) error {
	dependencies := make(map[string][]string, len(workflows))
	for _, role := range sortedMapKeys(workflows) {
		binding := workflows[role]
		for _, inputName := range sortedMapKeys(binding.Inputs) {
			mapping := binding.Inputs[inputName]
			if mapping.Source != AuditInputFromRetainedOutput && mapping.Source != AuditInputFromPreparation {
				continue
			}
			source, exists := workflows[mapping.Role]
			if !exists {
				return fmt.Errorf("spec.workflows.%s.inputs.%s names unknown output producer role %q", role, inputName, mapping.Role)
			}
			if mapping.Source == AuditInputFromPreparation && source.Kind != AuditWorkflowPrepare {
				return fmt.Errorf("spec.workflows.%s.inputs.%s prepare-output requires a prepare producer", role, inputName)
			}
			if mapping.Source == AuditInputFromRetainedOutput && (source.Kind == AuditWorkflowPrepare || binding.Kind == AuditWorkflowPrepare) {
				return fmt.Errorf("spec.workflows.%s.inputs.%s retained-output is confined to Round roles", role, inputName)
			}
			workflowOutput, exists := source.Outputs[mapping.Name]
			if !exists {
				return fmt.Errorf("spec.workflows.%s.inputs.%s names unknown logical output %q on role %q", role, inputName, mapping.Name, mapping.Role)
			}
			if !mediaTypesIntersect(source.Workflow.Outputs[workflowOutput].MediaTypes, binding.Workflow.Inputs[inputName].MediaTypes) {
				return fmt.Errorf("spec.workflows.%s.inputs.%s retained output media types are incompatible", role, inputName)
			}
			dependencies[role] = append(dependencies[role], mapping.Role)
		}
	}
	const (
		unseen uint8 = iota
		visiting
		visited
	)
	state := make(map[string]uint8, len(workflows))
	var visit func(string) error
	visit = func(role string) error {
		switch state[role] {
		case visiting:
			return fmt.Errorf("spec.workflows output dependencies contain a cycle at role %q", role)
		case visited:
			return nil
		}
		state[role] = visiting
		sort.Strings(dependencies[role])
		for _, dependency := range dependencies[role] {
			if err := visit(dependency); err != nil {
				return err
			}
		}
		state[role] = visited
		return nil
	}
	for _, role := range sortedMapKeys(workflows) {
		if err := visit(role); err != nil {
			return err
		}
	}
	phase := map[AuditWorkflowRoleKind]int{
		AuditWorkflowPrepare:    -1,
		AuditWorkflowDiscovery:  0,
		AuditWorkflowCheck:      1,
		AuditWorkflowAssessment: 2,
	}
	for _, role := range sortedMapKeys(workflows) {
		for _, dependency := range dependencies[role] {
			source, destination := workflows[dependency], workflows[role]
			if source.Kind == AuditWorkflowCheck {
				return fmt.Errorf(
					"spec.workflows.%s cannot consume retained output from check role %q",
					role, dependency,
				)
			}
			if phase[source.Kind] > phase[destination.Kind] {
				return fmt.Errorf(
					"spec.workflows.%s retained-output dependency %q belongs to a later execution phase",
					role, dependency,
				)
			}
		}
	}
	return nil
}
