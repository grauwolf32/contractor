package config

import "fmt"

// Inventory artifacts and Workflow inputs use the same explicit reference.
// Only Audit inputs and accepted Audit-scoped prepare outputs exist before a Round.
func resolveAuditInventoryArtifact(
	field string, source *auditWorkflowInputMappingSource,
	inputs map[string]AuditProfileInput, workflows map[string]ResolvedAuditWorkflowBinding,
	mediaTypes []string,
) (*AuditWorkflowInputMapping, error) {
	if source == nil {
		return nil, fmt.Errorf("%s is required", field)
	}
	mapping, err := resolveAuditWorkflowInputMapping(field, *source, ArtifactSlot{Required: true, MediaTypes: mediaTypes}, inputs)
	if err != nil {
		return nil, err
	}
	switch mapping.Source {
	case AuditInputFromAudit:
	case AuditInputFromPreparation:
		producer, exists := workflows[mapping.Role]
		if !exists || producer.Kind != AuditWorkflowPrepare {
			return nil, fmt.Errorf("%s requires a prepare producer", field)
		}
		output, exists := producer.Outputs[mapping.Name]
		slot, declared := producer.Workflow.Outputs[output]
		if !exists || !declared || !slot.Required || !mediaTypesIntersect(slot.MediaTypes, mediaTypes) {
			return nil, fmt.Errorf("%s requires a declared, required output with compatible media types", field)
		}
	default:
		return nil, fmt.Errorf("%s must use audit-input or prepare-output", field)
	}
	return &mapping, nil
}

func cloneAuditInputMapping(source *AuditWorkflowInputMapping) *AuditWorkflowInputMapping {
	if source == nil {
		return nil
	}
	value := *source
	return &value
}

func authoredAuditInput(mapping *AuditWorkflowInputMapping) *auditWorkflowInputMappingSource {
	if mapping == nil {
		return nil
	}
	return &auditWorkflowInputMappingSource{Source: string(mapping.Source), Name: mapping.Name, Role: mapping.Role}
}

func (profile ResolvedAuditProfile) HasPreparation() bool {
	for _, binding := range profile.Workflows {
		if binding.Kind == AuditWorkflowPrepare {
			return true
		}
	}
	return false
}

// ValidateAuditPreparationProfile also runs when decoding pinned snapshots.
// Validation establishes the contract; it does not enable prepare dispatch.
func ValidateAuditPreparationProfile(profile ResolvedAuditProfile) error {
	if len(profile.Workflows) == 0 || len(profile.Workflows) > MaxAuditProfileWorkflows || len(profile.Inputs) == 0 || len(profile.Inputs) > MaxAuditProfileInputs {
		return fmt.Errorf("Audit profile role/input count exceeds bounds")
	}
	descriptor, exists := auditInventories[profile.Inventory.Implementation]
	if !exists {
		return fmt.Errorf("inventory implementation is unsupported")
	}
	if len(descriptor.mediaTypes) != 0 {
		if _, err := resolveAuditInventoryArtifact("inventory.source", authoredAuditInput(profile.Inventory.Source), profile.Inputs, profile.Workflows, descriptor.mediaTypes); err != nil {
			return err
		}
	} else if profile.Inventory.Source != nil {
		return fmt.Errorf("standard inventory forbids source")
	}
	if descriptor.requiresSettings {
		if _, err := resolveAuditInventoryArtifact("inventory.settings", authoredAuditInput(profile.Inventory.Settings), profile.Inputs, profile.Workflows, []string{"application/json"}); err != nil {
			return err
		}
		if *profile.Inventory.Source == *profile.Inventory.Settings {
			return fmt.Errorf("inventory source and settings must be distinct")
		}
	} else if profile.Inventory.Settings != nil {
		return fmt.Errorf("inventory does not accept settings")
	}
	for _, role := range sortedMapKeys(profile.Workflows) {
		binding := profile.Workflows[role]
		if len(binding.Inputs) > MaxAuditWorkflowMappings || len(binding.Parameters) > MaxAuditWorkflowMappings || len(binding.Outputs) == 0 || len(binding.Outputs) > MaxAuditWorkflowMappings {
			return fmt.Errorf("workflows.%s mappings exceed bounds or have no output", role)
		}
		if _, err := resolveAuditWorkflowOutputs(role, &binding.Outputs, binding.Workflow); err != nil {
			return err
		}
		for name, slot := range binding.Workflow.Inputs {
			if _, exists := binding.Inputs[name]; slot.Required && !exists {
				return fmt.Errorf("workflows.%s missing required input %s", role, name)
			}
		}
		if binding.Kind == AuditWorkflowPrepare {
			// The existing Run-attempt ceiling applies independently to each
			// prepare role; the profile's cumulative Run budget still applies.
			if binding.MaxRunAttempts < 1 || binding.MaxRunAttempts > MaxAuditRunAttempts || binding.MaxRunAttempts > profile.Execution.MaxSubmittedRuns {
				return fmt.Errorf("workflows.%s.maxRunAttempts must be 1..%d and fit maxSubmittedRuns", role, MaxAuditRunAttempts)
			}
			if binding.WorkerCompletion != nil || binding.Workflow.AuditTask != nil {
				return fmt.Errorf("workflows.%s prepare role cannot execute item tasks", role)
			}
			for _, mapping := range binding.Parameters {
				if mapping.Source == AuditParameterItemField {
					return fmt.Errorf("workflows.%s prepare role cannot use item-field", role)
				}
			}
		} else if binding.MaxRunAttempts != 0 {
			return fmt.Errorf("workflows.%s.maxRunAttempts is only valid for prepare roles", role)
		}
		for name, mapping := range binding.Inputs {
			if _, exists := binding.Workflow.Inputs[name]; !exists {
				return fmt.Errorf("workflows.%s names unknown input %s", role, name)
			}
			if binding.Kind == AuditWorkflowPrepare && mapping.Source != AuditInputFromAudit && mapping.Source != AuditInputFromPreparation {
				return fmt.Errorf("workflows.%s.inputs.%s prepare role requires audit-input or prepare-output", role, name)
			}
			if _, err := resolveAuditWorkflowInputMapping("workflows."+role+".inputs."+name, *authoredAuditInput(&mapping), binding.Workflow.Inputs[name], profile.Inputs); err != nil {
				return err
			}
		}
	}
	return validateAuditWorkflowDependencies(profile.Workflows)
}
