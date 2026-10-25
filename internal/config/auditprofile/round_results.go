package auditprofile

import "fmt"

// ValidateRoundResultsBinding is also called while decoding a pinned profile;
// catalog identity never grants an undeclared or cross-phase source.
func ValidateRoundResultsBinding(binding ResolvedAuditWorkflowBinding) error {
	count := 0
	for name, mapping := range binding.Inputs {
		if mapping.Source != AuditInputFromRoundResults {
			continue
		}
		count++
		if binding.Kind != AuditWorkflowAssessment || !binding.Workflow.Inputs[name].Required {
			return fmt.Errorf("round-results requires a required assessment input")
		}
	}
	if count > 1 {
		return fmt.Errorf("one assessment binding may declare at most one round-results input")
	}
	return nil
}
