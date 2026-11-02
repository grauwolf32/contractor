package auditprofile

import (
	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// ValidateTaskInputs checks explicit selections against the pinned binding.
// Artifact access, full metadata membership and retention belong to the caller.
// Required Audit defaults were already checked by ResolveProgramWorkflows;
// system-context inputs are assembled by the trusted Run-creation path.
func ValidateTaskInputs(binding auditcontract.ProgramWorkflow, workflow resolved.ResolvedWorkflow, inputs map[string]auditcontract.Material) error {
	for name, slot := range workflow.Inputs {
		if slot.Required && binding.Inputs[name].Source == "task-input" {
			if _, selected := inputs[name]; !selected {
				return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: required task material is missing")
			}
		}
	}
	for name, selected := range inputs {
		mapping, declared := binding.Inputs[name]
		slot, exists := workflow.Inputs[name]
		if !declared || !exists || mapping.Source != "task-input" && mapping.Source != "audit-input" || !contracts.AcceptsMediaType(slot.MediaTypes, selected.MediaType) {
			return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: incompatible task material")
		}
	}
	return nil
}
