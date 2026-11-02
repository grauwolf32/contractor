package config

import (
	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// ValidateAuditProgram validates a common Audit program against this exact
// loaded Workflow catalog. It creates no runnable profile, grant or Run.
func ValidateAuditProgram(s *Snapshot, data []byte) error {
	program, err := auditcontract.DecodeProgram(data)
	if err != nil {
		return err
	}
	inputs := program.Inputs
	workflowInputs := map[string]map[string]resolved.ArtifactSlot{}
	for alias, binding := range program.Workflows {
		workflow, err := s.Workflow(binding.Ref)
		if err != nil {
			return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: Workflow selection is unavailable")
		}
		workflowInputs[alias] = workflow.Inputs
		mappings := binding.Inputs
		for name, slot := range workflow.Inputs {
			if _, ok := mappings[name]; !ok && slot.Required {
				return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: required Workflow input is missing")
			}
		}
		for name, source := range mappings {
			slot, ok := workflow.Inputs[name]
			if !ok {
				return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: unknown Workflow input")
			}
			if source.Source != "audit-input" {
				if !contracts.AcceptsMediaType(slot.MediaTypes, "application/json") {
					return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: context input must accept JSON")
				}
				continue
			}
			input := inputs[source.Name]
			if err := validateAuditMaterialInput(slot, input); err != nil {
				return err
			}
		}
		parameters := binding.Parameters
		for name, slot := range workflow.Parameters {
			if _, ok := parameters[name]; !ok && slot.Required {
				return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: required Workflow parameter is missing")
			}
		}
		for name := range parameters {
			if _, ok := workflow.Parameters[name]; !ok {
				return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: unknown Workflow parameter")
			}
		}
		for logical, output := range binding.Outputs {
			slot, ok := workflow.Outputs[output]
			if !ok || !slot.Required || logical == "result" && !contracts.AcceptsMediaType(slot.MediaTypes, "application/json") {
				return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: output must name a required Workflow output")
			}
		}
	}
	for _, task := range program.InitialWork {
		for name, selected := range task.Inputs {
			// Decode already checked binding and material membership. The
			// loaded Workflow owns requiredness of the effective input slot.
			input := inputs[selected]
			if err := validateAuditMaterialInput(workflowInputs[task.Workflow][name], input); err != nil {
				return err
			}
		}
	}
	return nil
}

func validateAuditMaterialInput(slot resolved.ArtifactSlot, input auditcontract.ProgramInput) error {
	if slot.Required && !input.Required {
		return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: required Workflow input uses optional material")
	}
	for _, media := range input.MediaTypes {
		if !contracts.AcceptsMediaType(slot.MediaTypes, media) {
			return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: incompatible input media type")
		}
	}
	return nil
}
