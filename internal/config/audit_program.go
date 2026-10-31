package config

import (
	"github.com/grauwolf32/contractor/internal/apierror"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/strictjson"
)

// ValidateAuditProgram validates a common Audit program against this exact
// loaded Workflow catalog. It creates no runnable profile, grant or Run.
func ValidateAuditProgram(s *Snapshot, data []byte) error {
	canonical, err := auditcontract.Decode(data, nil)
	if err != nil {
		return err
	}
	var program map[string]any
	if err = strictjson.Decode(canonical, &program); err != nil {
		return err
	}
	if program["schema"] != "contractor.audit.program.v1" {
		return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: expected program document")
	}
	inputs := program["inputs"].(map[string]any)
	for _, raw := range program["workflows"].(map[string]any) {
		binding := raw.(map[string]any)
		workflow, err := s.Workflow(binding["ref"].(string))
		if err != nil {
			return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: Workflow selection is unavailable")
		}
		mappings := binding["inputs"].(map[string]any)
		for name, slot := range workflow.Inputs {
			if _, ok := mappings[name]; !ok && slot.Required {
				return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: required Workflow input is missing")
			}
		}
		for name, raw := range mappings {
			slot, ok := workflow.Inputs[name]
			if !ok {
				return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: unknown Workflow input")
			}
			source := raw.(map[string]any)
			if source["source"] != "audit-input" {
				if !contracts.AcceptsMediaType(slot.MediaTypes, "application/json") {
					return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: context input must accept JSON")
				}
				continue
			}
			input := inputs[source["name"].(string)].(map[string]any)
			if slot.Required && !input["required"].(bool) {
				return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: required Workflow input uses optional material")
			}
			for _, media := range input["media_types"].([]any) {
				if !contracts.AcceptsMediaType(slot.MediaTypes, media.(string)) {
					return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: incompatible input media type")
				}
			}
		}
		parameters := binding["parameters"].(map[string]any)
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
		for logical, raw := range binding["outputs"].(map[string]any) {
			slot, ok := workflow.Outputs[raw.(string)]
			if !ok || !slot.Required || logical == "result" && !contracts.AcceptsMediaType(slot.MediaTypes, "application/json") {
				return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: output must name a required Workflow output")
			}
		}
	}
	return nil
}
