package auditprofile

import (
	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
)

// ResolveProgramWorkflows applies actual catalog slot rules and returns the
// exact allowed Workflow closures. Offline validation and accepted execution
// share this rule; a resolved closure alone grants no execution authority.
func ResolveProgramWorkflows(program auditcontract.Program, lookup func(string) (resolved.ResolvedWorkflow, error)) (map[string]resolved.ResolvedWorkflow, error) {
	inputs := program.Inputs
	workflows := map[string]resolved.ResolvedWorkflow{}
	workflowInputs := map[string]map[string]resolved.ArtifactSlot{}
	for alias, binding := range program.Workflows {
		workflow, err := lookup(binding.Ref)
		if err != nil || workflow.Ref.Name+"@"+workflow.Ref.Version != binding.Ref {
			return nil, apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: Workflow selection is unavailable")
		}
		workflows[alias] = workflow
		workflowInputs[alias] = workflow.Inputs
		mappings := binding.Inputs
		for name, slot := range workflow.Inputs {
			if _, ok := mappings[name]; !ok && slot.Required {
				return nil, apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: required Workflow input is missing")
			}
		}
		for name, source := range mappings {
			slot, ok := workflow.Inputs[name]
			if !ok {
				return nil, apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: unknown Workflow input")
			}
			if source.Source == "task-input" {
				continue
			}
			if source.Source != "audit-input" {
				if !contracts.AcceptsMediaType(slot.MediaTypes, "application/json") {
					return nil, apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: context input must accept JSON")
				}
				continue
			}
			input := inputs[source.Name]
			if err := validateAuditMaterialInput(slot, input); err != nil {
				return nil, err
			}
		}
		parameters := binding.Parameters
		for name, slot := range workflow.Parameters {
			if _, ok := parameters[name]; !ok && slot.Required {
				return nil, apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: required Workflow parameter is missing")
			}
		}
		for name := range parameters {
			if _, ok := workflow.Parameters[name]; !ok {
				return nil, apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: unknown Workflow parameter")
			}
		}
		for logical, output := range binding.Outputs {
			slot, ok := workflow.Outputs[output]
			if !ok || !slot.Required || (logical == "result" || logical == "work") && !contracts.AcceptsMediaType(slot.MediaTypes, "application/json") {
				return nil, apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: output must name a required Workflow output")
			}
		}
		if _, err := ResolveCompletionTarget(workflow, binding); err != nil {
			return nil, err
		}
	}
	for _, task := range program.InitialWork {
		for name, slot := range workflowInputs[task.Workflow] {
			if slot.Required && program.Workflows[task.Workflow].Inputs[name].Source == "task-input" && task.Inputs[name] == "" {
				return nil, apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: required initial task material is missing")
			}
		}
		for name, selected := range task.Inputs {
			// Decode already checked binding and material membership. The
			// loaded Workflow owns requiredness of the effective input slot.
			input := inputs[selected]
			if err := validateAuditMaterialInput(workflowInputs[task.Workflow][name], input); err != nil {
				return nil, err
			}
		}
	}
	return workflows, nil
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
