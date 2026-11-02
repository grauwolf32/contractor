package config

import (
	"github.com/grauwolf32/contractor/internal/apierror"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/config/resolved"

	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

// ValidateAuditProgram validates a common Audit program against this exact
// loaded Workflow catalog. It creates no runnable profile, grant or Run.
func ValidateAuditProgram(s *Snapshot, data []byte) error {
	program, err := auditcontract.DecodeProgram(data)
	if err != nil {
		return err
	}
	_, err = auditprofile.ResolveProgramWorkflows(program, s.Workflow)
	return err
}

// ValidatePreparedWork applies the shared source-set contract and the actual
// catalog's slot rules. It neither accepts tasks nor grants Artifact access.
func ValidatePreparedWork(s *Snapshot, data, context []byte) error {
	work, trusted, err := auditcontract.DecodePreparedWork(data, context)
	if err != nil {
		return err
	}
	program, err := strictjson.Canonical(trusted.Program)
	if err != nil {
		return err
	}
	if err := ValidateAuditProgram(s, program); err != nil {
		return err
	}
	workflows := map[string]resolved.ResolvedWorkflow{}
	for _, item := range work.Work {
		binding := trusted.Program.Workflows[item.Workflow]
		workflow, found := workflows[item.Workflow]
		if !found {
			workflow, err = s.Workflow(binding.Ref)
			if err != nil {
				return apierror.New(apierror.AuditDocumentInvalid, "audit_document_invalid: Workflow selection is unavailable")
			}
			workflows[item.Workflow] = workflow
		}
		if err := auditprofile.ValidateTaskInputs(binding, workflow, item.Inputs); err != nil {
			return err
		}
	}
	return nil
}
