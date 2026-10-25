package resolved

import (
	"fmt"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/scan"
)

// AuditTaskContract identifies the task payload accepted by an executor. It is
// independent of the inventory implementation that produced that payload.
type AuditTaskContract string

const AuditTaskWebScanV1 AuditTaskContract = "web-scan@1"

const AuditTaskOpenAPIScanV1 AuditTaskContract = "openapi-scan@1"

// AuditTaskExecution declares which task contract a Workflow accepts and which
// Stage owns its execution/result. Surrounding Stages remain ordinary Workflow
// graph nodes; the task Stage need not be the entry Stage.
type AuditTaskExecution struct {
	Contract AuditTaskContract `json:"contract" yaml:"contract"`
	Stage    string            `json:"stage" yaml:"stage"`
}

func cloneAuditTask(value *AuditTaskExecution) *AuditTaskExecution {
	if value == nil {
		return nil
	}
	copy := *value
	return &copy
}

// ValidateAuditTaskWorkflow checks executor capabilities without consulting any
// inventory or profile. A declaration alone cannot grant an unsupported one.
func ValidateAuditTaskWorkflow(workflow ResolvedWorkflow) error {
	execution := workflow.AuditTask
	for name, stage := range workflow.Stages {
		if stage.AuditScan != nil && (execution == nil || name != execution.Stage) {
			return fmt.Errorf("Stage %q must be the declared auditTask executor", name)
		}
	}
	if execution == nil {
		return nil
	}
	stage, exists := workflow.Stages[execution.Stage]
	if !exists {
		return fmt.Errorf("auditTask.stage names an absent Stage")
	}
	switch execution.Contract {
	case AuditTaskOpenAPIScanV1, AuditTaskWebScanV1:
		if stage.AuditScan == nil {
			return fmt.Errorf("%s requires an auditScan executor", execution.Contract)
		}
		for _, agent := range stage.Agents {
			if execution.Contract == AuditTaskWebScanV1 {
				if stage.AuditScan.Scanner != "nuclei" || agent.Template.Execution == nil {
					return fmt.Errorf("web-scan@1 requires Nuclei")
				}
				for _, name := range []string{"template_ids", "tags", "severity", "rate_limit"} {
					if !scan.IsNucleiSelectionBinding(name, agent.Template.Execution.Arguments[name]) {
						return fmt.Errorf("web-scan@1 requires exact-settings selection bindings")
					}
				}
				pin := agent.Template.Execution.Arguments["templates_digest"]
				digest, ok := pin.Value.(string)
				if pin.Source != "literal" || !ok || contracts.ValidateDigest("templates_digest", digest) != nil {
					return fmt.Errorf("web-scan@1 requires a pinned installed package")
				}
			}
		}
		if err := ValidateScanPlanStage(stage); err != nil {
			return err
		}
		return validateScanPlanInputMedia(workflow, stage)
	default:
		return fmt.Errorf("unsupported auditTask contract %q", execution.Contract)
	}
}
