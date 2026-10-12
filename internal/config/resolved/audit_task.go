package resolved

import (
	"fmt"
)

// AuditTaskContract identifies the task payload accepted by an executor. It is
// independent of the inventory implementation that produced that payload.
type AuditTaskContract string

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
	case AuditTaskOpenAPIScanV1:
		if stage.AuditScan == nil {
			return fmt.Errorf("%s requires an auditScan executor", execution.Contract)
		}
		if err := ValidateScanPlanStage(stage); err != nil {
			return err
		}
		return validateScanPlanInputMedia(workflow, stage)
	default:
		return fmt.Errorf("unsupported auditTask contract %q", execution.Contract)
	}
}
