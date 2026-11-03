package resolved

import (
	"fmt"
)

// AuditTaskContract identifies the task payload accepted by an executor. It is
// independent of the inventory implementation that produced that payload.
type AuditTaskContract string

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
	if workflow.AuditTask != nil {
		return fmt.Errorf("auditTask is retired; declare ordinary scanPreparation")
	}
	count := 0
	for _, stage := range workflow.Stages {
		if stage.ScanPreparation != nil {
			count++
		}
	}
	if count > 1 {
		return fmt.Errorf("a Workflow permits one prepared scan Stage")
	}
	return nil
}
