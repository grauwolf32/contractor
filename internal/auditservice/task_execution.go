package auditservice

import (
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// Validate actual task requirements after inventory construction and before any
// task is retained or dispatched. Builders do not inspect executor topology.
func validateInventoryTaskExecution(profile config.ResolvedAuditProfile, inventory auditdomain.Inventory) error {
	for _, generated := range inventory.Tasks {
		task := generated.Document
		binding, exists := profile.Workflows[task.WorkflowRole]
		if !exists {
			return fmt.Errorf("%w: task names an unknown Workflow role", ErrInvalid)
		}
		execution := binding.Workflow.AuditTask
		contract := task.ExecutionContract()
		if execution == nil {
			if contract != "" {
				return fmt.Errorf("%w: task requires %s executor", ErrInvalid, contract)
			}
			continue
		}
		if execution.Contract != contract {
			return fmt.Errorf("%w: task contract differs from Workflow executor", ErrInvalid)
		}
		switch contract {
		case contracts.AuditTaskOpenAPIScanV1:
			stage, exists := binding.Workflow.Stages[execution.Stage]
			if !exists || stage.AuditScan == nil || task.Scan == nil || stage.AuditScan.Scanner != task.Scan.Scanner {
				return fmt.Errorf("%w: assigned scanner differs from task executor", ErrInvalid)
			}
		default:
			return fmt.Errorf("%w: unsupported task execution contract %q", ErrInvalid, contract)
		}
	}
	return nil
}
