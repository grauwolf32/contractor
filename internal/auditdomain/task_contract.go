package auditdomain

import "github.com/grauwolf32/contractor/internal/config"

// ExecutionContract identifies typed execution requirements without recording
// which inventory implementation created the task. Empty selects the generic
// Audit item/completion contract.
func (task ItemTask) ExecutionContract() config.AuditTaskContract {
	if task.Kind == "openapi-scan" {
		return config.AuditTaskOpenAPIScanV1
	}
	return ""
}
