package auditdomain

import "github.com/grauwolf32/contractor/internal/contracts"

// ExecutionContract identifies typed execution requirements without recording
// which inventory implementation created the task. Empty preserves the legacy
// Audit item/completion contract.
func (task ItemTask) ExecutionContract() contracts.AuditTaskContract {
	if task.Kind == "openapi-scan" {
		return contracts.AuditTaskOpenAPIScanV1
	}
	return ""
}
