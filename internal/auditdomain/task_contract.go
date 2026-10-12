package auditdomain

import (
	"github.com/grauwolf32/contractor/internal/config/resolved"
)

// ExecutionContract identifies typed execution requirements without recording
// which inventory implementation created the task. Empty selects the generic
// Audit item/completion contract.
func (task ItemTask) ExecutionContract() resolved.AuditTaskContract {
	if task.Kind == "openapi-scan" {
		return resolved.AuditTaskOpenAPIScanV1
	}
	return ""
}
