package contracts

// AuditTaskContract identifies the task payload accepted by an executor. It is
// independent of the inventory implementation that produced that payload.
type AuditTaskContract string

const AuditTaskOpenAPIScanV1 AuditTaskContract = "openapi-scan@1"
