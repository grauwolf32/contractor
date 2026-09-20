package config

import "github.com/grauwolf32/contractor/internal/contracts"

// AuditInventoryOpenAPIScans selects a versioned inventory builder, independently
// of the Workflow that consumes its tasks.
const AuditInventoryOpenAPIScans = "openapi-scans@1"

type auditInventoryDescriptor struct {
	mediaTypes       []string
	taskContract     contracts.AuditTaskContract
	requiresSettings bool
}

// Legacy builders retain their existing task/completion contracts. Explicit
// executor contracts are opt-in and do not change old snapshots or digests.
var auditInventories = map[string]auditInventoryDescriptor{
	AuditInventoryOpenAPIScans: {
		mediaTypes:       []string{"application/json", "application/yaml"},
		taskContract:     contracts.AuditTaskOpenAPIScanV1,
		requiresSettings: true,
	},
	"openapi-operations@1": {mediaTypes: []string{"application/json", "application/yaml", "application/zip"}},
	"checklist@1":          {mediaTypes: []string{"application/json", "application/yaml", "application/zip"}},
	"finding-candidates@1": {mediaTypes: []string{"application/json", "application/zip"}},
	"standard-mappings@1":  {},
}
