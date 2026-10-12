package auditprofile

import (
	"github.com/grauwolf32/contractor/internal/config/resolved"
)

// AuditInventoryOpenAPIScans selects a versioned inventory builder, independently
// of the Workflow that consumes its tasks.
const AuditInventoryOpenAPIScans = "openapi-scans@1"

type auditInventoryDescriptor struct {
	mediaTypes       []string
	taskContract     resolved.AuditTaskContract
	requiresSettings bool
}

// A builder without a taskContract produces tasks for the generic Audit item
// and completion contract; a builder opts in to a typed executor contract here.
var auditInventories = map[string]auditInventoryDescriptor{
	AuditInventoryOpenAPIScans: {
		mediaTypes:       []string{"application/json", "application/yaml"},
		taskContract:     resolved.AuditTaskOpenAPIScanV1,
		requiresSettings: true,
	},
	"openapi-operations@1": {mediaTypes: []string{"application/json", "application/yaml", "application/zip"}},
	"checklist@1":          {mediaTypes: []string{"application/json", "application/yaml", "application/zip"}},
	"finding-candidates@1": {mediaTypes: []string{"application/json", "application/zip"}},
	"standard-mappings@1":  {},
}
