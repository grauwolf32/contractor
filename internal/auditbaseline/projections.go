package auditbaseline

import (
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
)

// Read projections preserve downstream readers' existing unknown-field and
// map semantics. Full baseline validation remains owned by the start service.
type ReportProjection struct {
	Schema    string                              `json:"schema"`
	Inputs    map[string]auditstore.ExactArtifact `json:"inputs"`
	Scope     map[string]string                   `json:"scope"`
	Standards []auditstandards.PinnedPackage      `json:"standards"`
	Inventory BaselineInventory                   `json:"inventory"`
}

type StandardsProjection struct {
	Schema    string                         `json:"schema"`
	Standards []auditstandards.PinnedPackage `json:"standards"`
}
