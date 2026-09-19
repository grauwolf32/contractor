package auditbaseline

import (
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
)

// Read projections preserve downstream readers' existing unknown-field and
// map semantics. Full baseline validation remains owned by the start service.
type ReportProjection struct {
	Schema    string                              `json:"schema"`
	Inputs    map[string]auditstore.ExactArtifact `json:"inputs"`
	Scope     map[string]string                   `json:"scope"`
	Standards []auditstandards.PinnedPackage      `json:"standards"`
	Inventory struct {
		SourceContentDigest      string                         `json:"sourceContentDigest"`
		CanonicalInventoryDigest string                         `json:"canonicalInventoryDigest"`
		Worklist                 auditstore.ExactArtifact       `json:"worklist"`
		Gaps                     []string                       `json:"gaps"`
		StandardSelection        *config.AuditStandardSelection `json:"standardSelection,omitempty"`
	} `json:"inventory"`
}

type StandardsProjection struct {
	Schema    string                         `json:"schema"`
	Standards []auditstandards.PinnedPackage `json:"standards"`
}
