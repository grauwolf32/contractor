package engagementcontract

import (
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

// AllocationProjection is the private, task-restricted view of a pinned
// engagement. Only the typed HTTP assignment authorizes target work.
type AllocationProjection struct {
	Schema           string                `json:"schema" strictjson:"required"`
	AuditID          string                `json:"auditId" strictjson:"required"`
	Engagement       contracts.ArtifactRef `json:"engagement" strictjson:"required"`
	EngagementDigest string                `json:"engagementDigest" strictjson:"required"`
	AuthorizedBy     string                `json:"authorizedBy" strictjson:"required"`
	AuthorizedAt     time.Time             `json:"authorizedAt" strictjson:"required"`
	HTTP             *HTTPAssignment       `json:"http,omitempty"`
}

func (p AllocationProjection) Validate() error {
	if p.Schema != "contractor.engagement.allocation.v1" ||
		contracts.ValidateOpaqueID("auditId", p.AuditID) != nil ||
		contracts.ValidateOpaqueID("authorizedBy", p.AuthorizedBy) != nil || p.AuthorizedAt.IsZero() ||
		p.Engagement.ValidateExact() != nil || !contentdigest.Valid(p.EngagementDigest) {
		return contracts.Invalidf("engagement allocation authority is invalid")
	}
	if p.HTTP != nil {
		return p.HTTP.Validate()
	}
	return nil
}
