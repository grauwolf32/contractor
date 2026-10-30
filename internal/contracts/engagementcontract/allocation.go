package engagementcontract

import (
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"slices"
	"time"

	"github.com/grauwolf32/contractor/internal/contentdigest"
	"github.com/grauwolf32/contractor/internal/contracts"
)

// AllocationProjection is the private, task-restricted view of a pinned
// engagement. Only the typed HTTP assignment authorizes target work.
type AllocationProjection struct {
	Schema           string                `json:"schema" strictjson:"required"`
	AuditID          string                `json:"auditId" strictjson:"required"`
	Engagement       contracts.ArtifactRef `json:"engagement" strictjson:"required"`
	EngagementDigest string                `json:"engagementDigest" strictjson:"required"`
	PolicyDigest     string                `json:"policyDigest" strictjson:"required"`
	AuthorizedBy     string                `json:"authorizedBy" strictjson:"required"`
	AuthorizedAt     time.Time             `json:"authorizedAt" strictjson:"required"`
	AccessMode       string                `json:"accessMode" strictjson:"required"`
	Policy           Policy                `json:"policy" strictjson:"required"`
	HTTP             *HTTPAssignment       `json:"http,omitempty"`
}

func (p AllocationProjection) Validate() error {
	if p.Schema != "contractor.engagement.allocation.v1" ||
		contracts.ValidateOpaqueID("auditId", p.AuditID) != nil ||
		contracts.ValidateOpaqueID("authorizedBy", p.AuthorizedBy) != nil || p.AuthorizedAt.IsZero() ||
		p.Engagement.ValidateExact() != nil || !contentdigest.Valid(p.EngagementDigest) || !contentdigest.Valid(p.PolicyDigest) ||
		(p.AccessMode != "blackbox" && p.AccessMode != "source-assisted") {
		return contracts.Invalidf("engagement allocation authority is invalid")
	}
	if err := p.Policy.Validate(); err != nil {
		return err
	}
	digest, err := p.Policy.Digest()
	if err != nil || digest != p.PolicyDigest {
		return contracts.Invalidf("engagement allocation policy digest differs")
	}
	for _, capability := range p.Policy.RequiredCapabilities {
		if capability != Capability && capability != string(runtimesettings.RuntimeAdapterHTTPTarget) {
			return contracts.Invalidf("engagement execution capability is unavailable")
		}
	}
	if p.HTTP != nil {
		if !slices.Contains(p.Policy.RequiredCapabilities, string(runtimesettings.RuntimeAdapterHTTPTarget)) {
			return contracts.Invalidf("controlled HTTP capability is absent")
		}
		return p.HTTP.Validate()
	}
	if slices.Contains(p.Policy.RequiredCapabilities, string(runtimesettings.RuntimeAdapterHTTPTarget)) {
		return contracts.Invalidf("engagement execution capability needs its assignment")
	}
	return nil
}
