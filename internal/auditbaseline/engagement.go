package auditbaseline

import (
	"time"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts/engagementcontract"
)

// EngagementSnapshot is trusted start metadata around one exact public document.
// It grants no allocation access: task assignment narrows the private projection.
type EngagementSnapshot struct {
	Schema       string                      `json:"schema" strictjson:"required"`
	Artifact     auditstore.ExactArtifact    `json:"artifact" strictjson:"required"`
	Document     engagementcontract.Document `json:"document" strictjson:"required"`
	PolicyDigest string                      `json:"policyDigest" strictjson:"required"`
	AuthorizedBy string                      `json:"authorizedBy" strictjson:"required"`
	AuthorizedAt time.Time                   `json:"authorizedAt" strictjson:"required"`
}

const EngagementSchema = "contractor.audit.engagement-baseline.v1"
