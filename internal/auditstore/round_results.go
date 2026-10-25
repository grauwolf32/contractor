package auditstore

import "github.com/grauwolf32/contractor/internal/auditdomain"

// RoundResultsBuild is durable construction progress of exactly one settled
// Round. Captured pages cannot dispatch a consumer until Manifest is accepted.
type RoundResultsBuild struct {
	AuditID           string
	RoundID           string
	Generation        uint64
	Sealed            bool
	ItemCount         int
	ExpectedItemCount int
	MetadataBytes     int64
	PageCount         int
	Pages             []RoundResultsBuildPage
	Manifest          *ExactArtifact
}

type RoundResultsBuildPage struct {
	Ordinal      int
	ItemCount    int
	ContentBytes int
	Artifact     *ExactArtifact
}

// RoundResultsStepParams carries the existing Controller authority. A capture
// step reads at most 128 selected Round records, with a separate related-row
// budget; publication never rereads mutable review state after source sealing.
type RoundResultsStepParams struct {
	Claim   ControllerClaim
	RoundID string
}

type PublishRoundResultsPageParams struct {
	RoundResultsStepParams
	Generation uint64
	Ordinal    int
	Artifact   ExactArtifact
}

type PublishRoundResultsManifestParams struct {
	RoundResultsStepParams
	Generation uint64
	Manifest   ExactArtifact
	Content    auditdomain.RoundResultsManifest
}
