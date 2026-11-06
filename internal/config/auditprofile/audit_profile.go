package auditprofile

// Kind is the manifest kind of an AuditProfile.
const Kind = "AuditProfile"

const (
	MaxAuditProfileInputs = 32

	MaxAuditRounds = 32

	MaxAuditItemsTotal = 100_000

	MaxAuditRunAttempts     = 10
	MaxAuditDeadlineSeconds = 365 * 24 * 60 * 60
)

type AuditProfileRef struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Digest  string `json:"digest"`
}
