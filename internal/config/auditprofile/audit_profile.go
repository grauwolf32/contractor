package auditprofile

// Kind is the manifest kind of an AuditProfile.
const Kind = "AuditProfile"

const (
	MaxAuditProfileInputs = 32

	MaxAuditWorkflowMappings = 128
	MaxAuditStandardLevels   = 16
	MaxAuditStandardEntries  = 4_096
	MaxAuditRounds           = 32

	MaxAuditItemsTotal = 100_000

	MaxAuditRunAttempts     = 10
	MaxAuditDeadlineSeconds = 365 * 24 * 60 * 60

	MaxAuditLiteralParamBytes = 4096
)

type AuditProfileRef struct {
	Name    string `json:"name"`
	Version string `json:"version"`
	Digest  string `json:"digest"`
}

// AuditStandardSelection is an exact, profile-authored denominator. EntryIDs
// are authoritative; Scope and Levels describe that bounded selection without
// asking the Server or a model to infer a standard-specific hierarchy.
type AuditStandardSelection struct {
	Scope    string   `json:"scope"`
	Levels   []string `json:"levels"`
	EntryIDs []string `json:"entryIds"`
}
