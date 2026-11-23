package auditstore

import "github.com/grauwolf32/contractor/internal/contracts/auditcontract"

// HypothesisWrite contains owner-assigned provenance. Public commands accept
// content only; an update carries its expected prior revision.
type HypothesisWrite struct {
	Document  auditcontract.Hypothesis `json:"document"`
	SourceKey string                   `json:"source_key"`
	Fixed     bool                     `json:"fixed"`
}

type WriteHypothesesParams struct {
	Claim            *ControllerClaim
	InitialProgress  *InitialWorkProgress
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	IdempotencyKey   string
	RequestDigest    string
	Writes           []HypothesisWrite
}

type HypothesisAcceptance struct {
	Hypotheses    []auditcontract.HypothesisRef `json:"hypotheses"`
	AuditRevision uint64                        `json:"audit_revision"`
}

type Hypothesis struct {
	Document  auditcontract.Hypothesis `json:"document"`
	SourceKey string                   `json:"source_key"`
	Fixed     bool                     `json:"fixed"`
	Ordinal   int                      `json:"ordinal"`
}

type ListHypothesesParams struct {
	OwnerID      string
	AuditID      string
	AfterOrdinal int
	Limit        int
}

type HypothesisPage struct {
	CheckingWorkflows []string     `json:"checking_workflows"`
	PendingFixedCount int          `json:"pending_fixed_count"`
	FixedDenominator  int          `json:"fixed_denominator"`
	Items             []Hypothesis `json:"items"`
	NextOrdinal       *int         `json:"next_ordinal,omitempty"`
}
