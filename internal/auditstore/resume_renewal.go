package auditstore

import (
	"time"
)

// ResumeRenewalParams identifies the paused Audit whose expired item reviews
// Resume renews. DeadlineAt is the deadline Resume applies, which also bounds
// the fresh requests; nil leaves them without expiry.
type ResumeRenewalParams struct {
	OwnerID          string
	AuditID          string
	ExpectedRevision uint64
	DeadlineAt       *time.Time
}
