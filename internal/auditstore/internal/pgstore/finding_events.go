package pgstore

import (
	"context"

	"github.com/grauwolf32/contractor/internal/auditstore"
)

// RecordRejectedFindingProposal advances one Audit revision for a proposal
// rejected by finding intake. The caller owns the receipt replay check and
// transaction; the Audit revision and event are written in one statement.
func (s *FindingEvents) RecordRejectedFindingProposal(
	ctx context.Context, params auditstore.RejectedFindingProposalParams,
) error {
	var sequence int64
	return s.db.QueryRow(ctx, recordRejectedFindingProposalSQL,
		params.AuditID, params.ReceiptID, params.RunID, params.Reason,
	).Scan(&sequence)
}

// directFindingAssessmentEvent is the finding.assessed event of a direct
// verification at the finding revision the assessment produced, with the
// evidence bytes it charges to the Audit.
type directFindingAssessmentEvent struct {
	AuditID         string
	FindingID       string
	FindingRevision int64
	AssessmentID    string
	RetainedBytes   int64
}

// recordDirectFindingAssessment charges the Audit evidence quota and records
// the finding assessment for CommitDirectFindingAssessment. A failed quota
// check returns pgx.ErrNoRows without changing the Audit or reserving a
// sequence.
func (s *FindingEvents) recordDirectFindingAssessment(
	ctx context.Context, event directFindingAssessmentEvent,
) error {
	var sequence int64
	return s.db.QueryRow(ctx, recordDirectFindingAssessmentSQL,
		event.AuditID, event.RetainedBytes, event.FindingID,
		event.FindingRevision, event.AssessmentID,
	).Scan(&sequence)
}
