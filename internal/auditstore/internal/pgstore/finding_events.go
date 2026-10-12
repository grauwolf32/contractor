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

// RecordDirectFindingAssessment charges the Audit evidence quota and records
// the finding assessment in the caller's transaction. A failed quota check
// returns pgx.ErrNoRows without changing the Audit or reserving a sequence.
func (s *FindingEvents) RecordDirectFindingAssessment(
	ctx context.Context, params auditstore.DirectFindingAssessmentParams,
) error {
	var sequence int64
	return s.db.QueryRow(ctx, recordDirectFindingAssessmentSQL,
		params.AuditID, params.RetainedBytes, params.FindingID,
		params.FindingRevision, params.AssessmentID,
	).Scan(&sequence)
}
