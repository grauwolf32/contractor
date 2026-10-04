package auditstore

import "context"

type RejectedFindingProposalParams struct {
	AuditID   string
	ReceiptID string
	RunID     string
	Reason    string
}

// RecordRejectedFindingProposal advances one Audit revision for a proposal
// rejected by finding intake. The caller owns the receipt replay check and
// transaction; the Audit revision and event are written in one statement.
func (s *PostgresStore) RecordRejectedFindingProposal(
	ctx context.Context, params RejectedFindingProposalParams,
) error {
	var sequence int64
	return s.db.QueryRow(ctx, recordRejectedFindingProposalSQL,
		params.AuditID, params.ReceiptID, params.RunID, params.Reason,
	).Scan(&sequence)
}

type DirectFindingAssessmentParams struct {
	AuditID         string
	FindingID       string
	FindingRevision int64
	AssessmentID    string
	RetainedBytes   int64
}

// RecordDirectFindingAssessment charges the Audit evidence quota and records
// the finding assessment in the caller's transaction. A failed quota check
// returns pgx.ErrNoRows without changing the Audit or reserving a sequence.
func (s *PostgresStore) RecordDirectFindingAssessment(
	ctx context.Context, params DirectFindingAssessmentParams,
) error {
	var sequence int64
	return s.db.QueryRow(ctx, recordDirectFindingAssessmentSQL,
		params.AuditID, params.RetainedBytes, params.FindingID,
		params.FindingRevision, params.AssessmentID,
	).Scan(&sequence)
}
