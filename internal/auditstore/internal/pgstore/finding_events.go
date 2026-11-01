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
	AuditID       string
	ReceiptID     string
	AssessmentID  string
	RetainedBytes int64
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
		event.AuditID, event.RetainedBytes, event.ReceiptID, event.AssessmentID,
	).Scan(&sequence)
}

// RecordFindingSourceAdded invalidates the Audit view for one new source
// association. The receipt has no Audit-owned Finding identity or state.
func (s *FindingEventsTx) RecordFindingSourceAdded(ctx context.Context, auditID, receiptID, runID string) error {
	if err := s.linkFindingSourceAudit(ctx, receiptID, auditID); err != nil {
		return err
	}
	var sequence int64
	if err := s.db.QueryRow(ctx, recordFindingSourceAddedSQL, auditID, receiptID, runID).Scan(&sequence); err != nil {
		return err
	}
	_, err := s.db.Exec(ctx, "SELECT contractor_advance_finding_correlation(ARRAY[(SELECT project_id FROM audits WHERE audit_id=$1)])", auditID)
	return err
}
