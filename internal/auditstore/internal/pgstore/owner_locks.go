package pgstore

import "context"

// The steps of an owner's resume inside the caller's transaction. Each lock
// lasts until the caller commits, so these steps exist only on the
// transaction-bound OwnerTxStore (PER-R18).

// LockAuditForResume locks an owner's Audit row until the caller commits. A
// missing Audit locks nothing; the caller's next read reports it.
func (s *OwnerTxStore) LockAuditForResume(ctx context.Context, ownerID, auditID string) error {
	_, err := s.tx.Exec(ctx, `SELECT audit_id FROM audits WHERE audit_id=$1 AND owner_id=$2 FOR UPDATE`, auditID, ownerID)
	return err
}

// HasReportReview reports whether an Audit ever proposed a report for
// review, which a resume cannot reopen.
func (s *OwnerTxStore) HasReportReview(ctx context.Context, auditID string) (bool, error) {
	var candidate bool
	err := s.tx.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audit_review_requests WHERE audit_id=$1 AND subject_kind='audit-report')`, auditID).Scan(&candidate)
	return candidate, err
}
