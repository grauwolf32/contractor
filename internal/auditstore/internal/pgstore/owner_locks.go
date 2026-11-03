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
