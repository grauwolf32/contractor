package auditstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

// LockForStartStaging reads ownerID's Audit and holds FOR KEY SHARE on its row
// until the caller's transaction ends. Audit start writes its content-named
// artifacts under this lock before its own transaction (AUD-R225): purge locks
// the row FOR UPDATE, so it either waits and then removes what the start
// staged, or it has already deleted the Audit and this read is ErrNotFound. No
// staged binding outlives its Audit. The lock leaves draft edits, which change
// no key column, unblocked. The lock must last until the caller commits, so
// the step exists only on the transaction-bound store (PER-R18).
func (s *PostgresTxStore) LockForStartStaging(ctx context.Context, ownerID, auditID string) (Audit, error) {
	if err := validateText("ownerID", ownerID, 256, true); err != nil {
		return Audit{}, err
	}
	if err := validateID("auditID", auditID); err != nil {
		return Audit{}, err
	}
	audit, err := scanAudit(s.tx.QueryRow(ctx, `
SELECT `+prefixedAuditColumns("audit")+`
  FROM audits AS audit
 WHERE audit.owner_id = $1 AND audit.audit_id = $2
   FOR KEY SHARE`, ownerID, auditID))
	if errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, ErrNotFound
	}
	if err != nil {
		return Audit{}, fmt.Errorf("lock Audit for start staging: %w", err)
	}
	return audit, nil
}
