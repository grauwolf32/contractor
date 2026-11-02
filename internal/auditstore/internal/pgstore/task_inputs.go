package pgstore

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/auditstore"
	postgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

// LockTaskInputs holds the normal Project/claim/Audit fences while the caller
// retains one bounded set of exact inputs through ArtifactStore. Draft permits
// Start staging; actual Task acceptance still requires active or paused.
func (s *OwnerTxStore) LockTaskInputs(ctx context.Context, ownerID, auditID string, claim *auditstore.ControllerClaim) (auditstore.Audit, error) {
	if validateID("auditID", auditID) != nil || ownerID == "" || claim != nil && (claim.AuditID != auditID || validateClaimIdentity(*claim) != nil) {
		return auditstore.Audit{}, auditstore.ErrInvalid
	}
	var projectID string
	err := s.tx.QueryRow(ctx, `SELECT project_id FROM audits WHERE audit_id=$1 AND owner_id=$2`, auditID, ownerID).Scan(&projectID)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.Audit{}, err
	}
	if _, err := projectstore.NewPostgresStore(s.tx).LockActiveAuditProject(ctx, ownerID, projectID); err != nil {
		return auditstore.Audit{}, err
	}
	if claim != nil {
		if err := auditClaims.Lock(ctx, s.tx, auditID, claim.HolderID, int64(claim.Epoch)); err != nil {
			if errors.Is(err, postgres.ErrControllerClaimLost) {
				err = auditstore.ErrClaimLost
			}
			return auditstore.Audit{}, err
		}
	}
	audit, err := scanAudit(s.tx.QueryRow(ctx, `SELECT `+prefixedAuditColumns("audit")+` FROM audits AS audit WHERE audit_id=$1 AND owner_id=$2 FOR UPDATE`, auditID, ownerID))
	if err != nil {
		return auditstore.Audit{}, err
	}
	if audit.State != auditstore.AuditDraft && audit.State != auditstore.AuditActive && audit.State != auditstore.AuditPaused || claim != nil && audit.State == auditstore.AuditDraft {
		return auditstore.Audit{}, auditstore.ErrPrecondition
	}
	return audit, nil
}

// CollectedRunInput checks the original immutable association, including its
// historical Run tombstone. It reads no other Audit or Run history.
func (s *QueryStore) CollectedRunInput(ctx context.Context, auditID, runID string) (bool, error) {
	if validateID("auditID", auditID) != nil || validateID("runID", runID) != nil {
		return false, auditstore.ErrInvalid
	}
	var found bool
	err := s.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM audit_executions WHERE audit_id=$1 AND run_id=$2 AND state='collected')`, auditID, runID).Scan(&found)
	return found, err
}
