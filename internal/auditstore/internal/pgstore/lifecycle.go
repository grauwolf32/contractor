package pgstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

// SettleUndispatched closes items for which the durable dispatch fence won
// before another execution intent could be created. Created executions always
// take the ordinary terminal-observation and collection-receipt path instead.
func (s *ClaimStore) SettleUndispatched(
	ctx context.Context, claim auditstore.ControllerClaim, limit int,
) (int, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return 0, err
	}
	if limit < 1 || limit > auditstore.MaxReconcileRows {
		return 0, invalidf("Audit settlement batch is invalid")
	}
	var authorized bool
	var changed int
	err := s.db.QueryRow(ctx, settleUndispatchedSQL,
		claim.AuditID, claim.HolderID, claim.Epoch, limit,
	).Scan(&authorized, &changed)
	if err != nil {
		return 0, fmt.Errorf("settle undispatched Audit items: %w", err)
	}
	if authorized {
		return changed, nil
	}
	if live, liveErr := s.claimLive(ctx, claim); liveErr != nil {
		return 0, liveErr
	} else if !live {
		return 0, auditstore.ErrClaimLost
	}
	return 0, auditstore.ErrPrecondition
}

// ReleaseDispatchHold releases only the Audit-level future-dispatch hold.
// Per-Run credential references and retained evidence have independent
// lifetimes and are intentionally untouched.
func (s *ClaimStore) ReleaseDispatchHold(
	ctx context.Context, claim auditstore.ControllerClaim,
) (auditstore.Audit, bool, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return auditstore.Audit{}, false, err
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, releaseDispatchHoldSQL+prefixedAuditColumns("changed")+` FROM changed`,
		claim.AuditID, claim.HolderID, claim.Epoch,
	))
	if err == nil {
		return audit, true, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Audit{}, false, fmt.Errorf("release Audit dispatch hold: %w", err)
	}
	if live, liveErr := s.claimLive(ctx, claim); liveErr != nil {
		return auditstore.Audit{}, false, liveErr
	} else if !live {
		return auditstore.Audit{}, false, auditstore.ErrClaimLost
	}
	existing, getErr := s.getAuditTrusted(ctx, claim.AuditID)
	if getErr != nil {
		return auditstore.Audit{}, false, getErr
	}
	if existing.Hold == auditstore.HoldReleased || existing.Hold == auditstore.HoldPending || existing.Dispatch == auditstore.DispatchClosed {
		return existing, false, nil
	}
	return auditstore.Audit{}, false, auditstore.ErrPrecondition
}

// NextLiveRunForDeletion returns one still-present child Run after collection.
// run_id remains on AuditExecution as tombstone provenance after hard deletion.
func (s *ClaimStore) NextLiveRunForDeletion(
	ctx context.Context, claim auditstore.ControllerClaim,
) (string, bool, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return "", false, err
	}
	var runID string
	err := s.db.QueryRow(ctx, nextLiveRunForDeletionSQL, claim.AuditID, claim.HolderID, claim.Epoch).Scan(&runID)
	if err == nil {
		return runID, true, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return "", false, fmt.Errorf("list Audit child Runs for deletion: %w", err)
	}
	if live, liveErr := s.claimLive(ctx, claim); liveErr != nil {
		return "", false, liveErr
	} else if !live {
		return "", false, auditstore.ErrClaimLost
	}
	audit, getErr := s.getAuditTrusted(ctx, claim.AuditID)
	if getErr != nil {
		return "", false, getErr
	}
	if audit.State != auditstore.AuditDeleting {
		return "", false, auditstore.ErrPrecondition
	}
	return "", false, nil
}

// PurgeClaimed atomically removes one drained Audit's protected Project
// bindings and domain rows. The claim row is cascaded, so the caller treats a
// subsequent ReleaseClaim miss as the expected successful terminal outcome.
func (s *ClaimStore) PurgeClaimed(
	ctx context.Context, claim auditstore.ControllerClaim, namespace string,
) error {
	if err := validateClaimIdentity(claim); err != nil {
		return err
	}
	if namespace != auditdomain.ArtifactNamespace(claim.AuditID) {
		return invalidf("Audit purge namespace is invalid")
	}
	if s.purgeDependencies == nil {
		return invalidf("Audit purge requires dependent-row cleanup")
	}
	return persistencepostgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		return purgeClaimedAudit(ctx, tx, claim, namespace, s.purgeDependencies)
	})
}

func purgeClaimedAudit(
	ctx context.Context, tx pgx.Tx, claim auditstore.ControllerClaim, namespace string, purgeDependencies auditstore.PurgeDependencies,
) error {
	if err := auditClaims.Lock(ctx, tx, claim.AuditID, claim.HolderID, int64(claim.Epoch)); err != nil {
		if errors.Is(err, persistencepostgres.ErrControllerClaimLost) {
			return auditstore.ErrClaimLost
		}
		return err
	}
	var projectID string
	err := tx.QueryRow(ctx, purgeClaimedAuditSQL, claim.AuditID, claim.HolderID, claim.Epoch).Scan(&projectID)
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.ErrPrecondition
	}
	if err != nil {
		return fmt.Errorf("lock drained Audit for purge: %w", err)
	}
	if _, err := tx.Exec(ctx, `SELECT set_config('contractor.lifecycle_purge', 'audit', true)`); err != nil {
		return persistencepostgres.WrapError("enable Audit lifecycle purge", err)
	}
	remaining := persistencepostgres.CleanupBatchRows
	for _, statement := range purgeAuditDomainSQL {
		tag, err := tx.Exec(ctx, statement, claim.AuditID, remaining)
		if err != nil {
			return persistencepostgres.WrapError("purge Audit domain batch", err)
		}
		remaining -= int(tag.RowsAffected())
		if remaining == 0 {
			return nil
		}
	}
	removed, err := purgeDependencies(ctx, tx, claim.AuditID, remaining)
	if err != nil {
		return err
	}
	if removed < 0 || removed > remaining {
		return invalidf("Audit purge dependency exceeded its row bound")
	}
	remaining -= removed
	if remaining == 0 {
		return nil
	}
	removed, err = persistencepostgres.DeleteAuditReceiptsBatch(ctx, tx, claim.AuditID, remaining)
	if err != nil || removed == remaining {
		return err
	}
	// References from decided history are gone before releasing holds; held
	// material remains available until every hold has been released.
	purger, err := artifacts.NewPostgresPurger(tx)
	if err != nil {
		return err
	}
	complete, err := purger.PurgeAuditNamespace(ctx, projectID, namespace)
	if err != nil {
		return fmt.Errorf("purge Audit artifacts: %w", err)
	}
	if !complete {
		return nil
	}
	tag, err := tx.Exec(ctx, `
DELETE FROM audits AS audit
 USING audit_controller_claims AS claim
 WHERE audit.audit_id = $1 AND audit.state = 'deleting'
   AND claim.audit_id = audit.audit_id
   AND `+auditClaims.HeldCondition("claim", "$2", "$3"),
		claim.AuditID, claim.HolderID, claim.Epoch)
	if err != nil {
		return fmt.Errorf("delete drained Audit: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return auditstore.ErrClaimLost
	}
	return nil
}
