package auditstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// SettleUndispatched closes items for which the durable dispatch fence won
// before another execution intent could be created. Created executions always
// take the ordinary terminal-observation and collection-receipt path instead.
func (s *PostgresStore) SettleUndispatched(
	ctx context.Context, claim ControllerClaim, limit int,
) (int, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return 0, err
	}
	if limit < 1 || limit > MaxReconcileRows {
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
		return 0, ErrClaimLost
	}
	return 0, ErrPrecondition
}

// ReleaseDispatchHold releases only the Audit-level future-dispatch hold.
// Per-Run credential references and retained evidence have independent
// lifetimes and are intentionally untouched.
func (s *PostgresStore) ReleaseDispatchHold(
	ctx context.Context, claim ControllerClaim,
) (Audit, bool, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return Audit{}, false, err
	}
	audit, err := scanAudit(s.db.QueryRow(ctx, releaseDispatchHoldSQL+prefixedAuditColumns("changed")+` FROM changed`,
		claim.AuditID, claim.HolderID, claim.Epoch,
	))
	if err == nil {
		return audit, true, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Audit{}, false, fmt.Errorf("release Audit dispatch hold: %w", err)
	}
	if live, liveErr := s.claimLive(ctx, claim); liveErr != nil {
		return Audit{}, false, liveErr
	} else if !live {
		return Audit{}, false, ErrClaimLost
	}
	existing, getErr := s.getAuditTrusted(ctx, claim.AuditID)
	if getErr != nil {
		return Audit{}, false, getErr
	}
	if existing.Hold == HoldReleased || existing.Hold == HoldPending || existing.Dispatch == DispatchClosed {
		return existing, false, nil
	}
	return Audit{}, false, ErrPrecondition
}

// NextLiveRunForDeletion returns one still-present child Run after collection.
// run_id remains on AuditExecution as tombstone provenance after hard deletion.
func (s *PostgresStore) NextLiveRunForDeletion(
	ctx context.Context, claim ControllerClaim,
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
		return "", false, ErrClaimLost
	}
	audit, getErr := s.getAuditTrusted(ctx, claim.AuditID)
	if getErr != nil {
		return "", false, getErr
	}
	if audit.State != AuditDeleting {
		return "", false, ErrPrecondition
	}
	return "", false, nil
}

// PurgeClaimed atomically removes one drained Audit's protected Project
// bindings and domain rows. The claim row is cascaded, so the caller treats a
// subsequent ReleaseClaim miss as the expected successful terminal outcome.
func (s *PostgresStore) PurgeClaimed(
	ctx context.Context, claim ControllerClaim, namespace string,
) error {
	if err := validateClaimIdentity(claim); err != nil {
		return err
	}
	if namespace != auditdomain.ArtifactNamespace(claim.AuditID) {
		return invalidf("Audit purge namespace is invalid")
	}
	switch db := s.db.(type) {
	case *pgxpool.Pool:
		return persistencepostgres.InTxWithRetry(ctx, db, pgx.TxOptions{}, func(tx pgx.Tx) error {
			return purgeClaimedAudit(ctx, tx, claim, namespace)
		})
	case pgx.Tx:
		return purgeClaimedAudit(ctx, db, claim, namespace)
	default:
		return errors.New("purge Audit: PostgreSQL transaction support is required")
	}
}

func purgeClaimedAudit(
	ctx context.Context, tx pgx.Tx, claim ControllerClaim, namespace string,
) error {
	var projectID string
	err := tx.QueryRow(ctx, purgeClaimedAuditSQL, claim.AuditID, claim.HolderID, claim.Epoch).Scan(&projectID)
	if errors.Is(err, pgx.ErrNoRows) {
		return ErrPrecondition
	}
	if err != nil {
		return fmt.Errorf("lock drained Audit for purge: %w", err)
	}
	purger, err := artifacts.NewPostgresPurger(tx)
	if err != nil {
		return err
	}
	if err := purger.PurgeAuditNamespace(ctx, projectID, namespace); err != nil {
		return fmt.Errorf("purge Audit artifacts: %w", err)
	}
	tag, err := tx.Exec(ctx, `
DELETE FROM audits AS audit
 USING audit_controller_claims AS claim
 WHERE audit.audit_id = $1 AND audit.state = 'deleting'
   AND claim.audit_id = audit.audit_id
   AND claim.holder_id = $2 AND claim.epoch = $3`,
		claim.AuditID, claim.HolderID, claim.Epoch)
	if err != nil {
		return fmt.Errorf("delete drained Audit: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return ErrClaimLost
	}
	return nil
}
