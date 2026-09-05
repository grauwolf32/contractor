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
	err := s.db.QueryRow(ctx, `
WITH live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), target_audit AS MATERIALIZED (
    SELECT audit.audit_id, audit.state
      FROM audits AS audit JOIN live_claim USING (audit_id)
     WHERE audit.state IN ('finalizing', 'cancelling', 'deleting')
     FOR UPDATE OF audit
), candidates AS MATERIALIZED (
    SELECT item.item_id
      FROM audit_items AS item JOIN target_audit USING (audit_id)
     WHERE item.state IN ('pending', 'awaiting_review', 'ready')
     ORDER BY item.round_id, item.ordinal, item.item_id
     FOR UPDATE OF item SKIP LOCKED
     LIMIT $4
), changed_items AS (
    UPDATE audit_items AS item
       SET state = 'settled',
           final_disposition = CASE WHEN target_audit.state = 'finalizing'
               THEN 'excluded' ELSE 'execution-cancelled' END,
           updated_at = GREATEST(clock_timestamp(), item.updated_at + interval '1 microsecond')
      FROM candidates, target_audit
     WHERE item.item_id = candidates.item_id
    RETURNING item.item_id, item.audit_id
), changed_coverage AS (
    UPDATE audit_coverage_rows AS coverage
       SET status = CASE WHEN target_audit.state = 'finalizing'
               THEN 'not-tested' ELSE 'blocked' END,
           gaps = CASE
               WHEN coverage.gaps ? CASE WHEN target_audit.state = 'finalizing'
                   THEN 'audit-closed-before-dispatch' ELSE 'audit-cancelled-before-dispatch' END
               THEN coverage.gaps
               ELSE coverage.gaps || jsonb_build_array(
                   CASE WHEN target_audit.state = 'finalizing'
                       THEN 'audit-closed-before-dispatch' ELSE 'audit-cancelled-before-dispatch' END
               )
           END,
           rationale = CASE WHEN target_audit.state = 'finalizing'
               THEN 'Audit dispatch closed before this item was submitted.'
               ELSE 'Audit cancellation closed this item before dispatch.' END,
           updated_at = GREATEST(clock_timestamp(), coverage.updated_at + interval '1 microsecond')
      FROM changed_items, target_audit
     WHERE coverage.item_id = changed_items.item_id
), advanced_audit AS (
    UPDATE audits AS audit
       SET revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM target_audit
     WHERE audit.audit_id = target_audit.audit_id
       AND EXISTS (SELECT 1 FROM changed_items)
    RETURNING audit.audit_id, audit.next_event_sequence
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, summary)
    SELECT advanced_audit.audit_id, advanced_audit.next_event_sequence - 1,
           'items.dispatch_closed', advanced_audit.audit_id,
           jsonb_build_object('count', (SELECT count(*) FROM changed_items))
      FROM advanced_audit
)
SELECT EXISTS(SELECT 1 FROM target_audit), count(*) FROM changed_items`,
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
	audit, err := scanAudit(s.db.QueryRow(ctx, `
WITH live_claim AS MATERIALIZED (
    SELECT claim.audit_id
      FROM audit_controller_claims AS claim
     WHERE claim.audit_id = $1 AND claim.holder_id = $2 AND claim.epoch = $3
       AND claim.expires_at > clock_timestamp()
     FOR UPDATE OF claim
), changed AS (
    UPDATE audits AS audit
       SET hold_state = 'released', revision = audit.revision + 1,
           next_event_sequence = audit.next_event_sequence + 1,
           updated_at = GREATEST(clock_timestamp(), audit.updated_at + interval '1 microsecond')
      FROM live_claim
     WHERE audit.audit_id = live_claim.audit_id
       AND audit.dispatch_state = 'closed' AND audit.hold_state = 'held'
       AND NOT EXISTS (
           SELECT 1 FROM audit_executions AS execution
            WHERE execution.audit_id = audit.audit_id AND execution.state = 'intent'
       )
    RETURNING audit.*
), event_row AS (
    INSERT INTO audit_events (audit_id, sequence_number, kind, entity_id, entity_revision, summary)
    SELECT audit_id, next_event_sequence - 1, 'audit.dispatch_hold_released', audit_id, revision, '{}'::jsonb
      FROM changed
)
SELECT `+prefixedAuditColumns("changed")+` FROM changed`,
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
	err := s.db.QueryRow(ctx, `
SELECT run.run_id
  FROM audits AS audit
  JOIN audit_controller_claims AS claim USING (audit_id)
  JOIN audit_executions AS execution USING (audit_id)
  JOIN workflow_runs AS run ON run.run_id = execution.run_id
 WHERE audit.audit_id = $1 AND audit.state = 'deleting'
   AND claim.holder_id = $2 AND claim.epoch = $3
   AND claim.expires_at > clock_timestamp()
 ORDER BY execution.created_at, execution.execution_id
 LIMIT 1`, claim.AuditID, claim.HolderID, claim.Epoch).Scan(&runID)
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
		return persistencepostgres.InTx(ctx, db, pgx.TxOptions{}, func(tx pgx.Tx) error {
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
	err := tx.QueryRow(ctx, `
SELECT audit.project_id
  FROM audits AS audit
  JOIN audit_controller_claims AS claim USING (audit_id)
 WHERE audit.audit_id = $1 AND audit.state = 'deleting'
   AND audit.dispatch_state = 'closed' AND audit.hold_state <> 'held'
   AND audit.deletion_requested_at IS NOT NULL
   AND claim.holder_id = $2 AND claim.epoch = $3
   AND claim.expires_at > clock_timestamp()
   AND NOT EXISTS (
       SELECT 1 FROM audit_items AS item
        WHERE item.audit_id = audit.audit_id AND item.state <> 'settled'
   )
   AND NOT EXISTS (
       SELECT 1 FROM audit_executions AS execution
        WHERE execution.audit_id = audit.audit_id AND execution.state <> 'collected'
   )
   AND NOT EXISTS (
       SELECT 1
         FROM audit_executions AS execution
         JOIN workflow_runs AS run ON run.run_id = execution.run_id
        WHERE execution.audit_id = audit.audit_id
   )
 FOR UPDATE OF audit, claim`, claim.AuditID, claim.HolderID, claim.Epoch).Scan(&projectID)
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
