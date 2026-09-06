package runstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

// DeleteReleasedTerminalRun permanently removes one owner-visible terminal
// Run and its Run-owned data. A pool-backed store creates the transaction; a
// transaction-backed store participates in its caller's transaction so Project
// lifecycle deletion can compose this operation atomically.
func (s *PostgresStore) DeleteReleasedTerminalRun(
	ctx context.Context,
	ownerID string,
	runID string,
) error {
	if err := validateOpaque("ownerID", ownerID); err != nil {
		return err
	}
	if err := validateOpaque("runID", runID); err != nil {
		return err
	}
	switch db := s.db.(type) {
	case *pgxpool.Pool:
		return persistencepostgres.InTxWithRetry(ctx, db, pgx.TxOptions{}, func(tx pgx.Tx) error {
			return deleteReleasedTerminalRun(ctx, tx, ownerID, runID)
		})
	case pgx.Tx:
		return deleteReleasedTerminalRun(ctx, db, ownerID, runID)
	default:
		return fmt.Errorf("delete WorkflowRun: PostgreSQL transaction support is required")
	}
}

func deleteReleasedTerminalRun(ctx context.Context, tx pgx.Tx, ownerID, runID string) error {
	var state WorkflowRunState
	var publicationMode OutputPublicationMode
	var auditExecutionID *string
	err := tx.QueryRow(ctx, `
SELECT state, publication_mode, audit_execution_id
FROM workflow_runs
WHERE run_id = $1 AND owner_id = $2
FOR UPDATE`, runID, ownerID).Scan(&state, &publicationMode, &auditExecutionID)
	if errors.Is(err, pgx.ErrNoRows) {
		return fmt.Errorf("delete WorkflowRun %q: %w", runID, ErrNotFound)
	}
	if err != nil {
		return fmt.Errorf("lock WorkflowRun %q for deletion: %w", runID, err)
	}
	if !RunLifecycleTerminal.Includes(state) {
		return &RunNotDeletableError{RunID: runID, Reason: RunNotTerminal}
	}

	var pendingRelease bool
	if err := tx.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1
    FROM stage_executions AS execution
    JOIN stage_allocations AS allocation
      ON allocation.stage_execution_id = execution.stage_execution_id
    WHERE execution.run_id = $1
      AND allocation.release_completed_at IS NULL
)`, runID).Scan(&pendingRelease); err != nil {
		return fmt.Errorf("check WorkflowRun %q allocation release: %w", runID, err)
	}
	if pendingRelease {
		return &RunNotDeletableError{RunID: runID, Reason: RunAllocationReleasePending}
	}
	if publicationMode == PublicationAuditManaged {
		if auditExecutionID == nil {
			return fmt.Errorf("delete WorkflowRun %q: invalid Audit authority", runID)
		}
		var collected bool
		if err := tx.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1
      FROM audit_executions AS execution
      JOIN audit_collection_receipts AS receipt
        ON receipt.execution_id = execution.execution_id
       AND receipt.audit_id = execution.audit_id
       AND receipt.run_id = execution.run_id
     WHERE execution.execution_id = $1 AND execution.run_id = $2
       AND execution.state = 'collected' AND execution.run_provenance IS NOT NULL
)`, *auditExecutionID, runID).Scan(&collected); err != nil {
			return fmt.Errorf("check WorkflowRun %q Audit collection: %w", runID, err)
		}
		if !collected {
			return &RunNotDeletableError{RunID: runID, Reason: RunAuditCollectionPending}
		}
		tag, err := tx.Exec(ctx, `
UPDATE audit_executions
   SET run_deleted_at = COALESCE(run_deleted_at, clock_timestamp()),
       updated_at = GREATEST(clock_timestamp(), updated_at + interval '1 microsecond')
 WHERE execution_id = $1 AND run_id = $2 AND state = 'collected'
   AND run_provenance IS NOT NULL
   AND EXISTS (
       SELECT 1 FROM audit_collection_receipts AS receipt
        WHERE receipt.execution_id = audit_executions.execution_id
          AND receipt.audit_id = audit_executions.audit_id
          AND receipt.run_id = audit_executions.run_id
   )`, *auditExecutionID, runID)
		if err != nil {
			return fmt.Errorf("mark WorkflowRun %q Audit tombstone: %w", runID, err)
		}
		if tag.RowsAffected() != 1 {
			return &RunNotDeletableError{RunID: runID, Reason: RunAuditCollectionPending}
		}
	}
	// Finding receipt identity and safe provenance outlive an ordinary source
	// Run. Imported proposals already have destination-Audit bindings; every
	// other proposal becomes a durable discarded tombstone before Run-owned
	// pins and bindings are purged below.
	if _, err := tx.Exec(ctx, `
UPDATE finding_proposal_retention AS retention
   SET source_run_deleted_at = COALESCE(retention.source_run_deleted_at, clock_timestamp()),
       state = CASE WHEN EXISTS (
           SELECT 1 FROM finding_proposal_audit_holds AS hold
            WHERE hold.receipt_id = retention.receipt_id
       ) THEN 'audit-held' ELSE 'discarded' END,
       discarded_at = CASE WHEN EXISTS (
           SELECT 1 FROM finding_proposal_audit_holds AS hold
            WHERE hold.receipt_id = retention.receipt_id
       ) THEN NULL ELSE COALESCE(retention.discarded_at, clock_timestamp()) END,
       updated_at = clock_timestamp()
  FROM finding_proposal_receipts AS receipt
 WHERE receipt.receipt_id = retention.receipt_id AND receipt.run_id = $1`, runID); err != nil {
		return fmt.Errorf("dispose WorkflowRun %q finding proposals: %w", runID, err)
	}

	purger, err := artifacts.NewPostgresPurger(tx)
	if err != nil {
		return fmt.Errorf("prepare WorkflowRun %q Artifact purge: %w", runID, err)
	}
	if err := purger.PurgeRun(ctx, runID); err != nil {
		return fmt.Errorf("purge WorkflowRun %q Artifacts: %w", runID, err)
	}
	tag, err := tx.Exec(ctx, `
DELETE FROM workflow_runs
WHERE run_id = $1 AND owner_id = $2`, runID, ownerID)
	if err != nil {
		return fmt.Errorf("delete WorkflowRun %q: %w", runID, err)
	}
	if tag.RowsAffected() != 1 {
		return fmt.Errorf("delete WorkflowRun %q: %w", runID, ErrConflict)
	}
	return nil
}

// RunDeletionBlocker reports the first durable lifecycle gate for an
// owner-visible Run without mutating it. The answer is advisory; deletion
// repeats every check under a row lock in one transaction.
func (s *PostgresStore) RunDeletionBlocker(
	ctx context.Context, ownerID, runID string,
) (*RunNotDeletableReason, error) {
	if err := validateOpaque("ownerID", ownerID); err != nil {
		return nil, err
	}
	if err := validateOpaque("runID", runID); err != nil {
		return nil, err
	}
	var state WorkflowRunState
	var releasePending, collectionPending bool
	err := s.db.QueryRow(ctx, `
SELECT run.state,
       EXISTS (
           SELECT 1
             FROM stage_executions AS execution
             JOIN stage_allocations AS allocation
               ON allocation.stage_execution_id = execution.stage_execution_id
            WHERE execution.run_id = run.run_id
              AND allocation.release_completed_at IS NULL
       ),
       run.publication_mode = 'audit-managed' AND NOT EXISTS (
           SELECT 1
             FROM audit_executions AS audit_execution
             JOIN audit_collection_receipts AS receipt
               ON receipt.execution_id = audit_execution.execution_id
              AND receipt.audit_id = audit_execution.audit_id
              AND receipt.run_id = run.run_id
            WHERE audit_execution.execution_id = run.audit_execution_id
              AND audit_execution.run_id = run.run_id
              AND audit_execution.state = 'collected'
              AND audit_execution.run_provenance IS NOT NULL
       )
  FROM workflow_runs AS run
 WHERE run.run_id = $1 AND run.owner_id = $2`, runID, ownerID).Scan(
		&state, &releasePending, &collectionPending,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, fmt.Errorf("inspect WorkflowRun %q deletion: %w", runID, ErrNotFound)
	}
	if err != nil {
		return nil, fmt.Errorf("inspect WorkflowRun %q deletion: %w", runID, err)
	}
	var reason RunNotDeletableReason
	switch {
	case !RunLifecycleTerminal.Includes(state):
		reason = RunNotTerminal
	case releasePending:
		reason = RunAllocationReleasePending
	case collectionPending:
		reason = RunAuditCollectionPending
	default:
		return nil, nil
	}
	return &reason, nil
}
