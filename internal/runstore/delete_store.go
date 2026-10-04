package runstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditstore"
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
	// ImportIntoAudit holds the source Run FOR KEY SHARE before creating a
	// destination hold. Our Run lock makes this destination set stable. Take
	// Audit locks before execution, retention and Artifact locks, matching
	// import and Audit purge rather than adding an Audit lock at the end.
	auditIDs, err := lockRunDeletionAudits(ctx, tx, ownerID, runID)
	if err != nil {
		return err
	}
	if publicationMode == PublicationAuditManaged {
		if auditExecutionID == nil {
			return fmt.Errorf("delete WorkflowRun %q: invalid Audit authority", runID)
		}
		var collected bool
		if err := tx.QueryRow(ctx, runAuditCollectedSQL, *auditExecutionID, runID).Scan(&collected); err != nil {
			return fmt.Errorf("check WorkflowRun %q Audit collection: %w", runID, err)
		}
		if !collected {
			return &RunNotDeletableError{RunID: runID, Reason: RunAuditCollectionPending}
		}
		tag, err := tx.Exec(ctx, markAuditRunDeletedSQL, *auditExecutionID, runID)
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
	if _, err := tx.Exec(ctx, markProposalSourceRunDeletedSQL, runID); err != nil {
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
	// Only external Run availability changed: finding assessments and pending
	// review subjects keep their own revisions. The Audit revision invalidates
	// every enclosing read/cursor, including terminal Audits.
	// While finalizing, UpdatedAt is also the frozen report's generatedFrom.
	// Keep it stable so a retry after a failed revision CAS reuses the exact
	// immutable report bytes; Run availability is absent from that payload.
	if err := auditstore.NewPostgresStore(tx).InvalidateDeletedRunProjections(ctx, auditIDs); err != nil {
		return fmt.Errorf("invalidate deleted WorkflowRun %q Audit projections: %w", runID, err)
	}
	return nil
}

func lockRunDeletionAudits(ctx context.Context, tx pgx.Tx, ownerID, runID string) ([]string, error) {
	rows, err := tx.Query(ctx, lockRunDeletionAuditsSQL, ownerID, runID)
	if err != nil {
		return nil, fmt.Errorf("lock deleted WorkflowRun %q Audit projections: %w", runID, err)
	}
	defer rows.Close()
	ids := make([]string, 0)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("read deleted WorkflowRun %q Audit identity: %w", runID, err)
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("read deleted WorkflowRun %q Audit identities: %w", runID, err)
	}
	return ids, nil
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
	err := s.db.QueryRow(ctx, runDeletionBlockerSQL, runID, ownerID).Scan(
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
