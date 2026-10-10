package auditstore

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
)

// LockRunDeletionAudits locks FOR UPDATE, in audit_id order, the owner's
// Audits that refer to Run runID through an AuditExecution, a finding proposal
// receipt or a proposal Audit hold, and returns their IDs. internal/rundeletion
// takes these locks after the Run lock and before any execution, retention or
// Artifact lock, matching ImportIntoAudit and Audit purge. The locks must be
// held until the Run is deleted, so the store must be transaction-backed.
func (s *PostgresStore) LockRunDeletionAudits(
	ctx context.Context, ownerID, runID string,
) ([]string, error) {
	tx, ok := s.db.(pgx.Tx)
	if !ok {
		return nil, fmt.Errorf("lock deleted WorkflowRun %q Audit projections: a transaction-backed store is required", runID)
	}
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

// MarkExecutionRunDeleted stamps the deletion of Run runID on its collected
// AuditExecution executionID. It reports false and changes nothing while that
// execution is not collected with Run provenance and a collection receipt,
// which means the Audit-managed Run may not be deleted yet.
func (s *PostgresStore) MarkExecutionRunDeleted(
	ctx context.Context, executionID, runID string,
) (bool, error) {
	tag, err := s.db.Exec(ctx, markExecutionRunDeletedSQL, executionID, runID)
	if err != nil {
		return false, fmt.Errorf("mark WorkflowRun %q Audit tombstone: %w", runID, err)
	}
	return tag.RowsAffected() == 1, nil
}

// CollectedExecutionRuns reports, without locking, which AuditExecutions
// pass the collection gate of MarkExecutionRunDeleted for the Run each one
// maps to. internal/rundeletion passes at most one Run page. The answer is
// advisory.
func (s *PostgresStore) CollectedExecutionRuns(
	ctx context.Context, runByExecution map[string]string,
) (map[string]bool, error) {
	result := make(map[string]bool, len(runByExecution))
	if len(runByExecution) == 0 {
		return result, nil
	}
	executionIDs := make([]string, 0, len(runByExecution))
	runIDs := make([]string, 0, len(runByExecution))
	for executionID, runID := range runByExecution {
		executionIDs = append(executionIDs, executionID)
		runIDs = append(runIDs, runID)
	}
	rows, err := s.db.Query(ctx, collectedExecutionRunsSQL, executionIDs, runIDs)
	if err != nil {
		return nil, fmt.Errorf("check WorkflowRun Audit collection: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var executionID string
		if err := rows.Scan(&executionID); err != nil {
			return nil, fmt.Errorf("scan WorkflowRun Audit collection: %w", err)
		}
		result[executionID] = true
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("check WorkflowRun Audit collection: %w", err)
	}
	return result, nil
}

// InvalidateDeletedRunProjections advances the revisions of Audits whose
// external Run availability changed. It deliberately does not append an
// Audit event: no finding assessment or review subject changed. A finalizing
// Audit's updated_at remains frozen for its immutable report snapshot.
func (s *PostgresStore) InvalidateDeletedRunProjections(
	ctx context.Context, auditIDs []string,
) error {
	_, err := s.db.Exec(ctx, `
UPDATE audits
   SET revision = revision + 1,
       updated_at = CASE WHEN state = 'finalizing' THEN updated_at
           ELSE GREATEST(clock_timestamp(), updated_at + interval '1 microsecond') END
 WHERE audit_id = ANY($1::text[])`, auditIDs)
	return err
}
