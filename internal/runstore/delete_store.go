package runstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/pagelimit"
	"github.com/jackc/pgx/v5"
)

// RunDeletionGates are the Run-owned facts that decide whether a Run may be
// hard-deleted. internal/rundeletion reads them and composes the deletion
// with the Audit, finding and Artifact state that refers to the Run.
type RunDeletionGates struct {
	State                    WorkflowRunState
	PublicationMode          OutputPublicationMode
	AuditExecutionID         *string
	AllocationReleasePending bool
}

// LockRunForDeletion locks owner-visible Run runID FOR UPDATE and reads its
// deletion gates. The lock is held until DeleteLockedRun in the same
// transaction.
func (s *PostgresTxStore) LockRunForDeletion(
	ctx context.Context, ownerID, runID string,
) (RunDeletionGates, error) {
	if err := validateRunDeletionIdentity(ownerID, runID); err != nil {
		return RunDeletionGates{}, err
	}
	var gates RunDeletionGates
	err := s.tx.QueryRow(ctx, lockRunForDeletionSQL, runID, ownerID).Scan(
		&gates.State, &gates.PublicationMode, &gates.AuditExecutionID,
	)
	if errors.Is(err, pgx.ErrNoRows) {
		return RunDeletionGates{}, fmt.Errorf("delete WorkflowRun %q: %w", runID, ErrNotFound)
	}
	if err != nil {
		return RunDeletionGates{}, fmt.Errorf("lock WorkflowRun %q for deletion: %w", runID, err)
	}
	if err := s.tx.QueryRow(ctx, runAllocationReleasePendingSQL, runID).Scan(&gates.AllocationReleasePending); err != nil {
		return RunDeletionGates{}, fmt.Errorf("check WorkflowRun %q allocation release: %w", runID, err)
	}
	return gates, nil
}

// ReadRunDeletionGates reads the same gates as LockRunForDeletion, without a
// lock, for at most one public page (pagelimit.MaxRows) of the owner's Runs. Runs that do not exist
// or belong to another owner are absent from the result. The answer is
// advisory; deletion repeats the read under the lock.
func (s *PostgresStore) ReadRunDeletionGates(
	ctx context.Context, ownerID string, runIDs []string,
) (map[string]RunDeletionGates, error) {
	if err := validateOpaque("ownerID", ownerID); err != nil {
		return nil, err
	}
	if len(runIDs) > pagelimit.MaxRows {
		return nil, invalidf("at most %d Runs can be inspected for deletion at once", pagelimit.MaxRows)
	}
	for _, runID := range runIDs {
		if err := validateOpaque("runID", runID); err != nil {
			return nil, err
		}
	}
	result := make(map[string]RunDeletionGates, len(runIDs))
	if len(runIDs) == 0 {
		return result, nil
	}
	rows, err := s.db.Query(ctx, readRunDeletionGatesSQL, ownerID, runIDs)
	if err != nil {
		return nil, fmt.Errorf("inspect WorkflowRun deletion: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var runID string
		var gates RunDeletionGates
		if err := rows.Scan(
			&runID, &gates.State, &gates.PublicationMode, &gates.AuditExecutionID,
			&gates.AllocationReleasePending,
		); err != nil {
			return nil, fmt.Errorf("scan WorkflowRun deletion gates: %w", err)
		}
		result[runID] = gates
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("inspect WorkflowRun deletion: %w", err)
	}
	return result, nil
}

// DeleteLockedRun deletes the Run row locked by LockRunForDeletion in the
// same transaction; Run-owned rows cascade through their foreign keys.
func (s *PostgresTxStore) DeleteLockedRun(ctx context.Context, ownerID, runID string) error {
	if err := validateRunDeletionIdentity(ownerID, runID); err != nil {
		return err
	}
	tag, err := s.tx.Exec(ctx, `
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

func validateRunDeletionIdentity(ownerID, runID string) error {
	if err := validateOpaque("ownerID", ownerID); err != nil {
		return err
	}
	return validateOpaque("runID", runID)
}
