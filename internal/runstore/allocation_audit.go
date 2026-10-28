package runstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5"
)

// AllocationAuditBinding is the narrow immutable Run ownership needed to read
// an Audit allocation's baseline. It contains no Workflow or Stage snapshots.
type AllocationAuditBinding struct {
	OwnerID     string
	ProjectID   *string
	ExecutionID *string
}

// ReadAllocationAuditBinding proves the Stage belongs to this Run before a
// private prepare resolves Audit metadata. Ordinary Runs have no execution.
func (s *PostgresStore) ReadAllocationAuditBinding(ctx context.Context, runID, stageID string) (AllocationAuditBinding, error) {
	var result AllocationAuditBinding
	var mode OutputPublicationMode
	err := s.db.QueryRow(ctx, `SELECT r.owner_id, r.project_id, r.audit_execution_id, r.publication_mode
 FROM workflow_runs r JOIN stage_executions s ON s.run_id=r.run_id
 WHERE r.run_id=$1 AND s.stage_execution_id=$2`, runID, stageID).Scan(&result.OwnerID, &result.ProjectID, &result.ExecutionID, &mode)
	if errors.Is(err, pgx.ErrNoRows) {
		return AllocationAuditBinding{}, ErrNotFound
	}
	if err != nil {
		return AllocationAuditBinding{}, fmt.Errorf("read allocation Audit binding: %w", err)
	}
	if !mode.Valid() || (mode == PublicationAuditManaged) != (result.ExecutionID != nil) || (result.ExecutionID != nil && result.ProjectID == nil) {
		return AllocationAuditBinding{}, invalidf("allocation Audit binding is invalid")
	}
	return result, nil
}
