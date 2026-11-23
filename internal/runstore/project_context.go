package runstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/jackc/pgx/v5"
)

// ProjectContext is the immutable owning Project identity of one durable Run.
// It contains no Run snapshot, Artifact descriptors or payloads.
type ProjectContext struct {
	OwnerID   string
	ProjectID string
}

func (s *PostgresStore) ProjectContext(ctx context.Context, runID string) (ProjectContext, error) {
	if !contracts.ValidResourceID(runID) {
		return ProjectContext{}, invalidf("Run ID is invalid")
	}
	var result ProjectContext
	err := s.db.QueryRow(ctx, `SELECT owner_id, project_id FROM workflow_runs WHERE run_id=$1 AND project_id IS NOT NULL`, runID).Scan(&result.OwnerID, &result.ProjectID)
	if errors.Is(err, pgx.ErrNoRows) {
		return ProjectContext{}, ErrNotFound
	}
	if err != nil {
		return ProjectContext{}, fmt.Errorf("read Run Project context: %w", err)
	}
	return result, nil
}
