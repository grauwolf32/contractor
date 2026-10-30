package pgstore

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/jackc/pgx/v5"
)

// ReadAllocationBaseline reads one immutable baseline through the exact child
// execution and Run association. It cannot cross owner or Project boundaries.
// It does not filter Audit lifecycle: already-submitted Runs must retain their
// pins while paused or draining. Run/Project cancellation and the live Registry
// grant recheck own admission; reading pinned data grants no new execution.
func (s *QueryStore) ReadAllocationBaseline(ctx context.Context, ownerID, projectID, executionID, runID string) (string, json.RawMessage, error) {
	var auditID string
	var baseline json.RawMessage
	err := s.db.QueryRow(ctx, `SELECT a.audit_id, a.baseline_snapshot
 FROM audit_executions e JOIN audits a ON a.audit_id=e.audit_id
 WHERE e.execution_id=$1 AND e.run_id=$2 AND a.owner_id=$3 AND a.project_id=$4`, executionID, runID, ownerID, projectID).Scan(&auditID, &baseline)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", nil, auditstore.ErrNotFound
	}
	if err != nil {
		return "", nil, fmt.Errorf("read allocation Audit baseline: %w", err)
	}
	return auditID, baseline, nil
}
