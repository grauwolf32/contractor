package runstore

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// AuditCompletionSnapshot contains only Server-derived RunScope authority.
// It is written inside the child Run creation transaction after input forks.
type AuditCompletionSnapshot struct {
	Stage    string                             `json:"stage"`
	Agent    string                             `json:"agent"`
	Contract contracts.WorkerCompletionContract `json:"contract"`
}

func (s AuditCompletionSnapshot) Validate() error {
	if contracts.ValidateArtifactName(s.Stage) != nil || contracts.ValidateArtifactName(s.Agent) != nil {
		return invalidf("invalid Audit completion target")
	}
	return s.Contract.Validate()
}

func (s *PostgresStore) SetAuditCompletion(ctx context.Context, runID string, completion AuditCompletionSnapshot) error {
	if err := completion.Validate(); err != nil {
		return err
	}
	encoded, err := json.Marshal(completion)
	if err != nil {
		return err
	}
	tag, err := s.db.Exec(ctx, `UPDATE workflow_runs SET audit_completion = $2::jsonb
 WHERE run_id = $1 AND publication_mode = 'audit-managed' AND state = 'initializing'
 AND audit_completion IS NULL`, runID, encoded)
	if err != nil {
		return fmt.Errorf("pin Audit completion: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return ErrConflict
	}
	return nil
}
