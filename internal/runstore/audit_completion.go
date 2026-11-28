package runstore

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/auditcontract"
	"github.com/grauwolf32/contractor/internal/helpers/contentdigest"
)

// AuditCompletionSnapshot contains only Server-derived RunScope authority.
// It is written inside the child Run creation transaction after input forks.
type AuditCompletionSnapshot struct {
	Finding  *contracts.FindingCheck            `json:"finding,omitempty"`
	Stage    string                             `json:"stage"`
	Agent    string                             `json:"agent"`
	Contract contracts.WorkerCompletionContract `json:"contract"`
}

func (s AuditCompletionSnapshot) Validate() error {
	if contracts.ValidateArtifactName(s.Stage) != nil || contracts.ValidateArtifactName(s.Agent) != nil {
		return invalidf("invalid Audit completion target")
	}
	if s.Finding != nil {
		if err := s.Finding.Validate(); err != nil {
			return err
		}
	}
	return s.Contract.Validate()
}

func (s *PostgresStore) SetAuditCompletion(ctx context.Context, runID string, completion AuditCompletionSnapshot) error {
	if err := completion.Validate(); err != nil {
		return err
	}
	if completion.Finding != nil {
		run, err := s.GetRun(ctx, runID)
		if err != nil {
			return err
		}
		if run.ProjectID == nil {
			return ErrInvalid
		}
		scoped, err := artifacts.NewService(s.storage.Repository(s.db)).Run(runID)
		if err != nil {
			return err
		}
		read, err := scoped.Read(ctx, completion.Contract.Context)
		if err != nil {
			return err
		}
		digest, err := contentdigest.CanonicalJSON(run.WorkflowSnapshot)
		if err != nil {
			return err
		}
		if read.Payload.MediaType != "application/json" || auditcontract.ValidateFindingContext(read.Payload.Data, *completion.Finding, runID, *run.ProjectID, run.WorkflowName+"@"+run.WorkflowVersion, digest) != nil {
			return ErrInvalid
		}
	}
	encoded, err := json.Marshal(completion)
	if err != nil {
		return err
	}
	tag, err := s.db.Exec(ctx, `UPDATE workflow_runs SET audit_completion = $2::jsonb
 WHERE run_id = $1 AND ((publication_mode = 'audit-managed' AND NOT $2::jsonb ? 'finding') OR
 (publication_mode = 'ordinary' AND project_id IS NOT NULL AND $2::jsonb ? 'finding')) AND state = 'initializing'
 AND audit_completion IS NULL`, runID, encoded)
	if err != nil {
		return fmt.Errorf("pin Audit completion: %w", err)
	}
	if tag.RowsAffected() != 1 {
		return ErrConflict
	}
	return nil
}
