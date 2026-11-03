package pgstore

import (
	"context"
	"errors"
	"fmt"

	"slices"

	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/runlabels"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/projectstore"
	"github.com/jackc/pgx/v5"
)

func (s *ClaimStore) claimLive(ctx context.Context, claim auditstore.ControllerClaim) (bool, error) {
	live, err := auditClaims.Live(ctx, s.db, claim.AuditID, claim.HolderID, int64(claim.Epoch))
	if err != nil {
		return false, fmt.Errorf("check Audit claim: %w", err)
	}
	return live, nil
}

func (s *ClaimStore) BindRun(ctx context.Context, params auditstore.BindRunParams) (auditstore.Execution, error) {
	if err := validateBindRun(params); err != nil {
		return auditstore.Execution{}, err
	}
	execution, err := ScanExecution(s.db.QueryRow(ctx, bindRunSQL+ExecutionColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExecutionID, params.RunID,
	))
	if err == nil {
		return execution, nil
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDeleting {
		return auditstore.Execution{}, auditstore.ErrProjectDeleting
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		if persistencepostgres.Classify(err) == persistencepostgres.ClassDuplicate {
			return auditstore.Execution{}, auditstore.ErrConflict
		}
		return auditstore.Execution{}, fmt.Errorf("bind Audit execution Run: %w", err)
	}
	existing, getErr := s.getExecution(ctx, params.Claim.AuditID, params.ExecutionID)
	if getErr == nil && existing.RunID != nil && *existing.RunID == params.RunID && existing.State != auditstore.ExecutionIntent {
		return existing, nil
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return auditstore.Execution{}, liveErr
	} else if !live {
		return auditstore.Execution{}, auditstore.ErrClaimLost
	}
	if getErr != nil {
		return auditstore.Execution{}, getErr
	}
	return auditstore.Execution{}, auditstore.ErrPrecondition
}

// GetRunCreationIntent locks the exact claim, Audit and Execution used by one
// trusted child-Run submission. Submitted executions remain readable so a
// response-loss replay can return the already-associated Run without touching
// mutable configuration.
func (s *ClaimStore) GetRunCreationIntent(
	ctx context.Context,
	claim auditstore.ControllerClaim,
	executionID string,
) (auditstore.RunCreationIntent, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return auditstore.RunCreationIntent{}, err
	}
	if err := validateID("executionID", executionID); err != nil {
		return auditstore.RunCreationIntent{}, err
	}
	// Run creation and Project deletion share the same lock order. The
	// production caller keeps this store and the Run writer on one transaction.
	var ownerID, projectID string
	if err := s.db.QueryRow(ctx, `SELECT owner_id,project_id FROM audits WHERE audit_id=$1`, claim.AuditID).Scan(&ownerID, &projectID); err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return auditstore.RunCreationIntent{}, auditstore.ErrNotFound
		}
		return auditstore.RunCreationIntent{}, err
	}
	if _, err := projectstore.NewPostgresStore(s.db).LockActiveAuditProject(ctx, ownerID, projectID); err != nil {
		return auditstore.RunCreationIntent{}, err
	}
	var result auditstore.RunCreationIntent
	var placement []byte
	var metadata []byte
	execution, err := ScanExecution(withColumns{row: s.db.QueryRow(ctx, `
SELECT audit.state, audit.dispatch_state, audit.deadline_at, audit.owner_id, audit.project_id, audit.baseline_snapshot->'labels', audit.baseline_snapshot->'requiredPlacementLabels', audit.profile_snapshot, audit.baseline_snapshot, (audit.state='active' AND audit.dispatch_state='open' AND (audit.deadline_at IS NULL OR audit.deadline_at>clock_timestamp())), `+profileRoleSQL("#>", "execution.workflow_role")+`, `+ExecutionColumns("execution")+`
  FROM audit_executions AS execution
  JOIN audits AS audit USING (audit_id)
  JOIN audit_controller_claims AS claim USING (audit_id)
 WHERE execution.audit_id = $1 AND execution.execution_id = $4
   AND `+auditClaims.LiveCondition("claim", "$2", "$3")+`
 FOR UPDATE OF claim, audit, execution`,
		claim.AuditID, claim.HolderID, claim.Epoch, executionID,
	), before: []any{&result.AuditState, &result.DispatchState, &result.DeadlineAt, &result.OwnerID, &result.ProjectID, &metadata, &placement, &result.ProgramSnapshot, &result.BaselineSnapshot, &result.CanDispatch, &result.WorkflowBinding}})
	if errors.Is(err, pgx.ErrNoRows) {
		if live, liveErr := s.claimLive(ctx, claim); liveErr != nil {
			return auditstore.RunCreationIntent{}, liveErr
		} else if !live {
			return auditstore.RunCreationIntent{}, auditstore.ErrClaimLost
		}
		return auditstore.RunCreationIntent{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.RunCreationIntent{}, fmt.Errorf("read Audit Run creation intent: %w", err)
	}
	if strictjson.Decode(placement, &result.RequiredPlacementLabels) != nil || result.RequiredPlacementLabels == nil ||
		contracts.ValidatePlacementLabels(result.RequiredPlacementLabels) != nil {
		return auditstore.RunCreationIntent{}, invalidf("stored Audit placement requirements are invalid")
	}
	if strictjson.Decode(metadata, &result.Labels) != nil || result.Labels == nil {
		return auditstore.RunCreationIntent{}, invalidf("stored Audit labels are invalid")
	}
	labels, labelErr := runlabels.NormalizeAuditLabels(result.Labels)
	if labelErr != nil {
		return auditstore.RunCreationIntent{}, invalidf("stored Audit labels are invalid")
	}
	derived, _ := labels.PlacementRequirements()
	if !slices.Equal(derived, result.RequiredPlacementLabels) {
		return auditstore.RunCreationIntent{}, invalidf("stored Audit placement requirements differ from labels")
	}
	result.Execution = execution
	if execution.Role == auditstore.ExecutionTask {
		ids, err := taskExecutionIDs(ctx, s.db, execution.AuditID, execution.ExecutionID)
		if err != nil {
			return auditstore.RunCreationIntent{}, err
		}
		result.Tasks, err = loadSelectedTasks(ctx, s.db, auditstore.Audit{AuditID: execution.AuditID}, ids)
		if err != nil {
			return auditstore.RunCreationIntent{}, err
		}
		return result, nil
	}
	result.Items, err = s.listExecutionItems(ctx, executionID)
	if err != nil {
		return auditstore.RunCreationIntent{}, err
	}
	return result, nil
}

func (s *ClaimStore) ObserveTerminal(
	ctx context.Context,
	params auditstore.ObserveTerminalParams,
) (auditstore.Execution, error) {
	if err := validateObserveTerminal(params); err != nil {
		return auditstore.Execution{}, err
	}
	execution, err := ScanExecution(s.db.QueryRow(ctx, observeTerminalSQL+ExecutionColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExecutionID, params.RunID, params.Generation, int64(params.Sequence),
	))
	if err == nil {
		return execution, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Execution{}, fmt.Errorf("observe terminal Audit Run: %w", err)
	}
	existing, getErr := s.getExecution(ctx, params.Claim.AuditID, params.ExecutionID)
	if getErr == nil && (existing.State == auditstore.ExecutionCollecting || existing.State == auditstore.ExecutionCollected) && existing.RunID != nil &&
		*existing.RunID == params.RunID && existing.TerminalRunGeneration != nil &&
		*existing.TerminalRunGeneration == params.Generation && existing.TerminalRunSequence != nil &&
		*existing.TerminalRunSequence == params.Sequence {
		return existing, nil
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return auditstore.Execution{}, liveErr
	} else if !live {
		return auditstore.Execution{}, auditstore.ErrClaimLost
	}
	if getErr != nil {
		return auditstore.Execution{}, getErr
	}
	return auditstore.Execution{}, auditstore.ErrPrecondition
}

func (s *ClaimStore) ObserveSubmissionFailure(
	ctx context.Context,
	params auditstore.ObserveSubmissionFailureParams,
) (auditstore.Execution, error) {
	if err := validateSubmissionFailure(params); err != nil {
		return auditstore.Execution{}, err
	}
	execution, err := ScanExecution(s.db.QueryRow(ctx, observeSubmissionFailureSQL+ExecutionColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch, params.ExecutionID,
	))
	if err == nil {
		return execution, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Execution{}, fmt.Errorf("observe Audit submission failure: %w", err)
	}
	existing, getErr := s.getExecution(ctx, params.Claim.AuditID, params.ExecutionID)
	if getErr == nil && (existing.State == auditstore.ExecutionCollecting || existing.State == auditstore.ExecutionCollected) && existing.TerminalOutcome != nil &&
		*existing.TerminalOutcome == auditstore.TerminalSubmissionFailed {
		return existing, nil
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return auditstore.Execution{}, liveErr
	} else if !live {
		return auditstore.Execution{}, auditstore.ErrClaimLost
	}
	if getErr != nil {
		return auditstore.Execution{}, getErr
	}
	return auditstore.Execution{}, auditstore.ErrPrecondition
}

func (s *ClaimStore) getExecution(ctx context.Context, auditID, executionID string) (auditstore.Execution, error) {
	execution, err := ScanExecution(s.db.QueryRow(ctx, `
SELECT `+ExecutionColumns("audit_executions")+` FROM audit_executions
 WHERE audit_id = $1 AND execution_id = $2`, auditID, executionID))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Execution{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.Execution{}, fmt.Errorf("read Audit execution: %w", err)
	}
	return execution, nil
}
