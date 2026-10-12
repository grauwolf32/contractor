package auditstore

import (
	"context"
	"errors"
	"fmt"

	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/jackc/pgx/v5"
)

func (s *PostgresStore) CreateExecutionIntent(
	ctx context.Context,
	params CreateExecutionIntentParams,
) (Execution, bool, error) {
	if err := validateExecutionIntent(params); err != nil {
		return Execution{}, false, err
	}
	if replay, found, err := s.lookupExecutionReplay(
		ctx, params.Claim.AuditID, params.SubmissionKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	prepared := prepareExecutionIntentWrite(params)
	execution, err := scanExecution(persistencepostgres.SavepointQueryRow(ctx, s.db, createExecutionIntentSQL,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExecutionID, string(params.Role), prepared.roundID, prepared.roleAttempt,
		prepared.manifestRef, params.Manifest.Digest, params.SubmissionKey,
		params.RequestDigest, prepared.members,
		params.WorkflowRole, prepared.preparation,
	))
	if err == nil {
		return execution, true, nil
	}
	failure := persistencepostgres.Classify(err)
	if failure == persistencepostgres.ClassDeleting {
		return Execution{}, false, ErrProjectDeleting
	}
	if params.Role == ExecutionPrepare && failure == persistencepostgres.ClassRejected {
		return Execution{}, false, ErrPrecondition
	}
	if failure == persistencepostgres.ClassDuplicate || errors.Is(err, pgx.ErrNoRows) {
		if existing, found, replayErr := s.lookupExecutionReplay(
			ctx, params.Claim.AuditID, params.SubmissionKey, params.RequestDigest,
		); replayErr != nil || found {
			return existing, false, replayErr
		}
		if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
			return Execution{}, false, liveErr
		} else if !live {
			return Execution{}, false, ErrClaimLost
		}
		if failure == persistencepostgres.ClassDuplicate {
			return Execution{}, false, ErrConflict
		}
		return Execution{}, false, ErrPrecondition
	}
	return Execution{}, false, fmt.Errorf("create Audit execution intent: %w", err)
}

func (s *PostgresStore) lookupExecutionReplay(
	ctx context.Context, auditID, submissionKey, digest string,
) (Execution, bool, error) {
	execution, err := scanExecution(s.db.QueryRow(ctx, `
SELECT `+prefixedExecutionColumns("audit_executions")+` FROM audit_executions
 WHERE submission_key = $1`, submissionKey))
	if errors.Is(err, pgx.ErrNoRows) {
		return Execution{}, false, nil
	}
	if err != nil {
		return Execution{}, false, fmt.Errorf("resolve execution submission replay: %w", err)
	}
	if execution.AuditID != auditID || execution.RequestDigest != digest {
		return Execution{}, true, ErrConflict
	}
	return execution, true, nil
}

func (s *PostgresStore) claimLive(ctx context.Context, claim ControllerClaim) (bool, error) {
	var live bool
	err := s.db.QueryRow(ctx, `
SELECT EXISTS (
    SELECT 1 FROM audit_controller_claims
     WHERE audit_id = $1 AND holder_id = $2 AND epoch = $3
       AND expires_at > clock_timestamp()
)`, claim.AuditID, claim.HolderID, claim.Epoch).Scan(&live)
	if err != nil {
		return false, fmt.Errorf("check Audit claim: %w", err)
	}
	return live, nil
}

func (s *PostgresStore) BindRun(ctx context.Context, params BindRunParams) (Execution, error) {
	if err := validateBindRun(params); err != nil {
		return Execution{}, err
	}
	execution, err := scanExecution(s.db.QueryRow(ctx, bindRunSQL+prefixedExecutionColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExecutionID, params.RunID,
	))
	if err == nil {
		return execution, nil
	}
	if persistencepostgres.Classify(err) == persistencepostgres.ClassDeleting {
		return Execution{}, ErrProjectDeleting
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		if persistencepostgres.Classify(err) == persistencepostgres.ClassDuplicate {
			return Execution{}, ErrConflict
		}
		return Execution{}, fmt.Errorf("bind Audit execution Run: %w", err)
	}
	existing, getErr := s.getExecution(ctx, params.Claim.AuditID, params.ExecutionID)
	if getErr == nil && existing.RunID != nil && *existing.RunID == params.RunID && existing.State != ExecutionIntent {
		return existing, nil
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return Execution{}, liveErr
	} else if !live {
		return Execution{}, ErrClaimLost
	}
	if getErr != nil {
		return Execution{}, getErr
	}
	return Execution{}, ErrPrecondition
}

// GetRunCreationIntent locks the exact claim, Audit and Execution used by one
// trusted child-Run submission. Submitted executions remain readable so a
// response-loss replay can return the already-associated Run without touching
// mutable configuration.
func (s *PostgresStore) GetRunCreationIntent(
	ctx context.Context,
	claim ControllerClaim,
	executionID string,
) (RunCreationIntent, error) {
	if err := validateClaimIdentity(claim); err != nil {
		return RunCreationIntent{}, err
	}
	if err := validateID("executionID", executionID); err != nil {
		return RunCreationIntent{}, err
	}
	var result RunCreationIntent
	execution, err := scanExecution(withColumns{row: s.db.QueryRow(ctx, `
SELECT audit.owner_id, audit.project_id, `+profileRoleSQL("#>", "execution.workflow_role")+`, `+prefixedExecutionColumns("execution")+`
  FROM audit_executions AS execution
  JOIN audits AS audit USING (audit_id)
  JOIN audit_controller_claims AS claim USING (audit_id)
 WHERE execution.audit_id = $1 AND execution.execution_id = $4
   AND claim.holder_id = $2 AND claim.epoch = $3
   AND claim.expires_at > clock_timestamp()
 FOR UPDATE OF claim, audit, execution`,
		claim.AuditID, claim.HolderID, claim.Epoch, executionID,
	), before: []any{&result.OwnerID, &result.ProjectID, &result.WorkflowBinding}})
	if errors.Is(err, pgx.ErrNoRows) {
		if live, liveErr := s.claimLive(ctx, claim); liveErr != nil {
			return RunCreationIntent{}, liveErr
		} else if !live {
			return RunCreationIntent{}, ErrClaimLost
		}
		return RunCreationIntent{}, ErrNotFound
	}
	if err != nil {
		return RunCreationIntent{}, fmt.Errorf("read Audit Run creation intent: %w", err)
	}
	result.Execution = execution
	result.Items, err = s.ListExecutionItems(ctx, executionID)
	if err != nil {
		return RunCreationIntent{}, err
	}
	return result, nil
}

func (s *PostgresStore) ObserveTerminal(
	ctx context.Context,
	params ObserveTerminalParams,
) (Execution, error) {
	if err := validateObserveTerminal(params); err != nil {
		return Execution{}, err
	}
	execution, err := scanExecution(s.db.QueryRow(ctx, observeTerminalSQL+prefixedExecutionColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
		params.ExecutionID, params.RunID, params.Generation, int64(params.Sequence),
	))
	if err == nil {
		return execution, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Execution{}, fmt.Errorf("observe terminal Audit Run: %w", err)
	}
	existing, getErr := s.getExecution(ctx, params.Claim.AuditID, params.ExecutionID)
	if getErr == nil && (existing.State == ExecutionCollecting || existing.State == ExecutionCollected) && existing.RunID != nil &&
		*existing.RunID == params.RunID && existing.TerminalRunGeneration != nil &&
		*existing.TerminalRunGeneration == params.Generation && existing.TerminalRunSequence != nil &&
		*existing.TerminalRunSequence == params.Sequence {
		return existing, nil
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return Execution{}, liveErr
	} else if !live {
		return Execution{}, ErrClaimLost
	}
	if getErr != nil {
		return Execution{}, getErr
	}
	return Execution{}, ErrPrecondition
}

func (s *PostgresStore) ObserveSubmissionFailure(
	ctx context.Context,
	params ObserveSubmissionFailureParams,
) (Execution, error) {
	if err := validateSubmissionFailure(params); err != nil {
		return Execution{}, err
	}
	execution, err := scanExecution(s.db.QueryRow(ctx, observeSubmissionFailureSQL+prefixedExecutionColumns("changed")+` FROM changed`,
		params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch, params.ExecutionID,
	))
	if err == nil {
		return execution, nil
	}
	if !errors.Is(err, pgx.ErrNoRows) {
		return Execution{}, fmt.Errorf("observe Audit submission failure: %w", err)
	}
	existing, getErr := s.getExecution(ctx, params.Claim.AuditID, params.ExecutionID)
	if getErr == nil && (existing.State == ExecutionCollecting || existing.State == ExecutionCollected) && existing.TerminalOutcome != nil &&
		*existing.TerminalOutcome == TerminalSubmissionFailed {
		return existing, nil
	}
	if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
		return Execution{}, liveErr
	} else if !live {
		return Execution{}, ErrClaimLost
	}
	if getErr != nil {
		return Execution{}, getErr
	}
	return Execution{}, ErrPrecondition
}

func (s *PostgresStore) getExecution(ctx context.Context, auditID, executionID string) (Execution, error) {
	execution, err := scanExecution(s.db.QueryRow(ctx, `
SELECT `+prefixedExecutionColumns("audit_executions")+` FROM audit_executions
 WHERE audit_id = $1 AND execution_id = $2`, auditID, executionID))
	if errors.Is(err, pgx.ErrNoRows) {
		return Execution{}, ErrNotFound
	}
	if err != nil {
		return Execution{}, fmt.Errorf("read Audit execution: %w", err)
	}
	return execution, nil
}
