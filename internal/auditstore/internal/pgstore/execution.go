package pgstore

import (
	"context"
	"errors"
	"fmt"

	"github.com/grauwolf32/contractor/internal/auditstore"
	persistencepostgres "github.com/grauwolf32/contractor/internal/persistence/postgres"
	"github.com/grauwolf32/contractor/internal/settingsstore"
	"github.com/jackc/pgx/v5"
)

func (s *ClaimStore) CreateExecutionIntent(
	ctx context.Context,
	params auditstore.CreateExecutionIntentParams,
) (auditstore.Execution, bool, error) {
	if err := validateExecutionIntent(params); err != nil {
		return auditstore.Execution{}, false, err
	}
	if replay, found, err := s.lookupExecutionReplay(
		ctx, params.Claim.AuditID, params.SubmissionKey, params.RequestDigest,
	); err != nil || found {
		return replay, false, err
	}
	prepared := prepareExecutionIntentWrite(params)
	var execution auditstore.Execution
	err := persistencepostgres.Within(ctx, s.db, pgx.TxOptions{}, func(tx pgx.Tx) error {
		// The reservation holds the Scheduler settings row until it commits,
		// so a settings change is ordered before or after it (AUX-R6), and
		// takes that lock before its claim, Audit and Round locks.
		settings, err := settingsstore.NewPostgresTxStore(tx).LockSchedulerSettings(ctx)
		if err != nil {
			return fmt.Errorf("lock Audit dispatch window: %w", err)
		}
		execution, err = scanExecution(persistencepostgres.SavepointQueryRow(ctx, tx, createExecutionIntentSQL,
			params.Claim.AuditID, params.Claim.HolderID, params.Claim.Epoch,
			params.ExecutionID, string(params.Role), prepared.roundID, prepared.roleAttempt,
			prepared.manifestRef, params.Manifest.Digest, params.SubmissionKey,
			params.RequestDigest, prepared.members,
			params.WorkflowRole, prepared.preparation, settings.MaxConcurrentRuns,
		))
		return err
	})
	if err == nil {
		return execution, true, nil
	}
	failure := persistencepostgres.Classify(err)
	if failure == persistencepostgres.ClassDeleting {
		return auditstore.Execution{}, false, auditstore.ErrProjectDeleting
	}
	if params.Role == auditstore.ExecutionPrepare && failure == persistencepostgres.ClassRejected {
		return auditstore.Execution{}, false, auditstore.ErrPrecondition
	}
	if failure == persistencepostgres.ClassDuplicate || errors.Is(err, pgx.ErrNoRows) {
		if existing, found, replayErr := s.lookupExecutionReplay(
			ctx, params.Claim.AuditID, params.SubmissionKey, params.RequestDigest,
		); replayErr != nil || found {
			return existing, false, replayErr
		}
		if live, liveErr := s.claimLive(ctx, params.Claim); liveErr != nil {
			return auditstore.Execution{}, false, liveErr
		} else if !live {
			return auditstore.Execution{}, false, auditstore.ErrClaimLost
		}
		if failure == persistencepostgres.ClassDuplicate {
			return auditstore.Execution{}, false, auditstore.ErrConflict
		}
		return auditstore.Execution{}, false, auditstore.ErrPrecondition
	}
	return auditstore.Execution{}, false, fmt.Errorf("create Audit execution intent: %w", err)
}

func (s *ClaimStore) lookupExecutionReplay(
	ctx context.Context, auditID, submissionKey, digest string,
) (auditstore.Execution, bool, error) {
	execution, err := scanExecution(s.db.QueryRow(ctx, `
SELECT `+prefixedExecutionColumns("audit_executions")+` FROM audit_executions
 WHERE submission_key = $1`, submissionKey))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Execution{}, false, nil
	}
	if err != nil {
		return auditstore.Execution{}, false, fmt.Errorf("resolve execution submission replay: %w", err)
	}
	if execution.AuditID != auditID || execution.RequestDigest != digest {
		return auditstore.Execution{}, true, auditstore.ErrConflict
	}
	return execution, true, nil
}

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
	execution, err := scanExecution(s.db.QueryRow(ctx, bindRunSQL+prefixedExecutionColumns("changed")+` FROM changed`,
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
	var result auditstore.RunCreationIntent
	execution, err := scanExecution(withColumns{row: s.db.QueryRow(ctx, `
SELECT audit.owner_id, audit.project_id, `+profileRoleSQL("#>", "execution.workflow_role")+`, `+prefixedExecutionColumns("execution")+`
  FROM audit_executions AS execution
  JOIN audits AS audit USING (audit_id)
  JOIN audit_controller_claims AS claim USING (audit_id)
 WHERE execution.audit_id = $1 AND execution.execution_id = $4
   AND `+auditClaims.LiveCondition("claim", "$2", "$3")+`
 FOR UPDATE OF claim, audit, execution`,
		claim.AuditID, claim.HolderID, claim.Epoch, executionID,
	), before: []any{&result.OwnerID, &result.ProjectID, &result.WorkflowBinding}})
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
	result.Execution = execution
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
	execution, err := scanExecution(s.db.QueryRow(ctx, observeTerminalSQL+prefixedExecutionColumns("changed")+` FROM changed`,
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
	execution, err := scanExecution(s.db.QueryRow(ctx, observeSubmissionFailureSQL+prefixedExecutionColumns("changed")+` FROM changed`,
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
	execution, err := scanExecution(s.db.QueryRow(ctx, `
SELECT `+prefixedExecutionColumns("audit_executions")+` FROM audit_executions
 WHERE audit_id = $1 AND execution_id = $2`, auditID, executionID))
	if errors.Is(err, pgx.ErrNoRows) {
		return auditstore.Execution{}, auditstore.ErrNotFound
	}
	if err != nil {
		return auditstore.Execution{}, fmt.Errorf("read Audit execution: %w", err)
	}
	return execution, nil
}
