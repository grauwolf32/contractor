package runservice

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func validateRoundResultsAuthority(ctx context.Context, writer AuditExecutionWriter, intent auditstore.RunCreationIntent, params AuditCreateParams) error {
	if len(params.RoundResults) == 0 {
		return nil
	}
	if intent.Execution.Role != auditstore.ExecutionAssessment || intent.Execution.RoundID == nil {
		return fmt.Errorf("%w: round results require the assessment Round", ErrInvalid)
	}
	var binding auditprofile.ResolvedAuditWorkflowBinding
	if strictjson.Decode(intent.WorkflowBinding, &binding) != nil {
		return fmt.Errorf("%w: stored role binding is invalid", ErrInvalid)
	}
	accepted, err := writer.GetAcceptedRoundResults(ctx, intent.Execution.AuditID, *intent.Execution.RoundID)
	if err != nil {
		return err
	}
	for slot, source := range params.RoundResults {
		mapping, declared := binding.Inputs[slot]
		input, supplied := params.Inputs[slot]
		if !declared || mapping.Source != "round-results" || !supplied || !sameExactIdentity(source, input) ||
			!sameExactIdentity(source, accepted) || source.MediaType != accepted.MediaType || source.SizeBytes != accepted.SizeBytes {
			return fmt.Errorf("%w: round-results input is outside the accepted closure", auditstore.ErrConflict)
		}
	}
	return nil
}

func completeAuditInputSlice(ctx context.Context, writer AuditRunWriter, service *artifacts.Service, intent auditstore.RunCreationIntent, params AuditCreateParams, run runstore.WorkflowRun, now time.Time) (runstore.WorkflowRun, error) {
	if intent.AuditState != auditstore.AuditActive || intent.DispatchState != auditstore.DispatchOpen ||
		intent.DeadlineAt != nil && !now.Before(*intent.DeadlineAt) {
		return runstore.WorkflowRun{}, auditstore.ErrPrecondition
	}
	previous, err := writer.LockAuditInputMaterialization(ctx, run.RunID)
	if err != nil {
		return runstore.WorkflowRun{}, err
	}
	if intent.Execution.CreatedAt.IsZero() {
		return runstore.WorkflowRun{}, fmt.Errorf("%w: Audit input intent has no creation time", ErrInvalid)
	}
	deadline := intent.Execution.CreatedAt.UTC().Add(runstore.AuditInputMaterializationTimeout)
	state := runstore.AuditInputMaterialization{
		Identity: params.InputMaterializationID, DeadlineAt: deadline,
		Checkpoint: []byte(`{}`), Refs: []contracts.ArtifactRef{},
	}
	if previous != nil {
		state = *previous
		if state.Identity != params.InputMaterializationID {
			return runstore.WorkflowRun{}, fmt.Errorf("%w: Audit input identity changed during recovery", auditstore.ErrConflict)
		}
		if !state.DeadlineAt.Equal(deadline) {
			return runstore.WorkflowRun{}, fmt.Errorf("%w: Audit input deadline differs from its original intent", ErrInvalid)
		}
		if state.Complete {
			return run, nil
		}
	}
	if !now.Before(state.DeadlineAt) {
		return failAuditInputInitialization(ctx, writer, run)
	}
	next, err := params.MaterializeInputs(ctx, service, intent, run, state)
	if err != nil {
		// A PostgreSQL statement may have aborted this transaction. Roll it
		// back before recording a permanent failure of an existing child.
		return runstore.WorkflowRun{}, err
	}
	if next.Identity != state.Identity || !next.DeadlineAt.Equal(state.DeadlineAt) ||
		!next.Complete && bytes.Equal(next.Checkpoint, state.Checkpoint) && len(next.Refs) == len(state.Refs) {
		return runstore.WorkflowRun{}, fmt.Errorf("%w: Audit input materializer changed identity or made no progress", ErrInvalid)
	}
	next.Revision = state.Revision + 1
	if err := writer.SetAuditInputMaterialization(ctx, run.RunID, state.Revision, next); err != nil {
		return runstore.WorkflowRun{}, err
	}
	if !next.Complete {
		run.StateReason = runstore.Reason{Code: runstore.AuditInputInitializationPendingReason}
		return run, nil
	}
	if len(params.Skills) != 0 {
		run.StateReason = runstore.Reason{Code: runstore.SkillInitializationPendingReason}
		return run, nil
	}
	return writer.TransitionRun(ctx, run.RunID, runstore.RunInitializing, runstore.RunPending, runstore.Reason{Code: "initialized"})
}

// failBoundAuditInput reconciles only a previously committed association after
// the failed slice rolled back. A first creation that failed has no child and
// keeps the existing submission-failure path. This never retries target work.
func (s *Service) failBoundAuditInput(ctx context.Context, params AuditCreateParams) (CreateResult, bool, error) {
	var result CreateResult
	handled := false
	err := s.auditTransaction(ctx, func(runs AuditRunWriter, _ *artifacts.Service, audits AuditExecutionWriter) error {
		intent, err := audits.GetRunCreationIntent(ctx, params.Claim, params.ExecutionID)
		if err != nil {
			return err
		}
		if err := validateAuditIntent(intent, params); err != nil {
			return err
		}
		if intent.Execution.RunID == nil {
			return nil
		}
		run, err := runs.GetRun(ctx, *intent.Execution.RunID)
		if err != nil {
			return err
		}
		if !matchingAuditRun(run, intent) {
			return fmt.Errorf("%w: Audit input child association changed", auditstore.ErrConflict)
		}
		if run.State == runstore.RunInitializing {
			if intent.AuditState != auditstore.AuditActive || intent.DispatchState != auditstore.DispatchOpen {
				return auditstore.ErrPrecondition
			}
			run, err = failAuditInputInitialization(ctx, runs, run)
			if err != nil {
				return err
			}
		}
		result, handled = CreateResult{Run: run, Replayed: true}, true
		return nil
	})
	return result, handled, err
}

func permanentAuditInputFailure(err error) bool {
	var validation *auditdomain.ValidationError
	return errors.As(err, &validation) || errors.Is(err, auditstore.ErrInvalid) ||
		errors.Is(err, ErrInvalid) || errors.Is(err, runstore.ErrInvalid) ||
		errors.Is(err, contracts.ErrValidation) || errors.Is(err, contracts.ErrExactArtifactMismatch) || errors.Is(err, artifacts.ErrArtifactNotFound) ||
		errors.Is(err, artifacts.ErrInvalidScope) || errors.Is(err, artifacts.ErrArtifactIntegrity) ||
		errors.Is(err, artifacts.ErrInvalidName) || errors.Is(err, artifacts.ErrExactRevisionRequired) ||
		errors.Is(err, artifacts.ErrArtifactConflict) || errors.Is(err, artifacts.ErrHistoryProtected) ||
		errors.Is(err, artifacts.ErrRunRevisionLimit) || artifacts.IsExactRevisionMissing(err)
}

func failAuditInputInitialization(ctx context.Context, writer AuditRunWriter, run runstore.WorkflowRun) (runstore.WorkflowRun, error) {
	return writer.TransitionRun(ctx, run.RunID, runstore.RunInitializing, runstore.RunFailed,
		runstore.Reason{Code: runstore.AuditInputInitializationFailedReason, Message: "The exact Audit input closure could not be initialized."})
}
