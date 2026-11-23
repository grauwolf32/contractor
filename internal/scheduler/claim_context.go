package scheduler

import (
	"context"
	"errors"

	"github.com/grauwolf32/contractor/internal/runstore"
)

type runOwnershipContextKey struct{}

// claimCleanupContext permits bounded cleanup after ordinary cancellation, but
// still observes loss of the Run claim. Keep the ownership context separately:
// an execution already cancelled by the user cannot acquire a second cause.
func claimCleanupContext(ctx context.Context) (context.Context, context.CancelFunc) {
	ownership, ok := ctx.Value(runOwnershipContextKey{}).(context.Context)
	if !ok {
		ownership = ctx
	}
	cleanup, cancel := context.WithCancelCause(context.WithoutCancel(ctx))
	checkClaim := func() {
		if cause := context.Cause(ownership); errors.Is(cause, ErrClaimLost) {
			cancel(cause)
		}
	}
	stop := context.AfterFunc(ownership, checkClaim)
	// AfterFunc is asynchronous even when ownership is already cancelled.
	// Prevent starting a new operation before that callback has run.
	checkClaim()
	return cleanup, func() {
		stop()
		cancel(nil)
	}
}

func (s *Scheduler) terminalOperationContext(ctx context.Context) (context.Context, context.CancelFunc) {
	cleanup, cancelCleanup := claimCleanupContext(ctx)
	operation, cancelOperation := context.WithTimeout(cleanup, s.options.OperationTimeout)
	return operation, func() {
		cancelOperation()
		cancelCleanup()
	}
}

type runClaimContextKey struct{}
type runClaim struct{ runID, claimID string }

// WithRunClaim binds lifecycle writes to the exact claim acquired by a lane.
// The scope survives execution cancellation and cleanup, and never derives its
// identity from a later Run read, which may already name a replacement owner.
func WithRunClaim(ctx context.Context, runID, claimID string) context.Context {
	return context.WithValue(ctx, runClaimContextKey{}, runClaim{runID, claimID})
}

func claimIDForRun(ctx context.Context, runID string) (string, error) {
	claim, ok := ctx.Value(runClaimContextKey{}).(runClaim)
	if !ok || claim.runID != runID || claim.claimID == "" {
		return "", ErrClaimLost
	}
	return claim.claimID, nil
}

func lockClaimedRun(ctx context.Context, store *runstore.PostgresTxStore, runID string, states ...runstore.WorkflowRunState) (runstore.WorkflowRunState, error) {
	claimID, err := claimIDForRun(ctx, runID)
	if err != nil {
		return "", err
	}
	state, err := store.LockClaimedRunState(ctx, runID, claimID, states...)
	var conflict *runstore.StateConflictError
	if errors.As(err, &conflict) && conflict.Resource == "WorkflowRun claim" {
		err = errors.Join(ErrClaimLost, err)
	}
	return state, err
}
