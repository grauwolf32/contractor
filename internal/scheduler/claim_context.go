package scheduler

import (
	"context"
	"errors"
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
