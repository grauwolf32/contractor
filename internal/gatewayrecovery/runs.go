package gatewayrecovery

import (
	"context"

	"github.com/jackc/pgx/v5"
)

// RunStates is what recovery needs of the Run store: the Run row lock, which
// orders recovery with the Scheduler's commit points (RUN-R29), and the moves
// of a Run between running and waiting (RUN-R30). The Run store owns those
// writes (PER-R23) and lies above this package in the layer stack, so
// recovery names them here and the Server passes the Run store's
// implementation. Each method runs in the caller's transaction tx; all but
// LockRun expect tx to hold the Run row lock already, under which every
// recovery wait of the Run changes.
type RunStates interface {
	// LockRun locks the row of Run runID until tx ends and returns the Run's
	// state, or "" when there is no such Run.
	LockRun(ctx context.Context, tx pgx.Tx, runID string) (string, error)
	// EnterWaiting moves a running Run to waiting with reason code, the
	// failure code of the blocked route, or gives a waiting Run that reason.
	// A Run in any other state is left as it is.
	EnterWaiting(ctx context.Context, tx pgx.Tx, runID, code string) error
	// LeaveWaiting returns a waiting Run to running. A Run in any other
	// state is left as it is.
	LeaveWaiting(ctx context.Context, tx pgx.Tx, runID string) error
	// RecordAdmissionWait gives a pending Run whose admission waits for a
	// blocked route reason code. A Run in any other state, or one that
	// already has that reason, is left as it is.
	RecordAdmissionWait(ctx context.Context, tx pgx.Tx, runID, code string) error
}

// lockRun locks an active Run through runs. A cancelling Run still accepts
// the terminal update of a granted attempt, so that cancellation
// relinquishes its probe; a Run that does not exist accepts nothing.
func lockRun(ctx context.Context, tx pgx.Tx, runs RunStates, runID string, terminalUpdate bool) error {
	state, err := runs.LockRun(ctx, tx, runID)
	if err != nil {
		return err
	}
	if !acceptsUpdate(state, terminalUpdate) {
		return ErrUnavailable
	}
	return nil
}
