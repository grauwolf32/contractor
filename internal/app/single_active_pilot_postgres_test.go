package app

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/persistence/leasetest"
)

// A cancelled startup must release the session lease while an old working
// transaction remains open. A restarted successor must still perform its own
// handover before reading the old transaction's eventual commit.
func TestServerLeaseHandoverPilotCancelledStartupCanRestart(t *testing.T) {
	ctx, pool := handoverTestPool(t)
	previous := activeLeaseForHandover(t, ctx, pool)
	writer, commit := openFencedWrite(t, ctx, pool, previous, 1)
	terminateLeaseBackend(t, ctx, pool, previous)

	startup, cancel := context.WithCancel(ctx)
	type result struct {
		lease *controlPlaneLease
		err   error
	}
	finished, joined := make(chan result, 1), make(chan struct{})
	go func() {
		defer close(joined)
		lease, err := awaitControlPlaneLease(startup, pool, "127.0.0.1:0", time.Second,
			slog.New(slog.NewTextHandler(io.Discard, nil)))
		finished <- result{lease, err}
	}()
	t.Cleanup(func() {
		cancel()
		<-joined
		// Also close an unexpected successful startup on a failed assertion.
		select {
		case got := <-finished:
			if got.lease != nil {
				got.lease.Close()
			}
		default:
		}
	})
	leasetest.AwaitWaiterBlockedBy(t, pool, writer)
	cancel()
	got := <-finished
	if got.lease != nil {
		got.lease.Close()
		t.Fatal("cancelled startup returned a lease")
	}
	if !errors.Is(got.err, context.Canceled) {
		t.Fatalf("cancelled startup = %v", got.err)
	}

	// Acquisition succeeding proves the cancelled startup closed its session.
	next, err := openControlPlaneLease(ctx, pool)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(next.Close)
	if acquired, err := next.tryAcquire(ctx); err != nil || !acquired {
		t.Fatalf("restart acquisition = (%v, %v)", acquired, err)
	}
	waiting, cancelWait := context.WithCancel(ctx)
	handover, handoverJoined := make(chan error, 1), make(chan struct{})
	go func() {
		defer close(handoverJoined)
		handover <- next.awaitHandover(waiting)
	}()
	t.Cleanup(func() { cancelWait(); <-handoverJoined })
	leasetest.AwaitWaiterBlockedBy(t, pool, writer)
	select {
	case err := <-handover:
		t.Fatalf("restart skipped the previous holder's pending commit: %v", err)
	default:
	}
	commit()
	if err := <-handover; err != nil {
		t.Fatal(err)
	}
	assertHandoverOutcome(t, ctx, pool, previous, next)
}
