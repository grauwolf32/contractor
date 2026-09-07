package controlplane

import (
	"context"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

type deadlineRuntime struct {
	recordingRuntime
	t        *testing.T
	want     time.Time
	terminal time.Time
}

func (r *deadlineRuntime) check(ctx context.Context, terminal time.Time) {
	r.t.Helper()
	got, ok := ctx.Deadline()
	if !ok || !got.Equal(r.want) || !terminal.Equal(r.terminal) {
		r.t.Errorf("call deadline = %v, terminal = %v; want %v, %v", got, terminal, r.want, r.terminal)
	}
}

func (r *deadlineRuntime) Finalize(ctx context.Context, reservation Reservation, id string, deadline time.Time) (contracts.AllocationFinalReport, error) {
	r.check(ctx, deadline)
	return r.recordingRuntime.Finalize(ctx, reservation, id, deadline)
}

func (r *deadlineRuntime) Abort(ctx context.Context, reservation Reservation, id string, reason contracts.TerminationError, deadline time.Time) (contracts.AllocationFinalReport, error) {
	r.check(ctx, deadline)
	return r.recordingRuntime.Abort(ctx, reservation, id, reason, deadline)
}

func TestTerminalCallsUseSavedOrEarlierCallerDeadlineIndependentlyOfCleanup(t *testing.T) {
	for _, earlierParent := range []bool{false, true} {
		for _, operation := range []string{"finalize", "abort"} {
			t.Run(operation+"/"+map[bool]string{false: "terminal", true: "caller"}[earlierParent], func(t *testing.T) {
				deadline := time.Now().Add(time.Minute)
				parent := deadline.Add(time.Minute)
				if earlierParent {
					parent = deadline.Add(-30 * time.Second)
				}
				want := deadline
				if parent.Before(want) {
					want = parent
				}
				ctx, cancel := context.WithDeadline(context.Background(), parent)
				defer cancel()
				runtime := &deadlineRuntime{t: t, want: want, terminal: deadline}
				controller, err := NewRuntimeBatchController(runtime, &recordingAllocationRegistry{}, RuntimeBatchOptions{CleanupTimeout: time.Nanosecond})
				if err != nil {
					t.Fatal(err)
				}
				reservations := []Reservation{testReservation("allocation_1", "first", "https://first.example", "https://first.example", testTemplate(t), deadline)}
				for attempt := 0; attempt < 2; attempt++ {
					if operation == "finalize" {
						_, err = controller.FinalizeAll(ctx, reservations, "finalization_1", deadline)
					} else {
						_, err = controller.AbortAll(ctx, reservations, "abort_1", contracts.TerminationError{Code: "test_abort", Message: "test abort"}, deadline)
					}
					if err != nil {
						t.Fatal(err)
					}
				}
			})
		}
	}
}

func (r *deadlineRuntime) Release(ctx context.Context, reservation Reservation) error {
	r.check(ctx, r.terminal)
	return r.recordingRuntime.Release(ctx, reservation)
}

func TestFailedPrepareAbortAndReleaseShareOneAbsoluteDeadline(t *testing.T) {
	now := time.Now()
	deadline := now.Add(time.Second)
	runtime := &deadlineRuntime{t: t, want: deadline, terminal: deadline}
	controller, err := NewRuntimeBatchController(runtime, &recordingAllocationRegistry{}, RuntimeBatchOptions{
		CleanupTimeout: time.Second, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	reservations := []Reservation{
		testReservation("allocation_1", "first", "https://first.example", "https://first.example", testTemplate(t), deadline),
		testReservation("allocation_2", "second", "https://second.example", "https://second.example", testTemplate(t), deadline),
	}
	if err := controller.cleanupFailedPrepare(reservations); err != nil {
		t.Fatal(err)
	}
	if len(runtime.aborted) != 2 || len(runtime.released) != 2 {
		t.Fatal("incomplete cleanup")
	}
}
