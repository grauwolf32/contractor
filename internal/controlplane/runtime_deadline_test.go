package controlplane

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/reporting"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
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

func (r *deadlineRuntime) Finalize(ctx context.Context, reservation Reservation, id string, deadline time.Time) (reporting.AllocationFinalReport, error) {
	r.check(ctx, deadline)
	return r.recordingRuntime.Finalize(ctx, reservation, id, deadline)
}

func (r *deadlineRuntime) Abort(ctx context.Context, reservation Reservation, id string, reason contracts.TerminationError, deadline time.Time) (reporting.AllocationFinalReport, error) {
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
				controller, err := NewRuntimeBatchController(runtime, &recordingAllocationRegistry{}, RuntimeBatchOptions{PrepareTimeout: time.Minute, CleanupTimeout: time.Nanosecond})
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
		PrepareTimeout: time.Minute, CleanupTimeout: time.Second, Now: func() time.Time { return now },
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

type batchBudgetRuntime struct {
	recordingRuntime
	deadlineMu       sync.Mutex
	prepareDeadlines []time.Time
	releaseDeadlines []time.Time
}

func (r *batchBudgetRuntime) Prepare(ctx context.Context, reservation Reservation, settings runtimesettings.WorkerExecutionSettings) (contracts.WorkerHandle, error) {
	deadline, _ := ctx.Deadline()
	r.deadlineMu.Lock()
	r.prepareDeadlines = append(r.prepareDeadlines, deadline)
	r.deadlineMu.Unlock()
	return r.recordingRuntime.Prepare(ctx, reservation, settings)
}

func (r *batchBudgetRuntime) Release(ctx context.Context, reservation Reservation) error {
	deadline, _ := ctx.Deadline()
	r.deadlineMu.Lock()
	r.releaseDeadlines = append(r.releaseDeadlines, deadline)
	r.deadlineMu.Unlock()
	return r.recordingRuntime.Release(ctx, reservation)
}

// Each Runtime batch has one deadline from its own configured budget
// (SCH-R65): a prepare batch from prepareTimeout, a release batch from
// cleanupTimeout, whatever the other budget is.
func TestRuntimeBatchesShareOneDeadlineFromTheirOwnBudget(t *testing.T) {
	const prepareTimeout, cleanupTimeout = 2 * time.Hour, 3 * time.Hour
	runtime := &batchBudgetRuntime{}
	controller, err := NewRuntimeBatchController(runtime, &recordingAllocationRegistry{}, RuntimeBatchOptions{
		PrepareTimeout: prepareTimeout, CleanupTimeout: cleanupTimeout,
	})
	if err != nil {
		t.Fatal(err)
	}
	template := testTemplate(t)
	lease := time.Now().Add(time.Minute)
	reservations := []Reservation{
		testReservation("allocation_1", "first", "https://first.example", "https://first.example", template, lease),
		testReservation("allocation_2", "second", "https://second.example", "https://second.example", template, lease),
	}
	before := time.Now()
	if _, err := controller.PrepareAll(context.Background(), reservations,
		testWorkerExecutionSettings(template, testRuntimeSettings(), "first", "second")); err != nil {
		t.Fatal(err)
	}
	if err := controller.ReleaseAll(context.Background(), reservations); err != nil {
		t.Fatal(err)
	}
	after := time.Now()
	for name, batch := range map[string]struct {
		deadlines []time.Time
		budget    time.Duration
	}{
		"prepare": {runtime.prepareDeadlines, prepareTimeout},
		"release": {runtime.releaseDeadlines, cleanupTimeout},
	} {
		if len(batch.deadlines) != 2 || !batch.deadlines[0].Equal(batch.deadlines[1]) ||
			batch.deadlines[0].Before(before.Add(batch.budget)) || batch.deadlines[0].After(after.Add(batch.budget)) {
			t.Fatalf("%s deadlines = %v; want one shared deadline %v after the batch started", name, batch.deadlines, batch.budget)
		}
	}
}

func TestRuntimeBatchControllerRequiresBothBudgets(t *testing.T) {
	for _, options := range []RuntimeBatchOptions{
		{CleanupTimeout: time.Minute},
		{PrepareTimeout: time.Minute},
		{PrepareTimeout: -time.Second, CleanupTimeout: time.Minute},
	} {
		if _, err := NewRuntimeBatchController(&recordingRuntime{}, &recordingAllocationRegistry{}, options); err == nil {
			t.Fatalf("budgets %+v accepted", options)
		}
	}
}
