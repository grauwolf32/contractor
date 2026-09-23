package controlplane

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPrepareAllCleansEveryReservationAfterPartialFailure(t *testing.T) {
	runtime := &recordingRuntime{prepareFailure: map[string]error{"allocation_2": errors.New("synthetic failure")}}
	registry := &recordingAllocationRegistry{}
	controller, err := NewRuntimeBatchController(runtime, registry, RuntimeBatchOptions{
		Now:            time.Now,
		NewID:          func(prefix string) (string, error) { return prefix + "cleanup", nil },
		CleanupTimeout: time.Second,
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

	handles, err := controller.PrepareAll(
		context.Background(), reservations,
		testWorkerExecutionSettings(template, testRuntimeSettings(), "first", "second"),
	)
	if err == nil || handles != nil {
		t.Fatalf("PrepareAll = (%+v, %v), want nil/error", handles, err)
	}
	for _, allocationID := range []string{"allocation_1", "allocation_2"} {
		if !slices.Contains(runtime.aborted, allocationID) || !slices.Contains(runtime.released, allocationID) ||
			!slices.Contains(registry.fenced, allocationID) || !slices.Contains(registry.released, allocationID) {
			t.Fatalf("allocation %s was not fully cleaned: runtime=%+v registry=%+v", allocationID, runtime, registry)
		}
	}
}

// Settings are validated as a whole before any Runtime call: a mismatch must
// not leave earlier Agents prepared and active without cleanup.
func TestPrepareAllRejectsMismatchedSettingsBeforePreparing(t *testing.T) {
	template := testTemplate(t)
	lease := time.Now().Add(time.Minute)
	reservations := []Reservation{
		testReservation("allocation_1", "first", "https://first.example", "https://first.example", template, lease),
		testReservation("allocation_2", "second", "https://second.example", "https://second.example", template, lease),
	}
	for name, settings := range map[string]map[string]contracts.WorkerExecutionSettings{
		"unknown": testWorkerExecutionSettings(template, testRuntimeSettings(), "first", "second", "third"),
		"missing": testWorkerExecutionSettings(template, testRuntimeSettings(), "first"),
	} {
		t.Run(name, func(t *testing.T) {
			runtime := &recordingRuntime{}
			registry := &recordingAllocationRegistry{}
			controller, err := NewRuntimeBatchController(runtime, registry, RuntimeBatchOptions{})
			if err != nil {
				t.Fatal(err)
			}
			handles, err := controller.PrepareAll(context.Background(), reservations, settings)
			if err == nil || handles != nil {
				t.Fatalf("PrepareAll = (%+v, %v), want nil/error", handles, err)
			}
			if len(runtime.prepared) != 0 || len(registry.phases) != 0 {
				t.Fatalf("mismatched settings reached Runtime: prepared=%v phases=%v", runtime.prepared, registry.phases)
			}
		})
	}
}

func TestFinalizeAllFencesBeforeRuntimeAndReleaseRetainsFailedGrant(t *testing.T) {
	runtime := &recordingRuntime{releaseFailure: map[string]error{"allocation_2": errors.New("unavailable")}}
	registry := &recordingAllocationRegistry{}
	controller, err := NewRuntimeBatchController(runtime, registry, RuntimeBatchOptions{})
	if err != nil {
		t.Fatal(err)
	}
	template := testTemplate(t)
	lease := time.Now().Add(time.Minute)
	reservations := []Reservation{
		testReservation("allocation_1", "first", "https://first.example", "https://first.example", template, lease),
		testReservation("allocation_2", "second", "https://second.example", "https://second.example", template, lease),
	}

	reports, err := controller.FinalizeAll(
		context.Background(), reservations, "finalization_1", time.Now().Add(time.Minute),
	)
	if err != nil || len(reports) != 2 {
		t.Fatalf("FinalizeAll = (%+v, %v)", reports, err)
	}
	if !slices.Equal(registry.fenced, []string{"allocation_1", "allocation_2"}) {
		t.Fatalf("fences = %v", registry.fenced)
	}
	if !slices.Equal(registry.phases, []allocationPhaseRecord{
		{"allocation_1", AllocationFinalizing}, {"allocation_2", AllocationFinalizing},
	}) || !slices.Equal(registry.reports, []string{"allocation_1", "allocation_2"}) {
		t.Fatalf("observed lifecycle = phases %v, reports %v", registry.phases, registry.reports)
	}
	if err := controller.ReleaseAll(context.Background(), reservations); err == nil {
		t.Fatal("ReleaseAll accepted a Runtime Agent release failure")
	}
	if !slices.Equal(registry.released, []string{"allocation_1"}) {
		t.Fatalf("registry releases = %v, want only confirmed Runtime release", registry.released)
	}
}

func TestRuntimeBatchTerminalCallsFanOutWithoutSiblingDeadlineStarvation(t *testing.T) {
	template := testTemplate(t)
	lease := time.Now().Add(time.Minute)
	reservations := []Reservation{
		testReservation("allocation_1", "first", "https://first.example", "https://first.example", template, lease),
		testReservation("allocation_2", "second", "https://second.example", "https://second.example", template, lease),
	}
	reason := contracts.TerminationError{Code: "test_abort", Message: "test abort", Retryable: true}

	tests := []struct {
		name       string
		configure  func(*recordingRuntime)
		invoke     func(*testing.T, context.Context, *RuntimeBatchController, []Reservation) error
		called     func(*recordingRuntime) []string
		wantReport bool
	}{
		{
			name:      "finalize",
			configure: func(runtime *recordingRuntime) { runtime.blockFinalize = map[string]bool{"allocation_1": true} },
			invoke: func(t *testing.T, ctx context.Context, controller *RuntimeBatchController, reservations []Reservation) error {
				reports, err := controller.FinalizeAll(ctx, reservations, "finalization_1", time.Now().Add(time.Minute))
				if len(reports) != 1 || reports["second"].AllocationID != "allocation_2" {
					t.Fatalf("finalize reports = %+v", reports)
				}
				return err
			},
			called:     func(runtime *recordingRuntime) []string { return runtime.finalized },
			wantReport: true,
		},
		{
			name:      "abort",
			configure: func(runtime *recordingRuntime) { runtime.blockAbort = map[string]bool{"allocation_1": true} },
			invoke: func(t *testing.T, ctx context.Context, controller *RuntimeBatchController, reservations []Reservation) error {
				reports, err := controller.AbortAll(ctx, reservations, "abort_1", reason, time.Now().Add(time.Minute))
				if len(reports) != 1 || reports["second"].AllocationID != "allocation_2" {
					t.Fatalf("abort reports = %+v", reports)
				}
				return err
			},
			called:     func(runtime *recordingRuntime) []string { return runtime.aborted },
			wantReport: true,
		},
		{
			name:      "release",
			configure: func(runtime *recordingRuntime) { runtime.blockRelease = map[string]bool{"allocation_1": true} },
			invoke: func(_ *testing.T, ctx context.Context, controller *RuntimeBatchController, reservations []Reservation) error {
				return controller.ReleaseAll(ctx, reservations)
			},
			called: func(runtime *recordingRuntime) []string { return runtime.released },
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			runtime := &recordingRuntime{}
			test.configure(runtime)
			registry := &recordingAllocationRegistry{}
			controller, err := NewRuntimeBatchController(runtime, registry, RuntimeBatchOptions{
				CleanupTimeout: 100 * time.Millisecond,
			})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
			defer cancel()
			if err := test.invoke(t, ctx, controller, reservations); err == nil {
				t.Fatal("batch accepted a blocked Runtime call")
			}
			if called := test.called(runtime); !slices.Contains(called, "allocation_2") {
				t.Fatalf("healthy sibling was not called: %v", called)
			}
			if test.name == "release" && !slices.Equal(registry.released, []string{"allocation_2"}) {
				t.Fatalf("registry releases = %v, want only healthy sibling", registry.released)
			}
			if test.wantReport && !slices.Equal(registry.reports, []string{"allocation_2"}) {
				t.Fatalf("recorded reports = %v", registry.reports)
			}
		})
	}
}

func TestFailedPrepareCleanupDoesNotHideHealthySibling(t *testing.T) {
	runtime := &recordingRuntime{blockAbort: map[string]bool{"allocation_1": true}}
	registry := &recordingAllocationRegistry{}
	controller, err := NewRuntimeBatchController(runtime, registry, RuntimeBatchOptions{
		CleanupTimeout: 50 * time.Millisecond,
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

	if err := controller.cleanupFailedPrepare(reservations); err == nil {
		t.Fatal("cleanup accepted a blocked Runtime abort")
	}
	if !slices.Contains(runtime.aborted, "allocation_2") || !slices.Contains(runtime.released, "allocation_2") {
		t.Fatalf("healthy sibling cleanup calls = aborted %v, released %v", runtime.aborted, runtime.released)
	}
	if !slices.Contains(registry.released, "allocation_2") {
		t.Fatalf("healthy sibling grant was not released: %v", registry.released)
	}
}

type recordingRuntime struct {
	mu             sync.Mutex
	prepared       []string
	finalized      []string
	aborted        []string
	released       []string
	prepareFailure map[string]error
	releaseFailure map[string]error
	blockFinalize  map[string]bool
	blockAbort     map[string]bool
	blockRelease   map[string]bool
}

func (r *recordingRuntime) Prepare(
	_ context.Context, reservation Reservation, _ contracts.WorkerExecutionSettings,
) (contracts.WorkerHandle, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	allocationID := reservation.Grant.AllocationID
	r.prepared = append(r.prepared, allocationID)
	if err := r.prepareFailure[allocationID]; err != nil {
		return contracts.WorkerHandle{}, err
	}
	return contracts.WorkerHandle{AllocationID: allocationID}, nil
}

func (r *recordingRuntime) Finalize(
	ctx context.Context, reservation Reservation, _ string, _ time.Time,
) (contracts.AllocationFinalReport, error) {
	if err := ctx.Err(); err != nil {
		return contracts.AllocationFinalReport{}, err
	}
	r.mu.Lock()
	allocationID := reservation.Grant.AllocationID
	r.finalized = append(r.finalized, allocationID)
	blocked := r.blockFinalize[allocationID]
	r.mu.Unlock()
	if blocked {
		<-ctx.Done()
		return contracts.AllocationFinalReport{}, ctx.Err()
	}
	return testExecutionReport(reservation.Grant.AllocationID), nil
}

func (r *recordingRuntime) Abort(
	ctx context.Context,
	reservation Reservation,
	_ string,
	_ contracts.TerminationError,
	_ time.Time,
) (contracts.AllocationFinalReport, error) {
	if err := ctx.Err(); err != nil {
		return contracts.AllocationFinalReport{}, err
	}
	r.mu.Lock()
	allocationID := reservation.Grant.AllocationID
	r.aborted = append(r.aborted, allocationID)
	blocked := r.blockAbort[allocationID]
	r.mu.Unlock()
	if blocked {
		<-ctx.Done()
		return contracts.AllocationFinalReport{}, ctx.Err()
	}
	return testExecutionReport(reservation.Grant.AllocationID), nil
}

func (r *recordingRuntime) Release(ctx context.Context, reservation Reservation) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	r.mu.Lock()
	allocationID := reservation.Grant.AllocationID
	r.released = append(r.released, allocationID)
	blocked := r.blockRelease[allocationID]
	failure := r.releaseFailure[allocationID]
	r.mu.Unlock()
	if blocked {
		<-ctx.Done()
		return ctx.Err()
	}
	return failure
}

type recordingAllocationRegistry struct {
	mu       sync.Mutex
	fenced   []string
	released []string
	phases   []allocationPhaseRecord
	reports  []string
}

type allocationPhaseRecord struct {
	allocationID string
	phase        AllocationAuthoritativePhase
}

func (r *recordingAllocationRegistry) SetWriteFence(allocationID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.fenced = append(r.fenced, allocationID)
	return nil
}

func (r *recordingAllocationRegistry) SetAllocationPhase(
	allocationID string,
	phase AllocationAuthoritativePhase,
	_ *SafeReason,
) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.phases = append(r.phases, allocationPhaseRecord{allocationID, phase})
	return nil
}

func (r *recordingAllocationRegistry) RecordAllocationReport(
	allocationID string,
	_ contracts.AllocationFinalReport,
) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.reports = append(r.reports, allocationID)
	return nil
}

func (r *recordingAllocationRegistry) Release(allocationID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.released = append(r.released, allocationID)
	return nil
}

func (r *recordingRuntime) String() string {
	return fmt.Sprintf("prepared=%v aborted=%v released=%v", r.prepared, r.aborted, r.released)
}
