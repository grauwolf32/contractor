package scheduler

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

// The memory doubles of the Scheduler tests must refuse what the PostgreSQL
// stores refuse, or a Scheduler regression that asks for an invalid operation
// passes its tests (V368-010). Each table below drives the double and the real
// store with the same operations and requires the same answer.

var parityRunStates = []runstore.WorkflowRunState{
	runstore.RunInitializing, runstore.RunPending, runstore.RunRunning, runstore.RunWaiting,
	runstore.RunCancelling, runstore.RunSucceeded, runstore.RunFailed, runstore.RunCancelled,
}

// TestMemorySchedulerStoreTransitionsMatchPostgres tries every transition
// between two WorkflowRun states on a Run that is in the expected state: the
// double and the store accept the same edges and refuse every other one as
// invalid.
func TestMemorySchedulerStoreTransitionsMatchPostgres(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	memoryErrors := map[[2]runstore.WorkflowRunState]error{}
	refused := 0
	for _, from := range parityRunStates {
		for _, to := range parityRunStates {
			store := &memorySchedulerStore{run: runstore.WorkflowRun{RunID: "run-parity", State: from}, events: &eventRecorder{}}
			_, err := store.TransitionRun(ctx, "run-parity", from, to, runstore.Reason{Code: "parity"})
			if err != nil && !errors.Is(err, runstore.ErrInvalid) {
				t.Errorf("memory %s -> %s on a Run in %s: %v, want acceptance or ErrInvalid", from, to, from, err)
			}
			if err != nil {
				refused++
			}
			memoryErrors[[2]runstore.WorkflowRunState{from, to}] = err
		}
	}
	if err := memoryErrors[[2]runstore.WorkflowRunState{runstore.RunSucceeded, runstore.RunRunning}]; !errors.Is(err, runstore.ErrInvalid) {
		t.Fatalf("memory succeeded -> running = %v, want ErrInvalid", err)
	}
	if refused == 0 || refused == len(memoryErrors) {
		t.Fatalf("memory store refused %d of %d transitions; the table tests nothing", refused, len(memoryErrors))
	}

	store := runstore.NewPostgresStore(isolatedSchedulerPool(t, ctx))
	for _, from := range parityRunStates {
		for _, to := range parityRunStates {
			runID := fmt.Sprintf("run-parity-%s-%s", from, to)
			createParityRun(t, ctx, store, runID, from, false)
			_, err := store.TransitionRun(ctx, runID, from, to, runstore.Reason{Code: "parity"})
			want := memoryErrors[[2]runstore.WorkflowRunState{from, to}]
			if (err == nil) != (want == nil) || err != nil && !errors.Is(err, runstore.ErrInvalid) {
				t.Errorf("PostgreSQL %s -> %s = %v, memory = %v", from, to, err, want)
			}
		}
	}
}

// TestMemorySchedulerStoreAllocationsMatchPostgres records the same Stage
// allocations in the double and the store: both refuse every malformed row
// as invalid and keep only the valid one.
func TestMemorySchedulerStoreAllocationsMatchPostgres(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	harness := newSchedulerHarness(t)
	reservation, err := harness.allocator.reservation("stage-parity")
	if err != nil {
		t.Fatal(err)
	}
	valid := stageAllocationFromReservation(reservation)
	invalid := []struct {
		name   string
		mutate func(*runstore.StageAllocation)
	}{
		{"no allocation ID", func(a *runstore.StageAllocation) { a.AllocationID = "" }},
		{"no logical agent", func(a *runstore.StageAllocation) { a.LogicalAgentName = " " }},
		{"Runtime Agent ID outside its grammar", func(a *runstore.StageAllocation) { a.RuntimeAgentID = "agent-1" }},
		{"no label revision", func(a *runstore.StageAllocation) { a.RuntimeAgentLabelRevision = 0 }},
		{"no Runtime configuration", func(a *runstore.StageAllocation) { a.RuntimeConfiguration = nil }},
		{"unknown configuration schema", func(a *runstore.StageAllocation) { a.RuntimeConfigurationSchemaVersion = "v1" }},
		{"AgentTemplate digest outside its grammar", func(a *runstore.StageAllocation) { a.AgentTemplateRef.Digest = "sha256:abc" }},
		{"tool Worker with a model policy", func(a *runstore.StageAllocation) {
			a.WorkerRuntimeRef = contracts.WorkerRuntimeRef{RuntimeID: "tool", Version: "1"}
		}},
		{"unpinned performance collection", func(a *runstore.StageAllocation) { a.PerformanceCollectionPolicy = "" }},
	}
	record := func(store interface {
		RecordStageAllocation(context.Context, runstore.StageAllocation) error
	}) []error {
		results := make([]error, 0, len(invalid)+1)
		for _, test := range invalid {
			allocation := valid
			configuration := *valid.RuntimeConfiguration
			allocation.RuntimeConfiguration = &configuration
			test.mutate(&allocation)
			results = append(results, store.RecordStageAllocation(ctx, allocation))
		}
		return append(results, store.RecordStageAllocation(ctx, valid))
	}
	check := func(name string, results []error) {
		t.Helper()
		for index, test := range invalid {
			if !errors.Is(results[index], runstore.ErrInvalid) {
				t.Errorf("%s recorded an allocation with %s: %v, want ErrInvalid", name, test.name, results[index])
			}
		}
		if err := results[len(invalid)]; err != nil {
			t.Fatalf("%s refused the valid allocation: %v", name, err)
		}
	}
	memory := &memorySchedulerStore{events: &eventRecorder{}}
	check("memory", record(memory))
	if len(memory.allocations) != 1 {
		t.Fatalf("memory store kept %d allocations, want the valid one", len(memory.allocations))
	}

	store := runstore.NewPostgresStore(isolatedSchedulerPool(t, ctx))
	createParityRun(t, ctx, store, "run-parity", runstore.RunRunning, true)
	check("PostgreSQL", record(store))
	recorded, err := store.ListStageAllocations(ctx, "stage-parity")
	if err != nil || len(recorded) != 1 || recorded[0].AllocationID != valid.AllocationID {
		t.Fatalf("PostgreSQL allocations = (%+v, %v), want the valid one", recorded, err)
	}
}

// TestMemoryAtomicPersistenceAbortRunStatesMatchPostgres starts an abort of a
// preparing Stage whose Run is in each state: the double and the PostgreSQL
// commit point accept the same Run states and refuse the others as a state
// conflict.
func TestMemoryAtomicPersistenceAbortRunStatesMatchPostgres(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	params := func() runstore.EnterAbortingParams {
		return runstore.EnterAbortingParams{
			StageExecutionID: "stage-parity", ExpectedState: runstore.StagePreparing,
			TerminationSchemaVersion: contracts.APIVersion,
			Termination: runstore.StageTermination{
				Outcome: runstore.TerminationInterrupted, Code: "parity", Message: "parity abort",
				Phase: runstore.TerminationPreparing, OccurredAt: time.Now().UTC(),
			},
			AbortID: "abort-parity", Deadline: time.Now().Add(time.Minute),
			Reason: runstore.Reason{Code: "stage_interrupted"},
		}
	}
	memoryErrors := map[runstore.WorkflowRunState]error{}
	for _, state := range parityRunStates {
		store := &memorySchedulerStore{
			run:    runstore.WorkflowRun{RunID: "run-parity", State: state},
			stages: []runstore.StageExecution{{StageExecutionID: "stage-parity", RunID: "run-parity", State: runstore.StagePreparing}},
			events: &eventRecorder{},
		}
		persistence := &memoryAtomicPersistence{store: store, allocator: &memoryAllocator{}, events: store.events}
		err := persistence.EnterAbortingWithTermination(ctx, "run-parity", params())
		if err != nil && !errors.Is(err, runstore.ErrConflict) {
			t.Errorf("memory abort in a %s Run: %v, want acceptance or ErrConflict", state, err)
		}
		memoryErrors[state] = err
	}
	for _, state := range parityRunStates {
		if accepted := memoryErrors[state] == nil; accepted != slices.Contains(abortRunStates, state) {
			t.Errorf("memory abort in a %s Run accepted=%t, want the PostgreSQL commit point's set %v", state, accepted, abortRunStates)
		}
	}

	pool := isolatedSchedulerPool(t, ctx)
	store := runstore.NewPostgresStore(pool)
	persistence, err := NewPostgresPersistence(pool)
	if err != nil {
		t.Fatal(err)
	}
	for _, state := range parityRunStates {
		runID := "run-abort-" + string(state)
		createParityRun(t, ctx, store, runID, state, true)
		abort := params()
		abort.StageExecutionID = "stage-" + runID
		err := persistence.EnterAbortingWithTermination(ctx, runID, abort)
		want := memoryErrors[state]
		if (err == nil) != (want == nil) || err != nil && !errors.Is(err, runstore.ErrConflict) {
			t.Errorf("PostgreSQL abort in a %s Run = %v, memory = %v", state, err, want)
		}
	}
}

// createParityRun moves a new Run into state through the store's own
// operations, so the Run is exactly what an operation in that state finds.
// withStage first creates its preparing StageExecution "stage-<runID>", or
// "stage-parity" for the Run named run-parity.
func createParityRun(
	t *testing.T, ctx context.Context, store *runstore.PostgresStore, runID string,
	state runstore.WorkflowRunState, withStage bool,
) {
	t.Helper()
	if _, err := store.CreateRun(ctx, runstore.CreateRunParams{
		RunID: runID, OwnerID: "user-parity", WorkflowName: "artifact-copy", WorkflowVersion: "1",
		WorkflowSchemaVersion: contracts.APIVersion, WorkflowSnapshot: json.RawMessage(`{}`),
		RuntimeConfig: runtimeconfig.BuiltInRunSnapshot(),
	}); err != nil {
		t.Fatal(err)
	}
	if withStage {
		stageID := "stage-" + runID
		if runID == "run-parity" {
			stageID = "stage-parity"
		}
		if _, err := store.CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
			StageExecutionID: stageID, RunID: runID, StageName: "copy", Attempt: 1,
			StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{}`),
			StageContextSchemaVersion: contracts.APIVersion, StageContext: runstore.StageContextSnapshot{},
		}); err != nil {
			t.Fatal(err)
		}
	}
	steps := map[runstore.WorkflowRunState][]runstore.WorkflowRunState{
		runstore.RunPending:   {runstore.RunPending},
		runstore.RunRunning:   {runstore.RunPending, runstore.RunRunning},
		runstore.RunWaiting:   {runstore.RunPending, runstore.RunRunning, runstore.RunWaiting},
		runstore.RunSucceeded: {runstore.RunPending, runstore.RunRunning, runstore.RunSucceeded},
		runstore.RunFailed:    {runstore.RunFailed},
		runstore.RunCancelled: {runstore.RunCancelled},
	}
	current := runstore.RunInitializing
	if state == runstore.RunCancelling || state == runstore.RunCancelled {
		if _, err := store.RequestRunCancellation(ctx, runID, runstore.WorkflowRunCancellation{
			Code: runstore.CancellationUserRequested, RequestedAt: time.Now(),
		}); err != nil {
			t.Fatal(err)
		}
		current = runstore.RunCancelling
	}
	for _, next := range steps[state] {
		if _, err := store.TransitionRun(ctx, runID, current, next, runstore.Reason{Code: "parity"}); err != nil {
			t.Fatalf("move Run %s from %s to %s: %v", runID, current, next, err)
		}
		current = next
	}
	run, err := store.GetRun(ctx, runID)
	if err != nil || run.State != state {
		t.Fatalf("Run %s = (%s, %v), want %s", runID, run.State, err, state)
	}
}
