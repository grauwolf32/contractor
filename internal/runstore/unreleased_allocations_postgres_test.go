package runstore

import (
	"context"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// ListStageExecutionsWithUnreleasedAllocations returns, in creation order,
// exactly the StageExecutions of one Run that have an allocation whose release
// is incomplete: never one whose releases completed, one without allocations
// or another Run's (SCH-R179).
func TestPostgresListStageExecutionsWithUnreleasedAllocations(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	store := NewPostgresStore(isolatedRunStorePool(t, ctx), artifactpgtest.NewStorage())
	for _, runID := range []string{"run-a", "run-b"} {
		if _, err := createRun(ctx, store, testRunParams(runID)); err != nil {
			t.Fatal(err)
		}
	}
	// stage creates a Stage of runID with the given allocations, attempts the
	// release of each, completes it for those listed in released, and ends the
	// Stage so the Run can start another.
	stage := func(runID, id string, allocations []string, released ...string) {
		t.Helper()
		if _, err := store.CreateStageExecution(ctx, testActiveStageParams(runID, id, id)); err != nil {
			t.Fatal(err)
		}
		for index, allocationID := range allocations {
			allocation := testStageAllocation(id, allocationID)
			allocation.LogicalAgentName = []string{"builder", "reviewer"}[index]
			allocation.Namespace = allocation.LogicalAgentName
			if err := store.RecordStageAllocation(ctx, allocation); err != nil {
				t.Fatal(err)
			}
			if err := store.MarkStageAllocationReleaseAttempt(ctx, allocationID); err != nil {
				t.Fatal(err)
			}
			if slices.Contains(released, allocationID) {
				if err := store.MarkStageAllocationReleased(ctx, allocationID); err != nil {
					t.Fatal(err)
				}
			}
		}
		if err := store.EnterAborting(ctx, EnterAbortingParams{
			StageExecutionID: id, ExpectedState: StagePreparing, TerminationSchemaVersion: contracts.APIVersion,
			Termination: StageTermination{
				Outcome: TerminationInterrupted, Code: "runtime_unavailable", Message: "Runtime Agent became unavailable",
				Retryable: true, Phase: TerminationPreparing, OccurredAt: time.Now(),
			},
			AbortID: "abort-" + id, Deadline: time.Now().Add(time.Minute), Reason: Reason{Code: "stage_interrupted"},
		}); err != nil {
			t.Fatal(err)
		}
		if err := store.CompleteStageTermination(ctx, id); err != nil {
			t.Fatal(err)
		}
	}
	stage("run-a", "a-released", []string{"a-released-1"}, "a-released-1")
	stage("run-a", "a-partial", []string{"a-partial-1", "a-partial-2"}, "a-partial-1")
	stage("run-a", "a-unallocated", nil)
	stage("run-a", "a-pending", []string{"a-pending-1"})
	stage("run-b", "b-pending", []string{"b-pending-1"})

	for runID, want := range map[string][]string{
		"run-a": {"a-partial", "a-pending"}, "run-b": {"b-pending"}, "run-unknown": nil,
	} {
		executions, err := store.ListStageExecutionsWithUnreleasedAllocations(ctx, runID)
		if err != nil {
			t.Fatalf("%s: %v", runID, err)
		}
		var got []string
		for _, execution := range executions {
			if execution.RunID != runID || execution.State != StageInterrupted {
				t.Fatalf("%s listed %+v", runID, execution)
			}
			got = append(got, execution.StageExecutionID)
		}
		if !slices.Equal(got, want) {
			t.Fatalf("%s StageExecutions with an incomplete release = %v, want %v", runID, got, want)
		}
	}
	if _, err := store.ListStageExecutionsWithUnreleasedAllocations(ctx, ""); !errors.Is(err, ErrInvalid) {
		t.Fatalf("empty Run ID error = %v, want invalid", err)
	}
}
