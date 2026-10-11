package scheduler

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// countingWorkflowSnapshotCache wraps decodeExecutableWorkflow so a test can
// count how often the snapshot is decoded and validated.
func countingWorkflowSnapshotCache(maxRuns, maxBytes int) (*workflowSnapshotCache, *atomic.Int64) {
	decodes := &atomic.Int64{}
	return newWorkflowSnapshotCache(func(run runstore.WorkflowRun) (executableWorkflow, error) {
		decodes.Add(1)
		return decodeExecutableWorkflow(run)
	}, maxRuns, maxBytes), decodes
}

// TestSchedulerStepsReuseTheDecodedWorkflowSnapshot is the regression for a
// Scheduler that decoded and revalidated the whole Workflow snapshot on every
// step of a Run: a Run deferred twice and then completed is decoded once
// (SCH-R163).
func TestSchedulerStepsReuseTheDecodedWorkflowSnapshot(t *testing.T) {
	h := newSchedulerHarness(t)
	cache, decodes := countingWorkflowSnapshotCache(workflowSnapshotCacheRuns, workflowSnapshotCacheBytes)
	h.scheduler.workflows = cache
	h.store.run.State = runstore.RunPending
	h.allocator.reserveError = controlplane.ErrInsufficientCapacity
	for range 2 {
		if worked, err := h.scheduler.RunOnce(t.Context()); !worked || !errors.Is(err, ErrDeferred) {
			t.Fatalf("capacity wait = (%v, %v)", worked, err)
		}
	}
	h.allocator.reserveError = nil
	if worked, err := h.scheduler.RunOnce(t.Context()); !worked || err != nil {
		t.Fatalf("admission = (%v, %v)", worked, err)
	}
	if h.store.run.State != runstore.RunSucceeded {
		t.Fatalf("Run state = %s, want succeeded", h.store.run.State)
	}
	if got := decodes.Load(); got != 1 {
		t.Fatalf("three steps decoded the Workflow snapshot %d times, want once", got)
	}
}

// TestWorkflowSnapshotCacheKeysByRunAndDigestAndHandsOutCopies proves a hit
// equals a fresh decode, no caller's edits reach the cache, a changed snapshot
// or another Run is decoded again and a snapshot that fails validation is
// never kept.
func TestWorkflowSnapshotCacheKeysByRunAndDigestAndHandsOutCopies(t *testing.T) {
	run := newSchedulerHarness(t).store.run
	want, err := decodeExecutableWorkflow(run)
	if err != nil {
		t.Fatal(err)
	}
	cache, decodes := countingWorkflowSnapshotCache(4, workflowSnapshotCacheBytes)
	first, err := cache.executableWorkflow(run)
	if err != nil || !reflect.DeepEqual(first, want) {
		t.Fatalf("first read = (%+v, %v), want the decoded snapshot", first, err)
	}
	first.workflow.Parameters["intruder"] = workflowconfig.ParameterSlot{Required: true}
	first.stage.Agents["intruder"] = first.stage.Agents["builder"]
	first.stage.Context.Artifacts["intruder"] = workflowconfig.ContextArtifact{Namespace: "inputs", Name: "intruder"}
	second, err := cache.executableWorkflow(run)
	if err != nil || !reflect.DeepEqual(second, want) || decodes.Load() != 1 {
		t.Fatalf("second read = (%+v, %v) after %d decodes, want the unchanged snapshot from one decode",
			second, err, decodes.Load())
	}

	edited := want.workflow.Clone()
	stage := edited.Stages[edited.EntryStage]
	stage.Objective = "A changed objective"
	edited.Stages[edited.EntryStage] = stage
	changed := run
	if changed.WorkflowSnapshot, err = json.Marshal(edited); err != nil {
		t.Fatal(err)
	}
	if got, err := cache.executableWorkflow(changed); err != nil || got.stage.Objective != "A changed objective" ||
		decodes.Load() != 2 {
		t.Fatalf("changed snapshot = (%q, %v) after %d decodes, want a new decode", got.stage.Objective, err, decodes.Load())
	}
	other := run
	other.RunID = "run-other"
	if _, err := cache.executableWorkflow(other); err != nil || decodes.Load() != 3 {
		t.Fatalf("another Run = %v after %d decodes, want its own decode", err, decodes.Load())
	}

	broken := run
	broken.RunID = "run-broken"
	broken.WorkflowSnapshot = json.RawMessage(`{}`)
	for range 2 {
		if _, err := cache.executableWorkflow(broken); !errors.Is(err, ErrUnsupportedWorkflow) {
			t.Fatalf("invalid snapshot error = %v", err)
		}
	}
	if decodes.Load() != 5 {
		t.Fatalf("invalid snapshot decoded %d times in total, want it never kept", decodes.Load()-3)
	}
}

// TestWorkflowSnapshotCacheServesConcurrentLanes runs lanes that share the
// cache, churn its entries and edit their copies; run with -race.
func TestWorkflowSnapshotCacheServesConcurrentLanes(t *testing.T) {
	base := newSchedulerHarness(t).store.run
	want, err := decodeExecutableWorkflow(base)
	if err != nil {
		t.Fatal(err)
	}
	cache, _ := countingWorkflowSnapshotCache(2, workflowSnapshotCacheBytes)
	var wait sync.WaitGroup
	failures := make(chan error, 8)
	for lane := range 8 {
		wait.Add(1)
		go func() {
			defer wait.Done()
			for step := range 20 {
				run := base
				run.RunID = []string{"run-a", "run-b", "run-c"}[(lane+step)%3]
				got, err := cache.executableWorkflow(run)
				if err != nil || !reflect.DeepEqual(got.stage, want.stage) {
					failures <- fmt.Errorf("lane %d step %d = %v", lane, step, err)
					return
				}
				got.stage.Agents["intruder"] = got.stage.Agents["builder"]
				got.workflow.Parameters["intruder"] = workflowconfig.ParameterSlot{}
			}
		}()
	}
	wait.Wait()
	close(failures)
	for failure := range failures {
		t.Error(failure)
	}
}

// TestWorkflowSnapshotCacheEvictsTheLeastRecentlyUsedRun bounds the cache by
// Runs and by stored snapshot bytes (LIM-SCH-15).
func TestWorkflowSnapshotCacheEvictsTheLeastRecentlyUsedRun(t *testing.T) {
	base := newSchedulerHarness(t).store.run
	size := len(base.WorkflowSnapshot)
	runs := map[string]runstore.WorkflowRun{}
	for _, runID := range []string{"run-a", "run-b", "run-c"} {
		run := base
		run.RunID = runID
		runs[runID] = run
	}
	for name, bounds := range map[string][2]int{
		"by Runs":  {2, 3 * size},
		"by bytes": {3, 2 * size},
	} {
		t.Run(name, func(t *testing.T) {
			cache, decodes := countingWorkflowSnapshotCache(bounds[0], bounds[1])
			for _, step := range []struct {
				runID   string
				decodes int64
			}{
				{"run-a", 1}, {"run-b", 2}, {"run-a", 2},
				// run-c evicts run-b, the least recently used Run.
				{"run-c", 3}, {"run-a", 3}, {"run-c", 3}, {"run-b", 4},
			} {
				if _, err := cache.executableWorkflow(runs[step.runID]); err != nil {
					t.Fatal(err)
				}
				if decodes.Load() != step.decodes {
					t.Fatalf("after %s: %d decodes, want %d", step.runID, decodes.Load(), step.decodes)
				}
			}
			if cache.order.Len() != 2 || cache.bytes != 2*size {
				t.Fatalf("cache holds %d Runs and %d bytes, want 2 and %d", cache.order.Len(), cache.bytes, 2*size)
			}
		})
	}

	tooLarge, decodes := countingWorkflowSnapshotCache(2, size-1)
	for range 2 {
		if _, err := tooLarge.executableWorkflow(base); err != nil {
			t.Fatal(err)
		}
	}
	if decodes.Load() != 2 || tooLarge.order.Len() != 0 || tooLarge.bytes != 0 {
		t.Fatalf("a snapshot over the byte bound was kept: %d decodes, %d Runs", decodes.Load(), tooLarge.order.Len())
	}
}
