package scheduler

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/controlplane"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/settingsstore"
)

// With the migration's pool sizes, one model lane and two tool lanes, the
// supervisor progresses one artifact-copy Run and two nuclei-target Runs at
// once from real PostgreSQL claims, and each freed lane claims the next Run
// of its own class (SCH-R175).
func TestPostgresSchedulerSupervisorRunsSeparatePools(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedSchedulerPool(t, ctx)
	postgresStore := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage())
	snapshot, _, err := configload.LoadUnionReadOnly("../../testdata/configs-scan", filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	toolWorkflow, err := snapshot.Workflow("nuclei-target@1")
	if err != nil {
		t.Fatal(err)
	}
	modelWorkflow := loadSchedulerWorkflow(t)
	for _, run := range []struct {
		id         string
		workflow   resolved.ResolvedWorkflow
		parameters map[string]string
	}{
		{id: "run-model-1", workflow: modelWorkflow, parameters: map[string]string{"objective": "hold lane"}},
		{id: "run-tool-1", workflow: toolWorkflow, parameters: map[string]string{"target": "https://target.test"}},
		{id: "run-tool-2", workflow: toolWorkflow, parameters: map[string]string{"target": "https://target.test"}},
		{id: "run-model-2", workflow: modelWorkflow, parameters: map[string]string{"objective": "hold lane"}},
		{id: "run-tool-3", workflow: toolWorkflow, parameters: map[string]string{"target": "https://target.test"}},
	} {
		createRunningPoolRun(t, ctx, postgresStore, run.id, run.workflow, run.parameters)
	}
	settings := settingsstore.NewPostgresStore(pool)
	if current, err := settings.GetSchedulerSettings(ctx); err != nil ||
		current.MaxConcurrentRuns != 1 || current.MaxConcurrentToolRuns != 2 {
		t.Fatalf("migrated Scheduler settings = (%+v, %v), want 1 model and 2 tool lanes", current, err)
	}
	store := newPostgresLaneBlockingStore(postgresStore)
	rounds := newRoundSettings(settings)
	clock := newManualPollClock()
	scheduler := newPostgresSupervisor(t, pool, store, rounds, clock)
	store.observeLaneStarts(scheduler)
	runContext, stop := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() { done <- scheduler.Run(runContext) }()

	initial := map[string]bool{}
	for range 3 {
		initial[awaitValue(t, ctx, store.claimed, "initial PostgreSQL claim")] = true
	}
	if !initial["run-model-1"] || !initial["run-tool-1"] || !initial["run-tool-2"] {
		t.Fatalf("initial PostgreSQL claims = %v, want one model and two tool Runs", initial)
	}
	completePollRound(t, ctx, clock, rounds, store)
	assertNoClaim(t, store, "PostgreSQL claim beyond both pool sizes")
	store.unblock("run-tool-1")
	if next := awaitValue(t, ctx, store.claimed, "tool replacement"); next != "run-tool-3" {
		t.Fatalf("freed tool lane claimed %q, want run-tool-3", next)
	}
	store.unblock("run-model-1")
	if next := awaitValue(t, ctx, store.claimed, "model replacement"); next != "run-model-2" {
		t.Fatalf("freed model lane claimed %q, want run-model-2", next)
	}
	for _, runID := range []string{"run-tool-2", "run-tool-3", "run-model-2"} {
		store.unblock(runID)
	}
	store.waitReleased(t, ctx, 5)
	stop()
	if err := awaitValue(t, ctx, done, "PostgreSQL Scheduler shutdown"); err != nil {
		t.Fatal(err)
	}
	if model, tool := store.classMaximum(resolved.LaneClassModel), store.classMaximum(resolved.LaneClassTool); model != 1 || tool != 2 {
		t.Fatalf("lanes held at once: model %d, tool %d, want 1 and 2", model, tool)
	}
	if active, maximum := store.counts(); active != 0 || maximum != 3 {
		t.Fatalf("PostgreSQL lane counts active=%d maximum=%d, want 0/3", active, maximum)
	}
}

// A Run whose stored lane class was changed to disagree with its entry Stage
// or its active StageExecution is claimed by a lane of the stored class and
// fails as unprogressable; the class is never corrected (SCH-R175).
func TestPostgresSchedulerFailsRunWhoseStoredLaneClassDisagrees(t *testing.T) {
	for _, active := range []bool{false, true} {
		t.Run(map[bool]string{false: "entry Stage", true: "active Stage"}[active], func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
			defer cancel()
			h := newDeferredPlacementHarness(t, ctx, nil)
			stages := 0
			if active {
				// No capacity: the Stage is created and the Run deferred.
				h.workers.allocator.reserveError = controlplane.ErrInsufficientCapacity
				if worked, err := h.scheduler.RunOnce(ctx); !worked || !errors.Is(err, ErrDeferred) {
					t.Fatalf("first RunOnce = (%v, %v), want deferred", worked, err)
				}
				h.workers.allocator.reserveError = nil
				stages = 1
			}
			if _, err := h.pool.Exec(ctx, `UPDATE workflow_runs
SET scheduler_lane_class = 'tool', scheduler_deferred = false WHERE run_id = 'run-1'`); err != nil {
				t.Fatal(err)
			}
			worked, err := h.scheduler.RunOnce(ctx)
			run, runErr := h.store.GetRun(ctx, "run-1")
			if err != nil || !worked || runErr != nil || run.State != runstore.RunFailed ||
				run.StateReason.Code != "scheduler_state_invalid" || run.SchedulerLaneClass != resolved.LaneClassTool {
				t.Fatalf("RunOnce = (%v, %v); Run = %s %+v class %q (%v)",
					worked, err, run.State, run.StateReason, run.SchedulerLaneClass, runErr)
			}
			executions, err := h.store.ListStageExecutions(ctx, "run-1")
			if err != nil || len(executions) != stages || h.invoker.calls != 0 {
				t.Fatalf("Stages = %d (%v), Worker calls = %d, want %d and none", len(executions), err, h.invoker.calls, stages)
			}
		})
	}
}
