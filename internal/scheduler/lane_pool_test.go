package scheduler

import (
	"context"
	"maps"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// The model and the tool pool are sized separately (SCH-R175): with one model
// lane and two tool lanes, one model Run and two tool Runs progress at once
// and each pool claims only its class. Lowering the tool pool drains only
// that pool; the model pool keeps claiming meanwhile.
func TestSchedulerSupervisorRunsSeparateModelAndToolPools(t *testing.T) {
	poll := 60 * time.Millisecond
	store := newLaneTestStore(
		"run-model-1", "run-model-2", "run-model-3",
		"run-tool-1", "run-tool-2", "run-tool-3", "run-tool-4",
	)
	store.toolRuns("run-tool-1", "run-tool-2", "run-tool-3", "run-tool-4")
	settings := newMutableSchedulerSettings(1, poll)
	settings.setToolLanes(2)
	scheduler := newLaneTestScheduler(t, store, settings, poll)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- scheduler.Run(ctx) }()

	initial := map[string]bool{}
	for range 3 {
		initial[receiveString(t, store.claimed, time.Second, "initial claim")] = true
	}
	if !initial["run-model-1"] || !initial["run-tool-1"] || !initial["run-tool-2"] {
		t.Fatalf("initial claims = %v, want one model and two tool Runs", initial)
	}
	assertNoString(t, store.claimed, 2*poll, "claim beyond both pool sizes")
	store.unblock("run-tool-1")
	if next := receiveString(t, store.claimed, time.Second, "tool replacement"); next != "run-tool-3" {
		t.Fatalf("freed tool lane claimed %q, want run-tool-3", next)
	}
	store.unblock("run-model-1")
	if next := receiveString(t, store.claimed, time.Second, "model replacement"); next != "run-model-2" {
		t.Fatalf("freed model lane claimed %q, want run-model-2", next)
	}

	settings.drainReads()
	settings.setToolLanes(1)
	scheduler.Wake()
	settings.waitToolRead(t, 1, time.Second)
	store.unblock("run-tool-2")
	store.waitReleased(t, 3, time.Second)
	assertNoString(t, store.claimed, 2*poll, "tool pool replaced a drained lane above one")
	store.unblock("run-model-2")
	if next := receiveString(t, store.claimed, time.Second, "model claim during the tool drain"); next != "run-model-3" {
		t.Fatalf("model lane claimed %q during the tool drain, want run-model-3", next)
	}
	store.unblock("run-tool-3")
	if next := receiveString(t, store.claimed, time.Second, "single tool lane after drain"); next != "run-tool-4" {
		t.Fatalf("drained tool pool claimed %q, want run-tool-4", next)
	}
	store.unblock("run-model-3")
	store.unblock("run-tool-4")
	store.waitReleased(t, 7, time.Second)
	cancel()
	if err := receiveError(t, done, time.Second, "Scheduler shutdown"); err != nil {
		t.Fatal(err)
	}
	if _, maximum := store.classCounts(resolved.LaneClassModel); maximum != 1 {
		t.Fatalf("model lanes held at once = %d, want 1", maximum)
	}
	if _, maximum := store.classCounts(resolved.LaneClassTool); maximum != 2 {
		t.Fatalf("tool lanes held at once = %d, want 2", maximum)
	}
	if active, maximum := store.counts(); active != 0 || maximum != 3 {
		t.Fatalf("lane counts active=%d maximum=%d, want 0/3", active, maximum)
	}
}

// A cancelling tool Run waits for a tool lane however idle the model pool is,
// and the next free tool lane takes it before ordinary tool work (SCH-R121,
// SCH-R175).
func TestSchedulerSupervisorClaimsCancellingToolRunInToolPool(t *testing.T) {
	poll := 40 * time.Millisecond
	store := newLaneTestStore("run-tool-1", "run-tool-2", "run-tool-cancelling")
	store.toolRuns("run-tool-1", "run-tool-2", "run-tool-cancelling")
	settings := newMutableSchedulerSettings(2, poll)
	scheduler := newLaneTestScheduler(t, store, settings, poll)
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- scheduler.Run(ctx) }()

	if first := receiveString(t, store.claimed, time.Second, "single tool lane"); first != "run-tool-1" {
		t.Fatalf("tool lane claimed %q, want run-tool-1", first)
	}
	store.cancellingRuns("run-tool-cancelling")
	assertNoString(t, store.claimed, 3*poll, "an idle model lane claimed a tool Run")
	store.unblock("run-tool-1")
	if next := receiveString(t, store.claimed, time.Second, "freed tool lane"); next != "run-tool-cancelling" {
		t.Fatalf("freed tool lane claimed %q before the cancelling Run", next)
	}
	store.unblock("run-tool-cancelling")
	if next := receiveString(t, store.claimed, time.Second, "ordinary tool work"); next != "run-tool-2" {
		t.Fatalf("tool lane claimed %q, want run-tool-2", next)
	}
	store.unblock("run-tool-2")
	store.waitReleased(t, 3, time.Second)
	cancel()
	if err := receiveError(t, done, time.Second, "Scheduler shutdown"); err != nil {
		t.Fatal(err)
	}
	if _, maximum := store.classCounts(resolved.LaneClassModel); maximum != 0 {
		t.Fatalf("model lanes claimed %d tool Runs", maximum)
	}
}

// Both pool sizes are read and validated before any claim: an invalid tool
// pool size stops the Scheduler as a storage invariant failure instead of
// starting the model pool alone (OPS-R50).
func TestSchedulerSupervisorRejectsInvalidToolPoolSizeBeforeAnyClaim(t *testing.T) {
	store := newLaneTestStore("run-1")
	settings := newMutableSchedulerSettings(1, time.Second)
	settings.setToolLanes(0)
	scheduler := newLaneTestScheduler(t, store, settings, 20*time.Millisecond)
	if err := scheduler.Run(t.Context()); err == nil || store.claimCalls() != 0 {
		t.Fatalf("Run = %v, claims = %d", err, store.claimCalls())
	}
}

// A Run whose scan Stage is followed by a model Stage moves between pools
// (SCH-R175): a tool lane progresses the scan Stage, whose result commit
// creates the model Stage and moves the Run to the model pool, so the tool
// lane releases the claim without deferral and a model lane claims the Run
// next. No lane of the wrong pool ever holds it.
func TestSchedulerHandsTheRunToTheModelPoolAfterAToolStage(t *testing.T) {
	snapshot, err := configload.Load("../../testdata/configs-scan")
	if err != nil {
		t.Fatal(err)
	}
	nuclei, err := snapshot.Workflow("nuclei-target@1")
	if err != nil {
		t.Fatal(err)
	}
	scan := nuclei.Stages[nuclei.EntryStage]
	harness := newSchedulerHarness(t)
	configureMultiStageWorkflow(t, harness)
	// The build Stage keeps its passthrough@1 Planner and binds the scanner's
	// tool Worker template in place of the ADK builder, reporting into the
	// Stage's copied result.
	build := harness.workflow.Stages["build"]
	binding := build.Agents["builder"]
	binding.Template = scan.Agents["scanner"].Template
	execution := *binding.Template.Execution
	execution.ResultArtifact = "copied"
	binding.Template.Execution = &execution
	build.Agents["builder"] = binding
	copied := build.Result.Artifacts["copied"]
	copied.MediaTypes = append(slices.Clone(copied.MediaTypes), "application/json")
	build.Result.Artifacts["copied"] = copied
	build.ExecutionConfig.Agents["builder"] = scan.ExecutionConfig.Agents["scanner"]
	harness.workflow.Stages["build"] = build
	maps.Copy(harness.workflow.Parameters, nuclei.Parameters)
	harness.store.run.Parameters["target"] = "https://target.test"
	installHarnessWorkflow(t, harness)
	harness.artifacts.current["run-1/builder/copied"] = "result-r1"
	if harness.store.run.SchedulerLaneClass != resolved.LaneClassTool {
		t.Fatalf("Run with a scan entry Stage has class %q", harness.store.run.SchedulerLaneClass)
	}

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("tool Stage RunOnce = (%v, %v)", worked, err)
	}
	if harness.store.run.SchedulerLaneClass != resolved.LaneClassModel || len(harness.store.stages) != 2 ||
		harness.store.stages[0].State != runstore.StageSucceeded || harness.store.stages[1].StageName != "review" {
		t.Fatalf("after the tool Stage: class=%q stages=%+v", harness.store.run.SchedulerLaneClass, harness.store.stages)
	}
	worked, err = harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked || harness.store.run.State != runstore.RunSucceeded {
		t.Fatalf("model Stage RunOnce = (%v, %v), run %s", worked, err, harness.store.run.State)
	}
	want := []resolved.LaneClass{resolved.LaneClassTool, resolved.LaneClassModel}
	if len(harness.store.claimedClasses) != 2 || harness.store.claimedClasses[0] != want[0] ||
		harness.store.claimedClasses[1] != want[1] || harness.store.deferrals != 0 {
		t.Fatalf("claims = %v with %d deferrals, want %v without deferral",
			harness.store.claimedClasses, harness.store.deferrals, want)
	}
}

// A stored lane class that disagrees with the Stage the Run progresses is a
// storage invariant failure: the Run ends as unprogressable and is never
// re-classified (SCH-R175), whether the Stage is still to be created or is
// already active.
func TestSchedulerFailsRunWhoseStoredLaneClassDisagrees(t *testing.T) {
	for _, active := range []bool{false, true} {
		t.Run(map[bool]string{false: "entry Stage", true: "active Stage"}[active], func(t *testing.T) {
			harness := newSchedulerHarness(t)
			if active {
				harness.store.stages = []runstore.StageExecution{harness.persistedExecution(t, runstore.StageRunning)}
			}
			harness.store.run.SchedulerLaneClass = resolved.LaneClassTool
			worked, err := harness.scheduler.RunOnce(context.Background())
			if err != nil || !worked {
				t.Fatalf("RunOnce = (%v, %v)", worked, err)
			}
			if harness.store.run.State != runstore.RunFailed ||
				harness.store.run.StateReason.Code != "scheduler_state_invalid" ||
				harness.store.run.SchedulerLaneClass != resolved.LaneClassTool {
				t.Fatalf("Run = %s %+v class %q, want failed unprogressable and unchanged",
					harness.store.run.State, harness.store.run.StateReason, harness.store.run.SchedulerLaneClass)
			}
			if !active && len(harness.store.stages) != 0 {
				t.Fatalf("mismatched Run created a Stage: %+v", harness.store.stages)
			}
			if harness.planners.runCalls != 0 || harness.allocator.reserveCalls != 0 {
				t.Fatalf("mismatched Run reached the Planner (%d) or allocation (%d)",
					harness.planners.runCalls, harness.allocator.reserveCalls)
			}
		})
	}
}
