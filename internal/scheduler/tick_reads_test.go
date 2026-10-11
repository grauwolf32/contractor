package scheduler

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// TestSchedulerProgressionReadsOnlyTheActiveStage recovers the finalizing
// attempt of a Run with a long retry history: the progression attempt reads
// the Run's state and its one active StageExecution and never lists the
// history (SCH-R164).
func TestSchedulerProgressionReadsOnlyTheActiveStage(t *testing.T) {
	harness := newSchedulerHarness(t)
	const historyLength = 63
	stages := make([]runstore.StageExecution, 0, historyLength+1)
	for attempt := 1; attempt <= historyLength; attempt++ {
		earlier := harness.persistedExecution(t, runstore.StageInterrupted)
		earlier.StageExecutionID = fmt.Sprintf("stage-history-%02d", attempt)
		earlier.Attempt = attempt
		if attempt > 1 {
			previous := fmt.Sprintf("stage-history-%02d", attempt-1)
			earlier.PreviousExecutionID = &previous
		}
		stages = append(stages, earlier)
	}
	execution := harness.persistedExecution(t, runstore.StageFinalizing)
	execution.Attempt = historyLength + 1
	execution.PreviousExecutionID = &stages[historyLength-1].StageExecutionID
	result := harness.planners.result
	finalizationID := "finalization-recovery"
	deadline := harness.clock.now.Add(time.Minute)
	execution.CandidateResultSchemaVersion = stringPointer(contracts.APIVersion)
	execution.CandidateResult = &result
	execution.FinalizationID = &finalizationID
	execution.FinalizationDeadline = &deadline
	harness.store.stages = append(stages, execution)
	harness.persistence.stageStates = []runstore.StageExecutionState{runstore.StageFinalizing}
	harness.installRecordedReservation(execution.StageExecutionID)
	harness.store.run.State = runstore.RunRunning

	worked, err := harness.scheduler.RunOnce(context.Background())
	if err != nil || !worked {
		t.Fatalf("RunOnce = (%v, %v)", worked, err)
	}
	if harness.store.run.State != runstore.RunSucceeded {
		t.Fatalf("recovered Run = %s, want succeeded", harness.store.run.State)
	}
	if harness.store.stageListReads != 0 || harness.store.activeStageReads != 1 {
		t.Fatalf("progression read the Stage history %d times and the active Stage %d times, want 0 and 1",
			harness.store.stageListReads, harness.store.activeStageReads)
	}
}

// TestSchedulerFailsRunWithBrokenActiveStageInvariant gives a running Run two
// active StageExecutions, or Stage history without an active one: durable
// state the Scheduler cannot progress, so it fails the Run instead of
// choosing a Stage or creating a first one again.
func TestSchedulerFailsRunWithBrokenActiveStageInvariant(t *testing.T) {
	for _, test := range []struct {
		name   string
		states []runstore.StageExecutionState
	}{
		{name: "two active Stages", states: []runstore.StageExecutionState{runstore.StageRunning, runstore.StagePreparing}},
		{name: "history without an active Stage", states: []runstore.StageExecutionState{runstore.StageInterrupted}},
	} {
		t.Run(test.name, func(t *testing.T) {
			harness := newSchedulerHarness(t)
			harness.store.run.State = runstore.RunRunning
			for index, state := range test.states {
				execution := harness.persistedExecution(t, state)
				execution.StageExecutionID = fmt.Sprintf("stage-broken-%d", index)
				harness.store.stages = append(harness.store.stages, execution)
			}

			if _, err := harness.scheduler.RunOnce(context.Background()); err != nil {
				t.Fatalf("RunOnce: %v", err)
			}
			if harness.store.run.State != runstore.RunFailed || harness.store.run.StateReason.Code != "scheduler_state_invalid" ||
				harness.planners.createCalls != 0 || len(harness.store.stages) != len(test.states) {
				t.Fatalf("Run = %s (%s), planners created %d, Stages %d; want it failed as scheduler_state_invalid",
					harness.store.run.State, harness.store.run.StateReason.Code, harness.planners.createCalls, len(harness.store.stages))
			}
		})
	}
}
