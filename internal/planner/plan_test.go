package planner

import (
	"reflect"
	"sync"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPlannerPlanUsesImmutableObjectiveOrderedIDsAndDeepSnapshots(t *testing.T) {
	controller, err := NewPlannerPlanController("Build an API description")
	if err != nil {
		t.Fatal(err)
	}
	first, planErr := controller.AddSubtask("Inspect routes", "Read the exact source inputs")
	if planErr != nil {
		t.Fatal(planErr)
	}
	second, planErr := controller.AddSubtask("Write the document", "Create the declared output")
	if planErr != nil {
		t.Fatal(planErr)
	}
	if first.Revision != 1 || second.Revision != 2 || second.Objective != "Build an API description" ||
		second.CurrentSubtaskID != "0" || len(second.Subtasks) != 2 ||
		second.Subtasks[0].ID != "0" || second.Subtasks[1].ID != "1" ||
		second.Subtasks[0].Status != PlannerSubtaskPending {
		t.Fatalf("plan = %+v", second)
	}
	second.Objective = "mutated"
	second.Subtasks[0].Objective = "mutated"
	snapshot := controller.Snapshot()
	if snapshot.Objective != "Build an API description" || snapshot.Subtasks[0].Objective != "Inspect routes" {
		t.Fatalf("authoritative plan was mutated through snapshot: %+v", snapshot)
	}
	if err := snapshot.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestPlannerPlanAllowsExactlyOneConcurrentCurrentClaim(t *testing.T) {
	controller, err := NewPlannerPlanController("Review the project")
	if err != nil {
		t.Fatal(err)
	}
	if _, planErr := controller.AddSubtask("Inspect", "Inspect all declared inputs"); planErr != nil {
		t.Fatal(planErr)
	}
	const callers = 16
	start := make(chan struct{})
	claims := make(chan PlannerDispatchClaim, callers)
	errors := make(chan *PlanError, callers)
	var group sync.WaitGroup
	for index := 0; index < callers; index++ {
		group.Add(1)
		go func() {
			defer group.Done()
			<-start
			claim, planErr := controller.ClaimCurrentSubtask("0", "reviewer")
			if planErr != nil {
				errors <- planErr
				return
			}
			claims <- claim
		}()
	}
	close(start)
	group.Wait()
	close(claims)
	close(errors)
	if len(claims) != 1 || len(errors) != callers-1 {
		t.Fatalf("claims=%d errors=%d", len(claims), len(errors))
	}
	claim := <-claims
	if claim.Subtask.ID != "0" || claim.CallID == "" {
		t.Fatalf("claim = %+v", claim)
	}
	snapshot := controller.Snapshot()
	if snapshot.Revision != 2 || snapshot.ActiveDispatch == nil ||
		snapshot.Subtasks[0].Status != PlannerSubtaskRunning {
		t.Fatalf("snapshot = %+v", snapshot)
	}
}

func TestPlannerPlanRejectsStaleClaimWithoutMutation(t *testing.T) {
	controller, err := NewPlannerPlanController("Review the project")
	if err != nil {
		t.Fatal(err)
	}
	if _, planErr := controller.AddSubtask("First", "Run first"); planErr != nil {
		t.Fatal(planErr)
	}
	if _, planErr := controller.AddSubtask("Second", "Run second"); planErr != nil {
		t.Fatal(planErr)
	}
	before := controller.Snapshot()
	if _, planErr := controller.ClaimCurrentSubtask("1", "reviewer"); planErr == nil || planErr.Code != "planner_subtask_stale" {
		t.Fatalf("error = %+v", planErr)
	}
	after := controller.Snapshot()
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("rejected claim mutated plan: before=%+v after=%+v", before, after)
	}
}

func TestPlannerPlanCompletesAndAdvancesWithLateResultFence(t *testing.T) {
	controller, err := NewPlannerPlanController("Review the project")
	if err != nil {
		t.Fatal(err)
	}
	if _, planErr := controller.AddSubtask("First", "Run first"); planErr != nil {
		t.Fatal(planErr)
	}
	if _, planErr := controller.AddSubtask("Second", "Run second"); planErr != nil {
		t.Fatal(planErr)
	}
	claim, planErr := controller.ClaimCurrentSubtask("0", "builder")
	if planErr != nil {
		t.Fatal(planErr)
	}
	completed, planErr := controller.CompleteDispatch(claim.CallID, contracts.StageFailed)
	if planErr != nil {
		t.Fatal(planErr)
	}
	if completed.Revision != 4 || completed.Subtasks[0].Status != PlannerSubtaskFailed ||
		completed.CurrentSubtaskID != "1" || completed.ActiveDispatch != nil {
		t.Fatalf("completed = %+v", completed)
	}
	before := controller.Snapshot()
	if _, planErr := controller.CompleteDispatch(claim.CallID, contracts.StageSucceeded); planErr == nil || planErr.Code != "planner_dispatch_stale" {
		t.Fatalf("late result error = %+v", planErr)
	}
	if after := controller.Snapshot(); !reflect.DeepEqual(before, after) {
		t.Fatalf("late result mutated plan: before=%+v after=%+v", before, after)
	}
}

func TestPlannerPlanFinishPreconditions(t *testing.T) {
	controller, err := NewPlannerPlanController("Review the project")
	if err != nil {
		t.Fatal(err)
	}
	if controller.CanFinish(contracts.StageFailed) != nil {
		t.Fatal("failed finish must be allowed without a dispatch")
	}
	if finishErr := controller.CanFinish(contracts.StageSucceeded); finishErr == nil || finishErr.Code != "planner_finish_incomplete" {
		t.Fatalf("empty successful finish error = %+v", finishErr)
	}
	if _, planErr := controller.AddSubtask("Inspect", "Inspect input"); planErr != nil {
		t.Fatal(planErr)
	}
	if finishErr := controller.CanFinish(contracts.StageSucceeded); finishErr == nil || finishErr.Code != "planner_finish_incomplete" {
		t.Fatalf("pending successful finish error = %+v", finishErr)
	}
	claim, planErr := controller.ClaimCurrentSubtask("0", "reviewer")
	if planErr != nil {
		t.Fatal(planErr)
	}
	if finishErr := controller.CanFinish(contracts.StageFailed); finishErr == nil || finishErr.Code != "planner_finish_dispatch_active" {
		t.Fatalf("active failed finish error = %+v", finishErr)
	}
	if _, planErr := controller.CompleteDispatch(claim.CallID, contracts.StageSucceeded); planErr != nil {
		t.Fatal(planErr)
	}
	if finishErr := controller.CanFinish(contracts.StageSucceeded); finishErr != nil {
		t.Fatalf("completed successful finish error = %+v", finishErr)
	}
}

func TestPlannerPlanRejectsBoundsWithoutRevisionChange(t *testing.T) {
	controller, err := NewPlannerPlanController("Review the project")
	if err != nil {
		t.Fatal(err)
	}
	if _, planErr := controller.AddSubtask(" ", "instructions"); planErr == nil || planErr.Code != "planner_subtask_invalid" {
		t.Fatalf("invalid text error = %+v", planErr)
	}
	for index := 0; index < MaxPlannerSubtasks; index++ {
		if _, planErr := controller.AddSubtask("Task", "Instructions"); planErr != nil {
			t.Fatalf("add %d: %v", index, planErr)
		}
	}
	before := controller.Snapshot()
	if _, planErr := controller.AddSubtask("Overflow", "Instructions"); planErr == nil || planErr.Code != "planner_subtask_limit" {
		t.Fatalf("limit error = %+v", planErr)
	}
	if after := controller.Snapshot(); !reflect.DeepEqual(before, after) {
		t.Fatalf("rejected add mutated plan: before=%+v after=%+v", before, after)
	}
}

func TestPlannerPlanTransitionValidationRejectsHistoryTampering(t *testing.T) {
	controller, err := NewPlannerPlanController("Review the project")
	if err != nil {
		t.Fatal(err)
	}
	added, planErr := controller.AddSubtask("Inspect", "Read every declared input")
	if planErr != nil {
		t.Fatal(planErr)
	}
	if err := ValidatePlannerPlanTransition(nil, added.Projection(), PlannerEventPlanChanged); err != nil {
		t.Fatalf("valid append transition: %v", err)
	}
	beforeDispatch := added.Projection()
	claim, planErr := controller.ClaimCurrentSubtask("0", "reviewer")
	if planErr != nil {
		t.Fatal(planErr)
	}
	dispatched := controller.Snapshot().Projection()
	if err := ValidatePlannerPlanTransition(
		&beforeDispatch, dispatched, PlannerEventDispatchStarted,
	); err != nil {
		t.Fatalf("valid dispatch transition: %v", err)
	}
	beforeCompletion := clonePlannerPlanProjection(dispatched)
	completed, planErr := controller.CompleteDispatch(claim.CallID, contracts.StageSucceeded)
	if planErr != nil {
		t.Fatal(planErr)
	}
	if err := ValidatePlannerPlanTransition(
		&beforeCompletion, completed.Projection(), PlannerEventDispatchCompleted,
	); err != nil {
		t.Fatalf("valid completion transition: %v", err)
	}

	tests := []struct {
		name     string
		previous *PlannerPlanProjection
		next     PlannerPlanProjection
		kind     PlannerEventKind
	}{
		{
			name: "revision gap", previous: &beforeDispatch,
			next: func() PlannerPlanProjection {
				value := clonePlannerPlanProjection(dispatched)
				value.Revision++
				return value
			}(),
			kind: PlannerEventDispatchStarted,
		},
		{
			name: "model rewrites objective", previous: &beforeDispatch,
			next: func() PlannerPlanProjection {
				value := clonePlannerPlanProjection(dispatched)
				value.Subtasks[0].Objective = "rewritten"
				return value
			}(),
			kind: PlannerEventDispatchStarted,
		},
		{
			name: "wrong event kind", previous: &beforeDispatch,
			next: dispatched, kind: PlannerEventPlanChanged,
		},
		{
			name: "forged call identity", previous: &beforeDispatch,
			next: func() PlannerPlanProjection {
				value := clonePlannerPlanProjection(dispatched)
				value.ActiveDispatch.CallID = ""
				return value
			}(),
			kind: PlannerEventDispatchStarted,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := ValidatePlannerPlanTransition(test.previous, test.next, test.kind); err == nil {
				t.Fatal("tampered transition was accepted")
			}
		})
	}
}
