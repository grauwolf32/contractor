package planner

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/planner/stageresult"
)

func TestPlannerConfiguredSubtaskLimitCountsResolvedRecords(t *testing.T) {
	for _, limit := range []contracts.PlannerSubtaskLimit{0, 1, 2, 64, contracts.MaxPlannerSubtasks} {
		t.Run(fmt.Sprint(limit), func(t *testing.T) {
			controller, err := NewPlannerPlanController("Review input", limit)
			if err != nil {
				t.Fatal(err)
			}
			for index := range limit.Effective() {
				if _, err := controller.AddSubtask("Inspect", "Read input"); err != nil {
					t.Fatalf("append %d: %v", index, err)
				}
				// Resolving a record must not free its place in the plan.
				if index < 2 {
					claim, err := controller.ClaimCurrentSubtask(fmt.Sprint(index), "builder")
					if err != nil {
						t.Fatal(err)
					}
					outcome := stageresult.StageSucceeded
					if index == 1 {
						outcome = stageresult.StageFailed
					}
					if _, err := controller.CompleteDispatch(claim.CallID, outcome); err != nil {
						t.Fatal(err)
					}
				}
			}
			before := controller.Snapshot()
			if _, err := controller.AddSubtask("Overflow", "Read input"); err == nil || err.Code != "planner_subtask_limit" {
				t.Fatalf("overflow = %v", err)
			}
			if after := controller.Snapshot(); !reflect.DeepEqual(before, after) {
				t.Fatal("rejected append changed the plan")
			}
			if err := before.Validate(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
