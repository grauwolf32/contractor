package runstore

import (
	"encoding/json"
	"fmt"
	"strconv"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedRunEventPlannerSubtaskBounds(t *testing.T) {
	var fixture struct {
		Plans []struct {
			Count int
			Valid bool
		}
		IDs []struct {
			Value string
			Valid bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "planner-subtask-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, tc := range fixture.Plans {
		t.Run(fmt.Sprintf("plan/%d", tc.Count), func(t *testing.T) {
			plan := plannerRunPlanProjection{Revision: 1, CurrentSubtaskID: "0"}
			for index := range tc.Count {
				plan.Subtasks = append(plan.Subtasks, plannerRunSubtask{
					ID: strconv.Itoa(index), Objective: "x", Instructions: "y", Status: "pending",
				})
			}
			if err := validatePlannerRunPlan(plan); (err == nil) != tc.Valid {
				t.Fatalf("valid=%v: %v", tc.Valid, err)
			}
		})
	}
	for _, tc := range fixture.IDs {
		t.Run("id/"+tc.Value, func(t *testing.T) {
			if valid := ValidPlannerSubtaskID(tc.Value); valid != tc.Valid {
				t.Fatalf("valid=%v, want %v", valid, tc.Valid)
			}
		})
	}
}
