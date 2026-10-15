package public

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/planner"
)

func TestSharedPublicPlannerSubtaskContracts(t *testing.T) {
	var fixture struct {
		Default int
		Limits  []struct {
			Name      string
			Present   bool
			Value     any
			Valid     bool
			Effective int
		}
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
	document := loadPublicOpenAPI(t)
	catalog, err := configload.Load(publicCatalogFixture)
	if err != nil {
		t.Fatal(err)
	}
	baseline, err := catalog.Workflow("artifact-copy@2")
	if err != nil {
		t.Fatal(err)
	}
	baselines := map[string]any{
		"WorkflowSummary":  workflowSummaryReadModel(baseline),
		"WorkflowResource": workflowResourceReadModel(baseline),
	}
	raw, err := os.ReadFile(publicEventsPath)
	if err != nil {
		t.Fatal(err)
	}
	var events map[string]any
	if err := json.Unmarshal(raw, &events); err != nil {
		t.Fatal(err)
	}
	definitions := events["$defs"].(map[string]any)
	eventSchema := func(name string) *openapi3.Schema {
		t.Helper()
		expanded, err := dereferenceLocalDefinitions(definitions[name], definitions, nil)
		if err != nil {
			t.Fatal(err)
		}
		encoded, err := json.Marshal(expanded)
		if err != nil {
			t.Fatal(err)
		}
		var schema openapi3.Schema
		if err := json.Unmarshal(encoded, &schema); err != nil {
			t.Fatal(err)
		}
		return &schema
	}
	for _, tc := range fixture.Limits {
		t.Run("limit/"+tc.Name, func(t *testing.T) {
			for _, name := range []string{"WorkflowSummary", "WorkflowResource"} {
				schema := document.Components.Schemas[name].Value
				if schema.Properties["maxSubtasks"].Value.Default != float64(fixture.Default) {
					t.Fatal("public schema default differs")
				}
				encoded, err := json.Marshal(baselines[name])
				if err != nil {
					t.Fatal(err)
				}
				var value map[string]any
				if err := json.Unmarshal(encoded, &value); err != nil {
					t.Fatal(err)
				}
				if tc.Present {
					value["maxSubtasks"] = tc.Value
				}
				if err := schema.VisitJSON(value, openapi3.EnableJSONSchema2020()); (err == nil) != tc.Valid {
					t.Fatalf("%s valid=%v: %v", name, tc.Valid, err)
				}
			}
			if !tc.Valid {
				return
			}
			limit := contracts.PlannerSubtaskLimit(0)
			if tc.Present {
				limit = contracts.PlannerSubtaskLimit(tc.Effective)
			}
			workflow := resolved.ResolvedWorkflow{MaxSubtasks: limit}
			for _, projection := range []any{workflowSummaryReadModel(workflow), workflowResourceReadModel(workflow)} {
				encoded, err := json.Marshal(projection)
				if err != nil {
					t.Fatal(err)
				}
				var value map[string]any
				if err := json.Unmarshal(encoded, &value); err != nil {
					t.Fatal(err)
				}
				actual, present := value["maxSubtasks"]
				if present != tc.Present || present && actual != tc.Value {
					t.Fatalf("projection limit=%v present=%v", actual, present)
				}
			}
		})
	}
	planSchemas := map[string]*openapi3.Schema{
		"public": document.Components.Schemas["PlannerPlan"].Value,
		"events": eventSchema("PlannerPlan"),
	}
	for _, tc := range fixture.Plans {
		t.Run(fmt.Sprintf("plan/%d", tc.Count), func(t *testing.T) {
			plan := planner.PlannerPlanProjection{Revision: 1, CurrentSubtaskID: "0"}
			for index := range tc.Count {
				plan.Subtasks = append(plan.Subtasks, planner.PlannerSubtask{
					ID: strconv.Itoa(index), Objective: "x", Instructions: "y", Status: planner.PlannerSubtaskPending,
				})
			}
			if err := plan.Validate(); (err == nil) != tc.Valid {
				t.Fatalf("Planner valid=%v: %v", tc.Valid, err)
			}
			encoded, err := json.Marshal(plan)
			if err != nil {
				t.Fatal(err)
			}
			var value any
			if err := json.Unmarshal(encoded, &value); err != nil {
				t.Fatal(err)
			}
			for name, schema := range planSchemas {
				if err := schema.VisitJSON(value, openapi3.EnableJSONSchema2020()); (err == nil) != tc.Valid {
					t.Fatalf("%s valid=%v: %v", name, tc.Valid, err)
				}
			}
		})
	}
	idSchemas := []*openapi3.Schema{
		document.Components.Schemas["PlannerSubtask"].Value.Properties["id"].Value,
		document.Components.Schemas["PlannerDispatch"].Value.Properties["subtaskId"].Value,
		document.Components.Schemas["PlannerPlan"].Value.Properties["currentSubtaskId"].Value,
		eventSchema("PlannerSubtask").Properties["id"].Value,
		eventSchema("ActiveDispatch").Properties["subtaskId"].Value,
		eventSchema("PlannerPlan").Properties["currentSubtaskId"].Value,
		eventSchema("PlannerEventData").Properties["subtaskId"].Value,
	}
	for _, tc := range fixture.IDs {
		t.Run("id/"+tc.Value, func(t *testing.T) {
			for index, schema := range idSchemas {
				if err := schema.VisitJSON(tc.Value, openapi3.EnableJSONSchema2020()); (err == nil) != tc.Valid {
					t.Fatalf("schema %d valid=%v: %v", index, tc.Valid, err)
				}
			}
		})
	}
}
