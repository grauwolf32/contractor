package resolved

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

// The descriptor registry lists exactly the shared Planners with their facts,
// and gives every shared Stage case the verdict the Studio gives it.
func TestSharedPlannerFactCases(t *testing.T) {
	t.Parallel()
	var fixture struct {
		Planners []struct {
			Ref          string `json:"ref"`
			SingleAgent  bool   `json:"singleAgent"`
			ModelFree    bool   `json:"modelFree"`
			AgentResults bool   `json:"agentResults"`
		} `json:"planners"`
		Cases []struct {
			Name            string `json:"name"`
			Planner         string `json:"planner"`
			Agents          int    `json:"agents"`
			ResultFromAgent bool   `json:"resultFromAgent"`
			Valid           bool   `json:"valid"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "planner-fact-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	shared := make([]string, 0, len(fixture.Planners))
	for _, planner := range fixture.Planners {
		shared = append(shared, planner.Ref)
		ref := plannerRef(t, planner.Ref)
		if !KnownPlanner(ref) {
			t.Errorf("shared Planner %q has no descriptor", planner.Ref)
			continue
		}
		if got := IsSingleAgentPlanner(ref); got != planner.SingleAgent {
			t.Errorf("%s singleAgent = %t, want %t", planner.Ref, got, planner.SingleAgent)
		}
		if got := IsModelFreePlanner(ref); got != planner.ModelFree {
			t.Errorf("%s modelFree = %t, want %t", planner.Ref, got, planner.ModelFree)
		}
		if got := PlannerBindsAgentResults(ref); got != planner.AgentResults {
			t.Errorf("%s agentResults = %t, want %t", planner.Ref, got, planner.AgentResults)
		}
	}
	for _, selector := range PlannerSelectors() {
		if !slices.Contains(shared, selector) {
			t.Errorf("PlannerFactory descriptor %q is not a shared Planner", selector)
		}
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("no shared Planner cases")
	}
	for _, tc := range fixture.Cases {
		ref := plannerRef(t, tc.Planner)
		valid := ValidatePlannerAgents(ref, tc.Agents) == nil && (tc.ResultFromAgent || !PlannerBindsAgentResults(ref))
		if valid != tc.Valid {
			t.Errorf("%s: valid = %t, want %t", tc.Name, valid, tc.Valid)
		}
	}
}

func plannerRef(t *testing.T, selector string) PlannerRef {
	t.Helper()
	parsed, err := ParseSelector(selector)
	if err != nil {
		t.Fatal(err)
	}
	return PlannerRef{PlannerID: parsed.ID, Version: parsed.Version}
}
