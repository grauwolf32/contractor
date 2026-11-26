package contracts

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPlannerExecutionBounds(t *testing.T) {
	var bounds struct{ OperationBytes, InvocationBytes, Tools, Operations, RecentGroups, Compactions, Instructions int }
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "planner-execution-limits.json"), &bounds); err != nil {
		t.Fatal(err)
	}
	if bounds.OperationBytes != MaxPlannerOperationBytes || bounds.InvocationBytes != MaxPlannerInvocationBytes || bounds.Tools != MaxPlannerTools || bounds.Operations != MaxPlannerOperations || bounds.RecentGroups != MaxPlannerKeepRecentGroups || bounds.Compactions != MaxPlannerCompactions || bounds.Instructions != MaxPlannerCompactorInstructions {
		t.Fatal("Planner wire bounds drifted")
	}
	for _, response := range []bool{false, true} {
		for _, extra := range []int{0, 1} {
			var value any
			makeValue := func(text string) any {
				encoded, _ := json.Marshal(map[string]string{"text": text})
				if response {
					return PlannerOperationResult{APIVersion: APIVersion, Result: encoded}
				}
				return PlannerOperation{APIVersion: APIVersion, Sequence: 1, Name: "before_model", Arguments: encoded}
			}
			size, err := ResultJSONSize(makeValue(""))
			if err != nil {
				t.Fatal(err)
			}
			padding := bounds.OperationBytes - size + extra
			value = makeValue(strings.Repeat("é", padding/2) + strings.Repeat("x", padding%2))
			if err := value.(Validatable).Validate(); (err == nil) != (extra == 0) {
				t.Fatalf("response=%v overflow=%d: %v", response, extra, err)
			}
		}
	}
}
