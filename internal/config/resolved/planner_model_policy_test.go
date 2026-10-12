package resolved

import (
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPlannerModelPolicyRequiresPlannerBudgetsAndNoToolBudget(t *testing.T) {
	valid := contracts.ResolvedModelPolicy{
		Ref:   contracts.ModelPolicyRef{PolicyID: "planner", Version: "1", Digest: "sha256:" + strings.Repeat("a", 64)},
		Model: "planner-model", MaxOutputTokens: 1024, MaxModelCalls: 8, MaxWorkerCalls: 4, MaxTotalTokens: 32768,
	}
	if err := ValidatePlannerModelPolicy(valid); err != nil {
		t.Fatalf("valid Planner policy rejected: %v", err)
	}
	for name, change := range map[string]func(*contracts.ResolvedModelPolicy){
		"without maxOutputTokens": func(policy *contracts.ResolvedModelPolicy) { policy.MaxOutputTokens = 0 },
		"without maxModelCalls":   func(policy *contracts.ResolvedModelPolicy) { policy.MaxModelCalls = 0 },
		"without maxWorkerCalls":  func(policy *contracts.ResolvedModelPolicy) { policy.MaxWorkerCalls = 0 },
		"without maxTotalTokens":  func(policy *contracts.ResolvedModelPolicy) { policy.MaxTotalTokens = 0 },
		"with maxToolCalls":       func(policy *contracts.ResolvedModelPolicy) { policy.MaxToolCalls = 1 },
		"without a model":         func(policy *contracts.ResolvedModelPolicy) { policy.Model = " " },
		"over the Worker call bound": func(policy *contracts.ResolvedModelPolicy) {
			policy.MaxWorkerCalls = contracts.MaxPlannerWorkerCalls + 1
		},
	} {
		policy := valid
		change(&policy)
		if err := ValidatePlannerModelPolicy(policy); err == nil {
			t.Errorf("Planner policy %s accepted", name)
		}
	}
}
