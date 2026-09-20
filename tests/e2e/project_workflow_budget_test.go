//go:build e2e

package e2e

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestProjectWorkerBudgetMatchesPinnedPolicy(t *testing.T) {
	catalog, err := config.Load(filepath.Join(repoRoot(t), "configs"), config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := catalog.Workflow("openapi-from-workspace@5")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages["dependency_discovery"]
	snapshot, err := json.Marshal(stage)
	if err != nil {
		t.Fatal(err)
	}
	policy := stage.ExecutionConfig.Agents["analyst"].ModelPolicy
	expected := contracts.WorkerBudgetMetrics{
		MaxModelCalls: int64(policy.MaxModelCalls), MaxToolCalls: int64(policy.MaxToolCalls),
		MaxTotalTokens: int64(policy.MaxTotalTokens), ObservedModelCalls: 5,
		ObservedToolCalls: 3, ObservedTotalTokens: 80,
	}
	if err := validateProjectWorkerBudget(snapshot, "analyst", &expected, 5, 3); err != nil {
		t.Fatalf("current pinned policy must accept the observed process counters: %v", err)
	}
	t.Run("immutable snapshot", func(t *testing.T) {
		selection := stage.ExecutionConfig.Agents["analyst"]
		selection.ModelPolicy.MaxModelCalls++
		stage.ExecutionConfig.Agents["analyst"] = selection
		if err := validateProjectWorkerBudget(snapshot, "analyst", &expected, 5, 3); err != nil {
			t.Fatalf("later authoring changes must not change pinned limits: %v", err)
		}
		// A separately pinned effective override must win over template defaults.
		overridden, err := json.Marshal(stage)
		if err != nil {
			t.Fatal(err)
		}
		if err := validateProjectWorkerBudget(overridden, "analyst", &expected, 5, 3); err == nil {
			t.Fatal("report used template defaults instead of the effective override")
		}
		matching := expected
		matching.MaxModelCalls++
		if err := validateProjectWorkerBudget(overridden, "analyst", &matching, 5, 3); err != nil {
			t.Fatalf("effective override was ignored: %v", err)
		}
	})
	for _, test := range []struct {
		name   string
		mutate func(*contracts.WorkerBudgetMetrics)
	}{
		{"model limit", func(b *contracts.WorkerBudgetMetrics) { b.MaxModelCalls++ }},
		{"tool limit", func(b *contracts.WorkerBudgetMetrics) { b.MaxToolCalls++ }},
		{"token limit", func(b *contracts.WorkerBudgetMetrics) { b.MaxTotalTokens++ }},
		{"model count", func(b *contracts.WorkerBudgetMetrics) { b.ObservedModelCalls++ }},
		{"tool count", func(b *contracts.WorkerBudgetMetrics) { b.ObservedToolCalls++ }},
		{"token count", func(b *contracts.WorkerBudgetMetrics) { b.ObservedTotalTokens++ }},
		{"unavailable usage", func(b *contracts.WorkerBudgetMetrics) { b.TokenUsageUnavailable++ }},
		{"exhausted", func(b *contracts.WorkerBudgetMetrics) { v := "model_calls"; b.Exhausted = &v }},
	} {
		t.Run(test.name, func(t *testing.T) {
			altered := expected
			test.mutate(&altered)
			if err := validateProjectWorkerBudget(snapshot, "analyst", &altered, 5, 3); err == nil {
				t.Fatal("altered budget report was accepted")
			}
		})
	}
	t.Run("missing budget", func(t *testing.T) {
		if err := validateProjectWorkerBudget(snapshot, "analyst", nil, 5, 3); err == nil {
			t.Fatal("missing budget was accepted")
		}
	})
	t.Run("missing allocated agent", func(t *testing.T) {
		if err := validateProjectWorkerBudget(snapshot, "absent", &expected, 5, 3); err == nil {
			t.Fatal("unknown allocated agent was accepted")
		}
	})
	t.Run("malformed snapshot", func(t *testing.T) {
		if err := validateProjectWorkerBudget([]byte("{"), "analyst", &expected, 5, 3); err == nil {
			t.Fatal("malformed pinned snapshot was accepted")
		}
	})
}
