package scheduler

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRuntimePlannerKeepsWorkerNamespaceAndSnapshotIsolated(t *testing.T) {
	for _, strategy := range []string{"streamline", "router"} {
		stage := resolved.ResolvedStage{
			Planner: resolved.PlannerRef{PlannerID: strategy, Version: "1"},
			Agents:  map[string]resolved.ResolvedAgentBinding{"builder": {Namespace: "planner_executor"}},
		}
		stage.ExecutionConfig.Agents = map[string]resolved.ResolvedConsumerExecutionConfig{
			"builder": {ModelPolicy: contracts.ResolvedModelPolicy{Model: "pinned-worker"}},
		}
		stage.ExecutionConfig.Planner = &resolved.ResolvedConsumerExecutionConfig{
			ModelPolicy:          contracts.ResolvedModelPolicy{Model: "pinned-planner"},
			CompactorModelPolicy: &contracts.ResolvedModelPolicy{Model: "pinned-compactor"},
		}
		got := executionStage(stage)
		executor, ok := got.Agents["planner_executor_1"]
		if !ok || !executor.Template.Runtime.IsPlanner() || executor.Namespace == stage.Agents["builder"].Namespace {
			t.Fatalf("Planner collided with the Worker's explicit namespace: %+v", got.Agents)
		}
		if len(stage.Agents) != 1 || len(stage.ExecutionConfig.Agents) != 1 || stage.ExecutionConfig.Planner.CompactorModelPolicy == nil {
			t.Fatal("allocation view changed the pinned Stage snapshot")
		}
		selection := got.ExecutionConfig.Agents["planner_executor_1"]
		if selection.ModelPolicy.Model != "pinned-planner" || selection.CompactorModelPolicy != nil {
			t.Fatalf("Planner allocation selection = %+v", selection)
		}
	}
}
