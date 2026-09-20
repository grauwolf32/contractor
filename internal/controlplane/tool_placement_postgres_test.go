package controlplane

import (
	"context"
	"testing"
	"time"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPlacementPostgresToolWorkerWithoutModel(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedPlacementPool(t, ctx)
	fixture := newPlacementFixture(t, ctx, pool, nil)
	catalog, err := workflowconfig.Load("../../configs/scan", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	fixture.template, err = catalog.AgentTemplate("nuclei-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	fixture.selection = workflowconfig.ResolvedConsumerExecutionConfig{}
	principalID := fixture.registerCandidate(t, ctx, "tool-runtime", "5", nil)
	request := ReservationRequest{
		RunID: fixture.runID, StageExecutionID: fixture.stageExecutionID, RuntimeConfig: &fixture.runtimeSnapshot,
		Bindings: []BindingRequirement{{
			LogicalAgentName: "scanner", Namespace: "scanner", AgentTemplate: fixture.template,
			WorkerSessionMode: contracts.WorkerSessionIsolated, RuntimeSelection: &fixture.selection,
		}},
	}
	resolved, err := fixture.allocator.resolveCandidate(ctx, pool, request, request.Bindings[0], AuthenticatedPrincipal{
		RuntimeAgentID: principalID, Labels: []string{}, LabelRevision: 1,
	}, placementCredentialLookups{llm: fixture.allocator.llmCredentials, runtime: fixture.allocator.runtimeCredentials})
	if err != nil || !resolved.ModelFree || !resolved.ModelPolicy.IsZero() {
		t.Fatalf("model-free resolution = (%+v, %v)", resolved, err)
	}
	reservations, err := fixture.allocator.ReserveAllContext(ctx, request)
	if err != nil || len(reservations) != 1 {
		t.Fatalf("model-free reservations = (%+v, %v)", reservations, err)
	}
	allocations, err := runstore.NewPostgresStore(pool).ListStageAllocations(ctx, fixture.stageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].RuntimeConfiguration == nil {
		t.Fatalf("model-free durable allocations = (%+v, %v)", allocations, err)
	}
	if got := allocations[0].RuntimeConfiguration; got.ModelPolicy != (contracts.ModelPolicyRef{}) || got.Provenance.LLMGatewayConfig != nil || got.Provenance.LLMCredential != nil {
		t.Fatalf("model route persisted for scanner: %+v", got)
	}
}
