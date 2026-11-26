package controlplane

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/artifacts/postgres/artifactpgtest"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/configload/configloadtest"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPlacementPostgresToolWorkerWithoutModel(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	pool := isolatedPlacementPool(t, ctx)
	fixture := newPlacementFixture(t, ctx, pool, nil)
	catalog, _, err := configload.LoadUnionReadOnly("../../testdata/configs-scan", filepath.Join(t.TempDir(), "managed"))
	if err != nil {
		t.Fatal(err)
	}
	fixture.template, err = configloadtest.AgentTemplate(catalog, "nuclei-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	fixture.selection = resolved.ResolvedConsumerExecutionConfig{}
	principalID := fixture.registerCandidate(t, ctx, "tool-runtime", "5", nil)
	request := ReservationRequest{
		RunID: fixture.runID, StageExecutionID: fixture.stageExecutionID, RuntimeConfig: &fixture.runtimeSnapshot,
		Bindings: []BindingRequirement{{
			LogicalAgentName: "scanner", Namespace: "scanner", AgentTemplate: fixture.template,
			WorkerSessionMode: contracts.WorkerSessionIsolated, ContextManagement: contracts.WorkerContextManagement{Strategy: contracts.ContextStrategyNone}, RuntimeSelection: &fixture.selection,
		}},
	}
	runtimeConfig, err := fixture.allocator.resolveCandidate(ctx, pool, request, request.Bindings[0], AuthenticatedPrincipal{
		RuntimeAgentID: principalID, Labels: []string{}, LabelRevision: 1,
	}, placementCredentialLookups{llm: fixture.allocator.llmCredentials, runtime: fixture.allocator.runtimeCredentials})
	if err != nil || !runtimeConfig.ModelFree || !runtimeConfig.ModelPolicy.IsZero() {
		t.Fatalf("model-free resolution = (%+v, %v)", runtimeConfig, err)
	}
	reservations, err := fixture.allocator.ReserveAllContext(ctx, request)
	if err != nil || len(reservations) != 1 {
		t.Fatalf("model-free reservations = (%+v, %v)", reservations, err)
	}
	allocations, err := runstore.NewPostgresStore(pool, artifactpgtest.NewStorage()).ListStageAllocations(ctx, fixture.stageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].RuntimeConfiguration == nil {
		t.Fatalf("model-free durable allocations = (%+v, %v)", allocations, err)
	}
	if got := allocations[0].RuntimeConfiguration; got.ModelPolicy != (contracts.ModelPolicyRef{}) || got.Provenance.LLMGatewayConfig != nil || got.Provenance.LLMCredential != nil {
		t.Fatalf("model route persisted for scanner: %+v", got)
	}
}

// Planner consumes the Stage's pinned route, independently of Worker settings
// and the selected process's mutable Agent-label configuration.
func TestPlacementPostgresPlannerExcludesWorkerConfiguration(t *testing.T) {
	ctx := t.Context()
	pool := isolatedPlacementPool(t, ctx)
	f := newPlacementFixture(t, ctx, pool, []string{"debug"})
	request := f.request()
	binding := request.Bindings[0]
	binding.AgentTemplate.Runtime = contracts.WorkerRuntimeRef{RuntimeID: "planner", Version: "1"}
	binding.AgentTemplate.Toolsets = nil
	binding.AgentTemplate.Skills = nil
	got, err := f.allocator.resolveCandidate(ctx, pool, request, binding, AuthenticatedPrincipal{
		Labels: []string{"unbound-worker-only-label"}, LabelRevision: 1,
	}, placementCredentialLookups{llm: f.allocator.llmCredentials, runtime: f.allocator.runtimeCredentials})
	if err != nil {
		t.Fatal(err)
	}
	if got.LLMGateway.Ref != f.selection.LLMGateway.Ref || got.ModelPolicy.Ref != f.selection.ModelPolicy.Ref ||
		got.LLMCredential == nil || *got.LLMCredential != *f.selection.Credential || got.WorkerTelemetry != nil ||
		got.HTTPProxy != nil || got.Caido != nil || got.Interactsh != nil || len(got.RequiredRuntimeAdapters) != 0 ||
		len(got.Provenance.AgentLabels) != 0 || len(got.Provenance.RuntimeCredentialRefs) != 0 {
		t.Fatalf("Planner inherited Worker configuration: %+v", got)
	}
	// Resolution must not mutate the cached Run-label snapshot used by Workers.
	worker, err := f.allocator.resolveCandidate(ctx, pool, request, request.Bindings[0], AuthenticatedPrincipal{
		Labels: []string{}, LabelRevision: 1,
	}, placementCredentialLookups{llm: f.allocator.llmCredentials, runtime: f.allocator.runtimeCredentials})
	if err != nil || worker.WorkerTelemetry == nil {
		t.Fatalf("Planner projection altered Worker settings: %+v, %v", worker, err)
	}
}
