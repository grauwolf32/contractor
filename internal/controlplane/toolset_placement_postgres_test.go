package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/credentials"
	"github.com/grauwolf32/contractor/internal/runstore"
	"github.com/grauwolf32/contractor/internal/runtimeconfig"
)

type mcpPlacementCredentialLookup struct{ calls []string }

func (l *mcpPlacementCredentialLookup) Get(_ context.Context, id string) (credentials.RuntimeCredentialMetadata, error) {
	l.calls = append(l.calls, id)
	if id != "knowledge-access" {
		return credentials.RuntimeCredentialMetadata{}, errors.New("unselected credential lookup")
	}
	return credentials.RuntimeCredentialMetadata{CredentialID: id, Kind: credentials.RuntimeCredentialMCPHeaders}, nil
}

func TestPlacementPostgresProjectsMCPBeforeLookupAndPersistsOnlySafeOrigins(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 45*time.Second)
	defer cancel()
	pool := isolatedPlacementPool(t, ctx)
	fixture := newPlacementFixture(t, ctx, pool, nil)
	publisher, err := runtimeconfig.NewPublisher(runtimeconfig.PublisherOptions{Pool: pool, RuntimeCredentials: placementRuntimeCatalog{}, PlannerTelemetryAdapters: runtimeconfig.PlannerTelemetryAdapterCatalogFunc(func(string) bool { return true })})
	if err != nil {
		t.Fatal(err)
	}
	published, err := publisher.Publish(ctx, []byte(`{"apiVersion":"contractor/v1alpha1","kind":"RuntimeConfig","metadata":{"name":"knowledge","version":"1"},"spec":{"worker":{"toolsets":{"mcp@1":{"endpoint":"https://knowledge.example/mcp","credential":"knowledge-access"}}}}}`), "publish-knowledge", "operator")
	if err != nil {
		t.Fatal(err)
	}
	fixture.runtimeSnapshot.Labels = []runtimeconfig.PinnedLabel{{Label: "knowledge", Explicit: true, BindingRevision: 1, Config: published.Version.Ref}}
	lookup := &mcpPlacementCredentialLookup{}
	fixture.allocator.runtimeCredentials = lookup
	request := fixture.request()
	native, err := fixture.allocator.resolveCandidate(ctx, pool, request, request.Bindings[0], AuthenticatedPrincipal{})
	if err != nil || len(lookup.calls) != 0 || len(native.Toolsets) != 0 || len(native.Provenance.RuntimeCredentialRefs) != 0 {
		t.Fatalf("native allocation accessed MCP: %v", err)
	}
	fixture.template.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "mcp", Version: "1"}, Tools: []string{"search"}}}
	request = fixture.request()
	first, err := fixture.allocator.resolveCandidate(ctx, pool, request, request.Bindings[0], AuthenticatedPrincipal{})
	if err != nil {
		t.Fatal(err)
	}
	request.Bindings[0].AgentTemplate.Toolsets[0].Tools = []string{"fetch"}
	second, err := fixture.allocator.resolveCandidate(ctx, pool, request, request.Bindings[0], AuthenticatedPrincipal{})
	if err != nil || !reflect.DeepEqual(first.Toolsets, second.Toolsets) || !reflect.DeepEqual(lookup.calls, []string{"knowledge-access", "knowledge-access"}) {
		t.Fatalf("shared consumer resolution differs: %v", err)
	}
	fixture.registerCandidate(t, ctx, "mcp-runtime", "e", nil)
	reservations, err := fixture.allocator.ReserveAllContext(ctx, fixture.request())
	if err != nil || len(reservations) != 1 {
		t.Fatalf("MCP placement failed: %v", err)
	}
	allocations, err := runstore.NewPostgresStore(pool).ListStageAllocations(ctx, fixture.stageExecutionID)
	if err != nil || len(allocations) != 1 {
		t.Fatal("MCP allocation not persisted")
	}
	safe := allocations[0].RuntimeConfiguration
	if safe.Origins.Toolsets["mcp@1"].Layer != runtimeconfig.LayerRunLabels || len(safe.Provenance.RuntimeCredentialRefs) != 1 {
		t.Fatal("durable MCP origin/credential hold is missing")
	}
	raw, err := json.Marshal(safe)
	if err != nil {
		t.Fatal(err)
	}
	var object map[string]any
	if json.Unmarshal(raw, &object) != nil || object["toolsets"] != nil || object["headers"] != nil {
		t.Fatal("allocation persistence contains connection body")
	}
}
