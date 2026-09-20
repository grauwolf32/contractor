package runtimeconfig

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestAllocationToolsetsProjectBeforeCredentialValidationAndKeepRunOrigins(t *testing.T) {
	input := resolverInput()
	input.Default.Spec.Worker.Toolsets = normalizedToolsetSpec(t, "default-mcp", map[string]any{"mcp@1": map[string]any{"endpoint": "https://old.example/mcp", "credential": "unavailable-old"}}).Worker.Toolsets
	input.RunLabels = []PinnedRuntimeConfig{testPin("knowledge", "1", normalizedToolsetSpec(t, "knowledge", map[string]any{"mcp@1": map[string]any{"endpoint": "https://new.example/mcp", "credential": "knowledge-key"}}))}
	native, err := ResolveRuntimeConfig(input)
	if err != nil || len(native.Toolsets) != 0 || len(native.Origins.Toolsets) != 0 || len(native.Provenance.RuntimeCredentialRefs) != 0 {
		t.Fatalf("unselected MCP configuration affected native allocation: %v", err)
	}
	input.ToolsetSelections = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "mcp", Version: "1"}, Tools: []string{"search"}}}
	if _, err := ResolveRuntimeConfig(input); err == nil {
		t.Fatal("missing selected credential accepted")
	}
	input.RuntimeCredentials = map[string]contracts.RuntimeCredentialKind{"knowledge-key": contracts.RuntimeCredentialMCPHeaders}
	first, err := ResolveRuntimeConfig(input)
	if err != nil || first.Validate() != nil {
		t.Fatalf("selected MCP resolution: %v", err)
	}
	input.ToolsetSelections[0].Tools = []string{"fetch"}
	second, err := ResolveRuntimeConfig(input)
	if err != nil || !reflect.DeepEqual(first.Toolsets, second.Toolsets) || len(first.RequiredRuntimeAdapters) != 0 {
		t.Fatalf("Run configuration differs across consumers: %v", err)
	}
	if first.Origins.Toolsets["mcp@1"].Layer != LayerRunLabels || len(first.Provenance.RuntimeCredentialRefs) != 1 {
		t.Fatal("toolset origin/credential hold is missing")
	}
	cloned := first.Clone()
	delete(cloned.Toolsets, "mcp@1")
	cloned.Origins.Toolsets["mcp@1"].Configs[0].Name = "changed"
	if first.Toolsets["mcp@1"] == nil || first.Origins.Toolsets["mcp@1"].Configs[0].Name == "changed" {
		t.Fatal("resolution clone shares mutable state")
	}
	raw, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	var stored ResolvedRuntimeConfig
	if err := json.Unmarshal(raw, &stored); err != nil || stored.Validate() != nil {
		t.Fatal("safe typed resolution cannot round-trip")
	}
	input.AgentLabels = []PinnedRuntimeConfig{testPin("host", "1", input.RunLabels[0].Spec)}
	if _, err := ResolveRuntimeConfig(input); err == nil {
		t.Fatal("Agent toolset override accepted")
	}
}
