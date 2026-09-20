package controlplane

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestMixedFleetOnlyAllocationDiscoveryCanServeMCP(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	old := testRegistration("agent-a-old")
	registerReadyWith(t, registry, old)
	forged := testRegistration("agent-b-static-mcp")
	forged.SupportedToolsets = append(forged.SupportedToolsets, contracts.ToolsetCapability{Ref: "mcp@1", Tools: []string{"search"}})
	registerReadyWith(t, registry, forged)
	ready := testRegistration("agent-c-mcp")
	ready.SupportedToolsets = append(ready.SupportedToolsets, contracts.ToolsetCapability{Ref: "mcp@1", Tools: []string{}, ToolDiscovery: "allocation"})
	registerReadyWith(t, registry, ready)
	template := testTemplate(t)
	template.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "mcp", Version: "1"}, Tools: []string{"search"}}}
	reservations, err := registry.ReserveAll(ReservationRequest{RunID: "run-mcp", StageExecutionID: "stage-mcp", Bindings: []BindingRequirement{testBinding(t, "builder", "builder", template)}})
	if err != nil {
		t.Fatal(err)
	}
	if len(reservations) != 1 || reservations[0].Grant.RuntimeInstanceID != "agent-c-mcp" {
		t.Fatal("wrong Runtime selected")
	}
	// Snapshot cloning and Operations projections must preserve the discriminator.
	snapshot := registry.SnapshotOperations()
	found := false
	for _, agent := range snapshot.RuntimeAgents {
		for _, capability := range agent.SupportedToolsets {
			if agent.InstanceID == "agent-c-mcp" && capability.Ref == "mcp@1" {
				found = capability.ToolDiscovery == "allocation" && capability.Tools != nil && len(capability.Tools) == 0
			}
		}
	}
	if !found {
		t.Fatal("Operations projection lost allocation discovery")
	}
}
