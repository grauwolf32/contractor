package controlplane

import (
	"errors"
	"testing"

	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestToolWorkerPlacementRequiresSelectedScannerWithoutModel(t *testing.T) {
	snapshot, err := config.Load("../../configs/scan", config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	template, err := snapshot.AgentTemplate("nuclei-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	registry := newTestRegistry(t, newTestClock())
	registration := testRegistration("tool-runtime")
	registration.SupportedRuntimes = []string{"tool@1"}
	registration.SupportedToolsets = []contracts.ToolsetCapability{{Ref: "scan@1", Tools: []string{"scan_naabu"}}}
	registerReadyWith(t, registry, registration)
	request := ReservationRequest{RunID: "tool-run", StageExecutionID: "tool-stage", Bindings: []BindingRequirement{{
		LogicalAgentName: "scanner", Namespace: "scanner", AgentTemplate: template, WorkerSessionMode: contracts.WorkerSessionIsolated,
	}}}
	if _, err := registry.ReserveAll(request); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("missing nuclei: %v", err)
	}
	registration.InstanceID = "nuclei-runtime"
	registration.ControlURL = "https://nuclei-runtime.example:9443"
	registration.A2AURL = "https://nuclei-runtime.example:9444"
	registration.SupportedToolsets[0].Tools = []string{"scan_nuclei"}
	registerReadyWith(t, registry, registration)
	reservations, err := registry.ReserveAll(request)
	if err != nil {
		t.Fatal(err)
	}
	if len(reservations) != 1 || reservations[0].Grant.RuntimeInstanceID != "nuclei-runtime" || reservations[0].ExecutionConfig != (AllocationExecutionConfig{}) {
		t.Fatalf("bad tool placement: %+v", reservations)
	}
}
