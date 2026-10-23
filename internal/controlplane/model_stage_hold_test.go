package controlplane

import (
	"errors"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
)

// A model Stage waiting for an allocated Agent holds every live Agent it
// could use against model-free placement until the hold window lapses; model
// placement, replay of a recorded reservation and Agents the model Stage
// cannot use are unaffected (CTL-R155).
func TestModelStageHoldKeepsSharedAgentsFromModelFreePlacement(t *testing.T) {
	clock := newTestClock()
	registry := newTestRegistry(t, clock)
	registerReadyWith(t, registry, sharedHoldRegistration("shared-a"))
	registerReadyWith(t, registry, sharedHoldRegistration("shared-b"))
	running := holdScanRequest(t, "stage-running-scan")
	if reservations, err := registry.ReserveAll(running); err != nil || reservations[0].Grant.RuntimeInstanceID != "shared-a" {
		t.Fatalf("running scan = (%+v, %v)", reservations, err)
	}
	template := testTemplate(t)
	model := ReservationRequest{RunID: "run-model", StageExecutionID: "stage-model", Bindings: []BindingRequirement{
		testBinding(t, "first", "first", template), testBinding(t, "second", "second", template),
	}}
	if _, err := registry.ReserveCandidateEdges(model, liveHoldEdges(registry, model)); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("model Stage with one free shared Agent: %v", err)
	}
	if err := registry.holdForModelStage(model, liveHoldEdges(registry, model)); err != nil {
		t.Fatal(err)
	}

	scan := holdScanRequest(t, "stage-scan")
	if _, err := registry.ReserveAll(scan); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("scan placed on the Agent a waiting model Stage holds: %v", err)
	}
	if replayed, err := registry.ReserveAll(running); err != nil || replayed[0].Grant.RuntimeInstanceID != "shared-a" {
		t.Fatalf("replay of the running scan = (%+v, %v)", replayed, err)
	}
	other := ReservationRequest{RunID: "run-other", StageExecutionID: "stage-other", Bindings: []BindingRequirement{
		testBinding(t, "only", "only", template),
	}}
	placed, err := registry.ReserveCandidateEdges(other, liveHoldEdges(registry, other))
	if err != nil || placed[0].Grant.RuntimeInstanceID != "shared-b" {
		t.Fatalf("model Stage on a held Agent = (%+v, %v)", placed, err)
	}
	if err := registry.DiscardCandidateReservations(other.StageExecutionID); err != nil {
		t.Fatal(err)
	}

	dedicated := sharedHoldRegistration("scan-only")
	dedicated.SupportedRuntimes = []string{"tool@1"}
	dedicated.SupportedToolsets = dedicated.SupportedToolsets[1:]
	registerReadyWith(t, registry, dedicated)
	if placed, err := registry.ReserveAll(scan); err != nil || placed[0].Grant.RuntimeInstanceID != "scan-only" {
		t.Fatalf("scan on the dedicated Agent = (%+v, %v)", placed, err)
	}

	clock.Advance(defaultModelStageHold - time.Second)
	next := holdScanRequest(t, "stage-next-scan")
	if _, err := registry.ReserveAll(next); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("scan placed before the hold window lapsed: %v", err)
	}
	clock.Advance(time.Second)
	if placed, err := registry.ReserveAll(next); err != nil || placed[0].Grant.RuntimeInstanceID != "shared-b" {
		t.Fatalf("scan after the hold window = (%+v, %v)", placed, err)
	}
}

// A model Stage that no set of live Agents could serve, even with every
// allocation released, holds nothing (CTL-R155): here both of its Workers
// need the one Agent with code analysis.
func TestModelStageHoldIgnoresStageWithoutCompleteLiveMatching(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	analysis := sharedHoldRegistration("shared-a")
	analysis.SupportedToolsets = append(analysis.SupportedToolsets,
		control.ToolsetCapability{Ref: "code-analysis@1", Tools: []string{"list_symbols"}})
	registerReadyWith(t, registry, analysis)
	registerReadyWith(t, registry, sharedHoldRegistration("shared-b"))
	template := codeAnalysisTemplate(t, "list_symbols")
	model := ReservationRequest{RunID: "run-model", StageExecutionID: "stage-model", Bindings: []BindingRequirement{
		testBinding(t, "first", "first", template), testBinding(t, "second", "second", template),
	}}
	if _, err := registry.ReserveCandidateEdges(model, liveHoldEdges(registry, model)); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("two Workers on one code-analysis Agent: %v", err)
	}
	if err := registry.holdForModelStage(model, liveHoldEdges(registry, model)); err != nil {
		t.Fatal(err)
	}
	if placed, err := registry.ReserveAll(holdScanRequest(t, "stage-scan")); err != nil || placed[0].Grant.RuntimeInstanceID != "shared-a" {
		t.Fatalf("scan beside an unplaceable model Stage = (%+v, %v)", placed, err)
	}
}

// Only a Stage whose every Worker is a tool Worker is model-free, and only a
// model Stage holds Agents.
func TestModelStageHoldValidatesTheLaneClassOfTheRequest(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	registerReadyWith(t, registry, sharedHoldRegistration("shared-a"))
	model := ReservationRequest{RunID: "run-model", StageExecutionID: "stage-model", ModelFree: true, Bindings: []BindingRequirement{
		testBinding(t, "first", "first", testTemplate(t)),
	}}
	if _, err := registry.ReserveAll(model); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("model-free request with a model Worker: %v", err)
	}
	scan := holdScanRequest(t, "stage-scan")
	if err := registry.holdForModelStage(scan, liveHoldEdges(registry, scan)); !errors.Is(err, ErrInvalidRequest) {
		t.Fatalf("hold for a model-free Stage: %v", err)
	}
}

func holdScanRequest(t *testing.T, stageExecutionID string) ReservationRequest {
	t.Helper()
	snapshot, err := configload.Load("../../testdata/configs-scan")
	if err != nil {
		t.Fatal(err)
	}
	template, err := snapshot.AgentTemplate("nuclei-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	return ReservationRequest{
		RunID: "run-" + stageExecutionID, StageExecutionID: stageExecutionID, ModelFree: true,
		Bindings: []BindingRequirement{{
			LogicalAgentName: "scanner", Namespace: "scanner", AgentTemplate: template,
			WorkerSessionMode: contracts.WorkerSessionIsolated,
		}},
	}
}

// sharedHoldRegistration advertises both the ADK builder of testTemplate and
// the nuclei scanner, so model and model-free Stages share it.
func sharedHoldRegistration(instanceID string) control.AgentRegistration {
	registration := testRegistration(instanceID)
	registration.SupportedRuntimes = []string{"adk@1", "tool@1"}
	registration.SupportedToolsets = append(registration.SupportedToolsets,
		control.ToolsetCapability{Ref: "scan@1", Tools: []string{"scan_nuclei"}})
	return registration
}

// liveHoldEdges builds the edges placement would resolve for every live Agent
// that has no label or adapter requirement.
func liveHoldEdges(registry *InMemoryRegistry, request ReservationRequest) []CandidateEdge {
	edges := []CandidateEdge{}
	for _, candidate := range registry.liveCandidates() {
		for _, binding := range request.Bindings {
			if !isBindingCompatible(candidate.Registration, binding) {
				continue
			}
			edges = append(edges, CandidateEdge{
				LogicalAgentName: binding.LogicalAgentName, RuntimeAgentID: candidate.Principal.RuntimeAgentID,
				RuntimeAgentInstanceID:    candidate.Registration.InstanceID,
				RuntimeAgentLabelRevision: candidate.Principal.LabelRevision,
				RequiredRuntimeAdapters:   []runtimesettings.RuntimeAdapterRef{},
			})
		}
	}
	return edges
}
