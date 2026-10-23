package controlplane

import (
	"context"
	"encoding/json"
	"errors"
	"slices"
	"testing"
	"time"

	"github.com/grauwolf32/contractor/internal/config/resolved"
	"github.com/grauwolf32/contractor/internal/configload"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/control"
	"github.com/grauwolf32/contractor/internal/contracts/runtimesettings"
	"github.com/grauwolf32/contractor/internal/runstore"
)

// A shared Agent released by a scan goes to the model Stage that waits for
// it, not to the next scan, while a dedicated scan Agent keeps taking scans
// (CTL-R155).
func TestModelStageHoldPostgresGivesReleasedSharedAgentToWaitingModelStage(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	pool := isolatedPlacementPool(t, ctx)
	fixture := newPlacementFixture(t, ctx, pool, nil)
	// The hold must not lapse while the test runs, however slow the database.
	fixture.registry.modelStageHold = time.Hour
	catalog, err := configload.Load("../../testdata/configs-scan")
	if err != nil {
		t.Fatal(err)
	}
	scanTemplate, err := catalog.AgentTemplate("nuclei-scan@1")
	if err != nil {
		t.Fatal(err)
	}
	shared := fixture.registerProcess(t, ctx, "shared", "5", nil, templateCapabilities(fixture.template, scanTemplate))
	if _, err := runstore.NewPostgresStore(pool).CreateStageExecution(ctx, runstore.CreateStageExecutionParams{
		LaneClass:        resolved.LaneClassTool,
		StageExecutionID: "stage-scan", RunID: fixture.runID, StageName: "scan", Attempt: 1,
		ExecutionConfigVariant: runstore.StageExecutionConfigBase,
		StageSpecSchemaVersion: contracts.APIVersion, StageSpecSnapshot: json.RawMessage(`{}`),
		StageContextSchemaVersion: contracts.APIVersion,
		StageContext: runstore.StageContextSnapshot{
			Parameters: map[string]string{}, Artifacts: map[string]runstore.PinnedContextArtifact{},
		},
	}); err != nil {
		t.Fatal(err)
	}
	scanRequest := func(stageExecutionID string) ReservationRequest {
		return ReservationRequest{
			RunID: fixture.runID, StageExecutionID: stageExecutionID, RuntimeConfig: &fixture.runtimeSnapshot,
			ModelFree: true,
			Bindings: []BindingRequirement{{
				LogicalAgentName: "scanner", Namespace: "scanner", AgentTemplate: scanTemplate,
				WorkerSessionMode: contracts.WorkerSessionIsolated,
				RuntimeSelection:  &resolved.ResolvedConsumerExecutionConfig{},
			}},
		}
	}
	if _, err := fixture.registry.ReserveCandidateEdges(scanRequest("stage-running-scan"), []CandidateEdge{{
		LogicalAgentName: "scanner", RuntimeAgentID: shared, RuntimeAgentInstanceID: "shared",
		RuntimeAgentLabelRevision: 1, RequiredRuntimeAdapters: []runtimesettings.RuntimeAdapterRef{},
	}}); err != nil {
		t.Fatal(err)
	}
	model := fixture.request()
	if _, err := fixture.allocator.ReserveAllContext(ctx, model); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("model Stage beside a running scan: %v", err)
	}

	if err := fixture.registry.DiscardCandidateReservations("stage-running-scan"); err != nil {
		t.Fatal(err)
	}
	scan := scanRequest("stage-scan")
	if _, err := fixture.allocator.ReserveAllContext(ctx, scan); !errors.Is(err, ErrInsufficientCapacity) {
		t.Fatalf("next scan took the Agent the model Stage waits for: %v", err)
	}
	dedicated := fixture.registerProcess(t, ctx, "scan-only", "6", nil, templateCapabilities(scanTemplate))
	placed, err := fixture.allocator.ReserveAllContext(ctx, scan)
	if err != nil || len(placed) != 1 || placed[0].Grant.RuntimeAgentID != dedicated {
		t.Fatalf("scan on the dedicated Agent = (%+v, %v)", placed, err)
	}
	placed, err = fixture.allocator.ReserveAllContext(ctx, model)
	if err != nil || len(placed) != 1 || placed[0].Grant.RuntimeAgentID != shared {
		t.Fatalf("model Stage on its next attempt = (%+v, %v)", placed, err)
	}
	allocations, err := runstore.NewPostgresStore(pool).ListStageAllocations(ctx, fixture.stageExecutionID)
	if err != nil || len(allocations) != 1 || allocations[0].RuntimeAgentID != shared {
		t.Fatalf("durable model allocations = (%+v, %v)", allocations, err)
	}
}

// templateCapabilities advertises exactly what templates need, so one process
// can serve several Stages.
func templateCapabilities(templates ...contracts.ResolvedAgentTemplate) control.AgentRegistration {
	registration := control.AgentRegistration{
		InitialLabels: []string{}, SupportedRuntimeAdapters: []runtimesettings.RuntimeAdapterRef{},
	}
	for _, template := range templates {
		runtime := template.Runtime.RuntimeID + "@" + template.Runtime.Version
		if !slices.Contains(registration.SupportedRuntimes, runtime) {
			registration.SupportedRuntimes = append(registration.SupportedRuntimes, runtime)
		}
		sandbox := template.SandboxProfile.SandboxProfileID + "@" + template.SandboxProfile.Version
		if !slices.Contains(registration.SupportedSandboxProfiles, sandbox) {
			registration.SupportedSandboxProfiles = append(registration.SupportedSandboxProfiles, sandbox)
		}
		for _, toolset := range template.Toolsets {
			registration.SupportedToolsets = append(registration.SupportedToolsets, control.ToolsetCapability{
				Ref: toolset.Ref.ToolsetID + "@" + toolset.Ref.Version, Tools: append([]string{}, toolset.Tools...),
			})
		}
	}
	return registration
}
