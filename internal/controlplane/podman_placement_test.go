package controlplane

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPodmanRegisteredFleetPreservesOrdinaryMemoryAndOverlaySlots(t *testing.T) {
	registry := newTestRegistry(t, newTestClock())
	for _, candidate := range []struct {
		id      string
		storage contracts.WorkspaceStorage
		mode    contracts.WorkspaceMode
	}{
		{"agent-a-memory", contracts.WorkspaceStorageMemory, contracts.WorkspaceModeDirect},
		{"agent-b-podman", contracts.WorkspaceStorageLocal, contracts.WorkspaceModeDirect},
		{"agent-c-overlay", contracts.WorkspaceStorageLocal, contracts.WorkspaceModeOverlay},
	} {
		registration := testRegistration(candidate.id)
		// Even an optimistic remote claim cannot bypass exact storage/mode checks.
		registration.SupportedSandboxProfiles = append(registration.SupportedSandboxProfiles, "podman@1")
		registration.SupportedToolsets = append(registration.SupportedToolsets,
			contracts.ToolsetCapability{Ref: "code-execution@1", Tools: []string{"exec_command"}})
		registration.WorkspaceCapabilities = &contracts.WorkspaceCapabilities{
			Storage: candidate.storage, Modes: []contracts.WorkspaceMode{candidate.mode},
			Limits: contracts.WorkspaceLimits{MaxFiles: 100, MaxExpandedBytes: 1024,
				MaxManagedTextBytes: 1024, MaxFileBytes: 1024},
		}
		principal := inProcessPrincipal(candidate.id)
		if _, err := registry.RegisterAuthenticated(principal, registration); err != nil {
			t.Fatal(err)
		}
		for _, beat := range []contracts.AgentHeartbeat{heartbeat(candidate.id, 1, 0), heartbeat(candidate.id, 2, 1)} {
			if _, err := registry.HeartbeatAuthenticated(principal.RuntimeAgentID, beat); err != nil {
				t.Fatal(err)
			}
		}
	}
	template := testTemplate(t)
	template.SandboxProfile = contracts.SandboxProfileRef{SandboxProfileID: "podman", Version: "1"}
	template.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "code-execution", Version: "1"}, Tools: []string{"exec_command"}}}
	bindings := []BindingRequirement{
		testBinding(t, "a-generic", "generic", testTemplate(t)),
		testBinding(t, "b-executor", "executor", template),
		testBinding(t, "c-overlay", "overlay", testTemplate(t)),
	}
	for i := range bindings {
		mode := contracts.WorkspaceModeDirect
		if i == 2 {
			mode = contracts.WorkspaceModeOverlay
		}
		bindings[i].Workspace = &contracts.AllocationWorkspaceSpec{Mode: mode,
			Sources: []contracts.AllocationWorkspaceSource{{Artifact: exactWorkspaceRef("source", "revision-1"), Target: ""}}}
	}
	reservations, err := registry.ReserveAll(ReservationRequest{RunID: "run-podman-fleet", StageExecutionID: "stage-podman-fleet", Bindings: bindings})
	if err != nil {
		t.Fatal(err)
	}
	if len(reservations) != 3 {
		t.Fatalf("reservations = %d", len(reservations))
	}
	for i, want := range []string{"agent-a-memory", "agent-b-podman", "agent-c-overlay"} {
		if reservations[i].Grant.RuntimeInstanceID != want {
			t.Fatalf("binding %d selected %s, want %s", i, reservations[i].Grant.RuntimeInstanceID, want)
		}
	}
}

func TestPodmanPlacementRequiresExactLocalDirectCapability(t *testing.T) {
	for _, test := range []struct {
		name                     string
		storage                  contracts.WorkspaceStorage
		mode                     contracts.WorkspaceMode
		profile, tool, workspace bool
		want                     bool
	}{
		{"local-direct", contracts.WorkspaceStorageLocal, contracts.WorkspaceModeDirect, true, true, true, true},
		{"memory-direct", contracts.WorkspaceStorageMemory, contracts.WorkspaceModeDirect, true, true, true, false},
		{"local-overlay", contracts.WorkspaceStorageLocal, contracts.WorkspaceModeOverlay, true, true, true, false},
		{"missing-workspace", contracts.WorkspaceStorageLocal, contracts.WorkspaceModeDirect, true, true, false, false},
		{"missing-profile", contracts.WorkspaceStorageLocal, contracts.WorkspaceModeDirect, false, true, true, false},
		{"missing-tool", contracts.WorkspaceStorageLocal, contracts.WorkspaceModeDirect, true, false, true, false},
	} {
		t.Run(test.name, func(t *testing.T) {
			registration := testRegistration("podman-test")
			if test.profile {
				registration.SupportedSandboxProfiles = append(registration.SupportedSandboxProfiles, "podman@1")
			}
			if test.tool {
				registration.SupportedToolsets = append(registration.SupportedToolsets, contracts.ToolsetCapability{Ref: "code-execution@1", Tools: []string{"exec_command"}})
			}
			registration.WorkspaceCapabilities = &contracts.WorkspaceCapabilities{Storage: test.storage, Modes: []contracts.WorkspaceMode{test.mode}}
			template := testTemplate(t)
			template.SandboxProfile = contracts.SandboxProfileRef{SandboxProfileID: "podman", Version: "1"}
			template.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "code-execution", Version: "1"}, Tools: []string{"exec_command"}}}
			var workspace *contracts.AllocationWorkspaceSpec
			if test.workspace {
				workspace = &contracts.AllocationWorkspaceSpec{Mode: test.mode}
			}
			if got := isCompatible(registration, template, workspace); got != test.want {
				t.Fatalf("compatible = %v, want %v", got, test.want)
			}
			// The same candidate is not globally excluded for ordinary Workers.
			if !isCompatible(registration, testTemplate(t), workspace) {
				t.Fatal("Podman requirement leaked to ordinary binding")
			}
		})
	}
}
