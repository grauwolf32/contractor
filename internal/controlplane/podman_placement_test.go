package controlplane

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPodmanPlacementRequiresExactLocalDirectCapability(t *testing.T) {
	for _, test := range []struct {
		name                     string
		storage                  contracts.WorkspaceStorageV2
		mode                     contracts.WorkspaceModeV2
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
			registration := testRegistrationV2("podman-test")
			if test.profile {
				registration.SupportedSandboxProfiles = append(registration.SupportedSandboxProfiles, "podman@1")
			}
			if test.tool {
				registration.SupportedToolsets = append(registration.SupportedToolsets, contracts.ToolsetCapability{Ref: "code-execution@1", Tools: []string{"exec_command"}})
			}
			registration.WorkspaceCapabilities = &contracts.WorkspaceCapabilitiesV2{Storage: test.storage, Modes: []contracts.WorkspaceModeV2{test.mode}}
			template := testTemplate(t)
			template.SandboxProfile = contracts.SandboxProfileRef{SandboxProfileID: "podman", Version: "1"}
			template.Toolsets = []contracts.ToolsetSelection{{Ref: contracts.ToolsetRef{ToolsetID: "code-execution", Version: "1"}, Tools: []string{"exec_command"}}}
			var workspace *contracts.AllocationWorkspaceSpecV2
			if test.workspace {
				workspace = &contracts.AllocationWorkspaceSpecV2{Mode: test.mode}
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
