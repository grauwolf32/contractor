package resolved

import (
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestPodmanRegisteredDescriptorsAndIsolation(t *testing.T) {
	d := registeredDescriptors()
	for _, selector := range []string{"podman@1", "podman-network@1"} {
		profile := d.SandboxProfiles[selector]
		if profile.WorkspaceMode != contracts.WorkspaceModeDirect || profile.WorkspaceStorage != contracts.WorkspaceStorageLocal {
			t.Fatalf("incorrect Podman requirements for %s: %+v", selector, profile)
		}
	}
	toolset := d.Toolsets["code-execution@1"]
	if !reflect.DeepEqual(toolset.Tools, []string{"exec_command"}) || !reflect.DeepEqual(toolset.RequiredSandboxProfiles, []string{"podman-network@1", "podman@1"}) || !reflect.DeepEqual(toolset.InfrastructureChannels["exec_command"], []infrastructureChannel{channelSandboxExecution}) {
		t.Fatalf("incorrect execution descriptor: %+v", toolset)
	}
	mixed := builtInDescriptors()
	mixed.Toolsets["code-execution@1"].InfrastructureChannels["exec_command"] = []infrastructureChannel{channelSandboxExecution, channelRuntimeSubprocessLauncher}
	if _, err := normalizeDescriptors(mixed); err == nil {
		t.Fatal("mixed host and sandbox authority accepted")
	}
}

func TestPodmanExecutionActiveClassificationUsesSelectedProfile(t *testing.T) {
	selection := contracts.ToolsetSelection{
		Ref:   contracts.ToolsetRef{ToolsetID: "code-execution", Version: "1"},
		Tools: []string{"exec_command"},
	}
	for _, profile := range []string{"podman", "podman-network"} {
		ref := contracts.SandboxProfileRef{SandboxProfileID: profile, Version: "1"}
		if got := SelectsActiveCheckTool(selection, ref); got != (profile == "podman-network") {
			t.Fatalf("incorrect active-check classification for %s: %v", profile, got)
		}
	}
}

func TestPodmanProfileAloneRequiresWorkspaceButAddsNoTools(t *testing.T) {
	template := contracts.ResolvedAgentTemplate{SandboxProfile: contracts.SandboxProfileRef{SandboxProfileID: "podman", Version: "1"}}
	agents := map[string]ResolvedAgentBinding{"worker": {Template: template}}
	if err := ValidateStageWorkspace(StageContext{}, StageResultContract{}, agents); err == nil {
		t.Fatal("profile without workspace accepted")
	}
	if err := ValidateStageWorkspace(StageContext{Workspace: &WorkspaceContext{Mode: contracts.WorkspaceModeDirect}}, StageResultContract{}, agents); err != nil {
		t.Fatal(err)
	}
	if len(agents["worker"].Template.Toolsets) != 0 {
		t.Fatal("profile added tools")
	}
}
