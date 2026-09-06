package scheduler

import (
	"testing"

	workflowconfig "github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/runstore"
)

func TestPodmanWorkspaceRequirementsStayBindingSpecific(t *testing.T) {
	snapshot, err := workflowconfig.Load("../../configs", workflowconfig.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	workflow, err := snapshot.Workflow("artifact-copy@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	ordinary := stage.Agents["builder"]
	container := ordinary
	container.Namespace = "executor"
	container.Template.SandboxProfile = contracts.SandboxProfileRef{SandboxProfileID: "podman", Version: "1"}
	stage.Agents["executor"] = container
	stage.ExecutionConfig.Agents["executor"] = stage.ExecutionConfig.Agents["builder"]
	stage.Context.Workspace = &workflowconfig.WorkspaceContext{Mode: contracts.WorkspaceModeDirect, Sources: []workflowconfig.WorkspaceSource{{Artifact: "source", Target: ""}}}
	revision := "revision-source"
	context := runstore.StageContextSnapshot{Artifacts: map[string]runstore.PinnedContextArtifact{
		"source": {Required: true, Artifact: &contracts.ArtifactRef{Namespace: "inputs", Name: "source", Revision: &revision}},
	}}
	bindings, err := bindingRequirements(stage, nil, context)
	if err != nil {
		t.Fatal(err)
	}
	if len(bindings) != 2 {
		t.Fatalf("bindings = %d", len(bindings))
	}
	memory := &contracts.WorkspaceCapabilitiesV2{Storage: contracts.WorkspaceStorageMemory, Modes: []contracts.WorkspaceModeV2{contracts.WorkspaceModeDirect}}
	local := *memory
	local.Storage = contracts.WorkspaceStorageLocal
	for _, binding := range bindings {
		if binding.Workspace == nil || *binding.Workspace.Sources[0].Artifact.Revision != revision {
			t.Fatal("workspace pin lost")
		}
		podman := binding.AgentTemplate.SandboxProfile.SandboxProfileID == "podman"
		if workflowconfig.SandboxWorkspaceCompatible(binding.AgentTemplate, binding.Workspace, memory) == podman {
			t.Fatal("incorrect memory placement")
		}
		if !workflowconfig.SandboxWorkspaceCompatible(binding.AgentTemplate, binding.Workspace, &local) {
			t.Fatal("local placement rejected")
		}
	}
}
