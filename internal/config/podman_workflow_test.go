package config

import (
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRepositoryPodmanWorkflowUsesExplicitLocalDirectToolsAndOrdinaryOutput(t *testing.T) {
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	workflow, err := snapshot.Workflow("podman-python-check@1")
	if err != nil {
		t.Fatal(err)
	}
	if workflow.EntryStage != "check" || len(workflow.Stages) != 1 {
		t.Fatalf("unexpected graph: %+v", workflow)
	}
	stage := workflow.Stages["check"]
	workspace := stage.Context.Workspace
	if workspace == nil || workspace.Mode != contracts.WorkspaceModeDirect || workspace.Export != nil || workspace.State != nil ||
		!reflect.DeepEqual(workspace.Sources, []WorkspaceSource{{Artifact: "source", Target: ""}}) {
		t.Fatalf("unexpected workspace: %+v", workspace)
	}
	template, err := snapshot.AgentTemplate("podman_python_fixer@1")
	if err != nil {
		t.Fatal(err)
	}
	if template.SandboxProfile.SandboxProfileID != "podman" || template.SandboxProfile.Version != "1" || len(template.Skills) != 0 {
		t.Fatalf("unexpected authority: %+v", template)
	}
	if !reflect.DeepEqual(selectedToolsets(template.Toolsets), map[string][]string{
		"filesystem@1": {"read_file"}, "edit-files@1": {"edit"},
		"code-execution@1": {"exec_command"}, "run-artifacts@1": {"write_artifact"},
	}) {
		t.Fatalf("unexpected tools: %+v", template.Toolsets)
	}
	if len(stage.Agents) != 1 || stage.Agents["builder"].Template.Ref != template.Ref {
		t.Fatalf("unexpected binding: %+v", stage.Agents)
	}
	assertStageResult(t, stage, "report", "application/json")
	if len(stage.Result.Artifacts) != 1 || !reflect.DeepEqual(stage.WorkflowOutputs, map[string]string{"report": "report"}) {
		t.Fatalf("unexpected outputs: %+v", stage)
	}
}
