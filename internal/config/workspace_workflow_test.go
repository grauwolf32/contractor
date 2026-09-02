package config

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestResolveWorkspaceContextAndExportContract(t *testing.T) {
	t.Parallel()
	required, optional := true, false
	artifacts := map[string]contextArtifactSource{
		"backend":        {Namespace: "inputs", Name: "backend", Required: &required},
		"frontend":       {Namespace: "inputs", Name: "frontend", Required: &required},
		"previous_state": {Namespace: "analysis", Name: "workspace_state", Required: &optional},
	}
	context, err := resolveStageContext(&stageContextSource{
		Artifacts: &artifacts,
		Workspace: &workspaceContextSource{
			Mode: "overlay",
			Sources: []workspaceSource{
				{Artifact: "backend", Target: "backend"},
				{Artifact: "frontend", Target: "frontend"},
			},
			State:  &workspaceStateInput{Artifact: "previous_state"},
			Export: &workspaceExport{State: "workspace_state", Diff: "workspace_diff"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := &WorkspaceContext{
		Mode: contracts.WorkspaceModeOverlay,
		Sources: []WorkspaceSource{
			{Artifact: "backend", Target: "backend"},
			{Artifact: "frontend", Target: "frontend"},
		},
		State:  &WorkspaceStateInput{Artifact: "previous_state"},
		Export: &WorkspaceExport{State: "workspace_state", Diff: "workspace_diff"},
	}
	if !reflect.DeepEqual(context.Workspace, want) {
		t.Fatalf("workspace = %+v, want %+v", context.Workspace, want)
	}
	result := StageResultContract{Artifacts: map[string]ArtifactSlot{
		"workspace_state": {Required: true, MediaTypes: []string{"application/vnd.contractor.workspace-overlay+json"}},
		"workspace_diff":  {Required: true, MediaTypes: []string{"text/x-diff"}},
	}}
	agents := workspaceAgents("workspace-changes")
	if err := validateStageWorkspace(context, result, agents); err != nil {
		t.Fatalf("valid workspace contract: %v", err)
	}
	if err := validateStageResultBindings(result, context.Workspace, agents); err != nil {
		t.Fatalf("valid Runtime-owned result bindings: %v", err)
	}
	badBinding := result
	badBinding.Artifacts = cloneArtifactSlots(result.Artifacts)
	state := badBinding.Artifacts["workspace_state"]
	state.From = &ArtifactBinding{Namespace: "worker", Name: "workspace_state"}
	badBinding.Artifacts["workspace_state"] = state
	if err := validateStageResultBindings(badBinding, context.Workspace, agents); err == nil ||
		!strings.Contains(err.Error(), "must be omitted") {
		t.Fatalf("Runtime-owned result binding error = %v", err)
	}

	stage := cloneStage(ResolvedStage{Context: context})
	stage.Context.Workspace.Sources[0].Target = "mutated"
	stage.Context.Workspace.Export.State = "mutated"
	if context.Workspace.Sources[0].Target != "backend" || context.Workspace.Export.State != "workspace_state" {
		t.Fatal("cloneStage aliases workspace state")
	}
}

func TestWorkspaceContextRejectsAliasesTargetsAndModeMismatch(t *testing.T) {
	t.Parallel()
	required := true
	artifacts := map[string]contextArtifactSource{
		"source": {Namespace: "inputs", Name: "source", Required: &required},
	}
	cases := []struct {
		name      string
		workspace workspaceContextSource
		fragment  string
	}{
		{"unknown alias", workspaceContextSource{Mode: "direct", Sources: []workspaceSource{{Artifact: "missing", Target: ""}}}, "unknown context artifact"},
		{"overlap", workspaceContextSource{Mode: "direct", Sources: []workspaceSource{{Artifact: "source", Target: "app"}, {Artifact: "source", Target: "app/api"}}}, "non-overlapping"},
		{"root overlap", workspaceContextSource{Mode: "direct", Sources: []workspaceSource{{Artifact: "source", Target: ""}, {Artifact: "source", Target: "app"}}}, "non-overlapping"},
		{"traversal", workspaceContextSource{Mode: "direct", Sources: []workspaceSource{{Artifact: "source", Target: "../app"}}}, "invalid component"},
		{"backslash", workspaceContextSource{Mode: "direct", Sources: []workspaceSource{{Artifact: "source", Target: `app\api`}}}, "relative POSIX"},
		{"direct export", workspaceContextSource{Mode: "direct", Sources: []workspaceSource{{Artifact: "source", Target: ""}}, Export: &workspaceExport{State: "state", Diff: "diff"}}, "requires overlay"},
	}
	for _, test := range cases {
		t.Run(test.name, func(t *testing.T) {
			_, err := resolveStageContext(&stageContextSource{Artifacts: &artifacts, Workspace: &test.workspace})
			if err == nil || !strings.Contains(err.Error(), test.fragment) {
				t.Fatalf("error = %v, want fragment %q", err, test.fragment)
			}
		})
	}
	optional := false
	optionalArtifacts := map[string]contextArtifactSource{
		"source": {Namespace: "inputs", Name: "source", Required: &optional},
	}
	_, err := resolveStageContext(&stageContextSource{
		Artifacts: &optionalArtifacts,
		Workspace: &workspaceContextSource{
			Mode: "direct", Sources: []workspaceSource{{Artifact: "source", Target: ""}},
		},
	})
	if err == nil || !strings.Contains(err.Error(), "required context artifact") {
		t.Fatalf("optional source error = %v", err)
	}
}

func TestWorkspaceToolsetAndResultCrossValidation(t *testing.T) {
	t.Parallel()
	overlay := StageContext{Artifacts: map[string]ContextArtifact{}, Workspace: &WorkspaceContext{
		Mode: contracts.WorkspaceModeOverlay, Sources: []WorkspaceSource{{Artifact: "source", Target: ""}},
		Export: &WorkspaceExport{State: "state", Diff: "diff"},
	}}
	validResult := StageResultContract{Artifacts: map[string]ArtifactSlot{
		"state": {MediaTypes: []string{"application/vnd.contractor.workspace-overlay+json"}},
		"diff":  {MediaTypes: []string{"text/x-diff"}},
	}}
	if err := validateStageWorkspace(StageContext{Artifacts: map[string]ContextArtifact{}}, StageResultContract{}, workspaceAgents("filesystem")); err == nil {
		t.Fatal("filesystem Toolset without workspace was accepted")
	}
	if err := validateStageWorkspace(StageContext{Artifacts: map[string]ContextArtifact{}}, StageResultContract{}, workspaceAgents("code-analysis")); err == nil ||
		!strings.Contains(err.Error(), "workspace-dependent Toolsets require context.workspace") {
		t.Fatalf("code-analysis@1 without workspace error = %v", err)
	}
	direct := overlay
	directWorkspace := *overlay.Workspace
	directWorkspace.Mode = contracts.WorkspaceModeDirect
	directWorkspace.Export = nil
	direct.Workspace = &directWorkspace
	if err := validateStageWorkspace(direct, StageResultContract{}, workspaceAgents("code-analysis")); err != nil {
		t.Fatalf("code-analysis@1 with direct workspace was rejected: %v", err)
	}
	if err := validateStageWorkspace(direct, StageResultContract{}, workspaceAgents("workspace-changes")); err == nil {
		t.Fatal("workspace-changes@1 with direct mode was accepted")
	}
	badResult := validResult
	badResult.Artifacts = cloneArtifactSlots(validResult.Artifacts)
	badResult.Artifacts["state"] = ArtifactSlot{MediaTypes: []string{"application/json"}}
	if err := validateStageWorkspace(overlay, badResult, workspaceAgents("workspace-changes")); err == nil {
		t.Fatal("overlay export with incorrect result media type was accepted")
	}
	if err := validateStageWorkspace(overlay, validResult, workspaceAgents("workspace-changes")); err != nil {
		t.Fatalf("valid workspace Toolset/result contract: %v", err)
	}
}

func TestWorkflowPublicationRequiresWorkspaceForCodeAnalysis(t *testing.T) {
	t.Parallel()

	root := copyConfigTree(t)
	path := filepath.Join(root, "agent-templates/artifact_builder.yaml")
	replaceFile(t, path, `    - ref: run-artifacts@1
      tools:
        - list_artifacts
        - read_artifact
        - write_artifact`, `    - ref: code-analysis@1
      tools: [search_def]`)
	snapshot, err := Load(root, MVPDescriptors())
	if err == nil || snapshot != nil ||
		!strings.Contains(err.Error(), "workspace-dependent Toolsets require context.workspace") {
		t.Fatalf("Load() = (%v, %v), want code-analysis workspace error", snapshot, err)
	}
}

func workspaceAgents(toolsetID string) map[string]ResolvedAgentBinding {
	return map[string]ResolvedAgentBinding{
		"worker": {Template: contracts.ResolvedAgentTemplate{Toolsets: []contracts.ToolsetSelection{{
			Ref: contracts.ToolsetRef{ToolsetID: toolsetID, Version: "1"}, Tools: []string{"tool"},
		}}}},
	}
}
