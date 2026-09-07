package config

import (
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRepositoryOpenAPIAuditGraphUsesPinnedSourceAndCanonicalResults(t *testing.T) {
	t.Parallel()
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	profile, err := snapshot.AuditProfile("openapi-operation-trace@2")
	if err != nil {
		t.Fatal(err)
	}
	binding := profile.Workflows[profile.Inventory.ItemWorkflowRole]
	stage := binding.Workflow.Stages[binding.Workflow.EntryStage]
	workspace := stage.Context.Workspace
	if workspace == nil || workspace.Mode != contracts.WorkspaceModeOverlay ||
		!reflect.DeepEqual(workspace.Sources, []WorkspaceSource{{Artifact: "source", Target: ""}}) ||
		workspace.State != nil || workspace.Export != nil {
		t.Fatalf("Audit graph workspace must contain only the pinned source: %+v", workspace)
	}
	if binding.Inputs["source"].Name != "source" ||
		stage.Context.Artifacts["source"] != (ContextArtifact{Namespace: "inputs", Name: "source", Required: true}) {
		t.Fatal("Audit source and graph workspace use different input bindings")
	}
	for _, alias := range []string{"task", "execution_manifest"} {
		if stage.Context.Artifacts[alias] != (ContextArtifact{Namespace: "inputs", Name: alias, Required: true}) {
			t.Fatalf("missing exact Audit context %s", alias)
		}
	}
	wantTools := map[string][]string{
		"run-artifacts@1": {"read_artifact"},
		"filesystem@1":    {"glob", "grep", "ls", "read_file"},
		"audit-results@1": {"read_audit_task", "submit_check_result"},
		"code-analysis@1": {
			"attack_surface", "complexity_hotspots", "entrypoint_paths_to",
			"find_callees", "find_callers", "find_symbol", "functions_that_raise",
			"graph_summary", "list_symbols", "paths_between", "search_def",
		},
	}
	if got := selectedToolsets(stage.Agents["checker"].Template.Toolsets); !reflect.DeepEqual(got, wantTools) {
		t.Fatalf("Audit graph tools = %+v, want %+v", got, wantTools)
	}
	checklist, err := snapshot.AuditProfile("source-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	plain := checklist.Workflows["check"].Workflow
	if !reflect.DeepEqual(binding.Workflow.Inputs, plain.Inputs) ||
		!reflect.DeepEqual(binding.Workflow.Outputs, plain.Outputs) ||
		!reflect.DeepEqual(stage.Result, plain.Stages[plain.EntryStage].Result) {
		t.Fatal("graph integration changed the canonical Audit task/result contract")
	}
	plainStage := plain.Stages[plain.EntryStage]
	if plainStage.Context.Workspace != nil || selectedTool(plainStage.Agents["checker"].Template.Toolsets, "code-analysis@1", "graph_summary") {
		t.Fatal("OpenAPI graph integration changed the checklist Worker")
	}
}
