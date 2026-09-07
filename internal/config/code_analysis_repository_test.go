package config

import (
	"reflect"
	"strings"
	"testing"
)

func TestRepositoryCodeAnalysisTemplateHasExactGraphToolSurface(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	graph, err := snapshot.AgentTemplate("workspace_source_graph_analyst@1")
	if err != nil {
		t.Fatal(err)
	}
	assertDigest(t, graph.Ref.Digest)
	assertDigest(t, graph.Instructions.Digest)
	if graph.Instructions.Ref != "instructions/workspace-source-graph-analyst-worker.md" {
		t.Fatalf("graph instructions ref = %q", graph.Instructions.Ref)
	}
	want := map[string][]string{
		"filesystem@1":     {"glob", "grep", "ls", "read_file"},
		"text-artifacts@1": {"read_text_artifact", "write_text_artifact"},
		"code-analysis@1": {
			"attack_surface", "complexity_hotspots", "entrypoint_paths_to",
			"find_callees", "find_callers", "find_symbol", "functions_that_raise",
			"graph_summary", "list_symbols", "paths_between", "search_def",
		},
	}
	if got := selectedToolsets(graph.Toolsets); !reflect.DeepEqual(got, want) {
		t.Fatalf("graph analyst tools = %+v, want %+v", got, want)
	}
}

func TestRepositoryCodeAnalysisWorkerPromptBoundary(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	template, err := snapshot.AgentTemplate("workspace_source_graph_analyst@1")
	if err != nil {
		t.Fatal(err)
	}
	text := strings.ToLower(template.Instructions.Text)
	for _, forbidden := range []string{
		"agenttemplate", "allocationspec", "runtime", "host path", "prompt protocol",
	} {
		if strings.Contains(text, forbidden) {
			t.Errorf("graph analyst instructions contain private concept %q", forbidden)
		}
	}
	for _, required := range []string{
		"coverage", "relative", "write_text_artifact", "symbolid", "find_symbol", "truncated",
	} {
		if !strings.Contains(text, required) {
			t.Errorf("graph analyst instructions omit %q", required)
		}
	}
}

func TestRepositoryCurrentProjectWorkflowsSelectGraphAnalysis(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	for _, selector := range []string{
		"openapi-from-workspace@5",
		"openapi-from-workspace-streamline@1",
		"likec4-from-workspace@5",
		"likec4-from-workspace-streamline@2",
	} {
		workflow, err := snapshot.Workflow(selector)
		if err != nil {
			t.Fatal(err)
		}
		for _, stageName := range []string{"dependency_discovery", "project_discovery"} {
			stage := workflow.Stages[stageName]
			binding, ok := stage.Agents["analyst"]
			if !ok {
				t.Fatalf("%s Stage %s omits analyst binding", selector, stageName)
			}
			if got := binding.Template.Ref.TemplateID + "@" + binding.Template.Ref.Version; got != "workspace_source_graph_analyst@1" {
				t.Fatalf("%s Stage %s template = %s", selector, stageName, got)
			}
		}
	}
}
