package config

import (
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRepositoryCodeAnalysisTemplatesHaveExactVersionedToolSurfaces(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	legacy, err := snapshot.AgentTemplate("workspace_source_analyst@1")
	if err != nil {
		t.Fatal(err)
	}
	if legacy.Ref.Digest != "sha256:c3fba16b47e501c33b5eac2f81b24256cea0eb296977b61e9247e350213ef4a2" {
		t.Fatalf("legacy workspace analyst digest = %q", legacy.Ref.Digest)
	}

	shallow, err := snapshot.AgentTemplate("workspace_source_analyst@2")
	if err != nil {
		t.Fatal(err)
	}
	graph, err := snapshot.AgentTemplate("workspace_source_graph_analyst@1")
	if err != nil {
		t.Fatal(err)
	}
	for _, expected := range []struct {
		name               string
		templateDigest     string
		instructionsRef    string
		instructionsDigest string
		actual             contracts.ResolvedAgentTemplate
	}{
		{
			name: "shallow", templateDigest: "sha256:c2f485636261a1a8bf70fa527e41fc3f873efe4851531b04d74c9f0197403269",
			instructionsRef:    "instructions/workspace-source-shallow-analyst-worker.md",
			instructionsDigest: "sha256:c0966cba9c5498730696db35cf653f8e33a30f458971845c059f0297527bf831",
			actual:             shallow,
		},
		{
			name: "graph", templateDigest: "sha256:a5cf2ca077ae2b223f5e4f1dd1f123b3b66c0d28532c41ded630b30d75d19070",
			instructionsRef:    "instructions/workspace-source-graph-analyst-worker.md",
			instructionsDigest: "sha256:8264d3672a0ee56614cb57680fdd62de7942b86083297db22beeff1ddb0b05b8",
			actual:             graph,
		},
	} {
		if expected.actual.Ref.Digest != expected.templateDigest ||
			expected.actual.Instructions.Ref != expected.instructionsRef ||
			expected.actual.Instructions.Digest != expected.instructionsDigest {
			t.Errorf("%s immutable digest snapshot = template %s instructions %+v", expected.name, expected.actual.Ref.Digest, expected.actual.Instructions)
		}
	}
	wantCommon := map[string][]string{
		"filesystem@1":     {"glob", "grep", "ls", "read_file"},
		"text-artifacts@1": {"read_text_artifact", "write_text_artifact"},
	}
	for ref, template := range map[string]struct {
		got      map[string][]string
		analysis []string
	}{
		"workspace_source_analyst@2": {
			got: selectedToolsets(shallow.Toolsets),
			analysis: []string{
				"list_symbols", "search_def",
			},
		},
		"workspace_source_graph_analyst@1": {
			got: selectedToolsets(graph.Toolsets),
			analysis: []string{
				"attack_surface", "complexity_hotspots", "entrypoint_paths_to",
				"find_callees", "find_callers", "find_symbol", "functions_that_raise",
				"graph_summary", "list_symbols", "paths_between", "search_def",
			},
		},
	} {
		want := map[string][]string{
			"filesystem@1":     wantCommon["filesystem@1"],
			"text-artifacts@1": wantCommon["text-artifacts@1"],
			"code-analysis@1":  template.analysis,
		}
		if !reflect.DeepEqual(template.got, want) {
			t.Errorf("%s tools = %+v, want %+v", ref, template.got, want)
		}
	}
	if shallow.Ref.Digest == legacy.Ref.Digest || graph.Ref.Digest == legacy.Ref.Digest ||
		shallow.Ref.Digest == graph.Ref.Digest {
		t.Fatalf("versioned template digests are not isolated: legacy=%s shallow=%s graph=%s",
			legacy.Ref.Digest, shallow.Ref.Digest, graph.Ref.Digest)
	}
	assertDigest(t, shallow.Ref.Digest)
	assertDigest(t, graph.Ref.Digest)
}

func TestRepositoryCodeAnalysisWorkerPromptBoundary(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	for _, selector := range []string{
		"workspace_source_analyst@2",
		"workspace_source_graph_analyst@1",
	} {
		template, err := snapshot.AgentTemplate(selector)
		if err != nil {
			t.Fatal(err)
		}
		text := strings.ToLower(template.Instructions.Text)
		for _, forbidden := range []string{
			"agenttemplate", "allocationspec", "runtime", "host path", "prompt protocol",
		} {
			if strings.Contains(text, forbidden) {
				t.Errorf("%s instructions contain private concept %q", selector, forbidden)
			}
		}
		for _, required := range []string{"coverage", "relative", "write_text_artifact"} {
			if !strings.Contains(text, required) {
				t.Errorf("%s instructions omit %q", selector, required)
			}
		}
	}
	graph, err := snapshot.AgentTemplate("workspace_source_graph_analyst@1")
	if err != nil {
		t.Fatal(err)
	}
	for _, required := range []string{"symbolId", "find_symbol", "truncated"} {
		if !strings.Contains(graph.Instructions.Text, required) {
			t.Errorf("graph instructions omit %q", required)
		}
	}
}

func TestRepositoryCodeAnalysisWorkflowVersionsPreserveTheWorkspaceGraph(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	cases := []struct {
		name           string
		discovery      []string
		terminalStages []string
	}{
		{name: "openapi-from-workspace", discovery: []string{"dependency_discovery", "project_discovery"}, terminalStages: []string{"openapi_build", "openapi_validate"}},
		{name: "likec4-from-workspace", discovery: []string{"dependency_discovery", "project_discovery"}, terminalStages: []string{"likec4_build", "likec4_validate"}},
	}
	for _, test := range cases {
		base, err := snapshot.Workflow(test.name + "@1")
		if err != nil {
			t.Fatal(err)
		}
		for _, variant := range []struct {
			version  string
			template string
		}{
			{version: "2", template: "workspace_source_analyst@2"},
			{version: "3", template: "workspace_source_graph_analyst@1"},
		} {
			resolved, err := snapshot.Workflow(test.name + "@" + variant.version)
			if err != nil {
				t.Fatal(err)
			}
			if resolved.Ref.Version != variant.version || resolved.EntryStage != base.EntryStage ||
				!reflect.DeepEqual(resolved.Parameters, base.Parameters) ||
				!reflect.DeepEqual(resolved.Inputs, base.Inputs) ||
				!reflect.DeepEqual(resolved.Outputs, base.Outputs) || len(resolved.Stages) != 4 {
				t.Fatalf("%s@%s top-level contract drifted", test.name, variant.version)
			}
			for _, stageName := range test.discovery {
				actual := cloneStage(resolved.Stages[stageName])
				baseline := base.Stages[stageName]
				binding := actual.Agents["analyst"]
				gotTemplate := binding.Template.Ref.TemplateID + "@" + binding.Template.Ref.Version
				if gotTemplate != variant.template {
					t.Fatalf("%s@%s Stage %s template = %s", test.name, variant.version, stageName, gotTemplate)
				}
				binding.Template = baseline.Agents["analyst"].Template
				actual.Agents["analyst"] = binding
				if !reflect.DeepEqual(actual, baseline) {
					t.Fatalf("%s@%s Stage %s drifted outside analyst version", test.name, variant.version, stageName)
				}
			}
			for _, stageName := range test.terminalStages {
				if !reflect.DeepEqual(resolved.Stages[stageName], base.Stages[stageName]) {
					t.Fatalf("%s@%s Stage %s changed builder/validator contract", test.name, variant.version, stageName)
				}
			}
		}
	}
}

func TestRepositoryCodeAnalysisShallowFitsMemoryButGraphRequiresMoreOperations(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	shallow, err := snapshot.AgentTemplate("workspace_source_analyst@2")
	if err != nil {
		t.Fatal(err)
	}
	graph, err := snapshot.AgentTemplate("workspace_source_graph_analyst@1")
	if err != nil {
		t.Fatal(err)
	}
	memoryTools := map[string]map[string]bool{
		"filesystem@1": {
			"glob": true, "grep": true, "ls": true, "read_file": true,
		},
		"text-artifacts@1": {
			"read_text_artifact": true, "write_text_artifact": true,
		},
		"code-analysis@1": {"list_symbols": true, "search_def": true},
	}
	if !templateToolsFit(shallow, memoryTools) {
		t.Fatal("portable shallow template does not fit the memory capability fixture")
	}
	if templateToolsFit(graph, memoryTools) {
		t.Fatal("graph template silently fits a shallow-only memory capability fixture")
	}
}

func templateToolsFit(
	template contracts.ResolvedAgentTemplate,
	available map[string]map[string]bool,
) bool {
	for _, selection := range template.Toolsets {
		ref := selection.Ref.ToolsetID + "@" + selection.Ref.Version
		operations, exists := available[ref]
		if !exists {
			return false
		}
		for _, operation := range selection.Tools {
			if !operations[operation] {
				return false
			}
		}
	}
	return true
}
