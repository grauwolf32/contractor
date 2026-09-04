package config

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRepositoryWorkspaceWorkflowsUseCumulativeOverlayState(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	cases := []struct {
		ref            string
		domainStages   map[string]string
		orderedStages  []string
		finalStage     string
		domainOutput   string
		domainMedia    string
		validationSlot string
	}{
		{
			ref: "openapi-from-workspace@3",
			domainStages: map[string]string{
				"dependency_discovery": "workspace_source_graph_analyst",
				"project_discovery":    "workspace_source_graph_analyst",
				"openapi_build":        "workspace_openapi_builder",
				"openapi_validate":     "workspace_openapi_validator",
			},
			orderedStages: []string{"dependency_discovery", "project_discovery", "openapi_build", "openapi_validate"},
			finalStage:    "openapi_validate", domainOutput: "openapi", domainMedia: "application/yaml",
			validationSlot: "validation_report",
		},
		{
			ref: "likec4-from-workspace@3",
			domainStages: map[string]string{
				"dependency_discovery": "workspace_source_graph_analyst",
				"project_discovery":    "workspace_source_graph_analyst",
				"likec4_build":         "workspace_likec4_builder",
				"likec4_validate":      "workspace_likec4_validator",
			},
			orderedStages: []string{"dependency_discovery", "project_discovery", "likec4_build", "likec4_validate"},
			finalStage:    "likec4_validate", domainOutput: "architecture", domainMedia: "text/vnd.likec4",
			validationSlot: "validation_report",
		},
	}
	for _, test := range cases {
		t.Run(test.ref, func(t *testing.T) {
			workflow, err := snapshot.Workflow(test.ref)
			if err != nil {
				t.Fatal(err)
			}
			if workflow.EntryStage != test.orderedStages[0] || len(workflow.Stages) != len(test.orderedStages) {
				t.Fatalf("unexpected workspace graph: entry=%q stages=%d", workflow.EntryStage, len(workflow.Stages))
			}
			for index, name := range test.orderedStages {
				stage := workflow.Stages[name]
				templateID := test.domainStages[name]
				if len(stage.Agents) != 1 {
					t.Fatalf("Stage %s agents = %+v", name, stage.Agents)
				}
				for _, binding := range stage.Agents {
					if binding.Template.Ref.TemplateID != templateID {
						t.Fatalf("Stage %s template = %s, want %s", name, binding.Template.Ref.TemplateID, templateID)
					}
				}
				workspace := stage.Context.Workspace
				if workspace == nil || workspace.Mode != contracts.WorkspaceModeOverlay ||
					!reflect.DeepEqual(workspace.Sources, []WorkspaceSource{{Artifact: "source", Target: ""}}) ||
					workspace.Export == nil || workspace.Export.State != "workspace_state" ||
					workspace.Export.Diff != "workspace_diff" {
					t.Fatalf("Stage %s workspace = %+v", name, workspace)
				}
				if index == 0 {
					if workspace.State != nil {
						t.Fatalf("entry Stage must start from the canonical empty overlay: %+v", workspace.State)
					}
				} else if workspace.State == nil || workspace.State.Artifact != "prior_workspace_state" {
					t.Fatalf("Stage %s cumulative state input = %+v", name, workspace.State)
				}
				assertStageResult(t, stage, "workspace_state", "application/vnd.contractor.workspace-overlay+json")
				assertStageResult(t, stage, "workspace_diff", "text/x-diff")
			}
			final := workflow.Stages[test.finalStage]
			assertStageResult(t, final, test.domainOutput, test.domainMedia)
			assertStageResult(t, final, test.validationSlot, "text/markdown")
			if !reflect.DeepEqual(final.WorkflowOutputs, map[string]string{
				test.domainOutput:   test.domainOutput,
				test.validationSlot: test.validationSlot,
				"workspace_state":   "workspace_state",
				"workspace_diff":    "workspace_diff",
			}) {
				t.Fatalf("final outputs = %+v", final.WorkflowOutputs)
			}
		})
	}
}

func TestRepositoryE2EWorkspaceRoundtripFixture(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, filepath.Join(repositoryConfigRoot, "e2e"), MVPDescriptors())
	workflow, err := snapshot.Workflow("workspace-roundtrip@1")
	if err != nil {
		t.Fatal(err)
	}
	stage := workflow.Stages[workflow.EntryStage]
	if stage.Context.Workspace == nil ||
		stage.Context.Workspace.Mode != contracts.WorkspaceModeOverlay ||
		stage.Context.Workspace.Export == nil ||
		stage.Context.Workspace.Export.State != "workspace_state" ||
		stage.Context.Workspace.Export.Diff != "workspace_diff" {
		t.Fatalf("workspace E2E Stage contract = %+v", stage.Context.Workspace)
	}
	template, err := snapshot.AgentTemplate("workspace_editor@1")
	if err != nil {
		t.Fatal(err)
	}
	if got := selectedToolsets(template.Toolsets); !reflect.DeepEqual(got, map[string][]string{
		"filesystem@1":        {"read_file"},
		"edit-files@1":        {"edit"},
		"workspace-changes@1": {"changed_paths", "diff"},
	}) {
		t.Fatalf("workspace E2E tools = %+v", got)
	}
}

func TestRepositoryWorkspaceTemplatesUseOnlyNarrowFilesystemContracts(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	for _, ref := range []string{
		"workspace_source_graph_analyst@1",
		"workspace_openapi_builder@1",
		"workspace_openapi_validator@1",
		"workspace_likec4_builder@1",
		"workspace_likec4_validator@1",
	} {
		template, err := snapshot.AgentTemplate(ref)
		if err != nil {
			t.Fatal(err)
		}
		toolsets := selectedToolsets(template.Toolsets)
		if _, exists := toolsets["source-analysis@1"]; exists {
			t.Fatalf("%s retains archive-local source-analysis@1", ref)
		}
		if !reflect.DeepEqual(toolsets["filesystem@1"], []string{"glob", "grep", "ls", "read_file"}) {
			t.Fatalf("%s filesystem tools = %+v", ref, toolsets)
		}
		if ref != "workspace_source_graph_analyst@1" &&
			!reflect.DeepEqual(toolsets["workspace-changes@1"], []string{"changed_paths", "diff", "rollback_changes"}) {
			t.Fatalf("%s workspace-change tools = %+v", ref, toolsets)
		}
		instructions := strings.Join(strings.Fields(template.Instructions.Text), " ")
		if ref != "workspace_source_graph_analyst@1" &&
			(!strings.Contains(strings.ToLower(instructions), "workspace export is automatic") ||
				!strings.Contains(instructions, "host path")) {
			t.Fatalf("%s instructions do not explain workspace boundaries", ref)
		}
	}
}
