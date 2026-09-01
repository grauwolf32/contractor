package config

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRepositoryWorkspaceWorkflowVariantsPreserveLegacyIDs(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	for _, legacyRef := range []string{"openapi-from-source@1", "likec4-from-source@1"} {
		legacy, err := snapshot.Workflow(legacyRef)
		if err != nil {
			t.Fatal(err)
		}
		for name, stage := range legacy.Stages {
			if stage.Context.Workspace != nil {
				t.Fatalf("legacy %s Stage %s unexpectedly gained a workspace", legacyRef, name)
			}
		}
	}

	cases := []struct {
		ref            string
		legacyRef      string
		domainStages   map[string]string
		finalStage     string
		domainOutput   string
		domainMedia    string
		validationSlot string
	}{
		{
			ref: "openapi-from-workspace@1", legacyRef: "openapi-from-source@1",
			domainStages: map[string]string{
				"dependency_discovery": "workspace_source_analyst",
				"project_discovery":    "workspace_source_analyst",
				"openapi_build":        "workspace_openapi_builder",
				"openapi_validate":     "workspace_openapi_validator",
			},
			finalStage: "openapi_validate", domainOutput: "openapi", domainMedia: "application/yaml",
			validationSlot: "validation_report",
		},
		{
			ref: "likec4-from-workspace@1", legacyRef: "likec4-from-source@1",
			domainStages: map[string]string{
				"dependency_discovery": "workspace_source_analyst",
				"project_discovery":    "workspace_source_analyst",
				"likec4_build":         "workspace_likec4_builder",
				"likec4_validate":      "workspace_likec4_validator",
			},
			finalStage: "likec4_validate", domainOutput: "architecture", domainMedia: "text/vnd.likec4",
			validationSlot: "validation_report",
		},
	}
	for _, test := range cases {
		t.Run(test.ref, func(t *testing.T) {
			workflow, err := snapshot.Workflow(test.ref)
			if err != nil {
				t.Fatal(err)
			}
			legacy, err := snapshot.Workflow(test.legacyRef)
			if err != nil {
				t.Fatal(err)
			}
			if workflow.EntryStage != legacy.EntryStage || !reflect.DeepEqual(workflow.Inputs, legacy.Inputs) ||
				len(workflow.Stages) != len(legacy.Stages) {
				t.Fatalf("workspace Workflow drifted from legacy graph: %+v", workflow)
			}
			for name, templateID := range test.domainStages {
				stage := workflow.Stages[name]
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
				if name == workflow.EntryStage {
					if workspace.State != nil {
						t.Fatalf("entry Stage imports state: %+v", workspace.State)
					}
				} else if workspace.State == nil || workspace.State.Artifact != "prior_workspace_state" {
					t.Fatalf("Stage %s state input = %+v", name, workspace.State)
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
		"workspace_source_analyst@1",
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
		if !reflect.DeepEqual(toolsets["filesystem@1"], []string{"glob", "grep", "ls", "read_file"}) ||
			!reflect.DeepEqual(toolsets["workspace-changes@1"], []string{"changed_paths", "diff", "rollback_changes"}) {
			t.Fatalf("%s workspace tools = %+v", ref, toolsets)
		}
		instructions := strings.Join(strings.Fields(template.Instructions.Text), " ")
		if !strings.Contains(instructions, "workspace_state") ||
			!strings.Contains(instructions, "workspace_diff") ||
			!strings.Contains(instructions, "host path") {
			t.Fatalf("%s instructions do not explain workspace boundaries", ref)
		}
	}
}
