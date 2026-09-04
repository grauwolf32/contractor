package config

import (
	"reflect"
	"strings"
	"testing"
)

func TestRepositoryLikeC4WorkflowTopology(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	workflow, err := snapshot.Workflow("likec4-from-workspace@4")
	if err != nil {
		t.Fatalf("resolve LikeC4 Workflow: %v", err)
	}
	if workflow.EntryStage != "dependency_discovery" || len(workflow.Stages) != 4 {
		t.Fatalf("unexpected LikeC4 topology: entry=%q stages=%d", workflow.EntryStage, len(workflow.Stages))
	}
	if source := workflow.Inputs["source"]; !source.Required || !equalStrings(source.MediaTypes, []string{"application/zip"}) {
		t.Fatalf("unexpected source input: %+v", source)
	}
	if seed := workflow.Inputs["existing_likec4"]; seed.Required || !equalStrings(seed.MediaTypes, []string{"text/vnd.likec4", "text/plain"}) {
		t.Fatalf("unexpected optional seed input: %+v", seed)
	}
	if got, want := workflow.Outputs, map[string]ArtifactSlot{
		"likec4":                   {Required: true, MediaTypes: []string{"text/vnd.likec4"}, Primary: true},
		"likec4_validation_report": {Required: true, MediaTypes: []string{"text/markdown"}},
		"workspace_state":          {Required: true, MediaTypes: []string{"application/vnd.contractor.workspace-overlay+json"}},
		"workspace_diff":           {Required: true, MediaTypes: []string{"text/x-diff"}},
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("LikeC4 outputs = %+v, want %+v", got, want)
	}

	dependency := workflow.Stages["dependency_discovery"]
	assertSingleAgent(t, dependency, "analyst", "workspace_source_graph_analyst", "analysis")
	assertContext(t, dependency, map[string]ContextArtifact{
		"source": {Namespace: "inputs", Name: "source", Required: true},
	})
	assertStageResult(t, dependency, "dependency_report", "text/markdown")
	assertNext(t, dependency.On.Succeeded, "project_discovery")
	assertBoundedRetry(t, dependency.On.Failed, 2)
	assertBoundedRetry(t, dependency.On.Interrupted, 2)

	project := workflow.Stages["project_discovery"]
	assertSingleAgent(t, project, "analyst", "workspace_source_graph_analyst", "analysis")
	assertContext(t, project, map[string]ContextArtifact{
		"source":                {Namespace: "inputs", Name: "source", Required: true},
		"dependency_report":     {Namespace: "analysis", Name: "dependencies", Required: true},
		"prior_workspace_state": {Namespace: "analysis", Name: "workspace_state", Required: true},
	})
	assertStageResult(t, project, "project_report", "text/markdown")
	assertNext(t, project.On.Succeeded, "likec4_build")
	assertBoundedRetry(t, project.On.Failed, 2)
	assertBoundedRetry(t, project.On.Interrupted, 2)

	build := workflow.Stages["likec4_build"]
	assertSingleAgent(t, build, "builder", "workspace_likec4_builder", "likec4")
	assertContext(t, build, map[string]ContextArtifact{
		"source":                {Namespace: "inputs", Name: "source", Required: true},
		"dependency_report":     {Namespace: "analysis", Name: "dependencies", Required: true},
		"project_report":        {Namespace: "analysis", Name: "project", Required: true},
		"existing_likec4":       {Namespace: "inputs", Name: "existing_likec4", Required: false},
		"prior_workspace_state": {Namespace: "analysis", Name: "workspace_state", Required: true},
	})
	assertStageResult(t, build, "architecture", "text/vnd.likec4")
	assertNext(t, build.On.Succeeded, "likec4_validate")
	assertBoundedRetry(t, build.On.Failed, 6)
	assertBoundedRetry(t, build.On.Interrupted, 6)

	validate := workflow.Stages["likec4_validate"]
	assertSingleAgent(t, validate, "validator", "workspace_likec4_validator", "likec4")
	assertContext(t, validate, map[string]ContextArtifact{
		"source":                 {Namespace: "inputs", Name: "source", Required: true},
		"dependency_report":      {Namespace: "analysis", Name: "dependencies", Required: true},
		"project_report":         {Namespace: "analysis", Name: "project", Required: true},
		"architecture_candidate": {Namespace: "likec4", Name: "architecture", Required: true},
		"prior_workspace_state":  {Namespace: "likec4", Name: "workspace_state", Required: true},
	})
	assertStageResult(t, validate, "architecture", "text/vnd.likec4")
	assertStageResult(t, validate, "validation_report", "text/markdown")
	if !reflect.DeepEqual(validate.WorkflowOutputs, map[string]string{
		"likec4": "architecture", "likec4_validation_report": "validation_report",
		"workspace_state": "workspace_state", "workspace_diff": "workspace_diff",
	}) {
		t.Fatalf("final Workflow output mappings = %+v", validate.WorkflowOutputs)
	}
	if validate.On.Succeeded.Kind != TransitionSucceed {
		t.Fatalf("validation success transition = %+v", validate.On.Succeeded)
	}
	assertBoundedRetry(t, validate.On.Failed, 2)
	assertBoundedRetry(t, validate.On.Interrupted, 2)

	openapi, err := snapshot.Workflow("openapi-from-workspace@4")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(dependency, openapi.Stages["dependency_discovery"]) {
		t.Fatal("LikeC4 dependency discovery drifted from the shared OpenAPI contract")
	}
	openapiProject := openapi.Stages["project_discovery"]
	if project.Instructions != openapiProject.Instructions ||
		!reflect.DeepEqual(project.Agents, openapiProject.Agents) ||
		!reflect.DeepEqual(project.Context, openapiProject.Context) ||
		!reflect.DeepEqual(project.Result, openapiProject.Result) {
		t.Fatal("LikeC4 project discovery drifted from the shared OpenAPI contract")
	}

	for name, stage := range workflow.Stages {
		if stage.Planner != (PlannerRef{PlannerID: "passthrough", Version: "1"}) {
			t.Fatalf("Stage %q planner = %+v", name, stage.Planner)
		}
		if stage.Context.Workspace == nil {
			t.Fatalf("Stage %q omits the project workspace", name)
		}
		if name != "likec4_validate" && len(stage.WorkflowOutputs) != 0 {
			t.Fatalf("Stage %q unexpectedly maps Workflow outputs: %+v", name, stage.WorkflowOutputs)
		}
	}
}

func TestRepositoryLikeC4WorkspaceAgentToolAllowlists(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	policy, err := snapshot.ModelPolicy("domain_worker@1")
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		ref      string
		expected map[string][]string
	}{
		{
			ref: "workspace_likec4_builder@1",
			expected: map[string][]string{
				"filesystem@1":        {"glob", "grep", "ls", "read_file"},
				"workspace-changes@1": {"changed_paths", "diff", "rollback_changes"},
				"text-artifacts@1":    {"read_text_artifact"},
				"likec4@1":            {"append_likec4", "load_likec4", "read_likec4", "replace_likec4", "validate_likec4", "write_likec4"},
			},
		},
		{
			ref: "workspace_likec4_validator@1",
			expected: map[string][]string{
				"filesystem@1":        {"glob", "grep", "ls", "read_file"},
				"workspace-changes@1": {"changed_paths", "diff", "rollback_changes"},
				"text-artifacts@1":    {"read_text_artifact", "write_text_artifact"},
				"likec4@1":            {"append_likec4", "load_likec4", "read_likec4", "replace_likec4", "validate_likec4", "write_likec4"},
			},
		},
	}
	for _, test := range cases {
		t.Run(test.ref, func(t *testing.T) {
			template, templateErr := snapshot.AgentTemplate(test.ref)
			if templateErr != nil {
				t.Fatal(templateErr)
			}
			if template.ModelPolicy.Ref != policy.Ref {
				t.Fatalf("template policy = %+v, want %+v", template.ModelPolicy.Ref, policy.Ref)
			}
			if actual := selectedToolsets(template.Toolsets); !reflect.DeepEqual(actual, test.expected) {
				t.Fatalf("selected Toolsets = %+v, want %+v", actual, test.expected)
			}
		})
	}
}

func TestRepositoryLikeC4WorkspaceInstructionsAreSelfContained(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	checks := map[string][]string{
		"instructions/workspace-likec4-builder-worker.md": {
			"specification", "model", "views", "relative/path:line", "existing_likec4",
		},
		"instructions/workspace-likec4-validator-worker.md": {
			"repair-only", "exactly once more", "validation-report",
		},
		"instructions/likec4-build-planner.md": {
			"specification` -> `model` -> `views", "likec4/architecture", "text/vnd.likec4",
		},
	}
	for ref, fragments := range checks {
		instructions, err := snapshot.Instructions(ref)
		if err != nil {
			t.Fatalf("resolve %s: %v", ref, err)
		}
		assertDigest(t, instructions.Digest)
		for _, fragment := range fragments {
			if !strings.Contains(instructions.Text, fragment) {
				t.Errorf("%s does not contain %q", ref, fragment)
			}
		}
	}
}
