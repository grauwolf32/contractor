package config

import (
	"reflect"
	"strings"
	"testing"
)

func TestRepositoryLikeC4WorkflowTopology(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	workflow, err := snapshot.Workflow("likec4-from-source@1")
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
		"architecture":      {Required: true, MediaTypes: []string{"text/vnd.likec4"}},
		"validation_report": {Required: true, MediaTypes: []string{"text/markdown"}},
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("LikeC4 outputs = %+v, want %+v", got, want)
	}

	dependency := workflow.Stages["dependency_discovery"]
	assertSingleAgent(t, dependency, "analyst", "source_analyst", "analysis")
	assertContext(t, dependency, map[string]ContextArtifact{
		"source": {Namespace: "inputs", Name: "source", Required: true},
	})
	assertStageResult(t, dependency, "dependency_report", "text/markdown")
	assertNext(t, dependency.On.Succeeded, "project_discovery")
	assertBoundedRetry(t, dependency.On.Failed, 2)
	assertBoundedRetry(t, dependency.On.Interrupted, 2)

	project := workflow.Stages["project_discovery"]
	assertSingleAgent(t, project, "analyst", "source_analyst", "analysis")
	assertContext(t, project, map[string]ContextArtifact{
		"source":            {Namespace: "inputs", Name: "source", Required: true},
		"dependency_report": {Namespace: "analysis", Name: "dependencies", Required: true},
	})
	assertStageResult(t, project, "project_report", "text/markdown")
	assertNext(t, project.On.Succeeded, "likec4_build")
	assertBoundedRetry(t, project.On.Failed, 2)
	assertBoundedRetry(t, project.On.Interrupted, 2)

	build := workflow.Stages["likec4_build"]
	assertSingleAgent(t, build, "builder", "likec4_builder", "likec4")
	assertContext(t, build, map[string]ContextArtifact{
		"source":            {Namespace: "inputs", Name: "source", Required: true},
		"dependency_report": {Namespace: "analysis", Name: "dependencies", Required: true},
		"project_report":    {Namespace: "analysis", Name: "project", Required: true},
		"existing_likec4":   {Namespace: "inputs", Name: "existing_likec4", Required: false},
	})
	assertStageResult(t, build, "architecture", "text/vnd.likec4")
	assertNext(t, build.On.Succeeded, "likec4_validate")
	assertBoundedRetry(t, build.On.Failed, 6)
	assertBoundedRetry(t, build.On.Interrupted, 6)

	validate := workflow.Stages["likec4_validate"]
	assertSingleAgent(t, validate, "validator", "likec4_validator", "likec4")
	assertContext(t, validate, map[string]ContextArtifact{
		"source":                 {Namespace: "inputs", Name: "source", Required: true},
		"dependency_report":      {Namespace: "analysis", Name: "dependencies", Required: true},
		"project_report":         {Namespace: "analysis", Name: "project", Required: true},
		"architecture_candidate": {Namespace: "likec4", Name: "architecture", Required: true},
	})
	assertStageResult(t, validate, "architecture", "text/vnd.likec4")
	assertStageResult(t, validate, "validation_report", "text/markdown")
	if !reflect.DeepEqual(validate.WorkflowOutputs, map[string]string{
		"architecture": "architecture", "validation_report": "validation_report",
	}) {
		t.Fatalf("final Workflow output mappings = %+v", validate.WorkflowOutputs)
	}
	if validate.On.Succeeded.Kind != TransitionSucceed {
		t.Fatalf("validation success transition = %+v", validate.On.Succeeded)
	}
	assertBoundedRetry(t, validate.On.Failed, 2)
	assertBoundedRetry(t, validate.On.Interrupted, 2)

	openapi, err := snapshot.Workflow("openapi-from-source@1")
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
		if _, ok := stage.Context.Artifacts["source"]; !ok {
			t.Fatalf("Stage %q omits source context", name)
		}
		if name != "likec4_validate" && len(stage.WorkflowOutputs) != 0 {
			t.Fatalf("Stage %q unexpectedly maps Workflow outputs: %+v", name, stage.WorkflowOutputs)
		}
	}
}

func TestRepositoryLikeC4HighBudgetWorkflowVariants(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	workerPolicy, err := snapshot.ModelPolicy("project_worker@1")
	if err != nil {
		t.Fatal(err)
	}
	if workerPolicy.Model != "worker-model" || workerPolicy.MaxOutputTokens != 32768 ||
		workerPolicy.MaxModelCalls != 48 || workerPolicy.MaxToolCalls != 256 ||
		workerPolicy.MaxTotalTokens != 1000000 || workerPolicy.MaxWorkerCalls != 0 {
		t.Fatalf("project Worker policy = %+v", workerPolicy)
	}
	plannerPolicy, err := snapshot.ModelPolicy("project_planner@1")
	if err != nil {
		t.Fatal(err)
	}
	if plannerPolicy.Model != "planner-model" || plannerPolicy.MaxOutputTokens != 8192 ||
		plannerPolicy.MaxModelCalls != 48 || plannerPolicy.MaxWorkerCalls != 64 ||
		plannerPolicy.MaxTotalTokens != 500000 || plannerPolicy.MaxToolCalls != 0 {
		t.Fatalf("project Planner policy = %+v", plannerPolicy)
	}

	baseline, err := snapshot.Workflow("likec4-from-source@1")
	if err != nil {
		t.Fatal(err)
	}
	passthrough, err := snapshot.Workflow("likec4-from-source@2")
	if err != nil {
		t.Fatal(err)
	}
	streamline, err := snapshot.Workflow("likec4-from-source-streamline@1")
	if err != nil {
		t.Fatal(err)
	}
	if passthrough.EntryStage != baseline.EntryStage ||
		!reflect.DeepEqual(passthrough.Inputs, baseline.Inputs) ||
		!reflect.DeepEqual(passthrough.Outputs, baseline.Outputs) ||
		len(passthrough.Stages) != len(baseline.Stages) ||
		streamline.EntryStage != baseline.EntryStage ||
		!reflect.DeepEqual(streamline.Inputs, baseline.Inputs) ||
		!reflect.DeepEqual(streamline.Outputs, baseline.Outputs) ||
		len(streamline.Stages) != len(baseline.Stages) {
		t.Fatal("high-budget LikeC4 variants drifted from the four-Stage graph")
	}

	for name, baselineStage := range baseline.Stages {
		passthroughStage := passthrough.Stages[name]
		streamlineStage := streamline.Stages[name]
		if passthroughStage.Planner != (PlannerRef{PlannerID: "passthrough", Version: "1"}) {
			t.Fatalf("passthrough Stage %q planner = %+v", name, passthroughStage.Planner)
		}
		if streamlineStage.Planner != (PlannerRef{PlannerID: "streamline", Version: "1"}) {
			t.Fatalf("streamline Stage %q planner = %+v", name, streamlineStage.Planner)
		}
		if !reflect.DeepEqual(passthroughStage.Agents, baselineStage.Agents) ||
			!reflect.DeepEqual(passthroughStage.Context, baselineStage.Context) ||
			!reflect.DeepEqual(passthroughStage.Result, baselineStage.Result) ||
			!reflect.DeepEqual(passthroughStage.On, baselineStage.On) ||
			!reflect.DeepEqual(passthroughStage.WorkflowOutputs, baselineStage.WorkflowOutputs) {
			t.Fatalf("passthrough Stage %q changed its semantic contract", name)
		}
		if !reflect.DeepEqual(streamlineStage.Agents, baselineStage.Agents) ||
			!reflect.DeepEqual(streamlineStage.Context, baselineStage.Context) ||
			!reflect.DeepEqual(streamlineStage.Result, baselineStage.Result) ||
			!reflect.DeepEqual(streamlineStage.On, baselineStage.On) ||
			!reflect.DeepEqual(streamlineStage.WorkflowOutputs, baselineStage.WorkflowOutputs) {
			t.Fatalf("streamline Stage %q changed its semantic contract", name)
		}
		for logicalName, selection := range passthroughStage.ExecutionConfig.Agents {
			if selection.ModelPolicy.Ref != workerPolicy.Ref {
				t.Fatalf("passthrough Stage %q Agent %q policy = %+v", name, logicalName, selection.ModelPolicy.Ref)
			}
		}
		if passthroughStage.ExecutionConfig.Planner != nil {
			t.Fatalf("passthrough Stage %q unexpectedly has modeled Planner config", name)
		}
		if streamlineStage.ExecutionConfig.Planner == nil ||
			streamlineStage.ExecutionConfig.Planner.ModelPolicy.Ref != plannerPolicy.Ref {
			t.Fatalf("streamline Stage %q Planner config = %+v", name, streamlineStage.ExecutionConfig.Planner)
		}
		for logicalName, selection := range streamlineStage.ExecutionConfig.Agents {
			if selection.ModelPolicy.Ref != workerPolicy.Ref {
				t.Fatalf("streamline Stage %q Agent %q policy = %+v", name, logicalName, selection.ModelPolicy.Ref)
			}
		}
	}
}

func TestRepositoryLikeC4AgentToolAllowlists(t *testing.T) {
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
			ref: "likec4_builder@1",
			expected: map[string][]string{
				"source-analysis@1": {"list_source_files", "open_source_archive", "read_source", "search_source"},
				"text-artifacts@1":  {"read_text_artifact"},
				"likec4@1":          {"append_likec4", "load_likec4", "read_likec4", "replace_likec4", "validate_likec4", "write_likec4"},
			},
		},
		{
			ref: "likec4_validator@1",
			expected: map[string][]string{
				"source-analysis@1": {"open_source_archive", "read_source", "search_source"},
				"text-artifacts@1":  {"read_text_artifact", "write_text_artifact"},
				"likec4@1":          {"append_likec4", "load_likec4", "read_likec4", "replace_likec4", "validate_likec4", "write_likec4"},
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
			actual := selectedToolsets(template.Toolsets)
			if !reflect.DeepEqual(actual, test.expected) {
				t.Fatalf("selected Toolsets = %+v, want %+v", actual, test.expected)
			}
			if _, generic := actual["run-artifacts@1"]; generic {
				t.Fatal("LikeC4 template exposes generic Run artifact tools")
			}
		})
	}
}

func TestRepositoryLikeC4InstructionsAreSelfContained(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	checks := map[string][]string{
		"instructions/likec4-builder-worker.md": {
			"specification", "model", "views", "relative/path:line", "autoLayout LeftRight",
			"inputs/existing_likec4", "valid: true",
		},
		"instructions/likec4-validator-worker.md": {
			"repair-only", "zero-based", "exactly once more", "validation-report", "valid: true",
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
