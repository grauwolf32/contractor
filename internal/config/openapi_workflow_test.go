package config

import (
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestRepositoryOpenAPIWorkflowTopology(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	workflow, err := snapshot.Workflow("openapi-from-source@1")
	if err != nil {
		t.Fatalf("resolve OpenAPI Workflow: %v", err)
	}
	if workflow.EntryStage != "dependency_discovery" || len(workflow.Stages) != 4 {
		t.Fatalf("unexpected OpenAPI topology: entry=%q stages=%d", workflow.EntryStage, len(workflow.Stages))
	}
	if source := workflow.Inputs["source"]; !source.Required || !equalStrings(source.MediaTypes, []string{"application/zip"}) {
		t.Fatalf("unexpected source input: %+v", source)
	}
	if seed := workflow.Inputs["existing_openapi"]; seed.Required || !equalStrings(seed.MediaTypes, []string{"application/yaml", "application/json"}) {
		t.Fatalf("unexpected optional seed input: %+v", seed)
	}
	if got, want := workflow.Outputs, map[string]ArtifactSlot{
		"openapi":           {Required: true, MediaTypes: []string{"application/yaml"}},
		"validation_report": {Required: true, MediaTypes: []string{"text/markdown"}},
	}; !reflect.DeepEqual(got, want) {
		t.Fatalf("OpenAPI outputs = %+v, want %+v", got, want)
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
	assertNext(t, project.On.Succeeded, "openapi_build")
	assertBoundedRetry(t, project.On.Failed, 2)
	assertBoundedRetry(t, project.On.Interrupted, 2)

	build := workflow.Stages["openapi_build"]
	assertSingleAgent(t, build, "builder", "openapi_builder", "openapi")
	assertContext(t, build, map[string]ContextArtifact{
		"source":            {Namespace: "inputs", Name: "source", Required: true},
		"dependency_report": {Namespace: "analysis", Name: "dependencies", Required: true},
		"project_report":    {Namespace: "analysis", Name: "project", Required: true},
		"existing_openapi":  {Namespace: "inputs", Name: "existing_openapi", Required: false},
	})
	assertStageResult(t, build, "openapi", "application/yaml")
	assertNext(t, build.On.Succeeded, "openapi_validate")
	assertBoundedRetry(t, build.On.Failed, 4)
	assertBoundedRetry(t, build.On.Interrupted, 4)

	validate := workflow.Stages["openapi_validate"]
	assertSingleAgent(t, validate, "validator", "openapi_validator", "openapi")
	assertContext(t, validate, map[string]ContextArtifact{
		"source":            {Namespace: "inputs", Name: "source", Required: true},
		"dependency_report": {Namespace: "analysis", Name: "dependencies", Required: true},
		"project_report":    {Namespace: "analysis", Name: "project", Required: true},
		"openapi_candidate": {Namespace: "openapi", Name: "openapi", Required: true},
	})
	assertStageResult(t, validate, "openapi", "application/yaml")
	assertStageResult(t, validate, "validation_report", "text/markdown")
	if !reflect.DeepEqual(validate.WorkflowOutputs, map[string]string{
		"openapi": "openapi", "validation_report": "validation_report",
	}) {
		t.Fatalf("final Workflow output mappings = %+v", validate.WorkflowOutputs)
	}
	if validate.On.Succeeded.Kind != TransitionSucceed {
		t.Fatalf("validation success transition = %+v", validate.On.Succeeded)
	}
	assertEscalation(t, validate.On.Failed, 1, "strong-oas-review")
	assertBoundedRetry(t, validate.On.Interrupted, 2)

	for name, stage := range workflow.Stages {
		if stage.Planner != (PlannerRef{PlannerID: "passthrough", Version: "1"}) {
			t.Fatalf("Stage %q planner = %+v", name, stage.Planner)
		}
		if _, ok := stage.Context.Artifacts["source"]; !ok {
			t.Fatalf("Stage %q omits source context", name)
		}
		if name != "openapi_validate" && len(stage.WorkflowOutputs) != 0 {
			t.Fatalf("Stage %q unexpectedly maps Workflow outputs: %+v", name, stage.WorkflowOutputs)
		}
	}
}

func TestRepositoryOpenAPIAgentToolAllowlists(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	policy, err := snapshot.ModelPolicy("domain_worker@1")
	if err != nil {
		t.Fatalf("resolve domain Worker policy: %v", err)
	}
	if policy.Model != "worker-model" || policy.MaxOutputTokens != 16384 || policy.Temperature == nil || *policy.Temperature != 0.1 {
		t.Fatalf("domain Worker policy = %+v", policy)
	}

	cases := []struct {
		ref      string
		expected map[string][]string
	}{
		{
			ref: "source_analyst@1",
			expected: map[string][]string{
				"source-analysis@1": {"list_source_files", "open_source_archive", "read_source", "search_source"},
				"text-artifacts@1":  {"read_text_artifact", "write_text_artifact"},
			},
		},
		{
			ref: "openapi_builder@1",
			expected: map[string][]string{
				"source-analysis@1": {"list_source_files", "open_source_archive", "read_source", "search_source"},
				"text-artifacts@1":  {"read_text_artifact"},
				"openapi@1": {
					"get_openapi_component", "get_openapi_info", "get_openapi_path",
					"initialize_openapi", "list_openapi_components", "list_openapi_paths",
					"list_openapi_servers", "list_openapi_tags", "load_openapi", "set_openapi_info",
					"set_openapi_servers", "set_openapi_tags", "upsert_openapi_component", "upsert_openapi_path",
					"validate_openapi",
				},
			},
		},
		{
			ref: "openapi_validator@1",
			expected: map[string][]string{
				"source-analysis@1": {"open_source_archive", "read_source", "search_source"},
				"text-artifacts@1":  {"read_text_artifact", "write_text_artifact"},
				"openapi@1": {
					"get_openapi_component", "get_openapi_info", "get_openapi_path",
					"list_openapi_components", "list_openapi_paths", "list_openapi_servers", "list_openapi_tags",
					"load_openapi", "remove_openapi_component", "remove_openapi_path",
					"set_openapi_info", "set_openapi_servers", "set_openapi_tags", "upsert_openapi_component",
					"upsert_openapi_path", "validate_openapi",
				},
			},
		},
	}
	for _, test := range cases {
		t.Run(test.ref, func(t *testing.T) {
			template, templateErr := snapshot.AgentTemplate(test.ref)
			if templateErr != nil {
				t.Fatalf("resolve template: %v", templateErr)
			}
			if template.ModelPolicy.Ref != policy.Ref {
				t.Fatalf("template policy = %+v, want %+v", template.ModelPolicy.Ref, policy.Ref)
			}
			actual := selectedToolsets(template.Toolsets)
			if !reflect.DeepEqual(actual, test.expected) {
				t.Fatalf("selected Toolsets = %+v, want %+v", actual, test.expected)
			}
			if _, generic := actual["run-artifacts@1"]; generic {
				t.Fatal("domain template exposes generic Run artifact tools")
			}
		})
	}
}

func TestRepositoryOpenAPIInstructionsPinArtifactAndValidationRules(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	checks := map[string][]string{
		"instructions/dependency-discovery-planner.md": {"relative/path:line", "analysis/dependencies", "dependency_report"},
		"instructions/project-discovery-planner.md":    {"relative/path:line", "analysis/project", "project_report"},
		"instructions/openapi-builder-worker.md":       {"inputs/existing_openapi", "generic artifact writer", "evidence_files"},
		"instructions/openapi-validator-worker.md":     {"exactly once more", "validation-report", "valid: true"},
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

func selectedToolsets(selections []contracts.ToolsetSelection) map[string][]string {
	result := make(map[string][]string, len(selections))
	for _, selection := range selections {
		ref := selection.Ref.ToolsetID + "@" + selection.Ref.Version
		result[ref] = selection.Tools
	}
	return result
}

func assertSingleAgent(t *testing.T, stage ResolvedStage, logicalName, templateID, namespace string) {
	t.Helper()
	if len(stage.Agents) != 1 {
		t.Fatalf("agents = %+v", stage.Agents)
	}
	agent, ok := stage.Agents[logicalName]
	if !ok || agent.Template.Ref.TemplateID != templateID || agent.Template.Ref.Version != "1" || agent.Namespace != namespace {
		t.Fatalf("Agent %q = %+v", logicalName, agent)
	}
}

func assertContext(t *testing.T, stage ResolvedStage, want map[string]ContextArtifact) {
	t.Helper()
	if !reflect.DeepEqual(stage.Context.Artifacts, want) {
		t.Fatalf("Stage context = %+v, want %+v", stage.Context.Artifacts, want)
	}
}

func assertStageResult(t *testing.T, stage ResolvedStage, name, mediaType string) {
	t.Helper()
	slot, ok := stage.Result.Artifacts[name]
	if !ok || !slot.Required || !equalStrings(slot.MediaTypes, []string{mediaType}) {
		t.Fatalf("result %q = %+v", name, slot)
	}
}

func assertNext(t *testing.T, action TransitionAction, want string) {
	t.Helper()
	if action.Kind != TransitionNext || action.NextStage != want {
		t.Fatalf("next transition = %+v, want %q", action, want)
	}
}

func assertBoundedRetry(t *testing.T, action TransitionAction, maxAttempts int) {
	t.Helper()
	if action.Kind != TransitionRetry || action.Retry == nil || action.Retry.MaxAttempts != maxAttempts || action.Retry.Then.Kind != TransitionFail {
		t.Fatalf("retry transition = %+v, want maxAttempts=%d then fail", action, maxAttempts)
	}
}

func assertEscalation(t *testing.T, action TransitionAction, maxAttempts int, configID string) {
	t.Helper()
	if action.Kind != TransitionEscalate || action.Escalate == nil ||
		action.Escalate.MaxAttempts != maxAttempts || action.Escalate.ExecutionConfig.Ref == nil ||
		action.Escalate.ExecutionConfig.Ref.ConfigID != configID ||
		action.Escalate.Then.Kind != TransitionFail {
		t.Fatalf("unexpected escalation Transition: %+v", action)
	}
}
