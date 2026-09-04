package config

import (
	"reflect"
	"testing"
)

func TestRepositoryPrecomputedAnalysisWorkflows(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	cases := []struct {
		workflowRef       string
		buildStage        string
		validateStage     string
		builderTemplate   string
		validatorTemplate string
		seedInput         string
		buildResult       string
		buildMediaType    string
		outputMediaTypes  map[string]ArtifactSlot
	}{
		{
			workflowRef: "openapi-from-analysis@1",
			buildStage:  "openapi_build", validateStage: "openapi_validate",
			builderTemplate: "openapi_builder@1", validatorTemplate: "openapi_validator@1",
			seedInput: "existing_openapi", buildResult: "openapi", buildMediaType: "application/yaml",
			outputMediaTypes: map[string]ArtifactSlot{
				"openapi":           {Required: true, MediaTypes: []string{"application/yaml"}},
				"validation_report": {Required: true, MediaTypes: []string{"text/markdown"}},
			},
		},
		{
			workflowRef: "likec4-from-analysis@2",
			buildStage:  "likec4_build", validateStage: "likec4_validate",
			builderTemplate: "likec4_builder@2", validatorTemplate: "likec4_validator@2",
			seedInput: "existing_likec4", buildResult: "architecture", buildMediaType: "text/vnd.likec4",
			outputMediaTypes: map[string]ArtifactSlot{
				"architecture":      {Required: true, MediaTypes: []string{"text/vnd.likec4"}},
				"validation_report": {Required: true, MediaTypes: []string{"text/markdown"}},
			},
		},
	}

	for _, test := range cases {
		t.Run(test.workflowRef, func(t *testing.T) {
			workflow, err := snapshot.Workflow(test.workflowRef)
			if err != nil {
				t.Fatal(err)
			}
			if workflow.EntryStage != test.buildStage || len(workflow.Stages) != 2 {
				t.Fatalf("topology = entry %q, stages %d", workflow.EntryStage, len(workflow.Stages))
			}
			for name, expected := range map[string]ArtifactSlot{
				"source":            {Required: true, MediaTypes: []string{"application/zip"}},
				"dependency_report": {Required: true, MediaTypes: []string{"text/markdown"}},
				"project_report":    {Required: true, MediaTypes: []string{"text/markdown"}},
			} {
				if !reflect.DeepEqual(workflow.Inputs[name], expected) {
					t.Fatalf("input %q = %+v, want %+v", name, workflow.Inputs[name], expected)
				}
			}
			if len(workflow.Inputs) != 4 || workflow.Inputs[test.seedInput].Required {
				t.Fatalf("inputs = %+v", workflow.Inputs)
			}
			if !reflect.DeepEqual(workflow.Outputs, test.outputMediaTypes) {
				t.Fatalf("outputs = %+v, want %+v", workflow.Outputs, test.outputMediaTypes)
			}

			build := workflow.Stages[test.buildStage]
			validate := workflow.Stages[test.validateStage]
			assertContext(t, build, map[string]ContextArtifact{
				"source":            {Namespace: "inputs", Name: "source", Required: true},
				"dependency_report": {Namespace: "inputs", Name: "dependency_report", Required: true},
				"project_report":    {Namespace: "inputs", Name: "project_report", Required: true},
				test.seedInput:      {Namespace: "inputs", Name: test.seedInput, Required: false},
			})
			candidateName, candidateNamespace, candidateArtifact := "openapi_candidate", "openapi", "openapi"
			if test.buildStage == "likec4_build" {
				candidateName, candidateNamespace, candidateArtifact = "architecture_candidate", "likec4", "architecture"
			}
			assertContext(t, validate, map[string]ContextArtifact{
				"source":            {Namespace: "inputs", Name: "source", Required: true},
				"dependency_report": {Namespace: "inputs", Name: "dependency_report", Required: true},
				"project_report":    {Namespace: "inputs", Name: "project_report", Required: true},
				candidateName:       {Namespace: candidateNamespace, Name: candidateArtifact, Required: true},
			})
			assertStageResult(t, build, test.buildResult, test.buildMediaType)
			assertTemplateSelector(t, build, test.builderTemplate)
			assertTemplateSelector(t, validate, test.validatorTemplate)
			assertNext(t, build.On.Succeeded, test.validateStage)
			if validate.On.Succeeded.Kind != TransitionSucceed {
				t.Fatalf("validation success transition = %+v", validate.On.Succeeded)
			}
			for stageName, stage := range workflow.Stages {
				if stage.Context.Workspace != nil {
					t.Fatalf("precomputed-analysis Stage %q unexpectedly hydrates workspace", stageName)
				}
				for contextName, artifact := range stage.Context.Artifacts {
					if artifact.Namespace == "analysis" {
						t.Fatalf("Stage %q context %q reads hidden prior-Run analysis", stageName, contextName)
					}
				}
			}
		})
	}
}

func assertTemplateSelector(t *testing.T, stage ResolvedStage, want string) {
	t.Helper()
	if len(stage.Agents) != 1 {
		t.Fatalf("agents = %+v", stage.Agents)
	}
	for _, binding := range stage.Agents {
		got := binding.Template.Ref.TemplateID + "@" + binding.Template.Ref.Version
		if got != want {
			t.Fatalf("template = %s, want %s", got, want)
		}
	}
}
