package config

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestRepositoryPrecomputedAnalysisWorkflowVariants(t *testing.T) {
	t.Parallel()

	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	cases := []struct {
		variantRef       string
		sourceRef        string
		buildStage       string
		validateStage    string
		seedInput        string
		buildResult      string
		buildMediaType   string
		outputMediaTypes map[string]ArtifactSlot
	}{
		{
			variantRef: "openapi-from-analysis@1", sourceRef: "openapi-from-source@1",
			buildStage: "openapi_build", validateStage: "openapi_validate",
			seedInput: "existing_openapi", buildResult: "openapi", buildMediaType: "application/yaml",
			outputMediaTypes: map[string]ArtifactSlot{
				"openapi":           {Required: true, MediaTypes: []string{"application/yaml"}},
				"validation_report": {Required: true, MediaTypes: []string{"text/markdown"}},
			},
		},
		{
			variantRef: "likec4-from-analysis@1", sourceRef: "likec4-from-source@1",
			buildStage: "likec4_build", validateStage: "likec4_validate",
			seedInput: "existing_likec4", buildResult: "architecture", buildMediaType: "text/vnd.likec4",
			outputMediaTypes: map[string]ArtifactSlot{
				"architecture":      {Required: true, MediaTypes: []string{"text/vnd.likec4"}},
				"validation_report": {Required: true, MediaTypes: []string{"text/markdown"}},
			},
		},
	}

	for _, test := range cases {
		t.Run(test.variantRef, func(t *testing.T) {
			variant, err := snapshot.Workflow(test.variantRef)
			if err != nil {
				t.Fatal(err)
			}
			source, err := snapshot.Workflow(test.sourceRef)
			if err != nil {
				t.Fatal(err)
			}
			if variant.EntryStage != test.buildStage || len(variant.Stages) != 2 {
				t.Fatalf("variant topology = entry %q, stages %d", variant.EntryStage, len(variant.Stages))
			}
			if _, exists := variant.Stages["dependency_discovery"]; exists {
				t.Fatal("variant unexpectedly contains dependency discovery")
			}
			if _, exists := variant.Stages["project_discovery"]; exists {
				t.Fatal("variant unexpectedly contains project discovery")
			}
			for name, expected := range map[string]ArtifactSlot{
				"source":            {Required: true, MediaTypes: []string{"application/zip"}},
				"dependency_report": {Required: true, MediaTypes: []string{"text/markdown"}},
				"project_report":    {Required: true, MediaTypes: []string{"text/markdown"}},
			} {
				if !reflect.DeepEqual(variant.Inputs[name], expected) {
					t.Fatalf("input %q = %+v, want %+v", name, variant.Inputs[name], expected)
				}
			}
			if len(variant.Inputs) != 4 || !reflect.DeepEqual(variant.Inputs[test.seedInput], source.Inputs[test.seedInput]) {
				t.Fatalf("variant inputs = %+v", variant.Inputs)
			}
			if !reflect.DeepEqual(variant.Outputs, test.outputMediaTypes) {
				t.Fatalf("variant outputs = %+v, want %+v", variant.Outputs, test.outputMediaTypes)
			}

			build := variant.Stages[test.buildStage]
			validate := variant.Stages[test.validateStage]
			assertContext(t, build, map[string]ContextArtifact{
				"source":            {Namespace: "inputs", Name: "source", Required: true},
				"dependency_report": {Namespace: "inputs", Name: "dependency_report", Required: true},
				"project_report":    {Namespace: "inputs", Name: "project_report", Required: true},
				test.seedInput:      {Namespace: "inputs", Name: test.seedInput, Required: false},
			})
			candidateName := "openapi_candidate"
			candidateNamespace := "openapi"
			candidateArtifact := "openapi"
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
			if !reflect.DeepEqual(build.Agents, source.Stages[test.buildStage].Agents) ||
				!reflect.DeepEqual(validate.Agents, source.Stages[test.validateStage].Agents) ||
				!reflect.DeepEqual(build.On, source.Stages[test.buildStage].On) ||
				!reflect.DeepEqual(validate.On, source.Stages[test.validateStage].On) ||
				!reflect.DeepEqual(validate.WorkflowOutputs, source.Stages[test.validateStage].WorkflowOutputs) {
				t.Fatal("variant does not preserve source Workflow templates, retries, or outputs")
			}
			for stageName, stage := range variant.Stages {
				for contextName, artifact := range stage.Context.Artifacts {
					if artifact.Namespace == "analysis" {
						t.Fatalf("Stage %q context %q reads hidden prior-Run analysis", stageName, contextName)
					}
				}
			}
			variantJSON, _ := json.Marshal(variant)
			sourceJSON, _ := json.Marshal(source)
			if reflect.DeepEqual(variantJSON, sourceJSON) || variant.Ref == source.Ref {
				t.Fatal("variant and source Workflow snapshots are not independent")
			}
		})
	}
}
