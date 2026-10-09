package config

import (
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestFindingFacadesAndOperationSelection(t *testing.T) {
	descriptors, err := normalizeDescriptors(MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	current := &loader{descriptors: descriptors}
	for _, test := range []struct {
		ref   string
		tools []string
		valid bool
	}{
		{"security-findings@1", []string{"finding"}, true},
		{"security-findings@1", []string{"list_findings"}, true},
		{"security-findings@1", []string{"finding", "list_findings"}, true},
		{"security-findings-code@1", []string{"finding"}, true},
		{"security-findings-http@1", []string{"finding"}, true},
		{"security-findings-code@1", []string{"list_findings"}, false},
		{"security-findings@2", []string{"finding"}, false},
		{"security-findings@1", []string{"confirm_finding"}, false},
	} {
		t.Run(test.ref+strings.Join(test.tools, "+"), func(t *testing.T) {
			got, err := current.resolveToolsets(&[]toolsetSelectionSource{{Ref: test.ref, Tools: test.tools}})
			if (err == nil) != test.valid {
				t.Fatalf("resolve tools = %v, %v", got, err)
			}
			if test.valid && !reflect.DeepEqual(got[0].Tools, test.tools) {
				t.Fatalf("selection changed: %+v", got)
			}
		})
	}
	if _, err := current.resolveToolsets(&[]toolsetSelectionSource{
		{Ref: "security-findings@1", Tools: []string{"finding"}},
		{Ref: "security-findings-code@1", Tools: []string{"finding"}},
	}); err == nil || !strings.Contains(err.Error(), "collides") {
		t.Fatalf("collision error = %v", err)
	}
}

func TestFindingsReaderRequiresDeclaredCollectionInputInWorkflowSnapshot(t *testing.T) {
	snapshot := mustLoad(t, copyCoreFixture(t), MVPDescriptors())
	for _, test := range []struct {
		name  string
		tools []string
		input *ArtifactSlot
		valid bool
	}{
		{"writer", []string{"finding"}, nil, true},
		{"missing", []string{"list_findings"}, nil, false},
		{"optional", []string{"list_findings"}, &ArtifactSlot{MediaTypes: []string{contracts.FindingCollectionMediaType}}, false},
		{"wrong-media", []string{"list_findings"}, &ArtifactSlot{Required: true, MediaTypes: []string{"application/json"}}, false},
		{"reader", []string{"list_findings"}, &ArtifactSlot{Required: true, MediaTypes: []string{contracts.FindingCollectionMediaType}}, true},
		{"both", []string{"finding", "list_findings"}, &ArtifactSlot{Required: true, MediaTypes: []string{contracts.FindingCollectionMediaType}}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			workflow, err := snapshot.Workflow("artifact-copy@1")
			if err != nil {
				t.Fatal(err)
			}
			stage := workflow.Stages[workflow.EntryStage]
			for name, agent := range stage.Agents {
				agent.Template.Toolsets = append(agent.Template.Toolsets, contracts.ToolsetSelection{
					Ref: contracts.ToolsetRef{ToolsetID: "security-findings", Version: "1"}, Tools: test.tools,
				})
				stage.Agents[name] = agent
				break
			}
			workflow.Stages[workflow.EntryStage] = stage
			if test.input != nil {
				workflow.Inputs["findings"] = *test.input
			}
			err = ValidateWorkflowGraph(workflow)
			if (err == nil) != test.valid {
				t.Fatalf("workflow validation error = %v", err)
			}
			const want = `Stage "copy" Agent "builder" security-findings@1 list_findings requires the required Workflow input findings with media type application/vnd.contractor.findings-collection+zip`
			if err != nil && err.Error() != want {
				t.Fatalf("workflow validation error = %q, want %q", err, want)
			}
		})
	}
}

// The requirement is data on the Toolset descriptor; the check knows no
// Toolset by name.
func TestToolWorkflowInputRequirementComesFromDescriptor(t *testing.T) {
	toolsets := map[string]ToolsetDescriptor{
		"corpus-reader@3": {
			Tools: []string{"index", "read_corpus"},
			RequiredWorkflowInputs: map[string]WorkflowInputRequirement{
				"read_corpus": {Input: "corpus", MediaType: "text/plain"},
			},
		},
	}
	for _, test := range []struct {
		name  string
		tools []string
		input *ArtifactSlot
		want  string
	}{
		{"unrequired tool", []string{"index"}, nil, ""},
		{"missing", []string{"read_corpus"}, nil, `Stage "read" Agent "reader" corpus-reader@3 read_corpus requires the required Workflow input corpus with media type text/plain`},
		{"any media type is not the declared one", []string{"read_corpus"}, &ArtifactSlot{Required: true, MediaTypes: []string{"*/*"}}, "requires the required Workflow input corpus"},
		{"declared", []string{"index", "read_corpus"}, &ArtifactSlot{Required: true, MediaTypes: []string{"text/markdown", "text/plain"}}, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			workflow := ResolvedWorkflow{
				Inputs: map[string]ArtifactSlot{},
				Stages: map[string]ResolvedStage{"read": {Agents: map[string]ResolvedAgentBinding{
					"reader": {Template: contracts.ResolvedAgentTemplate{Toolsets: []contracts.ToolsetSelection{{
						Ref: contracts.ToolsetRef{ToolsetID: "corpus-reader", Version: "3"}, Tools: test.tools,
					}}}},
				}}},
			}
			if test.input != nil {
				workflow.Inputs["corpus"] = *test.input
			}
			err := validateToolWorkflowInputs(workflow, toolsets)
			if test.want == "" && err != nil || test.want != "" && (err == nil || !strings.Contains(err.Error(), test.want)) {
				t.Fatalf("validation error = %v, want %q", err, test.want)
			}
		})
	}
}

func TestToolsetDescriptorWorkflowInputRequirementsAreValidated(t *testing.T) {
	for _, test := range []struct {
		name        string
		requirement map[string]WorkflowInputRequirement
		want        string
	}{
		{"unknown tool", map[string]WorkflowInputRequirement{"write": {Input: "corpus", MediaType: "text/plain"}}, `requires a Workflow input for unknown tool "write"`},
		{"invalid input", map[string]WorkflowInputRequirement{"read": {Input: "../corpus", MediaType: "text/plain"}}, "Workflow input"},
		{"non-canonical media type", map[string]WorkflowInputRequirement{"read": {Input: "corpus", MediaType: "Text/Plain"}}, `non-canonical media type "Text/Plain"`},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := normalizeDescriptors(Descriptors{Toolsets: map[string]ToolsetDescriptor{
				"corpus-reader@3": {Tools: []string{"read"}, RequiredWorkflowInputs: test.requirement},
			}})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("descriptor error = %v, want %q", err, test.want)
			}
		})
	}
	normalized, err := normalizeDescriptors(MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	want := WorkflowInputRequirement{Input: "findings", MediaType: contracts.FindingCollectionMediaType}
	if got := normalized.Toolsets["security-findings@1"].RequiredWorkflowInputs; len(got) != 1 || got["list_findings"] != want {
		t.Fatalf("security-findings@1 Workflow input requirements = %+v", got)
	}
}
