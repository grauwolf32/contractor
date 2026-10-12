package resolved

import (
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
)

// The requirement is data on the Toolset descriptor; the check knows no
// Toolset by name.
func TestToolWorkflowInputRequirementComesFromDescriptor(t *testing.T) {
	toolsets := map[string]toolsetDescriptor{
		"corpus-reader@3": {
			Tools: []string{"index", "read_corpus"},
			RequiredWorkflowInputs: map[string]workflowInputRequirement{
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
		requirement map[string]workflowInputRequirement
		want        string
	}{
		{"unknown tool", map[string]workflowInputRequirement{"write": {Input: "corpus", MediaType: "text/plain"}}, `requires a Workflow input for unknown tool "write"`},
		{"invalid input", map[string]workflowInputRequirement{"read": {Input: "../corpus", MediaType: "text/plain"}}, "Workflow input"},
		{"non-canonical media type", map[string]workflowInputRequirement{"read": {Input: "corpus", MediaType: "Text/Plain"}}, `non-canonical media type "Text/Plain"`},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := normalizeDescriptors(descriptorTable{Toolsets: map[string]toolsetDescriptor{
				"corpus-reader@3": {Tools: []string{"read"}, RequiredWorkflowInputs: test.requirement},
			}})
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("descriptor error = %v, want %q", err, test.want)
			}
		})
	}
	want := workflowInputRequirement{Input: "findings", MediaType: contracts.FindingCollectionMediaType}
	if got := registeredDescriptors().Toolsets["security-findings@1"].RequiredWorkflowInputs; len(got) != 1 || got["list_findings"] != want {
		t.Fatalf("security-findings@1 Workflow input requirements = %+v", got)
	}
}
