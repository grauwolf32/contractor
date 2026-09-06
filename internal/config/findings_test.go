package config

import (
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestFindingsToolVersionsAndOperationSelection(t *testing.T) {
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
		{"security-findings@1", []string{"list_findings"}, false},
		{"security-findings@2", []string{"finding"}, true},
		{"security-findings@2", []string{"list_findings"}, true},
		{"security-findings@2", []string{"finding", "list_findings"}, true},
		{"security-findings@2", []string{"confirm_finding"}, false},
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
		{Ref: "security-findings@2", Tools: []string{"finding"}},
	}); err == nil || !strings.Contains(err.Error(), "collides") {
		t.Fatalf("collision error = %v", err)
	}
}

func TestFindingsReaderRequiresDeclaredCollectionInputInWorkflowSnapshot(t *testing.T) {
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	for _, test := range []struct {
		name  string
		tools []string
		input *ArtifactSlot
		valid bool
	}{
		{"writer", []string{"finding"}, nil, true},
		{"missing", []string{"list_findings"}, nil, false},
		{"optional", []string{"list_findings"}, &ArtifactSlot{MediaTypes: []string{auditdomain.FindingCollectionMediaType}}, false},
		{"wrong-media", []string{"list_findings"}, &ArtifactSlot{Required: true, MediaTypes: []string{"application/json"}}, false},
		{"reader", []string{"list_findings"}, &ArtifactSlot{Required: true, MediaTypes: []string{auditdomain.FindingCollectionMediaType}}, true},
		{"both", []string{"finding", "list_findings"}, &ArtifactSlot{Required: true, MediaTypes: []string{auditdomain.FindingCollectionMediaType}}, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			workflow, err := snapshot.Workflow("openapi-from-workspace@5")
			if err != nil {
				t.Fatal(err)
			}
			stage := workflow.Stages[workflow.EntryStage]
			for name, agent := range stage.Agents {
				agent.Template.Toolsets = append(agent.Template.Toolsets, contracts.ToolsetSelection{
					Ref: contracts.ToolsetRef{ToolsetID: "security-findings", Version: "2"}, Tools: test.tools,
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
			if err != nil && !strings.Contains(err.Error(), "Workflow input findings") {
				t.Fatal(err)
			}
		})
	}
}
