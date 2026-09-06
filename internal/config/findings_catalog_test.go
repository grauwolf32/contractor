package config

import (
	"reflect"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestFindingsCatalogSeparatesProducerAndGenericReader(t *testing.T) {
	snapshot := mustLoad(t, repositoryConfigRoot, MVPDescriptors())
	profile, err := snapshot.AuditProfile("openapi-operation-trace@3")
	if err != nil {
		t.Fatal(err)
	}
	producer := profile.Workflows["trace"].Workflow
	stage := producer.Stages[producer.EntryStage]
	agent := stage.Agents["checker"].Template
	if agent.Ref.TemplateID != "audit_openapi_operation_tracer" || agent.Ref.Version != "2" ||
		producer.Ref.Name != "audit-openapi-operation-trace" || producer.Ref.Version != "2" ||
		profile.Interaction.FindingConfirmation != AuditFindingHumanRequired ||
		profile.Interaction.ActiveChecks != AuditActiveChecksProhibited {
		t.Fatalf("producer or policy is not explicitly versioned: %+v", profile)
	}
	if !selectedTool(agent.Toolsets, "security-findings@2", "finding") ||
		selectedTool(agent.Toolsets, "security-findings@2", "list_findings") ||
		!selectedTool(agent.Toolsets, "text-artifacts@1", "write_text_artifact") ||
		!selectedTool(agent.Toolsets, "code-analysis@1", "graph_summary") ||
		!selectedTool(agent.Toolsets, "audit-results@1", "submit_check_result") {
		t.Fatalf("producer selection = %+v", agent.Toolsets)
	}
	if stage.Context.Workspace == nil || stage.Context.Workspace.Mode != contracts.WorkspaceModeOverlay ||
		!reflect.DeepEqual(stage.Context.Workspace.Sources, []WorkspaceSource{{Artifact: "source", Target: ""}}) {
		t.Fatal("producer graph must use its pinned source workspace")
	}
	if !strings.Contains(agent.Instructions.Text, "proposal_keys") ||
		!strings.Contains(agent.Instructions.Text, "reproduction") ||
		!strings.Contains(agent.Instructions.Text, "operation-resolution") {
		t.Fatal("producer instructions omit receipt handoff or coverage/reproduction discipline")
	}
	reader, err := snapshot.Workflow("findings-review@1")
	if err != nil {
		t.Fatal(err)
	}
	if len(reader.Inputs) != 1 || !reader.Inputs["findings"].Required ||
		!reflect.DeepEqual(reader.Inputs["findings"].MediaTypes, []string{auditdomain.FindingCollectionMediaType}) ||
		!reader.Outputs["report"].Required || !reader.Outputs["report"].Primary {
		t.Fatalf("reader contract = %+v", reader)
	}
	readerStage := reader.Stages[reader.EntryStage]
	readerAgent := readerStage.Agents["analyst"].Template
	want := map[string][]string{
		"security-findings@2": {"list_findings"}, "run-artifacts@1": {"read_artifact"},
		"text-artifacts@1": {"write_text_artifact"},
	}
	got := map[string][]string{}
	for _, selection := range readerAgent.Toolsets {
		got[selection.Ref.ToolsetID+"@"+selection.Ref.Version] = selection.Tools
	}
	if !reflect.DeepEqual(got, want) || len(readerAgent.Skills) != 0 || readerStage.Context.Workspace != nil {
		t.Fatalf("generic reader acquired scenario-specific capabilities: %+v", readerAgent)
	}
	// All newly selected role instructions have their own paths. Existing
	// versioned instruction closures and the trace Skill are not rewritten.
	if agent.Instructions.Ref == "instructions/audit-openapi-operation-tracer-worker.md" {
		t.Fatal("new producer overwrote its prior instruction path")
	}
}
