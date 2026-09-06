package auditservice

import (
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestProfileCompatibilityReasonsAreClosedOrderedAndDeduplicated(t *testing.T) {
	profile, err := loadAuditServiceProfiles(t).AuditProfile("test-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	profile.Execution.MaxRounds = 2
	profile.Execution.BatchSize = 2
	profile.Interaction = config.AuditInteractionPolicy{
		ActiveChecks:        config.AuditActiveChecksApprovalRequired,
		FindingConfirmation: config.AuditFindingHumanRequired,
		NotApplicable:       config.AuditNotApplicableHumanRequired,
		ReportAcceptance:    config.AuditReportHumanRequired,
	}
	binding := profile.Workflows["check"]
	binding.Kind = config.AuditWorkflowDiscovery
	profile.Workflows["discovery"] = binding

	got := ProfileCompatibility(profile).Reasons
	want := []CompatibilityReason{
		ReasonBatchingUnsupported,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("compatibility reasons = %v, want %v", got, want)
	}
}

func TestProfileCompatibilityUsesExplicitToolAndOutputClassifications(t *testing.T) {
	profile, err := loadAuditServiceProfiles(t).AuditProfile("test-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	binding := profile.Workflows["check"]
	binding.Outputs["proposal-summary"] = "result"
	profile.Workflows["check"] = binding
	if compatibility := ProfileCompatibility(profile); !compatibility.ServerCompatible {
		t.Fatalf("arbitrary output name was classified as a finding surface: %v", compatibility.Reasons)
	}

	binding = profile.Workflows["check"]
	stage := binding.Workflow.Stages["check"]
	agent := stage.Agents["worker"]
	agent.Template.Toolsets = append(agent.Template.Toolsets, contracts.ToolsetSelection{
		Ref:   contracts.ToolsetRef{ToolsetID: "http-tools", Version: "1"},
		Tools: []string{"http_request"},
	})
	stage.Agents["worker"] = agent
	binding.Workflow.Stages["check"] = stage
	profile.Workflows["check"] = binding
	if got := ProfileCompatibility(profile).Reasons; !reflect.DeepEqual(got, []CompatibilityReason{
		ReasonAutomaticActiveChecksUnsupported,
	}) {
		t.Fatalf("active tool compatibility reasons = %v", got)
	}
	if !workflowRoleSelectsClassifiedTool(profile, "check", true) {
		t.Fatal("active tool was not attributed to its exact Workflow role")
	}
	profile.Interaction.ActiveChecks = config.AuditActiveChecksApprovalRequired
	revision := "checklist-r1"
	inventory, err := buildInventory(profile, DraftSelection{Scope: Scope{}}, map[string]artifacts.ReadResult{
		"checklist": {
			Ref: contracts.ArtifactRef{Namespace: "project", Name: "checklist", Revision: &revision},
			Payload: artifacts.Payload{MediaType: auditdomain.JSONMediaType, Data: []byte(
				`{"schema":"contractor.audit.checklist.v1","items":[{"key":"active","version":"1","statement":"Exercise the endpoint.","applicability":"always","allowed_methods":["http"],"required_evidence":[],"review_policy":"automatic"}]}`,
			)},
		},
	})
	if err != nil || len(inventory.Worklist.Items) != 1 ||
		inventory.Worklist.Items[0].ApprovalRequirement != auditdomain.ApprovalActiveCheck {
		t.Fatalf("active Workflow role did not strengthen its inventory gate: (%+v, %v)", inventory.Worklist.Items, err)
	}

	profile = mustAuditServiceProfile(t)
	binding = profile.Workflows["check"]
	binding.Outputs["proposals"] = "result"
	profile.Workflows["check"] = binding
	if got := ProfileCompatibility(profile).Reasons; !reflect.DeepEqual(got, []CompatibilityReason{
		ReasonFindingConfirmationUnsupported,
	}) {
		t.Fatalf("reserved proposal surface reasons = %v", got)
	}
}

func mustAuditServiceProfile(t *testing.T) config.ResolvedAuditProfile {
	t.Helper()
	profile, err := loadAuditServiceProfiles(t).AuditProfile("test-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	return profile
}
