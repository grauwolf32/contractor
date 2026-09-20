package auditservice

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/artifacts"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstandards"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestWSTGStandardInventoryRequiresActiveApprovalForEveryItem(t *testing.T) {
	root := filepath.Join("..", "..", "configs")
	snapshot, err := config.Load(root, config.MVPDescriptors())
	if err != nil {
		t.Fatal(err)
	}
	profile, err := snapshot.AuditProfile("owasp-wstg-4-2-active-http@1")
	if err != nil {
		t.Fatal(err)
	}
	_, pkg, err := auditstandards.PackageDirectory(filepath.Join(root, "audit-standards", "owasp-wstg-4.2-http.1"))
	if err != nil {
		t.Fatal(err)
	}
	if !ProfileCompatibility(profile).ServerCompatible || !workflowRoleSelectsClassifiedTool(profile, "check", true) {
		t.Fatal("active WSTG must use a compatible, classified HTTP workflow")
	}
	revision := "standard-r1"
	standard := auditstandards.ResolvedPackage{Package: *pkg}
	standard.Source.Artifact = contracts.ArtifactRef{Namespace: "audit-standards", Name: "wstg", Revision: &revision}
	inventory, err := buildInventory(profile, DraftSelection{Scope: Scope{
		Target: "https://example.test", AuthorizationScope: "Approved disposable HTTP test environment",
	}}, nil, []auditstandards.ResolvedPackage{standard})
	if err != nil || len(inventory.Tasks) != 94 {
		t.Fatalf("active WSTG inventory: %d tasks, %v", len(inventory.Tasks), err)
	}
	for _, task := range inventory.Tasks {
		if task.Item.ApprovalRequirement != auditdomain.ApprovalActiveCheck {
			t.Fatalf("standard item bypasses active approval: %s", task.Item.ItemKey)
		}
		kind, digest, state, err := materializedItemApproval("audit-wstg", profile, "item-"+task.Item.ItemKey, task.Item, auditstore.ExactArtifact{
			Ref: contracts.ArtifactRef{Namespace: "audit-wstg", Name: task.Document.ItemKey, Revision: &revision},
		})
		if err != nil || kind != auditstore.ItemApprovalActiveCheck || state != auditstore.ItemAwaitingReview || digest == "" {
			t.Fatalf("active WSTG item can run before review: %s, %s, %s, %v", task.Item.ItemKey, kind, state, err)
		}
	}

	// Approval applies only to roles that actually have active tools.
	binding := profile.Workflows["check"]
	stage := binding.Workflow.Stages["check"]
	agent := stage.Agents["verifier"]
	agent.Template.Toolsets = nil
	stage.Agents["verifier"] = agent
	binding.Workflow.Stages["check"] = stage
	profile.Workflows["check"] = binding
	inventory, err = buildInventory(profile, DraftSelection{}, nil, []auditstandards.ResolvedPackage{standard})
	if err != nil || inventory.Tasks[0].Item.ApprovalRequirement != auditdomain.ApprovalNone {
		t.Fatalf("non-active standard role inherited an active gate: %v", err)
	}
}

func TestProfileCompatibilityAcceptsImplementedAdvancedPoliciesAndBatching(t *testing.T) {
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
	want := []CompatibilityReason{}
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
	}, nil)
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
