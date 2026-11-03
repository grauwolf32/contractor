package auditservice

import (
	"encoding/json"
	"testing"

	artifactpolicy "github.com/grauwolf32/contractor/internal/artifacts/policy"
	"github.com/grauwolf32/contractor/internal/auditdomain"
	"github.com/grauwolf32/contractor/internal/auditstore"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func routedContextFixture(t *testing.T, policy contracts.ActiveChecksPolicy) (*RoutedCheckContext, auditprofile.ResolvedAuditProfile, auditdomain.AcceptedPlanSource) {
	t.Helper()
	profile, err := loadAuditServiceProfiles(t).AuditProfile("test-checklist@1")
	if err != nil {
		t.Fatal(err)
	}
	profile.Interaction.ActiveChecks = contracts.ActiveChecksApprovalRequired
	ordinary := profile.Workflows["check"]
	profile.Workflows["replay-looking-name"] = ordinary
	active := profile.Workflows["check"]
	// Clone through the resolved profile before adding classified tools so the
	// ordinary route does not inherit another route's active capability.
	clone := profile.Clone()
	active = clone.Workflows["check"]
	stage := active.Workflow.Stages["check"]
	agent := stage.Agents["worker"]
	agent.Template.Toolsets = append(agent.Template.Toolsets, contracts.ToolsetSelection{Ref: contracts.ToolsetRef{ToolsetID: "http-tools", Version: "1"}, Tools: []string{"http_request"}})
	stage.Agents["worker"] = agent
	active.Workflow.Stages["check"] = stage
	profile.Workflows["passive-looking-name"] = active
	profile.Methods = map[string]auditprofile.AuditCheckMethodMapping{
		"ordinary-route": {PlanSchema: contracts.FindingCheckPlanSchema, Method: "applicability", Phase: "applicability", WorkflowRole: "replay-looking-name", RequiredCapabilities: []contracts.CheckCapabilityRequirement{}},
		"active-route":   {PlanSchema: contracts.FindingCheckPlanSchema, Method: "verify", Phase: "verify", WorkflowRole: "passive-looking-name", RequiredCapabilities: []contracts.CheckCapabilityRequirement{}},
	}
	profile.Ref.Digest, err = auditprofile.Digest(profile)
	if err != nil {
		t.Fatal(err)
	}
	baseline := BaselineSnapshot{Schema: BaselineSchema, ActiveChecks: policy, Scope: Scope{Target: "https://fixture.invalid", AuthorizationScope: "only fixture"}}
	bytes, err := json.Marshal(baseline)
	if err != nil {
		t.Fatal(err)
	}
	audit := auditstore.Audit{AuditID: "audit", OwnerID: "owner", ProjectID: "project", Profile: auditstore.ProfileIdentity{Name: profile.Ref.Name, Version: profile.Ref.Version, Digest: profile.Ref.Digest}, BaselineSnapshot: bytes, Limits: auditstore.Limits{MaxItemsPerRound: 10, MaxItemsTotal: 20}}
	context, err := NewRoutedCheckContext(audit, profile, baseline)
	if err != nil {
		t.Fatal(err)
	}
	revision := "r1"
	ordinal := 7
	original := contracts.ExactArtifact{Ref: contracts.ArtifactRef{Namespace: "findings", Name: "proposal", Revision: &revision}, Digest: auditdomain.DigestBytes([]byte("proposal")), MediaType: auditdomain.JSONMediaType, SizeBytes: 8}
	retained := original.Clone()
	retained.Ref.Namespace = artifactpolicy.FindingMaterialsNamespace
	return context, profile, auditdomain.AcceptedPlanSource{Kind: "finding-receipt", ReceiptID: "receipt", ProposedCheckOrdinal: &ordinal, RunID: "producer", Proposal: original, RetainedProposal: retained}
}

func TestRoutedCheckContextPinsOriginalProfileAndPerRouteAuthority(t *testing.T) {
	context, profile, source := routedContextFixture(t, contracts.ActiveChecksApprovalRequired)
	for _, test := range []struct {
		method, phase, role string
		approval            auditdomain.ApprovalRequirement
	}{
		{"applicability", "applicability", "replay-looking-name", auditdomain.ApprovalNone},
		{"verify", "verify", "passive-looking-name", auditdomain.ApprovalActiveCheck},
	} {
		plan, err := context.build(source, auditdomain.CheckPlan{Schema: contracts.FindingCheckPlanSchema, Method: test.method, Phase: test.phase, Objective: "Inspect the exact claim <>&"}, nil)
		if err != nil {
			t.Fatal(err)
		}
		if plan.Route.WorkflowRole != test.role || plan.Authorization.ApprovalRequirement != test.approval || plan.ProfileDigest != profile.Ref.Digest || plan.Source.ProposedCheckOrdinal == source.ProposedCheckOrdinal {
			t.Fatalf("route/authority/source clone drift: %+v", plan)
		}
		workflowBytes, _ := json.Marshal(profile.Workflows[test.role].Workflow)
		if plan.Route.WorkflowDigest != auditdomain.DigestBytes(workflowBytes) {
			t.Fatal("opaque dependency consumer digest was reinterpreted")
		}
		if err := context.Validate(plan, nil, test.role); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := context.build(source, auditdomain.CheckPlan{Schema: contracts.FindingCheckPlanSchema, Method: "replay", Phase: "replay", Objective: "Not mapped"}, nil); err == nil {
		t.Fatal("unmapped method selected a fallback role")
	}
	// PRJ-R104 pins the effective value independently from the immutable
	// profile digest, including automatic execution of active routes.
	automatic, original, source := routedContextFixture(t, contracts.ActiveChecksAutomatic)
	plan, err := automatic.build(source, auditdomain.CheckPlan{Schema: contracts.FindingCheckPlanSchema, Method: "verify", Phase: "verify", Objective: "Exact automatic policy"}, nil)
	if err != nil || plan.Authorization.ActiveChecks != contracts.ActiveChecksAutomatic || plan.Authorization.ApprovalRequirement != auditdomain.ApprovalNone || plan.ProfileDigest != original.Ref.Digest {
		t.Fatalf("baseline overlay changed profile identity: %+v %v", plan, err)
	}
}

func TestRoutedCheckContextRejectsChangedRouteAndAuthorization(t *testing.T) {
	context, _, source := routedContextFixture(t, contracts.ActiveChecksApprovalRequired)
	for _, test := range []struct {
		name   string
		change func(*auditdomain.AcceptedPlan)
	}{
		{"owner", func(p *auditdomain.AcceptedPlan) { p.OwnerID = "foreign" }},
		{"ordinal", func(p *auditdomain.AcceptedPlan) { ordinal := 8; p.Source.ProposedCheckOrdinal = &ordinal }},
		{"scope", func(p *auditdomain.AcceptedPlan) {
			p.Authorization.ScopeDigest = auditdomain.DigestBytes([]byte("other"))
		}},
		{"limits", func(p *auditdomain.AcceptedPlan) {
			p.Authorization.LimitsDigest = auditdomain.DigestBytes([]byte("other"))
		}},
		{"policy", func(p *auditdomain.AcceptedPlan) { p.Authorization.ActiveChecks = contracts.ActiveChecksAutomatic }},
		{"approval", func(p *auditdomain.AcceptedPlan) { p.Authorization.ApprovalRequirement = auditdomain.ApprovalNone }},
		{"binding", func(p *auditdomain.AcceptedPlan) { p.Route.BindingDigest = auditdomain.DigestBytes([]byte("other")) }},
		{"mapping", func(p *auditdomain.AcceptedPlan) { p.Route.MappingID = "other" }},
	} {
		t.Run(test.name, func(t *testing.T) {
			plan, err := context.build(source, auditdomain.CheckPlan{Schema: contracts.FindingCheckPlanSchema, Method: "verify", Phase: "verify", Objective: "Verify exact claim"}, nil)
			if err != nil {
				t.Fatal(err)
			}
			test.change(&plan)
			if context.Validate(plan, nil, "passive-looking-name") == nil {
				t.Fatal("changed accepted route or authority was allowed")
			}
		})
	}
}
