package public

import (
	"testing"

	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
)

func TestAuditProfileMethodProjectionPreservesDetailAndSummaryBounds(t *testing.T) {
	profile := auditprofile.ResolvedAuditProfile{Methods: map[string]auditprofile.AuditCheckMethodMapping{"named": {PlanSchema: contracts.FindingCheckPlanSchema, Method: "verify", Phase: "verify", WorkflowRole: "renamed", RequiredCapabilities: []contracts.CheckCapabilityRequirement{{Stage: "check", Agent: "worker", Ref: contracts.ToolsetRef{ToolsetID: "audit-results", Version: "2"}, Tools: []string{"read_audit_task"}}}}}}
	projection := auditservice.ProfileProjection{Profile: profile}
	if len(auditProfileReadModel(projection, false).Methods) != 0 {
		t.Fatal("list hydrated the full method closure")
	}
	detail := auditProfileReadModel(projection, true)
	if detail.Methods["named"].WorkflowRole != "renamed" {
		t.Fatal("detail omitted exact method binding")
	}
	detail.Methods["named"].RequiredCapabilities[0].Tools[0] = "changed"
	if profile.Methods["named"].RequiredCapabilities[0].Tools[0] != "read_audit_task" {
		t.Fatal("response aliases retained profile")
	}
}
