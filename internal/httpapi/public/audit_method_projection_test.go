package public

import (
	"encoding/json"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/auditservice"
	"github.com/grauwolf32/contractor/internal/config/auditprofile"
	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
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
	data, err := json.Marshal(detail.Methods["named"])
	if err != nil {
		t.Fatal(err)
	}
	var mapping map[string]any
	if err := json.Unmarshal(data, &mapping); err != nil {
		t.Fatal(err)
	}
	if _, present := mapping["phase"]; present || len(mapping) != 4 {
		t.Fatal("public method exposes resolved-only fields")
	}
	if err := loadPublicOpenAPI(t).Components.Schemas["AuditCheckMethodMapping"].Value.VisitJSON(mapping, openapi3.EnableJSONSchema2020()); err != nil {
		t.Fatal(err)
	}
	detail.Methods["named"].RequiredCapabilities[0].Tools[0] = "changed"
	if profile.Methods["named"].RequiredCapabilities[0].Tools[0] != "read_audit_task" {
		t.Fatal("response aliases retained profile")
	}
}

func TestSharedCheckMethodAuthoringMatchesPublicMapping(t *testing.T) {
	var fixture struct {
		AuthoredMappingCases []struct {
			Name    string         `json:"name"`
			Valid   bool           `json:"valid"`
			Mapping map[string]any `json:"mapping"`
		} `json:"authoredMappingCases"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-check-plans", "accepted-plan-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.AuthoredMappingCases) == 0 {
		t.Fatal("shared authored mappings are empty")
	}
	schema := loadPublicOpenAPI(t).Components.Schemas["AuditCheckMethodMapping"].Value
	for _, test := range fixture.AuthoredMappingCases {
		t.Run(test.Name, func(t *testing.T) {
			if err := schema.VisitJSON(test.Mapping, openapi3.EnableJSONSchema2020()); (err == nil) != test.Valid {
				t.Fatalf("public valid=%t want %t: %v", err == nil, test.Valid, err)
			}
		})
	}
}
