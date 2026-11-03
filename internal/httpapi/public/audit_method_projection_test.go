package public

import (
	"encoding/json"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

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
