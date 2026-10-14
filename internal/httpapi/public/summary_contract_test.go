package public

import (
	"encoding/json"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPublicStageSummaryContracts(t *testing.T) {
	schema := loadPublicOpenAPI(t).Components.Schemas["StageContentResult"].Value
	data := contractstest.ReadFile(t, "api", "testdata", "stage-summary-cases.json")
	var cases []struct {
		Name   string
		Valid  bool
		Result any
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			if err := schema.VisitJSON(tc.Result, openapi3.EnableJSONSchema2020()); (err == nil) != tc.Valid {
				t.Fatalf("valid=%v: %v", tc.Valid, err)
			}
		})
	}
}
