package public

import (
	"encoding/json"
	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"testing"
)

func TestSharedPublicFactEvidenceProjection(t *testing.T) {
	var fixture struct {
		PublicDocuments []struct {
			Name, Schema string
			Valid        bool
			Document     any
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "finding-fact-evidence-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	document := loadPublicOpenAPI(t)
	for _, c := range fixture.PublicDocuments {
		t.Run(c.Name, func(t *testing.T) {
			err := document.Components.Schemas[c.Schema].Value.VisitJSON(c.Document, openapi3.EnableJSONSchema2020())
			if (err == nil) != c.Valid {
				t.Fatalf("valid=%v error=%v", c.Valid, err)
			}
		})
	}
}
