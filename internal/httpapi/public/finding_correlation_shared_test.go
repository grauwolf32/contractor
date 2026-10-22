package public

import (
	"encoding/json"
	"testing"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPublicFindingCorrelationReadPages(t *testing.T) {
	var fixture struct {
		ReadPages []struct {
			Name, Schema string
			Valid        bool
			Document     any
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "public", "finding-correlation-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.ReadPages) == 0 {
		t.Fatal("no correlation page fixtures")
	}
	document := loadPublicOpenAPI(t)
	for _, c := range fixture.ReadPages {
		t.Run(c.Name, func(t *testing.T) {
			schema := document.Components.Schemas[c.Schema]
			if schema == nil || schema.Value == nil {
				t.Fatalf("unknown schema %s", c.Schema)
			}
			err := schema.Value.VisitJSON(c.Document, openapi3.EnableJSONSchema2020())
			if (err == nil) != c.Valid {
				t.Fatalf("valid=%v want=%v: %v", err == nil, c.Valid, err)
			}
		})
	}
}
