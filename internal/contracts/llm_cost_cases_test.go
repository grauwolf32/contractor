package contracts_test

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedLLMCostReportCases(t *testing.T) {
	var cases struct {
		Reports []struct {
			Name        string
			Cost        json.RawMessage
			Valid       bool
			SchemaValid *bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "llm-cost-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	entry := readFixtureIndex(t).Valid["allocation-final-response.json"]
	compiler, ids := privateSchemaCompiler(t)
	schema, err := compiler.Compile(ids[entry.Schema])
	if err != nil {
		t.Fatal(err)
	}
	fixture := contractstest.ReadFixture(t, "valid", "allocation-final-response.json")
	if len(cases.Reports) == 0 {
		t.Fatal("empty report cost cases")
	}
	for _, c := range cases.Reports {
		t.Run(c.Name, func(t *testing.T) {
			var value any
			if err := json.Unmarshal(c.Cost, &value); err != nil {
				t.Fatal(err)
			}
			doc := withValue(t, fixture, "/report/worker/metrics/llmCost", value)
			schemaValid := c.Valid
			if c.SchemaValid != nil {
				schemaValid = *c.SchemaValid
			}
			if got := schema.Validate(decodeNumbers(t, doc)) == nil; got != schemaValid {
				t.Fatalf("schema accepted=%v, expected=%v", got, schemaValid)
			}
			if err := fixtureCodecs[entry.Type].privateReject(doc); (err == nil) != c.Valid {
				t.Fatalf("Go accepted=%v: %v", err == nil, err)
			}
		})
	}
}
