package contracts_test

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestSharedProjectSitemapAccess(t *testing.T) {
	var cases struct {
		Valid            []json.RawMessage `json:"valid"`
		Invalid          []json.RawMessage `json:"invalid"`
		InvariantInvalid []json.RawMessage `json:"invariantInvalid"`
		StrictInvalid    []string          `json:"strictInvalid"`
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "sitemap-access-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Valid) < 2 || len(cases.Invalid) < 10 || len(cases.StrictInvalid) < 2 {
		t.Fatal("incomplete sitemap access cases")
	}
	compiler, ids := privateSchemaCompiler(t)
	schema, err := compiler.Compile(ids["common.schema.json"] + "#/$defs/projectSitemapPolicy")
	if err != nil {
		t.Fatal(err)
	}
	for _, group := range []struct {
		data          []json.RawMessage
		valid, schema bool
	}{
		{cases.Valid, true, true}, {cases.Invalid, false, true}, {cases.InvariantInvalid, false, false},
	} {
		for _, data := range group.data {
			var policy contracts.ProjectSitemapPolicy
			err := strictjson.Decode(data, &policy)
			if err == nil {
				err = policy.Validate()
			}
			if (err == nil) != group.valid {
				t.Fatalf("policy valid=%t: %s: %v", group.valid, data, err)
			}
			if group.schema {
				var value any
				if err := json.Unmarshal(data, &value); err != nil {
					t.Fatal(err)
				}
				if err := schema.Validate(value); (err == nil) != group.valid {
					t.Fatalf("schema valid=%t: %s: %v", group.valid, data, err)
				}
			}
		}
	}
	for _, data := range cases.StrictInvalid {
		var policy contracts.ProjectSitemapPolicy
		if err := strictjson.Decode([]byte(data), &policy); err == nil {
			t.Fatalf("accepted strict-invalid policy: %s", data)
		}
	}
}
