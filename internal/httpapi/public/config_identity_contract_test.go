package public

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedPublicConfigIdentitySchemasMatchSharedCases(t *testing.T) {
	t.Parallel()
	schemas := loadPublicOpenAPI(t).Components.Schemas
	name, version, selector := schemas["ConfigurationName"].Value, schemas["ConfigVersion"].Value, schemas["Selector"].Value
	data := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "config-identity-cases.json")
	type caseGroup struct {
		MaxLength int      `json:"maxLength"`
		Valid     []string `json:"valid"`
		Invalid   []string `json:"invalid"`
	}
	var cases struct {
		ID      caseGroup `json:"id"`
		Version caseGroup `json:"version"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	checkID := func(value string, valid bool) {
		t.Helper()
		if err := name.VisitJSON(value); (err == nil) != valid {
			t.Errorf("ConfigurationName %q: valid=%v, error=%v", value, valid, err)
		}
		if err := schemas["EvalSelector"].Value.VisitJSON(value + "@1"); (err == nil) != valid {
			t.Errorf("EvalSelector %q@1: valid=%v, error=%v", value, valid, err)
		}
		if err := selector.VisitJSON(value + "@1"); (err == nil) != valid {
			t.Errorf("Selector %q@1: valid=%v, error=%v", value, valid, err)
		}
	}
	checkVersion := func(value string, valid bool) {
		t.Helper()
		if err := version.VisitJSON(value); (err == nil) != valid {
			t.Errorf("ConfigVersion %q: valid=%v, error=%v", value, valid, err)
		}
		if err := schemas["EvalSelector"].Value.VisitJSON("a@" + value); (err == nil) != valid {
			t.Errorf("EvalSelector a@%q: valid=%v, error=%v", value, valid, err)
		}
		if err := selector.VisitJSON("a@" + value); (err == nil) != valid {
			t.Errorf("Selector a@%q: valid=%v, error=%v", value, valid, err)
		}
	}
	for _, value := range cases.ID.Valid {
		checkID(value, true)
	}
	for _, value := range cases.ID.Invalid {
		checkID(value, false)
	}
	for _, value := range cases.Version.Valid {
		checkVersion(value, true)
	}
	for _, value := range cases.Version.Invalid {
		checkVersion(value, false)
	}
	checkID(strings.Repeat("a", cases.ID.MaxLength), true)
	checkID(strings.Repeat("a", cases.ID.MaxLength+1), false)
	checkVersion(strings.Repeat("1", cases.Version.MaxLength), true)
	checkVersion(strings.Repeat("1", cases.Version.MaxLength+1), false)
}
