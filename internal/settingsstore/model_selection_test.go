package settingsstore

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

type modelProfileIDCases struct {
	Valid   []string `json:"valid"`
	Invalid []string `json:"invalid"`
}

func sharedModelProfileIDCases(t *testing.T) modelProfileIDCases {
	t.Helper()
	var cases modelProfileIDCases
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "model-profile-id-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	return cases
}

func TestSharedModelProfileIDs(t *testing.T) {
	cases := sharedModelProfileIDCases(t)
	for _, group := range []struct {
		values []string
		valid  bool
	}{{cases.Valid, true}, {cases.Invalid, false}} {
		for _, value := range group.values {
			for _, selection := range []ModelSelectionValue{
				{DefaultModel: value, Roles: map[string]string{}},
				{DefaultModel: "fixture", Roles: map[string]string{"summarizer": value}},
			} {
				if got := selection.Validate() == nil; got != group.valid {
					t.Fatalf("model ID %q valid=%v, want %v", value, got, group.valid)
				}
			}
		}
	}
}
