package config

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedSlotMediaTypesMatchSharedCases(t *testing.T) {
	t.Parallel()
	data := contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "media-type-cases.json")
	var cases struct {
		Valid   []string `json:"valid"`
		Invalid []string `json:"invalid"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, value := range cases.Valid {
		if _, err := validateMediaTypes("slot", []string{value}); err != nil {
			t.Errorf("valid media type %q rejected: %v", value, err)
		}
	}
	for _, value := range cases.Invalid {
		// A slot may accept any media type with the explicit */* wildcard.
		if _, err := validateMediaTypes("slot", []string{value}); (err == nil) != (value == "*/*") {
			t.Errorf("media type %q: error=%v", value, err)
		}
	}
}
