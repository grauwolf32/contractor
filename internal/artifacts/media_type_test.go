package artifacts

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedStoreMediaTypeMatchesSharedCases(t *testing.T) {
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
		if err := validateMediaType(value); err != nil {
			t.Errorf("valid media type %q rejected: %v", value, err)
		}
	}
	for _, value := range cases.Invalid {
		if validateMediaType(value) == nil {
			t.Errorf("invalid media type %q accepted", value)
		}
	}
}
