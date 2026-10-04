package auditdomain

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestCanonicalPackageMediaTypesMatchSharedCases(t *testing.T) {
	t.Parallel()
	data, err := os.ReadFile(filepath.Join("..", "..", "api", "testdata", "v1alpha1", "media-type-cases.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cases struct {
		Valid   []string `json:"valid"`
		Invalid []string `json:"invalid"`
	}
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	// Package members and item origins accept only canonical media types.
	canonical := func(value string) bool { return validMediaType(value) && normalizedMediaType(value) == value }
	for _, value := range cases.Valid {
		if !canonical(value) {
			t.Errorf("valid media type %q rejected", value)
		}
	}
	for _, value := range cases.Invalid {
		if canonical(value) {
			t.Errorf("invalid media type %q accepted", value)
		}
	}
}
