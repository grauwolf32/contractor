package contracts

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestMediaTypeCases(t *testing.T) {
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
	for _, group := range []struct {
		values []string
		valid  bool
	}{{cases.Valid, true}, {cases.Invalid, false}} {
		for _, value := range group.values {
			if ValidMediaType(value) != group.valid {
				t.Errorf("ValidMediaType(%q) != %v", value, group.valid)
			}
			if err := validateArtifactMetadata(value, 0); (err == nil) != group.valid {
				t.Errorf("artifact metadata %q: valid=%v, error=%v", value, group.valid, err)
			}
		}
	}
}
