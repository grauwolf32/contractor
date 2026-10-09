package public

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestPublicMediaTypeSchemaMatchesSharedCases(t *testing.T) {
	t.Parallel()
	schemas := loadPublicOpenAPI(t).Components.Schemas
	schema := schemas["MediaType"].Value
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "api", "testdata", "v1alpha1", "media-type-cases.json"))
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
	for _, value := range cases.Valid {
		if err := schemas["EvalMedia"].Value.VisitJSON(value); err != nil {
			t.Errorf("EvalMedia %q rejected: %v", value, err)
		}
		if err := schema.VisitJSON(value); err != nil {
			t.Errorf("valid media type %q rejected: %v", value, err)
		}
	}
	for _, value := range cases.Invalid {
		if err := schemas["EvalMedia"].Value.VisitJSON(value); err == nil {
			t.Errorf("EvalMedia %q accepted", value)
		}
		// Slot declarations may use the explicit */* wildcard.
		if err := schema.VisitJSON(value); (err == nil) != (value == "*/*") {
			t.Errorf("media type %q: error=%v", value, err)
		}
	}
}
