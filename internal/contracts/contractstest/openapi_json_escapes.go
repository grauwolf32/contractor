package contractstest

import (
	"bytes"
	"encoding/json"
	"testing"
)

// OpenAPIJSONCase is a declared-format seed and its expected decoded value.
type OpenAPIJSONCase struct {
	Name      string         `json:"name"`
	MediaType string         `json:"mediaType"`
	Document  string         `json:"document"`
	Accepted  bool           `json:"accepted"`
	Value     map[string]any `json:"value"`
}

// ReadOpenAPIJSONCases reads the same JSON escape cases as Runtime authoring.
func ReadOpenAPIJSONCases(t testing.TB) []OpenAPIJSONCase {
	t.Helper()
	var fixture struct {
		Description string            `json:"description"`
		Cases       []OpenAPIJSONCase `json:"cases"`
	}
	decoder := json.NewDecoder(bytes.NewReader(ReadFile(t, "api", "testdata", "openapi-json-escape-cases.json")))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&fixture); err != nil {
		t.Fatalf("decode shared OpenAPI JSON cases: %v", err)
	}
	if len(fixture.Cases) == 0 {
		t.Fatal("shared OpenAPI JSON cases must not be empty")
	}
	for _, tc := range fixture.Cases {
		if tc.Name == "" || tc.MediaType != "application/json" || tc.Accepted != (tc.Value != nil) {
			t.Fatalf("invalid shared OpenAPI JSON case: %#v", tc)
		}
	}
	return fixture.Cases
}
