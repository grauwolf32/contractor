package public

import (
	"encoding/json"
	"github.com/getkin/kin-openapi/openapi3"
	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"testing"
)

func TestSharedArtifactStorageTimestamps(t *testing.T) {
	var cases []struct {
		Name  string
		Valid bool
		Value any
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "public", "artifact-storage-timestamps.json"), &cases); err != nil {
		t.Fatal(err)
	}
	schema := loadPublicOpenAPI(t).Components.Schemas["ArtifactMetadata"].Value
	for _, tc := range cases {
		t.Run(tc.Name, func(t *testing.T) {
			if err := schema.VisitJSON(tc.Value, openapi3.EnableJSONSchema2020()); (err == nil) != tc.Valid {
				t.Fatalf("valid=%t: %v", tc.Valid, err)
			}
		})
	}
}
