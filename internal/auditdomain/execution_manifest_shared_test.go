package auditdomain

import (
	"encoding/json"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
)

func TestSharedCompactExecutionManifest(t *testing.T) {
	var fixture struct {
		Schema string
		Base   json.RawMessage
		Cases  []struct {
			Name    string
			Valid   bool
			Patches []struct {
				Path   []any
				Value  any
				Delete bool
			}
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "audit-completion", "execution-manifest-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	if fixture.Schema != ExecutionManifestSchema || len(fixture.Cases) == 0 {
		t.Fatal("invalid shared execution fixture")
	}
	for _, test := range fixture.Cases {
		t.Run(test.Name, func(t *testing.T) {
			var value map[string]any
			if err := json.Unmarshal(fixture.Base, &value); err != nil {
				t.Fatal(err)
			}
			for _, patch := range test.Patches {
				var parent any = value
				for _, part := range patch.Path[:len(patch.Path)-1] {
					parent = dependencyPathValue(parent, part)
				}
				switch key := patch.Path[len(patch.Path)-1].(type) {
				case string:
					if patch.Delete {
						delete(parent.(map[string]any), key)
					} else {
						parent.(map[string]any)[key] = patch.Value
					}
				case float64:
					if patch.Delete {
						t.Fatal("fixture array deletion is unsupported")
					}
					parent.([]any)[int(key)] = patch.Value
				default:
					t.Fatal("invalid fixture path")
				}
			}
			data, err := canonicalJSON(value)
			if err != nil {
				t.Fatal(err)
			}
			manifest, err := DecodeExecutionManifest(data)
			if (err == nil) != test.Valid {
				t.Fatalf("accepted=%t want %t: %v", err == nil, test.Valid, err)
			}
			if err == nil {
				if err := ValidateDispatchExecutionManifest(manifest); err != nil {
					t.Fatal(err)
				}
				encoded, err := EncodeExecutionManifest(manifest)
				if err != nil || string(encoded) != string(data) {
					t.Fatalf("compact round trip differs: %v", err)
				}
			}
		})
	}
}
