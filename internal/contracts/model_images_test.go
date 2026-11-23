package contracts

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"github.com/grauwolf32/contractor/internal/helpers/strictjson"
)

func TestSharedModelImageSupportMetadata(t *testing.T) {
	var fixture struct {
		Base  map[string]any
		Cases []struct {
			Name    string
			Present bool
			Value   any
			Valid   bool
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "model-image-support-cases.json"), &fixture); err != nil {
		t.Fatal(err)
	}
	for _, c := range fixture.Cases {
		t.Run(c.Name, func(t *testing.T) {
			value := map[string]any{}
			for k, v := range fixture.Base {
				value[k] = v
			}
			value["ref"] = map[string]any{"policyId": "image-policy", "version": "1", "digest": "sha256:" + strings.Repeat("a", 64)}
			if c.Present {
				value["supportsImages"] = c.Value
			}
			raw, _ := json.Marshal(value)
			var policy ResolvedModelPolicy
			err := strictjson.Decode(raw, &policy)
			if err == nil {
				err = policy.Validate()
			}
			if (err == nil) != c.Valid {
				t.Fatalf("valid=%v expected=%v: %v", err == nil, c.Valid, err)
			}
			if err == nil && policy.SupportsImages != nil {
				original := *policy.SupportsImages
				cloned := policy.Clone()
				*cloned.SupportsImages = !original
				if *policy.SupportsImages != original {
					t.Fatal("clone shares image support metadata")
				}
			}
		})
	}
}
